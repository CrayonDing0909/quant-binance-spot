from __future__ import annotations
from pathlib import Path
from typing import Optional
import logging
import pandas as pd
import vectorbt as vbt

from ..strategy.base import StrategyContext
from ..strategy import get_strategy
from ..data.storage import load_klines
from ..data.quality import validate_data_quality, clean_data
from ..risk.risk_limits import RiskLimits, apply_risk_limits
from .metrics import benchmark_buy_and_hold

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# Shared constants — 所有回測相關模組共用
# ══════════════════════════════════════════════════════════════

# 我們的 direction → vectorbt direction 映射
VBT_DIRECTION_MAP: dict[str, str] = {
    "both": "both",
    "long_only": "longonly",
    "short_only": "shortonly",
}


def to_vbt_direction(direction: str) -> str:
    """將我們的 direction 字串轉為 vectorbt 接受的格式"""
    return VBT_DIRECTION_MAP.get(direction, "longonly")


def clip_positions_by_direction(
    pos: pd.Series,
    market_type: str,
    direction: str,
) -> pd.Series:
    """
    根據 market_type / direction 過濾持倉信號
    
    - spot / long_only  → clip 掉做空信號
    - short_only        → 轉換符號讓 vectorbt shortonly 正確運作
    - both              → 不做處理
    """
    if market_type == "spot" or direction == "long_only":
        return pos.clip(lower=0.0)
    elif direction == "short_only":
        # vectorbt shortonly: size>0 = 開空, size<0 = 平空
        # 策略的 pos=-1 表示做空 → 轉換為 +1
        return (-pos).clip(lower=0.0)
    return pos  # "both": 保留 [-1, 1]


def _bps_to_pct(bps: float) -> float:
    return bps / 10_000.0


def _resolve_backtest_params(cfg: dict, **kwargs) -> dict:
    """
    從 cfg dict + explicit kwargs 解析回測參數
    
    explicit kwargs 優先（如果傳入非 None 值），否則 fallback 到 cfg dict。
    這樣無論呼叫者是用 explicit args 還是 cfg dict 都能正確運作。
    """
    return {
        "market_type": kwargs.get("market_type") or cfg.get("market_type", "spot"),
        "direction": kwargs.get("direction") or cfg.get("direction", "both"),
        "validate_data": kwargs.get("validate_data") if kwargs.get("validate_data") is not None else cfg.get("validate_data", True),
        "clean_data_before": kwargs.get("clean_data_before") if kwargs.get("clean_data_before") is not None else cfg.get("clean_data_before", True),
        "start": kwargs.get("start") or cfg.get("start"),
        "end": kwargs.get("end") or cfg.get("end"),
    }


def _apply_date_filter(
    df: pd.DataFrame,
    pos: pd.Series,
    start: str | None,
    end: str | None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    根據 start / end 日期過濾數據和持倉信號
    
    策略在完整數據上計算（確保指標 warmup 正確），
    之後只截取 [start, end] 區間送入 VBT 回測。
    
    這樣做的好處：
    1. 指標不會有 NaN warmup 問題
    2. 回測結果只反映指定時間範圍
    3. Total Return / Sharpe / MDD 等指標更精確
    """
    if start is None and end is None:
        return df, pos
    
    original_len = len(df)
    
    if start is not None:
        start_ts = pd.Timestamp(start, tz="UTC") if df.index.tz is not None else pd.Timestamp(start)
        mask = df.index >= start_ts
        df = df.loc[mask]
        pos = pos.loc[mask]
    
    if end is not None:
        end_ts = pd.Timestamp(end, tz="UTC") if df.index.tz is not None else pd.Timestamp(end)
        mask = df.index <= end_ts
        df = df.loc[mask]
        pos = pos.loc[mask]
    
    if len(df) < original_len:
        logger.info(
            f"📅 日期過濾: {original_len} → {len(df)} bars "
            f"({df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')})"
        )
    
    return df, pos


# ══════════════════════════════════════════════════════════════
# 核心回測函數
# ══════════════════════════════════════════════════════════════

def run_symbol_backtest(
    symbol: str,
    data_path: Path,
    cfg: dict,
    strategy_name: str = None,
    validate_data: Optional[bool] = None,
    clean_data_before: Optional[bool] = None,
    risk_limits: Optional[RiskLimits] = None,
    market_type: str | None = None,
    direction: str | None = None,
) -> dict:
    """
    運行單個交易對的回測

    Args:
        symbol: 交易對
        data_path: K 線數據路徑
        cfg: 配置字典（可包含 market_type / direction，作為 fallback）
        strategy_name: 策略名稱
        validate_data: 是否驗證數據
        clean_data_before: 是否清洗數據
        risk_limits: 風險限制
        market_type: "spot" 或 "futures"（None → 從 cfg 讀取，預設 "spot"）
        direction: "both" / "long_only" / "short_only"（None → 從 cfg 讀取）

    Returns:
        {
            "pf":       策略 Portfolio,
            "pf_bh":    Buy & Hold Portfolio (基準),
            "stats":    策略原始 stats,
            "df":       K線 DataFrame,
            "pos":      持倉序列,
        }
    """
    df = load_klines(data_path)

    # 解析參數（explicit args 優先，fallback 到 cfg dict）
    resolved = _resolve_backtest_params(
        cfg,
        market_type=market_type,
        direction=direction,
        validate_data=validate_data,
        clean_data_before=clean_data_before,
    )
    mt = resolved["market_type"]
    dr = resolved["direction"]

    # 數據質量檢查
    if resolved["validate_data"]:
        quality_report = validate_data_quality(df)
        if not quality_report.is_valid:
            print(f"⚠️  警告: {symbol} 數據質量問題")
            for error in quality_report.errors:
                print(f"  - {error}")
            for warning in quality_report.warnings:
                print(f"  - {warning}")

    # 數據清洗
    if resolved["clean_data_before"]:
        df = clean_data(df, fill_method="forward", remove_outliers=False, remove_duplicates=True)

    ctx = StrategyContext(
        symbol=symbol,
        interval=cfg.get("interval", "1h"),
        market_type=mt,
        direction=dr,
    )

    # 獲取策略函數
    strategy_name = strategy_name or cfg.get("strategy_name", "ema_cross")
    strategy_func = get_strategy(strategy_name)

    # positions: [-1, 1] (Futures) 或 [0, 1] (Spot)
    pos = strategy_func(df, ctx, cfg["strategy_params"])
    
    # 根據 direction 過濾信號（使用共用函數）
    pos = clip_positions_by_direction(pos, mt, dr)

    # 應用風險限制（如果提供）
    if risk_limits is not None:
        equity_curve = (1 + df["close"].pct_change()).cumprod() * cfg["initial_cash"]
        adjusted_pos = pd.Series(0.0, index=pos.index)
        for i in range(len(pos)):
            current_equity = equity_curve.iloc[i] if i < len(equity_curve) else cfg["initial_cash"]
            adjusted_pos.iloc[i], _ = apply_risk_limits(
                pos.iloc[i],
                equity_curve.iloc[:i+1],
                risk_limits,
                current_equity=current_equity
            )
        pos = adjusted_pos

    # ── 日期過濾 ──────────────────────────────────────
    # 策略已在完整數據上計算完畢（確保指標 warmup），
    # 現在截取 [start, end] 區間送入 VBT 回測
    df, pos = _apply_date_filter(df, pos, resolved.get("start"), resolved.get("end"))

    close = df["close"]
    open_ = df["open"]
    fee = _bps_to_pct(cfg["fee_bps"])
    slippage = _bps_to_pct(cfg["slippage_bps"])

    # ── 策略 Portfolio ─────────────────────────────
    vbt_direction = to_vbt_direction(dr)
    
    pf = vbt.Portfolio.from_orders(
        close=close,
        size=pos,
        size_type="targetpercent",
        price=open_,
        fees=fee,
        slippage=slippage,
        init_cash=cfg["initial_cash"],
        freq="1h",
        direction=vbt_direction,
    )

    # ── Buy & Hold 基準 ────────────────────────────
    pf_bh = benchmark_buy_and_hold(
        df,
        initial_cash=cfg["initial_cash"],
        fee_bps=cfg["fee_bps"],
        slippage_bps=cfg["slippage_bps"],
    )

    stats = pf.stats()
    return {"pf": pf, "pf_bh": pf_bh, "stats": stats, "df": df, "pos": pos}
