from __future__ import annotations
from pathlib import Path
from typing import Optional
import logging
import numpy as np
import pandas as pd
import vectorbt as vbt

from ..strategy.base import StrategyContext
from ..strategy import get_strategy
from ..data.storage import load_klines
from ..data.quality import validate_data_quality, clean_data
from ..risk.risk_limits import RiskLimits, apply_risk_limits
from .metrics import benchmark_buy_and_hold
from .costs import (
    compute_funding_costs,
    adjust_equity_for_funding,
    compute_adjusted_stats,
    compute_volume_slippage,
    FundingCostResult,
    SlippageResult,
)
from ..data.funding_rate import (
    load_funding_rates,
    get_funding_rate_path,
    align_funding_to_klines,
)

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
# 波動率目標倉位縮放
# ══════════════════════════════════════════════════════════════

def _apply_vol_scaling(
    pos: pd.Series,
    df: pd.DataFrame,
    target_vol: float = 0.15,
    vol_lookback: int = 168,
    max_scale: float = 2.0,
    min_scale: float = 0.1,
    interval: str = "1h",
) -> pd.Series:
    """
    根據實現波動率反向縮放倉位（Volatility Targeting）

    高波動期 → 降低倉位，低波動期 → 提高倉位（但不超過 max_scale）
    
    公式: scale = target_vol / realized_vol
    
    Args:
        pos: 原始信號 [-1, 1]
        df: K 線 DataFrame（需要 close 欄位）
        target_vol: 目標年化波動率（預設 15%）
        vol_lookback: 波動率計算回看期（bar 數）
        max_scale: 最大縮放倍數
        min_scale: 最小縮放倍數
        interval: 時間間隔（用於年化）
    
    Returns:
        縮放後的倉位信號（連續值）
    """
    # 根據 interval 決定年化因子
    annualize_factors = {
        "1m": np.sqrt(525_600),
        "5m": np.sqrt(105_120),
        "15m": np.sqrt(35_040),
        "1h": np.sqrt(8_760),
        "4h": np.sqrt(2_190),
        "1d": np.sqrt(365),
    }
    annualize = annualize_factors.get(interval, np.sqrt(8_760))
    
    returns = df["close"].pct_change()
    realized_vol = returns.rolling(window=vol_lookback).std() * annualize
    
    # 避免除以零 & warmup 期用 target_vol
    realized_vol = realized_vol.replace(0, np.nan).ffill().fillna(target_vol)
    
    scale = (target_vol / realized_vol).clip(lower=min_scale, upper=max_scale)
    
    scaled_pos = pos * scale
    # 最終仍然限制在 [-1, 1]
    scaled_pos = scaled_pos.clip(lower=-1.0, upper=1.0)
    
    logger.info(
        f"📊 Vol Targeting: target={target_vol:.0%}, "
        f"avg_realized={realized_vol.mean():.1%}, "
        f"avg_scale={scale.mean():.2f}, "
        f"avg_|pos|={scaled_pos.abs().mean():.3f}"
    )
    
    return scaled_pos


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
    data_dir: Path | None = None,
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
        data_dir: 數據根目錄（用於載入 funding rate 等輔助數據）

    Returns:
        {
            "pf":       策略 Portfolio,
            "pf_bh":    Buy & Hold Portfolio (基準),
            "stats":    策略原始 stats,
            "df":       K線 DataFrame,
            "pos":      持倉序列,
            # ── 成本模型（如果啟用）──
            "funding_cost":       FundingCostResult | None,
            "slippage_result":    SlippageResult | None,
            "adjusted_stats":     dict | None,
            "adjusted_equity":    Series | None,
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

    # ── 波動率目標倉位縮放（已停用）─────────────────
    # 停用原因：此區塊只在 run_symbol_backtest 生效，
    # 但 run_portfolio_backtest 和 websocket_runner 都沒有 vol scaling，
    # 導致驗證測試的 Sharpe/收益率比實盤樂觀 30-70%。
    # 三條路徑統一為：不做 vol scaling，由 portfolio.allocation 控制曝險。
    # 如需重新啟用，請同步修改 run_portfolio_backtest.py 和 websocket_runner.py。
    # ───────────────────────────────────────────────

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

    # ── 構建執行價格（消除 SL/TP look-ahead bias）──────
    # exit_exec_prices: SL/TP 觸發時為實際出場價，其餘為 NaN
    exit_exec_prices = pos.attrs.get("exit_exec_prices")
    if exit_exec_prices is not None:
        # 對齊到日期過濾後的索引
        exit_exec_prices = exit_exec_prices.reindex(pos.index)
        # 自定義執行價格: SL/TP bar 使用出場價，其餘用 open
        exec_price = open_.copy()
        sl_tp_mask = exit_exec_prices.notna()
        exec_price[sl_tp_mask] = exit_exec_prices[sl_tp_mask]
        logger.info(
            f"🔧 SL/TP 出場價修正: {sl_tp_mask.sum()} bars 使用實際 SL/TP 價格"
        )
    else:
        exec_price = open_

    # ── 滑點模型 ──────────────────────────────────────
    sm_cfg = cfg.get("slippage_model", {})
    slippage_result: SlippageResult | None = None

    if sm_cfg.get("enabled", False):
        leverage = cfg.get("leverage", 1)
        slippage_result = compute_volume_slippage(
            pos=pos,
            df=df,
            capital=cfg["initial_cash"],
            base_bps=sm_cfg.get("base_bps", 2.0),
            impact_coefficient=sm_cfg.get("impact_coefficient", 0.1),
            impact_power=sm_cfg.get("impact_power", 0.5),
            adv_lookback=sm_cfg.get("adv_lookback", 20),
            participation_rate=sm_cfg.get("participation_rate", 0.10),
            leverage=leverage,
        )
        slippage = slippage_result.slippage_array
        logger.info(
            f"📊 {symbol} Volume slippage: "
            f"avg={slippage_result.avg_slippage_bps:.1f}bps, "
            f"max={slippage_result.max_slippage_bps:.1f}bps, "
            f"high_impact={slippage_result.high_impact_bars} bars"
        )
    else:
        slippage = _bps_to_pct(cfg["slippage_bps"])

    # ── 策略 Portfolio ─────────────────────────────
    vbt_direction = to_vbt_direction(dr)
    
    pf = vbt.Portfolio.from_orders(
        close=close,
        size=pos,
        size_type="targetpercent",
        price=exec_price,
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

    # ── Funding Rate 成本模型 ──────────────────────
    fr_cfg = cfg.get("funding_rate", {})
    funding_cost: FundingCostResult | None = None
    adjusted_stats: dict | None = None
    adjusted_equity: pd.Series | None = None

    if fr_cfg.get("enabled", False) and mt == "futures":
        # 嘗試載入歷史 funding rate
        funding_df = None
        if fr_cfg.get("use_historical", True) and data_dir is not None:
            fr_path = get_funding_rate_path(data_dir, symbol)
            funding_df = load_funding_rates(fr_path)
            if funding_df is not None:
                logger.info(f"📥 {symbol} 載入歷史 funding rate: {len(funding_df)} records")
            else:
                logger.info(f"ℹ️  {symbol} 無歷史 funding rate，使用預設費率")

        # 對齊到 kline 時間軸
        funding_rates = align_funding_to_klines(
            funding_df,
            df.index,
            default_rate_8h=fr_cfg.get("default_rate_8h", 0.0001),
        )

        # 計算 funding 成本
        leverage = cfg.get("leverage", 1)
        equity = pf.value()
        funding_cost = compute_funding_costs(
            pos=pos,
            equity=equity,
            funding_rates=funding_rates,
            leverage=leverage,
        )

        # 調整後的資金曲線和統計
        adjusted_equity = adjust_equity_for_funding(equity, funding_cost)
        adjusted_stats = compute_adjusted_stats(adjusted_equity, cfg["initial_cash"])

        logger.info(
            f"💰 {symbol} Funding cost: "
            f"total=${funding_cost.total_cost:,.2f} "
            f"({funding_cost.total_cost_pct*100:.2f}%), "
            f"annualized={funding_cost.annualized_cost_pct*100:.2f}%/yr, "
            f"settlements={funding_cost.n_settlements}"
        )

    return {
        "pf": pf,
        "pf_bh": pf_bh,
        "stats": stats,
        "df": df,
        "pos": pos,
        # 成本模型
        "funding_cost": funding_cost,
        "slippage_result": slippage_result,
        "adjusted_stats": adjusted_stats,
        "adjusted_equity": adjusted_equity,
    }
