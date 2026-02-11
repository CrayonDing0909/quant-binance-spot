from __future__ import annotations
from pathlib import Path
from typing import Optional
import pandas as pd
import vectorbt as vbt

from ..strategy.base import StrategyContext
from ..strategy import get_strategy
from ..data.storage import load_klines
from ..data.quality import validate_data_quality, clean_data
from ..risk.risk_limits import RiskLimits, apply_risk_limits
from .metrics import benchmark_buy_and_hold


def _bps_to_pct(bps: float) -> float:
    return bps / 10_000.0


def run_symbol_backtest(
    symbol: str,
    data_path: Path,
    cfg: dict,
    strategy_name: str = None,
    validate_data: Optional[bool] = None,
    clean_data_before: Optional[bool] = None,
    risk_limits: Optional[RiskLimits] = None,
    market_type: str = "spot",
    direction: str = "both",
) -> dict:
    """
    運行單個交易對的回測

    Args:
        symbol: 交易對
        data_path: K 線數據路徑
        cfg: 配置字典
        strategy_name: 策略名稱
        validate_data: 是否驗證數據
        clean_data_before: 是否清洗數據
        risk_limits: 風險限制
        market_type: "spot" 或 "futures"（futures 支援做空）
        direction: 交易方向
            - "both": 多空都做（預設）
            - "long_only": 只做多
            - "short_only": 只做空

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

    # 數據質量檢查
    should_validate = validate_data if validate_data is not None else cfg.get("validate_data", True)
    if should_validate:
        quality_report = validate_data_quality(df)
        if not quality_report.is_valid:
            print(f"⚠️  警告: {symbol} 數據質量問題")
            for error in quality_report.errors:
                print(f"  - {error}")
            for warning in quality_report.warnings:
                print(f"  - {warning}")

    # 數據清洗
    should_clean = clean_data_before if clean_data_before is not None else cfg.get("clean_data_before", True)
    if should_clean:
        df = clean_data(df, fill_method="forward", remove_outliers=False, remove_duplicates=True)

    ctx = StrategyContext(
        symbol=symbol,
        interval=cfg.get("interval", "1h"),
        market_type=market_type,
        direction=direction,
    )

    # 獲取策略函數
    strategy_name = strategy_name or cfg.get("strategy_name", "ema_cross")
    strategy_func = get_strategy(strategy_name)

    # positions: [-1, 1] (Futures) 或 [0, 1] (Spot)
    pos = strategy_func(df, ctx, cfg["strategy_params"])
    
    # 根據 direction 過濾信號
    if market_type == "spot" or direction == "long_only":
        # Spot 模式或 long_only：將做空信號 clip 到 0
        pos = pos.clip(lower=0.0)
    elif direction == "short_only":
        # short_only：只保留做空信號，並轉換符號讓 vectorbt 正確解讀
        # 策略的 pos=-1 表示做空，但 vectorbt shortonly 需要 size>0 才開空
        # 轉換：-1 → 1 (開空), 1 → -1 → clip → 0 (過濾掉做多)
        pos = (-pos).clip(lower=0.0)

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

    close = df["close"]
    open_ = df["open"]
    fee = _bps_to_pct(cfg["fee_bps"])
    slippage = _bps_to_pct(cfg["slippage_bps"])

    # ── 策略 Portfolio ─────────────────────────────
    # vectorbt direction 參數：
    # - "longonly": 只做多
    # - "shortonly": 只做空
    # - "both": 多空都做
    vbt_direction_map = {
        "both": "both",
        "long_only": "longonly",
        "short_only": "shortonly",
    }
    vbt_direction = vbt_direction_map.get(direction, "longonly")
    
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
