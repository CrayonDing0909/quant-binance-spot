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
    risk_limits: Optional[RiskLimits] = None
) -> dict:
    """
    运行单个交易对的回测

    Returns:
        {
            "pf":       策略 Portfolio,
            "pf_bh":    Buy & Hold Portfolio (基准),
            "stats":    策略原始 stats,
            "df":       K线 DataFrame,
            "pos":      持仓序列,
        }
    """
    df = load_klines(data_path)

    # 数据质量检查
    should_validate = validate_data if validate_data is not None else cfg.get("validate_data", True)
    if should_validate:
        quality_report = validate_data_quality(df)
        if not quality_report.is_valid:
            print(f"⚠️  警告: {symbol} 数据质量问题")
            for error in quality_report.errors:
                print(f"  - {error}")
            for warning in quality_report.warnings:
                print(f"  - {warning}")

    # 数据清洗
    should_clean = clean_data_before if clean_data_before is not None else cfg.get("clean_data_before", True)
    if should_clean:
        df = clean_data(df, fill_method="forward", remove_outliers=False, remove_duplicates=True)

    ctx = StrategyContext(symbol=symbol)

    # 获取策略函数
    strategy_name = strategy_name or cfg.get("strategy_name", "ema_cross")
    strategy_func = get_strategy(strategy_name)

    # positions: 0..1 (Spot long-only)
    pos = strategy_func(df, ctx, cfg["strategy_params"])

    # 应用风险限制（如果提供）
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
    pf = vbt.Portfolio.from_orders(
        close=close,
        size=pos,
        size_type="targetpercent",
        price=open_,
        fees=fee,
        slippage=slippage,
        init_cash=cfg["initial_cash"],
        freq="1h",
        direction="longonly",
    )

    # ── Buy & Hold 基准 ────────────────────────────
    pf_bh = benchmark_buy_and_hold(
        df,
        initial_cash=cfg["initial_cash"],
        fee_bps=cfg["fee_bps"],
        slippage_bps=cfg["slippage_bps"],
    )

    stats = pf.stats()
    return {"pf": pf, "pf_bh": pf_bh, "stats": stats, "df": df, "pos": pos}
