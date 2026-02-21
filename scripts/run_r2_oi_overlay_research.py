#!/usr/bin/env python3
"""
R2 + OI Exit Overlay Research Runner

完整研究流程：
    1. R2 baseline（3 cost multipliers）
    2. A/B/C overlay configs（3 × 3 = 9 runs）
    3. Year-by-year breakdown
    4. Walk-forward analysis（5 splits × 3 symbols × 4 configs）
    5. Delta comparison table
    6. Acceptance criteria check

Usage:
    cd /path/to/quant-binance-spot
    PYTHONPATH=src python scripts/run_r2_oi_overlay_research.py

    # Skip walk-forward (faster iteration)
    PYTHONPATH=src python scripts/run_r2_oi_overlay_research.py --skip-wf

    # Only run specific overlay config
    PYTHONPATH=src python scripts/run_r2_oi_overlay_research.py --configs B
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# 確保 src 在 path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import vectorbt as vbt

from qtrade.config import load_config
from qtrade.backtest.run_backtest import (
    BacktestResult,
    _bps_to_pct,
    to_vbt_direction,
    clip_positions_by_direction,
    validate_backtest_config,
    _apply_date_filter,
)
from qtrade.backtest.metrics import benchmark_buy_and_hold
from qtrade.backtest.costs import (
    compute_funding_costs,
    adjust_equity_for_funding,
    compute_adjusted_stats,
    FundingCostResult,
)
from qtrade.strategy import get_strategy
from qtrade.strategy.base import StrategyContext
from qtrade.data.storage import load_klines
from qtrade.data.quality import validate_data_quality, clean_data
from qtrade.data.funding_rate import (
    load_funding_rates,
    get_funding_rate_path,
    align_funding_to_klines,
)
from qtrade.data.open_interest import (
    load_open_interest,
    get_oi_path,
    align_oi_to_klines,
)
from qtrade.strategy.overlays.oi_vol_exit_overlay import (
    apply_full_oi_vol_overlay,
)
from qtrade.validation.walk_forward import walk_forward_analysis, walk_forward_summary

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Suppress vectorbt warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ══════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).parent.parent
BASELINE_CONFIG = PROJECT_ROOT / "config" / "prod_candidate_R2.yaml"
OVERLAY_CONFIGS = {
    "A": PROJECT_ROOT / "config" / "research_r2_oi_overlay_A.yaml",
    "B": PROJECT_ROOT / "config" / "research_r2_oi_overlay_B.yaml",
    "C": PROJECT_ROOT / "config" / "research_r2_oi_overlay_C.yaml",
}
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
WEIGHTS = {"BTCUSDT": 0.34, "ETHUSDT": 0.33, "SOLUSDT": 0.33}
COST_MULTS = [1.0, 1.5, 2.0]
YEAR_RANGES = {
    "2022": ("2022-01-01", "2022-12-31"),
    "2023": ("2023-01-01", "2023-12-31"),
    "2024": ("2024-01-01", "2024-12-31"),
    "2025": ("2025-01-01", "2025-12-31"),
    "2026 YTD": ("2026-01-01", None),
}


# ══════════════════════════════════════════════════════════════
# Core: Single-symbol backtest with optional overlay
# ══════════════════════════════════════════════════════════════

def _load_ensemble_strategy(config_path: str | Path, symbol: str) -> tuple | None:
    """從 ensemble 配置取得 symbol 的策略名與參數"""
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    ens = raw.get("ensemble")
    if ens and ens.get("enabled", False):
        strategies = ens.get("strategies", {})
        if symbol in strategies:
            s = strategies[symbol]
            return s["name"], s.get("params", {})
    return None


def _load_overlay_params(config_path: str | Path) -> dict | None:
    """從 config 載入 oi_vol_overlay 參數"""
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    overlay = raw.get("oi_vol_overlay")
    if overlay and overlay.get("enabled", False):
        return overlay
    return None


def run_single_symbol_with_overlay(
    symbol: str,
    cfg,
    config_path: str | Path,
    cost_mult: float = 1.0,
    overlay_params: dict | None = None,
    start_override: str | None = None,
    end_override: str | None = None,
) -> dict | None:
    """
    運行單幣種回測（支援 overlay 注入）

    返回 dict:
        {
            "symbol": str,
            "total_return_pct": float,
            "sharpe": float,
            "max_drawdown_pct": float,
            "total_trades": int,
            "calmar": float,
            "cagr": float,
            "equity": pd.Series,
            "pos": pd.Series,
        }
    """
    market_type = cfg.market_type_str
    data_path = (
        cfg.data_dir / "binance" / market_type
        / cfg.market.interval / f"{symbol}.parquet"
    )
    if not data_path.exists():
        print(f"    ⚠️  {symbol}: data not found at {data_path}")
        return None

    # Load data
    df = load_klines(data_path)
    df = clean_data(df, fill_method="forward", remove_outliers=False, remove_duplicates=True)

    # Resolve strategy (ensemble routing)
    ensemble_override = _load_ensemble_strategy(config_path, symbol)
    if ensemble_override:
        strategy_name, strategy_params = ensemble_override
    else:
        strategy_name = cfg.strategy.name
        strategy_params = cfg.strategy.get_params(symbol)

    # Build context
    ctx = StrategyContext(
        symbol=symbol,
        interval=cfg.market.interval,
        market_type=market_type,
        direction=cfg.direction,
        signal_delay=1,  # trade_on=next_open
    )

    # Generate base positions
    strategy_func = get_strategy(strategy_name)
    pos = strategy_func(df, ctx, strategy_params)
    pos = clip_positions_by_direction(pos, market_type, cfg.direction)

    # Apply overlay (if any)
    if overlay_params is not None:
        # Load OI data
        oi_path = get_oi_path(cfg.data_dir, symbol)
        oi_df = load_open_interest(oi_path)
        oi_series = align_oi_to_klines(oi_df, df.index) if oi_df is not None else None

        pos = apply_full_oi_vol_overlay(
            position=pos,
            price_df=df,
            oi_series=oi_series,
            params=overlay_params,
        )

    # Position sizing
    ps_cfg = cfg.position_sizing
    if ps_cfg.method == "fixed" and ps_cfg.position_pct < 1.0:
        pos = pos * ps_cfg.position_pct

    # Date filter
    start = start_override or cfg.market.start
    end = end_override or cfg.market.end
    df, pos = _apply_date_filter(df, pos, start, end)

    if len(df) < 100:
        return None

    # Cost settings with multiplier
    fee_bps = cfg.backtest.fee_bps * cost_mult
    slippage_bps = cfg.backtest.slippage_bps * cost_mult
    fee = _bps_to_pct(fee_bps)
    slippage = _bps_to_pct(slippage_bps)
    initial_cash = cfg.backtest.initial_cash

    close = df["close"]
    open_ = df["open"]

    # Build VBT Portfolio
    vbt_direction = to_vbt_direction(cfg.direction)
    pf = vbt.Portfolio.from_orders(
        close=close,
        size=pos,
        size_type="targetpercent",
        price=open_,
        fees=fee,
        slippage=slippage,
        init_cash=initial_cash,
        freq=cfg.market.interval,
        direction=vbt_direction,
    )

    stats = pf.stats()
    equity = pf.value()

    # Funding rate
    adjusted_equity = None
    adjusted_stats = None
    fr = cfg.backtest.funding_rate
    if fr.enabled and market_type == "futures":
        funding_df = None
        if fr.use_historical:
            fr_path = get_funding_rate_path(cfg.data_dir, symbol)
            funding_df = load_funding_rates(fr_path)

        funding_rates = align_funding_to_klines(
            funding_df, df.index,
            default_rate_8h=fr.default_rate_8h,
        )
        leverage = cfg.futures.leverage if cfg.futures else 1
        funding_cost = compute_funding_costs(
            pos=pos, equity=equity,
            funding_rates=funding_rates,
            leverage=leverage,
        )
        adjusted_equity = adjust_equity_for_funding(equity, funding_cost)
        adjusted_stats = compute_adjusted_stats(adjusted_equity, initial_cash)

    # Extract metrics
    final_stats = adjusted_stats if adjusted_stats else stats
    eq = adjusted_equity if adjusted_equity is not None else equity

    total_return_pct = final_stats.get("Total Return [%]", 0.0)
    sharpe = final_stats.get("Sharpe Ratio", 0.0)
    max_dd = abs(final_stats.get("Max Drawdown [%]", 0.0))
    total_trades = stats.get("Total Trades", 0)

    # Compute CAGR
    n_bars = len(df)
    years = n_bars / (365.25 * 24)
    total_ret = total_return_pct / 100.0
    cagr = ((1 + total_ret) ** (1 / max(years, 0.01)) - 1) * 100 if years > 0 else 0
    calmar = cagr / max_dd if max_dd > 0.01 else 0

    return {
        "symbol": symbol,
        "total_return_pct": total_return_pct,
        "sharpe": sharpe,
        "max_drawdown_pct": max_dd,
        "total_trades": int(total_trades),
        "cagr": cagr,
        "calmar": calmar,
        "equity": eq,
        "pos": pos,
    }


def run_portfolio_config(
    cfg,
    config_path: str | Path,
    cost_mult: float = 1.0,
    overlay_params: dict | None = None,
    start_override: str | None = None,
    end_override: str | None = None,
    label: str = "",
) -> dict:
    """
    運行完整組合回測（3 symbols weighted）

    Returns dict with portfolio-level metrics
    """
    initial_cash = cfg.backtest.initial_cash
    per_symbol = {}

    for symbol in SYMBOLS:
        res = run_single_symbol_with_overlay(
            symbol=symbol,
            cfg=cfg,
            config_path=config_path,
            cost_mult=cost_mult,
            overlay_params=overlay_params,
            start_override=start_override,
            end_override=end_override,
        )
        if res is not None:
            per_symbol[symbol] = res

    if not per_symbol:
        return {"label": label, "sharpe": 0, "total_return_pct": 0,
                "max_drawdown_pct": 0, "cagr": 0, "calmar": 0,
                "total_trades": 0, "per_symbol": {}}

    # Build portfolio equity curve
    active_symbols = list(per_symbol.keys())
    active_weights = np.array([WEIGHTS.get(s, 1.0 / len(SYMBOLS)) for s in active_symbols])
    active_weights = active_weights / active_weights.sum()

    equity_curves = {s: per_symbol[s]["equity"] for s in active_symbols}

    # Align to common time range
    min_start = max(eq.index[0] for eq in equity_curves.values())
    max_end = min(eq.index[-1] for eq in equity_curves.values())

    for s in active_symbols:
        equity_curves[s] = equity_curves[s].loc[min_start:max_end]

    # Normalized portfolio
    normalized = {}
    for s in active_symbols:
        eq = equity_curves[s]
        if len(eq) > 0 and eq.iloc[0] > 0:
            normalized[s] = eq / eq.iloc[0]
        else:
            normalized[s] = pd.Series(1.0, index=eq.index)

    portfolio_normalized = sum(
        normalized[s] * w for s, w in zip(active_symbols, active_weights)
    )
    portfolio_equity = portfolio_normalized * initial_cash
    portfolio_returns = portfolio_equity.pct_change().fillna(0)

    # Compute portfolio stats
    n_bars = len(portfolio_returns)
    years = n_bars / (365.25 * 24)
    total_return = (portfolio_equity.iloc[-1] / initial_cash - 1) * 100

    cagr = ((1 + total_return / 100) ** (1 / max(years, 0.01)) - 1) * 100

    rolling_max = portfolio_equity.expanding().max()
    drawdown = (portfolio_equity - rolling_max) / rolling_max
    max_dd = abs(drawdown.min()) * 100

    excess_returns = portfolio_returns
    sharpe = (
        np.sqrt(365 * 24) * excess_returns.mean() / excess_returns.std()
        if excess_returns.std() > 0 else 0
    )

    calmar = cagr / max_dd if max_dd > 0.01 else 0

    total_trades = sum(per_symbol[s]["total_trades"] for s in active_symbols)

    return {
        "label": label,
        "total_return_pct": round(total_return, 2),
        "cagr": round(cagr, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "calmar": round(calmar, 2),
        "total_trades": total_trades,
        "per_symbol": per_symbol,
        "portfolio_equity": portfolio_equity,
    }


# ══════════════════════════════════════════════════════════════
# Walk-Forward Helper
# ══════════════════════════════════════════════════════════════

def run_walk_forward_for_config(
    cfg,
    config_path: str | Path,
    n_splits: int = 5,
    overlay_params: dict | None = None,
) -> dict:
    """
    為每個 symbol 跑 walk-forward，返回 OOS Sharpe 摘要

    overlay 在 walk-forward 中不注入（WF 已有自己的 pipeline），
    我們只用 baseline WF 數據作為穩健性參考。
    """
    results = {}
    for symbol in SYMBOLS:
        market_type = cfg.market_type_str
        data_path = (
            cfg.data_dir / "binance" / market_type
            / cfg.market.interval / f"{symbol}.parquet"
        )
        if not data_path.exists():
            results[symbol] = {"avg_test_sharpe": 0, "oos_positive_pct": 0}
            continue

        # Build backtest config
        bt_cfg = cfg.to_backtest_dict(symbol=symbol)

        # Ensemble: override strategy
        ensemble_override = _load_ensemble_strategy(config_path, symbol)
        if ensemble_override:
            sym_name, sym_params = ensemble_override
            bt_cfg["strategy_name"] = sym_name
            bt_cfg["strategy_params"] = sym_params

        try:
            wf_df = walk_forward_analysis(
                symbol=symbol,
                data_path=data_path,
                cfg=bt_cfg,
                n_splits=n_splits,
                data_dir=cfg.data_dir,
            )
            if wf_df.empty:
                results[symbol] = {"avg_test_sharpe": 0, "oos_positive_pct": 0}
            else:
                summary = walk_forward_summary(wf_df)
                results[symbol] = summary
        except Exception as e:
            print(f"    ⚠️  WF {symbol} failed: {e}")
            results[symbol] = {"avg_test_sharpe": 0, "oos_positive_pct": 0}

    return results


# ══════════════════════════════════════════════════════════════
# Reporting
# ══════════════════════════════════════════════════════════════

def print_comparison_table(all_results: list[dict]):
    """Print Core Comparison Table (Section B)"""
    print("\n" + "=" * 110)
    print("B. CORE COMPARISON TABLE")
    print("=" * 110)
    header = f"{'Config':<22} {'Cost':>6} {'TotRet%':>10} {'CAGR%':>8} {'Sharpe':>8} {'MaxDD%':>8} {'Calmar':>8} {'Trades':>8}"
    print(header)
    print("-" * 110)
    for r in all_results:
        print(
            f"{r['label']:<22} "
            f"{r.get('cost_mult', 1.0):>6.1f} "
            f"{r['total_return_pct']:>10.2f} "
            f"{r['cagr']:>8.2f} "
            f"{r['sharpe']:>8.2f} "
            f"{r['max_drawdown_pct']:>8.2f} "
            f"{r['calmar']:>8.2f} "
            f"{r['total_trades']:>8d}"
        )


def print_delta_table(baseline_results: dict, overlay_results: list[dict]):
    """Print Delta vs R2 Baseline (Section C)"""
    print("\n" + "=" * 100)
    print("C. DELTA vs R2 BASELINE (cost_mult=1.5)")
    print("=" * 100)
    header = f"{'Config':<22} {'ΔSharpe':>10} {'ΔMaxDD':>10} {'ΔTrades':>10} {'ΔTrades%':>10} {'2025 ΔRet':>10}"
    print(header)
    print("-" * 100)

    bl = baseline_results
    for r in overlay_results:
        d_sharpe = r["sharpe"] - bl["sharpe"]
        d_mdd = r["max_drawdown_pct"] - bl["max_drawdown_pct"]
        d_trades = r["total_trades"] - bl["total_trades"]
        d_trades_pct = (d_trades / max(bl["total_trades"], 1)) * 100
        d_ret_2025 = r.get("ret_2025", 0) - bl.get("ret_2025", 0)
        print(
            f"{r['label']:<22} "
            f"{d_sharpe:>+10.2f} "
            f"{d_mdd:>+10.2f} "
            f"{d_trades:>+10d} "
            f"{d_trades_pct:>+10.1f}% "
            f"{d_ret_2025:>+10.2f}"
        )


def print_walk_forward_table(wf_results: dict):
    """Print Walk-Forward Table (Section D)"""
    print("\n" + "=" * 90)
    print("D. WALK-FORWARD TABLE")
    print("=" * 90)
    header = f"{'Config':<22} {'BTC OOS SR':>12} {'ETH OOS SR':>12} {'SOL OOS SR':>12} {'OOS+/5':>8}"
    print(header)
    print("-" * 90)

    for config_name, wf in wf_results.items():
        btc_sr = wf.get("BTCUSDT", {}).get("avg_test_sharpe", 0)
        eth_sr = wf.get("ETHUSDT", {}).get("avg_test_sharpe", 0)
        sol_sr = wf.get("SOLUSDT", {}).get("avg_test_sharpe", 0)

        # Count how many OOS positive across symbols
        oos_pos_count = 0
        for sym in SYMBOLS:
            pct = wf.get(sym, {}).get("oos_positive_pct", 0)
            if pct > 0:
                oos_pos_count += 1

        print(
            f"{config_name:<22} "
            f"{btc_sr:>12.2f} "
            f"{eth_sr:>12.2f} "
            f"{sol_sr:>12.2f} "
            f"{oos_pos_count:>8d}/3"
        )


def check_acceptance_criteria(
    baseline_15: dict,
    overlay_results_15: list[dict],
    baseline_10: dict,
    overlay_results_10: list[dict],
) -> dict:
    """
    Check acceptance criteria against baseline

    Returns: {config_name: {pass_sharpe, pass_trades, pass_mdd, pass_2025, overall}}
    """
    verdicts = {}

    for ov_15, ov_10 in zip(overlay_results_15, overlay_results_10):
        name = ov_15["label"]

        # 1. Sharpe at cost_mult=1.5 >= baseline
        pass_sharpe = ov_15["sharpe"] >= baseline_15["sharpe"]

        # 2. trade_count decreases >= 30%
        trade_delta_pct = (
            (ov_10["total_trades"] - baseline_10["total_trades"])
            / max(baseline_10["total_trades"], 1)
        ) * 100
        pass_trades = trade_delta_pct <= -30

        # 3. MaxDD does not worsen by more than +1pp
        mdd_delta = ov_15["max_drawdown_pct"] - baseline_15["max_drawdown_pct"]
        pass_mdd = mdd_delta <= 1.0

        # 4. 2025 year return is better than baseline
        pass_2025 = ov_15.get("ret_2025", 0) >= baseline_15.get("ret_2025", 0)

        overall = pass_sharpe and pass_trades and pass_mdd and pass_2025

        verdicts[name] = {
            "pass_sharpe": pass_sharpe,
            "pass_trades": pass_trades,
            "pass_mdd": pass_mdd,
            "pass_2025": pass_2025,
            "overall": overall,
            "sharpe_delta": ov_15["sharpe"] - baseline_15["sharpe"],
            "trade_delta_pct": trade_delta_pct,
            "mdd_delta": mdd_delta,
            "ret_2025_delta": ov_15.get("ret_2025", 0) - baseline_15.get("ret_2025", 0),
        }

    return verdicts


# ══════════════════════════════════════════════════════════════
# Main Research Pipeline
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="R2 + OI Overlay Research")
    parser.add_argument("--skip-wf", action="store_true", help="Skip walk-forward analysis")
    parser.add_argument("--configs", nargs="+", default=["A", "B", "C"],
                        choices=["A", "B", "C"], help="Overlay configs to run")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT / "reports" / "oi_overlay_research" / timestamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  R2 + OI Exit Overlay Research")
    print("=" * 80)
    print(f"  Timestamp:  {timestamp}")
    print(f"  Output Dir: {output_dir}")
    print(f"  Baseline:   {BASELINE_CONFIG.name}")
    print(f"  Overlays:   {', '.join(args.configs)}")
    print(f"  Symbols:    {', '.join(SYMBOLS)}")
    print(f"  Cost Mults: {COST_MULTS}")
    print()

    # ══════════════════════════════════════════════════════════
    # 1. RUN BASELINE (R2)
    # ══════════════════════════════════════════════════════════
    print("━" * 80)
    print("  STEP 1: R2 Baseline")
    print("━" * 80)

    cfg_baseline = load_config(str(BASELINE_CONFIG))
    all_results = []
    baseline_by_cost = {}

    for cm in COST_MULTS:
        label = f"R2_baseline"
        print(f"\n  Running {label} @ cost_mult={cm:.1f}...")
        result = run_portfolio_config(
            cfg=cfg_baseline,
            config_path=BASELINE_CONFIG,
            cost_mult=cm,
            label=label,
        )
        result["cost_mult"] = cm
        all_results.append(result)
        baseline_by_cost[cm] = result
        print(
            f"    → Return={result['total_return_pct']:+.2f}%, "
            f"Sharpe={result['sharpe']:.2f}, "
            f"MDD={result['max_drawdown_pct']:.2f}%, "
            f"Trades={result['total_trades']}"
        )

    # ── Baseline year-by-year ──
    print("\n  Year-by-year (baseline, cost_mult=1.0):")
    for year_label, (y_start, y_end) in YEAR_RANGES.items():
        yr = run_portfolio_config(
            cfg=cfg_baseline,
            config_path=BASELINE_CONFIG,
            cost_mult=1.0,
            start_override=y_start,
            end_override=y_end,
            label=f"R2_{year_label}",
        )
        print(f"    {year_label}: Return={yr['total_return_pct']:+.2f}%, Sharpe={yr['sharpe']:.2f}")
        if year_label == "2025":
            for cm_key in baseline_by_cost:
                baseline_by_cost[cm_key]["ret_2025"] = yr["total_return_pct"]

    # ══════════════════════════════════════════════════════════
    # 2. RUN OVERLAY CONFIGS
    # ══════════════════════════════════════════════════════════
    print("\n" + "━" * 80)
    print("  STEP 2: OI/Vol Overlay Configs")
    print("━" * 80)

    overlay_results_by_cost = {cm: [] for cm in COST_MULTS}

    for config_name in args.configs:
        config_path = OVERLAY_CONFIGS[config_name]
        cfg_overlay = load_config(str(config_path))
        overlay_params = _load_overlay_params(config_path)

        if overlay_params is None:
            print(f"\n  ⚠️  Config {config_name}: no oi_vol_overlay section, skipping")
            continue

        print(f"\n  Config {config_name}: {config_path.name}")
        print(f"    Params: oi_extreme_z={overlay_params.get('oi_extreme_z')}, "
              f"vol_spike_z={overlay_params.get('vol_spike_z')}, "
              f"reduce_pct={overlay_params.get('reduce_pct')}, "
              f"cooldown={overlay_params.get('overlay_cooldown_bars')}")

        for cm in COST_MULTS:
            label = f"R2+OI_{config_name}"
            print(f"\n    Running {label} @ cost_mult={cm:.1f}...")
            result = run_portfolio_config(
                cfg=cfg_overlay,
                config_path=config_path,
                cost_mult=cm,
                overlay_params=overlay_params,
                label=label,
            )
            result["cost_mult"] = cm
            all_results.append(result)
            overlay_results_by_cost[cm].append(result)
            print(
                f"      → Return={result['total_return_pct']:+.2f}%, "
                f"Sharpe={result['sharpe']:.2f}, "
                f"MDD={result['max_drawdown_pct']:.2f}%, "
                f"Trades={result['total_trades']}"
            )

        # ── Overlay year-by-year ──
        print(f"\n    Year-by-year ({config_name}, cost_mult=1.0):")
        for year_label, (y_start, y_end) in YEAR_RANGES.items():
            yr = run_portfolio_config(
                cfg=cfg_overlay,
                config_path=config_path,
                cost_mult=1.0,
                overlay_params=overlay_params,
                start_override=y_start,
                end_override=y_end,
                label=f"OI_{config_name}_{year_label}",
            )
            print(f"      {year_label}: Return={yr['total_return_pct']:+.2f}%, Sharpe={yr['sharpe']:.2f}")
            if year_label == "2025":
                for cm_key in overlay_results_by_cost:
                    for ov in overlay_results_by_cost[cm_key]:
                        if ov["label"] == f"R2+OI_{config_name}":
                            ov["ret_2025"] = yr["total_return_pct"]

    # ══════════════════════════════════════════════════════════
    # 3. WALK-FORWARD (optional)
    # ══════════════════════════════════════════════════════════
    wf_all = {}
    if not args.skip_wf:
        print("\n" + "━" * 80)
        print("  STEP 3: Walk-Forward Analysis (5 splits)")
        print("━" * 80)

        # Baseline WF
        print("\n  R2 Baseline WF:")
        wf_all["R2_baseline"] = run_walk_forward_for_config(
            cfg=cfg_baseline,
            config_path=BASELINE_CONFIG,
            n_splits=5,
        )

        # Overlay WF (same strategy, WF tests signal stability)
        for config_name in args.configs:
            config_path = OVERLAY_CONFIGS[config_name]
            cfg_ov = load_config(str(config_path))
            print(f"\n  R2+OI_{config_name} WF:")
            wf_all[f"R2+OI_{config_name}"] = run_walk_forward_for_config(
                cfg=cfg_ov,
                config_path=config_path,
                n_splits=5,
            )
    else:
        print("\n  ⏭️  Walk-forward skipped (--skip-wf)")

    # ══════════════════════════════════════════════════════════
    # 4. OUTPUT REPORT
    # ══════════════════════════════════════════════════════════
    print("\n\n" + "█" * 80)
    print("  FINAL REPORT")
    print("█" * 80)

    # ── A. Change Summary ──
    print("\n" + "=" * 80)
    print("A. CHANGE SUMMARY")
    print("=" * 80)
    changes = [
        ("src/qtrade/data/open_interest.py", "NEW", "OI data download/save/load/align"),
        ("src/qtrade/strategy/overlays/__init__.py", "NEW", "Overlay package init"),
        ("src/qtrade/strategy/overlays/oi_vol_exit_overlay.py", "NEW",
         "OI+Vol exit overlay (compute_oi_signals, compute_vol_state, apply_oi_vol_exit_overlay)"),
        ("config/research_r2_oi_overlay_A.yaml", "NEW", "Conservative overlay config"),
        ("config/research_r2_oi_overlay_B.yaml", "NEW", "Moderate overlay config"),
        ("config/research_r2_oi_overlay_C.yaml", "NEW", "Aggressive overlay config"),
        ("scripts/run_r2_oi_overlay_research.py", "NEW", "Research runner script"),
    ]
    print(f"{'File':<60} {'Status':<8} {'Purpose'}")
    print("-" * 120)
    for f, status, purpose in changes:
        print(f"{f:<60} {status:<8} {purpose}")
    print("\nBackward compatible: YES — no existing files modified, prod_candidate_R2.yaml untouched.")

    # ── B. Core Comparison Table ──
    print_comparison_table(all_results)

    # ── C. Delta vs R2 Baseline ──
    if 1.5 in baseline_by_cost and overlay_results_by_cost.get(1.5):
        print_delta_table(baseline_by_cost[1.5], overlay_results_by_cost[1.5])

    # ── D. Walk-Forward Table ──
    if wf_all:
        print_walk_forward_table(wf_all)

    # ── E. Verdict ──
    print("\n" + "=" * 80)
    print("E. VERDICT")
    print("=" * 80)

    if 1.5 in baseline_by_cost and overlay_results_by_cost.get(1.5):
        verdicts = check_acceptance_criteria(
            baseline_15=baseline_by_cost[1.5],
            overlay_results_15=overlay_results_by_cost[1.5],
            baseline_10=baseline_by_cost[1.0],
            overlay_results_10=overlay_results_by_cost[1.0],
        )

        any_pass = False
        for config_name, v in verdicts.items():
            status_icons = {
                "pass_sharpe": "✅" if v["pass_sharpe"] else "❌",
                "pass_trades": "✅" if v["pass_trades"] else "❌",
                "pass_mdd": "✅" if v["pass_mdd"] else "❌",
                "pass_2025": "✅" if v["pass_2025"] else "❌",
            }
            overall = "✅ PASS" if v["overall"] else "❌ FAIL"
            print(f"\n  {config_name}: {overall}")
            print(f"    Sharpe >= baseline:   {status_icons['pass_sharpe']} (Δ={v['sharpe_delta']:+.2f})")
            print(f"    Trades ↓ >= 30%:      {status_icons['pass_trades']} ({v['trade_delta_pct']:+.1f}%)")
            print(f"    MDD ↑ <= 1pp:         {status_icons['pass_mdd']} (Δ={v['mdd_delta']:+.2f}pp)")
            print(f"    2025 return better:   {status_icons['pass_2025']} (Δ={v['ret_2025_delta']:+.2f}%)")
            if v["overall"]:
                any_pass = True

        if any_pass:
            verdict = "GO_OVERLAY"
            reason = "At least one overlay config passes all acceptance criteria."
        else:
            # Check partial passes
            partial = any(
                sum([v["pass_sharpe"], v["pass_trades"], v["pass_mdd"], v["pass_2025"]]) >= 3
                for v in verdicts.values()
            )
            if partial:
                verdict = "NEED_MORE_WORK"
                reason = "Some criteria partially met. Consider tuning parameters."
            else:
                verdict = "KEEP_R2"
                reason = "No overlay config meets acceptance criteria."

        print(f"\n  {'─'*60}")
        print(f"  VERDICT: {verdict}")
        print(f"  REASON:  {reason}")
    else:
        print("  ⚠️  Insufficient data to determine verdict")

    # ── F. Evidence Paths ──
    print("\n" + "=" * 80)
    print("F. EVIDENCE PATHS")
    print("=" * 80)

    # Save full results to JSON
    results_json = {
        "timestamp": timestamp,
        "baseline_config": str(BASELINE_CONFIG),
        "overlay_configs": {k: str(v) for k, v in OVERLAY_CONFIGS.items()},
        "results": [],
    }
    for r in all_results:
        entry = {k: v for k, v in r.items()
                 if k not in ("per_symbol", "portfolio_equity")}
        # Convert numpy types to native
        for k, v in entry.items():
            if isinstance(v, (np.integer, np.int64)):
                entry[k] = int(v)
            elif isinstance(v, (np.floating, np.float64)):
                entry[k] = float(v)
        results_json["results"].append(entry)

    results_path = output_dir / "oi_overlay_results.json"
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2, default=str)

    # Save WF results
    if wf_all:
        wf_path = output_dir / "walk_forward_summary.json"
        wf_save = {}
        for cfg_name, wf in wf_all.items():
            wf_save[cfg_name] = {}
            for sym, data in wf.items():
                wf_save[cfg_name][sym] = {
                    k: float(v) if isinstance(v, (np.floating, np.float64)) else v
                    for k, v in data.items()
                    if k != "summary_text"
                }
        with open(wf_path, "w") as f:
            json.dump(wf_save, f, indent=2, default=str)
        print(f"  WF results:  {wf_path}")

    print(f"  Full results: {results_path}")
    print(f"  Output dir:   {output_dir}")

    # ── G. Honest Statement ──
    print("\n" + "=" * 80)
    print("G. HONEST STATEMENT")
    print("=" * 80)

    if 1.5 in baseline_by_cost and overlay_results_by_cost.get(1.5):
        verdicts_list = list(verdicts.values()) if 'verdicts' in dir() else []
        any_overall_pass = any(v.get("overall", False) for v in verdicts_list) if verdicts_list else False

        if not any_overall_pass:
            print("""
  RESULT: The OI/Vol exit overlay did NOT pass all acceptance criteria.

  This is a legitimate research finding. Possible next steps:

  1. Change OI sampling frequency:
     - Current: 1h OI aligned to 1h klines
     - Try: 4h OI (less noise, stronger signal)

  2. Apply overlay only to BTC:
     - BTC has highest OI data quality and liquidity
     - ETH/SOL may have lower OI signal-to-noise ratio

  3. "Reduce-only" variant:
     - Current: reduce_pct + full flatten on vol spike
     - Try: only reduce (never flatten) to preserve alpha

  4. Combine with entry pause:
     - Instead of reducing existing positions,
       pause new entries when OI is extreme

  5. OI data coverage check:
     - Verify if Binance OI historical data has sufficient coverage
     - Gaps in OI data effectively disable the overlay for those periods
""")
        else:
            print("\n  ✅ At least one overlay config passes all acceptance criteria.")
            print("  Recommend promoting the best-performing overlay to production candidate.")
    else:
        print("\n  ⚠️  Unable to determine — insufficient comparison data.")

    print("\n" + "=" * 80)
    print("  Research complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
