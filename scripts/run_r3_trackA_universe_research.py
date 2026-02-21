#!/usr/bin/env python3
"""
R3 Track A — Expanded Universe Research Runner
=================================================
Validates R3A micro-accel overlay on 19-symbol universe.

Research matrix:
    1. R2.1 baseline (19 symbols, vol-parity, 3 cost mults)
    2. Config B/C with micro accel overlay (2 × 3 = 6 runs)
    3. Year-by-year breakdown
    4. Walk-forward analysis (5 splits)
    5. +1 bar delay stress test
    6. Truncation invariance test
    7. Cost attribution (turnover/fee/slip/funding)
    8. Acceptance criteria check

Usage:
    cd /path/to/quant-binance-spot
    PYTHONPATH=src python scripts/run_r3_trackA_universe_research.py
    PYTHONPATH=src python scripts/run_r3_trackA_universe_research.py --skip-wf
    PYTHONPATH=src python scripts/run_r3_trackA_universe_research.py --configs C
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import vectorbt as vbt

from qtrade.config import load_config
from qtrade.backtest.run_backtest import (
    BacktestResult,
    _bps_to_pct,
    to_vbt_direction,
    clip_positions_by_direction,
    _apply_date_filter,
)
from qtrade.backtest.costs import (
    compute_funding_costs,
    adjust_equity_for_funding,
    compute_adjusted_stats,
)
from qtrade.strategy import get_strategy
from qtrade.strategy.base import StrategyContext
from qtrade.data.storage import load_klines
from qtrade.data.quality import clean_data
from qtrade.data.funding_rate import (
    load_funding_rates,
    get_funding_rate_path,
    align_funding_to_klines,
)
from qtrade.strategy.overlays.oi_vol_exit_overlay import (
    apply_overlay_by_mode,
    compute_flip_count,
)
from qtrade.strategy.overlays.microstructure_accel_overlay import (
    apply_full_micro_accel_overlay,
    load_multi_tf_klines,
)
from qtrade.validation.walk_forward import walk_forward_analysis, walk_forward_summary

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
warnings.filterwarnings("ignore", category=UserWarning)

# ══════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).parent.parent
BASELINE_CONFIG = PROJECT_ROOT / "config" / "research_r2_1_universe_baseline.yaml"
OVERLAY_CONFIGS = {
    "B": PROJECT_ROOT / "config" / "research_r3_trackA_B_universe.yaml",
    "C": PROJECT_ROOT / "config" / "research_r3_trackA_C_universe.yaml",
}

# 19-symbol expanded universe (MATICUSDT excluded — data ends 2024-09-11)
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
    "LTCUSDT", "NEARUSDT", "APTUSDT", "ARBUSDT", "OPUSDT",
    "FILUSDT", "ATOMUSDT", "UNIUSDT", "AAVEUSDT",
]

# Vol-parity weights (computed by build_universe.py, excluding MATICUSDT)
WEIGHTS = {
    "BTCUSDT": 0.0722, "ETHUSDT": 0.0538, "SOLUSDT": 0.0509,
    "BNBUSDT": 0.0707, "XRPUSDT": 0.0491, "DOGEUSDT": 0.0512,
    "ADAUSDT": 0.0511, "AVAXUSDT": 0.0545, "LINKUSDT": 0.0538,
    "DOTUSDT": 0.0540, "LTCUSDT": 0.0605, "NEARUSDT": 0.0495,
    "APTUSDT": 0.0483, "ARBUSDT": 0.0495, "OPUSDT": 0.0426,
    "FILUSDT": 0.0489, "ATOMUSDT": 0.0523, "UNIUSDT": 0.0394,
    "AAVEUSDT": 0.0477,
}

COST_MULTS = [1.0, 1.5, 2.0]
YEAR_RANGES = {
    "2022": ("2022-01-01", "2022-12-31"),
    "2023": ("2023-01-01", "2023-12-31"),
    "2024": ("2024-01-01", "2024-12-31"),
    "2025": ("2025-01-01", "2025-12-31"),
    "2026 YTD": ("2026-01-01", None),
}

# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════

def _load_ensemble_strategy(config_path: str | Path, symbol: str) -> tuple | None:
    """Get per-symbol ensemble strategy from config."""
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    ens = raw.get("ensemble")
    if ens and ens.get("enabled", False):
        strategies = ens.get("strategies", {})
        if symbol in strategies:
            s = strategies[symbol]
            return s["name"], s.get("params", {})
    return None


def _load_micro_accel_params(config_path: str | Path) -> dict | None:
    """Load micro_accel_overlay params from config."""
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    micro = raw.get("strategy", {}).get("micro_accel_overlay")
    if micro and micro.get("enabled", False):
        return micro.get("params", {})
    return None


def _load_vol_overlay_params(config_path: str | Path) -> dict | None:
    """Load vol overlay params from config."""
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    overlay = raw.get("strategy", {}).get("overlay")
    if overlay and overlay.get("enabled", False):
        return overlay
    return None


# ══════════════════════════════════════════════════════════════
# Core: Single-symbol backtest with optional micro accel overlay
# ══════════════════════════════════════════════════════════════

def run_single_symbol(
    symbol: str,
    cfg,
    config_path: str | Path,
    cost_mult: float = 1.0,
    micro_accel_params: dict | None = None,
    start_override: str | None = None,
    end_override: str | None = None,
    extra_signal_delay: int = 0,
) -> dict | None:
    """
    Run single-symbol backtest with vol overlay + optional micro accel overlay.

    Returns dict with metrics or None on failure.
    """
    market_type = cfg.market_type_str
    data_path = (
        cfg.data_dir / "binance" / market_type
        / cfg.market.interval / f"{symbol}.parquet"
    )
    if not data_path.exists():
        return None

    # Load 1h data
    df = load_klines(data_path)
    df = clean_data(df, fill_method="forward", remove_outliers=False, remove_duplicates=True)

    # Load multi-TF data for micro features
    multi_tf = load_multi_tf_klines(cfg.data_dir, symbol, market_type)
    df_5m = multi_tf.get("5m")
    df_15m = multi_tf.get("15m")

    # Resolve strategy (ensemble routing)
    ensemble_override = _load_ensemble_strategy(config_path, symbol)
    if ensemble_override:
        strategy_name, strategy_params = ensemble_override
    else:
        strategy_name = cfg.strategy.name
        strategy_params = cfg.strategy.get_params(symbol)

    # Build context
    total_delay = 1 + extra_signal_delay
    ctx = StrategyContext(
        symbol=symbol,
        interval=cfg.market.interval,
        market_type=market_type,
        direction=cfg.direction,
        signal_delay=total_delay,
    )

    # Generate base positions
    strategy_func = get_strategy(strategy_name)
    pos = strategy_func(df, ctx, strategy_params)

    # Apply vol overlay (R2.1)
    vol_overlay = _load_vol_overlay_params(config_path)
    if vol_overlay and vol_overlay.get("enabled", False):
        overlay_mode = vol_overlay.get("mode", "vol_pause")
        overlay_params = vol_overlay.get("params", {})
        pos = apply_overlay_by_mode(
            position=pos, price_df=df, oi_series=None,
            params=overlay_params, mode=overlay_mode,
        )

    # Apply micro accel overlay (R3 Track A)
    if micro_accel_params is not None:
        oi_series = None
        try:
            from qtrade.data.open_interest import get_oi_path, load_open_interest, align_oi_to_klines
            for prov in ["merged", "binance"]:
                oi_path = get_oi_path(cfg.data_dir, symbol, prov)
                oi_df = load_open_interest(oi_path)
                if oi_df is not None and not oi_df.empty:
                    oi_series = align_oi_to_klines(oi_df, df.index, max_ffill_bars=2)
                    break
        except Exception:
            pass

        pos = apply_full_micro_accel_overlay(
            base_position=pos, df_1h=df,
            df_5m=df_5m, df_15m=df_15m,
            oi_series=oi_series, params=micro_accel_params,
        )

    # Direction clip
    pos = clip_positions_by_direction(pos, market_type, cfg.direction)

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
        close=close, size=pos, size_type="targetpercent",
        price=open_, fees=fee, slippage=slippage,
        init_cash=initial_cash, freq=cfg.market.interval,
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
            funding_df, df.index, default_rate_8h=fr.default_rate_8h,
        )
        leverage = cfg.futures.leverage if cfg.futures else 1
        funding_cost = compute_funding_costs(
            pos=pos, equity=equity,
            funding_rates=funding_rates, leverage=leverage,
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

    n_bars = len(df)
    years = n_bars / (365.25 * 24)
    total_ret = total_return_pct / 100.0
    cagr = ((1 + total_ret) ** (1 / max(years, 0.01)) - 1) * 100 if years > 0 else 0
    calmar = cagr / max_dd if max_dd > 0.01 else 0
    flips = compute_flip_count(pos)

    # Cost attribution
    pos_diff = pos.diff().abs().fillna(0)
    turnover = pos_diff.sum()
    fee_cost = turnover * fee
    slip_cost = turnover * slippage

    return {
        "symbol": symbol,
        "total_return_pct": total_return_pct,
        "sharpe": sharpe,
        "max_drawdown_pct": max_dd,
        "total_trades": int(total_trades),
        "cagr": cagr,
        "calmar": calmar,
        "flips": flips,
        "equity": eq,
        "pos": pos,
        "turnover": turnover,
        "fee_cost": fee_cost,
        "slip_cost": slip_cost,
    }


def run_portfolio_config(
    cfg,
    config_path: str | Path,
    cost_mult: float = 1.0,
    micro_accel_params: dict | None = None,
    start_override: str | None = None,
    end_override: str | None = None,
    label: str = "",
    extra_signal_delay: int = 0,
    symbols: list[str] | None = None,
    weights: dict[str, float] | None = None,
) -> dict:
    """Run portfolio backtest across expanded universe."""
    initial_cash = cfg.backtest.initial_cash
    per_symbol = {}
    use_symbols = symbols or SYMBOLS
    use_weights = weights or WEIGHTS

    for symbol in use_symbols:
        try:
            res = run_single_symbol(
                symbol=symbol, cfg=cfg, config_path=config_path,
                cost_mult=cost_mult, micro_accel_params=micro_accel_params,
                start_override=start_override, end_override=end_override,
                extra_signal_delay=extra_signal_delay,
            )
            if res is not None:
                per_symbol[symbol] = res
        except Exception as e:
            logger.warning(f"    ⚠️  {symbol} failed: {e}")

    if not per_symbol:
        return {
            "label": label, "sharpe": 0, "total_return_pct": 0,
            "max_drawdown_pct": 0, "cagr": 0, "calmar": 0,
            "total_trades": 0, "flips": 0, "per_symbol": {},
            "turnover": 0, "fee_cost": 0, "slip_cost": 0,
        }

    # Build portfolio equity (handle different date ranges per symbol)
    active_symbols = list(per_symbol.keys())
    raw_weights = np.array([use_weights.get(s, 1.0 / len(use_symbols)) for s in active_symbols])
    active_weights = raw_weights / raw_weights.sum()  # renormalize for active symbols

    equity_curves = {s: per_symbol[s]["equity"] for s in active_symbols}

    # Find common date range
    min_start = max(eq.index[0] for eq in equity_curves.values())
    max_end = min(eq.index[-1] for eq in equity_curves.values())

    for s in active_symbols:
        equity_curves[s] = equity_curves[s].loc[min_start:max_end]

    # Normalized portfolio equity
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

    # Portfolio stats
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
    total_flips = sum(per_symbol[s]["flips"] for s in active_symbols)
    total_turnover = sum(per_symbol[s].get("turnover", 0) for s in active_symbols)
    total_fee = sum(per_symbol[s].get("fee_cost", 0) for s in active_symbols)
    total_slip = sum(per_symbol[s].get("slip_cost", 0) for s in active_symbols)

    return {
        "label": label,
        "total_return_pct": round(total_return, 2),
        "cagr": round(cagr, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "calmar": round(calmar, 2),
        "total_trades": total_trades,
        "flips": total_flips,
        "n_symbols": len(active_symbols),
        "per_symbol": per_symbol,
        "portfolio_equity": portfolio_equity,
        "turnover": round(total_turnover, 4),
        "fee_cost": round(total_fee, 4),
        "slip_cost": round(total_slip, 4),
    }


# ══════════════════════════════════════════════════════════════
# Walk-Forward (overlay-inclusive)
# ══════════════════════════════════════════════════════════════

def run_walk_forward_for_config(
    cfg,
    config_path: str | Path,
    n_splits: int = 5,
    symbols: list[str] | None = None,
) -> dict:
    """Run walk-forward for each symbol."""
    results = {}
    use_symbols = symbols or SYMBOLS

    for symbol in use_symbols:
        market_type = cfg.market_type_str
        data_path = (
            cfg.data_dir / "binance" / market_type
            / cfg.market.interval / f"{symbol}.parquet"
        )
        if not data_path.exists():
            results[symbol] = {"avg_test_sharpe": 0, "oos_positive_pct": 0}
            continue

        bt_cfg = cfg.to_backtest_dict(symbol=symbol)
        ensemble_override = _load_ensemble_strategy(config_path, symbol)
        if ensemble_override:
            sym_name, sym_params = ensemble_override
            bt_cfg["strategy_name"] = sym_name
            bt_cfg["strategy_params"] = sym_params

        try:
            wf_df = walk_forward_analysis(
                symbol=symbol, data_path=data_path,
                cfg=bt_cfg, n_splits=n_splits,
                data_dir=cfg.data_dir,
            )
            if wf_df.empty:
                results[symbol] = {"avg_test_sharpe": 0, "oos_positive_pct": 0}
            else:
                summary = walk_forward_summary(wf_df)
                results[symbol] = summary
        except Exception as e:
            logger.warning(f"    ⚠️  WF {symbol} failed: {e}")
            results[symbol] = {"avg_test_sharpe": 0, "oos_positive_pct": 0}

    return results


# ══════════════════════════════════════════════════════════════
# Robustness: Truncation Invariance
# ══════════════════════════════════════════════════════════════

def test_truncation_invariance(
    cfg, config_path, micro_accel_params=None, symbols=None,
) -> dict:
    """Truncation invariance: full vs 80% tail."""
    results = {}
    use_symbols = symbols or SYMBOLS

    for symbol in use_symbols:
        full_res = run_single_symbol(
            symbol=symbol, cfg=cfg, config_path=config_path,
            cost_mult=1.0, micro_accel_params=micro_accel_params,
        )
        if full_res is None:
            results[symbol] = {"pass": False, "reason": "no data"}
            continue

        market_type = cfg.market_type_str
        data_path = (
            cfg.data_dir / "binance" / market_type
            / cfg.market.interval / f"{symbol}.parquet"
        )
        df = load_klines(data_path)
        n = len(df)
        cutoff_idx = int(n * 0.20)
        cutoff_date = df.index[cutoff_idx].strftime("%Y-%m-%d")

        trunc_res = run_single_symbol(
            symbol=symbol, cfg=cfg, config_path=config_path,
            cost_mult=1.0, micro_accel_params=micro_accel_params,
            start_override=cutoff_date,
        )
        if trunc_res is None:
            results[symbol] = {"pass": False, "reason": "truncated too short"}
            continue

        full_sr = full_res["sharpe"]
        trunc_sr = trunc_res["sharpe"]
        denom = max(abs(full_sr), 0.01)
        relative_diff = abs(full_sr - trunc_sr) / denom
        passed = relative_diff < 0.30

        results[symbol] = {
            "pass": passed,
            "full_sharpe": round(full_sr, 3),
            "trunc_sharpe": round(trunc_sr, 3),
            "relative_diff": round(relative_diff, 3),
        }

    return results


# ══════════════════════════════════════════════════════════════
# Robustness: +1 Bar Delay Stress
# ══════════════════════════════════════════════════════════════

def test_delay_stress(
    cfg, config_path, micro_accel_params=None,
) -> dict:
    """Delay stress: normal vs +1 bar delay."""
    normal = run_portfolio_config(
        cfg=cfg, config_path=config_path, cost_mult=1.0,
        micro_accel_params=micro_accel_params, label="normal",
        extra_signal_delay=0,
    )
    delayed = run_portfolio_config(
        cfg=cfg, config_path=config_path, cost_mult=1.0,
        micro_accel_params=micro_accel_params, label="delayed",
        extra_signal_delay=1,
    )
    normal_sr = normal["sharpe"]
    delayed_sr = delayed["sharpe"]
    denom = max(abs(normal_sr), 0.01)
    sharpe_drop_pct = (normal_sr - delayed_sr) / denom * 100
    passed = sharpe_drop_pct <= 30.0

    return {
        "normal_sharpe": round(normal_sr, 3),
        "delayed_sharpe": round(delayed_sr, 3),
        "sharpe_drop_pct": round(sharpe_drop_pct, 1),
        "pass": passed,
    }


# ══════════════════════════════════════════════════════════════
# Reporting
# ══════════════════════════════════════════════════════════════

def print_comparison_table(all_results: list[dict]):
    """Core Comparison Table."""
    print("\n" + "=" * 130)
    print("B. CORE COMPARISON TABLE")
    print("=" * 130)
    header = (
        f"{'Config':<24} {'Cost':>6} {'#Sym':>5} {'TotRet%':>10} {'CAGR%':>8} "
        f"{'Sharpe':>8} {'MaxDD%':>8} {'Calmar':>8} {'Trades':>8} {'Flips':>8}"
    )
    print(header)
    print("-" * 130)
    for r in all_results:
        print(
            f"{r['label']:<24} "
            f"{r.get('cost_mult', 1.0):>6.1f} "
            f"{r.get('n_symbols', '?'):>5} "
            f"{r['total_return_pct']:>10.2f} "
            f"{r['cagr']:>8.2f} "
            f"{r['sharpe']:>8.2f} "
            f"{r['max_drawdown_pct']:>8.2f} "
            f"{r['calmar']:>8.2f} "
            f"{r['total_trades']:>8d} "
            f"{r.get('flips', 0):>8d}"
        )


def print_delta_table(baseline_results: dict, overlay_results: list[dict]):
    """Delta vs R2.1 Baseline."""
    print("\n" + "=" * 110)
    print("C. DELTA vs R2.1 UNIVERSE BASELINE (cost_mult=1.0)")
    print("=" * 110)
    header = (
        f"{'Config':<24} {'ΔCAGR':>10} {'ΔSharpe':>10} {'ΔMaxDD':>10} "
        f"{'ΔTrades':>10} {'ΔFlips':>10} {'Verdict':>12}"
    )
    print(header)
    print("-" * 110)

    bl = baseline_results
    for r in overlay_results:
        d_cagr = r["cagr"] - bl["cagr"]
        d_sharpe = r["sharpe"] - bl["sharpe"]
        d_mdd = r["max_drawdown_pct"] - bl["max_drawdown_pct"]
        d_trades = r["total_trades"] - bl["total_trades"]
        d_flips = r.get("flips", 0) - bl.get("flips", 0)

        if d_sharpe >= -0.05 and d_cagr > 0:
            verdict = "✅ GO"
        elif d_sharpe >= -0.10:
            verdict = "🟡 MAYBE"
        else:
            verdict = "❌ FAIL"

        print(
            f"{r['label']:<24} "
            f"{d_cagr:>+10.2f} "
            f"{d_sharpe:>+10.2f} "
            f"{d_mdd:>+10.2f} "
            f"{d_trades:>+10d} "
            f"{d_flips:>+10d} "
            f"{verdict:>12}"
        )


def print_per_symbol_table(per_symbol: dict, label: str):
    """Per-symbol breakdown."""
    print(f"\n  {label} — Per-Symbol:")
    print(f"  {'Symbol':<12} {'Return%':>10} {'CAGR%':>8} {'Sharpe':>8} {'MDD%':>8} {'Trades':>8} {'Flips':>8}")
    print("  " + "-" * 75)
    for sym in SYMBOLS:
        if sym in per_symbol:
            r = per_symbol[sym]
            print(
                f"  {sym:<12} "
                f"{r['total_return_pct']:>10.2f} "
                f"{r['cagr']:>8.2f} "
                f"{r['sharpe']:>8.2f} "
                f"{r['max_drawdown_pct']:>8.2f} "
                f"{r['total_trades']:>8d} "
                f"{r['flips']:>8d}"
            )


def print_cost_attribution(all_results: list[dict]):
    """Cost Attribution table."""
    print("\n" + "=" * 100)
    print("E. COST ATTRIBUTION")
    print("=" * 100)
    header = (
        f"{'Config':<24} {'Cost':>6} {'Turnover':>12} {'Fee Cost':>12} "
        f"{'Slip Cost':>12} {'CAGR%':>8} {'Sharpe':>8}"
    )
    print(header)
    print("-" * 100)
    for r in all_results:
        print(
            f"{r['label']:<24} "
            f"{r.get('cost_mult', 1.0):>6.1f} "
            f"{r.get('turnover', 0):>12.2f} "
            f"{r.get('fee_cost', 0):>12.4f} "
            f"{r.get('slip_cost', 0):>12.4f} "
            f"{r['cagr']:>8.2f} "
            f"{r['sharpe']:>8.2f}"
        )


def print_robustness_table(robustness: dict, config_names: list[str]):
    """Robustness Table."""
    print("\n" + "=" * 100)
    print("D. ROBUSTNESS TABLE")
    print("=" * 100)
    cols = " ".join(f"{'Config '+n:>15}" for n in config_names)
    header = f"{'Test':<30} {cols} {'Pass/Fail':>12}"
    print(header)
    print("-" * 100)

    for test_name, test_data in robustness.items():
        vals = []
        all_pass = True
        for cfg_name in config_names:
            if cfg_name in test_data:
                d = test_data[cfg_name]
                if isinstance(d, dict) and "pass" in d:
                    passed = d["pass"]
                    val_str = "✅" if passed else "❌"
                    if not passed:
                        all_pass = False
                    if "sharpe_drop_pct" in d:
                        val_str += f" ({d['sharpe_drop_pct']:+.0f}%)"
                    elif "relative_diff" in d:
                        val_str += f" ({d['relative_diff']:.2f})"
                elif isinstance(d, dict):
                    n_pass = sum(1 for v in d.values() if isinstance(v, dict) and v.get("pass", False))
                    n_total = len([v for v in d.values() if isinstance(v, dict)])
                    val_str = f"{n_pass}/{n_total}"
                    if n_pass < n_total:
                        all_pass = False
                else:
                    val_str = str(d)
                vals.append(val_str)
            else:
                vals.append("N/A")
                all_pass = False

        vals_str = " ".join(f"{v:>15}" for v in vals)
        overall = "✅ PASS" if all_pass else "❌ FAIL"
        print(f"{test_name:<30} {vals_str} {overall:>12}")


def print_walk_forward_table(wf_results: dict, symbols: list[str]):
    """Walk-Forward Table."""
    print("\n" + "=" * 100)
    print("F. WALK-FORWARD TABLE")
    print("=" * 100)

    for config_name, wf in wf_results.items():
        oos_pos = sum(
            1 for sym in symbols
            if wf.get(sym, {}).get("avg_test_sharpe", 0) > 0
        )
        n_total = sum(1 for sym in symbols if sym in wf)
        print(f"\n  {config_name}: OOS+/N = {oos_pos}/{n_total}")

        # Show top-level per-symbol OOS Sharpe
        for sym in symbols:
            sym_data = wf.get(sym, {})
            sr = sym_data.get("avg_test_sharpe", 0)
            icon = "✅" if sr > 0 else "❌"
            print(f"    {sym:<12} OOS Sharpe={sr:>8.2f} {icon}")


# ══════════════════════════════════════════════════════════════
# Acceptance Criteria (Expanded Universe)
# ══════════════════════════════════════════════════════════════

def check_acceptance_criteria(
    baseline_10: dict,
    baseline_15: dict,
    overlay_results_10: list[dict],
    overlay_results_15: list[dict],
    overlay_results_20: list[dict],
    delay_results: dict,
    wf_results: dict | None,
) -> dict:
    """Check expanded universe acceptance criteria."""
    verdicts = {}

    for i, ov_10 in enumerate(overlay_results_10):
        name = ov_10["label"]
        ov_15 = overlay_results_15[i] if i < len(overlay_results_15) else ov_10
        ov_20 = overlay_results_20[i] if i < len(overlay_results_20) else ov_10

        # 1. cost 1.5x Sharpe >= 1.2
        pass_sharpe_15 = ov_15["sharpe"] >= 1.2

        # 2. MaxDD <= 40%
        pass_mdd = ov_10["max_drawdown_pct"] <= 40.0

        # 3. WF OOS+/5 >= 4/5 (portfolio level)
        pass_wf = True
        wf_key = name
        if wf_results and wf_key in wf_results:
            wf = wf_results[wf_key]
            oos_pos = sum(
                1 for sym in SYMBOLS
                if wf.get(sym, {}).get("avg_test_sharpe", 0) > 0
            )
            n_total = sum(1 for sym in SYMBOLS if sym in wf)
            # 4/5 ratio for portfolio
            pass_wf = (oos_pos / max(n_total, 1)) >= 0.7  # ~70% symbols OOS+

        # 4. +1 bar delay Sharpe drop <= 30%
        delay_data = delay_results.get(name, {})
        pass_delay = delay_data.get("pass", True)

        # 5. Sharpe >= baseline - 0.05
        pass_sharpe_base = ov_10["sharpe"] >= baseline_10["sharpe"] - 0.05

        # 6. cost 2.0x positive return
        pass_cost_20 = ov_20["total_return_pct"] > 0

        # 7. CAGR improvement
        cagr_improvement = (
            (ov_10["cagr"] - baseline_10["cagr"])
            / max(abs(baseline_10["cagr"]), 0.01) * 100
        )
        mdd_delta = ov_10["max_drawdown_pct"] - baseline_10["max_drawdown_pct"]

        overall = (
            pass_sharpe_15 and pass_mdd and pass_wf
            and pass_delay and pass_sharpe_base and pass_cost_20
        )

        verdicts[name] = {
            "pass_sharpe_15x": pass_sharpe_15,
            "pass_mdd_40": pass_mdd,
            "pass_wf": pass_wf,
            "pass_delay": pass_delay,
            "pass_sharpe_base": pass_sharpe_base,
            "pass_cost_20": pass_cost_20,
            "overall": overall,
            "sharpe_10": round(ov_10["sharpe"], 3),
            "sharpe_15": round(ov_15["sharpe"], 3),
            "sharpe_delta": round(ov_10["sharpe"] - baseline_10["sharpe"], 3),
            "mdd_delta": round(mdd_delta, 2),
            "cagr_delta": round(ov_10["cagr"] - baseline_10["cagr"], 2),
            "cagr_improvement_pct": round(cagr_improvement, 1),
        }

    return verdicts


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="R3 Track A — Expanded Universe Research")
    parser.add_argument("--skip-wf", action="store_true", help="Skip walk-forward")
    parser.add_argument("--skip-robustness", action="store_true", help="Skip robustness tests")
    parser.add_argument("--configs", nargs="+", default=["B", "C"],
                        choices=["B", "C"], help="Overlay configs to run")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT / "reports" / "r3_trackA_universe_research" / timestamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("█" * 80)
    print("  R3 Track A — Expanded Universe Research")
    print("█" * 80)
    print(f"  Timestamp:  {timestamp}")
    print(f"  Output Dir: {output_dir}")
    print(f"  Baseline:   {BASELINE_CONFIG.name}")
    print(f"  Configs:    {', '.join(args.configs)}")
    print(f"  Symbols:    {len(SYMBOLS)} coins")
    print(f"  Cost Mults: {COST_MULTS}")
    print(f"  Weights:    vol_parity (min 3%, max 20%)")
    print()

    # ══════════════════════════════════════════════════════════
    # STEP 1: R2.1 UNIVERSE BASELINE
    # ══════════════════════════════════════════════════════════
    print("━" * 80)
    print("  STEP 1: R2.1 Universe Baseline (vol_only, 19 symbols)")
    print("━" * 80)

    cfg_baseline = load_config(str(BASELINE_CONFIG))
    all_results = []
    baseline_by_cost = {}

    for cm in COST_MULTS:
        label = "R2.1_universe"
        print(f"\n  Running {label} @ cost_mult={cm:.1f}...")
        result = run_portfolio_config(
            cfg=cfg_baseline, config_path=BASELINE_CONFIG,
            cost_mult=cm, label=label,
        )
        result["cost_mult"] = cm
        all_results.append(result)
        baseline_by_cost[cm] = result
        print(
            f"    → N={result.get('n_symbols', '?')}, "
            f"Return={result['total_return_pct']:+.2f}%, "
            f"Sharpe={result['sharpe']:.2f}, "
            f"MDD={result['max_drawdown_pct']:.2f}%, "
            f"CAGR={result['cagr']:.2f}%"
        )

    # Per-symbol breakdown for baseline
    if baseline_by_cost.get(1.0):
        print_per_symbol_table(baseline_by_cost[1.0].get("per_symbol", {}), "R2.1 Universe Baseline")

    # Yearly
    print("\n  Year-by-year (baseline, cost_mult=1.0):")
    baseline_yearly = {}
    for year_label, (y_start, y_end) in YEAR_RANGES.items():
        yr = run_portfolio_config(
            cfg=cfg_baseline, config_path=BASELINE_CONFIG,
            cost_mult=1.0, start_override=y_start, end_override=y_end,
            label=f"R2.1_{year_label}",
        )
        baseline_yearly[year_label] = yr
        print(f"    {year_label}: Return={yr['total_return_pct']:+.2f}%, Sharpe={yr['sharpe']:.2f}")

    # ══════════════════════════════════════════════════════════
    # STEP 2: OVERLAY CONFIGS
    # ══════════════════════════════════════════════════════════
    print("\n" + "━" * 80)
    print("  STEP 2: R3 Track A Micro Accel Overlays (Universe)")
    print("━" * 80)

    overlay_results_by_cost = {cm: [] for cm in COST_MULTS}
    overlay_yearly = {}

    for config_name in args.configs:
        config_path = OVERLAY_CONFIGS[config_name]
        cfg_overlay = load_config(str(config_path))
        micro_params = _load_micro_accel_params(config_path)

        if micro_params is None:
            print(f"\n  ⚠️  Config {config_name}: no micro_accel_overlay, skipping")
            continue

        print(f"\n  Config {config_name}: {config_path.name}")
        print(f"    accel={micro_params.get('accel_threshold')}, "
              f"boost={micro_params.get('boost_pct')}, "
              f"reduce={micro_params.get('reduce_pct')}, "
              f"cd={micro_params.get('cooldown_bars')}")

        for cm in COST_MULTS:
            label = f"R3A_{config_name}_univ"
            print(f"\n    Running {label} @ cost_mult={cm:.1f}...")
            result = run_portfolio_config(
                cfg=cfg_overlay, config_path=config_path,
                cost_mult=cm, micro_accel_params=micro_params,
                label=label,
            )
            result["cost_mult"] = cm
            all_results.append(result)
            overlay_results_by_cost[cm].append(result)
            print(
                f"      → N={result.get('n_symbols', '?')}, "
                f"Return={result['total_return_pct']:+.2f}%, "
                f"Sharpe={result['sharpe']:.2f}, "
                f"MDD={result['max_drawdown_pct']:.2f}%, "
                f"CAGR={result['cagr']:.2f}%"
            )

        # Per-symbol for cost 1.0
        if overlay_results_by_cost.get(1.0):
            latest = overlay_results_by_cost[1.0][-1]
            print_per_symbol_table(latest.get("per_symbol", {}), f"R3A_{config_name} Universe")

        # Yearly
        print(f"\n    Year-by-year ({config_name}, cost_mult=1.0):")
        overlay_yearly[config_name] = {}
        for year_label, (y_start, y_end) in YEAR_RANGES.items():
            yr = run_portfolio_config(
                cfg=cfg_overlay, config_path=config_path,
                cost_mult=1.0, micro_accel_params=micro_params,
                start_override=y_start, end_override=y_end,
                label=f"R3A_{config_name}_{year_label}",
            )
            overlay_yearly[config_name][year_label] = yr
            print(f"      {year_label}: Return={yr['total_return_pct']:+.2f}%, Sharpe={yr['sharpe']:.2f}")

    # ══════════════════════════════════════════════════════════
    # STEP 3: ROBUSTNESS
    # ══════════════════════════════════════════════════════════
    robustness = {}
    delay_results = {}

    if not args.skip_robustness:
        print("\n" + "━" * 80)
        print("  STEP 3: Robustness Tests")
        print("━" * 80)

        # Truncation invariance (sample 5 key symbols for speed)
        trunc_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "LINKUSDT"]
        print(f"\n  3a. Truncation Invariance ({', '.join(trunc_symbols)}):")
        robustness["Truncation invariance"] = {}
        for config_name in args.configs:
            config_path = OVERLAY_CONFIGS[config_name]
            cfg_ov = load_config(str(config_path))
            micro_params = _load_micro_accel_params(config_path)
            trunc = test_truncation_invariance(
                cfg_ov, config_path, micro_params, symbols=trunc_symbols,
            )
            robustness["Truncation invariance"][config_name] = trunc
            n_pass = sum(1 for r in trunc.values() if isinstance(r, dict) and r.get("pass", False))
            print(f"    Config {config_name}: {n_pass}/{len(trunc)} passed")

        # +1 bar delay stress
        print("\n  3b. +1 Bar Delay Stress:")
        robustness["+1 bar delay stress"] = {}
        for config_name in args.configs:
            config_path = OVERLAY_CONFIGS[config_name]
            cfg_ov = load_config(str(config_path))
            micro_params = _load_micro_accel_params(config_path)
            delay = test_delay_stress(cfg_ov, config_path, micro_params)
            robustness["+1 bar delay stress"][config_name] = delay
            delay_results[f"R3A_{config_name}_univ"] = delay
            icon = "✅" if delay["pass"] else "❌"
            print(
                f"    {config_name}: {icon} "
                f"(normal={delay['normal_sharpe']:.2f}, delayed={delay['delayed_sharpe']:.2f}, "
                f"drop={delay['sharpe_drop_pct']:.0f}%)"
            )

        # Cost stress
        print("\n  3c. Cost Stress (2.0x positive return):")
        robustness["Cost 2.0x positive"] = {}
        for i, config_name in enumerate(args.configs):
            if i < len(overlay_results_by_cost.get(2.0, [])):
                r20 = overlay_results_by_cost[2.0][i]
                passed = r20["total_return_pct"] > 0
                robustness["Cost 2.0x positive"][config_name] = {
                    "pass": passed, "return": round(r20["total_return_pct"], 2),
                }
                icon = "✅" if passed else "❌"
                print(f"    {config_name}: {icon} (ret={r20['total_return_pct']:.2f}%)")
    else:
        print("\n  ⏭️  Robustness tests skipped (--skip-robustness)")

    # ══════════════════════════════════════════════════════════
    # STEP 4: WALK-FORWARD
    # ══════════════════════════════════════════════════════════
    wf_all = {}
    if not args.skip_wf:
        print("\n" + "━" * 80)
        print("  STEP 4: Walk-Forward Analysis (5 splits)")
        print("━" * 80)

        # Use subset for WF (core + sample)
        wf_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
                       "DOGEUSDT", "LINKUSDT", "AVAXUSDT", "LTCUSDT"]
        print(f"  WF symbols: {', '.join(wf_symbols)} ({len(wf_symbols)} coins)")

        print("\n  R2.1 Universe Baseline WF:")
        wf_all["R2.1_universe"] = run_walk_forward_for_config(
            cfg=cfg_baseline, config_path=BASELINE_CONFIG,
            n_splits=5, symbols=wf_symbols,
        )

        for config_name in args.configs:
            config_path = OVERLAY_CONFIGS[config_name]
            cfg_ov = load_config(str(config_path))
            label = f"R3A_{config_name}_univ"
            print(f"\n  {label} WF:")
            wf_all[label] = run_walk_forward_for_config(
                cfg=cfg_ov, config_path=config_path,
                n_splits=5, symbols=wf_symbols,
            )
    else:
        print("\n  ⏭️  Walk-forward skipped (--skip-wf)")

    # ══════════════════════════════════════════════════════════
    # FINAL REPORT
    # ══════════════════════════════════════════════════════════
    print("\n\n" + "█" * 80)
    print("  FINAL REPORT — R3 Track A Expanded Universe")
    print("█" * 80)

    # ── A. Change Summary ──
    print("\n" + "=" * 80)
    print("A. CHANGE SUMMARY")
    print("=" * 80)
    changes = [
        ("scripts/build_universe.py", "NEW", "Universe selection (20 candidates → 19 selected)"),
        ("config/research_r2_1_universe_baseline.yaml", "NEW", "R2.1 baseline, 19 symbols, vol-parity"),
        ("config/research_r3_trackA_B_universe.yaml", "NEW", "R3A_B moderate, 19 symbols"),
        ("config/research_r3_trackA_C_universe.yaml", "NEW", "R3A_C aggressive, 19 symbols"),
        ("scripts/run_r3_trackA_universe_research.py", "NEW", "Expanded universe research runner"),
    ]
    print(f"{'File':<55} {'Status':<8} {'Purpose'}")
    print("-" * 120)
    for f, status, purpose in changes:
        print(f"{f:<55} {status:<8} {purpose}")
    print("\nUniverse: 19 symbols (MATICUSDT excluded — Binance POL migration, data ends 2024-09)")

    # ── B. Universe Selection ──
    print("\n" + "=" * 80)
    print("B1. UNIVERSE SELECTION")
    print("=" * 80)
    print(f"  Selected: {len(SYMBOLS)} symbols")
    print(f"  {'Symbol':<12} {'Weight':>8}")
    print("  " + "-" * 22)
    for sym in sorted(WEIGHTS.keys(), key=lambda x: -WEIGHTS[x]):
        print(f"  {sym:<12} {WEIGHTS[sym]:>8.1%}")
    print(f"  {'TOTAL':<12} {sum(WEIGHTS.values()):>8.1%}")
    print(f"\n  Excluded: MATICUSDT (data ends 2024-09-11, Binance POL migration)")

    # ── B2. Core Comparison ──
    print_comparison_table(all_results)

    # ── C. Delta ──
    if 1.0 in baseline_by_cost and overlay_results_by_cost.get(1.0):
        print_delta_table(baseline_by_cost[1.0], overlay_results_by_cost[1.0])

    # ── D. Robustness ──
    if robustness:
        print_robustness_table(robustness, args.configs)

    # ── E. Cost Attribution ──
    print_cost_attribution(all_results)

    # ── F. Walk-Forward ──
    wf_symbols_used = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
                        "DOGEUSDT", "LINKUSDT", "AVAXUSDT", "LTCUSDT"]
    if wf_all:
        print_walk_forward_table(wf_all, wf_symbols_used)

    # ── G. Verdict ──
    print("\n" + "=" * 80)
    print("G. FINAL VERDICT")
    print("=" * 80)

    if 1.0 in baseline_by_cost and overlay_results_by_cost.get(1.0):
        verdicts = check_acceptance_criteria(
            baseline_10=baseline_by_cost[1.0],
            baseline_15=baseline_by_cost.get(1.5, baseline_by_cost[1.0]),
            overlay_results_10=overlay_results_by_cost[1.0],
            overlay_results_15=overlay_results_by_cost.get(1.5, overlay_results_by_cost[1.0]),
            overlay_results_20=overlay_results_by_cost.get(2.0, overlay_results_by_cost[1.0]),
            delay_results=delay_results,
            wf_results=wf_all if wf_all else None,
        )

        for cfg_name, v in verdicts.items():
            icons = {
                "cost1.5x SR≥1.2":   "✅" if v["pass_sharpe_15x"] else "❌",
                "MDD≤40%":           "✅" if v["pass_mdd_40"] else "❌",
                "WF OOS+ ≥70%":      "✅" if v["pass_wf"] else "❌",
                "Delay≤30%":         "✅" if v["pass_delay"] else "❌",
                "SR≥bl-0.05":        "✅" if v["pass_sharpe_base"] else "❌",
                "Cost2.0x>0":        "✅" if v["pass_cost_20"] else "❌",
            }
            status = "✅ PASS" if v["overall"] else "❌ FAIL"

            print(f"\n  {cfg_name}:")
            print(f"    ── Acceptance Gates ── {status}")
            for k, icon in icons.items():
                print(f"      {k}: {icon}")
            print(
                f"    Sharpe@1.0x={v['sharpe_10']:.3f}, "
                f"Sharpe@1.5x={v['sharpe_15']:.3f}, "
                f"Δ Sharpe={v['sharpe_delta']:+.3f}, "
                f"Δ MDD={v['mdd_delta']:+.2f}pp, "
                f"Δ CAGR={v['cagr_delta']:+.2f}%"
            )

        # Final decision
        best_cfg = None
        for cfg_name, v in verdicts.items():
            if v["overall"]:
                if best_cfg is None or v["sharpe_15"] > verdicts[best_cfg]["sharpe_15"]:
                    best_cfg = cfg_name

        if best_cfg:
            if "C" in best_cfg:
                verdict = "GO_R3C_UNIVERSE_PAPER"
            else:
                verdict = "GO_R3B_UNIVERSE_PAPER"
            reason = f"{best_cfg} passes all acceptance gates on expanded universe."
        else:
            # Check if any are close
            partial = any(
                sum([v["pass_sharpe_15x"], v["pass_mdd_40"], v["pass_delay"],
                     v["pass_sharpe_base"], v["pass_cost_20"]]) >= 4
                for v in verdicts.values()
            )
            if partial:
                verdict = "NEED_MORE_WORK"
                reason = "Close to acceptance but not all gates passed. Consider parameter refinement."
            else:
                verdict = "KEEP_R2_1"
                reason = "R3 overlay does not pass acceptance criteria on expanded universe."

        print(f"\n  {'─'*60}")
        print(f"  VERDICT: {verdict}")
        print(f"  REASON:  {reason}")
    else:
        print("  ⚠️  Insufficient data to determine verdict")

    # ── H. Evidence Paths ──
    print("\n" + "=" * 80)
    print("H. EVIDENCE PATHS")
    print("=" * 80)

    # Save results JSON
    results_json = {
        "timestamp": timestamp,
        "universe": {
            "n_symbols": len(SYMBOLS),
            "symbols": SYMBOLS,
            "weights": WEIGHTS,
            "excluded": ["MATICUSDT"],
            "exclusion_reason": "Data ends 2024-09-11 (Binance MATIC→POL migration)",
        },
        "baseline_config": str(BASELINE_CONFIG),
        "overlay_configs": {k: str(v) for k, v in OVERLAY_CONFIGS.items()},
        "results": [],
        "robustness": {},
        "delay_results": {},
        "yearly": {},
    }

    for r in all_results:
        entry = {k: v for k, v in r.items()
                 if k not in ("per_symbol", "portfolio_equity")}
        # Add per-symbol summary (no equity)
        entry["per_symbol_summary"] = {}
        for sym, sym_data in r.get("per_symbol", {}).items():
            entry["per_symbol_summary"][sym] = {
                k: v for k, v in sym_data.items()
                if k not in ("equity", "pos")
            }
        for k, v in entry.items():
            if isinstance(v, (np.integer, np.int64)):
                entry[k] = int(v)
            elif isinstance(v, (np.floating, np.float64)):
                entry[k] = float(v)
        results_json["results"].append(entry)

    # Save robustness
    for test_name, test_data in robustness.items():
        results_json["robustness"][test_name] = {}
        for cfg_name, data in test_data.items():
            if isinstance(data, dict):
                clean = {}
                for k, v in data.items():
                    if isinstance(v, dict):
                        clean[k] = {
                            kk: (float(vv) if isinstance(vv, (np.floating, np.float64)) else
                                 int(vv) if isinstance(vv, (np.integer, np.int64)) else vv)
                            for kk, vv in v.items()
                        }
                    elif isinstance(v, (np.floating, np.float64)):
                        clean[k] = float(v)
                    elif isinstance(v, (np.integer, np.int64)):
                        clean[k] = int(v)
                    else:
                        clean[k] = v
                results_json["robustness"][test_name][cfg_name] = clean

    # Save delay
    for name, data in delay_results.items():
        results_json["delay_results"][name] = {
            k: (float(v) if isinstance(v, (np.floating, np.float64)) else v)
            for k, v in data.items()
        }

    # Save yearly
    results_json["yearly"]["baseline"] = {}
    for yr_name, yr_data in baseline_yearly.items():
        results_json["yearly"]["baseline"][yr_name] = {
            k: v for k, v in yr_data.items()
            if k not in ("per_symbol", "portfolio_equity")
        }
    for config_name, yr_dict in overlay_yearly.items():
        results_json["yearly"][config_name] = {}
        for yr_name, yr_data in yr_dict.items():
            results_json["yearly"][config_name][yr_name] = {
                k: v for k, v in yr_data.items()
                if k not in ("per_symbol", "portfolio_equity")
            }

    results_path = output_dir / "r3_trackA_universe_results.json"
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2, default=str)

    # Save WF results
    if wf_all:
        wf_path = output_dir / "walk_forward_universe_summary.json"
        wf_save = {}
        for cfg_name, wf in wf_all.items():
            wf_save[cfg_name] = {}
            for sym, data in wf.items():
                if isinstance(data, dict):
                    wf_save[cfg_name][sym] = {
                        k: (float(v) if isinstance(v, (np.floating, np.float64)) else v)
                        for k, v in data.items()
                        if k != "summary_text"
                    }
        with open(wf_path, "w") as f:
            json.dump(wf_save, f, indent=2, default=str)
        print(f"  WF results:  {wf_path}")

    # Save verdict
    if 1.0 in baseline_by_cost and overlay_results_by_cost.get(1.0):
        verdict_path = output_dir / "verdict.json"
        verdict_data = {
            "timestamp": timestamp,
            "verdict": verdict,
            "reason": reason,
            "verdicts": {
                k: {kk: vv for kk, vv in v.items()} for k, v in verdicts.items()
            },
        }
        with open(verdict_path, "w") as f:
            json.dump(verdict_data, f, indent=2, default=str)
        print(f"  Verdict:     {verdict_path}")

    print(f"  Full results: {results_path}")
    print(f"  Output dir:   {output_dir}")

    print("\n" + "=" * 80)
    print("  R3 Track A Expanded Universe Research complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
