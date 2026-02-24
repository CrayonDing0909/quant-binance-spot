#!/usr/bin/env python3
"""
R2 Overlay Ablation Runner

Phase A: Vol-only entry pause (immediate)
Phase B: OI coverage report (parallel)
Phase C: Vol-only / OI-only / OI+Vol ablation (after OI data ready)
Phase D: Final verdict

Usage:
    # Phase A: vol-only pause matrix
    PYTHONPATH=src python scripts/run_r2_overlay_ablation.py --phase A

    # Phase B: OI coverage report
    PYTHONPATH=src python scripts/run_r2_overlay_ablation.py --phase B --report-coverage

    # Phase C: full ablation (after OI data ready)
    PYTHONPATH=src python scripts/run_r2_overlay_ablation.py --phase C

    # All phases
    PYTHONPATH=src python scripts/run_r2_overlay_ablation.py --phase all
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

# Suppress vectorbt / numpy warnings during batch runs
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from qtrade.config import load_config
from qtrade.backtest.run_backtest import run_symbol_backtest, BacktestResult
from qtrade.data.storage import load_klines
from qtrade.data.open_interest import (
    get_oi_path,
    load_open_interest,
    align_oi_to_klines,
    compute_oi_coverage,
    print_oi_coverage_report,
)
from qtrade.strategy.overlays.oi_vol_exit_overlay import compute_flip_count


# ══════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════

BASELINE_CONFIG = "config/prod_candidate_R2.yaml"
VOL_PAUSE_CONFIGS = {
    "vol_pause_A": "config/research_r2_vol_pause_A.yaml",
    "vol_pause_B": "config/research_r2_vol_pause_B.yaml",
    "vol_pause_C": "config/research_r2_vol_pause_C.yaml",
}
ABLATION_CONFIGS = {
    "vol_only": "config/research_r2_oi_vol_ablation_vol_only.yaml",
    "oi_only": "config/research_r2_oi_vol_ablation_oi_only.yaml",
    "oi_vol": "config/research_r2_oi_vol_ablation_oi_vol.yaml",
}
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
WEIGHTS = [0.34, 0.33, 0.33]
COST_MULTS = [1.0, 1.5, 2.0]
OI_COVERAGE_THRESHOLD = 70.0  # 最低覆蓋率門檻


# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════

def _load_yaml_raw(path: str) -> dict:
    """Load raw YAML dict (for ensemble/overlay access)"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _get_ensemble_strategies(config_path: str) -> dict | None:
    """Load ensemble per-symbol strategy configs"""
    raw = _load_yaml_raw(config_path)
    ens = raw.get("ensemble", {})
    if ens.get("enabled", False):
        return ens.get("strategies", {})
    return None


def _load_oi_for_symbol(
    symbol: str,
    data_dir: Path,
    kline_index: pd.DatetimeIndex | None = None,
) -> pd.Series | None:
    """Load merged OI data for a symbol, optionally aligned to kline index."""
    # Try merged > coinglass > binance
    for provider_name in ["merged", "coinglass", "binance"]:
        path = get_oi_path(data_dir, symbol, provider_name)
        oi_df = load_open_interest(path)
        if oi_df is not None and not oi_df.empty:
            if kline_index is not None:
                return align_oi_to_klines(oi_df, kline_index, max_ffill_bars=2)
            return oi_df["sumOpenInterest"]
    return None


def _run_portfolio(
    config_path: str,
    cost_mult: float = 1.0,
    label: str = "",
    inject_oi: bool = False,
) -> dict:
    """
    Run portfolio backtest and return results dict.
    Integrates overlay through cfg dict (read from YAML).
    If inject_oi=True, loads OI data and injects into cfg for overlay.
    """
    cfg = load_config(config_path)
    raw = _load_yaml_raw(config_path)
    ensemble_strategies = _get_ensemble_strategies(config_path)
    overlay_cfg = raw.get("strategy", {}).get("overlay")

    market_type = cfg.market_type_str
    initial_cash = cfg.backtest.initial_cash

    per_symbol_results: dict[str, BacktestResult] = {}
    per_symbol_trades: dict[str, int] = {}
    per_symbol_flips: dict[str, int] = {}

    for symbol in SYMBOLS:
        # Build per-symbol backtest cfg
        if ensemble_strategies and symbol in ensemble_strategies:
            sym_strat = ensemble_strategies[symbol]
            strategy_name = sym_strat["name"]
            bt_cfg = cfg.to_backtest_dict(symbol=symbol)
            bt_cfg["strategy_params"] = sym_strat.get("params", bt_cfg["strategy_params"])
        else:
            strategy_name = cfg.strategy.name
            bt_cfg = cfg.to_backtest_dict(symbol=symbol)

        # Inject overlay config (if present in YAML)
        if overlay_cfg:
            bt_cfg["overlay"] = overlay_cfg

        # Cost multiplier
        if cost_mult != 1.0:
            bt_cfg["fee_bps"] = bt_cfg["fee_bps"] * cost_mult
            bt_cfg["slippage_bps"] = bt_cfg["slippage_bps"] * cost_mult

        data_path = (
            cfg.data_dir / "binance" / market_type
            / cfg.market.interval / f"{symbol}.parquet"
        )
        if not data_path.exists():
            print(f"  ⚠️  {symbol}: data not found ({data_path})")
            continue

        # Inject OI data if overlay needs it
        if inject_oi and overlay_cfg:
            overlay_mode = overlay_cfg.get("mode", "")
            if overlay_mode in ("oi_only", "oi_vol"):
                # Load klines to get index for alignment
                kline_df = load_klines(data_path)
                if kline_df is not None and not kline_df.empty:
                    oi_series = _load_oi_for_symbol(
                        symbol, cfg.data_dir, kline_df.index,
                    )
                    bt_cfg["_oi_series"] = oi_series

        res = run_symbol_backtest(
            symbol, data_path, bt_cfg,
            strategy_name=strategy_name,
            data_dir=cfg.data_dir,
        )
        per_symbol_results[symbol] = res
        per_symbol_trades[symbol] = int(res.stats.get("Total Trades", 0))
        per_symbol_flips[symbol] = compute_flip_count(res.pos)

    if not per_symbol_results:
        return {"error": "No results"}

    # Build portfolio equity curve
    equity_curves = {}
    for sym, res in per_symbol_results.items():
        equity_curves[sym] = res.equity()

    active_symbols = list(per_symbol_results.keys())
    active_weights = np.array([WEIGHTS[SYMBOLS.index(s)] for s in active_symbols])
    active_weights = active_weights / active_weights.sum()

    min_start = max(eq.index[0] for eq in equity_curves.values())
    max_end = min(eq.index[-1] for eq in equity_curves.values())

    normalized = {}
    for sym in active_symbols:
        eq = equity_curves[sym].loc[min_start:max_end]
        normalized[sym] = eq / eq.iloc[0]

    portfolio_eq = sum(normalized[s] * w for s, w in zip(active_symbols, active_weights))
    portfolio_eq = portfolio_eq * initial_cash

    returns = portfolio_eq.pct_change().fillna(0)
    total_return = (portfolio_eq.iloc[-1] - initial_cash) / initial_cash * 100
    n_periods = len(returns)
    years = n_periods / (365 * 24)
    cagr = ((1 + total_return / 100) ** (1 / years) - 1) * 100 if years > 0 else 0
    sharpe = (np.sqrt(365 * 24) * returns.mean() / returns.std()) if returns.std() > 0 else 0
    rolling_max = portfolio_eq.expanding().max()
    drawdown = (portfolio_eq - rolling_max) / rolling_max
    max_dd = abs(drawdown.min()) * 100
    calmar = cagr / max_dd if max_dd > 0 else 0

    total_trades = sum(per_symbol_trades.values())
    total_flips = sum(per_symbol_flips.values())

    # Year-by-year analysis
    yearly = {}
    for year in range(2022, 2027):
        yr_start = pd.Timestamp(f"{year}-01-01")
        yr_end = pd.Timestamp(f"{year}-12-31 23:59:59")
        if portfolio_eq.index.tz:
            yr_start = yr_start.tz_localize(portfolio_eq.index.tz)
            yr_end = yr_end.tz_localize(portfolio_eq.index.tz)
        yr_eq = portfolio_eq[(portfolio_eq.index >= yr_start) & (portfolio_eq.index <= yr_end)]
        if len(yr_eq) > 10:
            yr_ret = (yr_eq.iloc[-1] / yr_eq.iloc[0] - 1) * 100
            yr_returns = yr_eq.pct_change().fillna(0)
            yr_sharpe = (np.sqrt(365 * 24) * yr_returns.mean() / yr_returns.std()) if yr_returns.std() > 0 else 0
            yr_peak = yr_eq.expanding().max()
            yr_dd = ((yr_eq - yr_peak) / yr_peak).min() * 100
            yearly[str(year)] = {
                "return_pct": round(yr_ret, 2),
                "sharpe": round(yr_sharpe, 2),
                "max_dd_pct": round(abs(yr_dd), 2),
            }

    result = {
        "label": label,
        "config": config_path,
        "cost_mult": cost_mult,
        "total_return_pct": round(total_return, 2),
        "cagr_pct": round(cagr, 2),
        "sharpe": round(sharpe, 2),
        "max_dd_pct": round(max_dd, 2),
        "calmar": round(calmar, 3),
        "total_trades": total_trades,
        "total_flips": total_flips,
        "per_symbol_trades": per_symbol_trades,
        "per_symbol_flips": per_symbol_flips,
        "yearly": yearly,
        "start": str(min_start),
        "end": str(max_end),
    }

    print(
        f"  {label} (cost={cost_mult:.1f}x): "
        f"Return={total_return:+.1f}%, SR={sharpe:.2f}, "
        f"MDD={max_dd:.1f}%, Trades={total_trades}, Flips={total_flips}"
    )

    return result


def _run_walk_forward(
    config_path: str,
    n_splits: int = 5,
    label: str = "",
    inject_oi: bool = False,
) -> dict:
    """Run walk-forward analysis for all symbols with overlay integrated"""
    from qtrade.validation.walk_forward import walk_forward_analysis

    cfg = load_config(config_path)
    raw = _load_yaml_raw(config_path)
    ensemble_strategies = _get_ensemble_strategies(config_path)
    overlay_cfg = raw.get("strategy", {}).get("overlay")
    market_type = cfg.market_type_str

    wf_results = {}
    for symbol in SYMBOLS:
        if ensemble_strategies and symbol in ensemble_strategies:
            sym_strat = ensemble_strategies[symbol]
            strategy_name = sym_strat["name"]
            bt_cfg = cfg.to_backtest_dict(symbol=symbol)
            bt_cfg["strategy_params"] = sym_strat.get("params", bt_cfg["strategy_params"])
        else:
            strategy_name = cfg.strategy.name
            bt_cfg = cfg.to_backtest_dict(symbol=symbol)

        # Inject overlay
        if overlay_cfg:
            bt_cfg["overlay"] = overlay_cfg

        # Inject OI for oi_only/oi_vol modes
        if inject_oi and overlay_cfg:
            overlay_mode = overlay_cfg.get("mode", "")
            if overlay_mode in ("oi_only", "oi_vol"):
                data_path = (
                    cfg.data_dir / "binance" / market_type
                    / cfg.market.interval / f"{symbol}.parquet"
                )
                if data_path.exists():
                    kline_df = load_klines(data_path)
                    if kline_df is not None and not kline_df.empty:
                        oi_series = _load_oi_for_symbol(
                            symbol, cfg.data_dir, kline_df.index,
                        )
                        bt_cfg["_oi_series"] = oi_series

        bt_cfg["strategy_name"] = strategy_name

        data_path = (
            cfg.data_dir / "binance" / market_type
            / cfg.market.interval / f"{symbol}.parquet"
        )
        if not data_path.exists():
            print(f"  ⚠️  {symbol}: data not found")
            continue

        print(f"\n  === Walk-Forward: {symbol} ({label}) ===")
        try:
            wf_df = walk_forward_analysis(
                symbol=symbol,
                data_path=data_path,
                cfg=bt_cfg,
                n_splits=n_splits,
                data_dir=cfg.data_dir,
            )
            if not wf_df.empty:
                avg_oos_sr = wf_df["test_sharpe"].mean()
                oos_positive = (wf_df["test_sharpe"] > 0).sum()
                wf_results[symbol] = {
                    "avg_oos_sharpe": round(avg_oos_sr, 2),
                    "oos_positive": int(oos_positive),
                    "n_splits": len(wf_df),
                    "splits": wf_df.to_dict("records"),
                }
                print(f"  → {symbol}: OOS Sharpe avg={avg_oos_sr:.2f}, OOS+={oos_positive}/{len(wf_df)}")
            else:
                wf_results[symbol] = {"avg_oos_sharpe": 0, "oos_positive": 0, "n_splits": 0}
        except Exception as e:
            print(f"  ❌ {symbol} WF failed: {e}")
            wf_results[symbol] = {"avg_oos_sharpe": 0, "oos_positive": 0, "n_splits": 0, "error": str(e)}

    return {"label": label, "symbols": wf_results}


# ══════════════════════════════════════════════════════════════
# Phase A: Vol-only pause
# ══════════════════════════════════════════════════════════════

def run_phase_a(output_dir: Path, skip_wf: bool = False) -> dict:
    """Run Phase A: baseline + vol_pause A/B/C with WF"""
    print("\n" + "=" * 80)
    print("PHASE A: Vol-only Entry Pause")
    print("=" * 80)

    all_results = []

    # 1. Baseline
    print("\n--- R2 Baseline ---")
    for cm in COST_MULTS:
        r = _run_portfolio(BASELINE_CONFIG, cost_mult=cm, label="R2_baseline")
        r["variant"] = "R2_baseline"
        all_results.append(r)

    # 2. Vol Pause A/B/C
    for variant_name, config_path in VOL_PAUSE_CONFIGS.items():
        print(f"\n--- {variant_name} ---")
        for cm in COST_MULTS:
            r = _run_portfolio(config_path, cost_mult=cm, label=variant_name)
            r["variant"] = variant_name
            all_results.append(r)

    # 3. Walk-forward
    wf_results = {}
    if not skip_wf:
        print("\n--- Walk-Forward Analysis ---")
        wf_results["R2_baseline"] = _run_walk_forward(
            BASELINE_CONFIG, n_splits=5, label="R2_baseline"
        )
        for variant_name, config_path in VOL_PAUSE_CONFIGS.items():
            wf_results[variant_name] = _run_walk_forward(
                config_path, n_splits=5, label=variant_name
            )

    return {"backtest": all_results, "walk_forward": wf_results}


# ══════════════════════════════════════════════════════════════
# Phase B: OI coverage
# ══════════════════════════════════════════════════════════════

def run_phase_b(output_dir: Path) -> dict:
    """Run Phase B: OI data coverage report"""
    print("\n" + "=" * 80)
    print("PHASE B: OI Data Coverage Report")
    print("=" * 80)

    cfg = load_config(BASELINE_CONFIG)
    coverage_df = compute_oi_coverage(
        symbols=SYMBOLS,
        data_dir=cfg.data_dir,
        backtest_start="2022-01-01",
        interval="1h",
    )
    print_oi_coverage_report(coverage_df)

    # Overall coverage check
    overall_actual = coverage_df["actual_bars"].sum()
    overall_expected = coverage_df["expected_bars"].sum()
    overall_pct = (overall_actual / overall_expected * 100) if overall_expected > 0 else 0
    oi_valid = overall_pct >= OI_COVERAGE_THRESHOLD

    return {
        "coverage": coverage_df.to_dict("records"),
        "overall_coverage_pct": round(overall_pct, 1),
        "oi_valid": oi_valid,
        "verdict": "VALID" if oi_valid else "INVALID",
        "recommendation": (
            f"OI coverage {overall_pct:.1f}% >= {OI_COVERAGE_THRESHOLD}% — VALID for research"
            if oi_valid
            else f"OI coverage {overall_pct:.1f}% < {OI_COVERAGE_THRESHOLD}% — INVALID. "
                 "OI-based overlay conclusions are NOT trustworthy. "
                 "Set COINGLASS_API_KEY and run: "
                 "PYTHONPATH=src python scripts/download_oi_data.py --provider coinglass"
        ),
    }


# ══════════════════════════════════════════════════════════════
# Phase C: Ablation (requires OI data)
# ══════════════════════════════════════════════════════════════

def run_phase_c(output_dir: Path, skip_wf: bool = False) -> dict:
    """
    Run Phase C: vol-only / oi-only / oi+vol ablation

    Coverage gate:
        - If OI coverage < 70%: oi-only & oi+vol results marked INVALID
        - vol-only always runs (no OI dependency)
    """
    print("\n" + "=" * 80)
    print("PHASE C: Ablation (vol-only / oi-only / oi+vol)")
    print("=" * 80)

    # First check OI coverage
    cfg = load_config(BASELINE_CONFIG)
    coverage_df = compute_oi_coverage(
        symbols=SYMBOLS,
        data_dir=cfg.data_dir,
        backtest_start="2022-01-01",
        interval="1h",
    )
    overall_actual = coverage_df["actual_bars"].sum()
    overall_expected = coverage_df["expected_bars"].sum()
    overall_pct = (overall_actual / overall_expected * 100) if overall_expected > 0 else 0
    oi_valid = overall_pct >= OI_COVERAGE_THRESHOLD

    print(f"\n📊 OI Coverage: {overall_pct:.1f}% (threshold: {OI_COVERAGE_THRESHOLD}%)")
    if oi_valid:
        print("✅ OI data sufficient — running all 3 ablation groups")
    else:
        print(f"⚠️  OI coverage {overall_pct:.1f}% < {OI_COVERAGE_THRESHOLD}%")
        print("   oi-only and oi+vol results will be marked OI_RESULT_INVALID")

    results = {
        "oi_coverage_pct": round(overall_pct, 1),
        "oi_valid": oi_valid,
        "baseline": [],
        "vol_only": [],
        "oi_only": [],
        "oi_vol": [],
        "walk_forward": {},
    }

    # ── 0. Baseline ──
    print("\n--- R2 Baseline ---")
    for cm in COST_MULTS:
        r = _run_portfolio(BASELINE_CONFIG, cost_mult=cm, label="R2_baseline")
        r["variant"] = "R2_baseline"
        results["baseline"].append(r)

    # ── 1. Vol-only (always runs) ──
    print("\n--- Vol-only ---")
    for cm in COST_MULTS:
        r = _run_portfolio(
            ABLATION_CONFIGS["vol_only"],
            cost_mult=cm,
            label="vol_only",
            inject_oi=False,
        )
        r["variant"] = "vol_only"
        results["vol_only"].append(r)

    # ── 2. OI-only (runs even if coverage low, but result marked invalid) ──
    print("\n--- OI-only ---")
    for cm in COST_MULTS:
        r = _run_portfolio(
            ABLATION_CONFIGS["oi_only"],
            cost_mult=cm,
            label="oi_only",
            inject_oi=True,
        )
        r["variant"] = "oi_only"
        if not oi_valid:
            r["oi_result_status"] = "OI_RESULT_INVALID"
        results["oi_only"].append(r)

    # ── 3. OI+Vol (runs even if coverage low, but result marked invalid) ──
    print("\n--- OI+Vol ---")
    for cm in COST_MULTS:
        r = _run_portfolio(
            ABLATION_CONFIGS["oi_vol"],
            cost_mult=cm,
            label="oi_vol",
            inject_oi=True,
        )
        r["variant"] = "oi_vol"
        if not oi_valid:
            r["oi_result_status"] = "OI_RESULT_INVALID"
        results["oi_vol"].append(r)

    # ── Walk-Forward ──
    if not skip_wf:
        print("\n--- Walk-Forward Analysis ---")
        results["walk_forward"]["R2_baseline"] = _run_walk_forward(
            BASELINE_CONFIG, n_splits=5, label="R2_baseline"
        )
        results["walk_forward"]["vol_only"] = _run_walk_forward(
            ABLATION_CONFIGS["vol_only"], n_splits=5, label="vol_only",
            inject_oi=False,
        )
        results["walk_forward"]["oi_only"] = _run_walk_forward(
            ABLATION_CONFIGS["oi_only"], n_splits=5, label="oi_only",
            inject_oi=True,
        )
        results["walk_forward"]["oi_vol"] = _run_walk_forward(
            ABLATION_CONFIGS["oi_vol"], n_splits=5, label="oi_vol",
            inject_oi=True,
        )

    return results


# ══════════════════════════════════════════════════════════════
# Report generation
# ══════════════════════════════════════════════════════════════

def _print_phase_a_report(phase_a: dict, output_dir: Path) -> None:
    """Print Phase A report tables"""
    results = phase_a["backtest"]
    wf = phase_a.get("walk_forward", {})

    # ── Table 1: Phase A Result ──
    print("\n" + "=" * 130)
    print("1) Phase A Result")
    print("=" * 130)
    header = f"{'Config':<16} {'Cost':>5} {'Sharpe':>8} {'MaxDD%':>8} {'Trades':>8} {'Flips':>8} {'2025 Ret%':>10} {'CAGR%':>8} {'TotalRet%':>10}"
    print(header)
    print("-" * 130)

    for r in results:
        yr_2025 = r.get("yearly", {}).get("2025", {})
        ret_2025 = yr_2025.get("return_pct", "N/A")
        ret_2025_str = f"{ret_2025:+.1f}" if isinstance(ret_2025, (int, float)) else str(ret_2025)
        print(
            f"{r['variant']:<16} {r['cost_mult']:>5.1f} "
            f"{r['sharpe']:>8.2f} {r['max_dd_pct']:>8.1f} "
            f"{r['total_trades']:>8} {r['total_flips']:>8} "
            f"{ret_2025_str:>10} {r['cagr_pct']:>8.1f} "
            f"{r['total_return_pct']:>10.1f}"
        )

    # ── Table 2: Delta vs Baseline (cost=1.5) ──
    _print_delta_table(results, "R2_baseline")

    # ── Walk-Forward ──
    if wf:
        _print_wf_table(wf)

    # ── Year-by-year ──
    _print_yearly_table(results)


def _print_phase_c_report(phase_c: dict, output_dir: Path) -> None:
    """Print Phase C ablation report"""
    oi_valid = phase_c.get("oi_valid", False)
    oi_pct = phase_c.get("oi_coverage_pct", 0)

    print("\n" + "=" * 130)
    print("Phase C: Ablation Table")
    print(f"OI Coverage: {oi_pct}% — {'VALID' if oi_valid else 'INVALID'}")
    print("=" * 130)

    # Collect all results
    all_results = (
        phase_c.get("baseline", [])
        + phase_c.get("vol_only", [])
        + phase_c.get("oi_only", [])
        + phase_c.get("oi_vol", [])
    )

    header = (
        f"{'Variant':<14} {'Cost':>5} {'CAGR%':>8} {'Sharpe':>8} "
        f"{'MaxDD%':>8} {'Trades':>8} {'Flips':>8} "
        f"{'2025 Ret%':>10} {'Status':>18}"
    )
    print(header)
    print("-" * 130)

    for r in all_results:
        if isinstance(r, str):
            continue
        if "error" in r:
            continue

        yr_2025 = r.get("yearly", {}).get("2025", {})
        ret_2025 = yr_2025.get("return_pct", "N/A")
        ret_2025_str = f"{ret_2025:+.1f}" if isinstance(ret_2025, (int, float)) else str(ret_2025)
        status = r.get("oi_result_status", "OK")

        print(
            f"{r['variant']:<14} {r['cost_mult']:>5.1f} "
            f"{r['cagr_pct']:>8.1f} {r['sharpe']:>8.2f} "
            f"{r['max_dd_pct']:>8.1f} {r['total_trades']:>8} "
            f"{r['total_flips']:>8} {ret_2025_str:>10} "
            f"{status:>18}"
        )

    # Delta table
    _print_delta_table(all_results, "R2_baseline")

    # Walk-forward
    wf = phase_c.get("walk_forward", {})
    if wf:
        _print_wf_table(wf)

    # Year-by-year
    _print_yearly_table(all_results)


def _print_delta_table(results: list[dict], baseline_variant: str) -> None:
    """Print delta vs baseline (cost=1.5)"""
    print("\n" + "=" * 120)
    print(f"Delta vs {baseline_variant} (cost_mult=1.5)")
    print("=" * 120)

    baseline_15 = next(
        (r for r in results
         if isinstance(r, dict)
         and r.get("variant") == baseline_variant
         and r.get("cost_mult") == 1.5),
        None,
    )
    if not baseline_15:
        print("  (no baseline at cost_mult=1.5)")
        return

    header = (
        f"{'Config':<14} {'Δ Sharpe':>10} {'Δ MaxDD':>10} "
        f"{'Δ Trades':>16} {'Δ Flips':>16} {'Δ 2025 Ret':>12}"
    )
    print(header)
    print("-" * 120)

    for r in results:
        if not isinstance(r, dict):
            continue
        if r.get("variant") == baseline_variant or r.get("cost_mult") != 1.5:
            continue

        d_sharpe = r["sharpe"] - baseline_15["sharpe"]
        d_maxdd = r["max_dd_pct"] - baseline_15["max_dd_pct"]
        d_trades = r["total_trades"] - baseline_15["total_trades"]
        d_flips = r["total_flips"] - baseline_15["total_flips"]

        bl_2025 = baseline_15.get("yearly", {}).get("2025", {}).get("return_pct", 0)
        r_2025 = r.get("yearly", {}).get("2025", {}).get("return_pct", 0)
        d_2025 = r_2025 - bl_2025

        trade_pct = (d_trades / baseline_15["total_trades"] * 100) if baseline_15["total_trades"] > 0 else 0
        flip_pct = (d_flips / baseline_15["total_flips"] * 100) if baseline_15["total_flips"] > 0 else 0

        status = r.get("oi_result_status", "")
        if status:
            status = f" [{status}]"

        print(
            f"{r['variant']:<14}{status} "
            f"{d_sharpe:>+10.2f} "
            f"{d_maxdd:>+10.1f} "
            f"{d_trades:>+8} ({trade_pct:+.0f}%) "
            f"{d_flips:>+8} ({flip_pct:+.0f}%) "
            f"{d_2025:>+12.1f}"
        )


def _print_wf_table(wf: dict) -> None:
    """Print walk-forward table"""
    print("\n" + "=" * 90)
    print("Walk-Forward Table")
    print("=" * 90)
    header = f"{'Config':<16} {'BTC OOS SR':>12} {'ETH OOS SR':>12} {'SOL OOS SR':>12} {'OOS+/N':>10}"
    print(header)
    print("-" * 90)

    for variant_name, wf_data in wf.items():
        sym_data = wf_data.get("symbols", {})
        btc_sr = sym_data.get("BTCUSDT", {}).get("avg_oos_sharpe", "N/A")
        eth_sr = sym_data.get("ETHUSDT", {}).get("avg_oos_sharpe", "N/A")
        sol_sr = sym_data.get("SOLUSDT", {}).get("avg_oos_sharpe", "N/A")

        total_positive = sum(
            sym_data.get(s, {}).get("oos_positive", 0) for s in SYMBOLS
        )
        total_splits = sum(
            sym_data.get(s, {}).get("n_splits", 0) for s in SYMBOLS
        )

        def _fmt(v):
            return f"{v:+.2f}" if isinstance(v, (int, float)) else str(v)

        print(
            f"{variant_name:<16} "
            f"{_fmt(btc_sr):>12} "
            f"{_fmt(eth_sr):>12} "
            f"{_fmt(sol_sr):>12} "
            f"{total_positive}/{total_splits}"
        )


def _print_yearly_table(results: list[dict]) -> None:
    """Print year-by-year detail"""
    print("\n" + "=" * 100)
    print("Year-by-Year Detail (cost_mult=1.5)")
    print("=" * 100)
    header = f"{'Config':<14} {'2022':>10} {'2023':>10} {'2024':>10} {'2025':>10} {'2026 YTD':>10}"
    print(header)
    print("-" * 100)

    for r in results:
        if not isinstance(r, dict):
            continue
        if r.get("cost_mult") != 1.5:
            continue
        yr = r.get("yearly", {})
        cols = []
        for y in ["2022", "2023", "2024", "2025", "2026"]:
            v = yr.get(y, {}).get("return_pct", "N/A")
            cols.append(f"{v:+.1f}" if isinstance(v, (int, float)) else str(v))
        print(f"{r['variant']:<14} " + " ".join(f"{c:>10}" for c in cols))


def _determine_verdict(phase_c: dict | None, phase_b: dict | None) -> tuple[str, str]:
    """Determine final verdict based on Phase C ablation results"""
    if phase_c is None:
        return "KEEP_R2", "Phase C not run"

    oi_valid = phase_c.get("oi_valid", False)
    baseline_results = phase_c.get("baseline", [])
    vol_only_results = phase_c.get("vol_only", [])
    oi_only_results = phase_c.get("oi_only", [])
    oi_vol_results = phase_c.get("oi_vol", [])

    baseline_15 = next(
        (r for r in baseline_results if r.get("cost_mult") == 1.5), None
    )
    if not baseline_15:
        return "KEEP_R2", "No baseline data"

    def _check_variant(variant_results: list[dict], name: str) -> tuple[bool, str]:
        """Check if variant passes acceptance criteria at cost=1.5"""
        r = next((r for r in variant_results if isinstance(r, dict) and r.get("cost_mult") == 1.5), None)
        if not r:
            return False, f"{name}: no cost=1.5 result"

        d_sharpe = r["sharpe"] - baseline_15["sharpe"]
        d_maxdd = r["max_dd_pct"] - baseline_15["max_dd_pct"]
        d_trades = r["total_trades"] - baseline_15["total_trades"]
        trade_pct = (d_trades / baseline_15["total_trades"] * 100) if baseline_15["total_trades"] > 0 else 0
        bl_2025 = baseline_15.get("yearly", {}).get("2025", {}).get("return_pct", 0)
        r_2025 = r.get("yearly", {}).get("2025", {}).get("return_pct", 0)

        reasons = []
        if d_sharpe < 0:
            reasons.append(f"SR Δ{d_sharpe:+.2f}")
        if d_maxdd > 1.0:
            reasons.append(f"MDD Δ{d_maxdd:+.1f}pp")
        if r_2025 < bl_2025:
            reasons.append(f"2025 ret Δ{r_2025 - bl_2025:+.1f}%")

        passed = len(reasons) == 0
        detail = (
            f"{name}: SR={r['sharpe']:.2f}(Δ{d_sharpe:+.2f}), "
            f"Trades Δ{trade_pct:+.0f}%, MDD Δ{d_maxdd:+.1f}pp, "
            f"2025 ret {r_2025:+.1f}%"
        )
        if reasons:
            detail += f" — FAIL: {', '.join(reasons)}"
        return passed, detail

    # Check OI+Vol first (best case)
    if oi_valid:
        oi_vol_ok, oi_vol_reason = _check_variant(oi_vol_results, "oi_vol")
        if oi_vol_ok:
            return "GO_OI_VOL", oi_vol_reason

        oi_only_ok, oi_only_reason = _check_variant(oi_only_results, "oi_only")
        if oi_only_ok:
            return "GO_OI_VOL", oi_only_reason  # OI-only passes → use it

    # Check vol-only
    vol_ok, vol_reason = _check_variant(vol_only_results, "vol_only")
    if vol_ok:
        return "GO_R2_1_VOL_ONLY", vol_reason

    # Nothing passes
    all_reasons = []
    _, vr = _check_variant(vol_only_results, "vol_only")
    all_reasons.append(vr)
    if oi_valid:
        _, or_ = _check_variant(oi_only_results, "oi_only")
        _, ovr = _check_variant(oi_vol_results, "oi_vol")
        all_reasons.extend([or_, ovr])
    else:
        all_reasons.append(f"OI coverage {phase_c.get('oi_coverage_pct', 0):.1f}% < {OI_COVERAGE_THRESHOLD}% — OI results INVALID")

    return "KEEP_R2", "\n  ".join(all_reasons)


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="R2 Overlay Ablation Runner")
    parser.add_argument(
        "--phase", type=str, default="A",
        choices=["A", "B", "C", "all"],
        help="Phase to run (A/B/C/all)",
    )
    parser.add_argument("--skip-wf", action="store_true", help="Skip walk-forward")
    parser.add_argument("--report-coverage", action="store_true", help="Only report OI coverage")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("reports/overlay_ablation") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    phase_a_data = None
    phase_b_data = None
    phase_c_data = None

    # ── Phase B (coverage report — can run anytime) ──
    if args.report_coverage or args.phase in ("B", "all"):
        phase_b_data = run_phase_b(output_dir)
        with open(output_dir / "phase_b_coverage.json", "w") as f:
            json.dump(phase_b_data, f, indent=2, default=str)
        if args.report_coverage:
            return

    # ── Phase A ──
    if args.phase in ("A", "all"):
        phase_a_data = run_phase_a(output_dir, skip_wf=args.skip_wf)
        with open(output_dir / "phase_a_results.json", "w") as f:
            json.dump(phase_a_data, f, indent=2, default=str)

    # ── Phase C ──
    if args.phase in ("C", "all"):
        phase_c_data = run_phase_c(output_dir, skip_wf=args.skip_wf)
        with open(output_dir / "phase_c_results.json", "w") as f:
            json.dump(phase_c_data, f, indent=2, default=str)

    # ── Final Report ──
    print("\n\n")
    print("█" * 80)
    print("██  FINAL REPORT: R2 Overlay Ablation")
    print("█" * 80)

    if phase_a_data:
        _print_phase_a_report(phase_a_data, output_dir)

    if phase_b_data:
        print("\n" + "=" * 90)
        print("OI Coverage Report")
        print("=" * 90)
        for row in phase_b_data.get("coverage", []):
            usable = "✅" if row["usable"] else "❌"
            print(
                f"  {row['symbol']:<10} {row['year']} "
                f"coverage={row['coverage_pct']:.1f}% {usable}"
            )
        print(f"\n  Coverage Verdict: {phase_b_data.get('verdict', 'N/A')}")
        print(f"  {phase_b_data.get('recommendation', 'N/A')}")

    if phase_c_data:
        _print_phase_c_report(phase_c_data, output_dir)

    # ── Final Verdict ──
    verdict, reason = _determine_verdict(phase_c_data, phase_b_data)
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)
    print(f"\n  >>> {verdict} <<<")
    print(f"\n  Reason: {reason}")

    if verdict == "GO_OI_VOL":
        print("\n  Shortest path to production:")
        print("  1. Copy winning ablation config to prod_candidate_R2_1.yaml")
        print("  2. Run 1-week paper trading with overlay")
        print("  3. If stable: swap R2 → R2.1 in production")
        print("\n  Rollback condition: If OOS Sharpe < 0 for 2 consecutive weeks → revert to R2")
    elif verdict == "GO_R2_1_VOL_ONLY":
        print("\n  Shortest path to production:")
        print("  1. Copy vol_only config to prod_candidate_R2_1.yaml")
        print("  2. Run 1-week paper trading with vol overlay")
        print("  3. If stable: swap R2 → R2.1 in production")
        print("\n  Rollback: OOS Sharpe < 0 for 2 consecutive weeks → revert to R2")
    else:
        print("\n  Next steps:")
        print("  1. If OI coverage insufficient: set COINGLASS_API_KEY and download")
        print("     PYTHONPATH=src python scripts/download_oi_data.py --provider coinglass")
        print("  2. Re-run Phase C: PYTHONPATH=src python scripts/run_r2_overlay_ablation.py --phase C")
        print("  3. If still KEEP_R2: explore trend-aligned overlay or per-symbol overlay")

    # ── Evidence Paths ──
    print("\n" + "=" * 80)
    print("Evidence Paths")
    print("=" * 80)
    for f in sorted(output_dir.glob("*.json")):
        print(f"  {f}")

    print(f"\n✅ All output saved to: {output_dir}")


if __name__ == "__main__":
    main()
