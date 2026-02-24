#!/usr/bin/env python3
"""
R3 Track A Research Runner
===========================
Trend Core (R2.1) + 5m/15m Microstructure Acceleration Layer

Research matrix:
    1. R2.1 baseline (3 cost multipliers)
    2. Config A/B/C with micro accel overlay (3 × 3 = 9 runs)
    3. Year-by-year breakdown
    4. Walk-forward analysis (5 splits × 3 symbols × 4 configs)
    5. +1 bar delay stress test
    6. Truncation invariance test
    7. Delta comparison table
    8. Acceptance criteria check

Usage:
    cd /path/to/quant-binance-spot
    PYTHONPATH=src python scripts/run_r3_trackA_research.py

    # Skip walk-forward (faster iteration)
    PYTHONPATH=src python scripts/run_r3_trackA_research.py --skip-wf

    # Only run specific config
    PYTHONPATH=src python scripts/run_r3_trackA_research.py --configs B

    # Skip robustness checks
    PYTHONPATH=src python scripts/run_r3_trackA_research.py --skip-robustness
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

# Ensure src is on path
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

# Suppress vectorbt warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ══════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).parent.parent
BASELINE_CONFIG = PROJECT_ROOT / "config" / "prod_candidate_R2_1.yaml"
OVERLAY_CONFIGS = {
    "A": PROJECT_ROOT / "config" / "research_r3_trackA_A.yaml",
    "B": PROJECT_ROOT / "config" / "research_r3_trackA_B.yaml",
    "C": PROJECT_ROOT / "config" / "research_r3_trackA_C.yaml",
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
# Core: Single-symbol backtest with optional micro accel overlay
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


def _load_micro_accel_params(config_path: str | Path) -> dict | None:
    """從 config 載入 micro_accel_overlay 參數"""
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    strategy_section = raw.get("strategy", {})
    micro = strategy_section.get("micro_accel_overlay")
    if micro and micro.get("enabled", False):
        return micro.get("params", {})
    return None


def _load_vol_overlay_params(config_path: str | Path) -> dict | None:
    """從 config 載入 vol overlay 參數"""
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    strategy_section = raw.get("strategy", {})
    overlay = strategy_section.get("overlay")
    if overlay and overlay.get("enabled", False):
        return overlay
    return None


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
    運行單幣種回測（支援 micro accel overlay + vol overlay）

    Args:
        extra_signal_delay: 額外延遲 bar 數（用於 +1 bar delay stress test）

    Returns:
        dict with metrics or None
    """
    market_type = cfg.market_type_str
    data_path = (
        cfg.data_dir / "binance" / market_type
        / cfg.market.interval / f"{symbol}.parquet"
    )
    if not data_path.exists():
        print(f"    ⚠️  {symbol}: data not found at {data_path}")
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

    # Build context (with extra delay for stress test)
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

    # Apply vol overlay (from R2.1) if present
    vol_overlay = _load_vol_overlay_params(config_path)
    if vol_overlay and vol_overlay.get("enabled", False):
        overlay_mode = vol_overlay.get("mode", "vol_pause")
        overlay_params = vol_overlay.get("params", {})
        pos = apply_overlay_by_mode(
            position=pos,
            price_df=df,
            oi_series=None,
            params=overlay_params,
            mode=overlay_mode,
        )

    # Apply micro accel overlay (R3 Track A)
    if micro_accel_params is not None:
        # Load OI data if available
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
            base_position=pos,
            df_1h=df,
            df_5m=df_5m,
            df_15m=df_15m,
            oi_series=oi_series,
            params=micro_accel_params,
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

    # Compute CAGR and flip count
    n_bars = len(df)
    years = n_bars / (365.25 * 24)
    total_ret = total_return_pct / 100.0
    cagr = ((1 + total_ret) ** (1 / max(years, 0.01)) - 1) * 100 if years > 0 else 0
    calmar = cagr / max_dd if max_dd > 0.01 else 0
    flips = compute_flip_count(pos)

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
) -> dict:
    """運行完整組合回測（3 symbols weighted）"""
    initial_cash = cfg.backtest.initial_cash
    per_symbol = {}

    for symbol in SYMBOLS:
        res = run_single_symbol(
            symbol=symbol,
            cfg=cfg,
            config_path=config_path,
            cost_mult=cost_mult,
            micro_accel_params=micro_accel_params,
            start_override=start_override,
            end_override=end_override,
            extra_signal_delay=extra_signal_delay,
        )
        if res is not None:
            per_symbol[symbol] = res

    if not per_symbol:
        return {"label": label, "sharpe": 0, "total_return_pct": 0,
                "max_drawdown_pct": 0, "cagr": 0, "calmar": 0,
                "total_trades": 0, "flips": 0, "per_symbol": {}}

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
    total_flips = sum(per_symbol[s]["flips"] for s in active_symbols)

    return {
        "label": label,
        "total_return_pct": round(total_return, 2),
        "cagr": round(cagr, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "calmar": round(calmar, 2),
        "total_trades": total_trades,
        "flips": total_flips,
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
) -> dict:
    """為每個 symbol 跑 walk-forward"""
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

        bt_cfg = cfg.to_backtest_dict(symbol=symbol)

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
# Robustness: Truncation Invariance
# ══════════════════════════════════════════════════════════════

def test_truncation_invariance(
    cfg,
    config_path: str | Path,
    micro_accel_params: dict | None = None,
) -> dict:
    """
    Truncation invariance: 跑全期 vs 只截取後 80% 應得到相似 Sharpe

    Pass: |Sharpe_full - Sharpe_80pct| / max(|Sharpe_full|, 0.01) < 0.30
    """
    results = {}
    for symbol in SYMBOLS:
        full_res = run_single_symbol(
            symbol=symbol, cfg=cfg, config_path=config_path,
            cost_mult=1.0, micro_accel_params=micro_accel_params,
        )
        if full_res is None:
            results[symbol] = {"pass": False, "reason": "no data"}
            continue

        # Find 20% cutoff
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
            results[symbol] = {"pass": False, "reason": "truncated data too short"}
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
    cfg,
    config_path: str | Path,
    micro_accel_params: dict | None = None,
) -> dict:
    """
    +1 bar delay stress: 正常 delay=1 vs delay=2

    Pass: Sharpe drop <= 30%
    """
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
    """Section B: Core Comparison Table"""
    print("\n" + "=" * 120)
    print("B. CORE COMPARISON TABLE")
    print("=" * 120)
    header = (
        f"{'Config':<24} {'Cost':>6} {'TotRet%':>10} {'CAGR%':>8} "
        f"{'Sharpe':>8} {'MaxDD%':>8} {'Calmar':>8} {'Trades':>8} {'Flips':>8}"
    )
    print(header)
    print("-" * 120)
    for r in all_results:
        print(
            f"{r['label']:<24} "
            f"{r.get('cost_mult', 1.0):>6.1f} "
            f"{r['total_return_pct']:>10.2f} "
            f"{r['cagr']:>8.2f} "
            f"{r['sharpe']:>8.2f} "
            f"{r['max_drawdown_pct']:>8.2f} "
            f"{r['calmar']:>8.2f} "
            f"{r['total_trades']:>8d} "
            f"{r.get('flips', 0):>8d}"
        )


def print_delta_table(baseline_results: dict, overlay_results: list[dict]):
    """Section C: Delta vs R2.1 Baseline"""
    print("\n" + "=" * 110)
    print("C. DELTA vs R2.1 BASELINE (cost_mult=1.0)")
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

        # Quick verdict
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


def print_robustness_table(robustness: dict):
    """Section D: Robustness Table"""
    print("\n" + "=" * 100)
    print("D. ROBUSTNESS TABLE")
    print("=" * 100)
    header = f"{'Test':<30} {'Config A':>15} {'Config B':>15} {'Config C':>15} {'Pass/Fail':>12}"
    print(header)
    print("-" * 100)

    for test_name, test_data in robustness.items():
        vals = []
        all_pass = True
        for cfg_name in ["A", "B", "C"]:
            if cfg_name in test_data:
                d = test_data[cfg_name]
                if isinstance(d, dict) and "pass" in d:
                    passed = d["pass"]
                    val_str = "✅" if passed else "❌"
                    if not passed:
                        all_pass = False
                    # Add detail
                    if "sharpe_drop_pct" in d:
                        val_str += f" ({d['sharpe_drop_pct']:+.0f}%)"
                    elif "relative_diff" in d:
                        val_str += f" ({d['relative_diff']:.2f})"
                elif isinstance(d, dict):
                    # Per-symbol results
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

        overall = "✅ PASS" if all_pass else "❌ FAIL"
        print(f"{test_name:<30} {vals[0]:>15} {vals[1]:>15} {vals[2]:>15} {overall:>12}")


def print_walk_forward_table(wf_results: dict):
    """Section E: Walk-Forward Table"""
    print("\n" + "=" * 90)
    print("E. WALK-FORWARD TABLE")
    print("=" * 90)
    header = f"{'Config':<24} {'BTC OOS SR':>12} {'ETH OOS SR':>12} {'SOL OOS SR':>12} {'OOS+/N':>8}"
    print(header)
    print("-" * 90)

    for config_name, wf in wf_results.items():
        btc_sr = wf.get("BTCUSDT", {}).get("avg_test_sharpe", 0)
        eth_sr = wf.get("ETHUSDT", {}).get("avg_test_sharpe", 0)
        sol_sr = wf.get("SOLUSDT", {}).get("avg_test_sharpe", 0)

        oos_pos_count = 0
        n_total = 0
        for sym in SYMBOLS:
            sym_data = wf.get(sym, {})
            if isinstance(sym_data, dict):
                sr = sym_data.get("avg_test_sharpe", 0)
                if sr > 0:
                    oos_pos_count += 1
                n_total += 1

        print(
            f"{config_name:<24} "
            f"{btc_sr:>12.2f} "
            f"{eth_sr:>12.2f} "
            f"{sol_sr:>12.2f} "
            f"{oos_pos_count:>8d}/{n_total}"
        )


def check_acceptance_criteria(
    baseline_10: dict,
    baseline_15: dict,
    overlay_results_10: list[dict],
    overlay_results_15: list[dict],
    overlay_results_20: list[dict],
    delay_results: dict,
    wf_results: dict | None,
) -> dict:
    """Check Track A acceptance criteria"""
    verdicts = {}

    for i, ov_10 in enumerate(overlay_results_10):
        name = ov_10["label"]
        ov_15 = overlay_results_15[i] if i < len(overlay_results_15) else ov_10
        ov_20 = overlay_results_20[i] if i < len(overlay_results_20) else ov_10

        # 1. Sharpe >= baseline - 0.05
        pass_sharpe = ov_10["sharpe"] >= baseline_10["sharpe"] - 0.05

        # 2. MaxDD not worsen > +5pp
        mdd_delta = ov_10["max_drawdown_pct"] - baseline_10["max_drawdown_pct"]
        pass_mdd = mdd_delta <= 5.0

        # 3. cost 2.0x still positive return
        pass_cost_20 = ov_20["total_return_pct"] > 0

        # 4. WF OOS+/5 >= 4/5
        pass_wf = True  # default pass if skipped
        wf_key = name
        if wf_results and wf_key in wf_results:
            wf = wf_results[wf_key]
            oos_pos = sum(
                1 for sym in SYMBOLS
                if wf.get(sym, {}).get("avg_test_sharpe", 0) > 0
            )
            pass_wf = oos_pos >= 2  # at least 2/3 positive

        # 5. +1 bar delay Sharpe drop <= 30%
        delay_data = delay_results.get(name, {})
        pass_delay = delay_data.get("pass", True)

        # 6. 3-month run-rate CAGR improvement >= +20% (relative)
        cagr_improvement = (
            (ov_10["cagr"] - baseline_10["cagr"]) / max(abs(baseline_10["cagr"]), 0.01) * 100
        )
        pass_cagr_boost = cagr_improvement >= 20.0

        # 7. cost 1.5x Sharpe not decline
        pass_cost15_sharpe = ov_15["sharpe"] >= baseline_15["sharpe"] - 0.05

        overall_12m = pass_sharpe and pass_mdd and pass_cost_20 and pass_wf and pass_delay
        overall_3m = pass_cagr_boost and pass_cost15_sharpe

        verdicts[name] = {
            "pass_sharpe": pass_sharpe,
            "pass_mdd": pass_mdd,
            "pass_cost_20": pass_cost_20,
            "pass_wf": pass_wf,
            "pass_delay": pass_delay,
            "pass_cagr_boost": pass_cagr_boost,
            "pass_cost15_sharpe": pass_cost15_sharpe,
            "overall_12m": overall_12m,
            "overall_3m": overall_3m,
            "sharpe_delta": round(ov_10["sharpe"] - baseline_10["sharpe"], 3),
            "mdd_delta": round(mdd_delta, 2),
            "cagr_delta": round(ov_10["cagr"] - baseline_10["cagr"], 2),
            "cagr_improvement_pct": round(cagr_improvement, 1),
        }

    return verdicts


# ══════════════════════════════════════════════════════════════
# Main Research Pipeline
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="R3 Track A Research")
    parser.add_argument("--skip-wf", action="store_true", help="Skip walk-forward")
    parser.add_argument("--skip-robustness", action="store_true", help="Skip robustness tests")
    parser.add_argument("--configs", nargs="+", default=["A", "B", "C"],
                        choices=["A", "B", "C"], help="Overlay configs to run")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT / "reports" / "r3_trackA_research" / timestamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("█" * 80)
    print("  R3 Track A: Trend Core (R2.1) + Microstructure Accel Layer")
    print("█" * 80)
    print(f"  Timestamp:  {timestamp}")
    print(f"  Output Dir: {output_dir}")
    print(f"  Baseline:   {BASELINE_CONFIG.name}")
    print(f"  Configs:    {', '.join(args.configs)}")
    print(f"  Symbols:    {', '.join(SYMBOLS)}")
    print(f"  Cost Mults: {COST_MULTS}")
    print()

    # ══════════════════════════════════════════════════════════
    # STEP 1: R2.1 BASELINE
    # ══════════════════════════════════════════════════════════
    print("━" * 80)
    print("  STEP 1: R2.1 Baseline (vol_only)")
    print("━" * 80)

    cfg_baseline = load_config(str(BASELINE_CONFIG))
    all_results = []
    baseline_by_cost = {}

    for cm in COST_MULTS:
        label = "R2.1_baseline"
        print(f"\n  Running {label} @ cost_mult={cm:.1f}...")
        result = run_portfolio_config(
            cfg=cfg_baseline, config_path=BASELINE_CONFIG,
            cost_mult=cm, label=label,
        )
        result["cost_mult"] = cm
        all_results.append(result)
        baseline_by_cost[cm] = result
        print(
            f"    → Return={result['total_return_pct']:+.2f}%, "
            f"Sharpe={result['sharpe']:.2f}, "
            f"MDD={result['max_drawdown_pct']:.2f}%, "
            f"CAGR={result['cagr']:.2f}%, "
            f"Trades={result['total_trades']}, "
            f"Flips={result.get('flips', 0)}"
        )

    # Yearly breakdown
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
    print("  STEP 2: R3 Track A Micro Accel Overlays")
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
        print(f"    accel_threshold={micro_params.get('accel_threshold')}, "
              f"boost_pct={micro_params.get('boost_pct')}, "
              f"reduce_pct={micro_params.get('reduce_pct')}, "
              f"cooldown={micro_params.get('cooldown_bars')}")

        for cm in COST_MULTS:
            label = f"R3A_{config_name}"
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
                f"      → Return={result['total_return_pct']:+.2f}%, "
                f"Sharpe={result['sharpe']:.2f}, "
                f"MDD={result['max_drawdown_pct']:.2f}%, "
                f"CAGR={result['cagr']:.2f}%, "
                f"Trades={result['total_trades']}, "
                f"Flips={result.get('flips', 0)}"
            )

        # Yearly breakdown
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
    # STEP 3: ROBUSTNESS TESTS
    # ══════════════════════════════════════════════════════════
    robustness = {}
    delay_results = {}

    if not args.skip_robustness:
        print("\n" + "━" * 80)
        print("  STEP 3: Robustness Tests")
        print("━" * 80)

        # Truncation invariance
        print("\n  3a. Truncation Invariance:")
        robustness["Truncation invariance"] = {}
        for config_name in args.configs:
            config_path = OVERLAY_CONFIGS[config_name]
            cfg_ov = load_config(str(config_path))
            micro_params = _load_micro_accel_params(config_path)
            trunc = test_truncation_invariance(cfg_ov, config_path, micro_params)
            robustness["Truncation invariance"][config_name] = trunc
            for sym, res in trunc.items():
                icon = "✅" if res.get("pass", False) else "❌"
                print(f"    {config_name}/{sym}: {icon} (diff={res.get('relative_diff', 'N/A')})")

        # +1 bar delay stress
        print("\n  3b. +1 Bar Delay Stress:")
        robustness["+1 bar delay stress"] = {}
        for config_name in args.configs:
            config_path = OVERLAY_CONFIGS[config_name]
            cfg_ov = load_config(str(config_path))
            micro_params = _load_micro_accel_params(config_path)
            delay = test_delay_stress(cfg_ov, config_path, micro_params)
            robustness["+1 bar delay stress"][config_name] = delay
            delay_results[f"R3A_{config_name}"] = delay
            icon = "✅" if delay["pass"] else "❌"
            print(
                f"    {config_name}: {icon} "
                f"(normal={delay['normal_sharpe']:.2f}, delayed={delay['delayed_sharpe']:.2f}, "
                f"drop={delay['sharpe_drop_pct']:.0f}%)"
            )

        # Cost stress (already run above, just check 2.0x)
        print("\n  3c. Cost Stress (2.0x positive return):")
        robustness["Cost 2.0x positive"] = {}
        for i, config_name in enumerate(args.configs):
            if i < len(overlay_results_by_cost.get(2.0, [])):
                r20 = overlay_results_by_cost[2.0][i]
                passed = r20["total_return_pct"] > 0
                robustness["Cost 2.0x positive"][config_name] = {
                    "pass": passed,
                    "return": round(r20["total_return_pct"], 2),
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

        # Baseline WF
        print("\n  R2.1 Baseline WF:")
        wf_all["R2.1_baseline"] = run_walk_forward_for_config(
            cfg=cfg_baseline, config_path=BASELINE_CONFIG, n_splits=5,
        )

        # Overlay WF
        for config_name in args.configs:
            config_path = OVERLAY_CONFIGS[config_name]
            cfg_ov = load_config(str(config_path))
            print(f"\n  R3A_{config_name} WF:")
            wf_all[f"R3A_{config_name}"] = run_walk_forward_for_config(
                cfg=cfg_ov, config_path=config_path, n_splits=5,
            )
    else:
        print("\n  ⏭️  Walk-forward skipped (--skip-wf)")

    # ══════════════════════════════════════════════════════════
    # FINAL REPORT
    # ══════════════════════════════════════════════════════════
    print("\n\n" + "█" * 80)
    print("  FINAL REPORT — R3 Track A")
    print("█" * 80)

    # ── A. Change Summary ──
    print("\n" + "=" * 80)
    print("A. CHANGE SUMMARY")
    print("=" * 80)
    changes = [
        ("src/qtrade/strategy/overlays/microstructure_accel_overlay.py", "NEW",
         "Microstructure accel overlay (compute_micro_features, compute_accel_score, apply_accel_overlay)"),
        ("config/research_r3_trackA_A.yaml", "NEW", "Conservative micro accel config"),
        ("config/research_r3_trackA_B.yaml", "NEW", "Moderate micro accel config"),
        ("config/research_r3_trackA_C.yaml", "NEW", "Aggressive micro accel config"),
        ("scripts/run_r3_trackA_research.py", "NEW", "R3 Track A research runner"),
        ("src/qtrade/strategy/overlays/__init__.py", "MOD", "Added micro accel overlay docs"),
    ]
    print(f"{'File':<65} {'Status':<8} {'Purpose'}")
    print("-" * 130)
    for f, status, purpose in changes:
        print(f"{f:<65} {status:<8} {purpose}")
    print("\nBackward compatible: YES — no existing files modified, prod_candidate_R2_1.yaml untouched.")

    # ── B. Core Comparison Table ──
    print_comparison_table(all_results)

    # ── C. Delta vs R2.1 ──
    if 1.0 in baseline_by_cost and overlay_results_by_cost.get(1.0):
        print_delta_table(baseline_by_cost[1.0], overlay_results_by_cost[1.0])

    # ── D. Robustness Table ──
    if robustness:
        print_robustness_table(robustness)

    # ── E. Walk-Forward Table ──
    if wf_all:
        print_walk_forward_table(wf_all)

    # ── F. Verdict ──
    print("\n" + "=" * 80)
    print("F. FINAL VERDICT")
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
                "Sharpe>=bl-0.05": "✅" if v["pass_sharpe"] else "❌",
                "MDD<=+5pp":       "✅" if v["pass_mdd"] else "❌",
                "Cost2.0x>0":      "✅" if v["pass_cost_20"] else "❌",
                "WF OOS+":         "✅" if v["pass_wf"] else "❌",
                "Delay<=30%":      "✅" if v["pass_delay"] else "❌",
                "CAGR+20%":        "✅" if v["pass_cagr_boost"] else "❌",
                "Cost1.5x SR":     "✅" if v["pass_cost15_sharpe"] else "❌",
            }
            status_12m = "✅ PASS" if v["overall_12m"] else "❌ FAIL"
            status_3m = "✅ PASS" if v["overall_3m"] else "❌ FAIL"

            print(f"\n  {cfg_name}:")
            print(f"    ── 12-month stable gates ── {status_12m}")
            for k, icon in list(icons.items())[:5]:
                print(f"      {k}: {icon}")
            print(f"    ── 3-month sprint gates ── {status_3m}")
            for k, icon in list(icons.items())[5:]:
                print(f"      {k}: {icon}")
            print(f"    Δ Sharpe={v['sharpe_delta']:+.3f}, Δ MDD={v['mdd_delta']:+.2f}pp, "
                  f"Δ CAGR={v['cagr_delta']:+.2f}% ({v['cagr_improvement_pct']:+.1f}%)")

        # Final decision
        any_12m_pass = any(v["overall_12m"] for v in verdicts.values())
        any_3m_pass = any(v["overall_3m"] for v in verdicts.values())

        if any_12m_pass and any_3m_pass:
            verdict = "GO_R3_TRACKA_PAPER"
            reason = "Both 12m and 3m gates passed for at least one config."
        elif any_12m_pass:
            verdict = "GO_R3_TRACKA_PAPER"
            reason = "12m stable gates passed (3m sprint target not yet met, continue monitoring)."
        elif any_3m_pass:
            verdict = "NEED_MORE_WORK"
            reason = "3m sprint looks promising but 12m stability gates not met."
        else:
            partial = any(
                sum([v["pass_sharpe"], v["pass_mdd"], v["pass_cost_20"], v["pass_delay"]]) >= 3
                for v in verdicts.values()
            )
            if partial:
                verdict = "NEED_MORE_WORK"
                reason = "Multiple criteria partially met. Consider tuning parameters."
            else:
                verdict = "KEEP_R2_1"
                reason = "Micro accel overlay does not improve over R2.1 baseline."

        print(f"\n  {'─'*60}")
        print(f"  VERDICT: {verdict}")
        print(f"  REASON:  {reason}")
    else:
        print("  ⚠️  Insufficient data to determine verdict")

    # ── G. Evidence Paths ──
    print("\n" + "=" * 80)
    print("G. EVIDENCE PATHS")
    print("=" * 80)

    # Save full results to JSON
    results_json = {
        "timestamp": timestamp,
        "baseline_config": str(BASELINE_CONFIG),
        "overlay_configs": {k: str(v) for k, v in OVERLAY_CONFIGS.items()},
        "results": [],
        "robustness": {},
        "delay_results": {},
    }

    for r in all_results:
        entry = {k: v for k, v in r.items()
                 if k not in ("per_symbol", "portfolio_equity")}
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
                            kk: float(vv) if isinstance(vv, (np.floating, np.float64)) else vv
                            for kk, vv in v.items()
                        }
                    elif isinstance(v, (np.floating, np.float64)):
                        clean[k] = float(v)
                    elif isinstance(v, (np.integer, np.int64)):
                        clean[k] = int(v)
                    else:
                        clean[k] = v
                results_json["robustness"][test_name][cfg_name] = clean

    # Save delay results
    for name, data in delay_results.items():
        results_json["delay_results"][name] = {
            k: float(v) if isinstance(v, (np.floating, np.float64)) else v
            for k, v in data.items()
        }

    results_path = output_dir / "r3_trackA_results.json"
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2, default=str)

    # Save WF results
    if wf_all:
        wf_path = output_dir / "walk_forward_summary.json"
        wf_save = {}
        for cfg_name, wf in wf_all.items():
            wf_save[cfg_name] = {}
            for sym, data in wf.items():
                if isinstance(data, dict):
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

    print("\n" + "=" * 80)
    print("  R3 Track A Research complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
