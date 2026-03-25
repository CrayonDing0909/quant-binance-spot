#!/usr/bin/env python3
"""
Regime Gate Ablation Study

Compares 3 configs:
  A: Current production (scale_no_trend=0.0)
  B: Relaxed (scale_no_trend=0.3)
  C: No regime gate (disabled)

Runs portfolio backtest for each, reports SR, CAGR, MDD, TIM, turnover.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from qtrade.config import load_config
from qtrade.backtest.run_backtest import run_symbol_backtest
from qtrade.data.storage import load_klines


BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = BASE_DIR / "config" / "prod_candidate_simplified.yaml"

VARIANTS = {
    "A_prod (no_trend=0.0)": {"enabled": True, "scale_no_trend": 0.0, "scale_weak": 0.5, "scale_trending": 1.0},
    "B_relaxed (no_trend=0.3)": {"enabled": True, "scale_no_trend": 0.3, "scale_weak": 0.5, "scale_trending": 1.0},
    "C_no_gate (disabled)": {"enabled": False},
}


def _load_ref_df(cfg):
    """Load reference asset (BTC) klines for regime gate."""
    rg = getattr(cfg, '_regime_gate_cfg', None) or {}
    ref_sym = rg.get("reference_symbol", "BTCUSDT")
    mt = cfg.market_type_str
    ref_path = cfg.data_dir / "binance" / mt / cfg.market.interval / f"{ref_sym}.parquet"
    if ref_path.exists():
        return load_klines(ref_path)
    return None


def run_variant(variant_name: str, gate_override: dict) -> dict:
    """Run a full portfolio backtest with a regime gate variant."""
    cfg = load_config(CONFIG_PATH)
    ref_df = _load_ref_df(cfg)

    symbols = cfg.market.symbols
    results = {}

    for symbol in symbols:
        mt = cfg.market_type_str
        data_path = cfg.data_dir / "binance" / mt / cfg.market.interval / f"{symbol}.parquet"
        if not data_path.exists():
            print(f"  ⚠️ {symbol}: data not found at {data_path}")
            continue

        bt_cfg = cfg.to_backtest_dict(symbol)

        # Override regime gate
        if gate_override.get("enabled", False):
            rg_base = bt_cfg.get("regime_gate", {}) or {}
            rg_base.update(gate_override)
            bt_cfg["regime_gate"] = rg_base
            if ref_df is not None:
                bt_cfg["_regime_gate_ref_df"] = ref_df
        else:
            bt_cfg["regime_gate"] = {"enabled": False}

        try:
            result = run_symbol_backtest(
                symbol=symbol,
                data_path=data_path,
                cfg=bt_cfg,
            )
            results[symbol] = result
        except Exception as e:
            print(f"  ❌ {symbol}: {e}")

    return results


def compute_portfolio_stats(results: dict, cfg) -> dict:
    """Compute portfolio-level stats from per-symbol results."""
    alloc = cfg.portfolio.allocation
    equity_curves = {}

    for symbol, res in results.items():
        if res.pf is not None:
            eq = res.pf.value()
            # Normalize to returns
            eq_ret = eq.pct_change().fillna(0)
            weight = alloc.get(symbol, 0)
            equity_curves[symbol] = eq_ret * weight

    if not equity_curves:
        return {}

    port_ret = pd.DataFrame(equity_curves).sum(axis=1)
    cumret = (1 + port_ret).cumprod()

    total_ret = float(cumret.iloc[-1] - 1)
    n_years = len(port_ret) / (365.25 * 24)  # 1h bars
    cagr = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0

    # Sharpe
    sr = float(port_ret.mean() / port_ret.std() * np.sqrt(8760)) if port_ret.std() > 0 else 0

    # Max drawdown
    peak = cumret.cummax()
    dd = (cumret - peak) / peak
    mdd = float(dd.min())

    # Time in market (across symbols)
    tim_list = []
    turnover_list = []
    for symbol, res in results.items():
        if res.pos is not None:
            pos = res.pos
            tim_list.append((pos.abs() > 0.01).mean())
            turnover_list.append(pos.diff().abs().sum() / len(pos))

    avg_tim = np.mean(tim_list) if tim_list else 0
    avg_turnover = np.mean(turnover_list) if turnover_list else 0

    return {
        "SR": round(sr, 2),
        "CAGR": f"{cagr:.1%}",
        "MDD": f"{mdd:.1%}",
        "Return": f"{total_ret:.1%}",
        "Avg_TIM": f"{avg_tim:.1%}",
        "Avg_Turnover/bar": f"{avg_turnover:.5f}",
    }


def main():
    cfg = load_config(CONFIG_PATH)
    print(f"Config: {CONFIG_PATH.name}")
    print(f"Symbols: {cfg.market.symbols}")
    print(f"Period: {cfg.market.start} → {cfg.market.end or 'now'}")
    print("=" * 70)

    all_stats = {}
    all_results = {}

    for name, override in VARIANTS.items():
        print(f"\n{'─' * 50}")
        print(f"Running: {name}")
        print(f"{'─' * 50}")
        results = run_variant(name, override)
        stats = compute_portfolio_stats(results, cfg)
        all_stats[name] = stats
        all_results[name] = results
        print(f"  → {stats}")

    # Summary table
    print("\n" + "=" * 70)
    print("REGIME GATE ABLATION SUMMARY")
    print("=" * 70)
    df = pd.DataFrame(all_stats).T
    print(df.to_string())

    # Per-symbol SR comparison
    print("\n\nPer-Symbol Sharpe Ratio:")
    print("-" * 60)
    sym_sr = {}
    for name, results in all_results.items():
        sym_sr[name] = {}
        for symbol, res in results.items():
            if hasattr(res, 'stats') and res.stats:
                sr = res.stats.get("sharpe_ratio", res.stats.get("Sharpe Ratio", None))
                if sr is not None:
                    sym_sr[name][symbol] = round(float(sr), 2)
    sr_df = pd.DataFrame(sym_sr)
    print(sr_df.to_string())


if __name__ == "__main__":
    main()
