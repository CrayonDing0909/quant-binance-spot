#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════
  VP va_width_pct vs Simple Realized Vol — Quick Ablation
═══════════════════════════════════════════════════════════════

Hypothesis:
    Volume Profile EDA (20260228) found va_width_pct IC=0.024 (8/8 same sign),
    but G3 confounding analysis revealed corr(realized_vol)=0.79-0.85.
    
    This script tests whether VP provides information BEYOND simple volatility:
    - A (baseline): Current production config (no vol/VP gate)
    - B (VP gate):  Block new entries when va_width_pct pctrank < 0.3
    - C (vol gate): Block new entries when realized_vol pctrank < 0.3

    Decision rule: If B SR ≤ C SR → VP adds nothing beyond vol → FAIL

Usage:
    cd /Users/dylanting/Documents/spot_bot/quant-binance-spot
    source .venv/bin/activate
    PYTHONPATH=src python scripts/research_vp_vs_vol_ablation.py
"""
from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from qtrade.config import load_config
from qtrade.backtest.run_backtest import (
    run_symbol_backtest,
    BacktestResult,
    safe_portfolio_from_orders,
    _bps_to_pct,
    to_vbt_direction,
)
from qtrade.strategy.filters import oi_regime_filter

import logging

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("vp_vs_vol_ablation")
logger.setLevel(logging.INFO)


# ═══════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════

CONFIG_PATH = "config/prod_candidate_simplified.yaml"
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT",
]

VP_LOOKBACK = 168  # VP indicator lookback (bars)
PCTRANK_WINDOW = 720  # Rolling window for pctrank gating
MIN_PCTRANK = 0.3  # Block when pctrank < this
N_BINS = 50  # VP histogram bins


# ═══════════════════════════════════════════════════════════
#  VP and Vol Indicator Computation
# ═══════════════════════════════════════════════════════════

def _compute_vp_for_window(
    high_w: np.ndarray, low_w: np.ndarray, volume_w: np.ndarray,
    n_bins: int,
) -> float:
    """Compute VA width for a single window using np.histogram (vectorized bins)."""
    valid = (high_w > low_w) & (volume_w > 0)
    if not valid.any():
        return 0.0
    h, l, v = high_w[valid], low_w[valid], volume_w[valid]
    price_min, price_max = l.min(), h.max()
    if price_max <= price_min:
        return 0.0

    bin_edges = np.linspace(price_min, price_max, n_bins + 1)
    bin_width = bin_edges[1] - bin_edges[0]
    bin_volume = np.zeros(n_bins)

    # Vectorized per-bar: distribute volume uniformly across overlapping bins
    bar_range = h - l  # shape (m,)
    for b in range(n_bins):
        bl, bh = bin_edges[b], bin_edges[b + 1]
        overlap = np.maximum(0, np.minimum(h, bh) - np.maximum(l, bl))
        bin_volume[b] = np.sum(v * overlap / bar_range)

    total_vol = bin_volume.sum()
    if total_vol <= 0:
        return 0.0

    # Find VA: expand from POC until 70% volume
    poc_idx = np.argmax(bin_volume)
    va_lo, va_hi = poc_idx, poc_idx
    va_vol = bin_volume[poc_idx]

    while va_vol / total_vol < 0.70:
        exp_lo = bin_volume[va_lo - 1] if va_lo > 0 else 0
        exp_hi = bin_volume[va_hi + 1] if va_hi < n_bins - 1 else 0
        if exp_lo >= exp_hi and va_lo > 0:
            va_lo -= 1
            va_vol += bin_volume[va_lo]
        elif va_hi < n_bins - 1:
            va_hi += 1
            va_vol += bin_volume[va_hi]
        else:
            break

    return bin_edges[va_hi + 1] - bin_edges[va_lo]


def compute_va_width_pct(df: pd.DataFrame, lookback: int = 168, n_bins: int = 50) -> pd.Series:
    """
    Compute rolling Volume Profile VA (Value Area) width as % of close.
    
    VA = price range containing 70% of volume.
    Approximation: uniform distribution of volume across [low, high] for each 1h bar.
    
    Returns: va_width_pct = (VAH - VAL) / close, shifted by 1 for causality.
    """
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    volume = df["volume"].values
    n = len(df)

    va_width = np.full(n, np.nan)
    for i in range(lookback, n):
        s = i - lookback
        vaw = _compute_vp_for_window(high[s:i], low[s:i], volume[s:i], n_bins)
        va_width[i] = vaw / close[i] if close[i] > 0 else 0.0

    result = pd.Series(va_width, index=df.index, name="va_width_pct")
    return result.shift(1)  # Causal: use previous bar's VP


def compute_realized_vol(df: pd.DataFrame, lookback: int = 168) -> pd.Series:
    """
    Compute rolling realized volatility (std of log returns).
    Returns: realized_vol, shifted by 1 for causality.
    """
    log_ret = np.log(df["close"] / df["close"].shift(1))
    rv = log_ret.rolling(lookback).std()
    return rv.shift(1).rename("realized_vol")


# ═══════════════════════════════════════════════════════════
#  Metrics Helper
# ═══════════════════════════════════════════════════════════

def compute_equity_metrics(equity: pd.Series) -> dict:
    """Compute SR, Return%, MDD% from an equity curve."""
    rets = equity.pct_change().dropna()
    sr = rets.mean() / rets.std() * np.sqrt(8760) if rets.std() > 0 else 0.0
    total_ret = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
    cum_max = equity.cummax()
    mdd = ((equity - cum_max) / cum_max).min() * 100
    return {"sharpe": sr, "total_return": total_ret, "mdd": mdd}


def build_filtered_portfolio(
    baseline_result: BacktestResult,
    indicator: pd.Series,
    cfg_dict: dict,
) -> dict:
    """Apply gate filter to baseline positions and rebuild VBT portfolio."""
    filtered_pos = oi_regime_filter(
        baseline_result.pos,
        indicator,
        lookback=PCTRANK_WINDOW,
        min_pctrank=MIN_PCTRANK,
    )

    pf = safe_portfolio_from_orders(
        df=baseline_result.df,
        pos=filtered_pos,
        fee=_bps_to_pct(cfg_dict["fee_bps"]),
        slippage=_bps_to_pct(cfg_dict["slippage_bps"]),
        init_cash=cfg_dict["initial_cash"],
        freq=cfg_dict.get("interval", "1h"),
        direction=to_vbt_direction(cfg_dict.get("direction", "both")),
    )

    eq = pf.value()
    metrics = compute_equity_metrics(eq)
    metrics["equity"] = eq
    metrics["pos"] = filtered_pos
    metrics["time_in_market"] = (filtered_pos.abs() > 0.01).mean() * 100
    return metrics


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("  VP va_width_pct vs Simple Realized Vol — Quick Ablation")
    print("=" * 80)
    print(f"  Config: {CONFIG_PATH}")
    print(f"  VP lookback: {VP_LOOKBACK}, Pctrank window: {PCTRANK_WINDOW}")
    print(f"  Min pctrank: {MIN_PCTRANK}, VP bins: {N_BINS}")
    print(f"  Symbols: {len(SYMBOLS)}")
    print()

    cfg = load_config(CONFIG_PATH)
    cfg_dict_base = cfg.to_backtest_dict()

    # ── Step 1: Baseline backtest ──
    print("━" * 80)
    print("STEP 1: Baseline Backtest (A = current production)")
    print("━" * 80)

    baseline_results: dict[str, BacktestResult] = {}
    for sym in SYMBOLS:
        kline_path = cfg.resolve_kline_path(sym)
        if not kline_path.exists():
            logger.warning(f"Skipping {sym}: data not found at {kline_path}")
            continue

        sym_cfg = copy.deepcopy(cfg_dict_base)
        try:
            res = run_symbol_backtest(
                symbol=sym,
                data_path=kline_path,
                cfg=sym_cfg,
                data_dir=cfg.data_dir,
            )
            baseline_results[sym] = res
            eq = res.equity()
            m = compute_equity_metrics(eq)
            print(f"  {sym}: Return {m['total_return']:+.1f}%, SR {m['sharpe']:.2f}, "
                  f"MDD {m['mdd']:.1f}%")
        except Exception as e:
            logger.error(f"  {sym} failed: {e}")

    print(f"\n  ✅ {len(baseline_results)}/{len(SYMBOLS)} symbols completed\n")

    # ── Step 2: Compute VP and Vol indicators ──
    print("━" * 80)
    print("STEP 2: Computing VP and Vol Indicators")
    print("━" * 80)

    vp_indicators: dict[str, pd.Series] = {}
    vol_indicators: dict[str, pd.Series] = {}

    for sym, res in baseline_results.items():
        df = res.df
        print(f"  {sym}: computing VP (LB={VP_LOOKBACK}, bins={N_BINS})...", end="", flush=True)
        vp_indicators[sym] = compute_va_width_pct(df, lookback=VP_LOOKBACK, n_bins=N_BINS)
        print(f" done. Computing vol...", end="", flush=True)
        vol_indicators[sym] = compute_realized_vol(df, lookback=VP_LOOKBACK)
        
        valid = vp_indicators[sym].notna() & vol_indicators[sym].notna()
        corr = vp_indicators[sym][valid].corr(vol_indicators[sym][valid])
        print(f" done. corr(VP,vol)={corr:.3f}")

    # ── Step 3: Filtered backtests ──
    print("\n" + "━" * 80)
    print("STEP 3: Filtered Backtests (B=VP gate, C=vol gate)")
    print("━" * 80)

    # Use raw VBT metrics (no funding adjustment) for fair 3-way comparison
    per_sym: dict[str, dict] = {}

    for sym, res in baseline_results.items():
        sym_cfg = copy.deepcopy(cfg_dict_base)

        # A: baseline (raw VBT)
        pf_eq = res.pf.value()
        a_m = compute_equity_metrics(pf_eq)
        a_m["equity"] = pf_eq
        a_m["time_in_market"] = (res.pos.abs() > 0.01).mean() * 100

        # B: VP gate
        b_m = build_filtered_portfolio(res, vp_indicators[sym], sym_cfg)

        # C: vol gate
        c_m = build_filtered_portfolio(res, vol_indicators[sym], sym_cfg)

        per_sym[sym] = {"A": a_m, "B": b_m, "C": c_m}

    # ── Step 4: Results ──
    print("\n" + "━" * 80)
    print("STEP 4: Per-Symbol Comparison")
    print("━" * 80)

    header = (f"{'Symbol':<10}  {'A(base)SR':>10}  {'B(VP)SR':>10}  {'C(vol)SR':>10}  "
              f"{'B-A':>7}  {'C-A':>7}  {'B-C':>7}  {'B>C?':>6}")
    print(header)
    print("-" * 88)

    vp_wins = 0
    for sym in per_sym:
        sa = per_sym[sym]["A"]["sharpe"]
        sb = per_sym[sym]["B"]["sharpe"]
        sc = per_sym[sym]["C"]["sharpe"]
        bc = sb - sc
        vp_better = sb > sc
        if vp_better:
            vp_wins += 1
        print(f"{sym:<10}  {sa:10.2f}  {sb:10.2f}  {sc:10.2f}  "
              f"{sb - sa:+7.2f}  {sc - sa:+7.2f}  {bc:+7.2f}  {'✅ VP' if vp_better else '❌ Vol':>6}")

    vol_wins = len(per_sym) - vp_wins
    print("-" * 88)

    # ── Portfolio equity ──
    print("\n" + "━" * 80)
    print("STEP 5: Portfolio Comparison")
    print("━" * 80)

    weights = {}
    for sym in per_sym:
        weights[sym] = cfg.portfolio.get_weight(sym, len(per_sym))
    total_w = sum(weights.values())
    norm_w = {s: w / total_w for s, w in weights.items()}

    port = {}
    for key, label in [("A", "A (baseline)"), ("B", "B (VP gate)"), ("C", "C (vol gate)")]:
        equities = {s: per_sym[s][key]["equity"] for s in per_sym}
        min_start = max(eq.index[0] for eq in equities.values())
        max_end = min(eq.index[-1] for eq in equities.values())

        port_eq = None
        for s in per_sym:
            eq = equities[s].loc[min_start:max_end]
            eq_norm = eq / eq.iloc[0]
            if port_eq is None:
                port_eq = eq_norm * norm_w[s]
            else:
                port_eq = port_eq + eq_norm * norm_w[s]

        port_eq = port_eq * 10000
        port[key] = compute_equity_metrics(port_eq)

    print(f"\n{'Config':<20}  {'SR':>8}  {'Return%':>10}  {'MDD%':>8}")
    print("-" * 55)
    for key, label in [("A", "A (baseline)"), ("B", "B (VP gate)"), ("C", "C (vol gate)")]:
        r = port[key]
        print(f"{label:<20}  {r['sharpe']:8.2f}  {r['total_return']:+10.1f}  {r['mdd']:8.1f}")

    sr_a = port["A"]["sharpe"]
    sr_b = port["B"]["sharpe"]
    sr_c = port["C"]["sharpe"]

    incr_b = (sr_b - sr_a) / sr_a * 100 if sr_a else 0
    incr_c = (sr_c - sr_a) / sr_a * 100 if sr_a else 0

    print(f"\n{'='*80}")
    print(f"  DECISION")
    print(f"{'='*80}")
    print(f"  VP per-symbol wins: {vp_wins}/{len(per_sym)}")
    print(f"  Vol per-symbol wins: {vol_wins}/{len(per_sym)}")
    print(f"  Portfolio: VP SR={sr_b:.2f} vs Vol SR={sr_c:.2f} (Δ={sr_b - sr_c:+.2f})")
    print(f"  Baseline SR={sr_a:.2f}")
    print(f"  VP incremental vs baseline: {incr_b:+.1f}%")
    print(f"  Vol incremental vs baseline: {incr_c:+.1f}%")

    if sr_b <= sr_c:
        verdict = "FAIL"
        print(f"\n  🔴 VERDICT: FAIL — VP filter SR ({sr_b:.2f}) ≤ Vol filter SR ({sr_c:.2f})")
        print(f"     VP does NOT provide information beyond simple realized volatility.")
        print(f"     va_width_pct is a volatility regime proxy, not a genuine VP signal.")
        print(f"     Close this research direction.")
    elif vp_wins >= 5 and incr_b >= 5.0:
        verdict = "WEAK_GO"
        print(f"\n  🟢 VERDICT: WEAK GO — VP > Vol with {vp_wins}/{len(per_sym)} wins, +{incr_b:.1f}%")
    else:
        verdict = "FAIL"
        print(f"\n  🔴 VERDICT: FAIL — VP marginal ({vp_wins}/{len(per_sym)} wins, +{incr_b:.1f}% incr)")
        if sr_b > sr_c:
            print(f"     VP slightly > Vol but insufficient improvement")
        print(f"     Close this research direction.")

    # ── Save ──
    output_dir = Path("reports/research/vp_vs_vol_ablation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_json = {
        "timestamp": datetime.now().isoformat(),
        "config": CONFIG_PATH,
        "vp_lookback": VP_LOOKBACK,
        "pctrank_window": PCTRANK_WINDOW,
        "min_pctrank": MIN_PCTRANK,
        "portfolio": {k: {kk: vv for kk, vv in v.items() if kk != "equity"}
                      for k, v in port.items()},
        "per_symbol": {
            s: {
                "A_sr": per_sym[s]["A"]["sharpe"],
                "B_vp_sr": per_sym[s]["B"]["sharpe"],
                "C_vol_sr": per_sym[s]["C"]["sharpe"],
                "A_tim": per_sym[s]["A"]["time_in_market"],
                "B_tim": per_sym[s]["B"]["time_in_market"],
                "C_tim": per_sym[s]["C"]["time_in_market"],
            }
            for s in per_sym
        },
        "vp_wins": vp_wins,
        "vol_wins": vol_wins,
        "verdict": verdict,
    }
    
    with open(output_dir / "ablation_results.json", "w") as f:
        json.dump(results_json, f, indent=2, default=str)
    
    print(f"\n  📄 Results saved to {output_dir / 'ablation_results.json'}")


if __name__ == "__main__":
    main()
