"""
Entropy Regime Indicator EDA
=============================
Date: 2026-03-02
Author: Alpha Researcher
Target Gap: NEW dimension — Price Entropy / Predictability Regime
Integration Mode: Filter (Regime Gate)
Priority Score: 4.1

Hypothesis:
  Low entropy (structured, predictable) → trend signals reliable → allow trading
  High entropy (random, noisy) → trend signals fail → block new entries

Key Kill Criterion (from VP va_width_pct failure):
  corr(rolling_entropy, realized_volatility) > 0.6 → IMMEDIATE KILL
  VP had corr=0.71 with vol → was vol proxy → no alpha

Entropy measures:
  1. Permutation Entropy (PE) — Bandt & Pompe 2002, ordinal pattern randomness
  2. Shannon Entropy (SE) — return distribution randomness
  3. Approximate Entropy (ApEn) — Pincus 1991, self-similarity

× 3 lookback windows: 24h, 168h, 720h = 9 indicators + ApEn 168h only
"""

import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
import math
import time
import json
import warnings
warnings.filterwarnings("ignore")

# ── Config ──
DATA_DIR = Path("data/binance/futures/1h")
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT"]
LOOKBACKS = [24, 168, 720]
START_DATE = "2020-10-01"
VOL_CORR_KILL_THRESHOLD = 0.6
IC_THRESHOLD = 0.01  # A5

# ══════════════════════════════════════════════════════════════
# Entropy Computation Functions
# ══════════════════════════════════════════════════════════════

def rolling_permutation_entropy(series: pd.Series, window: int, m: int = 3, delay: int = 1) -> pd.Series:
    """
    Rolling Permutation Entropy (Bandt & Pompe 2002).
    PE = 0 → perfectly predictable; PE = 1 → maximally random.
    """
    values = series.values
    n = len(values)
    max_entropy = np.log2(math.factorial(m))
    result = np.full(n, np.nan)

    for i in range(window - 1, n):
        window_data = values[i - window + 1 : i + 1]
        n_patterns = len(window_data) - (m - 1) * delay
        if n_patterns < 10:
            continue

        patterns = []
        for j in range(n_patterns):
            subseq = tuple(window_data[j + k * delay] for k in range(m))
            pattern = tuple(np.argsort(subseq))
            patterns.append(pattern)

        counts = Counter(patterns)
        total = len(patterns)
        probs = np.array(list(counts.values())) / total
        entropy = -np.sum(probs * np.log2(probs))
        result[i] = entropy / max_entropy if max_entropy > 0 else np.nan

    return pd.Series(result, index=series.index, name=f"pe_{window}")


def rolling_shannon_entropy(returns: pd.Series, window: int, n_bins: int = 10) -> pd.Series:
    """
    Rolling Shannon Entropy of discretized returns.
    SE = 0 → deterministic; SE = 1 → max randomness.
    """
    values = returns.values
    n = len(values)
    max_entropy = np.log2(n_bins)
    result = np.full(n, np.nan)

    for i in range(window - 1, n):
        window_data = values[i - window + 1 : i + 1]
        valid = window_data[~np.isnan(window_data)]
        if len(valid) < window * 0.5:
            continue

        try:
            bins = np.percentile(valid, np.linspace(0, 100, n_bins + 1))
            bins[0] -= 1e-10
            bins[-1] += 1e-10
            digitized = np.digitize(valid, bins) - 1
            digitized = np.clip(digitized, 0, n_bins - 1)
            counts = np.bincount(digitized, minlength=n_bins)
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            entropy = -np.sum(probs * np.log2(probs))
            result[i] = entropy / max_entropy
        except Exception:
            continue

    return pd.Series(result, index=returns.index, name=f"se_{window}")


def rolling_approx_entropy(series: pd.Series, window: int, m: int = 2, r_factor: float = 0.2) -> pd.Series:
    """
    Rolling Approximate Entropy (Pincus 1991).
    Low ApEn → regular, predictable; High ApEn → irregular.
    Note: O(n²) per window, capped at 200 points.
    """
    values = series.values
    n = len(values)
    result = np.full(n, np.nan)
    max_apen_window = min(window, 200)

    for i in range(window - 1, n):
        window_data = values[i - max_apen_window + 1 : i + 1]
        valid = window_data[~np.isnan(window_data)]
        if len(valid) < 50:
            continue

        r = r_factor * np.std(valid)
        if r < 1e-10:
            continue

        N = len(valid)

        def _phi(m_dim):
            templates = np.array([valid[j : j + m_dim] for j in range(N - m_dim + 1)])
            n_templates = len(templates)
            counts = np.zeros(n_templates)
            for j in range(n_templates):
                diffs = np.abs(templates - templates[j])
                max_diffs = diffs.max(axis=1)
                counts[j] = np.sum(max_diffs <= r)
            counts /= n_templates
            return np.mean(np.log(counts[counts > 0]))

        try:
            phi_m = _phi(m)
            phi_m1 = _phi(m + 1)
            apen = phi_m - phi_m1
            result[i] = max(apen, 0)
        except Exception:
            continue

    return pd.Series(result, index=series.index, name=f"apen_{window}")


def compute_realized_volatility(returns: pd.Series, window: int) -> pd.Series:
    """Rolling realized volatility for confounding check."""
    return returns.rolling(window, min_periods=window // 2).std().rename(f"rvol_{window}")


# ══════════════════════════════════════════════════════════════
# Main EDA
# ══════════════════════════════════════════════════════════════

def main():
    from qtrade.data.storage import load_klines

    print("=" * 80)
    print("ENTROPY REGIME INDICATOR EDA")
    print("=" * 80)
    print(f"Symbols: {SYMBOLS}")
    print(f"Lookbacks: {LOOKBACKS}")
    print(f"Start: {START_DATE}")
    print(f"Kill threshold: |corr(entropy, vol)| > {VOL_CORR_KILL_THRESHOLD}")
    print()

    # ── Load Data ──
    print("─── Loading Data ───")
    kline_data = {}
    for sym in SYMBOLS:
        path = DATA_DIR / f"{sym}.parquet"
        if path.exists():
            df = load_klines(path)
            df = df[df.index >= START_DATE]
            kline_data[sym] = df
            print(f"  {sym}: {len(df)} bars, {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")
        else:
            print(f"  ⚠️  {sym}: NOT FOUND")

    print(f"\n✅ Loaded {len(kline_data)}/{len(SYMBOLS)} symbols\n")

    # ── Compute Entropy Indicators ──
    print("─── Computing Entropy Indicators ───")
    print("  (PE and SE for all lookbacks, ApEn for 168h only)")

    all_indicators = {}

    for sym in SYMBOLS:
        if sym not in kline_data:
            continue
        df = kline_data[sym]
        close = df["close"]
        log_ret = np.log(close / close.shift(1))

        indicators = pd.DataFrame(index=df.index)
        indicators["fwd_return"] = log_ret.shift(-1)

        t0 = time.time()

        for lb in LOOKBACKS:
            pe = rolling_permutation_entropy(close, window=lb, m=3, delay=1)
            indicators[f"pe_{lb}"] = pe

            se = rolling_shannon_entropy(log_ret, window=lb, n_bins=10)
            indicators[f"se_{lb}"] = se

            rvol = compute_realized_volatility(log_ret, window=lb)
            indicators[f"rvol_{lb}"] = rvol

        # ApEn only for 168h (O(n²) too slow for all)
        apen = rolling_approx_entropy(close, window=168, m=2, r_factor=0.2)
        indicators["apen_168"] = apen

        elapsed = time.time() - t0
        all_indicators[sym] = indicators
        n_cols = len([c for c in indicators.columns if c != "fwd_return"])
        print(f"  {sym}: {n_cols} indicators in {elapsed:.1f}s ({len(indicators)} bars)")

    print(f"\n✅ All indicators computed\n")

    # ══════════════════════════════════════════════════════════
    # 🚨 CONFOUNDING CHECK — Entropy vs Realized Volatility
    # ══════════════════════════════════════════════════════════
    print("=" * 80)
    print("🚨 CONFOUNDING CHECK: Entropy vs Realized Volatility")
    print(f"   Kill threshold: |corr| > {VOL_CORR_KILL_THRESHOLD}")
    print("=" * 80)

    entropy_cols = [
        c for c in all_indicators[SYMBOLS[0]].columns
        if c.startswith(("pe_", "se_", "apen_"))
    ]

    confounding_results = []

    for ent_col in entropy_cols:
        lb = ent_col.split("_")[-1]
        vol_col = f"rvol_{lb}"

        corrs = []
        for sym in SYMBOLS:
            if sym not in all_indicators:
                continue
            ind = all_indicators[sym]
            if ent_col in ind.columns and vol_col in ind.columns:
                c = ind[ent_col].corr(ind[vol_col])
                corrs.append(c)

        if not corrs:
            continue

        avg_corr = np.mean(corrs)
        min_corr = np.min(corrs)
        max_corr = np.max(corrs)
        killed = abs(avg_corr) > VOL_CORR_KILL_THRESHOLD

        status = "❌ KILLED" if killed else "✅ SURVIVES"
        confounding_results.append({
            "indicator": ent_col,
            "vs": vol_col,
            "avg_corr": float(avg_corr),
            "min_corr": float(min_corr),
            "max_corr": float(max_corr),
            "killed": killed,
        })
        print(f"  {ent_col:12s} vs {vol_col:10s} | avg corr = {avg_corr:+.3f} [{min_corr:+.3f} ~ {max_corr:+.3f}] | {status}")

    surviving = [r for r in confounding_results if not r["killed"]]
    killed_list = [r for r in confounding_results if r["killed"]]
    surviving_cols = [r["indicator"] for r in surviving]

    print(f"\n{'=' * 80}")
    print(f"CONFOUNDING SUMMARY: {len(surviving)}/{len(confounding_results)} survive")
    print(f"  Killed ({len(killed_list)}): {[r['indicator'] for r in killed_list]}")
    print(f"  Survived ({len(surviving)}): {surviving_cols}")

    if len(surviving) == 0:
        print("\n🔴 ALL ENTROPY MEASURES ARE VOL PROXIES → DIRECTION KILLED")
        print("   Same fate as VP va_width_pct. Entropy ≈ volatility in crypto.")

        # Save results
        _save_results(confounding_results, {}, {}, "FAIL", "All entropy measures are vol proxies (|corr|>0.6)")
        return

    # ══════════════════════════════════════════════════════════
    # Cross-Confounding: Entropy vs ATR pctrank & Momentum
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 80}")
    print("CROSS-CONFOUNDING: Entropy vs ATR_pctrank and |Momentum|")
    print("=" * 80)

    for ent_col in surviving_cols:
        lb = int(ent_col.split("_")[-1])

        corrs_atr = []
        corrs_mom = []

        for sym in SYMBOLS:
            if sym not in kline_data or sym not in all_indicators:
                continue
            df = kline_data[sym]
            ind = all_indicators[sym]

            tr = pd.concat([
                df["high"] - df["low"],
                (df["high"] - df["close"].shift(1)).abs(),
                (df["low"] - df["close"].shift(1)).abs(),
            ], axis=1).max(axis=1)
            atr = tr.rolling(lb, min_periods=lb // 2).mean()
            atr_pctrank = atr.rolling(lb, min_periods=lb // 2).rank(pct=True)

            c_atr = ind[ent_col].corr(atr_pctrank)
            corrs_atr.append(c_atr)

            mom = df["close"].pct_change(lb)
            c_mom = ind[ent_col].corr(mom.abs())
            corrs_mom.append(c_mom)

        avg_atr = np.mean(corrs_atr) if corrs_atr else 0
        avg_mom = np.mean(corrs_mom) if corrs_mom else 0

        atr_status = "⚠️  HIGH" if abs(avg_atr) > 0.5 else "✅ OK"
        mom_status = "⚠️  HIGH" if abs(avg_mom) > 0.5 else "✅ OK"

        print(f"  {ent_col:12s} vs ATR_pctrank: corr = {avg_atr:+.3f} {atr_status}")
        print(f"  {ent_col:12s} vs |Momentum| : corr = {avg_mom:+.3f} {mom_status}")
        print()

    # ══════════════════════════════════════════════════════════
    # Causal IC Analysis
    # ══════════════════════════════════════════════════════════
    print("=" * 80)
    print("CAUSAL IC ANALYSIS: signal.shift(1).corr(forward_return)")
    print(f"   A5 threshold: |IC| > {IC_THRESHOLD}")
    print("=" * 80)

    ic_results = {}

    for ent_col in surviving_cols:
        ic_results[ent_col] = {}
        print(f"\n  {ent_col}:")

        for sym in SYMBOLS:
            if sym not in all_indicators:
                continue
            ind = all_indicators[sym]
            signal = ind[ent_col].shift(1)
            fwd_ret = ind["fwd_return"]

            valid = pd.DataFrame({"signal": signal, "return": fwd_ret}).dropna()
            if len(valid) < 500:
                print(f"    {sym}: SKIP ({len(valid)} obs)")
                continue

            ic = valid["signal"].corr(valid["return"], method="spearman")
            ic_results[ent_col][sym] = float(ic)
            print(f"    {sym}: IC = {ic:+.4f} ({len(valid)} obs)")

        if ic_results[ent_col]:
            ics = list(ic_results[ent_col].values())
            avg_ic = np.mean(ics)
            same_sign = sum(1 for x in ics if np.sign(x) == np.sign(avg_ic))
            a5_pass = abs(avg_ic) > IC_THRESHOLD
            a3_pass = same_sign >= 6
            print(f"    → avg IC = {avg_ic:+.4f}, same sign: {same_sign}/{len(ics)}")
            print(f"    → A5 (|IC|>{IC_THRESHOLD}): {'✅ PASS' if a5_pass else '❌ FAIL'}")
            print(f"    → A3 (≥6/8 same sign): {'✅ PASS' if a3_pass else '❌ FAIL'}")

    # ══════════════════════════════════════════════════════════
    # Year-by-Year IC Stability (A1 + A2)
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 80}")
    print("YEAR-BY-YEAR IC STABILITY (A1 + A2)")
    print("=" * 80)

    best_indicator = None
    best_avg_ic = 0
    for ent_col in surviving_cols:
        if ent_col in ic_results and ic_results[ent_col]:
            avg = abs(np.mean(list(ic_results[ent_col].values())))
            if avg > best_avg_ic:
                best_avg_ic = avg
                best_indicator = ent_col

    a1_pass = False
    a2_pass = False

    if best_indicator is None:
        print("No valid IC results.")
    else:
        print(f"\nBest indicator: {best_indicator} (avg |IC| = {best_avg_ic:.4f})")

        # A1: Year-by-year
        print(f"\n--- A1: Year-by-Year IC ---")
        yearly_ic = {}
        for sym in SYMBOLS:
            if sym not in all_indicators:
                continue
            ind = all_indicators[sym]
            signal = ind[best_indicator].shift(1)
            fwd_ret = ind["fwd_return"]
            combined = pd.DataFrame({"signal": signal, "return": fwd_ret}).dropna()
            combined["year"] = combined.index.year
            for year, group in combined.groupby("year"):
                if len(group) < 200:
                    continue
                ic = group["signal"].corr(group["return"], method="spearman")
                if year not in yearly_ic:
                    yearly_ic[year] = []
                yearly_ic[year].append(ic)

        overall_ics = list(ic_results[best_indicator].values())
        avg_direction = np.sign(np.mean(overall_ics))
        consistent_years = 0

        for year in sorted(yearly_ic.keys()):
            ics = yearly_ic[year]
            yr_avg = np.mean(ics)
            n_same = sum(1 for x in ics if np.sign(x) == avg_direction)
            consistent = np.sign(yr_avg) == avg_direction
            if consistent:
                consistent_years += 1
            status = "✅" if consistent else "❌"
            print(f"  {year}: avg IC = {yr_avg:+.4f}, {n_same}/{len(ics)} same sign {status}")

        a1_pass = consistent_years >= 3
        print(f"\n  A1: {consistent_years}/{len(yearly_ic)} years → {'✅ PASS' if a1_pass else '❌ FAIL'}")

        # A2: Shift impact
        print(f"\n--- A2: Causal Shift Impact ---")
        unshifted_ics = []
        shifted_ics = []
        for sym in SYMBOLS:
            if sym not in all_indicators:
                continue
            ind = all_indicators[sym]
            fwd_ret = ind["fwd_return"]
            ic_no_shift = ind[best_indicator].corr(fwd_ret, method="spearman")
            ic_shifted = ind[best_indicator].shift(1).corr(fwd_ret, method="spearman")
            unshifted_ics.append(ic_no_shift)
            shifted_ics.append(ic_shifted)

        avg_unshifted = np.mean([abs(x) for x in unshifted_ics])
        avg_shifted = np.mean([abs(x) for x in shifted_ics])
        shift_impact = (avg_unshifted - avg_shifted) / avg_unshifted * 100 if avg_unshifted > 0 else 0
        a2_pass = shift_impact < 50

        print(f"  Unshifted avg |IC| = {avg_unshifted:.4f}")
        print(f"  Shifted avg |IC|   = {avg_shifted:.4f}")
        print(f"  Shift impact = {shift_impact:.1f}%")
        print(f"  A2: {'✅ PASS' if a2_pass else '❌ FAIL'} (threshold: < 50%)")

    # ══════════════════════════════════════════════════════════
    # Quintile Analysis
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 80}")
    print("QUINTILE ANALYSIS: Entropy Level → Forward Sharpe")
    print("=" * 80)

    quintile_results = {}

    if best_indicator is not None:
        print(f"\nAnalyzing: {best_indicator}")
        header = f"{'Symbol':>10} | {'Q1(Low)':>9} | {'Q2':>9} | {'Q3':>9} | {'Q4':>9} | {'Q5(High)':>9} | {'Q1-Q5':>8} | Mono?"
        print(header)
        print("-" * len(header))

        all_q_sharpes = []

        for sym in SYMBOLS:
            if sym not in all_indicators:
                continue
            ind = all_indicators[sym]
            signal = ind[best_indicator].shift(1)
            fwd_ret = ind["fwd_return"]
            combined = pd.DataFrame({"signal": signal, "return": fwd_ret}).dropna()

            try:
                combined["quintile"] = pd.qcut(combined["signal"], 5, labels=[1, 2, 3, 4, 5], duplicates="drop")
            except ValueError:
                continue

            q_sharpes = []
            for q in [1, 2, 3, 4, 5]:
                q_data = combined[combined["quintile"] == q]["return"]
                if len(q_data) > 100:
                    sr = q_data.mean() / q_data.std() * np.sqrt(8760)
                else:
                    sr = np.nan
                q_sharpes.append(sr)

            spread = q_sharpes[0] - q_sharpes[4] if not np.isnan(q_sharpes[0]) and not np.isnan(q_sharpes[4]) else np.nan
            valid_qs = [q for q in q_sharpes if not np.isnan(q)]
            if len(valid_qs) >= 4:
                diffs = [valid_qs[i + 1] - valid_qs[i] for i in range(len(valid_qs) - 1)]
                monotonic = all(d >= 0 for d in diffs) or all(d <= 0 for d in diffs)
            else:
                monotonic = False

            all_q_sharpes.append(q_sharpes)
            quintile_results[sym] = {"quintiles": q_sharpes, "spread": float(spread) if not np.isnan(spread) else None}

            qs_str = " | ".join(f"{s:>9.2f}" for s in q_sharpes)
            sp_str = f"{spread:>+8.2f}" if not np.isnan(spread) else "     N/A"
            print(f"{sym:>10} | {qs_str} | {sp_str} | {'✅' if monotonic else '❌'}")

        if all_q_sharpes:
            avg_qs = np.nanmean(all_q_sharpes, axis=0)
            avg_spread = avg_qs[0] - avg_qs[4]
            qs_str = " | ".join(f"{s:>9.2f}" for s in avg_qs)
            print(f"\n{'AVG':>10} | {qs_str} | {avg_spread:>+8.2f}")
            print(f"\nQuintile Spread (Q1_low - Q5_high) = {avg_spread:.2f} ann. Sharpe")
            if avg_spread > 0:
                print("→ Low entropy → HIGHER Sharpe → supports hypothesis")
            else:
                print("→ Low entropy → LOWER Sharpe → contradicts hypothesis")

    # ══════════════════════════════════════════════════════════
    # Binary Filter Effect (Block High Entropy)
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 80}")
    print("BINARY FILTER: Block trading when entropy pctrank > 0.80")
    print("=" * 80)

    filter_results = {}

    if best_indicator is not None:
        lb = int(best_indicator.split("_")[-1])
        print(f"\nUsing: {best_indicator}, pctrank window={lb}")
        header = f"{'Symbol':>10} | {'Unfiltered':>11} | {'Filtered':>9} | {'Δ SR':>7} | {'Freq Loss':>10} | OK?"
        print(header)
        print("-" * len(header))

        n_improved = 0
        for sym in SYMBOLS:
            if sym not in all_indicators:
                continue
            ind = all_indicators[sym]
            signal = ind[best_indicator].shift(1)
            fwd_ret = ind["fwd_return"]
            combined = pd.DataFrame({"signal": signal, "return": fwd_ret}).dropna()

            ent_pctrank = combined["signal"].rolling(lb, min_periods=lb // 2).rank(pct=True)

            all_ret = combined["return"]
            sr_all = all_ret.mean() / all_ret.std() * np.sqrt(8760)

            mask = ent_pctrank <= 0.80
            filtered_ret = combined.loc[mask, "return"]
            if len(filtered_ret) > 100:
                sr_filtered = filtered_ret.mean() / filtered_ret.std() * np.sqrt(8760)
            else:
                sr_filtered = np.nan

            delta_sr = sr_filtered - sr_all if not np.isnan(sr_filtered) else np.nan
            freq_loss = 1 - mask.mean()
            improved = delta_sr > 0 if not np.isnan(delta_sr) else False
            if improved:
                n_improved += 1

            filter_results[sym] = {
                "sr_unfiltered": float(sr_all),
                "sr_filtered": float(sr_filtered) if not np.isnan(sr_filtered) else None,
                "delta_sr": float(delta_sr) if not np.isnan(delta_sr) else None,
                "freq_loss": float(freq_loss),
            }

            sr_f_str = f"{sr_filtered:>9.2f}" if not np.isnan(sr_filtered) else "      N/A"
            d_str = f"{delta_sr:>+7.2f}" if not np.isnan(delta_sr) else "    N/A"
            print(f"{sym:>10} | {sr_all:>11.2f} | {sr_f_str} | {d_str} | {freq_loss:>9.1%} | {'✅' if improved else '❌'}")

        print(f"\n✅ Improved: {n_improved}/{len(SYMBOLS)} symbols")

    # ══════════════════════════════════════════════════════════
    # Indicator Summary Table
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 80}")
    print("INDICATOR COMPARISON SUMMARY")
    print("=" * 80)

    header = f"{'Indicator':>12} | {'Avg IC':>8} | {'|IC|':>6} | {'Same Sign':>10} | {'Vol Corr':>9} | {'A3':>3} | {'A5':>3} | Score"
    print(header)
    print("-" * len(header))

    for ent_col in surviving_cols:
        if ent_col not in ic_results or not ic_results[ent_col]:
            continue
        ics = list(ic_results[ent_col].values())
        avg_ic = np.mean(ics)
        abs_ic = abs(avg_ic)
        same_sign = sum(1 for x in ics if np.sign(x) == np.sign(avg_ic))
        vol_corr = [r for r in confounding_results if r["indicator"] == ent_col][0]["avg_corr"]
        a3 = same_sign >= 6
        a5 = abs_ic > IC_THRESHOLD
        score = abs_ic * 100 * (same_sign / 8) * (1 - abs(vol_corr))
        print(f"{ent_col:>12} | {avg_ic:>+8.4f} | {abs_ic:>6.4f} | {same_sign:>7}/8   | {vol_corr:>+9.3f} | {'✅' if a3 else '❌':>3} | {'✅' if a5 else '❌':>3} | {score:>5.3f}")

    # ══════════════════════════════════════════════════════════
    # FINAL VERDICT
    # ══════════════════════════════════════════════════════════
    print(f"\n{'=' * 80}")
    print("FINAL VERDICT: Entropy Regime Indicator")
    print("=" * 80)

    n_killed = len(killed_list)
    n_total = len(confounding_results)
    print(f"\n  Confounding: {len(surviving)}/{n_total} survive (|corr(vol)| < {VOL_CORR_KILL_THRESHOLD})")

    if len(surviving_cols) == 0:
        verdict = "FAIL"
        reason = f"All {n_total} entropy measures are vol proxies (|corr| > {VOL_CORR_KILL_THRESHOLD})"
        print(f"\n  🔴 VERDICT: {verdict}")
        print(f"     {reason}")
        _save_results(confounding_results, ic_results, filter_results, verdict, reason)
        return

    # Check quality gates for best indicator
    any_pass = False
    if best_indicator and best_indicator in ic_results and ic_results[best_indicator]:
        ics = list(ic_results[best_indicator].values())
        avg_ic = np.mean(ics)
        same_sign = sum(1 for x in ics if np.sign(x) == np.sign(avg_ic))
        a3 = same_sign >= 6
        a5 = abs(avg_ic) > IC_THRESHOLD
        any_pass = a3 and a5

    if any_pass:
        vol_corr_val = [r for r in confounding_results if r["indicator"] == best_indicator][0]["avg_corr"]
        print(f"\n  Best: {best_indicator}")
        print(f"  Avg IC: {avg_ic:+.4f}")
        print(f"  Vol corr: {vol_corr_val:+.3f}")
        print(f"  A1 (yearly): {'✅' if a1_pass else '❌'}, A2 (shift): {'✅' if a2_pass else '❌'}, A3 (cross-sym): ✅, A5 (strength): ✅")

        if a1_pass and a2_pass:
            verdict = "WEAK GO"
            reason = f"Best={best_indicator}, IC={avg_ic:+.4f}, vol_corr={vol_corr_val:+.3f}, A1-A5 all PASS"
            print(f"\n  🟡 VERDICT: WEAK GO — Entropy has signal beyond vol")
            print(f"     → Handoff Quant Dev: ablation A/B/C")
        else:
            gates_failed = []
            if not a1_pass:
                gates_failed.append("A1(yearly)")
            if not a2_pass:
                gates_failed.append("A2(shift)")
            verdict = "FAIL"
            reason = f"Best={best_indicator}, IC={avg_ic:+.4f}, but {', '.join(gates_failed)} FAIL"
            print(f"\n  🔴 VERDICT: FAIL — {', '.join(gates_failed)} failed")
    else:
        verdict = "FAIL"
        reason = "No indicator passes both A3 (≥6/8 same sign) and A5 (|IC| > 0.01)"
        print(f"\n  🔴 VERDICT: {verdict}")
        print(f"     {reason}")

    _save_results(confounding_results, ic_results, filter_results, verdict, reason)

    print(f"\n{'=' * 80}")
    print(f"VERDICT: {verdict}")
    print(f"Reason: {reason}")
    print("=" * 80)


def _save_results(confounding, ic_results, filter_results, verdict, reason):
    """Save results to JSON for reproducibility."""
    out_dir = Path("reports/research/entropy_regime_eda")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "date": "2026-03-02",
        "author": "Alpha Researcher",
        "verdict": verdict,
        "reason": reason,
        "confounding_check": confounding,
        "ic_results": ic_results,
        "filter_results": filter_results,
    }

    out_path = out_dir / "entropy_regime_eda_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n📄 Results saved to: {out_path}")


if __name__ == "__main__":
    main()
