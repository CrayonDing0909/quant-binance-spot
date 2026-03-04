#!/usr/bin/env python3
# EMBARGO_EXEMPT: EDA uses aggTrades data (not kline), temporal embargo not applicable to pre-aggregated tick data
"""
Tick-level OFI EDA — Alpha Research #22

Research question: Do tick-level microstructure features from aggTrades
provide alpha beyond the failed proxy OFI (taker_vol_ratio)?

Key features to test:
    1. Raw tick OFI (buy-sell)/total from aggTrades
    2. avg_trade_size = total_volume / num_trades (whale detection)
    3. OFI cumulative (rolling sum over various windows)
    4. OFI z-score and percentile rank
    5. Comparison: tick OFI vs proxy OFI

Quality Gates (Alpha Research Map A1-A5, G1-G6):
    A1: Year-by-year consistency (≥5 years same sign)
    A2: Shift sensitivity (<15% IC change with 1-bar shift)
    A3: Cross-symbol consistency (≥6/8 same sign)
    A5: IC ≥ 0.01
    G3: Confounding check (|corr(TSMOM)| < 0.3, |corr(ATR)| < 0.6)
    G6: Independence from existing signals

Data: data/binance/futures/aggtrades_agg/*_ofi.parquet + *_hourly.parquet
      data/binance/futures/1h/*.parquet (klines for forward returns)

EMBARGO_EXEMPT: This is new research using aggTrades data (not temporal embargo scope)

Usage:
    PYTHONPATH=src python scripts/research_tick_ofi_eda.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("tick_ofi_eda")

# ══════════════════════════════════════════════════════════════
#  Config
# ══════════════════════════════════════════════════════════════

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "LINKUSDT",
           "ADAUSDT", "BNBUSDT", "XRPUSDT"]
AGGTRADES_DIR = Path("data/binance/futures/aggtrades_agg")
KLINE_DIR = Path("data/binance/futures/1h")
DERIV_DIR = Path("data/binance/futures/derivatives/taker_vol_ratio")
REPORT_DIR = Path("reports/research/tick_ofi_eda")

FORWARD_BARS = [6, 24, 72]  # forward return horizons (hours)
IC_HORIZONS = [24]           # primary IC horizon for ranking

# Feature definitions: (name, compute_fn)
# Each compute_fn takes (ofi: Series, hourly: DataFrame, kline: DataFrame) -> Series
LOOKBACKS = [6, 24, 72, 168, 720]


def _safe_spearmanr(x, y):
    """Spearman rank correlation with NaN handling."""
    mask = x.notna() & y.notna()
    if mask.sum() < 100:
        return np.nan, 1.0
    return spearmanr(x[mask], y[mask])


# ══════════════════════════════════════════════════════════════
#  Feature Engineering
# ══════════════════════════════════════════════════════════════

def compute_features(ofi: pd.Series, hourly: pd.DataFrame,
                     kline: pd.DataFrame) -> pd.DataFrame:
    """Compute all OFI-derived features."""
    features = pd.DataFrame(index=ofi.index)

    # 1. Raw OFI
    features["ofi_raw"] = ofi

    # 2. Average trade size (USDT per trade)
    avg_trade_size = hourly["total_volume"] / hourly["num_trades"].replace(0, np.nan)
    features["avg_trade_size"] = avg_trade_size.reindex(ofi.index)

    # 3. num_trades
    features["num_trades"] = hourly["num_trades"].reindex(ofi.index)

    # 4. OFI cumulative (rolling sum)
    for lb in LOOKBACKS:
        features[f"ofi_cum_{lb}"] = ofi.rolling(lb, min_periods=max(lb // 2, 1)).sum()

    # 5. OFI z-score
    for lb in [168, 720]:
        mu = ofi.rolling(lb, min_periods=max(lb // 2, 1)).mean()
        sigma = ofi.rolling(lb, min_periods=max(lb // 2, 1)).std()
        features[f"ofi_zscore_{lb}"] = (ofi - mu) / sigma.replace(0, np.nan)

    # 6. OFI percentile rank
    for lb in [168, 720]:
        features[f"ofi_pctrank_{lb}"] = ofi.rolling(
            lb, min_periods=max(lb // 2, 1)
        ).rank(pct=True)

    # 7. avg_trade_size percentile rank (whale detection)
    for lb in [168, 720]:
        features[f"ats_pctrank_{lb}"] = avg_trade_size.reindex(ofi.index).rolling(
            lb, min_periods=max(lb // 2, 1)
        ).rank(pct=True)

    # 8. OFI momentum (change in OFI level)
    for lb in [24, 72]:
        features[f"ofi_mom_{lb}"] = ofi.rolling(lb).mean().diff(lb)

    # 9. Volume imbalance acceleration
    features["ofi_accel_24"] = ofi.rolling(24).mean() - ofi.rolling(72).mean()

    return features


# ══════════════════════════════════════════════════════════════
#  IC Analysis
# ══════════════════════════════════════════════════════════════

def compute_ic_table(features: pd.DataFrame, kline: pd.DataFrame,
                     forward_bars_list: list[int]) -> pd.DataFrame:
    """Compute IC for all features × forward horizons."""
    close = kline["close"].reindex(features.index)
    results = []

    for fwd in forward_bars_list:
        fwd_ret = close.pct_change(fwd, fill_method=None).shift(-fwd)

        for col in features.columns:
            ic, pval = _safe_spearmanr(features[col], fwd_ret)
            results.append({
                "feature": col,
                "forward_bars": fwd,
                "ic": ic,
                "pval": pval,
                "n_obs": features[col].notna().sum(),
            })

    return pd.DataFrame(results)


def compute_yearly_ic(feature: pd.Series, fwd_ret: pd.Series) -> dict:
    """Compute IC per year for A1 stability check."""
    yearly = {}
    for year in sorted(feature.index.year.unique()):
        mask = feature.index.year == year
        f_year = feature[mask]
        r_year = fwd_ret.reindex(f_year.index)
        ic, _ = _safe_spearmanr(f_year, r_year)
        if not np.isnan(ic):
            yearly[int(year)] = round(ic, 6)
    return yearly


def compute_shift_sensitivity(feature: pd.Series, fwd_ret: pd.Series) -> float:
    """A2: How much does IC change if we shift the feature by 1 bar?"""
    ic_orig, _ = _safe_spearmanr(feature, fwd_ret)
    ic_shifted, _ = _safe_spearmanr(feature.shift(1), fwd_ret)
    if abs(ic_orig) < 1e-6:
        return 0.0
    return abs(ic_shifted - ic_orig) / abs(ic_orig)


# ══════════════════════════════════════════════════════════════
#  Confounding Analysis
# ══════════════════════════════════════════════════════════════

def compute_confounding(feature: pd.Series, kline: pd.DataFrame,
                        ofi_proxy: pd.Series | None = None) -> dict:
    """Check correlation with existing signals (G3/G6)."""
    close = kline["close"].reindex(feature.index)
    returns = close.pct_change(24, fill_method=None).fillna(0.0)  # 24h momentum (TSMOM proxy)
    atr = (kline["high"] - kline["low"]).reindex(feature.index)
    atr_pctrank = atr.rolling(720, min_periods=100).rank(pct=True)

    result = {}
    for name, ref in [("tsmom_24h", returns), ("atr_pctrank", atr_pctrank)]:
        c, _ = _safe_spearmanr(feature, ref)
        result[name] = round(c, 4) if not np.isnan(c) else None

    if ofi_proxy is not None:
        c, _ = _safe_spearmanr(feature, ofi_proxy.reindex(feature.index))
        result["proxy_ofi"] = round(c, 4) if not np.isnan(c) else None

    return result


# ══════════════════════════════════════════════════════════════
#  Quintile Analysis
# ══════════════════════════════════════════════════════════════

def quintile_analysis(feature: pd.Series, fwd_ret: pd.Series,
                      n_quantiles: int = 5) -> dict:
    """Compute quintile spread and monotonicity."""
    valid = pd.DataFrame({"f": feature, "r": fwd_ret}).dropna()
    if len(valid) < 500:
        return {"spread": np.nan, "monotonic": False, "quintile_means": {}}

    valid["q"] = pd.qcut(valid["f"], n_quantiles, labels=False, duplicates="drop")
    qmeans = valid.groupby("q")["r"].mean()

    spread = qmeans.iloc[-1] - qmeans.iloc[0]
    # Check monotonicity
    diffs = qmeans.diff().iloc[1:]
    monotonic = (diffs > 0).all() or (diffs < 0).all()

    return {
        "spread": round(float(spread) * np.sqrt(8760 / 24), 4),  # annualized
        "monotonic": bool(monotonic),
        "quintile_means": {int(k): round(float(v), 6) for k, v in qmeans.items()},
    }


# ══════════════════════════════════════════════════════════════
#  Main EDA
# ══════════════════════════════════════════════════════════════

def run_eda():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}
    top_features_per_sym = {}  # symbol → best feature name + IC

    print("=" * 70)
    print("  Tick-level OFI EDA — Alpha Research #22")
    print("=" * 70)
    print()

    # ── Phase 1: Per-symbol feature IC scan ──
    sym_ic_tables = {}
    sym_feature_data = {}  # for cross-symbol consistency later

    for sym in SYMBOLS:
        ofi_path = AGGTRADES_DIR / f"{sym}_ofi.parquet"
        hourly_path = AGGTRADES_DIR / f"{sym}_hourly.parquet"
        kline_path = KLINE_DIR / f"{sym}.parquet"

        if not ofi_path.exists() or not kline_path.exists():
            logger.warning(f"⚠️  {sym}: Missing data, skipping")
            continue

        logger.info(f"📊 Processing {sym}...")

        ofi = pd.read_parquet(ofi_path)
        ofi = ofi.iloc[:, 0] if isinstance(ofi, pd.DataFrame) else ofi
        hourly = pd.read_parquet(hourly_path) if hourly_path.exists() else pd.DataFrame()
        kline = pd.read_parquet(kline_path)

        # Compute features
        features = compute_features(ofi, hourly, kline)

        # IC table
        ic_table = compute_ic_table(features, kline, FORWARD_BARS)
        sym_ic_tables[sym] = ic_table

        # Store feature data for cross-symbol analysis
        sym_feature_data[sym] = (features, kline, ofi)

        # Load proxy OFI for comparison
        proxy_path = DERIV_DIR / f"{sym}.parquet"
        ofi_proxy = None
        if proxy_path.exists():
            tvr = pd.read_parquet(proxy_path)
            tvr_ratio = tvr.iloc[:, 0]
            ofi_proxy = (tvr_ratio - 1.0) / (tvr_ratio + 1.0)

        # Best feature at 24h
        ic_24 = ic_table[ic_table["forward_bars"] == 24].copy()
        ic_24["abs_ic"] = ic_24["ic"].abs()
        best = ic_24.nlargest(5, "abs_ic")

        print(f"\n{'─' * 50}")
        print(f"  {sym} — Top 5 features (24h forward)")
        print(f"{'─' * 50}")
        for _, row in best.iterrows():
            sig = "***" if row["pval"] < 0.01 else "** " if row["pval"] < 0.05 else "   "
            print(f"  {row['feature']:25s}  IC={row['ic']:+.6f}  p={row['pval']:.4f} {sig}")

        top_features_per_sym[sym] = {
            "best_feature": best.iloc[0]["feature"],
            "best_ic": round(best.iloc[0]["ic"], 6),
            "best_pval": round(best.iloc[0]["pval"], 4),
        }

        # Proxy comparison
        if ofi_proxy is not None:
            common = ofi.index.intersection(ofi_proxy.index)
            fwd_ret = kline["close"].pct_change(24, fill_method=None).shift(-24)
            common_all = common.intersection(fwd_ret.dropna().index)
            ic_tick, _ = _safe_spearmanr(ofi.loc[common_all], fwd_ret.loc[common_all])
            ic_proxy, _ = _safe_spearmanr(ofi_proxy.loc[common_all], fwd_ret.loc[common_all])
            tick_proxy_corr = ofi.loc[common].corr(ofi_proxy.loc[common])
            print(f"  ── Tick vs Proxy comparison ──")
            print(f"  corr(tick, proxy) = {tick_proxy_corr:.4f}")
            print(f"  Tick OFI IC = {ic_tick:+.6f}, Proxy OFI IC = {ic_proxy:+.6f}")

    # ── Phase 2: Cross-symbol consistency (A3) ──
    print(f"\n{'=' * 70}")
    print("  Phase 2: Cross-Symbol Consistency (A3)")
    print(f"{'=' * 70}")

    # Get all unique features
    if not sym_ic_tables:
        logger.error("No symbols processed!")
        return

    all_features = sym_ic_tables[list(sym_ic_tables.keys())[0]][
        sym_ic_tables[list(sym_ic_tables.keys())[0]]["forward_bars"] == 24
    ]["feature"].unique()

    cross_symbol_results = {}
    for feat in all_features:
        signs = {}
        ics = {}
        for sym, ic_tab in sym_ic_tables.items():
            row = ic_tab[(ic_tab["feature"] == feat) & (ic_tab["forward_bars"] == 24)]
            if not row.empty:
                ic_val = row.iloc[0]["ic"]
                if not np.isnan(ic_val):
                    signs[sym] = "+" if ic_val > 0 else "-"
                    ics[sym] = ic_val

        if not signs:
            continue

        n_pos = sum(1 for s in signs.values() if s == "+")
        n_neg = len(signs) - n_pos
        same_sign = max(n_pos, n_neg)
        avg_ic = np.mean(list(ics.values()))
        dominant = "+" if n_pos >= n_neg else "-"

        cross_symbol_results[feat] = {
            "avg_ic": round(avg_ic, 6),
            "same_sign": f"{same_sign}/{len(signs)}",
            "dominant_sign": dominant,
            "signs": signs,
            "ics": {k: round(v, 6) for k, v in ics.items()},
        }

    # Sort by |avg_ic|
    sorted_features = sorted(
        cross_symbol_results.items(),
        key=lambda x: abs(x[1]["avg_ic"]),
        reverse=True,
    )

    print(f"\n  Top 15 features by |avg IC| (24h forward):")
    print(f"  {'Feature':25s}  {'Avg IC':>10s}  {'Same Sign':>10s}  {'A3':>4s}  {'A5':>4s}")
    print(f"  {'─' * 65}")
    for feat, data in sorted_features[:15]:
        a3 = "✅" if int(data["same_sign"].split("/")[0]) >= 6 else "❌"
        a5 = "✅" if abs(data["avg_ic"]) >= 0.01 else "❌"
        print(f"  {feat:25s}  {data['avg_ic']:+10.6f}  {data['same_sign']:>10s}  {a3:>4s}  {a5:>4s}")

    # ── Phase 3: Year-by-year stability (A1) for top features ──
    print(f"\n{'=' * 70}")
    print("  Phase 3: Year-by-Year Stability (A1) — Top Features")
    print(f"{'=' * 70}")

    top_n = 5
    top_feature_names = [f for f, _ in sorted_features[:top_n]]
    yearly_results = {}

    for feat_name in top_feature_names:
        yearly_results[feat_name] = {}
        for sym, (features, kline, _ofi) in sym_feature_data.items():
            if feat_name not in features.columns:
                continue
            fwd_ret = kline["close"].pct_change(24, fill_method=None).shift(-24)
            yearly = compute_yearly_ic(features[feat_name], fwd_ret)
            yearly_results[feat_name][sym] = yearly

        # Aggregate across symbols
        all_years = set()
        for sym_yearly in yearly_results[feat_name].values():
            all_years.update(sym_yearly.keys())

        print(f"\n  {feat_name}:")
        for year in sorted(all_years):
            year_ics = [
                sym_yearly[year]
                for sym_yearly in yearly_results[feat_name].values()
                if year in sym_yearly
            ]
            if year_ics:
                avg = np.mean(year_ics)
                n_pos = sum(1 for ic in year_ics if ic > 0)
                print(f"    {year}: avg IC={avg:+.4f}, {n_pos}/{len(year_ics)} positive")

    # ── Phase 4: Confounding Analysis (G3/G6) ──
    print(f"\n{'=' * 70}")
    print("  Phase 4: Confounding Analysis (G3/G6)")
    print(f"{'=' * 70}")

    confounding_results = {}
    for feat_name in top_feature_names:
        conf_all = {}
        for sym, (features, kline, _ofi) in sym_feature_data.items():
            if feat_name not in features.columns:
                continue

            proxy_path = DERIV_DIR / f"{sym}.parquet"
            ofi_proxy = None
            if proxy_path.exists():
                tvr = pd.read_parquet(proxy_path)
                ofi_proxy = (tvr.iloc[:, 0] - 1.0) / (tvr.iloc[:, 0] + 1.0)

            conf = compute_confounding(features[feat_name], kline, ofi_proxy)
            conf_all[sym] = conf

        # Average confounding across symbols
        avg_conf = {}
        for key in ["tsmom_24h", "atr_pctrank", "proxy_ofi"]:
            vals = [c[key] for c in conf_all.values() if key in c and c[key] is not None]
            if vals:
                avg_conf[key] = round(np.mean(vals), 4)

        confounding_results[feat_name] = avg_conf
        print(f"\n  {feat_name}:")
        for key, val in avg_conf.items():
            flag = "⚠️" if abs(val) > 0.3 else "✅"
            print(f"    corr({key}) = {val:+.4f} {flag}")

    # ── Phase 5: Quintile Analysis ──
    print(f"\n{'=' * 70}")
    print("  Phase 5: Quintile Analysis — Top Features")
    print(f"{'=' * 70}")

    quintile_results = {}
    for feat_name in top_feature_names:
        q_all = {}
        for sym, (features, kline, _ofi) in sym_feature_data.items():
            if feat_name not in features.columns:
                continue
            fwd_ret = kline["close"].pct_change(24, fill_method=None).shift(-24)
            qa = quintile_analysis(features[feat_name], fwd_ret)
            q_all[sym] = qa

        spreads = [q["spread"] for q in q_all.values() if not np.isnan(q["spread"])]
        mono = [q["monotonic"] for q in q_all.values()]
        quintile_results[feat_name] = {
            "avg_spread": round(np.mean(spreads), 4) if spreads else None,
            "n_monotonic": sum(mono),
            "n_total": len(mono),
            "per_symbol": {k: round(v["spread"], 4) for k, v in q_all.items()
                           if not np.isnan(v["spread"])},
        }

        print(f"\n  {feat_name}:")
        print(f"    Avg quintile spread (annualized): {quintile_results[feat_name]['avg_spread']}")
        print(f"    Monotonic: {sum(mono)}/{len(mono)}")
        for sym, qa in q_all.items():
            print(f"      {sym}: spread={qa['spread']:+.4f} {'📈' if qa['monotonic'] else ''}")

    # ── Phase 6: Shift Sensitivity (A2) ──
    print(f"\n{'=' * 70}")
    print("  Phase 6: Shift Sensitivity (A2)")
    print(f"{'=' * 70}")

    shift_results = {}
    for feat_name in top_feature_names:
        shifts = {}
        for sym, (features, kline, _ofi) in sym_feature_data.items():
            if feat_name not in features.columns:
                continue
            fwd_ret = kline["close"].pct_change(24, fill_method=None).shift(-24)
            shift_pct = compute_shift_sensitivity(features[feat_name], fwd_ret)
            shifts[sym] = round(shift_pct, 4)

        avg_shift = np.mean(list(shifts.values())) if shifts else None
        shift_results[feat_name] = {
            "avg_shift_pct": round(avg_shift, 4) if avg_shift is not None else None,
            "per_symbol": shifts,
        }
        a2_flag = "✅" if avg_shift is not None and avg_shift < 0.15 else "❌"
        print(f"  {feat_name}: avg shift impact = {avg_shift:.1%} {a2_flag}")

    # ── Phase 7: Quality Gates Summary ──
    print(f"\n{'=' * 70}")
    print("  Phase 7: Quality Gates Summary")
    print(f"{'=' * 70}")

    gate_summary = {}
    for feat_name in top_feature_names:
        cs = cross_symbol_results.get(feat_name, {})
        avg_ic = cs.get("avg_ic", 0)
        same_sign = cs.get("same_sign", "0/0")
        n_same = int(same_sign.split("/")[0]) if "/" in same_sign else 0

        # Yearly consistency (A1): check if 5+ years same sign
        yearly = yearly_results.get(feat_name, {})
        all_year_ics = {}
        for sym_yearly in yearly.values():
            for yr, ic in sym_yearly.items():
                all_year_ics.setdefault(yr, []).append(ic)
        avg_yearly = {yr: np.mean(ics) for yr, ics in all_year_ics.items()}
        n_pos_years = sum(1 for v in avg_yearly.values() if v > 0)
        n_neg_years = sum(1 for v in avg_yearly.values() if v < 0)
        a1_consistent_years = max(n_pos_years, n_neg_years)

        conf = confounding_results.get(feat_name, {})
        shift = shift_results.get(feat_name, {})

        gates = {
            "A1_yearly": f"{a1_consistent_years}/{len(avg_yearly)} same sign",
            "A1_pass": a1_consistent_years >= 5,
            "A2_shift": shift.get("avg_shift_pct"),
            "A2_pass": shift.get("avg_shift_pct", 1.0) < 0.15,
            "A3_cross_symbol": same_sign,
            "A3_pass": n_same >= 6,
            "A5_ic": round(avg_ic, 6),
            "A5_pass": abs(avg_ic) >= 0.01,
            "G3_tsmom_corr": conf.get("tsmom_24h"),
            "G3_atr_corr": conf.get("atr_pctrank"),
            "G3_pass": (
                abs(conf.get("tsmom_24h", 0)) < 0.3
                and abs(conf.get("atr_pctrank", 0)) < 0.6
            ),
        }

        n_pass = sum(1 for k, v in gates.items() if k.endswith("_pass") and v)
        n_total = sum(1 for k in gates if k.endswith("_pass"))
        gates["total_pass"] = f"{n_pass}/{n_total}"

        gate_summary[feat_name] = gates

        print(f"\n  {feat_name}:")
        print(f"    A1 Year-by-year: {gates['A1_yearly']} {'✅' if gates['A1_pass'] else '❌'}")
        print(f"    A2 Shift sens:   {gates['A2_shift']:.1%} {'✅' if gates['A2_pass'] else '❌'}" if gates['A2_shift'] is not None else "    A2 Shift sens:   N/A")
        print(f"    A3 Cross-symbol: {gates['A3_cross_symbol']} {'✅' if gates['A3_pass'] else '❌'}")
        print(f"    A5 IC ≥ 0.01:   {gates['A5_ic']:+.6f} {'✅' if gates['A5_pass'] else '❌'}")
        print(f"    G3 Confounding:  tsmom={gates['G3_tsmom_corr']}, atr={gates['G3_atr_corr']} {'✅' if gates['G3_pass'] else '❌'}")
        print(f"    ── Total: {gates['total_pass']} gates PASS ──")

    # ── Save Report ──
    report = {
        "metadata": {
            "symbols": SYMBOLS,
            "n_symbols_processed": len(sym_ic_tables),
            "forward_bars": FORWARD_BARS,
        },
        "top_features_per_symbol": top_features_per_sym,
        "cross_symbol_top15": {
            f: d for f, d in sorted_features[:15]
        },
        "yearly_ic": {
            feat: {
                sym: yearly
                for sym, yearly in sym_data.items()
            }
            for feat, sym_data in yearly_results.items()
        },
        "confounding": confounding_results,
        "quintile": quintile_results,
        "shift_sensitivity": shift_results,
        "quality_gates": gate_summary,
    }

    report_path = REPORT_DIR / "tick_ofi_eda_results.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n💾 Report saved: {report_path}")

    # ── Final Verdict ──
    print(f"\n{'=' * 70}")
    print("  VERDICT")
    print(f"{'=' * 70}")

    # Find the best feature overall
    if sorted_features:
        best_feat, best_data = sorted_features[0]
        best_gates = gate_summary.get(best_feat, {})
        n_pass = int(best_gates.get("total_pass", "0/0").split("/")[0])
        n_total = int(best_gates.get("total_pass", "0/0").split("/")[1])

        print(f"\n  Best feature: {best_feat}")
        print(f"  Avg IC: {best_data['avg_ic']:+.6f}")
        print(f"  Cross-symbol: {best_data['same_sign']}")
        print(f"  Quality gates: {best_gates.get('total_pass', 'N/A')}")

        if n_pass >= 4:
            print(f"\n  → WEAK GO: Proceed to ablation")
        elif n_pass >= 3:
            print(f"\n  → BORDERLINE: Need more investigation")
        else:
            print(f"\n  → FAIL: Insufficient alpha")

    return report


if __name__ == "__main__":
    run_eda()
