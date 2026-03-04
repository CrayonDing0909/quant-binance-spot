#!/usr/bin/env python3
"""
Macro Cross-Market Regime EDA — Alpha Researcher
==================================================
研究目標：Macro risk regime (VIX/DXY/US10Y/SPY/Gold) 是否提供
crypto returns 的信息，且不被 HTF filter / TVL 所捕捉。

核心假說 H1：
  macro risk-off 時 crypto 表現顯著較差 → 可作為 portfolio-level filter

5-Factor Score: 3.9（高優先級）
  分散化=5, 數據=4, Alpha=2(未知), 複雜度=4, 文獻=4

Integration Target: Portfolio Layer / Filter

Usage:
  PYTHONPATH=src python scripts/research_macro_regime_eda.py
"""
import json
import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT"]
DATA_DIR = Path("data/binance/futures/1h")
ONCHAIN_DIR = Path("data/onchain/defillama")
REPORT_DIR = Path("reports/research/macro_regime_eda")

# Macro tickers (yfinance)
MACRO_TICKERS = {
    "VIX": "^VIX",          # CBOE Volatility Index
    "SPY": "SPY",            # S&P 500 ETF
    "QQQ": "QQQ",            # NASDAQ 100 ETF
    "DXY": "DX-Y.NYB",      # US Dollar Index
    "US10Y": "^TNX",         # US 10-Year Treasury Yield
    "GLD": "GLD",            # Gold ETF
}

# Regime indicator lookbacks
MOMENTUM_WINDOWS = [30, 60, 90]  # calendar days
PCTRANK_WINDOW = 252  # ~1 trading year for daily data
IC_SHIFT = 1  # causal shift: signal at day t → return at day t+1

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════════════
# Phase 1: Data Acquisition
# ════════════════════════════════════════════════════════════
def download_macro_data(start: str = "2019-01-01", end: str = "2026-03-01") -> pd.DataFrame:
    """Download macro data via yfinance."""
    import yfinance as yf

    logger.info("📥 Downloading macro data via yfinance...")
    frames = {}
    for name, ticker in MACRO_TICKERS.items():
        logger.info(f"  Downloading {name} ({ticker})...")
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if df.empty:
                logger.warning(f"  ⚠️ {name}: empty data, trying alternative...")
                if name == "DXY":
                    # Fallback: UUP ETF as DXY proxy
                    df = yf.download("UUP", start=start, end=end, progress=False, auto_adjust=True)
                    name = "DXY_UUP"
            if not df.empty:
                frames[name] = df["Close"].squeeze()
                logger.info(f"  ✅ {name}: {len(df)} bars, {df.index[0].date()} → {df.index[-1].date()}")
            else:
                logger.warning(f"  ❌ {name}: no data available")
        except Exception as e:
            logger.error(f"  ❌ {name}: download failed: {e}")

    if not frames:
        raise RuntimeError("No macro data downloaded!")

    macro_df = pd.DataFrame(frames)
    macro_df.index = pd.to_datetime(macro_df.index).tz_localize(None)
    macro_df = macro_df.sort_index()

    logger.info(f"\n📊 Macro data summary:")
    logger.info(f"  Date range: {macro_df.index[0].date()} → {macro_df.index[-1].date()}")
    logger.info(f"  Columns: {list(macro_df.columns)}")
    logger.info(f"  Coverage:\n{macro_df.notna().sum()}")

    return macro_df


def load_crypto_data() -> dict[str, pd.DataFrame]:
    """Load 1h crypto OHLCV data."""
    logger.info("\n📥 Loading crypto 1h OHLCV data...")
    data = {}
    for sym in SYMBOLS:
        fp = DATA_DIR / f"{sym}.parquet"
        if fp.exists():
            df = pd.read_parquet(fp)
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            data[sym] = df
            logger.info(f"  ✅ {sym}: {len(df)} bars, {df.index[0].date()} → {df.index[-1].date()}")
        else:
            logger.warning(f"  ❌ {sym}: file not found")
    return data


def resample_crypto_to_daily(crypto_data: dict) -> pd.DataFrame:
    """Resample 1h crypto to daily close prices and returns."""
    daily_close = {}
    daily_ret = {}
    for sym, df in crypto_data.items():
        dc = df["close"].resample("1D").last().dropna()
        daily_close[sym] = dc
        daily_ret[sym] = dc.pct_change()
    return pd.DataFrame(daily_close), pd.DataFrame(daily_ret)


def load_tvl_data() -> pd.Series | None:
    """Load TVL data for confounding check."""
    fp = ONCHAIN_DIR / "tvl_total.parquet"
    if fp.exists():
        df = pd.read_parquet(fp)
        if "totalLiquidityUSD" in df.columns:
            tvl = df["totalLiquidityUSD"]
        elif "tvl" in df.columns:
            tvl = df["tvl"]
        else:
            tvl = df.iloc[:, 0]
        tvl.index = pd.to_datetime(tvl.index)
        if tvl.index.tz is not None:
            tvl.index = tvl.index.tz_localize(None)
        logger.info(f"  ✅ TVL: {len(tvl)} bars, {tvl.index[0].date()} → {tvl.index[-1].date()}")
        return tvl
    return None


# ════════════════════════════════════════════════════════════
# Phase 2: Compute Macro Regime Indicators
# ════════════════════════════════════════════════════════════
def compute_macro_indicators(macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute regime indicators from macro data.

    Indicators per macro variable:
    - momentum: pct_change over 30/60/90 days
    - pctrank: percentile rank over 252 days (level regime)
    - For VIX: also raw level and z-score
    """
    logger.info("\n🔧 Computing macro regime indicators...")
    indicators = {}

    for col in macro_df.columns:
        series = macro_df[col].dropna()

        # Momentum (pct_change over N days)
        for window in MOMENTUM_WINDOWS:
            mom = series.pct_change(window)
            indicators[f"{col}_mom_{window}d"] = mom

        # Level percentile rank (rolling 252-day)
        pctrank = series.rolling(PCTRANK_WINDOW, min_periods=PCTRANK_WINDOW // 2).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
        indicators[f"{col}_pctrank"] = pctrank

        # For VIX: additional z-score
        if "VIX" in col:
            mean = series.rolling(PCTRANK_WINDOW, min_periods=60).mean()
            std = series.rolling(PCTRANK_WINDOW, min_periods=60).std()
            indicators[f"{col}_zscore"] = (series - mean) / std

    result = pd.DataFrame(indicators)
    logger.info(f"  Generated {len(result.columns)} indicators")
    logger.info(f"  Indicator list: {list(result.columns)}")

    return result


def compute_composite_risk_score(macro_df: pd.DataFrame) -> pd.Series:
    """
    Composite Risk-Off Score:
    - VIX momentum UP → risk-off
    - DXY momentum UP → risk-off
    - SPY momentum DOWN → risk-off
    - US10Y momentum UP → risk-off (tightening)

    Score: 0 (extreme risk-on) to 1 (extreme risk-off)
    """
    logger.info("  Computing composite risk-off score...")
    components = []

    # VIX: higher = more risk-off
    if "VIX" in macro_df.columns:
        vix_rank = macro_df["VIX"].rolling(PCTRANK_WINDOW, min_periods=60).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
        components.append(("VIX_level", vix_rank))

    # DXY: higher = more risk-off (strong dollar → risk-off)
    for col in macro_df.columns:
        if "DXY" in col:
            dxy_mom = macro_df[col].pct_change(60)
            dxy_rank = dxy_mom.rolling(PCTRANK_WINDOW, min_periods=60).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
            )
            components.append(("DXY_mom60", dxy_rank))
            break

    # SPY: LOWER = more risk-off (inverted)
    if "SPY" in macro_df.columns:
        spy_mom = macro_df["SPY"].pct_change(60)
        spy_rank = spy_mom.rolling(PCTRANK_WINDOW, min_periods=60).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
        components.append(("SPY_mom60_inv", 1 - spy_rank))

    # US10Y: higher = more risk-off
    if "US10Y" in macro_df.columns:
        us10y_mom = macro_df["US10Y"].pct_change(60)
        us10y_rank = us10y_mom.rolling(PCTRANK_WINDOW, min_periods=60).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
        components.append(("US10Y_mom60", us10y_rank))

    if not components:
        return pd.Series(dtype=float)

    # Equal-weight composite
    comp_df = pd.DataFrame({name: s for name, s in components})
    composite = comp_df.mean(axis=1)
    logger.info(f"  Composite uses {len(components)} components: {[c[0] for c in components]}")

    return composite


# ════════════════════════════════════════════════════════════
# Phase 3: Causal IC Analysis
# ════════════════════════════════════════════════════════════
def compute_causal_ic(
    indicators: pd.DataFrame,
    crypto_returns: pd.DataFrame,
    shift: int = IC_SHIFT,
) -> pd.DataFrame:
    """
    Compute causal IC: corr(signal_t, return_{t+shift}).
    In practice: signal.shift(shift).corr(return).
    """
    logger.info(f"\n📊 Computing Causal IC (shift={shift})...")

    results = []
    for ind_name in indicators.columns:
        signal = indicators[ind_name].dropna()
        if len(signal) < 100:
            continue

        per_symbol_ic = {}
        for sym in crypto_returns.columns:
            ret = crypto_returns[sym].dropna()

            # Align
            common_idx = signal.index.intersection(ret.index)
            if len(common_idx) < 100:
                continue

            s = signal.reindex(common_idx)
            r = ret.reindex(common_idx)

            # Causal: signal at t → return at t+1
            # Equivalent to: corr(signal.shift(1), return) = corr(signal, return.shift(-1))
            ic = s.shift(shift).corr(r)
            per_symbol_ic[sym] = ic

        if per_symbol_ic:
            ic_values = list(per_symbol_ic.values())
            avg_ic = np.nanmean(ic_values)
            same_sign = sum(1 for x in ic_values if np.sign(x) == np.sign(avg_ic) and not np.isnan(x))
            n_valid = sum(1 for x in ic_values if not np.isnan(x))

            results.append({
                "indicator": ind_name,
                "avg_ic": avg_ic,
                "abs_ic": abs(avg_ic),
                "n_symbols": n_valid,
                "same_sign": same_sign,
                "same_sign_pct": same_sign / n_valid if n_valid > 0 else 0,
                "max_ic": max(ic_values, key=abs) if ic_values else np.nan,
                "min_ic": min(ic_values, key=abs) if ic_values else np.nan,
                **per_symbol_ic,
            })

    df = pd.DataFrame(results).sort_values("abs_ic", ascending=False)
    return df


def compute_yearly_ic(
    indicator: pd.Series,
    crypto_returns: pd.DataFrame,
    indicator_name: str,
    shift: int = IC_SHIFT,
) -> pd.DataFrame:
    """Compute IC per year for stability check (A1)."""
    all_years = sorted(set(indicator.dropna().index.year).intersection(
        set(crypto_returns.dropna(how="all").index.year)
    ))

    results = []
    for year in all_years:
        mask_ind = indicator.index.year == year
        mask_ret = crypto_returns.index.year == year

        yearly_ics = []
        for sym in crypto_returns.columns:
            sig = indicator[mask_ind].dropna()
            ret = crypto_returns[sym][mask_ret].dropna()
            common = sig.index.intersection(ret.index)
            if len(common) < 30:
                continue
            ic = sig.reindex(common).shift(shift).corr(ret.reindex(common))
            if not np.isnan(ic):
                yearly_ics.append(ic)

        if yearly_ics:
            avg_ic = np.mean(yearly_ics)
            results.append({
                "year": year,
                "indicator": indicator_name,
                "avg_ic": avg_ic,
                "n_symbols": len(yearly_ics),
                "direction": "+" if avg_ic > 0 else "-",
            })

    return pd.DataFrame(results)


# ════════════════════════════════════════════════════════════
# Phase 4: Quality Gates (A1-A5)
# ════════════════════════════════════════════════════════════
def run_quality_gates(
    ic_df: pd.DataFrame,
    indicators: pd.DataFrame,
    crypto_returns: pd.DataFrame,
    top_n: int = 10,
) -> dict:
    """Run quality gates A1-A5 on top indicators."""
    logger.info("\n🔍 Running Quality Gates (A1-A5)...")

    top_indicators = ic_df.head(top_n)
    gate_results = {}

    for _, row in top_indicators.iterrows():
        ind_name = row["indicator"]
        avg_ic = row["avg_ic"]
        abs_ic = row["abs_ic"]
        same_sign_pct = row["same_sign_pct"]
        n_symbols = row["n_symbols"]

        # A1: Multi-year consistency
        yearly = compute_yearly_ic(indicators[ind_name], crypto_returns, ind_name)
        if not yearly.empty:
            directions = yearly["direction"].tolist()
            dominant_dir = "+" if avg_ic > 0 else "-"
            consistent_years = sum(1 for d in directions if d == dominant_dir)
            a1_pass = consistent_years >= len(directions) * 0.6  # >60% years same direction
            a1_detail = f"{consistent_years}/{len(directions)} years same direction"
        else:
            a1_pass = False
            a1_detail = "insufficient data"

        # A2: Shift sensitivity (compare shift=1 vs shift=2)
        ic_shift2_vals = []
        for sym in crypto_returns.columns:
            sig = indicators[ind_name].dropna()
            ret = crypto_returns[sym].dropna()
            common = sig.index.intersection(ret.index)
            if len(common) < 100:
                continue
            ic2 = sig.reindex(common).shift(2).corr(ret.reindex(common))
            if not np.isnan(ic2):
                ic_shift2_vals.append(ic2)
        avg_ic_shift2 = np.mean(ic_shift2_vals) if ic_shift2_vals else 0
        shift_diff = abs(abs_ic - abs(avg_ic_shift2))
        a2_pass = shift_diff < abs_ic * 0.5  # Less than 50% change = stable
        a2_detail = f"IC@shift1={avg_ic:.4f}, IC@shift2={avg_ic_shift2:.4f}, Δ={shift_diff:.4f}"

        # A3: Cross-symbol consistency
        a3_pass = same_sign_pct >= 0.75  # ≥6/8 same sign
        a3_detail = f"{int(same_sign_pct * n_symbols)}/{n_symbols} same sign ({same_sign_pct:.0%})"

        # A4: Direction interaction — skipped for this phase (need position data)
        a4_pass = True
        a4_detail = "N/A (regime-level signal, not directional)"

        # A5: IC magnitude
        a5_pass = abs_ic >= 0.01
        a5_detail = f"|IC|={abs_ic:.4f} {'≥' if a5_pass else '<'} 0.01"

        overall_pass = sum([a1_pass, a2_pass, a3_pass, a4_pass, a5_pass])

        gate_results[ind_name] = {
            "avg_ic": avg_ic,
            "abs_ic": abs_ic,
            "A1_yearly": {"pass": a1_pass, "detail": a1_detail},
            "A2_shift": {"pass": a2_pass, "detail": a2_detail},
            "A3_cross_sym": {"pass": a3_pass, "detail": a3_detail},
            "A4_direction": {"pass": a4_pass, "detail": a4_detail},
            "A5_magnitude": {"pass": a5_pass, "detail": a5_detail},
            "gates_passed": f"{overall_pass}/5",
            "yearly_detail": yearly.to_dict("records") if not yearly.empty else [],
        }

        status = "✅" if overall_pass >= 4 else "🟡" if overall_pass >= 3 else "❌"
        logger.info(f"\n  {status} {ind_name} (IC={avg_ic:.4f}, |IC|={abs_ic:.4f})")
        logger.info(f"    A1 Yearly:     {'✅' if a1_pass else '❌'} {a1_detail}")
        logger.info(f"    A2 Shift:      {'✅' if a2_pass else '❌'} {a2_detail}")
        logger.info(f"    A3 Cross-sym:  {'✅' if a3_pass else '❌'} {a3_detail}")
        logger.info(f"    A4 Direction:  {'✅' if a4_pass else '❌'} {a4_detail}")
        logger.info(f"    A5 Magnitude:  {'✅' if a5_pass else '❌'} {a5_detail}")

    return gate_results


# ════════════════════════════════════════════════════════════
# Phase 5: Confounding Check
# ════════════════════════════════════════════════════════════
def check_confounding(
    indicators: pd.DataFrame,
    crypto_daily_close: pd.DataFrame,
    tvl_data: pd.Series | None,
    top_indicators: list[str],
) -> dict:
    """Check correlation with existing signals (HTF proxy, TVL, BTC momentum)."""
    logger.info("\n🔍 Confounding Check...")

    # BTC momentum as proxy for HTF filter
    btc_close = crypto_daily_close.get("BTCUSDT")
    if btc_close is not None:
        btc_mom_30 = btc_close.pct_change(30)
        btc_mom_60 = btc_close.pct_change(60)
    else:
        btc_mom_30 = btc_mom_60 = None

    # TVL momentum
    tvl_mom_30 = None
    if tvl_data is not None:
        tvl_mom_30 = tvl_data.pct_change(30)

    confounding = {}
    for ind_name in top_indicators:
        if ind_name not in indicators.columns:
            continue
        signal = indicators[ind_name].dropna()
        corr_results = {}

        # vs BTC momentum (proxy for HTF)
        if btc_mom_30 is not None:
            common = signal.index.intersection(btc_mom_30.dropna().index)
            if len(common) > 100:
                corr_results["corr_btc_mom30"] = signal.reindex(common).corr(btc_mom_30.reindex(common))
        if btc_mom_60 is not None:
            common = signal.index.intersection(btc_mom_60.dropna().index)
            if len(common) > 100:
                corr_results["corr_btc_mom60"] = signal.reindex(common).corr(btc_mom_60.reindex(common))

        # vs TVL momentum
        if tvl_mom_30 is not None:
            common = signal.index.intersection(tvl_mom_30.dropna().index)
            if len(common) > 100:
                corr_results["corr_tvl_mom30"] = signal.reindex(common).corr(tvl_mom_30.reindex(common))

        confounding[ind_name] = corr_results

        logger.info(f"  {ind_name}:")
        for k, v in corr_results.items():
            independence = "✅ independent" if abs(v) < 0.3 else "⚠️ correlated" if abs(v) < 0.5 else "❌ redundant"
            logger.info(f"    {k}: {v:.3f} ({independence})")

    return confounding


# ════════════════════════════════════════════════════════════
# Phase 6: Filter Effect Analysis
# ════════════════════════════════════════════════════════════
def analyze_filter_effect(
    indicators: pd.DataFrame,
    crypto_returns: pd.DataFrame,
    top_indicators: list[str],
) -> dict:
    """Analyze quintile returns for top indicators."""
    logger.info("\n📊 Filter Effect Analysis (Quintile Returns)...")

    results = {}
    for ind_name in top_indicators:
        if ind_name not in indicators.columns:
            continue

        signal = indicators[ind_name].dropna()
        sym_results = {}

        for sym in crypto_returns.columns:
            ret = crypto_returns[sym].dropna()
            common = signal.index.intersection(ret.index)
            if len(common) < 200:
                continue

            s = signal.reindex(common).shift(IC_SHIFT)  # causal
            r = ret.reindex(common)

            # Remove NaN
            valid = s.notna() & r.notna()
            s = s[valid]
            r = r[valid]

            if len(s) < 200:
                continue

            # Quintile analysis
            quintiles = pd.qcut(s, 5, labels=False, duplicates="drop")
            q_returns = {}
            for q in sorted(quintiles.unique()):
                mask = quintiles == q
                ann_ret = r[mask].mean() * 365
                ann_vol = r[mask].std() * np.sqrt(365)
                sr = ann_ret / ann_vol if ann_vol > 0 else 0
                q_returns[f"Q{q+1}"] = {"ann_ret": ann_ret, "sharpe": sr, "count": int(mask.sum())}

            # Filter effect: top quintile vs bottom
            if q_returns:
                q_keys = sorted(q_returns.keys())
                top_sr = q_returns[q_keys[-1]]["sharpe"]
                bot_sr = q_returns[q_keys[0]]["sharpe"]
                spread = top_sr - bot_sr

                sym_results[sym] = {
                    "quintiles": q_returns,
                    "Q5_Q1_spread": spread,
                    "Q5_sharpe": top_sr,
                    "Q1_sharpe": bot_sr,
                }

        if sym_results:
            avg_spread = np.mean([v["Q5_Q1_spread"] for v in sym_results.values()])
            n_positive = sum(1 for v in sym_results.values() if v["Q5_Q1_spread"] > 0)
            n_total = len(sym_results)

            results[ind_name] = {
                "per_symbol": sym_results,
                "avg_Q5_Q1_spread": avg_spread,
                "n_positive_spread": n_positive,
                "n_total": n_total,
                "monotonic": _check_monotonic(sym_results),
            }

            logger.info(f"\n  {ind_name}:")
            logger.info(f"    Avg Q5-Q1 Spread: {avg_spread:+.3f} Sharpe")
            logger.info(f"    Positive: {n_positive}/{n_total} symbols")
            logger.info(f"    Monotonic: {results[ind_name]['monotonic']}")

            # Per-symbol detail
            for sym, v in sym_results.items():
                qs = v["quintiles"]
                q_srs = [qs[k]["sharpe"] for k in sorted(qs.keys())]
                logger.info(f"    {sym}: Q1→Q5 SR = {' → '.join(f'{s:.2f}' for s in q_srs)}, spread={v['Q5_Q1_spread']:+.2f}")

    return results


def _check_monotonic(sym_results: dict) -> str:
    """Check if quintile returns are monotonically increasing across symbols."""
    monotonic_count = 0
    total = 0
    for sym, data in sym_results.items():
        qs = data["quintiles"]
        srs = [qs[k]["sharpe"] for k in sorted(qs.keys())]
        if len(srs) >= 3:
            total += 1
            # Check if roughly monotonic (allow 1 violation)
            violations = sum(1 for i in range(len(srs) - 1) if srs[i] > srs[i + 1])
            if violations <= 1:
                monotonic_count += 1
    return f"{monotonic_count}/{total} symbols roughly monotonic"


# ════════════════════════════════════════════════════════════
# Phase 7: Verdict
# ════════════════════════════════════════════════════════════
def determine_verdict(
    ic_df: pd.DataFrame,
    gate_results: dict,
    confounding: dict,
    filter_effects: dict,
) -> dict:
    """Determine GO/WEAK GO/FAIL verdict."""
    logger.info("\n" + "=" * 60)
    logger.info("📋 VERDICT DETERMINATION")
    logger.info("=" * 60)

    # Find best indicator
    if ic_df.empty:
        verdict = "FAIL"
        reason = "No indicators with sufficient data"
        return {"verdict": verdict, "reason": reason}

    best = ic_df.iloc[0]
    best_name = best["indicator"]
    best_ic = best["abs_ic"]

    logger.info(f"\n  Best indicator: {best_name}")
    logger.info(f"  |IC| = {best_ic:.4f}")

    # Check gates
    if best_name in gate_results:
        gates = gate_results[best_name]
        n_pass = int(gates["gates_passed"].split("/")[0])
    else:
        n_pass = 0

    # Check confounding
    max_corr_existing = 0
    if best_name in confounding:
        for k, v in confounding[best_name].items():
            max_corr_existing = max(max_corr_existing, abs(v))

    # Check filter effect
    avg_spread = 0
    n_positive_spread = 0
    n_total_spread = 0
    if best_name in filter_effects:
        fe = filter_effects[best_name]
        avg_spread = fe["avg_Q5_Q1_spread"]
        n_positive_spread = fe["n_positive_spread"]
        n_total_spread = fe["n_total"]

    logger.info(f"\n  Quality gates: {n_pass}/5")
    logger.info(f"  Max corr with existing: {max_corr_existing:.3f}")
    logger.info(f"  Filter spread: {avg_spread:+.3f}, positive {n_positive_spread}/{n_total_spread}")

    # Decision logic
    if best_ic < 0.01:
        verdict = "FAIL"
        reason = f"IC too weak: |IC|={best_ic:.4f} < 0.01"
    elif n_pass < 3:
        verdict = "FAIL"
        reason = f"Only {n_pass}/5 quality gates passed"
    elif max_corr_existing > 0.5:
        verdict = "FAIL"
        reason = f"Too correlated with existing signals: max_corr={max_corr_existing:.3f}"
    elif best_ic >= 0.03 and n_pass >= 4 and max_corr_existing < 0.3 and n_positive_spread >= 6:
        verdict = "GO"
        reason = f"|IC|={best_ic:.4f}, {n_pass}/5 gates, corr={max_corr_existing:.3f}, spread {n_positive_spread}/{n_total_spread}"
    elif best_ic >= 0.01 and n_pass >= 3:
        verdict = "WEAK GO"
        reason = f"|IC|={best_ic:.4f}, {n_pass}/5 gates, corr={max_corr_existing:.3f}, spread {n_positive_spread}/{n_total_spread}"
    else:
        verdict = "FAIL"
        reason = "Combined criteria not met"

    logger.info(f"\n  {'🟢' if verdict == 'GO' else '🟡' if verdict == 'WEAK GO' else '🔴'} VERDICT: {verdict}")
    logger.info(f"  Reason: {reason}")

    return {
        "verdict": verdict,
        "reason": reason,
        "best_indicator": best_name,
        "best_ic": float(best_ic),
        "gates_passed": n_pass,
        "max_corr_existing": float(max_corr_existing),
        "avg_quintile_spread": float(avg_spread),
        "n_positive_spread": n_positive_spread,
    }


# ════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════
def main():
    logger.info("=" * 60)
    logger.info("🌍 Macro Cross-Market Regime EDA")
    logger.info("=" * 60)
    logger.info("Hypothesis: Macro risk regime (VIX/DXY/US10Y/SPY) provides")
    logger.info("information about crypto returns not captured by HTF or TVL.\n")

    # Phase 1: Data
    macro_df = download_macro_data()
    crypto_data = load_crypto_data()
    crypto_daily_close, crypto_daily_ret = resample_crypto_to_daily(crypto_data)
    tvl_data = load_tvl_data()

    # Phase 2: Indicators
    indicators = compute_macro_indicators(macro_df)
    composite = compute_composite_risk_score(macro_df)
    if not composite.empty:
        indicators["composite_risk_off"] = composite

    # Phase 3: Causal IC
    ic_df = compute_causal_ic(indicators, crypto_daily_ret)

    logger.info("\n📊 IC Ranking (Top 15):")
    cols_to_show = ["indicator", "avg_ic", "abs_ic", "same_sign", "n_symbols"]
    logger.info(ic_df[cols_to_show].head(15).to_string(index=False))

    # Phase 4: Quality Gates on top 10
    top_indicator_names = ic_df.head(10)["indicator"].tolist()
    gate_results = run_quality_gates(ic_df, indicators, crypto_daily_ret, top_n=10)

    # Phase 5: Confounding
    confounding = check_confounding(indicators, crypto_daily_close, tvl_data, top_indicator_names)

    # Phase 6: Filter Effect
    filter_effects = analyze_filter_effect(indicators, crypto_daily_ret, top_indicator_names[:5])

    # Phase 7: Verdict
    verdict = determine_verdict(ic_df, gate_results, confounding, filter_effects)

    # Save full results
    report = {
        "meta": {
            "research": "Macro Cross-Market Regime EDA",
            "date": "2026-03-01",
            "hypothesis": "H1: Macro risk regime provides crypto return information not captured by HTF/TVL",
            "score": 3.9,
            "integration_target": "Portfolio Layer / Filter",
        },
        "ic_ranking": ic_df.head(20).to_dict("records"),
        "quality_gates": gate_results,
        "confounding": confounding,
        "filter_effects": {
            k: {kk: vv for kk, vv in v.items() if kk != "per_symbol"}
            for k, v in filter_effects.items()
        },
        "verdict": verdict,
    }

    report_path = REPORT_DIR / "macro_regime_eda_results.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"\n📁 Report saved to {report_path}")

    return verdict


if __name__ == "__main__":
    verdict = main()
    sys.exit(0 if verdict["verdict"] in ("GO", "WEAK GO") else 1)
