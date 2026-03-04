"""
Orderflow Composite 獨立策略 — EDA + Standalone Backtest + Portfolio Blend

Phase 1-3 of the Orderflow Composite strategy development plan.
Downloads data, evaluates signal quality, runs standalone backtest,
and computes portfolio-level blend with TSMOM.

Key questions answered:
    1. Do OFI, CVD, VPIN proxy individually have predictive IC?
    2. Does a composite signal have better IC than components?
    3. What is the correlation with existing TSMOM strategy?
    4. Is standalone Sharpe > 1.0?
    5. Does portfolio blend improve overall SR?

Usage:
    PYTHONPATH=src python scripts/research_orderflow_composite_eda.py

    # Skip download if data already exists:
    PYTHONPATH=src python scripts/research_orderflow_composite_eda.py --skip-download

    # Only run EDA (no backtest):
    PYTHONPATH=src python scripts/research_orderflow_composite_eda.py --eda-only

# EMBARGO_EXEMPT: Completed research (2026-03-02, FAIL).
# Results already obtained before embargo infrastructure was built.
# Future orderflow research should apply embargo via enforce_temporal_embargo().
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)

# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
    "DOGEUSDT", "ADAUSDT", "XRPUSDT", "LINKUSDT",
]
START = "2022-01-01"
END = "2026-02-28"
INTERVAL = "1h"
MARKET_TYPE = "futures"
DATA_DIR = Path("data")
KLINE_DIR = DATA_DIR / "binance" / MARKET_TYPE / INTERVAL
DERIVATIVES_DIR = DATA_DIR / "binance" / "futures" / "derivatives"
REPORT_DIR = Path("reports/research/orderflow_composite")
CONFIG_PATH = Path("config/research_orderflow_composite.yaml")


# ═══════════════════════════════════════════════════════════════
# Section 1: Data Download
# ═══════════════════════════════════════════════════════════════


def download_all_data() -> None:
    """Download klines + derivatives (taker_vol_ratio, funding_rate) for all symbols."""
    from qtrade.data.binance_vision import download_binance_vision_klines
    from qtrade.data.funding_rate import (
        download_funding_rates,
        get_funding_rate_path,
        save_funding_rates,
    )
    from qtrade.data.storage import save_klines
    from qtrade.data.taker_volume import (
        compute_cvd,
        download_taker_volume,
        save_cvd,
        save_taker_volume,
    )

    print("\n" + "=" * 70)
    print("  📥 Phase 0: Data Download")
    print("=" * 70)

    for sym in SYMBOLS:
        # ── Klines ──
        kline_path = KLINE_DIR / f"{sym}.parquet"
        if kline_path.exists():
            print(f"  ✅ {sym} klines: already exists")
        else:
            print(f"  📥 {sym} klines: downloading...")
            try:
                df = download_binance_vision_klines(
                    sym, INTERVAL, START, END, market_type=MARKET_TYPE
                )
                if not df.empty:
                    kline_path.parent.mkdir(parents=True, exist_ok=True)
                    save_klines(df, kline_path)
                    print(f"  ✅ {sym} klines: {len(df)} rows saved")
                else:
                    print(f"  ⚠️  {sym} klines: empty")
            except Exception as e:
                print(f"  ❌ {sym} klines: {e}")

        # ── Taker Volume Ratio + CVD ──
        taker_path = DERIVATIVES_DIR / "taker_vol_ratio" / f"{sym}.parquet"
        if taker_path.exists():
            print(f"  ✅ {sym} taker_vol_ratio: already exists")
        else:
            print(f"  📥 {sym} taker_vol_ratio: downloading...")
            try:
                taker = download_taker_volume(sym, start=START, end=END, provider="vision")
                if not taker.empty:
                    save_taker_volume(taker, sym, data_dir=DERIVATIVES_DIR)
                    cvd = compute_cvd(taker)
                    save_cvd(cvd, sym, data_dir=DERIVATIVES_DIR)
                    print(f"  ✅ {sym} taker+cvd: {len(taker)} rows saved")
                else:
                    print(f"  ⚠️  {sym} taker_vol: empty")
            except Exception as e:
                print(f"  ❌ {sym} taker_vol: {e}")

        # ── Funding Rate ──
        fr_path = get_funding_rate_path(DATA_DIR, sym)
        if fr_path.exists():
            print(f"  ✅ {sym} funding_rate: already exists")
        else:
            print(f"  📥 {sym} funding_rate: downloading...")
            try:
                fr = download_funding_rates(sym, START, END)
                if not fr.empty:
                    save_funding_rates(fr, fr_path)
                    print(f"  ✅ {sym} funding_rate: {len(fr)} rows saved")
            except Exception as e:
                print(f"  ❌ {sym} funding_rate: {e}")

    print()


# ═══════════════════════════════════════════════════════════════
# Section 2: Signal Construction Helpers
# ═══════════════════════════════════════════════════════════════


def load_symbol_data(sym: str) -> dict | None:
    """Load klines + taker data for a symbol."""
    from qtrade.data.storage import load_klines
    from qtrade.data.taker_volume import align_taker_to_klines, load_taker_volume

    kline_path = KLINE_DIR / f"{sym}.parquet"
    if not kline_path.exists():
        return None

    df = load_klines(kline_path)
    if df.empty:
        return None

    taker = load_taker_volume(sym, data_dir=DERIVATIVES_DIR)
    if taker is None or taker.empty:
        return None

    aligned = align_taker_to_klines(taker, df.index, max_ffill_bars=4)
    if aligned is None:
        return None

    coverage = 1.0 - aligned.isna().mean()
    if coverage < 0.10:
        logger.warning(f"⚠️ {sym}: taker coverage {coverage:.1%} < 10%")
        return None

    return {"df": df, "taker": aligned, "coverage": coverage}


def compute_orderflow_signals(taker: pd.Series) -> dict[str, pd.Series]:
    """Compute all orderflow signal components from taker vol ratio."""
    from qtrade.strategy.orderflow_composite_strategy import (
        _cvd_trend_signal,
        _ofi_direction_signal,
        _vpin_proxy_pctrank,
        compute_ofi_from_taker_ratio,
    )

    ofi = compute_ofi_from_taker_ratio(taker)

    return {
        "ofi_raw": ofi,
        "ofi_signal_12": _ofi_direction_signal(ofi, ema_span=12, lookback=12),
        "ofi_signal_24": _ofi_direction_signal(ofi, ema_span=12, lookback=24),
        "ofi_signal_72": _ofi_direction_signal(ofi, ema_span=24, lookback=72),
        "vpin_proxy": _vpin_proxy_pctrank(ofi, window=50, lookback=720),
        "cvd_trend": _cvd_trend_signal(ofi, lookback=72, ema_span=24),
        "cvd": ofi.cumsum(),
    }


def compute_tsmom_signal(close: pd.Series, lookback: int = 168) -> pd.Series:
    """Compute TSMOM signal for correlation comparison."""
    returns = close.pct_change()
    cum_ret = returns.rolling(lookback).sum()
    vol = returns.rolling(lookback).std() * np.sqrt(8760)
    vol = vol.replace(0, np.nan).ffill().fillna(0.15)
    raw = np.sign(cum_ret)
    scale = (0.15 / vol).clip(0.1, 2.0)
    tsmom = (raw * scale).clip(-1.0, 1.0).fillna(0.0)
    return tsmom


def compute_composite_signal(
    signals: dict, params: dict | None = None, contrarian: bool = True
) -> pd.Series:
    """Build composite orderflow signal from components.
    
    Args:
        signals: dict of signal components
        params: strategy parameters
        contrarian: if True, invert OFI and CVD signals (fade the flow)
    """
    if params is None:
        params = {}

    ofi_signal = signals["ofi_signal_24"].copy()
    vpin_pctrank = signals["vpin_proxy"]
    cvd_trend = signals["cvd_trend"].copy()

    # Contrarian mode: fade the flow
    if contrarian:
        ofi_signal = -ofi_signal
        cvd_trend = -cvd_trend

    vpin_min = params.get("vpin_min_scale", 0.3)
    cvd_agree = params.get("cvd_agree_scale", 1.0)
    cvd_disagree = params.get("cvd_disagree_scale", 0.5)
    cvd_neutral = params.get("cvd_neutral_scale", 0.7)

    vpin_scale = vpin_min + (1.0 - vpin_min) * vpin_pctrank

    ofi_dir = np.sign(ofi_signal)
    agreement = ofi_dir * cvd_trend
    cvd_scale = pd.Series(cvd_neutral, index=ofi_signal.index)
    cvd_scale[agreement > 0] = cvd_agree
    cvd_scale[agreement < 0] = cvd_disagree

    composite = ofi_signal * vpin_scale * cvd_scale

    # Light EMA smoothing
    ema_span = params.get("composite_ema", 6)
    if ema_span > 0:
        composite = composite.ewm(span=ema_span, adjust=False).mean()

    return composite.clip(-1.0, 1.0).fillna(0.0)


# ═══════════════════════════════════════════════════════════════
# Section 3: EDA Analysis
# ═══════════════════════════════════════════════════════════════


def compute_ic(signal: pd.Series, fwd_ret: pd.Series) -> float:
    """Compute rank IC (Spearman correlation) between signal and forward returns."""
    valid = signal.notna() & fwd_ret.notna()
    if valid.sum() < 100:
        return np.nan
    return signal[valid].corr(fwd_ret[valid], method="spearman")


def run_eda() -> dict:
    """
    Run comprehensive EDA on orderflow signals.

    Returns dict with per-symbol and aggregate results.
    """
    print("\n" + "=" * 70)
    print("  📊 Phase 1: Orderflow Composite EDA")
    print("=" * 70)

    results: dict = {"symbols": {}, "aggregate": {}}
    all_corrs = []  # correlation with TSMOM
    all_ics: dict[str, list] = {}

    for sym in SYMBOLS:
        print(f"\n  ── {sym} ──")
        data = load_symbol_data(sym)
        if data is None:
            print(f"    ⚠️  Skip: no data")
            results["symbols"][sym] = {"status": "no_data"}
            continue

        df = data["df"]
        taker = data["taker"]
        coverage = data["coverage"]

        # Forward returns (1h, 6h, 24h)
        close = df["close"]
        fwd_1h = close.pct_change().shift(-1)
        fwd_6h = close.pct_change(6).shift(-6)
        fwd_24h = close.pct_change(24).shift(-24)

        # Compute signals
        signals = compute_orderflow_signals(taker)
        composite_momentum = compute_composite_signal(signals, contrarian=False)
        composite_contrarian = compute_composite_signal(signals, contrarian=True)
        tsmom = compute_tsmom_signal(close)

        # ── IC Analysis (momentum raw signals + both composite modes) ──
        signal_names = [
            "ofi_signal_12", "ofi_signal_24", "ofi_signal_72",
            "vpin_proxy", "cvd_trend",
            "composite_momentum", "composite_contrarian",
        ]
        ic_results = {}
        for name in signal_names:
            if name == "composite_momentum":
                sig = composite_momentum
            elif name == "composite_contrarian":
                sig = composite_contrarian
            else:
                sig = signals.get(name)
            if sig is None:
                continue
            ic_1h = compute_ic(sig, fwd_1h)
            ic_6h = compute_ic(sig, fwd_6h)
            ic_24h = compute_ic(sig, fwd_24h)
            ic_results[name] = {"1h": ic_1h, "6h": ic_6h, "24h": ic_24h}

            if name not in all_ics:
                all_ics[name] = []
            all_ics[name].append({"symbol": sym, "ic_1h": ic_1h, "ic_6h": ic_6h, "ic_24h": ic_24h})

        # Use contrarian composite as the primary signal
        composite = composite_contrarian

        # ── Correlation with TSMOM ──
        valid = composite.notna() & tsmom.notna()
        if valid.sum() > 100:
            corr_tsmom = composite[valid].corr(tsmom[valid])
        else:
            corr_tsmom = np.nan
        all_corrs.append(corr_tsmom)

        # ── Year-by-Year IC stability ──
        yearly_ic = {}
        composite_aligned = composite.copy()
        for year in range(2022, 2027):
            mask = composite_aligned.index.year == year
            if mask.sum() < 100:
                continue
            ic_y = compute_ic(composite_aligned[mask], fwd_24h[mask])
            yearly_ic[str(year)] = ic_y

        # ── Quick standalone PnL simulation (both modes) ──
        def _quick_pnl(pos_series, ret_series):
            pos_delayed = pos_series.shift(1).fillna(0.0)
            strategy_ret = pos_delayed * ret_series
            strategy_ret = strategy_ret.dropna()
            if len(strategy_ret) > 100:
                ann_ret = strategy_ret.mean() * 8760
                ann_vol = strategy_ret.std() * np.sqrt(8760)
                sr = ann_ret / ann_vol if ann_vol > 0 else 0
                cum_ret = (1 + strategy_ret).cumprod()
                mdd = (cum_ret / cum_ret.cummax() - 1).min()
                turnover = pos_delayed.diff().abs().mean() * 8760
                return sr, mdd, turnover, ann_ret
            return 0, 0, 0, 0

        ret = close.pct_change()
        sr_m, mdd_m, to_m, ar_m = _quick_pnl(composite_momentum, ret)
        sr_c, mdd_c, to_c, ar_c = _quick_pnl(composite_contrarian, ret)

        # Use contrarian as primary
        sr, mdd, turnover, ann_ret = sr_c, mdd_c, to_c, ar_c

        sym_result = {
            "coverage": float(coverage),
            "n_bars": len(df),
            "ic": ic_results,
            "corr_tsmom": float(corr_tsmom) if not np.isnan(corr_tsmom) else None,
            "yearly_ic": yearly_ic,
            "standalone_sr_contrarian": float(sr_c),
            "standalone_sr_momentum": float(sr_m),
            "standalone_sr": float(sr),
            "standalone_mdd": float(mdd),
            "standalone_ann_return": float(ann_ret),
            "turnover_annual": float(turnover),
        }
        results["symbols"][sym] = sym_result

        # Print summary
        print(f"    Coverage:    {coverage:.1%}")
        print(f"    Bars:        {len(df):,}")
        c_ic_m = ic_results.get("composite_momentum", {})
        c_ic_c = ic_results.get("composite_contrarian", {})
        print(f"    Momentum IC:    1h={c_ic_m.get('1h', 0):.4f}, 6h={c_ic_m.get('6h', 0):.4f}, 24h={c_ic_m.get('24h', 0):.4f}")
        print(f"    Contrarian IC:  1h={c_ic_c.get('1h', 0):.4f}, 6h={c_ic_c.get('6h', 0):.4f}, 24h={c_ic_c.get('24h', 0):.4f}")
        print(f"    SR: momentum={sr_m:.3f}, contrarian={sr_c:.3f}")
        print(f"    Corr(TSMOM): {corr_tsmom:.3f}")
        print(f"    Standalone:  SR={sr:.3f}, MDD={mdd:.2%}, Turnover={turnover:.0f}/yr")

    # ── Aggregate Statistics ──
    print("\n" + "-" * 70)
    print("  📋 Aggregate Results")
    print("-" * 70)

    valid_syms = [s for s in results["symbols"]
                  if results["symbols"][s].get("standalone_sr") is not None
                  and results["symbols"][s].get("status") != "no_data"]

    if valid_syms:
        avg_corr = np.nanmean(all_corrs)
        avg_sr = np.mean([results["symbols"][s]["standalone_sr"] for s in valid_syms])
        avg_mdd = np.mean([results["symbols"][s]["standalone_mdd"] for s in valid_syms])

        # IC consistency: how many symbols have same-sign IC?
        for sig_name in all_ics:
            ics_24h = [x["ic_24h"] for x in all_ics[sig_name] if not np.isnan(x["ic_24h"])]
            n_pos = sum(1 for x in ics_24h if x > 0)
            n_neg = sum(1 for x in ics_24h if x < 0)
            same_sign = max(n_pos, n_neg)
            avg_ic = np.mean(ics_24h) if ics_24h else 0
            print(f"    {sig_name:<20} avg IC(24h)={avg_ic:+.4f}  same_sign={same_sign}/{len(ics_24h)}")

        print(f"\n    Avg corr(TSMOM):  {avg_corr:.3f}  {'✅ < 0.30' if abs(avg_corr) < 0.30 else '❌ >= 0.30'}")
        print(f"    Avg standalone SR: {avg_sr:.3f}  {'✅ > 1.0' if avg_sr > 1.0 else '⚠️ <= 1.0' if avg_sr > 0.5 else '❌ <= 0.5'}")
        print(f"    Avg MDD:          {avg_mdd:.2%}")
        print(f"    Symbols positive: {sum(1 for s in valid_syms if results['symbols'][s]['standalone_sr'] > 0)}/{len(valid_syms)}")

        results["aggregate"] = {
            "avg_corr_tsmom": float(avg_corr),
            "avg_standalone_sr": float(avg_sr),
            "avg_mdd": float(avg_mdd),
            "n_symbols": len(valid_syms),
            "n_positive_sr": sum(1 for s in valid_syms if results["symbols"][s]["standalone_sr"] > 0),
        }

    return results


# ═══════════════════════════════════════════════════════════════
# Section 4: Formal Backtest (per-symbol + portfolio)
# ═══════════════════════════════════════════════════════════════


def run_standalone_backtest() -> dict:
    """Run formal backtest using the strategy framework."""
    print("\n" + "=" * 70)
    print("  📈 Phase 2: Standalone Backtest (vbt Portfolio)")
    print("=" * 70)

    from qtrade.backtest.run_backtest import run_symbol_backtest
    from qtrade.config import load_config

    cfg = load_config(str(CONFIG_PATH))
    bt_dict = cfg.to_backtest_dict()

    results = {}
    all_equity = {}

    for sym in SYMBOLS:
        kline_path = KLINE_DIR / f"{sym}.parquet"
        if not kline_path.exists():
            print(f"  ⚠️ {sym}: no kline data, skip")
            continue

        print(f"\n  📊 {sym}...")
        try:
            bt = run_symbol_backtest(
                symbol=sym,
                data_path=kline_path,
                cfg=copy.deepcopy(bt_dict),
                strategy_name="orderflow_composite",
                market_type=MARKET_TYPE,
                direction="both",
                data_dir=DATA_DIR,
            )

            stats = bt.stats
            results[sym] = {
                "sharpe": float(stats.get("Sharpe Ratio", 0)),
                "total_return": float(stats.get("Total Return [%]", 0)) / 100.0,
                "max_drawdown": float(stats.get("Max Drawdown [%]", 0)) / 100.0,
                "win_rate": float(stats.get("Win Rate [%]", 0)),
                "num_trades": int(stats.get("Total Trades", 0)),
                "calmar": float(stats.get("Calmar Ratio", 0)),
            }

            print(
                f"    SR={results[sym]['sharpe']:.3f}, "
                f"Return={results[sym]['total_return']:.1%}, "
                f"MDD={results[sym]['max_drawdown']:.2%}, "
                f"Trades={results[sym]['num_trades']}"
            )

            # Store equity for portfolio
            if bt.pf is not None:
                all_equity[sym] = bt.pf.value()

        except Exception as e:
            print(f"    ❌ {sym}: {e}")
            results[sym] = {"error": str(e)}

    # ── Portfolio Summary ──
    if all_equity:
        print("\n  📋 Per-Symbol Summary:")
        print(f"    {'Symbol':<12} {'SR':>8} {'Return':>10} {'MDD':>8} {'Trades':>8}")
        print("    " + "-" * 50)
        for sym in results:
            if "sharpe" in results[sym]:
                r = results[sym]
                print(
                    f"    {sym:<12} {r['sharpe']:>8.3f} {r['total_return']:>9.1%} "
                    f"{r['max_drawdown']:>7.2%} {r['num_trades']:>8}"
                )

    return results


# ═══════════════════════════════════════════════════════════════
# Section 5: Portfolio Blend with TSMOM
# ═══════════════════════════════════════════════════════════════


def run_portfolio_blend() -> dict:
    """
    Evaluate portfolio-level blend of Orderflow + TSMOM.

    Computes:
    - TSMOM-only portfolio SR
    - Orderflow-only portfolio SR
    - 50/50 blend portfolio SR
    - Marginal SR contribution
    """
    print("\n" + "=" * 70)
    print("  🔀 Phase 3: Portfolio Blend Evaluation")
    print("=" * 70)

    from qtrade.backtest.run_backtest import run_symbol_backtest
    from qtrade.config import load_config

    # ── Orderflow returns per symbol ──
    of_cfg = load_config(str(CONFIG_PATH))
    of_dict = of_cfg.to_backtest_dict()

    # ── TSMOM returns per symbol ──
    tsmom_cfg_path = "config/prod_candidate_simplified.yaml"
    try:
        tsmom_cfg = load_config(tsmom_cfg_path)
        tsmom_dict = tsmom_cfg.to_backtest_dict()
    except Exception as e:
        print(f"  ⚠️ Cannot load TSMOM config: {e}")
        print("  Skipping portfolio blend (TSMOM config required)")
        return {}

    portfolio_alloc = {
        "BTCUSDT": 0.40, "ETHUSDT": 0.40,
        "SOLUSDT": 0.35, "BNBUSDT": 0.35,
        "DOGEUSDT": 0.35, "ADAUSDT": 0.35,
        "XRPUSDT": 0.35, "LINKUSDT": 0.35,
    }

    of_returns = {}
    tsmom_returns = {}

    for sym in SYMBOLS:
        kline_path = KLINE_DIR / f"{sym}.parquet"
        if not kline_path.exists():
            continue

        # Orderflow
        try:
            bt_of = run_symbol_backtest(
                sym, kline_path, copy.deepcopy(of_dict),
                strategy_name="orderflow_composite",
                market_type=MARKET_TYPE, direction="both",
                data_dir=DATA_DIR,
            )
            if bt_of.pf is not None:
                of_returns[sym] = bt_of.pf.returns()
        except Exception as e:
            logger.warning(f"⚠️ {sym} orderflow backtest failed: {e}")

        # TSMOM
        try:
            bt_ts = run_symbol_backtest(
                sym, kline_path, copy.deepcopy(tsmom_dict),
                strategy_name="meta_blend",
                market_type=MARKET_TYPE, direction="both",
                data_dir=DATA_DIR,
            )
            if bt_ts.pf is not None:
                tsmom_returns[sym] = bt_ts.pf.returns()
        except Exception as e:
            logger.warning(f"⚠️ {sym} TSMOM backtest failed: {e}")

    if not of_returns or not tsmom_returns:
        print("  ❌ Insufficient data for portfolio blend")
        return {}

    # ── Compute portfolio returns ──
    common_syms = sorted(set(of_returns.keys()) & set(tsmom_returns.keys()))
    print(f"\n  Common symbols: {len(common_syms)} — {', '.join(common_syms)}")

    # Align all return series to common index
    common_idx = None
    for sym in common_syms:
        idx = of_returns[sym].index.intersection(tsmom_returns[sym].index)
        if common_idx is None:
            common_idx = idx
        else:
            common_idx = common_idx.intersection(idx)

    if common_idx is None or len(common_idx) < 100:
        print("  ❌ Insufficient overlapping data")
        return {}

    # Weighted portfolio returns
    total_weight = sum(portfolio_alloc.get(s, 0.35) for s in common_syms)

    of_portfolio_ret = pd.Series(0.0, index=common_idx)
    ts_portfolio_ret = pd.Series(0.0, index=common_idx)

    for sym in common_syms:
        w = portfolio_alloc.get(sym, 0.35) / total_weight
        of_portfolio_ret += of_returns[sym].reindex(common_idx, fill_value=0) * w
        ts_portfolio_ret += tsmom_returns[sym].reindex(common_idx, fill_value=0) * w

    # Blend: 50% TSMOM + 50% Orderflow
    blend_ret = 0.5 * ts_portfolio_ret + 0.5 * of_portfolio_ret

    def portfolio_stats(ret_series: pd.Series, name: str) -> dict:
        ann_ret = ret_series.mean() * 8760
        ann_vol = ret_series.std() * np.sqrt(8760)
        sr = ann_ret / ann_vol if ann_vol > 0 else 0
        cum = (1 + ret_series).cumprod()
        mdd = (cum / cum.cummax() - 1).min()
        return {"name": name, "sr": sr, "ann_return": ann_ret, "mdd": mdd}

    ts_stats = portfolio_stats(ts_portfolio_ret, "TSMOM Only")
    of_stats = portfolio_stats(of_portfolio_ret, "Orderflow Only")
    blend_stats = portfolio_stats(blend_ret, "50/50 Blend")

    # Correlation between portfolio returns
    corr = of_portfolio_ret.corr(ts_portfolio_ret)

    print(f"\n    {'Portfolio':<20} {'SR':>8} {'Return':>10} {'MDD':>8}")
    print("    " + "-" * 50)
    for s in [ts_stats, of_stats, blend_stats]:
        print(f"    {s['name']:<20} {s['sr']:>8.3f} {s['ann_return']:>9.1%} {s['mdd']:>7.2%}")
    print(f"\n    Portfolio return correlation: {corr:.3f}")
    print(f"    Marginal SR from blend:     {blend_stats['sr'] - ts_stats['sr']:+.3f}")

    result = {
        "tsmom": {k: float(v) for k, v in ts_stats.items() if k != "name"},
        "orderflow": {k: float(v) for k, v in of_stats.items() if k != "name"},
        "blend_50_50": {k: float(v) for k, v in blend_stats.items() if k != "name"},
        "portfolio_corr": float(corr),
        "marginal_sr": float(blend_stats["sr"] - ts_stats["sr"]),
        "n_common_symbols": len(common_syms),
        "n_bars": len(common_idx),
    }

    return result


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Orderflow Composite Strategy — EDA + Backtest + Portfolio Blend"
    )
    parser.add_argument("--skip-download", action="store_true", help="Skip data download")
    parser.add_argument("--eda-only", action="store_true", help="Only run EDA, skip formal backtest")
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Phase 0: Download ──
    if not args.skip_download:
        download_all_data()

    # ── Phase 1: EDA ──
    eda_results = run_eda()

    # Save EDA results
    eda_path = REPORT_DIR / "orderflow_eda_results.json"

    def _clean(obj):
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        return obj

    with open(eda_path, "w") as f:
        json.dump(_clean(eda_results), f, indent=2)
    print(f"\n  💾 EDA results saved: {eda_path}")

    if args.eda_only:
        # ── Quality Gate Check ──
        _print_quality_gate_verdict(eda_results)
        return

    # ── Phase 2: Formal Backtest ──
    bt_results = run_standalone_backtest()

    bt_path = REPORT_DIR / "orderflow_backtest_results.json"
    with open(bt_path, "w") as f:
        json.dump(_clean(bt_results), f, indent=2)
    print(f"\n  💾 Backtest results saved: {bt_path}")

    # ── Phase 3: Portfolio Blend ──
    blend_results = run_portfolio_blend()

    blend_path = REPORT_DIR / "orderflow_blend_results.json"
    with open(blend_path, "w") as f:
        json.dump(_clean(blend_results), f, indent=2)
    print(f"\n  💾 Blend results saved: {blend_path}")

    # ── Final Verdict ──
    _print_final_verdict(eda_results, bt_results, blend_results)


def _print_quality_gate_verdict(eda: dict) -> None:
    """Print EDA quality gate summary."""
    agg = eda.get("aggregate", {})
    if not agg:
        print("\n  ❌ No aggregate results — insufficient data")
        return

    print("\n" + "=" * 70)
    print("  🏁 EDA Quality Gate Verdict")
    print("=" * 70)

    corr = agg.get("avg_corr_tsmom", 1.0)
    sr = agg.get("avg_standalone_sr", 0)
    n_pos = agg.get("n_positive_sr", 0)
    n_total = agg.get("n_symbols", 0)

    gates = [
        ("G1: avg |corr(TSMOM)| < 0.30", abs(corr) < 0.30),
        ("G2: avg standalone SR > 0.5", sr > 0.5),
        ("G3: ≥6/8 symbols positive SR", n_pos >= 6),
    ]

    for name, passed in gates:
        print(f"    {'✅' if passed else '❌'} {name}")

    n_pass = sum(1 for _, p in gates if p)
    if n_pass == len(gates):
        print(f"\n    → GO: {n_pass}/{len(gates)} PASS — proceed to formal backtest")
    elif n_pass >= 2:
        print(f"\n    → WEAK GO: {n_pass}/{len(gates)} PASS — proceed with caution")
    else:
        print(f"\n    → FAIL: {n_pass}/{len(gates)} PASS — orderflow composite insufficient alpha")


def _print_final_verdict(eda: dict, bt: dict, blend: dict) -> None:
    """Print final verdict across all phases."""
    print("\n" + "=" * 70)
    print("  🏁 Final Verdict")
    print("=" * 70)

    # Backtest SR
    bt_srs = [v["sharpe"] for v in bt.values() if isinstance(v, dict) and "sharpe" in v]
    avg_bt_sr = np.mean(bt_srs) if bt_srs else 0
    n_pos_bt = sum(1 for s in bt_srs if s > 0)

    # Blend
    marginal_sr = blend.get("marginal_sr", 0) if blend else 0
    portfolio_corr = blend.get("portfolio_corr", 1.0) if blend else 1.0

    print(f"    Avg per-symbol SR:     {avg_bt_sr:.3f}")
    print(f"    Positive SR symbols:   {n_pos_bt}/{len(bt_srs)}")
    print(f"    Portfolio correlation:  {portfolio_corr:.3f}")
    print(f"    Marginal SR from blend: {marginal_sr:+.3f}")

    if avg_bt_sr > 1.0 and abs(portfolio_corr) < 0.30 and marginal_sr > 0:
        print("\n    → ✅ GO: Proceed to WFA/CPCV/DSR validation")
    elif avg_bt_sr > 0.5 and abs(portfolio_corr) < 0.40:
        print("\n    → ⚠️ WEAK GO: Marginal, parameter optimization may help")
    else:
        print("\n    → ❌ FAIL: Orderflow composite insufficient for production")


if __name__ == "__main__":
    main()
