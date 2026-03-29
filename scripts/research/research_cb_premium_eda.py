#!/usr/bin/env python3
"""
CB Premium EDA — Coinbase-Binance Price Spread as Standalone Signal

Economic intuition (华山论剑):
  - CB Premium > 0 → US institutional buying → bullish
  - CB Premium < 0 → Asia-led selling → bearish
  - Premium direction is a leading indicator of institutional sentiment

Signal construction:
  1. Fetch BTC/USD from Coinbase + BTC/USDT from Binance (1h)
  2. premium = (coinbase_close - binance_close) / binance_close
  3. Signal variants: raw premium, rolling z-score, EMA-smoothed
  4. Position: pctrank > threshold → long, pctrank < threshold → short

Gates (from research playbook):
  A1: IC sign consistency across time windows
  A2: IC magnitude (|IC| > 0.01 target)
  A3: IC stability (rolling IC std)
  A4: Quintile monotonicity
  A5: Symbol breadth (BTC-only for now, but test ETH too)
  G6: Correlation with existing factors (TSMOM, LSR, HTF)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from qtrade.data.ccxt_client import fetch_ccxt_klines
from qtrade.data.storage import load_klines

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
CB_CACHE = DATA_DIR / "coinbase"


# ═══════════════════════════════════════════════════════════
#  Step 1: Data Fetching & Alignment
# ═══════════════════════════════════════════════════════════

def fetch_and_cache_coinbase(symbol: str, interval: str, start: str) -> pd.DataFrame:
    """Fetch Coinbase klines with local parquet cache."""
    CB_CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = CB_CACHE / f"{symbol.replace('/', '_')}_{interval}.parquet"

    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        print(f"  📂 Coinbase cache loaded: {len(df)} bars ({df.index[0]} → {df.index[-1]})")
        return df

    print(f"  📥 Fetching Coinbase {symbol} {interval} from {start}...")
    # Coinbase API caps at 300 bars per request — must set limit to match
    df = fetch_ccxt_klines(
        symbol, interval, start,
        exchange="coinbaseexchange", limit_per_request=300,
    )

    if len(df) > 0:
        df.to_parquet(cache_path)
        print(f"  💾 Cached: {cache_path} ({len(df)} bars)")

    return df


def compute_cb_premium(binance_close: pd.Series, coinbase_close: pd.Series) -> pd.Series:
    """
    Compute Coinbase Premium = (CB_price - BN_price) / BN_price.

    Both series must share the same DatetimeIndex.
    Returns premium as a fraction (e.g., 0.001 = 0.1% premium).
    """
    # Align on common index
    common = binance_close.index.intersection(coinbase_close.index)
    bn = binance_close.loc[common]
    cb = coinbase_close.loc[common]

    premium = (cb - bn) / bn
    premium = premium.replace([np.inf, -np.inf], np.nan)

    return premium


# ═══════════════════════════════════════════════════════════
#  Step 2: Signal Variants
# ═══════════════════════════════════════════════════════════

def build_signal_variants(premium: pd.Series) -> dict[str, pd.Series]:
    """Build multiple signal variants from raw premium."""
    signals = {}

    # V1: Raw premium (smoothed with 4h EMA to remove noise)
    signals["premium_ema4h"] = premium.ewm(span=4, adjust=False).mean()

    # V2: 24h rolling z-score
    roll_24 = premium.rolling(24, min_periods=12)
    signals["premium_zscore_24h"] = (premium - roll_24.mean()) / roll_24.std().replace(0, np.nan)

    # V3: 168h (7d) rolling z-score — captures weekly institutional flow cycle
    roll_168 = premium.rolling(168, min_periods=84)
    signals["premium_zscore_168h"] = (premium - roll_168.mean()) / roll_168.std().replace(0, np.nan)

    # V4: EMA crossover (fast 12h vs slow 72h premium)
    ema_fast = premium.ewm(span=12, adjust=False).mean()
    ema_slow = premium.ewm(span=72, adjust=False).mean()
    signals["premium_ema_cross"] = ema_fast - ema_slow

    # V5: 24h SMA of premium (simplest smoothing)
    signals["premium_sma24h"] = premium.rolling(24, min_periods=12).mean()

    return signals


# ═══════════════════════════════════════════════════════════
#  Step 3: IC Analysis (A1-A5 gates)
# ═══════════════════════════════════════════════════════════

def compute_ic_table(
    signals: dict[str, pd.Series],
    forward_returns: pd.Series,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """Compute Information Coefficient for each signal variant × forward return horizon."""
    if horizons is None:
        horizons = [1, 4, 12, 24]

    rows = []
    for name, sig in signals.items():
        for h in horizons:
            fwd = forward_returns.shift(-h)
            common = sig.dropna().index.intersection(fwd.dropna().index)
            if len(common) < 500:
                continue
            ic = sig.loc[common].corr(fwd.loc[common])

            # Rolling IC stability (252-bar windows)
            rolling_ic = sig.loc[common].rolling(720).corr(fwd.loc[common])
            ic_std = rolling_ic.std()
            ic_mean = rolling_ic.mean()
            pct_positive = (rolling_ic > 0).mean()

            rows.append({
                "signal": name,
                "horizon_h": h,
                "IC": round(ic, 5),
                "IC_mean_rolling": round(ic_mean, 5) if not np.isnan(ic_mean) else np.nan,
                "IC_std_rolling": round(ic_std, 5) if not np.isnan(ic_std) else np.nan,
                "IC_IR": round(ic_mean / ic_std, 3) if ic_std > 0 else np.nan,
                "pct_positive": round(pct_positive, 3),
                "n_obs": len(common),
            })

    return pd.DataFrame(rows)


def quintile_analysis(
    signal: pd.Series,
    forward_returns: pd.Series,
    n_quantiles: int = 5,
    horizon: int = 24,
) -> pd.DataFrame:
    """Quintile spread analysis — does sorting by signal predict forward returns?"""
    fwd = forward_returns.shift(-horizon)
    common = signal.dropna().index.intersection(fwd.dropna().index)
    s = signal.loc[common]
    f = fwd.loc[common]

    # Percentile rank into quintiles
    pctrank = s.rolling(720, min_periods=180).rank(pct=True)
    bins = pd.cut(pctrank, bins=n_quantiles, labels=[f"Q{i+1}" for i in range(n_quantiles)])

    result = f.groupby(bins, observed=True).agg(["mean", "std", "count"])
    result.columns = ["mean_ret", "std_ret", "count"]
    result["sharpe"] = result["mean_ret"] / result["std_ret"] * np.sqrt(8760 / horizon)

    return result


# ═══════════════════════════════════════════════════════════
#  Step 4: Position Construction & Backtest
# ═══════════════════════════════════════════════════════════

def signal_to_position(
    signal: pd.Series,
    lookback: int = 720,
    long_threshold_pctile: float = 0.6,
    short_threshold_pctile: float = 0.4,
    rebalance_hours: int = 24,
) -> pd.Series:
    """Convert signal to position via percentile ranking with rebalance cadence."""
    pctrank = signal.rolling(lookback, min_periods=max(lookback // 4, 30)).rank(pct=True)

    n = len(signal)
    pos = np.zeros(n, dtype=np.float64)
    current = 0.0

    for i in range(n):
        if np.isnan(pctrank.iloc[i]):
            pos[i] = current
            continue

        if i % rebalance_hours == 0:
            if pctrank.iloc[i] > long_threshold_pctile:
                current = 1.0
            elif pctrank.iloc[i] < short_threshold_pctile:
                current = -1.0
            else:
                current = 0.0

        pos[i] = current

    return pd.Series(pos, index=signal.index)


def backtest_position(
    pos: pd.Series,
    close: pd.Series,
    fee_bps: float = 5,
    label: str = "",
) -> dict:
    """Simple backtest: pos * returns - fees on trades. Signal delay = 1 bar."""
    returns = close.pct_change().fillna(0)
    pos_delayed = pos.shift(1).fillna(0)

    strat_ret = pos_delayed * returns

    trades = pos_delayed.diff().abs().fillna(0)
    fee_cost = trades * (fee_bps / 10000)
    strat_ret -= fee_cost

    cumret = (1 + strat_ret).cumprod()
    total_ret = float(cumret.iloc[-1] - 1)
    n_years = len(strat_ret) / (365.25 * 24)
    cagr = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
    sr = float(strat_ret.mean() / strat_ret.std() * np.sqrt(8760)) if strat_ret.std() > 0 else 0
    peak = cumret.cummax()
    mdd = float(((cumret - peak) / peak).min())
    tim = float((pos_delayed.abs() > 0.01).mean())
    turnover = float(trades.sum() / len(trades))
    n_trades = int((trades > 0.01).sum())

    return {
        "label": label,
        "SR": round(sr, 3),
        "CAGR": f"{cagr:.1%}",
        "MDD": f"{mdd:.1%}",
        "Return": f"{total_ret:.1%}",
        "TIM": f"{tim:.1%}",
        "Turnover/bar": f"{turnover:.5f}",
        "N_trades": n_trades,
        "returns": strat_ret,
    }


# ═══════════════════════════════════════════════════════════
#  Step 5: Factor Orthogonality (G6 gate)
# ═══════════════════════════════════════════════════════════

def compute_factor_correlation(
    cb_premium_returns: pd.Series,
    symbol: str,
) -> dict:
    """Compute daily return correlation with existing factors (TSMOM baseline)."""
    corrs = {}

    # Load TSMOM baseline
    try:
        from qtrade.config import load_config
        from qtrade.backtest.run_backtest import run_symbol_backtest

        tsmom_cfg_path = BASE_DIR / "config" / "prod_candidate_simplified.yaml"
        tsmom_cfg = load_config(tsmom_cfg_path)
        mt = tsmom_cfg.market_type_str
        data_path = tsmom_cfg.data_dir / "binance" / mt / tsmom_cfg.market.interval / f"{symbol}.parquet"
        ref_path = DATA_DIR / "binance" / "futures" / "1h" / "BTCUSDT.parquet"
        ref_df = load_klines(ref_path)

        bt_cfg = tsmom_cfg.to_backtest_dict(symbol)
        if ref_df is not None:
            bt_cfg["_regime_gate_ref_df"] = ref_df

        res = run_symbol_backtest(symbol=symbol, data_path=data_path, cfg=bt_cfg)
        tsmom_ret = res.pf.value().pct_change().fillna(0)

        # Daily correlation
        strat_daily = cb_premium_returns.resample("D").sum()
        tsmom_daily = tsmom_ret.resample("D").sum()
        common = strat_daily.index.intersection(tsmom_daily.index)
        if len(common) > 30:
            corrs["TSMOM_baseline"] = round(strat_daily.loc[common].corr(tsmom_daily.loc[common]), 3)
    except Exception as e:
        print(f"    ⚠️ TSMOM correlation failed: {e}")

    return corrs


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("CB PREMIUM EDA — Coinbase-Binance Spread as Standalone Signal")
    print("Source: 华山论剑 (@huashanlunjians) methodology")
    print("=" * 70)

    # ── 1. Load data ──
    # Binance starts 2017-08, Coinbase 2015-01. Overlap from 2017-08.
    # Use 2020-01 for sufficient history with modern market structure.
    START = "2020-01-01"

    symbols_map = {
        "BTCUSDT": "BTC/USD",
        "ETHUSDT": "ETH/USD",
    }

    all_results = []

    for bn_sym, cb_sym in symbols_map.items():
        print(f"\n{'═' * 60}")
        print(f"  {bn_sym} vs Coinbase {cb_sym}")
        print(f"{'═' * 60}")

        # Load Binance klines
        bn_path = DATA_DIR / "binance" / "futures" / "1h" / f"{bn_sym}.parquet"
        if not bn_path.exists():
            print(f"  ⚠️ Binance data not found: {bn_path}")
            continue
        bn_df = load_klines(bn_path)
        bn_df = bn_df[bn_df.index >= START]
        print(f"  Binance: {len(bn_df)} bars ({bn_df.index[0]} → {bn_df.index[-1]})")

        # Fetch/load Coinbase klines
        cb_df = fetch_and_cache_coinbase(cb_sym, "1h", START)
        if len(cb_df) == 0:
            print(f"  ⚠️ Coinbase data empty")
            continue

        # ── 2. Compute premium ──
        premium = compute_cb_premium(bn_df["close"], cb_df["close"])
        coverage = len(premium.dropna()) / len(bn_df)
        print(f"\n  Premium computed: {len(premium.dropna())} bars (coverage: {coverage:.1%})")
        print(f"  Premium stats: mean={premium.mean():.6f}, std={premium.std():.6f}, "
              f"min={premium.min():.6f}, max={premium.max():.6f}")

        if coverage < 0.70:
            print(f"  ❌ Coverage {coverage:.1%} < 70% threshold — SKIP")
            continue

        # ── 3. Build signal variants ──
        signals = build_signal_variants(premium)
        print(f"\n  Signal variants: {list(signals.keys())}")

        # ── 4. Forward returns ──
        # Use Binance close for forward returns (this is what we'd trade)
        bn_close_aligned = bn_df["close"].loc[premium.dropna().index]
        fwd_ret_1h = bn_close_aligned.pct_change().shift(-1)

        # ── 5. IC Analysis (A1-A5) ──
        print(f"\n  {'─' * 50}")
        print(f"  IC ANALYSIS (A1-A5 gates)")
        print(f"  {'─' * 50}")

        ic_table = compute_ic_table(signals, fwd_ret_1h, horizons=[1, 4, 12, 24])
        if len(ic_table) > 0:
            for _, row in ic_table.iterrows():
                flag = "✅" if abs(row["IC"]) >= 0.005 else "⚠️"
                print(f"    {flag} {row['signal']:25s} h={row['horizon_h']:2d}h  "
                      f"IC={row['IC']:+.5f}  IC_IR={row['IC_IR']:.3f}  "
                      f"pct+={row['pct_positive']:.1%}  n={row['n_obs']}")

        # Find best signal
        if len(ic_table) == 0:
            print("  ❌ No valid IC results — STOP")
            continue

        # Best by |IC| at 24h horizon
        ic_24h = ic_table[ic_table["horizon_h"] == 24].copy()
        if len(ic_24h) > 0:
            best_row = ic_24h.loc[ic_24h["IC"].abs().idxmax()]
            best_signal_name = best_row["signal"]
            print(f"\n  🏆 Best signal (24h): {best_signal_name} IC={best_row['IC']:+.5f}")
        else:
            best_signal_name = list(signals.keys())[0]

        # ── 6. Quintile Analysis (A4) ──
        print(f"\n  {'─' * 50}")
        print(f"  QUINTILE ANALYSIS — {best_signal_name}")
        print(f"  {'─' * 50}")

        best_sig = signals[best_signal_name]
        qt = quintile_analysis(best_sig, fwd_ret_1h, n_quantiles=5, horizon=24)
        print(qt.to_string())

        # Check monotonicity
        mean_rets = qt["mean_ret"].values
        if len(mean_rets) >= 3:
            q1_q5_spread = mean_rets[-1] - mean_rets[0]
            monotonic = all(mean_rets[i] <= mean_rets[i + 1] for i in range(len(mean_rets) - 1))
            print(f"\n  Q1→Q5 spread: {q1_q5_spread:.6f}  Monotonic: {'✅' if monotonic else '❌'}")

        # ── 7. Backtest — multiple parameter sets ──
        print(f"\n  {'─' * 50}")
        print(f"  BACKTEST — Signal: {best_signal_name}")
        print(f"  {'─' * 50}")

        close_for_bt = bn_close_aligned

        param_sets = [
            {"long_th": 0.7, "short_th": 0.3, "rebal": 24, "tag": "70/30 daily"},
            {"long_th": 0.6, "short_th": 0.4, "rebal": 24, "tag": "60/40 daily"},
            {"long_th": 0.7, "short_th": 0.3, "rebal": 4,  "tag": "70/30 4h"},
            {"long_th": 0.6, "short_th": 0.4, "rebal": 4,  "tag": "60/40 4h"},
            # Long-only variants
            {"long_th": 0.6, "short_th": -1.0, "rebal": 24, "tag": "LO 60 daily"},
            {"long_th": 0.5, "short_th": -1.0, "rebal": 24, "tag": "LO 50 daily"},
        ]

        best_bt = None
        for ps in param_sets:
            pos = signal_to_position(
                best_sig,
                lookback=720,
                long_threshold_pctile=ps["long_th"],
                short_threshold_pctile=ps["short_th"],
                rebalance_hours=ps["rebal"],
            )
            if ps["short_th"] < 0:
                pos = pos.clip(0, 1)  # long-only

            bt = backtest_position(pos, close_for_bt, fee_bps=5, label=f"{bn_sym} {ps['tag']}")
            print(f"    {ps['tag']:18s}  SR={bt['SR']:+.3f}  CAGR={bt['CAGR']}  "
                  f"MDD={bt['MDD']}  TIM={bt['TIM']}  trades={bt['N_trades']}")

            if best_bt is None or abs(bt["SR"]) > abs(best_bt["SR"]):
                best_bt = bt

            all_results.append(bt)

        # ── 8. G6: Factor Correlation ──
        if best_bt is not None:
            print(f"\n  {'─' * 50}")
            print(f"  G6: FACTOR CORRELATION — {best_bt['label']}")
            print(f"  {'─' * 50}")

            corrs = compute_factor_correlation(best_bt["returns"], bn_sym)
            for factor, corr in corrs.items():
                flag = "✅" if abs(corr) < 0.3 else "⚠️"
                print(f"    {flag} {factor}: {corr:+.3f}")

    # ═══════════════════════════════════════════════════════════
    #  Summary & Verdict
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'═' * 70}")
    print("SUMMARY & VERDICT")
    print(f"{'═' * 70}")

    if not all_results:
        print("  ❌ No results — check data availability")
        return

    print(f"\n  Total configs tested: {len(all_results)}")
    for r in all_results:
        print(f"    {r['label']:30s}  SR={r['SR']:+.3f}  MDD={r['MDD']}  TIM={r['TIM']}")

    # Gate summary
    print(f"\n  {'─' * 40}")
    print("  GATE CHECKLIST:")
    positive_sr = [r for r in all_results if r["SR"] > 0]
    print(f"    Configs with SR > 0: {len(positive_sr)}/{len(all_results)}")

    if len(positive_sr) > 0:
        best_overall = max(positive_sr, key=lambda x: x["SR"])
        print(f"    Best config: {best_overall['label']} SR={best_overall['SR']}")
        print(f"\n  → VERDICT: {'GO' if best_overall['SR'] > 0.5 else 'WEAK GO' if best_overall['SR'] > 0 else 'FAIL'}")
        print(f"    Next step: Implement as standalone strategy if IC > 0.01 and SR > 0.5")
    else:
        print(f"\n  → VERDICT: FAIL — no positive SR configs")


if __name__ == "__main__":
    main()
