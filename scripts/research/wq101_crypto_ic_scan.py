#!/usr/bin/env python3
"""
WorldQuant 101 Alphas — Crypto IC Scan

Adapts the 101 Formulaic Alphas (Kakushadze 2016, arXiv:1601.00991) from
equities cross-sectional to single-asset time-series for BTC/ETH on 1h candles.

Key adaptations:
  - rank() → time-series percentile rank (rolling 720 bars) instead of cross-sectional rank
  - IndNeutralize → removed (single asset, no sectors)
  - adv20 → 20-period rolling average volume
  - vwap → (high + low + close) / 3 (typical price proxy on 1h OHLCV)
  - Returns → 1-bar log returns

For each alpha, computes:
  - IC (Spearman corr with forward 1h, 4h, 24h returns)
  - IC sign consistency (pct of rolling windows with same sign)
  - Auto-correlation (signal persistence)

Output: ranked table of alphas by |IC| at 24h horizon.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from qtrade.data.storage import load_klines

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"


# ═══════════════════════════════════════════════════════════
#  Adapted helper functions (time-series, not cross-sectional)
# ═══════════════════════════════════════════════════════════

def ts_sum(s: pd.Series, window: int = 10) -> pd.Series:
    return s.rolling(window, min_periods=window // 2).sum()

def ts_mean(s: pd.Series, window: int = 10) -> pd.Series:
    return s.rolling(window, min_periods=window // 2).mean()

def ts_std(s: pd.Series, window: int = 10) -> pd.Series:
    return s.rolling(window, min_periods=window // 2).std()

def ts_rank(s: pd.Series, window: int = 10) -> pd.Series:
    return s.rolling(window, min_periods=window // 2).rank(pct=True)

def ts_corr(x: pd.Series, y: pd.Series, window: int = 10) -> pd.Series:
    return x.rolling(window, min_periods=window // 2).corr(y)

def ts_cov(x: pd.Series, y: pd.Series, window: int = 10) -> pd.Series:
    return x.rolling(window, min_periods=window // 2).cov(y)

def ts_argmax(s: pd.Series, window: int = 10) -> pd.Series:
    return s.rolling(window).apply(np.argmax, raw=True) + 1

def ts_argmin(s: pd.Series, window: int = 10) -> pd.Series:
    return s.rolling(window).apply(np.argmin, raw=True) + 1

def ts_min(s: pd.Series, window: int = 10) -> pd.Series:
    return s.rolling(window, min_periods=window // 2).min()

def ts_max(s: pd.Series, window: int = 10) -> pd.Series:
    return s.rolling(window, min_periods=window // 2).max()

def ts_product(s: pd.Series, window: int = 10) -> pd.Series:
    return s.rolling(window).apply(np.prod, raw=True)

def delta(s: pd.Series, period: int = 1) -> pd.Series:
    return s.diff(period)

def delay(s: pd.Series, period: int = 1) -> pd.Series:
    return s.shift(period)

def rank(s: pd.Series, window: int = 720) -> pd.Series:
    """Time-series percentile rank (replacing cross-sectional rank)."""
    return s.rolling(window, min_periods=window // 4).rank(pct=True)

def signed_power(s: pd.Series, exp: float) -> pd.Series:
    return s.abs().pow(exp) * np.sign(s)

def decay_linear(s: pd.Series, period: int = 10) -> pd.Series:
    """Linear weighted moving average."""
    weights = np.arange(1, period + 1, dtype=float)
    weights /= weights.sum()
    return s.rolling(period, min_periods=period // 2).apply(
        lambda x: np.dot(x[-len(weights):], weights[-len(x):]) if len(x) >= len(weights) else np.nan,
        raw=True,
    )


# ═══════════════════════════════════════════════════════════
#  Alpha definitions (crypto-adapted subset)
# ═══════════════════════════════════════════════════════════

def compute_alphas(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Compute selected WQ101 alphas adapted for single-asset crypto."""
    close = df["close"]
    open_ = df["open"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    returns = np.log(close / close.shift(1))
    vwap = (high + low + close) / 3  # typical price as VWAP proxy
    adv20 = volume.rolling(20, min_periods=10).mean()

    alphas = {}

    # Alpha#1: rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5
    cond = returns.copy()
    cond[returns >= 0] = close[returns >= 0]
    cond[returns < 0] = ts_std(returns, 20)[returns < 0]
    alphas["wq001"] = rank(ts_argmax(signed_power(cond, 2), 5)) - 0.5

    # Alpha#5: -(ts_rank(open, 5) * ts_rank(returns, 5) * ts_rank(volume, 5))
    # Skipped — needs cross-sectional rank, not useful for single asset

    # Alpha#7: (adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1)
    cond7 = pd.Series(-1.0, index=df.index)
    mask = adv20 < volume
    cond7[mask] = (-1 * ts_rank(delta(close, 7).abs(), 60) * np.sign(delta(close, 7)))[mask]
    alphas["wq007"] = cond7

    # Alpha#9: trend/reversion switcher
    d1 = delta(close, 1)
    cond_a = ts_min(d1, 5) > 0
    cond_b = ts_max(d1, 5) < 0
    a9 = d1.copy()
    a9[~cond_a & ~cond_b] = -d1[~cond_a & ~cond_b]
    alphas["wq009"] = a9

    # Alpha#11: (rank(ts_max(vwap-close, 3)) + rank(ts_min(vwap-close, 3))) * rank(delta(volume, 3))
    alphas["wq011"] = (rank(ts_max(vwap - close, 3)) + rank(ts_min(vwap - close, 3))) * rank(delta(volume, 3))

    # Alpha#12: sign(delta(volume, 1)) * (-1 * delta(close, 1))
    alphas["wq012"] = np.sign(delta(volume, 1)) * (-1 * delta(close, 1))

    # Alpha#17: -rank(ts_rank(close, 10)) * rank(delta(delta(close, 1), 1)) * rank(ts_rank(volume/adv20, 5))
    alphas["wq017"] = -rank(ts_rank(close, 10)) * rank(delta(delta(close, 1), 1)) * rank(ts_rank(volume / adv20, 5))

    # Alpha#25: rank(((-1 * returns) * adv20 * vwap * (high - close)))
    alphas["wq025"] = rank((-1 * returns) * adv20 * vwap * (high - close))

    # Alpha#26: -ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3)
    alphas["wq026"] = -ts_max(ts_corr(ts_rank(volume, 5), ts_rank(high, 5), 5), 3)

    # Alpha#28: scale(corr(adv20, low, 5) + ((high+low)/2 - close))
    # scale() doesn't make sense for single-asset; use raw signal
    alphas["wq028"] = ts_corr(adv20, low, 5) + ((high + low) / 2 - close)

    # Alpha#33: rank(-1 + (open/close))
    alphas["wq033"] = rank(-1 + (open_ / close))

    # Alpha#34: rank(((1-rank(stddev(returns, 2)/stddev(returns, 5))) + (1-rank(delta(close, 1)))))
    alphas["wq034"] = rank((1 - rank(ts_std(returns, 2) / ts_std(returns, 5).replace(0, np.nan))) + (1 - rank(delta(close, 1))))

    # Alpha#38: -rank(ts_rank(close, 10)) * rank(close / open)
    alphas["wq038"] = -rank(ts_rank(close, 10)) * rank(close / open_)

    # Alpha#41: pow(high * low, 0.5) - vwap
    alphas["wq041"] = np.sqrt(high * low) - vwap

    # Alpha#42: rank(vwap - close) / rank(vwap + close)
    alphas["wq042"] = rank(vwap - close) / rank(vwap + close).replace(0, np.nan)

    # Alpha#43: ts_rank(volume/adv20, 20) * ts_rank(-delta(close, 7), 8)
    alphas["wq043"] = ts_rank(volume / adv20, 20) * ts_rank(-delta(close, 7), 8)

    # Alpha#44: -corr(high, rank(volume), 5)
    alphas["wq044"] = -ts_corr(high, rank(volume, 168), 5)

    # Alpha#46: ((delay(close,20)-delay(close,10))/10-(delay(close,10)-close)/10) conditional
    d20_10 = (delay(close, 20) - delay(close, 10)) / 10
    d10_0 = (delay(close, 10) - close) / 10
    mean_chg = d20_10 - d10_0
    alphas["wq046"] = pd.Series(np.where(mean_chg > 0.25, -1, np.where(mean_chg < 0, 1, -(close - delay(close, 1)))), index=df.index)

    # Alpha#52: (-delta(ts_min(low, 5), 5)) * rank((sum(returns, 240) - sum(returns, 20)) / 220) * ts_rank(volume, 5)
    alphas["wq052"] = (-delta(ts_min(low, 5), 5)) * rank((ts_sum(returns, 240) - ts_sum(returns, 20)) / 220) * ts_rank(volume, 5)

    # Alpha#53: -delta((close-low)-(high-close))/(close-low), 9)
    cl_ratio = ((close - low) - (high - close)) / (close - low).replace(0, np.nan)
    alphas["wq053"] = -delta(cl_ratio, 9)

    # Alpha#55: -corr(rank((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12))), rank(volume), 6)
    pct_k = (close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12)).replace(0, np.nan)
    alphas["wq055"] = -ts_corr(rank(pct_k), rank(volume, 168), 6)

    # Alpha#60: -scale(rank(((close-low)-(high-close))/(high-low))*volume) - scale(rank(ts_argmax(close,10)))
    body_ratio = ((close - low) - (high - close)) / (high - low).replace(0, np.nan) * volume
    alphas["wq060"] = -rank(body_ratio) - rank(ts_argmax(close, 10))

    # Alpha#101: (close - open) / ((high - low) + 0.001)
    alphas["wq101"] = (close - open_) / ((high - low) + 0.001)

    # ── Custom crypto-specific alphas ──

    # C1: Volume surge × price reversal (inspired by WQ#43)
    vol_surge = volume / adv20
    alphas["crypto_vol_reversal"] = ts_rank(vol_surge, 24) * ts_rank(-delta(close, 4), 12)

    # C2: High-low range expansion (volatility breakout)
    hl_range = (high - low) / close
    range_z = (hl_range - ts_mean(hl_range, 168)) / ts_std(hl_range, 168).replace(0, np.nan)
    alphas["crypto_range_breakout"] = range_z * np.sign(returns)

    # C3: Weekend momentum (ACR Journal 2025: BTC weekend returns 2x weekday)
    is_weekend = pd.Series(df.index.dayofweek.isin([5, 6]).astype(float), index=df.index)
    alphas["crypto_weekend_mom"] = is_weekend * ts_rank(close, 168)

    # C4: Time-of-day weight (European/US overlap 13:00-16:30 UTC strongest)
    hour = pd.Series(df.index.hour, index=df.index, dtype=float)
    time_weight = pd.Series(1.0, index=df.index)
    time_weight[(hour >= 13) & (hour <= 16)] = 1.5  # EU/US overlap
    time_weight[(hour >= 0) & (hour <= 6)] = 0.7     # Asian hours
    alphas["crypto_time_weight"] = time_weight * returns

    return alphas


# ═══════════════════════════════════════════════════════════
#  IC Scan
# ═══════════════════════════════════════════════════════════

def run_ic_scan(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Compute IC for all alphas against forward returns."""
    print(f"\n{'═' * 60}")
    print(f"  IC SCAN — {symbol} ({len(df)} bars)")
    print(f"{'═' * 60}")

    alphas = compute_alphas(df)
    close = df["close"]
    fwd_1h = close.pct_change().shift(-1)
    fwd_4h = close.pct_change(4).shift(-4)
    fwd_24h = close.pct_change(24).shift(-24)

    rows = []
    for name, signal in alphas.items():
        sig = signal.replace([np.inf, -np.inf], np.nan).dropna()
        if len(sig) < 1000:
            continue

        for horizon, fwd, label in [(1, fwd_1h, "1h"), (4, fwd_4h, "4h"), (24, fwd_24h, "24h")]:
            common = sig.index.intersection(fwd.dropna().index)
            if len(common) < 500:
                continue

            ic = sig.loc[common].corr(fwd.loc[common])

            # Rolling IC stability
            rolling_ic = sig.loc[common].rolling(720).corr(fwd.loc[common])
            pct_positive = (rolling_ic > 0).mean() if len(rolling_ic.dropna()) > 0 else np.nan

            # Signal autocorrelation (persistence)
            autocorr = sig.autocorr(lag=1)

            rows.append({
                "alpha": name,
                "symbol": symbol,
                "horizon": label,
                "IC": round(ic, 5),
                "pct_positive": round(pct_positive, 3) if not np.isnan(pct_positive) else np.nan,
                "autocorr": round(autocorr, 3),
                "n_obs": len(common),
            })

    return pd.DataFrame(rows)


def main():
    print("=" * 70)
    print("WorldQuant 101 Alphas — Crypto IC Scan")
    print("Adapted from Kakushadze 2016 (arXiv:1601.00991)")
    print("=" * 70)

    symbols = ["BTCUSDT", "ETHUSDT"]
    all_results = []

    for sym in symbols:
        kline_path = DATA_DIR / "binance" / "futures" / "1h" / f"{sym}.parquet"
        if not kline_path.exists():
            print(f"⚠️ {sym} not found: {kline_path}")
            continue

        df = load_klines(kline_path)
        df = df[df.index >= "2020-01-01"]

        result = run_ic_scan(df, sym)
        all_results.append(result)

    if not all_results:
        print("❌ No results")
        return

    combined = pd.concat(all_results, ignore_index=True)

    # Summary: best alphas at 24h horizon
    print(f"\n{'═' * 70}")
    print("TOP ALPHAS BY |IC| AT 24h HORIZON")
    print(f"{'═' * 70}")

    ic_24h = combined[combined["horizon"] == "24h"].copy()
    ic_24h["abs_IC"] = ic_24h["IC"].abs()
    ic_24h_avg = ic_24h.groupby("alpha").agg(
        avg_IC=("IC", "mean"),
        avg_abs_IC=("abs_IC", "mean"),
        avg_pct_pos=("pct_positive", "mean"),
        avg_autocorr=("autocorr", "mean"),
        symbols=("symbol", "count"),
    ).sort_values("avg_abs_IC", ascending=False)

    for _, row in ic_24h_avg.head(20).iterrows():
        flag = "✅" if row["avg_abs_IC"] >= 0.005 else "⚠️"
        print(f"  {flag} {row.name:25s}  IC={row['avg_IC']:+.5f}  "
              f"|IC|={row['avg_abs_IC']:.5f}  pct+={row['avg_pct_pos']:.1%}  "
              f"autocorr={row['avg_autocorr']:.2f}  syms={int(row['symbols'])}")

    # Multi-horizon view for top alphas
    top_names = ic_24h_avg.head(10).index.tolist()
    print(f"\n{'═' * 70}")
    print("MULTI-HORIZON IC — TOP 10 ALPHAS")
    print(f"{'═' * 70}")

    for name in top_names:
        rows = combined[combined["alpha"] == name]
        print(f"\n  {name}:")
        for _, r in rows.iterrows():
            print(f"    {r['symbol']} h={r['horizon']:3s}  IC={r['IC']:+.5f}  pct+={r['pct_positive']:.1%}")

    # Save results
    out_path = BASE_DIR / "reports" / "research" / "wq101_ic_scan.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    print(f"\n✅ Full results saved: {out_path}")


if __name__ == "__main__":
    main()
