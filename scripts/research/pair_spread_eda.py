#!/usr/bin/env python3
"""
Crypto Pair Spread — Quick EDA

Evaluate whether crypto pair spreads (BTC/ETH, BTC/SOL, ETH/SOL)
exhibit mean-reversion characteristics suitable for a standalone strategy.

Key questions:
1. Does the log spread mean-revert? (Half-life < 168h = 1 week)
2. Is the spread z-score predictive? (IC of z-score → next-period spread return)
3. What is the Hurst exponent? (H < 0.5 = mean-reverting)
4. How many trading opportunities per year?
5. What is the gross expectancy of a simple z-score strategy?

Task: research_20260325_163400_higher_freq_complement
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings("ignore", category=UserWarning)
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from qtrade.data.storage import load_klines

DATA_DIR = Path("data/binance/futures")
PAIRS = [
    ("BTCUSDT", "ETHUSDT"),
    ("BTCUSDT", "SOLUSDT"),
    ("ETHUSDT", "SOLUSDT"),
]
INTERVALS = ["1h", "15m"]


def load_pair(sym_a: str, sym_b: str, interval: str) -> pd.DataFrame | None:
    pa = DATA_DIR / interval / f"{sym_a}.parquet"
    pb = DATA_DIR / interval / f"{sym_b}.parquet"
    if not pa.exists() or not pb.exists():
        return None
    da = load_klines(pa)
    db = load_klines(pb)
    common = da.index.intersection(db.index)
    if len(common) < 1000:
        return None
    df = pd.DataFrame({
        f"{sym_a}_close": da.loc[common, "close"],
        f"{sym_b}_close": db.loc[common, "close"],
    })
    df["log_spread"] = np.log(df.iloc[:, 0]) - np.log(df.iloc[:, 1])
    return df


def half_life(spread: pd.Series) -> float:
    """OLS regression to estimate mean-reversion half-life."""
    spread = spread.dropna()
    y = spread.diff().dropna()
    x = spread.shift(1).dropna()
    common = y.index.intersection(x.index)
    y, x = y.loc[common], x.loc[common]
    if len(y) < 100:
        return float("inf")
    slope, intercept, r_value, p_value, std_err = sp_stats.linregress(x, y)
    if slope >= 0:
        return float("inf")
    hl = -np.log(2) / slope
    return hl


def hurst_exponent(ts: pd.Series, max_lag: int = 100) -> float:
    """Simplified Hurst exponent via R/S analysis."""
    ts = ts.dropna().values
    if len(ts) < max_lag * 2:
        return 0.5
    lags = range(2, max_lag + 1)
    rs = []
    for lag in lags:
        chunks = [ts[i:i + lag] for i in range(0, len(ts) - lag, lag)]
        if len(chunks) < 5:
            continue
        vals = []
        for chunk in chunks:
            mean_adj = chunk - chunk.mean()
            cumdev = np.cumsum(mean_adj)
            r = cumdev.max() - cumdev.min()
            s = chunk.std()
            if s > 0:
                vals.append(r / s)
        if vals:
            rs.append((lag, np.mean(vals)))
    if len(rs) < 3:
        return 0.5
    log_lags = np.log([r[0] for r in rs])
    log_rs = np.log([r[1] for r in rs])
    slope, _, _, _, _ = sp_stats.linregress(log_lags, log_rs)
    return slope


def zscore_ic(spread: pd.Series, lookback: int, horizon: int) -> float:
    """IC: z-score of spread → future spread return."""
    mu = spread.rolling(lookback).mean()
    sigma = spread.rolling(lookback).std()
    z = (spread - mu) / sigma
    fwd_ret = spread.diff(horizon).shift(-horizon)
    valid = z.dropna().index.intersection(fwd_ret.dropna().index)
    if len(valid) < 100:
        return 0.0
    return z.loc[valid].corr(fwd_ret.loc[valid])


def simulate_zscore_strategy(
    spread: pd.Series,
    lookback: int = 168,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    tp_z: float = 0.0,
    max_hold: int = 168,
) -> dict:
    """Simple z-score MR strategy on the spread. Returns trade-level stats."""
    mu = spread.rolling(lookback).mean()
    sigma = spread.rolling(lookback).std()
    z = ((spread - mu) / sigma).values
    spread_vals = spread.values
    n = len(spread)

    trades = []
    holding = 0
    entry_idx = -1
    entry_spread = 0.0

    warmup = lookback + 10
    for i in range(warmup, n):
        if np.isnan(z[i]):
            continue

        if holding != 0:
            bars_held = i - entry_idx
            spread_pnl = (spread_vals[i] - entry_spread) * (-holding)

            exit_signal = False
            if holding == -1 and z[i] <= exit_z:
                exit_signal = True
            elif holding == 1 and z[i] >= -exit_z:
                exit_signal = True
            elif bars_held >= max_hold:
                exit_signal = True

            if exit_signal:
                trades.append({
                    "entry_idx": entry_idx,
                    "exit_idx": i,
                    "direction": holding,
                    "bars_held": bars_held,
                    "spread_pnl": spread_pnl,
                })
                holding = 0
        else:
            if z[i] > entry_z:
                holding = -1
                entry_idx = i
                entry_spread = spread_vals[i]
            elif z[i] < -entry_z:
                holding = 1
                entry_idx = i
                entry_spread = spread_vals[i]

    if not trades:
        return {"n_trades": 0}

    df_trades = pd.DataFrame(trades)
    wins = df_trades[df_trades["spread_pnl"] > 0]
    losses = df_trades[df_trades["spread_pnl"] <= 0]

    years = (n - warmup) / (365.25 * 24 * (4 if len(spread) > 50000 else 1))
    if years <= 0:
        years = 1

    return {
        "n_trades": len(df_trades),
        "trades_per_year": round(len(df_trades) / years, 1),
        "win_rate": round(len(wins) / len(df_trades) * 100, 1) if len(df_trades) > 0 else 0,
        "avg_win": round(wins["spread_pnl"].mean(), 6) if len(wins) > 0 else 0,
        "avg_loss": round(losses["spread_pnl"].mean(), 6) if len(losses) > 0 else 0,
        "avg_hold": round(df_trades["bars_held"].mean(), 1),
        "total_spread_pnl": round(df_trades["spread_pnl"].sum(), 4),
        "gross_expectancy": round(df_trades["spread_pnl"].mean(), 6),
    }


def main():
    print("=" * 70)
    print("CRYPTO PAIR SPREAD — EDA")
    print("=" * 70)

    for interval in INTERVALS:
        print(f"\n{'━' * 70}")
        print(f"  INTERVAL: {interval}")
        print(f"{'━' * 70}")

        for sym_a, sym_b in PAIRS:
            pair_label = f"{sym_a[:3]}/{sym_b[:3]}"
            print(f"\n  ── {pair_label} ({interval}) ──")

            df = load_pair(sym_a, sym_b, interval)
            if df is None:
                print(f"    Data not available")
                continue

            spread = df["log_spread"]
            print(f"    Bars: {len(spread):,}")
            print(f"    Period: {spread.index[0].date()} → {spread.index[-1].date()}")

            hl = half_life(spread)
            print(f"    Half-life: {hl:.1f} bars", end="")
            if interval == "1h":
                print(f" ({hl:.1f}h)")
            else:
                print(f" ({hl * 0.25:.1f}h)")

            h = hurst_exponent(spread)
            print(f"    Hurst exponent: {h:.4f}", end="")
            if h < 0.45:
                print(" ✅ (mean-reverting)")
            elif h < 0.55:
                print(" ⚠️ (random walk)")
            else:
                print(" ❌ (trending)")

            for lookback in [24, 72, 168]:
                ic = zscore_ic(spread, lookback, 24 if interval == "1h" else 96)
                label = f"{lookback}{'h' if interval == '1h' else 'b'}"
                print(f"    IC(z_{label} → 24h): {ic:+.4f}", end="")
                if abs(ic) > 0.03:
                    print(" ✅")
                elif abs(ic) > 0.01:
                    print(" ~")
                else:
                    print(" ❌")

            for entry_z, label in [(1.5, "z1.5"), (2.0, "z2.0"), (2.5, "z2.5")]:
                result = simulate_zscore_strategy(
                    spread,
                    lookback=168 if interval == "1h" else 672,
                    entry_z=entry_z,
                    exit_z=0.5,
                    max_hold=168 if interval == "1h" else 672,
                )
                if result["n_trades"] == 0:
                    print(f"    Strategy({label}): 0 trades")
                    continue
                wr = result["win_rate"]
                ge = result["gross_expectancy"]
                tpy = result["trades_per_year"]
                ah = result["avg_hold"]
                print(
                    f"    Strategy({label}): "
                    f"trades/yr={tpy:.0f}, "
                    f"WR={wr:.0f}%, "
                    f"avg_hold={ah:.0f}bars, "
                    f"gross_exp={ge:+.6f}"
                    f" {'✅' if ge > 0 else '❌'}"
                )

    print(f"\n{'=' * 70}")
    print("INTERPRETATION GUIDE")
    print(f"{'=' * 70}")
    print("  Half-life < 168h (1 week): spread mean-reverts fast enough for trading")
    print("  Hurst < 0.45: strong mean-reversion tendency")
    print("  IC > 0.03: z-score has predictive power for spread returns")
    print("  Gross exp > 0: basic z-score strategy has positive edge before costs")
    print()
    print("NOTE: This EDA tests LOG SPREAD mean-reversion (ratio trading).")
    print("  Actual implementation would need: hedged position sizing,")
    print("  rebalancing costs, and funding rate differential consideration.")


if __name__ == "__main__":
    main()
