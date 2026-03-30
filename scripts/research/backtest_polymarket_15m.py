#!/usr/bin/env python3
"""
Polymarket 15-Minute Strategy — Correct Backtest

Fixes 5 flaws from the original inline backtest:
  1. TA computed on 5m sub-bars (matching live bot's 1m resolution)
  2. Sweet spot simulation (TA includes first 1-2 bars of current window)
  3. Odds estimation (maps price displacement to implied Polymarket odds)
  4. Signal selectivity tracking
  5. Proper window alignment

Tests 6 strategies head-to-head:
  - Random, Always Up, Momentum, Mean Reversion, Contrarian Cheap, Krajekis TA
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from qtrade.indicators import calculate_rsi, calculate_macd, calculate_ema, calculate_vwap, calculate_atr
from qtrade.polymarket.binance_feed import TASignals
from qtrade.polymarket.krajekis_strategy import determine_session, classify_volatility, determine_direction

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


# ═══════════════════════════════════════════════════════════
#  Step 1: Build 15-minute windows from 5m sub-bars
# ═══════════════════════════════════════════════════════════

def build_windows(df_5m: pd.DataFrame) -> list[dict]:
    """
    Group 5m bars into 15m windows.
    Each window has 3 sub-bars (0m, 5m, 10m).

    Returns list of dicts with:
        window_start, window_open, window_close, window_up (bool),
        sub_bars (list of 3 rows), sweet_spot_idx (index into df_5m)
    """
    windows = []

    # Align to 15-minute boundaries
    df_5m = df_5m.copy()
    df_5m["window_start"] = df_5m.index.floor("15min")

    grouped = df_5m.groupby("window_start")

    for window_start, group in grouped:
        if len(group) != 3:
            continue  # incomplete window

        window_open = group["open"].iloc[0]
        window_close = group["close"].iloc[-1]
        window_up = window_close > window_open

        # Sweet spot: after 2nd sub-bar (10 min into window)
        sweet_spot_idx = group.index[1]  # end of 2nd 5m bar = 10 min in

        windows.append({
            "window_start": window_start,
            "window_open": window_open,
            "window_close": window_close,
            "window_up": window_up,
            "sweet_spot_idx": sweet_spot_idx,
            "price_at_sweet_spot": group["close"].iloc[1],  # price at 10 min
        })

    return windows


# ═══════════════════════════════════════════════════════════
#  Step 2: Compute TA at sweet spot (matching live bot)
# ═══════════════════════════════════════════════════════════

def compute_ta_at_sweet_spot(
    df_5m: pd.DataFrame,
    sweet_spot_idx: pd.Timestamp,
    lookback: int = 100,
) -> TASignals | None:
    """
    Compute TA signals using 5m data up to the sweet spot.
    Mimics live bot's compute_ta_signals() but on historical data.
    """
    # Get position of sweet_spot_idx
    idx_pos = df_5m.index.get_loc(sweet_spot_idx)
    if idx_pos < lookback:
        return None

    # Slice: last 'lookback' bars ending at sweet spot (inclusive)
    window = df_5m.iloc[idx_pos - lookback + 1:idx_pos + 1]
    close = window["close"]
    volume = window["volume"]

    if len(close) < 60:
        return None

    # TA indicators (same as binance_feed.py)
    rsi = calculate_rsi(close, 14)
    macd_result = calculate_macd(close)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    vwap = calculate_vwap(window)

    # MACD histogram
    if isinstance(macd_result, dict):
        macd_hist = macd_result.get("histogram", pd.Series(0, index=close.index))
    elif isinstance(macd_result, pd.DataFrame):
        cols = [c for c in macd_result.columns if "hist" in c.lower()]
        macd_hist = macd_result[cols[0]] if cols else macd_result.iloc[:, -1]
    else:
        macd_hist = pd.Series(0.0, index=close.index)

    # Volume ratio
    vol_avg = volume.rolling(20).mean()
    vol_ratio = float(volume.iloc[-1] / vol_avg.iloc[-1]) if vol_avg.iloc[-1] > 0 else 1.0

    # ATR: resample 5m to 15m for ATR calculation
    df_15m = window.resample("15min").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna()

    if len(df_15m) >= 15:
        atr = calculate_atr(df_15m, 14)
        atr_val = float(atr.iloc[-1]) if not atr.empty and not pd.isna(atr.iloc[-1]) else 0.0
    else:
        atr_val = 0.0

    return TASignals(
        close=float(close.iloc[-1]),
        vwap=float(vwap.iloc[-1]) if not vwap.empty else float(close.iloc[-1]),
        ema_21=float(ema_21.iloc[-1]) if not ema_21.empty else float(close.iloc[-1]),
        ema_50=float(ema_50.iloc[-1]) if not ema_50.empty else float(close.iloc[-1]),
        rsi_14=float(rsi.iloc[-1]) if not rsi.empty and not pd.isna(rsi.iloc[-1]) else 50.0,
        macd_hist=float(macd_hist.iloc[-1]) if len(macd_hist) > 0 and not pd.isna(macd_hist.iloc[-1]) else 0.0,
        macd_hist_prev=float(macd_hist.iloc[-2]) if len(macd_hist) > 1 and not pd.isna(macd_hist.iloc[-2]) else 0.0,
        atr_14=atr_val,
        volume_ratio=vol_ratio,
    )


# ═══════════════════════════════════════════════════════════
#  Step 3: Odds estimation
# ═══════════════════════════════════════════════════════════

def estimate_odds(price_at_entry: float, window_open: float) -> tuple[float, float]:
    """
    Estimate Polymarket share prices based on price displacement.

    If price has moved significantly from window open, the market
    would price the leading direction higher.

    Returns: (price_up, price_down) — estimated share prices [0, 1]
    """
    if window_open == 0:
        return 0.50, 0.50

    displacement = (price_at_entry - window_open) / window_open

    # Map displacement to up probability (logistic-like)
    # Based on: larger displacement → market more confident in direction
    # Calibrated roughly from observed Polymarket data
    if displacement > 0.003:      # strongly up
        p_up = min(0.90, 0.50 + displacement * 100)
    elif displacement > 0.001:    # mildly up
        p_up = 0.55 + displacement * 50
    elif displacement < -0.003:   # strongly down
        p_up = max(0.10, 0.50 + displacement * 100)
    elif displacement < -0.001:   # mildly down
        p_up = 0.45 + displacement * 50
    else:                         # flat
        p_up = 0.50

    p_up = max(0.05, min(0.95, p_up))
    p_down = 1.0 - p_up

    return p_up, p_down


# ═══════════════════════════════════════════════════════════
#  Step 4: Strategy functions
# ═══════════════════════════════════════════════════════════

def strategy_random(ta, session, vol_regime, window, **kw) -> str | None:
    return "up" if np.random.random() > 0.5 else "down"

def strategy_always_up(ta, session, vol_regime, window, **kw) -> str | None:
    return "up"

def strategy_momentum(ta, session, vol_regime, window, **kw) -> str | None:
    """If price moved up in first 10 min → predict up."""
    if window["price_at_sweet_spot"] > window["window_open"]:
        return "up"
    elif window["price_at_sweet_spot"] < window["window_open"]:
        return "down"
    return None

def strategy_mean_reversion(ta, session, vol_regime, window, **kw) -> str | None:
    """If price moved up in first 10 min → predict down (revert)."""
    if window["price_at_sweet_spot"] > window["window_open"]:
        return "down"
    elif window["price_at_sweet_spot"] < window["window_open"]:
        return "up"
    return None

def strategy_contrarian_cheap(ta, session, vol_regime, window, **kw) -> str | None:
    """Always buy the cheaper side (no TA). This is what the friend likely does."""
    p_up, p_down = kw.get("odds", (0.5, 0.5))
    if p_up < p_down:
        return "up"
    elif p_down < p_up:
        return "down"
    return None

def strategy_krajekis(ta, session, vol_regime, window, **kw) -> str | None:
    """Full Krajekis 5-layer strategy."""
    if ta is None:
        return None
    return determine_direction(ta, session)


# ═══════════════════════════════════════════════════════════
#  Step 5: Run backtest
# ═══════════════════════════════════════════════════════════

def run_backtest(
    windows: list[dict],
    df_5m: pd.DataFrame,
    strategy_fn,
    strategy_name: str,
    vol_low: float = 80,
    vol_high: float = 200,
    bet_size: float = 1.0,
) -> dict:
    """Run a single strategy across all windows. Returns results dict."""
    results = []

    for w in windows:
        # Compute TA at sweet spot
        ta = compute_ta_at_sweet_spot(df_5m, w["sweet_spot_idx"])

        # Session + volatility
        session = determine_session(w["window_start"].to_pydatetime())
        atr_val = ta.atr_14 if ta else 0
        vol_regime = classify_volatility(atr_val, vol_low, vol_high)

        # Estimate odds
        p_up, p_down = estimate_odds(w["price_at_sweet_spot"], w["window_open"])

        # Get prediction
        prediction = strategy_fn(
            ta=ta, session=session, vol_regime=vol_regime,
            window=w, odds=(p_up, p_down),
        )

        if prediction is None:
            results.append({"signal": False})
            continue

        # Check result
        actual_up = w["window_up"]
        predicted_up = prediction == "up"
        correct = predicted_up == actual_up

        # PnL with odds
        if prediction == "up":
            share_price = p_up
        else:
            share_price = p_down

        if correct:
            pnl = bet_size * (1.0 / share_price - 1) if share_price > 0 else 0
        else:
            pnl = -bet_size

        results.append({
            "signal": True,
            "correct": correct,
            "pnl": pnl,
            "prediction": prediction,
            "actual_up": actual_up,
            "share_price": share_price,
            "odds": 1.0 / share_price if share_price > 0 else 0,
            "session": session,
            "vol_regime": vol_regime,
            "month": w["window_start"].strftime("%Y-%m"),
        })

    # Summarize
    trades = [r for r in results if r.get("signal")]
    if not trades:
        return {"name": strategy_name, "trades": 0, "wr": 0, "pnl": 0, "signal_rate": 0}

    wins = sum(1 for t in trades if t.get("correct"))
    total_pnl = sum(t.get("pnl", 0) for t in trades)
    wr = wins / len(trades) * 100

    return {
        "name": strategy_name,
        "trades": len(trades),
        "wins": wins,
        "wr": round(wr, 2),
        "total_pnl": round(total_pnl, 2),
        "avg_pnl": round(total_pnl / len(trades), 4),
        "signal_rate": round(len(trades) / len(windows) * 100, 1),
        "total_windows": len(windows),
        "results": trades,  # for detailed analysis
    }


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("POLYMARKET 15m BACKTEST — Corrected Methodology")
    print("=" * 70)

    symbols = {"BTCUSDT": ("BTC", 80, 200), "ETHUSDT": ("ETH", 3, 8)}

    strategies = [
        ("Random", strategy_random),
        ("Always Up", strategy_always_up),
        ("Momentum", strategy_momentum),
        ("Mean Reversion", strategy_mean_reversion),
        ("Contrarian Cheap", strategy_contrarian_cheap),
        ("Krajekis TA", strategy_krajekis),
    ]

    for symbol, (coin, vol_low, vol_high) in symbols.items():
        path = DATA_DIR / "binance" / "futures" / "5m" / f"{symbol}.parquet"
        if not path.exists():
            print(f"  {symbol}: no 5m data")
            continue

        df_5m = pd.read_parquet(path)
        df_5m = df_5m[df_5m.index >= "2025-01-01"]  # match friend's period

        print(f"\n{'═' * 60}")
        print(f"  {coin} — {len(df_5m)} 5m bars ({df_5m.index[0].date()} → {df_5m.index[-1].date()})")
        print(f"{'═' * 60}")

        # Build windows
        windows = build_windows(df_5m)
        print(f"  {len(windows)} 15m windows")

        # Run all strategies
        print(f"\n  {'Strategy':<20s} {'Trades':>7s} {'WR%':>7s} {'PnL':>10s} {'Avg PnL':>9s} {'Signal%':>8s}")
        print(f"  {'-'*65}")

        for name, fn in strategies:
            np.random.seed(42)  # reproducible random
            result = run_backtest(windows, df_5m, fn, name, vol_low, vol_high)
            print(
                f"  {result['name']:<20s} "
                f"{result['trades']:>7d} "
                f"{result['wr']:>6.1f}% "
                f"{result.get('total_pnl', 0):>+10.1f} "
                f"{result.get('avg_pnl', 0):>+8.4f} "
                f"{result.get('signal_rate', 0):>7.1f}%"
            )

            # Monthly breakdown for Krajekis
            if name == "Krajekis TA" and result.get("results"):
                trades = result["results"]
                months = sorted(set(t["month"] for t in trades))

                if len(months) > 1:
                    print(f"\n  Monthly breakdown ({name}):")
                    for m in months:
                        m_trades = [t for t in trades if t["month"] == m]
                        m_wins = sum(1 for t in m_trades if t.get("correct"))
                        m_pnl = sum(t.get("pnl", 0) for t in m_trades)
                        m_wr = m_wins / len(m_trades) * 100 if m_trades else 0
                        print(f"    {m}: {len(m_trades):>4d} trades, WR={m_wr:>5.1f}%, PnL={m_pnl:>+8.1f}")

                # Session breakdown
                print(f"\n  Session breakdown ({name}):")
                sessions = sorted(set(t["session"] for t in trades))
                for s in sessions:
                    s_trades = [t for t in trades if t["session"] == s]
                    s_wins = sum(1 for t in s_trades if t.get("correct"))
                    s_pnl = sum(t.get("pnl", 0) for t in s_trades)
                    s_wr = s_wins / len(s_trades) * 100 if s_trades else 0
                    print(f"    {s:15s}: {len(s_trades):>4d} trades, WR={s_wr:>5.1f}%, PnL={s_pnl:>+8.1f}")

    print(f"\n{'═' * 70}")
    print("KEY QUESTION: Is Contrarian Cheap > 50% WR?")
    print("If yes → edge is in Polymarket pricing, not TA")
    print("If no → need a different approach entirely")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()
