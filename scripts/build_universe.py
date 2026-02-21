#!/usr/bin/env python3
"""
Universe Builder for Expanded R3 Track A Research
===================================================

Selects symbols from Binance USDT perpetual futures based on:
1. Minimum history length (>= 2 years)
2. Liquidity (top N by available data length, proxy for volume)
3. Data quality (gap rate < 1%)
4. Funding rate / OI coverage

Usage:
    cd /path/to/quant-binance-spot
    PYTHONPATH=src python scripts/build_universe.py --market futures --quote USDT --min-years 2 --top-n 20

    # Download missing data first
    PYTHONPATH=src python scripts/build_universe.py --download --top-n 20

    # Just report on existing data
    PYTHONPATH=src python scripts/build_universe.py --report-only
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qtrade.data.storage import load_klines

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# ══════════════════════════════════════════════════════════════
#  Candidate Pool — Top Binance USDT Perpetual Futures
#  (pre-defined, sorted roughly by 30D volume rank)
# ══════════════════════════════════════════════════════════════

CANDIDATE_SYMBOLS = [
    # Tier 1: mega-cap, highest liquidity
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
    # Tier 2: large-cap, high liquidity
    "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT", "MATICUSDT", "LTCUSDT",
    # Tier 3: mid-cap, medium-high liquidity
    "NEARUSDT", "APTUSDT", "ARBUSDT", "OPUSDT", "FILUSDT", "ATOMUSDT",
    "UNIUSDT", "AAVEUSDT",
    # Tier 4: additional liquid pairs
    "ETCUSDT", "XLMUSDT", "ALGOUSDT", "ICPUSDT", "TRXUSDT",
    "FTMUSDT", "SANDUSDT", "MANAUSDT", "GALAUSDT", "VETUSDT",
]

# Symbols known to be delisted, rebranded, or problematic
EXCLUDED_SYMBOLS = {
    "LUNAUSDT",   # collapsed
    "FTTUSDT",    # collapsed
    "SRMUSDT",    # delisted
}


def check_symbol_data(
    symbol: str,
    data_dir: Path,
    market_type: str = "futures",
    interval: str = "1h",
) -> dict:
    """Check local data availability and quality for a symbol."""
    result = {
        "symbol": symbol,
        "has_1h": False,
        "has_5m": False,
        "has_15m": False,
        "has_funding_rate": False,
        "has_oi": False,
        "start_date": None,
        "end_date": None,
        "total_bars": 0,
        "total_years": 0.0,
        "gap_rate_pct": 0.0,
        "status": "NO_DATA",
        "rejection_reason": None,
    }

    # Check 1h data
    path_1h = data_dir / "binance" / market_type / interval / f"{symbol}.parquet"
    if path_1h.exists():
        try:
            df = load_klines(path_1h)
            if len(df) > 0:
                result["has_1h"] = True
                result["start_date"] = str(df.index[0])
                result["end_date"] = str(df.index[-1])
                result["total_bars"] = len(df)
                result["total_years"] = round(len(df) / (365.25 * 24), 2)

                # Gap rate: expected bars vs actual
                time_span = (df.index[-1] - df.index[0]).total_seconds() / 3600
                expected_bars = int(time_span) + 1
                if expected_bars > 0:
                    result["gap_rate_pct"] = round(
                        (1 - len(df) / expected_bars) * 100, 2
                    )
        except Exception:
            pass

    # Check 5m data
    path_5m = data_dir / "binance" / market_type / "5m" / f"{symbol}.parquet"
    result["has_5m"] = path_5m.exists()

    # Check 15m data
    path_15m = data_dir / "binance" / market_type / "15m" / f"{symbol}.parquet"
    result["has_15m"] = path_15m.exists()

    # Check funding rate
    fr_path = data_dir / "binance" / market_type / "funding_rate" / f"{symbol}.parquet"
    result["has_funding_rate"] = fr_path.exists()

    # Check OI (various locations)
    for oi_dir in [
        data_dir / "binance" / market_type / "open_interest",
        data_dir / "binance" / market_type / "open_interest" / "merged",
        data_dir / "binance" / market_type / "open_interest" / "binance",
    ]:
        oi_path = oi_dir / f"{symbol}.parquet"
        if oi_path.exists():
            result["has_oi"] = True
            break

    return result


def select_universe(
    candidates: list[dict],
    min_years: float = 2.0,
    max_gap_rate: float = 1.0,
    top_n: int = 20,
) -> tuple[list[dict], list[dict]]:
    """
    Select symbols for the expanded universe.

    Returns:
        (selected, rejected) — both are lists of dicts
    """
    selected = []
    rejected = []

    for c in candidates:
        if c["symbol"] in EXCLUDED_SYMBOLS:
            c["status"] = "EXCLUDED"
            c["rejection_reason"] = "Known problematic (delisted/collapsed)"
            rejected.append(c)
            continue

        if not c["has_1h"]:
            c["status"] = "NO_DATA"
            c["rejection_reason"] = "No 1h data available locally"
            rejected.append(c)
            continue

        if c["total_years"] < min_years:
            c["status"] = "INSUFFICIENT_HISTORY"
            c["rejection_reason"] = f"Only {c['total_years']:.1f} years (need >= {min_years})"
            rejected.append(c)
            continue

        if c["gap_rate_pct"] > max_gap_rate:
            c["status"] = "DATA_QUALITY"
            c["rejection_reason"] = f"Gap rate {c['gap_rate_pct']:.1f}% > {max_gap_rate}%"
            rejected.append(c)
            continue

        c["status"] = "SELECTED"
        selected.append(c)

    # Sort selected by history length (longer = better), then take top_n
    selected.sort(key=lambda x: x["total_years"], reverse=True)
    if len(selected) > top_n:
        overflow = selected[top_n:]
        for o in overflow:
            o["status"] = "OVERFLOW"
            o["rejection_reason"] = f"Exceeded top-{top_n} limit"
            rejected.append(o)
        selected = selected[:top_n]

    return selected, rejected


def compute_vol_parity_weights(
    symbols: list[str],
    data_dir: Path,
    market_type: str = "futures",
    lookback: int = 720,
    min_weight: float = 0.03,
    max_weight: float = 0.20,
) -> dict[str, float]:
    """
    Compute volatility-parity weights for the expanded universe.

    Weight_i = (1/vol_i) / sum(1/vol_j for all j)
    Then clip to [min_weight, max_weight] and renormalize.
    """
    vols = {}
    for sym in symbols:
        path = data_dir / "binance" / market_type / "1h" / f"{sym}.parquet"
        if not path.exists():
            continue
        df = load_klines(path)
        if len(df) < lookback:
            vols[sym] = df["close"].pct_change().std() * np.sqrt(8760)
        else:
            vols[sym] = (
                df["close"].pct_change().iloc[-lookback:].std() * np.sqrt(8760)
            )

    if not vols:
        return {sym: 1.0 / len(symbols) for sym in symbols}

    # Inverse volatility
    inv_vols = {sym: 1.0 / max(v, 0.01) for sym, v in vols.items()}
    total_inv = sum(inv_vols.values())
    raw_weights = {sym: iv / total_inv for sym, iv in inv_vols.items()}

    # Clip and renormalize
    clipped = {sym: max(min_weight, min(max_weight, w)) for sym, w in raw_weights.items()}
    total_clipped = sum(clipped.values())
    final = {sym: round(w / total_clipped, 4) for sym, w in clipped.items()}

    return final


def main():
    parser = argparse.ArgumentParser(description="Build Universe for R3 Track A")
    parser.add_argument("--market", type=str, default="futures")
    parser.add_argument("--quote", type=str, default="USDT")
    parser.add_argument("--min-years", type=float, default=2.0)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--max-gap-rate", type=float, default=1.0)
    parser.add_argument("--report-only", action="store_true",
                        help="Only report existing data, don't select")
    parser.add_argument("--download", action="store_true",
                        help="Download missing 1h/5m/15m/funding data")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "reports" / "universe_selection" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print("█" * 80)
    print("  Universe Builder — R3 Track A Expanded Universe")
    print("█" * 80)
    print(f"  Timestamp:   {timestamp}")
    print(f"  Market:      {args.market}")
    print(f"  Min Years:   {args.min_years}")
    print(f"  Top N:       {args.top_n}")
    print(f"  Max Gap:     {args.max_gap_rate}%")
    print(f"  Candidates:  {len(CANDIDATE_SYMBOLS)}")
    print()

    # ── Step 1: Check all candidates ──
    print("━" * 80)
    print("  Step 1: Data Availability Check")
    print("━" * 80)

    candidates = []
    for sym in CANDIDATE_SYMBOLS:
        info = check_symbol_data(sym, DATA_DIR, args.market)
        candidates.append(info)
        status = "✅" if info["has_1h"] else "❌"
        years_str = f"{info['total_years']:.1f}y" if info["has_1h"] else "N/A"
        extras = []
        if info["has_5m"]:
            extras.append("5m")
        if info["has_15m"]:
            extras.append("15m")
        if info["has_funding_rate"]:
            extras.append("FR")
        if info["has_oi"]:
            extras.append("OI")
        extra_str = f" [{','.join(extras)}]" if extras else ""
        print(f"    {status} {sym:<15} {years_str:>6} "
              f"gap={info['gap_rate_pct']:.1f}%{extra_str}")

    if args.report_only:
        print("\n  Report-only mode. Exiting.")
        return

    # ── Step 2: Select universe ──
    print(f"\n{'━' * 80}")
    print("  Step 2: Universe Selection")
    print("━" * 80)

    selected, rejected = select_universe(
        candidates, min_years=args.min_years,
        max_gap_rate=args.max_gap_rate, top_n=args.top_n,
    )

    print(f"\n  ✅ Selected: {len(selected)} symbols")
    for s in selected:
        print(f"    {s['symbol']:<15} {s['total_years']:.1f}y "
              f"({s['start_date'][:10]} → {s['end_date'][:10]})")

    print(f"\n  ❌ Rejected: {len(rejected)} symbols")
    for r in rejected:
        print(f"    {r['symbol']:<15} {r['status']}: {r['rejection_reason']}")

    # ── Step 3: Compute weights ──
    print(f"\n{'━' * 80}")
    print("  Step 3: Vol-Parity Weights")
    print("━" * 80)

    selected_symbols = [s["symbol"] for s in selected]
    weights = compute_vol_parity_weights(
        selected_symbols, DATA_DIR, args.market,
        min_weight=0.03, max_weight=0.20,
    )

    print(f"\n  {'Symbol':<15} {'Weight':>8} {'Annualized Vol':>16}")
    print("  " + "-" * 42)
    for sym, w in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"  {sym:<15} {w:>8.1%}")
    print(f"  {'TOTAL':<15} {sum(weights.values()):>8.1%}")

    # ── Step 4: Identify missing data needs ──
    print(f"\n{'━' * 80}")
    print("  Step 4: Missing Data Summary")
    print("━" * 80)

    needs_5m = [s["symbol"] for s in selected if not s["has_5m"]]
    needs_15m = [s["symbol"] for s in selected if not s["has_15m"]]
    needs_fr = [s["symbol"] for s in selected if not s["has_funding_rate"]]

    if needs_5m:
        print(f"\n  Need 5m data:  {', '.join(needs_5m)}")
    if needs_15m:
        print(f"  Need 15m data: {', '.join(needs_15m)}")
    if needs_fr:
        print(f"  Need funding:  {', '.join(needs_fr)}")
    if not needs_5m and not needs_15m and not needs_fr:
        print("\n  ✅ All data complete!")

    # ── Step 5: Download if requested ──
    if args.download and (needs_5m or needs_15m or needs_fr):
        print(f"\n{'━' * 80}")
        print("  Step 5: Downloading Missing Data")
        print("━" * 80)

        # Create a minimal config for download
        from qtrade.data.klines import fetch_klines
        from qtrade.data.storage import save_klines as _save, merge_klines

        for interval_str, symbols_need in [("5m", needs_5m), ("15m", needs_15m)]:
            for sym in symbols_need:
                print(f"\n  📥 Downloading {sym} {interval_str}...")
                out_path = DATA_DIR / "binance" / args.market / interval_str / f"{sym}.parquet"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    df = fetch_klines(sym, interval_str, "2022-01-01", None,
                                     market_type=args.market)
                    if not df.empty:
                        _save(df, out_path)
                        print(f"    ✅ {len(df)} bars → {out_path}")
                    else:
                        print(f"    ⚠️  No data returned")
                except Exception as e:
                    print(f"    ❌ Failed: {e}")

        # Download funding rates
        if needs_fr:
            try:
                from qtrade.data.funding_rate import (
                    download_funding_rates,
                    save_funding_rates,
                    get_funding_rate_path,
                )
                for sym in needs_fr:
                    print(f"\n  📥 Downloading {sym} funding rate...")
                    fr_path = get_funding_rate_path(DATA_DIR, sym)
                    try:
                        fr_df = download_funding_rates(sym, "2022-01-01", None)
                        if not fr_df.empty:
                            save_funding_rates(fr_df, fr_path)
                            print(f"    ✅ {len(fr_df)} entries → {fr_path}")
                        else:
                            print(f"    ⚠️  No funding rate data")
                    except Exception as e:
                        print(f"    ❌ Failed: {e}")
            except ImportError:
                print("  ⚠️  Funding rate module not available")

    # ── Save outputs ──
    # CSV
    csv_path = output_dir / "universe_selected.csv"
    all_entries = selected + rejected
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "symbol", "status", "has_1h", "has_5m", "has_15m",
                "has_funding_rate", "has_oi", "start_date", "end_date",
                "total_bars", "total_years", "gap_rate_pct",
                "rejection_reason",
            ],
        )
        writer.writeheader()
        for entry in all_entries:
            row = {k: entry.get(k) for k in writer.fieldnames}
            writer.writerow(row)

    # JSON
    json_path = output_dir / "universe_selection.json"
    save_data = {
        "timestamp": timestamp,
        "params": {
            "market": args.market,
            "min_years": args.min_years,
            "top_n": args.top_n,
            "max_gap_rate": args.max_gap_rate,
        },
        "selected_symbols": selected_symbols,
        "weights": weights,
        "selected": [{k: v for k, v in s.items()} for s in selected],
        "rejected": [{k: v for k, v in r.items()} for r in rejected],
    }
    with open(json_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    print(f"\n{'━' * 80}")
    print(f"  Output: {csv_path}")
    print(f"  Output: {json_path}")
    print(f"  Selected: {len(selected)} symbols")
    print(f"  Weights: {json.dumps(weights, indent=2)}")
    print("━" * 80)


if __name__ == "__main__":
    main()
