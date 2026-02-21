#!/usr/bin/env python3
"""
Download OI (Open Interest) historical data from multiple providers.

Usage:
    # Best option: data.binance.vision (free, full history 2021-12 → now)
    PYTHONPATH=src python scripts/download_oi_data.py

    # Specific provider
    PYTHONPATH=src python scripts/download_oi_data.py --provider binance_vision

    # Coverage report only (no download)
    PYTHONPATH=src python scripts/download_oi_data.py --coverage-only

Providers:
    binance_vision — data.binance.vision public repo (2021-12-01 → now, 5m→1h, FREE)
    binance        — Binance API (last ~20 days only)
    coinglass      — Coinglass API (needs COINGLASS_API_KEY)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import pandas as pd

from qtrade.data.open_interest import (
    download_open_interest,
    save_open_interest,
    load_open_interest,
    get_oi_path,
    merge_oi_sources,
    compute_oi_coverage,
    print_oi_coverage_report,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
DEFAULT_START = "2022-01-01"
DEFAULT_INTERVAL = "1h"


def download_for_provider(
    provider_name: str,
    symbols: list[str],
    data_dir: Path,
    start: str,
    end: str | None,
    interval: str,
) -> dict[str, pd.DataFrame]:
    """Download OI from a single provider and save."""
    results = {}
    for symbol in symbols:
        print(f"\n{'─'*60}")
        print(f"  {symbol} via {provider_name}")
        print(f"{'─'*60}")

        try:
            df = download_open_interest(
                symbol=symbol,
                start=start,
                end=end,
                interval=interval,
                provider=provider_name,
            )
        except Exception as e:
            logger.error(f"❌ {symbol} download failed: {e}")
            df = pd.DataFrame(columns=["sumOpenInterest", "sumOpenInterestValue"])

        if not df.empty:
            path = get_oi_path(data_dir, symbol, provider_name)
            save_open_interest(df, path)
            results[symbol] = df
        else:
            logger.warning(f"⚠️  {symbol}: no data from {provider_name}")
            results[symbol] = df

    return results


def merge_all_sources(
    symbols: list[str],
    data_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Merge OI from all available providers."""
    merged = {}
    for symbol in symbols:
        sources = []
        for provider_name in ["binance_vision", "coinglass", "binance"]:
            path = get_oi_path(data_dir, symbol, provider_name)
            df = load_open_interest(path)
            if df is not None and not df.empty:
                sources.append(df)
                logger.info(
                    f"  {symbol}/{provider_name}: {len(df)} rows "
                    f"({df.index[0]:%Y-%m-%d} → {df.index[-1]:%Y-%m-%d})"
                )

        if sources:
            combined = merge_oi_sources(sources, max_ffill_bars=2)
            save_path = get_oi_path(data_dir, symbol, "merged")
            save_open_interest(combined, save_path)
            merged[symbol] = combined
            logger.info(
                f"✅ {symbol} merged: {len(combined)} rows "
                f"({combined.index[0]:%Y-%m-%d} → {combined.index[-1]:%Y-%m-%d})"
            )
        else:
            merged[symbol] = pd.DataFrame(
                columns=["sumOpenInterest", "sumOpenInterestValue"]
            )
            logger.warning(f"⚠️  {symbol}: no OI data from any provider")

    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Download OI historical data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--provider", default="auto",
        choices=["auto", "binance", "binance_vision", "coinglass", "all"],
        help="Data provider (default: auto = binance_vision)",
    )
    parser.add_argument("--start", default=DEFAULT_START, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: now)")
    parser.add_argument("--interval", default=DEFAULT_INTERVAL, help="Interval (1h/4h/1d)")
    parser.add_argument(
        "--symbols", nargs="+", default=SYMBOLS,
        help="Symbols to download",
    )
    parser.add_argument(
        "--data-dir", default="data",
        help="Data directory (default: data)",
    )
    parser.add_argument(
        "--coverage-only", action="store_true",
        help="Only print coverage report (no download)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    print("=" * 70)
    print("OI Data Download Pipeline")
    print("=" * 70)

    print(f"  Provider: {args.provider}")
    print(f"  Symbols: {args.symbols}")
    print(f"  Range: {args.start} → {args.end or 'now'}")
    print(f"  Interval: {args.interval}")
    print(f"  Data dir: {data_dir}")

    if args.coverage_only:
        print("\n📊 Coverage report (no download):")
        coverage_df = compute_oi_coverage(
            symbols=args.symbols,
            data_dir=data_dir,
            backtest_start=args.start,
            interval=args.interval,
        )
        print_oi_coverage_report(coverage_df)
        return

    # Determine which providers to use
    providers_to_run = []
    if args.provider == "auto":
        # binance_vision is free and has full history — always use it
        providers_to_run = ["binance_vision"]
    elif args.provider == "all":
        providers_to_run = ["binance_vision", "binance"]
        if os.getenv("COINGLASS_API_KEY"):
            providers_to_run.append("coinglass")
    else:
        providers_to_run = [args.provider]

    # Download from each provider
    for prov in providers_to_run:
        print(f"\n{'═'*70}")
        print(f"  Downloading from: {prov}")
        print(f"{'═'*70}")
        download_for_provider(
            prov, args.symbols, data_dir,
            args.start, args.end, args.interval,
        )

    # Merge all sources
    print(f"\n{'═'*70}")
    print(f"  Merging all sources")
    print(f"{'═'*70}")
    merge_all_sources(args.symbols, data_dir)

    # Coverage report
    coverage_df = compute_oi_coverage(
        symbols=args.symbols,
        data_dir=data_dir,
        backtest_start=args.start,
        interval=args.interval,
    )
    print_oi_coverage_report(coverage_df)

    # Summary
    overall_actual = coverage_df["actual_bars"].sum()
    overall_expected = coverage_df["expected_bars"].sum()
    overall_pct = (overall_actual / overall_expected * 100) if overall_expected > 0 else 0

    if overall_pct >= 70:
        print("\n✅ OI data ready for ablation study.")
        print("   Run: PYTHONPATH=src python scripts/run_r2_overlay_ablation.py --phase C")
    else:
        print(f"\n❌ OI coverage {overall_pct:.1f}% < 70% — insufficient for ablation.")


if __name__ == "__main__":
    main()
