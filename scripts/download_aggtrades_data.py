"""
aggTrades 數據下載 & 聚合工具

從 Binance Vision 下載 Futures USDT-M aggTrades（逐筆成交），
逐月處理並聚合為 VPIN + hourly metrics + Real CVD + OFI。

⚠️ 注意：aggTrades 原始檔案非常大（BTC 單月 100-500 MB zip）。
    此腳本逐月下載後立即聚合，不保留原始 tick data。
    8 symbols × 72 months ≈ 預計下載 50-100 GB（聚合後 ~100 MB）。

儲存路徑：
    data/binance/futures/aggtrades_agg/{SYMBOL}_hourly.parquet
    data/binance/futures/aggtrades_agg/{SYMBOL}_vpin_1h.parquet
    data/binance/futures/aggtrades_agg/{SYMBOL}_vpin_daily.parquet
    data/binance/futures/aggtrades_agg/{SYMBOL}_cvd.parquet
    data/binance/futures/aggtrades_agg/{SYMBOL}_ofi.parquet

使用範例：
    # 下載全部 8 symbols (2020-01 → 2026-02)
    PYTHONPATH=src python scripts/download_aggtrades_data.py

    # 只下載指定 symbols
    PYTHONPATH=src python scripts/download_aggtrades_data.py --symbols BTCUSDT ETHUSDT

    # 自訂日期範圍
    PYTHONPATH=src python scripts/download_aggtrades_data.py --start 2023-01-01 --end 2025-12-31

    # 只檢查已下載數據的覆蓋率
    PYTHONPATH=src python scripts/download_aggtrades_data.py --coverage

    # 自訂 VPIN 參數
    PYTHONPATH=src python scripts/download_aggtrades_data.py --buckets-per-day 50 --vpin-window 50
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# 8 production symbols
DEFAULT_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "XRPUSDT",
    "LINKUSDT",
]

DEFAULT_START = "2020-01-01"
DEFAULT_END = "2026-02-28"
DEFAULT_DATA_DIR = Path("data/binance/futures/aggtrades_agg")


def coverage_report(symbols: list[str], data_dir: Path) -> None:
    """印出已下載數據的覆蓋率報告"""
    print("\n📊 aggTrades Aggregated Data Coverage Report")
    print("=" * 90)

    metrics = ["hourly", "vpin_1h", "vpin_daily", "cvd", "ofi"]

    for symbol in symbols:
        print(f"\n  {symbol}:")
        for metric in metrics:
            path = data_dir / f"{symbol}_{metric}.parquet"
            if not path.exists():
                print(f"    {metric:<15} ❌ 無數據")
                continue
            try:
                df = pd.read_parquet(path)
                n_rows = len(df)
                start = df.index.min()
                end = df.index.max()
                days = (end - start).days if n_rows > 0 else 0
                print(
                    f"    {metric:<15} ✅ {n_rows:>8,} rows  "
                    f"{start.strftime('%Y-%m-%d')} → {end.strftime('%Y-%m-%d')}  "
                    f"({days}d)"
                )
            except Exception as e:
                print(f"    {metric:<15} ⚠️  讀取失敗: {e}")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download aggTrades from Binance Vision and compute VPIN/CVD/OFI metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=DEFAULT_SYMBOLS,
        help=f"Symbols to download (default: {', '.join(DEFAULT_SYMBOLS)})",
    )
    parser.add_argument(
        "--start",
        default=DEFAULT_START,
        help=f"Start date YYYY-MM-DD (default: {DEFAULT_START})",
    )
    parser.add_argument(
        "--end",
        default=DEFAULT_END,
        help=f"End date YYYY-MM-DD (default: {DEFAULT_END})",
    )
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help=f"Output directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--buckets-per-day",
        type=int,
        default=50,
        help="Target volume-clock buckets per day for VPIN (default: 50)",
    )
    parser.add_argument(
        "--vpin-window",
        type=int,
        default=50,
        help="VPIN rolling window in buckets (default: 50)",
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Print coverage report for existing data and exit",
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    # ── 覆蓋率報告模式 ──
    if args.coverage:
        coverage_report(args.symbols, data_dir)
        return

    # ── 下載模式 ──
    from qtrade.data.agg_trades import download_and_process_aggtrades

    total = len(args.symbols)
    success = 0
    failed = []

    for i, symbol in enumerate(args.symbols):
        print(f"\n{'='*70}")
        print(f"  [{i + 1}/{total}] 📥 {symbol}")
        print(f"{'='*70}")

        try:
            result = download_and_process_aggtrades(
                symbol=symbol,
                start=args.start,
                end=args.end,
                data_dir=data_dir,
                buckets_per_day=args.buckets_per_day,
                n_vpin_buckets=args.vpin_window,
            )

            if result:
                hourly = result.get("hourly")
                vpin_1h = result.get("vpin_1h")
                vpin_daily = result.get("vpin_daily")
                real_cvd = result.get("real_cvd")

                print(f"\n  📊 {symbol} Summary:")
                if hourly is not None and not hourly.empty:
                    print(
                        f"    Hourly metrics: {len(hourly):,} rows "
                        f"({hourly.index.min().strftime('%Y-%m-%d')} → "
                        f"{hourly.index.max().strftime('%Y-%m-%d')})"
                    )
                if vpin_1h is not None and not vpin_1h.empty:
                    print(
                        f"    VPIN (1h):      {len(vpin_1h):,} rows, "
                        f"mean={vpin_1h.mean():.4f}, "
                        f"std={vpin_1h.std():.4f}"
                    )
                if vpin_daily is not None and not vpin_daily.empty:
                    print(
                        f"    VPIN (daily):   {len(vpin_daily):,} rows, "
                        f"mean={vpin_daily.mean():.4f}"
                    )
                if real_cvd is not None and not real_cvd.empty:
                    print(f"    Real CVD:       {len(real_cvd):,} rows")

                success += 1
            else:
                logger.warning(f"⚠️  {symbol}: 無數據")
                failed.append(symbol)

        except Exception as e:
            logger.error(f"❌ {symbol}: {e}")
            failed.append(symbol)

    # ── 總結 ──
    print(f"\n{'='*70}")
    print(f"  📋 Download Complete")
    print(f"{'='*70}")
    print(f"  ✅ Success: {success}/{total}")
    if failed:
        print(f"  ❌ Failed:  {', '.join(failed)}")
    print(f"  📁 Data dir: {data_dir}")
    print()

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
