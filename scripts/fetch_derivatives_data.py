"""
衍生品數據下載工具 (Phase 0A)

從 Binance Futures API 下載以下衍生品數據：
    1. Long/Short Ratio (Account-level)
    2. Top Trader Long/Short Ratio (Account + Position)
    3. Taker Buy/Sell Volume Ratio
    4. CVD (Cumulative Volume Delta) — 從 Taker Volume 衍生

數據來源：
    A. Binance Vision (data.binance.vision) — 完整歷史 (2021-12 至今, 5m)
       每日 metrics CSV 已包含 LSR, Taker Vol Ratio 等
    B. Binance Futures API — 即時但僅 ~30 天歷史 (500 records)

儲存路徑：
    data/binance/futures/derivatives/{metric}/{SYMBOL}.parquet

使用範例：
    # 從 Binance Vision 下載全部衍生品數據（推薦，完整歷史）
    PYTHONPATH=src python scripts/fetch_derivatives_data.py --symbols BTCUSDT ETHUSDT

    # 從 Binance API 下載最近 30 天（即時，用於補齊最新數據）
    PYTHONPATH=src python scripts/fetch_derivatives_data.py --symbols BTCUSDT --source api

    # 只下載特定指標
    PYTHONPATH=src python scripts/fetch_derivatives_data.py --symbols BTCUSDT --metrics lsr taker_vol

    # 查看已下載數據的覆蓋率報告
    PYTHONPATH=src python scripts/fetch_derivatives_data.py --symbols BTCUSDT --coverage
"""
from __future__ import annotations

import argparse
import io
import logging
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════
#  常數
# ══════════════════════════════════════════════════════════════

DATA_DIR = Path("data/binance/futures/derivatives")

# Binance Vision metrics CSV 欄位映射
# CSV columns: create_time, symbol, sum_open_interest, sum_open_interest_value,
#   count_toptrader_long_short_ratio, sum_toptrader_long_short_ratio,
#   count_long_short_ratio, sum_taker_long_short_vol_ratio
VISION_METRIC_MAP = {
    "lsr": {
        "raw_col": "count_long_short_ratio",
        "description": "Long/Short Account Ratio (全帳戶)",
    },
    "top_lsr_account": {
        "raw_col": "count_toptrader_long_short_ratio",
        "description": "Top Trader Long/Short Ratio (帳戶數)",
    },
    "top_lsr_position": {
        "raw_col": "sum_toptrader_long_short_ratio",
        "description": "Top Trader Long/Short Ratio (持倉量)",
    },
    "taker_vol_ratio": {
        "raw_col": "sum_taker_long_short_vol_ratio",
        "description": "Taker Buy/Sell Volume Ratio",
    },
}

# Binance API endpoint 映射
API_ENDPOINTS = {
    "lsr": "/futures/data/globalLongShortAccountRatio",
    "top_lsr_account": "/futures/data/topLongShortAccountRatio",
    "top_lsr_position": "/futures/data/topLongShortPositionRatio",
    "taker_vol_ratio": "/futures/data/takerlongshortRatio",
}

ALL_METRICS = list(VISION_METRIC_MAP.keys())

VISION_BASE_URL = "https://data.binance.vision/data/futures/um/daily/metrics"
VISION_EARLIEST_DATE = "2021-12-01"


# ══════════════════════════════════════════════════════════════
#  Binance Vision 下載（完整歷史）
# ══════════════════════════════════════════════════════════════

def fetch_vision_metrics(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1h",
    cache_dir: Path | None = None,
) -> dict[str, pd.Series]:
    """
    從 data.binance.vision 下載每日 metrics CSV，提取所有衍生品指標

    Returns:
        dict[metric_name, pd.Series] — 每個指標一個 Series
    """
    import requests

    start_date = pd.Timestamp(start or VISION_EARLIEST_DATE, tz="UTC").normalize()
    end_date = (
        pd.Timestamp(end, tz="UTC").normalize()
        if end
        else pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=2)
    )
    earliest = pd.Timestamp(VISION_EARLIEST_DATE, tz="UTC")
    if start_date < earliest:
        start_date = earliest

    if cache_dir is None:
        cache_dir = Path("data/binance/futures/open_interest/vision_cache") / symbol
    cache_dir.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    all_dfs: list[pd.DataFrame] = []
    n_cached = n_downloaded = n_failed = 0

    logger.info(
        f"📥 Vision metrics: {symbol} {start_date:%Y-%m-%d} → {end_date:%Y-%m-%d} "
        f"({len(dates)} days)"
    )

    for dt in dates:
        date_str = dt.strftime("%Y-%m-%d")
        csv_cache = cache_dir / f"{symbol}-metrics-{date_str}.csv"

        if csv_cache.exists():
            try:
                df_day = pd.read_csv(csv_cache)
                if not df_day.empty:
                    all_dfs.append(df_day)
                    n_cached += 1
                    continue
            except Exception:
                pass

        zip_url = f"{VISION_BASE_URL}/{symbol}/{symbol}-metrics-{date_str}.zip"
        try:
            resp = requests.get(zip_url, timeout=15)
            if resp.status_code == 404:
                continue
            resp.raise_for_status()

            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                csv_name = zf.namelist()[0]
                with zf.open(csv_name) as f:
                    df_day = pd.read_csv(f)

            df_day.to_csv(csv_cache, index=False)
            all_dfs.append(df_day)
            n_downloaded += 1

        except Exception as e:
            n_failed += 1
            if n_failed <= 3:
                logger.debug(f"  skip {date_str}: {e}")

        total = n_cached + n_downloaded + n_failed
        if total > 0 and total % 100 == 0:
            logger.info(
                f"  ... {total}/{len(dates)} days "
                f"(cached={n_cached}, dl={n_downloaded}, fail={n_failed})"
            )

    if not all_dfs:
        logger.warning(f"⚠️  No vision metrics data for {symbol}")
        return {}

    raw = pd.concat(all_dfs, ignore_index=True)
    raw["create_time"] = pd.to_datetime(raw["create_time"], utc=True)
    raw = raw.sort_values("create_time")
    raw = raw.set_index("create_time")
    raw = raw[~raw.index.duplicated(keep="last")]

    logger.info(
        f"✅ Vision raw: {symbol} {len(raw)} rows "
        f"(cached={n_cached}, dl={n_downloaded}, fail={n_failed})"
    )

    # 提取各指標並 resample
    resample_map = {
        "5m": "5min", "15m": "15min", "30m": "30min",
        "1h": "1h", "2h": "2h", "4h": "4h", "1d": "1D",
    }
    freq = resample_map.get(interval, "1h")

    results: dict[str, pd.Series] = {}
    for metric_name, info in VISION_METRIC_MAP.items():
        col = info["raw_col"]
        if col not in raw.columns:
            logger.warning(f"  {metric_name}: column '{col}' not found, skipping")
            continue

        series = pd.to_numeric(raw[col], errors="coerce")
        if freq != "5min":
            series = series.resample(freq).last().dropna()
        else:
            series = series.dropna()

        series.name = metric_name
        results[metric_name] = series

        if not series.empty:
            logger.info(
                f"  {metric_name}: {len(series)} bars @ {interval} "
                f"({series.index[0]:%Y-%m-%d} → {series.index[-1]:%Y-%m-%d})"
            )

    return results


# ══════════════════════════════════════════════════════════════
#  Binance API 下載（最近 30 天）
# ══════════════════════════════════════════════════════════════

def fetch_api_metric(
    symbol: str,
    metric: str,
    interval: str = "1h",
    limit: int = 500,
) -> pd.Series:
    """
    從 Binance Futures API 下載單一衍生品指標（最近 ~30 天）

    Returns:
        pd.Series indexed by UTC timestamp
    """
    from qtrade.data.binance_futures_client import BinanceFuturesHTTP

    endpoint = API_ENDPOINTS.get(metric)
    if endpoint is None:
        raise ValueError(f"Unknown metric: {metric}. Available: {list(API_ENDPOINTS.keys())}")

    # Binance API 使用 'period' 參數
    period_map = {
        "5m": "5m", "15m": "15m", "30m": "30m",
        "1h": "1h", "2h": "2h", "4h": "4h",
        "6h": "6h", "8h": "8h", "12h": "12h", "1d": "1d",
    }
    period = period_map.get(interval, "1h")

    client = BinanceFuturesHTTP()
    try:
        records = client.get(endpoint, {
            "symbol": symbol,
            "period": period,
            "limit": min(limit, 500),
        })
    except Exception as e:
        logger.error(f"❌ API fetch {metric} {symbol}: {e}")
        return pd.Series(dtype=float, name=metric)

    if not records:
        return pd.Series(dtype=float, name=metric)

    df = pd.DataFrame(records)

    # API 回傳格式：timestamp, symbol, longShortRatio / buySellRatio 等
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    # 根據不同 endpoint 取得值欄位
    value_col = None
    for col_candidate in ["longShortRatio", "buySellRatio", "longAccount", "longPosition"]:
        if col_candidate in df.columns:
            value_col = col_candidate
            break

    if value_col is None:
        # fallback: 取第一個數值欄
        num_cols = [c for c in df.columns if c not in ("timestamp", "symbol")]
        value_col = num_cols[0] if num_cols else None

    if value_col is None:
        logger.warning(f"  {metric} {symbol}: no value column found")
        return pd.Series(dtype=float, name=metric)

    series = pd.to_numeric(df.set_index("timestamp")[value_col], errors="coerce")
    series = series.sort_index()
    series = series[~series.index.duplicated(keep="last")]
    series.name = metric

    if not series.empty:
        logger.info(
            f"✅ API {metric} {symbol}: {len(series)} records "
            f"({series.index[0]:%Y-%m-%d} → {series.index[-1]:%Y-%m-%d})"
        )

    return series


# ══════════════════════════════════════════════════════════════
#  CVD (Cumulative Volume Delta) 計算
# ══════════════════════════════════════════════════════════════

def compute_cvd(taker_vol_ratio: pd.Series) -> pd.Series:
    """
    從 Taker Buy/Sell Volume Ratio 近似計算 CVD

    公式：
        taker_vol_ratio > 1 → 買方主導（買入量 > 賣出量）
        delta = (ratio - 1) / (ratio + 1)  → 標準化到 [-1, 1]
        CVD = cumsum(delta)

    這是一個近似值 — 真正的 CVD 需要逐筆成交數據。
    但 taker ratio 的累積變化能捕捉同樣的趨勢訊號。

    Args:
        taker_vol_ratio: Taker Buy/Sell Volume Ratio Series

    Returns:
        CVD 累積序列
    """
    if taker_vol_ratio.empty:
        return pd.Series(dtype=float, name="cvd")

    # 標準化 delta: ratio > 1 → positive, ratio < 1 → negative
    ratio = taker_vol_ratio.copy()
    delta = (ratio - 1.0) / (ratio + 1.0)
    delta = delta.fillna(0.0).clip(-1.0, 1.0)

    cvd = delta.cumsum()
    cvd.name = "cvd"
    return cvd


# ══════════════════════════════════════════════════════════════
#  儲存 / 載入
# ══════════════════════════════════════════════════════════════

def save_derivative(
    series: pd.Series,
    symbol: str,
    metric: str,
    data_dir: Path = DATA_DIR,
) -> Path:
    """儲存衍生品指標到 parquet"""
    path = data_dir / metric / f"{symbol}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df = series.to_frame(name=metric)
    df.to_parquet(path, index=True)
    logger.info(f"💾 Saved {metric}/{symbol}: {len(df)} rows → {path}")
    return path


def load_derivative(
    symbol: str,
    metric: str,
    data_dir: Path = DATA_DIR,
) -> pd.Series | None:
    """載入衍生品指標"""
    path = data_dir / metric / f"{symbol}.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        if metric in df.columns:
            return df[metric]
        return df.iloc[:, 0]
    except Exception as e:
        logger.warning(f"⚠️  Load {metric}/{symbol} failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════
#  覆蓋率報告
# ══════════════════════════════════════════════════════════════

def coverage_report(
    symbols: list[str],
    metrics: list[str] | None = None,
    data_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    """生成衍生品數據覆蓋率報告"""
    if metrics is None:
        metrics = ALL_METRICS + ["cvd"]

    rows = []
    for symbol in symbols:
        for metric in metrics:
            series = load_derivative(symbol, metric, data_dir)
            if series is None or series.empty:
                rows.append({
                    "symbol": symbol,
                    "metric": metric,
                    "rows": 0,
                    "start": None,
                    "end": None,
                    "coverage_days": 0,
                })
            else:
                days = (series.index[-1] - series.index[0]).days
                rows.append({
                    "symbol": symbol,
                    "metric": metric,
                    "rows": len(series),
                    "start": series.index[0].strftime("%Y-%m-%d"),
                    "end": series.index[-1].strftime("%Y-%m-%d"),
                    "coverage_days": days,
                })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
#  主程式
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Binance 衍生品數據下載工具（LSR, Taker Vol, CVD）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 下載全部指標（從 Binance Vision，完整歷史）
  PYTHONPATH=src python scripts/fetch_derivatives_data.py --symbols BTCUSDT ETHUSDT

  # 從 Binance API 下載最近 30 天
  PYTHONPATH=src python scripts/fetch_derivatives_data.py --symbols BTCUSDT --source api

  # 只下載特定指標
  PYTHONPATH=src python scripts/fetch_derivatives_data.py --symbols BTCUSDT --metrics lsr taker_vol_ratio

  # 查看覆蓋率報告
  PYTHONPATH=src python scripts/fetch_derivatives_data.py --symbols BTCUSDT ETHUSDT --coverage
        """,
    )
    parser.add_argument(
        "--symbols", nargs="+", required=True,
        help="交易對列表 (e.g. BTCUSDT ETHUSDT)",
    )
    parser.add_argument(
        "--metrics", nargs="+", default=None,
        choices=ALL_METRICS,
        help=f"要下載的指標（預設全部: {ALL_METRICS}）",
    )
    parser.add_argument(
        "--source", default="vision",
        choices=["vision", "api", "both"],
        help="數據來源: vision=完整歷史, api=最近30天, both=合併",
    )
    parser.add_argument(
        "--interval", default="1h",
        help="K 線週期 (預設: 1h)",
    )
    parser.add_argument(
        "--start", default=None,
        help="開始日期 (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end", default=None,
        help="結束日期 (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--coverage", action="store_true",
        help="只顯示覆蓋率報告，不下載",
    )
    parser.add_argument(
        "--data-dir", default=str(DATA_DIR),
        help=f"數據儲存目錄 (預設: {DATA_DIR})",
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    metrics = args.metrics or ALL_METRICS

    # 覆蓋率報告模式
    if args.coverage:
        report = coverage_report(args.symbols, data_dir=data_dir)
        if report.empty:
            print("❌ 無已下載的數據")
            return
        print("\n📊 衍生品數據覆蓋率報告")
        print("=" * 80)
        for symbol in args.symbols:
            sym_data = report[report["symbol"] == symbol]
            print(f"\n  {symbol}:")
            for _, row in sym_data.iterrows():
                if row["rows"] == 0:
                    print(f"    {row['metric']:<20} ❌ 無數據")
                else:
                    print(
                        f"    {row['metric']:<20} ✅ {row['rows']:>6} rows  "
                        f"{row['start']} → {row['end']}  ({row['coverage_days']}d)"
                    )
        print()
        return

    # 下載模式
    for symbol in args.symbols:
        print(f"\n{'='*60}")
        print(f"  📥 {symbol}")
        print(f"{'='*60}")

        all_series: dict[str, pd.Series] = {}

        # 1. Vision 來源
        if args.source in ("vision", "both"):
            vision_data = fetch_vision_metrics(
                symbol, start=args.start, end=args.end, interval=args.interval,
            )
            for m in metrics:
                if m in vision_data:
                    all_series[m] = vision_data[m]

        # 2. API 來源
        if args.source in ("api", "both"):
            for m in metrics:
                api_series = fetch_api_metric(
                    symbol, m, interval=args.interval,
                )
                if api_series.empty:
                    continue

                if m in all_series and args.source == "both":
                    # 合併：vision 為主，api 補齊尾部
                    combined = pd.concat([all_series[m], api_series])
                    combined = combined[~combined.index.duplicated(keep="last")]
                    combined = combined.sort_index()
                    all_series[m] = combined
                    logger.info(f"  {m}: merged vision + api → {len(combined)} rows")
                else:
                    all_series[m] = api_series

            # API 模式之間做 rate limiting
            time.sleep(0.5)

        # 3. 儲存各指標
        for m, series in all_series.items():
            save_derivative(series, symbol, m, data_dir)

        # 4. 計算並儲存 CVD
        if "taker_vol_ratio" in all_series:
            cvd = compute_cvd(all_series["taker_vol_ratio"])
            save_derivative(cvd, symbol, "cvd", data_dir)
            logger.info(f"  CVD: {len(cvd)} bars computed from taker_vol_ratio")

    print(f"\n✅ 下載完成！數據目錄: {data_dir}")


if __name__ == "__main__":
    main()
