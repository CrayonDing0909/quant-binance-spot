"""
衍生品數據通用下載/載入模組

提供 Vision 和 API 兩種模式的統一介面，供 long_short_ratio / taker_volume / liquidation 模組使用。
"""
from __future__ import annotations

import io
import logging
import time
import zipfile
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Binance Vision metrics CSV 欄位映射
VISION_METRIC_MAP = {
    "lsr": "count_long_short_ratio",
    "top_lsr_account": "count_toptrader_long_short_ratio",
    "top_lsr_position": "sum_toptrader_long_short_ratio",
    "taker_vol_ratio": "sum_taker_long_short_vol_ratio",
}

# Binance API endpoints
API_ENDPOINTS = {
    "lsr": "/futures/data/globalLongShortAccountRatio",
    "top_lsr_account": "/futures/data/topLongShortAccountRatio",
    "top_lsr_position": "/futures/data/topLongShortPositionRatio",
    "taker_vol_ratio": "/futures/data/takerlongshortRatio",
}

VISION_BASE_URL = "https://data.binance.vision/data/futures/um/daily/metrics"
VISION_EARLIEST_DATE = "2021-12-01"


def fetch_vision_single_metric(
    symbol: str,
    metric: str,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1h",
) -> pd.Series:
    """
    從 data.binance.vision 下載單一衍生品指標

    利用 OI 模組已下載的 vision_cache CSV（避免重複下載）。
    """
    import requests

    raw_col = VISION_METRIC_MAP.get(metric)
    if raw_col is None:
        raise ValueError(f"Unknown metric: {metric}. Available: {list(VISION_METRIC_MAP.keys())}")

    start_date = pd.Timestamp(start or VISION_EARLIEST_DATE, tz="UTC").normalize()
    end_date = (
        pd.Timestamp(end, tz="UTC").normalize()
        if end
        else pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=2)
    )
    earliest = pd.Timestamp(VISION_EARLIEST_DATE, tz="UTC")
    if start_date < earliest:
        start_date = earliest

    # 共用 OI vision cache
    cache_dir = Path("data/binance/futures/open_interest/vision_cache") / symbol
    cache_dir.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    all_dfs: list[pd.DataFrame] = []
    n_cached = n_downloaded = n_failed = 0

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

    if not all_dfs:
        return pd.Series(dtype=float, name=metric)

    raw = pd.concat(all_dfs, ignore_index=True)
    raw["create_time"] = pd.to_datetime(raw["create_time"], utc=True)
    raw = raw.sort_values("create_time").set_index("create_time")
    raw = raw[~raw.index.duplicated(keep="last")]

    if raw_col not in raw.columns:
        logger.warning(f"Column '{raw_col}' not found for {metric}")
        return pd.Series(dtype=float, name=metric)

    series = pd.to_numeric(raw[raw_col], errors="coerce")

    # Resample
    resample_map = {
        "5m": "5min", "15m": "15min", "30m": "30min",
        "1h": "1h", "2h": "2h", "4h": "4h", "1d": "1D",
    }
    freq = resample_map.get(interval, "1h")
    if freq != "5min":
        series = series.resample(freq).last().dropna()

    series.name = metric
    if not series.empty:
        logger.info(
            f"✅ Vision {metric} {symbol}: {len(series)} bars @ {interval} "
            f"(cached={n_cached}, dl={n_downloaded})"
        )
    return series


def fetch_api_single_metric(
    symbol: str,
    metric: str,
    interval: str = "1h",
    limit: int = 500,
) -> pd.Series:
    """
    從 Binance Futures API 下載單一衍生品指標 (~30 天)
    """
    from qtrade.data.binance_futures_client import BinanceFuturesHTTP

    endpoint = API_ENDPOINTS.get(metric)
    if endpoint is None:
        raise ValueError(f"Unknown metric: {metric}")

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
        logger.error(f"❌ API {metric} {symbol}: {e}")
        return pd.Series(dtype=float, name=metric)

    if not records:
        return pd.Series(dtype=float, name=metric)

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    value_col = None
    for col in ["longShortRatio", "buySellRatio", "longAccount", "longPosition"]:
        if col in df.columns:
            value_col = col
            break
    if value_col is None:
        num_cols = [c for c in df.columns if c not in ("timestamp", "symbol")]
        value_col = num_cols[0] if num_cols else None

    if value_col is None:
        return pd.Series(dtype=float, name=metric)

    series = pd.to_numeric(df.set_index("timestamp")[value_col], errors="coerce")
    series = series.sort_index()
    series = series[~series.index.duplicated(keep="last")]
    series.name = metric
    return series
