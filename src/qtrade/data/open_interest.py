"""
Open Interest (OI) 歷史資料下載、快取與對齊

支援 provider：
    1. BinanceVisionOIProvider — data.binance.vision 公開資料庫 (2021-12 至今，5m→1h)
    2. BinanceOIProvider  — 免費 API，但僅 ~20 天歷史（500 records @ 1h）
    3. CoinglassOIProvider — 需 API key，支援 2020 年至今

Provider 優先級：binance_vision > coinglass > binance。

使用方式：
    from qtrade.data.open_interest import (
        download_open_interest,
        load_open_interest,
        merge_oi_sources,
        compute_oi_coverage,
    )
"""
from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# Provider Base
# ══════════════════════════════════════════════════════════════

class OIProvider(ABC):
    """OI 資料提供者基底類"""

    name: str = "base"

    @abstractmethod
    def fetch(
        self,
        symbol: str,
        interval: str = "1h",
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """
        下載 OI 資料

        Args:
            symbol: 交易對 (e.g. "BTCUSDT")
            interval: 時間間隔 ("1h", "4h", "1d")
            start: 開始日期 "YYYY-MM-DD"
            end: 結束日期 "YYYY-MM-DD"

        Returns:
            DataFrame, index=timestamp (UTC), columns=[sumOpenInterest, sumOpenInterestValue]
        """
        ...


# ══════════════════════════════════════════════════════════════
# Binance Provider
# ══════════════════════════════════════════════════════════════

class BinanceOIProvider(OIProvider):
    """
    Binance /futures/data/openInterestHist

    限制：
        - 不支援 startTime/endTime
        - 最多返回 500 筆（1h ≈ 20 天）
        - 無法分頁
    """

    name = "binance"

    def fetch(
        self,
        symbol: str,
        interval: str = "1h",
        start: str | None = None,
        end: str | None = None,
        max_retries: int = 3,
    ) -> pd.DataFrame:
        from .binance_futures_client import BinanceFuturesHTTP

        period_map = {"1h": "1h", "4h": "4h", "1d": "1d", "5m": "5m", "15m": "15m"}
        period = period_map.get(interval, "1h")

        client = BinanceFuturesHTTP()
        records = []
        for attempt in range(max_retries):
            try:
                records = client.get(
                    "/futures/data/openInterestHist",
                    {"symbol": symbol, "period": period, "limit": 500},
                )
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"⚠️  Binance OI retry ({attempt+1}): {e}")
                    time.sleep(1.0 * (attempt + 1))
                else:
                    logger.warning(f"⚠️  Binance OI failed: {e}")

        if not records:
            return pd.DataFrame(columns=["sumOpenInterest", "sumOpenInterestValue"])

        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["sumOpenInterest"] = pd.to_numeric(df["sumOpenInterest"], errors="coerce")
        df["sumOpenInterestValue"] = pd.to_numeric(df["sumOpenInterestValue"], errors="coerce")
        df = df.set_index("timestamp")[["sumOpenInterest", "sumOpenInterestValue"]]
        df = df.sort_index().pipe(lambda d: d[~d.index.duplicated(keep="last")])

        if start:
            df = df[df.index >= pd.Timestamp(start, tz="UTC")]
        if end:
            df = df[df.index <= pd.Timestamp(end, tz="UTC")]

        if not df.empty:
            logger.info(
                f"✅ Binance OI {symbol}: {len(df)} rows "
                f"({df.index[0]:%Y-%m-%d} → {df.index[-1]:%Y-%m-%d})"
            )
        return df


# ══════════════════════════════════════════════════════════════
# Binance Vision Provider (data.binance.vision)
# ══════════════════════════════════════════════════════════════

class BinanceVisionOIProvider(OIProvider):
    """
    Binance Public Data Repository — data.binance.vision

    完整歷史：2021-12-01 → 至今（約 2 天延遲）
    原始間隔：5 分鐘
    自動 resample 到策略 interval（預設 1h）

    URL 格式：
        https://data.binance.vision/data/futures/um/daily/metrics/{SYMBOL}/{SYMBOL}-metrics-{YYYY-MM-DD}.zip
    CSV 欄位：
        create_time, symbol, sum_open_interest, sum_open_interest_value,
        count_toptrader_long_short_ratio, sum_toptrader_long_short_ratio,
        count_long_short_ratio, sum_taker_long_short_vol_ratio
    """

    name = "binance_vision"
    BASE_URL = "https://data.binance.vision/data/futures/um/daily/metrics"
    # First available date on data.binance.vision
    EARLIEST_DATE = "2021-12-01"

    def fetch(
        self,
        symbol: str,
        interval: str = "1h",
        start: str | None = None,
        end: str | None = None,
        cache_dir: Path | None = None,
    ) -> pd.DataFrame:
        """
        Download daily metrics from data.binance.vision, concat, resample to target interval.

        Caches raw daily CSVs to avoid re-downloading.
        """
        import io
        import zipfile

        import requests as _requests

        start_date = pd.Timestamp(start or self.EARLIEST_DATE, tz="UTC").normalize()
        end_date = (
            pd.Timestamp(end, tz="UTC").normalize()
            if end
            else pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=2)
        )
        # Clamp to earliest available
        earliest = pd.Timestamp(self.EARLIEST_DATE, tz="UTC")
        if start_date < earliest:
            start_date = earliest

        # Cache directory
        if cache_dir is None:
            cache_dir = Path("data/binance/futures/open_interest/vision_cache") / symbol
        cache_dir.mkdir(parents=True, exist_ok=True)

        dates = pd.date_range(start=start_date, end=end_date, freq="D")
        all_dfs: list[pd.DataFrame] = []
        n_cached = 0
        n_downloaded = 0
        n_failed = 0

        logger.info(
            f"📥 BinanceVision OI: {symbol} {start_date.strftime('%Y-%m-%d')} → "
            f"{end_date.strftime('%Y-%m-%d')} ({len(dates)} days)"
        )

        for dt in dates:
            date_str = dt.strftime("%Y-%m-%d")
            csv_cache = cache_dir / f"{symbol}-metrics-{date_str}.csv"

            # Use cached CSV if exists
            if csv_cache.exists():
                try:
                    df_day = pd.read_csv(csv_cache)
                    if not df_day.empty:
                        all_dfs.append(df_day)
                        n_cached += 1
                        continue
                except Exception:
                    pass

            # Download from data.binance.vision
            zip_url = (
                f"{self.BASE_URL}/{symbol}/"
                f"{symbol}-metrics-{date_str}.zip"
            )
            try:
                resp = _requests.get(zip_url, timeout=15)
                if resp.status_code == 404:
                    # Date not available (e.g., future date or before listing)
                    continue
                resp.raise_for_status()

                with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                    csv_name = zf.namelist()[0]
                    with zf.open(csv_name) as f:
                        df_day = pd.read_csv(f)

                # Cache locally
                df_day.to_csv(csv_cache, index=False)
                all_dfs.append(df_day)
                n_downloaded += 1

            except Exception as e:
                n_failed += 1
                if n_failed <= 3:
                    logger.debug(f"  skip {date_str}: {e}")

            # Progress log every 100 days
            total = n_cached + n_downloaded + n_failed
            if total > 0 and total % 100 == 0:
                logger.info(
                    f"  ... {total}/{len(dates)} days processed "
                    f"(cached={n_cached}, downloaded={n_downloaded}, failed={n_failed})"
                )

        if not all_dfs:
            logger.warning(f"⚠️  BinanceVision: no OI data for {symbol}")
            return pd.DataFrame(columns=["sumOpenInterest", "sumOpenInterestValue"])

        # Concat all daily CSVs
        raw = pd.concat(all_dfs, ignore_index=True)
        raw["create_time"] = pd.to_datetime(raw["create_time"], utc=True)
        raw = raw.rename(columns={
            "sum_open_interest": "sumOpenInterest",
            "sum_open_interest_value": "sumOpenInterestValue",
        })
        raw["sumOpenInterest"] = pd.to_numeric(raw["sumOpenInterest"], errors="coerce")
        raw["sumOpenInterestValue"] = pd.to_numeric(raw["sumOpenInterestValue"], errors="coerce")
        raw = raw.set_index("create_time")[["sumOpenInterest", "sumOpenInterestValue"]]
        raw = raw.sort_index()
        raw = raw[~raw.index.duplicated(keep="last")]

        # Resample 5m → target interval (use last value = closing OI of each bar)
        resample_map = {
            "5m": "5min", "15m": "15min", "30m": "30min",
            "1h": "1h", "2h": "2h", "4h": "4h", "1d": "1D",
        }
        freq = resample_map.get(interval, "1h")
        if freq != "5min":
            resampled = raw.resample(freq).last().dropna(subset=["sumOpenInterest"])
        else:
            resampled = raw

        logger.info(
            f"✅ BinanceVision OI {symbol}: {len(resampled)} bars @ {interval} "
            f"({resampled.index[0]:%Y-%m-%d} → {resampled.index[-1]:%Y-%m-%d}) "
            f"[cached={n_cached}, downloaded={n_downloaded}, failed={n_failed}]"
        )

        return resampled


# ══════════════════════════════════════════════════════════════
# Coinglass Provider
# ══════════════════════════════════════════════════════════════

class CoinglassOIProvider(OIProvider):
    """
    Coinglass Open API v3

    API docs: https://coinglass.com/pricing  (free tier: 30 req/min)
    Base URL: https://open-api-v3.coinglass.com

    需要 env var: COINGLASS_API_KEY

    端點:
        GET /api/futures/openInterest/ohlc-aggregated-history
            symbol=BTC, timeType=h1, currency=USD
        回傳 OHLC OI (合計全交易所)

        GET /api/futures/openInterest/ohlc-history
            symbol=BTC, timeType=h1, exchangeName=Binance
        回傳 單交易所 OI OHLC
    """

    name = "coinglass"
    BASE_URL = "https://open-api-v3.coinglass.com"

    # Coinglass 使用幣種簡稱，非交易對
    _SYMBOL_MAP = {
        "BTCUSDT": "BTC",
        "ETHUSDT": "ETH",
        "SOLUSDT": "SOL",
        "BNBUSDT": "BNB",
        "XRPUSDT": "XRP",
        "DOGEUSDT": "DOGE",
        "ADAUSDT": "ADA",
        "AVAXUSDT": "AVAX",
    }

    _INTERVAL_MAP = {
        "5m": "m5",
        "15m": "m15",
        "30m": "m30",
        "1h": "h1",
        "2h": "h2",
        "4h": "h4",
        "6h": "h6",
        "12h": "h12",
        "1d": "1d",
    }

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("COINGLASS_API_KEY", "")
        if not self.api_key:
            logger.warning(
                "⚠️  COINGLASS_API_KEY not set. "
                "Get free key at https://www.coinglass.com/pricing"
            )

    def _request(self, endpoint: str, params: dict) -> dict:
        """Single authenticated GET request"""
        import requests as _requests

        headers = {
            "accept": "application/json",
            "CoinGlass-API-Key": self.api_key,
        }
        url = f"{self.BASE_URL}{endpoint}"
        resp = _requests.get(url, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        body = resp.json()

        if body.get("code") == "0" and body.get("success"):
            return body
        else:
            code = body.get("code", "?")
            msg = body.get("msg", "unknown")
            raise RuntimeError(f"Coinglass error [{code}]: {msg}")

    def fetch(
        self,
        symbol: str,
        interval: str = "1h",
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """
        Paginated download of Coinglass aggregated OI history.

        Uses /api/futures/openInterest/ohlc-aggregated-history for all-exchange
        aggregated OI (more robust than single-exchange).
        """
        if not self.api_key:
            logger.error("❌ COINGLASS_API_KEY not set — cannot fetch OI")
            return pd.DataFrame(columns=["sumOpenInterest", "sumOpenInterestValue"])

        cg_symbol = self._SYMBOL_MAP.get(symbol, symbol.replace("USDT", ""))
        cg_interval = self._INTERVAL_MAP.get(interval, "h1")

        # Determine time range
        if start:
            start_ts = int(pd.Timestamp(start, tz="UTC").timestamp())
        else:
            start_ts = int(pd.Timestamp("2020-01-01", tz="UTC").timestamp())
        if end:
            end_ts = int(pd.Timestamp(end, tz="UTC").timestamp())
        else:
            end_ts = int(pd.Timestamp.now(tz="UTC").timestamp())

        all_records: list[dict] = []
        current_end = end_ts
        page = 0
        max_pages = 200  # safety limit

        logger.info(
            f"📥 Coinglass OI download: {symbol} ({cg_symbol}) "
            f"{interval} from {start or '2020-01-01'} to {end or 'now'}"
        )

        while page < max_pages:
            params = {
                "symbol": cg_symbol,
                "timeType": cg_interval,
                "currency": "USD",
                "endTime": current_end,
                "limit": 500,
            }

            try:
                body = self._request(
                    "/api/futures/openInterest/ohlc-aggregated-history",
                    params,
                )
            except Exception as e:
                logger.warning(f"⚠️  Coinglass page {page} error: {e}")
                # Rate limit: wait and retry once
                if "429" in str(e) or "rate" in str(e).lower():
                    time.sleep(5)
                    try:
                        body = self._request(
                            "/api/futures/openInterest/ohlc-aggregated-history",
                            params,
                        )
                    except Exception as e2:
                        logger.error(f"❌ Coinglass retry failed: {e2}")
                        break
                else:
                    break

            data = body.get("data", [])
            if not data:
                break

            all_records.extend(data)
            page += 1

            # Find earliest timestamp in this batch
            timestamps = [r.get("t", 0) for r in data if r.get("t")]
            if not timestamps:
                break
            earliest = min(timestamps)
            if earliest <= start_ts:
                break  # We've gone past our target start

            # Next page: end just before the earliest record
            current_end = earliest - 1

            # Rate limiting: respect free tier (30 req/min)
            time.sleep(2.5)

            if page % 10 == 0:
                logger.info(
                    f"  ... page {page}, records so far: {len(all_records)}, "
                    f"earliest: {datetime.fromtimestamp(earliest, tz=timezone.utc):%Y-%m-%d}"
                )

        if not all_records:
            logger.warning(f"⚠️  Coinglass: no OI data for {symbol}")
            return pd.DataFrame(columns=["sumOpenInterest", "sumOpenInterestValue"])

        # Parse records
        rows = []
        for r in all_records:
            ts = r.get("t")
            if ts is None:
                continue
            # Use 'c' (close) as the representative OI value
            oi_val = r.get("c", r.get("o", 0))
            oi_usd = r.get("cv", oi_val)  # cv = close value in USD (if available)
            rows.append({
                "timestamp": pd.Timestamp(ts, unit="s", tz="UTC") if ts > 1e12 else pd.Timestamp(ts, unit="s", tz="UTC"),
                "sumOpenInterest": float(oi_val) if oi_val else 0.0,
                "sumOpenInterestValue": float(oi_usd) if oi_usd else 0.0,
            })

        df = pd.DataFrame(rows)
        df = df.set_index("timestamp")[["sumOpenInterest", "sumOpenInterestValue"]]
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]

        # Filter by range
        if start:
            df = df[df.index >= pd.Timestamp(start, tz="UTC")]
        if end:
            df = df[df.index <= pd.Timestamp(end, tz="UTC")]

        if not df.empty:
            logger.info(
                f"✅ Coinglass OI {symbol}: {len(df)} rows, {page} pages "
                f"({df.index[0]:%Y-%m-%d} → {df.index[-1]:%Y-%m-%d})"
            )
        else:
            logger.warning(f"⚠️  Coinglass OI {symbol}: no data after filtering")

        return df


# ══════════════════════════════════════════════════════════════
# Provider Registry
# ══════════════════════════════════════════════════════════════

_PROVIDERS: dict[str, type[OIProvider]] = {
    "binance": BinanceOIProvider,
    "binance_vision": BinanceVisionOIProvider,
    "coinglass": CoinglassOIProvider,
}


def get_oi_provider(name: str = "binance") -> OIProvider:
    """取得 OI 資料提供者"""
    cls = _PROVIDERS.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown OI provider: {name}. "
            f"Available: {list(_PROVIDERS.keys())}"
        )
    return cls()


def register_oi_provider(name: str, cls: type[OIProvider]) -> None:
    """註冊新的 OI provider"""
    _PROVIDERS[name] = cls


# ══════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════

def download_open_interest(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1h",
    provider: str = "auto",
    max_retries: int = 3,
) -> pd.DataFrame:
    """
    從指定 provider 下載歷史 Open Interest

    provider="auto" → 嘗試 coinglass（若有 API key），否則 fallback 到 binance

    Args:
        symbol: 交易對, e.g. "BTCUSDT"
        start: 開始日期 "YYYY-MM-DD"
        end: 結束日期 "YYYY-MM-DD"
        interval: 時間間隔
        provider: "binance" / "coinglass" / "auto"
        max_retries: 重試次數

    Returns:
        DataFrame, index=timestamp (UTC)
    """
    if provider == "auto":
        # Priority: binance_vision (free, full history) > coinglass > binance
        provider = "binance_vision"
        logger.info("📡 Auto-selecting BinanceVision provider (full OI history, free)")

    p = get_oi_provider(provider)
    return p.fetch(symbol, interval, start, end)


def save_open_interest(df: pd.DataFrame, path: Path) -> None:
    """儲存 OI 資料到 parquet"""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=True)
    logger.info(f"💾 OI saved: {path} ({len(df)} rows)")


def load_open_interest(path: Path) -> Optional[pd.DataFrame]:
    """載入 OI 資料"""
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        if not df.empty:
            logger.debug(f"📂 OI loaded: {path} ({len(df)} rows)")
        return df
    except Exception as e:
        logger.warning(f"⚠️  OI load failed: {e}")
        return None


def get_oi_path(data_dir: Path, symbol: str, provider: str = "merged") -> Path:
    """取得 OI 標準儲存路徑"""
    return data_dir / "binance" / "futures" / "open_interest" / provider / f"{symbol}.parquet"


# ══════════════════════════════════════════════════════════════
# Merge / Dedup / Align
# ══════════════════════════════════════════════════════════════

def merge_oi_sources(
    sources: list[pd.DataFrame],
    max_ffill_bars: int = 2,
) -> pd.DataFrame:
    """
    合併多來源 OI 資料

    規則：
        1. 合併所有 source，按 timestamp 去重（後者覆蓋前者）
        2. 僅 forward-fill 短缺口（<= max_ffill_bars）
        3. 不使用未來值填補

    Args:
        sources: list of OI DataFrames
        max_ffill_bars: 允許 forward-fill 的最大缺口（bars）

    Returns:
        merged DataFrame
    """
    valid = [df for df in sources if df is not None and not df.empty]
    if not valid:
        return pd.DataFrame(columns=["sumOpenInterest", "sumOpenInterestValue"])

    # Concat & deduplicate (later sources take priority)
    combined = pd.concat(valid, axis=0)
    combined = combined.sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]

    # Forward-fill short gaps only
    if max_ffill_bars > 0:
        combined = combined.ffill(limit=max_ffill_bars)

    # Drop any remaining NaN rows
    combined = combined.dropna(subset=["sumOpenInterest"])

    return combined


def align_oi_to_klines(
    oi_df: pd.DataFrame | None,
    kline_index: pd.DatetimeIndex,
    max_ffill_bars: int = 2,
) -> pd.Series | None:
    """
    將 OI 對齊到 K 線時間軸

    使用 forward-fill（最多 max_ffill_bars）。不使用未來值。
    """
    if oi_df is None or oi_df.empty:
        return None

    oi_series = oi_df["sumOpenInterest"]

    # Timezone alignment
    if kline_index.tz is None and oi_series.index.tz is not None:
        oi_series = oi_series.copy()
        oi_series.index = oi_series.index.tz_localize(None)
    elif kline_index.tz is not None and oi_series.index.tz is None:
        oi_series = oi_series.copy()
        oi_series.index = oi_series.index.tz_localize(kline_index.tz)

    # Reindex with limited forward-fill
    aligned = oi_series.reindex(kline_index, method="ffill", limit=max_ffill_bars)

    n_missing = aligned.isna().sum()
    n_total = len(aligned)
    if n_missing > 0:
        coverage = (n_total - n_missing) / n_total * 100
        logger.info(
            f"📊 OI alignment: {n_total - n_missing}/{n_total} bars "
            f"({coverage:.1f}% coverage), {n_missing} missing"
        )

    return aligned


# ══════════════════════════════════════════════════════════════
# Coverage Report
# ══════════════════════════════════════════════════════════════

def compute_oi_coverage(
    symbols: list[str],
    data_dir: Path,
    backtest_start: str = "2022-01-01",
    backtest_end: str | None = None,
    interval: str = "1h",
) -> pd.DataFrame:
    """
    計算 OI 資料覆蓋率報告

    對每個 symbol 按年計算覆蓋率。
    覆蓋率 < 70% 的年份標記為「不可用」。

    Returns:
        DataFrame: symbol, year, expected_bars, actual_bars, coverage_pct, provider, usable
    """
    interval_hours = {
        "5m": 1/12, "15m": 0.25, "30m": 0.5, "1h": 1,
        "2h": 2, "4h": 4, "6h": 6, "12h": 12, "1d": 24,
    }
    hours_per_bar = interval_hours.get(interval, 1)

    if backtest_end is None:
        backtest_end_ts = pd.Timestamp.now(tz="UTC")
    else:
        backtest_end_ts = pd.Timestamp(backtest_end, tz="UTC")
    start_ts = pd.Timestamp(backtest_start, tz="UTC")

    rows = []
    for symbol in symbols:
        # Try merged first, then raw providers
        oi_df = None
        provider_used = "none"
        for provider_name in ["merged", "coinglass", "binance"]:
            path = get_oi_path(data_dir, symbol, provider_name)
            oi_df = load_open_interest(path)
            if oi_df is not None and not oi_df.empty:
                provider_used = provider_name
                break

        for year in range(start_ts.year, backtest_end_ts.year + 1):
            yr_start = max(pd.Timestamp(f"{year}-01-01", tz="UTC"), start_ts)
            yr_end = min(pd.Timestamp(f"{year}-12-31 23:59:59", tz="UTC"), backtest_end_ts)
            days_in_year = (yr_end - yr_start).days + 1
            expected_bars = int(days_in_year * 24 / hours_per_bar)

            actual_bars = 0
            if oi_df is not None and not oi_df.empty:
                oi_idx = oi_df.index
                if oi_idx.tz is None:
                    oi_idx = oi_idx.tz_localize("UTC")
                yr_mask = (oi_idx >= yr_start) & (oi_idx <= yr_end)
                actual_bars = int(yr_mask.sum())

            coverage_pct = (actual_bars / expected_bars * 100) if expected_bars > 0 else 0.0

            rows.append({
                "symbol": symbol,
                "year": year,
                "expected_bars": expected_bars,
                "actual_bars": actual_bars,
                "coverage_pct": round(coverage_pct, 1),
                "provider": provider_used,
                "usable": coverage_pct >= 70.0,
            })

    return pd.DataFrame(rows)


def print_oi_coverage_report(coverage_df: pd.DataFrame) -> None:
    """印出 OI 覆蓋率報告"""
    print("\n" + "=" * 90)
    print("OI COVERAGE REPORT")
    print("=" * 90)
    print(
        f"{'Symbol':<10} {'Year':<6} {'Expected':>10} {'Actual':>10} "
        f"{'Coverage%':>10} {'Provider':<12} {'Usable?':<8}"
    )
    print("-" * 90)
    for _, row in coverage_df.iterrows():
        usable_str = "✅ YES" if row["usable"] else "❌ NO"
        print(
            f"{row['symbol']:<10} {row['year']:<6} {row['expected_bars']:>10,} "
            f"{row['actual_bars']:>10,} {row['coverage_pct']:>9.1f}% "
            f"{row['provider']:<12} {usable_str:<8}"
        )

    total_usable = coverage_df["usable"].sum()
    total_rows = len(coverage_df)
    overall_actual = coverage_df["actual_bars"].sum()
    overall_expected = coverage_df["expected_bars"].sum()
    overall_pct = (overall_actual / overall_expected * 100) if overall_expected > 0 else 0

    print("-" * 90)
    print(
        f"Summary: {total_usable}/{total_rows} year-symbol pairs usable (>= 70% coverage)"
    )
    print(f"Overall: {overall_actual:,} / {overall_expected:,} bars ({overall_pct:.1f}%)")

    if overall_pct >= 70:
        print("✅ OI data coverage SUFFICIENT — OI-based overlay conclusions are VALID")
    elif overall_pct >= 50:
        print("⚠️  OI coverage MARGINAL — results should be interpreted with caution")
    else:
        print(
            "❌ OI coverage INSUFFICIENT — OI-based overlay conclusions are INVALID\n"
            "   → OI overlay results should NOT be used for strategy decisions."
        )
    print("=" * 90)
