"""
Incremental K-Line Cache — 增量 K 線快取

解決問題：
    回測使用完整歷史數據（從第 1 bar 到最後），策略的狀態機從 bar 0 開始走。
    實盤原本每次只拉最近 300 bar（滑動窗口），窗口偏移 1 bar 就可能讓
    狀態機走向完全不同的路徑，導致信號不一致。

方案：
    首次啟動拉取 seed_bars 根 K 線作為種子，存入本地 Parquet。
    後續每次 cron 只拉「自快取最後一根以來的新 K 線」並 append。
    策略從 bar 0 跑到最新 bar → 與回測行為一致。

格式：
    cache/{symbol}.parquet — 僅含已收盤 K 線（OHLCV）
    close_time 用於過濾未收盤 bar 後即刻移除，記憶體中不保留。

典型大小：
    1h K 線 × 1 年 ≈ 8,760 bar × ~50 bytes ≈ 430 KB（可忽略）

記憶體管理：
    max_bars 參數限制快取保留的最大 bar 數（預設 1000）。
    超過時自動裁剪最舊的 bar，避免長期運行 OOM。
    生產策略最長 lookback 約 200 bar（TSMOM 168h + EMA warmup），
    1000 bar 綽綽有餘。
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

from ..data.klines import fetch_klines, FUTURES_BASE_URL, KLINE_COLS
from ..data.binance_client import BinanceHTTP
from ..data.quality import clean_data
from ..utils.log import get_logger

logger = get_logger("kline_cache")


class IncrementalKlineCache:
    """
    增量 K 線快取

    Usage:
        cache = IncrementalKlineCache(cache_dir, interval="1h")
        df = cache.get_klines("BTCUSDT")   # 首次拉 300 bar，後續只拉增量
        # df 會越來越長，等效回測的完整歷史
    """

    def __init__(
        self,
        cache_dir: Path,
        interval: str = "1h",
        seed_bars: int = 300,
        market_type: str = "futures",
        max_bars: int = 1000,
    ):
        """
        Args:
            cache_dir:    快取目錄，例如 reports/futures/rsi_adx_atr/live/kline_cache/
            interval:     K 線週期，例如 "1h"
            seed_bars:    首次拉取的 K 線數量（種子）
            market_type:  "spot" 或 "futures"
            max_bars:     記憶體中保留的最大 bar 數量（防止無限增長導致 OOM）
                          設為 0 或 None 則不限制
        """
        self.cache_dir = Path(cache_dir)
        self.interval = interval
        self.seed_bars = seed_bars
        self.market_type = market_type
        self.max_bars = max_bars or 0

        # 記憶體快取（避免每次都讀 Parquet）
        self._mem_cache: dict[str, pd.DataFrame] = {}

        # interval → 分鐘
        self._interval_minutes = {
            "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
            "1h": 60, "2h": 120, "4h": 240, "6h": 360, "8h": 480,
            "12h": 720, "1d": 1440,
        }.get(interval, 60)

    # ── 公開 API ──────────────────────────────────────────────

    def get_klines(self, symbol: str) -> pd.DataFrame:
        """
        取得完整的已收盤 K 線（含歷史快取 + 最新增量）

        首次呼叫 → 拉 seed_bars 根，存入快取
        後續呼叫 → 從快取最後一根往後拉新的 bar，append

        Returns:
            DataFrame (index=open_time UTC, cols=[open, high, low, close, volume])
        """
        cached = self._load(symbol)

        if cached is not None and len(cached) > 0:
            # ── 增量更新 ──
            new_bars = self._fetch_since(symbol, cached.index[-1])

            if new_bars is not None and len(new_bars) > 0:
                combined = pd.concat([cached, new_bars])
                combined = combined[~combined.index.duplicated(keep="last")]
                combined = combined.sort_index()
                combined = self._drop_unclosed(combined)
                combined = clean_data(
                    combined,
                    fill_method="forward",
                    remove_outliers=False,
                    remove_duplicates=True,
                )
                self._save(symbol, combined)
                logger.info(
                    f"📦 {symbol}: 快取增量更新 +{len(new_bars)} bar "
                    f"→ 總計 {len(combined)} bar "
                    f"({combined.index[0].strftime('%Y-%m-%d')} ~ "
                    f"{combined.index[-1].strftime('%Y-%m-%d %H:%M')})"
                )
                return combined
            else:
                logger.debug(f"  {symbol}: 快取已是最新 ({len(cached)} bar)")
                return cached
        else:
            # ── 首次啟動：拉取種子數據 ──
            seed = self._fetch_seed(symbol)
            if seed is not None and len(seed) > 0:
                self._save(symbol, seed)
                logger.info(
                    f"🌱 {symbol}: 首次建立快取 {len(seed)} bar "
                    f"({seed.index[0].strftime('%Y-%m-%d')} ~ "
                    f"{seed.index[-1].strftime('%Y-%m-%d %H:%M')})"
                )
            else:
                logger.warning(f"⚠️  {symbol}: 無法取得種子數據")
            return seed if seed is not None else pd.DataFrame()

    def get_bar_count(self, symbol: str) -> int:
        """取得快取中的 bar 數量（不觸發更新）"""
        cached = self._load(symbol)
        return len(cached) if cached is not None else 0

    def clear(self, symbol: str | None = None) -> None:
        """清除快取（symbol=None 清全部）"""
        if symbol:
            path = self._cache_path(symbol)
            if path.exists():
                path.unlink()
            self._mem_cache.pop(symbol, None)
            logger.info(f"🗑️  {symbol}: 快取已清除")
        else:
            if self.cache_dir.exists():
                for f in self.cache_dir.glob("*.parquet"):
                    f.unlink()
            self._mem_cache.clear()
            logger.info("🗑️  所有快取已清除")

    # ── WebSocket 整合 ─────────────────────────────────────────

    def get_cached(self, symbol: str) -> pd.DataFrame | None:
        """
        取得記憶體中的快取數據（不觸發 HTTP 更新）

        適用於 WebSocket 模式：由 WS 負責增量更新，策略讀取時不需要 HTTP。

        Returns:
            DataFrame or None if no cache exists
        """
        return self._load(symbol)

    def append_bar(self, symbol: str, bar_df: pd.DataFrame) -> pd.DataFrame:
        """
        追加單根 K 線到快取（WebSocket 用）

        不做 HTTP 請求，直接追加到記憶體 + 磁碟快取。
        與 get_klines() 的 HTTP 增量更新互補。

        Args:
            symbol:  交易對
            bar_df:  單行 DataFrame (index=open_time UTC,
                     cols=[open, high, low, close, volume] + optional close_time)

        Returns:
            更新後的完整 DataFrame
        """
        cached = self._load(symbol)

        # 確保 UTC index
        if bar_df.index.tz is None:
            bar_df.index = bar_df.index.tz_localize("UTC")

        # WS bar 已確認收盤，移除 close_time（記憶體中不保留）
        if "close_time" in bar_df.columns:
            bar_df = bar_df.drop(columns=["close_time"])

        if cached is not None and len(cached) > 0:
            combined = pd.concat([cached, bar_df])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined = combined.sort_index()
        else:
            combined = bar_df

        self._save(symbol, combined)
        logger.debug(f"  {symbol}: 追加 1 bar → 總計 {len(combined)} bar")
        return combined

    def fill_gap(self, symbol: str, last_cached_time: pd.Timestamp) -> pd.DataFrame | None:
        """
        補齊快取缺口（WebSocket 斷線重連後使用）

        從 last_cached_time 往後拉取遺漏的 K 線。

        Returns:
            更新後的完整 DataFrame, 或 None 如果失敗
        """
        try:
            new_bars = self._fetch_since(symbol, last_cached_time)
            if new_bars is not None and len(new_bars) > 0:
                cached = self._load(symbol)
                if cached is not None and len(cached) > 0:
                    combined = pd.concat([cached, new_bars])
                    combined = combined[~combined.index.duplicated(keep="last")]
                    combined = combined.sort_index()
                    combined = self._drop_unclosed(combined)
                    self._save(symbol, combined)
                    logger.info(
                        f"📦 {symbol}: 補齊缺口 +{len(new_bars)} bar → 總計 {len(combined)} bar"
                    )
                    return combined
            return self._load(symbol)
        except Exception as e:
            logger.warning(f"⚠️  {symbol}: 補齊缺口失敗: {e}")
            return self._load(symbol)

    # ── 內部方法 ──────────────────────────────────────────────

    def _cache_path(self, symbol: str) -> Path:
        return self.cache_dir / f"{symbol}.parquet"

    def _load(self, symbol: str) -> pd.DataFrame | None:
        """從記憶體快取或磁碟 Parquet 載入"""
        # 記憶體快取
        if symbol in self._mem_cache:
            return self._mem_cache[symbol]

        # 磁碟
        path = self._cache_path(symbol)
        if not path.exists():
            return None

        try:
            df = pd.read_parquet(path)

            # 確保 index 是 DatetimeIndex (UTC)
            if not isinstance(df.index, pd.DatetimeIndex):
                if "open_time" in df.columns:
                    df = df.set_index("open_time")
                df.index = pd.to_datetime(df.index, utc=True)

            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")

            # 向後相容：舊 parquet 可能還有 close_time，載入記憶體時移除
            if "close_time" in df.columns:
                df = df.drop(columns=["close_time"])

            self._mem_cache[symbol] = df
            logger.debug(f"  {symbol}: 從磁碟載入快取 {len(df)} bar")
            return df
        except Exception as e:
            logger.warning(f"⚠️  {symbol}: 載入快取失敗: {e}")
            return None

    def _save(self, symbol: str, df: pd.DataFrame) -> None:
        """保存到記憶體快取和磁碟 Parquet（超過 max_bars 時裁剪舊資料）"""
        if self.max_bars > 0 and len(df) > self.max_bars:
            trimmed = len(df) - self.max_bars
            df = df.iloc[-self.max_bars:]
            logger.debug(f"  {symbol}: 裁剪快取 -{trimmed} bar → 保留 {self.max_bars} bar")

        self._mem_cache[symbol] = df

        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            df.to_parquet(self._cache_path(symbol))
        except Exception as e:
            logger.warning(f"⚠️  {symbol}: 保存快取失敗: {e}")

    def _fetch_seed(self, symbol: str) -> pd.DataFrame | None:
        """首次啟動：拉取 seed_bars 根已收盤 K 線"""
        try:
            start_dt = datetime.now(timezone.utc) - timedelta(
                minutes=self._interval_minutes * (self.seed_bars + 10)
            )
            start_str = start_dt.strftime("%Y-%m-%d")

            df = fetch_klines(
                symbol=symbol,
                interval=self.interval,
                start=start_str,
                market_type=self.market_type,
            )
            df = clean_data(
                df, fill_method="forward",
                remove_outliers=False, remove_duplicates=True,
            )
            df = self._drop_unclosed(df)

            # 種子只取最近 seed_bars 根
            if len(df) > self.seed_bars:
                df = df.iloc[-self.seed_bars:]

            return df
        except Exception as e:
            logger.error(f"❌ {symbol}: 拉取種子數據失敗: {e}")
            return None

    def _fetch_since(
        self, symbol: str, last_time: pd.Timestamp,
    ) -> pd.DataFrame | None:
        """增量拉取：從 last_time 之後的新 K 線"""
        try:
            # 從快取最後一根的開盤時間往後拉
            # +1 ms 避免重複拉最後一根
            start_ms = int(last_time.timestamp() * 1000) + 1

            if self.market_type == "futures":
                http = BinanceHTTP(base_url=FUTURES_BASE_URL)
                endpoint = "/fapi/v1/klines"
            else:
                http = BinanceHTTP()
                endpoint = "/api/v3/klines"

            params = {
                "symbol": symbol,
                "interval": self.interval,
                "startTime": start_ms,
                "limit": 1000,
            }

            chunk = http.get(endpoint, params=params)
            if not chunk:
                return None

            df = pd.DataFrame(chunk, columns=KLINE_COLS)
            for c in ["open", "high", "low", "close", "volume"]:
                df[c] = df[c].astype(float)
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
            df = df.set_index("open_time").sort_index()
            df = df[["open", "high", "low", "close", "volume", "close_time"]]

            df = self._drop_unclosed(df)
            return df if len(df) > 0 else None

        except Exception as e:
            logger.warning(f"⚠️  {symbol}: 增量拉取失敗: {e}")
            return None

    @staticmethod
    def _drop_unclosed(df: pd.DataFrame) -> pd.DataFrame:
        """
        丟棄未收盤的 K 線，然後移除 close_time 欄位以節省記憶體。

        策略不需要 close_time（只用 OHLCV），此欄位僅用於判斷是否已收盤。
        移除後每個 symbol 可節省 ~8 bytes/bar 的 datetime64 記憶體。
        """
        if len(df) == 0:
            return df
        if "close_time" not in df.columns:
            return df
        now = pd.Timestamp.now(tz="UTC")
        # 保留 close_time 為 NaN 的行（舊快取已移除 close_time 的資料）
        has_ct = df["close_time"].notna()
        unclosed = has_ct & (df["close_time"] > now)
        n_dropped = unclosed.sum()
        if n_dropped > 0:
            logger.debug(f"  丟棄 {n_dropped} 根未收盤 K 線")
        df = df[~unclosed]
        # 移除 close_time — 策略不需要，節省記憶體
        df = df.drop(columns=["close_time"], errors="ignore")
        return df
