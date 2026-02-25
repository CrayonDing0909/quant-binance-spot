"""
多時間框架數據載入器 (Multi-TF Loader)

載入多個 timeframe 的 K 線數據並對齊到主要 interval。
所有對齊操作嚴格因果（forward-fill only，不使用未來資訊）。

使用方式：
    from qtrade.data.multi_tf_loader import MultiTFLoader

    loader = MultiTFLoader(data_dir=Path("data"), market_type="futures")

    # 載入多 TF 數據
    tf_data = loader.load_multi_tf(
        symbol="BTCUSDT",
        primary_interval="1h",
        auxiliary_intervals=["4h", "1d"],
        start="2022-01-01",
        end="2026-01-01",
    )

    # tf_data["1h"] = 主 DataFrame (OHLCV)
    # tf_data["4h"] = 對齊到 1h index 的 4h DataFrame
    # tf_data["1d"] = 對齊到 1h index 的 1d DataFrame
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# 時間框架排序（低→高頻率）
_TF_ORDER = {
    "1M": 0, "1w": 1, "1d": 2, "12h": 3, "8h": 4,
    "6h": 5, "4h": 6, "2h": 7, "1h": 8, "30m": 9,
    "15m": 10, "5m": 11, "3m": 12, "1m": 13,
}

# Resample 頻率映射
_RESAMPLE_FREQ = {
    "1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min",
    "30m": "30min", "1h": "1h", "2h": "2h", "4h": "4h",
    "6h": "6h", "8h": "8h", "12h": "12h", "1d": "1D",
    "1w": "1W", "1M": "1ME",
}


def _tf_rank(interval: str) -> int:
    """取得 timeframe 的排序值（高頻 = 高數值）"""
    return _TF_ORDER.get(interval, 8)


def _is_higher_tf(target: str, source: str) -> bool:
    """target 是否比 source 更高頻（更低 timeframe）"""
    return _tf_rank(target) > _tf_rank(source)


class MultiTFLoader:
    """
    多時間框架數據載入器

    支援：
        1. 直接從已下載的 parquet 檔載入不同 TF 的 K 線
        2. 如果目標 TF 不存在，從更低 TF 的數據 resample
        3. 所有 auxiliary TF 對齊到 primary interval 的 index（因果 ffill）
    """

    def __init__(
        self,
        data_dir: Path = Path("data"),
        market_type: str = "futures",
    ):
        self.data_dir = data_dir
        self.market_type = market_type

    def _kline_path(self, symbol: str, interval: str) -> Path:
        """K 線數據的標準路徑"""
        return self.data_dir / "binance" / self.market_type / interval / f"{symbol}.parquet"

    def _load_klines(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        """載入單一 TF 的 K 線數據"""
        from qtrade.data.storage import load_klines

        path = self._kline_path(symbol, interval)
        if not path.exists():
            return None

        df = load_klines(path)
        if df is None or df.empty:
            return None
        return df

    def _resample_ohlcv(self, df: pd.DataFrame, target_freq: str) -> pd.DataFrame:
        """
        從低 TF resample 到高 TF (e.g. 1h → 4h)

        使用標準 OHLCV resample 規則（嚴格因果）
        """
        freq = _RESAMPLE_FREQ.get(target_freq, target_freq)

        resampled = pd.DataFrame()
        resampled["open"] = df["open"].resample(freq).first()
        resampled["high"] = df["high"].resample(freq).max()
        resampled["low"] = df["low"].resample(freq).min()
        resampled["close"] = df["close"].resample(freq).last()
        resampled["volume"] = df["volume"].resample(freq).sum()
        resampled = resampled.dropna(subset=["open", "close"])

        return resampled

    def _align_to_primary(
        self,
        aux_df: pd.DataFrame,
        primary_index: pd.DatetimeIndex,
        max_ffill_bars: int = 1,
    ) -> pd.DataFrame:
        """
        將 auxiliary TF 數據對齊到 primary index

        規則（嚴格因果）：
            - 使用 forward-fill（只用過去的值填未來）
            - max_ffill_bars 限制最大 ffill 距離
            - 不使用 backward-fill（避免 look-ahead）
        """
        # Timezone alignment
        if primary_index.tz is None and aux_df.index.tz is not None:
            aux_df = aux_df.copy()
            aux_df.index = aux_df.index.tz_localize(None)
        elif primary_index.tz is not None and aux_df.index.tz is None:
            aux_df = aux_df.copy()
            aux_df.index = aux_df.index.tz_localize(primary_index.tz)

        aligned = aux_df.reindex(primary_index, method="ffill", limit=max_ffill_bars)
        return aligned

    def load_single_tf(
        self,
        symbol: str,
        interval: str,
        start: str | None = None,
        end: str | None = None,
        fallback_resample_from: str | None = None,
    ) -> Optional[pd.DataFrame]:
        """
        載入單一 TF 的數據

        如果目標 TF 不存在且提供了 fallback_resample_from，
        會從 fallback TF resample 生成。

        Args:
            symbol: 交易對
            interval: 目標 timeframe
            start: 開始日期
            end: 結束日期
            fallback_resample_from: 當目標 TF 不存在時的 resample 來源

        Returns:
            K 線 DataFrame or None
        """
        df = self._load_klines(symbol, interval)

        # Fallback: 從更低 TF resample
        if df is None and fallback_resample_from:
            source_df = self._load_klines(symbol, fallback_resample_from)
            if source_df is not None and not _is_higher_tf(interval, fallback_resample_from):
                # fallback_resample_from 是更低 TF，可以 resample 到更高 TF
                logger.info(
                    f"📊 Resampling {symbol} {fallback_resample_from} → {interval}"
                )
                df = self._resample_ohlcv(source_df, interval)

        if df is None:
            return None

        # 過濾日期範圍
        if start:
            start_ts = pd.Timestamp(start)
            if df.index.tz is not None:
                start_ts = start_ts.tz_localize(df.index.tz)
            df = df[df.index >= start_ts]
        if end:
            end_ts = pd.Timestamp(end)
            if df.index.tz is not None:
                end_ts = end_ts.tz_localize(df.index.tz)
            df = df[df.index <= end_ts]

        return df

    def load_multi_tf(
        self,
        symbol: str,
        primary_interval: str,
        auxiliary_intervals: list[str],
        start: str | None = None,
        end: str | None = None,
        max_ffill_bars: int = 1,
    ) -> dict[str, pd.DataFrame]:
        """
        載入多 TF 數據並對齊到 primary interval

        Args:
            symbol: 交易對
            primary_interval: 主要執行 timeframe (e.g. "1h")
            auxiliary_intervals: 輔助 TF 列表 (e.g. ["4h", "1d"])
            start: 開始日期
            end: 結束日期
            max_ffill_bars: 每個 aux TF 的最大 ffill bars

        Returns:
            dict[interval, DataFrame]
            primary_interval 的 DataFrame 是原始 OHLCV
            auxiliary intervals 的 DataFrame 已對齊到 primary index
        """
        result: dict[str, pd.DataFrame] = {}

        # 1. 載入 primary
        primary_df = self.load_single_tf(symbol, primary_interval, start, end)
        if primary_df is None:
            logger.warning(f"⚠️  No primary data: {symbol} @ {primary_interval}")
            return result

        result[primary_interval] = primary_df
        primary_index = primary_df.index

        # 2. 載入並對齊 auxiliary TFs
        for aux_interval in auxiliary_intervals:
            if aux_interval == primary_interval:
                continue

            aux_df = self.load_single_tf(
                symbol, aux_interval, start, end,
                fallback_resample_from=primary_interval,
            )

            if aux_df is None:
                logger.warning(f"⚠️  No auxiliary data: {symbol} @ {aux_interval}")
                continue

            # 對齊到 primary index
            # 高 TF (e.g. 4h, 1d) → ffill 到 primary bars
            # 低 TF (e.g. 5m, 15m) → 取最後一個值（聚合到 primary bar）
            if _is_higher_tf(primary_interval, aux_interval):
                # aux 是更高 TF → ffill 到每個 primary bar
                ffill_limit = max_ffill_bars
                aligned = self._align_to_primary(aux_df, primary_index, ffill_limit)
            else:
                # aux 是更低 TF → resample 到 primary TF
                aligned = self._resample_ohlcv(aux_df, primary_interval)
                aligned = self._align_to_primary(aligned, primary_index, 1)

            # 加上 prefix 避免欄位衝突
            aligned.columns = [f"{c}_{aux_interval}" for c in aligned.columns]
            result[aux_interval] = aligned

            n_available = aligned.notna().all(axis=1).sum()
            logger.info(
                f"  {aux_interval}: {n_available}/{len(primary_index)} bars aligned "
                f"({n_available / len(primary_index) * 100:.1f}%)"
            )

        return result

    def load_derivatives(
        self,
        symbol: str,
        kline_index: pd.DatetimeIndex,
        load_lsr: bool = False,
        load_taker_vol: bool = False,
        load_cvd: bool = False,
        load_liquidation: bool = False,
    ) -> dict[str, pd.Series]:
        """
        載入衍生品數據並對齊到 K 線時間軸

        Args:
            symbol: 交易對
            kline_index: primary K 線的 DatetimeIndex
            load_lsr: 載入 Long/Short Ratio
            load_taker_vol: 載入 Taker Buy/Sell Ratio
            load_cvd: 載入 CVD
            load_liquidation: 載入清算數據

        Returns:
            dict[metric_name, pd.Series] 已對齊到 kline_index
        """
        result: dict[str, pd.Series] = {}

        if load_lsr:
            from qtrade.data.long_short_ratio import load_lsr as _load_lsr, align_lsr_to_klines
            for lsr_type in ["lsr", "top_lsr_account", "top_lsr_position"]:
                series = _load_lsr(symbol, lsr_type)
                if series is not None:
                    aligned = align_lsr_to_klines(series, kline_index)
                    if aligned is not None:
                        result[lsr_type] = aligned

        if load_taker_vol:
            from qtrade.data.taker_volume import load_taker_volume as _load_tv, align_taker_to_klines
            tv = _load_tv(symbol)
            if tv is not None:
                aligned = align_taker_to_klines(tv, kline_index)
                if aligned is not None:
                    result["taker_vol_ratio"] = aligned

        if load_cvd:
            from qtrade.data.taker_volume import load_cvd as _load_cvd, align_taker_to_klines
            cvd = _load_cvd(symbol)
            if cvd is not None:
                aligned = align_taker_to_klines(cvd, kline_index)
                if aligned is not None:
                    result["cvd"] = aligned

        if load_liquidation:
            from qtrade.data.liquidation import load_liquidation as _load_liq
            liq_df = _load_liq(symbol)
            if liq_df is not None and not liq_df.empty:
                # 對齊清算欄位
                for col in ["liq_total", "liq_imbalance", "liq_cascade_z"]:
                    if col in liq_df.columns:
                        s = liq_df[col]
                        if kline_index.tz is None and s.index.tz is not None:
                            s = s.copy()
                            s.index = s.index.tz_localize(None)
                        elif kline_index.tz is not None and s.index.tz is None:
                            s = s.copy()
                            s.index = s.index.tz_localize(kline_index.tz)
                        aligned = s.reindex(kline_index, method="ffill", limit=2).fillna(0)
                        result[col] = aligned

        return result
