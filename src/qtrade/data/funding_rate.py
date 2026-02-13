"""
Binance Futures Funding Rate 歷史資料下載與快取

Binance USDT-M 永續合約每 8 小時結算一次 funding：
- 結算時間：00:00, 08:00, 16:00 UTC
- funding_rate > 0 → 多頭付費給空頭
- funding_rate < 0 → 空頭付費給多頭
- cost = position_value × funding_rate

使用方式：
    from qtrade.data.funding_rate import download_funding_rates, load_funding_rates

    # 下載並儲存
    df = download_funding_rates("BTCUSDT", "2022-01-01", "2024-12-31")
    save_funding_rates(df, Path("data/funding/BTCUSDT.parquet"))

    # 載入
    df = load_funding_rates(Path("data/funding/BTCUSDT.parquet"))
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def download_funding_rates(
    symbol: str,
    start: str,
    end: str | None = None,
    max_retries: int = 3,
) -> pd.DataFrame:
    """
    從 Binance Futures API 下載歷史 funding rate（自動分頁）

    Args:
        symbol: 交易對, e.g. "BTCUSDT"
        start: 開始日期 "YYYY-MM-DD"
        end: 結束日期 "YYYY-MM-DD"（None = 到現在）
        max_retries: 每次請求最大重試次數

    Returns:
        DataFrame, index=funding_time (UTC), columns=[funding_rate, mark_price]
    """
    from .binance_futures_client import BinanceFuturesHTTP

    client = BinanceFuturesHTTP()

    start_ts = int(
        datetime.strptime(start, "%Y-%m-%d")
        .replace(tzinfo=timezone.utc)
        .timestamp()
        * 1000
    )
    end_ts = (
        int(
            datetime.strptime(end, "%Y-%m-%d")
            .replace(tzinfo=timezone.utc)
            .timestamp()
            * 1000
        )
        if end
        else int(datetime.now(timezone.utc).timestamp() * 1000)
    )

    all_records: list[dict] = []
    cursor = start_ts
    page = 0
    limit = 1000  # Binance API max per request

    while cursor < end_ts:
        page += 1
        params = {
            "symbol": symbol,
            "startTime": cursor,
            "endTime": end_ts,
            "limit": limit,
        }

        try:
            records = client.get("/fapi/v1/fundingRate", params)
        except Exception as e:
            logger.warning(f"⚠️  Funding rate 下載失敗 (page {page}): {e}")
            break

        if not records:
            break

        all_records.extend(records)
        logger.info(
            f"  📥 Funding rate page {page}: {len(records)} records "
            f"(累計 {len(all_records)})"
        )

        # 移動 cursor 到最後一筆的下一毫秒
        last_time = int(records[-1]["fundingTime"])
        cursor = last_time + 1

        if len(records) < limit:
            break  # 已經是最後一頁

        time.sleep(0.2)  # Rate limit 保護

    if not all_records:
        logger.warning(f"⚠️  {symbol} 沒有 funding rate 資料")
        return pd.DataFrame(columns=["funding_rate", "mark_price"])

    # 轉換為 DataFrame
    df = pd.DataFrame(all_records)
    df["funding_time"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["funding_rate"] = df["fundingRate"].astype(float)
    df["mark_price"] = df["markPrice"].astype(float)
    df = df.set_index("funding_time")[["funding_rate", "mark_price"]]
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    logger.info(
        f"✅ {symbol} funding rate: {len(df)} records "
        f"({df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')})"
    )
    return df


def save_funding_rates(df: pd.DataFrame, path: Path) -> None:
    """儲存 funding rate 資料"""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=True)


def load_funding_rates(path: Path) -> Optional[pd.DataFrame]:
    """
    載入 funding rate 資料

    Returns:
        DataFrame 或 None（檔案不存在）
    """
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception as e:
        logger.warning(f"⚠️  載入 funding rate 失敗: {e}")
        return None


def get_funding_rate_path(data_dir: Path, symbol: str) -> Path:
    """取得 funding rate 的標準儲存路徑"""
    return data_dir / "binance" / "futures" / "funding_rate" / f"{symbol}.parquet"


def align_funding_to_klines(
    funding_df: pd.DataFrame,
    kline_index: pd.DatetimeIndex,
    default_rate_8h: float = 0.0001,
) -> pd.Series:
    """
    將 8h funding rate 對齊到 K 線時間軸

    每 8 小時的結算時刻（00:00, 08:00, 16:00 UTC）標記 funding rate，
    其他 bar 填 0（因為 funding 只在結算時刻發生）。

    Args:
        funding_df: Funding rate DataFrame (index=funding_time)
        kline_index: K 線的時間 index
        default_rate_8h: 無資料時的預設費率（每 8h）

    Returns:
        Series, index 與 kline_index 相同, 值為該 bar 的 funding rate（非結算時刻=0）
    """
    if funding_df is None or funding_df.empty:
        # 用預設費率：在結算時刻填入，其他填 0
        result = pd.Series(0.0, index=kline_index, name="funding_rate")
        for ts in kline_index:
            if ts.hour in (0, 8, 16) and ts.minute == 0:
                result.loc[ts] = default_rate_8h
        return result

    # 確保時區一致
    if kline_index.tz is None and funding_df.index.tz is not None:
        funding_df = funding_df.copy()
        funding_df.index = funding_df.index.tz_localize(None)
    elif kline_index.tz is not None and funding_df.index.tz is None:
        funding_df = funding_df.copy()
        funding_df.index = funding_df.index.tz_localize(kline_index.tz)

    # Reindex 到 kline 時間軸，非結算時刻為 0
    aligned = funding_df["funding_rate"].reindex(kline_index, fill_value=0.0)

    # 對於 kline_index 範圍內但 funding_df 缺少的結算時刻，用預設費率
    for ts in kline_index:
        if ts.hour in (0, 8, 16) and ts.minute == 0:
            if pd.isna(aligned.loc[ts]) or aligned.loc[ts] == 0.0:
                # 檢查這個時刻是否在 funding_df 的範圍內但缺失
                if funding_df.empty or ts < funding_df.index[0] or ts > funding_df.index[-1]:
                    aligned.loc[ts] = default_rate_8h

    return aligned.fillna(0.0)
