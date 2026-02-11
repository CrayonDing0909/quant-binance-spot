"""
Binance Data Vision - 官方歷史數據批量下載

從 data.binance.vision 下載 Binance 的完整歷史 K 線數據。
數據從 2017-08-17 (Binance 上線日) 開始提供。

優點:
- 官方數據，品質保證
- 批量下載，速度快
- 支援 Spot / Futures 所有交易對

使用方式:
    from qtrade.data.binance_vision import download_binance_vision_klines
    
    df = download_binance_vision_klines("BTCUSDT", "1h", "2017-08-17", "2024-01-01")
"""
from __future__ import annotations

import pandas as pd
import requests
import zipfile
import io
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

# Binance Data Vision base URL
BASE_URL = "https://data.binance.vision/data"

# 可用的數據類型
DATA_TYPES = {
    "spot": "spot/monthly/klines",
    "futures": "futures/um/monthly/klines",  # USDT-M Futures
    "futures_coin": "futures/cm/monthly/klines",  # Coin-M Futures
}

# K 線欄位（Binance 標準格式）
KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "num_trades",
    "taker_buy_base", "taker_buy_quote", "ignore"
]


def _generate_monthly_urls(
    symbol: str,
    interval: str,
    start: str,
    end: str,
    market_type: str = "spot",
) -> list[tuple[str, str]]:
    """
    生成每個月的下載 URL
    
    Returns:
        list of (url, year_month) tuples
    """
    data_path = DATA_TYPES.get(market_type, DATA_TYPES["spot"])
    
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    
    urls = []
    current = start_dt.replace(day=1)  # 從月初開始
    
    while current <= end_dt:
        year_month = current.strftime("%Y-%m")
        
        # 構建 URL
        # 格式: https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2023-01.zip
        filename = f"{symbol}-{interval}-{year_month}.zip"
        url = f"{BASE_URL}/{data_path}/{symbol}/{interval}/{filename}"
        
        urls.append((url, year_month))
        
        # 移到下個月
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
    
    return urls


def _download_single_month(
    url: str,
    year_month: str,
    session: requests.Session,
    max_retries: int = 3,
) -> Optional[pd.DataFrame]:
    """下載並解析單個月的數據"""
    
    for attempt in range(max_retries):
        try:
            response = session.get(url, timeout=30)
            
            if response.status_code == 404:
                # 數據不存在（可能是未來的月份或太早的數據）
                logger.debug(f"⚠️  {year_month}: 數據不存在")
                return None
            
            response.raise_for_status()
            
            # 解壓縮 ZIP
            with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                # ZIP 內應該只有一個 CSV 文件
                csv_filename = zf.namelist()[0]
                with zf.open(csv_filename) as csv_file:
                    df = pd.read_csv(csv_file, header=None, names=KLINE_COLS)
            
            return df
            
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.warning(f"⚠️  {year_month}: 下載失敗，{wait_time}s 後重試 ({attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                logger.error(f"❌ {year_month}: 下載失敗 - {e}")
                return None
        except Exception as e:
            logger.error(f"❌ {year_month}: 解析失敗 - {e}")
            return None
    
    return None


def download_binance_vision_klines(
    symbol: str,
    interval: str,
    start: str,
    end: str | None = None,
    market_type: str = "spot",
    max_workers: int = 4,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    從 Binance Data Vision 下載歷史 K 線數據
    
    Args:
        symbol: 交易對，如 "BTCUSDT"
        interval: K 線週期，如 "1h", "4h", "1d"
        start: 開始日期，格式 "YYYY-MM-DD"
        end: 結束日期，格式 "YYYY-MM-DD"（None = 到上個月底）
        market_type: "spot", "futures", "futures_coin"
        max_workers: 並行下載的線程數
        show_progress: 是否顯示進度
        
    Returns:
        DataFrame with columns: open, high, low, close, volume, close_time
        Index: open_time (UTC timezone-aware)
        
    Note:
        Binance Data Vision 只提供到上個月的月度數據。
        如需最近的數據，請配合 fetch_klines() 使用。
    """
    # 設定結束日期（默認到上個月）
    if end is None:
        now = datetime.now(timezone.utc)
        # 上個月最後一天
        first_of_month = now.replace(day=1)
        end = (first_of_month - timedelta(days=1)).strftime("%Y-%m-%d")
    
    logger.info(f"📥 Binance Vision: 下載 {symbol} {interval} ({start} → {end}) [{market_type}]")
    
    # 生成 URL 列表
    urls = _generate_monthly_urls(symbol, interval, start, end, market_type)
    
    if not urls:
        logger.warning("⚠️  沒有需要下載的數據")
        return pd.DataFrame()
    
    logger.info(f"📊 共 {len(urls)} 個月的數據需要下載")
    
    # 使用 Session 重用連接
    session = requests.Session()
    
    # 並行下載
    all_dfs = []
    completed = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任務
        future_to_month = {
            executor.submit(_download_single_month, url, month, session): month
            for url, month in urls
        }
        
        # 收集結果
        for future in as_completed(future_to_month):
            month = future_to_month[future]
            completed += 1
            
            try:
                df = future.result()
                if df is not None and not df.empty:
                    all_dfs.append(df)
                    if show_progress:
                        logger.info(f"  ✅ {month}: {len(df)} 筆 ({completed}/{len(urls)})")
                else:
                    if show_progress:
                        logger.debug(f"  ⏭️  {month}: 無數據 ({completed}/{len(urls)})")
            except Exception as e:
                logger.error(f"  ❌ {month}: {e}")
    
    session.close()
    
    if not all_dfs:
        logger.warning("⚠️  所有月份都沒有數據")
        return pd.DataFrame()
    
    # 合併所有數據
    df = pd.concat(all_dfs, ignore_index=True)
    
    # 處理時間
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    
    # 排序並去重
    df = df.sort_values("open_time")
    df = df.drop_duplicates(subset=["open_time"], keep="last")
    df = df.set_index("open_time")
    
    # 過濾時間範圍
    start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    df = df[(df.index >= start_dt) & (df.index < end_dt)]
    
    # 轉換類型
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    
    # 只保留需要的欄位
    df = df[["open", "high", "low", "close", "volume", "close_time"]]
    
    logger.info(f"✅ Binance Vision: 完成，共 {len(df)} 筆 ({df.index[0]} → {df.index[-1]})")
    
    return df


def get_available_symbols(market_type: str = "spot") -> list[str]:
    """
    獲取 Binance Data Vision 上可用的交易對列表
    
    注意: 這會發起一個 HTTP 請求來獲取目錄列表
    """
    data_path = DATA_TYPES.get(market_type, DATA_TYPES["spot"])
    
    # Binance Data Vision 沒有提供 API 列出所有 symbols
    # 這裡返回常見的交易對
    common_symbols = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
        "DOGEUSDT", "SOLUSDT", "DOTUSDT", "MATICUSDT", "LTCUSDT",
        "LINKUSDT", "AVAXUSDT", "ATOMUSDT", "UNIUSDT", "XLMUSDT",
        "TRXUSDT", "ETCUSDT", "BCHUSDT", "NEARUSDT", "ICPUSDT",
    ]
    
    return common_symbols


def check_data_availability(
    symbol: str,
    interval: str,
    market_type: str = "spot",
) -> dict:
    """
    檢查指定交易對在 Binance Data Vision 上的數據可用性
    
    Returns:
        dict: {available: bool, earliest_month: str, message: str}
    """
    data_path = DATA_TYPES.get(market_type, DATA_TYPES["spot"])
    
    # 從 2017-08 開始嘗試
    test_months = [
        "2017-08", "2017-09", "2018-01", "2019-01", "2020-01"
    ]
    
    session = requests.Session()
    earliest = None
    
    for month in test_months:
        filename = f"{symbol}-{interval}-{month}.zip"
        url = f"{BASE_URL}/{data_path}/{symbol}/{interval}/{filename}"
        
        try:
            response = session.head(url, timeout=10)
            if response.status_code == 200:
                earliest = month
                break
        except:
            pass
    
    session.close()
    
    if earliest:
        return {
            "available": True,
            "earliest_month": earliest,
            "message": f"✅ {symbol} 數據可用，最早從 {earliest} 開始",
        }
    else:
        return {
            "available": False,
            "earliest_month": None,
            "message": f"❌ {symbol} 在 Binance Data Vision 上找不到數據",
        }
