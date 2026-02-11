"""
Yahoo Finance 數據源 - 提供長期歷史加密貨幣數據

支援的交易對 (BTC 可追溯至 2014-09):
- BTC-USD, ETH-USD, BNB-USD, SOL-USD, XRP-USD, ADA-USD, DOGE-USD 等

使用方式:
    from qtrade.data.yfinance_client import fetch_yfinance_klines
    
    df = fetch_yfinance_klines("BTC-USD", "1h", "2015-01-01", "2024-01-01")
"""
from __future__ import annotations

import pandas as pd
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

# Yahoo Finance 交易對名稱映射 (Binance style -> Yahoo style)
SYMBOL_MAPPING = {
    "BTCUSDT": "BTC-USD",
    "ETHUSDT": "ETH-USD",
    "BNBUSDT": "BNB-USD",
    "SOLUSDT": "SOL-USD",
    "XRPUSDT": "XRP-USD",
    "ADAUSDT": "ADA-USD",
    "DOGEUSDT": "DOGE-USD",
    "DOTUSDT": "DOT-USD",
    "MATICUSDT": "MATIC-USD",
    "LTCUSDT": "LTC-USD",
    "LINKUSDT": "LINK-USD",
    "AVAXUSDT": "AVAX-USD",
    "ATOMUSDT": "ATOM-USD",
    "UNIUSDT": "UNI-USD",
    "XLMUSDT": "XLM-USD",
}

# Yahoo Finance interval 映射
INTERVAL_MAPPING = {
    "1m": "1m",      # 最近 7 天
    "2m": "2m",      # 最近 60 天
    "5m": "5m",      # 最近 60 天
    "15m": "15m",    # 最近 60 天
    "30m": "30m",    # 最近 60 天
    "1h": "1h",      # 最近 730 天
    "1d": "1d",      # 全部歷史
    "1wk": "1wk",    # 全部歷史
    "1mo": "1mo",    # 全部歷史
}


def convert_symbol(binance_symbol: str) -> str:
    """將 Binance 風格的交易對轉換為 Yahoo Finance 格式"""
    # 直接映射
    if binance_symbol in SYMBOL_MAPPING:
        return SYMBOL_MAPPING[binance_symbol]
    
    # 嘗試自動轉換 (XXXUSDT -> XXX-USD)
    if binance_symbol.endswith("USDT"):
        base = binance_symbol[:-4]
        return f"{base}-USD"
    
    # 原樣返回（可能已經是 Yahoo 格式）
    return binance_symbol


def fetch_yfinance_klines(
    symbol: str,
    interval: str,
    start: str,
    end: str | None = None,
) -> pd.DataFrame:
    """
    從 Yahoo Finance 下載 K 線資料
    
    Args:
        symbol: 交易對，支援 Binance 格式 (如 "BTCUSDT") 或 Yahoo 格式 (如 "BTC-USD")
        interval: K 線週期，如 "1h", "1d" (注意: 小週期數據有時間限制)
        start: 開始日期，格式 "YYYY-MM-DD"
        end: 結束日期，格式 "YYYY-MM-DD"（None = 到現在）
        
    Returns:
        DataFrame with columns: open, high, low, close, volume, close_time
        Index: open_time (UTC timezone-aware)
        
    Note:
        - 1m 數據只有最近 7 天
        - 2m/5m/15m/30m 數據只有最近 60 天
        - 1h 數據只有最近 730 天
        - 1d/1wk/1mo 數據有完整歷史
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError(
            "yfinance 未安裝。請執行: pip install yfinance\n"
            "或將 yfinance 加入 requirements.txt"
        )
    
    # 轉換交易對格式
    yf_symbol = convert_symbol(symbol)
    
    # 轉換 interval 格式
    yf_interval = INTERVAL_MAPPING.get(interval, interval)
    
    logger.info(f"📥 yfinance: 下載 {yf_symbol} {yf_interval} ({start} → {end or '現在'})")
    
    # 下載數據
    ticker = yf.Ticker(yf_symbol)
    
    try:
        df = ticker.history(
            start=start,
            end=end,
            interval=yf_interval,
            auto_adjust=False,  # 保持原始 OHLC
        )
    except Exception as e:
        logger.error(f"❌ yfinance 下載失敗: {e}")
        return pd.DataFrame()
    
    if df.empty:
        logger.warning(f"⚠️  yfinance 返回空數據: {yf_symbol}")
        return pd.DataFrame()
    
    # 重命名欄位 (Yahoo 用首字母大寫)
    df = df.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })
    
    # 只保留需要的欄位
    df = df[["open", "high", "low", "close", "volume"]].copy()
    
    # 處理 index (Yahoo Finance 的 index 是 timezone-aware)
    df.index.name = "open_time"
    
    # 確保 timezone 是 UTC
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    
    # 計算 close_time（開盤時間 + interval）
    interval_seconds = _interval_to_seconds(interval)
    df["close_time"] = df.index + pd.Timedelta(seconds=interval_seconds - 1)
    
    # 轉換類型
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    
    logger.info(f"✅ yfinance: 下載完成，共 {len(df)} 筆 ({df.index[0]} → {df.index[-1]})")
    
    return df


def _interval_to_seconds(interval: str) -> int:
    """將 interval 字串轉換為秒數"""
    mapping = {
        "1m": 60,
        "2m": 120,
        "5m": 300,
        "15m": 900,
        "30m": 1800,
        "1h": 3600,
        "2h": 7200,
        "4h": 14400,
        "1d": 86400,
        "1wk": 604800,
        "1mo": 2592000,
    }
    return mapping.get(interval, 3600)


def get_yfinance_data_range(symbol: str) -> tuple[str, str]:
    """
    獲取 Yahoo Finance 上該交易對的可用數據範圍
    
    Returns:
        (earliest_date, latest_date) 格式 "YYYY-MM-DD"
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance 未安裝。請執行: pip install yfinance")
    
    yf_symbol = convert_symbol(symbol)
    ticker = yf.Ticker(yf_symbol)
    
    # 用 1d interval 獲取最長歷史
    df = ticker.history(period="max", interval="1d")
    
    if df.empty:
        return None, None
    
    earliest = df.index[0].strftime("%Y-%m-%d")
    latest = df.index[-1].strftime("%Y-%m-%d")
    
    return earliest, latest


# 支援的交易對列表
SUPPORTED_CRYPTOS = list(SYMBOL_MAPPING.keys())
