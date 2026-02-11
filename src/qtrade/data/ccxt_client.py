"""
CCXT 多交易所數據源 - 統一 API 訪問 100+ 交易所

支援的交易所歷史數據起始時間 (BTC/USDT 或 BTC/USD):
- Coinbase Pro: 2015-01
- Kraken: 2013-10 (非常長的歷史！)
- Bitfinex: 2013-04
- Bitstamp: 2011-08 (最早的交易所之一)
- Binance: 2017-08
- OKX: 2017-08

使用方式:
    from qtrade.data.ccxt_client import fetch_ccxt_klines, list_available_exchanges
    
    # 查看可用交易所
    exchanges = list_available_exchanges()
    
    # 從 Kraken 下載 BTC 數據
    df = fetch_ccxt_klines("BTC/USD", "1h", "2015-01-01", exchange="kraken")
"""
from __future__ import annotations

import pandas as pd
from datetime import datetime, timezone
from typing import Optional
import time
import logging

logger = logging.getLogger(__name__)

# 推薦的交易所及其 BTC 歷史數據起始時間
EXCHANGE_HISTORY = {
    "bitstamp": {"btc_start": "2011-08-18", "note": "最早的交易所之一"},
    "kraken": {"btc_start": "2013-10-06", "note": "歐洲主要交易所，數據完整"},
    "bitfinex": {"btc_start": "2013-04-01", "note": "專業交易所"},
    "coinbasepro": {"btc_start": "2015-01-26", "note": "美國合規交易所，改名 Coinbase Exchange"},
    "binance": {"btc_start": "2017-08-17", "note": "全球最大"},
    "okx": {"btc_start": "2017-08-17", "note": "前 OKEx"},
    "huobi": {"btc_start": "2017-09-01", "note": "中國起家"},
    "kucoin": {"btc_start": "2017-09-15", "note": "小幣種豐富"},
    "bybit": {"btc_start": "2019-11-01", "note": "衍生品為主"},
}

# Binance 風格 symbol 轉 CCXT 風格
SYMBOL_MAPPING = {
    "BTCUSDT": "BTC/USDT",
    "ETHUSDT": "ETH/USDT",
    "BNBUSDT": "BNB/USDT",
    "SOLUSDT": "SOL/USDT",
    "XRPUSDT": "XRP/USDT",
    "ADAUSDT": "ADA/USDT",
    "DOGEUSDT": "DOGE/USDT",
    "DOTUSDT": "DOT/USDT",
    "LTCUSDT": "LTC/USDT",
    "LINKUSDT": "LINK/USDT",
    # USD pairs (for exchanges without USDT)
    "BTCUSD": "BTC/USD",
    "ETHUSD": "ETH/USD",
}

# 早期交易所通常只有 BTC/USD，沒有 USDT
USDT_TO_USD_EXCHANGES = {"bitstamp", "kraken", "coinbasepro", "bitfinex"}


def convert_symbol(binance_symbol: str, exchange: str) -> str:
    """
    將 Binance 風格的交易對轉換為 CCXT 格式
    
    對於早期交易所，自動將 USDT 轉為 USD
    """
    # 直接映射
    if binance_symbol in SYMBOL_MAPPING:
        ccxt_symbol = SYMBOL_MAPPING[binance_symbol]
    elif "/" in binance_symbol:
        # 已經是 CCXT 格式
        ccxt_symbol = binance_symbol
    else:
        # 嘗試自動轉換 (XXXUSDT -> XXX/USDT)
        if binance_symbol.endswith("USDT"):
            base = binance_symbol[:-4]
            ccxt_symbol = f"{base}/USDT"
        elif binance_symbol.endswith("USD"):
            base = binance_symbol[:-3]
            ccxt_symbol = f"{base}/USD"
        else:
            ccxt_symbol = binance_symbol
    
    # 對於早期交易所，自動轉換 USDT -> USD
    if exchange in USDT_TO_USD_EXCHANGES and "/USDT" in ccxt_symbol:
        ccxt_symbol = ccxt_symbol.replace("/USDT", "/USD")
        logger.info(f"📝 {exchange} 不支援 USDT，自動轉換為 {ccxt_symbol}")
    
    return ccxt_symbol


def list_available_exchanges() -> dict:
    """
    列出推薦的交易所及其歷史數據資訊
    
    Returns:
        dict: {exchange_id: {btc_start, note}}
    """
    return EXCHANGE_HISTORY.copy()


def _get_exchange(exchange_id: str):
    """獲取 CCXT 交易所實例"""
    try:
        import ccxt
    except ImportError:
        raise ImportError(
            "ccxt 未安裝。請執行: pip install ccxt\n"
            "或將 ccxt 加入 requirements.txt"
        )
    
    exchange_class = getattr(ccxt, exchange_id, None)
    if exchange_class is None:
        raise ValueError(f"交易所 '{exchange_id}' 不存在。可用: {list(EXCHANGE_HISTORY.keys())}")
    
    return exchange_class({
        "enableRateLimit": True,  # 自動處理速率限制
        "timeout": 30000,
    })


def fetch_ccxt_klines(
    symbol: str,
    interval: str,
    start: str,
    end: str | None = None,
    exchange: str = "binance",
    limit_per_request: int = 1000,
) -> pd.DataFrame:
    """
    使用 CCXT 從指定交易所下載 K 線資料
    
    Args:
        symbol: 交易對，支援 Binance 格式 (如 "BTCUSDT") 或 CCXT 格式 (如 "BTC/USDT")
        interval: K 線週期，如 "1h", "4h", "1d"
        start: 開始日期，格式 "YYYY-MM-DD"
        end: 結束日期，格式 "YYYY-MM-DD"（None = 到現在）
        exchange: 交易所 ID，如 "binance", "kraken", "coinbasepro"
        limit_per_request: 每次 API 請求的最大 K 線數量
        
    Returns:
        DataFrame with columns: open, high, low, close, volume, close_time
        Index: open_time (UTC timezone-aware)
    """
    try:
        import ccxt
    except ImportError:
        raise ImportError(
            "ccxt 未安裝。請執行: pip install ccxt\n"
            "或將 ccxt 加入 requirements.txt"
        )
    
    # 獲取交易所實例
    ex = _get_exchange(exchange)
    
    # 轉換 symbol 格式
    ccxt_symbol = convert_symbol(symbol, exchange)
    
    # 轉換時間
    start_ms = int(datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = (
        int(datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
        if end
        else int(datetime.now(timezone.utc).timestamp() * 1000)
    )
    
    # CCXT timeframe 格式（通常與我們的 interval 格式相同）
    timeframe = interval
    
    logger.info(f"📥 CCXT/{exchange}: 下載 {ccxt_symbol} {timeframe} ({start} → {end or '現在'})")
    
    # 逐批下載
    all_ohlcv = []
    current_since = start_ms
    
    while current_since < end_ms:
        try:
            ohlcv = ex.fetch_ohlcv(
                ccxt_symbol,
                timeframe=timeframe,
                since=current_since,
                limit=limit_per_request,
            )
        except Exception as e:
            logger.error(f"❌ CCXT 下載失敗: {e}")
            break
        
        if not ohlcv:
            break
        
        all_ohlcv.extend(ohlcv)
        
        # 更新起始時間（最後一根 K 線的時間 + 1ms）
        last_timestamp = ohlcv[-1][0]
        current_since = last_timestamp + 1
        
        # 如果返回的數據少於請求的數量，說明已經到達末尾
        if len(ohlcv) < limit_per_request:
            break
        
        # 進度提示（每 10000 筆）
        if len(all_ohlcv) % 10000 == 0:
            logger.info(f"  📊 已下載 {len(all_ohlcv)} 筆...")
        
        # 小延遲避免觸發速率限制
        time.sleep(ex.rateLimit / 1000)  # rateLimit 是毫秒
    
    if not all_ohlcv:
        logger.warning(f"⚠️  CCXT/{exchange} 返回空數據: {ccxt_symbol}")
        return pd.DataFrame()
    
    # 轉換為 DataFrame
    df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    
    # 處理時間
    df["open_time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("open_time").sort_index()
    
    # 過濾時間範圍
    end_dt = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc) if end else datetime.now(timezone.utc)
    df = df[df.index < end_dt]
    
    # 去除重複
    df = df[~df.index.duplicated(keep="last")]
    
    # 計算 close_time
    interval_ms = _interval_to_ms(interval)
    df["close_time"] = df.index + pd.Timedelta(milliseconds=interval_ms - 1)
    
    # 只保留需要的欄位
    df = df[["open", "high", "low", "close", "volume", "close_time"]].copy()
    
    # 確保類型
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    
    logger.info(f"✅ CCXT/{exchange}: 下載完成，共 {len(df)} 筆 ({df.index[0]} → {df.index[-1]})")
    
    return df


def _interval_to_ms(interval: str) -> int:
    """將 interval 字串轉換為毫秒"""
    mapping = {
        "1m": 60_000,
        "3m": 180_000,
        "5m": 300_000,
        "15m": 900_000,
        "30m": 1_800_000,
        "1h": 3_600_000,
        "2h": 7_200_000,
        "4h": 14_400_000,
        "6h": 21_600_000,
        "8h": 28_800_000,
        "12h": 43_200_000,
        "1d": 86_400_000,
        "1w": 604_800_000,
    }
    return mapping.get(interval, 3_600_000)


def check_exchange_symbol(exchange: str, symbol: str) -> dict:
    """
    檢查指定交易所是否支援該交易對，並返回可用資訊
    
    Returns:
        dict: {available: bool, symbol: str, message: str}
    """
    try:
        ex = _get_exchange(exchange)
        ex.load_markets()
        
        ccxt_symbol = convert_symbol(symbol, exchange)
        
        if ccxt_symbol in ex.markets:
            market = ex.markets[ccxt_symbol]
            return {
                "available": True,
                "symbol": ccxt_symbol,
                "message": f"✅ {ccxt_symbol} 可用於 {exchange}",
                "info": {
                    "base": market.get("base"),
                    "quote": market.get("quote"),
                    "active": market.get("active"),
                }
            }
        else:
            # 嘗試找類似的
            alternatives = [s for s in ex.markets if symbol.replace("USDT", "") in s]
            return {
                "available": False,
                "symbol": ccxt_symbol,
                "message": f"❌ {ccxt_symbol} 不存在於 {exchange}",
                "alternatives": alternatives[:5],
            }
    except Exception as e:
        return {
            "available": False,
            "symbol": symbol,
            "message": f"❌ 檢查失敗: {e}",
        }


def get_earliest_data_timestamp(exchange: str, symbol: str, interval: str = "1d") -> Optional[str]:
    """
    獲取該交易對在指定交易所的最早可用數據時間
    
    Returns:
        最早日期字串 "YYYY-MM-DD" 或 None
    """
    try:
        # 嘗試從很早的時間開始請求，看返回的第一筆數據
        ex = _get_exchange(exchange)
        ccxt_symbol = convert_symbol(symbol, exchange)
        
        # 從 2010 年開始（比特幣誕生後不久）
        since = int(datetime(2010, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        
        ohlcv = ex.fetch_ohlcv(ccxt_symbol, timeframe=interval, since=since, limit=1)
        
        if ohlcv:
            earliest_ts = ohlcv[0][0]
            earliest_dt = datetime.fromtimestamp(earliest_ts / 1000, tz=timezone.utc)
            return earliest_dt.strftime("%Y-%m-%d")
        
        return None
    except Exception as e:
        logger.error(f"無法獲取 {exchange} {symbol} 的最早數據: {e}")
        return None
