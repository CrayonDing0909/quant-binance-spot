from __future__ import annotations
from .storage import save_klines, load_klines
from .klines import fetch_klines
from .binance_client import BinanceHTTP
from .binance_futures_client import BinanceFuturesHTTP
from .quality import (
    DataQualityChecker,
    DataQualityReport,
    DataQualityIssue,
    validate_data_quality,
    clean_data,
)

# 多數據源支援 (lazy import to avoid import errors if dependencies not installed)
def _lazy_import_yfinance():
    from .yfinance_client import fetch_yfinance_klines, get_yfinance_data_range
    return fetch_yfinance_klines, get_yfinance_data_range

def _lazy_import_ccxt():
    from .ccxt_client import fetch_ccxt_klines, list_available_exchanges
    return fetch_ccxt_klines, list_available_exchanges

def _lazy_import_binance_vision():
    from .binance_vision import download_binance_vision_klines
    return download_binance_vision_klines

__all__ = [
    # Core
    "save_klines",
    "load_klines",
    "fetch_klines",
    "BinanceHTTP",
    "BinanceFuturesHTTP",
    # Quality
    "DataQualityChecker",
    "DataQualityReport",
    "DataQualityIssue",
    "validate_data_quality",
    "clean_data",
    # Multi-source (lazy import helpers)
    "_lazy_import_yfinance",
    "_lazy_import_ccxt",
    "_lazy_import_binance_vision",
]
