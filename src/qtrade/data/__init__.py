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

# 衍生品數據模組 (lazy import)
def _lazy_import_long_short_ratio():
    from .long_short_ratio import download_lsr, load_lsr, align_lsr_to_klines, compute_lsr_coverage
    return download_lsr, load_lsr, align_lsr_to_klines, compute_lsr_coverage

def _lazy_import_taker_volume():
    from .taker_volume import download_taker_volume, load_taker_volume, compute_cvd, load_cvd, align_taker_to_klines
    return download_taker_volume, load_taker_volume, compute_cvd, load_cvd, align_taker_to_klines

def _lazy_import_liquidation():
    from .liquidation import load_liquidation, align_liquidation_to_klines
    return load_liquidation, align_liquidation_to_klines

def _lazy_import_onchain():
    from .onchain import load_onchain, save_onchain, align_onchain_to_klines, compute_onchain_coverage
    return load_onchain, save_onchain, align_onchain_to_klines, compute_onchain_coverage

def _lazy_import_multi_tf_loader():
    from .multi_tf_loader import MultiTFLoader
    return MultiTFLoader

def _lazy_import_order_book():
    from .order_book import OrderBookSnapshot, OrderBookCache, compute_imbalance, compute_depth_profile
    return OrderBookSnapshot, OrderBookCache, compute_imbalance, compute_depth_profile

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
    # Derivatives data (lazy import helpers)
    "_lazy_import_long_short_ratio",
    "_lazy_import_taker_volume",
    "_lazy_import_liquidation",
    "_lazy_import_onchain",
    # Multi-TF loader
    "_lazy_import_multi_tf_loader",
    # Order book depth
    "_lazy_import_order_book",
]
