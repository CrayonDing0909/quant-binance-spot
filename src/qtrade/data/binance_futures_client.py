"""
Binance Futures HTTP Client

專門用於 USDT-M 永續合約的 REST API 客戶端。
繼承自 BinanceHTTP，覆寫 base_url 和備用端點。

主要差異：
    - Base URL: https://fapi.binance.com（而非 api.binance.com）
    - 端點前綴: /fapi/v1/ 或 /fapi/v2/
    - 備用端點: fapi1~4.binance.com
"""
from __future__ import annotations
import logging
import os

from .binance_client import BinanceHTTP

logger = logging.getLogger(__name__)

# Binance Futures API 備用端點
BINANCE_FUTURES_ENDPOINTS = [
    "https://fapi.binance.com",
    "https://fapi1.binance.com",
    "https://fapi2.binance.com",
    "https://fapi3.binance.com",
    "https://fapi4.binance.com",
]


class BinanceFuturesHTTP(BinanceHTTP):
    """
    Binance USDT-M Futures REST client

    繼承自 BinanceHTTP，共享重試、簽名、端點切換邏輯。
    只覆寫 base_url 和備用端點列表。
    """

    # 覆寫備用端點列表
    _FALLBACK_ENDPOINTS = BINANCE_FUTURES_ENDPOINTS

    def __init__(self, base_url: str | None = None):
        default_url = os.getenv("BINANCE_FUTURES_BASE_URL", "https://fapi.binance.com")
        super().__init__(base_url=base_url or default_url)

    # ── Futures 專用便利方法 ──────────────────────────

    def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> list:
        """
        獲取 K 線數據

        Args:
            symbol: 交易對, e.g. "BTCUSDT"
            interval: K 線週期, e.g. "1h", "4h", "1d"
            limit: 數量（最大 1500）
            start_time: 開始時間（毫秒）
            end_time: 結束時間（毫秒）
        """
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1500),
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        return self.get("/fapi/v1/klines", params)

    def get_mark_price(self, symbol: str) -> dict:
        """獲取標記價格和資金費率"""
        return self.get("/fapi/v1/premiumIndex", {"symbol": symbol})

    def get_funding_rate(self, symbol: str, limit: int = 100) -> list:
        """獲取歷史資金費率"""
        return self.get("/fapi/v1/fundingRate", {"symbol": symbol, "limit": limit})

    def get_exchange_info(self) -> dict:
        """獲取交易所信息"""
        return self.get("/fapi/v1/exchangeInfo")
