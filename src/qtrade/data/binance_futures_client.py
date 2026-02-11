"""
Binance Futures HTTP Client

專門用於 USDT-M 永續合約的 REST API 客戶端。
繼承自 BinanceHTTP，覆寫端點為 fapi.binance.com

主要差異：
    - Base URL: https://fapi.binance.com（而非 api.binance.com）
    - 端點前綴: /fapi/v1/ 或 /fapi/v2/
    - 備用端點: fapi1~4.binance.com
"""
from __future__ import annotations
import logging
import os
import time
import hmac
import hashlib
import requests
from urllib.parse import urlencode

logger = logging.getLogger(__name__)

# ── 密鑰管理（複用現有邏輯）──────────────────────────────
KEYRING_SERVICE = "spot_bot"


def _get_secret(key: str) -> str | None:
    """取得敏感憑證，優先 keyring → 環境變數"""
    try:
        import keyring
        val = keyring.get_password(KEYRING_SERVICE, key)
        if val:
            return val
    except ImportError:
        pass
    except Exception:
        pass
    return os.getenv(key)


# ── 重試配置 ──────────────────────────────────
MAX_RETRIES = 3
RETRY_DELAYS = [2, 5, 10]
RETRYABLE_HTTP_CODES = {500, 502, 503, 504, 429}

# Binance Futures API 端點
BINANCE_FUTURES_ENDPOINTS = [
    "https://fapi.binance.com",
    "https://fapi1.binance.com",
    "https://fapi2.binance.com",
    "https://fapi3.binance.com",
    "https://fapi4.binance.com",
]


class BinanceFuturesHTTP:
    """
    Binance USDT-M Futures REST client
    
    特性：
    - 自動重試：網路錯誤 / 5xx / 429 自動指數退避重試
    - 自動切換：連線失敗自動切換備用端點
    - 支援簽名請求（交易、查餘額）
    """

    def __init__(self, base_url: str | None = None):
        self.base_url = (
            base_url or 
            os.getenv("BINANCE_FUTURES_BASE_URL", "https://fapi.binance.com")
        ).rstrip("/")
        self.api_key = _get_secret("BINANCE_API_KEY")
        self.api_secret = _get_secret("BINANCE_API_SECRET")
        self._fallback_tested = False

    def _headers(self) -> dict:
        h = {}
        if self.api_key:
            h["X-MBX-APIKEY"] = self.api_key
        return h

    @staticmethod
    def _should_retry(exc: Exception) -> bool:
        """判斷異常是否值得重試"""
        if isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
            return True
        if isinstance(exc, requests.exceptions.HTTPError) and exc.response is not None:
            return exc.response.status_code in RETRYABLE_HTTP_CODES
        return False

    def get(self, path: str, params: dict | None = None) -> dict | list:
        """公開 GET 請求"""
        last_exc: Exception | None = None

        for attempt in range(MAX_RETRIES + 1):
            url = f"{self.base_url}{path}"
            try:
                r = requests.get(url, params=params, headers=self._headers(), timeout=30)
                r.raise_for_status()
                return r.json()
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 451 and not self._fallback_tested:
                    return self._try_fallback_endpoints(path, params)
                if self._should_retry(e) and attempt < MAX_RETRIES:
                    delay = RETRY_DELAYS[attempt]
                    logger.warning(
                        f"⚠️  Futures API {e.response.status_code} — "
                        f"重試 {attempt + 1}/{MAX_RETRIES}（等待 {delay}s）"
                    )
                    time.sleep(delay)
                    last_exc = e
                    continue
                raise
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                if attempt < MAX_RETRIES:
                    delay = RETRY_DELAYS[attempt]
                    logger.warning(
                        f"⚠️  Futures API 網路錯誤: {type(e).__name__} — "
                        f"重試 {attempt + 1}/{MAX_RETRIES}（等待 {delay}s）"
                    )
                    time.sleep(delay)
                    last_exc = e
                    continue
                raise

        raise last_exc or RuntimeError("Unexpected retry exhaustion")

    def _try_fallback_endpoints(self, path: str, params: dict | None) -> dict | list:
        """嘗試所有備用端點"""
        self._fallback_tested = True
        for endpoint in BINANCE_FUTURES_ENDPOINTS:
            if endpoint.rstrip("/") == self.base_url:
                continue
            url = f"{endpoint.rstrip('/')}{path}"
            try:
                r = requests.get(url, params=params, headers=self._headers(), timeout=15)
                if r.status_code == 200:
                    self.base_url = endpoint.rstrip("/")
                    logger.info(f"✅ 自動切換 Futures API → {endpoint}")
                    return r.json()
            except Exception:
                continue
        raise RuntimeError(
            f"❌ 所有 Binance Futures API 端點均不可用\n"
            f"   嘗試在環境變數中設置 BINANCE_FUTURES_BASE_URL"
        )

    def _sign_params(self, params: dict) -> dict:
        """簽名參數"""
        if not self.api_secret:
            raise RuntimeError("Missing BINANCE_API_SECRET")
        params = dict(params)
        params["timestamp"] = int(time.time() * 1000)
        query = urlencode(params)
        sig = hmac.new(self.api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()
        params["signature"] = sig
        return params

    def signed_get(self, path: str, params: dict) -> dict | list:
        """簽名 GET 請求"""
        last_exc: Exception | None = None

        for attempt in range(MAX_RETRIES + 1):
            signed = self._sign_params(params)
            url = f"{self.base_url}{path}"
            try:
                r = requests.get(url, params=signed, headers=self._headers(), timeout=30)
                r.raise_for_status()
                return r.json()
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout,
                    requests.exceptions.HTTPError) as e:
                if self._should_retry(e) and attempt < MAX_RETRIES:
                    delay = RETRY_DELAYS[attempt]
                    logger.warning(f"⚠️  signed_get 重試 {attempt + 1}/{MAX_RETRIES}（等待 {delay}s）")
                    time.sleep(delay)
                    last_exc = e
                    continue
                raise

        raise last_exc or RuntimeError("Unexpected retry exhaustion")

    def signed_post(self, path: str, params: dict) -> dict:
        """簽名 POST 請求"""
        last_exc: Exception | None = None

        for attempt in range(MAX_RETRIES + 1):
            signed = self._sign_params(params)
            url = f"{self.base_url}{path}"
            try:
                r = requests.post(url, params=signed, headers=self._headers(), timeout=30)
                r.raise_for_status()
                return r.json()
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout,
                    requests.exceptions.HTTPError) as e:
                if self._should_retry(e) and attempt < MAX_RETRIES:
                    delay = RETRY_DELAYS[attempt]
                    logger.warning(f"⚠️  signed_post 重試 {attempt + 1}/{MAX_RETRIES}（等待 {delay}s）")
                    time.sleep(delay)
                    last_exc = e
                    continue
                raise

        raise last_exc or RuntimeError("Unexpected retry exhaustion")

    def signed_delete(self, path: str, params: dict) -> dict:
        """簽名 DELETE 請求"""
        last_exc: Exception | None = None

        for attempt in range(MAX_RETRIES + 1):
            signed = self._sign_params(params)
            url = f"{self.base_url}{path}"
            try:
                r = requests.delete(url, params=signed, headers=self._headers(), timeout=30)
                r.raise_for_status()
                return r.json()
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout,
                    requests.exceptions.HTTPError) as e:
                if self._should_retry(e) and attempt < MAX_RETRIES:
                    delay = RETRY_DELAYS[attempt]
                    logger.warning(f"⚠️  signed_delete 重試 {attempt + 1}/{MAX_RETRIES}（等待 {delay}s）")
                    time.sleep(delay)
                    last_exc = e
                    continue
                raise

        raise last_exc or RuntimeError("Unexpected retry exhaustion")

    # ── 便利方法 ──────────────────────────────────────────

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
