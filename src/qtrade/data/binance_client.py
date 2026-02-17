from __future__ import annotations
import logging
import os
import time
import hmac
import hashlib
import requests
from urllib.parse import urlencode

logger = logging.getLogger(__name__)

# ── 密鑰管理 ──────────────────────────────────
KEYRING_SERVICE = "spot_bot"  # keyring 服務名稱


def _get_secret(key: str) -> str | None:
    """
    取得敏感憑證
    
    優先順序：
    1. keyring（系統安全存儲）
    2. 環境變數
    
    Args:
        key: 憑證名稱，例如 "BINANCE_API_KEY"
        
    Returns:
        憑證值或 None
    """
    # 優先從 keyring 讀取
    try:
        import keyring
        val = keyring.get_password(KEYRING_SERVICE, key)
        if val:
            logger.debug(f"🔐 {key} 從 keyring 讀取")
            return val
    except ImportError:
        pass  # keyring 未安裝，使用環境變數
    except Exception as e:
        logger.warning(f"⚠️  keyring 讀取 {key} 失敗: {e}")
    
    # Fallback 到環境變數
    val = os.getenv(key)
    if val:
        logger.debug(f"📄 {key} 從環境變數讀取")
    return val

# ── 重試配置 ──────────────────────────────────
MAX_RETRIES = 3                 # 最多重試 3 次
RETRY_DELAYS = [2, 5, 10]      # 指數退避延遲（秒）
RETRYABLE_HTTP_CODES = {500, 502, 503, 504, 429}   # 可重試的 HTTP 狀態碼

# Binance API 端點列表（按優先級排序）
# api.binance.com 會封鎖美國 IP (HTTP 451)
# data-api.binance.vision 是公開數據 API，不受地區限制
# api1~4 是鏡像端點
BINANCE_ENDPOINTS = [
    "https://api.binance.com",
    "https://data-api.binance.vision",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://api4.binance.com",
]


class BinanceHTTP:
    """
    Minimal Binance Spot REST client.
    Public endpoints (klines) don't require key.
    Signed endpoints are for live later.

    特性：
    - 自動重試：網路錯誤 / 5xx / 429 自動指數退避重試（最多 3 次）
    - 自動切換：HTTP 451 地區封鎖自動切換備用端點
    - 也可透過環境變數 BINANCE_BASE_URL 手動指定

    子類可覆寫 _FALLBACK_ENDPOINTS 提供不同的備用端點列表。
    """

    # 子類可覆寫此列表（例如 BinanceFuturesHTTP 覆寫為 fapi 端點）
    _FALLBACK_ENDPOINTS = BINANCE_ENDPOINTS

    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or os.getenv("BINANCE_BASE_URL", "https://api.binance.com")).rstrip("/")
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
        # 網路層錯誤：連接超時、DNS 失敗等
        if isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
            return True
        # HTTP 服務端錯誤或限流
        if isinstance(exc, requests.exceptions.HTTPError) and exc.response is not None:
            return exc.response.status_code in RETRYABLE_HTTP_CODES
        return False

    def get(self, path: str, params: dict | None = None) -> dict | list:
        last_exc: Exception | None = None

        for attempt in range(MAX_RETRIES + 1):
            url = f"{self.base_url}{path}"
            try:
                r = requests.get(url, params=params, headers=self._headers(), timeout=30)
                r.raise_for_status()
                return r.json()
            except requests.exceptions.HTTPError as e:
                # HTTP 451 = 地區封鎖 → 切換端點（不重試）
                if e.response is not None and e.response.status_code == 451 and not self._fallback_tested:
                    return self._try_fallback_endpoints(path, params)
                # 可重試的 HTTP 錯誤
                if self._should_retry(e) and attempt < MAX_RETRIES:
                    delay = RETRY_DELAYS[attempt]
                    logger.warning(
                        f"⚠️  Binance API {e.response.status_code} — "
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
                        f"⚠️  Binance API 網路錯誤: {type(e).__name__} — "
                        f"重試 {attempt + 1}/{MAX_RETRIES}（等待 {delay}s）"
                    )
                    time.sleep(delay)
                    last_exc = e
                    continue
                raise

        # 理論上不會到這裡，但保險起見
        raise last_exc or RuntimeError("Unexpected retry exhaustion")

    def _try_fallback_endpoints(self, path: str, params: dict | None) -> dict | list:
        """嘗試所有備用端點，找到能用的就切換過去"""
        self._fallback_tested = True
        for endpoint in self._FALLBACK_ENDPOINTS:
            if endpoint.rstrip("/") == self.base_url:
                continue  # 跳過已失敗的
            url = f"{endpoint.rstrip('/')}{path}"
            try:
                r = requests.get(url, params=params, headers=self._headers(), timeout=15)
                if r.status_code == 200:
                    self.base_url = endpoint.rstrip("/")
                    logger.info(f"✅ 自動切換 Binance API → {endpoint}")
                    return r.json()
            except Exception:
                continue
        raise RuntimeError(
            f"❌ 所有 Binance API 端點均不可用（可能是 IP 地區限制）\n"
            f"   嘗試在環境變數中設置 BINANCE_BASE_URL=https://data-api.binance.vision"
        )

    def _sign_params(self, params: dict) -> dict:
        if not self.api_secret:
            raise RuntimeError("Missing BINANCE_API_SECRET")
        params = dict(params)
        params["timestamp"] = int(time.time() * 1000)
        query = urlencode(params)
        sig = hmac.new(self.api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()
        params["signature"] = sig
        return params

    def signed_get(self, path: str, params: dict) -> dict | list:
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
        """簽名 DELETE 請求（用於取消訂單）"""
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
