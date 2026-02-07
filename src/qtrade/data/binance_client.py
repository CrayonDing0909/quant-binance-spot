from __future__ import annotations
import os
import time
import hmac
import hashlib
import requests
from urllib.parse import urlencode


# Binance API 端点列表（按优先级排序）
# api.binance.com 会封锁美国 IP (HTTP 451)
# data-api.binance.vision 是公开数据 API，不受地区限制
# api1~4 是镜像端点
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

    自动处理地区封锁：如果主端点返回 451，自动切换到备用端点。
    也可通过环境变量 BINANCE_BASE_URL 手动指定。
    """

    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or os.getenv("BINANCE_BASE_URL", "https://api.binance.com")).rstrip("/")
        self.api_key = os.getenv("BINANCE_API_KEY")
        self.api_secret = os.getenv("BINANCE_API_SECRET")
        self._fallback_tested = False

    def _headers(self) -> dict:
        h = {}
        if self.api_key:
            h["X-MBX-APIKEY"] = self.api_key
        return h

    def get(self, path: str, params: dict | None = None) -> dict | list:
        url = f"{self.base_url}{path}"
        try:
            r = requests.get(url, params=params, headers=self._headers(), timeout=30)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as e:
            # HTTP 451 = 地区封锁，自动尝试备用端点
            if e.response is not None and e.response.status_code == 451 and not self._fallback_tested:
                return self._try_fallback_endpoints(path, params)
            raise

    def _try_fallback_endpoints(self, path: str, params: dict | None) -> dict | list:
        """尝试所有备用端点，找到能用的就切换过去"""
        self._fallback_tested = True
        for endpoint in BINANCE_ENDPOINTS:
            if endpoint.rstrip("/") == self.base_url:
                continue  # 跳过已失败的
            url = f"{endpoint.rstrip('/')}{path}"
            try:
                r = requests.get(url, params=params, headers=self._headers(), timeout=15)
                if r.status_code == 200:
                    self.base_url = endpoint.rstrip("/")
                    print(f"✅ 自动切换 Binance API → {endpoint}")
                    return r.json()
            except Exception:
                continue
        raise RuntimeError(
            f"❌ 所有 Binance API 端点均不可用（可能是 IP 地区限制）\n"
            f"   尝试在环境变量中设置 BINANCE_BASE_URL=https://data-api.binance.vision"
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
        params = self._sign_params(params)
        url = f"{self.base_url}{path}"
        r = requests.get(url, params=params, headers=self._headers(), timeout=30)
        r.raise_for_status()
        return r.json()

    def signed_post(self, path: str, params: dict) -> dict:
        params = self._sign_params(params)
        url = f"{self.base_url}{path}"
        r = requests.post(url, params=params, headers=self._headers(), timeout=30)
        r.raise_for_status()
        return r.json()
