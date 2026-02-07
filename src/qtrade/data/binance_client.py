from __future__ import annotations
import os
import time
import hmac
import hashlib
import requests
from urllib.parse import urlencode


class BinanceHTTP:
    """
    Minimal Binance Spot REST client.
    Public endpoints (klines) don't require key.
    Signed endpoints are for live later.
    """

    def __init__(self, base_url: str = "https://api.binance.com"):
        self.base_url = base_url.rstrip("/")
        self.api_key = os.getenv("BINANCE_API_KEY")
        self.api_secret = os.getenv("BINANCE_API_SECRET")

    def _headers(self) -> dict:
        h = {}
        if self.api_key:
            h["X-MBX-APIKEY"] = self.api_key
        return h

    def get(self, path: str, params: dict | None = None) -> dict | list:
        url = f"{self.base_url}{path}"
        r = requests.get(url, params=params, headers=self._headers(), timeout=30)
        r.raise_for_status()
        return r.json()

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
