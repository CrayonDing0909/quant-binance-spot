#!/usr/bin/env python3
"""
最簡單的 Futures API 連線測試
不依賴任何第三方模組，使用內建 urllib
"""
import json
import time
import hmac
import hashlib
import os
import ssl
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

BASE_URL = "https://fapi.binance.com"

# 創建 SSL context（忽略證書驗證，僅用於測試）
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE


def http_get(url: str, params: dict | None = None, headers: dict | None = None) -> dict | list:
    """簡單的 HTTP GET"""
    if params:
        url = f"{url}?{urlencode(params)}"
    req = Request(url)
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    with urlopen(req, timeout=10, context=ssl_context) as resp:
        return json.loads(resp.read().decode())


def test_public_api():
    """測試公開 API"""
    print("\n📡 測試公開 API（不需要 API Key）")
    print("-" * 40)
    
    # 1. 伺服器時間
    try:
        data = http_get(f"{BASE_URL}/fapi/v1/time")
        print(f"✅ 伺服器時間: {data['serverTime']}")
    except Exception as e:
        print(f"❌ 伺服器時間查詢失敗: {e}")
        return False
    
    # 2. 標記價格
    try:
        data = http_get(f"{BASE_URL}/fapi/v1/premiumIndex", {"symbol": "BTCUSDT"})
        price = float(data['markPrice'])
        funding_rate = float(data['lastFundingRate'])
        print(f"✅ BTCUSDT 標記價格: ${price:,.2f}")
        print(f"   資金費率: {funding_rate * 100:.4f}%")
    except Exception as e:
        print(f"❌ 標記價格查詢失敗: {e}")
    
    # 3. K 線
    try:
        klines = http_get(f"{BASE_URL}/fapi/v1/klines", 
                         {"symbol": "BTCUSDT", "interval": "1h", "limit": "5"})
        print(f"✅ K 線數據: {len(klines)} 根")
        if klines:
            latest = klines[-1]
            print(f"   最新: O={float(latest[1]):.2f}, H={float(latest[2]):.2f}, "
                  f"L={float(latest[3]):.2f}, C={float(latest[4]):.2f}")
    except Exception as e:
        print(f"❌ K 線查詢失敗: {e}")
    
    # 4. 交易所信息
    try:
        data = http_get(f"{BASE_URL}/fapi/v1/exchangeInfo")
        symbols = [s['symbol'] for s in data.get('symbols', []) if s['symbol'].endswith('USDT')]
        print(f"✅ 交易所信息: {len(symbols)} 個 USDT 交易對")
    except Exception as e:
        print(f"❌ 交易所信息查詢失敗: {e}")
    
    return True


def test_signed_api():
    """測試簽名 API（需要 API Key）"""
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    
    if not api_key or not api_secret:
        print("\n⚠️  未設置 API Key，跳過簽名 API 測試")
        print("   設置環境變數後重試：")
        print("   export BINANCE_API_KEY=your_key")
        print("   export BINANCE_API_SECRET=your_secret")
        return True
    
    print("\n🔐 測試簽名 API（需要 API Key）")
    print("-" * 40)
    
    def sign_request(params: dict) -> dict:
        params = dict(params)
        params["timestamp"] = int(time.time() * 1000)
        query = urlencode(params)
        sig = hmac.new(api_secret.encode(), query.encode(), hashlib.sha256).hexdigest()
        params["signature"] = sig
        return params
    
    headers = {"X-MBX-APIKEY": api_key}
    
    # 1. 帳戶餘額
    try:
        params = sign_request({})
        data = http_get(f"{BASE_URL}/fapi/v2/balance", params, headers)
        usdt = next((b for b in data if b['asset'] == 'USDT'), None)
        if usdt:
            print(f"✅ USDT 餘額: ${float(usdt['balance']):,.2f}")
            print(f"   可用: ${float(usdt['availableBalance']):,.2f}")
    except Exception as e:
        print(f"❌ 餘額查詢失敗: {e}")
        return False
    
    # 2. 帳戶資訊
    try:
        params = sign_request({})
        data = http_get(f"{BASE_URL}/fapi/v2/account", params, headers)
        print(f"✅ 帳戶資訊:")
        print(f"   總權益: ${float(data.get('totalWalletBalance', 0)):,.2f}")
        print(f"   未實現盈虧: ${float(data.get('totalUnrealizedProfit', 0)):+,.2f}")
        print(f"   可交易: {data.get('canTrade', False)}")
    except Exception as e:
        print(f"❌ 帳戶資訊查詢失敗: {e}")
    
    # 3. 持倉
    try:
        params = sign_request({})
        data = http_get(f"{BASE_URL}/fapi/v2/positionRisk", params, headers)
        positions = [p for p in data if float(p['positionAmt']) != 0]
        if positions:
            print(f"✅ 當前持倉: {len(positions)} 個")
            for p in positions:
                qty = float(p['positionAmt'])
                side = "LONG" if qty > 0 else "SHORT"
                print(f"   {p['symbol']} [{side}]: {abs(qty):.4f} @ {float(p['entryPrice']):.2f}")
        else:
            print(f"✅ 當前無持倉")
    except Exception as e:
        print(f"❌ 持倉查詢失敗: {e}")
    
    return True


def main():
    print("=" * 60)
    print("  Binance Futures API 連線測試")
    print("=" * 60)
    
    # 公開 API
    if not test_public_api():
        print("\n❌ 公開 API 測試失敗，請檢查網路連線")
        return 1
    
    # 簽名 API
    test_signed_api()
    
    print("\n" + "=" * 60)
    print("  測試完成")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    exit(main())
