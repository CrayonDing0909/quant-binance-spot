#!/usr/bin/env python3
"""
測試 Binance Futures Broker

功能：
    1. 連線測試（無需 API Key）
    2. 帳戶查詢（需要 API Key）
    3. DRY-RUN 下單測試
    4. 目標倉位執行測試

使用方式：
    # 僅連線測試（不需要 API Key）
    python scripts/test_futures_broker.py --connection-only

    # 完整測試（需要 API Key，但不會真的下單）
    python scripts/test_futures_broker.py

    # 真實下單測試（危險！會實際交易）
    python scripts/test_futures_broker.py --live
"""
import sys
from pathlib import Path

# 添加專案根目錄到 path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import argparse
from qtrade.data.binance_futures_client import BinanceFuturesHTTP
from qtrade.live.binance_futures_broker import BinanceFuturesBroker


def test_connection():
    """測試 API 連線（不需要 API Key）"""
    print("\n" + "=" * 60)
    print("  1. 測試 Futures API 連線")
    print("=" * 60)
    
    http = BinanceFuturesHTTP()
    
    # 伺服器時間
    try:
        data = http.get("/fapi/v1/time")
        print(f"✅ 伺服器時間: {data}")
    except Exception as e:
        print(f"❌ 連線失敗: {e}")
        return False
    
    # 標記價格
    try:
        data = http.get_mark_price("BTCUSDT")
        print(f"✅ BTCUSDT 標記價格: ${float(data['markPrice']):,.2f}")
        print(f"   資金費率: {float(data['lastFundingRate']) * 100:.4f}%")
    except Exception as e:
        print(f"⚠️  獲取標記價格失敗: {e}")
    
    # K 線
    try:
        klines = http.get_klines("BTCUSDT", "1h", limit=5)
        print(f"✅ 獲取 K 線: {len(klines)} 根")
    except Exception as e:
        print(f"⚠️  獲取 K 線失敗: {e}")
    
    return True


def test_broker_dry_run():
    """測試 Broker（DRY-RUN 模式）"""
    print("\n" + "=" * 60)
    print("  2. 測試 Broker（DRY-RUN 模式）")
    print("=" * 60)
    
    try:
        broker = BinanceFuturesBroker(dry_run=True, leverage=10)
    except RuntimeError as e:
        print(f"⚠️  Broker 初始化失敗（需要 API Key）: {e}")
        print("   設置環境變數 BINANCE_API_KEY 和 BINANCE_API_SECRET 後重試")
        return False
    
    symbol = "BTCUSDT"
    
    # 連線檢查
    print("\n📡 連線檢查:")
    result = broker.check_connection([symbol])
    
    # 獲取價格
    price = broker.get_price(symbol)
    print(f"\n📊 {symbol} 當前價格: ${price:,.2f}")
    
    # 測試做多
    print("\n🟢 測試做多:")
    order = broker.market_long(symbol, usdt_value=100, reason="test_long")
    if order:
        print(f"   訂單: {order}")
    
    # 測試做空
    print("\n🔴 測試做空:")
    order = broker.market_short(symbol, usdt_value=100, reason="test_short")
    if order:
        print(f"   訂單: {order}")
    
    # 測試目標倉位
    print("\n🎯 測試目標倉位:")
    order = broker.execute_target_position(symbol, target_pct=0.5, reason="test_target")
    if order:
        print(f"   訂單: {order}")
    
    return True


def test_broker_live():
    """測試 Broker（真實模式，危險！）"""
    print("\n" + "=" * 60)
    print("  ⚠️  真實交易測試（會實際下單！）")
    print("=" * 60)
    
    confirm = input("確定要進行真實交易測試嗎？輸入 'YES' 繼續: ")
    if confirm != "YES":
        print("已取消")
        return False
    
    try:
        broker = BinanceFuturesBroker(dry_run=False, leverage=5)
    except RuntimeError as e:
        print(f"❌ Broker 初始化失敗: {e}")
        return False
    
    symbol = "BTCUSDT"
    
    # 連線檢查
    print("\n📡 連線檢查:")
    result = broker.check_connection([symbol])
    
    # 查詢餘額
    balance = broker.get_balance("USDT")
    equity = broker.get_equity()
    print(f"\n💰 可用餘額: ${balance:,.2f}")
    print(f"   總權益: ${equity:,.2f}")
    
    # 查詢持倉
    pos = broker.get_position(symbol)
    if pos:
        print(f"\n📊 當前持倉: {pos}")
    else:
        print(f"\n📊 {symbol} 無持倉")
    
    # 小額測試（5 USDT）
    test_amount = 5.0
    print(f"\n🧪 小額測試（{test_amount} USDT）:")
    
    price = broker.get_price(symbol)
    qty = test_amount / price
    
    # 開多
    print(f"   開多 {qty:.6f} {symbol}...")
    order = broker.market_long(symbol, usdt_value=test_amount, reason="live_test")
    if order:
        print(f"   ✅ 開多成功: {order.order_id}")
    
        # 立即平倉
        import time
        time.sleep(1)
        print(f"   平倉...")
        close = broker.market_close(symbol, reason="close_test")
        if close:
            print(f"   ✅ 平倉成功: {close.order_id}, PnL: {close.pnl:+.4f}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="測試 Binance Futures Broker")
    parser.add_argument("--connection-only", action="store_true", help="僅測試連線（不需要 API Key）")
    parser.add_argument("--live", action="store_true", help="真實交易測試（危險！）")
    args = parser.parse_args()
    
    print("🚀 Binance Futures Broker 測試")
    print("=" * 60)
    
    # 1. 連線測試
    if not test_connection():
        print("\n❌ 連線測試失敗")
        return 1
    
    if args.connection_only:
        print("\n✅ 連線測試完成")
        return 0
    
    # 2. DRY-RUN 測試
    if not args.live:
        if not test_broker_dry_run():
            print("\n⚠️  DRY-RUN 測試未完成（可能需要 API Key）")
        else:
            print("\n✅ DRY-RUN 測試完成")
    
    # 3. 真實測試
    if args.live:
        if not test_broker_live():
            print("\n⚠️  真實測試未完成")
        else:
            print("\n✅ 真實測試完成")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
