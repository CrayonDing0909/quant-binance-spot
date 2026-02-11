#!/usr/bin/env python3
"""
測試合約風險管理模組

功能：
    1. 資金費率查詢（不需要 API Key）
    2. 強平價格計算（模擬）
    3. 風險報告生成（需要 API Key 查詢持倉）

使用方式：
    # 僅查詢資金費率（不需要 API Key）
    python scripts/test_futures_risk.py --funding-only

    # 完整測試（需要 API Key）
    python scripts/test_futures_risk.py
"""
import sys
from pathlib import Path

# 添加專案根目錄到 path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import argparse
from dataclasses import dataclass


def test_funding_rate():
    """測試資金費率查詢（不需要 API Key）"""
    print("\n" + "=" * 60)
    print("  資金費率查詢測試")
    print("=" * 60)
    
    from qtrade.data.binance_futures_client import BinanceFuturesHTTP
    
    http = BinanceFuturesHTTP()
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    
    print("\n📊 當前資金費率:")
    print("-" * 50)
    print(f"{'幣種':<12} {'費率':>10} {'年化':>10} {'下次結算':<20}")
    print("-" * 50)
    
    for symbol in symbols:
        try:
            data = http.get_mark_price(symbol)
            rate = float(data['lastFundingRate'])
            annualized = rate * 1095  # 每 8 小時，一年 1095 次
            next_time = int(data['nextFundingTime']) / 1000
            
            from datetime import datetime, timezone
            next_dt = datetime.fromtimestamp(next_time, tz=timezone.utc)
            
            print(f"{symbol:<12} {rate*100:>9.4f}% {annualized*100:>9.2f}% {next_dt.strftime('%Y-%m-%d %H:%M')}")
        except Exception as e:
            print(f"{symbol:<12} ❌ 錯誤: {e}")
    
    print("-" * 50)
    
    # 獲取 BTCUSDT 歷史費率
    print("\n📈 BTCUSDT 歷史資金費率（最近 10 期）:")
    try:
        history = http.get_funding_rate("BTCUSDT", limit=10)
        for h in history:
            rate = float(h['fundingRate'])
            time = int(h['fundingTime']) / 1000
            from datetime import datetime, timezone
            dt = datetime.fromtimestamp(time, tz=timezone.utc)
            print(f"   {dt.strftime('%Y-%m-%d %H:%M')} : {rate*100:>8.4f}%")
    except Exception as e:
        print(f"   ❌ 錯誤: {e}")
    
    return True


def test_liquidation_calculation():
    """測試強平價格計算（模擬數據）"""
    print("\n" + "=" * 60)
    print("  強平價格計算測試（模擬）")
    print("=" * 60)
    
    # 模擬持倉數據
    @dataclass
    class MockPosition:
        symbol: str = "BTCUSDT"
        qty: float = 0.1
        entry_price: float = 68000.0
        leverage: int = 10
        unrealized_pnl: float = 0.0
        liquidation_price: float = 0.0
        
        @property
        def is_open(self) -> bool:
            return abs(self.qty) > 0
    
    print("\n📊 模擬持倉強平價格計算:")
    print("-" * 50)
    
    test_cases = [
        MockPosition(qty=0.1, entry_price=68000, leverage=10),   # 10x 做多
        MockPosition(qty=0.1, entry_price=68000, leverage=20),   # 20x 做多
        MockPosition(qty=-0.1, entry_price=68000, leverage=10),  # 10x 做空
        MockPosition(qty=-0.1, entry_price=68000, leverage=20),  # 20x 做空
    ]
    
    # 維持保證金率（簡化）
    mmr = 0.004  # 0.4%
    
    for pos in test_cases:
        side = "LONG" if pos.qty > 0 else "SHORT"
        
        if pos.qty > 0:  # 多倉
            liq = pos.entry_price * (1 - 1/pos.leverage + mmr)
        else:  # 空倉
            liq = pos.entry_price * (1 + 1/pos.leverage - mmr)
        
        distance = abs(pos.entry_price - liq) / pos.entry_price
        
        print(f"\n  {side} {pos.leverage}x @ ${pos.entry_price:,.0f}")
        print(f"    強平價格: ${liq:,.2f}")
        print(f"    距強平:   {distance:.2%}")
    
    print("\n💡 說明:")
    print("   - 槓桿越高，強平價格越近")
    print("   - 多倉強平價格 < 開倉價格")
    print("   - 空倉強平價格 > 開倉價格")
    
    return True


def test_risk_manager():
    """測試風險管理器（需要 API Key）"""
    print("\n" + "=" * 60)
    print("  風險管理器測試")
    print("=" * 60)
    
    try:
        from qtrade.live import BinanceFuturesBroker, FuturesRiskManager
        
        broker = BinanceFuturesBroker(dry_run=True)
        risk_manager = FuturesRiskManager(broker)
        
        # 獲取資金費率
        print("\n📊 資金費率資訊（透過 RiskManager）:")
        for symbol in ["BTCUSDT", "ETHUSDT"]:
            info = risk_manager.get_funding_rate_info(symbol)
            if info:
                print(f"\n  {symbol}:")
                print(f"    當前費率: {info.current_rate*100:.4f}%")
                print(f"    8h 平均: {info.rate_8h_avg*100:.4f}%")
                print(f"    24h 平均: {info.rate_24h_avg*100:.4f}%")
                print(f"    年化: {info.annualized_rate*100:.2f}%")
                print(f"    下次結算: {info.next_funding_time.strftime('%Y-%m-%d %H:%M UTC')}")
        
        # 生成風險報告
        print("\n📊 風險報告:")
        risk_manager.print_risk_report()
        
        return True
        
    except RuntimeError as e:
        print(f"\n⚠️  風險管理器初始化失敗（需要 API Key）: {e}")
        print("   設置環境變數 BINANCE_API_KEY 和 BINANCE_API_SECRET 後重試")
        return False


def main():
    parser = argparse.ArgumentParser(description="測試合約風險管理模組")
    parser.add_argument("--funding-only", action="store_true", help="僅測試資金費率查詢")
    args = parser.parse_args()
    
    print("🛡️  合約風險管理模組測試")
    print("=" * 60)
    
    # 1. 資金費率測試
    test_funding_rate()
    
    # 2. 強平計算測試（模擬）
    test_liquidation_calculation()
    
    if args.funding_only:
        print("\n✅ 資金費率測試完成")
        return 0
    
    # 3. 風險管理器測試
    test_risk_manager()
    
    print("\n✅ 測試完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
