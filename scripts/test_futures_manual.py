#!/usr/bin/env python3
"""
合約功能手動測試腳本

執行方式：
    # 在專案目錄下執行
    python scripts/test_futures_manual.py
    
    # 或使用 venv
    .venv/bin/python scripts/test_futures_manual.py
"""
from __future__ import annotations
import sys
from pathlib import Path

# 加入 src 目錄
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_config_futures():
    """測試 Config 合約相關功能"""
    print("\n" + "=" * 60)
    print("  測試 1: Config 合約配置載入")
    print("=" * 60)
    
    from qtrade.config import load_config, MarketType
    
    # 測試 Spot 配置
    cfg = load_config("config/rsi_adx_atr.yaml")
    print(f"\n  [Spot Config]")
    print(f"    market_type: {cfg.market.market_type}")
    print(f"    is_futures: {cfg.is_futures}")
    print(f"    supports_short: {cfg.supports_short}")
    assert cfg.market.market_type == MarketType.SPOT, "Spot 配置錯誤"
    assert not cfg.is_futures, "Spot 不應該是 futures"
    print("    ✅ Spot 配置正確")
    
    # 測試 Futures 配置
    cfg_futures = load_config("config/futures_rsi_adx_atr.yaml")
    print(f"\n  [Futures Config]")
    print(f"    market_type: {cfg_futures.market.market_type}")
    print(f"    is_futures: {cfg_futures.is_futures}")
    print(f"    supports_short: {cfg_futures.supports_short}")
    print(f"    leverage: {cfg_futures.futures.leverage}")
    print(f"    margin_type: {cfg_futures.futures.margin_type}")
    assert cfg_futures.market.market_type == MarketType.FUTURES, "Futures 配置錯誤"
    assert cfg_futures.is_futures, "Futures 應該是 futures"
    assert cfg_futures.supports_short, "Futures 應該支援做空"
    print("    ✅ Futures 配置正確")


def test_strategy_context():
    """測試策略上下文"""
    print("\n" + "=" * 60)
    print("  測試 2: StrategyContext 做空判斷")
    print("=" * 60)
    
    from qtrade.strategy.base import StrategyContext
    
    # Spot 上下文
    ctx_spot = StrategyContext(symbol="BTCUSDT", interval="1h", market_type="spot")
    print(f"\n  [Spot Context]")
    print(f"    supports_short: {ctx_spot.supports_short}")
    print(f"    is_futures: {ctx_spot.is_futures}")
    assert not ctx_spot.supports_short, "Spot 不應該支援做空"
    print("    ✅ Spot Context 正確")
    
    # Futures 上下文
    ctx_futures = StrategyContext(symbol="BTCUSDT", interval="1h", market_type="futures")
    print(f"\n  [Futures Context]")
    print(f"    supports_short: {ctx_futures.supports_short}")
    print(f"    is_futures: {ctx_futures.is_futures}")
    assert ctx_futures.supports_short, "Futures 應該支援做空"
    print("    ✅ Futures Context 正確")


def test_paper_broker_short():
    """測試 PaperBroker 做空功能"""
    print("\n" + "=" * 60)
    print("  測試 3: PaperBroker 做空功能")
    print("=" * 60)
    
    from qtrade.live.paper_broker import PaperBroker
    
    # Spot 模式
    print(f"\n  [Spot 模式]")
    broker_spot = PaperBroker(
        initial_cash=10000,
        market_type="spot",
    )
    print(f"    supports_short: {broker_spot.supports_short}")
    assert not broker_spot.supports_short, "Spot 不應該支援做空"
    
    # 嘗試做空（應該被忽略）
    trade = broker_spot.execute_target_position(
        symbol="BTCUSDT",
        target_pct=-0.5,
        current_price=50000,
    )
    assert trade is None, "Spot 模式做空應該無效"
    print("    ✅ Spot 模式正確阻止做空")
    
    # Futures 模式
    print(f"\n  [Futures 模式]")
    broker_futures = PaperBroker(
        initial_cash=10000,
        market_type="futures",
        leverage=2,
    )
    print(f"    supports_short: {broker_futures.supports_short}")
    print(f"    leverage: {broker_futures.leverage}")
    
    # 開空倉 50%
    trade = broker_futures.execute_target_position(
        symbol="BTCUSDT",
        target_pct=-0.5,
        current_price=50000,
    )
    
    assert trade is not None, "Futures 模式應該可以做空"
    assert trade.side == "SHORT", f"應該是 SHORT，但得到 {trade.side}"
    
    pos = broker_futures.get_position("BTCUSDT")
    print(f"    交易後:")
    print(f"      qty: {pos.qty:.6f} (負數表示空倉)")
    print(f"      avg_entry: ${pos.avg_entry:.2f}")
    print(f"      is_short: {pos.is_short}")
    print(f"      side: {pos.side}")
    
    assert pos.is_short, "應該是空倉"
    assert pos.qty < 0, "空倉數量應該是負數"
    print("    ✅ Futures 模式做空成功")
    
    # 價格下跌後平空（應該盈利）
    print(f"\n  [平空倉測試 - 價格下跌]")
    initial_cash = broker_futures.account.cash
    
    trade = broker_futures.execute_target_position(
        symbol="BTCUSDT",
        target_pct=0,
        current_price=45000,  # 跌 10%
    )
    
    assert trade is not None, "應該有平倉交易"
    assert trade.side == "CLOSE_SHORT", f"應該是 CLOSE_SHORT，但得到 {trade.side}"
    assert trade.pnl is not None, "應該有 PnL"
    assert trade.pnl > 0, f"價格下跌做空應該盈利，但 PnL = {trade.pnl}"
    
    print(f"    平倉價格: ${trade.price:.2f}")
    print(f"    PnL: ${trade.pnl:+.2f} 📈")
    print("    ✅ 平空倉盈虧計算正確")


def test_paper_broker_long_and_short_cycle():
    """測試完整的多空循環"""
    print("\n" + "=" * 60)
    print("  測試 4: 多空循環測試")
    print("=" * 60)
    
    from qtrade.live.paper_broker import PaperBroker
    
    broker = PaperBroker(
        initial_cash=10000,
        fee_bps=0,  # 無手續費方便觀察
        slippage_bps=0,
        market_type="futures",
        leverage=1,
    )
    
    print(f"\n  初始狀態:")
    print(f"    現金: ${broker.account.cash:,.2f}")
    
    # 1. 開多倉 50%
    print(f"\n  步驟 1: 開多倉 50% @ $50000")
    broker.execute_target_position("BTCUSDT", 0.5, 50000)
    pos = broker.get_position("BTCUSDT")
    print(f"    現金: ${broker.account.cash:,.2f}")
    print(f"    持倉: {pos.qty:.6f} BTC (${pos.qty * 50000:,.2f})")
    print(f"    side: {pos.side}")
    equity = broker.get_equity({"BTCUSDT": 50000})
    print(f"    權益: ${equity:,.2f}")
    
    # 2. 平多開空（價格上漲到 55000）
    print(f"\n  步驟 2: 平多開空 50% @ $55000")
    broker.execute_target_position("BTCUSDT", 0, 55000)  # 先平多
    broker.execute_target_position("BTCUSDT", -0.5, 55000)  # 再開空
    pos = broker.get_position("BTCUSDT")
    print(f"    現金: ${broker.account.cash:,.2f}")
    print(f"    持倉: {pos.qty:.6f} BTC (空倉)")
    print(f"    side: {pos.side}")
    equity = broker.get_equity({"BTCUSDT": 55000})
    print(f"    權益: ${equity:,.2f}")
    
    # 3. 價格下跌到 50000，平空
    print(f"\n  步驟 3: 平空 @ $50000 (價格下跌，空倉盈利)")
    broker.execute_target_position("BTCUSDT", 0, 50000)
    pos = broker.get_position("BTCUSDT")
    print(f"    現金: ${broker.account.cash:,.2f}")
    print(f"    side: {pos.side}")
    equity = broker.get_equity({"BTCUSDT": 50000})
    print(f"    權益: ${equity:,.2f}")
    
    # 計算總收益
    total_return = (equity / 10000 - 1) * 100
    print(f"\n  總收益: {total_return:+.2f}%")
    print("    ✅ 多空循環測試完成")


def test_paper_broker_summary():
    """測試帳戶摘要顯示"""
    print("\n" + "=" * 60)
    print("  測試 5: 帳戶摘要顯示")
    print("=" * 60)
    
    from qtrade.live.paper_broker import PaperBroker
    
    broker = PaperBroker(
        initial_cash=10000,
        market_type="futures",
        leverage=3,
    )
    
    # 開空倉
    broker.execute_target_position("BTCUSDT", -0.5, 50000)
    broker.execute_target_position("ETHUSDT", 0.3, 3000)
    
    summary = broker.summary({"BTCUSDT": 48000, "ETHUSDT": 3100})
    print(f"\n{summary}")
    
    assert "FUTURES" in summary, "摘要應該顯示 FUTURES"
    assert "SHORT" in summary, "摘要應該顯示 SHORT"
    assert "LONG" in summary, "摘要應該顯示 LONG"
    print("\n    ✅ 帳戶摘要顯示正確")


def test_position_pct():
    """測試倉位比例計算"""
    print("\n" + "=" * 60)
    print("  測試 6: 倉位比例計算")
    print("=" * 60)
    
    from qtrade.live.paper_broker import PaperBroker
    
    broker = PaperBroker(
        initial_cash=10000,
        fee_bps=0,
        slippage_bps=0,
        market_type="futures",
        leverage=1,
    )
    
    # 做空 50%
    broker.execute_target_position("BTCUSDT", -0.5, 50000)
    
    pct = broker.get_position_pct("BTCUSDT", 50000)
    print(f"\n  目標: -50%, 實際: {pct:.1%}")
    
    assert -0.55 < pct < -0.45, f"倉位比例應該約 -50%，但得到 {pct:.1%}"
    print("    ✅ 倉位比例計算正確")


def main():
    """執行所有測試"""
    print("\n" + "🔴" * 30)
    print("  合約交易功能測試")
    print("🔴" * 30)
    
    tests = [
        test_config_futures,
        test_strategy_context,
        test_paper_broker_short,
        test_paper_broker_long_and_short_cycle,
        test_paper_broker_summary,
        test_position_pct,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"\n    ❌ 測試失敗: {e}")
            failed += 1
        except Exception as e:
            print(f"\n    ❌ 測試錯誤: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"  測試結果: {passed} 通過, {failed} 失敗")
    print("=" * 60)
    
    if failed > 0:
        sys.exit(1)
    else:
        print("\n  🎉 所有測試通過！")
        print("\n  下一步:")
        print("    # Paper Trading 測試（Futures）")
        print("    python scripts/run_live.py -c config/futures_rsi_adx_atr.yaml --paper --once")
        print()


if __name__ == "__main__":
    main()
