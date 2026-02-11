"""
合約交易功能測試

測試項目：
1. Config 載入 - market_type 解析
2. PaperBroker 做空功能
3. StrategyContext 做空判斷
4. 信號範圍 [-1, 1] 處理
"""
import pytest
import tempfile
from pathlib import Path
import yaml

from qtrade.config import load_config, MarketType, FuturesConfig
from qtrade.live.paper_broker import PaperBroker, SymbolPosition
from qtrade.strategy.base import StrategyContext


class TestConfigFutures:
    """測試 Config 合約相關功能"""

    def test_market_type_spot_default(self, tmp_path):
        """測試預設市場類型為 spot"""
        config_content = """
market:
  symbols: ["BTCUSDT"]
  interval: "1h"
  start: "2022-01-01"
  end: null

strategy:
  name: "test"
  params: {}

backtest:
  initial_cash: 10000
  fee_bps: 6
  slippage_bps: 5
  trade_on: "next_open"

output:
  report_dir: "./reports"
"""
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(config_content)
        
        cfg = load_config(str(config_path))
        
        assert cfg.market.market_type == MarketType.SPOT
        assert not cfg.is_futures
        assert not cfg.supports_short
        assert cfg.futures is None

    def test_market_type_futures(self, tmp_path):
        """測試 market_type = futures"""
        config_content = """
market:
  symbols: ["BTCUSDT"]
  interval: "1h"
  start: "2022-01-01"
  end: null
  market_type: "futures"

futures:
  leverage: 3
  margin_type: "ISOLATED"
  position_mode: "ONE_WAY"

strategy:
  name: "test"
  params: {}

backtest:
  initial_cash: 10000
  fee_bps: 4
  slippage_bps: 3
  trade_on: "next_open"

output:
  report_dir: "./reports"
"""
        config_path = tmp_path / "test_futures.yaml"
        config_path.write_text(config_content)
        
        cfg = load_config(str(config_path))
        
        assert cfg.market.market_type == MarketType.FUTURES
        assert cfg.is_futures
        assert cfg.supports_short
        assert cfg.futures is not None
        assert cfg.futures.leverage == 3
        assert cfg.futures.margin_type == "ISOLATED"
        assert cfg.futures.position_mode == "ONE_WAY"


class TestStrategyContext:
    """測試策略上下文"""

    def test_spot_context(self):
        """測試 Spot 上下文"""
        ctx = StrategyContext(symbol="BTCUSDT", interval="1h", market_type="spot")
        assert not ctx.supports_short
        assert not ctx.is_futures

    def test_futures_context(self):
        """測試 Futures 上下文"""
        ctx = StrategyContext(symbol="BTCUSDT", interval="1h", market_type="futures")
        assert ctx.supports_short
        assert ctx.is_futures


class TestPaperBrokerShort:
    """測試 PaperBroker 做空功能"""

    def test_spot_mode_no_short(self):
        """Spot 模式不能做空"""
        broker = PaperBroker(
            initial_cash=10000,
            market_type="spot",
        )
        
        assert not broker.supports_short
        
        # 嘗試執行做空目標（應該被 clip 到 0）
        trade = broker.execute_target_position(
            symbol="BTCUSDT",
            target_pct=-0.5,
            current_price=50000,
        )
        
        # Spot 模式下，target_pct 會被 clip 到 0，如果當前是空倉則不交易
        # 所以不會有交易
        assert trade is None
        
        pos = broker.get_position("BTCUSDT")
        assert not pos.is_short

    def test_futures_mode_can_short(self):
        """Futures 模式可以做空"""
        broker = PaperBroker(
            initial_cash=10000,
            market_type="futures",
            leverage=1,
        )
        
        assert broker.supports_short
        
        # 執行做空
        trade = broker.execute_target_position(
            symbol="BTCUSDT",
            target_pct=-0.5,
            current_price=50000,
        )
        
        assert trade is not None
        assert trade.side == "SHORT"
        
        pos = broker.get_position("BTCUSDT")
        assert pos.is_short
        assert pos.qty < 0

    def test_short_then_close(self):
        """測試開空倉後平倉"""
        broker = PaperBroker(
            initial_cash=10000,
            market_type="futures",
            leverage=1,
        )
        
        # 開空倉 50%
        broker.execute_target_position(
            symbol="BTCUSDT",
            target_pct=-0.5,
            current_price=50000,
        )
        
        pos = broker.get_position("BTCUSDT")
        assert pos.is_short
        initial_qty = abs(pos.qty)
        
        # 平空倉（價格下跌，應該盈利）
        trade = broker.execute_target_position(
            symbol="BTCUSDT",
            target_pct=0,
            current_price=45000,  # 下跌 10%
        )
        
        assert trade is not None
        assert trade.side == "CLOSE_SHORT"
        assert trade.pnl is not None
        assert trade.pnl > 0  # 做空後價格下跌 = 盈利
        
        pos = broker.get_position("BTCUSDT")
        assert not pos.is_open

    def test_short_pnl_calculation(self):
        """測試做空盈虧計算"""
        broker = PaperBroker(
            initial_cash=10000,
            fee_bps=0,  # 無手續費方便計算
            slippage_bps=0,  # 無滑點方便計算
            market_type="futures",
            leverage=1,
        )
        
        # 全倉做空 @ 50000
        broker.execute_target_position(
            symbol="BTCUSDT",
            target_pct=-1.0,
            current_price=50000,
        )
        
        pos = broker.get_position("BTCUSDT")
        entry_price = pos.avg_entry
        qty = abs(pos.qty)
        
        # 價格下跌到 40000（跌 20%）
        # 未實現盈虧 = (entry - current) * qty = (50000 - 40000) * qty = 10000 * qty
        equity = broker.get_equity({"BTCUSDT": 40000})
        
        # 盈利應該約為 20% (因為價格跌了 20%，做空應該賺 20%)
        expected_return = (equity / 10000 - 1) * 100
        assert expected_return > 15  # 考慮一些誤差

    def test_position_pct_short(self):
        """測試做空時的倉位比例計算"""
        broker = PaperBroker(
            initial_cash=10000,
            fee_bps=0,
            slippage_bps=0,
            market_type="futures",
            leverage=1,
        )
        
        # 50% 做空
        broker.execute_target_position(
            symbol="BTCUSDT",
            target_pct=-0.5,
            current_price=50000,
        )
        
        # 檢查倉位比例是否約為 -50%
        pct = broker.get_position_pct("BTCUSDT", 50000)
        assert -0.55 < pct < -0.45  # 考慮一些誤差

    def test_leverage(self):
        """測試槓桿功能"""
        broker = PaperBroker(
            initial_cash=10000,
            fee_bps=0,
            slippage_bps=0,
            market_type="futures",
            leverage=3,  # 3 倍槓桿
        )
        
        assert broker.leverage == 3
        
        # 開 3 倍槓桿空倉
        broker.execute_target_position(
            symbol="BTCUSDT",
            target_pct=-1.0,
            current_price=50000,
        )
        
        pos = broker.get_position("BTCUSDT")
        # 使用槓桿，倉位價值應該約等於 cash * leverage
        position_value = abs(pos.qty) * pos.avg_entry
        assert position_value > 10000 * 2  # 至少 2 倍（考慮保證金扣除）


class TestSymbolPosition:
    """測試 SymbolPosition 類"""

    def test_long_position(self):
        """測試多倉"""
        pos = SymbolPosition(symbol="BTCUSDT", qty=0.1, avg_entry=50000)
        assert pos.is_open
        assert pos.is_long
        assert not pos.is_short
        assert pos.side == "LONG"

    def test_short_position(self):
        """測試空倉"""
        pos = SymbolPosition(symbol="BTCUSDT", qty=-0.1, avg_entry=50000)
        assert pos.is_open
        assert not pos.is_long
        assert pos.is_short
        assert pos.side == "SHORT"

    def test_no_position(self):
        """測試空倉位"""
        pos = SymbolPosition(symbol="BTCUSDT", qty=0, avg_entry=0)
        assert not pos.is_open
        assert not pos.is_long
        assert not pos.is_short
        assert pos.side == "NONE"


class TestPaperBrokerStatePersistence:
    """測試 PaperBroker 狀態持久化"""

    def test_save_and_load_short_position(self, tmp_path):
        """測試保存和載入空倉狀態"""
        state_path = tmp_path / "paper_state.json"
        
        # 建立 broker 並開空倉
        broker1 = PaperBroker(
            initial_cash=10000,
            market_type="futures",
            leverage=2,
            state_path=state_path,
        )
        
        broker1.execute_target_position(
            symbol="BTCUSDT",
            target_pct=-0.5,
            current_price=50000,
        )
        
        pos1 = broker1.get_position("BTCUSDT")
        assert pos1.is_short
        
        # 建立新 broker，應該自動載入狀態
        broker2 = PaperBroker(
            initial_cash=10000,  # 會被覆蓋
            state_path=state_path,
        )
        
        assert broker2.market_type == "futures"
        assert broker2.leverage == 2
        
        pos2 = broker2.get_position("BTCUSDT")
        assert pos2.is_short
        assert abs(pos2.qty - pos1.qty) < 0.0001
        assert abs(pos2.avg_entry - pos1.avg_entry) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
