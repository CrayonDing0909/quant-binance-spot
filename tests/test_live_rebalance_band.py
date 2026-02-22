"""
Rebalance Band Gate — 單元測試

測試場景：
  1) diff < 3% 且同方向 → skip
  2) diff >= 3% 且同方向 → execute
  3) 方向翻轉且 apply_on_same_direction_only=true → execute（不 skip）
  4) enabled=false → 行為與舊版一致
  5) YAML 未配置 rebalance_band 時 → 預設不影響舊行為
"""
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path

from qtrade.live.signal_generator import SignalResult
from qtrade.config import RebalanceBandConfig


# ══════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════

def _make_mock_config(tmp_path: Path, band_enabled=False, band_threshold=0.03,
                       band_same_dir_only=True):
    """建立最小化的 mock AppConfig（含 rebalance_band 配置）"""
    cfg = MagicMock()
    cfg.strategy.name = "tsmom_ema"
    cfg.market.symbols = ["BTCUSDT", "ETHUSDT"]
    cfg.market.interval = "1h"
    cfg.market_type_str = "futures"
    cfg.direction = "both"
    cfg.position_sizing.method = "fixed"
    cfg.position_sizing.position_pct = 1.0
    cfg.risk.max_drawdown_pct = 0.2
    cfg.futures = None
    cfg.live.flip_confirmation = False
    cfg.live.kline_cache = False
    cfg.portfolio.get_weight = MagicMock(return_value=0.5)
    cfg.notification = MagicMock()
    cfg.notification.telegram_bot_token = None
    cfg.get_report_dir.return_value = tmp_path / "reports"
    (tmp_path / "reports").mkdir(parents=True, exist_ok=True)
    cfg.strategy.get_params.return_value = {}

    # Rebalance band config
    rb = RebalanceBandConfig(
        enabled=band_enabled,
        threshold_pct=band_threshold,
        apply_on_same_direction_only=band_same_dir_only,
    )
    cfg.live.rebalance_band = rb

    return cfg


def _make_mock_broker(current_pct=0.0):
    """建立 mock broker"""
    from qtrade.live.paper_broker import PaperBroker
    broker = MagicMock(spec=PaperBroker)
    broker.get_position_pct.return_value = current_pct
    broker.execute_target_position.return_value = None
    broker.get_equity = MagicMock(return_value=10000.0)
    return broker


def _make_runner(cfg, broker):
    """建立 TestRunner 實例"""
    with patch("qtrade.live.base_runner.TelegramNotifier") as mock_notifier_cls:
        mock_notifier_cls.from_config.return_value = MagicMock(enabled=False)
        from qtrade.live.base_runner import BaseRunner

        class TestRunner(BaseRunner):
            def run(self):
                pass

        return TestRunner(cfg, broker, "paper")


def _make_signal(symbol="BTCUSDT", signal=1.0, price=50000.0):
    return SignalResult(
        symbol=symbol, signal=signal, price=price,
        timestamp="2026-01-01", strategy="tsmom_ema",
        indicators={"rsi": 55.0, "adx": 25.0},
    )


# ══════════════════════════════════════════════════════════
#  Test Cases
# ══════════════════════════════════════════════════════════

class TestRebalanceBandConfig:
    """RebalanceBandConfig dataclass 測試"""

    def test_defaults(self):
        rb = RebalanceBandConfig()
        assert rb.enabled is False
        assert rb.threshold_pct == 0.03
        assert rb.apply_on_same_direction_only is True

    def test_custom(self):
        rb = RebalanceBandConfig(enabled=True, threshold_pct=0.05,
                                  apply_on_same_direction_only=False)
        assert rb.enabled is True
        assert rb.threshold_pct == 0.05
        assert rb.apply_on_same_direction_only is False


class TestRebalanceBandSkip:
    """Case 1: diff < 3% 且不被 step 5 捕獲 → band skip

    Note: step 5 (fill_ratio >= 0.80) 已經處理了大部分同方向微調。
    Band gate 主要攔截：
      - 從 flat (current=0) 到小倉位（step 5 需 current!=0 才生效）
      - 同方向但 fill_ratio < 0.80 且 diff < threshold 的邊界情況
    """

    def test_flat_to_small_position_skipped(self, tmp_path):
        """從 flat(0) 到 target=0.025 → diff=0.025 < 3%
        Without band: diff=0.025 >= 0.02 → would trade
        With band:    diff=0.025 <  0.03 → skip
        """
        cfg = _make_mock_config(tmp_path, band_enabled=True, band_threshold=0.03)
        broker = _make_mock_broker(current_pct=0.0)
        runner = _make_runner(cfg, broker)

        # signal=0.05, weight=0.5 → target=0.025, diff=0.025
        sig = _make_signal(signal=0.05)
        result = runner._process_signal("BTCUSDT", sig)

        assert result is None
        broker.execute_target_position.assert_not_called()
        assert runner._band_skip_count == 1

    def test_flat_to_borderline_position_skipped(self, tmp_path):
        """從 flat(0) 到 target=0.029 → diff=0.029 < 3% → skip"""
        cfg = _make_mock_config(tmp_path, band_enabled=True, band_threshold=0.03)
        broker = _make_mock_broker(current_pct=0.0)
        runner = _make_runner(cfg, broker)

        # signal=0.058, weight=0.5 → target=0.029, diff=0.029 < 0.03
        sig = _make_signal(signal=0.058)
        result = runner._process_signal("BTCUSDT", sig)

        assert result is None
        broker.execute_target_position.assert_not_called()
        assert runner._band_skip_count == 1


class TestRebalanceBandExecute:
    """Case 2: diff >= 3% 且同方向 → execute"""

    def test_large_same_direction_adjustment_executes(self, tmp_path):
        """持倉 +0.10, 目標 +0.50 → diff=0.40 >= 3% → execute"""
        cfg = _make_mock_config(tmp_path, band_enabled=True, band_threshold=0.03)
        broker = _make_mock_broker(current_pct=0.10)
        mock_trade = MagicMock()
        mock_trade.side = "BUY"
        mock_trade.qty = 0.1
        mock_trade.price = 50000.0
        mock_trade.pnl = None
        broker.execute_target_position.return_value = mock_trade
        runner = _make_runner(cfg, broker)

        sig = _make_signal(signal=1.0)  # target = 0.50
        result = runner._process_signal("BTCUSDT", sig)

        assert result is not None
        broker.execute_target_position.assert_called_once()
        assert runner._band_skip_count == 0


class TestRebalanceBandDirectionFlip:
    """Case 3: 方向翻轉 + apply_on_same_direction_only=true → execute"""

    def test_direction_flip_bypasses_band(self, tmp_path):
        """持倉 -0.01, 目標 +0.50 → 方向翻轉 → 不受 band 限制"""
        cfg = _make_mock_config(tmp_path, band_enabled=True, band_threshold=0.03,
                                 band_same_dir_only=True)
        broker = _make_mock_broker(current_pct=-0.02)
        mock_trade = MagicMock()
        mock_trade.side = "BUY"
        mock_trade.qty = 0.1
        mock_trade.price = 50000.0
        mock_trade.pnl = None
        broker.execute_target_position.return_value = mock_trade
        runner = _make_runner(cfg, broker)

        sig = _make_signal(signal=1.0)  # target = +0.50
        result = runner._process_signal("BTCUSDT", sig)

        # Should execute because direction flip bypasses band
        assert result is not None
        broker.execute_target_position.assert_called_once()
        assert runner._band_skip_count == 0

    def test_direction_flip_blocked_when_same_dir_only_false(self, tmp_path):
        """apply_on_same_direction_only=false → 方向翻轉也受 band 限制（極端 threshold）"""
        cfg = _make_mock_config(tmp_path, band_enabled=True, band_threshold=0.99,
                                 band_same_dir_only=False)
        broker = _make_mock_broker(current_pct=-0.02)
        runner = _make_runner(cfg, broker)

        # target = +0.50, current = -0.02, diff = 0.52
        # Band=99% → diff=0.52 < 0.99 → skip (even direction flip)
        sig = _make_signal(signal=1.0)
        result = runner._process_signal("BTCUSDT", sig)

        assert result is None
        assert runner._band_skip_count == 1


class TestRebalanceBandDisabled:
    """Case 4: enabled=false → 行為與舊版一致"""

    def test_disabled_band_allows_small_adjustment(self, tmp_path):
        """Band disabled → small diff 仍正常走到 diff >= 0.02 判斷"""
        cfg = _make_mock_config(tmp_path, band_enabled=False)
        broker = _make_mock_broker(current_pct=0.0)
        mock_trade = MagicMock()
        mock_trade.side = "BUY"
        mock_trade.qty = 0.1
        mock_trade.price = 50000.0
        mock_trade.pnl = None
        broker.execute_target_position.return_value = mock_trade
        runner = _make_runner(cfg, broker)

        sig = _make_signal(signal=1.0)  # diff=0.50 > 0.02 → trade
        result = runner._process_signal("BTCUSDT", sig)

        assert result is not None
        assert runner._band_skip_count == 0


class TestRebalanceBandNotConfigured:
    """Case 5: YAML 未配置 rebalance_band → 預設 disabled"""

    def test_default_config_no_band(self):
        """預設 RebalanceBandConfig 不啟用"""
        rb = RebalanceBandConfig()
        assert rb.enabled is False

    def test_default_in_appconfig(self, tmp_path):
        """使用 default LiveConfig 時 band 是 disabled"""
        cfg = _make_mock_config(tmp_path)  # band_enabled=False by default
        broker = _make_mock_broker(current_pct=0.0)
        mock_trade = MagicMock()
        mock_trade.side = "BUY"
        mock_trade.qty = 0.1
        mock_trade.price = 50000.0
        mock_trade.pnl = None
        broker.execute_target_position.return_value = mock_trade
        runner = _make_runner(cfg, broker)

        sig = _make_signal(signal=1.0)
        result = runner._process_signal("BTCUSDT", sig)

        assert result is not None
        assert runner._band_skip_count == 0


class TestRebalanceBandCounter:
    """監控計數器測試"""

    def test_skip_counter_increments(self, tmp_path):
        """多次 skip 時計數器正確累加"""
        cfg = _make_mock_config(tmp_path, band_enabled=True, band_threshold=0.03)
        broker = _make_mock_broker(current_pct=0.0)  # flat → small position
        runner = _make_runner(cfg, broker)

        sig = _make_signal(signal=0.05)  # target=0.025, diff=0.025 < 3% → skip

        runner._process_signal("BTCUSDT", sig)
        runner._process_signal("BTCUSDT", sig)
        runner._process_signal("BTCUSDT", sig)

        assert runner._band_skip_count == 3
        assert runner._band_skip_notional_est > 0

    def test_execute_does_not_increment_counter(self, tmp_path):
        """正常交易不增加 skip 計數"""
        cfg = _make_mock_config(tmp_path, band_enabled=True, band_threshold=0.03)
        broker = _make_mock_broker(current_pct=0.0)
        mock_trade = MagicMock()
        mock_trade.side = "BUY"
        mock_trade.qty = 0.1
        mock_trade.price = 50000.0
        mock_trade.pnl = None
        broker.execute_target_position.return_value = mock_trade
        runner = _make_runner(cfg, broker)

        sig = _make_signal(signal=1.0)  # target=0.50, diff=0.50 → execute
        runner._process_signal("BTCUSDT", sig)

        assert runner._band_skip_count == 0


class TestRebalanceBandConfigParsing:
    """Config YAML 解析測試"""

    def test_load_config_with_rebalance_band(self, tmp_path):
        """確認 load_config 能正確解析 rebalance_band"""
        import yaml
        config_data = {
            "market": {
                "symbols": ["BTCUSDT"],
                "interval": "1h",
                "start": "2024-01-01",
                "end": None,
                "market_type": "futures",
            },
            "backtest": {
                "initial_cash": 10000,
                "fee_bps": 4.5,
                "slippage_bps": 2.0,
                "trade_on": "next_open",
            },
            "strategy": {
                "name": "tsmom_ema",
                "params": {"ema_span": 24},
            },
            "live": {
                "rebalance_band": {
                    "enabled": True,
                    "threshold_pct": 0.05,
                    "apply_on_same_direction_only": False,
                },
            },
        }
        config_path = tmp_path / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        from qtrade.config import load_config
        cfg = load_config(str(config_path))

        assert cfg.live.rebalance_band.enabled is True
        assert cfg.live.rebalance_band.threshold_pct == 0.05
        assert cfg.live.rebalance_band.apply_on_same_direction_only is False

    def test_load_config_without_rebalance_band(self, tmp_path):
        """YAML 無 rebalance_band 時預設 disabled"""
        import yaml
        config_data = {
            "market": {
                "symbols": ["BTCUSDT"],
                "interval": "1h",
                "start": "2024-01-01",
                "end": None,
                "market_type": "futures",
            },
            "backtest": {
                "initial_cash": 10000,
                "fee_bps": 4.5,
                "slippage_bps": 2.0,
                "trade_on": "next_open",
            },
            "strategy": {
                "name": "tsmom_ema",
                "params": {"ema_span": 24},
            },
        }
        config_path = tmp_path / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        from qtrade.config import load_config
        cfg = load_config(str(config_path))

        assert cfg.live.rebalance_band.enabled is False
        assert cfg.live.rebalance_band.threshold_pct == 0.03
        assert cfg.live.rebalance_band.apply_on_same_direction_only is True
