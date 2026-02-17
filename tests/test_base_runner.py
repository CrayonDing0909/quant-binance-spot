"""
BaseRunner 安全機制單元測試

測試所有共享的安全機制：
  - 熔斷 (circuit breaker)
  - 倉位計算 (position sizing)
  - 信號處理 (_process_signal)
  - 信號狀態持久化
  - SignalResult dataclass
"""
import pytest
import json
from unittest.mock import MagicMock, patch
from pathlib import Path

from qtrade.live.signal_generator import SignalResult, PositionInfo


# ══════════════════════════════════════════════════════════
#  SignalResult dataclass
# ══════════════════════════════════════════════════════════


class TestSignalResult:
    """SignalResult dataclass 基本測試"""

    def test_creation(self):
        sig = SignalResult(
            symbol="BTCUSDT", signal=0.75, price=50000.0,
            timestamp="2026-01-01", strategy="rsi_adx_atr",
            indicators={"rsi": 65.0, "adx": 30.0},
        )
        assert sig.symbol == "BTCUSDT"
        assert sig.signal == 0.75
        assert sig.price == 50000.0
        assert sig.indicators["rsi"] == 65.0

    def test_default_indicators(self):
        sig = SignalResult(
            symbol="ETHUSDT", signal=0.0, price=3000.0,
            timestamp="", strategy="test",
        )
        assert sig.indicators == {}
        assert sig.position_info.pct == 0.0

    def test_to_dict_basic(self):
        sig = SignalResult(
            symbol="SOLUSDT", signal=-1.0, price=100.0,
            timestamp="2026-01-01 00:00", strategy="test",
            indicators={"rsi": 25.0},
        )
        d = sig.to_dict()
        assert d["symbol"] == "SOLUSDT"
        assert d["signal"] == -1.0
        assert d["price"] == 100.0
        assert d["indicators"]["rsi"] == 25.0
        assert "_position" not in d  # no position info

    def test_to_dict_with_position(self):
        sig = SignalResult(
            symbol="BTCUSDT", signal=1.0, price=50000.0,
            timestamp="", strategy="test",
        )
        sig.position_info = PositionInfo(
            pct=0.5, entry=49000.0, qty=0.1, side="LONG",
            sl=48000.0, tp=55000.0,
        )
        d = sig.to_dict()
        assert "_position" in d
        assert d["_position"]["pct"] == 0.5
        assert d["_position"]["sl"] == 48000.0
        assert d["_position"]["tp"] == 55000.0

    def test_to_dict_no_position_when_flat(self):
        sig = SignalResult(
            symbol="X", signal=0.0, price=1.0,
            timestamp="", strategy="test",
        )
        sig.position_info = PositionInfo(pct=0.0)
        d = sig.to_dict()
        assert "_position" not in d


class TestPositionInfo:
    """PositionInfo dataclass 測試"""

    def test_defaults(self):
        pi = PositionInfo()
        assert pi.pct == 0.0
        assert pi.sl is None
        assert pi.tp is None

    def test_full(self):
        pi = PositionInfo(
            pct=1.0, entry=50000.0, qty=0.1,
            side="LONG", sl=48000.0, tp=55000.0,
        )
        assert pi.side == "LONG"
        assert pi.sl == 48000.0


# ══════════════════════════════════════════════════════════
#  BaseRunner 安全機制（需要 mock AppConfig + broker）
# ══════════════════════════════════════════════════════════


def _make_mock_config(tmp_path: Path):
    """建立最小化的 mock AppConfig"""
    cfg = MagicMock()
    cfg.strategy.name = "rsi_adx_atr"
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
    cfg.strategy.get_params.return_value = {"stop_loss_atr": 2.0, "take_profit_atr": 3.0}
    return cfg


def _make_mock_broker():
    """建立 mock broker"""
    from qtrade.live.paper_broker import PaperBroker
    broker = MagicMock(spec=PaperBroker)
    broker.get_position_pct.return_value = 0.0
    broker.execute_target_position.return_value = None
    broker.get_equity = MagicMock(return_value=10000.0)
    return broker


class TestBaseRunnerCircuitBreaker:
    """熔斷機制測試"""

    def test_no_circuit_breaker_when_no_config(self, tmp_path):
        cfg = _make_mock_config(tmp_path)
        cfg.risk.max_drawdown_pct = None
        broker = _make_mock_broker()

        with patch("qtrade.live.base_runner.TelegramNotifier") as mock_notifier_cls:
            mock_notifier_cls.from_config.return_value = MagicMock(enabled=False)
            from qtrade.live.base_runner import BaseRunner

            class TestRunner(BaseRunner):
                def run(self):
                    pass

            runner = TestRunner(cfg, broker, "paper")
            assert runner._check_circuit_breaker() is False

    def test_circuit_breaker_triggers_on_drawdown(self, tmp_path):
        cfg = _make_mock_config(tmp_path)
        cfg.risk.max_drawdown_pct = 0.10  # 10%
        # Use a non-PaperBroker mock so _get_equity goes through the simpler path
        broker = MagicMock()
        broker.get_position_pct.return_value = 0.0

        with patch("qtrade.live.base_runner.TelegramNotifier") as mock_notifier_cls:
            mock_notifier = MagicMock(enabled=False)
            mock_notifier_cls.from_config.return_value = mock_notifier
            from qtrade.live.base_runner import BaseRunner

            class TestRunner(BaseRunner):
                def run(self):
                    pass

            runner = TestRunner(cfg, broker, "paper")

            # First call: sets initial equity
            broker.get_equity.return_value = 10000.0
            assert runner._check_circuit_breaker() is False
            assert runner._initial_equity == 10000.0

            # Second call: equity dropped 15% → trigger
            broker.get_equity.return_value = 8500.0
            assert runner._check_circuit_breaker() is True
            assert runner._circuit_breaker_triggered is True


class TestBaseRunnerPositionSizing:
    """倉位計算測試"""

    def test_fixed_position_sizer(self, tmp_path):
        cfg = _make_mock_config(tmp_path)
        broker = _make_mock_broker()

        with patch("qtrade.live.base_runner.TelegramNotifier") as mock_notifier_cls:
            mock_notifier_cls.from_config.return_value = MagicMock(enabled=False)
            from qtrade.live.base_runner import BaseRunner

            class TestRunner(BaseRunner):
                def run(self):
                    pass

            runner = TestRunner(cfg, broker, "paper")
            from qtrade.risk.position_sizing import FixedPositionSizer
            assert isinstance(runner.position_sizer, FixedPositionSizer)


class TestBaseRunnerSignalState:
    """信號狀態持久化測試"""

    def test_save_and_load_signal_state(self, tmp_path):
        cfg = _make_mock_config(tmp_path)
        broker = _make_mock_broker()

        with patch("qtrade.live.base_runner.TelegramNotifier") as mock_notifier_cls:
            mock_notifier_cls.from_config.return_value = MagicMock(enabled=False)
            from qtrade.live.base_runner import BaseRunner

            class TestRunner(BaseRunner):
                def run(self):
                    pass

            runner = TestRunner(cfg, broker, "paper")

            # Save state
            runner._save_signal_state({"BTCUSDT": 1.0, "ETHUSDT": -0.5})

            # Load state (new runner instance)
            runner2 = TestRunner(cfg, broker, "paper")
            assert runner2._signal_state.get("BTCUSDT") == 1.0
            assert runner2._signal_state.get("ETHUSDT") == -0.5


class TestBaseRunnerProcessSignal:
    """信號處理 _process_signal 測試"""

    def test_process_signal_no_trade_when_flat(self, tmp_path):
        cfg = _make_mock_config(tmp_path)
        broker = _make_mock_broker()
        broker.get_position_pct.return_value = 0.0

        with patch("qtrade.live.base_runner.TelegramNotifier") as mock_notifier_cls:
            mock_notifier_cls.from_config.return_value = MagicMock(enabled=False)
            from qtrade.live.base_runner import BaseRunner

            class TestRunner(BaseRunner):
                def run(self):
                    pass

            runner = TestRunner(cfg, broker, "paper")

            sig = SignalResult(
                symbol="BTCUSDT", signal=0.0, price=50000.0,
                timestamp="", strategy="test",
                indicators={"rsi": 50.0, "adx": 20.0},
            )
            result = runner._process_signal("BTCUSDT", sig)
            assert result is None
            broker.execute_target_position.assert_not_called()

    def test_process_signal_opens_long(self, tmp_path):
        cfg = _make_mock_config(tmp_path)
        broker = _make_mock_broker()
        broker.get_position_pct.return_value = 0.0
        mock_trade = MagicMock()
        mock_trade.side = "BUY"
        mock_trade.qty = 0.1
        mock_trade.price = 50000.0
        mock_trade.pnl = None
        broker.execute_target_position.return_value = mock_trade

        with patch("qtrade.live.base_runner.TelegramNotifier") as mock_notifier_cls:
            mock_notifier = MagicMock(enabled=False)
            mock_notifier_cls.from_config.return_value = mock_notifier
            from qtrade.live.base_runner import BaseRunner

            class TestRunner(BaseRunner):
                def run(self):
                    pass

            runner = TestRunner(cfg, broker, "paper")

            sig = SignalResult(
                symbol="BTCUSDT", signal=1.0, price=50000.0,
                timestamp="", strategy="test",
                indicators={"rsi": 70.0, "adx": 30.0},
            )
            result = runner._process_signal("BTCUSDT", sig)
            assert result is not None
            broker.execute_target_position.assert_called_once()
            assert runner.trade_count == 1

    def test_process_signal_spot_clips_short(self, tmp_path):
        """Spot 模式下做空信號被 clip 到 0"""
        cfg = _make_mock_config(tmp_path)
        cfg.market_type_str = "spot"
        broker = _make_mock_broker()
        broker.get_position_pct.return_value = 0.0

        with patch("qtrade.live.base_runner.TelegramNotifier") as mock_notifier_cls:
            mock_notifier_cls.from_config.return_value = MagicMock(enabled=False)
            from qtrade.live.base_runner import BaseRunner

            class TestRunner(BaseRunner):
                def run(self):
                    pass

            runner = TestRunner(cfg, broker, "paper")

            sig = SignalResult(
                symbol="BTCUSDT", signal=-1.0, price=50000.0,
                timestamp="", strategy="test",
                indicators={},
            )
            result = runner._process_signal("BTCUSDT", sig)
            assert result is None  # No trade, signal clipped to 0
            broker.execute_target_position.assert_not_called()
