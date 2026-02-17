"""
回測一致性測試

確保：
1. BacktestResult dataclass 正確返回
2. 配置驗證能偵測遺漏的成本模型
3. run_symbol_backtest 是唯一的 VBT 入口
4. 成本模型旗標正確傳遞

防止「快樂表」問題再次發生。
"""
from __future__ import annotations

import warnings
import pytest
import pandas as pd
import numpy as np

from qtrade.backtest.run_backtest import (
    BacktestResult,
    validate_backtest_config,
)


# ══════════════════════════════════════════════════════════════
# 1. validate_backtest_config
# ══════════════════════════════════════════════════════════════


class TestValidateBacktestConfig:
    """配置安全驗證測試"""

    def test_spot_no_warnings(self):
        """Spot 模式不應觸發 funding rate 警告"""
        cfg = {"market_type": "spot"}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_backtest_config(cfg)
            assert len(w) == 0, f"Spot 不應有警告，但得到 {len(w)} 個"

    def test_futures_missing_funding_rate_warns(self):
        """Futures 缺少 funding_rate 配置應警告"""
        cfg = {"market_type": "futures"}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_backtest_config(cfg)
            msgs = [str(x.message) for x in w]
            assert any("funding_rate" in m for m in msgs), (
                f"應警告缺少 funding_rate，但警告為: {msgs}"
            )

    def test_futures_missing_slippage_model_warns(self):
        """Futures 缺少 slippage_model 配置應警告"""
        cfg = {
            "market_type": "futures",
            "funding_rate": {"enabled": True},
        }
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_backtest_config(cfg)
            msgs = [str(x.message) for x in w]
            assert any("slippage_model" in m for m in msgs), (
                f"應警告缺少 slippage_model，但警告為: {msgs}"
            )

    def test_futures_disabled_funding_rate_warns(self):
        """Futures 明確關閉 funding_rate 應警告"""
        cfg = {
            "market_type": "futures",
            "funding_rate": {"enabled": False},
            "slippage_model": {"enabled": True},
        }
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_backtest_config(cfg)
            msgs = [str(x.message) for x in w]
            assert any("funding_rate" in m.lower() or "enabled=false" in m.lower() for m in msgs), (
                f"應警告 FR 被關閉，但警告為: {msgs}"
            )

    def test_futures_all_enabled_no_critical_warnings(self):
        """Futures 全開時不應有 UserWarning"""
        cfg = {
            "market_type": "futures",
            "funding_rate": {"enabled": True},
            "slippage_model": {"enabled": True},
        }
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_backtest_config(cfg)
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            assert len(user_warnings) == 0, (
                f"全開時不應有 UserWarning，但得到: "
                f"{[str(x.message) for x in user_warnings]}"
            )


# ══════════════════════════════════════════════════════════════
# 2. BacktestResult dataclass
# ══════════════════════════════════════════════════════════════


class TestBacktestResult:
    """BacktestResult 功能測試"""

    def _make_mock_result(self, **overrides) -> BacktestResult:
        """建立 mock BacktestResult"""
        defaults = dict(
            pf=None,
            pf_bh=None,
            stats=pd.Series({
                "Total Return [%]": 100.0,
                "Sharpe Ratio": 1.5,
                "Max Drawdown [%]": -20.0,
            }),
            df=pd.DataFrame({"close": [100, 110]}),
            pos=pd.Series([0.0, 1.0]),
            funding_cost=None,
            slippage_result=None,
            adjusted_stats=None,
            adjusted_equity=None,
            funding_rate_enabled=False,
            slippage_model_enabled=False,
        )
        defaults.update(overrides)
        return BacktestResult(**defaults)

    def test_total_return_uses_raw_when_no_adjustment(self):
        """沒有 adjusted_stats 時使用原始 stats"""
        res = self._make_mock_result()
        assert res.total_return_pct() == 100.0

    def test_total_return_uses_adjusted_when_available(self):
        """有 adjusted_stats 時使用調整後數值"""
        res = self._make_mock_result(
            adjusted_stats={"Total Return [%]": 50.0, "Sharpe Ratio": 0.8, "Max Drawdown [%]": 25.0}
        )
        assert res.total_return_pct() == 50.0
        assert res.sharpe() == 0.8

    def test_cost_summary_off(self):
        """成本模型關閉時的摘要"""
        res = self._make_mock_result()
        summary = res.cost_summary()
        assert "FR: OFF" in summary
        assert "Slip: fixed" in summary

    def test_cost_summary_on(self):
        """成本模型開啟時的摘要"""
        res = self._make_mock_result(
            funding_rate_enabled=True,
            slippage_model_enabled=True,
        )
        summary = res.cost_summary()
        assert "FR:" in summary
        assert "Slip:" in summary

    def test_is_dataclass_not_dict(self):
        """BacktestResult 是 dataclass，不是 dict"""
        res = self._make_mock_result()
        assert not isinstance(res, dict)
        assert hasattr(res, "pf")
        assert hasattr(res, "cost_summary")
        # 不能用 dict-style 存取
        with pytest.raises(TypeError):
            _ = res["pf"]


# ══════════════════════════════════════════════════════════════
# 3. to_backtest_dict 包含成本模型
# ══════════════════════════════════════════════════════════════


class TestToBacktestDict:
    """確認 AppConfig.to_backtest_dict() 包含成本模型配置"""

    def test_to_backtest_dict_has_funding_rate(self, tmp_path):
        """to_backtest_dict 必須包含 funding_rate"""
        from qtrade.config import load_config
        config_content = """
market:
  symbols: ["ETHUSDT"]
  interval: "1h"
  start: "2024-01-01"
  end: null
  market_type: "futures"
futures:
  leverage: 5
  margin_type: "ISOLATED"
  position_mode: "ONE_WAY"
  direction: "both"
strategy:
  name: "rsi_adx_atr"
  params:
    rsi_period: 14
backtest:
  initial_cash: 10000
  fee_bps: 4
  slippage_bps: 3
  trade_on: "next_open"
  funding_rate:
    enabled: true
    default_rate_8h: 0.0001
  slippage_model:
    enabled: true
    base_bps: 2
"""
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(config_content)

        cfg = load_config(str(config_path))
        bt_dict = cfg.to_backtest_dict()

        # 必須有成本模型配置
        assert "funding_rate" in bt_dict, "to_backtest_dict 缺少 funding_rate！"
        assert bt_dict["funding_rate"]["enabled"] is True
        assert "slippage_model" in bt_dict, "to_backtest_dict 缺少 slippage_model！"
        assert bt_dict["slippage_model"]["enabled"] is True

    def test_to_backtest_dict_has_market_type(self, tmp_path):
        """to_backtest_dict 必須包含 market_type"""
        from qtrade.config import load_config
        config_content = """
market:
  symbols: ["ETHUSDT"]
  interval: "1h"
  start: "2024-01-01"
  end: null
  market_type: "futures"
futures:
  leverage: 5
  direction: "both"
strategy:
  name: "rsi_adx_atr"
  params: {}
backtest:
  initial_cash: 10000
  fee_bps: 4
  slippage_bps: 3
  trade_on: "next_open"
"""
        config_path = tmp_path / "test_config2.yaml"
        config_path.write_text(config_content)

        cfg = load_config(str(config_path))
        bt_dict = cfg.to_backtest_dict()

        assert bt_dict["market_type"] == "futures"
        assert bt_dict["direction"] == "both"
        assert bt_dict["leverage"] == 5
