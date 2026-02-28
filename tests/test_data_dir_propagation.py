"""
data_dir 傳播測試

確認 data_dir 參數能從頂層驗證函數一路穿透到 run_symbol_backtest。
使用 mock 替換 run_symbol_backtest 並驗證 data_dir kwarg 正確到達。
"""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ══════════════════════════════════════════════════════════════════════════════
# 測試 run_symbol_backtest 接受 data_dir
# ══════════════════════════════════════════════════════════════════════════════

class TestRunSymbolBacktestSignature:
    """確認 run_symbol_backtest 的 data_dir 參數存在"""

    def test_has_data_dir_param(self):
        from qtrade.backtest.run_backtest import run_symbol_backtest
        sig = inspect.signature(run_symbol_backtest)
        assert "data_dir" in sig.parameters
        param = sig.parameters["data_dir"]
        assert param.default is None


# ══════════════════════════════════════════════════════════════════════════════
# 測試 walk_forward_analysis 傳遞 data_dir
# ══════════════════════════════════════════════════════════════════════════════

class TestWalkForwardDataDirPropagation:
    """確認 walk_forward_analysis 把 data_dir 傳到 run_symbol_backtest"""

    def test_walk_forward_passes_data_dir(self, tmp_path):
        """WFA 呼叫 run_symbol_backtest 時應攜帶 data_dir"""
        # 建立假的 parquet 資料
        n = 2000
        dates = pd.date_range("2022-01-01", periods=n, freq="1h")
        df = pd.DataFrame({
            "open": np.random.uniform(100, 200, n),
            "high": np.random.uniform(200, 300, n),
            "low": np.random.uniform(50, 100, n),
            "close": np.random.uniform(100, 200, n),
            "volume": np.random.uniform(1000, 5000, n),
        }, index=dates)
        data_path = tmp_path / "test.parquet"
        df.to_parquet(data_path)

        sentinel_data_dir = tmp_path / "sentinel_data_dir"

        # Mock BacktestResult
        mock_result = MagicMock()
        mock_result.pf = MagicMock()
        mock_result.pf.stats.return_value = {
            "Sharpe Ratio": 1.0,
            "Total Return [%]": 10.0,
            "Max Drawdown [%]": -5.0,
            "Win Rate [%]": 55.0,
        }
        mock_result.stats = mock_result.pf.stats()
        mock_result.adjusted_stats = None

        with patch(
            "qtrade.backtest.run_backtest.run_symbol_backtest",
            return_value=mock_result,
        ) as mock_bt:
            from qtrade.validation.walk_forward import walk_forward_analysis

            walk_forward_analysis(
                symbol="BTCUSDT",
                data_path=data_path,
                cfg={"strategy_name": "test", "fee_bps": 10},
                n_splits=2,
                data_dir=sentinel_data_dir,
            )

            # 每次呼叫 run_symbol_backtest 都應帶 data_dir
            assert mock_bt.call_count >= 2  # 至少 train + test per split
            for call in mock_bt.call_args_list:
                assert call.kwargs.get("data_dir") == sentinel_data_dir


# ══════════════════════════════════════════════════════════════════════════════
# 測試 kelly_backtest_comparison 接受 data_dir
# ══════════════════════════════════════════════════════════════════════════════

class TestKellyDataDirSignature:
    """確認 kelly_backtest_comparison 接受 data_dir"""

    def test_has_data_dir_param(self):
        from qtrade.backtest.kelly_validation import kelly_backtest_comparison
        sig = inspect.signature(kelly_backtest_comparison)
        assert "data_dir" in sig.parameters
        param = sig.parameters["data_dir"]
        assert param.default is None


# ══════════════════════════════════════════════════════════════════════════════
# 測試 leave_one_asset_out 傳遞 data_dir
# ══════════════════════════════════════════════════════════════════════════════

class TestLOAODataDirPropagation:
    """確認 leave_one_asset_out 把 data_dir 傳給 BacktestFunction"""

    def test_loao_passes_data_dir(self):
        """LOAO 呼叫 backtest_func 時應攜帶 data_dir"""
        from qtrade.validation.cross_asset import leave_one_asset_out

        sentinel = Path("/sentinel/data/dir")
        calls_received = []

        def tracking_backtest(
            symbol, data_path, cfg, strategy_name=None, data_dir=None
        ):
            calls_received.append({"symbol": symbol, "data_dir": data_dir})
            return {
                "stats": {
                    "Sharpe Ratio": 1.0,
                    "Total Return [%]": 10.0,
                    "Max Drawdown [%]": -5.0,
                    "Win Rate [%]": 55.0,
                }
            }

        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        data_paths = {s: Path(f"/fake/{s}.parquet") for s in symbols}

        result = leave_one_asset_out(
            symbols=symbols,
            data_paths=data_paths,
            cfg={"strategy_name": "test"},
            backtest_func=tracking_backtest,
            parallel=False,
            data_dir=sentinel,
        )

        # 所有 backtest 呼叫都應帶 sentinel data_dir
        assert len(calls_received) > 0
        for call in calls_received:
            assert call["data_dir"] == sentinel


# ══════════════════════════════════════════════════════════════════════════════
# 測試 market_regime_validation 接受 data_dir
# ══════════════════════════════════════════════════════════════════════════════

class TestMarketRegimeDataDirSignature:
    """確認 market_regime_validation 接受 data_dir"""

    def test_has_data_dir_param(self):
        from qtrade.validation.cross_asset import market_regime_validation
        sig = inspect.signature(market_regime_validation)
        assert "data_dir" in sig.parameters


# ══════════════════════════════════════════════════════════════════════════════
# 測試 resolve_kline_path
# ══════════════════════════════════════════════════════════════════════════════

class TestResolveKlinePath:
    """確認 AppConfig.resolve_kline_path 產出正確路徑"""

    def test_resolve_kline_path_format(self):
        """路徑格式：data_dir / binance / market_type / interval / symbol.parquet"""
        from qtrade.config import AppConfig

        # 建構最小 AppConfig（使用 load_config 會需要檔案，改用 mock）
        cfg = MagicMock(spec=AppConfig)
        cfg.data_dir = Path("/data")
        cfg.market_type_str = "futures"
        cfg.market = MagicMock()
        cfg.market.interval = "1h"

        # 直接呼叫 AppConfig 的方法（因為是 mock，需要手動綁定）
        result = AppConfig.resolve_kline_path(cfg, "BTCUSDT")
        expected = Path("/data/binance/futures/1h/BTCUSDT.parquet")
        assert result == expected
