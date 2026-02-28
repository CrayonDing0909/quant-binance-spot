"""
Cross-Asset 驗證模組測試

測試內容：
- 配置類別的正確性
- 結果類別的不可變性
- 驗證器的核心邏輯
- 便捷函數的介面
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from qtrade.validation import (
    # 配置類
    CrossAssetValidationConfig,
    CorrelationStratifiedConfig,
    MarketRegimeConfig,
    # 結果類
    AssetValidationResult,
    CrossAssetValidationResult,
    # 列舉
    ValidationMethod,
    MarketRegimeIndicator,
    RobustnessLevel,
    # 驗證器
    LeaveOneAssetOutValidator,
    ValidationResultAnalyzer,
    # 便捷函數
    leave_one_asset_out,
)


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def mock_backtest_func():
    """模擬回測函數"""
    def _backtest(symbol: str, data_path: Path, cfg: dict, strategy_name=None, data_dir=None):
        # 根據 symbol 返回不同結果，模擬真實情況
        sharpe_map = {
            "BTCUSDT": 1.5,
            "ETHUSDT": 1.2,
            "BNBUSDT": 0.8,
            "SOLUSDT": 1.0,
            "ADAUSDT": 0.5,
        }
        sharpe = sharpe_map.get(symbol, 1.0)
        
        return {
            "stats": {
                "Sharpe Ratio": sharpe,
                "Total Return [%]": sharpe * 20,
                "Max Drawdown [%]": -15,
                "Win Rate [%]": 55,
            }
        }
    return _backtest


@pytest.fixture
def mock_data_loader():
    """模擬數據載入函數"""
    def _loader(data_path: Path):
        n = 1000
        dates = pd.date_range("2023-01-01", periods=n, freq="1h")
        
        # 生成模擬價格數據
        np.random.seed(42)
        returns = np.random.normal(0.0001, 0.02, n)
        close = 100 * np.cumprod(1 + returns)
        
        return pd.DataFrame({
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.random.uniform(1000, 10000, n),
        }, index=dates)
    return _loader


@pytest.fixture
def sample_symbols():
    """測試用資產列表"""
    return ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT"]


@pytest.fixture
def sample_data_paths(sample_symbols, tmp_path):
    """測試用數據路徑"""
    return {s: tmp_path / f"{s}.parquet" for s in sample_symbols}


@pytest.fixture
def sample_config():
    """測試用策略配置"""
    return {
        "strategy_name": "test_strategy",
        "strategy_params": {},
        "initial_cash": 10000,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 配置類測試
# ══════════════════════════════════════════════════════════════════════════════

class TestCrossAssetValidationConfig:
    """CrossAssetValidationConfig 測試"""
    
    def test_default_values(self):
        """測試默認值"""
        config = CrossAssetValidationConfig()
        
        assert config.min_bars_per_asset == 500
        assert config.min_assets == 3
        assert config.parallel is True
        assert config.max_workers == 4
        assert config.degradation_threshold == 0.3
        assert config.severe_degradation_threshold == 0.5
    
    def test_custom_values(self):
        """測試自定義值"""
        config = CrossAssetValidationConfig(
            min_bars_per_asset=1000,
            parallel=False,
        )
        
        assert config.min_bars_per_asset == 1000
        assert config.parallel is False
    
    def test_immutability(self):
        """測試不可變性"""
        config = CrossAssetValidationConfig()
        
        with pytest.raises(FrozenInstanceError):
            config.min_bars_per_asset = 100


class TestCorrelationStratifiedConfig:
    """CorrelationStratifiedConfig 測試"""
    
    def test_default_values(self):
        """測試默認值"""
        config = CorrelationStratifiedConfig()
        
        assert config.n_groups == 3
        assert config.clustering_method == "ward"


class TestMarketRegimeConfig:
    """MarketRegimeConfig 測試"""
    
    def test_default_values(self):
        """測試默認值"""
        config = MarketRegimeConfig()
        
        assert config.indicator == MarketRegimeIndicator.VOLATILITY
        assert config.lookback_period == 20


# ══════════════════════════════════════════════════════════════════════════════
# 結果類測試
# ══════════════════════════════════════════════════════════════════════════════

class TestAssetValidationResult:
    """AssetValidationResult 測試"""
    
    def test_creation(self):
        """測試建立"""
        result = AssetValidationResult(
            asset="BTCUSDT",
            train_sharpe=1.5,
            test_sharpe=1.2,
            train_return=30.0,
            test_return=25.0,
            train_drawdown=10.0,
            test_drawdown=12.0,
            sharpe_degradation=0.2,
            return_degradation=0.17,
        )
        
        assert result.asset == "BTCUSDT"
        assert result.sharpe_degradation == 0.2
    
    def test_is_overfitted_true(self):
        """測試過擬合判斷（True）"""
        result = AssetValidationResult(
            asset="BTCUSDT",
            train_sharpe=2.0,
            test_sharpe=0.5,
            train_return=40.0,
            test_return=10.0,
            train_drawdown=10.0,
            test_drawdown=20.0,
            sharpe_degradation=0.75,  # > 0.5
            return_degradation=0.75,
        )
        
        assert result.is_overfitted is True
    
    def test_is_overfitted_false(self):
        """測試過擬合判斷（False）"""
        result = AssetValidationResult(
            asset="BTCUSDT",
            train_sharpe=1.5,
            test_sharpe=1.3,
            train_return=30.0,
            test_return=26.0,
            train_drawdown=10.0,
            test_drawdown=11.0,
            sharpe_degradation=0.13,  # < 0.5
            return_degradation=0.13,
        )
        
        assert result.is_overfitted is False


class TestCrossAssetValidationResult:
    """CrossAssetValidationResult 測試"""
    
    def test_to_dataframe(self):
        """測試轉換為 DataFrame"""
        asset_results = (
            AssetValidationResult(
                asset="BTCUSDT",
                train_sharpe=1.5, test_sharpe=1.2,
                train_return=30, test_return=25,
                train_drawdown=10, test_drawdown=12,
                sharpe_degradation=0.2, return_degradation=0.17,
            ),
            AssetValidationResult(
                asset="ETHUSDT",
                train_sharpe=1.3, test_sharpe=1.0,
                train_return=26, test_return=20,
                train_drawdown=12, test_drawdown=15,
                sharpe_degradation=0.23, return_degradation=0.23,
            ),
        )
        
        result = CrossAssetValidationResult(
            method=ValidationMethod.LEAVE_ONE_OUT,
            asset_results=asset_results,
            avg_train_sharpe=1.4,
            avg_test_sharpe=1.1,
            avg_sharpe_degradation=0.215,
            std_sharpe_degradation=0.02,
            overfitted_assets=(),
            robustness_level=RobustnessLevel.ROBUST,
            warnings=(),
        )
        
        df = result.to_dataframe()
        
        assert len(df) == 2
        assert "asset" in df.columns
        assert "train_sharpe" in df.columns
        assert df.iloc[0]["asset"] == "BTCUSDT"


# ══════════════════════════════════════════════════════════════════════════════
# 驗證器測試
# ══════════════════════════════════════════════════════════════════════════════

class TestLeaveOneAssetOutValidator:
    """LeaveOneAssetOutValidator 測試"""
    
    def test_validate_minimum_assets(
        self,
        mock_backtest_func,
        mock_data_loader,
        sample_config,
    ):
        """測試最少資產數量檢查"""
        validator = LeaveOneAssetOutValidator(
            mock_backtest_func,
            mock_data_loader,
        )
        
        # 只有 2 個資產，應該失敗
        with pytest.raises(ValueError, match="至少需要"):
            validator.validate(
                symbols=["BTCUSDT", "ETHUSDT"],
                data_paths={"BTCUSDT": Path("a.parquet"), "ETHUSDT": Path("b.parquet")},
                cfg=sample_config,
            )
    
    def test_validate_success(
        self,
        mock_backtest_func,
        mock_data_loader,
        sample_symbols,
        sample_data_paths,
        sample_config,
    ):
        """測試成功驗證"""
        config = CrossAssetValidationConfig(parallel=False)
        validator = LeaveOneAssetOutValidator(
            mock_backtest_func,
            mock_data_loader,
            config,
        )
        
        result = validator.validate(
            symbols=sample_symbols,
            data_paths=sample_data_paths,
            cfg=sample_config,
        )
        
        assert result.method == ValidationMethod.LEAVE_ONE_OUT
        assert len(result.asset_results) == len(sample_symbols)
        assert result.robustness_level is not None
    
    def test_calculate_degradation(
        self,
        mock_backtest_func,
        mock_data_loader,
    ):
        """測試績效衰退計算"""
        validator = LeaveOneAssetOutValidator(
            mock_backtest_func,
            mock_data_loader,
        )
        
        # 正常情況
        assert validator._calculate_degradation(1.0, 0.8) == pytest.approx(0.2)
        
        # 訓練值為 0
        assert validator._calculate_degradation(0.0, 0.5) == 0.0
        
        # 負衰退（測試比訓練好）
        assert validator._calculate_degradation(1.0, 1.2) == pytest.approx(-0.2)


class TestValidationResultAnalyzer:
    """ValidationResultAnalyzer 測試"""
    
    def test_summarize(self):
        """測試結果摘要"""
        result = CrossAssetValidationResult(
            method=ValidationMethod.LEAVE_ONE_OUT,
            asset_results=(),
            avg_train_sharpe=1.5,
            avg_test_sharpe=1.2,
            avg_sharpe_degradation=0.2,
            std_sharpe_degradation=0.05,
            overfitted_assets=("ADAUSDT",),
            robustness_level=RobustnessLevel.MODERATE,
            warnings=("警告訊息",),
        )
        
        summary = ValidationResultAnalyzer.summarize(result)
        
        assert summary["method"] == "leave_one_out"
        assert summary["robustness"] == "moderate"
        assert "ADAUSDT" in summary["overfitted_assets"]
    
    def test_is_strategy_robust(self):
        """測試策略穩健性判斷"""
        # 穩健
        robust_result = CrossAssetValidationResult(
            method=ValidationMethod.LEAVE_ONE_OUT,
            asset_results=(),
            avg_train_sharpe=1.5,
            avg_test_sharpe=1.3,
            avg_sharpe_degradation=0.13,
            std_sharpe_degradation=0.03,
            overfitted_assets=(),
            robustness_level=RobustnessLevel.ROBUST,
            warnings=(),
        )
        assert ValidationResultAnalyzer.is_strategy_robust(robust_result) is True
        
        # 過擬合
        overfitted_result = CrossAssetValidationResult(
            method=ValidationMethod.LEAVE_ONE_OUT,
            asset_results=(),
            avg_train_sharpe=2.0,
            avg_test_sharpe=0.5,
            avg_sharpe_degradation=0.75,
            std_sharpe_degradation=0.2,
            overfitted_assets=("BTCUSDT", "ETHUSDT"),
            robustness_level=RobustnessLevel.OVERFITTED,
            warnings=(),
        )
        assert ValidationResultAnalyzer.is_strategy_robust(overfitted_result) is False
    
    def test_get_recommendations(self):
        """測試建議生成"""
        # 過擬合
        result = CrossAssetValidationResult(
            method=ValidationMethod.LEAVE_ONE_OUT,
            asset_results=(),
            avg_train_sharpe=2.0,
            avg_test_sharpe=0.5,
            avg_sharpe_degradation=0.75,
            std_sharpe_degradation=0.5,
            overfitted_assets=("BTCUSDT",),
            robustness_level=RobustnessLevel.OVERFITTED,
            warnings=(),
        )
        
        recs = ValidationResultAnalyzer.get_recommendations(result)
        
        assert len(recs) > 0
        assert any("過擬合" in r for r in recs)


# ══════════════════════════════════════════════════════════════════════════════
# 便捷函數測試
# ══════════════════════════════════════════════════════════════════════════════

class TestConvenienceFunctions:
    """便捷函數測試"""
    
    def test_leave_one_asset_out_with_custom_functions(
        self,
        mock_backtest_func,
        mock_data_loader,
        sample_symbols,
        sample_data_paths,
        sample_config,
    ):
        """測試 leave_one_asset_out 便捷函數"""
        result = leave_one_asset_out(
            symbols=sample_symbols,
            data_paths=sample_data_paths,
            cfg=sample_config,
            backtest_func=mock_backtest_func,
            data_loader=mock_data_loader,
            parallel=False,
        )
        
        assert isinstance(result, CrossAssetValidationResult)
        assert result.method == ValidationMethod.LEAVE_ONE_OUT


# ══════════════════════════════════════════════════════════════════════════════
# 列舉測試
# ══════════════════════════════════════════════════════════════════════════════

class TestEnums:
    """列舉測試"""
    
    def test_validation_method_values(self):
        """測試 ValidationMethod 值"""
        assert ValidationMethod.LEAVE_ONE_OUT.value == "leave_one_out"
        assert ValidationMethod.CORRELATION_STRATIFIED.value == "correlation_stratified"
        assert ValidationMethod.MARKET_REGIME.value == "market_regime"
    
    def test_market_regime_indicator_values(self):
        """測試 MarketRegimeIndicator 值"""
        assert MarketRegimeIndicator.VOLATILITY.value == "volatility"
        assert MarketRegimeIndicator.TREND.value == "trend"
        assert MarketRegimeIndicator.MOMENTUM.value == "momentum"
    
    def test_robustness_level_values(self):
        """測試 RobustnessLevel 值"""
        assert RobustnessLevel.ROBUST.value == "robust"
        assert RobustnessLevel.MODERATE.value == "moderate"
        assert RobustnessLevel.WEAK.value == "weak"
        assert RobustnessLevel.OVERFITTED.value == "overfitted"
