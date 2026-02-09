"""
Monte Carlo 模擬模組測試

測試內容：
- 配置類別的正確性
- 結果類別的不可變性
- VaR 計算的正確性
- Bootstrap 模擬的統計特性
- 路徑模擬的合理性
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from qtrade.risk.monte_carlo import (
    # 配置類
    MonteCarloConfig,
    BootstrapConfig,
    PathSimulationConfig,
    # 結果類
    VaRResult,
    MonteCarloVaRResult,
    BootstrapResult,
    StrategyBootstrapResult,
    DrawdownDistributionResult,
    PathSimulationResult,
    # 列舉
    SimulationMethod,
    VaRMethod,
    # 模擬器
    VaRCalculator,
    BootstrapSimulator,
    PathSimulator,
    DrawdownAnalyzer,
    PortfolioMonteCarloSimulator,
    MonteCarloSimulator,
    RandomGenerator,
    # 便捷函數
    monte_carlo_var,
    bootstrap_strategy_ci,
    simulate_strategy_outcomes,
)


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_returns():
    """模擬日收益率序列"""
    np.random.seed(42)
    n = 500
    returns = np.random.normal(0.0005, 0.02, n)  # 日均 0.05%，波動率 2%
    return pd.Series(returns, index=pd.date_range("2023-01-01", periods=n, freq="D"))


@pytest.fixture
def sample_trade_returns():
    """模擬交易收益率序列"""
    np.random.seed(42)
    n = 100
    # 模擬勝率 55% 的交易
    wins = np.random.uniform(0.01, 0.05, int(n * 0.55))
    losses = np.random.uniform(-0.03, -0.01, int(n * 0.45))
    returns = np.concatenate([wins, losses])
    np.random.shuffle(returns)
    return pd.Series(returns)


@pytest.fixture
def sample_portfolio_returns():
    """模擬多資產收益率"""
    np.random.seed(42)
    n = 300
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    
    # 生成相關的收益率
    cov = np.array([
        [0.0004, 0.0002, 0.0001],
        [0.0002, 0.0003, 0.0001],
        [0.0001, 0.0001, 0.0002],
    ])
    mean = [0.0005, 0.0003, 0.0002]
    
    returns = np.random.multivariate_normal(mean, cov, n)
    
    return {
        "BTCUSDT": pd.Series(returns[:, 0], index=dates),
        "ETHUSDT": pd.Series(returns[:, 1], index=dates),
        "BNBUSDT": pd.Series(returns[:, 2], index=dates),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 配置類測試
# ══════════════════════════════════════════════════════════════════════════════

class TestMonteCarloConfig:
    """MonteCarloConfig 測試"""
    
    def test_default_values(self):
        """測試默認值"""
        config = MonteCarloConfig()
        
        assert config.n_simulations == 10000
        assert config.confidence_levels == (0.95, 0.99)
        assert config.random_seed == 42
        assert config.horizon_days == 1
        assert config.annualization_factor == 252
    
    def test_custom_values(self):
        """測試自定義值"""
        config = MonteCarloConfig(
            n_simulations=5000,
            confidence_levels=(0.90, 0.95),
            random_seed=123,
        )
        
        assert config.n_simulations == 5000
        assert config.confidence_levels == (0.90, 0.95)
        assert config.random_seed == 123
    
    def test_immutability(self):
        """測試不可變性"""
        config = MonteCarloConfig()
        
        with pytest.raises(FrozenInstanceError):
            config.n_simulations = 5000


class TestBootstrapConfig:
    """BootstrapConfig 測試"""
    
    def test_default_values(self):
        """測試默認值"""
        config = BootstrapConfig()
        
        assert config.n_simulations == 10000
        assert config.block_size == 20
        assert config.replacement is True
        assert config.confidence_level == 0.95


class TestPathSimulationConfig:
    """PathSimulationConfig 測試"""
    
    def test_default_values(self):
        """測試默認值"""
        config = PathSimulationConfig()
        
        assert config.n_simulations == 1000
        assert config.n_steps == 252
        assert config.dt == 1 / 252
        assert config.method == SimulationMethod.PARAMETRIC


# ══════════════════════════════════════════════════════════════════════════════
# 結果類測試
# ══════════════════════════════════════════════════════════════════════════════

class TestVaRResult:
    """VaRResult 測試"""
    
    def test_creation(self):
        """測試建立"""
        result = VaRResult(
            var=1000.0,
            cvar=1200.0,
            confidence_level=0.95,
            method=VaRMethod.MONTE_CARLO,
            horizon_days=1,
        )
        
        assert result.var == 1000.0
        assert result.cvar == 1200.0
        assert result.confidence_level == 0.95


class TestMonteCarloVaRResult:
    """MonteCarloVaRResult 測試"""
    
    def test_get_var(self):
        """測試 get_var 方法"""
        results = (
            VaRResult(var=1000, cvar=1200, confidence_level=0.95,
                     method=VaRMethod.MONTE_CARLO, horizon_days=1),
            VaRResult(var=1500, cvar=1800, confidence_level=0.99,
                     method=VaRMethod.MONTE_CARLO, horizon_days=1),
        )
        
        mc_result = MonteCarloVaRResult(
            results=results,
            expected_return=0.1,
            expected_volatility=0.2,
            simulated_returns=np.array([0.01, -0.02, 0.03]),
            percentiles={50: 0.01},
        )
        
        assert mc_result.get_var(0.95) == 1000
        assert mc_result.get_var(0.99) == 1500
    
    def test_get_var_not_found(self):
        """測試找不到置信水平"""
        results = (
            VaRResult(var=1000, cvar=1200, confidence_level=0.95,
                     method=VaRMethod.MONTE_CARLO, horizon_days=1),
        )
        
        mc_result = MonteCarloVaRResult(
            results=results,
            expected_return=0.1,
            expected_volatility=0.2,
            simulated_returns=np.array([0.01]),
            percentiles={},
        )
        
        with pytest.raises(ValueError, match="找不到置信水平"):
            mc_result.get_var(0.99)


class TestBootstrapResult:
    """BootstrapResult 測試"""
    
    def test_confidence_interval(self):
        """測試信賴區間屬性"""
        result = BootstrapResult(
            metric_name="sharpe_ratio",
            point_estimate=1.5,
            lower_bound=1.0,
            upper_bound=2.0,
            confidence_level=0.95,
        )
        
        assert result.confidence_interval == (1.0, 2.0)


# ══════════════════════════════════════════════════════════════════════════════
# 隨機數生成器測試
# ══════════════════════════════════════════════════════════════════════════════

class TestRandomGenerator:
    """RandomGenerator 測試"""
    
    def test_reproducibility(self):
        """測試可重複性"""
        rng1 = RandomGenerator(seed=42)
        rng2 = RandomGenerator(seed=42)
        
        arr1 = rng1.normal(0, 1, 100)
        arr2 = rng2.normal(0, 1, 100)
        
        np.testing.assert_array_equal(arr1, arr2)
    
    def test_different_seeds(self):
        """測試不同種子產生不同結果"""
        rng1 = RandomGenerator(seed=42)
        rng2 = RandomGenerator(seed=123)
        
        arr1 = rng1.normal(0, 1, 100)
        arr2 = rng2.normal(0, 1, 100)
        
        assert not np.allclose(arr1, arr2)
    
    def test_choice(self):
        """測試 choice 方法"""
        rng = RandomGenerator(seed=42)
        
        choices = rng.choice(10, size=5, replace=False)
        
        assert len(choices) == 5
        assert len(set(choices)) == 5  # 無重複


# ══════════════════════════════════════════════════════════════════════════════
# VaR 計算器測試
# ══════════════════════════════════════════════════════════════════════════════

class TestVaRCalculator:
    """VaRCalculator 測試"""
    
    def test_parametric_var_basic(self, sample_returns):
        """測試基本參數化 VaR 計算"""
        config = MonteCarloConfig(n_simulations=10000, random_seed=42)
        calculator = VaRCalculator(config)
        
        result = calculator.parametric_var(sample_returns, portfolio_value=100000)
        
        # VaR 應該為正數
        assert result.get_var(0.95) > 0
        assert result.get_var(0.99) > 0
        
        # 99% VaR 應該大於 95% VaR
        assert result.get_var(0.99) > result.get_var(0.95)
        
        # CVaR 應該大於等於 VaR
        assert result.get_cvar(0.95) >= result.get_var(0.95)
    
    def test_parametric_var_reproducibility(self, sample_returns):
        """測試 VaR 計算的可重複性"""
        config = MonteCarloConfig(n_simulations=10000, random_seed=42)
        
        calc1 = VaRCalculator(config)
        calc2 = VaRCalculator(config)
        
        result1 = calc1.parametric_var(sample_returns)
        result2 = calc2.parametric_var(sample_returns)
        
        assert result1.get_var(0.95) == result2.get_var(0.95)
    
    def test_historical_var(self, sample_returns):
        """測試歷史模擬法 VaR"""
        config = MonteCarloConfig()
        calculator = VaRCalculator(config)
        
        result = calculator.historical_var(sample_returns, portfolio_value=100000)
        
        assert result.get_var(0.95) > 0
        assert result.results[0].method == VaRMethod.HISTORICAL
    
    def test_empty_returns_error(self):
        """測試空收益率序列錯誤"""
        config = MonteCarloConfig()
        calculator = VaRCalculator(config)
        
        with pytest.raises(ValueError, match="不能為空"):
            calculator.parametric_var(pd.Series(dtype=float))


# ══════════════════════════════════════════════════════════════════════════════
# Bootstrap 模擬器測試
# ══════════════════════════════════════════════════════════════════════════════

class TestBootstrapSimulator:
    """BootstrapSimulator 測試"""
    
    def test_strategy_performance_basic(self, sample_trade_returns):
        """測試基本策略績效 Bootstrap"""
        config = BootstrapConfig(n_simulations=1000, random_seed=42)
        simulator = BootstrapSimulator(config)
        
        result = simulator.strategy_performance(sample_trade_returns)
        
        assert isinstance(result, StrategyBootstrapResult)
        assert result.n_simulations == 1000
        
        # 檢查信賴區間合理性
        assert result.total_return.lower_bound < result.total_return.upper_bound
        assert result.sharpe_ratio.lower_bound < result.sharpe_ratio.upper_bound
    
    def test_strategy_performance_distributions(self, sample_trade_returns):
        """測試 Bootstrap 分布的統計特性"""
        config = BootstrapConfig(n_simulations=5000, random_seed=42)
        simulator = BootstrapSimulator(config)
        
        result = simulator.strategy_performance(sample_trade_returns)
        
        # 分布應該有足夠的變異
        assert result.win_rate.distribution.std() > 0
        
        # 中位數應該在信賴區間內
        assert result.total_return.lower_bound <= result.total_return.point_estimate
        assert result.total_return.point_estimate <= result.total_return.upper_bound
    
    def test_block_bootstrap(self, sample_returns):
        """測試 Block Bootstrap"""
        config = BootstrapConfig(n_simulations=100, block_size=10, random_seed=42)
        simulator = BootstrapSimulator(config)
        
        result = simulator.block_bootstrap(sample_returns)
        
        assert result.shape == (100, len(sample_returns))
    
    def test_empty_returns_error(self):
        """測試空收益率序列錯誤"""
        config = BootstrapConfig()
        simulator = BootstrapSimulator(config)
        
        with pytest.raises(ValueError, match="不能為空"):
            simulator.strategy_performance(pd.Series(dtype=float))


# ══════════════════════════════════════════════════════════════════════════════
# 路徑模擬器測試
# ══════════════════════════════════════════════════════════════════════════════

class TestPathSimulator:
    """PathSimulator 測試"""
    
    def test_gbm_paths_shape(self):
        """測試 GBM 路徑形狀"""
        config = PathSimulationConfig(
            n_simulations=100,
            n_steps=50,
            random_seed=42,
        )
        simulator = PathSimulator(config)
        
        result = simulator.gbm_paths(
            initial_price=100,
            mu=0.1,
            sigma=0.2,
        )
        
        # 路徑形狀: (n_simulations, n_steps + 1)
        assert result.paths.shape == (100, 51)
        
        # 初始價格應該都是 100
        np.testing.assert_array_equal(result.paths[:, 0], 100)
    
    def test_gbm_paths_positive(self):
        """測試 GBM 路徑始終為正"""
        config = PathSimulationConfig(
            n_simulations=1000,
            n_steps=252,
            random_seed=42,
        )
        simulator = PathSimulator(config)
        
        result = simulator.gbm_paths(
            initial_price=100,
            mu=-0.1,  # 負漂移
            sigma=0.3,  # 高波動
        )
        
        # GBM 路徑應該始終為正
        assert (result.paths > 0).all()
    
    def test_equity_curve_simulation(self, sample_returns):
        """測試權益曲線模擬"""
        config = PathSimulationConfig(n_simulations=100, random_seed=42)
        simulator = PathSimulator(config)
        
        result = simulator.equity_curve_simulation(
            sample_returns,
            initial_capital=100000,
        )
        
        # 初始資金應該都是 100000
        np.testing.assert_array_equal(result.paths[:, 0], 100000)
        
        # 應該有信賴區間帶
        assert 0.95 in result.confidence_bands
    
    def test_confidence_bands(self):
        """測試信賴區間帶"""
        config = PathSimulationConfig(
            n_simulations=1000,
            n_steps=100,
            random_seed=42,
        )
        simulator = PathSimulator(config)
        
        result = simulator.gbm_paths(initial_price=100, mu=0.1, sigma=0.2)
        
        # 95% 區間應該包含大部分路徑終點
        lower_95, upper_95 = result.confidence_bands[0.95]
        final_prices = result.paths[:, -1]
        
        in_band = (final_prices >= lower_95[-1]) & (final_prices <= upper_95[-1])
        assert in_band.mean() >= 0.90  # 至少 90% 在區間內


# ══════════════════════════════════════════════════════════════════════════════
# Drawdown 分析器測試
# ══════════════════════════════════════════════════════════════════════════════

class TestDrawdownAnalyzer:
    """DrawdownAnalyzer 測試"""
    
    def test_analyze_basic(self):
        """測試基本 Drawdown 分析"""
        # 建立簡單的權益曲線
        equity_curves = np.array([
            [100, 110, 105, 115, 120],  # 最大 DD: (110-105)/110 ≈ 4.5%
            [100, 90, 85, 95, 100],     # 最大 DD: (100-85)/100 = 15%
        ])
        
        analyzer = DrawdownAnalyzer()
        result = analyzer.analyze(equity_curves)
        
        assert len(result.max_drawdowns) == 2
        assert result.max_drawdowns[1] > result.max_drawdowns[0]  # 第二條 DD 更大
    
    def test_analyze_percentiles(self):
        """測試百分位數計算"""
        np.random.seed(42)
        
        # 建立模擬權益曲線
        n_sims = 100
        n_steps = 50
        equity_curves = np.zeros((n_sims, n_steps))
        equity_curves[:, 0] = 100
        
        for i in range(1, n_steps):
            returns = np.random.normal(0.001, 0.02, n_sims)
            equity_curves[:, i] = equity_curves[:, i-1] * (1 + returns)
        
        analyzer = DrawdownAnalyzer()
        result = analyzer.analyze(equity_curves)
        
        # 百分位數應該遞增
        assert result.percentiles["max_drawdown"][25] < result.percentiles["max_drawdown"][75]
    
    def test_probability_gt_threshold(self):
        """測試超越機率計算"""
        np.random.seed(42)
        
        # 建立有明顯 Drawdown 的權益曲線
        equity_curves = np.array([
            [100, 95, 90, 85, 80],  # DD = 20%
            [100, 98, 96, 94, 92],  # DD = 8%
            [100, 75, 70, 65, 60],  # DD = 40%
            [100, 99, 98, 97, 96],  # DD = 4%
        ])
        
        analyzer = DrawdownAnalyzer()
        result = analyzer.analyze(equity_curves)
        
        # 2/4 = 50% 超過 10%
        assert 0.4 <= result.probability_gt_threshold[0.10] <= 0.6


# ══════════════════════════════════════════════════════════════════════════════
# 組合 Monte Carlo 測試
# ══════════════════════════════════════════════════════════════════════════════

class TestPortfolioMonteCarloSimulator:
    """PortfolioMonteCarloSimulator 測試"""
    
    def test_simulate_basic(self, sample_portfolio_returns):
        """測試基本組合 Monte Carlo"""
        config = MonteCarloConfig(n_simulations=5000, random_seed=42)
        simulator = PortfolioMonteCarloSimulator(config)
        
        weights = {"BTCUSDT": 0.5, "ETHUSDT": 0.3, "BNBUSDT": 0.2}
        
        result = simulator.simulate(sample_portfolio_returns, weights)
        
        assert result.get_var(0.95) > 0
        assert result.get_cvar(0.95) >= result.get_var(0.95)
    
    def test_simulate_weight_normalization(self, sample_portfolio_returns):
        """測試權重歸一化"""
        config = MonteCarloConfig(n_simulations=1000, random_seed=42)
        simulator = PortfolioMonteCarloSimulator(config)
        
        # 權重不等於 1
        weights = {"BTCUSDT": 1.0, "ETHUSDT": 1.0, "BNBUSDT": 1.0}
        
        # 應該正常執行（內部會歸一化）
        result = simulator.simulate(sample_portfolio_returns, weights)
        
        assert result is not None


# ══════════════════════════════════════════════════════════════════════════════
# Facade 模擬器測試
# ══════════════════════════════════════════════════════════════════════════════

class TestMonteCarloSimulator:
    """MonteCarloSimulator (Facade) 測試"""
    
    def test_calculate_var(self, sample_returns):
        """測試 calculate_var 方法"""
        simulator = MonteCarloSimulator()
        
        result = simulator.calculate_var(sample_returns)
        
        assert result.get_var(0.95) > 0
    
    def test_bootstrap_performance(self, sample_trade_returns):
        """測試 bootstrap_performance 方法"""
        simulator = MonteCarloSimulator()
        
        result = simulator.bootstrap_performance(sample_trade_returns)
        
        assert isinstance(result, StrategyBootstrapResult)
    
    def test_simulate_paths(self, sample_returns):
        """測試 simulate_paths 方法"""
        simulator = MonteCarloSimulator()
        
        result = simulator.simulate_paths(sample_returns, initial_capital=100000)
        
        assert result.paths.shape[1] == len(sample_returns) + 1
    
    def test_analyze_drawdown(self, sample_returns):
        """測試 analyze_drawdown 方法"""
        simulator = MonteCarloSimulator()
        
        result = simulator.analyze_drawdown(sample_returns)
        
        assert isinstance(result, DrawdownDistributionResult)


# ══════════════════════════════════════════════════════════════════════════════
# 便捷函數測試
# ══════════════════════════════════════════════════════════════════════════════

class TestConvenienceFunctions:
    """便捷函數測試"""
    
    def test_monte_carlo_var(self, sample_returns):
        """測試 monte_carlo_var 便捷函數"""
        var = monte_carlo_var(
            sample_returns,
            confidence_level=0.95,
            n_simulations=5000,
            portfolio_value=100000,
        )
        
        assert isinstance(var, float)
        assert var > 0
    
    def test_bootstrap_strategy_ci(self, sample_trade_returns):
        """測試 bootstrap_strategy_ci 便捷函數"""
        ci = bootstrap_strategy_ci(
            sample_trade_returns,
            confidence=0.95,
            n_simulations=1000,
        )
        
        assert "total_return" in ci
        assert "sharpe_ratio" in ci
        assert "max_drawdown" in ci
        assert "win_rate" in ci
        
        # 檢查格式: (lower, median, upper)
        assert len(ci["total_return"]) == 3
    
    def test_simulate_strategy_outcomes(self, sample_returns):
        """測試 simulate_strategy_outcomes 便捷函數"""
        outcomes = simulate_strategy_outcomes(
            sample_returns,
            n_simulations=1000,
        )
        
        assert "final_return_distribution" in outcomes
        assert "max_drawdown_distribution" in outcomes
        assert "probability_of_loss" in outcomes
        assert "probability_of_drawdown_gt_20" in outcomes
        
        # 機率應該在 0-1 之間
        assert 0 <= outcomes["probability_of_loss"] <= 1


# ══════════════════════════════════════════════════════════════════════════════
# 列舉測試
# ══════════════════════════════════════════════════════════════════════════════

class TestEnums:
    """列舉測試"""
    
    def test_simulation_method_values(self):
        """測試 SimulationMethod 值"""
        assert SimulationMethod.PARAMETRIC.value == "parametric"
        assert SimulationMethod.HISTORICAL.value == "historical"
        assert SimulationMethod.BOOTSTRAP.value == "bootstrap"
        assert SimulationMethod.BLOCK_BOOTSTRAP.value == "block_bootstrap"
    
    def test_var_method_values(self):
        """測試 VaRMethod 值"""
        assert VaRMethod.PARAMETRIC.value == "parametric"
        assert VaRMethod.HISTORICAL.value == "historical"
        assert VaRMethod.MONTE_CARLO.value == "monte_carlo"
