"""
Monte Carlo 模擬模組

提供多種蒙特卡羅模擬功能：
- 參數化 VaR / CVaR 計算
- Bootstrap 重抽樣
- 權益曲線路徑模擬
- Drawdown 分布分析
- 組合風險模擬

設計原則：
- 策略模式：不同的模擬方法可插拔替換
- 結果不可變：使用 frozen dataclass
- 可重複性：支援隨機種子設定
- 效能考量：支援向量化運算
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# 列舉與常數
# ══════════════════════════════════════════════════════════════════════════════

class SimulationMethod(Enum):
    """模擬方法類型"""
    PARAMETRIC = "parametric"  # 參數化（假設正態分布）
    HISTORICAL = "historical"  # 歷史模擬
    BOOTSTRAP = "bootstrap"  # Bootstrap 重抽樣
    BLOCK_BOOTSTRAP = "block_bootstrap"  # 區塊 Bootstrap


class VaRMethod(Enum):
    """VaR 計算方法"""
    PARAMETRIC = "parametric"
    HISTORICAL = "historical"
    MONTE_CARLO = "monte_carlo"


# ══════════════════════════════════════════════════════════════════════════════
# 配置類別
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class MonteCarloConfig:
    """
    Monte Carlo 模擬配置
    
    Attributes:
        n_simulations: 模擬次數
        confidence_levels: 置信水平列表
        random_seed: 隨機種子（None 表示不固定）
        horizon_days: 風險評估期間（天）
        annualization_factor: 年化因子（252 為交易日）
    """
    n_simulations: int = 10000
    confidence_levels: Tuple[float, ...] = (0.95, 0.99)
    random_seed: Optional[int] = 42
    horizon_days: int = 1
    annualization_factor: int = 252


@dataclass(frozen=True)
class BootstrapConfig:
    """
    Bootstrap 模擬配置
    
    Attributes:
        n_simulations: 模擬次數
        block_size: 區塊大小（用於 Block Bootstrap）
        replacement: 是否放回抽樣
        confidence_level: 信賴區間置信水平
    """
    n_simulations: int = 10000
    block_size: int = 20
    replacement: bool = True
    confidence_level: float = 0.95
    random_seed: Optional[int] = 42


@dataclass(frozen=True)
class PathSimulationConfig:
    """
    路徑模擬配置
    
    Attributes:
        n_simulations: 模擬路徑數量
        n_steps: 模擬步數
        dt: 時間步長（1/252 為日頻）
        method: 模擬方法
    """
    n_simulations: int = 1000
    n_steps: int = 252
    dt: float = 1 / 252
    method: SimulationMethod = SimulationMethod.PARAMETRIC
    random_seed: Optional[int] = 42


# ══════════════════════════════════════════════════════════════════════════════
# 結果類別
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class VaRResult:
    """
    VaR 計算結果
    
    Attributes:
        var: VaR 值（正數，表示潛在損失）
        cvar: CVaR / Expected Shortfall
        confidence_level: 置信水平
        method: 計算方法
        horizon_days: 評估期間
    """
    var: float
    cvar: float
    confidence_level: float
    method: VaRMethod
    horizon_days: int


@dataclass(frozen=True)
class MonteCarloVaRResult:
    """
    Monte Carlo VaR 完整結果
    
    包含多個置信水平的 VaR/CVaR 結果。
    """
    results: Tuple[VaRResult, ...]
    expected_return: float
    expected_volatility: float
    simulated_returns: np.ndarray
    percentiles: Dict[int, float]
    
    def get_var(self, confidence_level: float) -> float:
        """取得指定置信水平的 VaR"""
        for r in self.results:
            if abs(r.confidence_level - confidence_level) < 1e-6:
                return r.var
        raise ValueError(f"找不到置信水平 {confidence_level} 的 VaR")
    
    def get_cvar(self, confidence_level: float) -> float:
        """取得指定置信水平的 CVaR"""
        for r in self.results:
            if abs(r.confidence_level - confidence_level) < 1e-6:
                return r.cvar
        raise ValueError(f"找不到置信水平 {confidence_level} 的 CVaR")


@dataclass(frozen=True)
class BootstrapResult:
    """
    Bootstrap 模擬結果
    
    Attributes:
        metric_name: 指標名稱
        point_estimate: 點估計值
        lower_bound: 信賴區間下界
        upper_bound: 信賴區間上界
        confidence_level: 信賴水平
        distribution: 模擬分布（可選）
    """
    metric_name: str
    point_estimate: float
    lower_bound: float
    upper_bound: float
    confidence_level: float
    distribution: Optional[np.ndarray] = None
    
    @property
    def confidence_interval(self) -> Tuple[float, float]:
        """取得信賴區間"""
        return (self.lower_bound, self.upper_bound)


@dataclass(frozen=True)
class StrategyBootstrapResult:
    """策略 Bootstrap 完整結果"""
    total_return: BootstrapResult
    sharpe_ratio: BootstrapResult
    max_drawdown: BootstrapResult
    win_rate: BootstrapResult
    n_simulations: int


@dataclass(frozen=True)
class DrawdownDistributionResult:
    """
    Drawdown 分布結果
    
    Attributes:
        max_drawdowns: 最大回撤分布
        drawdown_durations: 回撤持續期分布
        percentiles: 各百分位數
        probability_gt_threshold: 超過特定閾值的機率
    """
    max_drawdowns: np.ndarray
    drawdown_durations: np.ndarray
    percentiles: Dict[str, Dict[int, float]]
    probability_gt_threshold: Dict[float, float]


@dataclass(frozen=True)
class PathSimulationResult:
    """
    路徑模擬結果
    
    Attributes:
        paths: 模擬路徑矩陣 (n_simulations, n_steps + 1)
        expected_path: 期望路徑
        confidence_bands: 信賴區間帶
    """
    paths: np.ndarray
    expected_path: np.ndarray
    confidence_bands: Dict[float, Tuple[np.ndarray, np.ndarray]]


# ══════════════════════════════════════════════════════════════════════════════
# 隨機數生成器
# ══════════════════════════════════════════════════════════════════════════════

class RandomGenerator:
    """
    隨機數生成器包裝器
    
    提供可重複的隨機數生成，支援多種分布。
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Args:
            seed: 隨機種子（None 表示不固定）
        """
        self._rng = np.random.default_rng(seed)
    
    def normal(
        self,
        mean: float = 0.0,
        std: float = 1.0,
        size: Union[int, Tuple[int, ...]] = 1,
    ) -> np.ndarray:
        """生成正態分布隨機數"""
        return self._rng.normal(mean, std, size)
    
    def standard_normal(
        self,
        size: Union[int, Tuple[int, ...]] = 1,
    ) -> np.ndarray:
        """生成標準正態分布隨機數"""
        return self._rng.standard_normal(size)
    
    def choice(
        self,
        a: Union[int, np.ndarray],
        size: Optional[Union[int, Tuple[int, ...]]] = None,
        replace: bool = True,
    ) -> np.ndarray:
        """隨機選取"""
        return self._rng.choice(a, size=size, replace=replace)
    
    def integers(
        self,
        low: int,
        high: int,
        size: Optional[Union[int, Tuple[int, ...]]] = None,
    ) -> np.ndarray:
        """生成整數隨機數"""
        return self._rng.integers(low, high, size=size)


# ══════════════════════════════════════════════════════════════════════════════
# VaR 計算器
# ══════════════════════════════════════════════════════════════════════════════

class VaRCalculator:
    """
    VaR 計算器
    
    支援多種 VaR 計算方法：
    - 參數化方法（假設正態分布）
    - 歷史模擬法
    - Monte Carlo 方法
    
    使用範例:
        calculator = VaRCalculator(MonteCarloConfig())
        result = calculator.parametric_var(returns, portfolio_value=100000)
        print(f"95% VaR: {result.get_var(0.95):,.0f}")
    """
    
    def __init__(self, config: MonteCarloConfig | None = None):
        """
        Args:
            config: Monte Carlo 配置
        """
        self._config = config or MonteCarloConfig()
        self._rng = RandomGenerator(self._config.random_seed)
    
    def parametric_var(
        self,
        returns: pd.Series,
        portfolio_value: float = 1.0,
    ) -> MonteCarloVaRResult:
        """
        參數化 Monte Carlo VaR
        
        假設收益率服從正態分布，模擬未來收益率分布。
        
        Args:
            returns: 歷史日收益率序列
            portfolio_value: 投資組合價值
        
        Returns:
            MonteCarloVaRResult
        """
        returns_arr = self._validate_returns(returns)
        
        mu = np.mean(returns_arr)
        sigma = np.std(returns_arr, ddof=1)
        
        # 模擬收益率
        horizon = self._config.horizon_days
        simulated = self._rng.normal(
            mu * horizon,
            sigma * np.sqrt(horizon),
            self._config.n_simulations,
        )
        
        # 計算 VaR 和 CVaR
        var_results = []
        for conf in self._config.confidence_levels:
            var, cvar = self._compute_var_cvar(simulated, conf, portfolio_value)
            var_results.append(VaRResult(
                var=var,
                cvar=cvar,
                confidence_level=conf,
                method=VaRMethod.MONTE_CARLO,
                horizon_days=horizon,
            ))
        
        return MonteCarloVaRResult(
            results=tuple(var_results),
            expected_return=mu * self._config.annualization_factor,
            expected_volatility=sigma * np.sqrt(self._config.annualization_factor),
            simulated_returns=simulated,
            percentiles=self._compute_percentiles(simulated),
        )
    
    def historical_var(
        self,
        returns: pd.Series,
        portfolio_value: float = 1.0,
    ) -> MonteCarloVaRResult:
        """
        歷史模擬法 VaR
        
        使用歷史收益率分布直接估計 VaR。
        
        Args:
            returns: 歷史日收益率序列
            portfolio_value: 投資組合價值
        """
        returns_arr = self._validate_returns(returns)
        
        # 直接使用歷史收益率
        var_results = []
        for conf in self._config.confidence_levels:
            var, cvar = self._compute_var_cvar(returns_arr, conf, portfolio_value)
            var_results.append(VaRResult(
                var=var,
                cvar=cvar,
                confidence_level=conf,
                method=VaRMethod.HISTORICAL,
                horizon_days=1,  # 歷史法預設為 1 天
            ))
        
        return MonteCarloVaRResult(
            results=tuple(var_results),
            expected_return=np.mean(returns_arr) * self._config.annualization_factor,
            expected_volatility=np.std(returns_arr) * np.sqrt(self._config.annualization_factor),
            simulated_returns=returns_arr,
            percentiles=self._compute_percentiles(returns_arr),
        )
    
    def _validate_returns(self, returns: pd.Series) -> np.ndarray:
        """驗證並轉換收益率數據"""
        if len(returns) == 0:
            raise ValueError("收益率序列不能為空")
        
        returns_arr = returns.dropna().values
        
        if len(returns_arr) < 30:
            warnings.warn("收益率數據少於 30 筆，結果可能不可靠")
        
        return returns_arr
    
    def _compute_var_cvar(
        self,
        returns: np.ndarray,
        confidence_level: float,
        portfolio_value: float,
    ) -> Tuple[float, float]:
        """計算 VaR 和 CVaR"""
        percentile = (1 - confidence_level) * 100
        var_return = np.percentile(returns, percentile)
        var = abs(var_return) * portfolio_value
        
        # CVaR (Expected Shortfall)
        threshold = np.percentile(returns, percentile)
        tail_returns = returns[returns <= threshold]
        
        if len(tail_returns) > 0:
            cvar = abs(np.mean(tail_returns)) * portfolio_value
        else:
            cvar = var
        
        return var, cvar
    
    def _compute_percentiles(
        self,
        values: np.ndarray,
        percentiles: Tuple[int, ...] = (1, 5, 10, 25, 50, 75, 90, 95, 99),
    ) -> Dict[int, float]:
        """計算百分位數"""
        return {p: np.percentile(values, p) for p in percentiles}


# ══════════════════════════════════════════════════════════════════════════════
# Bootstrap 模擬器
# ══════════════════════════════════════════════════════════════════════════════

class BootstrapSimulator:
    """
    Bootstrap 模擬器
    
    提供多種 Bootstrap 重抽樣方法：
    - 標準 Bootstrap
    - Block Bootstrap（保留時間序列結構）
    
    使用範例:
        simulator = BootstrapSimulator(BootstrapConfig())
        result = simulator.strategy_performance(trade_returns)
        print(f"Sharpe 95% CI: {result.sharpe_ratio.confidence_interval}")
    """
    
    def __init__(self, config: BootstrapConfig | None = None):
        """
        Args:
            config: Bootstrap 配置
        """
        self._config = config or BootstrapConfig()
        self._rng = RandomGenerator(self._config.random_seed)
    
    def strategy_performance(
        self,
        trade_returns: pd.Series,
        n_trades_per_sim: Optional[int] = None,
    ) -> StrategyBootstrapResult:
        """
        Bootstrap 策略績效模擬
        
        重新抽樣歷史交易，估計策略績效的分布。
        
        Args:
            trade_returns: 每筆交易的收益率
            n_trades_per_sim: 每次模擬的交易數（默認為原始數量）
        
        Returns:
            StrategyBootstrapResult
        """
        returns_arr = trade_returns.dropna().values
        
        if len(returns_arr) == 0:
            raise ValueError("交易收益率序列不能為空")
        
        if n_trades_per_sim is None:
            n_trades_per_sim = len(returns_arr)
        
        n_sims = self._config.n_simulations
        
        # 存儲結果
        total_returns = np.zeros(n_sims)
        sharpe_ratios = np.zeros(n_sims)
        max_drawdowns = np.zeros(n_sims)
        win_rates = np.zeros(n_sims)
        
        for i in range(n_sims):
            # 重新抽樣
            if self._config.replacement:
                indices = self._rng.choice(len(returns_arr), size=n_trades_per_sim)
            else:
                indices = self._rng.choice(
                    len(returns_arr),
                    size=min(n_trades_per_sim, len(returns_arr)),
                    replace=False,
                )
            
            sampled = returns_arr[indices]
            
            # 計算績效指標
            cumulative = np.cumprod(1 + sampled)
            total_returns[i] = cumulative[-1] - 1
            
            if sampled.std() > 1e-10:
                sharpe_ratios[i] = (
                    sampled.mean() / sampled.std() *
                    np.sqrt(n_trades_per_sim)
                )
            
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = (cumulative - running_max) / np.maximum(running_max, 1e-10)
            max_drawdowns[i] = abs(np.min(drawdowns))
            
            win_rates[i] = (sampled > 0).mean()
        
        # 建立結果
        return StrategyBootstrapResult(
            total_return=self._create_bootstrap_result(
                "total_return", total_returns
            ),
            sharpe_ratio=self._create_bootstrap_result(
                "sharpe_ratio", sharpe_ratios
            ),
            max_drawdown=self._create_bootstrap_result(
                "max_drawdown", max_drawdowns
            ),
            win_rate=self._create_bootstrap_result(
                "win_rate", win_rates
            ),
            n_simulations=n_sims,
        )
    
    def block_bootstrap(
        self,
        returns: pd.Series,
    ) -> np.ndarray:
        """
        Block Bootstrap 重抽樣
        
        保留時間序列的局部結構。
        
        Args:
            returns: 收益率序列
        
        Returns:
            重抽樣後的收益率陣列 (n_simulations, len(returns))
        """
        returns_arr = returns.dropna().values
        n = len(returns_arr)
        block_size = self._config.block_size
        n_sims = self._config.n_simulations
        
        result = np.zeros((n_sims, n))
        
        for i in range(n_sims):
            sampled = []
            while len(sampled) < n:
                start = self._rng.integers(0, n - block_size + 1)
                sampled.extend(returns_arr[start:start + block_size])
            result[i] = sampled[:n]
        
        return result
    
    def _create_bootstrap_result(
        self,
        name: str,
        distribution: np.ndarray,
    ) -> BootstrapResult:
        """建立 Bootstrap 結果"""
        alpha = 1 - self._config.confidence_level
        
        return BootstrapResult(
            metric_name=name,
            point_estimate=np.median(distribution),
            lower_bound=np.percentile(distribution, alpha / 2 * 100),
            upper_bound=np.percentile(distribution, (1 - alpha / 2) * 100),
            confidence_level=self._config.confidence_level,
            distribution=distribution,
        )


# ══════════════════════════════════════════════════════════════════════════════
# 路徑模擬器
# ══════════════════════════════════════════════════════════════════════════════

class PathSimulator:
    """
    路徑模擬器
    
    模擬價格或權益曲線的未來路徑。
    
    支援方法:
    - 幾何布朗運動 (GBM)
    - Bootstrap 路徑模擬
    
    使用範例:
        simulator = PathSimulator(PathSimulationConfig())
        result = simulator.gbm_paths(initial_price=100, mu=0.1, sigma=0.2)
        print(result.paths.shape)  # (n_simulations, n_steps + 1)
    """
    
    def __init__(self, config: PathSimulationConfig | None = None):
        """
        Args:
            config: 路徑模擬配置
        """
        self._config = config or PathSimulationConfig()
        self._rng = RandomGenerator(self._config.random_seed)
    
    def gbm_paths(
        self,
        initial_price: float,
        mu: float,
        sigma: float,
    ) -> PathSimulationResult:
        """
        幾何布朗運動 (GBM) 路徑模擬
        
        dS = μS dt + σS dW
        
        Args:
            initial_price: 初始價格
            mu: 年化漂移率
            sigma: 年化波動率
        
        Returns:
            PathSimulationResult
        """
        n_sims = self._config.n_simulations
        n_steps = self._config.n_steps
        dt = self._config.dt
        
        # 生成隨機增量
        Z = self._rng.standard_normal((n_sims, n_steps))
        
        # GBM 離散化
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * Z
        
        # 累積對數收益
        log_returns = drift + diffusion
        log_prices = np.cumsum(log_returns, axis=1)
        
        # 轉換回價格
        prices = initial_price * np.exp(log_prices)
        
        # 加入初始價格
        paths = np.column_stack([np.full(n_sims, initial_price), prices])
        
        # 計算期望路徑和信賴區間
        expected_path = np.mean(paths, axis=0)
        confidence_bands = self._compute_confidence_bands(paths)
        
        return PathSimulationResult(
            paths=paths,
            expected_path=expected_path,
            confidence_bands=confidence_bands,
        )
    
    def equity_curve_simulation(
        self,
        returns: pd.Series,
        initial_capital: float = 100000,
    ) -> PathSimulationResult:
        """
        權益曲線模擬
        
        使用 Bootstrap 方法模擬權益曲線。
        
        Args:
            returns: 歷史收益率序列
            initial_capital: 初始資金
        
        Returns:
            PathSimulationResult
        """
        returns_arr = returns.dropna().values
        n = len(returns_arr)
        n_sims = self._config.n_simulations
        
        # Block Bootstrap
        block_size = min(20, n // 10)
        equity_curves = np.zeros((n_sims, n + 1))
        equity_curves[:, 0] = initial_capital
        
        for i in range(n_sims):
            sampled_returns = []
            while len(sampled_returns) < n:
                start = self._rng.integers(0, max(1, n - block_size + 1))
                sampled_returns.extend(returns_arr[start:start + block_size])
            
            sampled_returns = np.array(sampled_returns[:n])
            equity_curves[i, 1:] = initial_capital * np.cumprod(1 + sampled_returns)
        
        expected_path = np.mean(equity_curves, axis=0)
        confidence_bands = self._compute_confidence_bands(equity_curves)
        
        return PathSimulationResult(
            paths=equity_curves,
            expected_path=expected_path,
            confidence_bands=confidence_bands,
        )
    
    def _compute_confidence_bands(
        self,
        paths: np.ndarray,
        confidence_levels: Tuple[float, ...] = (0.90, 0.95, 0.99),
    ) -> Dict[float, Tuple[np.ndarray, np.ndarray]]:
        """計算信賴區間帶"""
        bands = {}
        
        for conf in confidence_levels:
            alpha = 1 - conf
            lower = np.percentile(paths, alpha / 2 * 100, axis=0)
            upper = np.percentile(paths, (1 - alpha / 2) * 100, axis=0)
            bands[conf] = (lower, upper)
        
        return bands


# ══════════════════════════════════════════════════════════════════════════════
# Drawdown 分析器
# ══════════════════════════════════════════════════════════════════════════════

class DrawdownAnalyzer:
    """
    Drawdown 分析器
    
    分析模擬路徑的最大回撤分布。
    
    使用範例:
        analyzer = DrawdownAnalyzer()
        result = analyzer.analyze(path_result.paths)
        print(f"50th percentile Max DD: {result.percentiles['max_drawdown'][50]:.1%}")
    """
    
    def analyze(
        self,
        equity_curves: np.ndarray,
        thresholds: Tuple[float, ...] = (0.10, 0.20, 0.30, 0.50),
    ) -> DrawdownDistributionResult:
        """
        分析 Drawdown 分布
        
        Args:
            equity_curves: 權益曲線矩陣 (n_simulations, n_steps)
            thresholds: 要計算超越機率的閾值
        
        Returns:
            DrawdownDistributionResult
        """
        n_sims = equity_curves.shape[0]
        
        max_drawdowns = np.zeros(n_sims)
        drawdown_durations = np.zeros(n_sims)
        
        for i in range(n_sims):
            equity = equity_curves[i]
            running_max = np.maximum.accumulate(equity)
            drawdowns = (equity - running_max) / np.maximum(running_max, 1e-10)
            
            max_drawdowns[i] = abs(np.min(drawdowns))
            drawdown_durations[i] = self._compute_max_duration(drawdowns)
        
        # 計算百分位數
        percentiles = {
            "max_drawdown": {
                p: np.percentile(max_drawdowns, p)
                for p in (5, 25, 50, 75, 95)
            },
            "duration": {
                p: np.percentile(drawdown_durations, p)
                for p in (5, 25, 50, 75, 95)
            },
        }
        
        # 計算超越機率
        prob_gt_threshold = {
            t: (max_drawdowns > t).mean()
            for t in thresholds
        }
        
        return DrawdownDistributionResult(
            max_drawdowns=max_drawdowns,
            drawdown_durations=drawdown_durations,
            percentiles=percentiles,
            probability_gt_threshold=prob_gt_threshold,
        )
    
    def _compute_max_duration(self, drawdowns: np.ndarray) -> int:
        """計算最大回撤持續期"""
        underwater = drawdowns < 0
        
        if not underwater.any():
            return 0
        
        # 計算連續 underwater 的長度
        max_duration = 0
        current_duration = 0
        
        for is_underwater in underwater:
            if is_underwater:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
        
        return max_duration


# ══════════════════════════════════════════════════════════════════════════════
# 組合 Monte Carlo
# ══════════════════════════════════════════════════════════════════════════════

class PortfolioMonteCarloSimulator:
    """
    組合 Monte Carlo 模擬器
    
    考慮資產間相關性的組合風險模擬。
    
    使用範例:
        simulator = PortfolioMonteCarloSimulator(MonteCarloConfig())
        result = simulator.simulate(returns_dict, weights)
        print(f"Portfolio 95% VaR: {result.get_var(0.95):.2%}")
    """
    
    def __init__(self, config: MonteCarloConfig | None = None):
        """
        Args:
            config: Monte Carlo 配置
        """
        self._config = config or MonteCarloConfig()
        self._rng = RandomGenerator(self._config.random_seed)
    
    def simulate(
        self,
        returns_dict: Dict[str, pd.Series],
        weights: Dict[str, float],
        correlation_matrix: Optional[pd.DataFrame] = None,
    ) -> MonteCarloVaRResult:
        """
        組合 Monte Carlo VaR 模擬
        
        Args:
            returns_dict: 各資產的收益率序列
            weights: 各資產的權重
            correlation_matrix: 相關性矩陣（None 則從數據計算）
        
        Returns:
            MonteCarloVaRResult
        """
        # 對齊收益率
        returns_df = pd.DataFrame(returns_dict).dropna()
        
        if len(returns_df) < 30:
            warnings.warn("數據少於 30 筆，結果可能不可靠")
        
        symbols = list(weights.keys())
        symbols = [s for s in symbols if s in returns_df.columns]
        
        if not symbols:
            raise ValueError("無有效資產")
        
        # 歸一化權重
        weight_vec = np.array([weights.get(s, 0) for s in symbols])
        weight_vec = weight_vec / weight_vec.sum()
        
        # 計算各資產參數
        means = returns_df[symbols].mean().values
        stds = returns_df[symbols].std().values
        
        # 計算或使用相關性矩陣
        if correlation_matrix is None:
            corr = returns_df[symbols].corr().values
        else:
            corr = correlation_matrix.loc[symbols, symbols].values
        
        # Cholesky 分解
        corr = self._ensure_positive_definite(corr)
        L = np.linalg.cholesky(corr)
        
        # 生成相關隨機數
        n_sims = self._config.n_simulations
        horizon = self._config.horizon_days
        n_assets = len(symbols)
        
        Z = self._rng.standard_normal((n_sims, n_assets))
        correlated_Z = Z @ L.T
        
        # 模擬各資產收益率
        simulated = np.zeros((n_sims, n_assets))
        for j in range(n_assets):
            simulated[:, j] = (
                means[j] * horizon +
                stds[j] * np.sqrt(horizon) * correlated_Z[:, j]
            )
        
        # 計算組合收益率
        portfolio_returns = simulated @ weight_vec
        
        # 計算 VaR 和 CVaR
        var_results = []
        for conf in self._config.confidence_levels:
            percentile = (1 - conf) * 100
            var = -np.percentile(portfolio_returns, percentile)
            
            threshold = np.percentile(portfolio_returns, percentile)
            tail = portfolio_returns[portfolio_returns <= threshold]
            cvar = -np.mean(tail) if len(tail) > 0 else var
            
            var_results.append(VaRResult(
                var=var,
                cvar=cvar,
                confidence_level=conf,
                method=VaRMethod.MONTE_CARLO,
                horizon_days=horizon,
            ))
        
        return MonteCarloVaRResult(
            results=tuple(var_results),
            expected_return=portfolio_returns.mean(),
            expected_volatility=portfolio_returns.std(),
            simulated_returns=portfolio_returns,
            percentiles={
                p: np.percentile(portfolio_returns, p)
                for p in (1, 5, 10, 25, 50, 75, 90, 95, 99)
            },
        )
    
    def _ensure_positive_definite(self, matrix: np.ndarray) -> np.ndarray:
        """確保矩陣為正定"""
        try:
            np.linalg.cholesky(matrix)
            return matrix
        except np.linalg.LinAlgError:
            # 使用最近正定矩陣
            eigvals, eigvecs = np.linalg.eigh(matrix)
            eigvals = np.maximum(eigvals, 1e-10)
            return eigvecs @ np.diag(eigvals) @ eigvecs.T


# ══════════════════════════════════════════════════════════════════════════════
# 整合模擬器（Facade）
# ══════════════════════════════════════════════════════════════════════════════

class MonteCarloSimulator:
    """
    Monte Carlo 模擬器（Facade）
    
    整合所有模擬功能的統一介面。
    
    使用範例:
        simulator = MonteCarloSimulator()
        
        # VaR 計算
        var_result = simulator.calculate_var(returns)
        
        # Bootstrap 績效估計
        bootstrap_result = simulator.bootstrap_performance(trade_returns)
        
        # 路徑模擬
        path_result = simulator.simulate_paths(returns, initial_capital=100000)
    """
    
    def __init__(
        self,
        mc_config: MonteCarloConfig | None = None,
        bootstrap_config: BootstrapConfig | None = None,
        path_config: PathSimulationConfig | None = None,
    ):
        """
        Args:
            mc_config: Monte Carlo 配置
            bootstrap_config: Bootstrap 配置
            path_config: 路徑模擬配置
        """
        self._mc_config = mc_config or MonteCarloConfig()
        self._bootstrap_config = bootstrap_config or BootstrapConfig()
        self._path_config = path_config or PathSimulationConfig()
        
        self._var_calculator = VaRCalculator(self._mc_config)
        self._bootstrap_simulator = BootstrapSimulator(self._bootstrap_config)
        self._path_simulator = PathSimulator(self._path_config)
        self._drawdown_analyzer = DrawdownAnalyzer()
        self._portfolio_simulator = PortfolioMonteCarloSimulator(self._mc_config)
    
    def calculate_var(
        self,
        returns: pd.Series,
        method: str = "parametric",
        portfolio_value: float = 1.0,
    ) -> MonteCarloVaRResult:
        """
        計算 VaR
        
        Args:
            returns: 收益率序列
            method: 計算方法 ("parametric", "historical")
            portfolio_value: 投資組合價值
        """
        if method == "historical":
            return self._var_calculator.historical_var(returns, portfolio_value)
        return self._var_calculator.parametric_var(returns, portfolio_value)
    
    def bootstrap_performance(
        self,
        trade_returns: pd.Series,
        n_trades_per_sim: Optional[int] = None,
    ) -> StrategyBootstrapResult:
        """
        Bootstrap 策略績效估計
        
        Args:
            trade_returns: 交易收益率序列
            n_trades_per_sim: 每次模擬的交易數
        """
        return self._bootstrap_simulator.strategy_performance(
            trade_returns,
            n_trades_per_sim,
        )
    
    def simulate_paths(
        self,
        returns: pd.Series,
        initial_capital: float = 100000,
    ) -> PathSimulationResult:
        """
        模擬權益曲線路徑
        
        Args:
            returns: 收益率序列
            initial_capital: 初始資金
        """
        return self._path_simulator.equity_curve_simulation(returns, initial_capital)
    
    def analyze_drawdown(
        self,
        returns: pd.Series,
        initial_capital: float = 100000,
    ) -> DrawdownDistributionResult:
        """
        分析 Drawdown 分布
        
        Args:
            returns: 收益率序列
            initial_capital: 初始資金
        """
        path_result = self.simulate_paths(returns, initial_capital)
        return self._drawdown_analyzer.analyze(path_result.paths)
    
    def portfolio_var(
        self,
        returns_dict: Dict[str, pd.Series],
        weights: Dict[str, float],
    ) -> MonteCarloVaRResult:
        """
        計算組合 VaR
        
        Args:
            returns_dict: 各資產收益率字典
            weights: 權重字典
        """
        return self._portfolio_simulator.simulate(returns_dict, weights)


# ══════════════════════════════════════════════════════════════════════════════
# 便捷函數
# ══════════════════════════════════════════════════════════════════════════════

def monte_carlo_var(
    returns: pd.Series,
    confidence_level: float = 0.95,
    n_simulations: int = 10000,
    horizon_days: int = 1,
    portfolio_value: float = 100000,
) -> float:
    """
    計算 Monte Carlo VaR 便捷函數
    
    Args:
        returns: 歷史日收益率
        confidence_level: 置信水平
        n_simulations: 模擬次數
        horizon_days: 風險評估期間
        portfolio_value: 投資組合價值
    
    Returns:
        VaR 值（正數，表示潛在損失）
    
    Example:
        var = monte_carlo_var(daily_returns, confidence_level=0.95)
        print(f"95% VaR: ${var:,.0f}")
    """
    config = MonteCarloConfig(
        n_simulations=n_simulations,
        confidence_levels=(confidence_level,),
        horizon_days=horizon_days,
    )
    calculator = VaRCalculator(config)
    result = calculator.parametric_var(returns, portfolio_value)
    return result.get_var(confidence_level)


def bootstrap_strategy_ci(
    trade_returns: pd.Series,
    confidence: float = 0.95,
    n_simulations: int = 10000,
) -> Dict[str, Tuple[float, float, float]]:
    """
    Bootstrap 策略績效信賴區間便捷函數
    
    Args:
        trade_returns: 交易收益率序列
        confidence: 信賴水平
        n_simulations: 模擬次數
    
    Returns:
        {metric: (lower, median, upper)}
    
    Example:
        ci = bootstrap_strategy_ci(trade_returns)
        print(f"Sharpe 95% CI: [{ci['sharpe_ratio'][0]:.2f}, {ci['sharpe_ratio'][2]:.2f}]")
    """
    config = BootstrapConfig(
        n_simulations=n_simulations,
        confidence_level=confidence,
    )
    simulator = BootstrapSimulator(config)
    result = simulator.strategy_performance(trade_returns)
    
    return {
        "total_return": (
            result.total_return.lower_bound,
            result.total_return.point_estimate,
            result.total_return.upper_bound,
        ),
        "sharpe_ratio": (
            result.sharpe_ratio.lower_bound,
            result.sharpe_ratio.point_estimate,
            result.sharpe_ratio.upper_bound,
        ),
        "max_drawdown": (
            result.max_drawdown.lower_bound,
            result.max_drawdown.point_estimate,
            result.max_drawdown.upper_bound,
        ),
        "win_rate": (
            result.win_rate.lower_bound,
            result.win_rate.point_estimate,
            result.win_rate.upper_bound,
        ),
    }


def simulate_strategy_outcomes(
    returns: pd.Series,
    n_simulations: int = 10000,
    initial_capital: float = 100000,
) -> Dict:
    """
    模擬策略可能的結果分布便捷函數
    
    Args:
        returns: 收益率序列
        n_simulations: 模擬次數
        initial_capital: 初始資金
    
    Returns:
        包含各種結果分布的字典
    
    Example:
        outcomes = simulate_strategy_outcomes(daily_returns)
        print(f"Probability of loss: {outcomes['probability_of_loss']:.1%}")
    """
    path_config = PathSimulationConfig(n_simulations=n_simulations)
    simulator = MonteCarloSimulator(path_config=path_config)
    
    path_result = simulator.simulate_paths(returns, initial_capital)
    dd_result = simulator._drawdown_analyzer.analyze(path_result.paths)
    
    final_returns = (path_result.paths[:, -1] / path_result.paths[:, 0]) - 1
    
    return {
        "final_return_distribution": final_returns,
        "max_drawdown_distribution": dd_result.max_drawdowns,
        "percentiles": {
            "final_return": {
                p: np.percentile(final_returns, p) for p in (5, 25, 50, 75, 95)
            },
            "max_drawdown": dd_result.percentiles["max_drawdown"],
        },
        "probability_of_loss": (final_returns < 0).mean(),
        "probability_of_drawdown_gt_20": dd_result.probability_gt_threshold.get(0.20, 0),
    }
