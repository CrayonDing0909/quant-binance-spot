"""
風險管理模組

提供倉位管理、風險限制、組合風險控制和 Monte Carlo 模擬功能。
"""
from __future__ import annotations

from .position_sizing import (
    PositionSizer,
    FixedPositionSizer,
    KellyPositionSizer,
    VolatilityPositionSizer,
)
from .risk_limits import (
    RiskLimits,
    apply_risk_limits,
    check_max_position_size,
    check_max_drawdown,
    check_max_leverage,
)
from .portfolio_risk import (
    PortfolioRiskManager,
    calculate_portfolio_var,
    calculate_correlation_matrix,
)
from .monte_carlo import (
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
    # 便捷函數
    monte_carlo_var,
    bootstrap_strategy_ci,
    simulate_strategy_outcomes,
)

__all__ = [
    # Position sizing
    "PositionSizer",
    "FixedPositionSizer",
    "KellyPositionSizer",
    "VolatilityPositionSizer",
    # Risk limits
    "RiskLimits",
    "apply_risk_limits",
    "check_max_position_size",
    "check_max_drawdown",
    "check_max_leverage",
    # Portfolio risk
    "PortfolioRiskManager",
    "calculate_portfolio_var",
    "calculate_correlation_matrix",
    # Monte Carlo - Config
    "MonteCarloConfig",
    "BootstrapConfig",
    "PathSimulationConfig",
    # Monte Carlo - Results
    "VaRResult",
    "MonteCarloVaRResult",
    "BootstrapResult",
    "StrategyBootstrapResult",
    "DrawdownDistributionResult",
    "PathSimulationResult",
    # Monte Carlo - Enums
    "SimulationMethod",
    "VaRMethod",
    # Monte Carlo - Simulators
    "VaRCalculator",
    "BootstrapSimulator",
    "PathSimulator",
    "DrawdownAnalyzer",
    "PortfolioMonteCarloSimulator",
    "MonteCarloSimulator",
    # Monte Carlo - Convenience
    "monte_carlo_var",
    "bootstrap_strategy_ci",
    "simulate_strategy_outcomes",
]
