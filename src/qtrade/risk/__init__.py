"""
风险管理模块

提供仓位管理、风险限制和组合风险控制功能。
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
]

