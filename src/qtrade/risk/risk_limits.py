"""
风险限制模块

提供各种风险限制检查功能：
- 最大仓位限制
- 最大回撤限制
- 最大杠杆限制
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np


@dataclass
class RiskLimits:
    """风险限制配置"""
    max_position_pct: float = 1.0  # 最大仓位比例 [0, 1]
    max_drawdown_pct: float = 0.5  # 最大回撤限制 [0, 1]
    max_leverage: float = 1.0  # 最大杠杆（现货为 1.0）
    max_single_position_pct: float = 0.5  # 单个资产最大仓位 [0, 1]
    min_cash_reserve_pct: float = 0.0  # 最小现金储备比例 [0, 1]


def check_max_position_size(
    position_pct: float,
    limits: RiskLimits
) -> tuple[bool, float]:
    """
    检查仓位是否超过最大限制
    
    Args:
        position_pct: 当前仓位比例
        limits: 风险限制配置
    
    Returns:
        (是否通过, 调整后的仓位比例)
    """
    if position_pct > limits.max_position_pct:
        return False, limits.max_position_pct
    return True, position_pct


def check_max_drawdown(
    equity_curve: pd.Series,
    limits: RiskLimits,
    current_equity: Optional[float] = None
) -> tuple[bool, float]:
    """
    检查回撤是否超过最大限制
    
    Args:
        equity_curve: 权益曲线
        limits: 风险限制配置
        current_equity: 当前权益（如果提供，会计算当前回撤）
    
    Returns:
        (是否通过, 当前回撤比例)
    """
    if len(equity_curve) == 0:
        return True, 0.0
    
    # 计算历史最大回撤
    peak = equity_curve.expanding().max()
    drawdown = (equity_curve - peak) / peak
    max_dd = abs(drawdown.min())
    
    # 如果提供了当前权益，计算当前回撤
    if current_equity is not None and len(equity_curve) > 0:
        current_peak = equity_curve.max()
        current_dd = abs((current_equity - current_peak) / current_peak) if current_peak > 0 else 0
        max_dd = max(max_dd, current_dd)
    
    if max_dd > limits.max_drawdown_pct:
        return False, max_dd
    return True, max_dd


def check_max_leverage(
    total_position_value: float,
    equity: float,
    limits: RiskLimits
) -> tuple[bool, float]:
    """
    检查杠杆是否超过最大限制
    
    Args:
        total_position_value: 总持仓价值
        equity: 当前权益
        limits: 风险限制配置
    
    Returns:
        (是否通过, 当前杠杆)
    """
    if equity <= 0:
        return False, float('inf')
    
    current_leverage = total_position_value / equity
    
    if current_leverage > limits.max_leverage:
        return False, current_leverage
    return True, current_leverage


def check_cash_reserve(
    cash: float,
    equity: float,
    limits: RiskLimits
) -> tuple[bool, float]:
    """
    检查现金储备是否满足最小要求
    
    Args:
        cash: 当前现金
        equity: 当前权益
        limits: 风险限制配置
    
    Returns:
        (是否通过, 当前现金比例)
    """
    if equity <= 0:
        return False, 0.0
    
    cash_pct = cash / equity
    
    if cash_pct < limits.min_cash_reserve_pct:
        return False, cash_pct
    return True, cash_pct


def apply_risk_limits(
    position_pct: float,
    equity_curve: pd.Series,
    limits: RiskLimits,
    current_equity: Optional[float] = None,
    cash: Optional[float] = None
) -> tuple[float, dict]:
    """
    应用所有风险限制，返回调整后的仓位和检查结果
    
    Args:
        position_pct: 原始仓位比例
        equity_curve: 权益曲线
        limits: 风险限制配置
        current_equity: 当前权益
        cash: 当前现金
    
    Returns:
        (调整后的仓位比例, 检查结果字典)
    """
    adjusted_position = position_pct
    checks = {}
    
    # 1. 检查最大仓位
    passed, adjusted_position = check_max_position_size(adjusted_position, limits)
    checks["max_position"] = {"passed": passed, "value": adjusted_position}
    
    # 2. 检查最大回撤
    passed, current_dd = check_max_drawdown(equity_curve, limits, current_equity)
    checks["max_drawdown"] = {"passed": passed, "value": current_dd}
    
    # 如果回撤超过限制，强制降低仓位
    if not passed:
        # 回撤越大，仓位越小
        dd_adjustment = 1.0 - (current_dd / limits.max_drawdown_pct)
        adjusted_position = adjusted_position * max(0.1, dd_adjustment)
        checks["max_drawdown"]["adjusted_position"] = adjusted_position
    
    # 3. 检查现金储备
    if cash is not None and current_equity is not None:
        passed, cash_pct = check_cash_reserve(cash, current_equity, limits)
        checks["cash_reserve"] = {"passed": passed, "value": cash_pct}
        
        if not passed:
            # 确保保留足够的现金
            max_position_from_cash = 1.0 - limits.min_cash_reserve_pct
            adjusted_position = min(adjusted_position, max_position_from_cash)
            checks["cash_reserve"]["adjusted_position"] = adjusted_position
    
    # 4. 确保最终仓位在合理范围内
    adjusted_position = max(0.0, min(adjusted_position, limits.max_position_pct))
    
    return adjusted_position, checks

