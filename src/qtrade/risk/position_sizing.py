"""
仓位管理模块

提供多种仓位计算方法：
- 固定仓位
- Kelly 公式
- 基于波动率的仓位
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd
import numpy as np


class PositionSizer(ABC):
    """仓位计算器基类"""
    
    @abstractmethod
    def calculate_size(
        self,
        signal: float,
        equity: float,
        price: float,
        **kwargs
    ) -> float:
        """
        计算仓位大小
        
        Args:
            signal: 信号强度 [0, 1]
            equity: 当前权益
            price: 当前价格
            **kwargs: 其他参数
        
        Returns:
            仓位大小（股数或金额）
        """
        raise NotImplementedError


class FixedPositionSizer(PositionSizer):
    """
    固定仓位计算器
    
    使用固定的仓位比例，不考虑信号强度。
    """
    
    def __init__(self, position_pct: float = 1.0):
        """
        Args:
            position_pct: 固定仓位比例 [0, 1]，默认 1.0（满仓）
        """
        if not 0 <= position_pct <= 1:
            raise ValueError("position_pct must be between 0 and 1")
        self.position_pct = position_pct
    
    def calculate_size(
        self,
        signal: float,
        equity: float,
        price: float,
        **kwargs
    ) -> float:
        """计算固定仓位"""
        target_value = equity * self.position_pct * signal
        return target_value / price if price > 0 else 0.0


class KellyPositionSizer(PositionSizer):
    """
    Kelly 公式仓位计算器
    
    根据胜率和盈亏比计算最优仓位。
    Kelly% = (P * W - L) / W
    其中：
    - P = 胜率
    - W = 平均盈利 / 平均亏损
    - L = 1 - P
    """
    
    def __init__(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        kelly_fraction: float = 1.0
    ):
        """
        Args:
            win_rate: 胜率 [0, 1]
            avg_win: 平均盈利（正数）
            avg_loss: 平均亏损（正数）
            kelly_fraction: Kelly 比例因子，默认 1.0（全 Kelly），通常用 0.25-0.5 更保守
        """
        if not 0 <= win_rate <= 1:
            raise ValueError("win_rate must be between 0 and 1")
        if avg_win <= 0 or avg_loss <= 0:
            raise ValueError("avg_win and avg_loss must be positive")
        if not 0 < kelly_fraction <= 1:
            raise ValueError("kelly_fraction must be between 0 and 1")
        
        self.win_rate = win_rate
        self.avg_win = avg_win
        self.avg_loss = avg_loss
        self.kelly_fraction = kelly_fraction
        
        # 计算 Kelly 比例
        W = avg_win / avg_loss
        L = 1 - win_rate
        self.kelly_pct = (win_rate * W - L) / W
        self.kelly_pct = max(0, min(self.kelly_pct, 1))  # 限制在 [0, 1]
        self.kelly_pct *= kelly_fraction  # 应用保守因子
    
    def calculate_size(
        self,
        signal: float,
        equity: float,
        price: float,
        **kwargs
    ) -> float:
        """根据 Kelly 公式计算仓位"""
        target_value = equity * self.kelly_pct * signal
        return target_value / price if price > 0 else 0.0


class VolatilityPositionSizer(PositionSizer):
    """
    基于波动率的仓位计算器
    
    根据资产波动率调整仓位，波动率越高，仓位越小。
    """
    
    def __init__(
        self,
        base_position_pct: float = 1.0,
        target_volatility: float = 0.15,
        lookback: int = 20
    ):
        """
        Args:
            base_position_pct: 基础仓位比例 [0, 1]
            target_volatility: 目标波动率（年化），默认 15%
            lookback: 计算波动率的回看期，默认 20
        """
        if not 0 <= base_position_pct <= 1:
            raise ValueError("base_position_pct must be between 0 and 1")
        if target_volatility <= 0:
            raise ValueError("target_volatility must be positive")
        
        self.base_position_pct = base_position_pct
        self.target_volatility = target_volatility
        self.lookback = lookback
    
    def calculate_size(
        self,
        signal: float,
        equity: float,
        price: float,
        returns: Optional[pd.Series] = None,
        **kwargs
    ) -> float:
        """
        根据波动率计算仓位
        
        Args:
            returns: 收益率序列，用于计算波动率
        """
        if returns is None or len(returns) < self.lookback:
            # 如果没有收益率数据，使用基础仓位
            target_value = equity * self.base_position_pct * signal
            return target_value / price if price > 0 else 0.0
        
        # 计算当前波动率（年化）
        current_vol = returns.tail(self.lookback).std() * np.sqrt(252)  # 假设日线数据
        
        # 根据波动率调整仓位
        vol_adjustment = min(self.target_volatility / current_vol, 2.0)  # 限制最大调整到 2 倍
        adjusted_position_pct = self.base_position_pct * vol_adjustment
        
        target_value = equity * adjusted_position_pct * signal
        return target_value / price if price > 0 else 0.0

