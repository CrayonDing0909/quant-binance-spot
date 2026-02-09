"""
倉位管理模組

提供多種倉位計算方法：
- 固定倉位
- Kelly 公式
- 基於波動率的倉位
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd
import numpy as np


class PositionSizer(ABC):
    """倉位計算器基類"""
    
    @abstractmethod
    def calculate_size(
        self,
        signal: float,
        equity: float,
        price: float,
        **kwargs
    ) -> float:
        """
        計算倉位大小
        
        Args:
            signal: 信號強度 [0, 1]
            equity: 當前權益
            price: 當前價格
            **kwargs: 其他參數
        
        Returns:
            倉位大小（股數或金額）
        """
        raise NotImplementedError


class FixedPositionSizer(PositionSizer):
    """
    固定倉位計算器
    
    使用固定的倉位比例，不考慮信號強度。
    """
    
    def __init__(self, position_pct: float = 1.0):
        """
        Args:
            position_pct: 固定倉位比例 [0, 1]，預設 1.0（滿倉）
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
        """計算固定倉位"""
        target_value = equity * self.position_pct * signal
        return target_value / price if price > 0 else 0.0


class KellyPositionSizer(PositionSizer):
    """
    Kelly 公式倉位計算器
    
    根據勝率和盈虧比計算最優倉位。
    Kelly% = (P * W - L) / W
    其中：
    - P = 勝率
    - W = 平均盈利 / 平均虧損
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
            win_rate: 勝率 [0, 1]
            avg_win: 平均盈利（正數）
            avg_loss: 平均虧損（正數）
            kelly_fraction: Kelly 比例因子，預設 1.0（全 Kelly），通常用 0.25-0.5 更保守
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
        
        # 計算 Kelly 比例
        W = avg_win / avg_loss
        L = 1 - win_rate
        self.kelly_pct = (win_rate * W - L) / W
        self.kelly_pct = max(0, min(self.kelly_pct, 1))  # 限制在 [0, 1]
        self.kelly_pct *= kelly_fraction  # 應用保守因子
    
    def calculate_size(
        self,
        signal: float,
        equity: float,
        price: float,
        **kwargs
    ) -> float:
        """根據 Kelly 公式計算倉位"""
        target_value = equity * self.kelly_pct * signal
        return target_value / price if price > 0 else 0.0


class VolatilityPositionSizer(PositionSizer):
    """
    基於波動率的倉位計算器
    
    根據資產波動率調整倉位，波動率越高，倉位越小。
    """
    
    def __init__(
        self,
        base_position_pct: float = 1.0,
        target_volatility: float = 0.15,
        lookback: int = 20
    ):
        """
        Args:
            base_position_pct: 基礎倉位比例 [0, 1]
            target_volatility: 目標波動率（年化），預設 15%
            lookback: 計算波動率的回看期，預設 20
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
        根據波動率計算倉位
        
        Args:
            returns: 收益率序列，用於計算波動率
        """
        if returns is None or len(returns) < self.lookback:
            # 如果沒有收益率數據，使用基礎倉位
            target_value = equity * self.base_position_pct * signal
            return target_value / price if price > 0 else 0.0
        
        # 計算當前波動率（年化）
        current_vol = returns.tail(self.lookback).std() * np.sqrt(252)  # 假設日線數據
        
        # 根據波動率調整倉位
        vol_adjustment = min(self.target_volatility / current_vol, 2.0)  # 限制最大調整到 2 倍
        adjusted_position_pct = self.base_position_pct * vol_adjustment
        
        target_value = equity * adjusted_position_pct * signal
        return target_value / price if price > 0 else 0.0
