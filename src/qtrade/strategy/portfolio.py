"""
策略組合模組

提供多策略組合和動態權重調整功能。
"""
from __future__ import annotations
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from enum import Enum

from .base import StrategyContext
from . import get_strategy


class WeightMethod(Enum):
    """權重分配方法"""
    EQUAL = "equal"  # 等權重
    FIXED = "fixed"  # 固定權重
    PERFORMANCE = "performance"  # 基於歷史表現
    VOLATILITY = "volatility"  # 基於波動率（波動率越低，權重越高）
    SHARPE = "sharpe"  # 基於夏普比率
    DYNAMIC = "dynamic"  # 動態調整


@dataclass
class StrategyWeight:
    """策略權重配置"""
    strategy_name: str
    weight: float
    min_weight: float = 0.0
    max_weight: float = 1.0


@dataclass
class PortfolioStrategy:
    """組合策略配置"""
    strategies: List[StrategyWeight]
    weight_method: WeightMethod = WeightMethod.EQUAL
    rebalance_freq: str = "D"  # 再平衡頻率："D", "W", "M"
    lookback_period: int = 30  # 用於計算動態權重的回看期


class StrategyPortfolio:
    """策略組合管理器"""
    
    def __init__(self, config: PortfolioStrategy):
        """
        Args:
            config: 組合策略配置
        """
        self.config = config
        self.strategies = {s.strategy_name: s for s in config.strategies}
    
    def generate_positions(
        self,
        df: pd.DataFrame,
        ctx: StrategyContext,
        params: dict,
        historical_returns: Optional[Dict[str, pd.Series]] = None,
        historical_sharpe: Optional[Dict[str, float]] = None
    ) -> pd.Series:
        """
        生成組合持倉信號
        
        Args:
            df: K線數據
            ctx: 策略上下文
            params: 策略參數（可能包含各子策略的參數）
            historical_returns: 各策略的歷史收益率（用於動態權重）
            historical_sharpe: 各策略的歷史夏普比率（用於動態權重）
        
        Returns:
            組合持倉比例序列 [0, 1]
        """
        # 獲取各策略的持倉信號
        strategy_positions = {}
        
        for strategy_name in self.strategies.keys():
            strategy_func = get_strategy(strategy_name)
            strategy_params = params.get(strategy_name, {})
            pos = strategy_func(df, ctx, strategy_params)
            strategy_positions[strategy_name] = pos
        
        # 計算權重
        weights = self._calculate_weights(
            strategy_positions,
            historical_returns,
            historical_sharpe
        )
        
        # 組合持倉 = 加權平均
        portfolio_pos = pd.Series(0.0, index=df.index)
        
        for strategy_name, pos in strategy_positions.items():
            weight = weights.get(strategy_name, 0.0)
            portfolio_pos += pos * weight
        
        # 確保持倉在 [0, 1] 範圍內
        portfolio_pos = portfolio_pos.clip(0.0, 1.0)
        
        return portfolio_pos
    
    def _calculate_weights(
        self,
        strategy_positions: Dict[str, pd.Series],
        historical_returns: Optional[Dict[str, pd.Series]] = None,
        historical_sharpe: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """計算各策略的權重"""
        if self.config.weight_method == WeightMethod.EQUAL:
            return self._equal_weights()
        
        elif self.config.weight_method == WeightMethod.FIXED:
            return self._fixed_weights()
        
        elif self.config.weight_method == WeightMethod.PERFORMANCE:
            return self._performance_weights(historical_returns)
        
        elif self.config.weight_method == WeightMethod.VOLATILITY:
            return self._volatility_weights(historical_returns)
        
        elif self.config.weight_method == WeightMethod.SHARPE:
            return self._sharpe_weights(historical_sharpe)
        
        elif self.config.weight_method == WeightMethod.DYNAMIC:
            return self._dynamic_weights(strategy_positions, historical_returns)
        
        else:
            return self._equal_weights()
    
    def _equal_weights(self) -> Dict[str, float]:
        """等權重分配"""
        n = len(self.strategies)
        if n == 0:
            return {}
        weight = 1.0 / n
        return {name: weight for name in self.strategies.keys()}
    
    def _fixed_weights(self) -> Dict[str, float]:
        """固定權重分配"""
        weights = {}
        total_weight = sum(s.weight for s in self.config.strategies)
        
        if total_weight == 0:
            return self._equal_weights()
        
        # 歸一化權重
        for strategy_weight in self.config.strategies:
            normalized = strategy_weight.weight / total_weight
            # 應用最小/最大限制
            normalized = max(strategy_weight.min_weight, min(normalized, strategy_weight.max_weight))
            weights[strategy_weight.strategy_name] = normalized
        
        # 再次歸一化以確保總和為 1
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        return weights
    
    def _performance_weights(
        self,
        historical_returns: Optional[Dict[str, pd.Series]]
    ) -> Dict[str, float]:
        """基於歷史表現的權重分配（收益率越高，權重越高）"""
        if historical_returns is None or len(historical_returns) == 0:
            return self._equal_weights()
        
        # 計算各策略的總收益率
        total_returns = {}
        for name, returns in historical_returns.items():
            if len(returns) > 0:
                total_returns[name] = (1 + returns).prod() - 1
            else:
                total_returns[name] = 0.0
        
        # 如果所有收益率為負，使用等權重
        if all(r <= 0 for r in total_returns.values()):
            return self._equal_weights()
        
        # 使用收益率作為權重（只考慮正收益）
        weights = {}
        for name, ret in total_returns.items():
            weights[name] = max(0.0, ret)
        
        # 歸一化
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        else:
            return self._equal_weights()
        
        # 應用最小/最大限制
        for strategy_weight in self.config.strategies:
            name = strategy_weight.strategy_name
            if name in weights:
                weights[name] = max(
                    strategy_weight.min_weight,
                    min(weights[name], strategy_weight.max_weight)
                )
        
        # 再次歸一化
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        return weights
    
    def _volatility_weights(
        self,
        historical_returns: Optional[Dict[str, pd.Series]]
    ) -> Dict[str, float]:
        """基於波動率的權重分配（波動率越低，權重越高）"""
        if historical_returns is None or len(historical_returns) == 0:
            return self._equal_weights()
        
        # 計算各策略的波動率
        volatilities = {}
        for name, returns in historical_returns.items():
            if len(returns) > 0:
                volatilities[name] = returns.std() * np.sqrt(252)  # 年化
            else:
                volatilities[name] = float('inf')
        
        # 使用波動率的倒數作為權重
        inv_vol = {}
        for name, vol in volatilities.items():
            inv_vol[name] = 1.0 / vol if vol > 0 else 0.0
        
        # 歸一化
        total = sum(inv_vol.values())
        if total > 0:
            weights = {k: v / total for k, v in inv_vol.items()}
        else:
            return self._equal_weights()
        
        # 應用最小/最大限制
        for strategy_weight in self.config.strategies:
            name = strategy_weight.strategy_name
            if name in weights:
                weights[name] = max(
                    strategy_weight.min_weight,
                    min(weights[name], strategy_weight.max_weight)
                )
        
        # 再次歸一化
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        return weights
    
    def _sharpe_weights(
        self,
        historical_sharpe: Optional[Dict[str, float]]
    ) -> Dict[str, float]:
        """基於夏普比率的權重分配（夏普比率越高，權重越高）"""
        if historical_sharpe is None or len(historical_sharpe) == 0:
            return self._equal_weights()
        
        # 使用夏普比率作為權重（只考慮正夏普）
        weights = {}
        for name, sharpe in historical_sharpe.items():
            weights[name] = max(0.0, sharpe)
        
        # 歸一化
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        else:
            return self._equal_weights()
        
        # 應用最小/最大限制
        for strategy_weight in self.config.strategies:
            name = strategy_weight.strategy_name
            if name in weights:
                weights[name] = max(
                    strategy_weight.min_weight,
                    min(weights[name], strategy_weight.max_weight)
                )
        
        # 再次歸一化
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        return weights
    
    def _dynamic_weights(
        self,
        strategy_positions: Dict[str, pd.Series],
        historical_returns: Optional[Dict[str, pd.Series]]
    ) -> Dict[str, float]:
        """動態權重分配（結合多個因素）"""
        # 結合表現和波動率
        perf_weights = self._performance_weights(historical_returns)
        vol_weights = self._volatility_weights(historical_returns)
        
        # 加權組合（表現 60%，波動率 40%）
        combined_weights = {}
        for name in self.strategies.keys():
            perf_w = perf_weights.get(name, 0.0)
            vol_w = vol_weights.get(name, 0.0)
            combined_weights[name] = 0.6 * perf_w + 0.4 * vol_w
        
        # 歸一化
        total = sum(combined_weights.values())
        if total > 0:
            combined_weights = {k: v / total for k, v in combined_weights.items()}
        else:
            return self._equal_weights()
        
        # 應用最小/最大限制
        for strategy_weight in self.config.strategies:
            name = strategy_weight.strategy_name
            if name in combined_weights:
                combined_weights[name] = max(
                    strategy_weight.min_weight,
                    min(combined_weights[name], strategy_weight.max_weight)
                )
        
        # 再次歸一化
        total = sum(combined_weights.values())
        if total > 0:
            combined_weights = {k: v / total for k, v in combined_weights.items()}
        
        return combined_weights
