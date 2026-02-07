"""
策略基类和接口定义
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import pandas as pd


@dataclass(frozen=True)
class StrategyContext:
    """策略上下文信息"""
    symbol: str
    interval: str = "1h"


@dataclass
class StrategyState:
    """
    策略状态
    
    用于维护策略的内部状态，如：
    - 当前持仓
    - 止损价格
    - 止盈价格
    - 其他自定义状态
    """
    current_position: float = 0.0  # 当前持仓比例 [0, 1]
    entry_price: float | None = None  # 入场价格
    stop_loss: float | None = None  # 止损价格
    take_profit: float | None = None  # 止盈价格
    custom_state: dict[str, Any] = field(default_factory=dict)  # 自定义状态


class BaseStrategy(ABC):
    """
    策略基类
    
    支持状态管理的策略基类。如果你的策略需要维护状态
    （如止损、止盈），可以继承此类。
    
    示例:
        class MyStrategy(BaseStrategy):
            def generate_positions(self, df, ctx, params, state):
                # 使用 state 维护策略状态
                if state.current_position > 0:
                    # 检查止损
                    if df["close"].iloc[-1] < state.stop_loss:
                        return 0.0
                return 1.0
    """
    
    def __init__(self, name: str):
        self.name = name
        self._state: StrategyState | None = None
    
    @abstractmethod
    def generate_positions(
        self,
        df: pd.DataFrame,
        ctx: StrategyContext,
        params: dict,
        state: StrategyState | None = None
    ) -> pd.Series:
        """
        生成持仓信号
        
        Args:
            df: K线数据
            ctx: 策略上下文
            params: 策略参数
            state: 策略状态（可选）
        
        Returns:
            持仓比例序列 [0, 1]
        """
        raise NotImplementedError
    
    def update_state(
        self,
        state: StrategyState,
        df: pd.DataFrame,
        position: pd.Series
    ) -> StrategyState:
        """
        更新策略状态
        
        子类可以重写此方法来实现自定义状态更新逻辑。
        
        Args:
            state: 当前状态
            df: K线数据
            position: 当前持仓序列
        
        Returns:
            更新后的状态
        """
        # 默认实现：更新当前持仓
        if len(position) > 0:
            state.current_position = float(position.iloc[-1])
        
        return state


# 为了向后兼容，保留函数式策略接口
def generate_positions(df: pd.DataFrame, ctx: StrategyContext, params: dict) -> pd.Series:
    """
    函数式策略接口（向后兼容）
    
    这是函数式策略的接口定义。实际策略应该使用 @register_strategy 装饰器。
    
    Returns:
        持仓比例序列 [0, 1]
    """
    raise NotImplementedError
