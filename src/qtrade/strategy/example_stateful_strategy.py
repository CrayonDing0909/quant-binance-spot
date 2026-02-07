"""
带状态管理的策略示例

展示如何使用策略基类实现需要维护状态的策略（如止损、止盈）。
"""
from __future__ import annotations
import pandas as pd
from .base import BaseStrategy, StrategyContext, StrategyState
from . import register_strategy
from ..indicators import calculate_ema


class StatefulEMAStrategy(BaseStrategy):
    """
    带止损止盈的 EMA 交叉策略
    
    这个策略展示了如何使用状态管理来实现止损和止盈。
    """
    
    def generate_positions(
        self,
        df: pd.DataFrame,
        ctx: StrategyContext,
        params: dict,
        state: StrategyState | None = None
    ) -> pd.Series:
        """
        生成持仓信号，带止损止盈逻辑
        """
        if state is None:
            state = StrategyState()
        
        fast = int(params.get("fast", 20))
        slow = int(params.get("slow", 60))
        stop_loss_pct = float(params.get("stop_loss_pct", 0.05))  # 5% 止损
        take_profit_pct = float(params.get("take_profit_pct", 0.10))  # 10% 止盈
        
        close = df["close"]
        ema_fast = calculate_ema(close, fast)
        ema_slow = calculate_ema(close, slow)
        
        # 基础信号
        base_signal = (ema_fast > ema_slow).astype(float)
        
        # 初始化持仓序列
        pos = pd.Series(0.0, index=df.index)
        
        # 逐 bar 处理，维护状态
        for i in range(1, len(df)):
            prev_pos = state.current_position
            current_price = close.iloc[i]
            
            # 如果有持仓，检查止损止盈
            if prev_pos > 0 and state.entry_price is not None:
                # 检查止损
                if current_price <= state.stop_loss:
                    pos.iloc[i] = 0.0
                    state.current_position = 0.0
                    state.entry_price = None
                    state.stop_loss = None
                    state.take_profit = None
                    continue
                
                # 检查止盈
                if current_price >= state.take_profit:
                    pos.iloc[i] = 0.0
                    state.current_position = 0.0
                    state.entry_price = None
                    state.stop_loss = None
                    state.take_profit = None
                    continue
            
            # 根据基础信号决定持仓
            if base_signal.iloc[i] > 0:
                if prev_pos == 0:
                    # 新开仓
                    pos.iloc[i] = 1.0
                    state.current_position = 1.0
                    state.entry_price = current_price
                    state.stop_loss = current_price * (1 - stop_loss_pct)
                    state.take_profit = current_price * (1 + take_profit_pct)
                else:
                    # 保持持仓
                    pos.iloc[i] = 1.0
            else:
                # 平仓
                pos.iloc[i] = 0.0
                state.current_position = 0.0
                state.entry_price = None
                state.stop_loss = None
                state.take_profit = None
        
        # 避免未来信息泄露
        pos = pos.shift(1).fillna(0.0)
        
        return pos


# 创建策略实例并注册
_stateful_ema_strategy = StatefulEMAStrategy("stateful_ema")


# 为了兼容函数式接口，创建一个包装函数
@register_strategy("stateful_ema")
def generate_positions_wrapper(
    df: pd.DataFrame,
    ctx: StrategyContext,
    params: dict
) -> pd.Series:
    """
    包装函数，将函数式接口转换为类式接口
    """
    state = StrategyState()
    return _stateful_ema_strategy.generate_positions(df, ctx, params, state)

