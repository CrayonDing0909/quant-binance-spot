"""
自定义策略模板

你可以在这里实现自己的策略逻辑。
"""
from __future__ import annotations
import pandas as pd
from ..strategy.base import StrategyContext
from ..strategy import register_strategy


@register_strategy("my_strategy")
def generate_positions(df: pd.DataFrame, ctx: StrategyContext, params: dict) -> pd.Series:
    """
    自定义策略
    
    Args:
        df: K线数据，包含以下列：
            - open: 开盘价
            - high: 最高价
            - low: 最低价
            - close: 收盘价
            - volume: 成交量
        ctx: 策略上下文，包含 symbol 等信息
        params: 策略参数，从 config 中读取
    
    Returns:
        持仓比例序列 [0, 1]
        - 1.0 = 满仓
        - 0.0 = 空仓
    """
    # TODO: 实现你的策略逻辑
    close = df["close"]
    
    # 示例：简单策略
    signal = (close > close.shift(1)).astype(float)
    
    # ⚠️ 重要：避免未来信息泄露，必须 shift(1)
    pos = signal.shift(1).fillna(0.0).clip(0.0, 1.0)
    
    return pos
