"""
自定義策略模板

你可以在這裡實現自己的策略邏輯。
"""
from __future__ import annotations
import pandas as pd
from ..strategy.base import StrategyContext
from ..strategy import register_strategy


@register_strategy("my_strategy")
def generate_positions(df: pd.DataFrame, ctx: StrategyContext, params: dict) -> pd.Series:
    """
    自定義策略
    
    Args:
        df: K線數據，包含以下列：
            - open: 開盤價
            - high: 最高價
            - low: 最低價
            - close: 收盤價
            - volume: 成交量
        ctx: 策略上下文，包含 symbol 等資訊
        params: 策略參數，從 config 中讀取
    
    Returns:
        持倉比例序列 [0, 1]
        - 1.0 = 滿倉
        - 0.0 = 空倉
    """
    # TODO: 實現你的策略邏輯
    close = df["close"]
    
    # 示例：簡單策略
    signal = (close > close.shift(1)).astype(float)
    
    # ⚠️ 重要：避免未來資訊洩露，必須 shift(1)
    pos = signal.shift(1).fillna(0.0).clip(0.0, 1.0)
    
    return pos
