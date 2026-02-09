"""
Bollinger Bands (布林帶) 指標
"""
from __future__ import annotations
import pandas as pd
from .moving_average import calculate_sma


def calculate_bollinger_bands(
    close: pd.Series,
    period: int = 20,
    std_mult: float = 2.0
) -> pd.DataFrame:
    """
    計算布林帶指標
    
    布林帶由三條線組成：
    - 中軌：移動平均線（SMA）
    - 上軌：中軌 + N倍標準差
    - 下軌：中軌 - N倍標準差
    
    Args:
        close: 收盤價序列
        period: 移動平均週期，預設 20
        std_mult: 標準差倍數，預設 2.0
    
    Returns:
        DataFrame 包含以下列：
        - upper: 上軌
        - middle: 中軌（SMA）
        - lower: 下軌
        - bandwidth: 帶寬（(上軌-下軌)/中軌）
        - %b: 價格在布林帶中的位置（0-1）
    
    Example:
        >>> bb = calculate_bollinger_bands(close, period=20, std_mult=2.0)
        >>> upper = bb['upper']
        >>> middle = bb['middle']
        >>> lower = bb['lower']
        >>> # 價格觸及下軌 -> 可能超賣
        >>> oversold = close < bb['lower']
    """
    middle = calculate_sma(close, period)
    std = close.rolling(window=period).std()
    
    upper = middle + (std * std_mult)
    lower = middle - (std * std_mult)
    
    # 帶寬：衡量布林帶的寬度
    bandwidth = (upper - lower) / middle
    
    # %b：價格在布林帶中的位置
    # 0 = 在下軌，0.5 = 在中軌，1 = 在上軌
    percent_b = (close - lower) / (upper - lower)
    
    return pd.DataFrame({
        "upper": upper,
        "middle": middle,
        "lower": lower,
        "bandwidth": bandwidth,
        "%b": percent_b,
    }, index=close.index)
