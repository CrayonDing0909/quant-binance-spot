"""
Bollinger Bands (布林带) 指标
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
    计算布林带指标
    
    布林带由三条线组成：
    - 中轨：移动平均线（SMA）
    - 上轨：中轨 + N倍标准差
    - 下轨：中轨 - N倍标准差
    
    Args:
        close: 收盘价序列
        period: 移动平均周期，默认 20
        std_mult: 标准差倍数，默认 2.0
    
    Returns:
        DataFrame 包含以下列：
        - upper: 上轨
        - middle: 中轨（SMA）
        - lower: 下轨
        - bandwidth: 带宽（(上轨-下轨)/中轨）
        - %b: 价格在布林带中的位置（0-1）
    
    Example:
        >>> bb = calculate_bollinger_bands(close, period=20, std_mult=2.0)
        >>> upper = bb['upper']
        >>> middle = bb['middle']
        >>> lower = bb['lower']
        >>> # 价格触及下轨 -> 可能超卖
        >>> oversold = close < bb['lower']
    """
    middle = calculate_sma(close, period)
    std = close.rolling(window=period).std()
    
    upper = middle + (std * std_mult)
    lower = middle - (std * std_mult)
    
    # 带宽：衡量布林带的宽度
    bandwidth = (upper - lower) / middle
    
    # %b：价格在布林带中的位置
    # 0 = 在下轨，0.5 = 在中轨，1 = 在上轨
    percent_b = (close - lower) / (upper - lower)
    
    return pd.DataFrame({
        "upper": upper,
        "middle": middle,
        "lower": lower,
        "bandwidth": bandwidth,
        "%b": percent_b,
    }, index=close.index)

