"""
移动平均线指标
"""
from __future__ import annotations
import pandas as pd


def calculate_sma(close: pd.Series, period: int) -> pd.Series:
    """
    计算 SMA (简单移动平均线)
    
    Args:
        close: 收盘价序列
        period: 移动平均周期
    
    Returns:
        SMA 值序列
    """
    return close.rolling(window=period).mean()


def calculate_ema(close: pd.Series, period: int, adjust: bool = False) -> pd.Series:
    """
    计算 EMA (指数移动平均线)
    
    EMA 对近期价格给予更高权重，反应更快。
    
    Args:
        close: 收盘价序列
        period: 移动平均周期
        adjust: 是否使用调整因子，默认 False
    
    Returns:
        EMA 值序列
    """
    return close.ewm(span=period, adjust=adjust).mean()


def calculate_wma(close: pd.Series, period: int) -> pd.Series:
    """
    计算 WMA (加权移动平均线)
    
    WMA 对近期价格给予线性递减的权重。
    
    Args:
        close: 收盘价序列
        period: 移动平均周期
    
    Returns:
        WMA 值序列
    """
    weights = pd.Series(range(1, period + 1), dtype=float)
    weights = weights / weights.sum()
    
    def weighted_mean(series):
        return (series * weights).sum()
    
    return close.rolling(window=period).apply(weighted_mean, raw=False)

