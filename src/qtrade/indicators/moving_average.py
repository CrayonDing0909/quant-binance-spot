"""
移動平均線指標
"""
from __future__ import annotations
import pandas as pd


def calculate_sma(close: pd.Series, period: int) -> pd.Series:
    """
    計算 SMA (簡單移動平均線)
    
    Args:
        close: 收盤價序列
        period: 移動平均週期
    
    Returns:
        SMA 值序列
    """
    return close.rolling(window=period).mean()


def calculate_ema(close: pd.Series, period: int, adjust: bool = False) -> pd.Series:
    """
    計算 EMA (指數移動平均線)
    
    EMA 對近期價格給予更高權重，反應更快。
    
    Args:
        close: 收盤價序列
        period: 移動平均週期
        adjust: 是否使用調整因子，預設 False
    
    Returns:
        EMA 值序列
    """
    return close.ewm(span=period, adjust=adjust).mean()


def calculate_wma(close: pd.Series, period: int) -> pd.Series:
    """
    計算 WMA (加權移動平均線)
    
    WMA 對近期價格給予線性遞減的權重。
    
    Args:
        close: 收盤價序列
        period: 移動平均週期
    
    Returns:
        WMA 值序列
    """
    weights = pd.Series(range(1, period + 1), dtype=float)
    weights = weights / weights.sum()
    
    def weighted_mean(series):
        return (series * weights).sum()
    
    return close.rolling(window=period).apply(weighted_mean, raw=False)
