"""
MACD (Moving Average Convergence Divergence) 指标
"""
from __future__ import annotations
import pandas as pd
from .moving_average import calculate_ema


def calculate_macd(
    close: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> pd.DataFrame:
    """
    计算 MACD 指标
    
    MACD 由三条线组成：
    - MACD 线：快线 EMA - 慢线 EMA
    - 信号线：MACD 线的 EMA
    - 柱状图：MACD 线 - 信号线
    
    Args:
        close: 收盘价序列
        fast_period: 快线周期，默认 12
        slow_period: 慢线周期，默认 26
        signal_period: 信号线周期，默认 9
    
    Returns:
        DataFrame 包含以下列：
        - macd: MACD 线
        - signal: 信号线
        - histogram: 柱状图（MACD - Signal）
    
    Example:
        >>> macd_data = calculate_macd(close, fast_period=12, slow_period=26)
        >>> macd_line = macd_data['macd']
        >>> signal_line = macd_data['signal']
        >>> histogram = macd_data['histogram']
    """
    ema_fast = calculate_ema(close, fast_period)
    ema_slow = calculate_ema(close, slow_period)
    
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    
    return pd.DataFrame({
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram,
    }, index=close.index)

