"""
MACD (Moving Average Convergence Divergence) 指標
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
    計算 MACD 指標
    
    MACD 由三條線組成：
    - MACD 線：快線 EMA - 慢線 EMA
    - 信號線：MACD 線的 EMA
    - 柱狀圖：MACD 線 - 信號線
    
    Args:
        close: 收盤價序列
        fast_period: 快線週期，預設 12
        slow_period: 慢線週期，預設 26
        signal_period: 信號線週期，預設 9
    
    Returns:
        DataFrame 包含以下列：
        - macd: MACD 線
        - signal: 信號線
        - histogram: 柱狀圖（MACD - Signal）
    
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
