"""
RSI (Relative Strength Index) 指标
"""
from __future__ import annotations
import pandas as pd
import numpy as np


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    计算 RSI (相对强弱指标)
    
    RSI 用于衡量价格变动的速度和幅度，范围在 0-100 之间。
    - RSI > 70: 通常被认为是超买
    - RSI < 30: 通常被认为是超卖
    
    Args:
        close: 收盘价序列
        period: RSI 计算周期，默认 14
    
    Returns:
        RSI 值序列，范围 [0, 100]
    
    Example:
        >>> import pandas as pd
        >>> close = pd.Series([100, 102, 101, 103, 105, ...])
        >>> rsi = calculate_rsi(close, period=14)
    """
    delta = close.diff()
    
    # 分离上涨和下跌
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    # 避免除零；loss=0 时 RSI=100，gain=0 时 RSI=0
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # loss=0 → RS=inf → RSI 应为 100
    rsi = rsi.where(~(loss == 0) | gain.isna(), 100.0)
    # gain=0 → RSI=0
    rsi = rsi.where(~(gain == 0) | loss.isna(), 0.0)
    # warmup 期间（gain/loss 都是 NaN）设为 50
    rsi = rsi.fillna(50.0)

    return rsi


def calculate_rsi_divergence(
    close: pd.Series,
    rsi: pd.Series,
    lookback: int = 20
) -> pd.Series:
    """
    检测 RSI 背离
    
    看涨背离：价格创新低，但 RSI 未创新低
    看跌背离：价格创新高，但 RSI 未创新高
    
    Args:
        close: 收盘价序列
        rsi: RSI 值序列
        lookback: 回看周期
    
    Returns:
        背离信号序列：1=看涨背离，-1=看跌背离，0=无背离
    """
    price_low = close.rolling(window=lookback).min()
    price_high = close.rolling(window=lookback).max()
    rsi_low = rsi.rolling(window=lookback).min()
    rsi_high = rsi.rolling(window=lookback).max()
    
    # 看涨背离：价格创新低但 RSI 未创新低
    bullish_divergence = (
        (close == price_low) &
        (rsi > rsi_low.shift(1))
    ).astype(float)
    
    # 看跌背离：价格创新高但 RSI 未创新高
    bearish_divergence = (
        (close == price_high) &
        (rsi < rsi_high.shift(1))
    ).astype(float) * -1
    
    divergence = bullish_divergence + bearish_divergence
    return divergence

