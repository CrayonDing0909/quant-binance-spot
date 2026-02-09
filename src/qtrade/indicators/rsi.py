"""
RSI (Relative Strength Index) 指標
"""
from __future__ import annotations
import pandas as pd
import numpy as np


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    計算 RSI (相對強弱指標)
    
    RSI 用於衡量價格變動的速度和幅度，範圍在 0-100 之間。
    - RSI > 70: 通常被認為是超買
    - RSI < 30: 通常被認為是超賣
    
    Args:
        close: 收盤價序列
        period: RSI 計算週期，預設 14
    
    Returns:
        RSI 值序列，範圍 [0, 100]
    
    Example:
        >>> import pandas as pd
        >>> close = pd.Series([100, 102, 101, 103, 105, ...])
        >>> rsi = calculate_rsi(close, period=14)
    """
    delta = close.diff()
    
    # 分離上漲和下跌
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    # 避免除零；loss=0 時 RSI=100，gain=0 時 RSI=0
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # loss=0 → RS=inf → RSI 應為 100
    rsi = rsi.where(~(loss == 0) | gain.isna(), 100.0)
    # gain=0 → RSI=0
    rsi = rsi.where(~(gain == 0) | loss.isna(), 0.0)
    # warmup 期間（gain/loss 都是 NaN）設為 50
    rsi = rsi.fillna(50.0)

    return rsi


def calculate_rsi_divergence(
    close: pd.Series,
    rsi: pd.Series,
    lookback: int = 20
) -> pd.Series:
    """
    檢測 RSI 背離
    
    看漲背離：價格創新低，但 RSI 未創新低
    看跌背離：價格創新高，但 RSI 未創新高
    
    Args:
        close: 收盤價序列
        rsi: RSI 值序列
        lookback: 回看週期
    
    Returns:
        背離信號序列：1=看漲背離，-1=看跌背離，0=無背離
    """
    price_low = close.rolling(window=lookback).min()
    price_high = close.rolling(window=lookback).max()
    rsi_low = rsi.rolling(window=lookback).min()
    rsi_high = rsi.rolling(window=lookback).max()
    
    # 看漲背離：價格創新低但 RSI 未創新低
    bullish_divergence = (
        (close == price_low) &
        (rsi > rsi_low.shift(1))
    ).astype(float)
    
    # 看跌背離：價格創新高但 RSI 未創新高
    bearish_divergence = (
        (close == price_high) &
        (rsi < rsi_high.shift(1))
    ).astype(float) * -1
    
    divergence = bullish_divergence + bearish_divergence
    return divergence
