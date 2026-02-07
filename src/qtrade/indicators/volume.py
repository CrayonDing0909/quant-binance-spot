"""
成交量指标

- OBV (On Balance Volume): 量价关系
- VWAP (Volume Weighted Average Price): 成交量加权均价
"""
from __future__ import annotations
import pandas as pd
import numpy as np


def calculate_obv(df: pd.DataFrame) -> pd.Series:
    """
    计算 OBV (On Balance Volume)

    价格上涨时累加成交量，下跌时累减。
    用于确认趋势：价格上涨 + OBV 上涨 = 趋势健康。

    Args:
        df: 包含 close, volume 列的 DataFrame

    Returns:
        OBV 序列

    Example:
        >>> obv = calculate_obv(df)
        >>> # OBV 趋势确认
        >>> obv_rising = obv > obv.rolling(20).mean()
    """
    close = df["close"]
    volume = df["volume"]

    direction = pd.Series(0.0, index=df.index)
    direction[close > close.shift(1)] = 1.0
    direction[close < close.shift(1)] = -1.0

    return (direction * volume).cumsum()


def calculate_vwap(df: pd.DataFrame, period: int | None = None) -> pd.Series:
    """
    计算 VWAP (Volume Weighted Average Price)

    Args:
        df: 包含 high, low, close, volume 列的 DataFrame
        period: 滚动窗口（None = 从头累计，常用于日内交易）

    Returns:
        VWAP 序列

    Example:
        >>> vwap = calculate_vwap(df)
        >>> # 价格在 VWAP 上方 = 多头
        >>> bullish = df["close"] > vwap
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    tp_volume = typical_price * df["volume"]

    if period is None:
        return tp_volume.cumsum() / df["volume"].cumsum()
    else:
        return tp_volume.rolling(period).sum() / df["volume"].rolling(period).sum()

