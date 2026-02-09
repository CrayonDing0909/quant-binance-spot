"""
成交量指標

- OBV (On Balance Volume): 量價關係
- VWAP (Volume Weighted Average Price): 成交量加權均價
"""
from __future__ import annotations
import pandas as pd
import numpy as np


def calculate_obv(df: pd.DataFrame) -> pd.Series:
    """
    計算 OBV (On Balance Volume)

    價格上漲時累加成交量，下跌時累減。
    用於確認趨勢：價格上漲 + OBV 上漲 = 趨勢健康。

    Args:
        df: 包含 close, volume 列的 DataFrame

    Returns:
        OBV 序列

    Example:
        >>> obv = calculate_obv(df)
        >>> # OBV 趨勢確認
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
    計算 VWAP (Volume Weighted Average Price)

    Args:
        df: 包含 high, low, close, volume 列的 DataFrame
        period: 滾動窗口（None = 從頭累計，常用於日內交易）

    Returns:
        VWAP 序列

    Example:
        >>> vwap = calculate_vwap(df)
        >>> # 價格在 VWAP 上方 = 多頭
        >>> bullish = df["close"] > vwap
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    tp_volume = typical_price * df["volume"]

    if period is None:
        return tp_volume.cumsum() / df["volume"].cumsum()
    else:
        return tp_volume.rolling(period).sum() / df["volume"].rolling(period).sum()
