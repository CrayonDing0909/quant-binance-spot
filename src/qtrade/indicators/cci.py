"""
CCI (Commodity Channel Index) 指標

衡量價格與其統計平均值的偏離程度：
- CCI > +100: 超買區域（價格遠高於平均值）
- CCI < -100: 超賣區域（價格遠低於平均值）
- CCI 在 -100 到 +100 之間: 正常波動

公式：
    TP = (High + Low + Close) / 3
    CCI = (TP - SMA(TP, n)) / (0.015 × Mean Deviation)

其中 Mean Deviation = mean(|TP_i - SMA(TP, n)|) for each i in the window.
常數 0.015 由 Donald Lambert（1980）選定，使得約 70-80% 的 CCI 值落在 [-100, +100]。
"""
from __future__ import annotations
import pandas as pd
import numpy as np


def calculate_cci(
    df: pd.DataFrame,
    period: int = 20,
) -> pd.Series:
    """
    計算 CCI (Commodity Channel Index)

    Args:
        df: 包含 high, low, close 列的 DataFrame
        period: CCI 計算週期，預設 20

    Returns:
        CCI 值序列（無上下界，通常在 [-300, +300]）

    Example:
        >>> cci = calculate_cci(df, period=20)
        >>> overbought = cci > 100
        >>> oversold = cci < -100
    """
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    sma_tp = tp.rolling(window=period, min_periods=period).mean()

    # Mean Absolute Deviation from the SMA within each window
    mad = tp.rolling(window=period, min_periods=period).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True
    )

    cci = (tp - sma_tp) / (0.015 * mad)
    return cci.fillna(0.0)
