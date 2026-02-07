"""
ATR (Average True Range) 指标

ATR 衡量价格波动幅度，常用于：
- 设定止损距离（例如 2x ATR）
- 仓位管理（波动越大仓位越小）
- 判断市场波动率
"""
from __future__ import annotations
import pandas as pd
import numpy as np


def calculate_true_range(df: pd.DataFrame) -> pd.Series:
    """
    计算 True Range

    TR = max(
        High - Low,
        abs(High - Previous Close),
        abs(Low - Previous Close)
    )

    Args:
        df: 包含 high, low, close 列的 DataFrame

    Returns:
        True Range 序列
    """
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    计算 ATR (Average True Range)

    Args:
        df: 包含 high, low, close 列的 DataFrame
        period: ATR 周期，默认 14

    Returns:
        ATR 值序列

    Example:
        >>> atr = calculate_atr(df, period=14)
        >>> # 2x ATR 止损
        >>> stop_loss = df["close"] - 2 * atr
        >>> # 基于波动率的仓位：波动越大仓位越小
        >>> position_size = target_risk / atr
    """
    tr = calculate_true_range(df)
    # 使用 RMA (Wilder's smoothing) 与传统 ATR 一致
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def calculate_atr_percent(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    计算 ATR 百分比（ATR / Close * 100）

    便于跨不同价位的标的比较波动率。

    Args:
        df: K线数据
        period: ATR 周期

    Returns:
        ATR% 序列
    """
    atr = calculate_atr(df, period)
    return (atr / df["close"]) * 100

