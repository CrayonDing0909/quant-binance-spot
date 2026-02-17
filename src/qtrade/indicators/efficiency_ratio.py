"""
Efficiency Ratio (ER) — Kaufman 1995

衡量價格移動的「效率」，即方向性移動與總路徑的比值：
    ER = |close[i] - close[i-N]| / Σ|close[j] - close[j-1]| for j in [i-N+1, i]

- ER → 1.0：趨勢市（價格走了很遠，路徑很直）
- ER → 0.0：震盪市（價格來回走，淨位移趨近零）

典型用途：
    - 作為倉位權重：position *= f(ER)
    - Kaufman Adaptive Moving Average (KAMA) 的核心
    - Regime detection（趨勢/震盪分類）

參考：Perry Kaufman, "Smarter Trading" (1995)
"""
from __future__ import annotations
import pandas as pd
import numpy as np


def calculate_efficiency_ratio(close: pd.Series, period: int = 10) -> pd.Series:
    """
    計算 Efficiency Ratio

    Args:
        close:  收盤價序列
        period: 回看週期，預設 10（Kaufman 原始建議）

    Returns:
        ER 序列，範圍 [0, 1]，NaN 用於 warmup 期
    """
    # 方向性移動：|close[i] - close[i-N]|
    direction = (close - close.shift(period)).abs()

    # 總路徑：Σ|close[j] - close[j-1]| for N bars
    volatility = close.diff().abs().rolling(window=period).sum()

    # ER = direction / volatility；volatility=0 時設為 0（完全不動）
    er = direction / volatility.replace(0, np.nan)
    er = er.fillna(0.0)

    return er


def calculate_choppiness_index(
    df: pd.DataFrame, period: int = 14
) -> pd.Series:
    """
    計算 Choppiness Index (CI)

    CI = 100 × log10(Σ(TR, N) / (Highest_High - Lowest_Low)) / log10(N)

    - CI > 61.8：震盪（Fibonacci 閾值）
    - CI < 38.2：趨勢
    - 範圍 [0, 100]

    Args:
        df:     K 線數據（需有 high, low, close）
        period: 回看週期，預設 14

    Returns:
        CI 序列，範圍 [0, 100]
    """
    from .atr import calculate_true_range

    tr = calculate_true_range(df)
    atr_sum = tr.rolling(window=period).sum()

    highest = df["high"].rolling(window=period).max()
    lowest = df["low"].rolling(window=period).min()
    hl_range = (highest - lowest).replace(0, np.nan)

    ci = 100.0 * np.log10(atr_sum / hl_range) / np.log10(period)
    return ci.fillna(50.0)
