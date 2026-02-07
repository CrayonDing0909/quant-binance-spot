"""
Stochastic Oscillator（随机指标）

衡量收盘价在一段时间内的价格范围中的位置。
- %K > 80: 超买
- %K < 20: 超卖
- %K 上穿 %D: 买入信号
- %K 下穿 %D: 卖出信号
"""
from __future__ import annotations
import pandas as pd


def calculate_stochastic(
    df: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3,
    smooth_k: int = 3,
) -> pd.DataFrame:
    """
    计算 Stochastic Oscillator

    Args:
        df: 包含 high, low, close 列的 DataFrame
        k_period: %K 回看周期，默认 14
        d_period: %D 平滑周期，默认 3
        smooth_k: %K 的额外平滑（1 = Fast Stoch, 3 = Slow Stoch），默认 3

    Returns:
        DataFrame 包含：
        - %K: 快线（经平滑后）
        - %D: 慢线（%K 的 SMA）

    Example:
        >>> stoch = calculate_stochastic(df, k_period=14, d_period=3)
        >>> oversold = stoch["%K"] < 20
        >>> overbought = stoch["%K"] > 80
        >>> # 金叉买入
        >>> buy = (stoch["%K"] > stoch["%D"]) & (stoch["%K"].shift(1) <= stoch["%D"].shift(1))
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # Raw %K
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()

    raw_k = ((close - lowest_low) / (highest_high - lowest_low)) * 100

    # Smooth %K (Fast -> Slow Stochastic)
    k = raw_k.rolling(window=smooth_k).mean() if smooth_k > 1 else raw_k

    # %D = SMA of %K
    d = k.rolling(window=d_period).mean()

    return pd.DataFrame({"%K": k, "%D": d}, index=df.index)

