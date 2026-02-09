"""
ADX (Average Directional Index) 指標

衡量趨勢強度（不區分方向）：
- ADX < 20: 無趨勢 / 震盪市
- ADX 20-40: 趨勢形成中
- ADX > 40: 強趨勢
- +DI > -DI: 上升趨勢
- -DI > +DI: 下降趨勢

典型用法：ADX > 25 時使用趨勢策略，ADX < 20 時使用均值回歸策略。
"""
from __future__ import annotations
import pandas as pd
import numpy as np


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    計算 ADX、+DI、-DI

    Args:
        df: 包含 high, low, close 列的 DataFrame
        period: ADX 週期，預設 14

    Returns:
        DataFrame 包含：
        - ADX:  趨勢強度（0-100）
        - +DI:  正方向指標
        - -DI:  負方向指標

    Example:
        >>> adx_data = calculate_adx(df, period=14)
        >>> strong_trend = adx_data["ADX"] > 25
        >>> uptrend = adx_data["+DI"] > adx_data["-DI"]
        >>> # 強上升趨勢
        >>> strong_uptrend = strong_trend & uptrend
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # +DM / -DM
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = pd.Series(0.0, index=df.index)
    minus_dm = pd.Series(0.0, index=df.index)

    plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
    minus_dm[(down_move > up_move) & (down_move > 0)] = down_move

    # True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Wilder's smoothing (RMA)
    alpha = 1.0 / period
    atr = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    smooth_plus_dm = plus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    smooth_minus_dm = minus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    # +DI / -DI
    plus_di = (smooth_plus_dm / atr) * 100
    minus_di = (smooth_minus_dm / atr) * 100

    # DX
    di_sum = plus_di + minus_di
    di_diff = (plus_di - minus_di).abs()
    dx = (di_diff / di_sum.replace(0, np.nan)) * 100

    # ADX = RMA of DX
    adx = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    return pd.DataFrame({
        "ADX": adx,
        "+DI": plus_di,
        "-DI": minus_di,
    }, index=df.index)
