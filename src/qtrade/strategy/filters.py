"""
通用信号过滤器

设计理念：
    过滤器作为后处理器，叠加在策略的原始信号上。
    如果当前 bar 不满足过滤条件，不允许新开仓（但不强制平仓）。

用法：
    from qtrade.strategy.filters import trend_filter, volume_filter, htf_trend_filter

    raw_pos = my_indicator_logic(df, params)
    pos = trend_filter(df, raw_pos, min_adx=25)
    pos = volume_filter(df, pos, min_volume_ratio=1.2)
    pos = htf_trend_filter(df, pos, htf_interval="4h")  # 高级时间框架趋势
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from ..indicators.adx import calculate_adx
from ..indicators.volume import calculate_obv
from ..indicators.moving_average import calculate_ema


def trend_filter(
    df: pd.DataFrame,
    raw_pos: pd.Series,
    min_adx: float = 25.0,
    adx_period: int = 14,
    require_uptrend: bool = True,
) -> pd.Series:
    """
    ADX 趋势过滤器

    规则：
    - ADX < min_adx → 无趋势，禁止新开仓
    - require_uptrend=True → 还要求 +DI > -DI（上升趋势）
    - 已有持仓不受影响（由出场规则或策略信号决定平仓）

    Args:
        df:              K线数据
        raw_pos:         原始持仓信号
        min_adx:         最小 ADX 值，低于此值不开新仓
        adx_period:      ADX 周期
        require_uptrend: 是否要求上升趋势（+DI > -DI）

    Returns:
        过滤后的持仓序列
    """
    adx_data = calculate_adx(df, adx_period)
    adx = adx_data["ADX"].values
    plus_di = adx_data["+DI"].values
    minus_di = adx_data["-DI"].values

    raw = raw_pos.values.copy()
    result = np.zeros(len(raw), dtype=float)
    was_in_position = False

    for i in range(len(raw)):
        has_trend = not np.isnan(adx[i]) and adx[i] >= min_adx
        is_uptrend = plus_di[i] > minus_di[i] if require_uptrend else True

        if raw[i] > 0:
            if was_in_position:
                # 已有持仓 → 保持（不因 ADX 下降而强制平仓）
                result[i] = raw[i]
            elif has_trend and is_uptrend:
                # 允许新开仓
                result[i] = raw[i]
                was_in_position = True
            else:
                # 不允许新开仓
                result[i] = 0.0
        else:
            result[i] = 0.0
            was_in_position = False

    return pd.Series(result, index=raw_pos.index)


def volume_filter(
    df: pd.DataFrame,
    raw_pos: pd.Series,
    min_volume_ratio: float = 1.2,
    volume_period: int = 20,
) -> pd.Series:
    """
    成交量过滤器

    规则：当前成交量 > 均量 × min_volume_ratio 时才允许新开仓。
    避免在低流动性时段进场。

    Args:
        df:                K线数据
        raw_pos:           原始持仓信号
        min_volume_ratio:  最小成交量倍数
        volume_period:     均量计算周期

    Returns:
        过滤后的持仓序列
    """
    volume = df["volume"].values
    vol_ma = df["volume"].rolling(volume_period).mean().values

    raw = raw_pos.values.copy()
    result = np.zeros(len(raw), dtype=float)
    was_in_position = False

    for i in range(len(raw)):
        vol_ok = not np.isnan(vol_ma[i]) and volume[i] >= vol_ma[i] * min_volume_ratio

        if raw[i] > 0:
            if was_in_position:
                result[i] = raw[i]
            elif vol_ok:
                result[i] = raw[i]
                was_in_position = True
            else:
                result[i] = 0.0
        else:
            result[i] = 0.0
            was_in_position = False

    return pd.Series(result, index=raw_pos.index)


# ── 高级时间框架重采样映射 ──────────────────────────────────
_RESAMPLE_MAP = {
    # 当前周期 → 可用的高级周期及其 pandas resample rule
    "1m":  {"5m": "5min",  "15m": "15min", "1h": "1h",  "4h": "4h"},
    "5m":  {"15m": "15min", "30m": "30min", "1h": "1h",  "4h": "4h"},
    "15m": {"1h": "1h",     "4h": "4h",     "1d": "1D"},
    "30m": {"2h": "2h",     "4h": "4h",     "1d": "1D"},
    "1h":  {"4h": "4h",     "6h": "6h",     "1d": "1D"},
    "2h":  {"8h": "8h",     "1d": "1D"},
    "4h":  {"1d": "1D"},
}


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    将低级时间框架 K 线重采样为高级时间框架

    Args:
        df:   低级 K 线数据（必须有 DatetimeIndex）
        rule: pandas resample rule, e.g. "4h", "1D"

    Returns:
        高级 K 线 DataFrame（OHLCV）
    """
    return df.resample(rule).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()


def htf_trend_filter(
    df: pd.DataFrame,
    raw_pos: pd.Series,
    htf_interval: str = "4h",
    ema_fast: int = 20,
    ema_slow: int = 50,
    current_interval: str = "1h",
) -> pd.Series:
    """
    多时间框架趋势过滤器

    逻辑：
        1. 将 1h 数据重采样为 4h
        2. 在 4h 上计算 EMA20 和 EMA50
        3. EMA20 > EMA50 = 上升趋势 → 允许做多
        4. EMA20 < EMA50 = 下降趋势 → 禁止新开仓（已有持仓不强制平仓）

    为什么用 EMA 交叉而不是 ADX？
        - ADX 只判断趋势强度，不判断方向
        - EMA 交叉同时判断趋势方向和强度（间距越大趋势越强）
        - 4h EMA20/50 对应日线 EMA5/12.5，是经典趋势判断

    Args:
        df:               低级 K 线数据（e.g. 1h）
        raw_pos:          原始持仓信号
        htf_interval:     高级时间框架，e.g. "4h"
        ema_fast:         快速 EMA 周期（在高级 TF 上），默认 20
        ema_slow:         慢速 EMA 周期（在高级 TF 上），默认 50
        current_interval: 当前 K 线周期，e.g. "1h"

    Returns:
        过滤后的持仓序列
    """
    # 获取重采样规则
    resample_options = _RESAMPLE_MAP.get(current_interval, {})
    resample_rule = resample_options.get(htf_interval)

    if resample_rule is None:
        # 无法重采样，原样返回
        return raw_pos

    # 重采样到高级时间框架
    htf_df = _resample_ohlcv(df, resample_rule)
    if len(htf_df) < ema_slow + 5:
        # 数据不足以计算 EMA，原样返回
        return raw_pos

    # 在高级 TF 上计算 EMA
    ema_f = calculate_ema(htf_df["close"], ema_fast)
    ema_s = calculate_ema(htf_df["close"], ema_slow)

    # 判断趋势方向：EMA_fast > EMA_slow → 上升趋势
    htf_uptrend = (ema_f > ema_s).astype(float)

    # 将高级 TF 的信号映射回低级 TF
    # 使用 reindex + forward fill：每根 4h K 线的趋势应用到其包含的所有 1h K 线
    htf_uptrend_ltf = htf_uptrend.reindex(df.index, method="ffill").fillna(0.0)

    # 应用过滤逻辑（同 trend_filter 的设计：不强制平已有仓位）
    raw = raw_pos.values.copy()
    trend = htf_uptrend_ltf.values
    result = np.zeros(len(raw), dtype=float)
    was_in_position = False

    for i in range(len(raw)):
        is_uptrend = trend[i] > 0.5

        if raw[i] > 0:
            if was_in_position:
                # 已有持仓 → 保持（不因高级 TF 转弱而强制平仓）
                result[i] = raw[i]
            elif is_uptrend:
                # 高级 TF 上升趋势 → 允许新开仓
                result[i] = raw[i]
                was_in_position = True
            else:
                # 高级 TF 下降趋势 → 禁止新开仓
                result[i] = 0.0
        else:
            result[i] = 0.0
            was_in_position = False

    return pd.Series(result, index=raw_pos.index)

