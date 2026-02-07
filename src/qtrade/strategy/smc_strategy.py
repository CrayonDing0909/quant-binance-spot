"""
SMC (Smart Money Concept) 策略

SMC 是一种基于机构交易者行为的交易方法，关注：
- 订单块 (Order Blocks)
- 流动性池 (Liquidity Pools)
- 市场结构 (Market Structure)
- 供需区域 (Supply/Demand Zones)
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from .base import StrategyContext
from . import register_strategy


def identify_order_blocks(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    识别订单块（Order Blocks）
    
    订单块是机构大量买入/卖出的区域，通常是反转点。
    
    Args:
        df: K线数据
        lookback: 回看周期
    
    Returns:
        包含订单块标记的 DataFrame
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]
    volume = df["volume"]
    
    # 计算价格变化和成交量
    price_change = close.pct_change()
    volume_ma = volume.rolling(window=lookback).mean()
    
    # 识别大成交量区域（可能的订单块）
    high_volume = volume > volume_ma * 1.5
    
    # 识别价格反转点
    # 下跌后的大成交量区域 -> 看涨订单块
    # 上涨后的大成交量区域 -> 看跌订单块
    bullish_ob = (
        (close < close.shift(lookback)) &  # 价格下跌
        high_volume &  # 大成交量
        (price_change < -0.02)  # 显著下跌
    )
    
    bearish_ob = (
        (close > close.shift(lookback)) &  # 价格上涨
        high_volume &  # 大成交量
        (price_change > 0.02)  # 显著上涨
    )
    
    return pd.DataFrame({
        "bullish_ob": bullish_ob.astype(float),
        "bearish_ob": bearish_ob.astype(float),
    }, index=df.index)


def identify_liquidity_pools(df: pd.DataFrame, lookback: int = 50) -> pd.Series:
    """
    识别流动性池（Liquidity Pools）
    
    流动性池是大量止损单聚集的区域，通常是：
    - 前期高点（做空止损）
    - 前期低点（做多止损）
    
    Args:
        df: K线数据
        lookback: 回看周期
    
    Returns:
        流动性池标记序列
    """
    high = df["high"]
    low = df["low"]
    
    # 识别前期高点和低点
    recent_high = high.rolling(window=lookback).max()
    recent_low = low.rolling(window=lookback).min()
    
    # 当前价格接近前期高点 -> 上方流动性池
    # 当前价格接近前期低点 -> 下方流动性池
    near_high = (high >= recent_high * 0.98) & (high <= recent_high * 1.02)
    near_low = (low >= recent_low * 0.98) & (low <= recent_low * 1.02)
    
    liquidity = pd.Series(0.0, index=df.index)
    liquidity[near_high] = -1  # 上方流动性（看跌）
    liquidity[near_low] = 1    # 下方流动性（看涨）
    
    return liquidity


def identify_market_structure(df: pd.DataFrame) -> pd.Series:
    """
    识别市场结构（Market Structure）
    
    市场结构包括：
    - 上升趋势：更高的高点和更高的低点
    - 下降趋势：更低的高点和更低的低点
    - 震荡：无明显趋势
    
    Returns:
        市场结构标记：1=上升，-1=下降，0=震荡
    """
    high = df["high"]
    low = df["low"]
    
    # 识别高点和低点
    higher_high = high > high.shift(20)
    higher_low = low > low.shift(20)
    lower_high = high < high.shift(20)
    lower_low = low < low.shift(20)
    
    structure = pd.Series(0.0, index=df.index)
    structure[(higher_high) & (higher_low)] = 1   # 上升趋势
    structure[(lower_high) & (lower_low)] = -1   # 下降趋势
    
    return structure


@register_strategy("smc_basic")
def generate_positions(df: pd.DataFrame, ctx: StrategyContext, params: dict) -> pd.Series:
    """
    SMC 基础策略
    
    策略逻辑：
    1. 识别订单块（机构买入/卖出区域）
    2. 识别流动性池（止损聚集区域）
    3. 结合市场结构判断方向
    
    Args:
        params:
            - order_block_lookback: 订单块回看周期，默认 20
            - liquidity_lookback: 流动性池回看周期，默认 50
            - min_volume_multiplier: 最小成交量倍数，默认 1.5
    """
    order_block_lookback = int(params.get("order_block_lookback", 20))
    liquidity_lookback = int(params.get("liquidity_lookback", 50))
    
    # 识别订单块
    order_blocks = identify_order_blocks(df, order_block_lookback)
    
    # 识别流动性池
    liquidity = identify_liquidity_pools(df, liquidity_lookback)
    
    # 识别市场结构
    structure = identify_market_structure(df)
    
    # 生成信号
    # 看涨订单块 + 下方流动性 + 上升趋势 = 买入
    bullish_signal = (
        (order_blocks["bullish_ob"] > 0) &
        (liquidity > 0) &
        (structure >= 0)
    ).astype(float)
    
    # 看跌订单块 + 上方流动性 + 下降趋势 = 卖出
    bearish_signal = (
        (order_blocks["bearish_ob"] > 0) &
        (liquidity < 0) &
        (structure <= 0)
    ).astype(float)
    
    # 生成持仓
    pos = bullish_signal.copy()
    pos[bearish_signal > 0] = 0
    
    # 避免未来信息泄露
    pos = pos.shift(1).fillna(0.0).clip(0.0, 1.0)
    
    return pos


@register_strategy("smc_orderblock")
def generate_positions_orderblock(df: pd.DataFrame, ctx: StrategyContext, params: dict) -> pd.Series:
    """
    SMC 订单块策略（简化版）
    
    专注于订单块信号，当价格回到订单块区域时交易。
    
    Args:
        params:
            - lookback: 回看周期
            - reentry_threshold: 重新进入订单块的阈值（价格距离订单块的百分比）
    """
    lookback = int(params.get("lookback", 20))
    reentry_threshold = float(params.get("reentry_threshold", 0.02))  # 2%
    
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    
    # 识别订单块
    order_blocks = identify_order_blocks(df, lookback)
    
    # 找到最近的订单块价格
    bullish_ob_price = close[order_blocks["bullish_ob"] > 0]
    bearish_ob_price = close[order_blocks["bearish_ob"] > 0]
    
    # 当价格回到看涨订单块附近时买入
    pos = pd.Series(0.0, index=df.index)
    
    for idx in bullish_ob_price.index:
        ob_price = bullish_ob_price[idx]
        # 在订单块之后，价格回到订单块附近
        future_prices = close[close.index > idx]
        reentry = future_prices[
            (future_prices >= ob_price * (1 - reentry_threshold)) &
            (future_prices <= ob_price * (1 + reentry_threshold))
        ]
        if len(reentry) > 0:
            pos[reentry.index[0]] = 1.0
    
    # 当价格回到看跌订单块附近时卖出
    for idx in bearish_ob_price.index:
        ob_price = bearish_ob_price[idx]
        future_prices = close[close.index > idx]
        reentry = future_prices[
            (future_prices >= ob_price * (1 - reentry_threshold)) &
            (future_prices <= ob_price * (1 + reentry_threshold))
        ]
        if len(reentry) > 0:
            pos[reentry.index[0]] = 0.0
    
    # 避免未来信息泄露
    pos = pos.shift(1).fillna(0.0).clip(0.0, 1.0)
    
    return pos

