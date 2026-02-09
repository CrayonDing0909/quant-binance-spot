"""
SMC (Smart Money Concept) 策略

SMC 是一種基於機構交易者行為的交易方法，關注：
- 訂單塊 (Order Blocks)
- 流動性池 (Liquidity Pools)
- 市場結構 (Market Structure)
- 供需區域 (Supply/Demand Zones)
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from .base import StrategyContext
from . import register_strategy


def identify_order_blocks(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    識別訂單塊（Order Blocks）
    
    訂單塊是機構大量買入/賣出的區域，通常是反轉點。
    
    Args:
        df: K線數據
        lookback: 回看週期
    
    Returns:
        包含訂單塊標記的 DataFrame
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]
    volume = df["volume"]
    
    # 計算價格變化和成交量
    price_change = close.pct_change()
    volume_ma = volume.rolling(window=lookback).mean()
    
    # 識別大成交量區域（可能的訂單塊）
    high_volume = volume > volume_ma * 1.5
    
    # 識別價格反轉點
    # 下跌後的大成交量區域 -> 看漲訂單塊
    # 上漲後的大成交量區域 -> 看跌訂單塊
    bullish_ob = (
        (close < close.shift(lookback)) &  # 價格下跌
        high_volume &  # 大成交量
        (price_change < -0.02)  # 顯著下跌
    )
    
    bearish_ob = (
        (close > close.shift(lookback)) &  # 價格上漲
        high_volume &  # 大成交量
        (price_change > 0.02)  # 顯著上漲
    )
    
    return pd.DataFrame({
        "bullish_ob": bullish_ob.astype(float),
        "bearish_ob": bearish_ob.astype(float),
    }, index=df.index)


def identify_liquidity_pools(df: pd.DataFrame, lookback: int = 50) -> pd.Series:
    """
    識別流動性池（Liquidity Pools）
    
    流動性池是大量止損單聚集的區域，通常是：
    - 前期高點（做空止損）
    - 前期低點（做多止損）
    
    Args:
        df: K線數據
        lookback: 回看週期
    
    Returns:
        流動性池標記序列
    """
    high = df["high"]
    low = df["low"]
    
    # 識別前期高點和低點
    recent_high = high.rolling(window=lookback).max()
    recent_low = low.rolling(window=lookback).min()
    
    # 當前價格接近前期高點 -> 上方流動性池
    # 當前價格接近前期低點 -> 下方流動性池
    near_high = (high >= recent_high * 0.98) & (high <= recent_high * 1.02)
    near_low = (low >= recent_low * 0.98) & (low <= recent_low * 1.02)
    
    liquidity = pd.Series(0.0, index=df.index)
    liquidity[near_high] = -1  # 上方流動性（看跌）
    liquidity[near_low] = 1    # 下方流動性（看漲）
    
    return liquidity


def identify_market_structure(df: pd.DataFrame) -> pd.Series:
    """
    識別市場結構（Market Structure）
    
    市場結構包括：
    - 上升趨勢：更高的高點和更高的低點
    - 下降趨勢：更低的高點和更低的低點
    - 震盪：無明顯趨勢
    
    Returns:
        市場結構標記：1=上升，-1=下降，0=震盪
    """
    high = df["high"]
    low = df["low"]
    
    # 識別高點和低點
    higher_high = high > high.shift(20)
    higher_low = low > low.shift(20)
    lower_high = high < high.shift(20)
    lower_low = low < low.shift(20)
    
    structure = pd.Series(0.0, index=df.index)
    structure[(higher_high) & (higher_low)] = 1   # 上升趨勢
    structure[(lower_high) & (lower_low)] = -1   # 下降趨勢
    
    return structure


@register_strategy("smc_basic")
def generate_positions(df: pd.DataFrame, ctx: StrategyContext, params: dict) -> pd.Series:
    """
    SMC 基礎策略
    
    策略邏輯：
    1. 識別訂單塊（機構買入/賣出區域）
    2. 識別流動性池（止損聚集區域）
    3. 結合市場結構判斷方向
    
    Args:
        params:
            - order_block_lookback: 訂單塊回看週期，預設 20
            - liquidity_lookback: 流動性池回看週期，預設 50
            - min_volume_multiplier: 最小成交量倍數，預設 1.5
    """
    order_block_lookback = int(params.get("order_block_lookback", 20))
    liquidity_lookback = int(params.get("liquidity_lookback", 50))
    
    # 識別訂單塊
    order_blocks = identify_order_blocks(df, order_block_lookback)
    
    # 識別流動性池
    liquidity = identify_liquidity_pools(df, liquidity_lookback)
    
    # 識別市場結構
    structure = identify_market_structure(df)
    
    # 生成信號
    # 看漲訂單塊 + 下方流動性 + 上升趨勢 = 買入
    bullish_signal = (
        (order_blocks["bullish_ob"] > 0) &
        (liquidity > 0) &
        (structure >= 0)
    ).astype(float)
    
    # 看跌訂單塊 + 上方流動性 + 下降趨勢 = 賣出
    bearish_signal = (
        (order_blocks["bearish_ob"] > 0) &
        (liquidity < 0) &
        (structure <= 0)
    ).astype(float)
    
    # 生成持倉
    pos = bullish_signal.copy()
    pos[bearish_signal > 0] = 0
    
    # 避免未來資訊洩漏
    pos = pos.shift(1).fillna(0.0).clip(0.0, 1.0)
    
    return pos


@register_strategy("smc_orderblock")
def generate_positions_orderblock(df: pd.DataFrame, ctx: StrategyContext, params: dict) -> pd.Series:
    """
    SMC 訂單塊策略（簡化版）
    
    專注於訂單塊信號，當價格回到訂單塊區域時交易。
    
    Args:
        params:
            - lookback: 回看週期
            - reentry_threshold: 重新進入訂單塊的閾值（價格距離訂單塊的百分比）
    """
    lookback = int(params.get("lookback", 20))
    reentry_threshold = float(params.get("reentry_threshold", 0.02))  # 2%
    
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    
    # 識別訂單塊
    order_blocks = identify_order_blocks(df, lookback)
    
    # 找到最近的訂單塊價格
    bullish_ob_price = close[order_blocks["bullish_ob"] > 0]
    bearish_ob_price = close[order_blocks["bearish_ob"] > 0]
    
    # 當價格回到看漲訂單塊附近時買入
    pos = pd.Series(0.0, index=df.index)
    
    for idx in bullish_ob_price.index:
        ob_price = bullish_ob_price[idx]
        # 在訂單塊之後，價格回到訂單塊附近
        future_prices = close[close.index > idx]
        reentry = future_prices[
            (future_prices >= ob_price * (1 - reentry_threshold)) &
            (future_prices <= ob_price * (1 + reentry_threshold))
        ]
        if len(reentry) > 0:
            pos[reentry.index[0]] = 1.0
    
    # 當價格回到看跌訂單塊附近時賣出
    for idx in bearish_ob_price.index:
        ob_price = bearish_ob_price[idx]
        future_prices = close[close.index > idx]
        reentry = future_prices[
            (future_prices >= ob_price * (1 - reentry_threshold)) &
            (future_prices <= ob_price * (1 + reentry_threshold))
        ]
        if len(reentry) > 0:
            pos[reentry.index[0]] = 0.0
    
    # 避免未來資訊洩漏
    pos = pos.shift(1).fillna(0.0).clip(0.0, 1.0)
    
    return pos
