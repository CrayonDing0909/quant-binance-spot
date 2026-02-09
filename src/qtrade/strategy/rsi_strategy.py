"""
RSI (Relative Strength Index) 策略

RSI 是常用的動量指標，用於判斷超買超賣狀態。
"""
from __future__ import annotations
import pandas as pd
from .base import StrategyContext
from . import register_strategy
from ..indicators import calculate_rsi, calculate_rsi_divergence


@register_strategy("rsi")
def generate_positions(df: pd.DataFrame, ctx: StrategyContext, params: dict) -> pd.Series:
    """
    RSI 策略：基於超買超賣信號
    
    策略邏輯：
    - RSI < 超賣線（如30）：買入信號
    - RSI > 超買線（如70）：賣出信號
    
    Args:
        df: K線數據
        ctx: 策略上下文
        params: 策略參數
            - period: RSI 週期，預設 14
            - oversold: 超賣閾值，預設 30
            - overbought: 超買閾值，預設 70
            - use_divergence: 是否使用背離信號，預設 False
    
    Returns:
        持倉比例序列 [0, 1]
    """
    period = int(params.get("period", 14))
    oversold = float(params.get("oversold", 30))
    overbought = float(params.get("overbought", 70))
    use_divergence = params.get("use_divergence", False)
    
    close = df["close"]
    rsi = calculate_rsi(close, period)
    
    # 基礎信號：RSI 超賣買入，超買賣出
    buy_signal = (rsi < oversold).astype(float)
    sell_signal = (rsi > overbought).astype(float)
    
    # 生成持倉信號
    # 當 RSI < 超賣線時，持倉 = 1（買入）
    # 當 RSI > 超買線時，持倉 = 0（賣出）
    # 否則保持當前狀態（這裡簡化處理，實際可以用更複雜的邏輯）
    signal = buy_signal - sell_signal
    
    # 將信號轉換為持倉（簡化版：買入=1，賣出=0，其他保持）
    pos = signal.copy()
    pos[pos < 0] = 0  # 賣出信號 -> 0
    pos[pos > 0] = 1  # 買入信號 -> 1
    
    # 如果使用背離信號（更複雜的邏輯）
    if use_divergence:
        # 使用指標庫的背離檢測函數
        divergence = calculate_rsi_divergence(close, rsi, lookback=period)
        bullish_divergence = (divergence > 0).astype(float)
        
        pos = pos + bullish_divergence
        pos = pos.clip(0.0, 1.0)
    
    # 避免未來資訊洩漏
    pos = pos.shift(1).fillna(0.0).clip(0.0, 1.0)
    
    return pos


@register_strategy("rsi_momentum")
def generate_positions_momentum(df: pd.DataFrame, ctx: StrategyContext, params: dict) -> pd.Series:
    """
    RSI 動量策略（改進版）：基於 RSI 的動量變化
    
    策略邏輯：
    - RSI 從超賣區域向上突破：買入
    - RSI 從超買區域向下突破：賣出
    - RSI 回到中性區域時退出（可選）
    - 使用狀態機管理持倉，保持持倉直到賣出信號
    
    Args:
        params:
            - period: RSI 週期
            - oversold: 超賣閾值
            - overbought: 超買閾值
            - exit_threshold: 退出閾值（RSI 回到中性區域時退出，設為 None 則禁用）
    """
    period = int(params.get("period", 14))
    oversold = float(params.get("oversold", 30))
    overbought = float(params.get("overbought", 70))
    exit_threshold = params.get("exit_threshold", 50)
    use_exit = exit_threshold is not None
    
    close = df["close"]
    rsi = calculate_rsi(close, period)
    rsi_prev = rsi.shift(1)
    
    # 從超賣區域向上突破（買入信號）
    # 改進：移除 exit_threshold 限制，只要從超賣區域向上突破就買入
    bullish = (rsi_prev < oversold) & (rsi >= oversold)
    
    # 從超買區域向下突破（賣出信號）
    # 改進：移除 exit_threshold 限制，只要從超買區域向下突破就賣出
    bearish = (rsi_prev > overbought) & (rsi <= overbought)
    
    # 可選的退出條件：如果啟用 exit_threshold，當 RSI 從高位回落到中性區域時退出
    # 注意：只有當 RSI 從 > exit_threshold 回落到 < exit_threshold 時才退出
    exit_signal = pd.Series(False, index=df.index)
    if use_exit:
        # RSI 從高於 exit_threshold 回落到低於 exit_threshold
        exit_signal = (rsi_prev >= exit_threshold) & (rsi < exit_threshold)
    
    # 使用狀態機生成持倉：買入時設為1，賣出時設為0，否則保持之前狀態
    pos_state = pd.Series(0.0, index=df.index)
    current_pos = 0.0
    entry_price = None
    
    for i in range(len(df)):
        if bullish.iloc[i] and current_pos == 0.0:
            # 買入信號：從空倉變為持倉
            current_pos = 1.0
            entry_price = close.iloc[i]
        elif bearish.iloc[i] and current_pos > 0.0:
            # 賣出信號：從持倉變為空倉
            current_pos = 0.0
            entry_price = None
        elif use_exit and exit_signal.iloc[i] and current_pos > 0.0:
            # 退出信號：如果啟用，當 RSI 從高位回落到中性區域時退出
            current_pos = 0.0
            entry_price = None
        
        pos_state.iloc[i] = current_pos
    
    # 避免未來資訊洩漏：信號在 t 時刻生成，在 t+1 時刻執行
    pos = pos_state.shift(1).fillna(0.0).clip(0.0, 1.0)
    
    return pos
