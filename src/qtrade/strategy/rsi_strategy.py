"""
RSI (Relative Strength Index) 策略

RSI 是常用的动量指标，用于判断超买超卖状态。
"""
from __future__ import annotations
import pandas as pd
from .base import StrategyContext
from . import register_strategy
from ..indicators import calculate_rsi, calculate_rsi_divergence


@register_strategy("rsi")
def generate_positions(df: pd.DataFrame, ctx: StrategyContext, params: dict) -> pd.Series:
    """
    RSI 策略：基于超买超卖信号
    
    策略逻辑：
    - RSI < 超卖线（如30）：买入信号
    - RSI > 超买线（如70）：卖出信号
    
    Args:
        df: K线数据
        ctx: 策略上下文
        params: 策略参数
            - period: RSI 周期，默认 14
            - oversold: 超卖阈值，默认 30
            - overbought: 超买阈值，默认 70
            - use_divergence: 是否使用背离信号，默认 False
    
    Returns:
        持仓比例序列 [0, 1]
    """
    period = int(params.get("period", 14))
    oversold = float(params.get("oversold", 30))
    overbought = float(params.get("overbought", 70))
    use_divergence = params.get("use_divergence", False)
    
    close = df["close"]
    rsi = calculate_rsi(close, period)
    
    # 基础信号：RSI 超卖买入，超买卖出
    buy_signal = (rsi < oversold).astype(float)
    sell_signal = (rsi > overbought).astype(float)
    
    # 生成持仓信号
    # 当 RSI < 超卖线时，持仓 = 1（买入）
    # 当 RSI > 超买线时，持仓 = 0（卖出）
    # 否则保持当前状态（这里简化处理，实际可以用更复杂的逻辑）
    signal = buy_signal - sell_signal
    
    # 将信号转换为持仓（简化版：买入=1，卖出=0，其他保持）
    pos = signal.copy()
    pos[pos < 0] = 0  # 卖出信号 -> 0
    pos[pos > 0] = 1  # 买入信号 -> 1
    
    # 如果使用背离信号（更复杂的逻辑）
    if use_divergence:
        # 使用指标库的背离检测函数
        divergence = calculate_rsi_divergence(close, rsi, lookback=period)
        bullish_divergence = (divergence > 0).astype(float)
        
        pos = pos + bullish_divergence
        pos = pos.clip(0.0, 1.0)
    
    # 避免未来信息泄露
    pos = pos.shift(1).fillna(0.0).clip(0.0, 1.0)
    
    return pos


@register_strategy("rsi_momentum")
def generate_positions_momentum(df: pd.DataFrame, ctx: StrategyContext, params: dict) -> pd.Series:
    """
    RSI 动量策略（改进版）：基于 RSI 的动量变化
    
    策略逻辑：
    - RSI 从超卖区域向上突破：买入
    - RSI 从超买区域向下突破：卖出
    - RSI 回到中性区域时退出（可选）
    - 使用状态机管理持仓，保持持仓直到卖出信号
    
    Args:
        params:
            - period: RSI 周期
            - oversold: 超卖阈值
            - overbought: 超买阈值
            - exit_threshold: 退出阈值（RSI 回到中性区域时退出，设为 None 则禁用）
    """
    period = int(params.get("period", 14))
    oversold = float(params.get("oversold", 30))
    overbought = float(params.get("overbought", 70))
    exit_threshold = params.get("exit_threshold", 50)
    use_exit = exit_threshold is not None
    
    close = df["close"]
    rsi = calculate_rsi(close, period)
    rsi_prev = rsi.shift(1)
    
    # 从超卖区域向上突破（买入信号）
    # 改进：移除 exit_threshold 限制，只要从超卖区域向上突破就买入
    bullish = (rsi_prev < oversold) & (rsi >= oversold)
    
    # 从超买区域向下突破（卖出信号）
    # 改进：移除 exit_threshold 限制，只要从超买区域向下突破就卖出
    bearish = (rsi_prev > overbought) & (rsi <= overbought)
    
    # 可选的退出条件：如果启用 exit_threshold，当 RSI 从高位回落到中性区域时退出
    # 注意：只有当 RSI 从 > exit_threshold 回落到 < exit_threshold 时才退出
    exit_signal = pd.Series(False, index=df.index)
    if use_exit:
        # RSI 从高于 exit_threshold 回落到低于 exit_threshold
        exit_signal = (rsi_prev >= exit_threshold) & (rsi < exit_threshold)
    
    # 使用状态机生成持仓：买入时设为1，卖出时设为0，否则保持之前状态
    pos_state = pd.Series(0.0, index=df.index)
    current_pos = 0.0
    entry_price = None
    
    for i in range(len(df)):
        if bullish.iloc[i] and current_pos == 0.0:
            # 买入信号：从空仓变为持仓
            current_pos = 1.0
            entry_price = close.iloc[i]
        elif bearish.iloc[i] and current_pos > 0.0:
            # 卖出信号：从持仓变为空仓
            current_pos = 0.0
            entry_price = None
        elif use_exit and exit_signal.iloc[i] and current_pos > 0.0:
            # 退出信号：如果启用，当 RSI 从高位回落到中性区域时退出
            current_pos = 0.0
            entry_price = None
        
        pos_state.iloc[i] = current_pos
    
    # 避免未来信息泄露：信号在 t 时刻生成，在 t+1 时刻执行
    pos = pos_state.shift(1).fillna(0.0).clip(0.0, 1.0)
    
    return pos

