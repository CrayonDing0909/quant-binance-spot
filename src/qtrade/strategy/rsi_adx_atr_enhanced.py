"""
RSI + ADX + ATR 增強版策略

只保留最有價值的優化：
    1. 部分止盈 → 到達第一目標先平一半，剩餘用移動止損讓利潤奔跑
    2. 動態冷卻期 → 止損後冷卻更久（可選）

其他入場邏輯與原版 rsi_adx_atr 完全一致，確保基本面不變。
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from .base import StrategyContext
from . import register_strategy
from ..indicators import calculate_rsi, calculate_adx, calculate_atr
from .filters import trend_filter, volatility_filter, htf_trend_filter


def apply_partial_tp_exit_rules(
    df: pd.DataFrame,
    raw_pos: pd.Series,
    stop_loss_atr: float | None = 2.0,
    take_profit_atr: float | None = 3.0,
    partial_tp_atr: float | None = 2.0,
    partial_tp_ratio: float = 0.5,
    trailing_stop_atr: float | None = None,
    atr_period: int = 14,
    cooldown_bars: int = 0,
    dynamic_cooldown: bool = False,
) -> pd.Series:
    """
    支援部分止盈的出場規則
    
    部分止盈邏輯：
        1. 價格到達 partial_tp_atr 時，倉位從 1.0 變為 (1 - partial_tp_ratio)
        2. 同時將止損移到入場價（保本止損）
        3. 剩餘倉位等待最終止盈或止損
        4. 如果有設定 trailing_stop_atr，部分止盈後啟動移動止損
    
    動態冷卻期：
        - 止損出場：冷卻期 × 2
        - 止盈出場：冷卻期 × 0.5
    """
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(df)
    
    atr = calculate_atr(df, atr_period).values
    raw = raw_pos.values.copy()
    result = np.zeros(n, dtype=float)
    
    position_state = 0
    entry_price = 0.0
    sl_price = 0.0
    tp_price = 0.0
    partial_tp_price = 0.0
    extreme_since_entry = 0.0
    partial_tp_taken = False
    remaining_size = 1.0
    cooldown_remaining = 0
    
    for i in range(n):
        current_close = close[i]
        current_high = high[i]
        current_low = low[i]
        current_atr = atr[i] if not np.isnan(atr[i]) else 0.0
        
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            result[i] = 0.0
            continue
        
        # ── 持有多倉 ──
        if position_state == 1:
            exit_triggered = False
            exit_type = None
            
            # 更新極值
            if current_high > extreme_since_entry:
                extreme_since_entry = current_high
                # 部分止盈後啟動移動止損
                if partial_tp_taken and trailing_stop_atr is not None and current_atr > 0:
                    sl_price = max(sl_price, extreme_since_entry - trailing_stop_atr * current_atr)
            
            # 檢查部分止盈
            if not partial_tp_taken and partial_tp_price > 0 and current_high >= partial_tp_price:
                remaining_size = 1.0 - partial_tp_ratio
                partial_tp_taken = True
                # 保本止損
                sl_price = max(sl_price, entry_price)
            
            # 檢查止損
            if sl_price > 0 and current_low <= sl_price:
                exit_triggered = True
                exit_type = "stop_loss"
            
            # 檢查止盈
            if tp_price > 0 and current_high >= tp_price:
                exit_triggered = True
                exit_type = "take_profit"
            
            # 策略信號
            if raw[i] <= 0:
                exit_triggered = True
                exit_type = "signal"
            
            if exit_triggered:
                if raw[i] < 0:
                    # 反手做空
                    position_state = -1
                    entry_price = current_close
                    extreme_since_entry = current_low
                    partial_tp_taken = False
                    remaining_size = 1.0
                    
                    if stop_loss_atr and current_atr > 0:
                        sl_price = entry_price + stop_loss_atr * current_atr
                    else:
                        sl_price = 0.0
                    if take_profit_atr and current_atr > 0:
                        tp_price = entry_price - take_profit_atr * current_atr
                    else:
                        tp_price = 0.0
                    if partial_tp_atr and current_atr > 0:
                        partial_tp_price = entry_price - partial_tp_atr * current_atr
                    else:
                        partial_tp_price = 0.0
                    
                    result[i] = -1.0
                else:
                    # 平倉
                    position_state = 0
                    remaining_size = 1.0
                    result[i] = 0.0
                    
                    # 動態冷卻期
                    if dynamic_cooldown and exit_type == "stop_loss":
                        cooldown_remaining = cooldown_bars * 2
                    elif dynamic_cooldown and exit_type == "take_profit":
                        cooldown_remaining = max(1, cooldown_bars // 2)
                    else:
                        cooldown_remaining = cooldown_bars
            else:
                result[i] = remaining_size
        
        # ── 持有空倉 ──
        elif position_state == -1:
            exit_triggered = False
            exit_type = None
            
            if current_low < extreme_since_entry:
                extreme_since_entry = current_low
                if partial_tp_taken and trailing_stop_atr is not None and current_atr > 0:
                    sl_price = min(sl_price, extreme_since_entry + trailing_stop_atr * current_atr)
            
            if not partial_tp_taken and partial_tp_price > 0 and current_low <= partial_tp_price:
                remaining_size = 1.0 - partial_tp_ratio
                partial_tp_taken = True
                sl_price = min(sl_price, entry_price) if sl_price > 0 else entry_price
            
            if sl_price > 0 and current_high >= sl_price:
                exit_triggered = True
                exit_type = "stop_loss"
            
            if tp_price > 0 and current_low <= tp_price:
                exit_triggered = True
                exit_type = "take_profit"
            
            if raw[i] >= 0:
                exit_triggered = True
                exit_type = "signal"
            
            if exit_triggered:
                if raw[i] > 0:
                    position_state = 1
                    entry_price = current_close
                    extreme_since_entry = current_high
                    partial_tp_taken = False
                    remaining_size = 1.0
                    
                    if stop_loss_atr and current_atr > 0:
                        sl_price = entry_price - stop_loss_atr * current_atr
                    else:
                        sl_price = 0.0
                    if take_profit_atr and current_atr > 0:
                        tp_price = entry_price + take_profit_atr * current_atr
                    else:
                        tp_price = 0.0
                    if partial_tp_atr and current_atr > 0:
                        partial_tp_price = entry_price + partial_tp_atr * current_atr
                    else:
                        partial_tp_price = 0.0
                    
                    result[i] = 1.0
                else:
                    position_state = 0
                    remaining_size = 1.0
                    result[i] = 0.0
                    
                    if dynamic_cooldown and exit_type == "stop_loss":
                        cooldown_remaining = cooldown_bars * 2
                    elif dynamic_cooldown and exit_type == "take_profit":
                        cooldown_remaining = max(1, cooldown_bars // 2)
                    else:
                        cooldown_remaining = cooldown_bars
            else:
                result[i] = -remaining_size
        
        # ── 空倉 ──
        else:
            if raw[i] > 0:
                position_state = 1
                entry_price = current_close
                extreme_since_entry = current_high
                partial_tp_taken = False
                remaining_size = 1.0
                
                if stop_loss_atr and current_atr > 0:
                    sl_price = entry_price - stop_loss_atr * current_atr
                else:
                    sl_price = 0.0
                if take_profit_atr and current_atr > 0:
                    tp_price = entry_price + take_profit_atr * current_atr
                else:
                    tp_price = 0.0
                if partial_tp_atr and current_atr > 0:
                    partial_tp_price = entry_price + partial_tp_atr * current_atr
                else:
                    partial_tp_price = 0.0
                
                result[i] = 1.0
                
            elif raw[i] < 0:
                position_state = -1
                entry_price = current_close
                extreme_since_entry = current_low
                partial_tp_taken = False
                remaining_size = 1.0
                
                if stop_loss_atr and current_atr > 0:
                    sl_price = entry_price + stop_loss_atr * current_atr
                else:
                    sl_price = 0.0
                if take_profit_atr and current_atr > 0:
                    tp_price = entry_price - take_profit_atr * current_atr
                else:
                    tp_price = 0.0
                if partial_tp_atr and current_atr > 0:
                    partial_tp_price = entry_price - partial_tp_atr * current_atr
                else:
                    partial_tp_price = 0.0
                
                result[i] = -1.0
            else:
                result[i] = 0.0
    
    return pd.Series(result, index=raw_pos.index)


@register_strategy("rsi_adx_atr_enhanced")
def generate_positions(df: pd.DataFrame, ctx: StrategyContext, params: dict) -> pd.Series:
    """
    RSI + ADX + ATR 增強版 — 保留原版入場邏輯，只優化出場
    
    與原版的差異：
        - 新增部分止盈：到達 partial_tp_atr 後平一半，止損移到保本
        - 可選動態冷卻：止損後冷卻更久
    
    params（除以下新增外，其餘與原版相同）:
        partial_tp_atr:    部分止盈 ATR 倍數，預設 2.0，None = 不用
        partial_tp_ratio:  部分止盈比例，預設 0.5（平一半）
        dynamic_cooldown:  是否動態冷卻，預設 False
    """
    # ── 參數 ──
    rsi_period = int(params.get("rsi_period", 14))
    oversold = float(params.get("oversold", 35))
    overbought = float(params.get("overbought", 70))
    min_adx = float(params.get("min_adx", 20))
    adx_period = int(params.get("adx_period", 14))
    
    supports_short = ctx.supports_short if hasattr(ctx, 'supports_short') else False
    close = df["close"]
    
    # ── 指標計算 ──
    rsi = calculate_rsi(close, rsi_period)
    rsi_prev = rsi.shift(1)
    
    # ── 原始信號（與原版完全相同）──
    long_entry = (rsi_prev < oversold) & (rsi >= oversold)
    long_exit = rsi > overbought
    
    short_entry = (rsi_prev > overbought) & (rsi <= overbought) if supports_short else pd.Series(False, index=df.index)
    short_exit = rsi < oversold
    
    # ── 狀態機（與原版完全相同）──
    raw_pos = pd.Series(0.0, index=df.index)
    state = 0
    
    for i in range(len(df)):
        if state == 0:
            if long_entry.iloc[i]:
                state = 1
                raw_pos.iloc[i] = 1.0
            elif supports_short and short_entry.iloc[i]:
                state = -1
                raw_pos.iloc[i] = -1.0
            else:
                raw_pos.iloc[i] = 0.0
        elif state == 1:
            if long_exit.iloc[i]:
                if supports_short:
                    state = -1
                    raw_pos.iloc[i] = -1.0
                else:
                    state = 0
                    raw_pos.iloc[i] = 0.0
            else:
                raw_pos.iloc[i] = 1.0
        else:  # state == -1
            if short_exit.iloc[i]:
                state = 1
                raw_pos.iloc[i] = 1.0
            else:
                raw_pos.iloc[i] = -1.0
    
    raw_pos = raw_pos.shift(1).fillna(0.0)
    
    # ── ADX 趨勢過濾（與原版相同）──
    filtered_pos = trend_filter(
        df, raw_pos,
        min_adx=min_adx,
        adx_period=adx_period,
        require_uptrend=True,
    )
    
    # ── 波動率過濾（可選，與原版相同）──
    min_atr_ratio = params.get("min_atr_ratio")
    if min_atr_ratio is not None:
        vol_mode = params.get("vol_filter_mode", "absolute")
        filtered_pos = volatility_filter(
            df, filtered_pos,
            min_atr_ratio=float(min_atr_ratio),
            atr_period=int(params.get("atr_period", 14)),
            use_percentile=(vol_mode == "percentile"),
            min_percentile=float(params.get("vol_min_percentile", 25)),
        )
    
    # ── 多時間框架過濾（可選，與原版相同）──
    htf_interval = params.get("htf_interval")
    if htf_interval:
        filtered_pos = htf_trend_filter(
            df, filtered_pos,
            htf_interval=htf_interval,
            ema_fast=int(params.get("htf_ema_fast", 20)),
            ema_slow=int(params.get("htf_ema_slow", 50)),
            current_interval=ctx.interval if hasattr(ctx, "interval") else "1h",
        )
    
    # ── 增強版出場規則（新增部分止盈）──
    pos = apply_partial_tp_exit_rules(
        df, filtered_pos,
        stop_loss_atr=params.get("stop_loss_atr", 2.0),
        take_profit_atr=params.get("take_profit_atr", 3.0),
        partial_tp_atr=params.get("partial_tp_atr", 2.0),
        partial_tp_ratio=float(params.get("partial_tp_ratio", 0.5)),
        trailing_stop_atr=params.get("trailing_stop_atr", None),
        atr_period=int(params.get("atr_period", 14)),
        cooldown_bars=int(params.get("cooldown_bars", 6)),
        dynamic_cooldown=params.get("dynamic_cooldown", False),
    )
    
    if supports_short:
        return pos.clip(-1.0, 1.0)
    else:
        return pos.clip(0.0, 1.0)


@register_strategy("rsi_adx_atr_partial_tp")
def generate_positions_partial_tp(df: pd.DataFrame, ctx: StrategyContext, params: dict) -> pd.Series:
    """
    專門啟用部分止盈的版本
    
    預設設定：
        - partial_tp_atr = 2.0（到達 2 ATR 先平一半）
        - partial_tp_ratio = 0.5（平 50%）
        - trailing_stop_atr = 2.5（部分止盈後啟動移動止損）
    """
    params_with_partial = {
        **params,
        "partial_tp_atr": params.get("partial_tp_atr", 2.0),
        "partial_tp_ratio": params.get("partial_tp_ratio", 0.5),
        "trailing_stop_atr": params.get("trailing_stop_atr", 2.5),
        "take_profit_atr": params.get("take_profit_atr", None),  # 用移動止損替代固定止盈
    }
    return generate_positions(df, ctx, params_with_partial)
