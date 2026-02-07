"""
RSI + ADX + ATR 组合策略

核心理念：
    1. ADX 过滤 → 只在有趋势的市场做单
    2. RSI 择时  → 趋势中找回调入场点
    3. ATR 止损  → 动态止损距离，适应波动率变化
    4. 冷却期    → 止损后不追单

与纯 RSI 策略的区别：
    - 纯 RSI：随时交易 → 震荡市频繁亏损
    - 本策略：趋势确认 + 回调入场 + 动态止损 → 减少无效交易

参数默认值经过初步调优，但建议使用 optimize_params.py 做网格搜索。
"""
from __future__ import annotations
import pandas as pd
from .base import StrategyContext
from . import register_strategy
from ..indicators import calculate_rsi, calculate_adx, calculate_atr
from .exit_rules import apply_exit_rules
from .filters import trend_filter, htf_trend_filter


@register_strategy("rsi_adx_atr")
def generate_positions(df: pd.DataFrame, ctx: StrategyContext, params: dict) -> pd.Series:
    """
    RSI 回调入场 + ADX 趋势过滤 + ATR 动态止损

    入场条件（同时满足）：
        1. ADX > min_adx（有趋势）
        2. +DI > -DI（上升趋势）
        3. RSI 从超卖区向上回升（回调结束）

    出场条件（任一触发）：
        1. RSI > overbought（动力衰竭）
        2. ATR 止损被触发
        3. ATR 止盈被触发
        4. 移动止损被触发（如果启用）

    params:
        rsi_period:        RSI 周期，默认 14
        oversold:          超卖线，默认 35
        overbought:        超买线，默认 70
        min_adx:           最小 ADX 值，默认 20
        adx_period:        ADX 周期，默认 14
        stop_loss_atr:     止损 ATR 倍数，默认 2.0
        take_profit_atr:   止盈 ATR 倍数，默认 3.0
        trailing_stop_atr: 移动止损 ATR 倍数，None = 不用，建议 2.5
        atr_period:        ATR 周期，默认 14
        cooldown_bars:     冷却期，默认 6
        htf_interval:      高级时间框架，e.g. "4h"，None = 不使用
        htf_ema_fast:      高级 TF 快速 EMA，默认 20
        htf_ema_slow:      高级 TF 慢速 EMA，默认 50
    """
    # ── 参数 ──
    rsi_period = int(params.get("rsi_period", 14))
    oversold = float(params.get("oversold", 35))
    overbought = float(params.get("overbought", 70))
    min_adx = float(params.get("min_adx", 20))
    adx_period = int(params.get("adx_period", 14))

    close = df["close"]

    # ── 指标计算 ──
    rsi = calculate_rsi(close, rsi_period)
    rsi_prev = rsi.shift(1)

    # ── 原始信号：RSI 从超卖回升 = 买入，RSI 超买 = 卖出 ──
    # 买入：RSI 上一根 < oversold 且当前 >= oversold（从超卖区回升）
    entry_signal = (rsi_prev < oversold) & (rsi >= oversold)

    # 卖出：RSI > overbought
    exit_signal = rsi > overbought

    # 状态机：生成持仓序列
    raw_pos = pd.Series(0.0, index=df.index)
    in_pos = False

    for i in range(len(df)):
        if not in_pos:
            if entry_signal.iloc[i]:
                in_pos = True
                raw_pos.iloc[i] = 1.0
            else:
                raw_pos.iloc[i] = 0.0
        else:
            if exit_signal.iloc[i]:
                in_pos = False
                raw_pos.iloc[i] = 0.0
            else:
                raw_pos.iloc[i] = 1.0

    # shift(1) 避免未来信息泄露
    raw_pos = raw_pos.shift(1).fillna(0.0)

    # ── ADX 趋势过滤 ──
    filtered_pos = trend_filter(
        df, raw_pos,
        min_adx=min_adx,
        adx_period=adx_period,
        require_uptrend=True,
    )

    # ── 多时间框架趋势过滤（可选）──
    htf_interval = params.get("htf_interval")
    if htf_interval:
        filtered_pos = htf_trend_filter(
            df, filtered_pos,
            htf_interval=htf_interval,
            ema_fast=int(params.get("htf_ema_fast", 20)),
            ema_slow=int(params.get("htf_ema_slow", 50)),
            current_interval=ctx.interval if hasattr(ctx, "interval") else "1h",
        )

    # ── ATR 止损 / 止盈 / 移动止损 ──
    pos = apply_exit_rules(
        df, filtered_pos,
        stop_loss_atr=params.get("stop_loss_atr", 2.0),
        take_profit_atr=params.get("take_profit_atr", 3.0),
        trailing_stop_atr=params.get("trailing_stop_atr", None),
        atr_period=int(params.get("atr_period", 14)),
        cooldown_bars=int(params.get("cooldown_bars", 6)),
    )

    return pos.clip(0.0, 1.0)


@register_strategy("rsi_adx_atr_trailing")
def generate_positions_trailing(df: pd.DataFrame, ctx: StrategyContext, params: dict) -> pd.Series:
    """
    同 rsi_adx_atr，但默认启用移动止损

    移动止损让利润奔跑，在趋势延续时不会过早止盈。
    适合趋势明显的行情（如 BTC 单边牛市）。
    """
    # 默认启用 trailing stop，取消固定 TP
    params_with_trailing = {
        **params,
        "trailing_stop_atr": params.get("trailing_stop_atr", 2.5),
        "take_profit_atr": params.get("take_profit_atr", None),  # 用 trailing 替代 TP
    }
    return generate_positions(df, ctx, params_with_trailing)

