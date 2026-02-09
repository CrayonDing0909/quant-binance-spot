"""
帶止損止盈的 EMA 交叉策略（使用通用 exit_rules）

展示如何將止損/止盈/趨勢過濾疊加到任意策略信號上。
"""
from __future__ import annotations
import pandas as pd
from .base import StrategyContext
from . import register_strategy
from ..indicators import calculate_ema
from .exit_rules import apply_exit_rules
from .filters import trend_filter


@register_strategy("ema_cross_protected")
def generate_positions(df: pd.DataFrame, ctx: StrategyContext, params: dict) -> pd.Series:
    """
    EMA 交叉 + ATR 止損 + ATR 止盈 + ADX 趨勢過濾

    params:
        fast:              EMA 快線週期，預設 20
        slow:              EMA 慢線週期，預設 60
        stop_loss_atr:     止損 ATR 倍數，預設 2.0
        take_profit_atr:   止盈 ATR 倍數，預設 3.0
        trailing_stop_atr: 移動止損 ATR 倍數，None=不用
        cooldown_bars:     出場後冷卻 bar 數，預設 6
        min_adx:           最小 ADX，低於不開倉，預設 25
    """
    fast = int(params.get("fast", 20))
    slow = int(params.get("slow", 60))

    close = df["close"]
    ema_fast = calculate_ema(close, fast)
    ema_slow = calculate_ema(close, slow)

    # ── 1. 原始信號 ──
    raw = (ema_fast > ema_slow).astype(float)
    raw = raw.shift(1).fillna(0.0)

    # ── 2. 趨勢過濾 ──
    min_adx = float(params.get("min_adx", 25))
    if min_adx > 0:
        raw = trend_filter(df, raw, min_adx=min_adx)

    # ── 3. 止損 / 止盈 / 移動止損 ──
    pos = apply_exit_rules(
        df, raw,
        stop_loss_atr=params.get("stop_loss_atr", 2.0),
        take_profit_atr=params.get("take_profit_atr", 3.0),
        trailing_stop_atr=params.get("trailing_stop_atr", None),
        atr_period=int(params.get("atr_period", 14)),
        cooldown_bars=int(params.get("cooldown_bars", 6)),
    )

    return pos.clip(0.0, 1.0)
