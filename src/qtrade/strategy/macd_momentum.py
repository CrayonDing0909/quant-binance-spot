"""
MACD 動量策略

核心理念：
    1. MACD 交叉 → 趨勢動量確認
    2. ADX 過濾  → 只在有趨勢的市場做單（避免震盪市假信號）
    3. MACD Histogram 加速 → 動量增強確認
    4. ATR 止損止盈 → 動態風控

與 RSI 系列策略的區別：
    - RSI 是反轉/回調指標 → 找超買超賣
    - MACD 是趨勢追蹤指標 → 跟著動量走
    - 互補性強：RSI 在震盪市好用，MACD 在趨勢市好用

策略邏輯：
    入場（做多）：
        1. MACD 線上穿信號線（金叉）
        2. ADX > min_adx（有趨勢）
        3. +DI > -DI（上升趨勢）
        4. MACD Histogram > 0 且遞增（動量加速，可選）
    入場（做空）：
        1. MACD 線下穿信號線（死叉）
        2. ADX > min_adx（有趨勢）
        3. -DI > +DI（下降趨勢）
    出場：
        1. 反向 MACD 交叉
        2. ATR 止損 / 止盈
        3. 冷卻期
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from .base import StrategyContext
from . import register_strategy
from ..indicators import calculate_macd, calculate_adx, calculate_ema
from .exit_rules import apply_exit_rules
from .filters import trend_filter, volatility_filter, htf_trend_filter


@register_strategy("macd_momentum")
def generate_positions(
    df: pd.DataFrame,
    ctx: StrategyContext,
    params: dict,
) -> pd.Series:
    """
    MACD 動量趨勢跟蹤策略

    params:
        # MACD 參數
        macd_fast:         快線 EMA 週期，預設 12
        macd_slow:         慢線 EMA 週期，預設 26
        macd_signal:       信號線週期，預設 9
        require_hist_accel: 是否要求 histogram 加速，預設 False

        # ADX 趨勢過濾
        min_adx:           最小 ADX 值，預設 20
        adx_period:        ADX 週期，預設 14

        # ATR 止損止盈
        stop_loss_atr:     止損 ATR 倍數，預設 2.0
        take_profit_atr:   止盈 ATR 倍數，預設 3.0
        trailing_stop_atr: 移動止損 ATR 倍數，None = 不用
        atr_period:        ATR 週期，預設 14
        cooldown_bars:     冷卻期，預設 4

        # 波動率過濾（可選）
        min_atr_ratio:     最小 ATR/Price 比率，None = 不使用

        # 多時間框架（可選）
        htf_interval:      高級時間框架，e.g. "4h"

    信號輸出：
        - Spot 模式：[0, 1]
        - Futures 模式：[-1, 1]
    """
    # ── 參數 ──
    macd_fast = int(params.get("macd_fast", 12))
    macd_slow = int(params.get("macd_slow", 26))
    macd_signal = int(params.get("macd_signal", 9))
    require_hist_accel = bool(params.get("require_hist_accel", False))
    min_adx = float(params.get("min_adx", 20))
    adx_period = int(params.get("adx_period", 14))

    supports_short = ctx.supports_short if hasattr(ctx, "supports_short") else False
    close = df["close"]

    # ── 指標計算 ──
    macd_data = calculate_macd(close, macd_fast, macd_slow, macd_signal)
    macd_line = macd_data["macd"]
    signal_line = macd_data["signal"]
    histogram = macd_data["histogram"]

    macd_prev = macd_line.shift(1)
    signal_prev = signal_line.shift(1)
    hist_prev = histogram.shift(1)

    # ── 入場信號 ──
    # 金叉：MACD 上穿信號線
    long_cross = (macd_prev <= signal_prev) & (macd_line > signal_line)
    # 死叉：MACD 下穿信號線
    short_cross = (macd_prev >= signal_prev) & (macd_line < signal_line)

    # 可選：要求 histogram 加速（遞增）
    if require_hist_accel:
        hist_accel = histogram > hist_prev
        long_cross = long_cross & hist_accel
        hist_decel = histogram < hist_prev
        short_cross = short_cross & hist_decel

    # ── 狀態機 ──
    raw_pos = pd.Series(0.0, index=df.index)
    state = 0  # 0 = flat, 1 = long, -1 = short

    for i in range(len(df)):
        if state == 0:  # 空倉
            if long_cross.iloc[i]:
                state = 1
                raw_pos.iloc[i] = 1.0
            elif supports_short and short_cross.iloc[i]:
                state = -1
                raw_pos.iloc[i] = -1.0
            else:
                raw_pos.iloc[i] = 0.0
        elif state == 1:  # 多倉
            if short_cross.iloc[i]:
                if supports_short:
                    state = -1
                    raw_pos.iloc[i] = -1.0
                else:
                    state = 0
                    raw_pos.iloc[i] = 0.0
            else:
                raw_pos.iloc[i] = 1.0
        else:  # state == -1，空倉
            if long_cross.iloc[i]:
                state = 1
                raw_pos.iloc[i] = 1.0
            else:
                raw_pos.iloc[i] = -1.0

    # shift(1) 避免未來資訊洩漏
    raw_pos = raw_pos.shift(1).fillna(0.0)

    # ── ADX 趨勢過濾 ──
    filtered_pos = trend_filter(
        df,
        raw_pos,
        min_adx=min_adx,
        adx_period=adx_period,
        require_uptrend=True,
    )

    # ── 波動率過濾（可選）──
    min_atr_ratio = params.get("min_atr_ratio")
    if min_atr_ratio is not None:
        vol_mode = params.get("vol_filter_mode", "absolute")
        filtered_pos = volatility_filter(
            df,
            filtered_pos,
            min_atr_ratio=float(min_atr_ratio),
            atr_period=int(params.get("atr_period", 14)),
            use_percentile=(vol_mode == "percentile"),
            min_percentile=float(params.get("vol_min_percentile", 25)),
        )

    # ── 多時間框架趨勢過濾（可選）──
    htf_interval = params.get("htf_interval")
    if htf_interval:
        filtered_pos = htf_trend_filter(
            df,
            filtered_pos,
            htf_interval=htf_interval,
            ema_fast=int(params.get("htf_ema_fast", 20)),
            ema_slow=int(params.get("htf_ema_slow", 50)),
            current_interval=ctx.interval if hasattr(ctx, "interval") else "1h",
        )

    # ── ATR 止損 / 止盈 / 移動止損 ──
    pos = apply_exit_rules(
        df,
        filtered_pos,
        stop_loss_atr=params.get("stop_loss_atr", 2.0),
        take_profit_atr=params.get("take_profit_atr", 3.0),
        trailing_stop_atr=params.get("trailing_stop_atr", None),
        atr_period=int(params.get("atr_period", 14)),
        cooldown_bars=int(params.get("cooldown_bars", 4)),
    )

    # clip 信號範圍
    if supports_short:
        return pos.clip(-1.0, 1.0)
    else:
        return pos.clip(0.0, 1.0)


@register_strategy("macd_momentum_conservative")
def generate_positions_conservative(
    df: pd.DataFrame,
    ctx: StrategyContext,
    params: dict,
) -> pd.Series:
    """
    MACD 動量策略（保守版）

    預設：
    - 要求 histogram 加速確認
    - 更嚴格的 ADX 門檻
    - 啟用移動止損讓利潤奔跑
    """
    conservative_params = {
        **params,
        "require_hist_accel": params.get("require_hist_accel", True),
        "min_adx": params.get("min_adx", 25),
        "trailing_stop_atr": params.get("trailing_stop_atr", 2.5),
        "take_profit_atr": params.get("take_profit_atr", None),
    }
    return generate_positions(df, ctx, conservative_params)
