"""
布林帶均值回歸策略

核心理念：
    1. 布林帶 %B  → 價格偏離均值的程度
    2. Stochastic  → 確認超買超賣動量反轉
    3. 帶寬過濾    → 帶寬收縮時不交易（突破行情，均值回歸失效）
    4. ATR 止損    → 嚴格風控，防止趨勢行情持續偏離

與 RSI 策略的區別：
    - RSI 只看動量強弱
    - BB + Stochastic 結合了統計偏離 + 動量反轉
    - 帶寬過濾避免在趨勢市場硬做均值回歸（最常見的虧損原因）

策略邏輯：
    入場（做多）：
        1. %B < lower_threshold（價格在下軌附近）
        2. Stochastic %K 從超賣區回升（%K 上穿 %D）
        3. 帶寬 > min_bandwidth（有足夠波動）
    入場（做空）：
        1. %B > upper_threshold（價格在上軌附近）
        2. Stochastic %K 從超買區回落（%K 下穿 %D）
        3. 帶寬 > min_bandwidth
    出場：
        1. %B 回歸中軌（0.5 附近）
        2. ATR 止損 / 止盈
        3. 反向 Stochastic 交叉
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from .base import StrategyContext
from . import register_strategy
from ..indicators import (
    calculate_bollinger_bands,
    calculate_stochastic,
    calculate_atr,
)
from .exit_rules import apply_exit_rules
from .filters import volatility_filter


@register_strategy("bb_mean_reversion")
def generate_positions(
    df: pd.DataFrame,
    ctx: StrategyContext,
    params: dict,
) -> pd.Series:
    """
    布林帶 + Stochastic 均值回歸策略

    params:
        # 布林帶參數
        bb_period:          布林帶週期，預設 20
        bb_std:             標準差倍數，預設 2.0
        lower_threshold:    %B 做多閾值，預設 0.05（下軌附近）
        upper_threshold:    %B 做空閾值，預設 0.95（上軌附近）
        exit_threshold:     %B 出場閾值，預設 0.5（回到中軌）
        min_bandwidth:      最小帶寬（避免窄幅震盪），預設 0.02

        # Stochastic 參數
        stoch_k:            %K 週期，預設 14
        stoch_d:            %D 週期，預設 3
        stoch_smooth:       %K 平滑（3 = Slow Stoch），預設 3
        stoch_oversold:     超賣線，預設 20
        stoch_overbought:   超買線，預設 80

        # ATR 止損止盈
        stop_loss_atr:      止損 ATR 倍數，預設 1.5（均值回歸止損要嚴格）
        take_profit_atr:    止盈 ATR 倍數，預設 2.0
        atr_period:         ATR 週期，預設 14
        cooldown_bars:      冷卻期，預設 4

    信號輸出：
        - Spot 模式：[0, 1]
        - Futures 模式：[-1, 1]
    """
    # ── 參數 ──
    bb_period = int(params.get("bb_period", 20))
    bb_std = float(params.get("bb_std", 2.0))
    lower_threshold = float(params.get("lower_threshold", 0.05))
    upper_threshold = float(params.get("upper_threshold", 0.95))
    exit_threshold_low = float(params.get("exit_threshold_low", 0.45))
    exit_threshold_high = float(params.get("exit_threshold_high", 0.55))
    min_bandwidth = float(params.get("min_bandwidth", 0.02))

    stoch_k = int(params.get("stoch_k", 14))
    stoch_d = int(params.get("stoch_d", 3))
    stoch_smooth = int(params.get("stoch_smooth", 3))
    stoch_oversold = float(params.get("stoch_oversold", 20))
    stoch_overbought = float(params.get("stoch_overbought", 80))

    supports_short = ctx.supports_short if hasattr(ctx, "supports_short") else False
    close = df["close"]

    # ── 指標計算 ──
    bb = calculate_bollinger_bands(close, bb_period, bb_std)
    percent_b = bb["%b"]
    bandwidth = bb["bandwidth"]

    stoch = calculate_stochastic(df, stoch_k, stoch_d, stoch_smooth)
    stoch_k_val = stoch["%K"]
    stoch_d_val = stoch["%D"]

    stoch_k_prev = stoch_k_val.shift(1)
    stoch_d_prev = stoch_d_val.shift(1)

    # ── 入場信號 ──
    # 做多：%B 低 + Stochastic 金叉（從超賣區回升）+ 帶寬足夠
    long_entry = (
        (percent_b < lower_threshold)
        & (stoch_k_prev <= stoch_d_prev)
        & (stoch_k_val > stoch_d_val)
        & (stoch_k_val < stoch_oversold + 20)  # 確保是從超賣區出來
        & (bandwidth > min_bandwidth)
    )

    # 做空：%B 高 + Stochastic 死叉（從超買區回落）+ 帶寬足夠
    short_entry = (
        (percent_b > upper_threshold)
        & (stoch_k_prev >= stoch_d_prev)
        & (stoch_k_val < stoch_d_val)
        & (stoch_k_val > stoch_overbought - 20)  # 確保是從超買區出來
        & (bandwidth > min_bandwidth)
    ) if supports_short else pd.Series(False, index=df.index)

    # 出場信號：%B 回到中軌附近
    long_exit = percent_b > exit_threshold_high
    short_exit = percent_b < exit_threshold_low

    # ── 狀態機 ──
    raw_pos = pd.Series(0.0, index=df.index)
    state = 0  # 0 = flat, 1 = long, -1 = short

    for i in range(len(df)):
        if state == 0:  # 空倉
            if long_entry.iloc[i]:
                state = 1
                raw_pos.iloc[i] = 1.0
            elif supports_short and short_entry.iloc[i]:
                state = -1
                raw_pos.iloc[i] = -1.0
            else:
                raw_pos.iloc[i] = 0.0
        elif state == 1:  # 多倉
            if long_exit.iloc[i]:
                # 均值回歸：回到中軌就出場，不反手
                state = 0
                raw_pos.iloc[i] = 0.0
            else:
                raw_pos.iloc[i] = 1.0
        else:  # state == -1，空倉
            if short_exit.iloc[i]:
                state = 0
                raw_pos.iloc[i] = 0.0
            else:
                raw_pos.iloc[i] = -1.0

    # shift(1) 避免未來資訊洩漏
    raw_pos = raw_pos.shift(1).fillna(0.0)

    # ── 波動率過濾（可選）──
    min_atr_ratio = params.get("min_atr_ratio")
    if min_atr_ratio is not None:
        raw_pos = volatility_filter(
            df,
            raw_pos,
            min_atr_ratio=float(min_atr_ratio),
            atr_period=int(params.get("atr_period", 14)),
        )

    # ── ATR 止損 / 止盈 ──
    # 均值回歸策略止損要嚴格（趨勢可能延續）
    pos = apply_exit_rules(
        df,
        raw_pos,
        stop_loss_atr=params.get("stop_loss_atr", 1.5),
        take_profit_atr=params.get("take_profit_atr", 2.0),
        trailing_stop_atr=params.get("trailing_stop_atr", None),
        atr_period=int(params.get("atr_period", 14)),
        cooldown_bars=int(params.get("cooldown_bars", 4)),
    )

    # clip 信號範圍
    if supports_short:
        return pos.clip(-1.0, 1.0)
    else:
        return pos.clip(0.0, 1.0)


@register_strategy("bb_mean_reversion_strict")
def generate_positions_strict(
    df: pd.DataFrame,
    ctx: StrategyContext,
    params: dict,
) -> pd.Series:
    """
    布林帶均值回歸策略（嚴格版）

    預設：
    - 更嚴格的入場門檻（%B 更極端）
    - 更窄的止損（均值回歸失敗 = 趨勢行情，快速認錯）
    - 不設移動止損（均值回歸不追趨勢）
    """
    strict_params = {
        **params,
        "lower_threshold": params.get("lower_threshold", 0.0),
        "upper_threshold": params.get("upper_threshold", 1.0),
        "stop_loss_atr": params.get("stop_loss_atr", 1.2),
        "take_profit_atr": params.get("take_profit_atr", 1.8),
        "stoch_oversold": params.get("stoch_oversold", 15),
        "stoch_overbought": params.get("stoch_overbought", 85),
    }
    return generate_positions(df, ctx, strict_params)
