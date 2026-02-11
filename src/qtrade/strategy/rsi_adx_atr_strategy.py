"""
RSI + ADX + ATR 組合策略

核心理念：
    1. ADX 過濾 → 只在有趨勢的市場做單
    2. RSI 擇時  → 趨勢中找回調入場點
    3. ATR 止損  → 動態止損距離，適應波動率變化
    4. 冷卻期    → 止損後不追單

與純 RSI 策略的區別：
    - 純 RSI：隨時交易 → 震盪市頻繁虧損
    - 本策略：趨勢確認 + 回調入場 + 動態止損 → 減少無效交易

參數預設值經過初步調優，但建議使用 optimize_params.py 做網格搜索。
"""
from __future__ import annotations
import pandas as pd
from .base import StrategyContext
from . import register_strategy
from ..indicators import calculate_rsi, calculate_adx, calculate_atr
from .exit_rules import apply_exit_rules
from .filters import trend_filter, htf_trend_filter, volatility_filter


@register_strategy("rsi_adx_atr")
def generate_positions(df: pd.DataFrame, ctx: StrategyContext, params: dict) -> pd.Series:
    """
    RSI 回調入場 + ADX 趨勢過濾 + ATR 動態止損

    入場條件（同時滿足）：
        1. ADX > min_adx（有趨勢）
        2. 做多：+DI > -DI（上升趨勢）+ RSI 從超賣區回升
        3. 做空：-DI > +DI（下降趨勢）+ RSI 從超買區回落（僅 Futures）

    出場條件（任一觸發）：
        1. RSI > overbought（動力衰竭）→ 平多（或開空）
        2. RSI < oversold（超賣）→ 平空（或開多）
        3. ATR 止損被觸發
        4. ATR 止盈被觸發
        5. 移動止損被觸發（如果啟用）

    params:
        rsi_period:        RSI 週期，預設 14
        oversold:          超賣線，預設 35
        overbought:        超買線，預設 70
        min_adx:           最小 ADX 值，預設 20
        adx_period:        ADX 週期，預設 14
        stop_loss_atr:     止損 ATR 倍數，預設 2.0
        take_profit_atr:   止盈 ATR 倍數，預設 3.0
        trailing_stop_atr: 移動止損 ATR 倍數，None = 不用，建議 2.5
        atr_period:        ATR 週期，預設 14
        cooldown_bars:     冷卻期，預設 6
        htf_interval:      高級時間框架，e.g. "4h"，None = 不使用
        htf_ema_fast:      高級 TF 快速 EMA，預設 20
        htf_ema_slow:      高級 TF 慢速 EMA，預設 50
        
        # ── 波動率過濾器（可選，防止低波動磨耗）──
        min_atr_ratio:     最小 ATR/Price 比率，None = 不使用，建議 0.005~0.01
        vol_filter_mode:   "absolute" 或 "percentile"，預設 "absolute"
        vol_min_percentile: 百分位模式下的閾值，預設 25
        
    信號輸出：
        - Spot 模式：[0, 1]，負數會被 clip 到 0
        - Futures 模式：[-1, 1]，支援做空
    """
    # ── 參數 ──
    rsi_period = int(params.get("rsi_period", 14))
    oversold = float(params.get("oversold", 35))
    overbought = float(params.get("overbought", 70))
    min_adx = float(params.get("min_adx", 20))
    adx_period = int(params.get("adx_period", 14))
    
    # 是否支援做空（從 context 取得）
    supports_short = ctx.supports_short if hasattr(ctx, 'supports_short') else False

    close = df["close"]

    # ── 指標計算 ──
    rsi = calculate_rsi(close, rsi_period)
    rsi_prev = rsi.shift(1)

    # ── 原始信號 ──
    # 做多信號：RSI 上一根 < oversold 且當前 >= oversold（從超賣區回升）
    long_entry = (rsi_prev < oversold) & (rsi >= oversold)
    long_exit = rsi > overbought
    
    # 做空信號（僅 Futures）：RSI 上一根 > overbought 且當前 <= overbought（從超買區回落）
    short_entry = (rsi_prev > overbought) & (rsi <= overbought) if supports_short else pd.Series(False, index=df.index)
    short_exit = rsi < oversold

    # 狀態機：生成持倉序列
    # 狀態：0 = 空倉，1 = 多倉，-1 = 空倉（做空）
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
        elif state == 1:  # 持有多倉
            if long_exit.iloc[i]:
                if supports_short:
                    # Futures: 平多後可以開空
                    state = -1
                    raw_pos.iloc[i] = -1.0
                else:
                    # Spot: 只能平倉
                    state = 0
                    raw_pos.iloc[i] = 0.0
            else:
                raw_pos.iloc[i] = 1.0
        else:  # state == -1，持有空倉
            if short_exit.iloc[i]:
                # 平空後開多
                state = 1
                raw_pos.iloc[i] = 1.0
            else:
                raw_pos.iloc[i] = -1.0

    # shift(1) 避免未來資訊洩漏
    raw_pos = raw_pos.shift(1).fillna(0.0)

    # ── ADX 趨勢過濾 ──
    filtered_pos = trend_filter(
        df, raw_pos,
        min_adx=min_adx,
        adx_period=adx_period,
        require_uptrend=True,
    )

    # ── 波動率過濾（可選，防止低波動磨耗）──
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

    # ── 多時間框架趨勢過濾（可選）──
    htf_interval = params.get("htf_interval")
    if htf_interval:
        filtered_pos = htf_trend_filter(
            df, filtered_pos,
            htf_interval=htf_interval,
            ema_fast=int(params.get("htf_ema_fast", 20)),
            ema_slow=int(params.get("htf_ema_slow", 50)),
            current_interval=ctx.interval if hasattr(ctx, "interval") else "1h",
        )

    # ── ATR 止損 / 止盈 / 移動止損 ──
    pos = apply_exit_rules(
        df, filtered_pos,
        stop_loss_atr=params.get("stop_loss_atr", 2.0),
        take_profit_atr=params.get("take_profit_atr", 3.0),
        trailing_stop_atr=params.get("trailing_stop_atr", None),
        atr_period=int(params.get("atr_period", 14)),
        cooldown_bars=int(params.get("cooldown_bars", 6)),
    )

    # 根據市場類型 clip 信號範圍
    # Spot: [0, 1]，Futures: [-1, 1]
    if supports_short:
        return pos.clip(-1.0, 1.0)
    else:
        return pos.clip(0.0, 1.0)


@register_strategy("rsi_adx_atr_trailing")
def generate_positions_trailing(df: pd.DataFrame, ctx: StrategyContext, params: dict) -> pd.Series:
    """
    同 rsi_adx_atr，但預設啟用移動止損

    移動止損讓利潤奔跑，在趨勢延續時不會過早止盈。
    適合趨勢明顯的行情（如 BTC 單邊牛市）。
    """
    # 預設啟用 trailing stop，取消固定 TP
    params_with_trailing = {
        **params,
        "trailing_stop_atr": params.get("trailing_stop_atr", 2.5),
        "take_profit_atr": params.get("take_profit_atr", None),  # 用 trailing 替代 TP
    }
    return generate_positions(df, ctx, params_with_trailing)
