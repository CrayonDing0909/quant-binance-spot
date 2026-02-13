"""
多因子組合策略

核心理念：
    結合多個獨立信號源，用加權評分機制決定持倉方向和強度。
    單一因子容易在特定市場環境下失效，多因子組合提高穩健性。

因子類別：
    1. 趨勢因子 — EMA 交叉 + ADX（跟隨趨勢方向）
    2. 動量因子 — MACD Histogram（動量加速/減速）
    3. 均值回歸因子 — Bollinger %B（偏離程度）
    4. 成交量因子 — OBV 趨勢確認（量價配合）

評分機制：
    每個因子產生 [-1, +1] 的分數
    加權平均後映射到持倉大小：
    - composite > entry_threshold → 做多
    - composite < -entry_threshold → 做空（Futures）
    - |composite| < exit_threshold → 平倉

與單因子策略的區別：
    - 單因子：一個信號決定全部 → 假信號率高
    - 多因子：多個信號投票 → 減少假信號，但可能錯過快速行情
    - 最適合中長期持倉的趨勢策略
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from .base import StrategyContext
from . import register_strategy
from ..indicators import (
    calculate_ema,
    calculate_macd,
    calculate_adx,
    calculate_bollinger_bands,
    calculate_obv,
)
from .exit_rules import apply_exit_rules
from .filters import volatility_filter


def _trend_factor(
    df: pd.DataFrame,
    ema_fast: int = 20,
    ema_slow: int = 50,
    adx_period: int = 14,
    min_adx: float = 15,
) -> pd.Series:
    """
    趨勢因子：EMA 交叉 + ADX 強度

    輸出 [-1, 1]：
    - EMA fast > slow 且 ADX > min → +1（強上升趨勢）
    - EMA fast < slow 且 ADX > min → -1（強下降趨勢）
    - ADX < min → 0（無趨勢）
    """
    close = df["close"]
    ema_f = calculate_ema(close, ema_fast)
    ema_s = calculate_ema(close, ema_slow)

    adx_data = calculate_adx(df, adx_period)
    adx = adx_data["ADX"]

    # 基礎方向
    direction = pd.Series(0.0, index=df.index)
    direction[ema_f > ema_s] = 1.0
    direction[ema_f < ema_s] = -1.0

    # ADX 過濾：無趨勢時歸零
    has_trend = adx >= min_adx
    factor = direction * has_trend.astype(float)

    # 用 ADX 強度調節幅度（ADX 越高信號越強）
    # 將 ADX 正規化到 [0, 1]（假設有效範圍 15~50）
    adx_strength = ((adx - min_adx) / (50 - min_adx)).clip(0.0, 1.0)
    factor = factor * adx_strength

    return factor.fillna(0.0)


def _momentum_factor(
    df: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    lookback: int = 10,
) -> pd.Series:
    """
    動量因子：MACD Histogram 方向 + 加速度

    輸出 [-1, 1]：
    - Histogram > 0 且遞增 → +1（多頭動量加速）
    - Histogram < 0 且遞減 → -1（空頭動量加速）
    - 其他 → 衰減
    """
    close = df["close"]
    macd_data = calculate_macd(close, macd_fast, macd_slow, macd_signal)
    histogram = macd_data["histogram"]

    # 方向
    direction = pd.Series(0.0, index=df.index)
    direction[histogram > 0] = 1.0
    direction[histogram < 0] = -1.0

    # 加速度：histogram 在增大還是縮小
    hist_change = histogram.diff()
    is_accelerating = (
        ((histogram > 0) & (hist_change > 0))
        | ((histogram < 0) & (hist_change < 0))
    )

    # 加速 = 強信號(1.0)，減速 = 弱信號(0.5)
    strength = pd.Series(0.5, index=df.index)
    strength[is_accelerating] = 1.0

    factor = direction * strength

    return factor.fillna(0.0)


def _mean_reversion_factor(
    df: pd.DataFrame,
    bb_period: int = 20,
    bb_std: float = 2.0,
) -> pd.Series:
    """
    均值回歸因子：Bollinger %B 偏離

    輸出 [-1, 1]：
    - %B 接近 0（下軌）→ +1（超賣，看多）
    - %B 接近 1（上軌）→ -1（超買，看空）
    - %B 接近 0.5（中軌）→ 0（均衡）

    注意：這個因子方向與趨勢因子相反！
    在趨勢市 → 被趨勢因子壓制
    在震盪市 → 成為主導因子
    """
    close = df["close"]
    bb = calculate_bollinger_bands(close, bb_period, bb_std)
    percent_b = bb["%b"]

    # 線性映射：%B 0→+1, 0.5→0, 1→-1
    factor = 1.0 - 2.0 * percent_b

    return factor.clip(-1.0, 1.0).fillna(0.0)


def _volume_factor(
    df: pd.DataFrame,
    obv_ema: int = 20,
) -> pd.Series:
    """
    成交量因子：OBV 趨勢

    輸出 [-1, 1]：
    - OBV 在均線上方 → +1（量價配合上升）
    - OBV 在均線下方 → -1（量價配合下降）
    """
    obv = calculate_obv(df)
    obv_ma = calculate_ema(obv, obv_ema)

    direction = pd.Series(0.0, index=df.index)
    direction[obv > obv_ma] = 1.0
    direction[obv < obv_ma] = -1.0

    return direction.fillna(0.0)


@register_strategy("multi_factor")
def generate_positions(
    df: pd.DataFrame,
    ctx: StrategyContext,
    params: dict,
) -> pd.Series:
    """
    多因子加權評分策略

    params:
        # 因子權重（權重總和不必為 1，會自動正規化）
        w_trend:           趨勢因子權重，預設 0.35
        w_momentum:        動量因子權重，預設 0.30
        w_mean_reversion:  均值回歸因子權重，預設 0.15
        w_volume:          成交量因子權重，預設 0.20

        # 閾值
        entry_threshold:   開倉閾值（composite > threshold），預設 0.3
        exit_threshold:    平倉閾值（|composite| < threshold），預設 0.1

        # 趨勢因子參數
        ema_fast:          快速 EMA 週期，預設 20
        ema_slow:          慢速 EMA 週期，預設 50
        min_adx:           最小 ADX 值，預設 15
        adx_period:        ADX 週期，預設 14

        # 動量因子參數
        macd_fast:         MACD 快線週期，預設 12
        macd_slow:         MACD 慢線週期，預設 26
        macd_signal:       MACD 信號線週期，預設 9

        # 均值回歸因子參數
        bb_period:         布林帶週期，預設 20
        bb_std:            標準差倍數，預設 2.0

        # 成交量因子參數
        obv_ema:           OBV EMA 週期，預設 20

        # ATR 止損止盈
        stop_loss_atr:     止損 ATR 倍數，預設 2.0
        take_profit_atr:   止盈 ATR 倍數，預設 3.5
        trailing_stop_atr: 移動止損，None = 不用
        atr_period:        ATR 週期，預設 14
        cooldown_bars:     冷卻期，預設 3

    信號輸出：
        - Spot 模式：[0, 1]
        - Futures 模式：[-1, 1]
    """
    # ── 因子權重 ──
    w_trend = float(params.get("w_trend", 0.35))
    w_momentum = float(params.get("w_momentum", 0.30))
    w_mean_reversion = float(params.get("w_mean_reversion", 0.15))
    w_volume = float(params.get("w_volume", 0.20))

    total_weight = w_trend + w_momentum + w_mean_reversion + w_volume
    if total_weight <= 0:
        total_weight = 1.0

    # 正規化權重
    w_trend /= total_weight
    w_momentum /= total_weight
    w_mean_reversion /= total_weight
    w_volume /= total_weight

    # ── 閾值 ──
    entry_threshold = float(params.get("entry_threshold", 0.3))
    exit_threshold = float(params.get("exit_threshold", 0.1))

    supports_short = ctx.supports_short if hasattr(ctx, "supports_short") else False

    # ── 計算各因子 ──
    f_trend = _trend_factor(
        df,
        ema_fast=int(params.get("ema_fast", 20)),
        ema_slow=int(params.get("ema_slow", 50)),
        adx_period=int(params.get("adx_period", 14)),
        min_adx=float(params.get("min_adx", 15)),
    )

    f_momentum = _momentum_factor(
        df,
        macd_fast=int(params.get("macd_fast", 12)),
        macd_slow=int(params.get("macd_slow", 26)),
        macd_signal=int(params.get("macd_signal", 9)),
    )

    f_mean_rev = _mean_reversion_factor(
        df,
        bb_period=int(params.get("bb_period", 20)),
        bb_std=float(params.get("bb_std", 2.0)),
    )

    f_volume = _volume_factor(
        df,
        obv_ema=int(params.get("obv_ema", 20)),
    )

    # ── 加權合成 ──
    composite = (
        w_trend * f_trend
        + w_momentum * f_momentum
        + w_mean_reversion * f_mean_rev
        + w_volume * f_volume
    )

    # ── 狀態機：基於 composite 分數決定持倉 ──
    raw_pos = pd.Series(0.0, index=df.index)
    state = 0  # 0 = flat, 1 = long, -1 = short

    composite_vals = composite.values

    for i in range(len(df)):
        score = composite_vals[i]
        if np.isnan(score):
            raw_pos.iloc[i] = 0.0
            continue

        if state == 0:  # 空倉
            if score > entry_threshold:
                state = 1
                raw_pos.iloc[i] = 1.0
            elif supports_short and score < -entry_threshold:
                state = -1
                raw_pos.iloc[i] = -1.0
            else:
                raw_pos.iloc[i] = 0.0
        elif state == 1:  # 多倉
            if score < exit_threshold:
                if supports_short and score < -entry_threshold:
                    # 反手做空
                    state = -1
                    raw_pos.iloc[i] = -1.0
                else:
                    # 平倉
                    state = 0
                    raw_pos.iloc[i] = 0.0
            else:
                raw_pos.iloc[i] = 1.0
        else:  # state == -1，空倉做空
            if score > -exit_threshold:
                if score > entry_threshold:
                    # 反手做多
                    state = 1
                    raw_pos.iloc[i] = 1.0
                else:
                    # 平倉
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
    pos = apply_exit_rules(
        df,
        raw_pos,
        stop_loss_atr=params.get("stop_loss_atr", 2.0),
        take_profit_atr=params.get("take_profit_atr", 3.5),
        trailing_stop_atr=params.get("trailing_stop_atr", None),
        atr_period=int(params.get("atr_period", 14)),
        cooldown_bars=int(params.get("cooldown_bars", 3)),
    )

    # clip 信號範圍
    if supports_short:
        return pos.clip(-1.0, 1.0)
    else:
        return pos.clip(0.0, 1.0)


@register_strategy("multi_factor_trend_heavy")
def generate_positions_trend_heavy(
    df: pd.DataFrame,
    ctx: StrategyContext,
    params: dict,
) -> pd.Series:
    """
    多因子策略（趨勢偏重版）

    偏重趨勢和動量因子，減少均值回歸權重。
    適合趨勢明顯的市場（如牛市/熊市）。
    """
    trend_params = {
        **params,
        "w_trend": params.get("w_trend", 0.45),
        "w_momentum": params.get("w_momentum", 0.35),
        "w_mean_reversion": params.get("w_mean_reversion", 0.05),
        "w_volume": params.get("w_volume", 0.15),
        "trailing_stop_atr": params.get("trailing_stop_atr", 2.5),
        "take_profit_atr": params.get("take_profit_atr", None),
    }
    return generate_positions(df, ctx, trend_params)


@register_strategy("multi_factor_balanced")
def generate_positions_balanced(
    df: pd.DataFrame,
    ctx: StrategyContext,
    params: dict,
) -> pd.Series:
    """
    多因子策略（均衡版）

    四個因子等權，適合不確定市場環境的情況。
    """
    balanced_params = {
        **params,
        "w_trend": params.get("w_trend", 0.25),
        "w_momentum": params.get("w_momentum", 0.25),
        "w_mean_reversion": params.get("w_mean_reversion", 0.25),
        "w_volume": params.get("w_volume", 0.25),
        "entry_threshold": params.get("entry_threshold", 0.25),
    }
    return generate_positions(df, ctx, balanced_params)
