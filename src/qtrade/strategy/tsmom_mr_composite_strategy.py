"""
TSMOM + Mean-Reversion Composite 策略

學術背景：
    - Moskowitz, Ooi & Pedersen (2012): Time-Series Momentum
    - Brock, Lakonishok & LeBaron (1992): Mean-Reversion via Bollinger Bands
    - Jegadeesh (1990): Short-Term Return Reversals

動機（Alpha Researcher 2026-02-24 IC 分析）：
    1. 純 TSMOM 在 2023-2025 加密市場 IC ≈ 0（regime 轉為 mean-reverting）
    2. Bollinger MR 信號在 9/10 symbols 有正 IC（avg +0.042）
    3. 24h Reversal 信號在 9/10 symbols 有正 IC（avg +0.033）
    4. TSMOM 與 MR/Reversal 負相關（-0.16 ~ -0.20）→ 天然對沖
    5. Naive composite IC: portfolio avg +0.038（vs TSMOM alone +0.001）

策略邏輯：
    1. TSMOM_EMA: 動量 + EMA 趨勢對齊（捕捉趨勢 regime）
    2. Bollinger MR: %B 超買超賣信號（捕捉均值回歸 regime）
    3. Short-Term Reversal: 24h 報酬反轉（捕捉短期過度反應）
    4. 加權組合：default 50% TSMOM + 25% MR + 25% Reversal
    5. 支援 per-symbol 權重 override

策略變體：
    1. tsmom_mr_composite     — TSMOM + MR + Reversal 三因子組合

Note:
    signal_delay 和 direction clip 由 @register_strategy 框架自動處理，
    策略函數只需回傳 raw position [-1, 1]。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import StrategyContext
from . import register_strategy
from ..indicators import calculate_ema, calculate_bollinger_bands


# ══════════════════════════════════════════════════════════════
#  子因子 1: TSMOM + EMA（沿用現有實作）
# ══════════════════════════════════════════════════════════════

def _tsmom_ema_signal(
    close: pd.Series,
    lookback: int = 168,
    vol_target: float = 0.15,
    ema_fast: int = 20,
    ema_slow: int = 50,
    agree_weight: float = 1.0,
    disagree_weight: float = 0.3,
) -> pd.Series:
    """
    TSMOM + EMA 趨勢對齊信號（與 tsmom_ema 策略相同邏輯）

    Args:
        close: 收盤價序列
        lookback: TSMOM 回看期
        vol_target: 年化波動率目標
        ema_fast: 快速 EMA 週期
        ema_slow: 慢速 EMA 週期
        agree_weight: 方向一致時權重
        disagree_weight: 方向不一致時權重

    Returns:
        持倉信號 [-1, 1]
    """
    returns = close.pct_change()

    # TSMOM 信號
    cum_ret = returns.rolling(lookback).sum()
    vol = returns.rolling(lookback).std() * np.sqrt(8760)
    vol = vol.replace(0, np.nan).ffill().fillna(vol_target)
    raw_signal = np.sign(cum_ret)
    scale = (vol_target / vol).clip(0.1, 2.0)
    tsmom = (raw_signal * scale).clip(-1.0, 1.0).fillna(0.0)

    # EMA 趨勢
    ema_f = calculate_ema(close, ema_fast)
    ema_s = calculate_ema(close, ema_slow)
    ema_trend = pd.Series(0.0, index=close.index)
    ema_trend[ema_f > ema_s] = 1.0
    ema_trend[ema_f < ema_s] = -1.0

    # 對齊權重
    agree = np.sign(tsmom) == np.sign(ema_trend)
    pos = tsmom.copy()
    pos[agree] *= agree_weight
    pos[~agree] *= disagree_weight

    return pos.clip(-1.0, 1.0)


# ══════════════════════════════════════════════════════════════
#  子因子 2: Bollinger Band Mean-Reversion
# ══════════════════════════════════════════════════════════════

def _bollinger_mr_signal(
    close: pd.Series,
    bb_period: int = 20,
    bb_std: float = 2.0,
    overbought_threshold: float = 0.8,
    oversold_threshold: float = 0.2,
    exit_upper: float = 0.65,
    exit_lower: float = 0.35,
) -> pd.Series:
    """
    Bollinger Band %B 均值回歸信號

    邏輯：
        - %B > overbought_threshold → 做空（超買回歸）
        - %B < oversold_threshold   → 做多（超賣回歸）
        - 回到中間區域 → 平倉

    連續信號模式（非 binary）：
        - 超買區：signal = -((%B - exit_upper) / (1 - exit_upper))
        - 超賣區：signal = +((exit_lower - %B) / exit_lower)
        - 中間區：signal = 0

    Args:
        close: 收盤價序列
        bb_period: Bollinger Band 計算週期
        bb_std: 標準差倍數
        overbought_threshold: 超買門檻（%B）
        oversold_threshold: 超賣門檻（%B）
        exit_upper: 多倉出場閾值
        exit_lower: 空倉出場閾值

    Returns:
        持倉信號 [-1, 1]
    """
    bb = calculate_bollinger_bands(close, period=bb_period, std_mult=bb_std)
    pct_b = bb["%b"]

    pos = pd.Series(0.0, index=close.index)

    # 超買 → 做空（連續強度）
    overbought = pct_b > overbought_threshold
    pos[overbought] = -((pct_b[overbought] - overbought_threshold) /
                         (1.0 - overbought_threshold)).clip(0.0, 1.0)

    # 超賣 → 做多（連續強度）
    oversold = pct_b < oversold_threshold
    pos[oversold] = ((oversold_threshold - pct_b[oversold]) /
                      oversold_threshold).clip(0.0, 1.0)

    return pos.clip(-1.0, 1.0).fillna(0.0)


# ══════════════════════════════════════════════════════════════
#  子因子 3: Short-Term Reversal (24h)
# ══════════════════════════════════════════════════════════════

def _reversal_signal(
    close: pd.Series,
    reversal_lookback: int = 24,
    vol_target: float = 0.15,
    vol_lookback: int = 168,
) -> pd.Series:
    """
    短期反轉信號

    邏輯：
        - 過去 N 小時回報為正 → 做空（預期反轉）
        - 過去 N 小時回報為負 → 做多（預期反轉）
        - 用波動率倒數縮放，控制曝險

    Args:
        close: 收盤價序列
        reversal_lookback: 反轉回看期（小時）
        vol_target: 年化波動率目標
        vol_lookback: 波動率計算回看期

    Returns:
        持倉信號 [-1, 1]
    """
    returns = close.pct_change()
    cum_ret = returns.rolling(reversal_lookback).sum()

    # 反轉信號 = 累積報酬的反向
    raw_signal = -np.sign(cum_ret)

    # 波動率目標縮放
    vol = returns.rolling(vol_lookback).std() * np.sqrt(8760)
    vol = vol.replace(0, np.nan).ffill().fillna(vol_target)
    scale = (vol_target / vol).clip(0.1, 2.0)

    pos = (raw_signal * scale).clip(-1.0, 1.0).fillna(0.0)
    return pos


# ══════════════════════════════════════════════════════════════
#  策略: TSMOM + MR Composite（三因子組合）
# ══════════════════════════════════════════════════════════════

@register_strategy("tsmom_mr_composite")
def generate_tsmom_mr_composite(
    df: pd.DataFrame, ctx: StrategyContext, params: dict
) -> pd.Series:
    """
    TSMOM + Mean-Reversion + Reversal 三因子組合策略（Regime-Aware）

    關鍵設計原則：
        1. TSMOM 是 anchor 信號，MR/Reversal 是 satellite
        2. 趨勢強時：MR/Reversal 只能 AGREE with TSMOM，不能 oppose
           → 保護 TSMOM 在強趨勢中的收益
        3. TSMOM 微弱時（flat/uncertain）：MR/Reversal 可自由發揮
           → 在盤整中補充 alpha
        4. MR/Reversal 信號先做 EMA 平滑 → 降低換手率

    因子組合原理：
        - TSMOM 與 MR 負相關（-0.16）→ 趨勢和均值回歸互補
        - TSMOM 與 Reversal 負相關（-0.20）→ 長短期動量對沖
        - Regime-aware gating 防止 MR/Rev 在趨勢行情中拖累

    params:
        # ── Factor Weights ──
        w_tsmom:                TSMOM 權重（預設 0.70）
        w_mr:                   Mean-Reversion 權重（預設 0.15）
        w_reversal:             Reversal 權重（預設 0.15）

        # ── Regime Gate ──
        regime_gate_enabled:    是否啟用 regime gate（預設 True）
        trend_strength_thresh:  TSMOM |signal| > thresh 視為強趨勢（預設 0.4）
                                強趨勢中，MR/Rev 只允許同向或靜默

        # ── Signal Smoothing ──
        mr_smooth_period:       MR 信號 EMA 平滑週期（預設 12，0=不平滑）
        rev_smooth_period:      Reversal 信號 EMA 平滑週期（預設 12）

        # ── TSMOM Sub-factor ──
        tsmom_lookback:         TSMOM 回看期（預設 168 = 7 天）
        tsmom_vol_target:       TSMOM 年化波動率目標（預設 0.15）
        tsmom_ema_fast:         EMA 快線（預設 20）
        tsmom_ema_slow:         EMA 慢線（預設 50）
        tsmom_agree_weight:     EMA 一致時權重（預設 1.0）
        tsmom_disagree_weight:  EMA 不一致時權重（預設 0.3）

        # ── Bollinger MR Sub-factor ──
        mr_bb_period:           BB 週期（預設 20）
        mr_bb_std:              BB 標準差倍數（預設 2.0）
        mr_overbought:          超買門檻（預設 0.8）
        mr_oversold:            超賣門檻（預設 0.2）
        mr_exit_upper:          多倉出場閾值（預設 0.65）
        mr_exit_lower:          空倉出場閾值（預設 0.35）

        # ── Reversal Sub-factor ──
        rev_lookback:           反轉回看期（預設 24）
        rev_vol_target:         反轉波動率目標（預設 0.15）
        rev_vol_lookback:       反轉 vol 回看期（預設 168）

        # ── Per-symbol weight override ──
        symbol_factor_weights:  per-symbol 因子權重 override dict
    """
    close = df["close"]

    # ── 解析因子權重 ──
    symbol_overrides = params.get("symbol_factor_weights", {})
    sym_weights = symbol_overrides.get(ctx.symbol, {})

    w_tsmom = float(sym_weights.get("w_tsmom", params.get("w_tsmom", 0.70)))
    w_mr = float(sym_weights.get("w_mr", params.get("w_mr", 0.15)))
    w_reversal = float(sym_weights.get("w_reversal", params.get("w_reversal", 0.15)))

    # 正規化權重（確保和為 1）
    w_total = w_tsmom + w_mr + w_reversal
    if w_total > 0:
        w_tsmom /= w_total
        w_mr /= w_total
        w_reversal /= w_total

    # ── Regime Gate 參數 ──
    regime_gate_enabled = params.get("regime_gate_enabled", True)
    trend_strength_thresh = float(params.get("trend_strength_thresh", 0.4))

    # ── Signal Smoothing 參數 ──
    mr_smooth = int(params.get("mr_smooth_period", 12))
    rev_smooth = int(params.get("rev_smooth_period", 12))

    # ── 子因子 1: TSMOM EMA ──
    tsmom_lookback = int(params.get("tsmom_lookback", 168))
    tsmom_vol_target = float(params.get("tsmom_vol_target", 0.15))
    tsmom_ema_fast = int(params.get("tsmom_ema_fast", 20))
    tsmom_ema_slow = int(params.get("tsmom_ema_slow", 50))
    tsmom_agree_w = float(params.get("tsmom_agree_weight", 1.0))
    tsmom_disagree_w = float(params.get("tsmom_disagree_weight", 0.3))

    sig_tsmom = _tsmom_ema_signal(
        close,
        lookback=tsmom_lookback,
        vol_target=tsmom_vol_target,
        ema_fast=tsmom_ema_fast,
        ema_slow=tsmom_ema_slow,
        agree_weight=tsmom_agree_w,
        disagree_weight=tsmom_disagree_w,
    )

    # ── 子因子 2: Bollinger MR ──
    mr_bb_period = int(params.get("mr_bb_period", 20))
    mr_bb_std = float(params.get("mr_bb_std", 2.0))
    mr_overbought = float(params.get("mr_overbought", 0.8))
    mr_oversold = float(params.get("mr_oversold", 0.2))
    mr_exit_upper = float(params.get("mr_exit_upper", 0.65))
    mr_exit_lower = float(params.get("mr_exit_lower", 0.35))

    sig_mr = _bollinger_mr_signal(
        close,
        bb_period=mr_bb_period,
        bb_std=mr_bb_std,
        overbought_threshold=mr_overbought,
        oversold_threshold=mr_oversold,
        exit_upper=mr_exit_upper,
        exit_lower=mr_exit_lower,
    )

    # ── 子因子 3: Short-Term Reversal ──
    rev_lookback = int(params.get("rev_lookback", 24))
    rev_vol_target = float(params.get("rev_vol_target", 0.15))
    rev_vol_lookback = int(params.get("rev_vol_lookback", 168))

    sig_rev = _reversal_signal(
        close,
        reversal_lookback=rev_lookback,
        vol_target=rev_vol_target,
        vol_lookback=rev_vol_lookback,
    )

    # ── Signal Smoothing（降低 MR/Reversal 換手率）──
    if mr_smooth > 0:
        sig_mr = sig_mr.ewm(span=mr_smooth, adjust=False).mean()
    if rev_smooth > 0:
        sig_rev = sig_rev.ewm(span=rev_smooth, adjust=False).mean()

    # ── Regime-Aware Gating ──
    # 當 TSMOM 有強方向性時，MR/Rev 信號只能同向或為零
    # 當 TSMOM 微弱時，MR/Rev 自由發揮
    if regime_gate_enabled:
        tsmom_dir = np.sign(sig_tsmom)
        tsmom_strong = sig_tsmom.abs() > trend_strength_thresh

        # MR: 在強趨勢中，只保留同向部分，反向歸零
        mr_gated = sig_mr.copy()
        oppose_mr = tsmom_strong & (np.sign(sig_mr) != tsmom_dir) & (sig_mr.abs() > 0.01)
        mr_gated[oppose_mr] = 0.0

        # Reversal: 同樣的 gating
        rev_gated = sig_rev.copy()
        oppose_rev = tsmom_strong & (np.sign(sig_rev) != tsmom_dir) & (sig_rev.abs() > 0.01)
        rev_gated[oppose_rev] = 0.0
    else:
        mr_gated = sig_mr
        rev_gated = sig_rev

    # ── 加權組合 ──
    composite = w_tsmom * sig_tsmom + w_mr * mr_gated + w_reversal * rev_gated
    pos = composite.clip(-1.0, 1.0).fillna(0.0)

    return pos
