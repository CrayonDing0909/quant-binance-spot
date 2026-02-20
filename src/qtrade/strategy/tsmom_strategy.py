"""
TSMOM（Time-Series Momentum）策略

學術背景：
    Moskowitz, Ooi & Pedersen (2012) "Time Series Momentum"
    - 過去 N 期的報酬 > 0 → 做多，< 0 → 做空
    - 波動率目標縮放 → 自動控制曝險
    - 在商品期貨、外匯、股指期貨都有顯著 alpha

加密貨幣適用性：
    - 加密強趨勢市場 → 動量因子更強
    - 信號變化慢 → 1-bar delay 影響極小
    - 波動率縮放 → MDD 控制在 10-25%（vs RSI 策略 80-100%）

策略變體：
    1. tsmom         — 單 lookback TSMOM
    2. tsmom_multi   — 多 lookback 集成（最穩健）
    3. tsmom_ema     — TSMOM + EMA 趨勢對齊（最佳風險調整）
    4. tsmom_multi_ema — 多 lookback + EMA（終極版）

Note:
    signal_delay 和 direction clip 由 @register_strategy 框架自動處理，
    策略函數只需回傳 raw position [-1, 1]。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import StrategyContext
from . import register_strategy
from ..indicators import calculate_ema, calculate_atr
from .filters import volatility_regime_scaler


# ──────────────────────────────────────────────
#  核心 TSMOM 信號
# ──────────────────────────────────────────────

def _tsmom_signal(
    close: pd.Series,
    lookback: int = 168,
    vol_target: float = 0.15,
) -> pd.Series:
    """
    單一 lookback 的 TSMOM 信號

    邏輯：
        1. 計算過去 lookback 期的累積報酬
        2. 報酬 > 0 → 做多信號，< 0 → 做空信號
        3. 用波動率倒數縮放倉位（年化波動率目標 = vol_target）

    Args:
        close: 收盤價序列
        lookback: 回看期（小時數）
        vol_target: 年化波動率目標（0.15 = 15%）

    Returns:
        倉位序列 [-1, 1]
    """
    returns = close.pct_change()

    # 過去 N 期累積報酬
    cum_ret = returns.rolling(lookback).sum()

    # 年化波動率（1h bar, 8760 = 365*24）
    vol = returns.rolling(lookback).std() * np.sqrt(8760)
    vol = vol.replace(0, np.nan).ffill().fillna(vol_target)

    # 信號 = 累積報酬的符號
    raw_signal = np.sign(cum_ret)

    # 波動率目標縮放
    scale = (vol_target / vol).clip(0.1, 2.0)
    pos = raw_signal * scale

    return pos.clip(-1.0, 1.0).fillna(0.0)


# ──────────────────────────────────────────────
#  策略 1: 基礎 TSMOM
# ──────────────────────────────────────────────

@register_strategy("tsmom")
def generate_tsmom(df: pd.DataFrame, ctx: StrategyContext, params: dict) -> pd.Series:
    """
    基礎 TSMOM 策略（單 lookback）

    params:
        lookback:       回看期小時數（預設 168 = 7 天）
        vol_target:     年化波動率目標（預設 0.15 = 15%）
    """
    lookback = int(params.get("lookback", 168))
    vol_target = float(params.get("vol_target", 0.15))

    return _tsmom_signal(df["close"], lookback, vol_target)


# ──────────────────────────────────────────────
#  策略 2: 多 Lookback TSMOM 集成
# ──────────────────────────────────────────────

@register_strategy("tsmom_multi")
def generate_tsmom_multi(df: pd.DataFrame, ctx: StrategyContext, params: dict) -> pd.Series:
    """
    多 Lookback TSMOM 集成策略（最穩健）

    原理：不同 lookback 期捕捉不同速度的趨勢
        - 短期 (3d)：快速趨勢
        - 中期 (7d/14d)：標準動量
        - 長期 (30d)：大趨勢
    等權平均 → 平滑收益曲線，降低單一週期風險

    params:
        lookbacks:      lookback 列表，預設 [72, 168, 336, 720]（3d/7d/14d/30d）
        vol_target:     年化波動率目標（預設 0.15）
        vol_regime_enabled: 是否啟用波動率 regime 倉位縮放
    """
    lookbacks_raw = params.get("lookbacks", [72, 168, 336, 720])
    if isinstance(lookbacks_raw, str):
        lookbacks = [int(x) for x in lookbacks_raw.split(",")]
    else:
        lookbacks = [int(x) for x in lookbacks_raw]

    vol_target = float(params.get("vol_target", 0.15))

    # 多 lookback 信號等權平均
    signals = [_tsmom_signal(df["close"], lb, vol_target) for lb in lookbacks]
    pos = sum(signals) / len(signals)
    pos = pos.clip(-1.0, 1.0)

    # 波動率 regime 縮放（可選）
    if params.get("vol_regime_enabled", False):
        pos = volatility_regime_scaler(
            df, pos,
            atr_period=int(params.get("atr_period", 14)),
            lookback=int(params.get("vol_regime_lookback", 168)),
            low_vol_percentile=float(params.get("vol_regime_low_pct", 30.0)),
            low_vol_weight=float(params.get("vol_regime_low_weight", 0.5)),
        )

    return pos


# ──────────────────────────────────────────────
#  策略 3: TSMOM + EMA 趨勢對齊
# ──────────────────────────────────────────────

@register_strategy("tsmom_ema")
def generate_tsmom_ema(df: pd.DataFrame, ctx: StrategyContext, params: dict) -> pd.Series:
    """
    TSMOM + EMA 趨勢對齊策略（最佳風險調整）

    原理：
        1. TSMOM 生成基礎信號（動量方向 + 波動率縮放）
        2. EMA 交叉判斷中期趨勢方向
        3. TSMOM 方向與 EMA 趨勢一致 → 全倉
        4. 不一致 → 大幅降倉（預設 30%）

    params:
        lookback:           TSMOM 回看期（預設 168 = 7 天）
        vol_target:         年化波動率目標（預設 0.15）
        ema_fast:           快速 EMA 週期（預設 20）
        ema_slow:           慢速 EMA 週期（預設 50）
        agree_weight:       趨勢一致時的權重（預設 1.0）
        disagree_weight:    趨勢不一致時的權重（預設 0.3）
        vol_regime_enabled: 是否啟用波動率 regime 倉位縮放
    """
    lookback = int(params.get("lookback", 168))
    vol_target = float(params.get("vol_target", 0.15))
    ema_fast = int(params.get("ema_fast", 20))
    ema_slow = int(params.get("ema_slow", 50))
    agree_weight = float(params.get("agree_weight", 1.0))
    disagree_weight = float(params.get("disagree_weight", 0.3))

    # TSMOM 信號
    tsmom = _tsmom_signal(df["close"], lookback, vol_target)

    # EMA 趨勢
    ema_f = calculate_ema(df["close"], ema_fast)
    ema_s = calculate_ema(df["close"], ema_slow)
    ema_trend = pd.Series(0.0, index=df.index)
    ema_trend[ema_f > ema_s] = 1.0
    ema_trend[ema_f < ema_s] = -1.0

    # 對齊權重
    agree = np.sign(tsmom) == np.sign(ema_trend)
    pos = tsmom.copy()
    pos[agree] *= agree_weight
    pos[~agree] *= disagree_weight
    pos = pos.clip(-1.0, 1.0)

    # 波動率 regime 縮放（可選）
    if params.get("vol_regime_enabled", False):
        pos = volatility_regime_scaler(
            df, pos,
            atr_period=int(params.get("atr_period", 14)),
            lookback=int(params.get("vol_regime_lookback", 168)),
            low_vol_percentile=float(params.get("vol_regime_low_pct", 30.0)),
            low_vol_weight=float(params.get("vol_regime_low_weight", 0.5)),
        )

    # ── Regime Filter（ADX chop scaler, E2）──
    if params.get("regime_filter_enabled", False):
        from ..indicators.adx import calculate_adx
        r_adx_period = int(params.get("regime_adx_period", 14))
        r_adx_thresh = float(params.get("regime_adx_threshold", 20))
        r_chop_scale = float(params.get("regime_chop_scale", 0.3))
        adx_data = calculate_adx(df, r_adx_period)
        adx_vals = adx_data["ADX"].shift(1).fillna(0)  # lagged — no lookahead
        chop_mask = adx_vals < r_adx_thresh
        pos = pos.copy()
        pos[chop_mask] = pos[chop_mask] * r_chop_scale

    return pos


# ──────────────────────────────────────────────
#  策略 4: TSMOM Multi + EMA（終極版）
# ──────────────────────────────────────────────

@register_strategy("tsmom_multi_ema")
def generate_tsmom_multi_ema(df: pd.DataFrame, ctx: StrategyContext, params: dict) -> pd.Series:
    """
    多 Lookback TSMOM + EMA 趨勢對齊（終極版）

    結合 tsmom_multi 的穩健性和 tsmom_ema 的風控：
        1. 多 lookback TSMOM 信號等權集成
        2. EMA 趨勢對齊過濾
        3. 波動率 regime 縮放

    params:
        lookbacks:          lookback 列表
        vol_target:         波動率目標
        ema_fast/ema_slow:  EMA 週期
        agree_weight:       趨勢一致權重
        disagree_weight:    趨勢不一致權重
    """
    lookbacks_raw = params.get("lookbacks", [72, 168, 336, 720])
    if isinstance(lookbacks_raw, str):
        lookbacks = [int(x) for x in lookbacks_raw.split(",")]
    else:
        lookbacks = [int(x) for x in lookbacks_raw]

    vol_target = float(params.get("vol_target", 0.15))
    ema_fast = int(params.get("ema_fast", 20))
    ema_slow = int(params.get("ema_slow", 50))
    agree_weight = float(params.get("agree_weight", 1.0))
    disagree_weight = float(params.get("disagree_weight", 0.3))

    # 多 lookback TSMOM
    signals = [_tsmom_signal(df["close"], lb, vol_target) for lb in lookbacks]
    tsmom = sum(signals) / len(signals)

    # EMA 趨勢
    ema_f = calculate_ema(df["close"], ema_fast)
    ema_s = calculate_ema(df["close"], ema_slow)
    ema_trend = pd.Series(0.0, index=df.index)
    ema_trend[ema_f > ema_s] = 1.0
    ema_trend[ema_f < ema_s] = -1.0

    # 對齊
    agree = np.sign(tsmom) == np.sign(ema_trend)
    pos = tsmom.copy()
    pos[agree] *= agree_weight
    pos[~agree] *= disagree_weight
    pos = pos.clip(-1.0, 1.0)

    # 波動率 regime 縮放
    if params.get("vol_regime_enabled", False):
        pos = volatility_regime_scaler(
            df, pos,
            atr_period=int(params.get("atr_period", 14)),
            lookback=int(params.get("vol_regime_lookback", 168)),
            low_vol_percentile=float(params.get("vol_regime_low_pct", 30.0)),
            low_vol_weight=float(params.get("vol_regime_low_weight", 0.5)),
        )

    # ── Regime Filter（ADX chop scaler, E2）──
    if params.get("regime_filter_enabled", False):
        from ..indicators.adx import calculate_adx
        r_adx_period = int(params.get("regime_adx_period", 14))
        r_adx_thresh = float(params.get("regime_adx_threshold", 20))
        r_chop_scale = float(params.get("regime_chop_scale", 0.3))
        adx_data = calculate_adx(df, r_adx_period)
        adx_vals = adx_data["ADX"].shift(1).fillna(0)  # lagged — no lookahead
        chop_mask = adx_vals < r_adx_thresh
        pos = pos.copy()
        pos[chop_mask] = pos[chop_mask] * r_chop_scale

    return pos
