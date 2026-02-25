"""
Multi-TF Resonance — 多時間框架共振策略

學術背景：
    - Moskowitz, Ooi & Pedersen (2012): Time-Series Momentum
    - Lempérière et al. (2014): 多 lookback 趨勢信號可提高穩定性
    - Murphy (1999): 多時間框架分析（higher-TF 趨勢 + lower-TF 入場）

核心概念：
    1h 信號僅在更高時間框架（4h + 1d）方向一致時才觸發交易
    這是一種「信號過濾」策略，透過多 TF 確認降低假信號

信號邏輯：
    - Daily (1d): Regime 判斷（ADX 趨勢 vs 盤整 + EMA 方向）
    - 4h:         趨勢方向（EMA crossover）
    - 1h:         入場時機（TSMOM 信號）
    - 最終信號 = 1h TSMOM × 4h 方向確認 × 1d regime 過濾

Multi-TF 數據來源：
    - 如果 ctx.auxiliary_data 有 "4h" 和 "1d" 數據 → 使用已對齊的 aux
    - 如果沒有 → 內部 resample 1h 數據到 4h/1d（向後相容）

Anti-Bias:
    - 所有 resample 使用 closed='left', label='left' 避免 look-ahead
    - signal_delay 由 @register_strategy 框架自動處理

Note:
    signal_delay 和 direction clip 由 @register_strategy 框架自動處理，
    策略函數只需回傳 raw position [-1, 1]。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import StrategyContext
from . import register_strategy

import logging

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
#  Component 1: Daily Regime Detection
# ══════════════════════════════════════════════════════════════

def _daily_regime(
    df_daily: pd.DataFrame,
    adx_period: int = 14,
    adx_trending_threshold: float = 25.0,
    ema_period: int = 20,
) -> pd.DataFrame:
    """
    日線 Regime 偵測

    回傳 DataFrame with:
        - regime_trend: 1 (trending) / 0 (ranging)
        - regime_direction: 1 (bullish) / -1 (bearish) / 0 (neutral)
        - regime_score: [-1, 1] 趨勢強度

    Args:
        df_daily: Daily OHLCV (可能是對齊到 1h index 的，欄位名可能帶 _1d 後綴)
        adx_period: ADX 計算週期
        adx_trending_threshold: ADX 趨勢門檻
        ema_period: 方向 EMA 週期
    """
    # 處理可能帶後綴的欄位名
    close_col = "close_1d" if "close_1d" in df_daily.columns else "close"
    high_col = "high_1d" if "high_1d" in df_daily.columns else "high"
    low_col = "low_1d" if "low_1d" in df_daily.columns else "low"

    close = df_daily[close_col]
    high = df_daily[high_col]
    low = df_daily[low_col]

    # ADX 計算
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    plus_dm = (high - high.shift(1)).clip(lower=0)
    minus_dm = (low.shift(1) - low).clip(lower=0)

    # 只保留較大者
    both_positive = (plus_dm > 0) & (minus_dm > 0)
    plus_dm[both_positive & (plus_dm < minus_dm)] = 0
    minus_dm[both_positive & (minus_dm < plus_dm)] = 0

    atr = tr.ewm(span=adx_period, min_periods=adx_period).mean()
    plus_di = 100 * (plus_dm.ewm(span=adx_period, min_periods=adx_period).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=adx_period, min_periods=adx_period).mean() / atr)

    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10))
    adx = dx.ewm(span=adx_period, min_periods=adx_period).mean()

    # EMA 方向
    ema = close.ewm(span=ema_period, min_periods=ema_period).mean()
    ema_direction = pd.Series(0.0, index=close.index)
    ema_direction[close > ema] = 1.0
    ema_direction[close < ema] = -1.0

    # Regime 判斷
    regime_trend = (adx > adx_trending_threshold).astype(float)
    regime_direction = ema_direction
    regime_score = regime_trend * regime_direction

    result = pd.DataFrame({
        "regime_trend": regime_trend,
        "regime_direction": regime_direction,
        "regime_score": regime_score,
        "adx": adx,
    }, index=close.index)

    return result


# ══════════════════════════════════════════════════════════════
#  Component 2: 4h Trend Direction
# ══════════════════════════════════════════════════════════════

def _htf_trend(
    df_4h: pd.DataFrame,
    ema_fast: int = 20,
    ema_slow: int = 50,
) -> pd.Series:
    """
    4h 趨勢方向（EMA crossover）

    回傳: Series of +1 / -1 / 0

    Args:
        df_4h: 4h OHLCV (可能有 _4h 後綴)
        ema_fast: 快速 EMA 週期
        ema_slow: 慢速 EMA 週期
    """
    close_col = "close_4h" if "close_4h" in df_4h.columns else "close"
    close = df_4h[close_col]

    ema_f = close.ewm(span=ema_fast, min_periods=ema_fast).mean()
    ema_s = close.ewm(span=ema_slow, min_periods=ema_slow).mean()

    trend = pd.Series(0.0, index=close.index)
    trend[ema_f > ema_s] = 1.0
    trend[ema_f < ema_s] = -1.0

    return trend


# ══════════════════════════════════════════════════════════════
#  Component 3: 1h TSMOM Entry Signal
# ══════════════════════════════════════════════════════════════

def _tsmom_signal(
    close: pd.Series,
    lookback: int = 168,
    vol_target: float = 0.15,
) -> pd.Series:
    """
    1h TSMOM 信號（驗證過的核心信號）

    Args:
        close: 1h 收盤價序列
        lookback: TSMOM 回看期（預設 168h = 1 week）
        vol_target: 年化波動率目標

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

    return tsmom


# ══════════════════════════════════════════════════════════════
#  Internal Resample (fallback when no aux data)
# ══════════════════════════════════════════════════════════════

def _resample_ohlcv(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Resample 1h OHLCV 到更高 TF

    使用 closed='left', label='left' 確保因果性（避免 look-ahead）
    """
    resampled = pd.DataFrame()
    resampled["open"] = df["open"].resample(freq, closed="left", label="left").first()
    resampled["high"] = df["high"].resample(freq, closed="left", label="left").max()
    resampled["low"] = df["low"].resample(freq, closed="left", label="left").min()
    resampled["close"] = df["close"].resample(freq, closed="left", label="left").last()
    resampled["volume"] = df["volume"].resample(freq, closed="left", label="left").sum()
    resampled = resampled.dropna(subset=["open", "close"])
    return resampled


# ══════════════════════════════════════════════════════════════
#  Main Strategy: Multi-TF Resonance
# ══════════════════════════════════════════════════════════════

@register_strategy("multi_tf_resonance")
def multi_tf_resonance(
    df: pd.DataFrame,
    ctx: StrategyContext,
    params: dict,
) -> pd.Series:
    """
    多時間框架共振策略

    信號邏輯：
        final = 1h_tsmom × htf_confirmation

    htf_confirmation:
        - 4h trend + daily regime 都與 1h 方向一致 → 1.0x（全倉）
        - 只有 4h 一致 → 0.7x
        - 4h 和 daily 都不一致 → 0.0x（不交易）

    Parameters (params):
        # 1h TSMOM
        tsmom_lookback: int = 168  (TSMOM 回看期)
        vol_target: float = 0.15   (波動率目標)

        # 4h Trend
        htf_ema_fast: int = 20     (4h EMA fast)
        htf_ema_slow: int = 50     (4h EMA slow)

        # Daily Regime
        adx_period: int = 14       (ADX 週期)
        adx_trending_threshold: float = 25.0  (趨勢門檻)
        regime_ema_period: int = 20 (Regime EMA 週期)

        # Confirmation weights
        full_confirm_weight: float = 1.0   (4h+daily 都一致)
        partial_confirm_weight: float = 0.7 (只有 4h 一致)
        no_confirm_weight: float = 0.0     (都不一致 → 不交易)

        # Turnover control
        composite_ema: int = 6     (最終信號 EMA 平滑)
        position_step: float = 0.25 (持倉量化步長)
    """
    close = df["close"]

    # ── 1. 1h TSMOM 信號 ──
    tsmom_lookback = params.get("tsmom_lookback", 168)
    vol_target = params.get("vol_target", 0.15)
    tsmom = _tsmom_signal(close, tsmom_lookback, vol_target)

    # ── 2. 取得 4h 數據 ──
    htf_ema_fast = params.get("htf_ema_fast", 20)
    htf_ema_slow = params.get("htf_ema_slow", 50)

    aux_4h = ctx.get_auxiliary_df("4h") if ctx.has_auxiliary else None
    if aux_4h is not None:
        # 使用已對齊的 4h 數據
        htf_4h_trend = _htf_trend(aux_4h, htf_ema_fast, htf_ema_slow)
    else:
        # Fallback: 從 1h resample 到 4h
        df_4h = _resample_ohlcv(df, "4h")
        htf_4h_raw = _htf_trend(df_4h, htf_ema_fast, htf_ema_slow)
        htf_4h_trend = htf_4h_raw.reindex(df.index, method="ffill").fillna(0)

    # ── 3. 取得 Daily 數據 ──
    adx_period = params.get("adx_period", 14)
    adx_threshold = params.get("adx_trending_threshold", 25.0)
    regime_ema = params.get("regime_ema_period", 20)

    aux_1d = ctx.get_auxiliary_df("1d") if ctx.has_auxiliary else None
    if aux_1d is not None:
        regime = _daily_regime(aux_1d, adx_period, adx_threshold, regime_ema)
    else:
        df_1d = _resample_ohlcv(df, "1D")
        regime_raw = _daily_regime(df_1d, adx_period, adx_threshold, regime_ema)
        regime = pd.DataFrame({
            c: regime_raw[c].reindex(df.index, method="ffill").fillna(0)
            for c in regime_raw.columns
        })

    # ── 4. Multi-TF Confirmation ──
    full_w = params.get("full_confirm_weight", 1.0)
    partial_w = params.get("partial_confirm_weight", 0.7)
    no_w = params.get("no_confirm_weight", 0.0)

    tsmom_dir = np.sign(tsmom)
    htf_4h_dir = np.sign(htf_4h_trend)
    daily_dir = regime["regime_direction"]

    # 4h 方向一致
    htf_agree = (tsmom_dir == htf_4h_dir) & (tsmom_dir != 0) & (htf_4h_dir != 0)
    # Daily 方向一致
    daily_agree = (tsmom_dir == daily_dir) & (tsmom_dir != 0) & (daily_dir != 0)
    # Daily 在趨勢 regime
    daily_trending = regime["regime_trend"] > 0.5

    # Confirmation 權重
    confirmation = pd.Series(no_w, index=df.index)

    # 4h 一致但 daily 不一致 → partial
    confirmation[htf_agree & ~daily_agree] = partial_w
    # 4h 一致且 daily 也一致（在趨勢 regime 中）→ full
    confirmation[htf_agree & daily_agree & daily_trending] = full_w
    # 4h 一致且 daily 也一致（但盤整 regime）→ partial + bonus
    confirmation[htf_agree & daily_agree & ~daily_trending] = min(full_w, partial_w + 0.15)

    # ── 5. 合成最終信號 ──
    raw_signal = tsmom * confirmation

    # Turnover control: EMA 平滑
    composite_ema_period = params.get("composite_ema", 6)
    if composite_ema_period > 1:
        raw_signal = raw_signal.ewm(span=composite_ema_period, min_periods=1).mean()

    # Position step 量化
    position_step = params.get("position_step", 0.25)
    if position_step > 0:
        raw_signal = (raw_signal / position_step).round() * position_step

    # ── 6. 最終裁剪 ──
    pos = raw_signal.clip(-1.0, 1.0).fillna(0.0)

    # Log 診斷
    n_full = (confirmation == full_w).sum()
    n_partial = ((confirmation > no_w) & (confirmation < full_w)).sum()
    n_filtered = (confirmation == no_w).sum()
    n_total = len(df)
    logger.info(
        f"  Multi-TF Resonance [{ctx.symbol}]: "
        f"full={n_full}/{n_total} ({n_full/n_total*100:.1f}%), "
        f"partial={n_partial}/{n_total}, "
        f"filtered={n_filtered}/{n_total} ({n_filtered/n_total*100:.1f}%)"
    )

    return pos
