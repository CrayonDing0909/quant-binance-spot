"""
Microstructure Acceleration Overlay (R3 Track A)

不改 R2.1 的 1h 方向判斷，只在執行層新增 5m/15m「加速/減速」：
    - 趨勢同向且微結構確認強 → 加速進場 / 加倉
    - 趨勢同向但微結構轉弱 → 延遲進場 / 減倉
    - 趨勢反向且微結構極端不利 → 快速降風險

微結構特徵（Binance-only 可得）：
    1. Taker Buy/Sell Imbalance（5m/15m OHLCV proxy）
    2. 短窗 Realized Vol / Vol Regime
    3. 價格短窗動能斜率（EMA slope / return burst）
    4. OI change rate（次要特徵，需 OI 資料）

Anti-lookahead 保證：
    - 所有特徵用 [0, i] 的資料計算
    - 5m/15m 特徵 resample 到 1h 時用 last（bar i 結束時可得）
    - 結果 position[i] 的改動在 bar[i+1] 開盤執行
    - 與 trade_on=next_open + signal_delay=1 一致

Usage:
    from qtrade.strategy.overlays.microstructure_accel_overlay import (
        compute_micro_features,
        compute_accel_score,
        apply_accel_overlay,
    )
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
#  1. Taker Buy/Sell Imbalance Proxy
# ══════════════════════════════════════════════════════════════

def _taker_imbalance_proxy(df: pd.DataFrame, window: int = 12) -> pd.Series:
    """
    從 OHLCV 估算 Taker Buy/Sell Imbalance

    Proxy 邏輯：
        close_position = (close - low) / (high - low)
        → 接近 1 表示買方主導，接近 0 表示賣方主導

    用 rolling mean 平滑後再標準化為 [-1, 1]：
        imbalance = (rolling_mean(close_position) - 0.5) * 2

    Args:
        df: OHLCV DataFrame（5m 或 15m）
        window: 滾動窗口（bars）

    Returns:
        Taker imbalance proxy [-1, 1]
    """
    hl_range = df["high"] - df["low"]
    # 避免除以零
    hl_range = hl_range.replace(0, np.nan)
    close_pos = (df["close"] - df["low"]) / hl_range
    close_pos = close_pos.fillna(0.5)  # flat bar → neutral

    # Volume-weighted close position for better signal
    vol = df["volume"].replace(0, np.nan).fillna(1.0)
    vol_weighted = close_pos * vol
    vol_sum = vol.rolling(window, min_periods=max(window // 2, 1)).sum()
    weighted_mean = (
        vol_weighted.rolling(window, min_periods=max(window // 2, 1)).sum()
        / vol_sum
    )

    # Normalize to [-1, 1]
    imbalance = (weighted_mean - 0.5) * 2.0
    return imbalance.clip(-1.0, 1.0).fillna(0.0)


# ══════════════════════════════════════════════════════════════
#  2. Short-Window Realized Volatility
# ══════════════════════════════════════════════════════════════

def _short_realized_vol(df: pd.DataFrame, window: int = 12) -> pd.Series:
    """
    短窗 Realized Volatility（年化）

    Args:
        df: OHLCV DataFrame
        window: 滾動窗口（bars）

    Returns:
        年化波動率
    """
    returns = df["close"].pct_change()
    # Detect bar frequency for annualization
    if len(df) >= 2:
        freq_sec = (df.index[1] - df.index[0]).total_seconds()
        bars_per_year = 365.25 * 24 * 3600 / max(freq_sec, 1)
    else:
        bars_per_year = 8760  # default 1h

    rv = returns.rolling(window, min_periods=max(window // 2, 1)).std()
    rv_annualized = rv * np.sqrt(bars_per_year)
    return rv_annualized.fillna(0.0)


def _vol_regime_zscore(
    rv: pd.Series,
    long_window: int = 168,
) -> pd.Series:
    """
    Volatility regime z-score

    正值 = 波動高於歷史平均（趨勢可能加速 or 反轉）
    負值 = 波動低於平均（盤整可能結束）

    Args:
        rv: 已計算的 realized vol
        long_window: z-score 計算用長窗口

    Returns:
        vol z-score
    """
    rolling_mean = rv.rolling(long_window, min_periods=max(long_window // 4, 1)).mean()
    rolling_std = rv.rolling(long_window, min_periods=max(long_window // 4, 1)).std()
    rolling_std = rolling_std.replace(0, np.nan)
    z = (rv - rolling_mean) / rolling_std
    return z.fillna(0.0).clip(-4.0, 4.0)


# ══════════════════════════════════════════════════════════════
#  3. EMA Slope / Return Burst
# ══════════════════════════════════════════════════════════════

def _ema_slope(df: pd.DataFrame, period: int = 12, norm_window: int = 48) -> pd.Series:
    """
    EMA 斜率（標準化為 z-score）

    衡量價格短期動能方向和強度。
    正值 = 上漲動能，負值 = 下跌動能。

    Args:
        df: OHLCV DataFrame
        period: EMA 週期
        norm_window: 用於標準化斜率的窗口

    Returns:
        EMA slope z-score
    """
    ema = df["close"].ewm(span=period, adjust=False).mean()
    # Slope = bar-to-bar change of EMA, normalized by price
    slope = ema.diff() / df["close"].replace(0, np.nan)
    slope = slope.fillna(0.0)

    # Z-score normalization for comparability across assets
    roll_mean = slope.rolling(norm_window, min_periods=max(norm_window // 4, 1)).mean()
    roll_std = slope.rolling(norm_window, min_periods=max(norm_window // 4, 1)).std()
    roll_std = roll_std.replace(0, np.nan)
    z = (slope - roll_mean) / roll_std
    return z.fillna(0.0).clip(-4.0, 4.0)


def _return_burst(df: pd.DataFrame, window: int = 6) -> pd.Series:
    """
    Return Burst — 短窗累計報酬 z-score

    捕捉價格突然加速或減速的時刻。

    Args:
        df: OHLCV DataFrame
        window: 累計報酬窗口（bars）

    Returns:
        Return burst z-score
    """
    returns = df["close"].pct_change()
    cum_ret = returns.rolling(window, min_periods=max(window // 2, 1)).sum()

    # Z-score over longer window for context
    long_w = window * 8
    roll_mean = cum_ret.rolling(long_w, min_periods=max(long_w // 4, 1)).mean()
    roll_std = cum_ret.rolling(long_w, min_periods=max(long_w // 4, 1)).std()
    roll_std = roll_std.replace(0, np.nan)
    z = (cum_ret - roll_mean) / roll_std
    return z.fillna(0.0).clip(-4.0, 4.0)


# ══════════════════════════════════════════════════════════════
#  4. OI Change Rate
# ══════════════════════════════════════════════════════════════

def _oi_change_rate(
    oi_series: pd.Series | None,
    lookback: int = 24,
    z_window: int = 168,
) -> pd.Series | None:
    """
    Open Interest 變化率 z-score

    正值 = OI 快速增加（新錢進場）
    負值 = OI 快速減少（平倉離場）

    Args:
        oi_series: OI 序列（已對齊到 kline index）
        lookback: OI 變化率回看期
        z_window: z-score 窗口

    Returns:
        OI change rate z-score, or None if no OI data
    """
    if oi_series is None or oi_series.empty:
        return None

    oi_change = oi_series.pct_change(lookback, fill_method=None)
    roll_mean = oi_change.rolling(z_window, min_periods=max(z_window // 4, 1)).mean()
    roll_std = oi_change.rolling(z_window, min_periods=max(z_window // 4, 1)).std()
    roll_std = roll_std.replace(0, np.nan)
    z = (oi_change - roll_mean) / roll_std
    return z.fillna(0.0).clip(-4.0, 4.0)


# ══════════════════════════════════════════════════════════════
#  Resample Helper: Sub-hourly → 1h
# ══════════════════════════════════════════════════════════════

def _resample_to_1h(series: pd.Series, method: str = "last") -> pd.Series:
    """
    Resample sub-hourly series to 1h frequency.

    使用 label='left', closed='left' 對齊到 1h bar 的 open_time，
    取最後一個值（bar i 結束時可得，不含未來資訊）。

    Args:
        series: sub-hourly pd.Series with DatetimeIndex
        method: "last" | "mean" | "sum"

    Returns:
        1h frequency pd.Series
    """
    resampler = series.resample("1h", label="left", closed="left")
    if method == "last":
        return resampler.last()
    elif method == "mean":
        return resampler.mean()
    elif method == "sum":
        return resampler.sum()
    else:
        return resampler.last()


# ══════════════════════════════════════════════════════════════
#  compute_micro_features — 公開 API
# ══════════════════════════════════════════════════════════════

def compute_micro_features(
    df_1h: pd.DataFrame,
    df_5m: pd.DataFrame | None = None,
    df_15m: pd.DataFrame | None = None,
    oi_series: pd.Series | None = None,
    params: dict | None = None,
) -> pd.DataFrame:
    """
    計算所有微結構特徵，回傳對齊到 1h index 的 DataFrame

    特徵欄位：
        - taker_imbalance:  Taker Buy/Sell Imbalance proxy [-1, 1]
        - vol_regime_z:     Short-window vol regime z-score
        - ema_slope_z:      EMA slope z-score
        - return_burst_z:   Return burst z-score
        - oi_change_z:      OI change rate z-score (may be NaN)

    Anti-lookahead:
        - 所有 sub-hourly 特徵 resample 到 1h 時用 "last"
          （= 該小時最後一個 sub-bar 的值，bar i 結束時已知）
        - 1h 上的特徵直接計算（與基礎策略同頻率）

    Args:
        df_1h: 1h OHLCV DataFrame
        df_5m: 5m OHLCV DataFrame (optional)
        df_15m: 15m OHLCV DataFrame (optional)
        oi_series: OI series aligned to 1h index (optional)
        params: feature computation parameters

    Returns:
        DataFrame aligned to df_1h.index with micro features
    """
    p = params or {}
    idx = df_1h.index

    # ── Choose best sub-hourly frame for each feature ──
    # Priority: 5m > 15m > fallback to 1h
    micro_df = df_5m if df_5m is not None and len(df_5m) > 0 else (
        df_15m if df_15m is not None and len(df_15m) > 0 else None
    )

    features = pd.DataFrame(index=idx)

    # ── Feature 1: Taker Imbalance ──
    taker_window = int(p.get("taker_window", 12))
    if micro_df is not None:
        raw_imb = _taker_imbalance_proxy(micro_df, window=taker_window)
        features["taker_imbalance"] = _resample_to_1h(raw_imb, "last").reindex(idx).ffill().fillna(0.0)
    else:
        # Fallback: compute from 1h directly (weaker signal)
        features["taker_imbalance"] = _taker_imbalance_proxy(df_1h, window=max(taker_window // 3, 3))

    # ── Feature 2: Vol Regime Z-score ──
    vol_short_window = int(p.get("vol_short_window", 12))
    vol_long_window = int(p.get("vol_long_window", 168))
    if micro_df is not None:
        rv_micro = _short_realized_vol(micro_df, window=vol_short_window)
        vz_micro = _vol_regime_zscore(rv_micro, long_window=vol_long_window)
        features["vol_regime_z"] = _resample_to_1h(vz_micro, "last").reindex(idx).ffill().fillna(0.0)
    else:
        rv_1h = _short_realized_vol(df_1h, window=max(vol_short_window // 3, 3))
        features["vol_regime_z"] = _vol_regime_zscore(rv_1h, long_window=max(vol_long_window // 12, 14))

    # ── Feature 3: EMA Slope ──
    ema_slope_period = int(p.get("ema_slope_period", 12))
    ema_slope_norm = int(p.get("ema_slope_norm_window", 48))
    if micro_df is not None:
        es_micro = _ema_slope(micro_df, period=ema_slope_period, norm_window=ema_slope_norm)
        features["ema_slope_z"] = _resample_to_1h(es_micro, "last").reindex(idx).ffill().fillna(0.0)
    else:
        features["ema_slope_z"] = _ema_slope(df_1h, period=max(ema_slope_period // 3, 3),
                                              norm_window=max(ema_slope_norm // 12, 6))

    # ── Feature 4: Return Burst ──
    burst_window = int(p.get("return_burst_window", 6))
    if micro_df is not None:
        rb_micro = _return_burst(micro_df, window=burst_window)
        features["return_burst_z"] = _resample_to_1h(rb_micro, "last").reindex(idx).ffill().fillna(0.0)
    else:
        features["return_burst_z"] = _return_burst(df_1h, window=max(burst_window // 3, 2))

    # ── Feature 5: OI Change Rate ──
    oi_lookback = int(p.get("oi_lookback", 24))
    oi_z_window = int(p.get("oi_z_window", 168))
    oi_z = _oi_change_rate(oi_series, lookback=oi_lookback, z_window=oi_z_window)
    if oi_z is not None:
        features["oi_change_z"] = oi_z.reindex(idx).ffill().fillna(0.0)
    else:
        features["oi_change_z"] = 0.0

    logger.info(
        f"📊 Micro features computed: {list(features.columns)}, "
        f"source={'5m' if df_5m is not None else '15m' if df_15m is not None else '1h'}, "
        f"shape={features.shape}"
    )

    return features


# ══════════════════════════════════════════════════════════════
#  compute_accel_score — 公開 API
# ══════════════════════════════════════════════════════════════

def compute_accel_score(
    features: pd.DataFrame,
    base_direction: pd.Series,
    params: dict | None = None,
) -> pd.Series:
    """
    將微結構特徵合成為「加速/減速」分數

    Score 含義：
        > 0: 微結構支持當前趨勢方向 → 加速
        < 0: 微結構反對當前趨勢方向 → 減速
        ≈ 0: 中性

    計算方式：
        1. 用 base_direction sign 對齊特徵方向
           (做多時 taker_imbalance > 0 = 順勢 → positive contribution)
        2. 加權組合
        3. Clip to [-1, 1]

    Args:
        features: compute_micro_features() 的輸出
        base_direction: 1h 基礎策略的 position sign (+1/-1/0)
        params: scoring weights and thresholds

    Returns:
        accel_score [-1, 1]
    """
    p = params or {}

    # Weights for each feature
    w_taker = float(p.get("w_taker", 0.35))
    w_vol = float(p.get("w_vol", 0.15))
    w_slope = float(p.get("w_slope", 0.25))
    w_burst = float(p.get("w_burst", 0.15))
    w_oi = float(p.get("w_oi", 0.10))

    # Direction alignment: multiply features by sign(base_position)
    # so "confirming" features become positive
    direction = np.sign(base_direction).fillna(0.0)

    # Taker imbalance: positive when aligned with position direction
    taker = features.get("taker_imbalance", pd.Series(0.0, index=features.index))
    taker_aligned = taker * direction

    # Vol regime: complex — moderate vol expansion during trend = good,
    # extreme vol = caution. Use inverted-U: best at z ∈ [0.5, 1.5]
    vol_z = features.get("vol_regime_z", pd.Series(0.0, index=features.index))
    # Convert to signal: moderate expansion = +, extreme = -
    vol_signal = pd.Series(0.0, index=features.index)
    vol_signal[vol_z.between(0.3, 2.0)] = 1.0  # healthy expansion
    vol_signal[vol_z > 3.0] = -1.0  # extreme — caution
    vol_signal[vol_z < -1.0] = -0.5  # low vol — weak signal

    # EMA slope: positive when aligned with direction
    slope_z = features.get("ema_slope_z", pd.Series(0.0, index=features.index))
    slope_aligned = slope_z.clip(-2, 2) / 2.0 * direction  # normalize to [-1, 1]

    # Return burst: positive when aligned with direction
    burst_z = features.get("return_burst_z", pd.Series(0.0, index=features.index))
    burst_aligned = burst_z.clip(-2, 2) / 2.0 * direction

    # OI change: positive OI + aligned direction = conviction
    oi_z = features.get("oi_change_z", pd.Series(0.0, index=features.index))
    oi_signal = oi_z.clip(-2, 2) / 2.0  # direction-neutral for now

    # Weighted sum
    total_weight = w_taker + w_vol + w_slope + w_burst + w_oi
    if total_weight <= 0:
        total_weight = 1.0

    score = (
        w_taker * taker_aligned
        + w_vol * vol_signal
        + w_slope * slope_aligned
        + w_burst * burst_aligned
        + w_oi * oi_signal
    ) / total_weight

    score = score.clip(-1.0, 1.0).fillna(0.0)

    # When base position is flat (0), accel score should be 0
    score[direction == 0] = 0.0

    logger.info(
        f"📊 Accel score: mean={score.mean():.4f}, std={score.std():.4f}, "
        f"pos_pct={( score > 0.1 ).mean()*100:.1f}%, neg_pct={( score < -0.1 ).mean()*100:.1f}%"
    )

    return score


# ══════════════════════════════════════════════════════════════
#  apply_accel_overlay — 公開 API
# ══════════════════════════════════════════════════════════════

def apply_accel_overlay(
    base_position: pd.Series,
    accel_score: pd.Series,
    params: dict | None = None,
) -> pd.Series:
    """
    根據 accel_score 調整基礎倉位

    規則：
        1. accel_score > accel_threshold → boost position
           new_size = base * (1 + boost_pct * score)，capped at size_cap
        2. accel_score < -decel_threshold → reduce position
           new_size = base * (1 - reduce_pct * |score|)，floored at size_floor
        3. |accel_score| <= threshold → no change
        4. adverse_micro_exit: if score < -adverse_threshold and position
           has been held for >= min_hold_bars → rapid exit

    Cooldown:
        After any accel/decel action, wait cooldown_bars before next action.

    Anti-lookahead:
        - accel_score[i] 用 [0, i] 的資料計算
        - 結果 position[i] 在 bar[i+1] 執行

    IMPORTANT:
        This overlay can INCREASE position size (unlike vol_pause which only reduces).
        This is by design for Track A (追求更高年化報酬).

    Args:
        base_position: 1h 策略的原始 position [-1, 1]
        accel_score: compute_accel_score() 的輸出 [-1, 1]
        params: overlay 參數

    Returns:
        adjusted position [-1, 1]
    """
    p = params or {}

    # Thresholds
    accel_threshold = float(p.get("accel_threshold", 0.2))
    decel_threshold = float(p.get("decel_threshold", 0.2))
    adverse_threshold = float(p.get("adverse_threshold", 0.6))

    # Size multipliers
    boost_pct = float(p.get("boost_pct", 0.3))      # max 30% increase
    reduce_pct = float(p.get("reduce_pct", 0.3))     # max 30% decrease
    size_floor = float(p.get("size_floor", 0.1))      # never go below 10%
    size_cap = float(p.get("size_cap", 1.0))           # never exceed 100%

    # Cooldown
    cooldown_bars = int(p.get("cooldown_bars", 3))

    # Adverse micro exit
    adverse_exit_enabled = bool(p.get("adverse_exit_enabled", True))
    adverse_exit_to = float(p.get("adverse_exit_to", 0.0))  # exit to flat

    n = len(base_position)
    base_arr = base_position.values.copy().astype(float)
    score_arr = accel_score.values.copy().astype(float)
    result = base_arr.copy()

    # Stats
    n_boost = 0
    n_reduce = 0
    n_adverse_exit = 0
    cooldown_remaining = 0

    for i in range(n):
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            continue

        if abs(base_arr[i]) < 0.001:
            # Position is flat → no accel/decel
            result[i] = base_arr[i]
            continue

        s = score_arr[i]
        abs_base = abs(base_arr[i])
        sign_base = np.sign(base_arr[i])

        # ── Adverse exit (highest priority) ──
        if adverse_exit_enabled and s < -adverse_threshold:
            result[i] = sign_base * adverse_exit_to * abs_base
            n_adverse_exit += 1
            cooldown_remaining = cooldown_bars
            continue

        # ── Boost (trend-confirming micro) ──
        if s > accel_threshold:
            strength = (s - accel_threshold) / (1.0 - accel_threshold + 1e-9)
            new_abs = abs_base * (1.0 + boost_pct * strength)
            new_abs = min(new_abs, size_cap)
            result[i] = sign_base * new_abs
            n_boost += 1
            cooldown_remaining = cooldown_bars
            continue

        # ── Reduce (trend-weakening micro) ──
        if s < -decel_threshold:
            strength = (-s - decel_threshold) / (1.0 - decel_threshold + 1e-9)
            new_abs = abs_base * (1.0 - reduce_pct * strength)
            new_abs = max(new_abs, size_floor * abs_base) if size_floor > 0 else max(new_abs, 0.0)
            result[i] = sign_base * new_abs
            n_reduce += 1
            cooldown_remaining = cooldown_bars
            continue

        # ── Neutral: keep original ──
        result[i] = base_arr[i]

    # Final clip
    result = np.clip(result, -1.0, 1.0)

    # Stats
    pct_boost = n_boost / max(n, 1) * 100
    pct_reduce = n_reduce / max(n, 1) * 100
    pct_adverse = n_adverse_exit / max(n, 1) * 100
    logger.info(
        f"📊 Accel Overlay: boost={n_boost} ({pct_boost:.1f}%), "
        f"reduce={n_reduce} ({pct_reduce:.1f}%), "
        f"adverse_exit={n_adverse_exit} ({pct_adverse:.1f}%), "
        f"cooldown={cooldown_bars}b"
    )

    return pd.Series(result, index=base_position.index)


# ══════════════════════════════════════════════════════════════
#  Convenience: Full Pipeline
# ══════════════════════════════════════════════════════════════

def apply_full_micro_accel_overlay(
    base_position: pd.Series,
    df_1h: pd.DataFrame,
    df_5m: pd.DataFrame | None = None,
    df_15m: pd.DataFrame | None = None,
    oi_series: pd.Series | None = None,
    params: dict | None = None,
) -> pd.Series:
    """
    完整微結構加速 overlay 流程

    1. compute_micro_features
    2. compute_accel_score
    3. apply_accel_overlay

    Args:
        base_position: 1h 策略的原始 position [-1, 1]
        df_1h: 1h OHLCV DataFrame
        df_5m: 5m OHLCV DataFrame (optional)
        df_15m: 15m OHLCV DataFrame (optional)
        oi_series: OI series (optional)
        params: all overlay parameters (feature + scoring + overlay)

    Returns:
        adjusted position [-1, 1]
    """
    p = params or {}

    # Feature params
    feature_params = {
        k: p[k] for k in [
            "taker_window", "vol_short_window", "vol_long_window",
            "ema_slope_period", "ema_slope_norm_window",
            "return_burst_window", "oi_lookback", "oi_z_window",
        ] if k in p
    }

    # Scoring params
    scoring_params = {
        k: p[k] for k in [
            "w_taker", "w_vol", "w_slope", "w_burst", "w_oi",
        ] if k in p
    }

    # Overlay params
    overlay_params = {
        k: p[k] for k in [
            "accel_threshold", "decel_threshold", "adverse_threshold",
            "boost_pct", "reduce_pct", "size_floor", "size_cap",
            "cooldown_bars", "adverse_exit_enabled", "adverse_exit_to",
        ] if k in p
    }

    features = compute_micro_features(
        df_1h=df_1h,
        df_5m=df_5m,
        df_15m=df_15m,
        oi_series=oi_series,
        params=feature_params,
    )

    accel_score = compute_accel_score(
        features=features,
        base_direction=base_position,
        params=scoring_params,
    )

    return apply_accel_overlay(
        base_position=base_position,
        accel_score=accel_score,
        params=overlay_params,
    )


# ══════════════════════════════════════════════════════════════
#  Data Helper: Load multi-timeframe klines
# ══════════════════════════════════════════════════════════════

def load_multi_tf_klines(
    data_dir: Path,
    symbol: str,
    market_type: str = "futures",
) -> dict[str, pd.DataFrame | None]:
    """
    載入多時間框架 K 線資料

    Returns:
        {
            "1h": df_1h,
            "5m": df_5m or None,
            "15m": df_15m or None,
        }
    """
    from ...data.storage import load_klines as _load

    result = {}
    for tf in ["1h", "5m", "15m"]:
        path = data_dir / "binance" / market_type / tf / f"{symbol}.parquet"
        if path.exists():
            try:
                result[tf] = _load(path)
            except Exception as e:
                logger.warning(f"⚠️  Failed to load {tf} data for {symbol}: {e}")
                result[tf] = None
        else:
            result[tf] = None

    return result
