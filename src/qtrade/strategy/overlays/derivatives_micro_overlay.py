"""
Derivatives-Enhanced Microstructure Overlay (Phase 3C)

升級 microstructure_accel_overlay.py，用真實衍生品數據替換 OHLCV proxy：
    - Taker Buy/Sell Ratio → 替換 close_position proxy
    - CVD momentum → 新增動能信號
    - LSR → 逆向制動器

如果 ctx.derivatives_data 沒有對應數據，自動 fallback 到原有 proxy。

Usage:
    from qtrade.strategy.overlays.derivatives_micro_overlay import (
        apply_derivatives_enhanced_overlay,
    )
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .microstructure_accel_overlay import (
    compute_micro_features,
    compute_accel_score,
    apply_accel_overlay,
    _taker_imbalance_proxy,
    _vol_regime_zscore,
    _short_realized_vol,
)

logger = logging.getLogger(__name__)


def _real_taker_imbalance(
    taker_vol_ratio: pd.Series,
    kline_index: pd.DatetimeIndex,
    window: int = 12,
) -> pd.Series:
    """
    從真實 Taker Buy/Sell Ratio 計算 Imbalance

    taker_vol_ratio = taker_buy_vol / total_vol
    > 0.5 = 買方主導, < 0.5 = 賣方主導

    正規化為 [-1, 1]：
        imbalance = (taker_vol_ratio - 0.5) * 2

    Args:
        taker_vol_ratio: Taker Buy/Sell Vol Ratio (已對齊 kline index)
        kline_index: K 線時間索引
        window: 滾動平滑窗口

    Returns:
        Real taker imbalance [-1, 1]
    """
    ratio = taker_vol_ratio.reindex(kline_index).ffill().fillna(0.5)

    # 平滑
    if window > 1:
        ratio = ratio.rolling(window, min_periods=1).mean()

    imbalance = (ratio - 0.5) * 2.0
    return imbalance.clip(-1.0, 1.0).fillna(0.0)


def _cvd_momentum(
    cvd: pd.Series,
    kline_index: pd.DatetimeIndex,
    lookback: int = 24,
    z_window: int = 168,
) -> pd.Series:
    """
    CVD Momentum z-score

    正值 = CVD 上升（淨買入壓力增加）
    負值 = CVD 下降（淨賣出壓力增加）

    Args:
        cvd: Cumulative Volume Delta series
        kline_index: K 線時間索引
        lookback: 動量回看期
        z_window: z-score 標準化窗口

    Returns:
        CVD momentum z-score
    """
    cvd_aligned = cvd.reindex(kline_index).ffill().fillna(0)

    # CVD 變化量
    cvd_change = cvd_aligned.diff(lookback)

    # z-score
    roll_mean = cvd_change.rolling(z_window, min_periods=z_window // 4).mean()
    roll_std = cvd_change.rolling(z_window, min_periods=z_window // 4).std()
    roll_std = roll_std.replace(0, np.nan).ffill().fillna(1.0)

    z = (cvd_change - roll_mean) / roll_std
    return z.fillna(0.0).clip(-4.0, 4.0)


def _lsr_contrarian_brake(
    lsr: pd.Series,
    kline_index: pd.DatetimeIndex,
    z_window: int = 168,
    brake_threshold: float = 1.5,
) -> pd.Series:
    """
    LSR 逆向制動器

    當 LSR 極端時（市場過度一致），降低順勢加速的力度。
    這不是直接的信號，而是 accel_score 的折扣因子。

    回傳 [0, 1]：
        1.0 = 正常（無擁擠）
        0.0 = 極端擁擠（應減速）

    Args:
        lsr: Long/Short Ratio series
        kline_index: K 線時間索引
        z_window: z-score 窗口
        brake_threshold: z-score 超過此值開始制動

    Returns:
        Brake multiplier [0, 1]
    """
    lsr_aligned = lsr.reindex(kline_index).ffill().fillna(1.0)

    lsr_mean = lsr_aligned.rolling(z_window, min_periods=z_window // 4).mean()
    lsr_std = lsr_aligned.rolling(z_window, min_periods=z_window // 4).std()
    lsr_std = lsr_std.replace(0, np.nan).ffill().fillna(1.0)
    lsr_z = ((lsr_aligned - lsr_mean) / lsr_std).abs()

    # 超過 brake_threshold 開始制動
    brake = pd.Series(1.0, index=kline_index)
    extreme = lsr_z > brake_threshold
    if extreme.any():
        # 線性降低：z=threshold → 1.0, z=threshold+2 → 0.0
        brake[extreme] = (1.0 - (lsr_z[extreme] - brake_threshold) / 2.0).clip(0.0, 1.0)

    return brake


def compute_enhanced_micro_features(
    df_1h: pd.DataFrame,
    derivatives_data: dict | None = None,
    df_5m: pd.DataFrame | None = None,
    df_15m: pd.DataFrame | None = None,
    oi_series: pd.Series | None = None,
    params: dict | None = None,
) -> pd.DataFrame:
    """
    增強版微結構特徵：優先使用真實衍生品數據

    新增特徵（相較原版）：
        - taker_imbalance: 優先用真實 taker_vol_ratio，否則 fallback proxy
        - cvd_momentum_z: CVD 動量 z-score（新增）
        - lsr_brake: LSR 擁擠制動器（新增）

    Args:
        df_1h: 1h OHLCV DataFrame
        derivatives_data: {metric: Series} from ctx.derivatives_data
        df_5m: 5m OHLCV (optional, for fallback)
        df_15m: 15m OHLCV (optional, for fallback)
        oi_series: OI series (optional)
        params: feature parameters

    Returns:
        Enhanced features DataFrame
    """
    p = params or {}
    idx = df_1h.index
    deriv = derivatives_data or {}

    # 先計算基礎特徵（從原版 overlay）
    features = compute_micro_features(
        df_1h=df_1h,
        df_5m=df_5m,
        df_15m=df_15m,
        oi_series=oi_series,
        params=params,
    )

    # ── 升級 1: 用真實 Taker Vol 替換 proxy ──
    taker_vol = deriv.get("taker_vol_ratio")
    if taker_vol is not None:
        taker_window = int(p.get("taker_window", 12))
        real_imb = _real_taker_imbalance(taker_vol, idx, taker_window)
        features["taker_imbalance"] = real_imb
        logger.info("  📊 Using real Taker Vol Ratio (replaced OHLCV proxy)")

    # ── 升級 2: CVD Momentum（新特徵）──
    cvd = deriv.get("cvd")
    if cvd is not None:
        cvd_lookback = int(p.get("cvd_lookback", 24))
        cvd_z_window = int(p.get("cvd_z_window", 168))
        features["cvd_momentum_z"] = _cvd_momentum(cvd, idx, cvd_lookback, cvd_z_window)
        logger.info("  📊 CVD momentum feature added")

    # ── 升級 3: LSR 逆向制動器（新特徵）──
    lsr = deriv.get("lsr")
    if lsr is None:
        lsr = deriv.get("top_lsr_account")
    if lsr is not None:
        z_window = int(p.get("lsr_z_window", 168))
        brake_thresh = float(p.get("lsr_brake_threshold", 1.5))
        features["lsr_brake"] = _lsr_contrarian_brake(lsr, idx, z_window, brake_thresh)
        logger.info("  📊 LSR contrarian brake feature added")

    return features


def compute_enhanced_accel_score(
    features: pd.DataFrame,
    base_direction: pd.Series,
    params: dict | None = None,
) -> pd.Series:
    """
    增強版加速分數（整合 CVD + LSR brake）

    在原版 accel_score 基礎上：
        1. 加入 CVD momentum 作為額外確認
        2. 乘以 LSR brake 進行制動

    Args:
        features: compute_enhanced_micro_features() 的輸出
        base_direction: 1h 基礎策略的 position sign
        params: scoring weights

    Returns:
        Enhanced accel_score [-1, 1]
    """
    p = params or {}

    # 基礎 accel score（使用原版邏輯）
    base_score = compute_accel_score(features, base_direction, params)

    direction = np.sign(base_direction).fillna(0.0)

    # ── CVD momentum bonus ──
    cvd_z = features.get("cvd_momentum_z")
    if cvd_z is not None:
        w_cvd = float(p.get("w_cvd", 0.20))
        cvd_aligned = (cvd_z.clip(-2, 2) / 2.0) * direction
        base_score = base_score * (1.0 - w_cvd) + cvd_aligned * w_cvd

    # ── LSR brake（乘法，不是加法）──
    lsr_brake = features.get("lsr_brake")
    if lsr_brake is not None:
        base_score = base_score * lsr_brake

    return base_score.clip(-1.0, 1.0).fillna(0.0)


def apply_derivatives_enhanced_overlay(
    base_position: pd.Series,
    df_1h: pd.DataFrame,
    derivatives_data: dict | None = None,
    df_5m: pd.DataFrame | None = None,
    df_15m: pd.DataFrame | None = None,
    oi_series: pd.Series | None = None,
    params: dict | None = None,
) -> pd.Series:
    """
    完整的衍生品增強微結構 overlay 流程

    1. compute_enhanced_micro_features (uses real taker/cvd/lsr if available)
    2. compute_enhanced_accel_score (includes CVD momentum + LSR brake)
    3. apply_accel_overlay (same as original)

    如果沒有衍生品數據，行為等同原版 overlay（graceful fallback）。

    Args:
        base_position: 1h 策略的原始 position [-1, 1]
        df_1h: 1h OHLCV DataFrame
        derivatives_data: {metric: Series} from ctx.derivatives_data
        df_5m: 5m OHLCV DataFrame (optional)
        df_15m: 15m OHLCV DataFrame (optional)
        oi_series: OI series (optional)
        params: all overlay parameters

    Returns:
        adjusted position [-1, 1]
    """
    p = params or {}

    features = compute_enhanced_micro_features(
        df_1h=df_1h,
        derivatives_data=derivatives_data,
        df_5m=df_5m,
        df_15m=df_15m,
        oi_series=oi_series,
        params=p,
    )

    accel_score = compute_enhanced_accel_score(
        features=features,
        base_direction=base_position,
        params=p,
    )

    return apply_accel_overlay(
        base_position=base_position,
        accel_score=accel_score,
        params=p,
    )
