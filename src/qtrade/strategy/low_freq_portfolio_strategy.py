"""
Low-Frequency Portfolio Layer — 日/週級別的組合管理策略

在 1h 策略之上疊加低頻組合層，用於：
  1. 風險 Regime 切換（risk-on → full exposure / risk-off → reduce）
  2. 月度動量因子的幣種輪動
  3. 鏈上數據/穩定幣流動性的宏觀 regime 偵測

使用方式：
    - 作為 overlay 疊加在 meta_blend 之上
    - 或作為獨立策略搭配 daily interval

策略信號範圍：
    - 輸出 exposure scalar [0.0, 1.0]（僅縮放，不改變方向）
    - 1.0 = full exposure, 0.0 = all cash (risk-off)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import register_strategy
from .base import StrategyContext


# ── 輔助函數 ────────────────────────────────────────────────


def compute_risk_regime(
    df: pd.DataFrame,
    adx_period: int = 14,
    vol_lookback: int = 20,
    vol_percentile_window: int = 252,
) -> pd.Series:
    """
    風險 Regime 偵測器

    - risk_on (1.0): 適度波動 + 趨勢存在 → 維持曝險
    - risk_off (0.0): 極端波動 + 趨勢崩潰 → 縮減曝險
    - neutral (0.5): 中間狀態

    基於：
      1. 已實現波動率 vs 長期百分位
      2. ADX 趨勢強度
      3. 跨資產動量（BTC 月回報作代理）
    """
    close = df["close"]

    # ── 1. 已實現波動率百分位 ──
    log_returns = np.log(close / close.shift(1))
    realized_vol = log_returns.rolling(vol_lookback).std() * np.sqrt(365 * 24)  # 年化

    vol_pctile = realized_vol.rolling(vol_percentile_window, min_periods=60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )

    # ── 2. ADX 趨勢強度 ──
    high, low = df["high"], df["low"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(adx_period).mean()

    plus_dm = (high - high.shift(1)).clip(lower=0)
    minus_dm = (low.shift(1) - low).clip(lower=0)
    mask = plus_dm < minus_dm
    plus_dm[mask] = 0
    mask2 = minus_dm < plus_dm
    minus_dm[mask2] = 0

    plus_di = 100 * (plus_dm.rolling(adx_period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(adx_period).mean() / atr)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10))
    adx = dx.rolling(adx_period).mean()

    # ── 3. Regime 組合邏輯 ──
    # risk_off: 波動率 > 90th percentile AND ADX < 20（高波動 + 無趨勢）
    # risk_on: 波動率 < 70th percentile OR ADX > 25（低波動 or 有趨勢）
    regime = pd.Series(0.5, index=close.index)  # 預設 neutral
    regime[vol_pctile > 0.90] = 0.2  # 極端波動 → 大幅縮減
    regime[(vol_pctile > 0.90) & (adx < 20)] = 0.0  # 極端波動 + 無趨勢 → risk off
    regime[(vol_pctile < 0.70) | (adx > 25)] = 1.0  # 正常/趨勢 → full exposure

    return regime


def compute_momentum_score(
    df: pd.DataFrame,
    lookback: int = 30 * 24,  # 30 天（1h bars）
    short_lookback: int = 7 * 24,  # 7 天
) -> pd.Series:
    """
    動量評分（月回報 vs 週回報）

    正分 = 上升趨勢，負分 = 下降趨勢
    用於幣種輪動：高動量 → 加碼，低動量 → 減碼
    """
    close = df["close"]
    monthly_ret = close.pct_change(lookback)
    weekly_ret = close.pct_change(short_lookback)

    # 月度 + 週度的加權平均
    score = 0.7 * monthly_ret + 0.3 * weekly_ret
    return score.fillna(0.0)


def compute_onchain_regime(
    ctx: StrategyContext,
) -> float:
    """
    鏈上數據 regime（佔位實作）

    若 ctx.derivatives_data 包含 stablecoin_mcap 或 tvl，
    則使用其變化率判斷 risk-on/off。

    Returns:
        float: 1.0 (risk-on), 0.5 (neutral), 0.0 (risk-off)
    """
    if not ctx.has_derivatives:
        return 1.0  # 無數據時預設 full exposure

    # 預留：未來接入 DeFi Llama TVL、穩定幣市值數據
    # 目前回傳 neutral
    return 1.0


# ── 策略函數 ─────────────────────────────────────────────────


@register_strategy("low_freq_portfolio")
def generate_positions(
    df: pd.DataFrame,
    ctx: StrategyContext,
    params: dict,
) -> pd.Series:
    """
    Low-Frequency Portfolio Layer

    組合層策略：不自己產生方向性信號，
    而是產生 exposure scalar [0, 1]，由上層 meta_blend 疊加。

    可作為獨立策略使用（此時 1.0 = 全倉做多，0.0 = 空倉）
    或作為 risk_regime overlay 使用。

    Params:
        adx_period (int): ADX 計算週期，預設 14
        vol_lookback (int): 波動率回溯窗口，預設 20
        vol_percentile_window (int): 波動率百分位窗口，預設 252
        momentum_lookback (int): 月度動量回溯（bars），預設 720 (30d @ 1h)
        momentum_short (int): 週動量回溯（bars），預設 168 (7d @ 1h)
        use_momentum_rotation (bool): 是否啟用動量輪動，預設 True
        momentum_threshold (float): 動量 > 此值才開倉，預設 0.0
        use_risk_regime (bool): 是否啟用 risk regime，預設 True
        use_onchain (bool): 是否使用鏈上數據 regime，預設 False
    """
    adx_period = params.get("adx_period", 14)
    vol_lookback = params.get("vol_lookback", 20)
    vol_pctile_window = params.get("vol_percentile_window", 252)
    momentum_lookback = params.get("momentum_lookback", 720)
    momentum_short = params.get("momentum_short", 168)
    use_momentum = params.get("use_momentum_rotation", True)
    mom_threshold = params.get("momentum_threshold", 0.0)
    use_risk_regime = params.get("use_risk_regime", True)
    use_onchain = params.get("use_onchain", False)

    # ── 1. Risk Regime ──
    if use_risk_regime:
        regime = compute_risk_regime(
            df,
            adx_period=adx_period,
            vol_lookback=vol_lookback,
            vol_percentile_window=vol_pctile_window,
        )
    else:
        regime = pd.Series(1.0, index=df.index)

    # ── 2. 動量輪動 ──
    if use_momentum:
        mom_score = compute_momentum_score(
            df,
            lookback=momentum_lookback,
            short_lookback=momentum_short,
        )
        # 動量 > threshold → 全額，否則 → 縮減
        mom_scalar = pd.Series(1.0, index=df.index)
        mom_scalar[mom_score < mom_threshold] = 0.5
        mom_scalar[mom_score < -abs(mom_threshold * 2)] = 0.0
    else:
        mom_scalar = pd.Series(1.0, index=df.index)

    # ── 3. 鏈上 regime ──
    if use_onchain:
        onchain_scalar = compute_onchain_regime(ctx)
    else:
        onchain_scalar = 1.0

    # ── 4. 合成 exposure ──
    exposure = regime * mom_scalar * onchain_scalar
    exposure = exposure.clip(0.0, 1.0)

    # 作為獨立策略時：exposure 就是持倉比例（做多）
    # 作為 overlay 時：上層策略乘以此 exposure scalar
    return exposure
