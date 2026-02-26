"""
TSMOM + Carry V2 — Per-Symbol 多因子複合策略

學術背景：
    - Moskowitz, Ooi & Pedersen (2012): Time-Series Momentum
    - Koijen et al. (2018): Carry factor 跨資產類別皆有效
    - BTC 720h TSMOM: OOS 驗證通過（avg OOS IC=+0.065, 7/7 gates PASS）

動機（Alpha Researcher 2026-02-24 BTC 增強報告）：
    1. BTC 需要更長的動量窗口（720h vs 168h），反映 BTC 獨有的長趨勢週期
    2. BTC FR 結構性做空偏差（>70% 時間 FR>0），用 Basis 替代 FR 解決
    3. ETH OI 信號 IC=+0.034，可補充 TSMOM
    4. 其他幣種用 FR+Basis (BasisCarry) 混合信號

Per-Symbol 配置分層：
    Tier 1 (BTC):       TSMOM(720h, EMA 30/100) + Basis confirmation
    Tier 2 (ETH):       TSMOM(168h) + OI/FR/Basis confirmation
    Tier 3 (SOL/AVAX):  TSMOM(168h) + BasisCarry confirmation
    Tier 4 (Default):   TSMOM(168h) + BasisCarry confirmation
    Tier 5 (tsmom_only): 純 TSMOM，不做 carry confirmation（用於 IC 不穩定的幣種如 XRP）

組合模式：
    confirmatory（預設）: Carry 作為 TSMOM 倉位縮放器，而非加性因子
        - Carry 與 TSMOM 方向一致 → 1.0x（全倉）
        - Carry 與 TSMOM 方向相反 → 0.5x（減倉）
        - 任一方向為零 → 0.7x（中性）
        這樣可以保留 TSMOM 趨勢 alpha，同時利用 carry 做風控

    additive: 傳統加權平均（已證實效果不佳——carry 的反趨勢性質
        會在趨勢期嚴重稀釋 TSMOM alpha）

HTF (Higher-TimeFrame) 過濾器（opt-in, 預設 disabled）：
    根據 Alpha Researcher P1 分析 (2026-02-26)：
    - C_4h+daily_hard 方案：24h IC +212%，8/8 幣種改善
    - 4h 趨勢 + 1d regime 共振過濾，過濾假信號
    - htf_filter_enabled: true 啟用
    - 4h/1d 由 1h resample（causal, closed='left', label='left'）

Turnover 控制機制：
    1. 信號 EMA 平滑（composite_ema）— 降低高頻雜訊
    2. 持倉量化（position_step）— 離散化持倉到步長
    3. 死區閾值（min_change_threshold）— 只在持倉變化超過閾值時更新

Note:
    signal_delay 和 direction clip 由 @register_strategy 框架自動處理，
    策略函數只需回傳 raw position [-1, 1]。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from .base import StrategyContext
from . import register_strategy
from ..indicators import calculate_ema

import logging

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
#  子因子 1: TSMOM + EMA（沿用驗證過的實作）
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
    TSMOM + EMA 趨勢對齊信號

    Args:
        close: 收盤價序列
        lookback: TSMOM 回看期（BTC=720h, others=168h）
        vol_target: 年化波動率目標
        ema_fast: 快速 EMA 週期（BTC=30, others=20）
        ema_slow: 慢速 EMA 週期（BTC=100, others=50）
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
#  子因子 2: Basis 信號
# ══════════════════════════════════════════════════════════════

def _basis_signal(
    close: pd.Series,
    ema_fast: int = 24,
    ema_slow: int = 168,
    tanh_scale: float = 50.0,
) -> pd.Series:
    """
    Basis spread 反向信號

    邏輯：價格遠離長期均線 → 預期回歸
    - EMA 快線 >> 慢線 → 過熱 → 做空
    - EMA 快線 << 慢線 → 恐慌 → 做多
    - tanh 壓縮到 [-1, 1]
    """
    ema_f = calculate_ema(close, ema_fast)
    ema_s = calculate_ema(close, ema_slow)
    spread = (ema_f - ema_s) / ema_s
    return (-np.tanh(spread * tanh_scale)).fillna(0.0)


# ══════════════════════════════════════════════════════════════
#  子因子 3: FR (Funding Rate) 信號
# ══════════════════════════════════════════════════════════════

def _fr_signal(
    fr_aligned: pd.Series,
    rolling_window: int = 72,
) -> pd.Series:
    """
    Funding Rate carry 信號

    邏輯：正 FR = 多頭擁擠 → 做空；負 FR = 空頭擁擠 → 做多
    """
    mean_fr = fr_aligned.rolling(rolling_window, min_periods=1).mean()
    return (-np.sign(mean_fr)).fillna(0.0)


# ══════════════════════════════════════════════════════════════
#  子因子 4: OI (Open Interest) z-score 信號
# ══════════════════════════════════════════════════════════════

def _oi_zscore_signal(
    oi_value: pd.Series,
    close: pd.Series,
    z_window: int = 168,
) -> pd.Series:
    """
    OI/Price z-score 信號

    邏輯：OI/價格比偏高 → 槓桿過高 → 預期回調 → 做空
    """
    ratio = oi_value / close
    rm = ratio.rolling(z_window).mean()
    rs = ratio.rolling(z_window).std()
    z = (ratio - rm) / rs.replace(0, np.nan)
    return (-np.tanh(z)).fillna(0.0)


# ══════════════════════════════════════════════════════════════
#  Confirmatory Combination: Carry 作為 TSMOM 倉位縮放器
# ══════════════════════════════════════════════════════════════

def _carry_confirmation_scale(
    sig_tsmom: pd.Series,
    sig_carry: pd.Series,
    agree_scale: float = 1.0,
    disagree_scale: float = 0.5,
    neutral_scale: float = 0.7,
    smoothing: int = 24,
) -> pd.Series:
    """
    Carry 確認縮放：用 carry 信號的方向性來縮放 TSMOM 倉位

    核心邏輯：
        - Carry 與 TSMOM 方向一致 → agree_scale (1.0x，全倉)
        - Carry 與 TSMOM 方向相反 → disagree_scale (0.5x，減倉但不反向)
        - 任一方為零 → neutral_scale (0.7x)

    這是 tsmom_ema 「agree_weight / disagree_weight」模式的推廣。
    相比加性組合，確認模式：
        1. 永遠不會反轉 TSMOM 的方向（保留趨勢 alpha）
        2. 在 carry 反對時減倉（風控）
        3. 在 carry 確認時維持全倉（保留最大收益）
        4. 自然降低 turnover（縮放變化比方向切換平滑）

    Args:
        sig_tsmom: TSMOM 基礎信號 [-1, 1]
        sig_carry: Carry 信號 [-1, 1]（Basis, FR, OI, 或其組合）
        agree_scale: TSMOM & carry 同方向時的倉位縮放
        disagree_scale: TSMOM & carry 反方向時的倉位縮放
        neutral_scale: 任一方為零時的倉位縮放
        smoothing: 縮放因子的 EMA 平滑週期（避免快速切換）

    Returns:
        縮放後的倉位 = sig_tsmom * scale
    """
    tsmom_dir = np.sign(sig_tsmom)
    carry_dir = np.sign(sig_carry)

    # Agreement: +1 (agree), -1 (disagree), 0 (one or both flat)
    agreement = tsmom_dir * carry_dir

    scale = pd.Series(neutral_scale, index=sig_tsmom.index)
    scale[agreement > 0] = agree_scale      # carry confirms TSMOM
    scale[agreement < 0] = disagree_scale    # carry opposes TSMOM

    # Smooth to avoid rapid scale oscillation
    if smoothing > 0:
        scale = scale.ewm(span=smoothing, adjust=False).mean()

    return sig_tsmom * scale


def _multi_carry_confirmation_scale(
    sig_tsmom: pd.Series,
    carry_signals: list[pd.Series],
    carry_weights: list[float],
    agree_scale: float = 1.0,
    disagree_scale: float = 0.5,
    neutral_scale: float = 0.7,
    smoothing: int = 24,
) -> pd.Series:
    """
    多 carry 信號的確認縮放（用於 ETH tier 等多因子場景）

    邏輯：計算加權的 agreement score，然後映射到 [disagree_scale, agree_scale]

    Args:
        sig_tsmom: TSMOM 基礎信號
        carry_signals: 多個 carry 信號的列表
        carry_weights: 對應的權重列表（會自動歸一化）
        agree_scale/disagree_scale/neutral_scale: 同 _carry_confirmation_scale
        smoothing: EMA 平滑週期
    """
    tsmom_dir = np.sign(sig_tsmom)

    # 加權 agreement score: ∈ [-1, +1]
    total_w = sum(carry_weights)
    if total_w == 0:
        return sig_tsmom * neutral_scale

    weighted_agree = pd.Series(0.0, index=sig_tsmom.index)
    for sig_c, w in zip(carry_signals, carry_weights):
        carry_dir = np.sign(sig_c)
        # per-signal agreement: +1, -1, or 0
        agree_i = tsmom_dir * carry_dir
        weighted_agree += agree_i * (w / total_w)

    # Map agreement score to scale:
    # agree_score = +1 → agree_scale, -1 → disagree_scale, 0 → neutral_scale
    # Linear interpolation:
    #   score > 0: scale = neutral + score * (agree - neutral)
    #   score < 0: scale = neutral + score * (neutral - disagree)
    scale = pd.Series(neutral_scale, index=sig_tsmom.index)
    pos_mask = weighted_agree > 0
    neg_mask = weighted_agree < 0
    scale[pos_mask] = neutral_scale + weighted_agree[pos_mask] * (agree_scale - neutral_scale)
    scale[neg_mask] = neutral_scale + weighted_agree[neg_mask] * (neutral_scale - disagree_scale)

    if smoothing > 0:
        scale = scale.ewm(span=smoothing, adjust=False).mean()

    return sig_tsmom * scale


# ══════════════════════════════════════════════════════════════
#  Turnover 控制：EMA 平滑 + 量化 + 死區
# ══════════════════════════════════════════════════════════════

def _apply_turnover_control(
    raw_pos: pd.Series,
    composite_ema: int = 12,
    position_step: float = 0.1,
    min_change_threshold: float = 0.05,
) -> pd.Series:
    """
    三層 turnover 控制，將連續信號轉為可交易的低頻持倉

    1. EMA 平滑：消除高頻雜訊，讓信號更持久
    2. 量化：離散化到 position_step 步長（如 0.1 → 0, ±0.1, ±0.2, ...）
    3. 死區：只有在新持倉與當前持倉差距超過閾值時才更新

    Args:
        raw_pos: 原始連續持倉信號
        composite_ema: EMA 平滑週期（0=不平滑）
        position_step: 持倉量化步長（如 0.1）
        min_change_threshold: 最小持倉變化閾值

    Returns:
        離散化、平滑後的持倉信號
    """
    # 1. EMA 平滑
    if composite_ema > 0:
        smoothed = raw_pos.ewm(span=composite_ema, adjust=False).mean()
    else:
        smoothed = raw_pos.copy()

    # 2. 量化到步長
    if position_step > 0:
        quantized = (smoothed / position_step).round() * position_step
    else:
        quantized = smoothed

    # 3. 死區（只在變化超過閾值時更新持倉）
    if min_change_threshold > 0:
        values = quantized.values.copy()
        current_pos = 0.0
        for i in range(len(values)):
            if np.isnan(values[i]):
                values[i] = current_pos
                continue
            if abs(values[i] - current_pos) >= min_change_threshold:
                current_pos = values[i]
            else:
                values[i] = current_pos
        return pd.Series(values, index=quantized.index)

    return quantized


# ══════════════════════════════════════════════════════════════
#  數據載入輔助函數
# ══════════════════════════════════════════════════════════════

def _load_and_align_fr(
    symbol: str,
    kline_index: pd.DatetimeIndex,
    data_dir: str | Path = "data",
) -> Optional[pd.Series]:
    """載入 FR 並對齊到 kline index（causal forward-fill）"""
    fpath = Path(data_dir) / "binance" / "futures" / "funding_rate" / f"{symbol}.parquet"
    if not fpath.exists():
        logger.warning(f"tsmom_carry_v2: {symbol} FR 數據不存在: {fpath}")
        return None

    try:
        fr = pd.read_parquet(fpath)
        fr_col = "funding_rate" if "funding_rate" in fr.columns else (
            "fundingRate" if "fundingRate" in fr.columns else fr.columns[0]
        )
        fr_series = fr[fr_col]
        return fr_series.reindex(kline_index, method="ffill")
    except Exception as e:
        logger.warning(f"tsmom_carry_v2: {symbol} FR 載入失敗: {e}")
        return None


def _load_and_align_oi(
    symbol: str,
    kline_index: pd.DatetimeIndex,
    data_dir: str | Path = "data",
) -> Optional[pd.Series]:
    """載入 OI 並對齊到 kline index（causal forward-fill）"""
    for subdir in ["merged", ""]:
        parts = ["binance", "futures", "open_interest"]
        if subdir:
            parts.append(subdir)
        fpath = Path(data_dir) / "/".join(parts) / f"{symbol}.parquet"
        if fpath.exists():
            try:
                oi = pd.read_parquet(fpath)
                if "sumOpenInterestValue" in oi.columns:
                    return oi["sumOpenInterestValue"].reindex(kline_index, method="ffill")
            except Exception as e:
                logger.warning(f"tsmom_carry_v2: {symbol} OI 載入失敗 ({fpath}): {e}")

    logger.info(f"tsmom_carry_v2: {symbol} 無 OI 數據")
    return None


# ══════════════════════════════════════════════════════════════
#  策略: TSMOM + Carry V2（Per-Symbol 多因子複合）
# ══════════════════════════════════════════════════════════════

@register_strategy("tsmom_carry_v2")
def generate_tsmom_carry_v2(
    df: pd.DataFrame, ctx: StrategyContext, params: dict
) -> pd.Series:
    """
    TSMOM + Carry V2 per-symbol 多因子策略

    依 symbol tier 自動選擇因子組合。
    Tier 由 symbol_overrides 中的 ``tier`` 參數決定：
      - "btc_enhanced": TSMOM(720h, EMA 30/100) + Basis confirmation
      - "eth_enhanced": TSMOM(168h) + OI/FR/Basis multi-confirmation
      - "tsmom_heavy":  TSMOM(168h) + BasisCarry confirmation (70/30)
      - "default":      TSMOM(168h) + BasisCarry confirmation (50/50)
      - "tsmom_only":   純 TSMOM（不做 carry confirmation，用於 IC 不穩定的幣種）

    組合模式（combination_mode）：
      - "confirmatory"（預設）: Carry 作為 TSMOM 倉位確認器
          TSMOM 決定方向，carry 決定倉位大小
          agree → agree_scale (1.0x), disagree → disagree_scale (0.5x)
      - "additive": 傳統加權平均（不推薦——carry 反趨勢性稀釋 TSMOM alpha）

    HTF Filter（opt-in, 預設 disabled）：
      - htf_filter_enabled: True 啟用多時間框架共振過濾
      - C_4h+daily_hard 方案（Alpha Researcher P1, 2026-02-26）
      - 4h agree + daily agree + trending → 1.0x
      - 4h agree + daily agree + ranging  → 0.85x
      - 4h agree + daily disagree         → 0.7x
      - 4h disagree                       → 0.0x（不交易）
      - 參數：htf_4h_ema_fast/slow, htf_adx_period/threshold, htf_regime_ema
      - 權重：htf_full_confirm, htf_partial_confirm, htf_4h_only_confirm, htf_no_confirm

    Turnover 控制（三層）：
      - composite_ema: 最終信號 EMA 平滑（預設 12h）
      - position_step: 持倉量化步長（預設 0.1）
      - min_change_threshold: 最小持倉變化閾值（預設 0.05）
    """
    close = df["close"]
    data_dir = params.get("_data_dir", params.get("data_dir", "data"))
    tier = params.get("tier", "default")
    mode = params.get("combination_mode", "confirmatory")

    # ── Turnover 控制參數 ──
    composite_ema = int(params.get("composite_ema", 12))
    position_step = float(params.get("position_step", 0.1))
    min_change = float(params.get("min_change_threshold", 0.05))

    # ── Confirmatory 模式參數 ──
    agree_scale = float(params.get("carry_agree_scale", 1.0))
    disagree_scale = float(params.get("carry_disagree_scale", 0.5))
    neutral_scale = float(params.get("carry_neutral_scale", 0.7))
    confirm_smoothing = int(params.get("carry_confirm_smoothing", 24))

    # ── TSMOM 信號（所有 tier 共用） ──
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

    # ── Basis 信號（所有 tier 共用） ──
    basis_ema_fast = int(params.get("basis_ema_fast", 24))
    basis_ema_slow = int(params.get("basis_ema_slow", 168))
    basis_tanh_scale = float(params.get("basis_tanh_scale", 50.0))

    sig_basis = _basis_signal(
        close,
        ema_fast=basis_ema_fast,
        ema_slow=basis_ema_slow,
        tanh_scale=basis_tanh_scale,
    )

    # ══════════════════════════════════════════════════
    # TIER 路由
    # ══════════════════════════════════════════════════

    if tier == "tsmom_only":
        # 純 TSMOM，不經過 carry confirmation
        # 用於 IC 不穩定的幣種（如 XRP），避免 carry 增加雜訊
        raw_pos = sig_tsmom.copy()

    elif tier == "btc_enhanced":
        raw_pos = _route_btc_enhanced(
            sig_tsmom, sig_basis, params, mode,
            agree_scale, disagree_scale, neutral_scale, confirm_smoothing,
        )

    elif tier == "eth_enhanced":
        raw_pos = _route_eth_enhanced(
            sig_tsmom, sig_basis, close, df.index, ctx.symbol, data_dir, params, mode,
            agree_scale, disagree_scale, neutral_scale, confirm_smoothing,
        )

    elif tier == "tsmom_heavy":
        raw_pos = _route_tsmom_carry(
            sig_tsmom, sig_basis, close, df.index, ctx.symbol, data_dir, params, mode,
            w_tsmom_default=0.70, w_bc_default=0.30,
            agree_scale=agree_scale, disagree_scale=disagree_scale,
            neutral_scale=neutral_scale, confirm_smoothing=confirm_smoothing,
        )

    else:  # "default"
        raw_pos = _route_tsmom_carry(
            sig_tsmom, sig_basis, close, df.index, ctx.symbol, data_dir, params, mode,
            w_tsmom_default=0.50, w_bc_default=0.50,
            agree_scale=agree_scale, disagree_scale=disagree_scale,
            neutral_scale=neutral_scale, confirm_smoothing=confirm_smoothing,
        )

    # ══════════════════════════════════════════════════
    # HTF Filter (opt-in, default disabled)
    # Source: Alpha Researcher P1 分析 (2026-02-26)
    # Result: C_4h+daily_hard, 24h IC +212%, 8/8 幣種改善
    # ══════════════════════════════════════════════════
    htf_filter_enabled = params.get("htf_filter_enabled", False)
    if htf_filter_enabled:
        raw_pos = _apply_htf_filter(
            raw_pos, df,
            ctx_symbol=ctx.symbol,
            htf_4h_ema_fast=int(params.get("htf_4h_ema_fast", 20)),
            htf_4h_ema_slow=int(params.get("htf_4h_ema_slow", 50)),
            htf_adx_period=int(params.get("htf_adx_period", 14)),
            htf_adx_threshold=float(params.get("htf_adx_threshold", 25.0)),
            htf_regime_ema=int(params.get("htf_regime_ema", 20)),
            htf_full_confirm=float(params.get("htf_full_confirm", 1.0)),
            htf_partial_confirm=float(params.get("htf_partial_confirm", 0.85)),
            htf_4h_only_confirm=float(params.get("htf_4h_only_confirm", 0.7)),
            htf_no_confirm=float(params.get("htf_no_confirm", 0.0)),
        )

    # ══════════════════════════════════════════════════
    # Turnover 控制
    # ══════════════════════════════════════════════════
    pos = _apply_turnover_control(
        raw_pos,
        composite_ema=composite_ema,
        position_step=position_step,
        min_change_threshold=min_change,
    )

    result = pos.clip(-1.0, 1.0).fillna(0.0)

    # ══════════════════════════════════════════════════
    # 附帶策略指標（供 signal_generator / Telegram 顯示）
    # ══════════════════════════════════════════════════
    try:
        _last = len(close) - 1
        _ind: dict = {"tier": tier}

        # TSMOM 分數（最後一根 K 線）
        _tsmom_val = float(sig_tsmom.iloc[_last]) if _last < len(sig_tsmom) else 0.0
        _ind["tsmom"] = round(_tsmom_val, 3)

        # EMA 趨勢方向
        _ema_fast_val = close.ewm(span=tsmom_ema_fast, adjust=False).mean().iloc[_last]
        _ema_slow_val = close.ewm(span=tsmom_ema_slow, adjust=False).mean().iloc[_last]
        _ind["ema_trend"] = "UP" if _ema_fast_val > _ema_slow_val else "DOWN"

        # Carry / Basis 分數（非 tsmom_only）
        if tier != "tsmom_only":
            _basis_val = float(sig_basis.iloc[_last]) if _last < len(sig_basis) else 0.0
            _ind["carry"] = round(_basis_val, 3)

        # HTF 狀態
        if htf_filter_enabled:
            _ind["htf"] = "ON"
        else:
            _ind["htf"] = "OFF"

        result.attrs["indicators"] = _ind
    except Exception:
        pass  # 指標附加失敗不影響信號

    # 最終裁剪 — direction clip 由 @register_strategy 框架處理
    return result


# ══════════════════════════════════════════════════════════════
#  HTF (Higher-TimeFrame) Filter — 多時間框架共振過濾
#  Source: Alpha Researcher P1 分析 (2026-02-26)
#  Result: C_4h+daily_hard 方案, 24h IC +212%, 8/8 幣種改善
# ══════════════════════════════════════════════════════════════

def _apply_htf_filter(
    raw_pos: pd.Series,
    df: pd.DataFrame,
    ctx_symbol: str = "",
    htf_4h_ema_fast: int = 20,
    htf_4h_ema_slow: int = 50,
    htf_adx_period: int = 14,
    htf_adx_threshold: float = 25.0,
    htf_regime_ema: int = 20,
    htf_full_confirm: float = 1.0,
    htf_partial_confirm: float = 0.85,
    htf_4h_only_confirm: float = 0.7,
    htf_no_confirm: float = 0.0,
) -> pd.Series:
    """
    多時間框架共振過濾器 (C_4h+daily_hard 方案)

    根據 Alpha Researcher P1 分析 (2026-02-26)：
      - 24h IC: +212% (0.016 → 0.048), 8/8 幣種改善
      - Turnover 降低 40%, 年節省 ~0.80% 交易成本
      - 在趨勢和盤整 regime 皆改善（盤整期改善更大）

    Confirmation weights (C_4h+daily_hard):
        4h agree + daily agree + trending → htf_full_confirm  (1.0)
        4h agree + daily agree + ranging  → htf_partial_confirm (0.85)
        4h agree + daily disagree         → htf_4h_only_confirm (0.7)
        4h disagree                       → htf_no_confirm (0.0)

    因果性保證：
        - 4h/1d 由 1h resample（closed='left', label='left'）
        - forward-fill 對齊到 1h index
        - 無 look-ahead bias（Alpha Researcher P1 已驗證等效）

    Args:
        raw_pos: carry confirmation 後的持倉信號 [-1, 1]
        df: 1h OHLCV DataFrame
        ctx_symbol: 幣種名（用於 log）
        htf_*: HTF 過濾參數

    Returns:
        HTF 過濾後的持倉信號
    """
    from .multi_tf_resonance_strategy import (
        _resample_ohlcv,
        _htf_trend,
        _daily_regime,
    )

    # ── 1. Resample 1h → 4h, 1d (causal) ──
    df_4h = _resample_ohlcv(df, "4h")
    df_1d = _resample_ohlcv(df, "1D")

    if len(df_4h) < htf_4h_ema_slow or len(df_1d) < htf_adx_period:
        logger.warning(
            f"tsmom_carry_v2 [{ctx_symbol}]: HTF filter 數據不足 "
            f"(4h={len(df_4h)}, 1d={len(df_1d)})，跳過 HTF 過濾"
        )
        return raw_pos

    # ── 2. Compute 4h trend → align to 1h ──
    htf_4h_trend = _htf_trend(df_4h, htf_4h_ema_fast, htf_4h_ema_slow)
    htf_4h_aligned = htf_4h_trend.reindex(df.index, method="ffill").fillna(0)

    # ── 3. Compute daily regime → align to 1h ──
    regime = _daily_regime(df_1d, htf_adx_period, htf_adx_threshold, htf_regime_ema)
    daily_dir = regime["regime_direction"].reindex(df.index, method="ffill").fillna(0)
    daily_trending = (
        regime["regime_trend"].reindex(df.index, method="ffill").fillna(0) > 0.5
    )

    # ── 4. Determine alignment ──
    pos_dir = np.sign(raw_pos)
    htf_4h_dir = np.sign(htf_4h_aligned)

    htf_agree = (pos_dir == htf_4h_dir) & (pos_dir != 0) & (htf_4h_dir != 0)
    daily_agree = (pos_dir == daily_dir) & (pos_dir != 0) & (daily_dir != 0)

    # ── 5. Apply C_4h+daily_hard confirmation ──
    confirmation = pd.Series(htf_no_confirm, index=df.index)

    # 4h agree + daily disagree → htf_4h_only_confirm
    confirmation[htf_agree & ~daily_agree] = htf_4h_only_confirm
    # 4h agree + daily agree + ranging → htf_partial_confirm
    confirmation[htf_agree & daily_agree & ~daily_trending] = htf_partial_confirm
    # 4h agree + daily agree + trending → htf_full_confirm
    confirmation[htf_agree & daily_agree & daily_trending] = htf_full_confirm

    filtered = raw_pos * confirmation

    # ── 6. Diagnostic logging ──
    has_signal = pos_dir != 0
    n_signal = has_signal.sum()
    if n_signal > 0:
        n_full = (confirmation[has_signal] == htf_full_confirm).sum()
        n_partial = (confirmation[has_signal] == htf_partial_confirm).sum()
        n_4h_only = (confirmation[has_signal] == htf_4h_only_confirm).sum()
        n_filtered = (confirmation[has_signal] == htf_no_confirm).sum()
        logger.info(
            f"  HTF filter [{ctx_symbol}]: "
            f"full={n_full}/{n_signal} ({n_full/n_signal*100:.1f}%), "
            f"partial={n_partial} ({n_partial/n_signal*100:.1f}%), "
            f"4h_only={n_4h_only} ({n_4h_only/n_signal*100:.1f}%), "
            f"filtered={n_filtered} ({n_filtered/n_signal*100:.1f}%)"
        )

    return filtered


# ══════════════════════════════════════════════════════════════
#  Tier 路由函數
# ══════════════════════════════════════════════════════════════

def _route_btc_enhanced(
    sig_tsmom, sig_basis, params, mode,
    agree_scale, disagree_scale, neutral_scale, confirm_smoothing,
):
    """BTC: TSMOM(720h, EMA 30/100) + Basis confirmation"""
    if mode == "confirmatory":
        return _carry_confirmation_scale(
            sig_tsmom, sig_basis,
            agree_scale=agree_scale,
            disagree_scale=disagree_scale,
            neutral_scale=neutral_scale,
            smoothing=confirm_smoothing,
        )
    else:  # additive
        w_tsmom = float(params.get("w_tsmom", 0.70))
        w_basis = float(params.get("w_basis", 0.30))
        total_w = w_tsmom + w_basis
        return sig_tsmom * (w_tsmom / total_w) + sig_basis * (w_basis / total_w)


def _route_eth_enhanced(
    sig_tsmom, sig_basis, close, kline_index, symbol, data_dir, params, mode,
    agree_scale, disagree_scale, neutral_scale, confirm_smoothing,
):
    """ETH: TSMOM + OI/FR/Basis multi-confirmation"""
    # Build carry signals
    carry_sigs = []
    carry_ws = []

    # OI
    oi_z_window = int(params.get("oi_z_window", 168))
    w_oi = float(params.get("w_oi", 0.25))
    oi_val = _load_and_align_oi(symbol, kline_index, data_dir)
    if oi_val is not None:
        sig_oi = _oi_zscore_signal(oi_val, close, z_window=oi_z_window)
        carry_sigs.append(sig_oi)
        carry_ws.append(w_oi)

    # FR
    fr_rolling = int(params.get("fr_rolling_window", 72))
    w_fr = float(params.get("w_fr", 0.15))
    fr_aligned = _load_and_align_fr(symbol, kline_index, data_dir)
    if fr_aligned is not None:
        sig_fr = _fr_signal(fr_aligned, rolling_window=fr_rolling)
        carry_sigs.append(sig_fr)
        carry_ws.append(w_fr)

    # Basis (always available)
    w_basis = float(params.get("w_basis", 0.20))
    carry_sigs.append(sig_basis)
    carry_ws.append(w_basis)

    if mode == "confirmatory":
        if carry_sigs:
            return _multi_carry_confirmation_scale(
                sig_tsmom, carry_sigs, carry_ws,
                agree_scale=agree_scale,
                disagree_scale=disagree_scale,
                neutral_scale=neutral_scale,
                smoothing=confirm_smoothing,
            )
        return sig_tsmom * neutral_scale
    else:  # additive
        w_tsmom = float(params.get("w_tsmom", 0.40))
        total_w = w_tsmom + sum(carry_ws)
        additive = sig_tsmom * (w_tsmom / total_w)
        for sig_c, w in zip(carry_sigs, carry_ws):
            additive += sig_c * (w / total_w)
        return additive


def _route_tsmom_carry(
    sig_tsmom, sig_basis, close, kline_index, symbol, data_dir, params, mode,
    w_tsmom_default, w_bc_default,
    agree_scale, disagree_scale, neutral_scale, confirm_smoothing,
):
    """tsmom_heavy / default: TSMOM + BasisCarry confirmation"""
    sig_bc = _make_basis_carry(
        symbol, close, kline_index, data_dir, params, sig_basis
    )

    if mode == "confirmatory":
        return _carry_confirmation_scale(
            sig_tsmom, sig_bc,
            agree_scale=agree_scale,
            disagree_scale=disagree_scale,
            neutral_scale=neutral_scale,
            smoothing=confirm_smoothing,
        )
    else:  # additive
        w_tsmom = float(params.get("w_tsmom", w_tsmom_default))
        w_bc = float(params.get("w_basis_carry", w_bc_default))
        total_w = w_tsmom + w_bc
        return sig_tsmom * (w_tsmom / total_w) + sig_bc * (w_bc / total_w)


# ══════════════════════════════════════════════════════════════
#  BasisCarry 混合信號（FR × 0.7 + Basis × 0.3）
# ══════════════════════════════════════════════════════════════

def _make_basis_carry(
    symbol: str,
    close: pd.Series,
    kline_index: pd.DatetimeIndex,
    data_dir: str | Path,
    params: dict,
    sig_basis: pd.Series,
) -> pd.Series:
    """
    BasisCarry = FR_sign × fr_weight + Basis × basis_weight

    如果 FR 不存在，100% fallback 到 Basis。
    """
    fr_w = float(params.get("basis_carry_fr_weight", 0.7))
    basis_w = float(params.get("basis_carry_basis_weight", 0.3))
    fr_rolling = int(params.get("fr_rolling_window", 72))

    fr_aligned = _load_and_align_fr(symbol, kline_index, data_dir)
    if fr_aligned is not None:
        sig_fr = _fr_signal(fr_aligned, rolling_window=fr_rolling)
        sig_bc = (sig_fr * fr_w + sig_basis * basis_w).clip(-1.0, 1.0)
    else:
        logger.info(f"tsmom_carry_v2: {symbol} 無 FR → 使用純 Basis")
        sig_bc = sig_basis.copy()

    return sig_bc.fillna(0.0)
