"""
LSR Confirmatory Overlay — 散戶多空比擇時增強層（v2: OI + FR 確認）

設計理念：
    不改變策略的進場方向，只根據散戶 LSR（Long/Short Ratio）
    的極端水平對已有持倉做「放大 / 縮減」：

    - TSMOM long + LSR 顯示空頭擁擠（支持做多）→ boost position
    - TSMOM long + LSR 顯示多頭擁擠（反對做多）→ reduce position
    - TSMOM short + vice versa
    - LSR 在中性區 → 不改變

    v2 升級（D_oi_fr_confirm）：在 LSR 極端判定後，加入 OI + FR 確認層
    梯度調整 boost/reduce 強度：
        - OI rising（24h pct_change > 0）→ 確認 +1
        - FR double crowding（LSR 高且 fr_pctrank > 0.7，或 LSR 低且 fr_pctrank < 0.3）→ 確認 +1
        - N ∈ {0, 1, 2}：boost_eff = base_boost + 0.15×N, reduce_eff = max(0.05, base_reduce - 0.10×N)

    與 vol_pause（急性風控）互補：
    - vol_pause：高波動期暫停交易（防禦性）
    - lsr_confirmatory：LSR 極端時放大/縮減（進攻 + 防禦）

Alpha 來源：
    散戶 LSR 反映群眾情緒，極端值是有效的反向指標。
    IC (168h LSR pctrank vs 24h fwd return) = -0.025（穩定為負）。
    OI rising + FR double crowding 提供獨立確認，D mode +0.090 SR vs A。

Anti-lookahead 保證：
    - LSR percentile rank 只用 [0, i] 的歷史數據
    - OI pct_change(24) 只用 [0, i] 的歷史數據
    - FR pctrank 只用 [0, i] 的歷史數據
    - forward-fill 對齊（嚴格因果）
    - 結果 position[i] 在 bar[i+1] 開盤執行（配合 trade_on=next_open）

Research Evidence:
    - notebooks/research/20260226_lsr_tsmom_hybrid_overlay.ipynb (A mode)
    - notebooks/research/20260227_lsr_full_alpha_exploration.ipynb (D mode)
    - A mode: Δ_Sharpe = +0.434, 8/8 symbols
    - D mode: Δ_Sharpe = +0.524, 8/8 symbols, all better than A
    - D vs A: +0.090 avg SR improvement
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


def compute_lsr_pctrank(
    lsr_series: pd.Series,
    window: int = 168,
) -> pd.Series:
    """
    計算 LSR rolling percentile rank

    Args:
        lsr_series: LSR 時間序列（已對齊到 K 線）
        window: 滾動窗口（bars）

    Returns:
        percentile rank [0, 1]，0 = 歷史最低，1 = 歷史最高
    """
    min_p = max(window // 2, 24)
    return lsr_series.rolling(window, min_periods=min_p).apply(
        lambda x: sp_stats.percentileofscore(x, x.iloc[-1]) / 100.0,
        raw=False,
    )


def _compute_oi_confirm(
    oi_series: pd.Series | None,
    index: pd.DatetimeIndex,
    oi_change_window: int = 24,
) -> np.ndarray:
    """
    計算 OI rising 確認信號

    OI 在過去 oi_change_window bars 上升 → 確認 = 1，否則 = 0。

    Args:
        oi_series: OI 時間序列（可為 None）
        index: 對齊用的 DatetimeIndex
        oi_change_window: OI 變化率計算窗口（預設 24）

    Returns:
        np.ndarray of {0, 1}，長度同 index
    """
    n = len(index)
    if oi_series is None:
        return np.zeros(n, dtype=float)

    oi_aligned = oi_series.reindex(index, method="ffill")
    oi_rising = oi_aligned.pct_change(oi_change_window) > 0
    # NaN → 0（不確認）
    return oi_rising.fillna(False).astype(float).values


def _compute_fr_double_crowding(
    fr_series: pd.Series | None,
    lsr_crowded_long: pd.Series,
    lsr_crowded_short: pd.Series,
    index: pd.DatetimeIndex,
    fr_pctrank_window: int = 168,
    fr_crowding_hi: float = 0.7,
    fr_crowding_lo: float = 0.3,
) -> np.ndarray:
    """
    計算 FR double crowding 確認信號

    當 LSR 極端 + FR 也極端（同方向擁擠）→ 確認 = 1：
    - LSR 多頭擁擠（high）且 fr_pctrank > fr_crowding_hi → double crowding
    - LSR 空頭擁擠（low） 且 fr_pctrank < fr_crowding_lo → double crowding

    Args:
        fr_series: Funding Rate 時間序列（可為 None）
        lsr_crowded_long: LSR 多頭擁擠 boolean Series
        lsr_crowded_short: LSR 空頭擁擠 boolean Series
        index: 對齊用的 DatetimeIndex
        fr_pctrank_window: FR percentile rank 窗口
        fr_crowding_hi: FR 高擁擠門檻（預設 0.7）
        fr_crowding_lo: FR 低擁擠門檻（預設 0.3）

    Returns:
        np.ndarray of {0, 1}，長度同 index
    """
    n = len(index)
    if fr_series is None:
        return np.zeros(n, dtype=float)

    fr_aligned = fr_series.reindex(index, method="ffill")
    coverage = (~fr_aligned.isna()).mean()
    if coverage < 0.1:
        logger.debug(f"  FR coverage {coverage:.1%} too low, skipping FR confirm")
        return np.zeros(n, dtype=float)

    # 計算 FR percentile rank
    min_p = max(fr_pctrank_window // 2, 24)
    fr_pctrank = fr_aligned.rolling(fr_pctrank_window, min_periods=min_p).apply(
        lambda x: sp_stats.percentileofscore(x, x.iloc[-1]) / 100.0,
        raw=False,
    )

    fr_hi = fr_pctrank > fr_crowding_hi
    fr_lo = fr_pctrank < fr_crowding_lo
    fr_valid = ~fr_pctrank.isna()

    crowded_long_arr = lsr_crowded_long.values.astype(bool)
    crowded_short_arr = lsr_crowded_short.values.astype(bool)
    fr_hi_arr = fr_hi.values.astype(bool)
    fr_lo_arr = fr_lo.values.astype(bool)
    fr_valid_arr = fr_valid.values.astype(bool)

    # LSR 多頭擁擠 + FR 高 → double crowding
    # LSR 空頭擁擠 + FR 低 → double crowding
    double_crowding = np.zeros(n, dtype=float)
    double_crowding[crowded_long_arr & fr_hi_arr & fr_valid_arr] = 1.0
    double_crowding[crowded_short_arr & fr_lo_arr & fr_valid_arr] = 1.0

    return double_crowding


def apply_lsr_confirmatory_overlay(
    position: pd.Series,
    price_df: pd.DataFrame,
    lsr_series: pd.Series | None,
    params: dict,
) -> pd.Series:
    """
    LSR Confirmatory Scaling Overlay (v2: OI + FR 確認層)

    核心邏輯：
        根據散戶 LSR percentile rank 對已有持倉做放大/縮減：

        1. 計算 LSR percentile rank（rolling window）
        2. 判斷 LSR 極端方向：
           - pctrank > entry_pctile → 多頭擁擠（支持做空）
           - pctrank < (1 - entry_pctile) → 空頭擁擠（支持做多）
        3. 與策略持倉方向交叉比對：
           - 持多 + 空頭擁擠（LSR 支持）→ scale_boost
           - 持多 + 多頭擁擠（LSR 反對）→ scale_reduce
           - 持空 + 多頭擁擠（LSR 支持）→ scale_boost
           - 持空 + 空頭擁擠（LSR 反對）→ scale_reduce
           - LSR 非極端 → 不改變（scale = 1.0）
        4. (v2) OI + FR 確認層梯度調整 boost/reduce：
           - OI rising（24h pct_change > 0）→ N += 1
           - FR double crowding（LSR+FR 同向極端）→ N += 1
           - boost_eff = base_boost + 0.15 × N
           - reduce_eff = max(0.05, base_reduce - 0.10 × N)

    特性：
        - 雙向有效（多空皆可 boost/reduce）
        - 不改變進場方向（scale ∈ [reduce, boost]，reduce > 0）
        - 與 vol_pause 互補（vol_pause 做急性平倉，LSR 做擇時縮放）
        - OI/FR 確認層為可選，無數據時 graceful fallback 到 A mode

    Args:
        position: 原始持倉信號 [-1, 1]
        price_df: K 線 DataFrame（用於索引對齊，本身不使用價格）
        lsr_series: LSR 時間序列（已對齊到 K 線 index），可為 None
        params: overlay 參數：
            lsr_window:          int    LSR percentile rank 窗口（預設 168）
            lsr_entry_pctile:    float  極端判定門檻（預設 0.85）
            lsr_scale_boost:     float  方向一致時放大倍率（預設 1.3）
            lsr_scale_reduce:    float  方向矛盾時縮減倍率（預設 0.5）
            lsr_type:            str    LSR 類型（預設 "lsr"，用於數據載入）
            lsr_min_coverage:    float  最低覆蓋率門檻（預設 0.3）
            lsr_pos_threshold:   float  持倉判定門檻（|pos| > threshold 才算持倉）
            oi_confirm_enabled:  bool   OI 確認層開關（預設 False，向後相容）
            fr_confirm_enabled:  bool   FR 確認層開關（預設 False，向後相容）
            _oi_series:          Series OI 數據（由 pipeline 注入）
            _fr_series:          Series FR 數據（由 pipeline 注入）

    Returns:
        修改後的持倉信號 [-1, 1]
    """
    # ── 參數解析 ──
    lsr_window = int(params.get("lsr_window", 168))
    entry_pctile = float(params.get("lsr_entry_pctile", 0.85))
    scale_boost = float(params.get("lsr_scale_boost", 1.3))
    scale_reduce = float(params.get("lsr_scale_reduce", 0.5))
    min_coverage = float(params.get("lsr_min_coverage", 0.3))
    pos_threshold = float(params.get("lsr_pos_threshold", 0.05))

    # v2: OI + FR 確認層參數
    oi_confirm_enabled = bool(params.get("oi_confirm_enabled", False))
    fr_confirm_enabled = bool(params.get("fr_confirm_enabled", False))

    # ── Guard: 無 LSR 數據時直接返回 ──
    if lsr_series is None:
        logger.warning("📊 LSR Confirmatory Overlay: no LSR data, skipping")
        return position

    # ── 對齊到 position index ──
    lsr_aligned = lsr_series.reindex(position.index).ffill()

    # ── 覆蓋率檢查 ──
    coverage = (~lsr_aligned.isna()).mean()
    if coverage < min_coverage:
        logger.warning(
            f"📊 LSR Confirmatory Overlay: coverage {coverage:.1%} < "
            f"{min_coverage:.0%}, skipping"
        )
        return position

    # ── 計算 LSR percentile rank ──
    lsr_pctrank = compute_lsr_pctrank(lsr_aligned, window=lsr_window)

    # ── 極端判定 ──
    entry_hi = entry_pctile       # e.g., 0.85 → top 15%（多頭擁擠）
    entry_lo = 1.0 - entry_pctile  # e.g., 0.15 → bottom 15%（空頭擁擠）

    lsr_crowded_long = lsr_pctrank > entry_hi   # 多頭擁擠 → 支持做空
    lsr_crowded_short = lsr_pctrank < entry_lo  # 空頭擁擠 → 支持做多

    # ── v2: OI + FR 確認層 ──
    n_confirm = np.zeros(len(position), dtype=float)

    if oi_confirm_enabled:
        oi_series = params.get("_oi_series")
        oi_confirm = _compute_oi_confirm(oi_series, position.index)
        # OI 確認只在 LSR 極端時有意義
        lsr_extreme = lsr_crowded_long.values | lsr_crowded_short.values
        n_confirm += oi_confirm * lsr_extreme
        n_oi_confirm = int((oi_confirm * lsr_extreme).sum())
    else:
        n_oi_confirm = 0

    if fr_confirm_enabled:
        fr_series = params.get("_fr_series")
        fr_confirm = _compute_fr_double_crowding(
            fr_series, lsr_crowded_long, lsr_crowded_short, position.index,
        )
        n_confirm += fr_confirm
        n_fr_confirm = int(fr_confirm.sum())
    else:
        n_fr_confirm = 0

    # ── 計算 effective boost/reduce（梯度調整）──
    # boost_eff = base_boost + 0.15 × N
    # reduce_eff = max(0.05, base_reduce - 0.10 × N)
    boost_arr = scale_boost + 0.15 * n_confirm
    reduce_arr = np.maximum(0.05, scale_reduce - 0.10 * n_confirm)

    # ── 計算 scale ──
    pos_arr = position.values.astype(float)
    scale = np.ones(len(position), dtype=float)

    is_long = pos_arr > pos_threshold
    is_short = pos_arr < -pos_threshold
    crowded_long_arr = lsr_crowded_long.values.astype(bool)
    crowded_short_arr = lsr_crowded_short.values.astype(bool)

    # 處理 NaN（pctrank 暖身期無值 → scale = 1.0）
    pctrank_valid = ~lsr_pctrank.isna().values

    # 持多 + 空頭擁擠（LSR 支持做多）→ boost
    mask = is_long & crowded_short_arr & pctrank_valid
    scale[mask] = boost_arr[mask]
    # 持多 + 多頭擁擠（LSR 反對做多）→ reduce
    mask = is_long & crowded_long_arr & pctrank_valid
    scale[mask] = reduce_arr[mask]
    # 持空 + 多頭擁擠（LSR 支持做空）→ boost
    mask = is_short & crowded_long_arr & pctrank_valid
    scale[mask] = boost_arr[mask]
    # 持空 + 空頭擁擠（LSR 反對做空）→ reduce
    mask = is_short & crowded_short_arr & pctrank_valid
    scale[mask] = reduce_arr[mask]

    result = (pos_arr * scale).clip(-1.0, 1.0)

    # ── 統計 ──
    n_boosted = int(((is_long & crowded_short_arr) | (is_short & crowded_long_arr)).sum())
    n_reduced = int(((is_long & crowded_long_arr) | (is_short & crowded_short_arr)).sum())
    n_total = len(position)
    n_with_pos = int((np.abs(pos_arr) > pos_threshold).sum())

    confirm_str = ""
    if oi_confirm_enabled or fr_confirm_enabled:
        confirm_str = (
            f", oi_confirm={n_oi_confirm}, fr_confirm={n_fr_confirm}, "
            f"avg_N={n_confirm[n_confirm > 0].mean():.2f}" if n_confirm.sum() > 0
            else f", oi_confirm={n_oi_confirm}, fr_confirm={n_fr_confirm}, avg_N=0"
        )

    logger.info(
        f"📊 LSR Confirmatory Overlay: "
        f"boosted={n_boosted} ({n_boosted/n_total*100:.1f}%), "
        f"reduced={n_reduced} ({n_reduced/n_total*100:.1f}%), "
        f"active_bars={n_with_pos}, "
        f"coverage={coverage:.1%}, "
        f"params(boost={scale_boost}, reduce={scale_reduce}, "
        f"pctile={entry_pctile}, window={lsr_window})"
        f"{confirm_str}"
    )

    return pd.Series(result, index=position.index)
