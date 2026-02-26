"""
LSR Confirmatory Overlay — 散戶多空比擇時增強層

設計理念：
    不改變策略的進場方向，只根據散戶 LSR（Long/Short Ratio）
    的極端水平對已有持倉做「放大 / 縮減」：

    - TSMOM long + LSR 顯示空頭擁擠（支持做多）→ boost position
    - TSMOM long + LSR 顯示多頭擁擠（反對做多）→ reduce position
    - TSMOM short + vice versa
    - LSR 在中性區 → 不改變

    與 vol_pause（急性風控）互補：
    - vol_pause：高波動期暫停交易（防禦性）
    - lsr_confirmatory：LSR 極端時放大/縮減（進攻 + 防禦）

Alpha 來源：
    散戶 LSR 反映群眾情緒，極端值是有效的反向指標。
    IC (168h LSR pctrank vs 24h fwd return) = -0.025（穩定為負）。

Anti-lookahead 保證：
    - LSR percentile rank 只用 [0, i] 的歷史數據
    - forward-fill 對齊（嚴格因果）
    - 結果 position[i] 在 bar[i+1] 開盤執行（配合 trade_on=next_open）

Research Evidence:
    - notebooks/research/20260226_lsr_tsmom_hybrid_overlay.ipynb
    - Confirmatory mode: Δ_Sharpe = +1.11, 8/8 symbols improved
    - MDD improvement: -6.8% vs -8.4% (baseline)
    - Conservative params: boost=1.3, reduce=0.5
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


def apply_lsr_confirmatory_overlay(
    position: pd.Series,
    price_df: pd.DataFrame,
    lsr_series: pd.Series | None,
    params: dict,
) -> pd.Series:
    """
    LSR Confirmatory Scaling Overlay

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

    特性：
        - 雙向有效（多空皆可 boost/reduce）
        - 不改變進場方向（scale ∈ [reduce, boost]，reduce > 0）
        - 與 vol_pause 互補（vol_pause 做急性平倉，LSR 做擇時縮放）

    Args:
        position: 原始持倉信號 [-1, 1]
        price_df: K 線 DataFrame（用於索引對齊，本身不使用價格）
        lsr_series: LSR 時間序列（已對齊到 K 線 index），可為 None
        params: overlay 參數：
            lsr_window:       int    LSR percentile rank 窗口（預設 168）
            lsr_entry_pctile: float  極端判定門檻（預設 0.85）
            lsr_scale_boost:  float  方向一致時放大倍率（預設 1.3）
            lsr_scale_reduce: float  方向矛盾時縮減倍率（預設 0.5）
            lsr_type:         str    LSR 類型（預設 "lsr"，用於數據載入）
            lsr_min_coverage: float  最低覆蓋率門檻（預設 0.3）
            lsr_pos_threshold:float  持倉判定門檻（|pos| > threshold 才算持倉）

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
    scale[is_long & crowded_short_arr & pctrank_valid] = scale_boost
    # 持多 + 多頭擁擠（LSR 反對做多）→ reduce
    scale[is_long & crowded_long_arr & pctrank_valid] = scale_reduce
    # 持空 + 多頭擁擠（LSR 支持做空）→ boost
    scale[is_short & crowded_long_arr & pctrank_valid] = scale_boost
    # 持空 + 空頭擁擠（LSR 反對做空）→ reduce
    scale[is_short & crowded_short_arr & pctrank_valid] = scale_reduce

    result = (pos_arr * scale).clip(-1.0, 1.0)

    # ── 統計 ──
    n_boosted = int(((is_long & crowded_short_arr) | (is_short & crowded_long_arr)).sum())
    n_reduced = int(((is_long & crowded_long_arr) | (is_short & crowded_short_arr)).sum())
    n_total = len(position)
    n_with_pos = int((np.abs(pos_arr) > pos_threshold).sum())

    logger.info(
        f"📊 LSR Confirmatory Overlay: "
        f"boosted={n_boosted} ({n_boosted/n_total*100:.1f}%), "
        f"reduced={n_reduced} ({n_reduced/n_total*100:.1f}%), "
        f"active_bars={n_with_pos}, "
        f"coverage={coverage:.1%}, "
        f"params(boost={scale_boost}, reduce={scale_reduce}, "
        f"pctile={entry_pctile}, window={lsr_window})"
    )

    return pd.Series(result, index=position.index)
