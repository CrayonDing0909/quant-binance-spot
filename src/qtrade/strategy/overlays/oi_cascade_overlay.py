"""
OI Cascade Confirmatory Overlay — OI 清算瀑布擇時增強層

設計理念：
    將 OI Liquidation Bounce 策略的核心 insight 從獨立策略轉化為 overlay：

    - 當 OI 急降 + 價格急跌（清算瀑布），市場傾向反彈
    - 持多 + 清算瀑布觸發 → boost（瀑布結束 = 支持多頭反彈）
    - 持空 + 清算瀑布觸發 → reduce（瀑布結束 = 不利空頭）
    - 空倉 → 不改變（overlay 不生成新進場）

    與 vol_pause + lsr_confirmatory 互補：
    - vol_pause: 高波動暫停（防禦性）
    - lsr_confirmatory: LSR 極端時放大/縮減（情緒面）
    - oi_cascade: 清算瀑布後 boost/reduce（結構面 — OI 清算事件）

Alpha 來源：
    大量清算（多頭爆倉）造成 OI 急劇下降伴隨價格下跌。
    清算瀑布結束後，賣壓消失 → 價格傾向反彈。
    獨立策略 SR=2.49, corr≈0.01（與 TSMOM 正交）。
    轉為 overlay 可避免佔用獨立 runner（Time-in-market 僅 4.2%）。

Anti-lookahead 保證：
    - OI z-score 和 Price z-score 只用 [0, i] 的歷史
    - cascade_active 在觸發 bar 開始，持續 hold_bars
    - 結果 position[i] 在 bar[i+1] 開盤執行（配合 trade_on=next_open）

Research Origin:
    - src/qtrade/strategy/oi_liq_bounce_strategy.py (v4.2 standalone)
    - notebooks/research/20260224_oi_liq_bounce_*.ipynb
    - Standalone metrics: SR=2.49, MDD=-1.3%, time-in-market=4.2%
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
#  核心信號計算（來自 oi_liq_bounce_strategy.py）
# ══════════════════════════════════════════════════════════════

def _compute_oi_change_zscore(
    oi_series: pd.Series,
    change_lookback: int = 24,
    z_window: int = 720,
) -> pd.Series:
    """
    計算 OI 變化率的滾動 z-score

    步驟：
        1. OI 24h 變化率 = (OI - OI_lag) / OI_lag
        2. z-score = (change_rate - rolling_mean) / rolling_std
    """
    if oi_series is None or oi_series.empty:
        return pd.Series(dtype=float)

    oi_lagged = oi_series.shift(change_lookback)
    change_rate = (oi_series - oi_lagged) / oi_lagged.replace(0, np.nan)

    min_periods = max(z_window // 4, 30)
    rolling_mean = change_rate.rolling(z_window, min_periods=min_periods).mean()
    rolling_std = change_rate.rolling(z_window, min_periods=min_periods).std()
    z = (change_rate - rolling_mean) / rolling_std.replace(0, np.nan)

    return z.fillna(0.0).clip(-5.0, 5.0)


def _compute_price_change_zscore(
    close: pd.Series,
    change_lookback: int = 8,
    z_window: int = 720,
) -> pd.Series:
    """
    計算價格變化率的滾動 z-score

    步驟：
        1. Price 8h 變化率 = (close - close_lag) / close_lag
        2. z-score = (change_rate - rolling_mean) / rolling_std
    """
    close_lagged = close.shift(change_lookback)
    change_rate = (close - close_lagged) / close_lagged.replace(0, np.nan)

    min_periods = max(z_window // 4, 30)
    rolling_mean = change_rate.rolling(z_window, min_periods=min_periods).mean()
    rolling_std = change_rate.rolling(z_window, min_periods=min_periods).std()
    z = (change_rate - rolling_mean) / rolling_std.replace(0, np.nan)

    return z.fillna(0.0).clip(-5.0, 5.0)


# ══════════════════════════════════════════════════════════════
#  OI Cascade Confirmatory Overlay
# ══════════════════════════════════════════════════════════════

def apply_oi_cascade_overlay(
    position: pd.Series,
    price_df: pd.DataFrame,
    oi_series: pd.Series | None,
    params: dict,
) -> pd.Series:
    """
    OI Cascade Confirmatory Overlay

    核心邏輯：
        1. 偵測清算瀑布事件：
           - OI 變化率 z-score < oi_cascade_z_threshold（OI 急降）
           - 價格變化率 z-score < oi_cascade_price_z_threshold（價格急跌）
           - 兩者同時滿足 → cascade event
        2. Cascade 觸發後保持 active 狀態 hold_bars 個 bar
        3. 在 cascade active 期間對持倉做確認性縮放：
           - 持多 + cascade active → boost（瀑布結束支持反彈做多）
           - 持空 + cascade active → reduce（瀑布結束不利空頭）
           - 空倉 → 不改變（overlay 不生成新進場）
        4. Cascade 結束後進入 cooldown（防止短間距連續觸發）

    特性：
        - 事件驅動：只在極端 OI+價格下跌時觸發（稀少但高品質）
        - 純確認性：不改變進場方向（只 scale，不 flip）
        - 與 vol_pause 互補：vol_pause 高波動退出，oi_cascade 瀑布後做多加碼
        - Time-in-market 極低（~4%），作為 overlay 不會頻繁干預

    Args:
        position: 原始持倉信號 [-1, 1]
        price_df: K 線 DataFrame（需要 close）
        oi_series: OI 數值序列（已對齊到 K 線 index），可為 None
        params: overlay 參數：
            oi_cascade_oi_lookback:       int    OI 變化率回看期（預設 24）
            oi_cascade_price_lookback:    int    價格變化率回看期（預設 8）
            oi_cascade_z_window:          int    z-score 滾動窗口（預設 720）
            oi_cascade_z_threshold:       float  OI z-score 觸發門檻（負值，預設 -1.5）
            oi_cascade_price_z_threshold: float  Price z-score 觸發門檻（負值，預設 -1.0）
            oi_cascade_boost:             float  cascade 確認做多時放大倍率（預設 1.3）
            oi_cascade_reduce:            float  cascade 反向持空時縮減倍率（預設 0.3）
            oi_cascade_hold_bars:         int    cascade 信號持續期（預設 36 bars）
            oi_cascade_cooldown_bars:     int    cascade 結束後冷卻期（預設 12 bars）
            oi_cascade_min_coverage:      float  OI 最低覆蓋率（預設 0.3）
            oi_cascade_pos_threshold:     float  持倉判定門檻（預設 0.05）

    Returns:
        修改後的持倉信號 [-1, 1]
    """
    # ── 參數解析 ──
    oi_lookback = int(params.get("oi_cascade_oi_lookback", 24))
    price_lookback = int(params.get("oi_cascade_price_lookback", 8))
    z_window = int(params.get("oi_cascade_z_window", 720))
    oi_z_threshold = float(params.get("oi_cascade_z_threshold", -1.5))
    price_z_threshold = float(params.get("oi_cascade_price_z_threshold", -1.0))
    scale_boost = float(params.get("oi_cascade_boost", 1.3))
    scale_reduce = float(params.get("oi_cascade_reduce", 0.3))
    hold_bars = int(params.get("oi_cascade_hold_bars", 36))
    cooldown_bars = int(params.get("oi_cascade_cooldown_bars", 12))
    min_coverage = float(params.get("oi_cascade_min_coverage", 0.3))
    pos_threshold = float(params.get("oi_cascade_pos_threshold", 0.05))

    n = len(position)

    # ── Guard: 無 OI 數據時直接返回 ──
    if oi_series is None or oi_series.empty:
        logger.warning("📊 OI Cascade Overlay: no OI data, skipping")
        return position

    # ── 對齊到 position index ──
    oi_aligned = oi_series.reindex(position.index).ffill()

    # ── 覆蓋率檢查 ──
    coverage = (~oi_aligned.isna()).mean()
    if coverage < min_coverage:
        logger.warning(
            f"📊 OI Cascade Overlay: OI coverage {coverage:.1%} < "
            f"{min_coverage:.0%}, skipping"
        )
        return position

    # ── 計算 z-scores ──
    oi_z = _compute_oi_change_zscore(oi_aligned, oi_lookback, z_window)
    price_z = _compute_price_change_zscore(
        price_df["close"], price_lookback, z_window,
    )

    oi_z_vals = oi_z.values
    price_z_vals = price_z.values
    pos_arr = position.values.copy().astype(float)
    result = pos_arr.copy()

    # ── Stateful bar-by-bar overlay ──
    cascade_remaining = 0   # cascade active 剩餘 bars
    cooldown_remaining = 0  # cooldown 剩餘 bars
    n_cascade_triggers = 0
    n_boosted = 0
    n_reduced = 0

    warmup = max(z_window, oi_lookback, price_lookback) + 50

    for i in range(n):
        if i < warmup:
            continue

        # ── Cooldown 中不觸發新 cascade ──
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            # cooldown 期間不做任何縮放
            continue

        # ── 檢查新 cascade 觸發（只在非 active 期間）──
        if cascade_remaining <= 0:
            oi_trigger = oi_z_vals[i] < oi_z_threshold
            price_trigger = price_z_vals[i] < price_z_threshold

            if oi_trigger and price_trigger:
                cascade_remaining = hold_bars
                n_cascade_triggers += 1

        # ── Cascade active 期間：縮放持倉 ──
        if cascade_remaining > 0:
            cascade_remaining -= 1

            # cascade 結束後進入 cooldown
            if cascade_remaining == 0:
                cooldown_remaining = cooldown_bars

            # 根據持倉方向決定 boost/reduce
            if pos_arr[i] > pos_threshold:
                # 持多 + cascade → boost（瀑布結束支持反彈）
                result[i] = min(pos_arr[i] * scale_boost, 1.0)
                n_boosted += 1
            elif pos_arr[i] < -pos_threshold:
                # 持空 + cascade → reduce（瀑布結束不利空頭）
                result[i] = max(pos_arr[i] * scale_reduce, -1.0)
                n_reduced += 1
            # 空倉 → 不改變

    # ── 統計 ──
    n_active_bars = n_boosted + n_reduced
    n_with_pos = int((np.abs(pos_arr) > pos_threshold).sum())

    logger.info(
        f"📊 OI Cascade Overlay: "
        f"cascade_triggers={n_cascade_triggers}, "
        f"boosted={n_boosted}, reduced={n_reduced}, "
        f"active_bars={n_active_bars}/{n} ({n_active_bars/n*100:.1f}%), "
        f"pos_bars={n_with_pos}, "
        f"OI_coverage={coverage:.1%}, "
        f"params(oi_z={oi_z_threshold}, price_z={price_z_threshold}, "
        f"boost={scale_boost}, reduce={scale_reduce}, "
        f"hold={hold_bars}, cooldown={cooldown_bars})"
    )

    return pd.Series(result, index=position.index)
