"""
OI + Volatility Exit Overlay

純風控層：只做「減倉 / 平倉 / 暫停新進場」，不改變策略主進場邏輯。

提供三種 overlay 模式：

1. apply_oi_vol_exit_overlay:  原版 OI+Vol 出場（reduce_pct / flatten）
2. apply_vol_pause_overlay:    Vol-only entry pause（⭐ Phase A 主力）
3. apply_full_oi_vol_overlay:  便捷函數（自動計算信號 + 套用）

Phase A 重點：vol_pause — 當波動率 spike 時「暫停所有進場 + 維持冷卻期」

Anti-lookahead 保證：
    - 所有信號只用 [0, i] 的資料計算
    - 不用未來 bar 的 OI / 價格 / 波動率
    - 與 trade_on=next_open 配合，overlay 的決定在 bar i 結尾做出，
      bar i+1 開盤執行

Metrics:
    - flip_count: 倉位方向翻轉次數（long↔short / long↔flat / flat↔long 等）
    - 比 VBT 的 total_trades 更能反映實際交易頻率
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# Metrics: flip_count（倉位翻轉計數）
# ══════════════════════════════════════════════════════════════

def compute_flip_count(pos: pd.Series, threshold: float = 0.01) -> int:
    """
    計算倉位翻轉次數

    定義：倉位方向在相鄰 bar 發生變化的次數。
    方向以 sign(pos) 定義：+1（多）、-1（空）、0（空倉）。
    只有方向改變才算一次 flip（倉位大小微調不算）。

    Args:
        pos: 倉位序列
        threshold: 低於此絕對值視為 0（避免浮點噪聲）

    Returns:
        int: flip 次數
    """
    arr = pos.values.copy().astype(float)
    # 低於 threshold 視為 flat
    signs = np.where(np.abs(arr) < threshold, 0, np.sign(arr))
    # 計算方向改變次數
    diffs = np.diff(signs)
    flips = int(np.count_nonzero(diffs))
    return flips


# ══════════════════════════════════════════════════════════════
# OI 信號計算
# ══════════════════════════════════════════════════════════════

def compute_oi_signals(
    oi_series: pd.Series,
    lookback: int = 24,
    z_window: int = 168,
) -> pd.DataFrame:
    """
    計算 Open Interest 衍生信號

    Args:
        oi_series: OI 數值序列（已對齊到 K 線 index）
        lookback: 計算 OI 變化率的回看期（bars）
        z_window: 計算 z-score 的滾動窗口（bars）

    Returns:
        DataFrame with columns:
            - oi_change_rate: (OI - OI_lagged) / OI_lagged
            - oi_zscore: (OI - rolling_mean) / rolling_std
            - oi_trend: OI 短期趨勢 (+1 / 0 / -1)
            - dOI: OI 差分（bar-to-bar 變化）
    """
    if oi_series is None or oi_series.empty:
        return pd.DataFrame({
            "oi_change_rate": pd.Series(dtype=float),
            "oi_zscore": pd.Series(dtype=float),
            "oi_trend": pd.Series(dtype=float),
            "dOI": pd.Series(dtype=float),
        })

    result = pd.DataFrame(index=oi_series.index)

    # 1. OI 變化率（lookback period）
    oi_lagged = oi_series.shift(lookback)
    result["oi_change_rate"] = (oi_series - oi_lagged) / oi_lagged.replace(0, np.nan)

    # 2. OI z-score（滾動標準化）
    rolling_mean = oi_series.rolling(z_window, min_periods=max(z_window // 4, 1)).mean()
    rolling_std = oi_series.rolling(z_window, min_periods=max(z_window // 4, 1)).std()
    result["oi_zscore"] = (oi_series - rolling_mean) / rolling_std.replace(0, np.nan)

    # 3. OI 趨勢（短期方向）
    oi_sma = oi_series.rolling(lookback, min_periods=max(lookback // 2, 1)).mean()
    oi_sma_prev = oi_sma.shift(1)
    result["oi_trend"] = np.sign(oi_sma - oi_sma_prev).fillna(0.0)

    # 4. OI bar-to-bar 差分
    result["dOI"] = oi_series.diff()

    return result


# ══════════════════════════════════════════════════════════════
# 波動率狀態計算
# ══════════════════════════════════════════════════════════════

def compute_vol_state(
    df: pd.DataFrame,
    atr_period: int = 14,
    z_window: int = 168,
) -> pd.Series:
    """
    計算波動率 z-score（衡量當前波動率是否異常偏高）

    Args:
        df: K 線 DataFrame（需要 high, low, close）
        atr_period: ATR 計算週期
        z_window: z-score 滾動窗口

    Returns:
        Series: vol_z — 波動率 z-score
    """
    from ...indicators.atr import calculate_atr

    atr = calculate_atr(df, atr_period)
    close = df["close"]
    atr_ratio = atr / close

    rolling_mean = atr_ratio.rolling(z_window, min_periods=max(z_window // 4, 1)).mean()
    rolling_std = atr_ratio.rolling(z_window, min_periods=max(z_window // 4, 1)).std()

    vol_z = (atr_ratio - rolling_mean) / rolling_std.replace(0, np.nan)

    return vol_z


# ══════════════════════════════════════════════════════════════
# ⭐ Phase A 主力：Vol-only Entry Pause Overlay
# ══════════════════════════════════════════════════════════════

def apply_vol_pause_overlay(
    position: pd.Series,
    price_df: pd.DataFrame,
    params: dict,
) -> pd.Series:
    """
    波動率暫停 overlay（Phase A 主力）

    核心邏輯：
        當 vol_z > vol_spike_z 時，進入「暫停期」：
        1. 整段 cooldown 期間強制 position = 0（完全退出市場）
        2. 這對 **連續信號策略**（TSMOM/breakout）至關重要：
           - 這類策略的 position 每 bar 微調 → VBT 計算出大量交易
           - 唯有 cooldown 期間「歸零」才能真正減少 VBT trade_count
        3. Spike bar 本身也強制平倉

    為什麼能降低 trade_count：
        - cooldown 期間 position 固定為 0 → VBT 不產生交易
        - 每次 spike 消除 cooldown_bars 個 bar 的微調交易
        - 等效於 「高波動期間不交易」

    Anti-lookahead：
        - vol_z[i] 只用 [0, i] 的 ATR 和 close 計算
        - 結果 position[i] 的改動在 bar[i+1] 開盤執行

    Args:
        position: 原始持倉信號 [-1, 1]
        price_df: K 線 DataFrame
        params: overlay 參數：
            vol_spike_z:          float  vol zscore 觸發閾值（預設 2.0）
            overlay_cooldown_bars: int   暫停期長度（預設 24）
            atr_period:           int    ATR 計算週期（預設 14）
            vol_z_window:         int    vol z-score 滾動窗口（預設 168）
            force_flat_on_spike:  bool   spike bar 是否強制平倉（預設 True）
            pause_new_entries:    bool   暫停期是否攔截全部信號（預設 True）

    Returns:
        修改後的持倉信號
    """
    # ── 解析參數 ──
    vol_spike_z = params.get("vol_spike_z", 2.0)
    cooldown_bars = params.get("overlay_cooldown_bars", 24)
    atr_period = params.get("atr_period", 14)
    vol_z_window = params.get("vol_z_window", 168)
    # force_flat_on_spike 預設 True（整段 cooldown 都歸零）
    force_flat_on_spike = params.get("force_flat_on_spike", True)
    pause_new_entries = params.get("pause_new_entries", True)

    n = len(position)
    pos_arr = position.values.copy().astype(float)
    result = pos_arr.copy()

    # ── 計算 vol_z ──
    vol_z = compute_vol_state(price_df, atr_period=atr_period, z_window=vol_z_window)
    vz = vol_z.values

    # ── 統計 ──
    n_spike_trigger = 0
    n_bars_zeroed = 0
    cooldown_remaining = 0

    # ── Bar-by-bar overlay ──
    for i in range(n):
        # 檢查是否觸發 vol spike（spike 可以在 cooldown 中 re-trigger 延長）
        if not np.isnan(vz[i]) and vz[i] > vol_spike_z:
            if cooldown_remaining <= 0:
                n_spike_trigger += 1
            # Re-trigger / extend cooldown
            cooldown_remaining = cooldown_bars

        # cooldown 中：強制 position = 0
        if cooldown_remaining > 0:
            if abs(pos_arr[i]) >= 0.001:
                result[i] = 0.0
                n_bars_zeroed += 1
            cooldown_remaining -= 1

    # ── 確保 overlay 只做限制性操作 ──
    # (position 從非零到零是限制性的；已符合要求)
    for i in range(n):
        if pos_arr[i] == 0:
            result[i] = 0.0
        elif pos_arr[i] > 0:
            result[i] = min(result[i], pos_arr[i])
            result[i] = max(result[i], 0.0)
        elif pos_arr[i] < 0:
            result[i] = max(result[i], pos_arr[i])
            result[i] = min(result[i], 0.0)

    # ── 日誌 ──
    pct_zeroed = n_bars_zeroed / n * 100 if n > 0 else 0
    logger.info(
        f"📊 Vol Pause Overlay: "
        f"spikes={n_spike_trigger}, bars_zeroed={n_bars_zeroed}/{n} "
        f"({pct_zeroed:.1f}%), cooldown={cooldown_bars}"
    )

    return pd.Series(result, index=position.index)


# ══════════════════════════════════════════════════════════════
# 原版 OI + Vol 出場 Overlay（保留不動）
# ══════════════════════════════════════════════════════════════

def apply_oi_vol_exit_overlay(
    position: pd.Series,
    price_df: pd.DataFrame,
    oi_signals: pd.DataFrame | None,
    vol_z: pd.Series | None,
    params: dict,
) -> pd.Series:
    """
    OI + Vol 出場覆蓋層（只減倉，不開新倉）

    核心規則：
        1. OI Extreme Reversal → reduce_pct 降倉
        2. Vol Spike + Counter-Trend → 平倉
        3. Cooldown 防抖

    Anti-lookahead 保證：
        - bar[i] 的 overlay 決策只用 [0, i] 的資料
        - 結果 position[i] 的改動在 bar[i+1] 開盤執行
    """
    # ── 解析參數 ──
    oi_extreme_z = params.get("oi_extreme_z", 2.0)
    oi_reversal_window = params.get("oi_reversal_window", 6)
    reduce_pct = params.get("reduce_pct", 0.5)
    vol_spike_z = params.get("vol_spike_z", 2.5)
    cooldown_bars = params.get("overlay_cooldown_bars", 12)
    trend_lookback = params.get("trend_lookback", 20)

    n = len(position)
    pos_arr = position.values.copy().astype(float)
    result = pos_arr.copy()

    # ── 預計算趨勢方向（SMA 方向）──
    close = price_df["close"].values
    trend_dir = np.zeros(n, dtype=float)
    for i in range(trend_lookback, n):
        sma = np.mean(close[i - trend_lookback + 1: i + 1])
        if close[i] > sma:
            trend_dir[i] = 1.0
        elif close[i] < sma:
            trend_dir[i] = -1.0

    # ── 準備 OI 信號陣列 ──
    has_oi = (
        oi_signals is not None
        and not oi_signals.empty
        and len(oi_signals) == n
    )
    if has_oi:
        oi_z = oi_signals["oi_zscore"].values
        dOI = oi_signals["dOI"].values
    else:
        oi_z = np.full(n, np.nan)
        dOI = np.full(n, np.nan)

    # ── 準備 Vol 信號陣列 ──
    has_vol = vol_z is not None and len(vol_z) == n
    if has_vol:
        vz = vol_z.values
    else:
        vz = np.full(n, np.nan)

    # ── 統計計數 ──
    n_oi_reduce = 0
    n_vol_flatten = 0
    cooldown_remaining = 0

    # ── Bar-by-bar overlay 邏輯 ──
    for i in range(n):
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            continue

        if pos_arr[i] == 0:
            continue

        triggered = False

        # ── Rule 1: OI Extreme Reversal ──
        if has_oi and not np.isnan(oi_z[i]):
            window_start = max(0, i - oi_reversal_window + 1)
            oi_z_window = oi_z[window_start: i + 1]
            was_extreme = np.nanmax(oi_z_window) > oi_extreme_z if len(oi_z_window) > 0 else False
            oi_turning_down = not np.isnan(dOI[i]) and dOI[i] < 0

            if was_extreme and oi_turning_down:
                new_size = result[i] * (1.0 - reduce_pct)
                if pos_arr[i] > 0:
                    new_size = max(0.0, new_size)
                else:
                    new_size = min(0.0, new_size)
                result[i] = new_size
                triggered = True
                n_oi_reduce += 1

        # ── Rule 2: Vol Spike + Counter-Trend ──
        if has_vol and not np.isnan(vz[i]) and not triggered:
            if vz[i] > vol_spike_z:
                is_counter_trend = False
                if pos_arr[i] > 0 and trend_dir[i] < 0:
                    is_counter_trend = True
                elif pos_arr[i] < 0 and trend_dir[i] > 0:
                    is_counter_trend = True

                if is_counter_trend:
                    result[i] = 0.0
                    triggered = True
                    n_vol_flatten += 1

        if triggered:
            cooldown_remaining = cooldown_bars

    # ── 確保只減倉 ──
    for i in range(n):
        if pos_arr[i] == 0:
            result[i] = 0.0
        elif pos_arr[i] > 0:
            result[i] = min(result[i], pos_arr[i])
            result[i] = max(result[i], 0.0)
        elif pos_arr[i] < 0:
            result[i] = max(result[i], pos_arr[i])
            result[i] = min(result[i], 0.0)

    total_triggers = n_oi_reduce + n_vol_flatten
    if total_triggers > 0:
        logger.info(
            f"📊 OI/Vol Overlay: "
            f"OI reduce={n_oi_reduce}, Vol flatten={n_vol_flatten}, "
            f"total triggers={total_triggers}/{n} bars "
            f"({total_triggers/n*100:.2f}%)"
        )
    else:
        logger.info("📊 OI/Vol Overlay: 0 triggers")

    return pd.Series(result, index=position.index)


# ══════════════════════════════════════════════════════════════
# 便捷函數
# ══════════════════════════════════════════════════════════════

def apply_full_oi_vol_overlay(
    position: pd.Series,
    price_df: pd.DataFrame,
    oi_series: pd.Series | None,
    params: dict,
) -> pd.Series:
    """
    完整 OI/Vol Overlay 流程（計算信號 + 套用）

    便捷函數，合併 compute_oi_signals + compute_vol_state + apply_oi_vol_exit_overlay。
    """
    # 計算 OI 信號
    oi_signals = None
    if oi_series is not None and not oi_series.empty:
        oi_signals = compute_oi_signals(
            oi_series,
            lookback=params.get("oi_lookback", 24),
            z_window=params.get("oi_z_window", 168),
        )
        if len(oi_signals) != len(position):
            oi_signals = oi_signals.reindex(position.index)

    # 計算波動率狀態
    vol_z = compute_vol_state(
        price_df,
        atr_period=params.get("atr_period", 14),
        z_window=params.get("vol_z_window", 168),
    )

    return apply_oi_vol_exit_overlay(
        position=position,
        price_df=price_df,
        oi_signals=oi_signals,
        vol_z=vol_z,
        params=params,
    )


def _apply_single_overlay(
    position: pd.Series,
    price_df: pd.DataFrame,
    oi_series: pd.Series | None,
    params: dict,
    mode: str,
) -> pd.Series:
    """
    套用單一 overlay 模式

    Modes:
        "vol_pause"           → Vol spike entry pause
        "oi_vol"              → OI + Vol 出場
        "oi_only"             → OI only（vol disabled）
        "vol_only"            → Vol only（等同 vol_pause）
        "lsr_confirmatory"    → LSR 散戶多空比擇時縮放
        "oi_cascade"          → OI 清算瀑布擇時增強（結構面確認）

    Args:
        position: 持倉信號
        price_df: K 線 DataFrame
        oi_series: OI 數值序列（可為 None）
        params: overlay 參數（各 overlay 讀取自己的子集）
        mode: 單一 overlay 模式名稱

    Returns:
        修改後的持倉信號
    """
    if mode == "vol_pause" or mode == "vol_only":
        return apply_vol_pause_overlay(
            position=position,
            price_df=price_df,
            params=params,
        )
    elif mode == "oi_only":
        oi_only_params = {**params, "vol_spike_z": 999.0}
        return apply_full_oi_vol_overlay(
            position=position,
            price_df=price_df,
            oi_series=oi_series,
            params=oi_only_params,
        )
    elif mode == "oi_vol":
        return apply_full_oi_vol_overlay(
            position=position,
            price_df=price_df,
            oi_series=oi_series,
            params=params,
        )
    elif mode == "lsr_confirmatory":
        from .lsr_confirmatory_overlay import apply_lsr_confirmatory_overlay
        lsr_series = params.get("_lsr_series")
        return apply_lsr_confirmatory_overlay(
            position=position,
            price_df=price_df,
            lsr_series=lsr_series,
            params=params,
        )
    elif mode == "oi_cascade":
        from .oi_cascade_overlay import apply_oi_cascade_overlay
        return apply_oi_cascade_overlay(
            position=position,
            price_df=price_df,
            oi_series=oi_series,
            params=params,
        )
    else:
        raise ValueError(f"Unknown overlay mode: {mode}")


def apply_overlay_by_mode(
    position: pd.Series,
    price_df: pd.DataFrame,
    oi_series: pd.Series | None,
    params: dict,
    mode: str = "vol_pause",
) -> pd.Series:
    """
    根據 mode 選擇對應的 overlay 函數（支援 '+' 連鎖複合模式）

    單一模式：
        "vol_pause"           → Vol spike entry pause（Phase A）
        "oi_vol"              → OI + Vol 出場（Phase C）
        "oi_only"             → OI only（vol disabled）
        "vol_only"            → Vol only（等同 vol_pause）
        "lsr_confirmatory"    → LSR 散戶多空比擇時縮放
        "oi_cascade"          → OI 清算瀑布擇時增強（結構面確認）

    複合模式（用 '+' 連接，依序套用）：
        "vol_pause+lsr_confirmatory"  → 先 vol_pause 再 LSR 縮放
        "oi_vol+lsr_confirmatory"     → 先 OI+Vol 再 LSR 縮放
        "oi_vol+lsr_confirmatory+oi_cascade" → 先 OI+Vol 再 LSR 再 OI cascade

    Args:
        position: 原始持倉信號
        price_df: K 線 DataFrame
        oi_series: OI 數值序列（可為 None）
        params: overlay 參數（各 overlay 讀取自己的子集，LSR 以 lsr_ 前綴區分）
        mode: overlay 模式（單一或 '+' 連鎖）

    Returns:
        修改後的持倉信號
    """
    # 支援 '+' 連鎖：依序套用
    if "+" in mode:
        sub_modes = [m.strip() for m in mode.split("+") if m.strip()]
        pos = position
        for sub_mode in sub_modes:
            pos = _apply_single_overlay(
                position=pos,
                price_df=price_df,
                oi_series=oi_series,
                params=params,
                mode=sub_mode,
            )
            logger.info(f"📊 Compound overlay step: {sub_mode} done")
        return pos

    # 單一模式
    return _apply_single_overlay(
        position=position,
        price_df=price_df,
        oi_series=oi_series,
        params=params,
        mode=mode,
    )
