"""
OI Liquidation Bounce 策略

Alpha 來源：
    OI 急降 + 價格急跌 → 清算瀑布結束 → 做多反彈

學術 / 實踐背景：
    - 大量清算（多頭爆倉）會造成 OI 急劇下降伴隨價格下跌
    - 清算瀑布結束後，賣壓消失 → 價格傾向反彈
    - 與趨勢動量（TSMOM）正交（相關性 ≈ 0.01），適合組合配置

信號定義：
    入場：OI_change_24h z-score < oi_z_threshold（預設 -1.5）
          AND Price_change_8h z-score < price_z_threshold（預設 -1.0）
          AND 價格在 Daily EMA50 之上（HTF 趨勢過濾，避免熊市）
    出場：固定持有 hold_bars（預設 24h）
    方向：Long-only

風控元件：
    1. HTF Trend Filter — Daily EMA50（熊市不做多）
    2. Vol Scaling — 根據波動率倒數縮放倉位
    3. Cooldown — 出場後 N bars 禁止再入場

Anti-lookahead：
    - OI z-score 使用 .shift(1) 延遲 1 bar（使用確認後的 OI 數據）
    - Price z-score 使用 .shift(1)
    - HTF trend 使用 resampled Daily 數據，ffill 到 1h（不含未來）
    - signal_delay 由 @register_strategy 框架自動處理（trade_on=next_open → shift(1)）

Note:
    使用 auto_delay=True，框架自動處理 signal_delay 和 direction clip。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import StrategyContext
from . import register_strategy
from ..indicators import calculate_ema, calculate_atr


# ══════════════════════════════════════════════════════════════
#  核心指標計算
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

    Args:
        oi_series: OI 數值序列（已對齊到 1h bar index）
        change_lookback: 計算變化率的回看期（bars），預設 24 = 24h
        z_window: 滾動 z-score 窗口（bars），預設 720 = 30 天

    Returns:
        OI 變化率 z-score 序列
    """
    if oi_series is None or oi_series.empty:
        return pd.Series(dtype=float)

    # 1. OI 變化率
    oi_lagged = oi_series.shift(change_lookback)
    change_rate = (oi_series - oi_lagged) / oi_lagged.replace(0, np.nan)

    # 2. 滾動 z-score
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

    Args:
        close: 收盤價序列
        change_lookback: 計算變化率的回看期（bars），預設 8 = 8h
        z_window: 滾動 z-score 窗口（bars），預設 720 = 30 天

    Returns:
        價格變化率 z-score 序列
    """
    # 1. 價格變化率
    close_lagged = close.shift(change_lookback)
    change_rate = (close - close_lagged) / close_lagged.replace(0, np.nan)

    # 2. 滾動 z-score
    min_periods = max(z_window // 4, 30)
    rolling_mean = change_rate.rolling(z_window, min_periods=min_periods).mean()
    rolling_std = change_rate.rolling(z_window, min_periods=min_periods).std()
    z = (change_rate - rolling_mean) / rolling_std.replace(0, np.nan)

    return z.fillna(0.0).clip(-5.0, 5.0)


def _compute_daily_ema_trend(
    df: pd.DataFrame,
    ema_period: int = 50,
) -> pd.Series:
    """
    計算 Daily EMA 趨勢過濾信號

    邏輯：
        1. 將 1h 數據重採樣為 Daily
        2. 計算 Daily EMA(ema_period)
        3. close > EMA → 上升趨勢（允許做多），反之不允許

    Args:
        df: 1h K 線 DataFrame（需含 close）
        ema_period: EMA 週期（在 Daily 上），預設 50

    Returns:
        1h 粒度的趨勢信號：1.0 = 上升趨勢，0.0 = 非上升趨勢
    """
    # 重採樣到 Daily
    daily_df = df.resample("1D").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()

    if len(daily_df) < ema_period + 5:
        # 數據不足 → 不過濾（全部允許）
        return pd.Series(1.0, index=df.index)

    # Daily EMA
    daily_ema = calculate_ema(daily_df["close"], ema_period)

    # close > EMA → 上升趨勢
    daily_trend = pd.Series(0.0, index=daily_df.index)
    daily_trend[daily_df["close"] > daily_ema] = 1.0

    # 映射回 1h（forward fill，不用未來數據）
    trend_1h = daily_trend.reindex(df.index, method="ffill").fillna(0.0)

    return trend_1h


def _vol_scale(
    close: pd.Series,
    vol_target: float = 0.15,
    vol_lookback: int = 168,
) -> pd.Series:
    """
    波動率目標縮放（倉位大小根據波動率反比調整）

    Args:
        close: 收盤價序列
        vol_target: 年化波動率目標（預設 0.15 = 15%）
        vol_lookback: 波動率計算回看期（bars）

    Returns:
        縮放因子序列，clip 到 [0.2, 1.0]
    """
    returns = close.pct_change()
    # 年化波動率（1h bar, 8760 = 365 * 24）
    vol = returns.rolling(vol_lookback, min_periods=max(vol_lookback // 4, 10)).std() * np.sqrt(8760)
    vol = vol.replace(0, np.nan).ffill().fillna(vol_target)

    scale = (vol_target / vol).clip(0.2, 1.0)
    return scale


# ══════════════════════════════════════════════════════════════
#  策略：OI Liquidation Bounce
# ══════════════════════════════════════════════════════════════

@register_strategy("oi_liq_bounce")
def generate_oi_liq_bounce(
    df: pd.DataFrame, ctx: StrategyContext, params: dict,
) -> pd.Series:
    """
    OI Liquidation Bounce 策略

    信號邏輯：
        Entry: OI_change_24h z-score < oi_z_threshold
               AND Price_change_8h z-score < price_z_threshold
               AND Daily EMA50 趨勢向上（HTF filter）
        Exit:  固定持有 hold_bars 後出場
        Direction: Long-only

    params:
        # Core signal
        oi_change_lookback:     OI 變化率回看期（bars），預設 24
        price_change_lookback:  價格變化率回看期（bars），預設 8
        z_window:               z-score 滾動窗口（bars），預設 720
        oi_z_threshold:         OI z-score 入場門檻（負值），預設 -1.5
        price_z_threshold:      Price z-score 入場門檻（負值），預設 -1.0

        # Exit
        hold_bars:              固定持有期（bars），預設 24

        # HTF trend filter
        htf_ema_period:         Daily EMA 週期，預設 50
        htf_filter_enabled:     是否啟用 HTF 過濾，預設 True

        # Vol scaling
        vol_scale_enabled:      是否啟用波動率縮放，預設 True
        vol_target:             年化波動率目標，預設 0.15
        vol_lookback:           波動率回看期（bars），預設 168

        # Cooldown
        cooldown_bars:          出場後冷卻期（bars），預設 12

        # OI data (injected externally)
        _oi_series:             預注入的 OI Series（由 runner 或 backtest script 注入）
    """
    close = df["close"]
    n = len(df)

    # ── 參數解析 ──
    oi_change_lookback = int(params.get("oi_change_lookback", 24))
    price_change_lookback = int(params.get("price_change_lookback", 8))
    z_window = int(params.get("z_window", 720))
    oi_z_threshold = float(params.get("oi_z_threshold", -1.5))
    price_z_threshold = float(params.get("price_z_threshold", -1.0))

    hold_bars = int(params.get("hold_bars", 24))

    htf_ema_period = int(params.get("htf_ema_period", 50))
    htf_filter_enabled = bool(params.get("htf_filter_enabled", True))

    vol_scale_enabled = bool(params.get("vol_scale_enabled", True))
    vol_target = float(params.get("vol_target", 0.15))
    vol_lookback = int(params.get("vol_lookback", 168))

    cooldown_bars = int(params.get("cooldown_bars", 12))

    # ── 1. OI 信號 ──
    oi_raw = params.get("_oi_series")
    if oi_raw is not None and isinstance(oi_raw, pd.Series) and not oi_raw.empty:
        # OI 可能需要對齊到 df index
        if not oi_raw.index.equals(df.index):
            # Timezone alignment
            if df.index.tz is None and oi_raw.index.tz is not None:
                oi_raw = oi_raw.copy()
                oi_raw.index = oi_raw.index.tz_localize(None)
            elif df.index.tz is not None and oi_raw.index.tz is None:
                oi_raw = oi_raw.copy()
                oi_raw.index = oi_raw.index.tz_localize(df.index.tz)
            oi_aligned = oi_raw.reindex(df.index, method="ffill", limit=2)
        else:
            oi_aligned = oi_raw
    else:
        # 無 OI 數據 → 無法產生信號，返回全 0
        return pd.Series(0.0, index=df.index)

    oi_coverage = (~oi_aligned.isna()).mean()
    if oi_coverage < 0.3:
        # OI 覆蓋率太低，返回全 0
        return pd.Series(0.0, index=df.index)

    # OI 變化率 z-score（lag 1 bar 避免 look-ahead）
    oi_z = _compute_oi_change_zscore(oi_aligned, oi_change_lookback, z_window)
    oi_z_lagged = oi_z.shift(1).fillna(0.0)

    # ── 2. Price 信號 ──
    price_z = _compute_price_change_zscore(close, price_change_lookback, z_window)
    price_z_lagged = price_z.shift(1).fillna(0.0)

    # ── 3. HTF 趨勢過濾 ──
    if htf_filter_enabled:
        htf_trend = _compute_daily_ema_trend(df, htf_ema_period)
        # lag 1 bar：使用昨日的 Daily trend 判斷
        htf_trend_lagged = htf_trend.shift(1).fillna(0.0)
    else:
        htf_trend_lagged = pd.Series(1.0, index=df.index)

    # ── 4. Vol Scaling ──
    if vol_scale_enabled:
        vol_scale = _vol_scale(close, vol_target, vol_lookback)
        # lag 1 bar
        vol_scale_lagged = vol_scale.shift(1).fillna(0.5)
    else:
        vol_scale_lagged = pd.Series(1.0, index=df.index)

    # ── 5. State Machine: 信號生成 ──
    oi_z_vals = oi_z_lagged.values
    price_z_vals = price_z_lagged.values
    htf_vals = htf_trend_lagged.values
    vol_vals = vol_scale_lagged.values

    pos = np.zeros(n, dtype=float)
    state = 0           # 0 = flat, 1 = long
    hold_count = 0      # bars since entry
    cooldown_remaining = 0  # cooldown counter

    warmup = max(z_window, oi_change_lookback, price_change_lookback) + 50

    for i in range(warmup, n):
        # ── Cooldown 檢查 ──
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            pos[i] = 0.0
            continue

        if state == 1:
            # ── 持倉中 ──
            hold_count += 1
            if hold_count >= hold_bars:
                # 到期出場
                pos[i] = 0.0
                state = 0
                hold_count = 0
                cooldown_remaining = cooldown_bars
            else:
                # 繼續持有
                pos[i] = vol_vals[i]  # vol scaling 持續作用
        else:
            # ── 空倉 → 檢查入場條件 ──
            oi_trigger = oi_z_vals[i] < oi_z_threshold
            price_trigger = price_z_vals[i] < price_z_threshold
            htf_ok = htf_vals[i] > 0.5  # 上升趨勢

            if oi_trigger and price_trigger and htf_ok:
                # 入場做多
                pos[i] = vol_vals[i]  # vol scaling 決定倉位大小
                state = 1
                hold_count = 0
            else:
                pos[i] = 0.0

    return pd.Series(pos, index=df.index)
