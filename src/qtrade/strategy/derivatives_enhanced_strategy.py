"""
衍生品增強策略 — Derivatives-Enhanced Strategies

包含三個子策略：
    1. crowding_contrarian: LSR 極端 + 價格背離 → 逆向入場
    2. cvd_divergence: 價格新高但 CVD 下降 → 做空信號（反之亦然）
    3. liq_cascade_v2: 實際清算數據增強 OI Liq Bounce 入場精度

這些策略需要 ctx.derivatives_data 中的對應數據。
如果數據不可用，策略輸出全 0（不交易）。

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
#  Strategy 1: Crowding Contrarian (LSR Extreme)
# ══════════════════════════════════════════════════════════════

@register_strategy("crowding_contrarian")
def crowding_contrarian(
    df: pd.DataFrame,
    ctx: StrategyContext,
    params: dict,
) -> pd.Series:
    """
    LSR 極端值逆向策略

    原理：
        當 Long/Short Ratio 達到極端值（過度擁擠）時，
        逆向交易。市場過度一致看多 → 做空；過度看空 → 做多。

    信號邏輯：
        - LSR z-score > threshold → 做空（多頭擁擠）
        - LSR z-score < -threshold → 做多（空頭擁擠）
        - 可選：需要價格背離確認（價格創新高但 LSR 不創新高）

    Required derivatives_data:
        - "lsr" or "top_lsr_account": Long/Short Ratio series

    Parameters:
        lsr_metric: str = "lsr"          (使用哪個 LSR 指標)
        zscore_window: int = 168          (z-score 計算窗口)
        entry_threshold: float = 2.0      (z-score 入場門檻)
        exit_threshold: float = 0.5       (z-score 出場門檻)
        require_divergence: bool = False   (是否需要價格背離確認)
        divergence_window: int = 48       (背離檢查窗口)
        vol_target: float = 0.15          (波動率目標)
        composite_ema: int = 4            (信號 EMA 平滑)
    """
    lsr_metric = params.get("lsr_metric", "lsr")
    zscore_window = params.get("zscore_window", 168)
    entry_thresh = params.get("entry_threshold", 2.0)
    exit_thresh = params.get("exit_threshold", 0.5)
    require_divergence = params.get("require_divergence", False)
    div_window = params.get("divergence_window", 48)
    vol_target = params.get("vol_target", 0.15)
    composite_ema = params.get("composite_ema", 4)

    close = df["close"]
    pos = pd.Series(0.0, index=df.index)

    # 取得 LSR 數據
    lsr = ctx.get_derivative(lsr_metric)
    if lsr is None:
        logger.warning(f"  ⚠️ Crowding Contrarian [{ctx.symbol}]: no {lsr_metric} data, skipping")
        return pos

    # 確保 index 對齊
    lsr = lsr.reindex(df.index).ffill()

    # LSR z-score
    lsr_mean = lsr.rolling(zscore_window, min_periods=zscore_window // 2).mean()
    lsr_std = lsr.rolling(zscore_window, min_periods=zscore_window // 2).std()
    lsr_std = lsr_std.replace(0, np.nan).ffill().fillna(1.0)
    lsr_zscore = (lsr - lsr_mean) / lsr_std

    # Vol scaling
    returns = close.pct_change()
    vol = returns.rolling(zscore_window).std() * np.sqrt(8760)
    vol = vol.replace(0, np.nan).ffill().fillna(vol_target)
    vol_scale = (vol_target / vol).clip(0.1, 2.0)

    # 基礎信號
    # LSR 過高 (多頭擁擠) → 做空
    signal = pd.Series(0.0, index=df.index)
    signal[lsr_zscore > entry_thresh] = -1.0  # 多頭擁擠 → short
    signal[lsr_zscore < -entry_thresh] = 1.0  # 空頭擁擠 → long
    # 出場
    signal[(lsr_zscore.abs() < exit_thresh) & (signal.shift(1) != 0)] = 0.0

    # 持倉延續（非反轉時維持方向）
    pos = signal.replace(0, np.nan).ffill().fillna(0.0)

    # 可選：價格背離確認
    if require_divergence:
        # 價格新高但 LSR 不新高 → 背離確認
        price_new_high = close == close.rolling(div_window).max()
        lsr_not_new_high = lsr < lsr.rolling(div_window).max()
        price_new_low = close == close.rolling(div_window).min()
        lsr_not_new_low = lsr > lsr.rolling(div_window).min()

        bearish_div = price_new_high & lsr_not_new_high
        bullish_div = price_new_low & lsr_not_new_low

        div_confirm = pd.Series(0.0, index=df.index)
        div_confirm[bearish_div] = -1.0
        div_confirm[bullish_div] = 1.0
        div_confirm = div_confirm.replace(0, np.nan).ffill(limit=div_window).fillna(0)

        # 需要 LSR 和背離同方向
        pos = pos * (np.sign(pos) == np.sign(div_confirm)).astype(float)

    # Vol scaling
    pos = pos * vol_scale

    # EMA 平滑
    if composite_ema > 1:
        pos = pos.ewm(span=composite_ema, min_periods=1).mean()

    pos = pos.clip(-1.0, 1.0).fillna(0.0)

    n_long = (pos > 0.1).sum()
    n_short = (pos < -0.1).sum()
    logger.info(
        f"  Crowding Contrarian [{ctx.symbol}]: "
        f"long={n_long}/{len(df)}, short={n_short}/{len(df)}"
    )

    return pos


# ══════════════════════════════════════════════════════════════
#  Strategy 2: CVD Divergence
# ══════════════════════════════════════════════════════════════

@register_strategy("cvd_divergence")
def cvd_divergence(
    df: pd.DataFrame,
    ctx: StrategyContext,
    params: dict,
) -> pd.Series:
    """
    CVD (Cumulative Volume Delta) 背離策略

    原理：
        價格創新高但 CVD 不跟隨（背離）→ 趨勢可能反轉
        CVD 反映 Taker 的實際買賣壓力，比價格更早反映方向變化

    信號邏輯：
        - 價格 N-bar 新高 + CVD 低於 M-bar 前水平 → 看空背離
        - 價格 N-bar 新低 + CVD 高於 M-bar 前水平 → 看多背離
        - 背離信號需要確認（連續 N bars 維持背離）

    Required derivatives_data:
        - "cvd": Cumulative Volume Delta series

    Parameters:
        price_lookback: int = 48       (價格新高/低檢查窗口)
        cvd_lookback: int = 48         (CVD 比較窗口)
        confirmation_bars: int = 3     (背離確認所需 bars)
        hold_period: int = 24          (信號持有期)
        vol_target: float = 0.15       (波動率目標)
        composite_ema: int = 4         (信號 EMA 平滑)
    """
    price_lookback = params.get("price_lookback", 48)
    cvd_lookback = params.get("cvd_lookback", 48)
    confirm_bars = params.get("confirmation_bars", 3)
    hold_period = params.get("hold_period", 24)
    vol_target = params.get("vol_target", 0.15)
    composite_ema = params.get("composite_ema", 4)

    close = df["close"]
    pos = pd.Series(0.0, index=df.index)

    # 取得 CVD 數據
    cvd = ctx.get_derivative("cvd")
    if cvd is None:
        logger.warning(f"  ⚠️ CVD Divergence [{ctx.symbol}]: no CVD data, skipping")
        return pos

    cvd = cvd.reindex(df.index).ffill()

    # 價格新高/低
    price_high = close.rolling(price_lookback, min_periods=price_lookback // 2).max()
    price_low = close.rolling(price_lookback, min_periods=price_lookback // 2).min()
    at_price_high = (close >= price_high * 0.999)  # 容許 0.1% 容差
    at_price_low = (close <= price_low * 1.001)

    # CVD 走勢
    cvd_lagged = cvd.shift(cvd_lookback)
    cvd_rising = cvd > cvd_lagged  # CVD 上升
    cvd_falling = cvd < cvd_lagged  # CVD 下降

    # 背離信號
    bearish_div = at_price_high & cvd_falling  # 價格高點 + CVD 下降
    bullish_div = at_price_low & cvd_rising    # 價格低點 + CVD 上升

    # 確認：連續 N bars 背離
    if confirm_bars > 1:
        bearish_confirmed = bearish_div.rolling(confirm_bars, min_periods=confirm_bars).sum() >= confirm_bars
        bullish_confirmed = bullish_div.rolling(confirm_bars, min_periods=confirm_bars).sum() >= confirm_bars
    else:
        bearish_confirmed = bearish_div
        bullish_confirmed = bullish_div

    # 生成信號（持有 hold_period bars）
    signal = pd.Series(0.0, index=df.index)
    signal[bearish_confirmed] = -1.0
    signal[bullish_confirmed] = 1.0
    # 信號持有（使用 ffill + limit）
    signal = signal.replace(0, np.nan).ffill(limit=hold_period).fillna(0)

    # Vol scaling
    returns = close.pct_change()
    vol = returns.rolling(168).std() * np.sqrt(8760)
    vol = vol.replace(0, np.nan).ffill().fillna(vol_target)
    vol_scale = (vol_target / vol).clip(0.1, 2.0)
    signal = signal * vol_scale

    # EMA 平滑
    if composite_ema > 1:
        signal = signal.ewm(span=composite_ema, min_periods=1).mean()

    pos = signal.clip(-1.0, 1.0).fillna(0.0)

    n_bearish = bearish_confirmed.sum() if hasattr(bearish_confirmed, 'sum') else 0
    n_bullish = bullish_confirmed.sum() if hasattr(bullish_confirmed, 'sum') else 0
    logger.info(
        f"  CVD Divergence [{ctx.symbol}]: "
        f"bearish_divs={n_bearish}, bullish_divs={n_bullish}"
    )

    return pos


# ══════════════════════════════════════════════════════════════
#  Strategy 3: Liquidation Cascade V2
# ══════════════════════════════════════════════════════════════

@register_strategy("liq_cascade_v2")
def liq_cascade_v2(
    df: pd.DataFrame,
    ctx: StrategyContext,
    params: dict,
) -> pd.Series:
    """
    清算瀑布增強版（OI Liq Bounce V2）

    原理：
        用實際清算數據（而非 OI proxy）偵測清算瀑布事件，
        在瀑布結束後入場做反彈。

    信號邏輯：
        1. 偵測清算瀑布（liq_cascade_z > threshold）
        2. 判斷清算方向（liq_imbalance）
           - 空頭被清算多 (imbalance > 0) → 做多（反彈）
           - 多頭被清算多 (imbalance < 0) → 做空（反彈）
        3. 等待瀑布結束（liq_cascade_z 回落）再入場
        4. 持有 hold_period bars

    Required derivatives_data:
        - "liq_cascade_z": 清算瀑布 z-score
        - "liq_imbalance": 清算不平衡 [-1, 1]

    Parameters:
        cascade_threshold: float = 2.0   (瀑布 z-score 門檻)
        cooldown_bars: int = 2           (瀑布結束後等待 bars)
        hold_period: int = 24            (信號持有期)
        imbalance_threshold: float = 0.3 (清算不平衡門檻)
        vol_target: float = 0.15         (波動率目標)
        composite_ema: int = 3           (信號 EMA 平滑)
    """
    cascade_thresh = params.get("cascade_threshold", 2.0)
    cooldown = params.get("cooldown_bars", 2)
    hold_period = params.get("hold_period", 24)
    imb_thresh = params.get("imbalance_threshold", 0.3)
    vol_target = params.get("vol_target", 0.15)
    composite_ema = params.get("composite_ema", 3)

    close = df["close"]
    pos = pd.Series(0.0, index=df.index)

    # 取得清算數據
    liq_z = ctx.get_derivative("liq_cascade_z")
    liq_imb = ctx.get_derivative("liq_imbalance")

    if liq_z is None or liq_imb is None:
        logger.warning(f"  ⚠️ Liq Cascade V2 [{ctx.symbol}]: no liquidation data, skipping")
        return pos

    liq_z = liq_z.reindex(df.index).ffill().fillna(0)
    liq_imb = liq_imb.reindex(df.index).ffill().fillna(0)

    # 偵測清算瀑布事件
    is_cascade = liq_z > cascade_thresh

    # 瀑布結束（z-score 從高位回落到 threshold 以下）
    was_cascade = is_cascade.shift(1).fillna(False)
    cascade_end = was_cascade & ~is_cascade

    # 加入 cooldown
    if cooldown > 0:
        # 瀑布結束後等 N bars
        cascade_end_shifted = cascade_end.shift(cooldown).fillna(False)
    else:
        cascade_end_shifted = cascade_end

    # 判斷反彈方向
    signal = pd.Series(0.0, index=df.index)

    # 在瀑布結束的 bar 上生成信號
    for idx in df.index[cascade_end_shifted]:
        imb_val = liq_imb.get(idx, 0)
        if abs(imb_val) >= imb_thresh:
            if imb_val > 0:
                # 空頭被清算多 → 做多反彈
                signal[idx] = 1.0
            else:
                # 多頭被清算多 → 做空反彈
                signal[idx] = -1.0

    # 信號持有
    signal = signal.replace(0, np.nan).ffill(limit=hold_period).fillna(0)

    # Vol scaling
    returns = close.pct_change()
    vol = returns.rolling(168).std() * np.sqrt(8760)
    vol = vol.replace(0, np.nan).ffill().fillna(vol_target)
    vol_scale = (vol_target / vol).clip(0.1, 2.0)
    signal = signal * vol_scale

    # EMA 平滑
    if composite_ema > 1:
        signal = signal.ewm(span=composite_ema, min_periods=1).mean()

    pos = signal.clip(-1.0, 1.0).fillna(0.0)

    n_events = cascade_end_shifted.sum()
    n_signals = (signal.abs() > 0.01).sum()
    logger.info(
        f"  Liq Cascade V2 [{ctx.symbol}]: "
        f"cascade_events={n_events}, signal_bars={n_signals}"
    )

    return pos
