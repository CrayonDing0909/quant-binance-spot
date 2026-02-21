"""
Mean Reversion 微觀結構策略（研究版 — Phase MR-1）

目標：找出可在成本後存活、且與 R2（TSMOM-based momentum）低相關的 MR 因子。
不進 production，僅供 research pipeline 使用。

三個 baseline 變體：
    MR-A: Z-score Reversion（VWAP 中心 + z-score 入出場）
    MR-B: Bollinger Reversion + Vol Filter（BB 反轉 + ADX/Vol 濾網）
    MR-C: RSI(短週期) + HTF Trend Blocker（RSI 過熱反轉 + 高時間框架趨勢阻擋）

風控共用：
    - stop_loss_atr
    - max_holding_bars（短持倉，防止趨勢市硬扛）
    - cooldown_bars（出場後冷卻）

Anti-Lookahead 保證：
    - signal_delay 和 direction clip 由 @register_strategy 框架自動處理
    - 策略函數只回傳 raw position [-1, 1]
    - 所有指標僅使用當下與過去資料（shift(1) 或 rolling 保證因果性）
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import StrategyContext
from . import register_strategy
from ..indicators import (
    calculate_atr,
    calculate_rsi,
    calculate_adx,
    calculate_bollinger_bands,
    calculate_ema,
)
from ..indicators.volume import calculate_vwap
from .exit_rules import apply_exit_rules
from .filters import (
    htf_soft_trend_filter,
    time_of_day_filter,
    _RESAMPLE_MAP,
    _resample_ohlcv,
)


# ══════════════════════════════════════════════════════════════
#  共用工具
# ══════════════════════════════════════════════════════════════

def _bars_per_year(interval: str) -> int:
    """各 interval 一年有多少 bars（365d × 24h 基準）"""
    m = {
        "1m": 525_600, "5m": 105_120, "15m": 35_040,
        "30m": 17_520, "1h": 8_760, "4h": 2_190, "1d": 365,
    }
    return m.get(interval, 8_760)


def _apply_mr_exit_rules(
    df: pd.DataFrame,
    raw_pos: pd.Series,
    params: dict,
) -> pd.Series:
    """
    共用出場規則包裝：stop_loss_atr + max_holding_bars + cooldown_bars

    回傳 (adjusted_pos, exec_prices)，並把 exec_prices 掛到
    pos.attrs["exit_exec_prices"] 讓 run_backtest 可用。
    """
    sl_atr = params.get("stop_loss_atr")
    tp_atr = params.get("take_profit_atr")
    max_hold = int(params.get("max_holding_bars", 0))
    cooldown = int(params.get("cooldown_bars", 0))
    atr_period = int(params.get("atr_period", 14))

    if sl_atr is None and tp_atr is None and max_hold == 0:
        return raw_pos

    pos, exec_prices = apply_exit_rules(
        df, raw_pos,
        stop_loss_atr=sl_atr,
        take_profit_atr=tp_atr,
        atr_period=atr_period,
        max_holding_bars=max_hold,
        cooldown_bars=cooldown,
    )
    pos.attrs["exit_exec_prices"] = exec_prices
    return pos


# ══════════════════════════════════════════════════════════════
#  MR-A: Z-score Reversion
# ══════════════════════════════════════════════════════════════

@register_strategy("mr_zscore", auto_delay=False)
def generate_mr_zscore(
    df: pd.DataFrame, ctx: StrategyContext, params: dict,
) -> pd.Series:
    """
    Z-score Mean Reversion

    中心：rolling VWAP（成交量加權均價）
    訊號：
        zscore < -z_enter → long（價格低於均值太多）
        zscore > +z_enter → short（價格高於均值太多）
        回到 |z| < z_exit → flat

    params:
        lookback:           VWAP/std 回看期（bar 數），預設 96
        z_enter:            入場 z-score 門檻，預設 2.0
        z_exit:             出場 z-score 門檻，預設 0.5
        use_vwap:           是否用 VWAP（True）或 SMA（False）作中心，預設 True
        stop_loss_atr:      ATR 止損倍數
        take_profit_atr:    ATR 止盈倍數
        max_holding_bars:   最大持倉 bar 數
        cooldown_bars:      出場冷卻 bar 數
        atr_period:         ATR 計算週期
        time_filter_hours:  封鎖的 UTC 小時列表（可選）
    """
    close = df["close"]
    lookback = int(params.get("lookback", 96))
    z_enter = float(params.get("z_enter", 2.0))
    z_exit = float(params.get("z_exit", 0.5))
    use_vwap = params.get("use_vwap", True)

    # ── 中心線 ──
    if use_vwap:
        center = calculate_vwap(df, period=lookback)
    else:
        center = close.rolling(lookback).mean()

    # ── z-score（用 close 偏離中心的標準化距離）──
    rolling_std = close.rolling(lookback).std()
    rolling_std = rolling_std.replace(0, np.nan).ffill().fillna(1.0)
    zscore = (close - center) / rolling_std

    # ── 信號生成（狀態機：防止 warmup 期亂入場）──
    n = len(df)
    raw = np.zeros(n, dtype=float)
    state = 0  # 0=flat, 1=long, -1=short

    zv = zscore.values
    for i in range(lookback, n):
        z = zv[i]
        if np.isnan(z):
            raw[i] = 0.0
            state = 0
            continue

        if state == 0:
            if z < -z_enter:
                state = 1
                raw[i] = 1.0
            elif z > z_enter:
                state = -1
                raw[i] = -1.0
            else:
                raw[i] = 0.0
        elif state == 1:  # holding long
            if z > -z_exit:
                state = 0
                raw[i] = 0.0
            else:
                raw[i] = 1.0
        elif state == -1:  # holding short
            if z < z_exit:
                state = 0
                raw[i] = 0.0
            else:
                raw[i] = -1.0

    pos = pd.Series(raw, index=df.index)

    # ── time-of-day filter（可選）──
    blocked = params.get("time_filter_hours")
    if blocked:
        pos = time_of_day_filter(df, pos, blocked_hours=blocked)

    # ── signal_delay（手動處理，因為 auto_delay=False + exit_rules）──
    signal_delay = getattr(ctx, "signal_delay", 0)
    if signal_delay > 0:
        pos = pos.shift(signal_delay).fillna(0.0)

    # ── exit rules ──
    pos = _apply_mr_exit_rules(df, pos, params)

    # ── direction clip（手動，因 auto_delay=False）──
    if not ctx.can_short:
        pos = pos.clip(lower=0.0)
    if not ctx.can_long:
        pos = pos.clip(upper=0.0)

    return pos


# ══════════════════════════════════════════════════════════════
#  MR-B: Bollinger Reversion + Vol Filter
# ══════════════════════════════════════════════════════════════

@register_strategy("mr_bollinger", auto_delay=False)
def generate_mr_bollinger(
    df: pd.DataFrame, ctx: StrategyContext, params: dict,
) -> pd.Series:
    """
    Bollinger Band Mean Reversion + Volatility Filter

    入場：
        close < lower band → long
        close > upper band → short
    出場：
        回到 middle band → flat

    波動濾網（避免趨勢爆發時硬接刀）：
        ADX > adx_block_threshold → 禁止新入場
        realized vol 在高百分位時（vol_block_pct） → 禁止新入場

    params:
        bb_period:             Bollinger 週期，預設 20
        bb_std:                標準差倍數，預設 2.0
        adx_period:            ADX 計算週期，預設 14
        adx_block_threshold:   ADX > 此值禁止入場，預設 30
        vol_block_enabled:     是否啟用 realized vol 濾網，預設 True
        vol_block_lookback:    vol 百分位回看期，預設 96
        vol_block_pct:         高 vol 百分位閾值，預設 80
        stop_loss_atr / take_profit_atr / max_holding_bars / cooldown_bars / atr_period
        time_filter_hours
    """
    close = df["close"]
    bb_period = int(params.get("bb_period", 20))
    bb_std = float(params.get("bb_std", 2.0))
    adx_period = int(params.get("adx_period", 14))
    adx_block = float(params.get("adx_block_threshold", 30))
    vol_block_on = params.get("vol_block_enabled", True)
    vol_block_lb = int(params.get("vol_block_lookback", 96))
    vol_block_pct = float(params.get("vol_block_pct", 80))

    # ── Bollinger Bands ──
    bb = calculate_bollinger_bands(close, period=bb_period, std_mult=bb_std)
    upper = bb["upper"].values
    middle = bb["middle"].values
    lower = bb["lower"].values

    # ── ADX（lagged — no lookahead）──
    adx_data = calculate_adx(df, adx_period)
    adx_vals = adx_data["ADX"].shift(1).fillna(0).values

    # ── Realized Vol percentile（lagged — no lookahead）──
    returns = close.pct_change()
    rvol = returns.rolling(vol_block_lb).std()
    rvol_pct = rvol.rolling(vol_block_lb, min_periods=max(vol_block_lb // 2, 1)).rank(pct=True) * 100
    rvol_pct = rvol_pct.shift(1).fillna(50).values  # lagged

    # ── 信號生成 ──
    n = len(df)
    raw = np.zeros(n, dtype=float)
    cv = close.values
    state = 0

    warmup = max(bb_period, adx_period, vol_block_lb) + 5
    for i in range(warmup, n):
        # 濾網：ADX 過高（趨勢太強）或 vol 過高（波動爆發）
        blocked = adx_vals[i] > adx_block
        if vol_block_on and rvol_pct[i] > vol_block_pct:
            blocked = True

        if state == 0:
            if not blocked:
                if not np.isnan(lower[i]) and cv[i] < lower[i]:
                    state = 1
                    raw[i] = 1.0
                elif not np.isnan(upper[i]) and cv[i] > upper[i]:
                    state = -1
                    raw[i] = -1.0
        elif state == 1:  # long
            if not np.isnan(middle[i]) and cv[i] >= middle[i]:
                state = 0
                raw[i] = 0.0
            else:
                raw[i] = 1.0
        elif state == -1:  # short
            if not np.isnan(middle[i]) and cv[i] <= middle[i]:
                state = 0
                raw[i] = 0.0
            else:
                raw[i] = -1.0

    pos = pd.Series(raw, index=df.index)

    # ── time filter ──
    blocked_hours = params.get("time_filter_hours")
    if blocked_hours:
        pos = time_of_day_filter(df, pos, blocked_hours=blocked_hours)

    # ── signal_delay ──
    signal_delay = getattr(ctx, "signal_delay", 0)
    if signal_delay > 0:
        pos = pos.shift(signal_delay).fillna(0.0)

    # ── exit rules ──
    pos = _apply_mr_exit_rules(df, pos, params)

    # ── direction clip ──
    if not ctx.can_short:
        pos = pos.clip(lower=0.0)
    if not ctx.can_long:
        pos = pos.clip(upper=0.0)

    return pos


# ══════════════════════════════════════════════════════════════
#  MR-C: RSI(短週期) + HTF Trend Blocker
# ══════════════════════════════════════════════════════════════

@register_strategy("mr_rsi_htf", auto_delay=False)
def generate_mr_rsi_htf(
    df: pd.DataFrame, ctx: StrategyContext, params: dict,
) -> pd.Series:
    """
    短週期 RSI Mean Reversion + 高時間框架趨勢阻擋

    入場（逆勢反轉）：
        RSI < rsi_oversold → long
        RSI > rsi_overbought → short
    出場：
        RSI 回到 rsi_exit_low ~ rsi_exit_high 中性區 → flat

    HTF Trend Blocker（軟過濾）：
        用更高 TF 的 EMA 交叉判斷大趨勢方向
        逆勢 MR 交易 → 降權（counter_weight，預設 0.3）
        順勢 MR 交易 → 全倉
        這避免在強趨勢中硬接飛刀

    params:
        rsi_period:            RSI 週期，預設 7（短）
        rsi_overbought:        超買門檻，預設 75
        rsi_oversold:          超賣門檻，預設 25
        rsi_exit_high:         多倉 RSI 出場上限，預設 55
        rsi_exit_low:          空倉 RSI 出場下限，預設 45
        htf_interval:          高 TF 週期，預設 "1h"（給 5m/15m 用）
        htf_ema_fast:          高 TF 快速 EMA，預設 20
        htf_ema_slow:          高 TF 慢速 EMA，預設 50
        htf_counter_weight:    逆勢權重，預設 0.3
        htf_align_weight:      順勢權重，預設 1.0
        htf_neutral_weight:    中性權重，預設 0.7
        stop_loss_atr / take_profit_atr / max_holding_bars / cooldown_bars / atr_period
        time_filter_hours
    """
    close = df["close"]
    rsi_period = int(params.get("rsi_period", 7))
    rsi_ob = float(params.get("rsi_overbought", 75))
    rsi_os = float(params.get("rsi_oversold", 25))
    rsi_exit_hi = float(params.get("rsi_exit_high", 55))
    rsi_exit_lo = float(params.get("rsi_exit_low", 45))

    # ── RSI（lagged by shift(1) 內建於 warmup 邏輯中）──
    rsi = calculate_rsi(close, rsi_period)

    # ── 信號生成 ──
    n = len(df)
    raw = np.zeros(n, dtype=float)
    rv = rsi.values
    state = 0

    warmup = rsi_period + 5
    for i in range(warmup, n):
        r = rv[i]
        if np.isnan(r):
            raw[i] = 0.0
            state = 0
            continue

        if state == 0:
            if r < rsi_os:
                state = 1
                raw[i] = 1.0
            elif r > rsi_ob:
                state = -1
                raw[i] = -1.0
        elif state == 1:  # long
            if r >= rsi_exit_hi:
                state = 0
                raw[i] = 0.0
            else:
                raw[i] = 1.0
        elif state == -1:  # short
            if r <= rsi_exit_lo:
                state = 0
                raw[i] = 0.0
            else:
                raw[i] = -1.0

    pos = pd.Series(raw, index=df.index)

    # ── HTF soft trend filter ──
    htf_interval = params.get("htf_interval", "1h")
    htf_ema_fast = int(params.get("htf_ema_fast", 20))
    htf_ema_slow = int(params.get("htf_ema_slow", 50))
    htf_counter = float(params.get("htf_counter_weight", 0.3))
    htf_align = float(params.get("htf_align_weight", 1.0))
    htf_neutral = float(params.get("htf_neutral_weight", 0.7))
    current_interval = ctx.interval

    pos = htf_soft_trend_filter(
        df, pos,
        htf_interval=htf_interval,
        ema_fast=htf_ema_fast,
        ema_slow=htf_ema_slow,
        current_interval=current_interval,
        align_weight=htf_align,
        counter_weight=htf_counter,
        neutral_weight=htf_neutral,
    )

    # ── time filter ──
    blocked_hours = params.get("time_filter_hours")
    if blocked_hours:
        pos = time_of_day_filter(df, pos, blocked_hours=blocked_hours)

    # ── signal_delay ──
    signal_delay = getattr(ctx, "signal_delay", 0)
    if signal_delay > 0:
        pos = pos.shift(signal_delay).fillna(0.0)

    # ── exit rules ──
    pos = _apply_mr_exit_rules(df, pos, params)

    # ── direction clip ──
    if not ctx.can_short:
        pos = pos.clip(lower=0.0)
    if not ctx.can_long:
        pos = pos.clip(upper=0.0)

    return pos
