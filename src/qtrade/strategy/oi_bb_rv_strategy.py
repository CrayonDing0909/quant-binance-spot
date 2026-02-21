"""
OI + Bollinger Band Breakout + Realized Volatility Filter (OI-BB-RV)

策略來源重現：
    市場：BTCUSDT 永續合約（15m）
    進場（多）：close 上穿布林上軌
                AND  OIMA 快線 > 慢線（OI 上升趨勢確認）
                AND  pctrank_RV 在上下界內（排除極端波動）
    進場（空）：close 下穿布林下軌
                AND  OIMA 快線 > 慢線
                AND  pctrank_RV 在上下界內
    出場：OIMA 快線下穿慢線

Anti-lookahead:
    - 所有指標只使用 bar[i] 及之前的資料
    - BB 穿越用 close[i] vs band[i]（band 由 close[:i] 算出，不含未來）
    - OIMA 用 .shift(1) 延遲 1 bar（避免使用當 bar 的 OI 數據確認交叉）
    - pctrank_RV 用 .shift(1)
    - signal_delay 由框架自動處理（trade_on=next_open → shift(1)）

OI 對齊：
    OI 原始資料為 1h，15m 策略中用 forward-fill 對齊到 15m bar。
    OIMA 快/慢線在對齊後的 OI 上計算。

Note:
    使用 auto_delay=True，讓框架自動處理 signal_delay 和 direction clip。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import StrategyContext
from . import register_strategy
from ..indicators import calculate_bollinger_bands


# ══════════════════════════════════════════════════════════════
#  核心指標計算
# ══════════════════════════════════════════════════════════════

def _compute_oima(
    oi_series: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
) -> tuple[pd.Series, pd.Series]:
    """
    OI Moving Average（快/慢線）

    Args:
        oi_series: OI 數值序列（已對齊到策略 bar index）
        fast_period: 快線 EMA 週期
        slow_period: 慢線 EMA 週期

    Returns:
        (oima_fast, oima_slow)
    """
    oima_fast = oi_series.ewm(span=fast_period, min_periods=fast_period).mean()
    oima_slow = oi_series.ewm(span=slow_period, min_periods=slow_period).mean()
    return oima_fast, oima_slow


def _compute_pctrank_rv(
    close: pd.Series,
    rv_period: int = 20,
    rank_window: int = 252,
) -> pd.Series:
    """
    Realized Volatility 的百分位排名

    Args:
        close: 收盤價序列
        rv_period: RV 計算週期（bar 數）
        rank_window: 百分位排名的回看窗口

    Returns:
        pctrank_rv: 0-100 的百分位排名
    """
    returns = close.pct_change()
    rv = returns.rolling(rv_period, min_periods=max(rv_period // 2, 2)).std()
    pctrank = rv.rolling(rank_window, min_periods=max(rank_window // 4, 10)).rank(pct=True) * 100.0
    return pctrank


def _align_oi_to_15m(
    oi_series: pd.Series | None,
    target_index: pd.DatetimeIndex,
    max_ffill_bars: int = 8,  # 1h OI → 15m means 4 bars per OI point, allow 2 hours gap
) -> pd.Series:
    """
    將 1h OI 數據 forward-fill 對齊到 15m bar index

    不使用未來值：只允許前向填充。
    """
    if oi_series is None or oi_series.empty:
        return pd.Series(np.nan, index=target_index)

    # Timezone alignment
    if target_index.tz is None and oi_series.index.tz is not None:
        oi_series = oi_series.copy()
        oi_series.index = oi_series.index.tz_localize(None)
    elif target_index.tz is not None and oi_series.index.tz is None:
        oi_series = oi_series.copy()
        oi_series.index = oi_series.index.tz_localize(target_index.tz)

    aligned = oi_series.reindex(target_index, method="ffill", limit=max_ffill_bars)
    return aligned


# ══════════════════════════════════════════════════════════════
#  V1: 嚴格重現版
# ══════════════════════════════════════════════════════════════

@register_strategy("oi_bb_rv")
def generate_oi_bb_rv(
    df: pd.DataFrame, ctx: StrategyContext, params: dict,
) -> pd.Series:
    """
    OI + Bollinger Band Breakout + RV Filter

    策略信號：
        Long:  close 上穿 BB upper  AND  OIMA fast > slow  AND  RV_pctrank in [rv_lower, rv_upper]
        Short: close 下穿 BB lower  AND  OIMA fast > slow  AND  RV_pctrank in [rv_lower, rv_upper]
        Exit:  OIMA fast 下穿 slow

    params:
        # Bollinger Bands
        bb_period:          BB 週期, 預設 20
        bb_std:             BB 標準差倍數, 預設 2.0
        # OIMA (OI Moving Average)
        oima_fast:          OI 快線 EMA 週期, 預設 12
        oima_slow:          OI 慢線 EMA 週期, 預設 26
        # RV Filter
        rv_period:          RV 計算週期, 預設 20
        rv_rank_window:     RV 排名回看窗口, 預設 252
        rv_lower:           RV 百分位下界, 預設 10
        rv_upper:           RV 百分位上界, 預設 90
        # Risk management (V2 extensions, V1 defaults are off)
        cooldown_bars:      出場後冷卻期 (0=off), 預設 0
        max_holding_bars:   最大持倉 bar 數 (0=off), 預設 0
        min_hold_bars:      最少持倉 bar 數 (0=off), 預設 0
        confirm_bars:       連續突破確認 bar 數 (1=off), 預設 1
        # OI data (injected externally)
        _oi_series:         預注入的 OI Series (由 runner 注入, 策略不直接下載)
    """
    close = df["close"]
    n = len(df)

    # ── 參數解析 ──
    bb_period = int(params.get("bb_period", 20))
    bb_std = float(params.get("bb_std", 2.0))
    oima_fast_period = int(params.get("oima_fast", 12))
    oima_slow_period = int(params.get("oima_slow", 26))
    rv_period = int(params.get("rv_period", 20))
    rv_rank_window = int(params.get("rv_rank_window", 252))
    rv_lower = float(params.get("rv_lower", 10.0))
    rv_upper = float(params.get("rv_upper", 90.0))

    # V2 risk params (defaults = off for V1)
    cooldown_bars = int(params.get("cooldown_bars", 0))
    max_holding_bars = int(params.get("max_holding_bars", 0))
    min_hold_bars = int(params.get("min_hold_bars", 0))
    confirm_bars = int(params.get("confirm_bars", 1))

    # ── 1. Bollinger Bands ──
    bb = calculate_bollinger_bands(close, period=bb_period, std_mult=bb_std)
    bb_upper = bb["upper"].values
    bb_lower = bb["lower"].values

    # BB crossover detection (close[i] vs band[i], both computed from data up to i)
    cv = close.values
    # Cross above upper: close[i-1] <= upper[i-1] and close[i] > upper[i]
    cross_above_upper = np.zeros(n, dtype=bool)
    cross_below_lower = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if not np.isnan(bb_upper[i]) and not np.isnan(bb_upper[i - 1]):
            cross_above_upper[i] = (cv[i - 1] <= bb_upper[i - 1]) and (cv[i] > bb_upper[i])
        if not np.isnan(bb_lower[i]) and not np.isnan(bb_lower[i - 1]):
            cross_below_lower[i] = (cv[i - 1] >= bb_lower[i - 1]) and (cv[i] < bb_lower[i])

    # ── 2. OIMA (OI Moving Average) ──
    oi_raw = params.get("_oi_series")
    if oi_raw is not None and isinstance(oi_raw, pd.Series) and not oi_raw.empty:
        # OI may be 1h → align to 15m
        oi_aligned = _align_oi_to_15m(oi_raw, df.index, max_ffill_bars=8)
    else:
        # No OI data → OIMA filter is always True (degrade gracefully)
        oi_aligned = pd.Series(np.nan, index=df.index)

    oima_fast, oima_slow = _compute_oima(
        oi_aligned, fast_period=oima_fast_period, slow_period=oima_slow_period
    )
    # Lag by 1 bar to avoid look-ahead (use confirmed OI state/crossover)
    oima_fast_lag1 = oima_fast.shift(1)
    oima_slow_lag1 = oima_slow.shift(1)
    oima_fast_lag2 = oima_fast.shift(2)
    oima_slow_lag2 = oima_slow.shift(2)

    # Entry filter: OIMA fast > slow (STATE at lag-1)
    oima_bull = (oima_fast_lag1 > oima_slow_lag1).fillna(False).values

    # Exit signal: OIMA fast CROSSES BELOW slow (CROSSOVER event at lag-1)
    # Cross = was above at lag-2, now below at lag-1
    oima_exit_cross = (
        (oima_fast_lag2 >= oima_slow_lag2) & (oima_fast_lag1 < oima_slow_lag1)
    ).fillna(False).values

    # If OI data is all NaN, let OIMA filter pass through (degrade to BB-RV only)
    oi_available = ~np.isnan(oi_aligned.values)
    has_oi = oi_available.any()

    # ── 3. RV Percentile Rank Filter ──
    rv_pctrank = _compute_pctrank_rv(close, rv_period=rv_period, rank_window=rv_rank_window)
    # Lag by 1 bar to avoid look-ahead
    rv_vals = rv_pctrank.shift(1).fillna(50.0).values
    rv_ok = (rv_vals >= rv_lower) & (rv_vals <= rv_upper)

    # ── 4. Signal generation (state machine) ──
    pos = np.zeros(n, dtype=float)
    state = 0        # 0=flat, 1=long, -1=short
    hold_count = 0   # bars since entry
    cooldown_remaining = 0  # cooldown counter
    confirm_count = 0       # consecutive breakout confirmation
    confirm_dir = 0         # direction being confirmed

    warmup = max(bb_period, oima_slow_period, rv_rank_window) + 5

    for i in range(warmup, n):
        # ── Cooldown check ──
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            pos[i] = 0.0
            continue

        # ── Entry confirmation logic ──
        if confirm_bars > 1 and state == 0:
            # Check if BB breakout is happening
            if cross_above_upper[i]:
                if confirm_dir == 1:
                    confirm_count += 1
                else:
                    confirm_dir = 1
                    confirm_count = 1
            elif cross_below_lower[i]:
                if confirm_dir == -1:
                    confirm_count += 1
                else:
                    confirm_dir = -1
                    confirm_count = 1
            else:
                confirm_count = 0
                confirm_dir = 0

        if state == 0:
            # ── ENTRY conditions ──
            # OIMA filter: if OI available, require oima_bull; otherwise pass
            oima_entry_ok = oima_bull[i] if has_oi else True

            if confirm_bars > 1:
                entry_confirmed = (confirm_count >= confirm_bars)
            else:
                entry_confirmed = True

            if oima_entry_ok and rv_ok[i]:
                if cross_above_upper[i] and entry_confirmed:
                    state = 1
                    hold_count = 0
                    confirm_count = 0
                    confirm_dir = 0
                elif cross_below_lower[i] and entry_confirmed:
                    state = -1
                    hold_count = 0
                    confirm_count = 0
                    confirm_dir = 0

        elif state == 1:
            hold_count += 1
            # ── EXIT conditions ──
            exit_now = False

            # Primary exit: OIMA fast CROSSES BELOW slow (crossover event)
            if has_oi and oima_exit_cross[i]:
                if min_hold_bars <= 0 or hold_count >= min_hold_bars:
                    exit_now = True

            # Max holding exit
            if max_holding_bars > 0 and hold_count >= max_holding_bars:
                exit_now = True

            if exit_now:
                state = 0
                if cooldown_bars > 0:
                    cooldown_remaining = cooldown_bars

        elif state == -1:
            hold_count += 1
            exit_now = False

            # Primary exit: OIMA fast CROSSES BELOW slow (crossover event)
            if has_oi and oima_exit_cross[i]:
                if min_hold_bars <= 0 or hold_count >= min_hold_bars:
                    exit_now = True

            if max_holding_bars > 0 and hold_count >= max_holding_bars:
                exit_now = True

            if exit_now:
                state = 0
                if cooldown_bars > 0:
                    cooldown_remaining = cooldown_bars

        pos[i] = float(state)

    return pd.Series(pos, index=df.index)
