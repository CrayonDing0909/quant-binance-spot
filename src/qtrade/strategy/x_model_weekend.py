"""
X-Model — ERL→IRL Weekend Liquidity Sweep Strategy v8 (ICT/SMC)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

基於 ICT (Inner Circle Trader) 的 ERL (External Range Liquidity) →
IRL (Internal Range Liquidity) 交易模型。

v8 Changes (from v7):
    - **Weekly Profile Management — "Monday Scalper" → "Weekly Swing Trader"**:

      Root Cause: v7 正確捕捉了 Monday Judas Swing (accumulation),
      但 force_close_window_end=True 在 Mon/Tue 晚間強制平倉，
      錯過了 Wed/Thu 的主要 expansion move (distribution)。
      我們在抓 accumulation 但錯過 distribution。

      修正:
      1. **Breakeven Logic (move-to-BE)**:
         - 當 price 順勢移動 breakeven_trigger_pct (e.g. 0.5%)
         - SL 移至 Entry Price → 消除風險 (Risk-Free Trade)
         - 保護免受 Complex Pullback 影響

      2. **Trailing Stop**:
         - Breakeven 觸發後，若 price 繼續順勢移動
         - SL 追蹤 peak favorable price 的 trailing_stop_pct (e.g. 1.5%)
         - 鎖定利潤，但給 expansion room

      3. **Swing Holding Period**:
         - force_close_window_end default → False
         - max_hold_bars default → 96 (4 天)
         - 允許 Mon 進場的單持有到 Thu/Fri

    SMC 邏輯: Weekly Profile = Mon accumulation → Wed/Thu distribution
    策略捕捉完整 weekly range expansion，而非只抓 Mon scalp。

v7: HTF Directional Bias (Daily EMA 20)
v6: Rejection Quality Filter + Volume Filter
v5: Static Weekend Range (Sat+Sun)
v4: Trend Filter (1H SMA), Kill Zone, Relaxed Confirmation

Signal Flow (v8, Bull Market Weekly Swing):
    1. HTF Bias: Daily Close(D-1) > EMA 20 → Bullish → ONLY LONG
    2. Key Levels: Weekend Range Low
    3. Active Window + Kill Zone (Mon London Open)
    4. Judas Swing: Price sweeps Weekend Low, closes above (SFP)
    5. Rejection + Volume filters
    6. Confirmation: MSS or FVG
    7. Entry: Next bar open (signal_delay=1)
    8. ★ Breakeven: Price +0.5% → SL moves to Entry (risk-free)
    9. ★ Trailing: After BE, trail SL at 1.5% from peak
   10. Hold: Up to 96 bars (4 days, capture Wed/Thu expansion)
   11. Exit: TP (STDV), Trailing SL, or max_hold_bars

Anti-Lookahead 保證:
    - 所有 level/filter 使用已完成的歷史 bar
    - Breakeven/Trailing: 使用 current bar 的 H/L (已完成)
    - Entry at bar[i+1] open (signal_delay=1)

Timeframe: H1 (主要) + Daily (HTF bias) + 15m (確認)
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .base import StrategyContext
from . import register_strategy


# ══════════════════════════════════════════════════════════════
#  Helper: Previous Day High/Low (PDH/PDL) — UTC+8
# ══════════════════════════════════════════════════════════════

def compute_daily_levels_utc8(
    index: pd.DatetimeIndex,
    high: np.ndarray,
    low: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    計算 PDH (Previous Day High) 與 PDL (Previous Day Low)，以 UTC+8 日曆計。

    一個 UTC+8 「日」= 00:00 UTC+8 (= 16:00 UTC 前日) ~ 23:59 UTC+8。
    PDH/PDL 來自 **已完成** 的前一個 UTC+8 日，不含當日任何資料。

    Anti-Lookahead:
        - 前日在當日開始時已全部完成 → 不存在未來資訊
        - 使用 shifted UTC+8 時間取 .date → 正確映射日曆日

    Returns:
        pdh: Previous Day High (每根 bar 一個值)
        pdl: Previous Day Low  (每根 bar 一個值)
    """
    n = len(index)
    pdh = np.full(n, np.nan)
    pdl = np.full(n, np.nan)

    # 將 UTC index 平移 +8h → UTC+8 calendar 時間
    utc8_times = index + pd.Timedelta(hours=8)
    utc8_dates = utc8_times.date  # python date objects in UTC+8

    # 逐 bar 建立每日 High/Low
    daily_high: dict = defaultdict(lambda: -np.inf)
    daily_low: dict = defaultdict(lambda: np.inf)

    for i in range(n):
        d = utc8_dates[i]
        daily_high[d] = max(daily_high[d], high[i])
        daily_low[d] = min(daily_low[d], low[i])

    # 建立 prev-day mapping
    sorted_dates = sorted(daily_high.keys())
    prev_high: dict = {}
    prev_low: dict = {}
    for j in range(1, len(sorted_dates)):
        curr = sorted_dates[j]
        prev = sorted_dates[j - 1]
        prev_high[curr] = daily_high[prev]
        prev_low[curr] = daily_low[prev]

    # 指派到每根 bar
    for i in range(n):
        d = utc8_dates[i]
        if d in prev_high:
            pdh[i] = prev_high[d]
            pdl[i] = prev_low[d]

    return pdh, pdl


# ══════════════════════════════════════════════════════════════
#  Helper: Weekend Range (Sat+Sun) — UTC+8
# ══════════════════════════════════════════════════════════════

def compute_weekend_levels_utc8(
    index: pd.DatetimeIndex,
    high: np.ndarray,
    low: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    計算週末 (Sat+Sun) 的靜態高低點，供週一/週二使用。
    """
    n = len(index)
    wh = np.full(n, np.nan)
    wl = np.full(n, np.nan)

    utc8_times = index + pd.Timedelta(hours=8)
    
    # Simple loop
    curr_wh = -np.inf
    curr_wl = np.inf
    last_wh = np.nan
    last_wl = np.nan
    
    # To detect week change: keep track of week number
    last_week = -1

    for i in range(n):
        t = utc8_times[i]
        iso_yr, iso_wk, iso_d = t.isocalendar() # d: 1=Mon, 7=Sun
        
        # New week starts on Monday (d=1)
        # If we just finished a week, finalize weekend stats
        if iso_wk != last_week:
            # Only if we had data from prev week
            if curr_wh > -np.inf:
                last_wh = curr_wh
                last_wl = curr_wl
            
            # Reset for new week
            curr_wh = -np.inf
            curr_wl = np.inf
            last_week = iso_wk
            
        # Update if Sat(6) or Sun(7)
        if iso_d >= 6:
            curr_wh = max(curr_wh, high[i])
            curr_wl = min(curr_wl, low[i])
            
        # Assign if Mon(1) or Tue(2)
        if iso_d <= 2 and not np.isnan(last_wh):
            wh[i] = last_wh
            wl[i] = last_wl
            
    return wh, wl



# ══════════════════════════════════════════════════════════════
#  Helper: HTF Directional Bias (Daily EMA) — UTC+8
# ══════════════════════════════════════════════════════════════

def compute_daily_ema_bias_utc8(
    index: pd.DatetimeIndex,
    close: np.ndarray,
    ema_period: int = 20,
) -> np.ndarray:
    """
    Compute HTF directional bias using Daily EMA (UTC+8 calendar).

    SMC Core Principle: "Trade the Judas Swing in the direction of the HTF Trend."
        - Bull Market (close > EMA): Only buy the dip (sweep Weekend Low → Long)
        - Bear Market (close < EMA): Only sell the rally (sweep Weekend High → Short)

    Logic:
        1. Resample 1H close → Daily close (last close of each UTC+8 day)
        2. Compute EMA(ema_period) on daily closes
        3. For each 1H bar on day D, use day D-1's close vs D-1's EMA
           to determine bias (no look-ahead)

    Anti-Lookahead:
        - Daily close is from the COMPLETED previous day (D-1)
        - EMA(D-1) uses only closes through D-1
        - Bias for day D is fully determined before D opens

    Args:
        index:      1H bar timestamps (UTC)
        close:      1H close prices
        ema_period: Daily EMA period (default 20 ≈ 1 trading month)

    Returns:
        bias: ndarray of shape (n,), values in {-1, 0, +1}
              +1 = Bullish (only long), -1 = Bearish (only short), 0 = neutral
    """
    n = len(index)
    bias = np.zeros(n, dtype=np.int8)

    if n == 0 or ema_period <= 0:
        return bias

    # Convert to UTC+8 dates
    utc8_times = index + pd.Timedelta(hours=8)
    utc8_dates = utc8_times.date

    # Build daily close (last 1H close of each UTC+8 day)
    daily_close: dict = {}
    for i in range(n):
        daily_close[utc8_dates[i]] = close[i]

    sorted_dates = sorted(daily_close.keys())
    if len(sorted_dates) < 2:
        return bias

    daily_closes = np.array([daily_close[d] for d in sorted_dates],
                            dtype=np.float64)

    # Compute EMA on daily closes
    ema = np.empty_like(daily_closes)
    alpha = 2.0 / (ema_period + 1)
    ema[0] = daily_closes[0]
    for j in range(1, len(daily_closes)):
        ema[j] = alpha * daily_closes[j] + (1 - alpha) * ema[j - 1]

    # Determine bias: day D uses D-1's close vs D-1's EMA (shifted by 1)
    day_bias: dict = {}
    for j in range(1, len(sorted_dates)):
        d = sorted_dates[j]
        prev_close = daily_closes[j - 1]
        prev_ema = ema[j - 1]
        if prev_close > prev_ema:
            day_bias[d] = 1    # Bullish → only long
        elif prev_close < prev_ema:
            day_bias[d] = -1   # Bearish → only short
        # else: 0 (neutral, allow both) — left as default

    # Map back to 1H bars
    for i in range(n):
        d = utc8_dates[i]
        if d in day_bias:
            bias[i] = day_bias[d]

    return bias


# ══════════════════════════════════════════════════════════════
#  Helper: Fair Value Gap (FVG) Detection
# ══════════════════════════════════════════════════════════════

def detect_fvg(
    high: np.ndarray,
    low: np.ndarray,
    min_gap_pct: float = 0.0,
    ref_price: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    偵測 Fair Value Gap (失衡區) — 3 根 K 棒模式。

    Bullish FVG at bar[i]:  high[i-2] < low[i]
        candle(i-1) 為向上衝擊棒，gap = [high[i-2], low[i]]
    Bearish FVG at bar[i]:  low[i-2] > high[i]
        candle(i-1) 為向下衝擊棒，gap = [high[i], low[i-2]]

    Returns:
        bull_fvg, bear_fvg, fvg_top, fvg_bot
    """
    n = len(high)
    bull_fvg = np.zeros(n, dtype=bool)
    bear_fvg = np.zeros(n, dtype=bool)
    fvg_top = np.full(n, np.nan)
    fvg_bot = np.full(n, np.nan)

    if n < 3:
        return bull_fvg, bear_fvg, fvg_top, fvg_bot

    bull_fvg[2:] = high[:-2] < low[2:]
    bear_fvg[2:] = low[:-2] > high[2:]

    bull_idx = np.where(bull_fvg)[0]
    if len(bull_idx) > 0:
        fvg_top[bull_idx] = low[bull_idx]
        fvg_bot[bull_idx] = high[bull_idx - 2]

    bear_idx = np.where(bear_fvg)[0]
    if len(bear_idx) > 0:
        fvg_top[bear_idx] = low[bear_idx - 2]
        fvg_bot[bear_idx] = high[bear_idx]

    if min_gap_pct > 0 and ref_price is not None:
        gap_size = np.abs(fvg_top - fvg_bot)
        gap_pct = np.where(ref_price > 0, gap_size / ref_price, 0)
        too_small = gap_pct < min_gap_pct
        bull_fvg[too_small] = False
        bear_fvg[too_small] = False
        fvg_top[too_small] = np.nan
        fvg_bot[too_small] = np.nan

    return bull_fvg, bear_fvg, fvg_top, fvg_bot


# ══════════════════════════════════════════════════════════════
#  Helper: Active Window (UTC+8)
# ══════════════════════════════════════════════════════════════

def is_x_model_window_utc8(index: pd.DatetimeIndex) -> np.ndarray:
    """
    判斷每根 bar 是否在 X-Model 活躍窗口內。

    v3 窗口 (兩個 ERL→IRL 周期):
        Cycle 1: Sat 00:00 ~ Sun 24:00 (UTC+8)  → sweep Fri PDH/PDL
        Cycle 2: Sun 00:00 ~ Mon 24:00 (UTC+8)  → sweep Sat/Sun PDH/PDL
        Union:   Sat 00:00 ~ Mon 24:00 (UTC+8)
               = Fri 16:00 ~ Mon 16:00 (UTC)

    邏輯: 週末盤整 + 週一全天操縱 = 假突破高發時段。

    Returns:
        bool ndarray
    """
    utc8_times = index + pd.Timedelta(hours=8)
    weekday = np.asarray(utc8_times.weekday)   # Mon=0 ... Sun=6

    saturday = weekday == 5    # Sat 00:00 ~ 24:00 UTC+8
    sunday = weekday == 6      # Sun 00:00 ~ 24:00 UTC+8
    monday = weekday == 0      # Mon 00:00 ~ 24:00 UTC+8

    return saturday | sunday | monday


# ══════════════════════════════════════════════════════════════
#  Helper: Swing Point Detection (No Look-Ahead)
# ══════════════════════════════════════════════════════════════

def find_recent_swing_low(
    low: np.ndarray, end_idx: int, lookback: int = 10,
) -> float:
    """
    找到 bar[end_idx] 之前最近的已確認 swing low。

    Swing low at bar[k]: low[k] <= low[k-1] AND low[k] <= low[k+1]
    Confirmed at bar[k+1]。在 end_idx 時，bar[k+1] 已完成 → 無未來偏差。

    搜索方向: 從 end_idx-2 向前搜索，返回第一個找到的 swing low。
    Fallback: lookback 窗口內的最低點。
    """
    start = max(1, end_idx - lookback)
    for k in range(end_idx - 2, start - 1, -1):
        if k < 1 or k + 1 >= end_idx:
            continue
        if low[k] <= low[k - 1] and low[k] <= low[k + 1]:
            return float(low[k])
    # Fallback
    fb_start = max(0, end_idx - lookback)
    if fb_start < end_idx:
        return float(np.min(low[fb_start:end_idx]))
    return float(low[end_idx]) if end_idx < len(low) else np.nan


def find_recent_swing_high(
    high: np.ndarray, end_idx: int, lookback: int = 10,
) -> float:
    """
    找到 bar[end_idx] 之前最近的已確認 swing high。

    Swing high at bar[k]: high[k] >= high[k-1] AND high[k] >= high[k+1]
    Confirmed at bar[k+1]。
    """
    start = max(1, end_idx - lookback)
    for k in range(end_idx - 2, start - 1, -1):
        if k < 1 or k + 1 >= end_idx:
            continue
        if high[k] >= high[k - 1] and high[k] >= high[k + 1]:
            return float(high[k])
    fb_start = max(0, end_idx - lookback)
    if fb_start < end_idx:
        return float(np.max(high[fb_start:end_idx]))
    return float(high[end_idx]) if end_idx < len(high) else np.nan


# ══════════════════════════════════════════════════════════════
#  Helper: Manipulation Leg & STDV Target
# ══════════════════════════════════════════════════════════════

def compute_manipulation_leg(
    high: np.ndarray,
    low: np.ndarray,
    sweep_bar: int,
    sweep_type: int,        # -1 = bearish sweep (above PDH), +1 = bullish sweep (below PDL)
    sweep_extreme: float,
    lookback: int = 12,
) -> float:
    """
    計算操縱段 (Manipulation Leg) 長度。

    操縱段 = [sweep 前的反向 swing extreme] → [sweep extreme]

    Bearish sweep (高於 PDH → 反轉做空):
        pre_swing_low = min(low) in lookback bars before sweep
        leg = sweep_high - pre_swing_low

    Bullish sweep (低於 PDL → 反轉做多):
        pre_swing_high = max(high) in lookback bars before sweep
        leg = pre_swing_high - sweep_low

    Returns:
        leg length (always >= 0), 0 if insufficient lookback
    """
    start = max(0, sweep_bar - lookback)
    end = sweep_bar  # exclusive — 不含 sweep bar 本身

    if start >= end:
        return 0.0

    if sweep_type == -1:  # bearish: upward manipulation
        pre_swing = float(np.min(low[start:end]))
        leg = sweep_extreme - pre_swing
    else:                 # bullish: downward manipulation
        pre_swing = float(np.max(high[start:end]))
        leg = pre_swing - sweep_extreme

    return max(0.0, leg)


# ══════════════════════════════════════════════════════════════
#  Helper: 15m Multi-Timeframe Confirmation
# ══════════════════════════════════════════════════════════════

def check_15m_mss_fvg(
    df_15m: pd.DataFrame,
    sweep_h1_time: pd.Timestamp,
    current_h1_time: pd.Timestamp,
    sweep_type: int,
) -> bool:
    """
    檢查 15m 級別是否出現 MSS + FVG 確認。

    以空頭為例 (sweep_type=-1, PDH sweep 後):
        1. 取 sweep 1H bar 到 current 1H bar 結束的 15m K 棒
        2. 前 4 根 15m (= sweep 1H bar 內部) 定義 "結構"
        3. MSS: 後續 15m close 跌破結構低點 (structure shift)
        4. FVG: 後續 15m 出現 bearish FVG (displacement)
        5. 兩者皆滿足 → 確認

    Anti-Lookahead:
        只使用已完成的 15m bars (current_h1_time 的 1H bar 已收盤)。

    Args:
        df_15m: 15 分鐘 OHLCV DataFrame (UTC index)
        sweep_h1_time: sweep 1H bar 的時間戳
        current_h1_time: 當前 1H bar 的時間戳
        sweep_type: -1 (bearish/空) 或 +1 (bullish/多)

    Returns:
        True if 15m MSS + FVG both confirmed
    """
    # Window: from sweep_h1_time to end of current_h1_bar
    end_time = current_h1_time + pd.Timedelta(hours=1)

    # Timezone alignment
    idx_15m = df_15m.index
    if idx_15m.tz is not None:
        if sweep_h1_time.tz is None:
            sweep_h1_time = sweep_h1_time.tz_localize("UTC")
        if end_time.tz is None:
            end_time = end_time.tz_localize("UTC")
    else:
        if sweep_h1_time.tz is not None:
            sweep_h1_time = sweep_h1_time.tz_localize(None)
        if end_time.tz is not None:
            end_time = end_time.tz_localize(None)

    mask = (idx_15m >= sweep_h1_time) & (idx_15m < end_time)
    sub = df_15m.loc[mask]

    n = len(sub)
    if n < 5:  # 至少 5 根 15m bar (1h structure + 1 bar after)
        return False

    h = sub["high"].values
    l = sub["low"].values
    c = sub["close"].values

    # Structure: 前 4 根 15m bars (= sweep 1H bar 的內部結構)
    n_struct = min(4, n - 1)

    # FVG on all 15m bars
    bull_fvg_15m, bear_fvg_15m, _, _ = detect_fvg(h, l)

    if sweep_type == -1:  # bearish: 掃 PDH 後看跌
        struct_low = float(np.min(l[:n_struct]))
        has_mss = bool(np.any(c[n_struct:] < struct_low))
        has_fvg = bool(np.any(bear_fvg_15m[n_struct:]))
    else:  # bullish: 掃 PDL 後看漲
        struct_high = float(np.max(h[:n_struct]))
        has_mss = bool(np.any(c[n_struct:] > struct_high))
        has_fvg = bool(np.any(bull_fvg_15m[n_struct:]))

    return has_mss and has_fvg


# ══════════════════════════════════════════════════════════════
#  Main Strategy
# ══════════════════════════════════════════════════════════════

@register_strategy("x_model_weekend", auto_delay=False)
def generate_x_model_weekend(
    df: pd.DataFrame,
    ctx: StrategyContext,
    params: dict,
) -> pd.Series:
    """
    X-Model v8: ERL→IRL Weekend Liquidity Sweep (ICT/SMC)

    Uses auto_delay=False for manual control of:
        - signal_delay (pending entry mechanism)
        - direction clip (ctx.can_long / ctx.can_short)
        - SL/TP trigger price execution (exec_prices)

    params:
      Core:
        confirm_window          確認等待 bar 數, 預設 4
        require_fvg             是否要求 1H FVG 確認, 預設 True
        require_mss             是否要求 1H MSS 確認, 預設 True
        require_15m             是否要求 15m MSS+FVG 確認, 預設 True
        _df_15m                 15m DataFrame (由外部注入), 預設 None
        sl_buffer_pct           SL 超出 sweep extreme 的緩衝%, 預設 0.001
        tp_mode                 TP 模式: "stdv" | "opposing" | "min_stdv_opp"
        stdv_mult               STDV 倍數, 預設 2.0
        swing_lookback          找 manipulation leg 的回看 bar 數, 預設 12
        mss_lookback            找 MSS swing point 的回看 bar 數, 預設 10
        max_hold_bars           最大持倉 bar 數 (time-stop), 預設 96 (v8: 4 days)
        force_close_window_end  窗口結束時強制平倉, 預設 False (v8 swing)
        cooldown_bars           出場後冷卻 bar 數, 預設 2
        min_range_pct           PDH-PDL 太小時跳過 (佔 close%), 預設 0.002
      v4 Filters:
        trend_filter_period     1H MA 週期 (0=off), 預設 0 (legacy)
        kill_zone_hours         (start, end) UTC+8 小時, 預設 None
        relax_confirmation      True=MSS OR FVG, 預設 False
      v5 Weekend Range:
        use_weekend_levels      True=使用 Sat+Sun 靜態高低點, 預設 False
      v6 Rejection Quality Filter:
        rejection_filter        True=啟用 candle morphology 過濾, 預設 False
        min_wick_ratio          上/下影線佔 candle range 的最低比例, 預設 0.5
        max_body_position       close 在 candle range 中的最高位置, 預設 0.4
        min_volume_factor       掃蕩 bar volume / avg volume, 預設 0.0 (0=off)
      v7 HTF Directional Bias:
        htf_trend_period        Daily EMA 週期 (0=off), 預設 0
                                20 = EMA(20) ≈ 1 trading month
                                Bull: ONLY LONG, Bear: ONLY SHORT
      v8 Weekly Profile Management:
        breakeven_trigger_pct   觸發 Move-to-BE 的 % (0=off), 預設 0.0
                                e.g. 0.005 = price 順勢移動 0.5% → SL→Entry
        trailing_stop_pct       BE 觸發後的 trailing %, 預設 0.0 (0=off)
                                e.g. 0.015 = 從 peak 回撤 1.5% 出場
                                Only active AFTER breakeven is triggered

    Returns:
        pd.Series: position [-1, 0, 1]
                   attrs["exit_exec_prices"] = SL/TP trigger prices
    """
    # ── Parameters ──
    confirm_window = int(params.get("confirm_window", params.get("fvg_window", 4)))
    require_fvg = bool(params.get("require_fvg", True))
    require_mss = bool(params.get("require_mss", True))
    require_15m_flag = bool(params.get("require_15m", True))
    sl_buffer_pct = float(params.get("sl_buffer_pct", 0.001))
    tp_mode = str(params.get("tp_mode", "stdv"))
    stdv_mult = float(params.get("stdv_mult", 2.0))
    swing_lookback = int(params.get("swing_lookback", 12))
    mss_lookback = int(params.get("mss_lookback", 10))
    max_hold_bars = int(params.get("max_hold_bars", 96))       # v8: 4 days (swing)
    force_close_end = bool(params.get("force_close_window_end", False))  # v8: no force close
    cooldown_bars = int(params.get("cooldown_bars", 2))
    min_range_pct = float(params.get("min_range_pct", 0.002))
    signal_delay = getattr(ctx, "signal_delay", 1)

    # v4 Filters
    trend_filter_period = int(params.get("trend_filter_period", 0))  # 0 to disable
    kill_zone_hours = params.get("kill_zone_hours", None)  # tuple (start, end) in UTC+8, e.g. (14, 18)
    relax_confirmation = bool(params.get("relax_confirmation", False)) # if True, MSS OR FVG (instead of AND)
    use_weekend_levels = bool(params.get("use_weekend_levels", False)) # v5: Use Sat+Sun levels for Mon sweep

    # v6 Rejection Quality Filter
    rejection_filter = bool(params.get("rejection_filter", False))
    min_wick_ratio = float(params.get("min_wick_ratio", 0.5))
    max_body_position = float(params.get("max_body_position", 0.4))
    min_volume_factor = float(params.get("min_volume_factor", 0.0))  # 0 = disabled

    # v7 HTF Directional Bias (Daily EMA)
    htf_trend_period = int(params.get("htf_trend_period", 0))  # 0 = disabled

    # v8 Weekly Profile Management (Breakeven + Trailing)
    breakeven_trigger_pct = float(params.get("breakeven_trigger_pct", 0.0))  # 0 = disabled
    trailing_stop_pct = float(params.get("trailing_stop_pct", 0.0))          # 0 = disabled

    # 15m multi-timeframe data (optional, injected by research script)
    df_15m = params.get("_df_15m")
    require_15m = require_15m_flag and df_15m is not None

    if confirm_window <= 0:
        require_fvg = False
        require_mss = False

    # ── Precompute arrays ──
    idx = df.index
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    open_ = df["open"].values
    n = len(df)

    pdh, pdl = compute_daily_levels_utc8(idx, high, low)
    if use_weekend_levels:
        pdh, pdl = compute_weekend_levels_utc8(idx, high, low)

    bull_fvg, bear_fvg, _, _ = detect_fvg(high, low)
    is_active = is_x_model_window_utc8(idx)

    # ── v6: Precompute Rejection Quality (vectorized) ──
    candle_range = high - low
    safe_range = np.where(candle_range > 0, candle_range, 1.0)  # avoid /0
    is_doji = candle_range <= 0  # doji = no rejection possible

    # Bearish rejection metrics (for SHORT signals — Shooting Star / Pinbar)
    #   wick_ratio: how much of the candle is upper wick (high→close)
    #   body_pos:   where close sits in range (0=bottom, 1=top)
    bear_wick_ratio = (high - close) / safe_range
    bear_body_pos = (close - low) / safe_range
    valid_bear_reject = ((bear_wick_ratio > min_wick_ratio) |
                         (bear_body_pos < max_body_position))
    valid_bear_reject[is_doji] = False

    # Bullish rejection metrics (for LONG signals — Hammer)
    #   wick_ratio: how much of the candle is lower wick (close→low)
    #   body_pos:   distance of close from top (0=top, 1=bottom)
    bull_wick_ratio = (close - low) / safe_range
    bull_body_pos = (high - close) / safe_range
    valid_bull_reject = ((bull_wick_ratio > min_wick_ratio) |
                         (bull_body_pos < max_body_position))
    valid_bull_reject[is_doji] = False

    # ── v6: Precompute Volume Filter (Stopping Volume) ──
    vol_ok = np.ones(n, dtype=bool)
    if min_volume_factor > 0 and "volume" in df.columns:
        volume = df["volume"].values.astype(np.float64)
        avg_vol = pd.Series(volume).rolling(20, min_periods=1).mean().values
        vol_ok = volume > (min_volume_factor * avg_vol)

    # ── v7: HTF Directional Bias (Daily EMA) ──
    htf_bias = np.zeros(n, dtype=np.int8)  # 0=neutral, +1=bull, -1=bear
    if htf_trend_period > 0:
        htf_bias = compute_daily_ema_bias_utc8(idx, close, htf_trend_period)

    # v4 Trend Filter (legacy, independent of v7 HTF bias)
    trend_ma = np.zeros(n, dtype=np.float64)
    if trend_filter_period > 0:
        ma_s = pd.Series(close).rolling(trend_filter_period, min_periods=1).mean()
        trend_ma = ma_s.values
    
    # v5: Day Filter + Kill Zone
    utc8_weekday = (idx + pd.Timedelta(hours=8)).weekday.values  # 0=Mon ... 6=Sun
    if use_weekend_levels:
        # v5 核心: 覆蓋 is_active → 只在 Mon+Tue 交易 (weekend levels 是靜態的)
        # Sat/Sun 用於累積 Weekend Range, 不進場
        is_active = (utc8_weekday == 0) | (utc8_weekday == 1)

    # Kill Zone (UTC+8 Hour filter, optionally combined with day filter)
    in_kill_zone = np.ones(n, dtype=bool)
    if kill_zone_hours:
        kz_start, kz_end = kill_zone_hours
        utc8_hours = (idx + pd.Timedelta(hours=8)).hour.values

        if kz_start <= kz_end:
            time_mask = (utc8_hours >= kz_start) & (utc8_hours < kz_end)
        else:  # Over midnight (e.g. 22 ~ 04)
            time_mask = (utc8_hours >= kz_start) | (utc8_hours < kz_end)

        in_kill_zone = time_mask

    # ── State machine ──
    pos = np.zeros(n, dtype=np.float64)
    exec_prices = np.full(n, np.nan, dtype=np.float64)

    IDLE = 0
    AWAIT_BEAR = 1
    AWAIT_BULL = 2

    sweep_state = IDLE
    sweep_bar = -1
    sweep_extreme = 0.0
    sweep_manip_leg = 0.0
    mss_level = 0.0
    h1_mss_confirmed = False
    h1_fvg_confirmed = False

    holding = 0
    entry_price = 0.0
    entry_bar = -999
    sl_price = 0.0
    tp_price = 0.0
    cooldown_until = -1

    # v8: Breakeven & Trailing state
    be_triggered = False       # has SL been moved to breakeven?
    peak_favorable = 0.0       # best price since entry (high for long, low for short)

    # Pending entry (for signal_delay)
    pending = False
    pending_bar = -1
    pending_dir = 0
    pending_sl = 0.0
    pending_manip_leg = 0.0
    pending_opp_level = 0.0  # opposing PDH/PDL for IRL target

    for i in range(2, n):  # start at 2 for FVG lookback

        # ═══════════════════════════════════════════
        #  1. Execute pending entry
        # ═══════════════════════════════════════════
        if pending and i == pending_bar:
            # Cancel if outside window and force_close enabled
            if force_close_end and not is_active[i]:
                pending = False
            else:
                entry_price = open_[i]
                entry_bar = i
                holding = pending_dir
                sl_price = pending_sl

                # ── Compute TP ──
                sl_dist = abs(entry_price - sl_price)
                leg = pending_manip_leg if pending_manip_leg > 0 else sl_dist

                # STDV target
                if holding == 1:
                    stdv_tp = entry_price + stdv_mult * leg
                else:
                    stdv_tp = entry_price - stdv_mult * leg

                opp_tp = pending_opp_level

                if tp_mode == "stdv":
                    tp_price = stdv_tp
                elif tp_mode == "opposing":
                    tp_price = opp_tp
                elif tp_mode == "min_stdv_opp":
                    # 取較近的目標 (更保守，避免回吐)
                    if holding == 1:
                        tp_price = min(stdv_tp, opp_tp) if opp_tp > entry_price else stdv_tp
                    else:
                        tp_price = max(stdv_tp, opp_tp) if opp_tp < entry_price else stdv_tp
                else:
                    tp_price = stdv_tp

                # 防止 TP 方向錯誤 (fallback)
                if holding == 1 and tp_price <= entry_price:
                    tp_price = entry_price + sl_dist * 2.0
                elif holding == -1 and tp_price >= entry_price:
                    tp_price = entry_price - sl_dist * 2.0

                # v8: Initialize breakeven/trailing state
                be_triggered = False
                peak_favorable = entry_price

                pos[i] = float(holding)
                pending = False
                continue  # 入場 bar 不檢查 SL/TP

        # Cancel stale pending
        if pending and i > pending_bar:
            pending = False

        # ═══════════════════════════════════════════
        #  2. Manage existing position
        # ═══════════════════════════════════════════
        if holding != 0:
            bars_held = i - entry_bar

            # ── Force close at window end ──
            if force_close_end and not is_active[i] and i > 0 and is_active[i - 1]:
                pos[i] = 0.0
                holding = 0
                sweep_state = IDLE
                cooldown_until = i + cooldown_bars
                continue

            # ── v8: Breakeven + Trailing Stop logic ──
            # Update peak favorable price BEFORE checking exits
            if holding == 1:
                peak_favorable = max(peak_favorable, high[i])
            else:
                peak_favorable = min(peak_favorable, low[i])

            # Move-to-Breakeven: once triggered, SL = entry_price
            if not be_triggered and breakeven_trigger_pct > 0:
                if holding == 1:
                    be_level = entry_price * (1.0 + breakeven_trigger_pct)
                    if high[i] >= be_level:
                        be_triggered = True
                        sl_price = max(sl_price, entry_price)
                else:  # holding == -1
                    be_level = entry_price * (1.0 - breakeven_trigger_pct)
                    if low[i] <= be_level:
                        be_triggered = True
                        sl_price = min(sl_price, entry_price)

            # Trailing Stop: only ratchet SL in favorable direction
            if be_triggered and trailing_stop_pct > 0:
                if holding == 1:
                    trail_sl = peak_favorable * (1.0 - trailing_stop_pct)
                    sl_price = max(sl_price, trail_sl)
                else:  # holding == -1
                    trail_sl = peak_favorable * (1.0 + trailing_stop_pct)
                    sl_price = min(sl_price, trail_sl)

            if holding == 1:  # Long
                if high[i] >= tp_price:
                    pos[i] = 0.0
                    exec_prices[i] = tp_price
                    holding = 0
                    sweep_state = IDLE
                    cooldown_until = i + cooldown_bars
                    continue
                if low[i] <= sl_price:
                    pos[i] = 0.0
                    exec_prices[i] = sl_price
                    holding = 0
                    sweep_state = IDLE
                    cooldown_until = i + cooldown_bars
                    continue
                if max_hold_bars > 0 and bars_held >= max_hold_bars:
                    pos[i] = 0.0
                    holding = 0
                    sweep_state = IDLE
                    cooldown_until = i + cooldown_bars
                    continue
                pos[i] = 1.0

            elif holding == -1:  # Short
                if low[i] <= tp_price:
                    pos[i] = 0.0
                    exec_prices[i] = tp_price
                    holding = 0
                    sweep_state = IDLE
                    cooldown_until = i + cooldown_bars
                    continue
                if high[i] >= sl_price:
                    pos[i] = 0.0
                    exec_prices[i] = sl_price
                    holding = 0
                    sweep_state = IDLE
                    cooldown_until = i + cooldown_bars
                    continue
                if max_hold_bars > 0 and bars_held >= max_hold_bars:
                    pos[i] = 0.0
                    holding = 0
                    sweep_state = IDLE
                    cooldown_until = i + cooldown_bars
                    continue
                pos[i] = -1.0

            continue  # 有持倉不檢查新入場

        # ═══════════════════════════════════════════
        #  3. Flat — check for new setups
        # ═══════════════════════════════════════════

        if i < cooldown_until:
            continue

        if not is_active[i]:
            sweep_state = IDLE
            continue

        if np.isnan(pdh[i]) or np.isnan(pdl[i]):
            continue

        # 日範圍太小 → 無明確流動性方向，跳過
        day_range = pdh[i] - pdl[i]
        if close[i] > 0 and (day_range / close[i]) < min_range_pct:
            continue

        # ── Sweep detection (IDLE → AWAIT) ──
        if sweep_state == IDLE:

            # Kill Zone check (v4): only detect sweeps inside specified hours
            if kill_zone_hours and not in_kill_zone[i]:
                continue

            # ── v7: HTF Directional Bias (Daily EMA) ──
            # Bull Market → ONLY LONG (sweep Low = Judas Swing down)
            # Bear Market → ONLY SHORT (sweep High = Judas Swing up)
            allow_short = True
            allow_long = True

            if htf_trend_period > 0:
                if htf_bias[i] == 1:      # Bullish: price > Daily EMA
                    allow_short = False   # Weekend High is a target, not resistance
                elif htf_bias[i] == -1:   # Bearish: price < Daily EMA
                    allow_long = False    # Weekend Low is a target, not support

            # v4 Trend Filter (legacy, additional constraint if enabled)
            if trend_filter_period > 0 and close[i] > trend_ma[i]:
                allow_short = False
            if trend_filter_period > 0 and close[i] < trend_ma[i]:
                allow_long = False

            # PDH sweep (bearish): high > PDH AND close < PDH = SFP
            bearish_sfp = (high[i] > pdh[i] and close[i] < pdh[i]
                           and allow_short)
            # v6: Rejection Quality — filter out full bullish candles
            if rejection_filter and bearish_sfp:
                bearish_sfp = valid_bear_reject[i] and vol_ok[i]

            if bearish_sfp:
                _extreme = high[i]
                _leg = compute_manipulation_leg(
                    high, low, i, -1, _extreme, swing_lookback
                )
                _mss_lvl = find_recent_swing_low(low, i, mss_lookback)

                # No confirmation required → immediate pending
                if not require_fvg and not require_mss and not require_15m:
                    if ctx.can_short and i + signal_delay < n:
                        pending = True
                        pending_bar = i + signal_delay
                        pending_dir = -1
                        pending_sl = _extreme * (1.0 + sl_buffer_pct)
                        pending_manip_leg = _leg
                        pending_opp_level = pdl[i]
                else:
                    sweep_state = AWAIT_BEAR
                    sweep_bar = i
                    sweep_extreme = _extreme
                    sweep_manip_leg = _leg
                    mss_level = _mss_lvl
                    h1_mss_confirmed = False
                    h1_fvg_confirmed = False

            # PDL sweep (bullish): low < PDL AND close > PDL = SFP
            bullish_sfp = (low[i] < pdl[i] and close[i] > pdl[i]
                           and allow_long)
            # v6: Rejection Quality — filter out full bearish candles
            if rejection_filter and bullish_sfp:
                bullish_sfp = valid_bull_reject[i] and vol_ok[i]

            if not bearish_sfp and bullish_sfp:
                _extreme = low[i]
                _leg = compute_manipulation_leg(
                    high, low, i, 1, _extreme, swing_lookback
                )
                _mss_lvl = find_recent_swing_high(high, i, mss_lookback)

                if not require_fvg and not require_mss and not require_15m:
                    if ctx.can_long and i + signal_delay < n:
                        pending = True
                        pending_bar = i + signal_delay
                        pending_dir = 1
                        pending_sl = _extreme * (1.0 - sl_buffer_pct)
                        pending_manip_leg = _leg
                        pending_opp_level = pdh[i]
                else:
                    sweep_state = AWAIT_BULL
                    sweep_bar = i
                    sweep_extreme = _extreme
                    sweep_manip_leg = _leg
                    mss_level = _mss_lvl
                    h1_mss_confirmed = False
                    h1_fvg_confirmed = False

        # ── Await bearish confirmation ──
        elif sweep_state == AWAIT_BEAR:
            # Re-sweep (wider extreme) — must also pass rejection filter
            re_bear = high[i] > pdh[i] and close[i] < pdh[i]
            if rejection_filter and re_bear:
                re_bear = valid_bear_reject[i] and vol_ok[i]
            if re_bear:
                sweep_bar = i
                sweep_extreme = max(sweep_extreme, high[i])
                sweep_manip_leg = compute_manipulation_leg(
                    high, low, i, -1, sweep_extreme, swing_lookback
                )
                mss_level = find_recent_swing_low(low, i, mss_lookback)
                h1_mss_confirmed = False  # reset MSS (level changed)
            elif i - sweep_bar > confirm_window:
                sweep_state = IDLE
                continue

            # Check 1H MSS: close breaks below recent swing low
            if not h1_mss_confirmed and close[i] < mss_level:
                h1_mss_confirmed = True

            # Check 1H FVG: bearish FVG
            if not h1_fvg_confirmed and bear_fvg[i]:
                h1_fvg_confirmed = True

            # Evaluate all conditions
            h1_ok = True
            if relax_confirmation:
                # v4: MSS OR FVG (one of them is enough)
                h1_ok = h1_mss_confirmed or h1_fvg_confirmed
            else:
                if require_mss and not h1_mss_confirmed:
                    h1_ok = False
                if require_fvg and not h1_fvg_confirmed:
                    h1_ok = False
            
            if h1_ok:
                m15_ok = True
                if require_15m:
                    m15_ok = check_15m_mss_fvg(
                        df_15m, idx[sweep_bar], idx[i], -1
                    )

                if m15_ok and ctx.can_short and i + signal_delay < n:
                    pending = True
                    pending_bar = i + signal_delay
                    pending_dir = -1
                    pending_sl = sweep_extreme * (1.0 + sl_buffer_pct)
                    pending_manip_leg = sweep_manip_leg
                    pending_opp_level = pdl[i]
                    sweep_state = IDLE

        # ── Await bullish confirmation ──
        elif sweep_state == AWAIT_BULL:
            # Re-sweep (wider extreme) — must also pass rejection filter
            re_bull = low[i] < pdl[i] and close[i] > pdl[i]
            if rejection_filter and re_bull:
                re_bull = valid_bull_reject[i] and vol_ok[i]
            if re_bull:
                sweep_bar = i
                sweep_extreme = min(sweep_extreme, low[i])
                sweep_manip_leg = compute_manipulation_leg(
                    high, low, i, 1, sweep_extreme, swing_lookback
                )
                mss_level = find_recent_swing_high(high, i, mss_lookback)
                h1_mss_confirmed = False
            elif i - sweep_bar > confirm_window:
                sweep_state = IDLE
                continue

            # Check 1H MSS: close breaks above recent swing high
            if not h1_mss_confirmed and close[i] > mss_level:
                h1_mss_confirmed = True

            # Check 1H FVG: bullish FVG
            if not h1_fvg_confirmed and bull_fvg[i]:
                h1_fvg_confirmed = True

            # Evaluate all conditions
            h1_ok = True
            if relax_confirmation:
                # v4: MSS OR FVG
                h1_ok = h1_mss_confirmed or h1_fvg_confirmed
            else:
                if require_mss and not h1_mss_confirmed:
                    h1_ok = False
                if require_fvg and not h1_fvg_confirmed:
                    h1_ok = False

            if h1_ok:
                m15_ok = True
                if require_15m:
                    m15_ok = check_15m_mss_fvg(
                        df_15m, idx[sweep_bar], idx[i], 1
                    )

                if m15_ok and ctx.can_long and i + signal_delay < n:
                    pending = True
                    pending_bar = i + signal_delay
                    pending_dir = 1
                    pending_sl = sweep_extreme * (1.0 - sl_buffer_pct)
                    pending_manip_leg = sweep_manip_leg
                    pending_opp_level = pdh[i]
                    sweep_state = IDLE

    # ── Build result ──
    result = pd.Series(pos, index=df.index, name="position")
    exec_series = pd.Series(exec_prices, index=df.index)
    result.attrs["exit_exec_prices"] = exec_series
    return result
