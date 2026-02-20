"""
Breakout + Volatility Expansion 策略

學術背景：
    - Donchian (1960): Channel Breakout（商品期貨經典）
    - Brock, Lakonishok & LeBaron (1992): Trading Range Breakout
    - Keltner (1960): ATR Channel

加密幣適用性：
    - 加密市場 regime 切換明顯：低波段 → 突破 → 趨勢
    - 假突破頻繁 → 必須搭配 volatility expansion 確認
    - 與 TSMOM（連續動量）正交：Breakout 捕捉 regime 切換起點

策略邏輯：
    1. Donchian Channel Breakout（N-bar 高/低點突破）
    2. Volatility Expansion 確認：ATR 必須處於上升態勢
       - ATR_fast > ATR_slow（短期波動超過長期均值 → 正在擴張）
       - 或 ATR percentile rank > threshold
    3. Fake Breakout Filter：
       - 突破後若在 K 根 bar 內回到通道內 → 失效，平倉
       - 避免 whipsaw
    4. Min Hold / Cooldown：
       - 入場後最少持倉 N bars（避免微小突破立即被噪音反轉出場）
       - SL/cooldown 後暫停入場

策略變體：
    1. breakout_vol         — Donchian + ATR expansion（基礎版）
    2. breakout_vol_atr     — ATR channel breakout + expansion（替代版）

Note:
    使用 auto_delay=False，因為需要在 exit_rules 之前手動 shift。
    框架自動處理 direction clip。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import StrategyContext
from . import register_strategy
from ..indicators import calculate_atr
from .exit_rules import apply_exit_rules


# ══════════════════════════════════════════════════════════════
#  核心指標計算
# ══════════════════════════════════════════════════════════════

def _donchian_channel(
    high: pd.Series,
    low: pd.Series,
    period: int = 96,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Donchian Channel（N 期最高/最低）

    Args:
        high:   最高價序列
        low:    最低價序列
        period: 回看期

    Returns:
        (upper, lower, mid)
    """
    upper = high.rolling(period, min_periods=period).max()
    lower = low.rolling(period, min_periods=period).min()
    mid = (upper + lower) / 2.0
    return upper, lower, mid


def _atr_channel(
    close: pd.Series,
    atr: pd.Series,
    period: int = 96,
    multiplier: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    ATR Channel（EMA ± mult × ATR）

    Args:
        close:      收盤價序列
        atr:        ATR 序列
        period:     EMA 回看期
        multiplier: ATR 倍數

    Returns:
        (upper, lower, mid)
    """
    mid = close.ewm(span=period, adjust=False).mean()
    upper = mid + multiplier * atr
    lower = mid - multiplier * atr
    return upper, lower, mid


def _volatility_expanding(
    atr: pd.Series,
    fast_period: int = 14,
    slow_period: int = 50,
    expansion_ratio: float = 1.0,
) -> pd.Series:
    """
    波動率擴張偵測

    邏輯：ATR_fast_ma > ATR_slow_ma × expansion_ratio
    → True 表示波動率正在放大（breakout 更可靠）

    Args:
        atr:              ATR 序列
        fast_period:      快速均值週期
        slow_period:      慢速均值週期
        expansion_ratio:  擴張倍數門檻

    Returns:
        bool Series（True = 波動率擴張中）
    """
    atr_fast = atr.rolling(fast_period, min_periods=fast_period).mean()
    atr_slow = atr.rolling(slow_period, min_periods=slow_period).mean()
    return atr_fast > (atr_slow * expansion_ratio)


# ══════════════════════════════════════════════════════════════
#  信號生成（stateful loop — 含 fake breakout filter）
# ══════════════════════════════════════════════════════════════

def _generate_breakout_signals(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    upper: np.ndarray,
    lower: np.ndarray,
    mid: np.ndarray,
    vol_expanding: np.ndarray,
    # params
    fake_breakout_bars: int = 3,
    min_hold_bars: int = 6,
    trail_exit_mid: bool = True,
) -> np.ndarray:
    """
    Stateful breakout 信號生成

    邏輯：
      Long entry:  close > upper（前一根） AND vol_expanding
      Short entry: close < lower（前一根） AND vol_expanding
      Long exit:   close < mid（回穿中軌）OR fake breakout
      Short exit:  close > mid（回穿中軌）OR fake breakout

    Fake breakout filter:
      入場後 fake_breakout_bars 根 bar 內，
      若 close 回到通道內（多倉: close < upper, 空倉: close > lower）→ 視為假突破，平倉

    Args:
        close, high, low:  價格序列
        upper, lower, mid: 通道邊界
        vol_expanding:     波動率擴張 bool 陣列
        fake_breakout_bars: 假突破監測窗口
        min_hold_bars:     最少持倉 bar 數
        trail_exit_mid:    是否用中軌止盈

    Returns:
        position 陣列 [-1, 0, 1]
    """
    n = len(close)
    pos = np.zeros(n, dtype=np.float64)

    # state
    state = 0       # 0=flat, 1=long, -1=short
    bars_held = 0
    entry_bar = -999

    for i in range(1, n):
        # 跳過 NaN
        if (np.isnan(upper[i - 1]) or np.isnan(lower[i - 1])
                or np.isnan(mid[i]) or np.isnan(close[i])):
            pos[i] = 0.0
            state = 0
            continue

        if state == 1:
            # --- 持有多倉 ---
            bars_held += 1

            # Fake breakout check（在觀察窗口內，close 落回通道內）
            if bars_held <= fake_breakout_bars and close[i] < upper[i - 1]:
                # 假突破 → 平倉（不受 min_hold 限制）
                pos[i] = 0.0
                state = 0
                continue

            # 中軌止盈 / 信號退出
            if bars_held >= min_hold_bars:
                if trail_exit_mid and close[i] < mid[i]:
                    pos[i] = 0.0
                    state = 0
                    continue

            pos[i] = 1.0

        elif state == -1:
            # --- 持有空倉 ---
            bars_held += 1

            # Fake breakout check
            if bars_held <= fake_breakout_bars and close[i] > lower[i - 1]:
                pos[i] = 0.0
                state = 0
                continue

            if bars_held >= min_hold_bars:
                if trail_exit_mid and close[i] > mid[i]:
                    pos[i] = 0.0
                    state = 0
                    continue

            pos[i] = -1.0

        else:
            # --- 空仓：檢查入場 ---
            # 用前一根的通道邊界（避免 look-ahead）
            vo = vol_expanding[i]
            if np.isnan(vo):
                vo = False

            if close[i] > upper[i - 1] and vo:
                pos[i] = 1.0
                state = 1
                bars_held = 0
                entry_bar = i
            elif close[i] < lower[i - 1] and vo:
                pos[i] = -1.0
                state = -1
                bars_held = 0
                entry_bar = i
            else:
                pos[i] = 0.0

    return pos


# ══════════════════════════════════════════════════════════════
#  策略 1: Donchian Breakout + Vol Expansion（基礎版）
# ══════════════════════════════════════════════════════════════

@register_strategy("breakout_vol", auto_delay=False)
def generate_breakout_vol(
    df: pd.DataFrame, ctx: StrategyContext, params: dict
) -> pd.Series:
    """
    Donchian Channel Breakout + Volatility Expansion

    params:
        # ── Channel ──
        channel_period:       Donchian 回看期（預設 96 = 4 天）
        # ── Vol Expansion ──
        atr_period:           ATR 計算週期（預設 14）
        vol_fast_period:      ATR 快速均值（預設 14）
        vol_slow_period:      ATR 慢速均值（預設 50）
        expansion_ratio:      擴張倍數門檻（預設 1.0 = 快 > 慢）
        # ── Filters ──
        fake_breakout_bars:   假突破監測窗口（預設 3）
        min_hold_bars:        最少持倉 bar 數（預設 6）
        trail_exit_mid:       是否用中軌止盈（預設 True）
        # ── Exit Rules ──
        stop_loss_atr:        止損 ATR 倍數（預設 2.5）
        take_profit_atr:      止盈 ATR 倍數（預設 null，用中軌退出）
        trailing_stop_atr:    移動止損 ATR 倍數（預設 null）
        cooldown_bars:        出場後冷卻 bar 數（預設 6）
    """
    # ── 參數解析 ──
    channel_period = int(params.get("channel_period", 96))
    atr_period = int(params.get("atr_period", 14))
    vol_fast_period = int(params.get("vol_fast_period", 14))
    vol_slow_period = int(params.get("vol_slow_period", 50))
    expansion_ratio = float(params.get("expansion_ratio", 1.0))
    fake_breakout_bars = int(params.get("fake_breakout_bars", 3))
    min_hold_bars = int(params.get("min_hold_bars", 6))
    trail_exit_mid = bool(params.get("trail_exit_mid", True))

    # exit rules
    sl_atr = params.get("stop_loss_atr", 2.5)
    tp_atr = params.get("take_profit_atr", None)
    trailing_atr = params.get("trailing_stop_atr", None)
    cooldown = int(params.get("cooldown_bars", 6))
    exit_min_hold = int(params.get("exit_min_hold_bars", 0))

    if sl_atr is not None:
        sl_atr = float(sl_atr)
    if tp_atr is not None:
        tp_atr = float(tp_atr)
    if trailing_atr is not None:
        trailing_atr = float(trailing_atr)

    # ── 計算指標 ──
    close = df["close"]
    high = df["high"]
    low = df["low"]

    atr = calculate_atr(df, atr_period)
    upper, lower, mid = _donchian_channel(high, low, channel_period)
    vol_exp = _volatility_expanding(atr, vol_fast_period, vol_slow_period, expansion_ratio)

    # ── 生成 raw 信號 ──
    raw_pos_arr = _generate_breakout_signals(
        close.values, high.values, low.values,
        upper.values, lower.values, mid.values,
        vol_exp.values.astype(float),
        fake_breakout_bars=fake_breakout_bars,
        min_hold_bars=min_hold_bars,
        trail_exit_mid=trail_exit_mid,
    )
    raw_pos = pd.Series(raw_pos_arr, index=df.index)

    # ── signal_delay（手動 shift） ──
    signal_delay = getattr(ctx, "signal_delay", 0)
    if signal_delay > 0:
        raw_pos = raw_pos.shift(signal_delay).fillna(0.0)

    # ── Exit Rules（SL/TP/Trailing） ──
    if sl_atr is not None or tp_atr is not None or trailing_atr is not None:
        pos, _exec_prices = apply_exit_rules(
            df, raw_pos,
            stop_loss_atr=sl_atr,
            take_profit_atr=tp_atr,
            trailing_stop_atr=trailing_atr,
            atr_period=atr_period,
            cooldown_bars=cooldown,
            min_hold_bars=exit_min_hold,
        )
    else:
        pos = raw_pos

    # ── Direction clip ──
    if not ctx.can_short:
        pos = pos.clip(lower=0.0)
    if not ctx.can_long:
        pos = pos.clip(upper=0.0)

    return pos


# ══════════════════════════════════════════════════════════════
#  策略 2: ATR Channel Breakout + Vol Expansion（替代版）
# ══════════════════════════════════════════════════════════════

@register_strategy("breakout_vol_atr", auto_delay=False)
def generate_breakout_vol_atr(
    df: pd.DataFrame, ctx: StrategyContext, params: dict
) -> pd.Series:
    """
    ATR Channel Breakout + Volatility Expansion

    與 Donchian 版差異：
      - 通道用 EMA ± mult × ATR（自適應寬度）
      - 更適合加密幣高波動環境（通道寬度隨波動率自動調整）

    params:
        # ── Channel ──
        channel_period:       EMA 週期（預設 96）
        channel_multiplier:   ATR 倍數（預設 2.0）
        # ── Vol Expansion ──
        atr_period:           ATR 計算週期（預設 14）
        vol_fast_period:      ATR 快速均值（預設 14）
        vol_slow_period:      ATR 慢速均值（預設 50）
        expansion_ratio:      擴張倍數門檻（預設 1.0）
        # ── Filters ──
        fake_breakout_bars:   假突破監測窗口（預設 3）
        min_hold_bars:        最少持倉 bar 數（預設 6）
        trail_exit_mid:       是否用中軌止盈（預設 True）
        # ── Exit Rules ──
        stop_loss_atr:        止損 ATR 倍數（預設 2.5）
        take_profit_atr:      止盈 ATR 倍數（預設 null）
        trailing_stop_atr:    移動止損 ATR 倍數（預設 null）
        cooldown_bars:        出場後冷卻 bar 數（預設 6）
        max_holding_bars:     最大持倉 bar 數（預設 0 = 不限）— time stop
        # ── Volatility Scaling ──
        vol_scale_enabled:    是否啟用波動率反比縮放（預設 false）
        vol_scale_lookback:   波動率計算回看期（預設 168 = 7天）
        vol_scale_target:     年化波動率目標（預設 0.80）
        vol_scale_floor:      最低縮放倍數（預設 0.2）
        vol_scale_cap:        最高縮放倍數（預設 1.0）
    """
    # ── 參數解析 ──
    channel_period = int(params.get("channel_period", 96))
    channel_mult = float(params.get("channel_multiplier", 2.0))
    atr_period = int(params.get("atr_period", 14))
    vol_fast_period = int(params.get("vol_fast_period", 14))
    vol_slow_period = int(params.get("vol_slow_period", 50))
    expansion_ratio = float(params.get("expansion_ratio", 1.0))
    fake_breakout_bars = int(params.get("fake_breakout_bars", 3))
    min_hold_bars = int(params.get("min_hold_bars", 6))
    trail_exit_mid = bool(params.get("trail_exit_mid", True))

    # exit rules
    sl_atr = params.get("stop_loss_atr", 2.5)
    tp_atr = params.get("take_profit_atr", None)
    trailing_atr = params.get("trailing_stop_atr", None)
    cooldown = int(params.get("cooldown_bars", 6))
    exit_min_hold = int(params.get("exit_min_hold_bars", 0))
    max_holding = int(params.get("max_holding_bars", 0))

    if sl_atr is not None:
        sl_atr = float(sl_atr)
    if tp_atr is not None:
        tp_atr = float(tp_atr)
    if trailing_atr is not None:
        trailing_atr = float(trailing_atr)

    # vol scaling
    vol_scale_enabled = bool(params.get("vol_scale_enabled", False))
    vol_scale_lookback = int(params.get("vol_scale_lookback", 168))
    vol_scale_target = float(params.get("vol_scale_target", 0.80))
    vol_scale_floor = float(params.get("vol_scale_floor", 0.2))
    vol_scale_cap = float(params.get("vol_scale_cap", 1.0))

    # ── 計算指標 ──
    close = df["close"]
    high = df["high"]
    low = df["low"]

    atr = calculate_atr(df, atr_period)
    upper, lower, mid = _atr_channel(close, atr, channel_period, channel_mult)
    vol_exp = _volatility_expanding(atr, vol_fast_period, vol_slow_period, expansion_ratio)

    # ── 生成 raw 信號 ──
    raw_pos_arr = _generate_breakout_signals(
        close.values, high.values, low.values,
        upper.values, lower.values, mid.values,
        vol_exp.values.astype(float),
        fake_breakout_bars=fake_breakout_bars,
        min_hold_bars=min_hold_bars,
        trail_exit_mid=trail_exit_mid,
    )
    raw_pos = pd.Series(raw_pos_arr, index=df.index)

    # ── signal_delay（手動 shift） ──
    signal_delay = getattr(ctx, "signal_delay", 0)
    if signal_delay > 0:
        raw_pos = raw_pos.shift(signal_delay).fillna(0.0)

    # ── Exit Rules（SL/TP/Trailing/TimeStop） ──
    if sl_atr is not None or tp_atr is not None or trailing_atr is not None or max_holding > 0:
        pos, _exec_prices = apply_exit_rules(
            df, raw_pos,
            stop_loss_atr=sl_atr,
            take_profit_atr=tp_atr,
            trailing_stop_atr=trailing_atr,
            atr_period=atr_period,
            cooldown_bars=cooldown,
            min_hold_bars=exit_min_hold,
            max_holding_bars=max_holding,
        )
    else:
        pos = raw_pos

    # ── Volatility Scaling（在 exit rules 之後） ──
    # exit_rules 輸出 binary positions (1/-1/0)
    # vol scaling 在其上縮放倉位大小，保留方向
    if vol_scale_enabled:
        import numpy as _np
        returns = close.pct_change()
        rolling_vol = returns.rolling(vol_scale_lookback, min_periods=max(24, vol_scale_lookback // 4)).std() * _np.sqrt(8760)
        rolling_vol = rolling_vol.replace(0, _np.nan).ffill().fillna(vol_scale_target)
        scale = (vol_scale_target / rolling_vol).clip(vol_scale_floor, vol_scale_cap)
        pos = pos * scale

    # ── Regime Filter（ADX chop scaler, E2）──
    # 當 ADX 低於門檻時，將倉位縮小到 chop_scale（預設不啟用）
    regime_filter = bool(params.get("regime_filter_enabled", False))
    if regime_filter:
        from ..indicators.adx import calculate_adx
        r_adx_period = int(params.get("regime_adx_period", 14))
        r_adx_thresh = float(params.get("regime_adx_threshold", 20))
        r_chop_scale = float(params.get("regime_chop_scale", 0.3))
        adx_data = calculate_adx(df, r_adx_period)
        adx_vals = adx_data["ADX"].shift(1).fillna(0)  # lagged — no lookahead
        chop_mask = adx_vals < r_adx_thresh
        pos = pos.copy()
        pos[chop_mask] = pos[chop_mask] * r_chop_scale

    # ── Volume Confirmation Filter（E3）──
    # 突破入場須 volume > MA(volume) * ratio
    vol_confirm = bool(params.get("volume_confirm_enabled", False))
    if vol_confirm:
        vc_period = int(params.get("volume_confirm_period", 20))
        vc_ratio = float(params.get("volume_confirm_ratio", 1.5))
        vol_ma = df["volume"].rolling(vc_period).mean().shift(1)  # lagged
        vol_ok = df["volume"] > (vol_ma * vc_ratio)
        # only zero-out new entries (transitions from 0 to non-0)
        pos_shifted = pos.shift(1).fillna(0)
        new_entry = (pos != 0) & (pos_shifted == 0)
        block = new_entry & (~vol_ok)
        pos = pos.copy()
        pos[block] = 0.0

    # ── Direction clip ──
    if not ctx.can_short:
        pos = pos.clip(lower=0.0)
    if not ctx.can_long:
        pos = pos.clip(upper=0.0)

    return pos
