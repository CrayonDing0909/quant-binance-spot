"""
NW Envelope + Regime Filter + MTF Gating 策略（Futures 多空）

v2: Alpha Enhancement — 雙模組 + 多時間框架

學術背景：
    - Nadaraya (1964), Watson (1964): Kernel Regression
    - Rational Quadratic Kernel: Rasmussen & Williams (2006)
    - Multi-timeframe gating: 4h regime filter → 1h entry

核心改進（相較 v1）：
    1. 4h NW regime 取代 1h regime → 更穩定的趨勢判斷
    2. 雙模組設計：
       Module A (Trend): NW center pullback / band breakout，持倉到趨勢反轉
       Module B (MR): 均值回歸 band touch，持倉到 NW center
    3. 趨勢回踩用 NW center（比 band touch 更頻繁但仍有品質過濾）
    4. 位置縮放：趨勢順向全倉、逆向半倉

策略設計防 look-ahead bias：
    - 使用 causal NW regression（只用 t 及之前的資料）
    - auto_delay=False，手動在 exit_rules 前 shift(signal_delay)
    - exit_rules 入場用 open 價，SL/TP 用 intra-bar 價
    - 回測 price=open，trade_on=next_open

Ablation features（可逐步啟用，用於實驗）：
    - use_mtf: 是否使用 4h regime（False → 用 1h regime）
    - module_a_enabled / module_b_enabled: 雙模組開關
    - use_vol_scaling: 是否啟用 vol targeting（由框架控制）
    - entry_mode: "pullback_nw"/"pullback_band"/"breakout"/"dual"

Note:
    使用 auto_delay=False，因為需要在 exit_rules 之前手動 shift。
    框架自動處理 direction clip。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import StrategyContext
from . import register_strategy
from ..indicators import calculate_adx, calculate_atr, calculate_ema
from .exit_rules import apply_exit_rules

# 重用 nwkl_strategy 的 kernel 計算函數
from .nwkl_strategy import _rq_kernel_weights, _causal_nw_regression


# ══════════════════════════════════════════════════════════════
#  NW Envelope 計算
# ══════════════════════════════════════════════════════════════

def _compute_nw_envelope(
    df: pd.DataFrame,
    bandwidth: float = 8.0,
    alpha: float = 1.0,
    lookback: int = 200,
    envelope_mult: float = 2.5,
    envelope_window: int = 200,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    計算因果 Nadaraya-Watson 核回歸中線 + MAE 包絡線

    因果 = 只看 t 及之前的資料，不會重繪（消除 look-ahead）

    Returns:
        (nw_estimate, upper_band, lower_band)
    """
    close = df["close"]
    close_arr = close.values.astype(np.float64)

    weights = _rq_kernel_weights(lookback, bandwidth, alpha)
    estimate = _causal_nw_regression(close_arr, weights, lookback)
    nw = pd.Series(estimate, index=df.index)

    # MAE (Mean Absolute Error) 包絡線
    residuals = (close - nw).abs()
    mae = residuals.rolling(window=envelope_window, min_periods=1).mean()

    upper = nw + envelope_mult * mae
    lower = nw - envelope_mult * mae

    return nw, upper, lower


# ══════════════════════════════════════════════════════════════
#  Regime Detection（1h / 4h 通用）
# ══════════════════════════════════════════════════════════════

def _detect_regime(
    df: pd.DataFrame,
    nw_estimate: pd.Series,
    regime_mode: str = "adx",
    adx_period: int = 14,
    adx_threshold: float = 25.0,
    slope_window: int = 10,
    slope_threshold: float = 0.001,
) -> tuple[pd.Series, pd.Series]:
    """
    Regime 偵測（震盪 vs 趨勢）+ 趨勢方向

    支援三種模式：
        - "adx":   僅用 ADX 判斷趨勢強度 + DI 判斷方向
        - "slope":  僅用 NW 中線斜率
        - "both":   兩者都要確認（最嚴格）

    Returns:
        (is_trending: bool Series, trend_direction: +1.0/0.0/-1.0 Series)
    """
    is_trending = pd.Series(False, index=df.index)
    trend_dir = pd.Series(0.0, index=df.index)

    # ── ADX regime ──
    adx_trending = pd.Series(False, index=df.index)
    adx_dir = pd.Series(0.0, index=df.index)

    if regime_mode in ("adx", "both"):
        adx_data = calculate_adx(df, adx_period)
        adx = adx_data["ADX"]
        plus_di = adx_data["+DI"]
        minus_di = adx_data["-DI"]

        adx_trending = adx >= adx_threshold
        adx_dir = pd.Series(0.0, index=df.index)
        adx_dir[plus_di > minus_di] = 1.0
        adx_dir[minus_di > plus_di] = -1.0

    # ── Slope regime ──
    slope_trending = pd.Series(False, index=df.index)
    slope_dir = pd.Series(0.0, index=df.index)

    if regime_mode in ("slope", "both"):
        nw_slope = (nw_estimate - nw_estimate.shift(slope_window)) / nw_estimate.shift(slope_window)
        nw_slope = nw_slope.fillna(0.0)

        slope_trending = nw_slope.abs() > slope_threshold
        slope_dir[nw_slope > slope_threshold] = 1.0
        slope_dir[nw_slope < -slope_threshold] = -1.0

    # ── 合併判斷 ──
    if regime_mode == "adx":
        is_trending = adx_trending
        trend_dir = adx_dir
    elif regime_mode == "slope":
        is_trending = slope_trending
        trend_dir = slope_dir
    else:  # "both" — 兩個都要確認
        is_trending = adx_trending & slope_trending
        agree = (adx_dir == slope_dir) & (adx_dir != 0)
        trend_dir = adx_dir.copy()
        trend_dir[~agree] = 0.0

    return is_trending, trend_dir


# ══════════════════════════════════════════════════════════════
#  4h MTF Regime 計算
# ══════════════════════════════════════════════════════════════

def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    將低級時間框架 K 線重採樣為高級時間框架（left/left — 交易所標準）

    時間語義（以 1h→4h 為例）：
        closed='left', label='left'
        → 4h bar 標記 00:00 包含 1h bars: 00:00, 01:00, 02:00, 03:00
        → close = 03:00 的 1h close
        → 此 bar 在 03:00 1h bar 收盤後完成
    """
    return df.resample(rule, closed="left", label="left").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()


def _resample_ohlcv_right(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    將低級時間框架 K 線重採樣為高級時間框架（right/right — 最小延遲）

    時間語義（以 1h→4h 為例）：
        closed='right', label='right'
        → 4h bar 標記 04:00 包含 1h bars: 01:00, 02:00, 03:00, 04:00
        → close = 04:00 的 1h close
        → 此 bar 在 04:00 1h bar 收盤時完成
        → label=04:00 = 完成時刻 → ffill 直接使用，無需 shift
    """
    return df.resample(rule, closed="right", label="right").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()


# ── 合法的 MTF 對齊模式 ──
_VALID_ALIGNMENT_MODES = {"legacy_left_ffill", "right_ffill", "left_shift1_ffill"}


def _compute_htf_regime(
    df_1h: pd.DataFrame,
    rule: str = "4h",
    bandwidth: float = 8.0,
    alpha: float = 1.0,
    lookback: int = 50,
    adx_period: int = 14,
    adx_threshold: float = 20.0,
    slope_window: int = 5,
    slope_threshold: float = 0.002,
    regime_mode: str = "adx",
    mtf_alignment_mode: str = "left_shift1_ffill",
) -> tuple[pd.Series, pd.Series]:
    """
    在指定 timeframe 上計算 NW regime，再對齊回 1h。
    支援 4h、12h、1D 等任意高級時間框架。

    Args:
        rule: 重採樣規則，如 "4h", "12h", "1D"

    ═══ 三種 MTF 對齊模式 ═══

    1. "legacy_left_ffill"（⚠️ LOOK-AHEAD — 僅供對照測試）
       resample(left/left) + 直接 ffill
       → 未來資訊洩漏

    2. "right_ffill"（✅ CAUSAL — 最小延遲）
       resample(right/right) + 直接 ffill
       bar 的 label = 完成時刻 → ffill 無需 shift
       → 0h 額外延遲

    3. "left_shift1_ffill"（✅ CAUSAL — 保守穩定）
       resample(left/left) + shift(1) + ffill
       → 多 1 個 HTF bar 延遲

    Returns:
        (htf_trending_1h, htf_direction_1h): 已對齊回 1h index
    """
    if mtf_alignment_mode not in _VALID_ALIGNMENT_MODES:
        raise ValueError(
            f"Unknown mtf_alignment_mode: {mtf_alignment_mode!r}. "
            f"Valid: {_VALID_ALIGNMENT_MODES}"
        )

    # ── Step 1: 依模式選擇 resample 方式 ──
    if mtf_alignment_mode == "right_ffill":
        df_4h = _resample_ohlcv_right(df_1h, rule)
    else:
        # legacy_left_ffill 和 left_shift1_ffill 都用 left/left（交易所標準）
        df_4h = _resample_ohlcv(df_1h, rule)

    if len(df_4h) < lookback + 20:
        return (
            pd.Series(0.0, index=df_1h.index),
            pd.Series(0.0, index=df_1h.index),
        )

    # ── Step 2: 在 4h 上計算 NW + Regime ──
    nw_4h, _, _ = _compute_nw_envelope(
        df_4h, bandwidth, alpha, lookback,
        envelope_mult=2.0,
        envelope_window=lookback,
    )

    htf_trending, htf_dir = _detect_regime(
        df_4h, nw_4h,
        regime_mode=regime_mode,
        adx_period=adx_period,
        adx_threshold=adx_threshold,
        slope_window=slope_window,
        slope_threshold=slope_threshold,
    )

    # ── Step 3: 依模式對齊回 1h ──

    if mtf_alignment_mode == "legacy_left_ffill":
        # ⚠️ LOOK-AHEAD: 4h bar data 在同 4h 區間開始即可見
        # 保留僅供仲裁測試對照，不應用於正式回測
        htf_trending_1h = htf_trending.astype(float).reindex(
            df_1h.index, method="ffill"
        ).fillna(0.0)
        htf_dir_1h = htf_dir.reindex(
            df_1h.index, method="ffill"
        ).fillna(0.0)

    elif mtf_alignment_mode == "right_ffill":
        # ✅ CAUSAL: right/right label = bar 完成時刻
        # 4h bar 04:00 的 close = 04:00 1h close
        # 04:00 1h bar 收盤時，此 regime 立即可見（0h 額外延遲）
        # 1h bars 01:00~03:00 ffill 自 00:00 bar（已完成的上一根）
        htf_trending_1h = htf_trending.astype(float).reindex(
            df_1h.index, method="ffill"
        ).fillna(0.0)
        htf_dir_1h = htf_dir.reindex(
            df_1h.index, method="ffill"
        ).fillna(0.0)

    elif mtf_alignment_mode == "left_shift1_ffill":
        # ✅ CAUSAL: left/left + shift(1)
        # 4h bar 00:00（close=03:00）的 regime 延遲到 04:00 才可見
        # 使用交易所標準 4h bar → 與 Binance K 線定義一致
        # 額外延遲 1h（相較 right_ffill）
        htf_trending = htf_trending.astype(float).shift(1).fillna(0.0)
        htf_dir = htf_dir.shift(1).fillna(0.0)

        htf_trending_1h = htf_trending.reindex(
            df_1h.index, method="ffill"
        ).fillna(0.0)
        htf_dir_1h = htf_dir.reindex(
            df_1h.index, method="ffill"
        ).fillna(0.0)

    return htf_trending_1h, htf_dir_1h


# ══════════════════════════════════════════════════════════════
#  Phase B: Alpha Rebuild — LTF Proxy + Momentum Confirm
# ══════════════════════════════════════════════════════════════

def _compute_ltf_proxy(
    df: pd.DataFrame,
    ema_fast: int = 20,
    ema_slow: int = 50,
) -> pd.Series:
    """
    1h causal regime proxy: EMA fast/slow 交叉。

    目的：降低 4h regime 延遲的影響。4h 給「結構方向」，
    1h EMA 交叉給「即時確認」，兩者一致時入場品質最高。

    +1.0 = bullish (ema_fast > ema_slow)
    -1.0 = bearish (ema_fast < ema_slow)
     0.0 = neutral / warmup
    """
    ef = calculate_ema(df["close"], ema_fast)
    es = calculate_ema(df["close"], ema_slow)
    proxy = pd.Series(0.0, index=df.index)
    proxy[ef > es] = 1.0
    proxy[ef < es] = -1.0
    return proxy


def _compute_momentum_confirm(
    close: pd.Series,
    lookback: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    動量恢復確認：close 相對 N bars 前的方向。

    目的：避免在持續下跌中做多（catching falling knives），
    或在持續上漲中做空。入場需要價格已開始「反轉」或「恢復」。

    long_ok[i] = close[i] > close[i-lookback]  → 做多需要近期有上漲
    short_ok[i] = close[i] < close[i-lookback]  → 做空需要近期有下跌
    """
    diff = close - close.shift(lookback)
    long_ok = (diff > 0).fillna(False).values.astype(bool)
    short_ok = (diff < 0).fillna(False).values.astype(bool)
    return long_ok, short_ok


def _compute_volume_expansion(
    df: pd.DataFrame,
    period: int = 20,
    min_ratio: float = 1.5,
) -> np.ndarray:
    """
    Volume expansion 過濾器：成交量 > 均量 × min_ratio 時才允許入場。

    目的：過濾低流動性時段的假信號，只在市場活躍時交易。
    用於 E3 實驗（entry trigger quality）。

    Returns:
        bool array: True = volume expansion ok
    """
    vol_ma = df["volume"].rolling(period, min_periods=1).mean()
    return (df["volume"] >= vol_ma * min_ratio).values.astype(bool)


# ══════════════════════════════════════════════════════════════
#  Stateful Signal Generator V2（雙模組 + MTF + Entry Filters）
# ══════════════════════════════════════════════════════════════

def _generate_signals_v2(
    close_arr: np.ndarray,
    nw_arr: np.ndarray,
    upper_arr: np.ndarray,
    lower_arr: np.ndarray,
    regime_trending_arr: np.ndarray,
    regime_dir_arr: np.ndarray,
    module_a_enabled: bool,     # 趨勢模組
    module_b_enabled: bool,     # 均值回歸模組
    entry_mode: str,            # "pullback_nw"/"pullback_band"/"breakout"/"dual"
    mr_exit_target: str,        # "nw"/"upper"/"band"
    trend_scale: float,         # 趨勢模組倉位 (1.0)
    range_scale: float,         # 均值回歸模組倉位 (0.5-1.0)
    # ── Phase B: Entry Quality Filters ──
    ltf_proxy_arr: np.ndarray | None = None,       # 1h EMA 交叉 proxy (+1/0/-1)
    momentum_long_arr: np.ndarray | None = None,    # bool: close > close[-n]
    momentum_short_arr: np.ndarray | None = None,   # bool: close < close[-n]
    min_pullback_depth: float = 0.0,                # 回踩深度門檻 (0=不過濾)
    reentry_lockout: bool = False,                  # 出場後需價格回穿 NW 才重入
    volume_ok_arr: np.ndarray | None = None,        # volume expansion gate
) -> np.ndarray:
    """
    帶狀態的信號生成器 V2（雙模組 + MTF gating + Entry Filters）

    Phase B Structural Improvements:
        1. LTF Proxy: 1h EMA 交叉方向必須與 4h regime 一致
        2. Momentum Confirm: 做多需近期上漲，做空需近期下跌
        3. Pullback Depth: 回踩深度 > min_pullback_depth 才入場
           depth = (NW-close)/(NW-lower) for long, (close-NW)/(upper-NW) for short
           0 = at NW center, 1 = at band → 更深的回踩 = 更高品質入場
        4. Re-entry Lockout: 出場後價格需回穿 NW center 才解鎖同方向再入場
           防止在弱趨勢中反覆入場 → 降低 turnover + cost drag

    Returns:
        signal array [-scale, 0, +scale]
    """
    n = len(close_arr)
    signal = np.zeros(n, dtype=np.float64)

    logical_pos = 0       # 0=flat, 1=long, -1=short
    entry_module = 0      # 0=none, 1=trend, 2=mean_reversion
    entry_sub_mode = 0    # 0=none, 1=pullback, 2=breakout
    pos_scale = 1.0

    # Re-entry lockout state
    lockout_dir = 0       # 上次出場方向: 1=long, -1=short, 0=none
    reentry_ok = True     # 是否已解鎖（價格回穿 NW）

    for i in range(n):
        c = close_arr[i]
        nw = nw_arr[i]
        up = upper_arr[i]
        lo = lower_arr[i]
        trending = bool(regime_trending_arr[i])
        t_dir = regime_dir_arr[i]

        # ── Re-entry lockout: 價格回穿 NW 則解鎖 ──
        if reentry_lockout and not reentry_ok:
            if lockout_dir == 1 and c > nw:
                reentry_ok = True
            elif lockout_dir == -1 and c < nw:
                reentry_ok = True

        # ── 已持多倉：檢查出場條件 ──
        if logical_pos == 1:
            should_exit = False

            if entry_module == 1:
                if (not trending) or (t_dir <= 0):
                    should_exit = True
                elif entry_sub_mode == 1 and c >= up:
                    should_exit = True
                elif entry_sub_mode == 2 and c < nw:
                    should_exit = True

            elif entry_module == 2:
                if mr_exit_target == "nw":
                    should_exit = c >= nw
                elif mr_exit_target == "upper":
                    should_exit = c >= up
                else:
                    should_exit = c >= lo

            if should_exit:
                signal[i] = 0.0
                # Set lockout
                if reentry_lockout:
                    lockout_dir = 1  # was long
                    reentry_ok = False
                logical_pos = 0
                entry_module = 0
                entry_sub_mode = 0
            else:
                signal[i] = pos_scale
            continue

        # ── 已持空倉：檢查出場條件 ──
        if logical_pos == -1:
            should_exit = False

            if entry_module == 1:
                if (not trending) or (t_dir >= 0):
                    should_exit = True
                elif entry_sub_mode == 1 and c <= lo:
                    should_exit = True
                elif entry_sub_mode == 2 and c > nw:
                    should_exit = True

            elif entry_module == 2:
                if mr_exit_target == "nw":
                    should_exit = c <= nw
                elif mr_exit_target == "upper":
                    should_exit = c <= lo
                else:
                    should_exit = c <= up

            if should_exit:
                signal[i] = 0.0
                if reentry_lockout:
                    lockout_dir = -1  # was short
                    reentry_ok = False
                logical_pos = 0
                entry_module = 0
                entry_sub_mode = 0
            else:
                signal[i] = -pos_scale
            continue

        # ── Flat：檢查入場條件 ──

        # Module A: Trend Following（趨勢 regime 下）
        if module_a_enabled and trending and t_dir != 0:

            # ── Phase B: Entry Quality Filters ──
            # 1. LTF proxy: 1h EMA 方向必須與 4h regime 一致
            ltf_ok = True
            if ltf_proxy_arr is not None:
                ltf_ok = (
                    (t_dir > 0 and ltf_proxy_arr[i] > 0)
                    or (t_dir < 0 and ltf_proxy_arr[i] < 0)
                )

            # 2. Momentum confirm: 做多需要近期上漲，做空需要近期下跌
            mom_ok = True
            if t_dir > 0 and momentum_long_arr is not None:
                mom_ok = bool(momentum_long_arr[i])
            elif t_dir < 0 and momentum_short_arr is not None:
                mom_ok = bool(momentum_short_arr[i])

            # 3. Re-entry lockout: 出場後需回穿 NW 才解鎖同方向
            lockout_ok = True
            if reentry_lockout and not reentry_ok:
                if lockout_dir == 1 and t_dir > 0:
                    lockout_ok = False   # 多倉出場後要買，但未回穿
                elif lockout_dir == -1 and t_dir < 0:
                    lockout_ok = False   # 空倉出場後要空，但未回穿

            # 4. Volume expansion: 成交量須達標
            vol_ok = True
            if volume_ok_arr is not None:
                vol_ok = bool(volume_ok_arr[i])

            if not (ltf_ok and mom_ok and lockout_ok and vol_ok):
                signal[i] = 0.0
                continue

            entered = False

            if entry_mode in ("pullback_nw", "dual"):
                # Pullback to NW center
                if t_dir > 0 and c < nw:
                    # 4. Pullback depth filter
                    bw = nw - lo
                    depth = (nw - c) / bw if bw > 0 else 0.0
                    if depth >= min_pullback_depth:
                        signal[i] = trend_scale
                        logical_pos = 1
                        entry_module = 1
                        entry_sub_mode = 1
                        pos_scale = trend_scale
                        entered = True
                        if reentry_lockout:
                            lockout_dir = 0
                            reentry_ok = True
                elif t_dir < 0 and c > nw:
                    bw = up - nw
                    depth = (c - nw) / bw if bw > 0 else 0.0
                    if depth >= min_pullback_depth:
                        signal[i] = -trend_scale
                        logical_pos = -1
                        entry_module = 1
                        entry_sub_mode = 1
                        pos_scale = trend_scale
                        entered = True
                        if reentry_lockout:
                            lockout_dir = 0
                            reentry_ok = True

            if not entered and entry_mode in ("pullback_band",):
                # Pullback to lower band（depth 總是 >= 1.0，自動通過 depth filter）
                if t_dir > 0 and c < lo:
                    signal[i] = trend_scale
                    logical_pos = 1
                    entry_module = 1
                    entry_sub_mode = 1
                    pos_scale = trend_scale
                    entered = True
                    if reentry_lockout:
                        lockout_dir = 0
                        reentry_ok = True
                elif t_dir < 0 and c > up:
                    signal[i] = -trend_scale
                    logical_pos = -1
                    entry_module = 1
                    entry_sub_mode = 1
                    pos_scale = trend_scale
                    entered = True
                    if reentry_lockout:
                        lockout_dir = 0
                        reentry_ok = True

            if not entered and entry_mode in ("breakout", "dual"):
                # Breakout past envelope
                if t_dir > 0 and c > up:
                    signal[i] = trend_scale
                    logical_pos = 1
                    entry_module = 1
                    entry_sub_mode = 2
                    pos_scale = trend_scale
                    entered = True
                    if reentry_lockout:
                        lockout_dir = 0
                        reentry_ok = True
                elif t_dir < 0 and c < lo:
                    signal[i] = -trend_scale
                    logical_pos = -1
                    entry_module = 1
                    entry_sub_mode = 2
                    pos_scale = trend_scale
                    entered = True
                    if reentry_lockout:
                        lockout_dir = 0
                        reentry_ok = True

            if entered:
                continue

        # Module B: Mean Reversion（非趨勢 regime 下）
        if module_b_enabled and not trending:
            # Volume check for MR entries
            b_vol_ok = True
            if volume_ok_arr is not None:
                b_vol_ok = bool(volume_ok_arr[i])
            if not b_vol_ok:
                signal[i] = 0.0
                continue
            if c < lo:
                signal[i] = range_scale
                logical_pos = 1
                entry_module = 2
                entry_sub_mode = 0
                pos_scale = range_scale
                continue
            elif c > up:
                signal[i] = -range_scale
                logical_pos = -1
                entry_module = 2
                entry_sub_mode = 0
                pos_scale = range_scale
                continue

        signal[i] = 0.0

    return signal


# ══════════════════════════════════════════════════════════════
#  Max Holding 後處理
# ══════════════════════════════════════════════════════════════

def _apply_max_holding(pos: pd.Series, max_bars: int) -> pd.Series:
    """強制平倉超過 max_bars 的持倉"""
    if max_bars <= 0:
        return pos
    values = pos.values.copy()
    bars_held = 0
    prev_sign = 0.0

    for i in range(len(values)):
        curr_sign = np.sign(values[i])
        if curr_sign != 0.0 and curr_sign == prev_sign:
            bars_held += 1
        elif curr_sign != 0.0:
            bars_held = 1
        else:
            bars_held = 0

        if bars_held > max_bars:
            values[i] = 0.0
            bars_held = 0

        prev_sign = np.sign(values[i])

    return pd.Series(values, index=pos.index)


# ══════════════════════════════════════════════════════════════
#  策略主函數
# ══════════════════════════════════════════════════════════════

@register_strategy("nw_envelope_regime", auto_delay=False)
def generate_nw_envelope_regime(
    df: pd.DataFrame, ctx: StrategyContext, params: dict
) -> pd.Series:
    """
    Nadaraya-Watson Envelope + Regime Filter V2

    雙模組 + MTF gating，支援 ablation 實驗。

    ── Ablation 參數 ──
        use_mtf:            使用 4h MTF regime (True) 還是 1h regime (False)
        module_a_enabled:   啟用趨勢模組 (Module A)
        module_b_enabled:   啟用均值回歸模組 (Module B)
        entry_mode:         入場模式:
                            "pullback_nw"  — close < NW center (最頻繁)
                            "pullback_band"— close < lower band (最嚴格)
                            "breakout"     — close > upper band (動量型)
                            "dual"         — pullback_nw + breakout (雙入場)

    ── NW Kernel ──
        kernel_bandwidth, kernel_alpha, kernel_lookback
        envelope_multiplier, envelope_window

    ── Regime Filter ──
        regime_mode, adx_period, adx_threshold, slope_window, slope_threshold

    ── MTF ──
        htf_bandwidth, htf_lookback, htf_adx_period, htf_adx_threshold

    ── Trade Mode ──
        trend_scale, range_scale, mr_exit_target

    ── Risk / Exit ──
        stop_loss_atr, take_profit_atr, trailing_stop_atr
        cooldown_bars, max_holding_bars, min_hold_bars
    """
    # ══════════════════════════════════════════════
    #  1. 參數解析
    # ══════════════════════════════════════════════

    # NW Kernel (1h)
    bandwidth = float(params.get("kernel_bandwidth", 8.0))
    alpha = float(params.get("kernel_alpha", 1.0))
    lookback = int(params.get("kernel_lookback", 200))
    envelope_mult = float(params.get("envelope_multiplier", 2.0))
    envelope_window = int(params.get("envelope_window", 200))

    # Regime Filter
    regime_mode = str(params.get("regime_mode", "adx"))
    adx_period = int(params.get("adx_period", 14))
    adx_threshold = float(params.get("adx_threshold", 20.0))
    slope_window = int(params.get("slope_window", 10))
    slope_threshold = float(params.get("slope_threshold", 0.001))

    # MTF
    use_mtf = bool(params.get("use_mtf", True))
    mtf_alignment_mode = str(params.get("mtf_alignment_mode", "left_shift1_ffill"))
    htf_bandwidth = float(params.get("htf_bandwidth", 8.0))
    htf_lookback = int(params.get("htf_lookback", 50))
    htf_adx_period = int(params.get("htf_adx_period", 14))
    htf_adx_threshold = float(params.get("htf_adx_threshold", 20.0))
    htf_slope_window = int(params.get("htf_slope_window", 5))
    htf_slope_threshold = float(params.get("htf_slope_threshold", 0.002))

    # Phase B: LTF Proxy + Momentum + Depth + Lockout
    use_ltf_proxy = bool(params.get("use_ltf_proxy", False))
    ltf_ema_fast = int(params.get("ltf_ema_fast", 20))
    ltf_ema_slow = int(params.get("ltf_ema_slow", 50))
    use_momentum_confirm = bool(params.get("use_momentum_confirm", False))
    momentum_lookback = int(params.get("momentum_lookback", 3))
    min_pullback_depth = float(params.get("min_pullback_depth", 0.0))
    reentry_lockout = bool(params.get("reentry_lockout", False))

    # ═══ Experiment Matrix: Multi-TF Gating ═══
    # 1D Risk Gate
    use_1d_risk_gate = bool(params.get("use_1d_risk_gate", False))
    risk_gate_lookback = int(params.get("risk_gate_lookback", 20))
    risk_gate_adx_period = int(params.get("risk_gate_adx_period", 14))
    risk_gate_adx_threshold = float(params.get("risk_gate_adx_threshold", 25.0))

    # 12H Regime
    use_12h_regime = bool(params.get("use_12h_regime", False))
    htf_12h_lookback = int(params.get("htf_12h_lookback", 25))
    htf_12h_adx_period = int(params.get("htf_12h_adx_period", 14))
    htf_12h_adx_threshold = float(params.get("htf_12h_adx_threshold", 20.0))
    htf_12h_slope_window = int(params.get("htf_12h_slope_window", 5))
    htf_12h_slope_threshold = float(params.get("htf_12h_slope_threshold", 0.002))
    dual_regime_require_agree = bool(params.get("dual_regime_require_agree", True))

    # Volume Expansion
    use_entry_volume_expansion = bool(params.get("use_entry_volume_expansion", False))
    volume_expansion_period = int(params.get("volume_expansion_period", 20))
    volume_expansion_ratio = float(params.get("volume_expansion_ratio", 1.5))

    # Modules
    module_a = bool(params.get("module_a_enabled", True))
    module_b = bool(params.get("module_b_enabled", True))
    entry_mode = str(params.get("entry_mode", "dual"))
    mr_exit_target = str(params.get("mr_exit_target", "nw"))

    # Position scale
    trend_scale = float(params.get("trend_scale", 1.0))
    range_scale = float(params.get("range_scale", 0.5))

    # Risk / Exit
    atr_period = int(params.get("atr_period", 14))
    sl_atr = params.get("stop_loss_atr", 2.0)
    tp_atr = params.get("take_profit_atr", 3.0)
    trailing_atr = params.get("trailing_stop_atr", None)
    cooldown = int(params.get("cooldown_bars", 6))
    max_hold = int(params.get("max_holding_bars", 0))
    min_hold = int(params.get("min_hold_bars", 2))

    signal_delay = getattr(ctx, "signal_delay", 0)

    # ══════════════════════════════════════════════
    #  2. 計算 1h NW Envelope
    # ══════════════════════════════════════════════
    nw, upper, lower = _compute_nw_envelope(
        df, bandwidth, alpha, lookback, envelope_mult, envelope_window,
    )

    close = df["close"]

    # ══════════════════════════════════════════════
    #  3. Regime Detection（1h 或 4h MTF）
    # ══════════════════════════════════════════════
    if use_mtf:
        # 4h regime — 更穩定，更少 whipsaw
        regime_trending, regime_dir = _compute_htf_regime(
            df,
            rule="4h",
            bandwidth=htf_bandwidth,
            alpha=alpha,
            lookback=htf_lookback,
            adx_period=htf_adx_period,
            adx_threshold=htf_adx_threshold,
            slope_window=htf_slope_window,
            slope_threshold=htf_slope_threshold,
            regime_mode=regime_mode,
            mtf_alignment_mode=mtf_alignment_mode,
        )
    else:
        # 1h regime（baseline ablation）
        regime_trending, regime_dir = _detect_regime(
            df, nw,
            regime_mode=regime_mode,
            adx_period=adx_period,
            adx_threshold=adx_threshold,
            slope_window=slope_window,
            slope_threshold=slope_threshold,
        )

    # ══════════════════════════════════════════════
    #  3d. 1D Risk Gate（日線層級的風險開關）
    # ══════════════════════════════════════════════
    if use_1d_risk_gate:
        risk_trending, risk_dir = _compute_htf_regime(
            df, rule="1D",
            bandwidth=bandwidth,
            alpha=alpha,
            lookback=risk_gate_lookback,
            adx_period=risk_gate_adx_period,
            adx_threshold=risk_gate_adx_threshold,
            regime_mode=regime_mode,
            mtf_alignment_mode=mtf_alignment_mode,
        )
        # Gate: 只有日線也顯示趨勢時才允許交易
        regime_trending = regime_trending * risk_trending
        # Direction: 日線與低級別方向必須一致
        dir_disagree = (risk_dir != 0) & (regime_dir != 0) & (risk_dir != regime_dir)
        regime_dir = regime_dir.copy()
        regime_dir[dir_disagree] = 0.0

    # ══════════════════════════════════════════════
    #  3e. 12H Regime（12小時 regime 額外確認）
    # ══════════════════════════════════════════════
    if use_12h_regime:
        r12h_trending, r12h_dir = _compute_htf_regime(
            df, rule="12h",
            bandwidth=htf_bandwidth,
            alpha=alpha,
            lookback=htf_12h_lookback,
            adx_period=htf_12h_adx_period,
            adx_threshold=htf_12h_adx_threshold,
            slope_window=htf_12h_slope_window,
            slope_threshold=htf_12h_slope_threshold,
            regime_mode=regime_mode,
            mtf_alignment_mode=mtf_alignment_mode,
        )
        if dual_regime_require_agree:
            # 12H 和 4H 必須同向
            regime_trending = regime_trending * r12h_trending
            dir_disagree = (r12h_dir != 0) & (regime_dir != 0) & (r12h_dir != regime_dir)
            regime_dir = regime_dir.copy()
            regime_dir[dir_disagree] = 0.0

    # ══════════════════════════════════════════════
    #  3b. LTF Proxy（1h EMA cross — 降低 regime 延遲）
    # ══════════════════════════════════════════════
    ltf_proxy = None
    if use_ltf_proxy:
        ltf_proxy = _compute_ltf_proxy(df, ltf_ema_fast, ltf_ema_slow)

    # ══════════════════════════════════════════════
    #  3c. Momentum Confirmation（避免 catching falling knives）
    # ══════════════════════════════════════════════
    mom_long = None
    mom_short = None
    if use_momentum_confirm:
        mom_long, mom_short = _compute_momentum_confirm(close, momentum_lookback)

    # ══════════════════════════════════════════════
    #  3f. Volume Expansion（成交量擴張過濾）
    # ══════════════════════════════════════════════
    vol_expansion = None
    if use_entry_volume_expansion:
        vol_expansion = _compute_volume_expansion(
            df, volume_expansion_period, volume_expansion_ratio,
        )

    # ══════════════════════════════════════════════
    #  4. Stateful Signal Generation V2
    # ══════════════════════════════════════════════
    raw_arr = _generate_signals_v2(
        close_arr=close.values.astype(np.float64),
        nw_arr=nw.values.astype(np.float64),
        upper_arr=upper.values.astype(np.float64),
        lower_arr=lower.values.astype(np.float64),
        regime_trending_arr=regime_trending.values.astype(np.float64),
        regime_dir_arr=regime_dir.values.astype(np.float64),
        module_a_enabled=module_a,
        module_b_enabled=module_b,
        entry_mode=entry_mode,
        mr_exit_target=mr_exit_target,
        trend_scale=trend_scale,
        range_scale=range_scale,
        ltf_proxy_arr=(
            ltf_proxy.values.astype(np.float64) if ltf_proxy is not None else None
        ),
        momentum_long_arr=mom_long,
        momentum_short_arr=mom_short,
        min_pullback_depth=min_pullback_depth,
        reentry_lockout=reentry_lockout,
        volume_ok_arr=vol_expansion,
    )

    raw_signal = pd.Series(raw_arr, index=df.index)

    # ══════════════════════════════════════════════
    #  5. Signal Delay（消除 look-ahead bias）
    # ══════════════════════════════════════════════
    if signal_delay > 0:
        raw_signal = raw_signal.shift(signal_delay).fillna(0.0)

    # ══════════════════════════════════════════════
    #  6. Exit Rules (SL/TP/Cooldown)
    # ══════════════════════════════════════════════
    _sl = float(sl_atr) if sl_atr is not None else None
    _tp = float(tp_atr) if tp_atr is not None else None
    _trailing = float(trailing_atr) if trailing_atr is not None else None

    pos, exit_exec_prices = apply_exit_rules(
        df, raw_signal,
        stop_loss_atr=_sl,
        take_profit_atr=_tp,
        trailing_stop_atr=_trailing,
        atr_period=atr_period,
        cooldown_bars=cooldown,
        min_hold_bars=min_hold,
    )

    # ══════════════════════════════════════════════
    #  7. Max Holding Bars（可選）
    # ══════════════════════════════════════════════
    if max_hold > 0:
        pos = _apply_max_holding(pos, max_hold)

    # ══════════════════════════════════════════════
    #  8. 附加出場價格（消除 SL/TP look-ahead bias）
    # ══════════════════════════════════════════════
    pos.attrs["exit_exec_prices"] = exit_exec_prices

    return pos
