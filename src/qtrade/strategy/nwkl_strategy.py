"""
NWKL v3.1（Nadaraya-Watson Kernel Regression + Lorentzian Distance k-NN Classifier）

v3.1 Dynamic Volatility Envelope — Long-Only + Panic Override

學術背景：
    - Nadaraya (1964), Watson (1964): Kernel Regression Estimation
    - Rational Quadratic Kernel: Rasmussen & Williams (2006)
    - Lorentzian Distance: d(x, y) = Σ ln(1 + |x_j - y_j|)

策略邏輯（v3.1 — Dynamic Envelope）：
    1. 快速 NW Kernel Regression（h=3, lookback=100）→ 響應更快的趨勢估計
    2. 動態波動率包絡線（v3.1 新增）：
       - Z_vol = (ATR_t - μ(ATR_168)) / σ(ATR_168)
       - 低波動（Z < -1）→ 1.5× MAE（牛市淺回調）
       - 正常            → 2.0× MAE
       - 高波動（Z > 2） → 3.0× MAE（避免接刀）
    3. Lorentzian Distance k-NN → ML 確認信號
    4. LONG-ONLY 強制：移除所有空倉邏輯
    5. Panic Override：(ADX < 30) OR (RSI < 12) → 極端超賣時覆蓋 ADX 限制
    6. 支持市價單 / Post-Only 限價單
    7. TP: 價格回到 NW Band → 更現實的止盈目標
    8. SL: 1.5× ATR → 適中的止損寬度
    9. 2× Leverage → 低曝光（~2.4%）下安全槓桿

Entry Rules:
    Long:  close < NW Lower Band AND ML = Bullish AND ((ADX < 30) OR (RSI < 12))

Exit Rules:
    TP: close >= NW Lower Band（Band crossing）
    SL: entry_price - 1.5 × ATR

Note:
    使用 auto_delay=False，因為倉位管理需要狀態機追蹤
    （限價單掛撤、SL/TP），信號延遲在循環中手動處理。
    框架自動處理 direction clip。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import StrategyContext
from . import register_strategy
from ..indicators import calculate_rsi, calculate_atr, calculate_adx
from ..indicators.cci import calculate_cci

# ── Optional numba ──
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator


# ══════════════════════════════════════════════════════════════
#  核心計算函數（numba 加速）
# ══════════════════════════════════════════════════════════════

@njit(cache=True)
def _rq_kernel_weights(lookback: int, bandwidth: float, alpha: float):
    """Rational Quadratic Kernel 權重預計算。"""
    weights = np.empty(lookback + 1)
    for d in range(lookback + 1):
        weights[d] = (1.0 + (d * d) / (2.0 * alpha * bandwidth * bandwidth)) ** (-alpha)
    return weights


@njit(cache=True)
def _causal_nw_regression(close_arr, weights, lookback):
    """因果 Nadaraya-Watson 核回歸（非重繪）。"""
    n = len(close_arr)
    estimate = np.empty(n)
    estimate[0] = close_arr[0]
    for t in range(1, n):
        window_size = min(t + 1, lookback + 1)
        num = 0.0
        den = 0.0
        for j in range(window_size):
            w = weights[j]
            num += w * close_arr[t - j]
            den += w
        estimate[t] = num / den
    return estimate


@njit(cache=True)
def _lorentzian_knn(features, close_arr, k, training_window, prediction_horizon):
    """Lorentzian Distance k-NN 分類器。"""
    n = features.shape[0]
    n_features = features.shape[1]
    predictions = np.zeros(n)
    min_start = training_window + prediction_horizon + 1

    for t in range(min_start, n):
        train_end = t - prediction_horizon
        train_start = max(0, train_end - training_window)
        n_train = train_end - train_start
        if n_train < k:
            continue

        distances = np.empty(n_train)
        labels = np.empty(n_train)

        for i in range(n_train):
            idx = train_start + i
            dist = 0.0
            for f in range(n_features):
                dist += np.log(1.0 + np.abs(features[t, f] - features[idx, f]))
            distances[i] = dist
            future_idx = idx + prediction_horizon
            if future_idx < n:
                labels[i] = 1.0 if close_arr[future_idx] > close_arr[idx] else -1.0
            else:
                labels[i] = 0.0

        k_actual = min(k, n_train)
        for ki in range(k_actual):
            min_idx = ki
            for j in range(ki + 1, n_train):
                if distances[j] < distances[min_idx]:
                    min_idx = j
            if min_idx != ki:
                distances[ki], distances[min_idx] = distances[min_idx], distances[ki]
                labels[ki], labels[min_idx] = labels[min_idx], labels[ki]

        vote = 0.0
        for ki in range(k_actual):
            vote += labels[ki]
        predictions[t] = 1.0 if vote > 0 else (-1.0 if vote < 0 else 0.0)

    return predictions


def _lorentzian_knn_numpy(features_arr, close_arr, k, training_window, prediction_horizon):
    """Numpy fallback for Lorentzian k-NN."""
    n = features_arr.shape[0]
    predictions = np.zeros(n)
    min_start = training_window + prediction_horizon + 1

    for t in range(min_start, n):
        train_end = t - prediction_horizon
        train_start = max(0, train_end - training_window)
        n_train = train_end - train_start
        if n_train < k:
            continue

        train_features = features_arr[train_start:train_end]
        current = features_arr[t]
        distances = np.sum(np.log(1.0 + np.abs(train_features - current)), axis=1)

        future_close = close_arr[train_start + prediction_horizon:train_end + prediction_horizon]
        past_close = close_arr[train_start:train_end]
        labels = np.where(future_close > past_close, 1.0, -1.0)

        k_actual = min(k, n_train)
        k_idx = np.argpartition(distances, k_actual)[:k_actual]
        vote = np.sum(labels[k_idx])
        predictions[t] = 1.0 if vote > 0 else (-1.0 if vote < 0 else 0.0)

    return predictions


# ══════════════════════════════════════════════════════════════
#  特徵工程
# ══════════════════════════════════════════════════════════════

def _build_features(
    df: pd.DataFrame,
    rsi_period: int = 14,
    cci_period: int = 20,
    adx_period: int = 14,
    norm_window: int = 500,
) -> np.ndarray:
    """構建 z-score 標準化的特徵矩陣。"""
    features = pd.DataFrame({
        "rsi": calculate_rsi(df["close"], rsi_period),
        "cci": calculate_cci(df, cci_period),
        "adx": calculate_adx(df, adx_period)["ADX"],
        "tr":  pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ], axis=1).max(axis=1),
    }, index=df.index)

    # Rolling z-score
    for col in features.columns:
        roll_mean = features[col].rolling(norm_window, min_periods=1).mean()
        roll_std = features[col].rolling(norm_window, min_periods=1).std().replace(0, 1.0)
        features[col] = (features[col] - roll_mean) / roll_std

    return features.fillna(0.0).values.astype(np.float64)


# ══════════════════════════════════════════════════════════════
#  位置管理狀態機（numba 加速版）
# ══════════════════════════════════════════════════════════════

# State constants for the position management state machine (v3.0: Long-Only)
_FLAT = 0
_PENDING_LONG = 1
_IN_LONG = 2


@njit(cache=True)
def _manage_positions_numba(
    close_arr, low_arr, high_arr, nw_arr, upper_band_arr, lower_band_arr,
    atr_arr, adx_arr, rsi_arr,
    long_signal_arr,
    adx_filter_enabled, adx_threshold, rsi_panic_threshold,
    use_limit_orders, limit_order_expiry,
    use_atr_stop, atr_stop_mult, stop_loss_pct,
    tp_mode, tp_atr_mult,
):
    """
    位置管理狀態機 v3.0（numba 加速）— Long-Only + Panic Override。

    狀態：
        FLAT → PENDING_LONG（Post-Only 限價單）→ IN_LONG（成交）→ FLAT（TP/SL）
        FLAT → IN_LONG（市價單，use_limit_orders=False）→ FLAT（TP/SL）

    Panic Override：(ADX < threshold) OR (RSI < rsi_panic_threshold)
    信號延遲：sig_i = i - 1（手動 delay = 1 bar）。
    限價單：Limit Buy @ Low[sig_i]。
    tp_mode: 0=mean（NW estimate），1=band（lower band），2=ATR
    """
    n = len(close_arr)
    pos = np.zeros(n)

    state = _FLAT
    pending_price = 0.0
    pending_bars_left = 0
    entry_price = 0.0
    stop_price = 0.0
    tp_price = 0.0

    for i in range(n):
        # ── 1. 檢查持倉的 TP / SL ──
        if state == _IN_LONG:
            tp_hit = False
            if tp_mode == 0:
                tp_hit = close_arr[i] >= nw_arr[i]
            elif tp_mode == 1:
                tp_hit = close_arr[i] >= lower_band_arr[i]
            elif tp_mode == 2:
                tp_hit = high_arr[i] >= tp_price
            if tp_hit:
                state = _FLAT
            elif low_arr[i] <= stop_price:
                state = _FLAT
            else:
                pos[i] = 1.0
                continue

        # ── 2. 檢查掛單是否成交 ──
        elif state == _PENDING_LONG:
            if low_arr[i] <= pending_price:
                state = _IN_LONG
                entry_price = pending_price
                atr_ref = atr_arr[max(0, i - 1)]
                if use_atr_stop:
                    stop_price = entry_price - atr_stop_mult * atr_ref
                else:
                    stop_price = entry_price * (1.0 - stop_loss_pct)
                if tp_mode == 2:
                    tp_price = entry_price + tp_atr_mult * atr_ref
                pos[i] = 1.0
                continue
            else:
                pending_bars_left -= 1
                if pending_bars_left <= 0:
                    state = _FLAT

        # ── 3. FLAT 狀態：檢查新信號（Long-Only + Panic Override） ──
        if state == _FLAT:
            sig_i = i - 1  # signal delay = 1 bar
            if sig_i < 0:
                continue

            # v3.0: Panic Override — (ADX < threshold) OR (RSI < panic)
            entry_allowed = True
            if adx_filter_enabled:
                adx_ok = adx_arr[sig_i] < adx_threshold
                rsi_panic = rsi_arr[sig_i] < rsi_panic_threshold
                entry_allowed = adx_ok or rsi_panic
            if not entry_allowed:
                continue

            if long_signal_arr[sig_i] == 1.0:
                if use_limit_orders:
                    state = _PENDING_LONG
                    pending_price = low_arr[sig_i]
                    pending_bars_left = limit_order_expiry
                else:
                    state = _IN_LONG
                    entry_price = close_arr[i]
                    atr_ref = atr_arr[sig_i]
                    if use_atr_stop:
                        stop_price = entry_price - atr_stop_mult * atr_ref
                    else:
                        stop_price = entry_price * (1.0 - stop_loss_pct)
                    if tp_mode == 2:
                        tp_price = entry_price + tp_atr_mult * atr_ref
                    pos[i] = 1.0

    return pos


def _manage_positions_python(
    close_arr, low_arr, high_arr, nw_arr, upper_band_arr, lower_band_arr,
    atr_arr, adx_arr, rsi_arr,
    long_signal_arr,
    adx_filter_enabled, adx_threshold, rsi_panic_threshold,
    use_limit_orders, limit_order_expiry,
    use_atr_stop, atr_stop_mult, stop_loss_pct,
    tp_mode, tp_atr_mult,
):
    """Pure Python fallback v3.0（邏輯與 numba 版本完全一致）— Long-Only。"""
    n = len(close_arr)
    pos = np.zeros(n)

    state = _FLAT
    pending_price = 0.0
    pending_bars_left = 0
    entry_price = 0.0
    stop_price = 0.0
    tp_price = 0.0

    for i in range(n):
        if state == _IN_LONG:
            tp_hit = False
            if tp_mode == 0:
                tp_hit = close_arr[i] >= nw_arr[i]
            elif tp_mode == 1:
                tp_hit = close_arr[i] >= lower_band_arr[i]
            elif tp_mode == 2:
                tp_hit = high_arr[i] >= tp_price
            if tp_hit:
                state = _FLAT
            elif low_arr[i] <= stop_price:
                state = _FLAT
            else:
                pos[i] = 1.0
                continue

        elif state == _PENDING_LONG:
            if low_arr[i] <= pending_price:
                state = _IN_LONG
                entry_price = pending_price
                atr_ref = atr_arr[max(0, i - 1)]
                if use_atr_stop:
                    stop_price = entry_price - atr_stop_mult * atr_ref
                else:
                    stop_price = entry_price * (1.0 - stop_loss_pct)
                if tp_mode == 2:
                    tp_price = entry_price + tp_atr_mult * atr_ref
                pos[i] = 1.0
                continue
            else:
                pending_bars_left -= 1
                if pending_bars_left <= 0:
                    state = _FLAT

        if state == _FLAT:
            sig_i = i - 1
            if sig_i < 0:
                continue

            # v3.0: Panic Override
            entry_allowed = True
            if adx_filter_enabled:
                adx_ok = adx_arr[sig_i] < adx_threshold
                rsi_panic = rsi_arr[sig_i] < rsi_panic_threshold
                entry_allowed = adx_ok or rsi_panic
            if not entry_allowed:
                continue

            if long_signal_arr[sig_i] == 1.0:
                if use_limit_orders:
                    state = _PENDING_LONG
                    pending_price = low_arr[sig_i]
                    pending_bars_left = limit_order_expiry
                else:
                    state = _IN_LONG
                    entry_price = close_arr[i]
                    atr_ref = atr_arr[sig_i]
                    if use_atr_stop:
                        stop_price = entry_price - atr_stop_mult * atr_ref
                    else:
                        stop_price = entry_price * (1.0 - stop_loss_pct)
                    if tp_mode == 2:
                        tp_price = entry_price + tp_atr_mult * atr_ref
                    pos[i] = 1.0

    return pos


# ══════════════════════════════════════════════════════════════
#  策略主函數（框架整合）
# ══════════════════════════════════════════════════════════════

@register_strategy("nwkl", auto_delay=False)
def generate_nwkl(df: pd.DataFrame, ctx: StrategyContext, params: dict) -> pd.Series:
    """
    Nadaraya-Watson + Lorentzian k-NN Classifier 策略 v3.1。

    v3.1 — Dynamic Volatility Envelope + Long-Only + Panic Override。

    Entry:
        Long:  close < NW Lower Band AND ML = Bullish AND ((ADX < 30) OR (RSI < 12))

    Exit (tp_mode):
        "band": TP close >= NW Lower Band（推薦）
        "mean": TP close >= NW Estimate
        "atr":  TP high >= entry + tp_atr_multiplier × ATR
        SL: entry - 1.5 × ATR

    v3.1 Dynamic Envelope:
        Z_vol = (ATR - μ(ATR, 168)) / σ(ATR, 168)
        low vol  (Z < -1.0) → multiplier = 1.5  (shallow dips)
        normal               → multiplier = 2.0
        high vol (Z >  2.0) → multiplier = 3.0  (avoid knives)

    使用 auto_delay=False → 信號延遲由狀態機手動處理。
    框架自動處理 direction clip（spot → [0,1]，futures → [-1,1]）。

    params:
        kernel_bandwidth:      NW 核帶寬（預設 3.0）
        kernel_alpha:          RQ 核 alpha 參數（預設 1.0）
        kernel_lookback:       最大回看窗口（預設 100）
        envelope_multiplier:   包絡線倍數 — 正常波動區（預設 2.0）
        envelope_window:       MAE 滾動窗口（預設 200）
        dynamic_envelope:      啟用動態波動率包絡線（預設 True）
        vol_zscore_window:     ATR Z-score 窗口（預設 168 = 1 週）
        envelope_mult_low:     低波動倍數（預設 1.5）
        envelope_mult_high:    高波動倍數（預設 3.0）
        vol_zscore_low:        低波動 Z-score 閾值（預設 -1.0）
        vol_zscore_high:       高波動 Z-score 閾值（預設 2.0）
        knn_k:                 k 近鄰數（預設 8）
        training_window:       訓練窗口大小（預設 2000）
        prediction_horizon:    預測 horizon（預設 4）
        rsi_period:            RSI 週期（預設 14）
        cci_period:            CCI 週期（預設 20）
        adx_period:            ADX 週期（預設 14）
        feature_norm_window:   特徵標準化窗口（預設 500）
        adx_filter_enabled:    啟用 ADX 過濾器（預設 True）
        adx_filter_threshold:  ADX 閾值（預設 30）
        rsi_panic_threshold:   RSI Panic Override 閾值（預設 12）
        use_limit_orders:      使用 Post-Only 限價單入場（預設 True）
        limit_order_expiry:    限價單到期 bars（預設 3）
        use_atr_stop:          使用 ATR 動態止損（預設 True）
        atr_stop_multiplier:   ATR 止損倍數（預設 1.5）
        atr_period:            ATR 週期（預設 14）
        stop_loss_pct:         固定止損 %（use_atr_stop=false 時使用，預設 0.02）
        tp_mode:               止盈模式："band"/"mean"/"atr"（預設 "band"）
        tp_atr_multiplier:     ATR 止盈倍數（tp_mode="atr" 時使用，預設 1.5）
    """
    # ── 1. 解析參數 (v3.1 預設) ──
    bandwidth = float(params.get("kernel_bandwidth", 3.0))
    alpha = float(params.get("kernel_alpha", 1.0))
    lookback = int(params.get("kernel_lookback", 100))
    envelope_mult = float(params.get("envelope_multiplier", 2.0))
    envelope_window = int(params.get("envelope_window", 200))

    # v3.1: Dynamic Volatility Envelope
    dynamic_envelope = bool(params.get("dynamic_envelope", True))
    vol_zscore_window = int(params.get("vol_zscore_window", 168))
    envelope_mult_low = float(params.get("envelope_mult_low", 1.5))
    envelope_mult_high = float(params.get("envelope_mult_high", 3.0))
    vol_zscore_low = float(params.get("vol_zscore_low", -1.0))
    vol_zscore_high = float(params.get("vol_zscore_high", 2.0))

    knn_k = int(params.get("knn_k", 8))
    train_window = int(params.get("training_window", 2000))
    pred_horizon = int(params.get("prediction_horizon", 4))

    rsi_period = int(params.get("rsi_period", 14))
    cci_period = int(params.get("cci_period", 20))
    adx_period = int(params.get("adx_period", 14))
    norm_window = int(params.get("feature_norm_window", 500))

    # ADX + Panic Override
    adx_filter_enabled = bool(params.get("adx_filter_enabled", True))
    adx_threshold = float(params.get("adx_filter_threshold", 30.0))
    rsi_panic_threshold = float(params.get("rsi_panic_threshold", 12.0))

    # Post-Only limit orders
    use_limit_orders = bool(params.get("use_limit_orders", True))
    limit_order_expiry = int(params.get("limit_order_expiry", 3))

    use_atr_stop = bool(params.get("use_atr_stop", True))
    atr_stop_mult = float(params.get("atr_stop_multiplier", 1.5))
    atr_period_val = int(params.get("atr_period", 14))
    stop_loss_pct = float(params.get("stop_loss_pct", 0.02))

    # TP mode
    tp_mode_str = str(params.get("tp_mode", "band")).lower()
    tp_mode_map = {"mean": 0, "band": 1, "atr": 2}
    tp_mode_int = tp_mode_map.get(tp_mode_str, 1)
    tp_atr_mult = float(params.get("tp_atr_multiplier", 1.5))

    close = df["close"]
    close_arr = close.values.astype(np.float64)

    # ── 2. NW Kernel Regression ──
    weights = _rq_kernel_weights(lookback, bandwidth, alpha)
    estimate = _causal_nw_regression(close_arr, weights, lookback)
    estimate_series = pd.Series(estimate, index=df.index)

    # MAE envelope
    residuals = (close - estimate_series).abs()
    mae = residuals.rolling(window=envelope_window, min_periods=1).mean()

    # v3.1: Dynamic Volatility Envelope
    if dynamic_envelope:
        atr_for_vol = calculate_atr(df, atr_period_val)
        w = vol_zscore_window
        atr_mean = atr_for_vol.rolling(w, min_periods=1).mean()
        atr_std = atr_for_vol.rolling(w, min_periods=1).std().replace(0, 1.0)
        vol_zscore = ((atr_for_vol - atr_mean) / atr_std).fillna(0.0)

        dyn_mult = pd.Series(envelope_mult, index=df.index, dtype=np.float64)
        dyn_mult[vol_zscore < vol_zscore_low] = envelope_mult_low
        dyn_mult[vol_zscore > vol_zscore_high] = envelope_mult_high

        upper_band = estimate_series + dyn_mult * mae
        lower_band = estimate_series - dyn_mult * mae
    else:
        upper_band = estimate_series + envelope_mult * mae
        lower_band = estimate_series - envelope_mult * mae

    # ── 3. Feature Engineering + ML ──
    features_arr = _build_features(df, rsi_period, cci_period, adx_period, norm_window)

    if HAS_NUMBA:
        ml_pred = _lorentzian_knn(features_arr, close_arr, knn_k, train_window, pred_horizon)
    else:
        ml_pred = _lorentzian_knn_numpy(features_arr, close_arr, knn_k, train_window, pred_horizon)

    ml_pred_series = pd.Series(ml_pred, index=df.index)

    # ── 4. Raw LONG entry signals (v3.0: Long-Only) ──
    long_signal = (close < lower_band) & (ml_pred_series == 1.0)

    # ── 5. 計算 ADX + RSI（用於 regime filter + panic override） ──
    adx_raw = calculate_adx(df, adx_period)["ADX"]
    rsi_raw = calculate_rsi(df["close"], rsi_period)

    # ── 6. 計算 ATR（用於止損） ──
    atr = calculate_atr(df, atr_period_val)

    # ── 7. 狀態機位置管理（Long-Only + Panic Override） ──
    low_arr = df["low"].values.astype(np.float64)
    high_arr = df["high"].values.astype(np.float64)
    nw_arr = estimate_series.values.astype(np.float64)
    upper_arr = upper_band.values.astype(np.float64)
    lower_arr = lower_band.values.astype(np.float64)
    atr_arr = atr.values.astype(np.float64)
    adx_arr_raw = adx_raw.fillna(100.0).values.astype(np.float64)
    rsi_arr_raw = rsi_raw.fillna(50.0).values.astype(np.float64)
    long_sig_arr = long_signal.astype(np.float64).values

    manage_fn = _manage_positions_numba if HAS_NUMBA else _manage_positions_python

    pos_arr = manage_fn(
        close_arr, low_arr, high_arr, nw_arr, upper_arr, lower_arr,
        atr_arr, adx_arr_raw, rsi_arr_raw,
        long_sig_arr,
        adx_filter_enabled, adx_threshold, rsi_panic_threshold,
        use_limit_orders, limit_order_expiry,
        use_atr_stop, atr_stop_mult, stop_loss_pct,
        tp_mode_int, tp_atr_mult,
    )

    return pd.Series(pos_arr, index=df.index)
