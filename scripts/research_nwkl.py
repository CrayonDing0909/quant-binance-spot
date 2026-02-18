#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════════
 Nadaraya-Watson Kernel Regression + Lorentzian Distance k-NN Classifier  v3.1
 Dynamic Volatility Envelope — Long-Only Mean-Reversion
═══════════════════════════════════════════════════════════════════════════════════

v3.1 Dynamic Volatility Scaling (extends v3.0):
    1. DYNAMIC ENVELOPE:    Multiplier adapts to ATR Z-score regime.
                            Low vol (Z < -1):  1.5× MAE  (catch shallow dips)
                            Normal:            2.0× MAE
                            High vol (Z > 2):  3.0× MAE  (avoid catching knives)
    2. LONG-ONLY ENFORCED:  All short logic removed; we only buy dips.
    3. PANIC OVERRIDE:      Enter Long IF (Price < LowerBand) AND
                            ((ADX < 30) OR (RSI < 12)).
    4. MARKET/POST-ONLY:    Supports both execution modes via CLI flags.
    5. 2× LEVERAGE:         Low exposure (~2.4%) → safe for 2× leverage.

Strategy Logic:
    1. Nadaraya-Watson Envelope (Causal / Non-Repainting):
       - Rational Quadratic Kernel smoothing of close price
       - Dynamic bands = Estimate ± multiplier(Z_vol) × MAE
       - Price below Lower Band → potential Long zone

    2. Dynamic Volatility Scaling (v3.1 NEW):
       - Z_vol = (ATR_t - μ(ATR_168)) / σ(ATR_168)
       - Quiet market (Z < -1) → tighter bands → catch bull market shallow dips
       - Panic/crash (Z > 2)   → wider bands  → avoid premature entries
       - Fixes "Bull Market Miss Rate" and "Bear Market Knife Catching"

    3. Lorentzian Distance k-NN Classifier (Machine Learning):
       - Features: RSI, CCI, ADX, True Range (z-score normalized)
       - Distance: d(x, y) = Σ ln(1 + |x_j - y_j|)
       - Predicts bullish → confirmation for Long entry

    4. Regime Filter + Panic Override:
       - Normal:  ADX < 30 → ranging → enter mean-reversion
       - Panic:   RSI < 12 → extreme oversold → override ADX barrier
       - Combined: (ADX < 30) OR (RSI < 12)

    5. Execution:
       - Market Orders (--no-limit-orders): entry at next bar open
       - Post-Only Limit (default): Limit Buy @ Low[signal_bar], 3-bar expiry

    6. Exit Rules:
       - TP: Price returns to NW Lower Band (tp_mode="band")
       - SL: Entry - 1.5 × ATR
       - Leverage: 2× applied to position PnL

Non-Repainting Guarantee:
    The Nadaraya-Watson estimator at time t uses ONLY data from [0, t].

Academic References:
    - Nadaraya (1964), Watson (1964): Kernel Regression Estimation
    - Rational Quadratic Kernel: Rasmussen & Williams (2006)
    - LuxAlgo (2023): Lorentzian Distance Classifier for TradingView

Usage:
    # Production candidate (2022-2023):
    python scripts/research_nwkl.py --symbol ETHUSDT --start 2022-01-01 --end 2023-12-31

    # Full period test:
    python scripts/research_nwkl.py --symbol ETHUSDT

    # Synthetic data:
    python scripts/research_nwkl.py --synthetic

Author: Quantitative Research Engineer
Date:   2026-02-19
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── Optional: numba for 10-50x speedup on core loops ──
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        """No-op decorator when numba is not installed."""
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

# ── Add project root to path for optional framework integration ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class NWKLConfig:
    """
    Strategy configuration v3.0 — Production Candidate.

    All parameters are tunable. Defaults are calibrated for ETHUSDT 1H.
    v3.0: Long-only + Panic Override + Post-Only + 2× Leverage.
    """
    # ── Kernel Regression ──
    kernel_bandwidth: float = 3.0       # h : smoothness
    kernel_alpha: float = 1.0           # α : tail weight (RQ kernel parameter)
    kernel_lookback: int = 100          # Max causal lookback
    envelope_multiplier: float = 2.0    # Band width = 2.0 × MAE (normal regime)
    envelope_window: int = 200          # Rolling window for MAE calculation

    # ── Dynamic Volatility Envelope (v3.1) ──
    dynamic_envelope: bool = True       # Enable dynamic volatility scaling
    vol_zscore_window: int = 168        # ATR Z-score window (168 = 1 week hourly)
    envelope_mult_low: float = 1.5     # Low vol regime: tighter bands (shallow dips)
    envelope_mult_high: float = 3.0    # High vol regime: wider bands (avoid knives)
    vol_zscore_low: float = -1.0       # Z_vol < -1.0 → Low vol (quiet market)
    vol_zscore_high: float = 2.0       # Z_vol >  2.0 → High vol (panic/crash)

    # ── Lorentzian Classifier ──
    knn_k: int = 8                      # Number of nearest neighbors
    training_window: int = 2000         # Training window (bars)
    prediction_horizon: int = 4         # Future bars for label generation

    # ── Feature Engineering ──
    rsi_period: int = 14
    cci_period: int = 20
    adx_period: int = 14
    feature_norm_window: int = 500      # Rolling z-score normalization window

    # ── Regime Filter + Panic Override (v3.0) ──
    adx_filter_enabled: bool = True     # ADX regime filter
    adx_filter_threshold: float = 30.0  # Normal: enter when ADX < 30
    rsi_panic_threshold: float = 12.0   # Panic: override ADX when RSI < 12

    # ── Execution: Post-Only Limit Orders (v3.0) ──
    use_limit_orders: bool = True       # ← v3.0: Post-Only limit orders
    limit_order_expiry: int = 3         # Cancel if not filled in 3 bars

    # ── Risk Management ──
    stop_loss_pct: float = 0.02         # 2% fixed stop loss (fallback)
    use_atr_stop: bool = True           # Use ATR-based stop instead
    atr_stop_multiplier: float = 1.5    # ATR stop multiplier
    atr_period: int = 14

    # ── Take Profit ──
    tp_mode: str = "band"               # "band" / "mean" / "atr"
    tp_atr_multiplier: float = 1.5      # ATR TP multiplier (for tp_mode="atr")

    # ── Leverage (v3.0) ──
    leverage: float = 2.0               # ← v3.0: 2× leverage (low exposure ≈ 2.4%)

    # ── Backtest ──
    initial_cash: float = 10_000.0
    fee_pct: float = 0.0002             # ← v3.0: 0.02% maker fee (Post-Only)
    slippage_pct: float = 0.0           # ← v3.0: no slippage for Post-Only
    signal_delay: int = 1               # 1 = execute on next bar's open (causal)
    direction: str = "long_only"        # ← v3.0: Long-only enforced

    @property
    def direction_code(self) -> int:
        """Convert direction string to integer for numba compatibility."""
        return {"both": 0, "long_only": 1, "short_only": 2}.get(self.direction, 0)

    @property
    def tp_mode_code(self) -> int:
        """Convert tp_mode string to integer for numba. 0=mean, 1=band, 2=atr."""
        return {"mean": 0, "band": 1, "atr": 2}.get(self.tp_mode, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

class FeatureEngine:
    """
    Calculate and normalize technical indicator features for the ML model.

    Features:
        1. RSI  — Momentum oscillator [0, 100]
        2. CCI  — Mean-deviation oscillator (unbounded)
        3. ADX  — Trend strength [0, 100]
        4. TR   — Volatility proxy (True Range)

    All features are z-score normalized using a rolling window to ensure
    scale-invariance for the Lorentzian distance metric.
    """

    @staticmethod
    def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """RSI using Wilder's smoothing (EMA with α = 1/period)."""
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).ewm(
            alpha=1 / period, min_periods=period, adjust=False
        ).mean()
        loss = (-delta.where(delta < 0, 0.0)).ewm(
            alpha=1 / period, min_periods=period, adjust=False
        ).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        rsi = rsi.where(~(loss == 0) | gain.isna(), 100.0)
        rsi = rsi.where(~(gain == 0) | loss.isna(), 0.0)
        return rsi.fillna(50.0)

    @staticmethod
    def calculate_cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        Commodity Channel Index.

        CCI = (TP - SMA(TP, n)) / (0.015 × Mean Deviation)
        where TP = (H + L + C) / 3
        """
        tp = (df["high"] + df["low"] + df["close"]) / 3.0
        sma = tp.rolling(period, min_periods=period).mean()
        mad = tp.rolling(period, min_periods=period).apply(
            lambda x: np.abs(x - x.mean()).mean(), raw=True
        )
        cci = (tp - sma) / (0.015 * mad)
        return cci.fillna(0.0)

    @staticmethod
    def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average Directional Index (ADX only, no DI± split)."""
        high, low, close = df["high"], df["low"], df["close"]
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = pd.Series(0.0, index=df.index)
        minus_dm = pd.Series(0.0, index=df.index)
        plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
        minus_dm[(down_move > up_move) & (down_move > 0)] = down_move

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)

        alpha = 1.0 / period
        atr = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        s_plus = plus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        s_minus = minus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

        plus_di = (s_plus / atr) * 100
        minus_di = (s_minus / atr) * 100
        di_sum = plus_di + minus_di
        dx = ((plus_di - minus_di).abs() / di_sum.replace(0, np.nan)) * 100
        adx = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        return adx.fillna(0.0)

    @staticmethod
    def calculate_true_range(df: pd.DataFrame) -> pd.Series:
        """True Range = max(H-L, |H-C_prev|, |L-C_prev|)."""
        return pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ], axis=1).max(axis=1)

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average True Range (Wilder's smoothing)."""
        tr = FeatureEngine.calculate_true_range(df)
        return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    @classmethod
    def build_features(
        cls,
        df: pd.DataFrame,
        config: NWKLConfig,
    ) -> pd.DataFrame:
        """
        Build normalized feature matrix.

        Returns DataFrame with columns: rsi, cci, adx, tr
        All z-score normalized using a rolling window.
        """
        logger.info("📐 Computing features: RSI, CCI, ADX, TR ...")

        features = pd.DataFrame({
            "rsi": cls.calculate_rsi(df["close"], config.rsi_period),
            "cci": cls.calculate_cci(df, config.cci_period),
            "adx": cls.calculate_adx(df, config.adx_period),
            "tr":  cls.calculate_true_range(df),
        }, index=df.index)

        # Rolling z-score normalization
        w = config.feature_norm_window
        for col in features.columns:
            roll_mean = features[col].rolling(w, min_periods=1).mean()
            roll_std = features[col].rolling(w, min_periods=1).std().replace(0, 1.0)
            features[col] = (features[col] - roll_mean) / roll_std

        features = features.fillna(0.0)
        logger.info(
            f"   Features shape: {features.shape}, "
            f"mean|z| = {features.abs().mean().mean():.3f}"
        )
        return features


# ═══════════════════════════════════════════════════════════════════════════════
# 3. NADARAYA-WATSON KERNEL REGRESSION (CAUSAL / NON-REPAINTING)
# ═══════════════════════════════════════════════════════════════════════════════

@njit(cache=True)
def _compute_rq_kernel_weights(lookback: int, bandwidth: float, alpha: float):
    """
    Precompute Rational Quadratic Kernel weights.

    K(d) = (1 + d² / (2·α·h²))^(-α)

    where d = temporal distance (0 = most recent, lookback = oldest).

    Properties:
        - Gaussian-like for α → ∞
        - Heavier tails (more weight to distant points) for small α
        - Bandwidth h controls overall width
    """
    weights = np.empty(lookback + 1)
    for d in range(lookback + 1):
        weights[d] = (1.0 + (d * d) / (2.0 * alpha * bandwidth * bandwidth)) ** (-alpha)
    return weights


@njit(cache=True)
def _causal_kernel_regression_numba(close_arr, weights, lookback):
    """
    Causal (non-repainting) Nadaraya-Watson kernel regression.

    ═══════════════════════════════════════════════════
    NON-REPAINTING GUARANTEE:
      At time t, ONLY close[0..t] is used.
      The estimate at bar t will NEVER change when
      future bars arrive.
    ═══════════════════════════════════════════════════

    Implementation:
        ŷ(t) = Σᵢ K(t-i) · close[i]  /  Σᵢ K(t-i)
        where i ∈ [max(0, t-lookback), t]

    For t < lookback (expanding window), fewer bars are used.
    For t ≥ lookback (steady state), exactly lookback+1 bars are used.
    """
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


def _causal_kernel_regression_numpy(close_arr, weights, lookback):
    """Pure numpy fallback (slower, ~2-5s for 50k bars)."""
    n = len(close_arr)
    estimate = np.empty(n)
    estimate[0] = close_arr[0]

    for t in range(1, n):
        w = min(t + 1, lookback + 1)
        window = close_arr[max(0, t - lookback): t + 1][::-1]
        wts = weights[:w]
        estimate[t] = np.dot(wts, window) / np.sum(wts)

    return estimate


class NadarayaWatsonEstimator:
    """
    Causal (Non-Repainting) Nadaraya-Watson Kernel Regression.

    Produces:
        - estimate: kernel regression line (smoothed price)
        - upper_band: estimate + multiplier × MAE
        - lower_band: estimate - multiplier × MAE

    v3.1: Dynamic Volatility Scaling
        The envelope multiplier adapts to the volatility regime via
        the rolling Z-score of ATR:
            Z_vol < -1.0 (quiet)  → multiplier = 1.5  (catch shallow dips)
            Z_vol >  2.0 (panic)  → multiplier = 3.0  (avoid catching knives)
            otherwise (normal)    → multiplier = 2.0

    The envelope bands define "overbought" and "oversold" zones
    relative to the non-parametric trend estimate.
    """

    def __init__(self, config: NWKLConfig):
        self.config = config
        self.weights = _compute_rq_kernel_weights(
            config.kernel_lookback,
            config.kernel_bandwidth,
            config.kernel_alpha,
        )

    def fit(self, close: pd.Series, atr: pd.Series) -> pd.DataFrame:
        """
        Compute NW estimate and envelope bands.

        Args:
            close: close price series
            atr:   ATR(14) series — used for dynamic volatility scaling

        Returns DataFrame with columns:
            nw_estimate, upper_band, lower_band, mae, dynamic_mult, vol_zscore
        """
        close_arr = close.values.astype(np.float64)
        cfg = self.config

        logger.info(
            f"🔬 Computing causal NW regression "
            f"(h={cfg.kernel_bandwidth}, α={cfg.kernel_alpha}, "
            f"lookback={cfg.kernel_lookback}) ..."
        )

        t0 = time.time()
        if HAS_NUMBA:
            estimate = _causal_kernel_regression_numba(
                close_arr, self.weights, cfg.kernel_lookback
            )
        else:
            logger.warning("⚠️  numba not available — using numpy fallback (slower)")
            estimate = _causal_kernel_regression_numpy(
                close_arr, self.weights, cfg.kernel_lookback
            )
        elapsed = time.time() - t0
        logger.info(f"   NW regression done in {elapsed:.2f}s ({len(close_arr)} bars)")

        estimate_series = pd.Series(estimate, index=close.index, name="nw_estimate")

        # MAE-based envelope
        residuals = (close - estimate_series).abs()
        mae = residuals.rolling(
            window=cfg.envelope_window, min_periods=1
        ).mean()

        # ── v3.1: Dynamic Volatility Scaling ──
        if cfg.dynamic_envelope:
            # Rolling Z-score of ATR over vol_zscore_window (e.g. 168 = 1 week)
            w = cfg.vol_zscore_window
            atr_mean = atr.rolling(w, min_periods=1).mean()
            atr_std = atr.rolling(w, min_periods=1).std().replace(0, 1.0)
            vol_zscore = (atr - atr_mean) / atr_std
            vol_zscore = vol_zscore.fillna(0.0)

            # Regime map: Z_vol → multiplier
            dynamic_mult = pd.Series(
                cfg.envelope_multiplier, index=close.index, dtype=np.float64,
            )
            dynamic_mult[vol_zscore < cfg.vol_zscore_low] = cfg.envelope_mult_low
            dynamic_mult[vol_zscore > cfg.vol_zscore_high] = cfg.envelope_mult_high

            upper = estimate_series + dynamic_mult * mae
            lower = estimate_series - dynamic_mult * mae

            # Regime statistics
            n_low = (vol_zscore < cfg.vol_zscore_low).sum()
            n_high = (vol_zscore > cfg.vol_zscore_high).sum()
            n_normal = len(vol_zscore) - n_low - n_high
            logger.info(
                f"   Dynamic Envelope v3.1: "
                f"Low-vol({cfg.envelope_mult_low}×)={n_low} bars, "
                f"Normal({cfg.envelope_multiplier}×)={n_normal} bars, "
                f"High-vol({cfg.envelope_mult_high}×)={n_high} bars"
            )
        else:
            dynamic_mult = pd.Series(
                cfg.envelope_multiplier, index=close.index, dtype=np.float64,
            )
            vol_zscore = pd.Series(0.0, index=close.index)
            upper = estimate_series + cfg.envelope_multiplier * mae
            lower = estimate_series - cfg.envelope_multiplier * mae

        logger.info(
            f"   Envelope: mean mult={dynamic_mult.mean():.2f}× MAE, "
            f"mean band width={((upper - lower) / close).mean() * 100:.2f}%"
        )

        return pd.DataFrame({
            "nw_estimate":  estimate_series,
            "upper_band":   upper,
            "lower_band":   lower,
            "mae":          mae,
            "dynamic_mult": dynamic_mult,
            "vol_zscore":   vol_zscore,
        }, index=close.index)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. LORENTZIAN DISTANCE k-NN CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════

@njit(cache=True)
def _lorentzian_knn_numba(features, close_arr, k, training_window, prediction_horizon):
    """
    Lorentzian Distance k-NN Classifier (numba-accelerated).

    Distance metric:
        d(x, y) = Σⱼ ln(1 + |xⱼ - yⱼ|)

    Properties vs Euclidean:
        - Logarithmic growth → robust to outliers
        - Better discrimination for nearby points
        - Naturally handles features of different magnitudes

    Training Logic:
        At time t, the training window is [train_start, t - prediction_horizon].
        This ensures all outcome labels are KNOWN at time t (no future leak).
        For each training sample i, the label is:
            +1 if close[i + prediction_horizon] > close[i]  (bullish)
            -1 otherwise  (bearish)

    Prediction:
        Majority vote of the k nearest neighbors' labels.
    """
    n = features.shape[0]
    n_features = features.shape[1]
    predictions = np.zeros(n)

    min_start = training_window + prediction_horizon + 1

    for t in range(min_start, n):
        # Training window: ensure all labels are known
        train_end = t - prediction_horizon
        train_start = max(0, train_end - training_window)
        n_train = train_end - train_start

        if n_train < k:
            continue

        # Compute Lorentzian distances & labels
        distances = np.empty(n_train)
        labels = np.empty(n_train)

        for i in range(n_train):
            idx = train_start + i
            # Lorentzian distance
            dist = 0.0
            for f in range(n_features):
                dist += np.log(1.0 + np.abs(features[t, f] - features[idx, f]))
            distances[i] = dist

            # Outcome label
            future_idx = idx + prediction_horizon
            if future_idx < n:
                labels[i] = 1.0 if close_arr[future_idx] > close_arr[idx] else -1.0
            else:
                labels[i] = 0.0

        # Partial sort: find k smallest distances
        # Use selection algorithm (efficient for small k)
        k_actual = min(k, n_train)
        for ki in range(k_actual):
            min_idx = ki
            for j in range(ki + 1, n_train):
                if distances[j] < distances[min_idx]:
                    min_idx = j
            # Swap
            if min_idx != ki:
                distances[ki], distances[min_idx] = distances[min_idx], distances[ki]
                labels[ki], labels[min_idx] = labels[min_idx], labels[ki]

        # Majority vote
        vote = 0.0
        for ki in range(k_actual):
            vote += labels[ki]

        if vote > 0:
            predictions[t] = 1.0
        elif vote < 0:
            predictions[t] = -1.0
        # else: 0 (tie → no prediction)

    return predictions


def _lorentzian_knn_numpy(features_arr, close_arr, k, training_window, prediction_horizon):
    """Pure numpy fallback (vectorized inner loop, Python outer loop)."""
    n = features_arr.shape[0]
    predictions = np.zeros(n)
    min_start = training_window + prediction_horizon + 1

    for t in range(min_start, n):
        train_end = t - prediction_horizon
        train_start = max(0, train_end - training_window)
        n_train = train_end - train_start

        if n_train < k:
            continue

        # Vectorized distance computation
        train_features = features_arr[train_start:train_end]  # (n_train, n_features)
        current = features_arr[t]  # (n_features,)
        distances = np.sum(np.log(1.0 + np.abs(train_features - current)), axis=1)

        # Labels
        future_close = close_arr[train_start + prediction_horizon: train_end + prediction_horizon]
        past_close = close_arr[train_start:train_end]
        labels = np.where(future_close > past_close, 1.0, -1.0)

        # k nearest
        k_actual = min(k, n_train)
        k_idx = np.argpartition(distances, k_actual)[:k_actual]
        vote = np.sum(labels[k_idx])

        predictions[t] = 1.0 if vote > 0 else (-1.0 if vote < 0 else 0.0)

    return predictions


class LorentzianClassifier:
    """
    k-Nearest Neighbors classifier using Lorentzian distance.

    Produces a prediction for each bar:
        +1 = Bullish (majority of neighbors saw price rise)
        -1 = Bearish (majority of neighbors saw price fall)
         0 = Tie / insufficient data
    """

    def __init__(self, config: NWKLConfig):
        self.config = config

    def predict(self, features: pd.DataFrame, close: pd.Series) -> pd.Series:
        """
        Generate ML predictions for each bar.

        Args:
            features: normalized feature DataFrame (RSI, CCI, ADX, TR)
            close:    close price series

        Returns:
            Prediction series: +1 (bullish), -1 (bearish), 0 (no signal)
        """
        features_arr = features.values.astype(np.float64)
        close_arr = close.values.astype(np.float64)
        cfg = self.config

        logger.info(
            f"🧠 Running Lorentzian k-NN classifier "
            f"(k={cfg.knn_k}, window={cfg.training_window}, "
            f"horizon={cfg.prediction_horizon}) ..."
        )

        t0 = time.time()
        if HAS_NUMBA:
            preds = _lorentzian_knn_numba(
                features_arr, close_arr,
                cfg.knn_k, cfg.training_window, cfg.prediction_horizon,
            )
        else:
            logger.warning("⚠️  numba not available — using numpy fallback (slower)")
            preds = _lorentzian_knn_numpy(
                features_arr, close_arr,
                cfg.knn_k, cfg.training_window, cfg.prediction_horizon,
            )
        elapsed = time.time() - t0
        logger.info(f"   Classifier done in {elapsed:.2f}s")

        result = pd.Series(preds, index=close.index, name="ml_prediction")

        # Summary statistics
        bullish = (result == 1).sum()
        bearish = (result == -1).sum()
        neutral = (result == 0).sum()
        logger.info(
            f"   Predictions: {bullish} bullish, {bearish} bearish, {neutral} neutral"
        )

        return result


# ═══════════════════════════════════════════════════════════════════════════════
# 5. COMBINED STRATEGY
# ═══════════════════════════════════════════════════════════════════════════════

class NWKLStrategy:
    """
    Combined Nadaraya-Watson + Lorentzian Classifier Strategy  v3.0.

    v3.0 — Production Candidate: Long-Only + Panic Override.

    Entry Rules:
        Long:  close < NW Lower Band  AND  ML = Bullish
               AND  ( ADX < 30  OR  RSI < 12 )
               → Post-Only Limit Buy @ Low[signal_bar], 3-bar expiry

    Exit Rules:
        TP: Price returns to NW Lower Band (tp_mode="band")
        SL: Entry - 1.5 × ATR
    """

    def __init__(self, config: NWKLConfig):
        self.config = config
        self.nw = NadarayaWatsonEstimator(config)
        self.classifier = LorentzianClassifier(config)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all signals and indicators for the strategy.

        Returns DataFrame with columns:
            nw_estimate, upper_band, lower_band, mae, dynamic_mult, vol_zscore,
            ml_prediction, atr, adx_raw, rsi_raw,
            raw_long_signal
        """
        logger.info("=" * 60)
        logger.info("🚀 NW-KL Strategy v3.1: Generating signals (LONG-ONLY + Dynamic Envelope) ...")
        logger.info("=" * 60)

        # Step 1: ATR (needed for both dynamic envelope and stop-loss)
        atr = FeatureEngine.calculate_atr(df, self.config.atr_period)

        # Step 2: Kernel Regression + Dynamic Volatility Envelope (v3.1)
        nw_df = self.nw.fit(df["close"], atr)

        # Step 3: Feature Engineering
        features = FeatureEngine.build_features(df, self.config)

        # Step 4: ML Classifier
        ml_pred = self.classifier.predict(features, df["close"])

        # Step 5: Raw ADX for regime filter
        adx_raw = FeatureEngine.calculate_adx(df, self.config.adx_period)

        # Step 6: Raw RSI for panic override
        rsi_raw = FeatureEngine.calculate_rsi(df["close"], self.config.rsi_period)

        # Step 7: Raw long entry signals (dynamic bands applied)
        close = df["close"]
        raw_long = (close < nw_df["lower_band"]) & (ml_pred == 1.0)

        # Step 8: Apply ADX + Panic Override filter
        if self.config.adx_filter_enabled:
            adx_ok = adx_raw < self.config.adx_filter_threshold
            rsi_panic = rsi_raw < self.config.rsi_panic_threshold
            regime_ok = adx_ok | rsi_panic  # Panic Override!
            filtered_long = raw_long & regime_ok

            n_adx_only = (raw_long & ~adx_ok).sum()
            n_panic_saved = (raw_long & ~adx_ok & rsi_panic).sum()
            n_filtered_out = (raw_long.sum() - filtered_long.sum())
            logger.info(
                f"🛡️  ADX filter (< {self.config.adx_filter_threshold}): "
                f"blocked {n_adx_only} signals"
            )
            logger.info(
                f"🚨 Panic Override (RSI < {self.config.rsi_panic_threshold}): "
                f"rescued {n_panic_saved} signals"
            )
            logger.info(
                f"   Net filtered out: {n_filtered_out} signals"
            )
        else:
            filtered_long = raw_long

        signals = nw_df.copy()
        signals["ml_prediction"] = ml_pred
        signals["atr"] = atr
        signals["adx_raw"] = adx_raw
        signals["rsi_raw"] = rsi_raw
        signals["raw_long_signal"] = filtered_long.astype(float)

        # Signal statistics
        n_long = filtered_long.sum()
        logger.info(f"📊 Filtered signals: {n_long} long entries (short disabled)")

        return signals


# ═══════════════════════════════════════════════════════════════════════════════
# 6. BACKTESTING ENGINE (v2: Limit Orders + Wide ATR Stop)
# ═══════════════════════════════════════════════════════════════════════════════

@njit(cache=True)
def _backtest_loop_numba(
    open_arr, high_arr, low_arr, close_arr,
    nw_estimate, upper_band, lower_band,
    ml_prediction, atr_arr, adx_arr, rsi_arr,
    stop_loss_pct, atr_stop_mult, use_atr_stop,
    signal_delay,
    fee_pct, slippage_pct, initial_cash,
    adx_filter_enabled, adx_threshold,
    rsi_panic_threshold,
    use_limit_orders, limit_order_expiry,
    tp_mode, tp_atr_mult,
    leverage,
):
    """
    Event-driven backtesting loop v3.0 — Long-Only + Panic Override + Leverage.

    v3.0 changes:
        - LONG-ONLY: all short logic removed
        - Panic Override: enter if (ADX < 30) OR (RSI < 12)
        - Post-Only: limit orders @ Low[signal_bar], 0.02% maker fee
        - Leverage: 2× applied to position PnL

    State Machine:
        FLAT → signal → PENDING → fill → IN_LONG → TP/SL → FLAT
        FLAT → signal → IN_LONG (if use_limit_orders=False)
        PENDING → expiry → FLAT (order cancelled)
    """
    n = len(close_arr)

    equity = np.empty(n)
    equity[0] = initial_cash
    positions = np.zeros(n)

    current_pos = 0.0
    entry_price = 0.0
    stop_price = 0.0
    tp_price = 0.0

    # Limit order pending state (long only)
    pending_active = False
    pending_price = 0.0
    pending_bars_left = 0

    n_trades = 0
    n_wins = 0
    gross_profit = 0.0
    gross_loss = 0.0

    for t in range(1, n):
        # ── Bar PnL from held long position (with leverage) ──
        if current_pos == 1.0 and close_arr[t - 1] > 0:
            bar_pnl = leverage * (close_arr[t] - close_arr[t - 1]) / close_arr[t - 1]
        else:
            bar_pnl = 0.0

        exit_triggered = False
        exit_price = close_arr[t]
        new_pos = current_pos
        fee_this_bar = 0.0

        # ── Check exit conditions for held LONG position ──
        if current_pos == 1.0:
            tp_hit = False
            if tp_mode == 0:  # mean
                tp_hit = close_arr[t] >= nw_estimate[t]
            elif tp_mode == 1:  # band
                tp_hit = close_arr[t] >= lower_band[t]
            elif tp_mode == 2:  # atr
                tp_hit = high_arr[t] >= tp_price
            if tp_hit:
                exit_triggered = True
                if tp_mode == 2:
                    exit_price = min(tp_price, high_arr[t])
                else:
                    exit_price = close_arr[t]
            # SL: intra-bar low breaches stop
            elif low_arr[t] <= stop_price:
                exit_triggered = True
                exit_price = stop_price
                bar_pnl = leverage * (stop_price - close_arr[t - 1]) / close_arr[t - 1]

        if exit_triggered:
            n_trades += 1
            trade_ret = leverage * (exit_price - entry_price) / entry_price
            if trade_ret > 0:
                n_wins += 1
                gross_profit += trade_ret
            else:
                gross_loss += abs(trade_ret)
            new_pos = 0.0
            pending_active = False
            fee_this_bar = fee_pct  # Post-Only exit also maker

        # ── Check pending limit order fill (long only) ──
        if new_pos == 0.0 and pending_active:
            if low_arr[t] <= pending_price:
                # Long limit buy filled at limit price
                new_pos = 1.0
                entry_price = pending_price
                atr_ref = atr_arr[max(0, t - 1)]
                if use_atr_stop:
                    stop_price = entry_price - atr_stop_mult * atr_ref
                else:
                    stop_price = entry_price * (1.0 - stop_loss_pct)
                if tp_mode == 2:
                    tp_price = entry_price + tp_atr_mult * atr_ref
                # PnL from fill price to close (with leverage)
                bar_pnl = leverage * (close_arr[t] - pending_price) / pending_price
                pending_active = False
                fee_this_bar = fee_pct  # Post-Only → maker fee only, no slippage
            else:
                pending_bars_left -= 1
                if pending_bars_left <= 0:
                    pending_active = False  # cancel order

        # ── Check LONG entry conditions (only if flat and no pending) ──
        if new_pos == 0.0 and not pending_active:
            sig_t = t - signal_delay
            if sig_t >= 0:
                # v3.0 Panic Override: (ADX < threshold) OR (RSI < panic)
                entry_allowed = True
                if adx_filter_enabled:
                    adx_ok = adx_arr[sig_t] < adx_threshold
                    rsi_panic = rsi_arr[sig_t] < rsi_panic_threshold
                    entry_allowed = adx_ok or rsi_panic

                if (entry_allowed
                        and close_arr[sig_t] < lower_band[sig_t]
                        and ml_prediction[sig_t] == 1.0):

                    if use_limit_orders:
                        # Post-Only Limit Buy @ Low[signal_bar]
                        pending_active = True
                        pending_price = low_arr[sig_t]
                        pending_bars_left = limit_order_expiry
                    else:
                        # Market order at open
                        new_pos = 1.0
                        entry_price = open_arr[t]
                        atr_ref = atr_arr[sig_t]
                        if use_atr_stop:
                            stop_price = entry_price - atr_stop_mult * atr_ref
                        else:
                            stop_price = entry_price * (1.0 - stop_loss_pct)
                        if tp_mode == 2:
                            tp_price = entry_price + tp_atr_mult * atr_ref
                        fee_this_bar = fee_pct + slippage_pct

        # ── Update equity ──
        equity[t] = equity[t - 1] * (1.0 + bar_pnl - fee_this_bar)

        positions[t] = new_pos
        current_pos = new_pos

    return equity, positions, n_trades, n_wins, gross_profit, gross_loss


def _backtest_loop_python(
    open_arr, high_arr, low_arr, close_arr,
    nw_estimate, upper_band, lower_band,
    ml_prediction, atr_arr, adx_arr, rsi_arr,
    stop_loss_pct, atr_stop_mult, use_atr_stop,
    signal_delay,
    fee_pct, slippage_pct, initial_cash,
    adx_filter_enabled, adx_threshold,
    rsi_panic_threshold,
    use_limit_orders, limit_order_expiry,
    tp_mode, tp_atr_mult,
    leverage,
):
    """Pure Python fallback v3.0 — Long-Only (same logic as numba version)."""
    n = len(close_arr)
    equity = np.empty(n)
    equity[0] = initial_cash
    positions = np.zeros(n)

    current_pos = 0.0
    entry_price = 0.0
    stop_price = 0.0
    tp_price = 0.0

    pending_active = False
    pending_price = 0.0
    pending_bars_left = 0

    n_trades = 0
    n_wins = 0
    gross_profit = 0.0
    gross_loss = 0.0

    for t in range(1, n):
        if current_pos == 1.0 and close_arr[t - 1] > 0:
            bar_pnl = leverage * (close_arr[t] - close_arr[t - 1]) / close_arr[t - 1]
        else:
            bar_pnl = 0.0

        exit_triggered = False
        exit_price = close_arr[t]
        new_pos = current_pos
        fee_this_bar = 0.0

        if current_pos == 1.0:
            tp_hit = False
            if tp_mode == 0:
                tp_hit = close_arr[t] >= nw_estimate[t]
            elif tp_mode == 1:
                tp_hit = close_arr[t] >= lower_band[t]
            elif tp_mode == 2:
                tp_hit = high_arr[t] >= tp_price
            if tp_hit:
                exit_triggered = True
                exit_price = close_arr[t] if tp_mode != 2 else min(tp_price, high_arr[t])
            elif low_arr[t] <= stop_price:
                exit_triggered = True
                exit_price = stop_price
                bar_pnl = leverage * (stop_price - close_arr[t - 1]) / close_arr[t - 1]

        if exit_triggered:
            n_trades += 1
            trade_ret = leverage * (exit_price - entry_price) / entry_price
            if trade_ret > 0:
                n_wins += 1
                gross_profit += trade_ret
            else:
                gross_loss += abs(trade_ret)
            new_pos = 0.0
            pending_active = False
            fee_this_bar = fee_pct

        # Check pending limit order fill (long only)
        if new_pos == 0.0 and pending_active:
            if low_arr[t] <= pending_price:
                new_pos = 1.0
                entry_price = pending_price
                atr_ref = atr_arr[max(0, t - 1)]
                if use_atr_stop:
                    stop_price = entry_price - atr_stop_mult * atr_ref
                else:
                    stop_price = entry_price * (1.0 - stop_loss_pct)
                if tp_mode == 2:
                    tp_price = entry_price + tp_atr_mult * atr_ref
                bar_pnl = leverage * (close_arr[t] - pending_price) / pending_price
                pending_active = False
                fee_this_bar = fee_pct
            else:
                pending_bars_left -= 1
                if pending_bars_left <= 0:
                    pending_active = False

        # Check LONG entry conditions
        if new_pos == 0.0 and not pending_active:
            sig_t = t - signal_delay
            if sig_t >= 0:
                entry_allowed = True
                if adx_filter_enabled:
                    adx_ok = adx_arr[sig_t] < adx_threshold
                    rsi_panic = rsi_arr[sig_t] < rsi_panic_threshold
                    entry_allowed = adx_ok or rsi_panic

                if (entry_allowed
                        and close_arr[sig_t] < lower_band[sig_t]
                        and ml_prediction[sig_t] == 1.0):
                    if use_limit_orders:
                        pending_active = True
                        pending_price = low_arr[sig_t]
                        pending_bars_left = limit_order_expiry
                    else:
                        new_pos = 1.0
                        entry_price = open_arr[t]
                        atr_ref = atr_arr[sig_t]
                        if use_atr_stop:
                            stop_price = entry_price - atr_stop_mult * atr_ref
                        else:
                            stop_price = entry_price * (1.0 - stop_loss_pct)
                        if tp_mode == 2:
                            tp_price = entry_price + tp_atr_mult * atr_ref
                        fee_this_bar = fee_pct + slippage_pct

        equity[t] = equity[t - 1] * (1.0 + bar_pnl - fee_this_bar)
        positions[t] = new_pos
        current_pos = new_pos

    return equity, positions, n_trades, n_wins, gross_profit, gross_loss


@dataclass
class BacktestResult:
    """Structured output from the backtesting engine."""
    equity: np.ndarray
    positions: np.ndarray
    n_trades: int
    n_wins: int
    gross_profit: float
    gross_loss: float
    config: NWKLConfig
    df: pd.DataFrame
    signals: pd.DataFrame


class BacktestEngine:
    """
    Event-driven backtesting engine v3.0 — Long-Only + Post-Only + Leverage.

    Features (v3.0):
        - LONG-ONLY: short logic removed
        - Panic Override: (ADX < 30) OR (RSI < 12)
        - Post-Only Limit Orders @ Low[signal_bar], 0.02% maker fee
        - 2× Leverage on position PnL
        - ATR-based stop-loss (1.5× ATR)
        - TP at NW band crossing (tp_mode="band")
    """

    def __init__(self, config: NWKLConfig):
        self.config = config

    def run(self, df: pd.DataFrame, signals: pd.DataFrame) -> BacktestResult:
        """Execute backtest and return results."""
        cfg = self.config

        logger.info("=" * 60)
        logger.info("💰 Running backtest v3.0 (LONG-ONLY) ...")
        logger.info(
            f"   Cash=${cfg.initial_cash:,.0f}, "
            f"Fee={cfg.fee_pct*100:.3f}% (maker), "
            f"Slip={cfg.slippage_pct*100:.3f}%, "
            f"Leverage={cfg.leverage:.0f}×"
        )
        logger.info(
            f"   ADX filter: {'ON (<' + str(cfg.adx_filter_threshold) + ')' if cfg.adx_filter_enabled else 'OFF'}, "
            f"Panic RSI: <{cfg.rsi_panic_threshold}, "
            f"Post-Only: {'ON (' + str(cfg.limit_order_expiry) + '-bar)' if cfg.use_limit_orders else 'OFF'}, "
            f"ATR SL: {cfg.atr_stop_multiplier}×"
        )
        logger.info("=" * 60)

        # Extract arrays
        open_arr = df["open"].values.astype(np.float64)
        high_arr = df["high"].values.astype(np.float64)
        low_arr = df["low"].values.astype(np.float64)
        close_arr = df["close"].values.astype(np.float64)
        nw_est = signals["nw_estimate"].values.astype(np.float64)
        upper = signals["upper_band"].values.astype(np.float64)
        lower = signals["lower_band"].values.astype(np.float64)
        ml_pred = signals["ml_prediction"].values.astype(np.float64)
        atr_arr = signals["atr"].values.astype(np.float64)
        adx_arr = signals["adx_raw"].values.astype(np.float64)
        rsi_arr = signals["rsi_raw"].values.astype(np.float64)

        t0 = time.time()
        backtest_fn = _backtest_loop_numba if HAS_NUMBA else _backtest_loop_python

        equity, positions, n_trades, n_wins, g_profit, g_loss = backtest_fn(
            open_arr, high_arr, low_arr, close_arr,
            nw_est, upper, lower, ml_pred, atr_arr, adx_arr, rsi_arr,
            cfg.stop_loss_pct, cfg.atr_stop_multiplier, cfg.use_atr_stop,
            cfg.signal_delay,
            cfg.fee_pct, cfg.slippage_pct, cfg.initial_cash,
            cfg.adx_filter_enabled, cfg.adx_filter_threshold,
            cfg.rsi_panic_threshold,
            cfg.use_limit_orders, cfg.limit_order_expiry,
            cfg.tp_mode_code, cfg.tp_atr_multiplier,
            cfg.leverage,
        )
        elapsed = time.time() - t0
        logger.info(f"   Backtest done in {elapsed:.2f}s")

        return BacktestResult(
            equity=equity,
            positions=positions,
            n_trades=n_trades,
            n_wins=n_wins,
            gross_profit=g_profit,
            gross_loss=g_loss,
            config=cfg,
            df=df,
            signals=signals,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 7. PERFORMANCE REPORT
# ═══════════════════════════════════════════════════════════════════════════════

class PerformanceReport:
    """
    Calculates and displays comprehensive strategy performance metrics.

    Metrics:
        - Total Return, Annualized Return
        - Sharpe Ratio, Sortino Ratio, Calmar Ratio
        - Max Drawdown (%)
        - Win Rate, Profit Factor
        - Number of Trades, Exposure Time
    """

    def __init__(self, result: BacktestResult):
        self.result = result
        self.metrics = self._compute_metrics()

    def _compute_metrics(self) -> dict:
        """Compute all performance metrics."""
        eq = self.result.equity
        r = self.result
        n = len(eq)

        # Returns
        eq_series = pd.Series(eq)
        returns = eq_series.pct_change().dropna()
        total_return = (eq[-1] / eq[0]) - 1.0

        # Annualized (assume 1h bars, 8760h/year)
        n_years = n / 8760.0
        ann_return = (1.0 + total_return) ** (1.0 / max(n_years, 0.01)) - 1.0

        # Sharpe (annualized)
        if returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(8760)
        else:
            sharpe = 0.0

        # Sortino (annualized, downside deviation)
        downside = returns[returns < 0]
        if len(downside) > 0 and downside.std() > 0:
            sortino = returns.mean() / downside.std() * np.sqrt(8760)
        else:
            sortino = 0.0

        # Max Drawdown
        peak = np.maximum.accumulate(eq)
        drawdown = (eq - peak) / peak
        max_dd = np.min(drawdown)

        # Calmar
        calmar = ann_return / abs(max_dd) if abs(max_dd) > 1e-10 else 0.0

        # Win Rate
        win_rate = r.n_wins / r.n_trades if r.n_trades > 0 else 0.0

        # Profit Factor
        profit_factor = (
            r.gross_profit / r.gross_loss
            if r.gross_loss > 0 else float("inf")
        )

        # Exposure (fraction of time in market)
        exposure = np.count_nonzero(r.positions) / n

        # Buy & Hold comparison
        bh_return = (
            self.result.df["close"].iloc[-1] / self.result.df["close"].iloc[0] - 1.0
        )

        # Limit order fill rate (approximate)
        n_signals = int(self.result.signals["raw_long_signal"].sum())

        return {
            "Total Return [%]": total_return * 100,
            "Annualized Return [%]": ann_return * 100,
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "Calmar Ratio": calmar,
            "Max Drawdown [%]": max_dd * 100,
            "Win Rate [%]": win_rate * 100,
            "Profit Factor": profit_factor,
            "Total Trades": r.n_trades,
            "Winning Trades": r.n_wins,
            "Losing Trades": r.n_trades - r.n_wins,
            "Exposure [%]": exposure * 100,
            "Buy & Hold Return [%]": bh_return * 100,
            "Final Equity": eq[-1],
            "Max Drawdown Duration": self._max_dd_duration(eq),
            "Total Signals": n_signals,
            "Fill Rate [%]": (r.n_trades / n_signals * 100) if n_signals > 0 else 0.0,
        }

    @staticmethod
    def _max_dd_duration(equity: np.ndarray) -> int:
        """Max number of bars in a drawdown."""
        peak = np.maximum.accumulate(equity)
        in_dd = equity < peak
        max_dur = 0
        current_dur = 0
        for v in in_dd:
            if v:
                current_dur += 1
                max_dur = max(max_dur, current_dur)
            else:
                current_dur = 0
        return max_dur

    def print_report(self) -> None:
        """Print formatted performance report."""
        m = self.metrics
        cfg = self.result.config
        print()
        print("═" * 65)
        print("  📊 PERFORMANCE REPORT — NW + Lorentzian Classifier v3.1")
        print("  🎯 Dynamic Volatility Envelope + Long-Only + Panic Override")
        print("═" * 65)
        if cfg.dynamic_envelope:
            print(f"  Envelope:   Dynamic [{cfg.envelope_mult_low}×/"
                  f"{cfg.envelope_multiplier}×/{cfg.envelope_mult_high}×]  |  "
                  f"ATR SL: {cfg.atr_stop_multiplier:.1f}×  |  "
                  f"Leverage: {cfg.leverage:.0f}×")
        else:
            print(f"  Envelope:   {cfg.envelope_multiplier:.1f}× MAE (fixed)  |  "
                  f"ATR SL: {cfg.atr_stop_multiplier:.1f}×  |  "
                  f"Leverage: {cfg.leverage:.0f}×")
        print(f"  ADX: {'<' + str(int(cfg.adx_filter_threshold)) if cfg.adx_filter_enabled else 'OFF'}  |  "
              f"Panic RSI: <{cfg.rsi_panic_threshold:.0f}  |  "
              f"Fee: {cfg.fee_pct*100:.2f}% (maker)")
        print(f"  Execution:  {'Post-Only Limit (' + str(cfg.limit_order_expiry) + '-bar)' if cfg.use_limit_orders else 'Market Orders'}")
        print("─" * 65)
        print(f"  Total Return:         {m['Total Return [%]']:>10.2f} %")
        print(f"  Annualized Return:    {m['Annualized Return [%]']:>10.2f} %")
        print(f"  Sharpe Ratio:         {m['Sharpe Ratio']:>10.3f}")
        print(f"  Sortino Ratio:        {m['Sortino Ratio']:>10.3f}")
        print(f"  Calmar Ratio:         {m['Calmar Ratio']:>10.3f}")
        print(f"  Max Drawdown:         {m['Max Drawdown [%]']:>10.2f} %")
        print(f"  Max DD Duration:      {m['Max Drawdown Duration']:>10d} bars")
        print("─" * 65)
        print(f"  Win Rate:             {m['Win Rate [%]']:>10.2f} %")
        pf_str = f"{m['Profit Factor']:.3f}" if m['Profit Factor'] != float('inf') else "∞"
        print(f"  Profit Factor:        {pf_str:>10s}")
        print(f"  Total Signals:        {m['Total Signals']:>10d}")
        print(f"  Total Trades:         {m['Total Trades']:>10d}")
        print(f"  Fill Rate:            {m['Fill Rate [%]']:>10.1f} %")
        print(f"  Winning Trades:       {m['Winning Trades']:>10d}")
        print(f"  Losing Trades:        {m['Losing Trades']:>10d}")
        print(f"  Exposure:             {m['Exposure [%]']:>10.2f} %")
        print("─" * 65)
        print(f"  Buy & Hold Return:    {m['Buy & Hold Return [%]']:>10.2f} %")
        print(f"  Final Equity:         ${m['Final Equity']:>12,.2f}")
        print("═" * 65)
        print()


# ═══════════════════════════════════════════════════════════════════════════════
# 8. PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def plot_results(
    result: BacktestResult,
    report: PerformanceReport,
    output_dir: Optional[Path] = None,
    show: bool = False,
) -> None:
    """
    Generate strategy visualization with 5 subplots.

    1. Price + NW Bands + Entry/Exit signals
    2. Lorentzian ML Predictions + ADX filter
    3. ADX with threshold line
    4. Equity curve vs Buy & Hold
    5. Drawdown
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    df = result.df
    sig = result.signals
    eq = result.equity
    pos = result.positions
    close = df["close"].values

    fig = plt.figure(figsize=(18, 20))
    gs = GridSpec(6, 1, height_ratios=[3, 1, 1, 1, 2, 1], hspace=0.15)

    dates = df.index

    # ── Subplot 1: Price + NW Bands + Signals ──
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(dates, close, color="#555555", linewidth=0.7, alpha=0.8, label="Close")
    ax1.plot(dates, sig["nw_estimate"].values, color="#2196F3",
             linewidth=1.5, label="NW Estimate")

    # Color bands by regime: green for tight (low vol), red for wide (high vol)
    ax1.fill_between(
        dates,
        sig["upper_band"].values,
        sig["lower_band"].values,
        alpha=0.12, color="#2196F3",
        label="NW Envelope (dynamic)" if result.config.dynamic_envelope
        else f"NW Envelope ({result.config.envelope_multiplier:.0f}× MAE)",
    )

    # Entry/Exit signals (Long only in v3.0)
    long_entries = (np.diff(pos, prepend=0) > 0) & (pos == 1)
    long_exits = (np.diff(pos, prepend=0) < 0) & (np.roll(pos, 1) == 1)

    ax1.scatter(dates[long_entries], close[long_entries],
                marker="^", color="#4CAF50", s=80, zorder=5, label="Long Entry")
    ax1.scatter(dates[long_exits], close[long_exits],
                marker="x", color="#F44336", s=50, zorder=5, alpha=0.6, label="Exit")

    ax1.set_title("Price + Nadaraya-Watson Dynamic Envelope + Trade Signals (v3.1 Long-Only)", fontsize=13)
    ax1.legend(loc="upper left", fontsize=8, ncol=3)
    ax1.set_ylabel("Price")
    ax1.grid(True, alpha=0.3)

    # ── Subplot 2: ML Predictions ──
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ml_pred = sig["ml_prediction"].values
    ax2.fill_between(
        dates, ml_pred, 0,
        where=ml_pred > 0, color="#4CAF50", alpha=0.4, label="Bullish"
    )
    ax2.fill_between(
        dates, ml_pred, 0,
        where=ml_pred < 0, color="#F44336", alpha=0.4, label="Bearish"
    )
    ax2.axhline(0, color="gray", linewidth=0.5)
    ax2.set_ylabel("ML Pred")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.set_ylim(-1.5, 1.5)
    ax2.grid(True, alpha=0.3)

    # ── Subplot 3: ADX with filter threshold ──
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    adx_vals = sig["adx_raw"].values
    ax3.plot(dates, adx_vals, color="#FF9800", linewidth=0.8, label="ADX")
    if result.config.adx_filter_enabled:
        ax3.axhline(
            result.config.adx_filter_threshold,
            color="#F44336", linewidth=1.0, linestyle="--",
            label=f"Threshold ({result.config.adx_filter_threshold})"
        )
        ax3.fill_between(
            dates, 0, result.config.adx_filter_threshold,
            alpha=0.08, color="#4CAF50"
        )
    ax3.set_ylabel("ADX")
    ax3.set_ylim(0, 80)
    ax3.legend(loc="upper left", fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ── Subplot 4: Volatility Z-Score + Dynamic Multiplier (v3.1) ──
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    vol_z = sig["vol_zscore"].values
    dyn_mult = sig["dynamic_mult"].values
    ax4_twin = ax4.twinx()
    ax4.plot(dates, vol_z, color="#9C27B0", linewidth=0.8, alpha=0.7, label="Vol Z-score")
    ax4.axhline(result.config.vol_zscore_low, color="#4CAF50", linewidth=0.8,
                linestyle="--", alpha=0.6, label=f"Z={result.config.vol_zscore_low}")
    ax4.axhline(result.config.vol_zscore_high, color="#F44336", linewidth=0.8,
                linestyle="--", alpha=0.6, label=f"Z={result.config.vol_zscore_high}")
    ax4.fill_between(dates, vol_z, result.config.vol_zscore_low,
                     where=vol_z < result.config.vol_zscore_low,
                     alpha=0.15, color="#4CAF50")
    ax4.fill_between(dates, vol_z, result.config.vol_zscore_high,
                     where=vol_z > result.config.vol_zscore_high,
                     alpha=0.15, color="#F44336")
    ax4_twin.step(dates, dyn_mult, color="#FF5722", linewidth=1.2, alpha=0.5,
                  where="post", label="Multiplier")
    ax4.set_ylabel("Vol Z-score", color="#9C27B0")
    ax4_twin.set_ylabel("Envelope Mult", color="#FF5722")
    ax4_twin.set_ylim(0.5, 4.0)
    ax4.legend(loc="upper left", fontsize=7)
    ax4_twin.legend(loc="upper right", fontsize=7)
    ax4.grid(True, alpha=0.3)

    # ── Subplot 5: Equity Curve ──
    ax5 = fig.add_subplot(gs[4], sharex=ax1)
    bh_equity = result.config.initial_cash * (close / close[0])
    ax5.plot(dates, eq, color="#2196F3", linewidth=1.5, label="Strategy v3.1")
    ax5.plot(dates, bh_equity, color="#999999", linewidth=1.0,
             linestyle="--", label="Buy & Hold")
    ax5.set_ylabel("Equity ($)")
    ax5.legend(loc="upper left", fontsize=9)
    ax5.grid(True, alpha=0.3)

    # ── Subplot 6: Drawdown ──
    ax6 = fig.add_subplot(gs[5], sharex=ax1)
    peak = np.maximum.accumulate(eq)
    dd_pct = (eq - peak) / peak * 100
    ax6.fill_between(dates, dd_pct, 0, color="#F44336", alpha=0.4)
    ax6.set_ylabel("Drawdown (%)")
    ax6.set_xlabel("Date")
    ax6.grid(True, alpha=0.3)

    env_label = (
        f"Dynamic [{result.config.envelope_mult_low}/"
        f"{result.config.envelope_multiplier}/"
        f"{result.config.envelope_mult_high}]"
        if result.config.dynamic_envelope
        else f"Fixed {result.config.envelope_multiplier}×"
    )
    plt.suptitle(
        f"NW + Lorentzian v3.1 (Long-Only {result.config.leverage:.0f}×, "
        f"Env={env_label})  |  "
        f"Sharpe={report.metrics['Sharpe Ratio']:.2f}  "
        f"MDD={report.metrics['Max Drawdown [%]']:.1f}%  "
        f"Trades={report.metrics['Total Trades']}  "
        f"Fill={report.metrics['Fill Rate [%]']:.0f}%",
        fontsize=13, fontweight="bold", y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "nwkl_v2_backtest.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"📈 Plot saved to {path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_project_data(symbol: str, interval: str = "1h") -> Optional[pd.DataFrame]:
    """Try to load data from the project's data directory."""
    try:
        from qtrade.data.storage import load_klines
        data_dir = PROJECT_ROOT / "data"
        # Search multiple possible path structures
        search_paths = [
            data_dir / "binance" / "futures" / interval / f"{symbol}.parquet",
            data_dir / "binance" / "spot" / interval / f"{symbol}.parquet",
            data_dir / "futures" / interval / f"{symbol}.parquet",
            data_dir / "spot" / interval / f"{symbol}.parquet",
        ]
        for path in search_paths:
            if path.exists():
                df = load_klines(path)
                logger.info(f"📦 Loaded {symbol} from {path} ({len(df)} bars)")
                return df
        logger.info(f"ℹ️  No project data found for {symbol}")
    except ImportError:
        logger.info("ℹ️  Project modules not available")
    return None


def generate_synthetic_data(
    n: int = 15000,
    initial_price: float = 3000.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data with GBM + jump diffusion.

    Calibrated to resemble ETH hourly data:
        - ~80% annualized volatility
        - ~10% annual drift
        - 1% chance of jumps per bar (3% jump std)
    """
    logger.info(f"🎲 Generating synthetic data ({n} bars, seed={seed}) ...")
    rng = np.random.RandomState(seed)

    # GBM parameters (hourly)
    mu = 0.10 / 8760
    sigma = 0.80 / np.sqrt(8760)

    # Jump diffusion
    jump_prob = 0.01
    jump_std = 0.03

    # Generate returns
    normal_ret = rng.normal(mu, sigma, n)
    jumps = rng.binomial(1, jump_prob, n) * rng.normal(0, jump_std, n)
    total_ret = normal_ret + jumps

    # Close prices
    close = initial_price * np.cumprod(1.0 + total_ret)

    # OHLV from close
    spread = (np.abs(total_ret) + sigma * 0.5) * close
    high = close + np.abs(rng.normal(0, 1, n)) * spread * 0.3
    low = close - np.abs(rng.normal(0, 1, n)) * spread * 0.3
    open_ = np.roll(close, 1) * (1 + rng.normal(0, sigma * 0.1, n))
    open_[0] = initial_price

    # Fix OHLC constraints
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))

    # Volume
    base_vol = 1e6
    volume = base_vol * np.exp(rng.normal(0, 0.5, n)) * (1 + np.abs(total_ret) * 10)

    dates = pd.date_range("2022-01-01", periods=n, freq="1h", tz="UTC")
    df = pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": close, "volume": volume,
    }, index=dates)

    logger.info(
        f"   Price range: ${df['close'].min():.0f} — ${df['close'].max():.0f}, "
        f"period: {dates[0].date()} → {dates[-1].date()}"
    )
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 10. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NW + Lorentzian Classifier Crypto Backtest v3.0 — Production Candidate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
v3.0 Production Candidate:
  • LONG-ONLY enforced — all short logic removed
  • Panic Override:  Enter if (ADX < 30) OR (RSI < 12)
  • Post-Only Limit: Entry @ Low[signal_bar], 0.02% maker fee
  • 2× Leverage:     Low exposure (~2.4%) → safe for 2× leverage

Examples:
  python scripts/research_nwkl.py --symbol ETHUSDT --start 2022-01-01 --end 2023-12-31
  python scripts/research_nwkl.py --symbol ETHUSDT --leverage 3
  python scripts/research_nwkl.py --synthetic
        """,
    )
    p.add_argument("--symbol", default="ETHUSDT", help="Trading pair (default: ETHUSDT)")
    p.add_argument("--interval", default="1h", help="K-line interval (default: 1h)")
    p.add_argument("--synthetic", action="store_true", help="Use synthetic data")

    # Strategy params
    p.add_argument("--bandwidth", type=float, default=3.0, help="Kernel bandwidth h")
    p.add_argument("--alpha", type=float, default=1.0, help="RQ kernel alpha")
    p.add_argument("--lookback", type=int, default=100, help="Kernel lookback")
    p.add_argument("--envelope-mult", type=float, default=2.0,
                   help="Envelope multiplier (× MAE, normal regime)")
    p.add_argument("--no-dynamic-envelope", action="store_true",
                   help="Disable dynamic volatility scaling (use fixed envelope)")
    p.add_argument("--vol-zscore-window", type=int, default=168,
                   help="ATR Z-score window (168=1wk hourly, default: 168)")
    p.add_argument("--envelope-mult-low", type=float, default=1.5,
                   help="Low vol regime multiplier (default: 1.5)")
    p.add_argument("--envelope-mult-high", type=float, default=3.0,
                   help="High vol regime multiplier (default: 3.0)")
    p.add_argument("--vol-z-low", type=float, default=-1.0,
                   help="Z-score threshold for low vol (default: -1.0)")
    p.add_argument("--vol-z-high", type=float, default=2.0,
                   help="Z-score threshold for high vol (default: 2.0)")
    p.add_argument("--knn-k", type=int, default=8, help="k for k-NN")
    p.add_argument("--train-window", type=int, default=2000, help="Training window")

    # v3.0: Regime filter + Panic Override
    p.add_argument("--no-adx-filter", action="store_true",
                   help="Disable ADX regime filter")
    p.add_argument("--adx-threshold", type=float, default=30.0,
                   help="ADX threshold for regime filter (default: 30)")
    p.add_argument("--rsi-panic", type=float, default=12.0,
                   help="RSI panic threshold — override ADX when RSI < this (default: 12)")

    # v3.0: Post-Only execution
    p.add_argument("--no-limit-orders", action="store_true",
                   help="Use market orders instead of Post-Only limit orders")
    p.add_argument("--limit-expiry", type=int, default=3,
                   help="Limit order expiry (bars, default: 3)")

    # Risk
    p.add_argument("--atr-sl-mult", type=float, default=1.5,
                   help="ATR stop-loss multiplier (default: 1.5)")
    p.add_argument("--tp-mode", default="band", choices=["band", "mean", "atr"],
                   help="Take-profit mode (default: band)")
    p.add_argument("--tp-atr-mult", type=float, default=1.5,
                   help="ATR take-profit multiplier (for --tp-mode atr)")

    # v3.0: Leverage
    p.add_argument("--leverage", type=float, default=2.0,
                   help="Leverage multiplier (default: 2.0)")

    # Backtest params
    p.add_argument("--fee", type=float, default=0.0002,
                   help="Trading fee rate (default: 0.0002 = 0.02%% maker)")
    p.add_argument("--slippage", type=float, default=0.0,
                   help="Slippage rate (default: 0.0 for Post-Only)")
    p.add_argument("--cash", type=float, default=10000.0, help="Initial cash")
    p.add_argument("--no-delay", action="store_true",
                   help="Disable signal delay (CAUTION: may introduce look-ahead bias)")

    # Date range
    p.add_argument("--start", type=str, default=None,
                   help="Start date for backtest (e.g. 2022-01-01)")
    p.add_argument("--end", type=str, default=None,
                   help="End date for backtest (e.g. 2023-12-31)")

    # Output
    p.add_argument("--output-dir", type=str, default=None,
                   help="Report output directory")
    p.add_argument("--show", action="store_true", help="Show plot interactively")

    return p.parse_args()


def main():
    args = parse_args()

    # v3.0: Post-Only limit orders ON by default (unless --no-limit-orders)
    use_limits = not args.no_limit_orders

    print()
    print("═" * 65)
    print("  🧪 Nadaraya-Watson + Lorentzian Distance Classifier  v3.1")
    print("  🎯 Dynamic Volatility Envelope + Long-Only + Panic Override")
    print("═" * 65)
    print(f"  numba: {'✅ available' if HAS_NUMBA else '❌ not installed (using numpy fallback)'}")
    print(f"  v3.1 config:")
    print(f"    • NW Kernel:     h={args.bandwidth:.1f}, lookback={args.lookback}")
    if not args.no_dynamic_envelope:
        print(f"    • Envelope:      DYNAMIC [{args.envelope_mult_low}×/"
              f"{args.envelope_mult}×/{args.envelope_mult_high}×] "
              f"(Z∈[{args.vol_z_low},{args.vol_z_high}], w={args.vol_zscore_window})")
    else:
        print(f"    • Envelope:      {args.envelope_mult:.1f}× MAE (fixed)")
    print(f"    • Direction:     LONG-ONLY (enforced)")
    print(f"    • ADX filter:    {'ON (<' + str(args.adx_threshold) + ')' if not args.no_adx_filter else 'OFF'}")
    print(f"    • Panic RSI:     <{args.rsi_panic:.0f} (override ADX)")
    print(f"    • Execution:     {'Post-Only Limit (' + str(args.limit_expiry) + '-bar)' if use_limits else 'Market Orders'}")
    print(f"    • Fee:           {args.fee*100:.2f}% {'(maker)' if use_limits else '(taker)'}")
    print(f"    • ATR SL:        {args.atr_sl_mult:.1f}×  |  TP: {args.tp_mode}")
    print(f"    • Leverage:      {args.leverage:.0f}×")
    print()

    # ── Configuration ──
    config = NWKLConfig(
        kernel_bandwidth=args.bandwidth,
        kernel_alpha=args.alpha,
        kernel_lookback=args.lookback,
        envelope_multiplier=args.envelope_mult,
        dynamic_envelope=not args.no_dynamic_envelope,
        vol_zscore_window=args.vol_zscore_window,
        envelope_mult_low=args.envelope_mult_low,
        envelope_mult_high=args.envelope_mult_high,
        vol_zscore_low=args.vol_z_low,
        vol_zscore_high=args.vol_z_high,
        knn_k=args.knn_k,
        training_window=args.train_window,
        initial_cash=args.cash,
        fee_pct=args.fee,
        slippage_pct=args.slippage,
        direction="long_only",  # v3.0: enforced
        signal_delay=0 if args.no_delay else 1,
        adx_filter_enabled=not args.no_adx_filter,
        adx_filter_threshold=args.adx_threshold,
        rsi_panic_threshold=args.rsi_panic,
        use_limit_orders=use_limits,
        limit_order_expiry=args.limit_expiry,
        atr_stop_multiplier=args.atr_sl_mult,
        tp_mode=args.tp_mode,
        tp_atr_multiplier=args.tp_atr_mult,
        leverage=args.leverage,
    )

    # ── Load Data ──
    if args.synthetic:
        df = generate_synthetic_data()
    else:
        df = load_project_data(args.symbol, args.interval)
        if df is None:
            logger.info("📭 No project data — falling back to synthetic data")
            df = generate_synthetic_data()

    # ── Date range filter ──
    if args.start:
        start_ts = pd.Timestamp(args.start, tz="UTC") if df.index.tz else pd.Timestamp(args.start)
        df = df[df.index >= start_ts]
        logger.info(f"  Start filter: >= {args.start} → {len(df)} bars")
    if args.end:
        end_ts = pd.Timestamp(args.end, tz="UTC") if df.index.tz else pd.Timestamp(args.end)
        df = df[df.index <= end_ts]
        logger.info(f"  End filter:   <= {args.end} → {len(df)} bars")

    # ── Run Strategy ──
    strategy = NWKLStrategy(config)
    signals = strategy.generate_signals(df)

    # ── Run Backtest ──
    engine = BacktestEngine(config)
    result = engine.run(df, signals)

    # ── Performance Report ──
    report = PerformanceReport(result)
    report.print_report()

    # ── Plot ──
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / "reports" / "research" / "nwkl_v3"

    plot_results(result, report, output_dir=output_dir, show=args.show)

    return result, report


if __name__ == "__main__":
    main()
