"""
WorldQuant Alpha 策略 — Price-Volume Microstructure Signals

Alpha 來源：
    Kakushadze 2016, "101 Formulaic Alphas" (arXiv:1601.00991)
    從 101 個 alpha 中篩選出在 crypto 1h candles 上 IC > 0.01 的信號。

IC Scan 結果 (2020-2026, BTC+ETH):
    wq012: sign(Δvolume) × (-Δclose)     IC=+0.011 (24h), pct+=65%, autocorr=0.06
    wq028: corr(adv20, low, 5) + midpoint-close  IC=+0.008 (24h), pct+=72%, autocorr=-0.07
    wq055: -corr(rank(%K), rank(volume), 6)  IC=-0.013 (24h), ETH stronger

信號特性：
    - Price-volume microstructure（與 TSMOM/華山的 derivatives sentiment 完全不同類型）
    - 低 autocorrelation → 快進快出，高 turnover
    - 預期與現有因子低相關

模式：
    A. 單一 alpha standalone
    B. Composite（多個 alpha 等權平均）

Changelog:
    v1 (2026-03-29): Initial — wq012, wq028, wq055 composite
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from qtrade.strategy.base import StrategyContext
from qtrade.strategy import register_strategy
from qtrade.utils.log import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════
#  Helper functions (adapted from WQ101 for single-asset)
# ═══════════════════════════════════════════════════════════

def _ts_corr(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    return x.rolling(window, min_periods=window // 2).corr(y)

def _ts_rank(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=window // 2).rank(pct=True)

def _ts_min(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=window // 2).min()

def _ts_max(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=window // 2).max()

def _rank(s: pd.Series, window: int = 720) -> pd.Series:
    """Time-series percentile rank (replacing cross-sectional rank)."""
    return s.rolling(window, min_periods=window // 4).rank(pct=True)


# ═══════════════════════════════════════════════════════════
#  Individual alpha computations
# ═══════════════════════════════════════════════════════════

def _compute_wq012(df: pd.DataFrame) -> pd.Series:
    """
    Alpha#12: sign(delta(volume, 1)) * (-1 * delta(close, 1))

    When volume increases and price drops → buy signal (+)
    When volume increases and price rises → sell signal (-)
    "Volume-direction reversal" — fade the move on volume expansion.

    IC: +0.011 (24h), pct+ 65%, autocorr 0.06
    """
    vol_delta = df["volume"].diff(1)
    price_delta = df["close"].diff(1)
    return np.sign(vol_delta) * (-1 * price_delta)


def _compute_wq028(df: pd.DataFrame) -> pd.Series:
    """
    Alpha#28: correlation(adv20, low, 5) + ((high + low) / 2 - close)

    Component 1: When 20d avg volume correlates with lows → institutional accumulation at dips
    Component 2: Midpoint vs close — close below midpoint = bearish intrabar
    Combined: positive when volume-low correlation is high AND close above midpoint.

    IC: +0.008 (24h), pct+ 72% (most stable signal), autocorr -0.07
    """
    adv20 = df["volume"].rolling(20, min_periods=10).mean()
    corr_component = _ts_corr(adv20, df["low"], 5)
    midpoint_component = (df["high"] + df["low"]) / 2 - df["close"]
    return corr_component + midpoint_component


def _compute_wq055(df: pd.DataFrame) -> pd.Series:
    """
    Alpha#55: -correlation(rank(%K), rank(volume), 6)

    %K = stochastic oscillator position within 12-bar range
    When %K and volume move together (high corr) → crowded momentum, fade it
    Negative correlation = divergence → trend continuation

    IC: -0.013 (24h), stronger on ETH (-0.022), autocorr 0.75
    """
    pct_k = (df["close"] - _ts_min(df["low"], 12)) / \
            (_ts_max(df["high"], 12) - _ts_min(df["low"], 12)).replace(0, np.nan)
    return -_ts_corr(_rank(pct_k, 168), _rank(df["volume"], 168), 6)


# ═══════════════════════════════════════════════════════════
#  Strategy
# ═══════════════════════════════════════════════════════════

@register_strategy("wq_alpha")
def generate_positions(
    df: pd.DataFrame,
    ctx: StrategyContext,
    params: dict,
) -> pd.Series:
    """
    WorldQuant Alpha composite strategy.

    Params:
        # Alpha selection
        use_wq012: bool = True
        use_wq028: bool = True
        use_wq055: bool = True

        # Signal processing
        pctrank_lookback: int = 720       # 30 days rolling percentile
        long_threshold: float = 0.65      # pctrank > this → long
        short_threshold: float = 0.35     # pctrank < this → short
        rebalance_hours: int = 4          # 4h rebalance (these are fast signals)

        # Exit design
        sl_atr_mult: float = 3.0
        sl_atr_lookback: int = 14
        max_hold_bars: int = 48           # 2 days max hold (short-term alphas)
        profit_target_pct: float = 0.03   # 3% profit target (smaller, faster)

        # Entry filter
        adx_gate_enabled: bool = False    # OFF by default (these alphas work in all regimes)
        persist_bars: int = 1             # less persist needed (fast signals)
        cooldown_bars: int = 4            # 4 bars × 4h = 16h cooldown
    """
    use_wq012 = bool(params.get("use_wq012", True))
    use_wq028 = bool(params.get("use_wq028", True))
    use_wq055 = bool(params.get("use_wq055", True))

    pctrank_lookback = int(params.get("pctrank_lookback", 720))
    long_threshold = float(params.get("long_threshold", 0.65))
    short_threshold = float(params.get("short_threshold", 0.35))
    rebalance_hours = int(params.get("rebalance_hours", 4))

    sl_atr_mult = float(params.get("sl_atr_mult", 3.0))
    sl_atr_lookback = int(params.get("sl_atr_lookback", 14))
    max_hold_bars = int(params.get("max_hold_bars", 48))
    profit_target_pct = float(params.get("profit_target_pct", 0.03))

    adx_gate_enabled = bool(params.get("adx_gate_enabled", False))
    adx_threshold = float(params.get("adx_threshold", 25.0))
    persist_bars = int(params.get("persist_bars", 1))
    cooldown_bars = int(params.get("cooldown_bars", 4))

    # ── 1. Compute selected alphas ──
    alpha_signals: list[pd.Series] = []

    if use_wq012:
        sig = _compute_wq012(df)
        alpha_signals.append(sig)
        logger.info(f"  WQ Alpha [{ctx.symbol}]: wq012 computed")

    if use_wq028:
        sig = _compute_wq028(df)
        alpha_signals.append(sig)
        logger.info(f"  WQ Alpha [{ctx.symbol}]: wq028 computed")

    if use_wq055:
        sig = _compute_wq055(df)
        alpha_signals.append(sig)
        logger.info(f"  WQ Alpha [{ctx.symbol}]: wq055 computed")

    if not alpha_signals:
        return pd.Series(0.0, index=df.index)

    # ── 2. Composite score (equal-weight z-scored) ──
    composite = pd.Series(0.0, index=df.index)
    for sig in alpha_signals:
        # Z-score each alpha independently before averaging
        roll = sig.rolling(pctrank_lookback, min_periods=pctrank_lookback // 4)
        z = (sig - roll.mean()) / roll.std().replace(0, np.nan)
        z = z.clip(-3, 3) / 3.0  # normalize to [-1, 1]
        composite += z / len(alpha_signals)

    # ── 3. Percentile rank for position sizing ──
    pctrank = composite.rolling(pctrank_lookback, min_periods=pctrank_lookback // 4).rank(pct=True)

    # ── 4. ADX gate (optional) ──
    adx_ok = pd.Series(True, index=df.index)
    if adx_gate_enabled:
        from qtrade.indicators import calculate_adx
        adx_result = calculate_adx(df, 14)
        adx_series = adx_result["ADX"] if isinstance(adx_result, (dict, pd.DataFrame)) else adx_result
        adx_ok = adx_series > adx_threshold

    # ── 5. ATR for stop-loss ──
    from qtrade.indicators import calculate_atr
    atr = calculate_atr(df, sl_atr_lookback)

    # ── 6. State machine ──
    n = len(df)
    pos_arr = np.zeros(n, dtype=np.float64)
    close_vals = df["close"].values
    high_vals = df["high"].values
    low_vals = df["low"].values
    pctrank_vals = pctrank.values
    adx_ok_vals = adx_ok.values
    atr_vals = atr.values

    current_pos = 0.0
    entry_price = 0.0
    entry_bar = -999
    bars_above = 0
    last_exit_bar = -999

    for i in range(n):
        pr = pctrank_vals[i]
        if np.isnan(pr):
            pos_arr[i] = current_pos
            continue

        # ── Exits (every bar) ──
        if current_pos != 0.0 and entry_price > 0:
            bars_held = i - entry_bar
            current_atr = atr_vals[i] if not np.isnan(atr_vals[i]) else 0.0

            # Profit target
            pnl_pct = (close_vals[i] - entry_price) / entry_price * current_pos
            if pnl_pct >= profit_target_pct:
                current_pos = 0.0
                entry_price = 0.0
                last_exit_bar = i

            # ATR stop-loss
            if current_pos != 0.0 and current_atr > 0:
                if current_pos > 0 and low_vals[i] <= entry_price - sl_atr_mult * current_atr:
                    current_pos = 0.0
                    entry_price = 0.0
                    last_exit_bar = i
                elif current_pos < 0 and high_vals[i] >= entry_price + sl_atr_mult * current_atr:
                    current_pos = 0.0
                    entry_price = 0.0
                    last_exit_bar = i

            # Time stop
            if current_pos != 0.0 and bars_held >= max_hold_bars:
                current_pos = 0.0
                entry_price = 0.0
                last_exit_bar = i

        # ── Entries (rebalance bars only) ──
        if current_pos == 0.0 and i % rebalance_hours == 0:
            if i - last_exit_bar < cooldown_bars:
                pos_arr[i] = 0.0
                continue

            if not adx_ok_vals[i]:
                bars_above = 0
                pos_arr[i] = 0.0
                continue

            if pr > long_threshold or (pr < short_threshold and ctx.can_short):
                bars_above += 1
            else:
                bars_above = 0

            if bars_above >= persist_bars:
                if pr > long_threshold:
                    current_pos = 1.0
                    entry_price = close_vals[i]
                    entry_bar = i
                    bars_above = 0
                elif pr < short_threshold and ctx.can_short:
                    current_pos = -1.0
                    entry_price = close_vals[i]
                    entry_bar = i
                    bars_above = 0

        pos_arr[i] = current_pos

    pos = pd.Series(pos_arr, index=df.index)

    tim = float((pos.abs() > 0.01).mean())
    logger.info(
        f"  WQ Alpha [{ctx.symbol}]: composite TIM={tim:.1%}, "
        f"long={float((pos > 0.01).mean()):.1%}, short={float((pos < -0.01).mean()):.1%}"
    )

    return pos
