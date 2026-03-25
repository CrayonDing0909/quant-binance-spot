"""
LSR Contrarian 策略 — 散戶多空比極端值逆向交易

Alpha 來源：
    散戶 Long/Short Ratio（LSR）達到極端高/低時，
    群眾情緒過度一致，逆向交易有統計優勢。

信號定義（v3 — Satellite Mode）：
    標準化：EW pctrank（halflife=84, span=168）替代簡單 rolling pctrank
    Regime gate：ADX(14) > 25 時才允許入場
    入場：
        LSR EW pctrank < entry_lo 且 persist >= persist_bars → 做多
        LSR EW pctrank > entry_hi 且 persist >= persist_bars → 做空（long_only=False 時）
        出場後 cooldown_bars 內不重新入場
    出場（三擇一，先觸發者優先）：
        1. TP: midpoint 模式 → pr 穿越 0.50；opposite_extreme 模式 → pr 到對面極端
        2. SL: 價格觸及 entry ± sl_atr_mult * ATR
        3. 時間止損: 超過 max_hold_bars 強制出場
    方向：long_only=True（衛星角色）或 both

v3 vs v2 差異：
    - v3: EW pctrank(hl=84) 替代 rolling pctrank — 更佳的 2025-2026 IC
    - v3: ADX regime gate — 只在趨勢環境交易
    - v3: persist filter — 連續 N bars 極端才入場，減少假信號
    - v3: cooldown — 出場後不立即重入場，消除重複虧損
    - v3: midpoint TP — 在中性點止盈，MDD 改善 11pp
    - v3: long_only 模式 — 衛星角色只做多

參數（v3 新增）：
    use_ew_pctrank: bool = True     使用 EW pctrank（False → 退回 v2 rolling）
    ew_halflife: int = 84           EW pctrank 半衰期
    adx_gate_enabled: bool = True   ADX regime gate
    adx_period: int = 14            ADX 計算週期
    adx_threshold: float = 25.0     ADX 趨勢門檻
    persist_bars: int = 2           連續極端 bars 門檻
    cooldown_bars: int = 24         出場後冷卻期（bars）
    tp_mode: str = "midpoint"       止盈模式 "midpoint" | "opposite_extreme"
    long_only: bool = True          只做多（衛星角色）

參數（v2 保留）：
    lsr_type, lsr_window, entry_pctile, tp_pctile,
    sl_atr_mult (default 3.0), sl_atr_lookback, max_hold_bars (default 168),
    vol_scale_enabled, vol_target, vol_lookback

Research Evidence (v3):
    - Proposal: docs/research/20260325_lsr_contrarian_research_alignment_proposal.md
    - Cycle 1 (entry): notebooks/research/20260325_lsr_entry_timing_cycle1.ipynb
    - Cycle 2 (exit): scripts/research/lsr_exit_design_cycle2.py
    - Cycle 3 (role): scripts/research/lsr_portfolio_role_cycle3.py
    - Long-only satellite: net exp +0.645%, SR 1.52, 55.6 tr/yr, corr=0.256 to prod BTC
    - Symmetric standalone: net exp +0.407%, SR 1.47, 105.0 tr/yr, MDD -19.56%

Changelog:
    v1 (2026-02-26): Initial implementation from research notebook
    v2 (2026-02-27): Swing mode — TP at opposite extreme, ATR SL, time stop
    v3 (2026-03-25): Satellite mode — EW pctrank, ADX gate, persist, cooldown, midpoint TP, long-only
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .base import StrategyContext
from . import register_strategy

logger = logging.getLogger(__name__)


def _compute_lsr_pctrank(
    lsr: pd.Series,
    window: int = 168,
) -> pd.Series:
    """v2 rolling percentile rank (kept for backward compat)"""
    min_periods = max(window // 2, 24)
    pctrank = lsr.rolling(window, min_periods=min_periods).apply(
        lambda x: sp_stats.percentileofscore(x, x.iloc[-1]) / 100.0,
        raw=False,
    )
    return pctrank


def _compute_ew_pctrank(
    lsr: pd.Series,
    halflife: int = 84,
    span_equiv: int = 168,
) -> pd.Series:
    """
    v3 EW-weighted z-score → rolling percentile rank.

    Uses exponentially-weighted mean/std to emphasise recent LSR
    levels, then maps the resulting z-score into a [0, 1] percentile
    via a trailing rolling window of size *span_equiv*.
    """
    min_p = max(halflife // 2, 24)
    ew_mean = lsr.ewm(halflife=halflife, min_periods=min_p).mean()
    ew_std = lsr.ewm(halflife=halflife, min_periods=min_p).std()
    ew_z = (lsr - ew_mean) / ew_std.replace(0, np.nan)
    return ew_z.rolling(span_equiv, min_periods=min_p).apply(
        lambda x: sp_stats.percentileofscore(x, x.iloc[-1]) / 100.0,
        raw=False,
    )


def _compute_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """ADX indicator for regime gating."""
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr_s = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr_s
    minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr_s
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    return dx.ewm(span=period, adjust=False).mean()


def _consecutive_count(series: np.ndarray) -> np.ndarray:
    """Count consecutive True values; resets to 0 on False."""
    out = np.zeros(len(series), dtype=int)
    for i in range(len(series)):
        if series[i]:
            out[i] = out[i - 1] + 1 if i > 0 else 1
    return out


def _vol_scale(
    close: pd.Series,
    vol_target: float = 0.15,
    vol_lookback: int = 168,
) -> pd.Series:
    """
    波動率目標縮放（倉位根據波動率反比調整）

    Args:
        close: 收盤價序列
        vol_target: 年化波動率目標
        vol_lookback: 波動率計算窗口

    Returns:
        縮放因子序列，clip 到 [0.2, 2.0]
    """
    returns = close.pct_change()
    vol = returns.rolling(vol_lookback, min_periods=max(vol_lookback // 4, 10)).std() * np.sqrt(8760)
    vol = vol.replace(0, np.nan).ffill().fillna(vol_target)
    scale = (vol_target / vol).clip(0.2, 2.0)
    return scale


def _compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    lookback: int = 14,
) -> pd.Series:
    """計算 ATR（Average True Range）"""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(lookback, min_periods=max(lookback // 2, 5)).mean()


@register_strategy("lsr_contrarian")
def generate_lsr_contrarian(
    df: pd.DataFrame,
    ctx: StrategyContext,
    params: dict,
) -> pd.Series:
    """
    LSR 逆向策略 v3 — Satellite Mode

    v3 params (new):
        use_ew_pctrank:   True → EW pctrank(hl=84); False → v2 rolling pctrank
        ew_halflife:      EW pctrank 半衰期，預設 84
        adx_gate_enabled: ADX regime gate，預設 True
        adx_period:       ADX 週期，預設 14
        adx_threshold:    ADX 趨勢門檻，預設 25.0
        persist_bars:     入場需連續極端 bars 數，預設 2
        cooldown_bars:    出場後冷卻期，預設 24
        tp_mode:          "midpoint"（pr=0.50）或 "opposite_extreme"（v2 行為）
        long_only:        True → 只做多（衛星角色）

    v2 params (kept, defaults updated):
        lsr_type, lsr_window=168, entry_pctile=0.85, tp_pctile=0.25,
        sl_atr_mult=3.0, sl_atr_lookback=14, max_hold_bars=168,
        vol_scale_enabled=False, vol_target=0.15, vol_lookback=168,
        _data_dir
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    n = len(df)

    # ── v2 params (defaults updated for v3) ──
    lsr_type = str(params.get("lsr_type", "lsr"))
    lsr_window = int(params.get("lsr_window", 168))
    entry_pctile = float(params.get("entry_pctile", 0.85))
    tp_pctile = float(params.get("tp_pctile", 0.25))
    sl_atr_mult = float(params.get("sl_atr_mult", 3.0))
    sl_atr_lookback = int(params.get("sl_atr_lookback", 14))
    max_hold_bars = int(params.get("max_hold_bars", 168))
    vol_scale_enabled = bool(params.get("vol_scale_enabled", False))
    vol_target = float(params.get("vol_target", 0.15))
    vol_lookback = int(params.get("vol_lookback", 168))

    # ── v3 params ──
    use_ew_pctrank = bool(params.get("use_ew_pctrank", True))
    ew_halflife = int(params.get("ew_halflife", 84))
    adx_gate_enabled = bool(params.get("adx_gate_enabled", True))
    adx_period = int(params.get("adx_period", 14))
    adx_threshold = float(params.get("adx_threshold", 25.0))
    persist_bars = int(params.get("persist_bars", 2))
    cooldown_bars = int(params.get("cooldown_bars", 24))
    tp_mode = str(params.get("tp_mode", "midpoint"))
    long_only = bool(params.get("long_only", True))

    entry_hi = entry_pctile
    entry_lo = 1.0 - entry_pctile
    tp_for_short = tp_pctile
    tp_for_long = 1.0 - tp_pctile

    # ── 1. 取得 LSR 數據 ──
    lsr_series = ctx.get_derivative(lsr_type)

    if lsr_series is None:
        data_dir = params.get("_data_dir")
        if data_dir is not None:
            try:
                from ..data.long_short_ratio import load_lsr, align_lsr_to_klines
                from pathlib import Path

                data_dir_path = Path(data_dir)
                deriv_dir = data_dir_path / "binance" / "futures" / "derivatives"
                lsr_raw = load_lsr(ctx.symbol, lsr_type, data_dir=deriv_dir)
                if lsr_raw is not None and not lsr_raw.empty:
                    lsr_series = align_lsr_to_klines(lsr_raw, df.index, max_ffill_bars=2)
                    logger.info(f"  LSR Contrarian [{ctx.symbol}]: 自動載入 {lsr_type} ({len(lsr_raw)} rows)")
            except Exception as e:
                logger.warning(f"  LSR Contrarian [{ctx.symbol}]: LSR 自動載入失敗: {e}")

    if lsr_series is None:
        logger.warning(f"  ⚠️ LSR Contrarian [{ctx.symbol}]: no {lsr_type} data, returning flat")
        return pd.Series(0.0, index=df.index)

    lsr_aligned = lsr_series.reindex(df.index).ffill()

    lsr_coverage = (~lsr_aligned.isna()).mean()
    if lsr_coverage < 0.3:
        logger.warning(f"  ⚠️ LSR Contrarian [{ctx.symbol}]: LSR coverage {lsr_coverage:.1%} < 30%, returning flat")
        return pd.Series(0.0, index=df.index)

    # ── 2. Percentile rank (v3: EW or v2: rolling) ──
    if use_ew_pctrank:
        lsr_pctrank = _compute_ew_pctrank(lsr_aligned, ew_halflife, lsr_window)
    else:
        lsr_pctrank = _compute_lsr_pctrank(lsr_aligned, lsr_window)

    # ── 3. ATR ──
    atr = _compute_atr(high, low, close, sl_atr_lookback)

    # ── 4. ADX regime gate (v3) ──
    if adx_gate_enabled:
        adx_series = _compute_adx(high, low, close, adx_period)
        adx_ok = (adx_series >= adx_threshold).fillna(False).values
    else:
        adx_ok = np.ones(n, dtype=bool)

    # ── 5. Persistence counts (v3) ──
    pr_filled = lsr_pctrank.fillna(0.5).values
    long_persist = _consecutive_count(pr_filled < entry_lo)
    short_persist = _consecutive_count(pr_filled > entry_hi)

    # ── 6. Vol scaling ──
    if vol_scale_enabled:
        vol_vals = _vol_scale(close, vol_target, vol_lookback).fillna(1.0).values
    else:
        vol_vals = np.ones(n, dtype=float)

    # ── 7. State machine (v3) ──
    atr_vals = atr.fillna(0.0).values
    high_vals = high.values
    low_vals = low.values
    open_vals = open_.values

    pos = np.zeros(n, dtype=float)
    state = 0
    entry_price = 0.0
    entry_bar = 0
    last_exit_bar = -9999
    warmup = max(lsr_window + 10, sl_atr_lookback + 5)

    n_tp = 0
    n_sl = 0
    n_time = 0

    for i in range(warmup, n):
        pr = pr_filled[i]
        cur_atr = atr_vals[i]

        if state == 0:
            # ── Cooldown (v3) ──
            if i - last_exit_bar < cooldown_bars:
                continue

            # ── ADX gate (v3) ──
            if not adx_ok[i]:
                continue

            # ── Entry with persistence (v3) ──
            can_long = pr < entry_lo and long_persist[i] >= persist_bars
            can_short = (not long_only) and pr > entry_hi and short_persist[i] >= persist_bars

            if can_long:
                pos[i] = 1.0 * vol_vals[i]
                state = 1
                entry_price = open_vals[min(i + 1, n - 1)]
                entry_bar = i
            elif can_short:
                pos[i] = -1.0 * vol_vals[i]
                state = -1
                entry_price = open_vals[min(i + 1, n - 1)]
                entry_bar = i

        elif state == 1:
            hold_time = i - entry_bar

            sl_price = entry_price - sl_atr_mult * cur_atr
            if low_vals[i] <= sl_price and cur_atr > 0:
                pos[i] = 0.0
                state = 0
                last_exit_bar = i
                n_sl += 1
            elif tp_mode == "midpoint" and pr > 0.50:
                pos[i] = 0.0
                state = 0
                last_exit_bar = i
                n_tp += 1
            elif tp_mode == "opposite_extreme" and pr > tp_for_long:
                pos[i] = 0.0
                state = 0
                last_exit_bar = i
                n_tp += 1
            elif hold_time >= max_hold_bars:
                pos[i] = 0.0
                state = 0
                last_exit_bar = i
                n_time += 1
            else:
                pos[i] = 1.0 * vol_vals[i]

        elif state == -1:
            hold_time = i - entry_bar

            sl_price = entry_price + sl_atr_mult * cur_atr
            if high_vals[i] >= sl_price and cur_atr > 0:
                pos[i] = 0.0
                state = 0
                last_exit_bar = i
                n_sl += 1
            elif tp_mode == "midpoint" and pr < 0.50:
                pos[i] = 0.0
                state = 0
                last_exit_bar = i
                n_tp += 1
            elif tp_mode == "opposite_extreme" and pr < tp_for_short:
                pos[i] = 0.0
                state = 0
                last_exit_bar = i
                n_tp += 1
            elif hold_time >= max_hold_bars:
                pos[i] = 0.0
                state = 0
                last_exit_bar = i
                n_time += 1
            else:
                pos[i] = -1.0 * vol_vals[i]

    pos_series = pd.Series(pos, index=df.index)
    pos_series = pos_series.clip(-1.0, 1.0).fillna(0.0)

    n_long = (pos_series > 0.1).sum()
    n_short = (pos_series < -0.1).sum()
    n_flat = n - n_long - n_short
    total_exits = n_tp + n_sl + n_time
    mode_tag = "long-only" if long_only else "both"
    tp_tag = tp_mode
    logger.info(
        f"  LSR Contrarian v3 [{ctx.symbol}] ({mode_tag}, tp={tp_tag}): "
        f"long={n_long}/{n}, short={n_short}/{n}, flat={n_flat}/{n}, "
        f"exits: TP={n_tp}, SL={n_sl}, Time={n_time} (total={total_exits}), "
        f"LSR coverage={lsr_coverage:.1%}"
    )

    return pos_series
