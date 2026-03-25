#!/usr/bin/env python3
"""
Forced Deleveraging Reversal EDA

Research question:
Can post-deleveraging reflexive rebounds become a low-correlation, configurable,
repeatable second strategy leg for the 1h baseline trend system?

This is an Alpha Research EDA, not a formal vectorbt promotion backtest.
It compares event definitions, HTF dependency, entry framing, bucket coverage,
independence vs baseline, density, concentration, and symbol eligibility.

Required methodology guards:
    - Uses temporal + symbol embargo from config/validation.yaml
    - Runs factor orthogonality check against existing production signals
    - Saves both machine-readable and human-readable outputs

Usage:
    PYTHONPATH=src .venv/bin/python scripts/research_forced_deleveraging_reversal.py
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qtrade.config import load_config
from qtrade.data.long_short_ratio import align_lsr_to_klines, load_lsr
from qtrade.data.open_interest import (
    OI_PROVIDER_SEARCH_ORDER,
    align_oi_to_klines,
    get_oi_path,
    load_open_interest,
)
from qtrade.data.quality import DataQualityChecker, clean_data
from qtrade.data.storage import load_klines
from qtrade.strategy import get_strategy
from qtrade.strategy.base import StrategyContext
from qtrade.strategy.oi_liq_bounce_strategy import (
    _compute_oi_change_zscore,
    _compute_price_change_zscore,
    _compute_volume_zscore,
)
from qtrade.strategy.overlays.overlay_pipeline import prepare_and_apply_overlay
from qtrade.strategy.tsmom_carry_v2_strategy import _apply_htf_filter
from qtrade.validation.embargo import (
    get_embargo_only_symbols,
    get_research_symbols,
    load_embargo_config,
    enforce_temporal_embargo,
)
from qtrade.validation.factor_orthogonality import marginal_information_ratio

np.random.seed(42)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("forced_deleveraging_reversal")


BASELINE_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT"]
TARGET_BUCKETS = [
    "panic_liquidation",
    "crash_rebound",
    "sharp_reversal",
    "high_vol_no_trend",
    "false_breakdown_reclaim",
]
ENTRY_VARIANTS = [
    "event_fires_immediately",
    "cascade_end",
    "cascade_end_reclaim",
    "cascade_end_absorption",
    "false_breakdown_reclaim",
]
HTF_VARIANTS = ["no_htf_gate", "soft_htf_veto", "hard_htf_gate"]


@dataclass
class SymbolData:
    symbol: str
    df_1h: pd.DataFrame
    df_5m: pd.DataFrame | None
    oi: pd.Series
    lsr: pd.Series | None
    baseline_pos: pd.Series
    old_oi_pos: pd.Series
    features: pd.DataFrame
    baseline_bar_ret: pd.Series
    data_quality: dict[str, Any]


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    return float(value)


def _annualized_sharpe(series: pd.Series) -> float:
    valid = series.dropna()
    if len(valid) < 50 or float(valid.std()) == 0.0:
        return 0.0
    return float(valid.mean() / valid.std() * np.sqrt(8760))


def _trade_sharpe(returns: list[float]) -> float:
    if len(returns) < 5:
        return 0.0
    arr = np.asarray(returns, dtype=float)
    if np.std(arr) == 0:
        return 0.0
    return float(arr.mean() / arr.std() * np.sqrt(len(arr)))


def _zscore(series: pd.Series, window: int = 720, min_periods: int | None = None) -> pd.Series:
    min_periods = min_periods or max(window // 4, 30)
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std()
    return ((series - mean) / std.replace(0, np.nan)).clip(-5.0, 5.0)


def _load_oi_series(data_dir: Path, symbol: str, index: pd.DatetimeIndex) -> pd.Series:
    for provider in OI_PROVIDER_SEARCH_ORDER:
        oi_path = get_oi_path(data_dir, symbol, provider)
        oi_df = load_open_interest(oi_path)
        if oi_df is not None and not oi_df.empty:
            aligned = align_oi_to_klines(oi_df, index, max_ffill_bars=2)
            if aligned is not None:
                return aligned
    raise FileNotFoundError(f"{symbol}: no OI data found in {data_dir}")


def _load_lsr_series(data_dir: Path, symbol: str, index: pd.DatetimeIndex) -> pd.Series | None:
    deriv_dir = data_dir / "binance" / "futures" / "derivatives"
    lsr_raw = load_lsr(symbol, "lsr", data_dir=deriv_dir)
    if lsr_raw is None or lsr_raw.empty:
        return None
    return align_lsr_to_klines(lsr_raw, index, max_ffill_bars=2)


def _load_symbol_df(
    path: Path,
    embargo,
    research_start: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = load_klines(path)
    df = enforce_temporal_embargo(df, embargo)
    if research_start is not None:
        df = df[df.index >= research_start].copy()
    df = clean_data(df, fill_method="forward", remove_outliers=False, remove_duplicates=True)
    report = DataQualityChecker().validate(df)
    return df, {
        "rows": int(report.total_rows),
        "is_valid": bool(report.is_valid),
        "missing_pct": round(float(report.missing_pct), 4),
        "outlier_pct": round(float(report.outlier_pct), 4),
        "warnings": report.warnings[:5],
        "errors": report.errors[:5],
    }


def _compute_positions(
    cfg_path: Path,
    symbol: str,
    df: pd.DataFrame,
    data_dir: Path,
) -> pd.Series:
    cfg = load_config(str(cfg_path))
    params = dict(cfg.strategy.get_params(symbol))
    params["_data_dir"] = data_dir
    ctx = StrategyContext(
        symbol=symbol,
        interval=cfg.market.interval,
        market_type=cfg.market_type_str,
        direction=cfg.direction,
        signal_delay=1,  # Backtest-style EDA: signal generated on close, trade on next open
    )
    pos = get_strategy(cfg.strategy.name)(df, ctx, params)
    overlay_cfg = getattr(cfg, "_overlay_cfg", None)
    if overlay_cfg and overlay_cfg.get("enabled", False):
        pos = prepare_and_apply_overlay(
            pos,
            df,
            overlay_cfg,
            symbol,
            data_dir=data_dir,
        )
    return pos.reindex(df.index).fillna(0.0)


def _compute_htf_masks(symbol: str, df: pd.DataFrame, baseline_cfg_path: Path) -> dict[str, pd.Series]:
    cfg = load_config(str(baseline_cfg_path))
    params = cfg.strategy.get_params(symbol)
    sub_params = {}
    for sub in params.get("sub_strategies", []):
        if sub.get("name") == "tsmom_carry_v2":
            sub_params = dict(sub.get("params", {}))
            break

    raw = pd.Series(1.0, index=df.index)
    confirmation = _apply_htf_filter(
        raw,
        df,
        ctx_symbol=symbol,
        htf_4h_ema_fast=int(sub_params.get("htf_4h_ema_fast", 20)),
        htf_4h_ema_slow=int(sub_params.get("htf_4h_ema_slow", 50)),
        htf_adx_period=int(sub_params.get("htf_adx_period", 14)),
        htf_adx_threshold=float(sub_params.get("htf_adx_threshold", 25.0)),
        htf_regime_ema=int(sub_params.get("htf_regime_ema", 20)),
        htf_full_confirm=1.0,
        htf_partial_confirm=0.85,
        htf_4h_only_confirm=0.7,
        htf_no_confirm=0.0,
        htf_daily_only=bool(sub_params.get("htf_daily_only", False)),
    )
    return {
        "no_htf_gate": pd.Series(True, index=df.index),
        "soft_htf_veto": confirmation > 0.0,
        "hard_htf_gate": confirmation >= 0.85,
    }


def _build_features(symbol: str, df: pd.DataFrame, oi: pd.Series) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    feat["oi_z"] = _compute_oi_change_zscore(oi, change_lookback=24, z_window=720)
    feat["price_z"] = _compute_price_change_zscore(df["close"], change_lookback=8, z_window=720)
    feat["volume_z"] = _compute_volume_zscore(df["volume"], vol_sum_window=24, z_window=720)

    true_range = (df["high"] - df["low"]) / df["close"].shift(1).replace(0, np.nan)
    feat["range_z"] = _zscore(true_range.rolling(8, min_periods=2).sum())
    feat["ret_1h"] = df["close"].pct_change(fill_method=None)
    feat["ret_6h"] = df["close"].pct_change(6, fill_method=None)
    feat["support_24h"] = df["low"].rolling(24, min_periods=8).min().shift(1)
    feat["support_72h"] = df["low"].rolling(72, min_periods=24).min().shift(1)

    candle_range = (df["high"] - df["low"]).replace(0, np.nan)
    lower_wick = (df[["open", "close"]].min(axis=1) - df["low"]).clip(lower=0.0)
    feat["lower_wick_ratio"] = (lower_wick / candle_range).fillna(0.0)

    shock_components = pd.concat(
        [
            (-feat["oi_z"]).clip(lower=0.0),
            (-feat["price_z"]).clip(lower=0.0),
            feat["volume_z"].clip(lower=0.0),
            feat["range_z"].clip(lower=0.0),
        ],
        axis=1,
    )
    feat["shock_score"] = shock_components.mean(axis=1)
    feat["proxy_event"] = (feat["oi_z"] < -1.25) & (feat["price_z"] < -1.0)
    feat["shock_event"] = (
        (feat["price_z"] < -1.0)
        & (
            (feat["oi_z"] < -1.0)
            | (feat["volume_z"] > 1.5)
            | (feat["range_z"] > 1.5)
        )
    )
    feat["recent_shock"] = feat["shock_event"].rolling(3, min_periods=1).max() > 0
    feat["cascade_end"] = (
        feat["recent_shock"]
        & (df["low"] >= df["low"].shift(1))
        & (df["close"] > df["close"].shift(1))
        & (feat["oi_z"] > feat["oi_z"].shift(1))
    )
    feat["reclaim_support"] = df["close"] > feat["support_24h"]
    feat["cascade_end_reclaim"] = feat["cascade_end"] & feat["reclaim_support"]
    feat["cascade_end_absorption"] = (
        feat["cascade_end"]
        & (feat["lower_wick_ratio"] > 0.45)
        & (feat["volume_z"] > 1.0)
        & (df["close"] > df["open"])
    )
    feat["recent_break_support"] = (
        (df["low"] < feat["support_24h"]).rolling(3, min_periods=1).max() > 0
    )
    feat["false_breakdown_reclaim"] = (
        feat["recent_break_support"]
        & (df["close"] > feat["support_24h"])
        & (df["close"] > df["open"])
    )
    feat["event_fires_immediately"] = feat["proxy_event"]

    feat["bucket_anchor"] = feat["shock_event"] | feat["false_breakdown_reclaim"]
    return feat.fillna(0.0)


def _select_anchor_indices(anchor: pd.Series, cooldown_bars: int = 12) -> list[int]:
    idxs: list[int] = []
    last = -10_000
    for i, flag in enumerate(anchor.values):
        if not flag:
            continue
        if i - last <= cooldown_bars:
            continue
        idxs.append(i)
        last = i
    return idxs


def _bucket_label(
    df: pd.DataFrame,
    feat: pd.DataFrame,
    i: int,
) -> str:
    if i + 25 >= len(df):
        return "insufficient_forward"

    entry_open = float(df["open"].iloc[i + 1])
    next_6h = float(df["open"].iloc[i + 7] / entry_open - 1.0) if i + 7 < len(df) else 0.0
    next_24h = float(df["open"].iloc[i + 25] / entry_open - 1.0)
    fwd_high_24 = float(df["high"].iloc[i + 1:i + 25].max() / entry_open - 1.0)
    fwd_low_24 = float(df["low"].iloc[i + 1:i + 25].min() / entry_open - 1.0)
    realized_range_24 = fwd_high_24 - fwd_low_24

    support = float(feat["support_24h"].iloc[i]) if feat["support_24h"].iloc[i] > 0 else np.nan
    broke_support = bool(df["low"].iloc[i] < support) if not np.isnan(support) else False
    reclaim_3h = bool((df["close"].iloc[i + 1:i + 4] > support).any()) if not np.isnan(support) else False

    if broke_support and reclaim_3h:
        return "false_breakdown_reclaim"
    if feat["proxy_event"].iloc[i] and next_24h > 0.03:
        return "crash_rebound"
    if next_6h > 0.015 and next_24h > 0.005:
        return "sharp_reversal"
    if realized_range_24 > 0.06 and abs(next_24h) < 0.01:
        return "high_vol_no_trend"
    if feat["proxy_event"].iloc[i]:
        return "panic_liquidation"
    if next_24h < -0.02:
        return "trend_continuation_disguised"
    return "unclassified"


def _event_map(symbol_data: dict[str, SymbolData]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sym, data in symbol_data.items():
        anchors = _select_anchor_indices(data.features["bucket_anchor"])
        old_entries = (data.old_oi_pos > 0.05) & (data.old_oi_pos.shift(1).fillna(0.0) <= 0.05)
        for i in anchors:
            if i + 25 >= len(data.df_1h):
                continue

            bucket = _bucket_label(data.df_1h, data.features, i)
            entry_open = float(data.df_1h["open"].iloc[i + 1])
            next_24h = float(data.df_1h["open"].iloc[i + 25] / entry_open - 1.0)

            old_hit = bool(old_entries.iloc[i:i + 3].any())
            baseline_entry_pos = float(data.baseline_pos.iloc[i + 1])
            baseline_state = "long" if baseline_entry_pos > 0.05 else "short" if baseline_entry_pos < -0.05 else "flat"

            baseline_window = data.baseline_bar_ret.iloc[i + 1:i + 25]
            baseline_24h_ret = float((1.0 + baseline_window.fillna(0.0)).prod() - 1.0)

            rows.append(
                {
                    "symbol": sym,
                    "idx": int(i),
                    "timestamp": str(data.df_1h.index[i]),
                    "bucket": bucket,
                    "proxy_event": bool(data.features["proxy_event"].iloc[i]),
                    "shock_score": round(float(data.features["shock_score"].iloc[i]), 4),
                    "event_24h_return": round(next_24h, 6),
                    "old_oi_liq_hit": old_hit,
                    "baseline_entry_pos": round(baseline_entry_pos, 4),
                    "baseline_state": baseline_state,
                    "baseline_24h_ret": round(baseline_24h_ret, 6),
                }
            )
    return pd.DataFrame(rows)


def _simulate_fixed_hold(
    entry_signal: pd.Series,
    open_prices: pd.Series,
    hold_bars: int = 24,
    cooldown_bars: int = 12,
) -> tuple[pd.Series, list[dict[str, Any]]]:
    pos = pd.Series(0.0, index=entry_signal.index)
    trades: list[dict[str, Any]] = []
    state = False
    entry_open_idx: int | None = None
    signal_idx: int | None = None
    cooldown_left = 0

    n = len(entry_signal)
    for i in range(n - 1):
        pos.iloc[i] = 1.0 if state else 0.0

        if cooldown_left > 0 and not state:
            cooldown_left -= 1

        if state and entry_open_idx is not None and signal_idx is not None:
            if i - entry_open_idx + 1 >= hold_bars:
                exit_open_idx = i + 1
                trades.append(
                    {
                        "signal_bar": signal_idx,
                        "entry_open_idx": entry_open_idx,
                        "exit_open_idx": exit_open_idx,
                        "entry_time": str(entry_signal.index[entry_open_idx]),
                        "exit_time": str(entry_signal.index[exit_open_idx]),
                        "return": float(open_prices.iloc[exit_open_idx] / open_prices.iloc[entry_open_idx] - 1.0),
                    }
                )
                state = False
                entry_open_idx = None
                signal_idx = None
                cooldown_left = cooldown_bars

        if (not state) and cooldown_left == 0 and bool(entry_signal.iloc[i]) and i + 1 < n:
            state = True
            signal_idx = i
            entry_open_idx = i + 1

    if state and entry_open_idx is not None and signal_idx is not None and entry_open_idx < n - 1:
        exit_open_idx = n - 1
        trades.append(
            {
                "signal_bar": signal_idx,
                "entry_open_idx": entry_open_idx,
                "exit_open_idx": exit_open_idx,
                "entry_time": str(entry_signal.index[entry_open_idx]),
                "exit_time": str(entry_signal.index[exit_open_idx]),
                "return": float(open_prices.iloc[exit_open_idx] / open_prices.iloc[entry_open_idx] - 1.0),
            }
        )

    return pos.fillna(0.0), trades


def _evaluate_variant(
    symbol: str,
    variant: str,
    htf_name: str,
    entry_signal: pd.Series,
    data: SymbolData,
    hold_bars: int,
) -> dict[str, Any]:
    pos, trades = _simulate_fixed_hold(entry_signal, data.df_1h["open"], hold_bars=hold_bars, cooldown_bars=12)
    open_ret = data.df_1h["open"].pct_change(fill_method=None).shift(-1).fillna(0.0)
    bar_ret = pos * open_ret
    trade_returns = [t["return"] for t in trades]
    corr = 0.0
    if bar_ret.abs().sum() > 0 and data.baseline_bar_ret.abs().sum() > 0:
        aligned = pd.DataFrame({"cand": bar_ret, "base": data.baseline_bar_ret}).dropna()
        if len(aligned) > 50:
            corr = float(aligned["cand"].corr(aligned["base"]))

    baseline_flat_or_wrong = 0
    baseline_wrong = 0
    baseline_profitable_overlap = 0
    for trade in trades:
        entry_idx = trade["entry_open_idx"]
        baseline_pos = float(data.baseline_pos.iloc[entry_idx])
        if baseline_pos <= 0.05:
            baseline_flat_or_wrong += 1
        if baseline_pos < -0.05:
            baseline_wrong += 1
        window = data.baseline_bar_ret.iloc[entry_idx:trade["exit_open_idx"]]
        base_ret = float((1.0 + window.fillna(0.0)).prod() - 1.0)
        if base_ret > 0:
            baseline_profitable_overlap += 1

    return {
        "symbol": symbol,
        "entry_variant": variant,
        "htf_variant": htf_name,
        "trades": len(trades),
        "tim": round(float(pos.mean()), 4),
        "bar_sharpe": round(_annualized_sharpe(bar_ret), 4),
        "trade_sharpe": round(_trade_sharpe(trade_returns), 4),
        "avg_trade_return": round(float(np.mean(trade_returns)) if trade_returns else 0.0, 6),
        "median_trade_return": round(float(np.median(trade_returns)) if trade_returns else 0.0, 6),
        "win_rate": round(float(np.mean(np.asarray(trade_returns) > 0.0)) if trade_returns else 0.0, 4),
        "baseline_corr": round(corr, 4),
        "baseline_flat_or_wrong_share": round(baseline_flat_or_wrong / len(trades), 4) if trades else 0.0,
        "baseline_wrong_share": round(baseline_wrong / len(trades), 4) if trades else 0.0,
        "baseline_profitable_overlap_share": round(baseline_profitable_overlap / len(trades), 4) if trades else 0.0,
        "bar_ret": bar_ret,
        "pos": pos,
        "trades_detail": trades,
    }


def _orthogonality_summary(symbol_data: dict[str, SymbolData]) -> dict[str, Any]:
    rows = []
    for sym, data in symbol_data.items():
        candidate = data.features["cascade_end_reclaim"].astype(float)
        tsmom_proxy = data.df_1h["close"].pct_change(24, fill_method=None).shift(1).fillna(0.0)
        htf_proxy = _compute_htf_masks(sym, data.df_1h, BASELINE_CFG_PATH)["hard_htf_gate"].astype(float)
        existing = {
            "baseline_pos": data.baseline_pos,
            "tsmom_24h": tsmom_proxy,
            "htf_hard": htf_proxy,
        }
        if data.lsr is not None:
            lsr_rank = data.lsr.rolling(168, min_periods=48).rank(pct=True).shift(1).fillna(0.5)
            existing["lsr_rank"] = lsr_rank

        fwd_ret_24h = data.df_1h["open"].pct_change(24, fill_method=None).shift(-24)
        result = marginal_information_ratio(candidate, existing, fwd_ret_24h)
        rows.append(
            {
                "symbol": sym,
                "r_squared": round(float(result.r_squared), 4),
                "residual_ic": round(float(result.residual_ic), 4),
                "is_redundant": bool(result.is_redundant),
                "coefficients": {k: round(float(v), 4) for k, v in result.coefficients.items()},
            }
        )

    df = pd.DataFrame(rows)
    return {
        "per_symbol": rows,
        "avg_r_squared": round(float(df["r_squared"].mean()), 4) if not df.empty else None,
        "avg_residual_ic": round(float(df["residual_ic"].mean()), 4) if not df.empty else None,
        "redundant_symbols": int(df["is_redundant"].sum()) if not df.empty else 0,
    }


def _entry_variant_scan(symbol_data: dict[str, SymbolData], hold_bars: int) -> pd.DataFrame:
    rows = []
    for sym, data in symbol_data.items():
        gates = _compute_htf_masks(sym, data.df_1h, BASELINE_CFG_PATH)
        for entry_variant in ENTRY_VARIANTS:
            raw_signal = data.features[entry_variant].astype(bool)
            for htf_name, mask in gates.items():
                result = _evaluate_variant(sym, entry_variant, htf_name, raw_signal & mask, data, hold_bars)
                rows.append({k: v for k, v in result.items() if k not in {"bar_ret", "pos", "trades_detail"}})
    return pd.DataFrame(rows)


def _pick_best_candidate(scan_df: pd.DataFrame) -> dict[str, Any]:
    agg = (
        scan_df.groupby(["entry_variant", "htf_variant"])
        .agg(
            mean_bar_sharpe=("bar_sharpe", "mean"),
            mean_trade_sharpe=("trade_sharpe", "mean"),
            mean_tim=("tim", "mean"),
            mean_corr=("baseline_corr", "mean"),
            mean_trade_return=("avg_trade_return", "mean"),
            total_trades=("trades", "sum"),
            improved_symbols=("bar_sharpe", lambda x: int((x > 0).sum())),
            flat_or_wrong_share=("baseline_flat_or_wrong_share", "mean"),
            wrong_share=("baseline_wrong_share", "mean"),
        )
        .reset_index()
    )

    # Favor low correlation, tradable density in the 8%-15% satellite zone,
    # and positive edge. Overly broad dip-buying proxies should be penalized.
    tim_target = 0.10
    agg["tim_distance"] = (agg["mean_tim"] - tim_target).abs()
    agg["tim_penalty"] = 6.0 * agg["tim_distance"]
    agg["tim_penalty"] += 3.0 * (agg["mean_tim"] > 0.18).astype(float)
    agg["tim_penalty"] += 2.0 * (agg["mean_tim"] < 0.05).astype(float)
    agg["score"] = (
        agg["mean_bar_sharpe"]
        + 0.5 * agg["mean_trade_sharpe"]
        + agg["flat_or_wrong_share"]
        + 0.5 * agg["wrong_share"]
        - 2.0 * agg["mean_corr"].clip(lower=0.0)
        - agg["tim_penalty"]
        - 1.0 * (agg["mean_corr"] > 0.25).astype(float)
    )
    best = agg.sort_values("score", ascending=False).iloc[0].to_dict()
    return {
        "aggregate_table": agg.sort_values("score", ascending=False).to_dict(orient="records"),
        "best": best,
    }


def _bucket_level_relative_edge(
    symbol_data: dict[str, SymbolData],
    event_df: pd.DataFrame,
    best_entry: str,
    best_htf: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sym, data in symbol_data.items():
        gates = _compute_htf_masks(sym, data.df_1h, BASELINE_CFG_PATH)
        candidate_signal = (data.features[best_entry].astype(bool) & gates[best_htf]).astype(bool)

        sym_events = event_df[event_df["symbol"] == sym]
        for bucket in TARGET_BUCKETS + ["trend_continuation_disguised"]:
            bucket_events = sym_events[sym_events["bucket"] == bucket]
            if bucket_events.empty:
                continue

            event_rets = []
            baseline_rets = []
            candidate_hits = 0
            candidate_rets = []
            baseline_states = {"flat": 0, "short": 0, "long": 0}
            for _, row in bucket_events.iterrows():
                i = int(row["idx"])
                event_rets.append(float(row["event_24h_return"]))
                baseline_rets.append(float(row["baseline_24h_ret"]))
                baseline_states[str(row["baseline_state"])] += 1
                if bool(candidate_signal.iloc[i]):
                    candidate_hits += 1
                    if i + 25 < len(data.df_1h):
                        entry_open = float(data.df_1h["open"].iloc[i + 1])
                        candidate_rets.append(float(data.df_1h["open"].iloc[i + 25] / entry_open - 1.0))

            rows.append(
                {
                    "symbol": sym,
                    "bucket": bucket,
                    "events": int(len(bucket_events)),
                    "candidate_coverage": round(candidate_hits / len(bucket_events), 4),
                    "avg_event_24h_return": round(float(np.mean(event_rets)), 6),
                    "avg_candidate_24h_return": round(float(np.mean(candidate_rets)) if candidate_rets else 0.0, 6),
                    "avg_baseline_24h_return": round(float(np.mean(baseline_rets)), 6),
                    "baseline_flat_share": round(baseline_states["flat"] / len(bucket_events), 4),
                    "baseline_short_share": round(baseline_states["short"] / len(bucket_events), 4),
                }
            )
    return rows


def _analyze_exit_paths(symbol_data: dict[str, SymbolData], best_entry: str, best_htf: str) -> dict[str, Any]:
    rows = []
    for sym, data in symbol_data.items():
        gates = _compute_htf_masks(sym, data.df_1h, BASELINE_CFG_PATH)
        signal = data.features[best_entry].astype(bool) & gates[best_htf]
        trade_idxs = _select_anchor_indices(signal.astype(bool), cooldown_bars=12)
        for i in trade_idxs:
            if i + 49 >= len(data.df_1h):
                continue
            entry = float(data.df_1h["open"].iloc[i + 1])
            support = float(data.features["support_24h"].iloc[i])
            ret_6h = float(data.df_1h["open"].iloc[i + 7] / entry - 1.0)
            ret_12h = float(data.df_1h["open"].iloc[i + 13] / entry - 1.0)
            ret_24h = float(data.df_1h["open"].iloc[i + 25] / entry - 1.0)
            ret_48h = float(data.df_1h["open"].iloc[i + 49] / entry - 1.0)
            max_up_24h = float(data.df_1h["high"].iloc[i + 1:i + 25].max() / entry - 1.0)
            max_down_24h = float(data.df_1h["low"].iloc[i + 1:i + 25].min() / entry - 1.0)
            reclaim_failed = bool(
                (support > 0)
                and (data.df_1h["close"].iloc[i + 1:i + 7] < support).any()
            )
            shock_decay_exit = 24
            for h in range(6, 25):
                if data.features["shock_score"].iloc[i + h] < 0.6:
                    shock_decay_exit = h
                    break
            event_decay_ret = float(data.df_1h["open"].iloc[i + shock_decay_exit + 1] / entry - 1.0)
            reclaim_failure_ret = ret_24h
            if reclaim_failed:
                fail_idx = i + 1 + int((data.df_1h["close"].iloc[i + 1:i + 7] < support).values.argmax())
                if fail_idx + 1 < len(data.df_1h):
                    reclaim_failure_ret = float(data.df_1h["open"].iloc[fail_idx + 1] / entry - 1.0)
            staged_tp_ret = 0.5 * ret_12h + 0.5 * ret_24h

            rows.append(
                {
                    "symbol": sym,
                    "ret_6h": ret_6h,
                    "ret_12h": ret_12h,
                    "ret_24h": ret_24h,
                    "ret_48h": ret_48h,
                    "max_up_24h": max_up_24h,
                    "max_down_24h": max_down_24h,
                    "event_decay_ret": event_decay_ret,
                    "reclaim_failure_ret": reclaim_failure_ret,
                    "staged_tp_ret": staged_tp_ret,
                    "reclaim_failed": reclaim_failed,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return {}

    return {
        "n_trades": int(len(df)),
        "fixed_hold": {
            "hold_12h": round(float(df["ret_12h"].mean()), 6),
            "hold_24h": round(float(df["ret_24h"].mean()), 6),
            "hold_48h": round(float(df["ret_48h"].mean()), 6),
        },
        "conceptual_exits": {
            "event_decay_exit": round(float(df["event_decay_ret"].mean()), 6),
            "reclaim_failure_exit": round(float(df["reclaim_failure_ret"].mean()), 6),
            "staged_take_profit": round(float(df["staged_tp_ret"].mean()), 6),
        },
        "path_shape": {
            "avg_max_up_24h": round(float(df["max_up_24h"].mean()), 6),
            "avg_max_down_24h": round(float(df["max_down_24h"].mean()), 6),
            "reclaim_failure_rate": round(float(df["reclaim_failed"].mean()), 4),
        },
    }


def _symbol_eligibility(
    symbol_data: dict[str, SymbolData],
    scan_df: pd.DataFrame,
    best_entry: str,
    best_htf: str,
) -> list[dict[str, Any]]:
    rows = []
    for sym in sorted(symbol_data.keys()):
        row = scan_df[
            (scan_df["symbol"] == sym)
            & (scan_df["entry_variant"] == best_entry)
            & (scan_df["htf_variant"] == best_htf)
        ].iloc[0]

        yearly_counts = scan_df[
            (scan_df["symbol"] == sym)
            & (scan_df["entry_variant"] == best_entry)
            & (scan_df["htf_variant"] == best_htf)
        ]
        eligible = (
            row["trades"] >= 10
            and row["tim"] >= 0.05
            and row["baseline_corr"] < 0.25
            and row["avg_trade_return"] > 0.0
        )
        reason = []
        if row["trades"] < 10:
            reason.append("too_sparse")
        if row["tim"] < 0.05:
            reason.append("tim_too_low")
        if row["baseline_corr"] >= 0.25:
            reason.append("corr_too_high")
        if row["avg_trade_return"] <= 0.0:
            reason.append("no_edge")
        if not reason:
            reason.append("eligible")

        rows.append(
            {
                "symbol": sym,
                "trades": int(row["trades"]),
                "tim": round(float(row["tim"]), 4),
                "bar_sharpe": round(float(row["bar_sharpe"]), 4),
                "avg_trade_return": round(float(row["avg_trade_return"]), 6),
                "baseline_corr": round(float(row["baseline_corr"]), 4),
                "baseline_flat_or_wrong_share": round(float(row["baseline_flat_or_wrong_share"]), 4),
                "eligible": bool(eligible),
                "reason": ",".join(reason),
            }
        )
    return rows


def _concentration_test(
    symbol_data: dict[str, SymbolData],
    best_entry: str,
    best_htf: str,
) -> dict[str, Any]:
    trade_rows = []
    for sym, data in symbol_data.items():
        gates = _compute_htf_masks(sym, data.df_1h, BASELINE_CFG_PATH)
        signal = data.features[best_entry].astype(bool) & gates[best_htf]
        _, trades = _simulate_fixed_hold(signal, data.df_1h["open"], hold_bars=24, cooldown_bars=12)
        for trade in trades:
            entry_idx = trade["entry_open_idx"]
            trade_rows.append(
                {
                    "symbol": sym,
                    "year": int(data.df_1h.index[entry_idx].year),
                    "entry_time": str(data.df_1h.index[entry_idx]),
                    "return": float(trade["return"]),
                    "abs_return": abs(float(trade["return"])),
                }
            )

    df = pd.DataFrame(trade_rows)
    if df.empty:
        return {}

    total_abs = float(df["abs_return"].sum()) or 1.0
    by_symbol = (
        df.groupby("symbol")["abs_return"].sum().div(total_abs).sort_values(ascending=False).round(4)
    )
    by_year = (
        df.groupby("year")["abs_return"].sum().div(total_abs).sort_values(ascending=False).round(4)
    )
    top_trade_share = round(float(df["abs_return"].max() / total_abs), 4)
    top5_trade_share = round(float(df["abs_return"].nlargest(5).sum() / total_abs), 4)

    return {
        "by_symbol_share": by_symbol.to_dict(),
        "by_year_share": {str(k): float(v) for k, v in by_year.to_dict().items()},
        "top_trade_share": top_trade_share,
        "top5_trade_share": top5_trade_share,
    }


def _verdict(summary: dict[str, Any]) -> tuple[str, str]:
    best = summary["best_candidate"]["best"]
    exit_summary = summary["exit_path_analysis"]
    true_liq_data = summary["data_gate"]["true_liquidation_state_available"]
    oi_gate_failed = bool(summary["data_gate"].get("oi_gate_failed_symbols"))

    if (
        best["mean_corr"] > 0.25
        or best["mean_tim"] < 0.05
        or best["mean_bar_sharpe"] <= 0
        or best["htf_variant"] != "no_htf_gate"
    ):
        return "FAIL", "This is still just an overlay thesis in disguise"

    if oi_gate_failed or (not true_liq_data) or best["mean_tim"] < 0.08:
        return "SHELVED", "This is still just an overlay thesis in disguise"

    if exit_summary and exit_summary["fixed_hold"]["hold_24h"] < exit_summary["fixed_hold"]["hold_12h"]:
        return "WEAK GO", "This is still just an overlay thesis in disguise"

    return "GO", "This is a second-leg candidate"


def _build_reopen_framework(summary: dict[str, Any]) -> dict[str, Any]:
    best = summary["best_candidate"]["best"]
    data_gate = summary["data_gate"]
    concentration = summary.get("concentration", {})

    bucket_df = pd.DataFrame(summary.get("bucket_relative_edge", []))
    panic_ok = False
    false_breakdown_ok = False
    if not bucket_df.empty:
        bucket_avg = (
            bucket_df.groupby("bucket")
            .agg(
                candidate_ret=("avg_candidate_24h_return", "mean"),
                baseline_ret=("avg_baseline_24h_return", "mean"),
                candidate_coverage=("candidate_coverage", "mean"),
            )
        )
        if "panic_liquidation" in bucket_avg.index:
            row = bucket_avg.loc["panic_liquidation"]
            panic_ok = bool(row["candidate_ret"] > row["baseline_ret"] and row["candidate_ret"] > 0)
        if "false_breakdown_reclaim" in bucket_avg.index:
            row = bucket_avg.loc["false_breakdown_reclaim"]
            false_breakdown_ok = bool(
                row["candidate_coverage"] >= 0.05
                and row["candidate_ret"] > row["baseline_ret"]
                and row["candidate_ret"] > 0
            )

    proxy_valid = bool(
        best["htf_variant"] == "no_htf_gate"
        and 0.08 <= best["mean_tim"] <= 0.15
        and best["mean_corr"] < 0.15
        and best["mean_trade_return"] > 0.0
        and best["improved_symbols"] >= max(len(summary["research_symbols"]) - 1, 1)
        and not data_gate["oi_gate_failed_symbols"]
        and data_gate["historical_lsr_available"]
    )
    state_valid = bool(proxy_valid and data_gate["true_liquidation_state_available"])
    handoff_ready = bool(
        state_valid
        and panic_ok
        and false_breakdown_ok
        and concentration.get("top5_trade_share", 1.0) <= 0.15
    )

    current_state = (
        "proxy_valid_state_invalid"
        if proxy_valid and not state_valid
        else "state_valid_handoff_blocked"
        if state_valid and not handoff_ready
        else "handoff_ready"
        if handoff_ready
        else "not_proxy_valid"
    )

    return {
        "current_state": current_state,
        "proxy_valid": proxy_valid,
        "state_valid": state_valid,
        "handoff_ready": handoff_ready,
        "status_banner": (
            "proxy-valid / state-invalid"
            if current_state == "proxy_valid_state_invalid"
            else "state-valid / handoff-blocked"
            if current_state == "state_valid_handoff_blocked"
            else "handoff-ready"
            if current_state == "handoff_ready"
            else "not proxy-valid"
        ),
        "current_blockers": [
            blocker
            for blocker, is_active in [
                ("missing_true_liquidation_state_history", not data_gate["true_liquidation_state_available"]),
                ("panic_bucket_not_solved", not panic_ok),
                ("false_breakdown_not_solved", not false_breakdown_ok),
                ("concentration_not_healthy_enough", concentration.get("top5_trade_share", 1.0) > 0.15),
            ]
            if is_active
        ],
        "phase_status": {
            "phase_a_proxy_validation": "pass" if proxy_valid else "fail",
            "phase_b_state_validation": "pass" if state_valid else "blocked",
            "phase_c_handoff_gate": "pass" if handoff_ready else "blocked",
        },
        "phases": [
            {
                "phase": "A",
                "name": "Proxy Validation",
                "goal": "Keep the thesis alive on a common-data window without HTF hard-gate dependence.",
                "required_gates": [
                    "OI coverage >= 70% on the research window",
                    "historical LSR replay available for the same window",
                    "best trigger survives with no_htf_gate",
                    "TIM between 8% and 15%",
                    "baseline correlation < 0.15 preferred",
                ],
                "status": "pass" if proxy_valid else "fail",
            },
            {
                "phase": "B",
                "name": "State Validation",
                "goal": "Upgrade the thesis from OI-state proxy to true liquidation-state evidence.",
                "required_gates": [
                    "historical liquidation-state data available for all target symbols",
                    "state-based trigger still selects cascade_end or a better successor",
                    "proxy edge is preserved after replacing proxy tags with state tags",
                ],
                "status": "pass" if state_valid else "blocked",
            },
            {
                "phase": "C",
                "name": "Developer Handoff Gate",
                "goal": "Only hand off after the strategy behaves like a genuine second leg, not an event-colored overlay.",
                "required_gates": [
                    "panic_liquidation or false_breakdown bucket is materially improved",
                    "corr < 0.15 preferred, < 0.25 maximum",
                    "no HTF hard gate dependence",
                    "concentration remains healthy",
                ],
                "status": "pass" if handoff_ready else "blocked",
            },
        ],
    }


def _build_report(summary: dict[str, Any]) -> str:
    best = summary["best_candidate"]["best"]
    verdict = summary["verdict"]["label"]
    second_leg_line = summary["verdict"]["second_leg_line"]
    framework = summary["reopen_framework"]
    lines = [
        "# Forced Deleveraging Reversal EDA",
        "",
        f"- Baseline: `{summary['baseline_config']}`",
        f"- Research start override: `{summary.get('research_start') or 'none'}`",
        f"- Research symbols: `{', '.join(summary['research_symbols'])}`",
        f"- Embargo-only symbols: `{', '.join(summary['embargo_only_symbols']) or 'none'}`",
        f"- Verdict: **{verdict}**",
        f"- Evidence tier: **{framework['status_banner']}**",
        f"- Conclusion: **{second_leg_line}**",
        "",
        "## Step 1. Archetype Check",
        "",
        "- Classified as `Event-Driven Reversal`.",
        "- Forced sellers in deleveraging are crowded longs unwinding into falling liquidity.",
        "- A tradable rebound can exist only if liquidation pressure exhausts and price reclaims structure after the cascade.",
        "- This should not be naturally absorbed by the baseline if profits arrive while the baseline is flat or short during rebound buckets.",
        "",
        "## Data Gate",
        "",
        f"- True liquidation-state history available: `{summary['data_gate']['true_liquidation_state_available']}`",
        f"- OI proxy available for all research symbols: `{summary['data_gate']['oi_proxy_available']}`",
        f"- Historical LSR available for full-period baseline replay: `{summary['data_gate']['historical_lsr_available']}`",
        f"- OI coverage by symbol: `{summary['data_gate']['oi_coverage_by_symbol']}`",
        f"- OI coverage gate failures (<70%): `{summary['data_gate']['oi_gate_failed_symbols']}`",
        "- Implication: proxy/state-style EDA is valid, but true liquidation-state evidence is capped.",
        "",
        "## Reopen Framework",
        "",
        f"- Current state: `{framework['current_state']}`",
        f"- Status banner: **{framework['status_banner']}**",
        f"- Proxy-valid: `{framework['proxy_valid']}`",
        f"- State-valid: `{framework['state_valid']}`",
        f"- Handoff-ready: `{framework['handoff_ready']}`",
        f"- Active blockers: `{framework['current_blockers']}`",
        "",
        "## Best Candidate",
        "",
        f"- Entry framing: `{best['entry_variant']}`",
        f"- HTF dependency winner: `{best['htf_variant']}`",
        f"- Mean bar Sharpe: `{best['mean_bar_sharpe']:.3f}`",
        f"- Mean TIM: `{best['mean_tim']:.3%}`",
        f"- Mean baseline correlation: `{best['mean_corr']:.3f}`",
        f"- Baseline flat-or-wrong share: `{best['flat_or_wrong_share']:.3%}`",
        "",
        "## Bucket-Level Findings",
        "",
    ]

    bucket_df = pd.DataFrame(summary["bucket_relative_edge"])
    if not bucket_df.empty:
        bucket_avg = (
            bucket_df.groupby("bucket")
            .agg(
                events=("events", "sum"),
                candidate_coverage=("candidate_coverage", "mean"),
                candidate_ret=("avg_candidate_24h_return", "mean"),
                baseline_ret=("avg_baseline_24h_return", "mean"),
                flat_share=("baseline_flat_share", "mean"),
                short_share=("baseline_short_share", "mean"),
            )
            .sort_values("events", ascending=False)
        )
        for bucket, row in bucket_avg.iterrows():
            lines.append(
                f"- `{bucket}`: events={int(row['events'])}, candidate coverage={row['candidate_coverage']:.1%}, "
                f"candidate 24h={row['candidate_ret']:+.3%}, baseline 24h={row['baseline_ret']:+.3%}, "
                f"baseline flat={row['flat_share']:.1%}, short={row['short_share']:.1%}"
            )

    lines.extend(
        [
            "",
            "## Definition Layer Check",
            "",
            "- `Proxy`: OI drop + price drop can find panic bars, but it is still noisy and catches trend continuation disguises.",
            "- `State-based`: cascade-end improves timing, but without true liquidation-state history this remains an OI-state approximation.",
            "- `Tradeable trigger`: reclaim-style entries are the most faithful mapping from event to tradable reversal engine.",
            "",
            "## HTF Dependency Check",
            "",
            "- If the best candidate still needs `hard_htf_gate`, it fails the independent-engine thesis.",
            "- If `no_htf_gate` or `soft_htf_veto` survives with acceptable density/correlation, the thesis remains alive.",
            "",
            "## Exit Redesign",
            "",
            f"- Fixed hold 12h/24h/48h mean return: "
            f"`{summary['exit_path_analysis'].get('fixed_hold', {}).get('hold_12h', 0.0):+.3%}` / "
            f"`{summary['exit_path_analysis'].get('fixed_hold', {}).get('hold_24h', 0.0):+.3%}` / "
            f"`{summary['exit_path_analysis'].get('fixed_hold', {}).get('hold_48h', 0.0):+.3%}`",
            f"- Conceptual event-decay exit mean return: `{summary['exit_path_analysis'].get('conceptual_exits', {}).get('event_decay_exit', 0.0):+.3%}`",
            f"- Conceptual reclaim-failure exit mean return: `{summary['exit_path_analysis'].get('conceptual_exits', {}).get('reclaim_failure_exit', 0.0):+.3%}`",
            f"- Conceptual staged TP mean return: `{summary['exit_path_analysis'].get('conceptual_exits', {}).get('staged_take_profit', 0.0):+.3%}`",
            "",
            "## Symbol Eligibility",
            "",
        ]
    )

    for row in summary["symbol_eligibility"]:
        lines.append(
            f"- `{row['symbol']}`: eligible={row['eligible']}, trades={row['trades']}, "
            f"TIM={row['tim']:.1%}, corr={row['baseline_corr']:.3f}, reason=`{row['reason']}`"
        )

    lines.extend(
        [
            "",
            "## Reopen Phases",
            "",
        ]
    )
    for phase in framework["phases"]:
        lines.append(
            f"- Phase {phase['phase']} `{phase['name']}`: status=`{phase['status']}`, goal=`{phase['goal']}`"
        )
        for gate in phase["required_gates"]:
            lines.append(f"  - {gate}")

    lines.extend(
        [
            "",
            "## Concentration Test",
            "",
            f"- By symbol share: `{summary['concentration'].get('by_symbol_share', {})}`",
            f"- By year share: `{summary['concentration'].get('by_year_share', {})}`",
            f"- Top trade share: `{summary['concentration'].get('top_trade_share', 0.0):.1%}`",
            f"- Top-5 trade share: `{summary['concentration'].get('top5_trade_share', 0.0):.1%}`",
            "",
            "## Final Verdict",
            "",
            f"**{verdict}**: {summary['verdict']['reason']}",
            "",
            f"**{second_leg_line}**",
            "",
        ]
    )
    return "\n".join(lines)


def run_research(args: argparse.Namespace) -> dict[str, Any]:
    baseline_cfg = load_config(str(args.baseline_config))
    data_dir = baseline_cfg.data_dir
    embargo = load_embargo_config(args.validation_config)
    research_start = pd.Timestamp(args.research_start, tz="UTC") if args.research_start else None
    research_symbols = get_research_symbols(BASELINE_SYMBOLS, embargo)
    embargo_only = get_embargo_only_symbols(BASELINE_SYMBOLS, embargo)

    symbol_data: dict[str, SymbolData] = {}
    for sym in research_symbols:
        logger.info("Loading %s", sym)
        df_1h, q_1h = _load_symbol_df(
            baseline_cfg.resolve_kline_path(sym),
            embargo,
            research_start=research_start,
        )

        df_5m = None
        p_5m = data_dir / "binance" / "futures" / "5m" / f"{sym}.parquet"
        if p_5m.exists():
            df_5m, _ = _load_symbol_df(p_5m, embargo, research_start=research_start)

        oi = _load_oi_series(data_dir, sym, df_1h.index)
        lsr = _load_lsr_series(data_dir, sym, df_1h.index)
        baseline_pos = _compute_positions(args.baseline_config, sym, df_1h, data_dir)
        old_oi_pos = _compute_positions(args.legacy_config, sym, df_1h, data_dir)
        features = _build_features(sym, df_1h, oi)

        open_ret_1h = df_1h["open"].pct_change(fill_method=None).shift(-1).fillna(0.0)
        baseline_bar_ret = baseline_pos * open_ret_1h

        symbol_data[sym] = SymbolData(
            symbol=sym,
            df_1h=df_1h,
            df_5m=df_5m,
            oi=oi,
            lsr=lsr,
            baseline_pos=baseline_pos,
            old_oi_pos=old_oi_pos,
            features=features,
            baseline_bar_ret=baseline_bar_ret,
            data_quality=q_1h,
        )

    event_df = _event_map(symbol_data)
    orthogonality = _orthogonality_summary(symbol_data)
    scan_df = _entry_variant_scan(symbol_data, hold_bars=args.hold_bars)
    best_candidate = _pick_best_candidate(scan_df)
    best_entry = best_candidate["best"]["entry_variant"]
    best_htf = best_candidate["best"]["htf_variant"]
    bucket_relative_edge = _bucket_level_relative_edge(symbol_data, event_df, best_entry, best_htf)
    exit_paths = _analyze_exit_paths(symbol_data, best_entry, best_htf)
    eligibility = _symbol_eligibility(symbol_data, scan_df, best_entry, best_htf)
    concentration = _concentration_test(symbol_data, best_entry, best_htf)

    summary = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "baseline_config": str(args.baseline_config),
        "legacy_config": str(args.legacy_config),
        "validation_config": str(args.validation_config),
        "research_start": args.research_start,
        "research_symbols": research_symbols,
        "embargo_only_symbols": embargo_only,
        "data_gate": {
            "oi_proxy_available": True,
            "true_liquidation_state_available": bool(
                list((data_dir / "binance" / "futures" / "liquidation").glob("*.parquet"))
            ),
            "historical_lsr_available": bool(
                np.mean(
                    [
                        float(data.lsr.notna().mean()) if data.lsr is not None else 0.0
                        for data in symbol_data.values()
                    ]
                ) >= 0.30
            ),
            "oi_coverage_by_symbol": {
                sym: round(float(data.oi.notna().mean() * 100), 1)
                for sym, data in symbol_data.items()
            },
            "oi_gate_failed_symbols": [
                sym
                for sym, data in symbol_data.items()
                if float(data.oi.notna().mean()) < 0.70
            ],
            "data_quality": {sym: data.data_quality for sym, data in symbol_data.items()},
        },
        "event_map": {
            "total_events": int(len(event_df)),
            "bucket_counts": event_df["bucket"].value_counts().to_dict() if not event_df.empty else {},
            "old_oi_liq_bucket_coverage": (
                event_df.groupby("bucket")["old_oi_liq_hit"].mean().round(4).to_dict() if not event_df.empty else {}
            ),
            "baseline_state_by_bucket": (
                event_df.groupby(["bucket", "baseline_state"]).size().unstack(fill_value=0).to_dict()
                if not event_df.empty
                else {}
            ),
        },
        "orthogonality": orthogonality,
        "entry_scan": best_candidate["aggregate_table"],
        "best_candidate": best_candidate,
        "bucket_relative_edge": bucket_relative_edge,
        "exit_path_analysis": exit_paths,
        "symbol_eligibility": eligibility,
        "concentration": concentration,
    }
    summary["reopen_framework"] = _build_reopen_framework(summary)
    verdict_label, second_leg_line = _verdict(summary)
    oi_gate_failed = bool(summary["data_gate"]["oi_gate_failed_symbols"])
    no_true_liq = not summary["data_gate"]["true_liquidation_state_available"]
    no_hist_lsr = not summary["data_gate"]["historical_lsr_available"]
    shelved_blockers = []
    if oi_gate_failed:
        shelved_blockers.append("OI coverage gate still fails on part of the universe")
    if no_true_liq:
        shelved_blockers.append("true liquidation-state history is unavailable")
    if no_hist_lsr:
        shelved_blockers.append("historical LSR is still missing for full baseline replay")
    shelved_reason = (
        "Edge survives in proxy form, but " + ", ".join(shelved_blockers) + "."
        if shelved_blockers
        else "Edge survives in proxy form, but promotion is still deferred pending a cleaner state-based evidence stack."
    )
    summary["verdict"] = {
        "label": verdict_label,
        "reason": (
            "Best trigger still depends on HTF gating, density is below second-leg range, "
            "or correlation/edge does not clear the satellite threshold."
            if verdict_label == "FAIL"
            else shelved_reason
            if verdict_label == "SHELVED"
            else "There is a tradable reclaim-style entry with acceptable density and independence, but it still looks more like a module than a full leg."
            if verdict_label == "WEAK GO"
            else "Edge survives without hard HTF dependence, remains low-correlation to baseline, and reaches portfolio-relevant density."
        ),
        "second_leg_line": second_leg_line,
        "recommended_single_v2_direction": best_entry,
        "hard_blockers": {
            "oi_coverage_gate_failed": oi_gate_failed,
            "true_liquidation_state_missing": no_true_liq,
            "historical_lsr_missing": no_hist_lsr,
        },
    }
    return summary


def _write_outputs(summary: dict[str, Any], output_root: Path) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = output_root / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "summary.json"
    report_path = out_dir / "report.md"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    report_path.write_text(_build_report(summary), encoding="utf-8")

    latest_path = output_root / "latest_summary.json"
    latest_path.write_text(summary_path.read_text(encoding="utf-8"), encoding="utf-8")
    latest_report = output_root / "latest_report.md"
    latest_report.write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Forced deleveraging reversal EDA")
    parser.add_argument(
        "-c",
        "--baseline-config",
        type=Path,
        default=Path("config/prod_candidate_simplified.yaml"),
        help="Baseline production config",
    )
    parser.add_argument(
        "--legacy-config",
        type=Path,
        default=Path("config/prod_live_oi_liq_bounce.yaml"),
        help="Legacy OI liquidation bounce config used for historical coverage comparison",
    )
    parser.add_argument(
        "-v",
        "--validation-config",
        type=Path,
        default=Path("config/validation.yaml"),
        help="Validation config for embargo settings",
    )
    parser.add_argument(
        "--hold-bars",
        type=int,
        default=24,
        help="Fixed-hold window used for entry-framing comparison",
    )
    parser.add_argument(
        "--research-start",
        type=str,
        default=None,
        help="Optional UTC start date override for the research window (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/research/forced_deleveraging_reversal"),
        help="Research output directory",
    )
    return parser.parse_args()


BASELINE_CFG_PATH = Path("config/prod_candidate_simplified.yaml")


def main() -> None:
    args = parse_args()
    summary = run_research(args)
    out_dir = _write_outputs(summary, args.output_dir)

    print("=" * 88)
    print("Forced Deleveraging Reversal EDA")
    print("=" * 88)
    print(f"Research symbols: {summary['research_symbols']}")
    print(f"Embargo-only symbols: {summary['embargo_only_symbols']}")
    print(f"Best candidate: {summary['best_candidate']['best']}")
    print(f"Verdict: {summary['verdict']['label']}")
    print(summary["verdict"]["second_leg_line"])
    print(f"Outputs: {out_dir}")


if __name__ == "__main__":
    main()
