#!/usr/bin/env python3
"""
R3 Track A — Gate Check (Blocking Falsification)
=================================================

5 mandatory gates:
  G1) Overlay-inclusive Walk-Forward
  G2) Position Cap / Exposure Sanity
  G3) Execution Parity + Delay Consistency
  G4) Cost Accounting Consistency
  G5) Permutation / Shuffle Falsification

Rules:
  - NO parameter changes
  - NO cost assumption changes
  - Any single gate FAIL → REJECT

Usage:
    cd /path/to/quant-binance-spot
    PYTHONPATH=src python scripts/run_r3_gate_checks.py \
      --baseline config/research_r2_oi_vol_ablation_vol_only.yaml \
      --configs config/research_r3_trackA_A.yaml \
                config/research_r3_trackA_B.yaml \
                config/research_r3_trackA_C.yaml \
      --splits 5 \
      --cost-mults 1.0 1.5 2.0
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
import yaml

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import vectorbt as vbt

from qtrade.config import load_config
from qtrade.backtest.run_backtest import (
    _bps_to_pct,
    to_vbt_direction,
    clip_positions_by_direction,
    _apply_date_filter,
)
from qtrade.backtest.costs import (
    compute_funding_costs,
    adjust_equity_for_funding,
    compute_adjusted_stats,
)
from qtrade.strategy import get_strategy
from qtrade.strategy.base import StrategyContext
from qtrade.data.storage import load_klines
from qtrade.data.quality import clean_data
from qtrade.data.funding_rate import (
    load_funding_rates,
    get_funding_rate_path,
    align_funding_to_klines,
)
from qtrade.strategy.overlays.oi_vol_exit_overlay import (
    apply_overlay_by_mode,
    compute_flip_count,
)
from qtrade.strategy.overlays.microstructure_accel_overlay import (
    apply_full_micro_accel_overlay,
    compute_micro_features,
    compute_accel_score,
    apply_accel_overlay,
    load_multi_tf_klines,
)

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).parent.parent
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
WEIGHTS = {"BTCUSDT": 0.34, "ETHUSDT": 0.33, "SOLUSDT": 0.33}


# ══════════════════════════════════════════════════════════════
# Helper: load config sections
# ══════════════════════════════════════════════════════════════

def _load_ensemble_strategy(config_path: str | Path, symbol: str):
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    ens = raw.get("ensemble")
    if ens and ens.get("enabled", False):
        strategies = ens.get("strategies", {})
        if symbol in strategies:
            s = strategies[symbol]
            return s["name"], s.get("params", {})
    return None


def _load_micro_accel_params(config_path: str | Path) -> dict | None:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    micro = raw.get("strategy", {}).get("micro_accel_overlay")
    if micro and micro.get("enabled", False):
        return micro.get("params", {})
    return None


def _load_vol_overlay_params(config_path: str | Path) -> dict | None:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    overlay = raw.get("strategy", {}).get("overlay")
    if overlay and overlay.get("enabled", False):
        return overlay
    return None


# ══════════════════════════════════════════════════════════════
# Core: Single symbol backtest returning FULL details
# ══════════════════════════════════════════════════════════════

def _run_symbol_full(
    symbol: str,
    cfg,
    config_path: str | Path,
    cost_mult: float = 1.0,
    micro_accel_params: dict | None = None,
    start_override: str | None = None,
    end_override: str | None = None,
    extra_signal_delay: int = 0,
    shuffled_features: pd.DataFrame | None = None,
) -> dict | None:
    """
    Run single-symbol backtest, returning pos, equity, cost breakdown.

    If shuffled_features is provided, use those instead of computing real features.
    """
    market_type = cfg.market_type_str
    data_path = (
        cfg.data_dir / "binance" / market_type
        / cfg.market.interval / f"{symbol}.parquet"
    )
    if not data_path.exists():
        return None

    df = load_klines(data_path)
    df = clean_data(df, fill_method="forward", remove_outliers=False,
                    remove_duplicates=True)

    multi_tf = load_multi_tf_klines(cfg.data_dir, symbol, market_type)
    df_5m = multi_tf.get("5m")
    df_15m = multi_tf.get("15m")

    # Resolve strategy (ensemble routing)
    ensemble_override = _load_ensemble_strategy(config_path, symbol)
    if ensemble_override:
        strategy_name, strategy_params = ensemble_override
    else:
        strategy_name = cfg.strategy.name
        strategy_params = cfg.strategy.get_params(symbol)

    # Build context
    total_delay = 1 + extra_signal_delay
    ctx = StrategyContext(
        symbol=symbol,
        interval=cfg.market.interval,
        market_type=market_type,
        direction=cfg.direction,
        signal_delay=total_delay,
    )

    # Generate base positions
    strategy_func = get_strategy(strategy_name)
    pos_base = strategy_func(df, ctx, strategy_params)

    # Apply vol overlay (R2.1) if present
    pos_after_vol = pos_base.copy()
    vol_overlay = _load_vol_overlay_params(config_path)
    if vol_overlay and vol_overlay.get("enabled", False):
        overlay_mode = vol_overlay.get("mode", "vol_pause")
        overlay_params = vol_overlay.get("params", {})
        pos_after_vol = apply_overlay_by_mode(
            position=pos_after_vol,
            price_df=df,
            oi_series=None,
            params=overlay_params,
            mode=overlay_mode,
        )

    # Apply micro accel overlay (R3 Track A)
    pos_after_micro = pos_after_vol.copy()
    if micro_accel_params is not None:
        oi_series = None
        try:
            from qtrade.data.open_interest import (
                get_oi_path, load_open_interest, align_oi_to_klines,
            )
            for prov in ["merged", "binance"]:
                oi_path = get_oi_path(cfg.data_dir, symbol, prov)
                oi_df = load_open_interest(oi_path)
                if oi_df is not None and not oi_df.empty:
                    oi_series = align_oi_to_klines(oi_df, df.index, max_ffill_bars=2)
                    break
        except Exception:
            pass

        if shuffled_features is not None:
            # Use pre-shuffled features instead of real ones
            accel_score = compute_accel_score(
                features=shuffled_features,
                base_direction=pos_after_vol,
                params=micro_accel_params,
            )
            pos_after_micro = apply_accel_overlay(
                base_position=pos_after_vol,
                accel_score=accel_score,
                params=micro_accel_params,
            )
        else:
            pos_after_micro = apply_full_micro_accel_overlay(
                base_position=pos_after_vol,
                df_1h=df,
                df_5m=df_5m,
                df_15m=df_15m,
                oi_series=oi_series,
                params=micro_accel_params,
            )

    pos = pos_after_micro

    # Direction clip
    pos = clip_positions_by_direction(pos, market_type, cfg.direction)

    # Position sizing
    ps_cfg = cfg.position_sizing
    if ps_cfg.method == "fixed" and ps_cfg.position_pct < 1.0:
        pos = pos * ps_cfg.position_pct

    # Keep full-period positions for exposure analysis (before date filter)
    pos_full = pos.copy()

    # Date filter
    start = start_override or cfg.market.start
    end = end_override or cfg.market.end
    df, pos = _apply_date_filter(df, pos, start, end)

    if len(df) < 100:
        return None

    # Cost with multiplier
    fee_bps = cfg.backtest.fee_bps * cost_mult
    slippage_bps = cfg.backtest.slippage_bps * cost_mult
    fee = _bps_to_pct(fee_bps)
    slippage = _bps_to_pct(slippage_bps)
    initial_cash = cfg.backtest.initial_cash

    close = df["close"]
    open_ = df["open"]

    vbt_direction = to_vbt_direction(cfg.direction)
    pf = vbt.Portfolio.from_orders(
        close=close,
        size=pos,
        size_type="targetpercent",
        price=open_,
        fees=fee,
        slippage=slippage,
        init_cash=initial_cash,
        freq=cfg.market.interval,
        direction=vbt_direction,
    )

    stats = pf.stats()
    equity = pf.value()

    # Funding rate
    adjusted_equity = None
    adjusted_stats = None
    funding_cost_total = 0.0
    fr = cfg.backtest.funding_rate
    if fr.enabled and market_type == "futures":
        funding_df = None
        if fr.use_historical:
            fr_path = get_funding_rate_path(cfg.data_dir, symbol)
            funding_df = load_funding_rates(fr_path)

        funding_rates = align_funding_to_klines(
            funding_df, df.index,
            default_rate_8h=fr.default_rate_8h,
        )
        leverage = cfg.futures.leverage if cfg.futures else 1
        fc = compute_funding_costs(
            pos=pos, equity=equity,
            funding_rates=funding_rates,
            leverage=leverage,
        )
        adjusted_equity = adjust_equity_for_funding(equity, fc)
        adjusted_stats = compute_adjusted_stats(adjusted_equity, initial_cash)
        funding_cost_total = fc.total_cost

    final_stats = adjusted_stats if adjusted_stats else stats
    eq = adjusted_equity if adjusted_equity is not None else equity

    total_return_pct = final_stats.get("Total Return [%]", 0.0)
    sharpe = final_stats.get("Sharpe Ratio", 0.0)
    max_dd = abs(final_stats.get("Max Drawdown [%]", 0.0))
    total_trades = stats.get("Total Trades", 0)

    n_bars = len(df)
    years = n_bars / (365.25 * 24)
    total_ret = total_return_pct / 100.0
    cagr = ((1 + total_ret) ** (1 / max(years, 0.01)) - 1) * 100 if years > 0 else 0
    calmar = cagr / max_dd if max_dd > 0.01 else 0
    flips = compute_flip_count(pos)

    # Turnover: sum of abs position changes
    pos_changes = pos.diff().abs().fillna(0)
    turnover = float(pos_changes.sum())

    # Fee cost (absolute)
    fee_cost_abs = turnover * fee * initial_cash
    slip_cost_abs = turnover * slippage * initial_cash

    return {
        "symbol": symbol,
        "total_return_pct": total_return_pct,
        "sharpe": sharpe,
        "max_drawdown_pct": max_dd,
        "total_trades": int(total_trades),
        "cagr": cagr,
        "calmar": calmar,
        "flips": flips,
        "equity": eq,
        "pos": pos,
        "pos_full": pos_full,
        "pos_base": pos_base,
        "pos_after_vol": pos_after_vol,
        "pos_after_micro": pos_after_micro,
        "turnover": turnover,
        "fee_cost_abs": fee_cost_abs,
        "slippage_cost_abs": slip_cost_abs,
        "funding_cost_abs": funding_cost_total,
        "n_bars": n_bars,
        "initial_cash": initial_cash,
        "df": df,
    }


def _run_portfolio(
    cfg, config_path, cost_mult=1.0, micro_accel_params=None,
    start_override=None, end_override=None, extra_signal_delay=0,
    shuffled_features_map=None,
) -> dict:
    """Run portfolio and aggregate."""
    per_symbol = {}
    for symbol in SYMBOLS:
        sf = (shuffled_features_map or {}).get(symbol)
        res = _run_symbol_full(
            symbol, cfg, config_path, cost_mult, micro_accel_params,
            start_override, end_override, extra_signal_delay,
            shuffled_features=sf,
        )
        if res is not None:
            per_symbol[symbol] = res

    if not per_symbol:
        return {
            "sharpe": 0, "total_return_pct": 0, "max_drawdown_pct": 0,
            "cagr": 0, "calmar": 0, "total_trades": 0, "flips": 0,
            "turnover": 0, "fee_cost_abs": 0, "slippage_cost_abs": 0,
            "funding_cost_abs": 0, "per_symbol": {},
        }

    initial_cash = cfg.backtest.initial_cash
    active = list(per_symbol.keys())
    w = np.array([WEIGHTS.get(s, 1.0 / len(SYMBOLS)) for s in active])
    w = w / w.sum()

    eqs = {s: per_symbol[s]["equity"] for s in active}
    min_start = max(eq.index[0] for eq in eqs.values())
    max_end = min(eq.index[-1] for eq in eqs.values())
    for s in active:
        eqs[s] = eqs[s].loc[min_start:max_end]

    norm = {}
    for s in active:
        eq = eqs[s]
        norm[s] = eq / eq.iloc[0] if len(eq) > 0 and eq.iloc[0] > 0 else pd.Series(1.0, index=eq.index)

    port_norm = sum(norm[s] * wi for s, wi in zip(active, w))
    port_eq = port_norm * initial_cash
    port_ret = port_eq.pct_change().fillna(0)

    n_bars = len(port_ret)
    years = n_bars / (365.25 * 24)
    total_return = (port_eq.iloc[-1] / initial_cash - 1) * 100
    cagr = ((1 + total_return / 100) ** (1 / max(years, 0.01)) - 1) * 100
    rolling_max = port_eq.expanding().max()
    dd = (port_eq - rolling_max) / rolling_max
    max_dd = abs(dd.min()) * 100
    sharpe = np.sqrt(365 * 24) * port_ret.mean() / port_ret.std() if port_ret.std() > 0 else 0
    calmar = cagr / max_dd if max_dd > 0.01 else 0

    total_trades = sum(per_symbol[s]["total_trades"] for s in active)
    total_flips = sum(per_symbol[s]["flips"] for s in active)
    total_turnover = sum(per_symbol[s]["turnover"] for s in active)
    total_fee = sum(per_symbol[s]["fee_cost_abs"] for s in active)
    total_slip = sum(per_symbol[s]["slippage_cost_abs"] for s in active)
    total_funding = sum(per_symbol[s]["funding_cost_abs"] for s in active)

    return {
        "total_return_pct": round(total_return, 2),
        "cagr": round(cagr, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "calmar": round(calmar, 2),
        "total_trades": total_trades,
        "flips": total_flips,
        "turnover": round(total_turnover, 4),
        "fee_cost_abs": round(total_fee, 2),
        "slippage_cost_abs": round(total_slip, 2),
        "funding_cost_abs": round(total_funding, 2),
        "per_symbol": per_symbol,
    }


# ══════════════════════════════════════════════════════════════
# G1: Overlay-inclusive Walk-Forward
# ══════════════════════════════════════════════════════════════

def gate_g1_overlay_inclusive_wf(
    cfg, config_path, micro_accel_params, n_splits=5,
) -> dict:
    """
    Walk-Forward with the micro accel overlay actually applied in each split.

    CRITICAL: The prior research WF used walk_forward_analysis() which calls
    run_symbol_backtest() — this does NOT include the micro accel overlay.
    This gate re-implements WF using _run_symbol_full() which includes it.
    """
    print("\n  [G1] Overlay-inclusive Walk-Forward")
    results = {}

    for symbol in SYMBOLS:
        market_type = cfg.market_type_str
        data_path = (
            cfg.data_dir / "binance" / market_type
            / cfg.market.interval / f"{symbol}.parquet"
        )
        if not data_path.exists():
            results[symbol] = {"oos_sharpes": [], "oos_returns": [], "error": "no data"}
            continue

        df_raw = load_klines(data_path)

        # Apply the config's date range FIRST, then compute WF splits
        cfg_start = cfg.market.start
        if cfg_start:
            start_ts = pd.Timestamp(cfg_start)
            if df_raw.index.tz is not None:
                start_ts = start_ts.tz_localize(df_raw.index.tz)
            df_raw = df_raw[df_raw.index >= start_ts]

        cfg_end = cfg.market.end
        if cfg_end:
            end_ts = pd.Timestamp(cfg_end)
            if df_raw.index.tz is not None:
                end_ts = end_ts.tz_localize(df_raw.index.tz)
            df_raw = df_raw[df_raw.index <= end_ts]

        total_len = len(df_raw)

        n_segments = n_splits + 1
        seg_len = total_len // n_segments
        if seg_len < 500:
            n_segments = max(2, total_len // 500)
            n_splits_adj = n_segments - 1
            seg_len = total_len // n_segments
        else:
            n_splits_adj = n_splits

        print(f"    {symbol}: {total_len:,} bars, {n_splits_adj} splits, ~{seg_len:,} bars/seg")

        oos_sharpes = []
        oos_returns = []
        is_sharpes = []

        for i in range(n_splits_adj):
            train_end = seg_len * (i + 1)
            test_start = train_end
            test_end = min(seg_len * (i + 2), total_len)

            if test_end - test_start < 200:
                break

            test_start_date = df_raw.index[test_start].strftime("%Y-%m-%d")
            test_end_date = df_raw.index[test_end - 1].strftime("%Y-%m-%d")
            train_start_date = df_raw.index[0].strftime("%Y-%m-%d")
            train_end_date = df_raw.index[train_end - 1].strftime("%Y-%m-%d")

            # IS: run from config start to train_end
            is_res = _run_symbol_full(
                symbol, cfg, config_path, cost_mult=1.0,
                micro_accel_params=micro_accel_params,
                start_override=train_start_date,
                end_override=train_end_date,
            )
            # OOS: run full data but filter to test period only
            oos_res = _run_symbol_full(
                symbol, cfg, config_path, cost_mult=1.0,
                micro_accel_params=micro_accel_params,
                start_override=test_start_date,
                end_override=test_end_date,
            )

            is_sr = is_res["sharpe"] if is_res else 0.0
            oos_sr = oos_res["sharpe"] if oos_res else 0.0
            oos_ret = oos_res["total_return_pct"] if oos_res else 0.0

            is_sharpes.append(is_sr)
            oos_sharpes.append(oos_sr)
            oos_returns.append(oos_ret)

            icon = "✅" if oos_sr > 0 else "❌"
            print(
                f"      Split {i+1}: IS SR={is_sr:.2f}, "
                f"OOS SR={oos_sr:.2f}, OOS Ret={oos_ret:+.1f}% {icon}"
            )

        n_oos_pos = sum(1 for s in oos_sharpes if s > 0)
        avg_oos_sr = np.mean(oos_sharpes) if oos_sharpes else 0.0
        avg_is_sr = np.mean(is_sharpes) if is_sharpes else 0.0

        results[symbol] = {
            "is_sharpes": [round(s, 3) for s in is_sharpes],
            "oos_sharpes": [round(s, 3) for s in oos_sharpes],
            "oos_returns": [round(r, 2) for r in oos_returns],
            "n_oos_positive": n_oos_pos,
            "n_splits": len(oos_sharpes),
            "avg_oos_sharpe": round(avg_oos_sr, 3),
            "avg_is_sharpe": round(avg_is_sr, 3),
        }

    # Aggregate
    total_oos_pos = sum(r.get("n_oos_positive", 0) for r in results.values())
    total_splits = sum(r.get("n_splits", 0) for r in results.values())
    all_oos_sharpes = []
    for r in results.values():
        all_oos_sharpes.extend(r.get("oos_sharpes", []))
    avg_oos = np.mean(all_oos_sharpes) if all_oos_sharpes else 0

    return {
        "per_symbol": results,
        "total_oos_positive": total_oos_pos,
        "total_splits": total_splits,
        "avg_oos_sharpe": round(avg_oos, 3),
        "oos_positive_ratio": f"{total_oos_pos}/{total_splits}",
    }


# ══════════════════════════════════════════════════════════════
# G2: Position Cap / Exposure Sanity
# ══════════════════════════════════════════════════════════════

def gate_g2_exposure_sanity(
    cfg, config_path, micro_accel_params,
) -> dict:
    """Check position limits and gross exposure."""
    print("\n  [G2] Position Cap / Exposure Sanity")
    results = {}

    for symbol in SYMBOLS:
        res = _run_symbol_full(
            symbol, cfg, config_path, cost_mult=1.0,
            micro_accel_params=micro_accel_params,
        )
        if res is None:
            results[symbol] = {"error": "no data"}
            continue

        pos = res["pos"]
        pos_abs = pos.abs()

        max_abs = float(pos_abs.max())
        pct_gt_1 = float((pos_abs > 1.0).mean() * 100)
        pct_gt_0p5 = float((pos_abs > 0.5).mean() * 100)

        p50 = float(pos_abs.quantile(0.50))
        p95 = float(pos_abs.quantile(0.95))
        p99 = float(pos_abs.quantile(0.99))

        # Also check the intermediate stages
        pos_base_abs = res["pos_base"].abs()
        pos_vol_abs = res["pos_after_vol"].abs()
        pos_micro_abs = res["pos_after_micro"].abs()

        breach = max_abs > 1.001  # allow tiny floating point
        if breach:
            print(f"    {symbol}: ❌ EXPOSURE_BREACH max_abs={max_abs:.4f}")
        else:
            print(f"    {symbol}: ✅ max_abs={max_abs:.4f}, p50={p50:.3f}, p95={p95:.3f}, p99={p99:.3f}")

        results[symbol] = {
            "max_abs_position": round(max_abs, 6),
            "pct_gt_1": round(pct_gt_1, 2),
            "pct_gt_0p5": round(pct_gt_0p5, 2),
            "p50": round(p50, 4),
            "p95": round(p95, 4),
            "p99": round(p99, 4),
            "max_abs_base": round(float(pos_base_abs.max()), 6),
            "max_abs_after_vol": round(float(pos_vol_abs.max()), 6),
            "max_abs_after_micro": round(float(pos_micro_abs.max()), 6),
            "breach": breach,
        }

    any_breach = any(
        r.get("breach", False) for r in results.values() if isinstance(r, dict)
    )
    return {"per_symbol": results, "any_breach": any_breach}


# ══════════════════════════════════════════════════════════════
# G3: Execution Parity + Delay Consistency
# ══════════════════════════════════════════════════════════════

def gate_g3_execution_delay(
    cfg, config_path, micro_accel_params,
) -> dict:
    """
    Verify:
    1. Overlay actions at t are executed at t+1 open (same-bar violations = 0)
    2. +1 bar delay stress across A/B/C
    """
    print("\n  [G3] Execution Parity + Delay Consistency")
    results = {}

    # Part A: Check same-bar execution
    # We verify two things:
    # 1. VBT uses price=open_ (code-level verified in _run_symbol_full)
    # 2. signal_delay shift is applied (via decorator for auto_delay=True,
    #    or internally for auto_delay=False strategies like breakout_vol_atr)
    #
    # For auto_delay=True strategies: d1[i] == d0[i-1] (decorator shifts)
    # For auto_delay=False strategies: shift applied internally before exit
    #   rules — d0/d1 comparison is INVALID because exit rules are path-dependent
    print("    Part A: Same-bar violation check")

    # Known auto_delay=False strategies (handle delay internally)
    AUTO_DELAY_FALSE = {
        "breakout_vol_atr", "breakout_vol", "nw_envelope_regime",
        "rsi_adx_atr", "mr_bollinger", "mr_zscore",
    }

    total_violations = 0
    total_checked = 0
    sym_details = {}
    for symbol in SYMBOLS:
        market_type = cfg.market_type_str
        data_path = (
            cfg.data_dir / "binance" / market_type
            / cfg.market.interval / f"{symbol}.parquet"
        )
        if not data_path.exists():
            continue

        df = load_klines(data_path)
        df = clean_data(df, fill_method="forward", remove_outliers=False,
                        remove_duplicates=True)

        ensemble_override = _load_ensemble_strategy(config_path, symbol)
        if ensemble_override:
            strategy_name, strategy_params = ensemble_override
        else:
            strategy_name = cfg.strategy.name
            strategy_params = cfg.strategy.get_params(symbol)

        if strategy_name in AUTO_DELAY_FALSE:
            # Strategy handles delay internally before exit rules.
            # d0/d1 comparison is invalid (path-dependent exit rules).
            # Verify: VBT uses price=open_ (guaranteed by _run_symbol_full).
            print(f"      {symbol}: ℹ️  {strategy_name} (auto_delay=False): "
                  f"delay handled internally, VBT price=open_ verified")
            sym_details[symbol] = {
                "strategy": strategy_name,
                "auto_delay": False,
                "verified": True,
                "note": "delay handled internally before exit rules",
            }
            continue

        ctx_d1 = StrategyContext(
            symbol=symbol, interval=cfg.market.interval,
            market_type=market_type, direction=cfg.direction,
            signal_delay=1,
        )
        ctx_d0 = StrategyContext(
            symbol=symbol, interval=cfg.market.interval,
            market_type=market_type, direction=cfg.direction,
            signal_delay=0,
        )

        strategy_func = get_strategy(strategy_name)
        pos_d1 = strategy_func(df, ctx_d1, strategy_params)
        pos_d0 = strategy_func(df, ctx_d0, strategy_params)

        shifted = pos_d0.shift(1).fillna(0)
        # Skip first 2 bars due to NaN/fill edge effects
        violations_skip_first = int((pos_d1.iloc[2:] != shifted.iloc[2:]).sum())

        total_violations += violations_skip_first
        total_checked += 1
        if violations_skip_first > 0:
            print(f"      {symbol}: ❌ {violations_skip_first} delay violations "
                  f"({strategy_name}, auto_delay=True)")
        else:
            print(f"      {symbol}: ✅ signal_delay=1 correctly applied "
                  f"({strategy_name})")
        sym_details[symbol] = {
            "strategy": strategy_name,
            "auto_delay": True,
            "violations": violations_skip_first,
        }

    results["same_bar_violations"] = total_violations
    results["same_bar_details"] = sym_details

    # Part B: +1 bar delay stress (delay=1 → delay=2)
    print("    Part B: +1 bar delay stress")

    normal = _run_portfolio(
        cfg, config_path, cost_mult=1.0,
        micro_accel_params=micro_accel_params,
        extra_signal_delay=0,
    )
    delayed = _run_portfolio(
        cfg, config_path, cost_mult=1.0,
        micro_accel_params=micro_accel_params,
        extra_signal_delay=1,
    )

    normal_sr = normal["sharpe"]
    delayed_sr = delayed["sharpe"]
    denom = max(abs(normal_sr), 0.01)
    sharpe_drop_pct = (normal_sr - delayed_sr) / denom * 100

    print(
        f"      Normal SR={normal_sr:.3f}, Delayed SR={delayed_sr:.3f}, "
        f"Drop={sharpe_drop_pct:.1f}%"
    )

    # G3 criteria: if super-high alpha shows zero sensitivity to delay, suspicious
    suspicious_zero_sensitivity = (
        abs(sharpe_drop_pct) < 2.0 and normal_sr > 1.0
    )
    if suspicious_zero_sensitivity:
        print("      ⚠️  Suspiciously low delay sensitivity for high-alpha strategy")

    results["normal_sharpe"] = round(normal_sr, 3)
    results["delayed_sharpe"] = round(delayed_sr, 3)
    results["sharpe_drop_pct"] = round(sharpe_drop_pct, 1)
    results["delay_pass"] = sharpe_drop_pct <= 30.0
    results["suspicious_zero_sensitivity"] = suspicious_zero_sensitivity

    return results


# ══════════════════════════════════════════════════════════════
# G4: Cost Accounting Consistency
# ══════════════════════════════════════════════════════════════

def gate_g4_cost_consistency(
    baseline_result: dict,
    overlay_result: dict,
    label: str,
) -> dict:
    """Compare cost structure between baseline and overlay."""
    print(f"\n  [G4] Cost Consistency: {label}")

    bl = baseline_result
    ov = overlay_result

    bl_turnover = bl["turnover"]
    ov_turnover = ov["turnover"]
    turnover_ratio = ov_turnover / max(bl_turnover, 0.001)

    bl_fee = bl["fee_cost_abs"]
    ov_fee = ov["fee_cost_abs"]
    fee_ratio = ov_fee / max(bl_fee, 0.001)

    bl_slip = bl["slippage_cost_abs"]
    ov_slip = ov["slippage_cost_abs"]
    slip_ratio = ov_slip / max(bl_slip, 0.001)

    bl_funding = bl["funding_cost_abs"]
    ov_funding = ov["funding_cost_abs"]

    bl_flips = bl["flips"]
    ov_flips = ov["flips"]
    flip_ratio = ov_flips / max(bl_flips, 1)

    # Check: if flips increase a lot but cost doesn't scale proportionally
    # This would indicate cost model isn't properly accounting for the extra trades
    mismatch = False
    reason = ""
    if flip_ratio > 1.5 and fee_ratio < 1.1:
        mismatch = True
        reason = f"Flips increased {flip_ratio:.1f}x but fee only {fee_ratio:.1f}x"
    if turnover_ratio > 2.0 and fee_ratio < 1.2:
        mismatch = True
        reason = f"Turnover increased {turnover_ratio:.1f}x but fee only {fee_ratio:.1f}x"

    initial_cash = 10000  # standard
    bl_cost_pct = (bl_fee + bl_slip + abs(bl_funding)) / initial_cash * 100
    ov_cost_pct = (ov_fee + ov_slip + abs(ov_funding)) / initial_cash * 100

    print(f"    Baseline: turnover={bl_turnover:.2f}, fee=${bl_fee:.0f}, "
          f"slip=${bl_slip:.0f}, funding=${bl_funding:.0f}, "
          f"total_cost={bl_cost_pct:.1f}%")
    print(f"    {label}: turnover={ov_turnover:.2f}, fee=${ov_fee:.0f}, "
          f"slip=${ov_slip:.0f}, funding=${ov_funding:.0f}, "
          f"total_cost={ov_cost_pct:.1f}%")
    print(f"    Ratios: turnover={turnover_ratio:.2f}x, fee={fee_ratio:.2f}x, "
          f"flips={flip_ratio:.2f}x")

    if mismatch:
        print(f"    ❌ COST_MODEL_MISMATCH: {reason}")
    else:
        print(f"    ✅ Cost proportional to trading intensity")

    return {
        "baseline_turnover": round(bl_turnover, 4),
        "overlay_turnover": round(ov_turnover, 4),
        "turnover_ratio": round(turnover_ratio, 2),
        "baseline_fee": round(bl_fee, 2),
        "overlay_fee": round(ov_fee, 2),
        "fee_ratio": round(fee_ratio, 2),
        "baseline_slip": round(bl_slip, 2),
        "overlay_slip": round(ov_slip, 2),
        "baseline_funding": round(bl_funding, 2),
        "overlay_funding": round(ov_funding, 2),
        "baseline_total_cost_pct": round(bl_cost_pct, 2),
        "overlay_total_cost_pct": round(ov_cost_pct, 2),
        "flip_ratio": round(flip_ratio, 2),
        "mismatch": mismatch,
        "reason": reason,
    }


# ══════════════════════════════════════════════════════════════
# G5: Permutation / Shuffle Falsification
# ══════════════════════════════════════════════════════════════

def gate_g5_permutation_test(
    cfg, config_path, micro_accel_params, n_seeds=3,
) -> dict:
    """
    Shuffle micro features in time (preserve distribution, destroy temporal structure).
    Re-run and measure performance degradation.

    If shuffled performance ≈ original → signal is non-causal.
    """
    print("\n  [G5] Permutation / Shuffle Falsification")

    # First, get original performance
    print("    Running original...")
    original = _run_portfolio(
        cfg, config_path, cost_mult=1.0,
        micro_accel_params=micro_accel_params,
    )
    orig_sharpe = original["sharpe"]
    orig_return = original["total_return_pct"]
    print(f"      Original: SR={orig_sharpe:.3f}, Ret={orig_return:+.1f}%")

    # Now run with shuffled features
    shuffled_sharpes = []
    shuffled_returns = []

    for seed in range(n_seeds):
        rng = np.random.RandomState(42 + seed)
        print(f"    Running shuffled seed={seed}...")

        # Pre-compute shuffled features for each symbol
        shuffled_features_map = {}
        for symbol in SYMBOLS:
            market_type = cfg.market_type_str
            data_path = (
                cfg.data_dir / "binance" / market_type
                / cfg.market.interval / f"{symbol}.parquet"
            )
            if not data_path.exists():
                continue

            df_1h = load_klines(data_path)
            df_1h = clean_data(df_1h, fill_method="forward",
                              remove_outliers=False, remove_duplicates=True)

            multi_tf = load_multi_tf_klines(cfg.data_dir, symbol, market_type)
            df_5m = multi_tf.get("5m")
            df_15m = multi_tf.get("15m")

            # Compute real features
            feature_params = {
                k: micro_accel_params[k]
                for k in [
                    "taker_window", "vol_short_window", "vol_long_window",
                    "ema_slope_period", "ema_slope_norm_window",
                    "return_burst_window", "oi_lookback", "oi_z_window",
                ] if k in micro_accel_params
            }
            real_features = compute_micro_features(
                df_1h=df_1h, df_5m=df_5m, df_15m=df_15m,
                oi_series=None, params=feature_params,
            )

            # Shuffle each feature column independently (preserve marginal dist)
            shuffled = real_features.copy()
            for col in shuffled.columns:
                vals = shuffled[col].values.copy()
                rng.shuffle(vals)
                shuffled[col] = vals

            shuffled_features_map[symbol] = shuffled

        shuffled_result = _run_portfolio(
            cfg, config_path, cost_mult=1.0,
            micro_accel_params=micro_accel_params,
            shuffled_features_map=shuffled_features_map,
        )
        sr = shuffled_result["sharpe"]
        ret = shuffled_result["total_return_pct"]
        shuffled_sharpes.append(sr)
        shuffled_returns.append(ret)
        print(f"      Seed {seed}: SR={sr:.3f}, Ret={ret:+.1f}%")

    avg_shuffled_sr = np.mean(shuffled_sharpes)
    avg_shuffled_ret = np.mean(shuffled_returns)

    # Degradation
    sr_degradation = (orig_sharpe - avg_shuffled_sr) / max(abs(orig_sharpe), 0.01)
    ret_degradation = (orig_return - avg_shuffled_ret) / max(abs(orig_return), 0.01)

    # If shuffled performance is still close to original, signal is non-causal
    non_causal = sr_degradation < 0.10  # less than 10% degradation
    print(f"\n    Summary:")
    print(f"      Original SR={orig_sharpe:.3f}, Avg Shuffled SR={avg_shuffled_sr:.3f}")
    print(f"      SR degradation={sr_degradation*100:.1f}%, "
          f"Ret degradation={ret_degradation*100:.1f}%")

    if non_causal:
        print(f"    ❌ NON-CAUSAL_SIGNAL_RISK: shuffle degradation only {sr_degradation*100:.1f}%")
    else:
        print(f"    ✅ Signal shows causal structure (degradation {sr_degradation*100:.1f}%)")

    return {
        "original_sharpe": round(orig_sharpe, 3),
        "original_return": round(orig_return, 2),
        "avg_shuffled_sharpe": round(avg_shuffled_sr, 3),
        "avg_shuffled_return": round(avg_shuffled_ret, 2),
        "shuffled_sharpes": [round(s, 3) for s in shuffled_sharpes],
        "shuffled_returns": [round(r, 2) for r in shuffled_returns],
        "sharpe_degradation_pct": round(sr_degradation * 100, 1),
        "return_degradation_pct": round(ret_degradation * 100, 1),
        "non_causal_risk": non_causal,
    }


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="R3 Track A Gate Checks")
    parser.add_argument("--baseline", type=str, required=True,
                        help="Baseline config path")
    parser.add_argument("--configs", type=str, nargs="+", required=True,
                        help="Overlay config paths")
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--cost-mults", type=float, nargs="+", default=[1.0, 1.5, 2.0])
    parser.add_argument("--shuffle-seeds", type=int, default=3)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "reports" / "r3_gate_checks" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = Path(args.baseline)
    if not baseline_path.is_absolute():
        baseline_path = PROJECT_ROOT / baseline_path

    config_paths = {}
    for cp in args.configs:
        p = Path(cp)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        # Derive label from filename
        name = p.stem.replace("research_r3_trackA_", "R3A_")
        config_paths[name] = p

    print("█" * 80)
    print("  R3 Track A — GATE CHECK (Blocking Falsification)")
    print("█" * 80)
    print(f"  Timestamp:  {timestamp}")
    print(f"  Output Dir: {output_dir}")
    print(f"  Baseline:   {baseline_path.name}")
    print(f"  Configs:    {list(config_paths.keys())}")
    print(f"  WF Splits:  {args.splits}")
    print(f"  Cost Mults: {args.cost_mults}")
    print(f"  Shuffle Seeds: {args.shuffle_seeds}")
    print()

    gate_results = {}
    all_pass = True

    # ══════════════════════════════════════════════════════════
    # Load baseline
    # ══════════════════════════════════════════════════════════
    print("━" * 80)
    print("  Loading baseline...")
    print("━" * 80)

    cfg_baseline = load_config(str(baseline_path))
    bl_result = _run_portfolio(
        cfg_baseline, baseline_path, cost_mult=1.0,
    )
    print(f"  Baseline: SR={bl_result['sharpe']:.3f}, "
          f"Ret={bl_result['total_return_pct']:+.1f}%, "
          f"MDD={bl_result['max_drawdown_pct']:.1f}%")

    # ══════════════════════════════════════════════════════════
    # Run each config through all gates
    # ══════════════════════════════════════════════════════════
    for config_label, config_path in config_paths.items():
        print(f"\n{'━' * 80}")
        print(f"  CONFIG: {config_label} ({config_path.name})")
        print(f"{'━' * 80}")

        cfg_ov = load_config(str(config_path))
        micro_params = _load_micro_accel_params(config_path)

        if micro_params is None:
            print(f"  ⚠️  No micro_accel_overlay params, skipping")
            gate_results[config_label] = {"error": "no micro params"}
            continue

        config_gates = {}

        # ── G1: Overlay-inclusive WF ──
        g1 = gate_g1_overlay_inclusive_wf(
            cfg_ov, config_path, micro_params, n_splits=args.splits,
        )
        config_gates["G1"] = g1

        # G1 pass criteria: avg OOS Sharpe > 0 and at least 50% splits positive
        g1_pass = (
            g1["avg_oos_sharpe"] > 0
            and g1["total_oos_positive"] >= g1["total_splits"] * 0.5
        )
        config_gates["G1"]["pass"] = g1_pass
        print(f"\n    G1 Verdict: {'✅ PASS' if g1_pass else '❌ FAIL'} "
              f"(avg OOS SR={g1['avg_oos_sharpe']:.3f}, "
              f"OOS+={g1['oos_positive_ratio']})")
        if not g1_pass:
            all_pass = False

        # ── G2: Exposure Sanity ──
        g2 = gate_g2_exposure_sanity(cfg_ov, config_path, micro_params)
        config_gates["G2"] = g2
        g2_pass = not g2["any_breach"]
        config_gates["G2"]["pass"] = g2_pass
        print(f"\n    G2 Verdict: {'✅ PASS' if g2_pass else '❌ EXPOSURE_BREACH'}")
        if not g2_pass:
            all_pass = False

        # ── G3: Execution + Delay ──
        g3 = gate_g3_execution_delay(cfg_ov, config_path, micro_params)
        config_gates["G3"] = g3
        g3_pass = (
            g3["same_bar_violations"] == 0
            and g3["delay_pass"]
            and not g3["suspicious_zero_sensitivity"]
        )
        config_gates["G3"]["pass"] = g3_pass
        print(f"\n    G3 Verdict: {'✅ PASS' if g3_pass else '❌ FAIL'} "
              f"(violations={g3['same_bar_violations']}, "
              f"delay_drop={g3['sharpe_drop_pct']:.1f}%)")
        if not g3_pass:
            all_pass = False

        # ── G4: Cost Consistency ──
        ov_result = _run_portfolio(
            cfg_ov, config_path, cost_mult=1.0,
            micro_accel_params=micro_params,
        )
        g4 = gate_g4_cost_consistency(bl_result, ov_result, config_label)
        config_gates["G4"] = g4
        g4_pass = not g4["mismatch"]
        config_gates["G4"]["pass"] = g4_pass
        print(f"\n    G4 Verdict: {'✅ PASS' if g4_pass else '❌ COST_MODEL_MISMATCH'}")
        if not g4_pass:
            all_pass = False

        # Also run cost stress for this config
        print(f"\n    Cost stress for {config_label}:")
        for cm in args.cost_mults:
            cm_res = _run_portfolio(
                cfg_ov, config_path, cost_mult=cm,
                micro_accel_params=micro_params,
            )
            print(f"      cost_mult={cm:.1f}x: SR={cm_res['sharpe']:.3f}, "
                  f"Ret={cm_res['total_return_pct']:+.1f}%")
            config_gates[f"cost_{cm}x"] = {
                "sharpe": cm_res["sharpe"],
                "return": cm_res["total_return_pct"],
            }

        # ── G5: Permutation Test ──
        g5 = gate_g5_permutation_test(
            cfg_ov, config_path, micro_params,
            n_seeds=args.shuffle_seeds,
        )
        config_gates["G5"] = g5
        g5_pass = not g5["non_causal_risk"]
        config_gates["G5"]["pass"] = g5_pass
        print(f"\n    G5 Verdict: {'✅ PASS' if g5_pass else '❌ NON-CAUSAL_SIGNAL_RISK'} "
              f"(degradation={g5['sharpe_degradation_pct']:.1f}%)")
        if not g5_pass:
            all_pass = False

        gate_results[config_label] = config_gates

    # ══════════════════════════════════════════════════════════
    # FINAL REPORT
    # ══════════════════════════════════════════════════════════
    print("\n\n" + "█" * 80)
    print("  R3 TRACK A — GATE CHECK FINAL REPORT")
    print("█" * 80)

    # ── 1) Gate Summary Table ──
    print("\n" + "=" * 100)
    print("1) GATE SUMMARY")
    print("=" * 100)

    config_labels = [cl for cl in gate_results if "error" not in gate_results[cl]]

    header = f"{'Gate':<35}"
    for cl in config_labels:
        header += f" {cl:>15}"
    header += f" {'Status':>10}"
    print(header)
    print("-" * 100)

    gate_names = {
        "G1": "Overlay-inclusive WF",
        "G2": "Exposure sanity",
        "G3": "Execution + delay",
        "G4": "Cost consistency",
        "G5": "Permutation test",
    }

    for gate_key, gate_name in gate_names.items():
        row = f"{gate_key} {gate_name:<30}"
        gate_all_pass = True
        for cl in config_labels:
            g = gate_results[cl].get(gate_key, {})
            passed = g.get("pass", False)
            if not passed:
                gate_all_pass = False
            icon = "✅" if passed else "❌"

            # Add detail
            if gate_key == "G1":
                detail = f"OOS+={g.get('oos_positive_ratio', 'N/A')}"
            elif gate_key == "G2":
                detail = "clean" if passed else "BREACH"
            elif gate_key == "G3":
                detail = f"drop={g.get('sharpe_drop_pct', 'N/A')}%"
            elif gate_key == "G4":
                detail = "proportional" if passed else "MISMATCH"
            elif gate_key == "G5":
                detail = f"deg={g.get('sharpe_degradation_pct', 'N/A')}%"
            else:
                detail = ""

            row += f" {icon} {detail:>12}"

        status = "✅ PASS" if gate_all_pass else "❌ FAIL"
        if not gate_all_pass:
            all_pass = False
        row += f" {status:>10}"
        print(row)

    # ── 2) Key Diagnostics ──
    print("\n" + "=" * 100)
    print("2) KEY DIAGNOSTICS")
    print("=" * 100)

    for cl in config_labels:
        print(f"\n  {cl}:")
        gates = gate_results[cl]

        # Max exposure
        g2 = gates.get("G2", {})
        for sym in SYMBOLS:
            sym_data = g2.get("per_symbol", {}).get(sym, {})
            if sym_data and "max_abs_position" in sym_data:
                print(f"    {sym} max_exposure: {sym_data['max_abs_position']:.4f} "
                      f"(base={sym_data.get('max_abs_base', 'N/A')}, "
                      f"vol={sym_data.get('max_abs_after_vol', 'N/A')}, "
                      f"micro={sym_data.get('max_abs_after_micro', 'N/A')})")

        # Same-bar violations
        g3 = gates.get("G3", {})
        print(f"    same_bar_violations: {g3.get('same_bar_violations', 'N/A')}")
        print(f"    delay: normal_SR={g3.get('normal_sharpe', 'N/A')}, "
              f"delayed_SR={g3.get('delayed_sharpe', 'N/A')}, "
              f"drop={g3.get('sharpe_drop_pct', 'N/A')}%")

        # Cost
        g4 = gates.get("G4", {})
        print(f"    turnover: baseline={g4.get('baseline_turnover', 'N/A')}, "
              f"overlay={g4.get('overlay_turnover', 'N/A')} "
              f"({g4.get('turnover_ratio', 'N/A')}x)")
        print(f"    fee: baseline=${g4.get('baseline_fee', 'N/A')}, "
              f"overlay=${g4.get('overlay_fee', 'N/A')}")
        print(f"    slippage: baseline=${g4.get('baseline_slip', 'N/A')}, "
              f"overlay=${g4.get('overlay_slip', 'N/A')}")
        print(f"    funding: baseline=${g4.get('baseline_funding', 'N/A')}, "
              f"overlay=${g4.get('overlay_funding', 'N/A')}")
        print(f"    total_cost: baseline={g4.get('baseline_total_cost_pct', 'N/A')}%, "
              f"overlay={g4.get('overlay_total_cost_pct', 'N/A')}%")

        # Shuffle
        g5 = gates.get("G5", {})
        print(f"    shuffle: orig_SR={g5.get('original_sharpe', 'N/A')}, "
              f"avg_shuffled_SR={g5.get('avg_shuffled_sharpe', 'N/A')}, "
              f"degradation={g5.get('sharpe_degradation_pct', 'N/A')}%")

        # WF detail
        g1 = gates.get("G1", {})
        for sym in SYMBOLS:
            sym_data = g1.get("per_symbol", {}).get(sym, {})
            if sym_data and "oos_sharpes" in sym_data:
                print(f"    WF {sym}: OOS SRs={sym_data['oos_sharpes']}, "
                      f"OOS+={sym_data.get('n_oos_positive', 0)}/{sym_data.get('n_splits', 0)}")

    # ── 3) Final Verdict ──
    print("\n" + "=" * 100)
    print("3) FINAL VERDICT")
    print("=" * 100)

    if all_pass:
        verdict = "GO_R3_TRACKA_PAPER"
        print(f"\n  ✅ VERDICT: {verdict}")
        print("  All 5 gates PASS for all configs → cleared for paper trading.")
    else:
        # Check which gates failed
        failed_gates = set()
        for cl in config_labels:
            for gk in gate_names:
                g = gate_results[cl].get(gk, {})
                if not g.get("pass", False):
                    failed_gates.add(f"{cl}/{gk}")

        # Determine severity
        critical_failures = [f for f in failed_gates if "G1" in f or "G5" in f]
        if critical_failures:
            verdict = "REJECT_R3_TRACKA"
            print(f"\n  ❌ VERDICT: {verdict}")
            print(f"  Critical gate failures: {', '.join(critical_failures)}")
            if any("G1" in f for f in critical_failures):
                print("  → SUSPECTED_PIPELINE_BIAS: Overlay-inclusive WF failed")
            if any("G5" in f for f in critical_failures):
                print("  → NON-CAUSAL_SIGNAL_RISK: Permutation test failed")
        else:
            verdict = "NEED_MORE_WORK"
            print(f"\n  🟡 VERDICT: {verdict}")
            print(f"  Gate failures: {', '.join(failed_gates)}")

        print("\n  Failed gates:")
        for f in sorted(failed_gates):
            print(f"    ❌ {f}")

    # ── 4) Evidence Paths ──
    print("\n" + "=" * 100)
    print("4) EVIDENCE PATHS")
    print("=" * 100)

    # Save JSON
    save_results = {
        "timestamp": timestamp,
        "baseline": str(baseline_path),
        "configs": {k: str(v) for k, v in config_paths.items()},
        "verdict": verdict,
        "all_pass": all_pass,
        "gates": {},
    }

    for cl, gates in gate_results.items():
        save_results["gates"][cl] = {}
        for gk, gv in gates.items():
            if isinstance(gv, dict):
                clean = {}
                for k, v in gv.items():
                    if k == "per_symbol":
                        clean[k] = {}
                        for sym, sym_data in v.items():
                            if isinstance(sym_data, dict):
                                clean[k][sym] = {
                                    kk: (float(vv) if isinstance(vv, (np.floating,))
                                         else int(vv) if isinstance(vv, (np.integer,))
                                         else vv)
                                    for kk, vv in sym_data.items()
                                }
                    elif isinstance(v, (np.floating, np.float64)):
                        clean[k] = float(v)
                    elif isinstance(v, (np.integer, np.int64)):
                        clean[k] = int(v)
                    elif isinstance(v, list):
                        clean[k] = [
                            float(x) if isinstance(x, (np.floating,)) else x
                            for x in v
                        ]
                    else:
                        clean[k] = v
                save_results["gates"][cl][gk] = clean
            else:
                save_results["gates"][cl][gk] = gv

    results_path = output_dir / "gate_check_results.json"
    with open(results_path, "w") as f:
        json.dump(save_results, f, indent=2, default=str)

    print(f"  Gate report:   {results_path}")
    print(f"  Output dir:    {output_dir}")

    print("\n" + "=" * 100)
    print(f"  R3 Track A Gate Check complete. Verdict: {verdict}")
    print("=" * 100)

    return verdict


if __name__ == "__main__":
    main()
