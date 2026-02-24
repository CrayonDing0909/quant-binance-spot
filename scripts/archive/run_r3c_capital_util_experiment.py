#!/usr/bin/env python3
"""
R3C Capital Utilization Experiment Matrix
==========================================

8 experiments evaluating higher gross exposure (target ~2x) for the R3C Universe:
  E0  Baseline (1.0x gross)
  E1  Gross 1.25x
  E2  Gross 1.50x
  E3  Gross 1.75x
  E4  Gross 2.00x
  E5  E3 + Wide SL (BTC stop_loss_atr 4.0)
  E6  E4 + Wide SL
  E7  E4 + Wide SL + DD Throttle

Measures:
  - CAGR, Sharpe, MDD, Calmar, annualised vol, trades, flips
  - +1 bar delay stress
  - Cost perturbation ±20%
  - Execution jitter 0–1 bar
  - Block bootstrap & trade shuffle
  - Margin utilisation estimates, worst week, consecutive losses

GO/NO-GO gates:
  - P5 CAGR > 0
  - Median Sharpe > 1.8
  - P95 MDD < 25%
  - +1 bar delay Sharpe decay ≤ 55%
  - Execution jitter P5 Sharpe ≥ 1.5
  - Margin utilisation (normal) < 65%, peak < 80%

Ranking:
  score = 0.35*Sharpe_n + 0.25*CAGR_n - 0.25*MDD_n - 0.15*JitterDecay_n

Usage:
    cd /path/to/quant-binance-spot
    source .venv/bin/activate
    PYTHONPATH=src python scripts/run_r3c_capital_util_experiment.py
    PYTHONPATH=src python scripts/run_r3c_capital_util_experiment.py --experiments E0 E1 E2
    PYTHONPATH=src python scripts/run_r3c_capital_util_experiment.py --mc-sims 50 --mc34-sims 5
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

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
    load_multi_tf_klines,
)

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).parent.parent
EXPERIMENT_DIR = PROJECT_ROOT / "config" / "experiments" / "r3c_capital_utilization"

# 19-symbol universe
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
    "LTCUSDT", "NEARUSDT", "APTUSDT", "ARBUSDT", "OPUSDT",
    "FILUSDT", "ATOMUSDT", "UNIUSDT", "AAVEUSDT",
]

BASE_WEIGHTS = {
    "BTCUSDT": 0.0722, "ETHUSDT": 0.0538, "SOLUSDT": 0.0509,
    "BNBUSDT": 0.0707, "XRPUSDT": 0.0491, "DOGEUSDT": 0.0512,
    "ADAUSDT": 0.0511, "AVAXUSDT": 0.0545, "LINKUSDT": 0.0538,
    "DOTUSDT": 0.0540, "LTCUSDT": 0.0605, "NEARUSDT": 0.0495,
    "APTUSDT": 0.0483, "ARBUSDT": 0.0495, "OPUSDT": 0.0426,
    "FILUSDT": 0.0489, "ATOMUSDT": 0.0523, "UNIUSDT": 0.0394,
    "AAVEUSDT": 0.0477,
}


# ══════════════════════════════════════════════════════════════
# 1. Experiment loading
# ══════════════════════════════════════════════════════════════

def load_experiment(yaml_path: Path) -> dict:
    """Load experiment overlay YAML."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    exp = raw.get("experiment", {})
    return {
        "id": exp.get("id", "?"),
        "name": exp.get("name", "?"),
        "description": exp.get("description", ""),
        "base_config": exp.get("base_config", "config/prod_candidate_R3C_universe.yaml"),
        "gross_multiplier": float(exp.get("gross_multiplier", 1.0)),
        "overrides": exp.get("overrides", {}),
        "yaml_path": str(yaml_path),
    }


def discover_experiments(exp_ids: list[str] | None = None) -> list[dict]:
    """Discover experiment YAML files in the experiment directory."""
    yamls = sorted(EXPERIMENT_DIR.glob("E*.yaml"))
    experiments = []
    for y in yamls:
        exp = load_experiment(y)
        if exp_ids is None or exp["id"] in exp_ids:
            experiments.append(exp)
    return experiments


# ══════════════════════════════════════════════════════════════
# 2. Config helpers (borrowed from MC/gate scripts)
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
# 3. Single-symbol backtest pipeline
# ══════════════════════════════════════════════════════════════

def _run_symbol_full(
    symbol: str,
    cfg,
    config_path: str | Path,
    cost_mult: float = 1.0,
    micro_accel_params: dict | None = None,
    extra_signal_delay: int = 0,
    funding_rate_mult: float = 1.0,
    btc_stop_loss_atr: float | None = None,
) -> dict | None:
    """
    Run single-symbol with both overlays, returning full details.
    Supports BTC stop_loss_atr override for E5-E7.
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

    # Apply BTC stop_loss_atr override
    if btc_stop_loss_atr is not None and symbol == "BTCUSDT":
        strategy_params = dict(strategy_params)
        strategy_params["stop_loss_atr"] = btc_stop_loss_atr

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

    # Apply vol overlay
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

    # Apply micro accel overlay
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

        pos_after_micro = apply_full_micro_accel_overlay(
            base_position=pos_after_vol,
            df_1h=df, df_5m=df_5m, df_15m=df_15m,
            oi_series=oi_series, params=micro_accel_params,
        )

    pos = pos_after_micro

    # Direction clip
    pos = clip_positions_by_direction(pos, market_type, cfg.direction)

    # Position sizing
    ps_cfg = cfg.position_sizing
    if ps_cfg.method == "fixed" and ps_cfg.position_pct < 1.0:
        pos = pos * ps_cfg.position_pct

    # Date filter
    start = cfg.market.start
    end = cfg.market.end
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
    funding_cost_total = 0.0
    fr = cfg.backtest.funding_rate
    if fr.enabled and market_type == "futures":
        funding_df = None
        if fr.use_historical:
            fr_path = get_funding_rate_path(cfg.data_dir, symbol)
            funding_df = load_funding_rates(fr_path)
        funding_rates = align_funding_to_klines(
            funding_df, df.index, default_rate_8h=fr.default_rate_8h,
        )
        funding_rates = funding_rates * funding_rate_mult
        leverage = cfg.futures.leverage if cfg.futures else 1
        fc = compute_funding_costs(
            pos=pos, equity=equity,
            funding_rates=funding_rates,
            leverage=leverage,
        )
        adjusted_equity = adjust_equity_for_funding(equity, fc)
        funding_cost_total = fc.total_cost

    eq = adjusted_equity if adjusted_equity is not None else equity

    total_return_pct = (eq.iloc[-1] / initial_cash - 1) * 100
    n_bars = len(df)
    years = n_bars / (365.25 * 24)
    cagr = ((1 + total_return_pct / 100) ** (1 / max(years, 0.01)) - 1) * 100
    flips = compute_flip_count(pos)
    total_trades = int(stats.get("Total Trades", 0))

    # Sharpe from hourly equity
    ret = eq.pct_change().fillna(0)
    sharpe = float(np.sqrt(365 * 24) * ret.mean() / ret.std()) if ret.std() > 0 else 0
    max_dd = float(((eq / eq.expanding().max()) - 1).min() * (-100))
    calmar = cagr / max_dd if max_dd > 0.01 else 0

    pos_changes = pos.diff().abs().fillna(0)
    turnover = float(pos_changes.sum())

    return {
        "symbol": symbol,
        "equity": eq,
        "pos": pos,
        "returns": ret,
        "sharpe": sharpe,
        "cagr": cagr,
        "max_drawdown_pct": max_dd,
        "calmar": calmar,
        "total_return_pct": total_return_pct,
        "total_trades": total_trades,
        "flips": flips,
        "turnover": turnover,
        "funding_cost_abs": funding_cost_total,
        "n_bars": n_bars,
        "initial_cash": initial_cash,
        "df": df,
    }


# ══════════════════════════════════════════════════════════════
# 4. Portfolio aggregation with gross scaling
# ══════════════════════════════════════════════════════════════

def _aggregate_portfolio(
    per_symbol: dict[str, dict],
    weights: dict[str, float],
    gross_multiplier: float,
    initial_cash: float,
    dd_throttle: dict | None = None,
) -> dict:
    """
    Aggregate per-symbol equity curves into portfolio.

    Key: weights are scaled by gross_multiplier (NOT normalized to 1.0)
    so total weight sum = gross_multiplier × 1.0
    """
    active = list(per_symbol.keys())
    if not active:
        return _empty_portfolio()

    # Scale weights by gross_multiplier
    raw_w = np.array([weights.get(s, 1.0 / len(SYMBOLS)) for s in active])
    raw_w = raw_w / raw_w.sum()  # normalize to 1.0 first
    scaled_w = raw_w * gross_multiplier  # then scale to target gross

    # Align equity curves
    eqs = {s: per_symbol[s]["equity"] for s in active}
    min_start = max(eq.index[0] for eq in eqs.values())
    max_end = min(eq.index[-1] for eq in eqs.values())
    for s in active:
        eqs[s] = eqs[s].loc[min_start:max_end]

    # Normalize each symbol to 1.0
    norm = {}
    for s in active:
        eq = eqs[s]
        if len(eq) > 0 and eq.iloc[0] > 0:
            norm[s] = eq / eq.iloc[0]
        else:
            norm[s] = pd.Series(1.0, index=eq.index)

    # Weighted portfolio: sum(w_i * norm_i) — sum(w) = gross_multiplier
    # Need to adjust: 1 unit + sum(w_i * (norm_i - 1)) to properly handle
    # leveraged portfolio (base = 1 + leveraged returns)
    port_ret_parts = []
    for s, w in zip(active, scaled_w):
        sym_ret = norm[s].pct_change().fillna(0)
        port_ret_parts.append(sym_ret * w)

    portfolio_returns = sum(port_ret_parts)

    # Apply DD throttle if configured
    if dd_throttle and dd_throttle.get("enabled", False):
        portfolio_returns = _apply_dd_throttle(
            portfolio_returns,
            dd_on=dd_throttle.get("dd_on", 0.08),
            dd_off=dd_throttle.get("dd_off", 0.05),
            scale=dd_throttle.get("scale", 0.50),
        )

    portfolio_equity = (1 + portfolio_returns).cumprod() * initial_cash

    # Compute metrics
    n_bars = len(portfolio_returns)
    years = n_bars / (365.25 * 24)
    total_return = (portfolio_equity.iloc[-1] / initial_cash - 1) * 100
    cagr = ((1 + total_return / 100) ** (1 / max(years, 0.01)) - 1) * 100

    rolling_max = portfolio_equity.expanding().max()
    dd = (portfolio_equity - rolling_max) / rolling_max
    max_dd = abs(dd.min()) * 100

    sharpe = (
        float(np.sqrt(365 * 24) * portfolio_returns.mean() / portfolio_returns.std())
        if portfolio_returns.std() > 0 else 0
    )
    ann_vol = float(portfolio_returns.std() * np.sqrt(365 * 24) * 100)
    calmar = cagr / max_dd if max_dd > 0.01 else 0

    total_trades = sum(per_symbol[s].get("total_trades", 0) for s in active)
    total_flips = sum(per_symbol[s].get("flips", 0) for s in active)

    # Trading feasibility metrics
    # Per-symbol gross exposure at each time step
    gross_series = pd.Series(0.0, index=portfolio_equity.index)
    for s, w in zip(active, scaled_w):
        if s in per_symbol and "pos" in per_symbol[s]:
            pos = per_symbol[s]["pos"]
            # Align
            common = pos.index.intersection(gross_series.index)
            gross_series.loc[common] += abs(pos.loc[common]) * w

    avg_gross = float(gross_series.mean())
    peak_gross = float(gross_series.max())

    # Worst week return
    weekly_ret = portfolio_returns.resample("7D").sum() if len(portfolio_returns) > 168 else portfolio_returns
    worst_week = float(weekly_ret.min() * 100)

    # Consecutive losing days
    daily_ret = portfolio_returns.resample("1D").sum()
    losing_days = (daily_ret < 0).astype(int)
    consecutive_loss = 0
    max_consecutive_loss = 0
    for v in losing_days:
        if v == 1:
            consecutive_loss += 1
            max_consecutive_loss = max(max_consecutive_loss, consecutive_loss)
        else:
            consecutive_loss = 0

    return {
        "cagr": round(cagr, 4),
        "sharpe": round(sharpe, 4),
        "max_drawdown_pct": round(max_dd, 4),
        "calmar": round(calmar, 4),
        "total_return_pct": round(total_return, 4),
        "ann_vol": round(ann_vol, 4),
        "total_trades": total_trades,
        "flips": total_flips,
        "avg_gross": round(avg_gross, 4),
        "peak_gross": round(peak_gross, 4),
        "worst_week_pct": round(worst_week, 4),
        "max_consecutive_loss_days": max_consecutive_loss,
        "portfolio_equity": portfolio_equity,
        "portfolio_returns": portfolio_returns,
        "gross_series": gross_series,
    }


def _empty_portfolio():
    return {
        "cagr": 0, "sharpe": 0, "max_drawdown_pct": 0, "calmar": 0,
        "total_return_pct": 0, "ann_vol": 0, "total_trades": 0, "flips": 0,
        "avg_gross": 0, "peak_gross": 0, "worst_week_pct": 0,
        "max_consecutive_loss_days": 0,
        "portfolio_equity": pd.Series(dtype=float),
        "portfolio_returns": pd.Series(dtype=float),
        "gross_series": pd.Series(dtype=float),
    }


def _apply_dd_throttle(
    portfolio_returns: pd.Series,
    dd_on: float = 0.08,
    dd_off: float = 0.05,
    scale: float = 0.50,
) -> pd.Series:
    """Portfolio-level drawdown throttle."""
    n = len(portfolio_returns)
    ret_arr = portfolio_returns.values.copy()
    throttled = np.zeros(n, dtype=float)

    equity = 1.0
    peak = 1.0
    throttle_active = False

    for i in range(n):
        current_scale = scale if throttle_active else 1.0
        throttled[i] = ret_arr[i] * current_scale
        equity *= (1.0 + throttled[i])
        if equity > peak:
            peak = equity
        running_dd = (peak - equity) / peak if peak > 0 else 0.0
        if not throttle_active and running_dd > dd_on:
            throttle_active = True
        elif throttle_active and running_dd < dd_off:
            throttle_active = False

    return pd.Series(throttled, index=portfolio_returns.index)


# ══════════════════════════════════════════════════════════════
# 5. Stress tests
# ══════════════════════════════════════════════════════════════

def _compute_metrics_from_returns(returns: pd.Series, initial_cash: float) -> dict:
    equity = (1 + returns).cumprod() * initial_cash
    ret = returns
    n_bars = len(ret)
    years = n_bars / (365.25 * 24)
    total_return = (equity.iloc[-1] / initial_cash - 1) * 100
    cagr = ((1 + total_return / 100) ** (1 / max(years, 0.01)) - 1) * 100
    rolling_max = equity.expanding().max()
    dd = (equity - rolling_max) / rolling_max
    max_dd = abs(dd.min()) * 100
    sharpe = (
        float(np.sqrt(365 * 24) * ret.mean() / ret.std())
        if ret.std() > 0 else 0
    )
    calmar = cagr / max_dd if max_dd > 0.01 else 0
    return {
        "cagr": round(cagr, 4),
        "sharpe": round(sharpe, 4),
        "max_drawdown_pct": round(max_dd, 4),
        "calmar": round(calmar, 4),
        "total_return_pct": round(total_return, 4),
    }


def _percentile_summary(results: list[dict], metric: str) -> dict:
    vals = [r[metric] for r in results if metric in r]
    if not vals:
        return {"p5": 0, "p25": 0, "p50": 0, "p75": 0, "p95": 0, "mean": 0, "std": 0}
    arr = np.array(vals)
    return {
        "p5": round(float(np.percentile(arr, 5)), 4),
        "p25": round(float(np.percentile(arr, 25)), 4),
        "p50": round(float(np.percentile(arr, 50)), 4),
        "p75": round(float(np.percentile(arr, 75)), 4),
        "p95": round(float(np.percentile(arr, 95)), 4),
        "mean": round(float(np.mean(arr)), 4),
        "std": round(float(np.std(arr)), 4),
    }


def stress_delay(
    per_symbol_delay: dict[str, dict],
    weights: dict[str, float],
    gross_mult: float,
    initial_cash: float,
    dd_throttle: dict | None = None,
) -> dict:
    """Stress test: +1 bar signal delay."""
    port = _aggregate_portfolio(
        per_symbol_delay, weights, gross_mult, initial_cash, dd_throttle,
    )
    return {
        "cagr": port["cagr"],
        "sharpe": port["sharpe"],
        "max_drawdown_pct": port["max_drawdown_pct"],
        "calmar": port["calmar"],
    }


def stress_bootstrap(
    portfolio_returns: pd.Series,
    initial_cash: float,
    n_sims: int,
    block_size: int,
    rng: np.random.Generator,
) -> list[dict]:
    """MC1: Block bootstrap of portfolio returns."""
    ret = portfolio_returns.values
    n = len(ret)
    n_blocks = int(np.ceil(n / block_size))
    max_start = n - block_size
    if max_start < 1:
        return []
    results = []
    for i in range(n_sims):
        starts = rng.integers(0, max_start, size=n_blocks)
        synthetic = np.concatenate([ret[s:s + block_size] for s in starts])[:n]
        metrics = _compute_metrics_from_returns(pd.Series(synthetic), initial_cash)
        metrics["sim_id"] = i
        results.append(metrics)
    return results


def stress_shuffle(
    portfolio_returns: pd.Series,
    initial_cash: float,
    n_sims: int,
    rng: np.random.Generator,
) -> list[dict]:
    """MC2: Trade-order shuffle."""
    ret = portfolio_returns.values.copy()
    results = []
    for i in range(n_sims):
        shuffled = ret.copy()
        rng.shuffle(shuffled)
        metrics = _compute_metrics_from_returns(pd.Series(shuffled), initial_cash)
        metrics["sim_id"] = i
        results.append(metrics)
    return results


def stress_cost_perturbation(
    cfg, config_path, micro_accel_params,
    weights: dict[str, float],
    gross_mult: float,
    n_sims: int,
    rng: np.random.Generator,
    btc_stop_loss_atr: float | None = None,
    dd_throttle: dict | None = None,
) -> list[dict]:
    """MC3: Cost perturbation ±20%."""
    initial_cash = cfg.backtest.initial_cash
    results = []
    for i in range(n_sims):
        fee_mult = float(rng.uniform(0.80, 1.20))
        slip_mult = float(rng.uniform(0.80, 1.20))
        funding_mult = float(rng.uniform(0.80, 1.20))
        combined_cost_mult = float(np.sqrt(fee_mult * slip_mult))

        per_sym = {}
        for symbol in SYMBOLS:
            try:
                res = _run_symbol_full(
                    symbol, cfg, config_path,
                    cost_mult=combined_cost_mult,
                    micro_accel_params=micro_accel_params,
                    funding_rate_mult=funding_mult,
                    btc_stop_loss_atr=btc_stop_loss_atr,
                )
                if res is not None:
                    per_sym[symbol] = res
            except Exception:
                continue

        port = _aggregate_portfolio(
            per_sym, weights, gross_mult, initial_cash, dd_throttle,
        )
        metrics = {
            "sim_id": i,
            "cagr": port["cagr"],
            "sharpe": port["sharpe"],
            "max_drawdown_pct": port["max_drawdown_pct"],
            "calmar": port["calmar"],
            "total_return_pct": port["total_return_pct"],
        }
        results.append(metrics)

        if (i + 1) % 3 == 0 or i == 0:
            print(
                f"      MC3 sim {i+1}/{n_sims}: "
                f"CAGR={metrics['cagr']:.1f}%, MDD={metrics['max_drawdown_pct']:.2f}%, "
                f"SR={metrics['sharpe']:.2f}"
            )
    return results


def stress_jitter(
    cfg, config_path, micro_accel_params,
    weights: dict[str, float],
    gross_mult: float,
    n_sims: int,
    rng: np.random.Generator,
    btc_stop_loss_atr: float | None = None,
    dd_throttle: dict | None = None,
) -> list[dict]:
    """
    MC4: Execution jitter (0–1 bar random delay per signal change).

    Pre-computes base positions once, then jitters them per sim.
    """
    market_type = cfg.market_type_str
    initial_cash = cfg.backtest.initial_cash
    fee = _bps_to_pct(cfg.backtest.fee_bps)
    slippage = _bps_to_pct(cfg.backtest.slippage_bps)
    vbt_direction = to_vbt_direction(cfg.direction)

    # Pre-compute base positions
    symbol_data = {}
    for symbol in SYMBOLS:
        data_path = (
            cfg.data_dir / "binance" / market_type
            / cfg.market.interval / f"{symbol}.parquet"
        )
        if not data_path.exists():
            continue

        df = load_klines(data_path)
        df = clean_data(df, fill_method="forward", remove_outliers=False,
                        remove_duplicates=True)

        multi_tf = load_multi_tf_klines(cfg.data_dir, symbol, market_type)
        df_5m = multi_tf.get("5m")
        df_15m = multi_tf.get("15m")

        ensemble_override = _load_ensemble_strategy(config_path, symbol)
        if ensemble_override:
            strategy_name, strategy_params = ensemble_override
        else:
            strategy_name = cfg.strategy.name
            strategy_params = cfg.strategy.get_params(symbol)

        if btc_stop_loss_atr is not None and symbol == "BTCUSDT":
            strategy_params = dict(strategy_params)
            strategy_params["stop_loss_atr"] = btc_stop_loss_atr

        ctx = StrategyContext(
            symbol=symbol, interval=cfg.market.interval,
            market_type=market_type, direction=cfg.direction,
            signal_delay=1,
        )

        strategy_func = get_strategy(strategy_name)
        pos_base = strategy_func(df, ctx, strategy_params)

        vol_overlay = _load_vol_overlay_params(config_path)
        if vol_overlay and vol_overlay.get("enabled", False):
            pos_base = apply_overlay_by_mode(
                position=pos_base, price_df=df, oi_series=None,
                params=vol_overlay.get("params", {}),
                mode=vol_overlay.get("mode", "vol_pause"),
            )

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
            pos_base = apply_full_micro_accel_overlay(
                base_position=pos_base, df_1h=df,
                df_5m=df_5m, df_15m=df_15m,
                oi_series=oi_series, params=micro_accel_params,
            )

        pos_base = clip_positions_by_direction(pos_base, market_type, cfg.direction)
        df, pos_base = _apply_date_filter(df, pos_base, cfg.market.start, cfg.market.end)
        if len(df) < 100:
            continue

        change_mask = pos_base.diff().abs() > 0.001
        change_indices = np.where(change_mask.values)[0]

        fr = cfg.backtest.funding_rate
        funding_rates = None
        if fr.enabled and market_type == "futures":
            funding_df = None
            if fr.use_historical:
                fr_path = get_funding_rate_path(cfg.data_dir, symbol)
                funding_df = load_funding_rates(fr_path)
            funding_rates = align_funding_to_klines(
                funding_df, df.index, default_rate_8h=fr.default_rate_8h,
            )

        symbol_data[symbol] = {
            "df": df, "pos": pos_base,
            "change_indices": change_indices,
            "funding_rates": funding_rates,
        }

    results = []
    for sim_i in range(n_sims):
        per_symbol_eq = {}

        for symbol, sdata in symbol_data.items():
            pos = sdata["pos"].copy()
            change_idx = sdata["change_indices"]
            df_sym = sdata["df"]

            jitter_pos = pos.values.copy()
            delays = rng.integers(0, 2, size=len(change_idx))
            for ci, delay in zip(change_idx, delays):
                if delay == 1 and ci + 1 < len(jitter_pos):
                    jitter_pos[ci] = jitter_pos[ci - 1] if ci > 0 else 0.0

            jitter_series = pd.Series(jitter_pos, index=pos.index)

            try:
                pf = vbt.Portfolio.from_orders(
                    close=df_sym["close"], size=jitter_series,
                    size_type="targetpercent", price=df_sym["open"],
                    fees=fee, slippage=slippage,
                    init_cash=initial_cash,
                    freq=cfg.market.interval, direction=vbt_direction,
                )
                equity = pf.value()

                if sdata["funding_rates"] is not None:
                    leverage = cfg.futures.leverage if cfg.futures else 1
                    fc = compute_funding_costs(
                        pos=jitter_series, equity=equity,
                        funding_rates=sdata["funding_rates"],
                        leverage=leverage,
                    )
                    equity = adjust_equity_for_funding(equity, fc)

                per_symbol_eq[symbol] = {"equity": equity, "pos": jitter_series}
            except Exception:
                continue

        if not per_symbol_eq:
            continue

        port = _aggregate_portfolio(
            per_symbol_eq, weights, gross_mult, initial_cash, dd_throttle,
        )
        metrics = {
            "sim_id": sim_i,
            "cagr": port["cagr"],
            "sharpe": port["sharpe"],
            "max_drawdown_pct": port["max_drawdown_pct"],
            "calmar": port["calmar"],
        }
        results.append(metrics)

        if (sim_i + 1) % 3 == 0 or sim_i == 0:
            print(
                f"      MC4 sim {sim_i+1}/{n_sims}: "
                f"CAGR={metrics['cagr']:.1f}%, MDD={metrics['max_drawdown_pct']:.2f}%, "
                f"SR={metrics['sharpe']:.2f}"
            )

    return results


# ══════════════════════════════════════════════════════════════
# 6. Margin utilisation estimate
# ══════════════════════════════════════════════════════════════

def estimate_margin_utilisation(
    gross_mult: float,
    leverage: int = 3,
    peak_gross: float = 0.0,
    avg_gross: float = 0.0,
) -> dict:
    """
    Estimate margin utilisation based on gross exposure and exchange leverage.

    margin_usage = gross_exposure / leverage
    e.g. 2.0x gross / 3x leverage = 66.7% margin usage
    """
    normal_margin = avg_gross / leverage * 100
    peak_margin = peak_gross / leverage * 100
    return {
        "normal_margin_pct": round(normal_margin, 2),
        "peak_margin_pct": round(peak_margin, 2),
    }


# ══════════════════════════════════════════════════════════════
# 7. GO/NO-GO evaluation
# ══════════════════════════════════════════════════════════════

def evaluate_go_nogo(
    exp_id: str,
    baseline: dict,
    delay_metrics: dict,
    mc_results: dict,
    margin: dict,
) -> dict:
    """
    Evaluate GO/NO-GO for an experiment.

    Gates:
      1. P5 CAGR > 0 (bootstrap)
      2. Median Sharpe > 1.8 (bootstrap)
      3. P95 MDD < 25% (bootstrap)
      4. +1 bar delay Sharpe decay ≤ 55%
      5. Execution jitter P5 Sharpe ≥ 1.5
      6. Margin (normal) < 65%, peak < 80%
    """
    gates = []

    # Gate 1: P5 CAGR > 0 (from bootstrap or baseline)
    bootstrap_cagr_p5 = 0
    if "bootstrap" in mc_results and mc_results["bootstrap"]:
        bootstrap_cagr_p5 = _percentile_summary(mc_results["bootstrap"], "cagr")["p5"]
    else:
        bootstrap_cagr_p5 = baseline["cagr"]
    g1_pass = bootstrap_cagr_p5 > 0
    gates.append(("P5_CAGR_gt_0", g1_pass, f"P5 CAGR={bootstrap_cagr_p5:.1f}%"))

    # Gate 2: Median Sharpe > 1.8
    bootstrap_sharpe_p50 = 0
    if "bootstrap" in mc_results and mc_results["bootstrap"]:
        bootstrap_sharpe_p50 = _percentile_summary(mc_results["bootstrap"], "sharpe")["p50"]
    else:
        bootstrap_sharpe_p50 = baseline["sharpe"]
    g2_pass = bootstrap_sharpe_p50 > 1.8
    gates.append(("Median_Sharpe_gt_1.8", g2_pass, f"Median Sharpe={bootstrap_sharpe_p50:.2f}"))

    # Gate 3: P95 MDD < 25%
    bootstrap_mdd_p95 = 0
    if "bootstrap" in mc_results and mc_results["bootstrap"]:
        bootstrap_mdd_p95 = _percentile_summary(mc_results["bootstrap"], "max_drawdown_pct")["p95"]
    else:
        bootstrap_mdd_p95 = baseline["max_drawdown_pct"]
    g3_pass = bootstrap_mdd_p95 < 25.0
    gates.append(("P95_MDD_lt_25", g3_pass, f"P95 MDD={bootstrap_mdd_p95:.2f}%"))

    # Gate 4: +1 bar delay Sharpe decay ≤ 55%
    delay_sharpe = delay_metrics.get("sharpe", 0)
    base_sharpe = baseline["sharpe"]
    sharpe_decay = (base_sharpe - delay_sharpe) / abs(base_sharpe) * 100 if abs(base_sharpe) > 0.01 else 0
    g4_pass = sharpe_decay <= 55.0
    gates.append(("Delay_Sharpe_decay_le_55pct", g4_pass,
                   f"Decay={sharpe_decay:.1f}% ({base_sharpe:.2f}→{delay_sharpe:.2f})"))

    # Gate 5: Jitter P5 Sharpe ≥ 1.5
    jitter_sharpe_p5 = 0
    if "jitter" in mc_results and mc_results["jitter"]:
        jitter_sharpe_p5 = _percentile_summary(mc_results["jitter"], "sharpe")["p5"]
    else:
        jitter_sharpe_p5 = delay_sharpe  # fallback
    g5_pass = jitter_sharpe_p5 >= 1.5
    gates.append(("Jitter_P5_Sharpe_ge_1.5", g5_pass,
                   f"Jitter P5 Sharpe={jitter_sharpe_p5:.2f}"))

    # Gate 6: Margin utilisation
    g6a_pass = margin["normal_margin_pct"] < 65.0
    g6b_pass = margin["peak_margin_pct"] < 80.0
    g6_pass = g6a_pass and g6b_pass
    gates.append(("Margin_normal_lt_65_peak_lt_80", g6_pass,
                   f"Normal={margin['normal_margin_pct']:.1f}%, Peak={margin['peak_margin_pct']:.1f}%"))

    all_pass = all(p for _, p, _ in gates)
    verdict = "GO" if all_pass else "NO-GO"
    failed = [name for name, p, _ in gates if not p]

    return {
        "exp_id": exp_id,
        "verdict": verdict,
        "all_pass": all_pass,
        "gates": gates,
        "failed_gates": failed,
        "sharpe_decay_pct": round(sharpe_decay, 2),
        "jitter_sharpe_p5": round(jitter_sharpe_p5, 4),
    }


# ══════════════════════════════════════════════════════════════
# 8. Ranking
# ══════════════════════════════════════════════════════════════

def compute_ranking(all_results: dict) -> list[dict]:
    """
    Rank experiments by composite score:
      score = 0.35*Sharpe_norm + 0.25*CAGR_norm - 0.25*MDD_norm - 0.15*JitterDecay_norm
    Eliminated experiments (NO-GO) are excluded from ranking.
    """
    eligible = {k: v for k, v in all_results.items()
                if v["evaluation"]["verdict"] == "GO"}
    eliminated = {k: v for k, v in all_results.items()
                  if v["evaluation"]["verdict"] != "GO"}

    if not eligible:
        ranking = []
        for k, v in all_results.items():
            ranking.append({
                "exp_id": k,
                "rank": None,
                "score": None,
                "verdict": v["evaluation"]["verdict"],
                "cagr": v["baseline"]["cagr"],
                "sharpe": v["baseline"]["sharpe"],
                "max_drawdown_pct": v["baseline"]["max_drawdown_pct"],
                "eliminated": True,
                "failed_gates": v["evaluation"]["failed_gates"],
            })
        return ranking

    # Collect raw values
    sharpe_vals = {k: v["baseline"]["sharpe"] for k, v in eligible.items()}
    cagr_vals = {k: v["baseline"]["cagr"] for k, v in eligible.items()}
    mdd_vals = {k: v["baseline"]["max_drawdown_pct"] for k, v in eligible.items()}
    jitter_decay_vals = {k: v["evaluation"]["sharpe_decay_pct"] for k, v in eligible.items()}

    # Normalize to [0, 1]
    def _normalize(vals: dict) -> dict:
        arr = np.array(list(vals.values()))
        vmin, vmax = arr.min(), arr.max()
        if vmax - vmin < 1e-9:
            return {k: 0.5 for k in vals}
        return {k: (v - vmin) / (vmax - vmin) for k, v in vals.items()}

    sharpe_n = _normalize(sharpe_vals)
    cagr_n = _normalize(cagr_vals)
    mdd_n = _normalize(mdd_vals)
    jitter_n = _normalize(jitter_decay_vals)

    # Compute scores
    scores = {}
    for k in eligible:
        scores[k] = (
            0.35 * sharpe_n[k]
            + 0.25 * cagr_n[k]
            - 0.25 * mdd_n[k]
            - 0.15 * jitter_n[k]
        )

    # Sort by score descending
    sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
    ranking = []
    for rank, k in enumerate(sorted_keys, 1):
        v = eligible[k]
        ranking.append({
            "exp_id": k,
            "rank": rank,
            "score": round(scores[k], 4),
            "verdict": "GO",
            "cagr": v["baseline"]["cagr"],
            "sharpe": v["baseline"]["sharpe"],
            "max_drawdown_pct": v["baseline"]["max_drawdown_pct"],
            "calmar": v["baseline"]["calmar"],
            "sharpe_decay_pct": v["evaluation"]["sharpe_decay_pct"],
            "eliminated": False,
            "failed_gates": [],
        })

    # Append eliminated
    for k, v in eliminated.items():
        ranking.append({
            "exp_id": k,
            "rank": None,
            "score": None,
            "verdict": v["evaluation"]["verdict"],
            "cagr": v["baseline"]["cagr"],
            "sharpe": v["baseline"]["sharpe"],
            "max_drawdown_pct": v["baseline"]["max_drawdown_pct"],
            "calmar": v["baseline"]["calmar"],
            "sharpe_decay_pct": v["evaluation"]["sharpe_decay_pct"],
            "eliminated": True,
            "failed_gates": v["evaluation"]["failed_gates"],
        })

    return ranking


# ══════════════════════════════════════════════════════════════
# 9. Report generation
# ══════════════════════════════════════════════════════════════

def generate_summary_csv(all_results: dict, ranking: list[dict], output_path: Path):
    """Generate summary_table.csv."""
    rows = []
    for r in ranking:
        k = r["exp_id"]
        v = all_results[k]
        bl = v["baseline"]
        ev = v["evaluation"]
        stress = v.get("stress", {})

        row = {
            "exp_id": k,
            "name": v["experiment"]["name"],
            "gross_mult": v["experiment"]["gross_multiplier"],
            "rank": r.get("rank", "—"),
            "score": r.get("score", "—"),
            "verdict": r["verdict"],
            "CAGR": bl["cagr"],
            "Sharpe": bl["sharpe"],
            "MDD": bl["max_drawdown_pct"],
            "Calmar": bl["calmar"],
            "Ann_Vol": bl["ann_vol"],
            "Trades": bl["total_trades"],
            "Flips": bl["flips"],
            "Avg_Gross": bl["avg_gross"],
            "Peak_Gross": bl["peak_gross"],
            "Worst_Week": bl["worst_week_pct"],
            "Max_Consec_Loss_Days": bl["max_consecutive_loss_days"],
            "Delay_Sharpe": stress.get("delay", {}).get("sharpe", "—"),
            "Delay_Decay_Pct": ev["sharpe_decay_pct"],
            "Normal_Margin_Pct": v.get("margin", {}).get("normal_margin_pct", "—"),
            "Peak_Margin_Pct": v.get("margin", {}).get("peak_margin_pct", "—"),
            "Failed_Gates": "; ".join(ev["failed_gates"]) if ev["failed_gates"] else "",
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df


def generate_ranking_md(
    all_results: dict,
    ranking: list[dict],
    output_path: Path,
    timestamp: str,
):
    """Generate ranking.md."""
    lines = [
        "# R3C Capital Utilization — Experiment Ranking",
        "",
        f"**Generated**: {timestamp}",
        f"**Experiments**: {len(ranking)}",
        "",
        "## Ranking Formula",
        "",
        "```",
        "score = 0.35 × Sharpe_norm + 0.25 × CAGR_norm - 0.25 × MDD_norm - 0.15 × JitterDecay_norm",
        "```",
        "",
        "NO-GO experiments are excluded from ranking.",
        "",
        "---",
        "",
        "## Results Table",
        "",
        "| Rank | Exp | Name | Gross | Score | Verdict | CAGR | Sharpe | MDD | Calmar | Delay Decay |",
        "|------|-----|------|-------|-------|---------|------|--------|-----|--------|-------------|",
    ]

    for r in ranking:
        k = r["exp_id"]
        v = all_results[k]
        rank_str = str(r["rank"]) if r["rank"] else "—"
        score_str = f"{r['score']:.3f}" if r["score"] is not None else "—"
        icon = "✅" if r["verdict"] == "GO" else "❌"
        lines.append(
            f"| {rank_str} | {k} | {v['experiment']['name']} "
            f"| {v['experiment']['gross_multiplier']:.2f}x "
            f"| {score_str} | {icon} {r['verdict']} "
            f"| {r['cagr']:.1f}% | {r['sharpe']:.2f} "
            f"| {r['max_drawdown_pct']:.2f}% | {r.get('calmar', 0):.1f} "
            f"| {r.get('sharpe_decay_pct', 0):.1f}% |"
        )

    lines += [
        "",
        "---",
        "",
        "## Per-Experiment Details",
        "",
    ]

    for r in ranking:
        k = r["exp_id"]
        v = all_results[k]
        exp = v["experiment"]
        bl = v["baseline"]
        ev = v["evaluation"]
        margin = v.get("margin", {})

        icon = "✅" if r["verdict"] == "GO" else "❌"
        lines.append(f"### {k}: {exp['name']} ({icon} {r['verdict']})")
        lines.append("")
        lines.append(f"**Description**: {exp['description']}")
        lines.append(f"**Gross multiplier**: {exp['gross_multiplier']:.2f}x")
        lines.append("")

        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| CAGR | {bl['cagr']:.2f}% |")
        lines.append(f"| Sharpe | {bl['sharpe']:.4f} |")
        lines.append(f"| MaxDD | {bl['max_drawdown_pct']:.2f}% |")
        lines.append(f"| Calmar | {bl['calmar']:.2f} |")
        lines.append(f"| Ann Vol | {bl['ann_vol']:.2f}% |")
        lines.append(f"| Trades | {bl['total_trades']} |")
        lines.append(f"| Flips | {bl['flips']} |")
        lines.append(f"| Avg Gross | {bl['avg_gross']:.3f} |")
        lines.append(f"| Peak Gross | {bl['peak_gross']:.3f} |")
        lines.append(f"| Worst Week | {bl['worst_week_pct']:.2f}% |")
        lines.append(f"| Max Consec Loss Days | {bl['max_consecutive_loss_days']} |")
        lines.append(f"| Normal Margin | {margin.get('normal_margin_pct', 0):.1f}% |")
        lines.append(f"| Peak Margin | {margin.get('peak_margin_pct', 0):.1f}% |")
        lines.append("")

        if ev["failed_gates"]:
            lines.append("**Failed gates**: " + ", ".join(ev["failed_gates"]))
        else:
            lines.append("**All gates PASS**")

        lines.append("")
        lines.append("**Gate details**:")
        for gname, gpass, gdesc in ev["gates"]:
            gicon = "✅" if gpass else "❌"
            lines.append(f"- {gicon} {gname}: {gdesc}")
        lines.append("")

        # Pros/Cons
        pros, cons = [], []
        if bl["cagr"] > 50:
            pros.append(f"High CAGR ({bl['cagr']:.1f}%)")
        if bl["sharpe"] > 3:
            pros.append(f"Excellent Sharpe ({bl['sharpe']:.2f})")
        if bl["max_drawdown_pct"] < 5:
            pros.append(f"Low MDD ({bl['max_drawdown_pct']:.2f}%)")
        if ev["sharpe_decay_pct"] < 20:
            pros.append(f"Robust to delay (decay {ev['sharpe_decay_pct']:.1f}%)")
        if margin.get("normal_margin_pct", 100) < 50:
            pros.append(f"Comfortable margin ({margin.get('normal_margin_pct', 0):.0f}%)")

        if bl["max_drawdown_pct"] > 10:
            cons.append(f"High MDD ({bl['max_drawdown_pct']:.2f}%)")
        if ev["sharpe_decay_pct"] > 40:
            cons.append(f"Fragile to delay (decay {ev['sharpe_decay_pct']:.1f}%)")
        if margin.get("peak_margin_pct", 0) > 65:
            cons.append(f"High peak margin ({margin.get('peak_margin_pct', 0):.1f}%)")
        if bl["worst_week_pct"] < -5:
            cons.append(f"Bad worst week ({bl['worst_week_pct']:.2f}%)")

        if not pros:
            pros.append("Standard performance")
        if not cons:
            cons.append("No significant concerns")

        lines.append(f"**Pros**: {'; '.join(pros)}")
        lines.append(f"**Cons**: {'; '.join(cons)}")

        if ev["failed_gates"]:
            lines.append(f"**Failure reason**: {'; '.join(ev['failed_gates'])}")

        lines.append("")
        lines.append("---")
        lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def generate_go_nogo_md(
    all_results: dict,
    ranking: list[dict],
    output_path: Path,
    timestamp: str,
):
    """Generate go_no_go.md with final recommendations."""
    go_exps = [r for r in ranking if r["verdict"] == "GO"]
    nogo_exps = [r for r in ranking if r["verdict"] != "GO"]

    lines = [
        "# R3C Capital Utilization — GO / NO-GO Decision",
        "",
        f"**Generated**: {timestamp}",
        f"**Total experiments**: {len(ranking)}",
        f"**GO**: {len(go_exps)} | **NO-GO**: {len(nogo_exps)}",
        "",
        "---",
        "",
        "## Summary",
        "",
        "| Exp | Name | Verdict | Reason |",
        "|-----|------|---------|--------|",
    ]

    for r in ranking:
        k = r["exp_id"]
        v = all_results[k]
        icon = "✅" if r["verdict"] == "GO" else "❌"
        reason = "; ".join(r["failed_gates"]) if r["failed_gates"] else "All gates PASS"
        lines.append(f"| {k} | {v['experiment']['name']} | {icon} {r['verdict']} | {reason} |")

    lines.append("")
    lines.append("---")
    lines.append("")

    # Top-2 candidates
    lines.append("## Top-2 Candidates for Live Deployment")
    lines.append("")

    if len(go_exps) >= 2:
        top2 = go_exps[:2]
        for i, r in enumerate(top2, 1):
            k = r["exp_id"]
            v = all_results[k]
            gross = v["experiment"]["gross_multiplier"]
            pilot_pct = 0.05 if gross <= 1.5 else 0.03

            lines.append(f"### Candidate #{i}: {k} — {v['experiment']['name']}")
            lines.append("")
            lines.append(f"- **Gross**: {gross:.2f}x")
            lines.append(f"- **CAGR**: {r['cagr']:.1f}%")
            lines.append(f"- **Sharpe**: {r['sharpe']:.2f}")
            lines.append(f"- **MDD**: {r['max_drawdown_pct']:.2f}%")
            lines.append(f"- **Score**: {r['score']:.3f}")
            lines.append(f"- **Suggested pilot capital**: {pilot_pct*100:.0f}%")
            lines.append(f"- **Suggested pilot duration**: {'4 weeks' if gross <= 1.5 else '6 weeks'}")
            lines.append("")
    elif len(go_exps) == 1:
        r = go_exps[0]
        k = r["exp_id"]
        v = all_results[k]
        lines.append(f"### Only candidate: {k} — {v['experiment']['name']}")
        lines.append(f"- CAGR: {r['cagr']:.1f}%, Sharpe: {r['sharpe']:.2f}, MDD: {r['max_drawdown_pct']:.2f}%")
        lines.append("")
    else:
        lines.append("**⚠️ No experiments passed all GO/NO-GO gates.**")
        lines.append("")
        lines.append("Recommendation: Stay at current 1.0x gross (E0 baseline) until further research.")
        lines.append("")

    # Recommended upgrade path
    lines.append("---")
    lines.append("")
    lines.append("## Recommended Upgrade Path")
    lines.append("")

    if go_exps:
        max_go_gross = max(all_results[r["exp_id"]]["experiment"]["gross_multiplier"] for r in go_exps)
        path_steps = []
        for target in [1.0, 1.25, 1.50, 1.75, 2.00]:
            if target <= max_go_gross:
                matching = [r for r in go_exps
                            if all_results[r["exp_id"]]["experiment"]["gross_multiplier"] == target]
                if matching:
                    path_steps.append(f"**{target:.2f}x** (E{all_results[matching[0]['exp_id']]['experiment']['id']})")

        if path_steps:
            lines.append("Gradual scale-up path (paper-trade each level for 2-4 weeks):")
            lines.append("")
            lines.append(" → ".join(path_steps))
            lines.append("")
            lines.append("**Key rule**: If any level shows live Sharpe < 70% of backtest Sharpe, STOP and evaluate.")
        else:
            lines.append("No safe upgrade path identified.")
    else:
        lines.append("No safe upgrade path — all experiments failed at least one gate.")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"*Report generated at {timestamp}*")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


# ══════════════════════════════════════════════════════════════
# 10. Main orchestrator
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="R3C Capital Utilization Experiment Matrix",
    )
    parser.add_argument("--experiments", nargs="+", default=None,
                        help="Specific experiments to run (e.g. E0 E1 E2). Default: all")
    parser.add_argument("--mc-sims", type=int, default=200,
                        help="MC1/MC2 simulations per experiment (default: 200)")
    parser.add_argument("--mc34-sims", type=int, default=5,
                        help="MC3/MC4 simulations per experiment (default: 5)")
    parser.add_argument("--block-size", type=int, default=168,
                        help="Block size for bootstrap (default: 168)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--skip-mc", action="store_true",
                        help="Skip Monte Carlo (faster screening)")
    parser.add_argument("--skip-jitter", action="store_true",
                        help="Skip MC4 jitter (most expensive)")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "reports" / "r3c_capital_utilization" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print("█" * 100)
    print("  R3C CAPITAL UTILIZATION EXPERIMENT MATRIX")
    print("█" * 100)
    print(f"  Timestamp:  {timestamp}")
    print(f"  Output:     {output_dir}")
    print(f"  MC sims:    MC1/MC2={args.mc_sims}, MC3/MC4={args.mc34_sims}")
    print(f"  Block size: {args.block_size}h")
    print(f"  Seed:       {args.seed}")
    print()

    # ── Load experiments ──
    experiments = discover_experiments(args.experiments)
    if not experiments:
        print("❌ No experiment configs found")
        return
    print(f"📋 Loaded {len(experiments)} experiments:")
    for exp in experiments:
        print(f"   {exp['id']}: {exp['name']} (gross={exp['gross_multiplier']:.2f}x)")
    print()

    # ── Load base config ──
    base_config_path = PROJECT_ROOT / experiments[0]["base_config"]
    cfg = load_config(str(base_config_path))
    micro_params = _load_micro_accel_params(base_config_path)
    initial_cash = cfg.backtest.initial_cash
    leverage = cfg.futures.leverage if cfg.futures else 3

    if micro_params is None:
        print("  ❌ FATAL: No micro_accel_overlay params in config!")
        return

    rng = np.random.default_rng(args.seed)

    # ══════════════════════════════════════════════════════════
    # PHASE 1: Run per-symbol backtests (cached)
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  PHASE 1: Per-symbol backtests")
    print("=" * 100)

    # 1a. Baseline (default SL)
    print("\n  ── Baseline per-symbol (default config) ──")
    baseline_per_sym = {}
    for symbol in SYMBOLS:
        try:
            res = _run_symbol_full(
                symbol, cfg, base_config_path,
                micro_accel_params=micro_params,
            )
            if res is not None:
                baseline_per_sym[symbol] = res
                print(f"    ✅ {symbol}: CAGR={res['cagr']:.1f}%, SR={res['sharpe']:.2f}")
        except Exception as e:
            print(f"    ⚠️  {symbol} failed: {e}")
    print(f"  → {len(baseline_per_sym)}/{len(SYMBOLS)} symbols OK")

    # 1b. Baseline with +1 bar delay
    print("\n  ── Delay +1 bar per-symbol ──")
    delay_per_sym = {}
    for symbol in SYMBOLS:
        try:
            res = _run_symbol_full(
                symbol, cfg, base_config_path,
                micro_accel_params=micro_params,
                extra_signal_delay=1,
            )
            if res is not None:
                delay_per_sym[symbol] = res
        except Exception:
            pass
    print(f"  → {len(delay_per_sym)}/{len(SYMBOLS)} symbols OK")

    # 1c. BTC with wide SL (for E5-E7)
    print("\n  ── BTC with wide SL (stop_loss_atr=4.0) ──")
    btc_wide_sl = None
    btc_wide_sl_delay = None
    try:
        btc_wide_sl = _run_symbol_full(
            "BTCUSDT", cfg, base_config_path,
            micro_accel_params=micro_params,
            btc_stop_loss_atr=4.0,
        )
        if btc_wide_sl:
            print(f"    ✅ BTCUSDT (SL=4.0): CAGR={btc_wide_sl['cagr']:.1f}%, SR={btc_wide_sl['sharpe']:.2f}")

        btc_wide_sl_delay = _run_symbol_full(
            "BTCUSDT", cfg, base_config_path,
            micro_accel_params=micro_params,
            btc_stop_loss_atr=4.0,
            extra_signal_delay=1,
        )
    except Exception as e:
        print(f"    ⚠️  BTC wide SL failed: {e}")

    # ══════════════════════════════════════════════════════════
    # PHASE 2: For each experiment, aggregate and stress-test
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  PHASE 2: Experiment evaluation")
    print("=" * 100)

    all_results = {}

    for exp in experiments:
        exp_id = exp["id"]
        gross_mult = exp["gross_multiplier"]
        overrides = exp.get("overrides", {}) or {}
        btc_sl_override = overrides.get("btc_stop_loss_atr")
        dd_throttle = overrides.get("dd_throttle")

        print(f"\n  ── {exp_id}: {exp['name']} (gross={gross_mult:.2f}x) ──")

        # Select per-symbol data
        per_sym = dict(baseline_per_sym)
        per_sym_delay = dict(delay_per_sym)

        # For experiments with BTC SL override, swap BTC data
        if btc_sl_override and btc_wide_sl:
            per_sym["BTCUSDT"] = btc_wide_sl
            if btc_wide_sl_delay:
                per_sym_delay["BTCUSDT"] = btc_wide_sl_delay

        # 2a. Baseline portfolio
        port = _aggregate_portfolio(
            per_sym, BASE_WEIGHTS, gross_mult, initial_cash, dd_throttle,
        )
        print(f"    Baseline: CAGR={port['cagr']:.1f}%, SR={port['sharpe']:.2f}, "
              f"MDD={port['max_drawdown_pct']:.2f}%, Calmar={port['calmar']:.1f}")

        # 2b. Delay stress
        delay_port = _aggregate_portfolio(
            per_sym_delay, BASE_WEIGHTS, gross_mult, initial_cash, dd_throttle,
        )
        delay_metrics = {
            "cagr": delay_port["cagr"],
            "sharpe": delay_port["sharpe"],
            "max_drawdown_pct": delay_port["max_drawdown_pct"],
            "calmar": delay_port["calmar"],
        }
        decay = (port["sharpe"] - delay_port["sharpe"]) / abs(port["sharpe"]) * 100 if abs(port["sharpe"]) > 0.01 else 0
        print(f"    Delay:    CAGR={delay_port['cagr']:.1f}%, SR={delay_port['sharpe']:.2f}, "
              f"decay={decay:.1f}%")

        # 2c. Margin estimate
        margin = estimate_margin_utilisation(
            gross_mult, leverage,
            peak_gross=port["peak_gross"],
            avg_gross=port["avg_gross"],
        )
        print(f"    Margin:   normal={margin['normal_margin_pct']:.1f}%, "
              f"peak={margin['peak_margin_pct']:.1f}%")

        # 2d. Monte Carlo stress tests
        mc_results_exp = {}

        if not args.skip_mc:
            # MC1: Block bootstrap
            print(f"    [MC1] Block bootstrap ({args.mc_sims} sims)...")
            mc1 = stress_bootstrap(
                port["portfolio_returns"], initial_cash,
                args.mc_sims, args.block_size, rng,
            )
            mc_results_exp["bootstrap"] = mc1
            if mc1:
                p = _percentile_summary(mc1, "sharpe")
                print(f"      → P50 SR={p['p50']:.2f}, P5 SR={p['p5']:.2f}")

            # MC2: Shuffle
            print(f"    [MC2] Trade shuffle ({args.mc_sims} sims)...")
            mc2 = stress_shuffle(
                port["portfolio_returns"], initial_cash,
                args.mc_sims, rng,
            )
            mc_results_exp["shuffle"] = mc2
            if mc2:
                p = _percentile_summary(mc2, "max_drawdown_pct")
                print(f"      → P50 MDD={p['p50']:.2f}%, P95 MDD={p['p95']:.2f}%")

            # MC3: Cost perturbation
            if args.mc34_sims > 0:
                print(f"    [MC3] Cost perturbation ({args.mc34_sims} sims)...")
                mc3 = stress_cost_perturbation(
                    cfg, base_config_path, micro_params,
                    BASE_WEIGHTS, gross_mult,
                    args.mc34_sims, rng,
                    btc_stop_loss_atr=btc_sl_override,
                    dd_throttle=dd_throttle,
                )
                mc_results_exp["cost"] = mc3

            # MC4: Jitter
            if args.mc34_sims > 0 and not args.skip_jitter:
                print(f"    [MC4] Execution jitter ({args.mc34_sims} sims)...")
                mc4 = stress_jitter(
                    cfg, base_config_path, micro_params,
                    BASE_WEIGHTS, gross_mult,
                    args.mc34_sims, rng,
                    btc_stop_loss_atr=btc_sl_override,
                    dd_throttle=dd_throttle,
                )
                mc_results_exp["jitter"] = mc4

        # 2e. GO/NO-GO evaluation
        evaluation = evaluate_go_nogo(
            exp_id, port, delay_metrics, mc_results_exp, margin,
        )
        print(f"    Verdict:  {'✅ GO' if evaluation['all_pass'] else '❌ NO-GO'}")
        if evaluation["failed_gates"]:
            print(f"    Failed:   {', '.join(evaluation['failed_gates'])}")

        all_results[exp_id] = {
            "experiment": exp,
            "baseline": {k: v for k, v in port.items()
                         if k not in ("portfolio_equity", "portfolio_returns", "gross_series")},
            "delay": delay_metrics,
            "margin": margin,
            "stress": {"delay": delay_metrics, "mc": mc_results_exp},
            "evaluation": evaluation,
        }

    # ══════════════════════════════════════════════════════════
    # PHASE 3: Ranking & Reports
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  PHASE 3: Ranking & Reports")
    print("=" * 100)

    ranking = compute_ranking(all_results)

    # Generate reports
    summary_path = output_dir / "summary_table.csv"
    summary_df = generate_summary_csv(all_results, ranking, summary_path)
    print(f"\n  ✅ Summary table:  {summary_path}")

    ranking_path = output_dir / "ranking.md"
    generate_ranking_md(all_results, ranking, ranking_path, timestamp)
    print(f"  ✅ Ranking:        {ranking_path}")

    go_nogo_path = output_dir / "go_no_go.md"
    generate_go_nogo_md(all_results, ranking, go_nogo_path, timestamp)
    print(f"  ✅ GO/NO-GO:       {go_nogo_path}")

    # Save raw JSON
    json_results = {}
    for k, v in all_results.items():
        r = {kk: vv for kk, vv in v.items() if kk != "stress"}
        # Clean stress data
        stress_clean = {}
        stress_data = v.get("stress", {})
        stress_clean["delay"] = stress_data.get("delay", {})
        mc_data = stress_data.get("mc", {})
        for mc_key, mc_list in mc_data.items():
            if isinstance(mc_list, list):
                # Summarise as percentiles
                if mc_list:
                    stress_clean[mc_key] = {
                        "n_sims": len(mc_list),
                        "cagr": _percentile_summary(mc_list, "cagr"),
                        "sharpe": _percentile_summary(mc_list, "sharpe"),
                        "max_drawdown_pct": _percentile_summary(mc_list, "max_drawdown_pct"),
                    }
        r["stress"] = stress_clean
        json_results[k] = r

    # Convert numpy types
    def _clean_json(obj):
        if isinstance(obj, dict):
            return {k: _clean_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean_json(x) for x in obj]
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, pd.Series):
            return None
        if isinstance(obj, pd.DataFrame):
            return None
        return obj

    json_path = output_dir / "full_results.json"
    with open(json_path, "w") as f:
        json.dump(_clean_json(json_results), f, indent=2, default=str)
    print(f"  ✅ Full JSON:      {json_path}")

    # ── Print final summary table ──
    print("\n" + "=" * 100)
    print("  FINAL SUMMARY")
    print("=" * 100)
    print()
    print(f"  {'Exp':<5} {'Name':<35} {'Gross':>5} {'CAGR':>8} {'Sharpe':>8} {'MDD':>8} "
          f"{'Calmar':>8} {'Delay%':>8} {'Margin':>8} {'Verdict':>10} {'Rank':>5} {'Score':>7}")
    print("  " + "-" * 115)

    for r in ranking:
        k = r["exp_id"]
        v = all_results[k]
        bl = v["baseline"]
        m = v.get("margin", {})
        icon = "✅" if r["verdict"] == "GO" else "❌"
        rank_str = str(r["rank"]) if r["rank"] else "—"
        score_str = f"{r['score']:.3f}" if r["score"] is not None else "—"
        print(
            f"  {k:<5} {v['experiment']['name']:<35} "
            f"{v['experiment']['gross_multiplier']:>5.2f} "
            f"{bl['cagr']:>7.1f}% "
            f"{bl['sharpe']:>8.2f} "
            f"{bl['max_drawdown_pct']:>7.2f}% "
            f"{bl['calmar']:>8.1f} "
            f"{v['evaluation']['sharpe_decay_pct']:>7.1f}% "
            f"{m.get('normal_margin_pct', 0):>7.1f}% "
            f"{icon} {r['verdict']:>7} "
            f"{rank_str:>5} "
            f"{score_str:>7}"
        )

    # Top-2
    go_exps = [r for r in ranking if r["verdict"] == "GO"]
    print()
    if go_exps:
        print(f"  🏆 Top candidates:")
        for i, r in enumerate(go_exps[:2], 1):
            k = r["exp_id"]
            v = all_results[k]
            print(f"     #{i}: {k} — {v['experiment']['name']} "
                  f"(Sharpe={r['sharpe']:.2f}, CAGR={r['cagr']:.1f}%, MDD={r['max_drawdown_pct']:.2f}%)")
    else:
        print("  ⚠️  No experiments passed all gates")
        print("  Recommendation: Stay at 1.0x gross (E0 baseline)")

    print()
    print(f"  📁 All reports: {output_dir}")
    print()
    print("█" * 100)
    print("  R3C Capital Utilization Experiment complete")
    print("█" * 100)


if __name__ == "__main__":
    main()
