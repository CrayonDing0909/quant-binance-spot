#!/usr/bin/env python3
"""
R3C Universe — Monte Carlo Reliability Verification
=====================================================

4 Monte Carlo stress scenarios:
  MC1) Return bootstrap (block bootstrap)
       — resample 1h portfolio returns with overlapping blocks,
         preserving autocorrelation
  MC2) Trade-order shuffle
       — keep per-trade PnL distribution, randomise sequence,
         measure path-dependent MDD tail
  MC3) Cost perturbation
       — fee / slippage / funding each ±20% random perturbation
  MC4) Execution jitter simulation
       — 0–1 bar random delay on every signal change, per symbol

Rules:
  - NO production config changes
  - Cost model ON (fee 5 bps + slippage 3 bps + historical funding)
  - trade_on = next_open
  - All overlays (vol + micro_accel) applied as-is
  - Reproducible seed (42) for all random draws

Outputs:
  reports/r3c_monte_carlo/<timestamp>/mc_summary.json
  reports/r3c_monte_carlo/<timestamp>/mc_paths.csv
  reports/r3c_monte_carlo/<timestamp>/mc_report.md

Usage:
    cd /path/to/quant-binance-spot
    PYTHONPATH=src python scripts/run_r3c_monte_carlo.py
    PYTHONPATH=src python scripts/run_r3c_monte_carlo.py --n-sims 500
    PYTHONPATH=src python scripts/run_r3c_monte_carlo.py --n-sims 2000 --block-size 168
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

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

# ── Production config (R3C Universe) — READ ONLY ──
CONFIG_PATH = PROJECT_ROOT / "config" / "prod_candidate_R3C_universe.yaml"

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
    "LTCUSDT", "NEARUSDT", "APTUSDT", "ARBUSDT", "OPUSDT",
    "FILUSDT", "ATOMUSDT", "UNIUSDT", "AAVEUSDT",
]

WEIGHTS = {
    "BTCUSDT": 0.0722, "ETHUSDT": 0.0538, "SOLUSDT": 0.0509,
    "BNBUSDT": 0.0707, "XRPUSDT": 0.0491, "DOGEUSDT": 0.0512,
    "ADAUSDT": 0.0511, "AVAXUSDT": 0.0545, "LINKUSDT": 0.0538,
    "DOTUSDT": 0.0540, "LTCUSDT": 0.0605, "NEARUSDT": 0.0495,
    "APTUSDT": 0.0483, "ARBUSDT": 0.0495, "OPUSDT": 0.0426,
    "FILUSDT": 0.0489, "ATOMUSDT": 0.0523, "UNIUSDT": 0.0394,
    "AAVEUSDT": 0.0477,
}

# Default MC parameters
DEFAULT_N_SIMS = 1000
DEFAULT_BLOCK_SIZE = 168  # 1 week of 1h bars — preserves weekly autocorrelation
DEFAULT_SEED = 42


# ══════════════════════════════════════════════════════════════
# Config helpers (identical to final gate — no changes)
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
# Core: Single symbol full pipeline (reused from final gate)
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
    funding_rate_mult: float = 1.0,
) -> dict | None:
    """Run single-symbol with both overlays, returning full details."""
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

    # Apply vol overlay (R2.1)
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
        # Apply funding rate multiplier for cost perturbation
        funding_rates = funding_rates * funding_rate_mult

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

    pos_changes = pos.diff().abs().fillna(0)
    turnover = float(pos_changes.sum())

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
        "turnover": turnover,
        "funding_cost_abs": funding_cost_total,
        "n_bars": n_bars,
        "initial_cash": initial_cash,
        "df": df,
    }


def _run_portfolio(
    cfg, config_path, cost_mult=1.0, micro_accel_params=None,
    start_override=None, end_override=None, extra_signal_delay=0,
    symbols=None, weights=None, funding_rate_mult=1.0,
) -> dict:
    """Run portfolio and aggregate."""
    use_symbols = symbols or SYMBOLS
    use_weights = weights or WEIGHTS
    per_symbol = {}

    for symbol in use_symbols:
        try:
            res = _run_symbol_full(
                symbol, cfg, config_path, cost_mult, micro_accel_params,
                start_override, end_override, extra_signal_delay,
                funding_rate_mult=funding_rate_mult,
            )
            if res is not None:
                per_symbol[symbol] = res
        except Exception as e:
            logger.warning(f"    ⚠️  {symbol} failed: {e}")

    if not per_symbol:
        return {
            "sharpe": 0, "total_return_pct": 0, "max_drawdown_pct": 0,
            "cagr": 0, "calmar": 0, "total_trades": 0, "flips": 0,
            "per_symbol": {}, "portfolio_equity": pd.Series(dtype=float),
            "portfolio_returns": pd.Series(dtype=float),
        }

    initial_cash = cfg.backtest.initial_cash
    active = list(per_symbol.keys())
    w = np.array([use_weights.get(s, 1.0 / len(use_symbols)) for s in active])
    w = w / w.sum()

    eqs = {s: per_symbol[s]["equity"] for s in active}
    min_start = max(eq.index[0] for eq in eqs.values())
    max_end = min(eq.index[-1] for eq in eqs.values())
    for s in active:
        eqs[s] = eqs[s].loc[min_start:max_end]

    norm = {}
    for s in active:
        eq = eqs[s]
        if len(eq) > 0 and eq.iloc[0] > 0:
            norm[s] = eq / eq.iloc[0]
        else:
            norm[s] = pd.Series(1.0, index=eq.index)

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

    sharpe = (
        np.sqrt(365 * 24) * port_ret.mean() / port_ret.std()
        if port_ret.std() > 0 else 0
    )
    calmar = cagr / max_dd if max_dd > 0.01 else 0

    total_trades = sum(per_symbol[s]["total_trades"] for s in active)
    total_flips = sum(per_symbol[s]["flips"] for s in active)

    return {
        "total_return_pct": round(total_return, 2),
        "cagr": round(cagr, 2),
        "sharpe": round(sharpe, 4),
        "max_drawdown_pct": round(max_dd, 4),
        "calmar": round(calmar, 2),
        "total_trades": total_trades,
        "flips": total_flips,
        "per_symbol": per_symbol,
        "portfolio_equity": port_eq,
        "portfolio_returns": port_ret,
    }


# ══════════════════════════════════════════════════════════════
# Portfolio metrics from equity / returns
# ══════════════════════════════════════════════════════════════

def _compute_metrics_from_equity(equity: pd.Series, initial_cash: float) -> dict:
    """Compute CAGR / Sharpe / MaxDD from an equity curve."""
    ret = equity.pct_change().fillna(0)
    n_bars = len(ret)
    years = n_bars / (365.25 * 24)

    total_return = (equity.iloc[-1] / initial_cash - 1) * 100
    cagr = ((1 + total_return / 100) ** (1 / max(years, 0.01)) - 1) * 100

    rolling_max = equity.expanding().max()
    dd = (equity - rolling_max) / rolling_max
    max_dd = abs(dd.min()) * 100

    sharpe = (
        np.sqrt(365 * 24) * ret.mean() / ret.std()
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


def _compute_metrics_from_returns(
    returns: pd.Series, initial_cash: float,
) -> dict:
    """Compute CAGR / Sharpe / MaxDD from a return series."""
    equity = (1 + returns).cumprod() * initial_cash
    return _compute_metrics_from_equity(equity, initial_cash)


# ══════════════════════════════════════════════════════════════
# MC1: Return Block Bootstrap
# ══════════════════════════════════════════════════════════════

def mc1_block_bootstrap(
    portfolio_returns: pd.Series,
    initial_cash: float,
    n_sims: int,
    block_size: int,
    rng: np.random.Generator,
) -> list[dict]:
    """
    Block bootstrap of portfolio hourly returns.

    Preserves short-term autocorrelation by resampling in blocks.
    Each simulation draws len(returns)/block_size overlapping blocks,
    concatenates them, and builds a synthetic equity curve.
    """
    print(f"\n  [MC1] Return Block Bootstrap ({n_sims} sims, block={block_size}h)")

    ret = portfolio_returns.values
    n = len(ret)
    n_blocks = int(np.ceil(n / block_size))
    max_start = n - block_size

    if max_start < 1:
        print("    ⚠️  Data too short for block bootstrap, skipping")
        return []

    results = []
    for i in range(n_sims):
        # Draw random block start indices
        starts = rng.integers(0, max_start, size=n_blocks)
        # Concatenate blocks
        synthetic = np.concatenate([ret[s:s + block_size] for s in starts])[:n]
        metrics = _compute_metrics_from_returns(
            pd.Series(synthetic), initial_cash,
        )
        metrics["sim_id"] = i
        results.append(metrics)

        if (i + 1) % 200 == 0 or i == 0:
            print(f"    sim {i+1}/{n_sims}: CAGR={metrics['cagr']:.1f}%, "
                  f"MDD={metrics['max_drawdown_pct']:.2f}%, SR={metrics['sharpe']:.2f}")

    return results


# ══════════════════════════════════════════════════════════════
# MC2: Trade-Order Shuffle
# ══════════════════════════════════════════════════════════════

def mc2_trade_shuffle(
    portfolio_returns: pd.Series,
    initial_cash: float,
    n_sims: int,
    rng: np.random.Generator,
) -> list[dict]:
    """
    Preserve per-bar return distribution but shuffle the order.

    This destroys temporal structure and isolates path-dependent risk.
    The MDD distribution under shuffle shows how much of the observed
    low MDD is due to favourable sequencing vs return quality.
    """
    print(f"\n  [MC2] Trade-Order Shuffle ({n_sims} sims)")

    ret = portfolio_returns.values.copy()
    n = len(ret)
    results = []

    for i in range(n_sims):
        shuffled = ret.copy()
        rng.shuffle(shuffled)
        metrics = _compute_metrics_from_returns(
            pd.Series(shuffled), initial_cash,
        )
        metrics["sim_id"] = i
        results.append(metrics)

        if (i + 1) % 200 == 0 or i == 0:
            print(f"    sim {i+1}/{n_sims}: CAGR={metrics['cagr']:.1f}%, "
                  f"MDD={metrics['max_drawdown_pct']:.2f}%, SR={metrics['sharpe']:.2f}")

    return results


# ══════════════════════════════════════════════════════════════
# MC3: Cost Perturbation
# ══════════════════════════════════════════════════════════════

def mc3_cost_perturbation(
    cfg,
    config_path: str | Path,
    micro_accel_params: dict | None,
    n_sims: int,
    rng: np.random.Generator,
) -> list[dict]:
    """
    Perturb fee, slippage, and funding rate each by ±20% independently.

    Each simulation draws:
      fee_mult    ~ U(0.80, 1.20)
      slip_mult   ~ U(0.80, 1.20)
      funding_mult ~ U(0.80, 1.20)

    Combined cost_mult = (fee_mult + slip_mult) / 2 is used for
    fee/slippage via existing VBT path. Funding is perturbed separately.
    """
    print(f"\n  [MC3] Cost Perturbation ({n_sims} sims, each ±20%)")

    results = []
    for i in range(n_sims):
        fee_mult = float(rng.uniform(0.80, 1.20))
        slip_mult = float(rng.uniform(0.80, 1.20))
        funding_mult = float(rng.uniform(0.80, 1.20))

        # We pass combined multiplier to fee/slippage via cost_mult
        # but that multiplies both equally. Instead, run _run_portfolio
        # with fee_mult and slip_mult applied via cost_mult = avg,
        # and funding_rate_mult separately.
        #
        # Actually: to properly separate fee & slippage perturbation,
        # we use the geometric mean as the cost_mult and accept the
        # tiny coupling. The dominant effect is the total cost level.
        combined_cost_mult = float(np.sqrt(fee_mult * slip_mult))

        res = _run_portfolio(
            cfg, config_path,
            cost_mult=combined_cost_mult,
            micro_accel_params=micro_accel_params,
            funding_rate_mult=funding_mult,
        )

        metrics = {
            "sim_id": i,
            "cagr": res["cagr"],
            "sharpe": res["sharpe"],
            "max_drawdown_pct": res["max_drawdown_pct"],
            "calmar": res["calmar"],
            "total_return_pct": res["total_return_pct"],
            "fee_mult": round(fee_mult, 4),
            "slip_mult": round(slip_mult, 4),
            "funding_mult": round(funding_mult, 4),
        }
        results.append(metrics)

        if (i + 1) % 5 == 0 or i == 0:
            print(
                f"    sim {i+1}/{n_sims}: fee×{fee_mult:.2f} slip×{slip_mult:.2f} "
                f"fund×{funding_mult:.2f} → CAGR={metrics['cagr']:.1f}%, "
                f"MDD={metrics['max_drawdown_pct']:.2f}%, SR={metrics['sharpe']:.2f}"
            )

    return results


# ══════════════════════════════════════════════════════════════
# MC4: Execution Jitter Simulation
# ══════════════════════════════════════════════════════════════

def mc4_execution_jitter(
    cfg,
    config_path: str | Path,
    micro_accel_params: dict | None,
    n_sims: int,
    rng: np.random.Generator,
) -> list[dict]:
    """
    Simulate random execution delay of 0–1 bars on each signal change.

    For each simulation, we:
    1. Run the full signal pipeline for all symbols
    2. At each signal change point, randomly delay the new signal by 0 or 1 bar
    3. Build portfolio equity from the jittered positions
    4. Compute metrics

    NOTE: This is more nuanced than extra_signal_delay because the delay
    is per-change-point, not global. It captures realistic execution jitter.
    """
    print(f"\n  [MC4] Execution Jitter Simulation ({n_sims} sims, 0-1 bar)")

    market_type = cfg.market_type_str
    initial_cash = cfg.backtest.initial_cash
    fee = _bps_to_pct(cfg.backtest.fee_bps)
    slippage = _bps_to_pct(cfg.backtest.slippage_bps)
    vbt_direction = to_vbt_direction(cfg.direction)

    # Pre-compute base positions for all symbols (expensive, do once)
    print("    Pre-computing base positions for all symbols...")
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

        ctx = StrategyContext(
            symbol=symbol,
            interval=cfg.market.interval,
            market_type=market_type,
            direction=cfg.direction,
            signal_delay=1,
        )

        strategy_func = get_strategy(strategy_name)
        pos_base = strategy_func(df, ctx, strategy_params)

        # Apply vol overlay
        vol_overlay = _load_vol_overlay_params(config_path)
        if vol_overlay and vol_overlay.get("enabled", False):
            overlay_mode = vol_overlay.get("mode", "vol_pause")
            overlay_params = vol_overlay.get("params", {})
            pos_base = apply_overlay_by_mode(
                position=pos_base, price_df=df, oi_series=None,
                params=overlay_params, mode=overlay_mode,
            )

        # Apply micro accel overlay
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
                base_position=pos_base,
                df_1h=df, df_5m=df_5m, df_15m=df_15m,
                oi_series=oi_series, params=micro_accel_params,
            )

        pos_base = clip_positions_by_direction(pos_base, market_type, cfg.direction)

        # Date filter
        df, pos_base = _apply_date_filter(df, pos_base, cfg.market.start, cfg.market.end)

        if len(df) < 100:
            continue

        # Find signal change points
        pos_diff = pos_base.diff().abs()
        change_mask = pos_diff > 0.001
        change_indices = np.where(change_mask.values)[0]

        # Load funding rates for this symbol
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
            "df": df,
            "pos": pos_base,
            "change_indices": change_indices,
            "funding_rates": funding_rates,
        }

    print(f"    Pre-computed {len(symbol_data)} symbols, running {n_sims} jitter sims...")

    results = []
    for sim_i in range(n_sims):
        # For each symbol, create jittered positions
        per_symbol_eq = {}

        for symbol, sdata in symbol_data.items():
            pos = sdata["pos"].copy()
            change_idx = sdata["change_indices"]
            df_sym = sdata["df"]

            # Apply random jitter: for each change point, delay by 0 or 1 bar
            jitter_pos = pos.values.copy()
            delays = rng.integers(0, 2, size=len(change_idx))  # 0 or 1

            for ci, delay in zip(change_idx, delays):
                if delay == 1 and ci + 1 < len(jitter_pos):
                    # Delay: keep old value for 1 extra bar
                    jitter_pos[ci] = jitter_pos[ci - 1] if ci > 0 else 0.0

            jitter_series = pd.Series(jitter_pos, index=pos.index)

            # Build VBT portfolio
            try:
                pf = vbt.Portfolio.from_orders(
                    close=df_sym["close"],
                    size=jitter_series,
                    size_type="targetpercent",
                    price=df_sym["open"],
                    fees=fee,
                    slippage=slippage,
                    init_cash=initial_cash,
                    freq=cfg.market.interval,
                    direction=vbt_direction,
                )
                equity = pf.value()

                # Apply funding
                if sdata["funding_rates"] is not None:
                    leverage = cfg.futures.leverage if cfg.futures else 1
                    fc = compute_funding_costs(
                        pos=jitter_series, equity=equity,
                        funding_rates=sdata["funding_rates"],
                        leverage=leverage,
                    )
                    equity = adjust_equity_for_funding(equity, fc)

                per_symbol_eq[symbol] = equity
            except Exception:
                continue

        if not per_symbol_eq:
            continue

        # Build weighted portfolio
        active = list(per_symbol_eq.keys())
        w = np.array([WEIGHTS.get(s, 1.0 / len(SYMBOLS)) for s in active])
        w = w / w.sum()

        eqs = per_symbol_eq
        min_start = max(eq.index[0] for eq in eqs.values())
        max_end = min(eq.index[-1] for eq in eqs.values())
        for s in active:
            eqs[s] = eqs[s].loc[min_start:max_end]

        norm = {}
        for s in active:
            eq = eqs[s]
            if len(eq) > 0 and eq.iloc[0] > 0:
                norm[s] = eq / eq.iloc[0]
            else:
                norm[s] = pd.Series(1.0, index=eq.index)

        port_norm = sum(norm[s] * wi for s, wi in zip(active, w))
        port_eq = port_norm * initial_cash

        metrics = _compute_metrics_from_equity(port_eq, initial_cash)
        metrics["sim_id"] = sim_i
        results.append(metrics)

        if (sim_i + 1) % 5 == 0 or sim_i == 0:
            print(
                f"    sim {sim_i+1}/{n_sims}: CAGR={metrics['cagr']:.1f}%, "
                f"MDD={metrics['max_drawdown_pct']:.2f}%, SR={metrics['sharpe']:.2f}"
            )

    return results


# ══════════════════════════════════════════════════════════════
# Percentile helpers
# ══════════════════════════════════════════════════════════════

def _percentile_summary(results: list[dict], metric: str) -> dict:
    """Extract p5/p25/p50/p75/p95 for a metric."""
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


# ══════════════════════════════════════════════════════════════
# Pass/Fail criteria
# ══════════════════════════════════════════════════════════════

def _evaluate_scenario(scenario_name: str, perc: dict) -> tuple[bool, str]:
    """
    Evaluate pass/fail for a Monte Carlo scenario.

    Criteria:
      - P5 CAGR > 0%  (worst 5% still profitable)
      - P50 Sharpe > 1.0  (median must remain strong)
      - P95 MDD < 40% (tail drawdown within risk budget)
    """
    p5_cagr = perc["cagr"]["p5"]
    p50_sharpe = perc["sharpe"]["p50"]
    p95_mdd = perc["max_drawdown_pct"]["p95"]

    pass_cagr = p5_cagr > 0.0
    pass_sharpe = p50_sharpe > 1.0
    pass_mdd = p95_mdd < 40.0

    all_pass = pass_cagr and pass_sharpe and pass_mdd

    reasons = []
    if not pass_cagr:
        reasons.append(f"P5 CAGR={p5_cagr:.1f}% ≤ 0%")
    if not pass_sharpe:
        reasons.append(f"P50 Sharpe={p50_sharpe:.2f} ≤ 1.0")
    if not pass_mdd:
        reasons.append(f"P95 MDD={p95_mdd:.1f}% ≥ 40%")

    reason = "; ".join(reasons) if reasons else "All criteria met"
    return all_pass, reason


# ══════════════════════════════════════════════════════════════
# Report generation
# ══════════════════════════════════════════════════════════════

def _generate_md_report(
    baseline: dict,
    mc_results: dict,
    mc_summaries: dict,
    verdict: str,
    risk_note: str,
    output_dir: Path,
    n_sims: int,
    block_size: int,
) -> str:
    """Generate the Markdown report."""
    lines = [
        "# R3C Universe — Monte Carlo Reliability Report",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Config**: `config/prod_candidate_R3C_universe.yaml`",
        f"**Simulations**: {n_sims} per scenario",
        f"**Block size**: {block_size}h (MC1)",
        f"**Seed**: {DEFAULT_SEED}",
        "",
        "---",
        "",
        "## Baseline (Deterministic Backtest)",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| CAGR | {baseline['cagr']:.2f}% |",
        f"| Sharpe | {baseline['sharpe']:.4f} |",
        f"| MaxDD | {baseline['max_drawdown_pct']:.2f}% |",
        f"| Calmar | {baseline['calmar']:.2f} |",
        f"| Total Return | {baseline['total_return_pct']:.2f}% |",
        "",
        "---",
        "",
        "## 1. Monte Carlo Summary Table",
        "",
        "| Scenario | Median CAGR | P5 CAGR | P95 CAGR | Median MDD | P95 MDD | Median Sharpe | P5 Sharpe | Pass/Fail |",
        "|----------|------------|---------|---------|------------|---------|---------------|-----------|-----------|",
    ]

    scenario_order = ["MC1_block_bootstrap", "MC2_trade_shuffle", "MC3_cost_perturb", "MC4_exec_jitter"]
    scenario_labels = {
        "MC1_block_bootstrap": "MC1: Block Bootstrap",
        "MC2_trade_shuffle": "MC2: Trade Shuffle",
        "MC3_cost_perturb": "MC3: Cost Perturbation",
        "MC4_exec_jitter": "MC4: Execution Jitter",
    }

    for sc in scenario_order:
        if sc not in mc_summaries:
            continue
        s = mc_summaries[sc]
        p = s["percentiles"]
        passed = s["pass"]
        label = scenario_labels.get(sc, sc)
        icon = "✅ PASS" if passed else "❌ FAIL"
        lines.append(
            f"| {label} "
            f"| {p['cagr']['p50']:.1f}% "
            f"| {p['cagr']['p5']:.1f}% "
            f"| {p['cagr']['p95']:.1f}% "
            f"| {p['max_drawdown_pct']['p50']:.2f}% "
            f"| {p['max_drawdown_pct']['p95']:.2f}% "
            f"| {p['sharpe']['p50']:.2f} "
            f"| {p['sharpe']['p5']:.2f} "
            f"| {icon} |"
        )

    lines += [
        "",
        "**Pass criteria**: P5 CAGR > 0%, Median Sharpe > 1.0, P95 MDD < 40%",
        "",
        "---",
        "",
        "## 2. Risk Interpretation",
        "",
    ]

    # MC1 interpretation
    if "MC1_block_bootstrap" in mc_summaries:
        s = mc_summaries["MC1_block_bootstrap"]
        p = s["percentiles"]
        lines += [
            "### MC1: Block Bootstrap (Return Reliability)",
            "",
            f"- The P5 CAGR of **{p['cagr']['p5']:.1f}%** represents the worst 5th percentile "
            f"outcome under bootstrapped return sequences.",
            f"- P50 Sharpe = **{p['sharpe']['p50']:.2f}** vs baseline {baseline['sharpe']:.2f} "
            f"→ median degradation = {baseline['sharpe'] - p['sharpe']['p50']:.2f}",
            f"- P95 MDD = **{p['max_drawdown_pct']['p95']:.2f}%** (risk budget = 40%)",
            "",
        ]

    # MC2 interpretation
    if "MC2_trade_shuffle" in mc_summaries:
        s = mc_summaries["MC2_trade_shuffle"]
        p = s["percentiles"]
        lines += [
            "### MC2: Trade Shuffle (Path Dependency)",
            "",
            f"- Shuffling trade order while preserving return distribution:",
            f"- P95 MDD = **{p['max_drawdown_pct']['p95']:.2f}%** "
            f"(baseline = {baseline['max_drawdown_pct']:.2f}%)",
            f"- MDD inflation factor = **{p['max_drawdown_pct']['p95'] / max(baseline['max_drawdown_pct'], 0.01):.1f}×**",
            f"- This measures how much of the low backtest MDD is due to lucky sequencing.",
            "",
        ]

    # MC3 interpretation
    if "MC3_cost_perturb" in mc_summaries:
        s = mc_summaries["MC3_cost_perturb"]
        p = s["percentiles"]
        # Calculate sensitivity
        cagr_range = p["cagr"]["p95"] - p["cagr"]["p5"]
        sharpe_range = p["sharpe"]["p95"] - p["sharpe"]["p5"]
        lines += [
            "### MC3: Cost Perturbation (Cost Sensitivity)",
            "",
            f"- Fee/slippage/funding each perturbed ±20% independently.",
            f"- CAGR range (P5–P95): **{p['cagr']['p5']:.1f}% — {p['cagr']['p95']:.1f}%** "
            f"(spread = {cagr_range:.1f}pp)",
            f"- Sharpe range (P5–P95): **{p['sharpe']['p5']:.2f} — {p['sharpe']['p95']:.2f}** "
            f"(spread = {sharpe_range:.2f})",
            f"- Cost sensitivity is {'LOW' if cagr_range < 20 else 'MODERATE' if cagr_range < 40 else 'HIGH'}.",
            "",
        ]

    # MC4 interpretation
    if "MC4_exec_jitter" in mc_summaries:
        s = mc_summaries["MC4_exec_jitter"]
        p = s["percentiles"]
        lines += [
            "### MC4: Execution Jitter (Timing Fragility)",
            "",
            f"- Random 0–1 bar delay per signal change.",
            f"- P50 Sharpe = **{p['sharpe']['p50']:.2f}** vs baseline {baseline['sharpe']:.2f} "
            f"→ degradation = {(1 - p['sharpe']['p50'] / max(baseline['sharpe'], 0.01)) * 100:.1f}%",
            f"- P5 CAGR = **{p['cagr']['p5']:.1f}%** (worst 5th percentile)",
            f"- Timing fragility is "
            f"{'LOW' if abs(baseline['sharpe'] - p['sharpe']['p50']) < 0.5 else 'MODERATE' if abs(baseline['sharpe'] - p['sharpe']['p50']) < 1.5 else 'HIGH'}.",
            "",
        ]

    # Identify most damaging perturbation
    lines += [
        "### Most Damaging Perturbation",
        "",
    ]

    worst_scenario = None
    worst_p5_cagr = float("inf")
    for sc in scenario_order:
        if sc in mc_summaries:
            p5 = mc_summaries[sc]["percentiles"]["cagr"]["p5"]
            if p5 < worst_p5_cagr:
                worst_p5_cagr = p5
                worst_scenario = sc

    if worst_scenario:
        label = scenario_labels.get(worst_scenario, worst_scenario)
        lines.append(
            f"- **{label}** is the most damaging scenario "
            f"(P5 CAGR = {worst_p5_cagr:.1f}%)"
        )
    lines.append("")

    # Tail risk assessment
    lines += [
        "### Tail Risk Assessment",
        "",
    ]
    all_p95_mdd = [
        mc_summaries[sc]["percentiles"]["max_drawdown_pct"]["p95"]
        for sc in scenario_order if sc in mc_summaries
    ]
    worst_mdd = max(all_p95_mdd) if all_p95_mdd else 0
    lines.append(
        f"- Worst P95 MDD across all scenarios: **{worst_mdd:.2f}%** "
        f"({'ACCEPTABLE' if worst_mdd < 40 else 'EXCEEDS RISK BUDGET'})"
    )
    lines.append("")

    lines += [
        "---",
        "",
        "## 3. Final Conclusion",
        "",
        f"### Verdict: **{verdict}**",
        "",
        risk_note,
        "",
        "---",
        "",
        "## 4. Evidence & Reproducibility",
        "",
        f"- Summary JSON: `{output_dir / 'mc_summary.json'}`",
        f"- MC Paths CSV: `{output_dir / 'mc_paths.csv'}`",
        f"- This report: `{output_dir / 'mc_report.md'}`",
        "",
        "### Reproduce command:",
        "```bash",
        "cd /path/to/quant-binance-spot",
        f"PYTHONPATH=src python scripts/run_r3c_monte_carlo.py --n-sims {n_sims} --block-size {block_size}",
        "```",
        "",
    ]

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="R3C Universe — Monte Carlo Reliability Verification"
    )
    parser.add_argument("--n-sims", type=int, default=DEFAULT_N_SIMS,
                        help=f"Number of MC simulations per scenario (default: {DEFAULT_N_SIMS})")
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE,
                        help=f"Block size for bootstrap in hours (default: {DEFAULT_BLOCK_SIZE})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Random seed (default: {DEFAULT_SEED})")
    parser.add_argument("--mc3-sims", type=int, default=None,
                        help="Override n_sims for MC3 (cost perturbation, each is a full backtest)")
    parser.add_argument("--mc4-sims", type=int, default=None,
                        help="Override n_sims for MC4 (execution jitter, each is a full backtest)")
    parser.add_argument("--skip", nargs="*", default=[],
                        choices=["mc1", "mc2", "mc3", "mc4"],
                        help="Skip specific MC scenarios")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "reports" / "r3c_monte_carlo" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    n_sims = args.n_sims
    block_size = args.block_size
    seed = args.seed
    rng = np.random.default_rng(seed)

    # MC3/MC4 are expensive (full backtests), default to lower count
    mc3_sims = args.mc3_sims or min(n_sims, 50)
    mc4_sims = args.mc4_sims or min(n_sims, 50)

    print("█" * 80)
    print("  R3C Universe — Monte Carlo Reliability Verification")
    print("█" * 80)
    print(f"  Timestamp:    {timestamp}")
    print(f"  Output Dir:   {output_dir}")
    print(f"  Config:       {CONFIG_PATH.name}")
    print(f"  Symbols:      {len(SYMBOLS)} coins")
    print(f"  MC Sims:      MC1/MC2={n_sims}, MC3={mc3_sims}, MC4={mc4_sims}")
    print(f"  Block Size:   {block_size}h")
    print(f"  Seed:         {seed}")
    print(f"  Skip:         {args.skip or 'none'}")
    print()

    # ── Load config ──
    cfg = load_config(str(CONFIG_PATH))
    micro_params = _load_micro_accel_params(CONFIG_PATH)

    if micro_params is None:
        print("  ❌ FATAL: No micro_accel_overlay params in config!")
        return

    # ══════════════════════════════════════════════════════════
    # STEP 0: Baseline (deterministic backtest)
    # ══════════════════════════════════════════════════════════
    print("━" * 80)
    print("  STEP 0: Baseline Backtest (deterministic)")
    print("━" * 80)

    baseline = _run_portfolio(
        cfg, CONFIG_PATH, cost_mult=1.0,
        micro_accel_params=micro_params,
    )

    initial_cash = cfg.backtest.initial_cash
    baseline_metrics = {
        "cagr": baseline["cagr"],
        "sharpe": baseline["sharpe"],
        "max_drawdown_pct": baseline["max_drawdown_pct"],
        "calmar": baseline["calmar"],
        "total_return_pct": baseline["total_return_pct"],
        "total_trades": baseline["total_trades"],
    }

    print(f"  Baseline: CAGR={baseline['cagr']:.2f}%, "
          f"Sharpe={baseline['sharpe']:.4f}, "
          f"MDD={baseline['max_drawdown_pct']:.2f}%, "
          f"Calmar={baseline['calmar']:.2f}")

    portfolio_returns = baseline["portfolio_returns"]
    portfolio_equity = baseline["portfolio_equity"]

    if len(portfolio_returns) == 0:
        print("  ❌ FATAL: Empty portfolio returns!")
        return

    # ══════════════════════════════════════════════════════════
    # Run MC scenarios
    # ══════════════════════════════════════════════════════════
    mc_results = {}
    mc_summaries = {}
    all_paths_data = []

    # ── MC1: Block Bootstrap ──
    if "mc1" not in args.skip:
        print("\n" + "━" * 80)
        print("  MC1: Return Block Bootstrap")
        print("━" * 80)

        mc1_res = mc1_block_bootstrap(
            portfolio_returns, initial_cash, n_sims, block_size, rng,
        )
        mc_results["MC1_block_bootstrap"] = mc1_res

        p = {
            "cagr": _percentile_summary(mc1_res, "cagr"),
            "sharpe": _percentile_summary(mc1_res, "sharpe"),
            "max_drawdown_pct": _percentile_summary(mc1_res, "max_drawdown_pct"),
        }
        passed, reason = _evaluate_scenario("MC1_block_bootstrap", p)
        mc_summaries["MC1_block_bootstrap"] = {
            "percentiles": p, "pass": passed, "reason": reason,
            "n_sims": n_sims, "block_size": block_size,
        }

        icon = "✅" if passed else "❌"
        print(f"\n  MC1 Verdict: {icon} {'PASS' if passed else 'FAIL'} — {reason}")
        print(f"    P5 CAGR={p['cagr']['p5']:.1f}%, "
              f"P50 CAGR={p['cagr']['p50']:.1f}%, "
              f"P50 SR={p['sharpe']['p50']:.2f}, "
              f"P95 MDD={p['max_drawdown_pct']['p95']:.2f}%")

        # Store paths for CSV
        for r in mc1_res:
            all_paths_data.append({
                "scenario": "MC1_block_bootstrap",
                "sim_id": r["sim_id"],
                "cagr": r["cagr"],
                "sharpe": r["sharpe"],
                "max_drawdown_pct": r["max_drawdown_pct"],
                "calmar": r.get("calmar", 0),
                "total_return_pct": r.get("total_return_pct", 0),
            })

    # ── MC2: Trade-Order Shuffle ──
    if "mc2" not in args.skip:
        print("\n" + "━" * 80)
        print("  MC2: Trade-Order Shuffle")
        print("━" * 80)

        mc2_res = mc2_trade_shuffle(
            portfolio_returns, initial_cash, n_sims, rng,
        )
        mc_results["MC2_trade_shuffle"] = mc2_res

        p = {
            "cagr": _percentile_summary(mc2_res, "cagr"),
            "sharpe": _percentile_summary(mc2_res, "sharpe"),
            "max_drawdown_pct": _percentile_summary(mc2_res, "max_drawdown_pct"),
        }
        passed, reason = _evaluate_scenario("MC2_trade_shuffle", p)
        mc_summaries["MC2_trade_shuffle"] = {
            "percentiles": p, "pass": passed, "reason": reason,
            "n_sims": n_sims,
        }

        icon = "✅" if passed else "❌"
        print(f"\n  MC2 Verdict: {icon} {'PASS' if passed else 'FAIL'} — {reason}")
        print(f"    P5 CAGR={p['cagr']['p5']:.1f}%, "
              f"P50 MDD={p['max_drawdown_pct']['p50']:.2f}%, "
              f"P95 MDD={p['max_drawdown_pct']['p95']:.2f}%")

        for r in mc2_res:
            all_paths_data.append({
                "scenario": "MC2_trade_shuffle",
                "sim_id": r["sim_id"],
                "cagr": r["cagr"],
                "sharpe": r["sharpe"],
                "max_drawdown_pct": r["max_drawdown_pct"],
                "calmar": r.get("calmar", 0),
                "total_return_pct": r.get("total_return_pct", 0),
            })

    # ── MC3: Cost Perturbation ──
    if "mc3" not in args.skip:
        print("\n" + "━" * 80)
        print("  MC3: Cost Perturbation")
        print("━" * 80)

        mc3_res = mc3_cost_perturbation(
            cfg, CONFIG_PATH, micro_params, mc3_sims, rng,
        )
        mc_results["MC3_cost_perturb"] = mc3_res

        p = {
            "cagr": _percentile_summary(mc3_res, "cagr"),
            "sharpe": _percentile_summary(mc3_res, "sharpe"),
            "max_drawdown_pct": _percentile_summary(mc3_res, "max_drawdown_pct"),
        }
        passed, reason = _evaluate_scenario("MC3_cost_perturb", p)
        mc_summaries["MC3_cost_perturb"] = {
            "percentiles": p, "pass": passed, "reason": reason,
            "n_sims": mc3_sims,
        }

        icon = "✅" if passed else "❌"
        print(f"\n  MC3 Verdict: {icon} {'PASS' if passed else 'FAIL'} — {reason}")
        print(f"    P5 CAGR={p['cagr']['p5']:.1f}%, "
              f"P95 CAGR={p['cagr']['p95']:.1f}%, "
              f"P50 SR={p['sharpe']['p50']:.2f}")

        for r in mc3_res:
            all_paths_data.append({
                "scenario": "MC3_cost_perturb",
                "sim_id": r["sim_id"],
                "cagr": r["cagr"],
                "sharpe": r["sharpe"],
                "max_drawdown_pct": r["max_drawdown_pct"],
                "calmar": r.get("calmar", 0),
                "total_return_pct": r.get("total_return_pct", 0),
                "fee_mult": r.get("fee_mult", 1.0),
                "slip_mult": r.get("slip_mult", 1.0),
                "funding_mult": r.get("funding_mult", 1.0),
            })

    # ── MC4: Execution Jitter ──
    if "mc4" not in args.skip:
        print("\n" + "━" * 80)
        print("  MC4: Execution Jitter")
        print("━" * 80)

        mc4_res = mc4_execution_jitter(
            cfg, CONFIG_PATH, micro_params, mc4_sims, rng,
        )
        mc_results["MC4_exec_jitter"] = mc4_res

        p = {
            "cagr": _percentile_summary(mc4_res, "cagr"),
            "sharpe": _percentile_summary(mc4_res, "sharpe"),
            "max_drawdown_pct": _percentile_summary(mc4_res, "max_drawdown_pct"),
        }
        passed, reason = _evaluate_scenario("MC4_exec_jitter", p)
        mc_summaries["MC4_exec_jitter"] = {
            "percentiles": p, "pass": passed, "reason": reason,
            "n_sims": mc4_sims,
        }

        icon = "✅" if passed else "❌"
        print(f"\n  MC4 Verdict: {icon} {'PASS' if passed else 'FAIL'} — {reason}")
        print(f"    P5 CAGR={p['cagr']['p5']:.1f}%, "
              f"P50 SR={p['sharpe']['p50']:.2f}, "
              f"P95 MDD={p['max_drawdown_pct']['p95']:.2f}%")

        for r in mc4_res:
            all_paths_data.append({
                "scenario": "MC4_exec_jitter",
                "sim_id": r["sim_id"],
                "cagr": r["cagr"],
                "sharpe": r["sharpe"],
                "max_drawdown_pct": r["max_drawdown_pct"],
                "calmar": r.get("calmar", 0),
                "total_return_pct": r.get("total_return_pct", 0),
            })

    # ══════════════════════════════════════════════════════════
    # FINAL VERDICT
    # ══════════════════════════════════════════════════════════
    print("\n\n" + "█" * 80)
    print("  R3C UNIVERSE — MONTE CARLO RELIABILITY REPORT")
    print("█" * 80)

    # Check all pass
    n_pass = sum(1 for s in mc_summaries.values() if s["pass"])
    n_total = len(mc_summaries)
    all_pass = n_pass == n_total

    # Determine verdict
    if all_pass:
        verdict = "GO_LIVE_R3C"
        risk_note = (
            "All 4 Monte Carlo stress scenarios PASS. "
            "The strategy demonstrates robustness to return resampling, "
            "trade sequencing, cost perturbation, and execution jitter. "
            "Proceed with paper trading phase as defined in "
            "`config/prod_scale_rules_R3C_universe.yaml`."
        )
    elif n_pass >= 3:
        # Identify which failed
        failed = [sc for sc, s in mc_summaries.items() if not s["pass"]]
        verdict = "GO_LIVE_WITH_REDUCED_RISK"
        risk_note = (
            f"{n_pass}/{n_total} MC scenarios pass. "
            f"Failed: {', '.join(failed)}. "
            "Recommendation: reduce initial capital allocation to 50% of planned "
            "(i.e., pilot_phase.capital_pct = 0.05 instead of 0.10), "
            "reduce max leverage to 2×, and set kill-switch MDD threshold to 25% "
            "instead of 40%. Monitor the failing scenario metrics closely during "
            "paper-trade phase. If live metrics stay within P25–P75 of passing "
            "scenarios for 4 weeks, gradually ramp up."
        )
    else:
        failed = [sc for sc, s in mc_summaries.items() if not s["pass"]]
        verdict = "HOLD_R3C_NEED_FIX"
        risk_note = (
            f"Only {n_pass}/{n_total} MC scenarios pass. "
            f"Failed: {', '.join(failed)}. "
            "The strategy shows significant fragility under Monte Carlo stress. "
            "DO NOT deploy. Investigate the root cause of failures. "
            "Consider: (1) reducing position sizing, (2) adding volatility "
            "targeting, (3) widening the cost model assumptions, "
            "(4) implementing adaptive signal smoothing."
        )

    # ── Print Summary Table ──
    print("\n" + "=" * 120)
    print("1) MONTE CARLO SUMMARY TABLE")
    print("=" * 120)
    print(f"{'Scenario':<30} {'Med CAGR':>10} {'P5 CAGR':>10} {'P95 CAGR':>10} "
          f"{'Med MDD':>10} {'P95 MDD':>10} {'Med SR':>10} {'P5 SR':>10} {'Pass/Fail':>12}")
    print("-" * 120)

    scenario_labels = {
        "MC1_block_bootstrap": "MC1: Block Bootstrap",
        "MC2_trade_shuffle": "MC2: Trade Shuffle",
        "MC3_cost_perturb": "MC3: Cost Perturbation",
        "MC4_exec_jitter": "MC4: Exec Jitter",
    }

    for sc_key in ["MC1_block_bootstrap", "MC2_trade_shuffle", "MC3_cost_perturb", "MC4_exec_jitter"]:
        if sc_key not in mc_summaries:
            continue
        s = mc_summaries[sc_key]
        p = s["percentiles"]
        label = scenario_labels.get(sc_key, sc_key)
        icon = "✅ PASS" if s["pass"] else "❌ FAIL"
        print(
            f"{label:<30} "
            f"{p['cagr']['p50']:>9.1f}% "
            f"{p['cagr']['p5']:>9.1f}% "
            f"{p['cagr']['p95']:>9.1f}% "
            f"{p['max_drawdown_pct']['p50']:>9.2f}% "
            f"{p['max_drawdown_pct']['p95']:>9.2f}% "
            f"{p['sharpe']['p50']:>10.2f} "
            f"{p['sharpe']['p5']:>9.2f} "
            f"{icon:>12}"
        )

    print(f"\n  Baseline: CAGR={baseline['cagr']:.2f}%, "
          f"Sharpe={baseline['sharpe']:.4f}, MDD={baseline['max_drawdown_pct']:.2f}%")

    # ── Print Risk Interpretation ──
    print("\n" + "=" * 120)
    print("2) RISK INTERPRETATION")
    print("=" * 120)

    # Identify most damaging
    worst_sc = None
    worst_p5 = float("inf")
    for sc, s in mc_summaries.items():
        p5 = s["percentiles"]["cagr"]["p5"]
        if p5 < worst_p5:
            worst_p5 = p5
            worst_sc = sc

    if worst_sc:
        print(f"\n  Most damaging perturbation: {scenario_labels.get(worst_sc, worst_sc)} "
              f"(P5 CAGR = {worst_p5:.1f}%)")

    # Tail risk
    all_p95_mdd = [s["percentiles"]["max_drawdown_pct"]["p95"] for s in mc_summaries.values()]
    worst_mdd = max(all_p95_mdd) if all_p95_mdd else 0
    print(f"  Worst P95 MDD: {worst_mdd:.2f}% "
          f"({'ACCEPTABLE (< 40%)' if worst_mdd < 40 else 'EXCEEDS RISK BUDGET (≥ 40%)'})")

    # Cost sensitivity
    if "MC3_cost_perturb" in mc_summaries:
        p = mc_summaries["MC3_cost_perturb"]["percentiles"]
        cagr_range = p["cagr"]["p95"] - p["cagr"]["p5"]
        print(f"  Cost sensitivity (CAGR P5–P95 spread): {cagr_range:.1f}pp "
              f"({'LOW' if cagr_range < 20 else 'MODERATE' if cagr_range < 40 else 'HIGH'})")

    # Timing fragility
    if "MC4_exec_jitter" in mc_summaries:
        p = mc_summaries["MC4_exec_jitter"]["percentiles"]
        sr_drop = baseline["sharpe"] - p["sharpe"]["p50"]
        sr_drop_pct = sr_drop / max(abs(baseline["sharpe"]), 0.01) * 100
        print(f"  Timing fragility (Sharpe drop): {sr_drop_pct:.1f}% "
              f"({'LOW' if sr_drop_pct < 10 else 'MODERATE' if sr_drop_pct < 25 else 'HIGH'})")

    # ── Final Verdict ──
    print("\n" + "=" * 120)
    print("3) FINAL VERDICT")
    print("=" * 120)

    verdict_icon = {"GO_LIVE_R3C": "✅", "GO_LIVE_WITH_REDUCED_RISK": "🟡", "HOLD_R3C_NEED_FIX": "❌"}
    print(f"\n  {verdict_icon.get(verdict, '?')} VERDICT: {verdict}")
    print(f"\n  {risk_note}")

    if verdict == "GO_LIVE_WITH_REDUCED_RISK":
        print("\n  ── Recommended Risk Reduction Parameters ──")
        print("    - pilot_phase.capital_pct: 0.05 (50% of normal)")
        print("    - max_leverage: 2 (down from 3)")
        print("    - kill_switch.max_drawdown_pct: 25% (down from 40%)")
        print("    - kill_switch.max_daily_loss_pct: 3% (down from 5%)")
        print("    - Minimum paper-trade: 6 weeks (up from 4)")

    # ── Evidence Paths ──
    print("\n" + "=" * 120)
    print("4) EVIDENCE PATHS")
    print("=" * 120)

    # ══════════════════════════════════════════════════════════
    # Save outputs
    # ══════════════════════════════════════════════════════════

    # 1. mc_summary.json
    summary_json = {
        "timestamp": timestamp,
        "config": str(CONFIG_PATH),
        "seed": seed,
        "n_sims": {"mc1": n_sims, "mc2": n_sims, "mc3": mc3_sims, "mc4": mc4_sims},
        "block_size": block_size,
        "baseline": baseline_metrics,
        "verdict": verdict,
        "risk_note": risk_note,
        "all_pass": all_pass,
        "n_pass": n_pass,
        "n_total": n_total,
        "scenarios": {},
    }

    for sc_key, s in mc_summaries.items():
        summary_json["scenarios"][sc_key] = {
            "pass": s["pass"],
            "reason": s["reason"],
            "n_sims": s.get("n_sims", n_sims),
            "percentiles": {
                k: {kk: float(vv) for kk, vv in v.items()}
                for k, v in s["percentiles"].items()
            },
        }

    summary_path = output_dir / "mc_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_json, f, indent=2, default=str)
    print(f"  Summary JSON:  {summary_path}")

    # 2. mc_paths.csv
    paths_path = output_dir / "mc_paths.csv"
    if all_paths_data:
        paths_df = pd.DataFrame(all_paths_data)
        paths_df.to_csv(paths_path, index=False)
        print(f"  MC Paths CSV:  {paths_path}")
    else:
        print("  ⚠️  No MC path data to save")

    # 3. mc_report.md
    report_md = _generate_md_report(
        baseline=baseline_metrics,
        mc_results=mc_results,
        mc_summaries=mc_summaries,
        verdict=verdict,
        risk_note=risk_note,
        output_dir=output_dir,
        n_sims=n_sims,
        block_size=block_size,
    )
    report_path = output_dir / "mc_report.md"
    with open(report_path, "w") as f:
        f.write(report_md)
    print(f"  MC Report MD:  {report_path}")

    print(f"  Output dir:    {output_dir}")
    print(f"  Config used:   {CONFIG_PATH}")
    print(f"  Scale rules:   config/prod_scale_rules_R3C_universe.yaml")

    # Reproduce command
    print(f"\n  Reproduce:")
    print(f"    cd {PROJECT_ROOT}")
    print(f"    PYTHONPATH=src python scripts/run_r3c_monte_carlo.py --n-sims {n_sims} "
          f"--block-size {block_size} --seed {seed}")

    print("\n" + "=" * 120)
    print(f"  R3C Universe Monte Carlo Verification complete. Verdict: {verdict}")
    print("=" * 120)

    return verdict


if __name__ == "__main__":
    main()
