#!/usr/bin/env python3
"""
Silver Bullet Validation (Case C)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Compare the full-overlay E3 baseline with "E3 + Gates only (no EMA)"
to validate whether execution gates alone reduce churn without hurting Sharpe.

Case A — Original E3 (10-symbol, ensemble, oi_vol + micro_accel overlays)
Case C — E3 + rb=3% + fg=80% execution gates ONLY (no EMA smoothing)

Pipeline per symbol:
  strategy(df) → oi_vol overlay → micro_accel overlay
  → direction clip → position sizing → date filter → [gates if Case C] → VBT

Output:
  reports/research/golden_combo/<ts>/
    equity_curve_comparison.png
    comparison_table.csv
    per_symbol_breakdown.csv
    summary.txt

Usage:
  cd /Users/dylanting/Documents/spot_bot/quant-binance-spot
  source .venv/bin/activate
  PYTHONPATH=src python scripts/golden_combo_validation.py
"""
from __future__ import annotations

import gc
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
warnings.filterwarnings("ignore", category=FutureWarning)

# ══════════════════════════════════════════════════════════════
#  Constants
# ══════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "prod_live_R3C_E3.yaml"
DATA_DIR = PROJECT_ROOT / "data"
OUT_ROOT = PROJECT_ROOT / "reports" / "research" / "silver_bullet"

# Date range (align with previous reports)
BT_START = "2023-03-01"
BT_END = "2026-02-28"

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT",
]

# E3 3.5x weights from prod config
E3_WEIGHTS = {
    "BTCUSDT": 0.4450, "ETHUSDT": 0.3316, "SOLUSDT": 0.3137,
    "BNBUSDT": 0.4358, "XRPUSDT": 0.3026, "DOGEUSDT": 0.3156,
    "ADAUSDT": 0.3150, "AVAXUSDT": 0.3359, "LINKUSDT": 0.3316,
    "LTCUSDT": 0.3729,
}
# sum = ~3.50, this is the actual gross leverage


# ══════════════════════════════════════════════════════════════
#  Config helpers
# ══════════════════════════════════════════════════════════════

def _load_ensemble_strategy(symbol: str) -> tuple[str, dict] | None:
    """Load per-symbol strategy override from the E3 config."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    ens = raw.get("ensemble")
    if ens and ens.get("enabled", False):
        strategies = ens.get("strategies", {})
        if symbol in strategies:
            s = strategies[symbol]
            return s["name"], s.get("params", {})
    return None


def _load_overlay_params() -> tuple[dict | None, dict | None]:
    """Load OI/Vol overlay and micro-accel overlay params from config."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    strat = raw.get("strategy", {})

    vol_overlay = strat.get("overlay")
    if vol_overlay and not vol_overlay.get("enabled", False):
        vol_overlay = None

    micro_accel = strat.get("micro_accel_overlay")
    if micro_accel and micro_accel.get("enabled", False):
        micro_params = micro_accel.get("params", {})
    else:
        micro_params = None

    return vol_overlay, micro_params


# ══════════════════════════════════════════════════════════════
#  Gate simulation (same as churn research & live runner)
# ══════════════════════════════════════════════════════════════

def apply_gates(raw_pos: pd.Series, rb: float, fg: float) -> pd.Series:
    """Simulate rebalance_band + fill_gate from live runner.

    rb: rebalance band (skip if |target - current| < rb, same direction)
    fg: fill gate (skip if current/target >= fg, same direction, non-zero target)
    """
    vals = raw_pos.values.astype(np.float64)
    out = np.empty(len(vals), dtype=np.float64)
    prev = 0.0
    for i in range(len(vals)):
        tgt = vals[i]
        diff = abs(tgt - prev)
        same_dir = (tgt > 0 and prev > 0) or (tgt < 0 and prev < 0)
        flip = (tgt > 0 and prev < 0) or (tgt < 0 and prev > 0)
        ok = True

        # Fill gate: if already filled ≥ fg fraction of target, skip
        if same_dir and fg < 1.0 and tgt != 0.0:
            if abs(prev / tgt) >= fg:
                ok = False

        # Rebalance band: skip if diff is small and same direction
        if ok and rb > 0 and not flip and diff < rb:
            ok = False

        out[i] = tgt if ok else prev
        prev = out[i]
    return pd.Series(out, index=raw_pos.index)


# ══════════════════════════════════════════════════════════════
#  Per-symbol backtest (supports EMA + gates injection)
# ══════════════════════════════════════════════════════════════

def run_symbol(
    symbol: str,
    cfg,
    vol_overlay: dict | None,
    micro_params: dict | None,
    ema_span: int = 0,
    rb: float = 0.0,
    fg: float = 1.0,
) -> dict | None:
    """Run a single symbol backtest with full overlay pipeline.

    Returns dict with equity curve, costs, turnover, positions.
    """
    market_type = cfg.market_type_str
    data_path = (
        cfg.data_dir / "binance" / market_type
        / cfg.market.interval / f"{symbol}.parquet"
    )
    if not data_path.exists():
        print(f"  ⚠ {symbol}: data not found")
        return None

    df = load_klines(data_path)
    df = clean_data(df, fill_method="forward", remove_outliers=False,
                    remove_duplicates=True)

    # Load multi-TF data for micro-accel overlay
    multi_tf = load_multi_tf_klines(cfg.data_dir, symbol, market_type)
    df_5m = multi_tf.get("5m")
    df_15m = multi_tf.get("15m")

    # ── 1. Strategy signal (ensemble routing) ──
    ensemble_override = _load_ensemble_strategy(symbol)
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
        signal_delay=1,  # trade_on=next_open → signal_delay=1
    )

    strategy_func = get_strategy(strategy_name)
    pos = strategy_func(df, ctx, strategy_params)

    # ── 2. [NEW] EMA signal smoothing (before overlays) ──
    if ema_span > 0:
        pos = pos.ewm(span=ema_span, adjust=False).mean().clip(-1.0, 1.0)

    # ── 3. OI/Vol overlay ──
    if vol_overlay and vol_overlay.get("enabled", False):
        # Load OI data
        oi_series = None
        try:
            from qtrade.data.open_interest import (
                get_oi_path, load_open_interest, align_oi_to_klines,
            )
            for prov in ["merged", "coinglass", "binance"]:
                oi_path = get_oi_path(cfg.data_dir, symbol, prov)
                oi_df = load_open_interest(oi_path)
                if oi_df is not None and not oi_df.empty:
                    oi_series = align_oi_to_klines(oi_df, df.index, max_ffill_bars=2)
                    break
        except Exception:
            pass

        pos = apply_overlay_by_mode(
            position=pos, price_df=df, oi_series=oi_series,
            params=vol_overlay.get("params", {}),
            mode=vol_overlay.get("mode", "vol_pause"),
        )

    # ── 4. Micro-accel overlay ──
    if micro_params is not None:
        oi_series_micro = None
        try:
            from qtrade.data.open_interest import (
                get_oi_path, load_open_interest, align_oi_to_klines,
            )
            for prov in ["merged", "coinglass", "binance"]:
                oi_path = get_oi_path(cfg.data_dir, symbol, prov)
                oi_df = load_open_interest(oi_path)
                if oi_df is not None and not oi_df.empty:
                    oi_series_micro = align_oi_to_klines(oi_df, df.index, max_ffill_bars=2)
                    break
        except Exception:
            pass

        pos = apply_full_micro_accel_overlay(
            base_position=pos,
            df_1h=df, df_5m=df_5m, df_15m=df_15m,
            oi_series=oi_series_micro, params=micro_params,
        )

    # ── 5. Direction clip ──
    pos = clip_positions_by_direction(pos, market_type, cfg.direction)

    # ── 6. Position sizing ──
    ps_cfg = cfg.position_sizing
    if ps_cfg.method == "fixed" and ps_cfg.position_pct < 1.0:
        pos = pos * ps_cfg.position_pct

    # ── 7. Date filter ──
    df, pos = _apply_date_filter(df, pos, BT_START, BT_END)
    if len(df) < 100:
        print(f"  ⚠ {symbol}: insufficient data after filter ({len(df)} bars)")
        return None

    # ── 8. [NEW] Gates (after all signal processing, before execution) ──
    if rb > 0 or fg < 1.0:
        pos = apply_gates(pos, rb, fg)

    # ── 9. VBT Portfolio ──
    close = df["close"]
    open_ = df["open"]
    fee_bps = cfg.backtest.fee_bps
    slippage_bps = cfg.backtest.slippage_bps
    fee = _bps_to_pct(fee_bps)
    slippage = _bps_to_pct(slippage_bps)
    initial_cash = cfg.backtest.initial_cash

    vbt_direction = to_vbt_direction(cfg.direction)
    pf = vbt.Portfolio.from_orders(
        close=close, size=pos,
        size_type="targetpercent",
        price=open_,
        fees=fee, slippage=slippage,
        init_cash=initial_cash,
        freq=cfg.market.interval,
        direction=vbt_direction,
    )

    stats = pf.stats()
    equity = pf.value()

    # ── 10. Funding rate ──
    adjusted_equity = None
    funding_cost_total = 0.0
    fr_cfg = cfg.backtest.funding_rate
    if fr_cfg.enabled and market_type == "futures":
        funding_df = None
        if fr_cfg.use_historical:
            fr_path = get_funding_rate_path(cfg.data_dir, symbol)
            funding_df = load_funding_rates(fr_path)
        funding_rates = align_funding_to_klines(
            funding_df, df.index, default_rate_8h=fr_cfg.default_rate_8h,
        )
        leverage = cfg.futures.leverage if cfg.futures else 1
        fc = compute_funding_costs(
            pos=pos, equity=equity,
            funding_rates=funding_rates,
            leverage=leverage,
        )
        adjusted_equity = adjust_equity_for_funding(equity, fc)
        funding_cost_total = fc.total_cost

    eq = adjusted_equity if adjusted_equity is not None else equity

    # Metrics
    total_fees_paid = float(stats.get("Total Fees Paid", 0))
    total_trades = int(stats.get("Total Trades", 0))
    turnover = float(pos.diff().abs().fillna(0).sum())
    flips = compute_flip_count(pos)

    return {
        "symbol": symbol,
        "strategy": strategy_name,
        "equity": eq,
        "pos": pos,
        "total_fees_paid": total_fees_paid,
        "funding_cost": funding_cost_total,
        "total_trades": total_trades,
        "turnover": turnover,
        "flips": flips,
        "initial_cash": initial_cash,
        "n_bars": len(df),
    }


# ══════════════════════════════════════════════════════════════
#  Portfolio aggregation
# ══════════════════════════════════════════════════════════════

def aggregate_portfolio(
    per_symbol: dict[str, dict],
    weights: dict[str, float],
    initial_cash: float,
) -> dict:
    """Aggregate per-symbol equity curves into a portfolio.

    Uses weights as-is (sum ≈ 3.5 for E3) to reflect actual leverage.
    Portfolio return = Σ(w_s × r_s(t)) where r_s(t) = pct_change of equity.
    """
    active = list(per_symbol.keys())
    if not active:
        return {}

    # Weights for active symbols (no renormalization — use E3 weights directly)
    w = np.array([weights.get(s, 0.0) for s in active])

    # Align equity curves
    eqs = {s: per_symbol[s]["equity"] for s in active}
    min_start = max(eq.index[0] for eq in eqs.values())
    max_end = min(eq.index[-1] for eq in eqs.values())
    for s in active:
        eqs[s] = eqs[s].loc[min_start:max_end]

    # Normalize to 1.0
    norm = {}
    for s in active:
        eq = eqs[s]
        if len(eq) > 0 and eq.iloc[0] > 0:
            norm[s] = eq / eq.iloc[0]
        else:
            norm[s] = pd.Series(1.0, index=eq.index)

    # Portfolio returns (weighted sum)
    port_ret = pd.Series(0.0, index=norm[active[0]].index)
    for s, ws in zip(active, w):
        sym_ret = norm[s].pct_change().fillna(0)
        port_ret += ws * sym_ret

    port_eq = (1 + port_ret).cumprod() * initial_cash

    # Metrics
    n_bars = len(port_ret)
    years = max(0.5, n_bars / (365.25 * 24))
    total_return = (port_eq.iloc[-1] / initial_cash - 1) * 100
    cagr = ((1 + total_return / 100) ** (1 / years) - 1) * 100

    dd = (port_eq - port_eq.cummax()) / port_eq.cummax()
    mdd = abs(float(dd.min())) * 100

    sharpe = (
        float(np.sqrt(365 * 24) * port_ret.mean() / port_ret.std())
        if port_ret.std() > 0 else 0
    )
    dn = port_ret[port_ret < 0]
    sortino = (
        float(np.sqrt(365 * 24) * port_ret.mean() / dn.std())
        if len(dn) > 1 and dn.std() > 0 else 0
    )
    calmar = cagr / mdd if mdd > 0.01 else 0

    # Aggregate costs (weighted)
    total_fee = sum(w_s * per_symbol[s].get("total_fees_paid", 0)
                    for s, w_s in zip(active, w))
    total_funding = sum(w_s * per_symbol[s].get("funding_cost", 0)
                        for s, w_s in zip(active, w))
    total_turnover = sum(w_s * per_symbol[s].get("turnover", 0)
                         for s, w_s in zip(active, w))
    total_trades = sum(per_symbol[s].get("total_trades", 0) for s in active)
    total_flips = sum(per_symbol[s].get("flips", 0) for s in active)

    fee_pct = total_fee / initial_cash * 100
    funding_pct = total_funding / initial_cash * 100

    # Gross exposure
    gross_parts = []
    for s, ws in zip(active, w):
        p = per_symbol[s]["pos"]
        p = p.reindex(port_eq.index, method="ffill").fillna(0)
        gross_parts.append(abs(p) * ws)
    gross_series = sum(gross_parts)
    avg_gross = float(gross_series.mean())

    return {
        "equity": port_eq,
        "returns": port_ret,
        "CAGR": round(cagr, 2),
        "Sharpe": round(sharpe, 3),
        "Sortino": round(sortino, 3),
        "MaxDD": round(mdd, 2),
        "Calmar": round(calmar, 3),
        "TotalReturn": round(total_return, 2),
        "Fee%": round(fee_pct, 2),
        "Funding%": round(funding_pct, 2),
        "Turnover": round(total_turnover, 1),
        "Trades": total_trades,
        "Flips": total_flips,
        "AvgGross": round(avg_gross, 4),
        "WeightSum": round(float(w.sum()), 4),
        "N_Symbols": len(active),
        "Start": str(port_eq.index[0])[:10],
        "End": str(port_eq.index[-1])[:10],
    }


# ══════════════════════════════════════════════════════════════
#  Run one case
# ══════════════════════════════════════════════════════════════

def run_case(
    label: str,
    cfg,
    vol_overlay: dict | None,
    micro_params: dict | None,
    ema_span: int = 0,
    rb: float = 0.0,
    fg: float = 1.0,
) -> tuple[dict[str, dict], dict]:
    """Run all symbols for a case, return (per_symbol_results, portfolio_metrics)."""
    print(f"\n{'━' * 72}")
    print(f"  🔬 {label}")
    if ema_span > 0:
        print(f"     EMA({ema_span}) + rb={rb*100:.0f}% + fg={fg*100:.0f}%")
    else:
        print(f"     (no EMA, no gates)")
    print(f"{'━' * 72}")

    per_symbol: dict[str, dict] = {}
    for sym in SYMBOLS:
        res = run_symbol(
            sym, cfg,
            vol_overlay=vol_overlay,
            micro_params=micro_params,
            ema_span=ema_span,
            rb=rb, fg=fg,
        )
        if res is not None:
            per_symbol[sym] = res
            eq = res["equity"]
            ret_pct = (eq.iloc[-1] / res["initial_cash"] - 1) * 100
            print(f"  ✓ {sym:>10} [{res['strategy']:>20}]: "
                  f"Ret={ret_pct:+6.1f}% "
                  f"Trades={res['total_trades']:>4} "
                  f"TO={res['turnover']:>6.0f} "
                  f"Fee=${res['total_fees_paid']:>8.0f}")
        else:
            print(f"  ✗ {sym:>10}: FAILED")

        gc.collect()

    portfolio = aggregate_portfolio(per_symbol, E3_WEIGHTS, cfg.backtest.initial_cash)
    return per_symbol, portfolio


# ══════════════════════════════════════════════════════════════
#  Chart
# ══════════════════════════════════════════════════════════════

def plot_equity_curves(eq_a: pd.Series, eq_b: pd.Series, out_path: Path):
    """Plot two equity curves side by side."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("  ⚠ matplotlib not available, skipping chart")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})

    # Equity curves
    ax1.plot(eq_a.index, eq_a.values, label="Case A: E3 Baseline", color="#2196F3", linewidth=1.5)
    ax1.plot(eq_b.index, eq_b.values, label="Case C: E3 + Gates only", color="#4CAF50", linewidth=1.5)
    ax1.set_title("Silver Bullet Validation — Portfolio Equity Curves\n"
                   "(10-symbol, Ensemble + OI/Vol + Micro-Accel, 3.5x leverage)",
                   fontsize=13, fontweight="bold")
    ax1.set_ylabel("Portfolio Equity ($)")
    ax1.legend(loc="upper left", fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    # Drawdown comparison
    dd_a = (eq_a - eq_a.cummax()) / eq_a.cummax() * 100
    dd_b = (eq_b - eq_b.cummax()) / eq_b.cummax() * 100
    ax2.fill_between(dd_a.index, dd_a.values, 0, alpha=0.3, color="#2196F3", label="Case A DD")
    ax2.fill_between(dd_b.index, dd_b.values, 0, alpha=0.3, color="#4CAF50", label="Case C DD")
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.legend(loc="lower left", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊 Equity curve chart saved: {out_path}")


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUT_ROOT / ts
    out.mkdir(parents=True, exist_ok=True)

    print(f"{'═' * 72}")
    print(f"  🥈 Silver Bullet Validation (Case C: Gates only)")
    print(f"  Config: {CONFIG_PATH}")
    print(f"  Period: {BT_START} → {BT_END}")
    print(f"  Output: {out}")
    print(f"  Time:   {datetime.now().isoformat()}")
    print(f"{'═' * 72}")

    # ── Look-ahead checks ──
    cfg = load_config(str(CONFIG_PATH))
    assert cfg.backtest.trade_on == "next_open", "trade_on must be next_open"
    print(f"  ✅ trade_on={cfg.backtest.trade_on} → signal_delay=1")
    print(f"  ✅ fee={cfg.backtest.fee_bps}bps, slippage={cfg.backtest.slippage_bps}bps")
    print(f"  ✅ leverage={cfg.futures.leverage if cfg.futures else 1}")
    print(f"  ✅ Weight sum (E3 3.5x): {sum(E3_WEIGHTS.values()):.4f}")

    # Load overlay params
    vol_overlay, micro_params = _load_overlay_params()
    print(f"  ✅ OI/Vol overlay: {'enabled' if vol_overlay else 'DISABLED'}")
    print(f"  ✅ Micro-accel:    {'enabled' if micro_params else 'DISABLED'}")

    # ── Case A: Original E3 baseline ──
    ps_a, port_a = run_case(
        "Case A — E3 Baseline (no EMA, no gates)",
        cfg, vol_overlay, micro_params,
        ema_span=0, rb=0.0, fg=1.0,
    )

    # ── Case C: E3 + Gates ONLY (no EMA) ──
    ps_b, port_b = run_case(
        "Case C — E3 + rb=3% + fg=80% (Gates only, no EMA)",
        cfg, vol_overlay, micro_params,
        ema_span=0, rb=0.03, fg=0.80,
    )

    # ══════════════════════════════════════════════════════════════
    #  Comparison Table
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'═' * 72}")
    print(f"  📊 Portfolio Comparison Table")
    print(f"{'═' * 72}")

    metrics = ["CAGR", "Sharpe", "Sortino", "MaxDD", "Calmar", "TotalReturn",
               "Fee%", "Funding%", "Turnover", "Trades", "Flips", "AvgGross",
               "WeightSum", "N_Symbols", "Start", "End"]

    print(f"\n  {'Metric':<20} {'Case A (Baseline)':>20} {'Case C (Gates only)':>20} {'Delta':>15}")
    print(f"  {'─' * 75}")
    comp_rows = []
    for m in metrics:
        va = port_a.get(m, "N/A")
        vb = port_b.get(m, "N/A")
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            delta = vb - va
            pct = f" ({delta / abs(va) * 100:+.1f}%)" if abs(va) > 0.001 else ""
            if m in ("MaxDD", "Fee%", "Funding%", "Turnover", "Trades"):
                # Lower is better for these
                marker = "✅" if delta < 0 else ("⚠" if delta > 0 else "—")
            else:
                marker = "✅" if delta > 0 else ("⚠" if delta < 0 else "—")
            print(f"  {m:<20} {va:>20} {vb:>20} {delta:>+10.2f}{pct} {marker}")
            comp_rows.append({"Metric": m, "CaseA": va, "CaseC": vb,
                              "Delta": round(delta, 3),
                              "Delta%": round(delta / abs(va) * 100, 1) if abs(va) > 0.001 else 0})
        else:
            print(f"  {m:<20} {str(va):>20} {str(vb):>20}")
            comp_rows.append({"Metric": m, "CaseA": va, "CaseC": vb, "Delta": "", "Delta%": ""})

    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(out / "comparison_table.csv", index=False)
    print(f"\n  💾 comparison_table.csv saved")

    # ══════════════════════════════════════════════════════════════
    #  Per-symbol breakdown
    # ══════════════════════════════════════════════════════════════
    sym_rows = []
    for sym in SYMBOLS:
        row = {"Symbol": sym}
        for case_label, ps in [("A", ps_a), ("C", ps_b)]:
            if sym in ps:
                d = ps[sym]
                eq = d["equity"]
                ret = (eq.iloc[-1] / d["initial_cash"] - 1) * 100
                r = eq.pct_change().dropna()
                sh = float(r.mean() / r.std() * np.sqrt(8760)) if len(r) > 1 and r.std() > 0 else 0
                row[f"{case_label}_Return%"] = round(ret, 1)
                row[f"{case_label}_Sharpe"] = round(sh, 3)
                row[f"{case_label}_Trades"] = d["total_trades"]
                row[f"{case_label}_TO"] = round(d["turnover"], 0)
                row[f"{case_label}_Fee$"] = round(d["total_fees_paid"], 0)
            else:
                for k in ["Return%", "Sharpe", "Trades", "TO", "Fee$"]:
                    row[f"{case_label}_{k}"] = "N/A"
        sym_rows.append(row)

    sym_df = pd.DataFrame(sym_rows)
    sym_df.to_csv(out / "per_symbol_breakdown.csv", index=False)
    print(f"  💾 per_symbol_breakdown.csv saved")

    # Per-symbol delta
    print(f"\n  Per-Symbol Comparison:")
    print(f"  {'Symbol':<10} {'A_Sh':>7} {'C_Sh':>7} {'ΔSh':>7} {'A_TO':>7} {'C_TO':>7} {'ΔTO':>7}")
    print(f"  {'─' * 52}")
    for _, r in sym_df.iterrows():
        try:
            d_sh = r["C_Sharpe"] - r["A_Sharpe"]
            d_to = r["C_TO"] - r["A_TO"]
            print(f"  {r['Symbol']:<10} {r['A_Sharpe']:>7.3f} {r['C_Sharpe']:>7.3f} {d_sh:>+7.3f} "
                  f"{r['A_TO']:>7.0f} {r['C_TO']:>7.0f} {d_to:>+7.0f}")
        except (TypeError, ValueError):
            print(f"  {r['Symbol']:<10} — data missing —")

    # ══════════════════════════════════════════════════════════════
    #  Equity curve chart
    # ══════════════════════════════════════════════════════════════
    if "equity" in port_a and "equity" in port_b:
        plot_equity_curves(port_a["equity"], port_b["equity"], out / "equity_curve_comparison.png")

    # ══════════════════════════════════════════════════════════════
    #  Summary
    # ══════════════════════════════════════════════════════════════
    summary_lines = [
        "=" * 72,
        "  Silver Bullet Validation — Summary (Case A vs Case C)",
        "=" * 72,
        "",
        f"  Config: {CONFIG_PATH}",
        f"  Period: {BT_START} → {BT_END}",
        f"  Symbols: {len(SYMBOLS)} ({', '.join(SYMBOLS)})",
        f"  Leverage: {cfg.futures.leverage if cfg.futures else 1}x, Weight sum: {sum(E3_WEIGHTS.values()):.2f}",
        "",
        "  ── Case A: E3 Baseline (ensemble + overlays, no EMA/gates) ──",
        f"    Sharpe:  {port_a.get('Sharpe', 'N/A')}",
        f"    CAGR:    {port_a.get('CAGR', 'N/A')}%",
        f"    MaxDD:   {port_a.get('MaxDD', 'N/A')}%",
        f"    Calmar:  {port_a.get('Calmar', 'N/A')}",
        f"    Turnover:{port_a.get('Turnover', 'N/A')}",
        f"    Fee%:    {port_a.get('Fee%', 'N/A')}%",
        "",
        "  ── Case C: E3 + rb=3% + fg=80% (Gates only, no EMA) ──",
        f"    Sharpe:  {port_b.get('Sharpe', 'N/A')}",
        f"    CAGR:    {port_b.get('CAGR', 'N/A')}%",
        f"    MaxDD:   {port_b.get('MaxDD', 'N/A')}%",
        f"    Calmar:  {port_b.get('Calmar', 'N/A')}",
        f"    Turnover:{port_b.get('Turnover', 'N/A')}",
        f"    Fee%:    {port_b.get('Fee%', 'N/A')}%",
        "",
        "  ── Delta (B - A) ──",
    ]

    delta_keys = ["Sharpe", "CAGR", "MaxDD", "Calmar", "Turnover", "Fee%"]
    for k in delta_keys:
        va = port_a.get(k, 0)
        vb = port_b.get(k, 0)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            summary_lines.append(f"    Δ{k}: {vb - va:+.3f}")

    summary_lines.extend([
        "",
        "  ── Verdict ──",
    ])

    sh_a = port_a.get("Sharpe", 0)
    sh_b = port_b.get("Sharpe", 0)
    d_sharpe = sh_b - sh_a
    to_a = port_a.get("Turnover", 0)
    to_b = port_b.get("Turnover", 0)
    fee_a = port_a.get("Fee%", 0)
    fee_b = port_b.get("Fee%", 0)

    if d_sharpe >= -0.1 and to_b < to_a and fee_b < fee_a:
        summary_lines.append(
            f"    ✅ Case C wins: Sharpe Δ={d_sharpe:+.3f} (negligible loss or gain), "
            f"Turnover ↓{to_b - to_a:.0f}, Fee% ↓{fee_b - fee_a:.2f}%"
        )
        summary_lines.append("    Gates-only 策略保留原始信號品質，同時有效降低 churn。")
        summary_lines.append("    建議注入 rb=3% + fg=80% 到 E3 config，進入 shadow 驗證。")
    elif d_sharpe > 0:
        summary_lines.append(
            f"    ✅ Case C improves Sharpe ({d_sharpe:+.3f}) with cost reduction."
        )
        summary_lines.append("    建議進入 shadow 驗證。")
    elif abs(d_sharpe) < 0.3:
        summary_lines.append(
            f"    ⚠ Case C: Sharpe Δ={d_sharpe:+.3f} (moderate loss), cost savings may justify."
        )
        summary_lines.append("    需衡量 Sharpe 損失 vs Fee/Turnover 節省是否值得。")
    else:
        summary_lines.append(
            f"    ❌ Case C underperforms: Sharpe Δ={d_sharpe:+.3f}."
        )
        summary_lines.append("    Gates-only 也無法解決問題，暫不建議注入。")

    summary_text = "\n".join(summary_lines)
    print(f"\n{summary_text}")

    with open(out / "summary.txt", "w") as f:
        f.write(summary_text)
    print(f"\n  💾 summary.txt saved")
    print(f"\n{'═' * 72}")
    print(f"  ✅ Silver Bullet Validation complete → {out}")
    print(f"{'═' * 72}")

    return out


if __name__ == "__main__":
    main()
