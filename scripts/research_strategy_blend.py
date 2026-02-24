#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════
  Strategy Blend Optimizer — R3C (tsmom_ema) vs tsmom_carry_v2
═══════════════════════════════════════════════════════════════

Phase 1 of the meta_blend plan:
  1. Run backtests for both strategies on shared symbols
  2. Extract per-symbol daily return series
  3. Sweep allocation weights (0/100 → 100/0) in 5% steps
  4. Compute portfolio Sharpe / Return / MDD for each blend
  5. Find optimal weight via max-Sharpe
  6. Report strategy-level return correlation

Usage:
  cd /Users/dylanting/Documents/spot_bot/quant-binance-spot
  source .venv/bin/activate
  PYTHONPATH=src python scripts/research_strategy_blend.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── project imports ──
from qtrade.config import load_config
from qtrade.backtest.run_backtest import run_symbol_backtest, BacktestResult

import logging

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("blend_optimizer")
logger.setLevel(logging.INFO)


# ═══════════════════════════════════════════════════════════
#  Config Paths
# ═══════════════════════════════════════════════════════════

R3C_CONFIG = "config/prod_live_R3C_E3.yaml"
V2_CONFIG  = "config/research_tsmom_carry_v2.yaml"

# Shared 8 symbols (V2 universe — excludes LTC and XRP)
SHARED_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT",
]

DATA_DIR = Path("data/binance")


# ═══════════════════════════════════════════════════════════
#  Helper: run backtest and extract daily equity series
# ═══════════════════════════════════════════════════════════

def _run_and_extract(
    symbol: str,
    cfg_path: str,
    strategy_name: str | None = None,
    strategy_params: dict | None = None,
    disable_overlay: bool = False,
) -> tuple[BacktestResult | None, pd.Series | None]:
    """
    Run backtest for a single symbol and return (BacktestResult, daily_returns).
    Daily returns are resampled from hourly equity curve.
    """
    cfg_obj = load_config(cfg_path)
    bt_dict = cfg_obj.to_backtest_dict(symbol=symbol)

    # Override strategy if requested (for R3C ensemble routing)
    if strategy_name:
        bt_dict["strategy_name"] = strategy_name
    if strategy_params:
        bt_dict["strategy_params"] = strategy_params

    # Disable overlay for fair comparison
    if disable_overlay:
        bt_dict.pop("overlay", None)

    data_path = DATA_DIR / "futures" / cfg_obj.market.interval / f"{symbol}.parquet"
    if not data_path.exists():
        logger.warning(f"⚠️  Missing data: {data_path}")
        return None, None

    try:
        result = run_symbol_backtest(
            symbol=symbol,
            data_path=data_path,
            cfg=bt_dict,
            market_type="futures",
            direction="both",
            data_dir=DATA_DIR,
        )
        # Extract equity curve → daily returns
        eq = result.equity()
        if eq is None or eq.empty:
            return result, None
        daily_eq = eq.resample("1D").last().dropna()
        daily_ret = daily_eq.pct_change().dropna()
        return result, daily_ret
    except Exception as e:
        logger.error(f"❌ Backtest failed for {symbol}: {e}")
        return None, None


def _get_r3c_strategy_for_symbol(symbol: str, cfg_obj) -> tuple[str, dict]:
    """
    Replicate R3C ensemble routing: BTC→breakout_vol_atr, ETH→tsmom_multi_ema, others→tsmom_ema.
    Returns (strategy_name, params) WITHOUT the oi_vol overlay.
    """
    import yaml
    with open(R3C_CONFIG, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    ens = raw.get("ensemble", {})
    strategies = ens.get("strategies", {})

    if symbol in strategies:
        sym_cfg = strategies[symbol]
        return sym_cfg["name"], sym_cfg.get("params", {})

    # Default: tsmom_ema with base params
    base_params = raw.get("strategy", {}).get("params", {})
    return "tsmom_ema", base_params


# ═══════════════════════════════════════════════════════════
#  Step 1: Run backtests for both strategies
# ═══════════════════════════════════════════════════════════

def run_all_backtests() -> tuple[dict, dict, dict, dict]:
    """
    Returns:
        r3c_results: {symbol: BacktestResult}
        r3c_daily:   {symbol: daily_returns_series}
        v2_results:  {symbol: BacktestResult}
        v2_daily:    {symbol: daily_returns_series}
    """
    r3c_results, r3c_daily = {}, {}
    v2_results, v2_daily = {}, {}

    cfg_r3c = load_config(R3C_CONFIG)
    cfg_v2  = load_config(V2_CONFIG)

    for sym in SHARED_SYMBOLS:
        logger.info(f"═══ {sym} ═══")

        # ── R3C (with proper ensemble routing, WITHOUT overlay) ──
        strat_name, strat_params = _get_r3c_strategy_for_symbol(sym, cfg_r3c)
        logger.info(f"  R3C: {strat_name}")
        r3c_bt_dict = cfg_r3c.to_backtest_dict(symbol=sym)
        r3c_bt_dict["strategy_name"] = strat_name
        r3c_bt_dict["strategy_params"] = strat_params
        r3c_bt_dict.pop("overlay", None)  # disable overlay for fair compare

        data_path = DATA_DIR / "futures" / cfg_r3c.market.interval / f"{sym}.parquet"
        try:
            res_r3c = run_symbol_backtest(
                symbol=sym,
                data_path=data_path,
                cfg=r3c_bt_dict,
                market_type="futures",
                direction="both",
                data_dir=DATA_DIR,
            )
            eq = res_r3c.equity()
            daily_eq = eq.resample("1D").last().dropna()
            daily_ret = daily_eq.pct_change().dropna()
            r3c_results[sym] = res_r3c
            r3c_daily[sym] = daily_ret
            logger.info(f"    Sharpe={res_r3c.sharpe():.3f}  Ret={res_r3c.total_return_pct():.1f}%")
        except Exception as e:
            logger.error(f"  R3C failed: {e}")

        # ── V2 ──
        logger.info(f"  V2: tsmom_carry_v2")
        v2_bt_dict = cfg_v2.to_backtest_dict(symbol=sym)
        try:
            res_v2 = run_symbol_backtest(
                symbol=sym,
                data_path=data_path,
                cfg=v2_bt_dict,
                market_type="futures",
                direction="both",
                data_dir=DATA_DIR,
            )
            eq = res_v2.equity()
            daily_eq = eq.resample("1D").last().dropna()
            daily_ret = daily_eq.pct_change().dropna()
            v2_results[sym] = res_v2
            v2_daily[sym] = daily_ret
            logger.info(f"    Sharpe={res_v2.sharpe():.3f}  Ret={res_v2.total_return_pct():.1f}%")
        except Exception as e:
            logger.error(f"  V2 failed: {e}")

    return r3c_results, r3c_daily, v2_results, v2_daily


# ═══════════════════════════════════════════════════════════
#  Step 2: Build portfolio return series for each blend
# ═══════════════════════════════════════════════════════════

# V2 allocation weights (from config)
V2_WEIGHTS = {
    "BTCUSDT": 0.1500, "ETHUSDT": 0.1500,
    "SOLUSDT": 0.1167, "BNBUSDT": 0.1167,
    "DOGEUSDT": 0.1167, "ADAUSDT": 0.1167,
    "AVAXUSDT": 0.1166, "LINKUSDT": 0.1166,
}


def build_portfolio_returns(
    daily_returns: dict[str, pd.Series],
    weights: dict[str, float] | None = None,
) -> pd.Series:
    """
    Build an equal-weight or custom-weight portfolio return series.
    All symbol returns are aligned to a common date index.
    """
    if not daily_returns:
        return pd.Series(dtype=float)

    # Align all to common index
    df = pd.DataFrame(daily_returns)
    df = df.dropna(how="all")

    if weights is None:
        # Equal weight
        w = 1.0 / len(df.columns)
        port_ret = df.mean(axis=1)
    else:
        # Normalize weights to sum=1 for the available symbols
        avail = [s for s in df.columns if s in weights]
        w_vals = np.array([weights[s] for s in avail])
        w_vals = w_vals / w_vals.sum()
        port_ret = (df[avail] * w_vals).sum(axis=1)

    return port_ret


def compute_metrics(daily_returns: pd.Series) -> dict:
    """Compute Sharpe, Total Return, Max Drawdown, Calmar from daily return series."""
    if daily_returns.empty or daily_returns.std() == 0:
        return {"sharpe": 0.0, "total_ret_pct": 0.0, "mdd_pct": 0.0, "calmar": 0.0}

    sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(365)
    cum = (1 + daily_returns).cumprod()
    total_ret = (cum.iloc[-1] - 1) * 100
    running_max = cum.cummax()
    dd = (cum - running_max) / running_max
    mdd = dd.min() * 100  # negative
    calmar = (total_ret / 100) / abs(mdd / 100) if mdd != 0 else 0.0

    return {
        "sharpe": sharpe,
        "total_ret_pct": total_ret,
        "mdd_pct": mdd,
        "calmar": calmar,
    }


# ═══════════════════════════════════════════════════════════
#  Step 3: Sweep blend weights and find optimal
# ═══════════════════════════════════════════════════════════

def sweep_blend_weights(
    r3c_daily: dict[str, pd.Series],
    v2_daily: dict[str, pd.Series],
    step: float = 0.05,
) -> pd.DataFrame:
    """
    Sweep w_r3c from 0.0 to 1.0, w_v2 = 1 - w_r3c.
    For each blend, compute the blended per-symbol daily returns, then portfolio metrics.

    Returns a DataFrame with columns: w_r3c, w_v2, sharpe, total_ret_pct, mdd_pct, calmar
    """
    common_syms = sorted(set(r3c_daily.keys()) & set(v2_daily.keys()))
    if not common_syms:
        logger.error("No common symbols between R3C and V2!")
        return pd.DataFrame()

    logger.info(f"Common symbols for blend: {common_syms}")

    results = []
    weights_range = np.arange(0.0, 1.0 + step / 2, step)

    for w_r3c in weights_range:
        w_v2 = 1.0 - w_r3c

        # Blend per-symbol returns, then form portfolio
        blended_daily = {}
        for sym in common_syms:
            r3c_ret = r3c_daily[sym]
            v2_ret = v2_daily[sym]
            # Align indices
            common_idx = r3c_ret.index.intersection(v2_ret.index)
            blended = w_r3c * r3c_ret.loc[common_idx] + w_v2 * v2_ret.loc[common_idx]
            blended_daily[sym] = blended

        port_ret = build_portfolio_returns(blended_daily, weights=V2_WEIGHTS)
        m = compute_metrics(port_ret)
        m["w_r3c"] = round(w_r3c * 100)
        m["w_v2"] = round(w_v2 * 100)
        results.append(m)

    df = pd.DataFrame(results)
    df = df[["w_r3c", "w_v2", "sharpe", "total_ret_pct", "mdd_pct", "calmar"]]
    return df


# ═══════════════════════════════════════════════════════════
#  Step 4: Report
# ═══════════════════════════════════════════════════════════

def compute_correlation(
    r3c_daily: dict[str, pd.Series],
    v2_daily: dict[str, pd.Series],
) -> pd.DataFrame:
    """Compute per-symbol daily return correlation between R3C and V2."""
    rows = []
    for sym in sorted(set(r3c_daily.keys()) & set(v2_daily.keys())):
        common_idx = r3c_daily[sym].index.intersection(v2_daily[sym].index)
        if len(common_idx) > 30:
            corr = r3c_daily[sym].loc[common_idx].corr(v2_daily[sym].loc[common_idx])
        else:
            corr = float("nan")
        rows.append({"symbol": sym, "corr": corr})
    return pd.DataFrame(rows)


def print_report(
    r3c_results: dict,
    v2_results: dict,
    sweep_df: pd.DataFrame,
    corr_df: pd.DataFrame,
):
    """Pretty print the full blend analysis report."""
    print("\n" + "═" * 70)
    print("  STRATEGY BLEND OPTIMIZER — R3C vs tsmom_carry_v2")
    print("═" * 70)

    # ── Per-symbol comparison ──
    print("\n┌─────────────────────────────────────────────────────────┐")
    print("│  Per-Symbol Backtest Results (NO overlay, 1x capital)  │")
    print("├──────────┬───────────────────┬───────────────────┬──────┤")
    print("│ Symbol   │  R3C Sharpe  Ret% │  V2  Sharpe  Ret% │ Corr │")
    print("├──────────┼───────────────────┼───────────────────┼──────┤")

    for sym in SHARED_SYMBOLS:
        r3c_sh = r3c_results[sym].sharpe() if sym in r3c_results else 0.0
        r3c_rt = r3c_results[sym].total_return_pct() if sym in r3c_results else 0.0
        v2_sh  = v2_results[sym].sharpe() if sym in v2_results else 0.0
        v2_rt  = v2_results[sym].total_return_pct() if sym in v2_results else 0.0
        corr_row = corr_df[corr_df["symbol"] == sym]
        corr_val = corr_row["corr"].values[0] if len(corr_row) > 0 else float("nan")
        better = "←V2" if v2_sh > r3c_sh else "←R3C"
        print(f"│ {sym:8s} │  {r3c_sh:+6.3f} {r3c_rt:+7.1f}% │  {v2_sh:+6.3f} {v2_rt:+7.1f}% │ {corr_val:+.2f} │ {better}")

    print("└──────────┴───────────────────┴───────────────────┴──────┘")

    avg_corr = corr_df["corr"].mean()
    print(f"\n  Average daily return correlation: {avg_corr:+.3f}")
    print(f"  (Lower = more diversification benefit from blending)")

    # ── Blend sweep ──
    print("\n┌──────────────────────────────────────────────────────────────┐")
    print("│  Blend Weight Sweep (R3C% / V2%)                           │")
    print("├───────┬───────┬────────┬──────────┬────────┬───────────────┤")
    print("│ R3C%  │  V2%  │ Sharpe │  Return  │  MDD   │   Calmar      │")
    print("├───────┼───────┼────────┼──────────┼────────┼───────────────┤")

    for _, row in sweep_df.iterrows():
        marker = ""
        if row["sharpe"] == sweep_df["sharpe"].max():
            marker = " ★ MAX SHARPE"
        print(
            f"│ {row['w_r3c']:5.0f} │ {row['w_v2']:5.0f} │ {row['sharpe']:+6.3f} │"
            f" {row['total_ret_pct']:+7.1f}% │ {row['mdd_pct']:+6.1f}% │"
            f" {row['calmar']:+6.2f}         │{marker}"
        )

    print("└───────┴───────┴────────┴──────────┴────────┴───────────────┘")

    # ── Optimal ──
    best = sweep_df.loc[sweep_df["sharpe"].idxmax()]
    print(f"\n  ★ Optimal Blend: R3C {best['w_r3c']:.0f}% / V2 {best['w_v2']:.0f}%")
    print(f"    Sharpe = {best['sharpe']:+.3f}")
    print(f"    Return = {best['total_ret_pct']:+.1f}%")
    print(f"    MDD    = {best['mdd_pct']:+.1f}%")
    print(f"    Calmar = {best['calmar']:+.2f}")

    # ── Pure strategy comparison ──
    pure_r3c = sweep_df[sweep_df["w_r3c"] == 100].iloc[0]
    pure_v2  = sweep_df[sweep_df["w_v2"] == 100].iloc[0]
    print(f"\n  Comparison:")
    print(f"    Pure R3C:  Sharpe={pure_r3c['sharpe']:+.3f}  Ret={pure_r3c['total_ret_pct']:+.1f}%  MDD={pure_r3c['mdd_pct']:+.1f}%")
    print(f"    Pure V2:   Sharpe={pure_v2['sharpe']:+.3f}  Ret={pure_v2['total_ret_pct']:+.1f}%  MDD={pure_v2['mdd_pct']:+.1f}%")
    print(f"    Blend:     Sharpe={best['sharpe']:+.3f}  Ret={best['total_ret_pct']:+.1f}%  MDD={best['mdd_pct']:+.1f}%")

    sharpe_improve = best['sharpe'] - max(pure_r3c['sharpe'], pure_v2['sharpe'])
    if sharpe_improve > 0:
        print(f"\n  ✅ Blend improves Sharpe by {sharpe_improve:+.3f} over the best pure strategy")
    else:
        print(f"\n  ⚠️  Blend does NOT improve over the best pure strategy ({sharpe_improve:+.3f})")

    return best


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def main():
    print("🚀 Starting Strategy Blend Optimizer...")
    print(f"   R3C Config: {R3C_CONFIG}")
    print(f"   V2  Config: {V2_CONFIG}")
    print(f"   Symbols:    {SHARED_SYMBOLS}")
    print()

    # Step 1: Run backtests
    r3c_results, r3c_daily, v2_results, v2_daily = run_all_backtests()

    if not r3c_daily or not v2_daily:
        print("❌ No backtest results to compare!")
        sys.exit(1)

    # Step 2: Compute correlations
    corr_df = compute_correlation(r3c_daily, v2_daily)

    # Step 3: Sweep blend weights
    sweep_df = sweep_blend_weights(r3c_daily, v2_daily, step=0.05)
    if sweep_df.empty:
        print("❌ Blend sweep failed!")
        sys.exit(1)

    # Step 4: Report
    best = print_report(r3c_results, v2_results, sweep_df, corr_df)

    # Output optimal for next phase
    print("\n" + "═" * 70)
    print("  NEXT STEP: Use these weights in meta_blend strategy")
    print(f"  → w_r3c = {best['w_r3c']/100:.2f}")
    print(f"  → w_v2  = {best['w_v2']/100:.2f}")
    print("═" * 70)


if __name__ == "__main__":
    main()
