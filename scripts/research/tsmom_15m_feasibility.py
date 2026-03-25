#!/usr/bin/env python3
"""
TSMOM 15m — Feasibility EDA

Test if trend-following alpha survives at 15m timeframe with 4x cost.
Also tests a "native" faster parameterization (shorter lookbacks).

Task: research_20260325_163400_higher_freq_complement
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from qtrade.config import load_config
from qtrade.backtest.run_backtest import run_symbol_backtest, BacktestResult


CONFIGS = {
    "TSMOM_15m_rescaled": "config/research_tsmom_15m.yaml",
}
PROD_CFG = "config/prod_candidate_simplified.yaml"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def run_bt(cfg_path: str, symbol: str) -> BacktestResult | None:
    cfg = load_config(cfg_path)
    bt = cfg.to_backtest_dict(symbol=symbol)
    data_path = cfg.resolve_kline_path(symbol)
    try:
        return run_symbol_backtest(
            symbol, data_path, bt,
            strategy_name=cfg.strategy.name,
            data_dir=cfg.data_dir,
        )
    except Exception as e:
        print(f"  ERROR {symbol}: {e}")
        import traceback; traceback.print_exc()
        return None


def extract_stats(result: BacktestResult, symbol: str) -> dict:
    pf = result.pf
    stats = result.stats
    total_return = stats.get("Total Return [%]", 0)
    sharpe = stats.get("Sharpe Ratio", 0)
    max_dd = stats.get("Max Drawdown [%]", 0)
    total_trades = int(stats.get("Total Trades", 0))
    start = pf.wrapper.index[0]
    end = pf.wrapper.index[-1]
    years = (end - start).total_seconds() / (365.25 * 86400)
    trades_per_year = total_trades / years if years > 0 else 0
    win_rate = stats.get("Win Rate [%]", 0)

    return {
        "symbol": symbol,
        "total_return_pct": round(total_return, 2),
        "sharpe": round(sharpe, 3),
        "max_dd_pct": round(max_dd, 2),
        "total_trades": total_trades,
        "trades_per_year": round(trades_per_year, 1),
        "years": round(years, 2),
        "win_rate_pct": round(win_rate, 2),
    }


def yearly_sr(result: BacktestResult) -> dict:
    eq = result.equity()
    daily = eq.pct_change().dropna()
    daily.index = daily.index.normalize()
    daily = daily.groupby(daily.index).sum()
    df = pd.DataFrame({"ret": daily})
    df["year"] = df.index.year
    out = {}
    for yr, g in df.groupby("year"):
        r = g["ret"]
        sr = r.mean() / r.std() * np.sqrt(365) if r.std() > 0 else 0
        ret = ((1 + r).prod() - 1) * 100
        out[yr] = {"sr": round(sr, 3), "ret": round(ret, 2)}
    return out


def corr_with_prod(result: BacktestResult, symbol: str) -> float | None:
    try:
        prod_cfg = load_config(PROD_CFG)
        prod_data = prod_cfg.resolve_kline_path(symbol)
        prod_bt = prod_cfg.to_backtest_dict(symbol=symbol)
        prod_result = run_symbol_backtest(
            symbol, prod_data, prod_bt,
            strategy_name=prod_cfg.strategy.name,
            data_dir=prod_cfg.data_dir,
        )
        eq_mr = result.equity()
        eq_prod = prod_result.equity()
        d1 = eq_mr.pct_change().dropna()
        d2 = eq_prod.pct_change().dropna()
        d1.index = d1.index.normalize()
        d2.index = d2.index.normalize()
        d1 = d1.groupby(d1.index).sum()
        d2 = d2.groupby(d2.index).sum()
        common = d1.index.intersection(d2.index)
        if len(common) < 30:
            return None
        return round(d1.loc[common].corr(d2.loc[common]), 4)
    except Exception as e:
        print(f"  Corr error {symbol}: {e}")
        return None


def main():
    print("=" * 70)
    print("TSMOM 15m — FEASIBILITY EDA")
    print("=" * 70)

    for label, cfg_path in CONFIGS.items():
        print(f"\n{'━' * 70}")
        print(f"  CONFIG: {label}")
        print(f"  Path:   {cfg_path}")
        print(f"{'━' * 70}")

        all_stats = []

        for sym in SYMBOLS:
            print(f"\n  ── {sym} ──")
            result = run_bt(cfg_path, sym)
            if result is None:
                continue

            s = extract_stats(result, sym)
            all_stats.append(s)

            print(f"    Return:     {s['total_return_pct']:>10.2f}%")
            print(f"    Sharpe:     {s['sharpe']:>10.3f}")
            print(f"    MaxDD:      {s['max_dd_pct']:>10.2f}%")
            print(f"    Trades/yr:  {s['trades_per_year']:>10.1f}")
            print(f"    Win Rate:   {s['win_rate_pct']:>10.2f}%")

            yrs = yearly_sr(result)
            print(f"    Yearly: ", end="")
            for yr, d in sorted(yrs.items()):
                print(f"{yr}(SR={d['sr']}, R={d['ret']}%) ", end="")
            print()

            c = corr_with_prod(result, sym)
            if c is not None:
                print(f"    Corr(Prod): {c:.4f} {'✅' if abs(c) < 0.3 else '⚠️ HIGH'}")

        if all_stats:
            df = pd.DataFrame(all_stats)
            avg_sr = df["sharpe"].mean()
            avg_trades = df["trades_per_year"].mean()
            print(f"\n  Summary: avg SR={avg_sr:.3f}, avg trades/yr={avg_trades:.0f}")
            if avg_sr > 0.5:
                print(f"  → WEAK GO — proceed to parameter sweep + cost stress")
            elif avg_sr > 0:
                print(f"  → NEED_MORE_WORK — positive but marginal")
            else:
                print(f"  → FAIL — negative post-cost SR")


if __name__ == "__main__":
    main()
