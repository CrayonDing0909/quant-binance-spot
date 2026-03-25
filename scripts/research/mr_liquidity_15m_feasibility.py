#!/usr/bin/env python3
"""
MR Liquidity Sweep 15m — Feasibility EDA

Evaluates whether the mean_revert_liquidity strategy at 15m has:
1. Positive gross expectancy per trade (MR Iron Rule)
2. Sufficient trade frequency for fast feedback
3. Survivable cost profile
4. Low correlation with production TSMOM

Task: research_20260325_163400_higher_freq_complement
"""
from __future__ import annotations

import sys
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from qtrade.config import load_config
from qtrade.backtest.run_backtest import run_symbol_backtest, BacktestResult
from qtrade.data.storage import load_klines
from qtrade.strategy.base import StrategyContext

RESEARCH_CFG = "config/research_mr_liquidity_15m.yaml"
PROD_CFG = "config/prod_candidate_simplified.yaml"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def run_single_backtest(cfg_path: str, symbol: str) -> BacktestResult | None:
    cfg = load_config(cfg_path)
    bt = cfg.to_backtest_dict(symbol=symbol)
    strategy_name = cfg.strategy.name
    data_path = cfg.resolve_kline_path(symbol)
    try:
        result = run_symbol_backtest(
            symbol, data_path, bt,
            strategy_name=strategy_name,
            data_dir=cfg.data_dir,
        )
        return result
    except Exception as e:
        print(f"  ERROR {symbol}: {e}")
        import traceback; traceback.print_exc()
        return None


def analyze_trades(result: BacktestResult, symbol: str) -> dict:
    """Extract trade-level statistics from backtest result."""
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
    avg_win = stats.get("Avg Winning Trade [%]", 0)
    avg_loss = stats.get("Avg Losing Trade [%]", 0)

    gross_expectancy = (win_rate / 100 * avg_win / 100) - ((1 - win_rate / 100) * abs(avg_loss) / 100)

    return {
        "symbol": symbol,
        "total_return_pct": round(total_return, 2),
        "sharpe": round(sharpe, 3),
        "max_dd_pct": round(max_dd, 2),
        "total_trades": total_trades,
        "trades_per_year": round(trades_per_year, 1),
        "years": round(years, 2),
        "win_rate_pct": round(win_rate, 2),
        "avg_win_pct": round(avg_win, 3),
        "avg_loss_pct": round(avg_loss, 3),
        "gross_expectancy": round(gross_expectancy, 6),
        "calmar": round(abs(total_return / max_dd) if max_dd != 0 else 0, 2),
    }


def compute_correlation_with_prod(mr_result: BacktestResult, symbol: str) -> float | None:
    """Compute daily return correlation between MR and production TSMOM."""
    try:
        prod_cfg = load_config(PROD_CFG)
        prod_bt = prod_cfg.to_backtest_dict(symbol=symbol)
        prod_data_path = prod_cfg.resolve_kline_path(symbol)
        prod_result = run_symbol_backtest(
            symbol, prod_data_path, prod_bt,
            strategy_name=prod_cfg.strategy.name,
            data_dir=prod_cfg.data_dir,
        )
        if prod_result is None:
            return None

        mr_eq = mr_result.equity()
        prod_eq = prod_result.equity()
        mr_daily = mr_eq.pct_change().dropna()
        prod_daily = prod_eq.pct_change().dropna()
        mr_daily.index = mr_daily.index.normalize()
        prod_daily.index = prod_daily.index.normalize()
        mr_daily = mr_daily.groupby(mr_daily.index).sum()
        prod_daily = prod_daily.groupby(prod_daily.index).sum()

        common_idx = mr_daily.index.intersection(prod_daily.index)
        if len(common_idx) < 30:
            return None

        corr = mr_daily.loc[common_idx].corr(prod_daily.loc[common_idx])
        return round(corr, 4)
    except Exception as e:
        print(f"  Correlation computation failed for {symbol}: {e}")
        return None


def run_yearly_breakdown(result: BacktestResult, symbol: str) -> pd.DataFrame:
    """Yearly PnL decomposition."""
    eq = result.equity()
    daily = eq.pct_change().dropna()
    daily.index = daily.index.normalize()
    daily = daily.groupby(daily.index).sum()
    daily_df = pd.DataFrame({"ret": daily})
    daily_df["year"] = daily_df.index.year

    yearly = daily_df.groupby("year").agg(
        total_return=("ret", lambda x: (1 + x).prod() - 1),
        sharpe=("ret", lambda x: x.mean() / x.std() * np.sqrt(365) if x.std() > 0 else 0),
        max_dd=("ret", lambda x: (
            (1 + x).cumprod().div((1 + x).cumprod().cummax()) - 1
        ).min()),
        trading_days=("ret", "count"),
    )
    yearly["total_return_pct"] = (yearly["total_return"] * 100).round(2)
    yearly["sharpe"] = yearly["sharpe"].round(3)
    yearly["max_dd_pct"] = (yearly["max_dd"] * 100).round(2)
    return yearly


def main():
    print("=" * 70)
    print("MR LIQUIDITY SWEEP 15m — FEASIBILITY EDA")
    print("=" * 70)
    print(f"Config: {RESEARCH_CFG}")
    print(f"Symbols: {SYMBOLS}")
    print()

    all_stats = []

    for sym in SYMBOLS:
        print(f"\n{'─' * 50}")
        print(f"  {sym}")
        print(f"{'─' * 50}")

        result = run_single_backtest(RESEARCH_CFG, sym)
        if result is None:
            continue

        stats = analyze_trades(result, sym)
        all_stats.append(stats)

        print(f"  Total Return:    {stats['total_return_pct']:>10.2f}%")
        print(f"  Sharpe Ratio:    {stats['sharpe']:>10.3f}")
        print(f"  Max Drawdown:    {stats['max_dd_pct']:>10.2f}%")
        print(f"  Calmar:          {stats['calmar']:>10.2f}")
        print(f"  Total Trades:    {stats['total_trades']:>10d}")
        print(f"  Trades/Year:     {stats['trades_per_year']:>10.1f}")
        print(f"  Win Rate:        {stats['win_rate_pct']:>10.2f}%")
        print(f"  Avg Win:         {stats['avg_win_pct']:>10.3f}%")
        print(f"  Avg Loss:        {stats['avg_loss_pct']:>10.3f}%")
        print(f"  Gross Expect:    {stats['gross_expectancy']:>10.6f}")

        # MR Iron Rule check
        if stats["gross_expectancy"] > 0:
            print(f"  ✅ MR Iron Rule: PASS (gross expectancy > 0)")
        else:
            print(f"  ❌ MR Iron Rule: FAIL (gross expectancy <= 0)")

        # Yearly breakdown
        yearly = run_yearly_breakdown(result, sym)
        print(f"\n  Yearly Breakdown:")
        print(f"  {'Year':>6} {'Return%':>10} {'Sharpe':>8} {'MaxDD%':>8} {'Days':>6}")
        for yr, row in yearly.iterrows():
            print(f"  {yr:>6} {row['total_return_pct']:>10.2f} {row['sharpe']:>8.3f} {row['max_dd_pct']:>8.2f} {int(row['trading_days']):>6}")

        # Correlation with production
        corr = compute_correlation_with_prod(result, sym)
        if corr is not None:
            print(f"\n  Corr with Prod TSMOM: {corr:.4f}")
            if abs(corr) < 0.3:
                print(f"  ✅ Diversification: PASS (|corr| < 0.3)")
            else:
                print(f"  ⚠️ Diversification: WARN (|corr| >= 0.3)")

    # Portfolio summary
    if all_stats:
        print(f"\n{'=' * 70}")
        print("PORTFOLIO SUMMARY")
        print(f"{'=' * 70}")
        df = pd.DataFrame(all_stats)
        print(f"\n{'Symbol':>10} {'Return%':>10} {'SR':>8} {'MDD%':>8} {'Trades/yr':>10} {'WinR%':>8} {'GrossExp':>10}")
        for _, row in df.iterrows():
            print(f"{row['symbol']:>10} {row['total_return_pct']:>10.2f} {row['sharpe']:>8.3f} {row['max_dd_pct']:>8.2f} {row['trades_per_year']:>10.1f} {row['win_rate_pct']:>8.2f} {row['gross_expectancy']:>10.6f}")

        avg_sr = df["sharpe"].mean()
        avg_trades = df["trades_per_year"].mean()
        avg_gross = df["gross_expectancy"].mean()
        all_positive_gross = all(s["gross_expectancy"] > 0 for s in all_stats)
        all_positive_return = all(s["total_return_pct"] > 0 for s in all_stats)

        print(f"\n  Avg Sharpe:         {avg_sr:.3f}")
        print(f"  Avg Trades/Year:    {avg_trades:.1f}")
        print(f"  Avg Gross Expect:   {avg_gross:.6f}")
        print(f"  All Gross Pos?      {'YES ✅' if all_positive_gross else 'NO ❌'}")
        print(f"  All Returns Pos?    {'YES ✅' if all_positive_return else 'NO ❌'}")

        # Final verdict
        print(f"\n{'=' * 70}")
        print("FEASIBILITY VERDICT")
        print(f"{'=' * 70}")

        passes = 0
        checks = 0

        checks += 1
        if all_positive_gross:
            print("  ✅ MR Iron Rule: All symbols have positive gross expectancy")
            passes += 1
        else:
            print("  ❌ MR Iron Rule: Some symbols have negative gross expectancy → FAIL")

        checks += 1
        if avg_trades > 100:
            print(f"  ✅ Trade Frequency: {avg_trades:.0f} trades/yr (>100 required)")
            passes += 1
        else:
            print(f"  ❌ Trade Frequency: {avg_trades:.0f} trades/yr (<100, insufficient feedback)")

        checks += 1
        if avg_sr > 0.3:
            print(f"  ✅ Post-Cost SR: {avg_sr:.3f} (>0.3 minimum)")
            passes += 1
        else:
            print(f"  ❌ Post-Cost SR: {avg_sr:.3f} (<0.3, cost kills alpha)")

        checks += 1
        if avg_sr > 0:
            print(f"  ✅ Net Positive: avg SR > 0 after costs")
            passes += 1
        else:
            print(f"  ❌ Net Negative: avg SR <= 0 after costs")

        print(f"\n  Result: {passes}/{checks} checks passed")
        if passes >= 3:
            print("  → WEAK GO / GO — proceed to full validation")
        elif passes >= 2:
            print("  → NEED_MORE_WORK — promising but parameters need tuning")
        else:
            print("  → FAIL — direction not viable at 15m with current parameters")


if __name__ == "__main__":
    main()
