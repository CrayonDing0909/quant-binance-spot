#!/usr/bin/env python3
"""
Mean Revert Liquidity Sweep — Research Backtest
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

回測「Liquidity Sweep 均值回歸策略」在 15m BTC/ETH/SOL 上的表現。

實驗矩陣：
  A) Baseline:    key=96, RSI(14), TP=1.5%, SL=1.0%, Hold=12
  B) 寬 SL:       SL=1.5%
  C) 嚴格 RSI:    RSI ob=75/os=25
  D) 短持倉:      Hold=8 bars (2h)
  E) 長持倉:      Hold=24 bars (6h)
  F) 大 TP:       TP=2.0%, SL=1.5%
  G) 快 RSI:      RSI(7), ob=65/os=35
  H) 48h Key:     key=192 bars

時間分段：
  - Full:   全部可用資料
  - IS:     2023-01-01 ~ 2024-06-30
  - OOS:    2024-07-01 ~ 2025-06-30
  - Recent: 2025-07-01 ~ 2026-02-28

輸出：
  reports/research/mean_revert_liquidity/<timestamp>/
    ├── performance_matrix.csv
    ├── per_symbol_detail.csv
    ├── trade_analysis.csv
    └── summary.txt

Usage:
  cd /Users/dylanting/Documents/spot_bot/quant-binance-spot
  source .venv/bin/activate
  PYTHONPATH=src python scripts/research_mean_revert_liquidity.py
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

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import vectorbt as vbt

from qtrade.strategy import get_strategy
from qtrade.strategy.base import StrategyContext
from qtrade.data.storage import load_klines
from qtrade.data.quality import clean_data

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ══════════════════════════════════════════════════════════════
#  Constants
# ══════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "binance" / "futures" / "15m"
OUT_ROOT = PROJECT_ROOT / "reports" / "research" / "mean_revert_liquidity"

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

INTERVALS = {
    "15m": {"freq": "15m", "bars_per_year": 35_040},
}

# Time segments
SEGMENTS = {
    "full":   (None, None),
    "IS":     ("2023-01-01", "2024-06-30"),
    "OOS":    ("2024-07-01", "2025-06-30"),
    "recent": ("2025-07-01", "2026-02-28"),
}

# Fee/slippage for 15m: slightly higher slippage due to more frequent trading
FEE_BPS = 5       # 5 bps maker/taker average
SLIPPAGE_BPS = 3   # 3 bps slippage
INITIAL_CASH = 100_000

# ══════════════════════════════════════════════════════════════
#  Experiment Configs
# ══════════════════════════════════════════════════════════════

CONFIGS = {
    "A_baseline": {
        "key_level_bars": 96,
        "rsi_period": 14,
        "rsi_overbought": 70,
        "rsi_oversold": 30,
        "tp_pct": 0.015,
        "sl_pct": 0.010,
        "max_hold_bars": 12,
        "cooldown_bars": 4,
    },
    "B_wide_sl": {
        "key_level_bars": 96,
        "rsi_period": 14,
        "rsi_overbought": 70,
        "rsi_oversold": 30,
        "tp_pct": 0.015,
        "sl_pct": 0.015,
        "max_hold_bars": 12,
        "cooldown_bars": 4,
    },
    "C_strict_rsi": {
        "key_level_bars": 96,
        "rsi_period": 14,
        "rsi_overbought": 75,
        "rsi_oversold": 25,
        "tp_pct": 0.015,
        "sl_pct": 0.010,
        "max_hold_bars": 12,
        "cooldown_bars": 4,
    },
    "D_short_hold": {
        "key_level_bars": 96,
        "rsi_period": 14,
        "rsi_overbought": 70,
        "rsi_oversold": 30,
        "tp_pct": 0.015,
        "sl_pct": 0.010,
        "max_hold_bars": 8,
        "cooldown_bars": 4,
    },
    "E_long_hold": {
        "key_level_bars": 96,
        "rsi_period": 14,
        "rsi_overbought": 70,
        "rsi_oversold": 30,
        "tp_pct": 0.015,
        "sl_pct": 0.010,
        "max_hold_bars": 24,
        "cooldown_bars": 4,
    },
    "F_big_tp": {
        "key_level_bars": 96,
        "rsi_period": 14,
        "rsi_overbought": 70,
        "rsi_oversold": 30,
        "tp_pct": 0.020,
        "sl_pct": 0.015,
        "max_hold_bars": 12,
        "cooldown_bars": 4,
    },
    "G_fast_rsi": {
        "key_level_bars": 96,
        "rsi_period": 7,
        "rsi_overbought": 65,
        "rsi_oversold": 35,
        "tp_pct": 0.015,
        "sl_pct": 0.010,
        "max_hold_bars": 12,
        "cooldown_bars": 4,
    },
    "H_48h_key": {
        "key_level_bars": 192,
        "rsi_period": 14,
        "rsi_overbought": 70,
        "rsi_oversold": 30,
        "tp_pct": 0.015,
        "sl_pct": 0.010,
        "max_hold_bars": 12,
        "cooldown_bars": 4,
    },
}


# ══════════════════════════════════════════════════════════════
#  Backtest engine
# ══════════════════════════════════════════════════════════════

def run_single(
    symbol: str,
    df: pd.DataFrame,
    params: dict,
    start: str | None,
    end: str | None,
) -> dict | None:
    """Run single symbol/config/segment backtest."""

    # Date filter
    df_seg = df.copy()
    if start:
        df_seg = df_seg.loc[start:]
    if end:
        df_seg = df_seg.loc[:end]

    if len(df_seg) < 200:
        return None

    # Strategy context (signal_delay=1 for next-bar execution)
    ctx = StrategyContext(
        symbol=symbol,
        interval="15m",
        market_type="futures",
        direction="both",
        signal_delay=1,
    )

    strategy_func = get_strategy("mean_revert_liquidity")
    pos = strategy_func(df_seg, ctx, params)

    # Extract exec_prices if available
    exec_prices_series = pos.attrs.get("exit_exec_prices", None)

    # Build custom price array for VBT:
    # - Default: use open (next-bar execution)
    # - When SL/TP triggers: use trigger price
    open_ = df_seg["open"]
    if exec_prices_series is not None:
        price = open_.copy()
        mask = exec_prices_series.notna()
        price[mask] = exec_prices_series[mask]
    else:
        price = open_

    fee = FEE_BPS / 10_000
    slippage = SLIPPAGE_BPS / 10_000

    try:
        pf = vbt.Portfolio.from_orders(
            close=df_seg["close"],
            size=pos,
            size_type="targetpercent",
            price=price,
            fees=fee,
            slippage=slippage,
            init_cash=INITIAL_CASH,
            freq="15min",
            direction="both",
        )
    except Exception as e:
        logger.warning(f"  ⚠ {symbol}: VBT error — {e}")
        return None

    stats = pf.stats()
    equity = pf.value()

    # ── Core metrics ──
    total_trades = int(stats.get("Total Trades", 0))
    if total_trades == 0:
        return None

    total_return = float(equity.iloc[-1] / INITIAL_CASH - 1)
    n_bars = len(df_seg)
    years = max(0.01, n_bars / 35_040)
    cagr = (1 + total_return) ** (1 / years) - 1

    r = equity.pct_change().dropna()
    sharpe = float(r.mean() / r.std() * np.sqrt(35_040)) if len(r) > 1 and r.std() > 0 else 0
    dn = r[r < 0]
    sortino = float(r.mean() / dn.std() * np.sqrt(35_040)) if len(dn) > 1 and dn.std() > 0 else 0

    dd = (equity - equity.cummax()) / equity.cummax()
    mdd = abs(float(dd.min()))
    calmar = cagr / mdd if mdd > 0.001 else 0

    win_rate = float(stats.get("Win Rate [%]", 0))
    pf_stat = float(stats.get("Profit Factor", 0))

    # Trade analysis
    trades = pf.trades.records_readable if hasattr(pf.trades, "records_readable") else None
    avg_hold_bars = 0
    avg_pnl_pct = 0
    if trades is not None and len(trades) > 0:
        if "Duration" in trades.columns:
            durations = pd.to_timedelta(trades["Duration"])
            avg_hold_bars = float(durations.mean().total_seconds() / 900)  # 900s = 15min
        if "PnL" in trades.columns:
            avg_pnl_pct = float(trades["PnL"].mean() / INITIAL_CASH * 100)

    total_fees = float(stats.get("Total Fees Paid", 0))
    fee_pct = total_fees / INITIAL_CASH * 100

    ann_trades = total_trades / years

    return {
        "CAGR%": round(cagr * 100, 2),
        "Sharpe": round(sharpe, 3),
        "Sortino": round(sortino, 3),
        "MaxDD%": round(mdd * 100, 2),
        "Calmar": round(calmar, 3),
        "WinRate%": round(win_rate, 1),
        "PF": round(pf_stat, 3),
        "Trades": total_trades,
        "Ann.Trades": round(ann_trades, 0),
        "AvgHold(bars)": round(avg_hold_bars, 1),
        "AvgPnL%": round(avg_pnl_pct, 4),
        "Fee%": round(fee_pct, 2),
        "TotalReturn%": round(total_return * 100, 2),
    }


# ══════════════════════════════════════════════════════════════
#  Aggregate across symbols
# ══════════════════════════════════════════════════════════════

def aggregate_results(rows: list[dict]) -> dict:
    """Equal-weight average of per-symbol metrics."""
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    numeric = df.select_dtypes(include=[np.number])
    agg = numeric.mean().to_dict()
    agg["N_Symbols"] = len(rows)
    # Round
    for k in agg:
        if isinstance(agg[k], float):
            agg[k] = round(agg[k], 3)
    return agg


# ══════════════════════════════════════════════════════════════
#  Trade-level analysis (for best config)
# ══════════════════════════════════════════════════════════════

def trade_analysis(
    symbol: str,
    df: pd.DataFrame,
    params: dict,
) -> pd.DataFrame | None:
    """Extract detailed trade records for analysis."""
    ctx = StrategyContext(
        symbol=symbol, interval="15m",
        market_type="futures", direction="both", signal_delay=1,
    )
    strategy_func = get_strategy("mean_revert_liquidity")
    pos = strategy_func(df, ctx, params)

    exec_prices_series = pos.attrs.get("exit_exec_prices", None)
    open_ = df["open"]
    if exec_prices_series is not None:
        price = open_.copy()
        mask = exec_prices_series.notna()
        price[mask] = exec_prices_series[mask]
    else:
        price = open_

    fee = FEE_BPS / 10_000
    slippage = SLIPPAGE_BPS / 10_000

    try:
        pf = vbt.Portfolio.from_orders(
            close=df["close"], size=pos, size_type="targetpercent",
            price=price, fees=fee, slippage=slippage,
            init_cash=INITIAL_CASH, freq="15min", direction="both",
        )
    except Exception:
        return None

    if not hasattr(pf.trades, "records_readable"):
        return None

    trades = pf.trades.records_readable.copy()
    if len(trades) == 0:
        return None

    trades["Symbol"] = symbol
    return trades


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUT_ROOT / ts
    out.mkdir(parents=True, exist_ok=True)

    print(f"{'═' * 72}")
    print(f"  📊 Mean Revert Liquidity Sweep — Research Backtest")
    print(f"  Output:  {out}")
    print(f"  Symbols: {SYMBOLS}")
    print(f"  Configs: {len(CONFIGS)}")
    print(f"  Segments: {list(SEGMENTS.keys())}")
    print(f"  Fee: {FEE_BPS}bps, Slip: {SLIPPAGE_BPS}bps")
    print(f"{'═' * 72}\n")

    # ── Load 15m data ──
    data = {}
    for sym in SYMBOLS:
        path = DATA_DIR / f"{sym}.parquet"
        if not path.exists():
            print(f"  ⚠ {sym}: 15m data not found at {path}")
            continue
        df = load_klines(path)
        df = clean_data(df, fill_method="forward", remove_outliers=False,
                        remove_duplicates=True)
        data[sym] = df
        print(f"  ✓ {sym}: {len(df)} bars, {df.index[0]} → {df.index[-1]}")

    if not data:
        print("  ❌ No data loaded. Exiting.")
        return

    print()

    # ── Run matrix ──
    all_rows = []
    per_symbol_rows = []

    total = len(CONFIGS) * len(SEGMENTS)
    done = 0

    for config_name, params in CONFIGS.items():
        for seg_name, (seg_start, seg_end) in SEGMENTS.items():
            done += 1
            label = f"[{done}/{total}] {config_name} / {seg_name}"

            sym_results = []
            for sym in SYMBOLS:
                if sym not in data:
                    continue
                res = run_single(sym, data[sym], params, seg_start, seg_end)
                if res is not None:
                    sym_results.append(res)
                    per_symbol_rows.append({
                        "Config": config_name,
                        "Segment": seg_name,
                        "Symbol": sym,
                        **res,
                    })

            # Aggregate
            agg = aggregate_results(sym_results)
            if agg:
                all_rows.append({
                    "Config": config_name,
                    "Segment": seg_name,
                    **agg,
                })
                sh = agg.get("Sharpe", 0)
                wr = agg.get("WinRate%", 0)
                tr = agg.get("Trades", 0)
                print(f"  {label:50s} Sh={sh:+6.3f}  WR={wr:5.1f}%  Trades={tr:>5}")
            else:
                print(f"  {label:50s} — no trades —")

            gc.collect()

    # ══════════════════════════════════════════════════════════════
    #  Output: Performance Matrix
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'═' * 72}")
    print(f"  📋 Results")
    print(f"{'═' * 72}\n")

    perf_df = pd.DataFrame(all_rows)
    if len(perf_df) == 0:
        print("  ❌ No results generated. Check data and parameters.")
        return

    perf_df.to_csv(out / "performance_matrix.csv", index=False)
    print(f"  💾 performance_matrix.csv ({len(perf_df)} rows)")

    # Per symbol
    sym_df = pd.DataFrame(per_symbol_rows)
    sym_df.to_csv(out / "per_symbol_detail.csv", index=False)
    print(f"  💾 per_symbol_detail.csv ({len(sym_df)} rows)")

    # ── Ranking table (full segment) ──
    full_df = perf_df[perf_df["Segment"] == "full"].copy()
    if len(full_df) > 0:
        full_df = full_df.sort_values("Sharpe", ascending=False)
        print(f"\n  ── Full Period Ranking ──")
        rank_cols = ["Config", "CAGR%", "Sharpe", "Sortino", "MaxDD%",
                     "Calmar", "WinRate%", "PF", "Trades", "Ann.Trades",
                     "AvgHold(bars)", "Fee%"]
        available_cols = [c for c in rank_cols if c in full_df.columns]
        print(full_df[available_cols].to_string(index=False))

    # ── OOS comparison ──
    oos_df = perf_df[perf_df["Segment"] == "OOS"].copy()
    if len(oos_df) > 0:
        oos_df = oos_df.sort_values("Sharpe", ascending=False)
        print(f"\n  ── OOS Period Ranking ──")
        available_cols = [c for c in rank_cols if c in oos_df.columns]
        print(oos_df[available_cols].to_string(index=False))

    # ══════════════════════════════════════════════════════════════
    #  Trade Analysis (for best full-period config)
    # ══════════════════════════════════════════════════════════════
    if len(full_df) > 0:
        best_config = full_df.iloc[0]["Config"]
        best_params = CONFIGS[best_config]
        print(f"\n  ── Trade Analysis: {best_config} ──")

        all_trades = []
        for sym in SYMBOLS:
            if sym not in data:
                continue
            tr = trade_analysis(sym, data[sym], best_params)
            if tr is not None and len(tr) > 0:
                all_trades.append(tr)

        if all_trades:
            trades_df = pd.concat(all_trades, ignore_index=True)
            trades_df.to_csv(out / "trade_analysis.csv", index=False)
            print(f"  💾 trade_analysis.csv ({len(trades_df)} trades)")

            # Summary stats
            if "PnL" in trades_df.columns:
                winners = trades_df[trades_df["PnL"] > 0]
                losers = trades_df[trades_df["PnL"] <= 0]
                print(f"    Total Trades:  {len(trades_df)}")
                print(f"    Winners:       {len(winners)} ({len(winners)/len(trades_df)*100:.1f}%)")
                print(f"    Losers:        {len(losers)} ({len(losers)/len(trades_df)*100:.1f}%)")
                print(f"    Avg Win:       ${winners['PnL'].mean():.2f}" if len(winners) > 0 else "")
                print(f"    Avg Loss:      ${losers['PnL'].mean():.2f}" if len(losers) > 0 else "")
                total_pnl = trades_df["PnL"].sum()
                top10_pnl = trades_df.nlargest(max(1, len(trades_df)//10), "PnL")["PnL"].sum()
                top10_pct = top10_pnl / total_pnl * 100 if abs(total_pnl) > 0 else 0
                print(f"    Top10% Contr:  {top10_pct:.1f}% ({'右尾驅動' if top10_pct > 50 else '高勝率驅動'})")

    # ══════════════════════════════════════════════════════════════
    #  Summary
    # ══════════════════════════════════════════════════════════════
    summary_lines = [
        "=" * 72,
        "  Mean Revert Liquidity Sweep — Summary",
        "=" * 72,
        "",
        f"  Timeframe: 15m",
        f"  Symbols:   {', '.join(SYMBOLS)}",
        f"  Configs:   {len(CONFIGS)}",
        f"  Fee: {FEE_BPS}bps, Slippage: {SLIPPAGE_BPS}bps",
        "",
        "  Strategy Logic:",
        "    - Detect 24h high/low liquidity sweep (false breakout)",
        "    - Confirm with RSI extreme + close reversal",
        "    - Fixed TP/SL/Time-stop exits",
        "",
        "  Anti-Lookahead:",
        "    - signal_delay=1 (entry at next bar open)",
        "    - SL/TP use intrabar high/low + trigger price execution",
        "    - Rolling high/low shifted by 1 (no current bar inclusion)",
        "",
    ]

    if len(full_df) > 0:
        best = full_df.iloc[0]
        summary_lines.extend([
            "  ── Best Full-Period Config ──",
            f"    Config:     {best['Config']}",
            f"    Sharpe:     {best.get('Sharpe', 'N/A')}",
            f"    CAGR:       {best.get('CAGR%', 'N/A')}%",
            f"    MaxDD:      {best.get('MaxDD%', 'N/A')}%",
            f"    Win Rate:   {best.get('WinRate%', 'N/A')}%",
            f"    PF:         {best.get('PF', 'N/A')}",
            f"    Avg Hold:   {best.get('AvgHold(bars)', 'N/A')} bars",
            f"    Ann.Trades: {best.get('Ann.Trades', 'N/A')}",
            "",
        ])

        # Check OOS consistency
        if len(oos_df) > 0:
            oos_best_row = oos_df[oos_df["Config"] == best["Config"]]
            if len(oos_best_row) > 0:
                oos_sh = oos_best_row.iloc[0].get("Sharpe", 0)
                is_sh = best.get("Sharpe", 0)
                summary_lines.extend([
                    f"  ── IS vs OOS Consistency ──",
                    f"    IS  Sharpe: {is_sh}",
                    f"    OOS Sharpe: {oos_sh}",
                    f"    Verdict:    {'✅ OOS positive' if oos_sh > 0 else '⚠ OOS degraded'}",
                    "",
                ])

    summary_lines.extend([
        "  ── Strategy Character ──",
        "    This is a HIGH WIN-RATE strategy (expected WR > 50%)",
        "    with FIXED TP/SL → consistent PnL distribution.",
        "    Complement to TSMOM (right-tail, low WR, trend-following).",
        "",
        "  ── Next Steps ──",
        "    1. If OOS Sharpe > 0: explore combining with TSMOM",
        "    2. Correlation analysis: daily returns corr(MR, TSMOM)",
        "    3. Paper trade best config for 2-4 weeks",
        "",
    ])

    summary_text = "\n".join(summary_lines)
    print(f"\n{summary_text}")
    with open(out / "summary.txt", "w") as f:
        f.write(summary_text)
    print(f"  💾 summary.txt saved")

    print(f"\n{'═' * 72}")
    print(f"  ✅ Research complete → {out}")
    print(f"{'═' * 72}")

    return out


if __name__ == "__main__":
    main()
