#!/usr/bin/env python3
"""
Governance Metrics Pipeline — compute SymbolMetrics from live trading.db

Reads the SQLite trading database and produces per-symbol SymbolMetrics JSON
that the governance review script can consume.

Usage:
    # Default: compute from prod config DB, output to reports/symbol_governance/
    PYTHONPATH=src python scripts/compute_governance_metrics.py \
        -c config/prod_candidate_simplified.yaml

    # Specify DB path and output
    PYTHONPATH=src python scripts/compute_governance_metrics.py \
        -c config/prod_candidate_simplified.yaml \
        --db reports/futures/meta_blend/live/trading.db \
        --output reports/symbol_governance/metrics_latest.json

    # Custom lookback window (default: 28 days = 4 weeks)
    PYTHONPATH=src python scripts/compute_governance_metrics.py \
        -c config/prod_candidate_simplified.yaml --days 14

Cron (every Monday 00:00 UTC):
    0 0 * * 1 cd ~/quant-binance-spot && .venv/bin/python scripts/compute_governance_metrics.py \
        -c config/prod_candidate_simplified.yaml && \
        .venv/bin/python scripts/run_symbol_governance_review.py \
        -c config/prod_candidate_simplified.yaml \
        --metrics-file reports/symbol_governance/metrics_latest.json \
        >> logs/governance.log 2>&1
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from qtrade.config import load_config
from qtrade.live.symbol_metrics_store import SymbolMetrics


def _get_db_path(cfg) -> Path:
    """Derive trading.db path from config (same logic as base_runner)."""
    return cfg.get_report_dir("live") / "trading.db"


def compute_symbol_metrics(
    db_path: Path,
    symbol: str,
    days: int = 28,
    model_slippage_bps: float = 3.0,
) -> SymbolMetrics | None:
    """
    Compute SymbolMetrics for one symbol from trading.db.

    Returns None if no data found for the symbol in the lookback window.
    """
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.row_factory = sqlite3.Row

    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    now_iso = datetime.now(timezone.utc).isoformat()

    # -- Trades data --
    trades = conn.execute(
        """
        SELECT timestamp, symbol, side, position_side, qty, price, value, fee, pnl
        FROM trades
        WHERE symbol = ? AND timestamp >= ?
        ORDER BY timestamp
        """,
        (symbol, since),
    ).fetchall()

    if not trades:
        conn.close()
        return None

    # Net PnL (sum of realized PnL from closed trades)
    pnl_values = [t["pnl"] for t in trades if t["pnl"] is not None]
    net_pnl = sum(pnl_values)

    # Turnover (total traded value)
    turnover = sum(t["value"] for t in trades)

    # Trade count (only count closing trades with PnL)
    trade_count = len(pnl_values)

    # Returns series — per-trade return as fraction of trade value
    returns_series = []
    for t in trades:
        if t["pnl"] is not None and t["value"] > 0:
            returns_series.append(t["pnl"] / t["value"])

    # Max drawdown from cumulative PnL
    max_drawdown_pct = 0.0
    if pnl_values:
        cum_pnl = 0.0
        peak = 0.0
        for pnl in pnl_values:
            cum_pnl += pnl
            peak = max(peak, cum_pnl)
            dd = peak - cum_pnl
            if peak > 0:
                dd_pct = (dd / peak) * 100
                max_drawdown_pct = max(max_drawdown_pct, dd_pct)

    # Realized slippage — compare signal-implied price vs actual fill price
    # Approximation: use fee as a proxy for realized cost per trade
    total_fee_bps = 0.0
    fee_trades = 0
    for t in trades:
        if t["value"] > 0:
            fee_bps = (t["fee"] / t["value"]) * 10000 if t["fee"] else 0
            total_fee_bps += fee_bps
            fee_trades += 1
    realized_slippage_bps = total_fee_bps / fee_trades if fee_trades > 0 else 0.0

    # Signal execution consistency — compare signal count vs trade count
    signal_count = conn.execute(
        "SELECT COUNT(*) as cnt FROM signals WHERE symbol = ? AND timestamp >= ?",
        (symbol, since),
    ).fetchone()["cnt"]

    expected_bars = days * 24  # 1h bars
    if signal_count > 0 and expected_bars > 0:
        consistency_pct = min(100.0, (signal_count / expected_bars) * 100)
    else:
        consistency_pct = 100.0

    # Missed signals — bars where signal exists but no trade when signal != 0
    missed_signals_pct = 0.0
    if signal_count > 0:
        nonzero_signals = conn.execute(
            "SELECT COUNT(*) as cnt FROM signals WHERE symbol = ? AND timestamp >= ? AND signal_value != 0",
            (symbol, since),
        ).fetchone()["cnt"]
        if nonzero_signals > 0 and trade_count > 0:
            missed_signals_pct = max(0.0, (1 - trade_count / nonzero_signals) * 100)

    conn.close()

    window_start = since[:19]
    window_end = now_iso[:19]

    m = SymbolMetrics(
        symbol=symbol,
        net_pnl=net_pnl,
        turnover=turnover,
        returns_series=returns_series,
        max_drawdown_pct=max_drawdown_pct,
        realized_slippage_bps=realized_slippage_bps,
        model_slippage_bps=model_slippage_bps,
        signal_execution_consistency_pct=consistency_pct,
        missed_signals_pct=missed_signals_pct,
        trade_count=trade_count,
        window_start=window_start,
        window_end=window_end,
    )
    m.compute()
    return m


def main():
    parser = argparse.ArgumentParser(
        description="Compute governance metrics from trading.db",
    )
    parser.add_argument("-c", "--config", required=True, help="YAML config path")
    parser.add_argument("--db", type=str, default=None, help="Override trading.db path")
    parser.add_argument("--days", type=int, default=28, help="Lookback window in days (default: 28)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    symbols = cfg.market.symbols

    db_path = Path(args.db) if args.db else _get_db_path(cfg)
    if not db_path.exists():
        print(f"❌ Trading DB not found: {db_path}")
        sys.exit(1)

    slippage_bps = cfg.backtest.slippage_bps if hasattr(cfg.backtest, "slippage_bps") else 3.0

    print(f"{'=' * 60}")
    print(f"  Governance Metrics Pipeline")
    print(f"  DB: {db_path}")
    print(f"  Window: {args.days} days")
    print(f"  Symbols: {', '.join(symbols)}")
    print(f"{'=' * 60}")

    all_metrics = {}
    for sym in symbols:
        m = compute_symbol_metrics(db_path, sym, days=args.days, model_slippage_bps=slippage_bps)
        if m is None:
            print(f"  ⚠️  {sym}: no data in window")
            continue
        all_metrics[sym] = m

        icon = "✅" if m.edge_sharpe_4w > 0.3 else "⚠️" if m.edge_sharpe_4w > 0 else "❌"
        print(
            f"  {icon} {sym:12s}  sharpe={m.edge_sharpe_4w:>+8.2f}  "
            f"pnl={m.net_pnl:>+10.4f}  trades={m.trade_count:>3d}  "
            f"dd={m.dd_4w:>5.1f}%  slip_ratio={m.slippage_ratio_4w:>.2f}"
        )

    # Output JSON
    output_path = Path(args.output) if args.output else Path(
        cfg.live.symbol_governance.artifacts_dir
    ) / "metrics_latest.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "window_days": args.days,
        "db_path": str(db_path),
        "symbols": [m.to_dict() for m in all_metrics.values()],
    }
    # Include returns_series in the full output for the review script
    for entry in payload["symbols"]:
        sym = entry["symbol"]
        if sym in all_metrics:
            entry["returns_series"] = all_metrics[sym].returns_series

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Metrics written to {output_path}")
    print(f"   Next step: python scripts/run_symbol_governance_review.py "
          f"-c {args.config} --metrics-file {output_path}")


if __name__ == "__main__":
    main()
