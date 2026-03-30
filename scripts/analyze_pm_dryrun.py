#!/usr/bin/env python3
"""
Polymarket Dry Run Analyzer — Are we making money?

Reads settlement logs and gives a clear verdict:
  GO (positive EV confirmed) / STOP (negative EV) / WAIT (not enough data)

Usage:
  PYTHONPATH=src python scripts/analyze_pm_dryrun.py
  # Or on Oracle Cloud:
  ssh ubuntu@140.83.57.255 "cd ~/quant-binance-spot && ..."
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from collections import Counter
from datetime import datetime

LOG_DIR = Path("logs/polymarket")


def load_settlements() -> list[dict]:
    settlements = []
    for f in sorted(LOG_DIR.glob("settlements_*.jsonl")):
        with open(f) as fh:
            for line in fh:
                if line.strip():
                    settlements.append(json.loads(line.strip()))
    return settlements


def load_trades() -> list[dict]:
    trades = []
    for f in sorted(LOG_DIR.glob("trades_15m_*.jsonl")):
        with open(f) as fh:
            for line in fh:
                if line.strip():
                    trades.append(json.loads(line.strip()))
    return trades


def load_snapshots() -> list[dict]:
    snapshots = []
    for f in sorted(LOG_DIR.glob("snapshots_*.jsonl")):
        with open(f) as fh:
            for line in fh:
                if line.strip():
                    snapshots.append(json.loads(line.strip()))
    return snapshots


def main():
    print("=" * 60)
    print("POLYMARKET DRY RUN ANALYSIS")
    print("=" * 60)

    settlements = load_settlements()
    trades = load_trades()
    snapshots = load_snapshots()

    print(f"\nData: {len(settlements)} settlements, {len(trades)} trades, {len(snapshots)} snapshots")

    if not settlements and not trades:
        print("\n⏳ VERDICT: WAIT — no data yet")
        print("   Let the bot run for 24-48 hours during London/NY hours")

        # Show snapshot stats if available
        if snapshots:
            # Deduplicate snapshots by slug
            seen = set()
            unique = []
            for s in snapshots:
                key = s.get("slug", "")
                if key not in seen:
                    seen.add(key)
                    unique.append(s)

            signals = [s for s in unique if s.get("signal")]
            print(f"\n   Snapshots: {len(unique)} unique windows observed")
            print(f"   Signals fired: {len(signals)}")
            if signals:
                by_strat = Counter(s.get("signal") for s in signals)
                print(f"   By strategy: {dict(by_strat)}")
        return

    # ── Settlement Analysis ──
    if settlements:
        print(f"\n{'─' * 40}")
        print("SETTLEMENT RESULTS")
        print(f"{'─' * 40}")

        wins = [s for s in settlements if s.get("won")]
        losses = [s for s in settlements if not s.get("won")]
        total_pnl = sum(s.get("pnl", 0) for s in settlements)

        print(f"  Total: {len(settlements)} settled")
        print(f"  Wins: {len(wins)} ({len(wins)/len(settlements)*100:.0f}%)")
        print(f"  Losses: {len(losses)} ({len(losses)/len(settlements)*100:.0f}%)")
        print(f"  PnL: ${total_pnl:+.2f}")
        print(f"  Avg PnL/trade: ${total_pnl/len(settlements):+.3f}")

        # By strategy
        by_strat = {}
        for s in settlements:
            strat = s.get("strategy", "unknown")
            by_strat.setdefault(strat, []).append(s)

        if len(by_strat) > 1:
            print(f"\n  By Strategy:")
            for strat, trades_list in by_strat.items():
                w = sum(1 for t in trades_list if t.get("won"))
                p = sum(t.get("pnl", 0) for t in trades_list)
                print(f"    {strat:15s}: {len(trades_list)} trades, WR={w/len(trades_list)*100:.0f}%, PnL=${p:+.2f}")

        # By hour
        print(f"\n  By Hour (UTC):")
        by_hour = {}
        for s in settlements:
            h = datetime.fromisoformat(s["timestamp"]).hour
            by_hour.setdefault(h, []).append(s)
        for h in sorted(by_hour):
            trades_h = by_hour[h]
            w = sum(1 for t in trades_h if t.get("won"))
            p = sum(t.get("pnl", 0) for t in trades_h)
            if len(trades_h) >= 3:
                print(f"    {h:>2d}:00: {len(trades_h)} trades, WR={w/len(trades_h)*100:.0f}%, PnL=${p:+.2f}")

    # ── Trade Log Analysis (if no settlements yet) ──
    if trades and not settlements:
        print(f"\n{'─' * 40}")
        print("TRADE SIGNALS (no settlements yet)")
        print(f"{'─' * 40}")

        dry = [t for t in trades if t.get("dry_run")]
        real = [t for t in trades if not t.get("dry_run")]
        print(f"  Dry run signals: {len(dry)}")
        print(f"  Real orders: {len(real)}")

        by_strat = Counter(t.get("strategy", "unknown") for t in trades)
        print(f"  By strategy: {dict(by_strat)}")

    # ── Snapshot Analysis ──
    if snapshots:
        seen = set()
        unique = []
        for s in snapshots:
            key = (s.get("slug", ""), s.get("coin", ""))
            if key not in seen:
                seen.add(key)
                unique.append(s)

        signals = [s for s in unique if s.get("signal")]
        no_signals = [s for s in unique if not s.get("signal")]

        print(f"\n{'─' * 40}")
        print("SNAPSHOT ANALYSIS")
        print(f"{'─' * 40}")
        print(f"  Windows observed: {len(unique)}")
        print(f"  Signals fired: {len(signals)} ({len(signals)/len(unique)*100:.0f}%)")
        print(f"  No signal: {len(no_signals)} ({len(no_signals)/len(unique)*100:.0f}%)")

        if signals:
            by_strat = Counter(s.get("signal") for s in signals)
            print(f"  Signal breakdown: {dict(by_strat)}")

    # ── Verdict ──
    print(f"\n{'═' * 60}")
    if settlements and len(settlements) >= 50:
        total_pnl = sum(s.get("pnl", 0) for s in settlements)
        if total_pnl > 0:
            print(f"✅ VERDICT: GO — Positive EV confirmed ({len(settlements)} trades, PnL ${total_pnl:+.2f})")
        else:
            print(f"❌ VERDICT: STOP — Negative EV ({len(settlements)} trades, PnL ${total_pnl:+.2f})")
    elif settlements and len(settlements) >= 20:
        total_pnl = sum(s.get("pnl", 0) for s in settlements)
        print(f"⚠️ VERDICT: PRELIMINARY — {len(settlements)} trades, PnL ${total_pnl:+.2f}")
        print(f"   Need 50+ trades for confidence. Keep running.")
    else:
        n = len(settlements) if settlements else 0
        print(f"⏳ VERDICT: WAIT — Only {n} settlements. Need 50+.")
        print(f"   Let bot run during London/NY hours (UTC 7-17)")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
