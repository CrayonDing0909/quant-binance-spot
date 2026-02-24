#!/usr/bin/env python3
"""
Symbol Governance Weekly Review — CLI Entry Point

Usage:
    # Dry-run (generate report only, do NOT apply changes)
    python scripts/run_symbol_governance_review.py -c config/prod_candidate_R3C_universe.yaml --dry-run

    # Apply (persist artifact + effective weights)
    python scripts/run_symbol_governance_review.py -c config/prod_candidate_R3C_universe.yaml

    # Reproducible run with a specific review date
    python scripts/run_symbol_governance_review.py -c config/prod_candidate_R3C_universe.yaml --review-date 2026-02-17

    # Inject metrics from JSON file (for testing / backfill)
    python scripts/run_symbol_governance_review.py -c config/prod_candidate_R3C_universe.yaml --metrics-file metrics.json --dry-run

Spec: docs/R3C_SYMBOL_GOVERNANCE_SPEC.md
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root on sys.path for direct `python scripts/...` execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from qtrade.config import load_config
from qtrade.live.symbol_governance import (
    SymbolGovernanceEngine,
    load_latest_decisions,
)
from qtrade.live.symbol_metrics_store import SymbolMetrics, build_metrics_from_dict


def _build_dummy_metrics(symbols: list[str]) -> dict[str, SymbolMetrics]:
    """
    Build placeholder metrics for symbols with no real data yet.

    All values default to "healthy" so that no transition fires on a cold-start.
    In production, replace this with actual data from trading_db or a metrics pipeline.
    """
    metrics = {}
    for sym in symbols:
        m = SymbolMetrics(
            symbol=sym,
            net_pnl=1.0,
            turnover=1000.0,
            returns_series=[0.001] * 672,  # 4 weeks of hourly bars
            max_drawdown_pct=1.0,
            realized_slippage_bps=2.0,
            model_slippage_bps=3.0,
            signal_execution_consistency_pct=100.0,
            missed_signals_pct=0.0,
            trade_count=50,
        )
        metrics[sym] = m.compute()
    return metrics


def _load_metrics_from_file(path: str) -> dict[str, SymbolMetrics]:
    """Load per-symbol metrics from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    metrics = {}
    items = raw if isinstance(raw, list) else raw.get("symbols", raw.values())
    for item in items:
        if isinstance(item, str):
            # raw is a dict keyed by symbol
            item = raw[item]
        m = build_metrics_from_dict(item)
        m.compute()
        metrics[m.symbol] = m
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Run Symbol Governance Weekly Review"
    )
    parser.add_argument(
        "-c", "--config", required=True, help="Path to YAML config"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate report only; do not persist or apply",
    )
    parser.add_argument(
        "--review-date",
        type=str,
        default=None,
        help="Override review date (YYYY-MM-DD) for reproducibility",
    )
    parser.add_argument(
        "--metrics-file",
        type=str,
        default=None,
        help="Path to JSON file with per-symbol metrics",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default=None,
        help="Override artifacts output directory",
    )
    args = parser.parse_args()

    # ── Load config ──
    cfg = load_config(args.config)
    gov_cfg = cfg.live.symbol_governance

    if not gov_cfg.enabled and not args.dry_run:
        print("⚠️  symbol_governance is disabled in config. Use --dry-run to preview.")
        sys.exit(0)

    # ── Determine review time ──
    if args.review_date:
        review_time = datetime.strptime(args.review_date, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )
    else:
        review_time = datetime.now(timezone.utc)

    # ── Build base weights ──
    symbols = cfg.market.symbols
    n = len(symbols)
    base_weights = {sym: cfg.portfolio.get_weight(sym, n) for sym in symbols}

    # ── Load metrics ──
    if args.metrics_file:
        metrics = _load_metrics_from_file(args.metrics_file)
    else:
        if not args.dry_run:
            print(
                "❌ 錯誤：正式 review（非 --dry-run）必須提供真實 metrics 來源。\n"
                "   請使用 --metrics-file <path> 指定 metrics JSON，\n"
                "   或加上 --dry-run 使用 placeholder 預覽。"
            )
            sys.exit(1)
        print("ℹ️  No --metrics-file provided; using placeholder healthy metrics (dry-run only).")
        metrics = _build_dummy_metrics(symbols)

    # ── Initialize engine & load previous state ──
    engine = SymbolGovernanceEngine(gov_cfg, base_weights)

    artifacts_dir = args.artifacts_dir or gov_cfg.artifacts_dir
    prev = load_latest_decisions(artifacts_dir)
    if prev:
        engine.load_state(prev.get("records"))
        print(f"✅ Loaded previous state from {artifacts_dir}/latest_decisions.json")
    else:
        print("ℹ️  No previous governance state found; starting fresh.")

    # ── Run review ──
    decision = engine.run_review(
        metrics=metrics,
        review_time=review_time,
        dry_run=args.dry_run,
    )

    # ── Print summary ──
    print(f"\n{'='*60}")
    print(f"Symbol Governance Review — {decision.review_date}")
    print(f"{'='*60}")
    print(f"  Active:      {decision.summary.get('active', 0)}")
    print(f"  Deweighted:  {decision.summary.get('deweighted', 0)}")
    print(f"  Quarantined: {decision.summary.get('quarantined', 0)}")
    print()

    for sym, detail in sorted(decision.symbols.items()):
        prev_s = detail["previous_state"]
        new_s = detail["new_state"]
        ew = detail["effective_weight"]
        reasons = detail.get("reason_codes", [])
        changed = " ⚡" if prev_s != new_s else ""
        print(
            f"  {sym:12s}  {prev_s:>12s} → {new_s:<12s}  "
            f"ew={ew:.4f}{changed}"
        )
        if reasons:
            print(f"               reasons: {', '.join(reasons)}")

    # ── Persist (unless dry-run) ──
    if args.dry_run:
        print(f"\n🔍 DRY-RUN complete. No artifacts written.")

        # Optionally dump to stdout
        out_path = Path(artifacts_dir) / "dry_run_preview.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(decision.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"   Preview saved to {out_path}")
    else:
        latest, history = engine.persist(decision, artifacts_dir)
        print(f"\n✅ Artifacts written:")
        print(f"   {latest}")
        print(f"   {history}")


if __name__ == "__main__":
    main()
