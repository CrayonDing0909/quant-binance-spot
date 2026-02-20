#!/usr/bin/env python3
"""
Production Report Generator — Daily & Weekly Operations Report

Generates standardized operations reports for the R1 ensemble.
Reads backtest replay data (or live equity if available) and produces
Markdown + JSON reports.

Usage:
    # Daily report
    PYTHONPATH=src python scripts/prod_report.py --daily

    # Weekly report
    PYTHONPATH=src python scripts/prod_report.py --weekly

    # Custom lookback
    PYTHONPATH=src python scripts/prod_report.py --daily --lookback 7

    # JSON only
    PYTHONPATH=src python scripts/prod_report.py --daily --json-only
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

REPORT_DIR = Path("reports/prod_reports")
RULES_PATH = "config/prod_scale_rules_R1.yaml"
CONFIG_PATH = "config/prod_candidate_R1.yaml"


def load_scale_rules(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_data_path(cfg, sym: str) -> Path:
    """Resolve data path trying multiple directory structures."""
    market_type = cfg.market_type_str
    interval = cfg.market.interval
    candidates = [
        cfg.data_dir / "binance" / market_type / interval / f"{sym}.parquet",
        cfg.data_dir / "binance" / market_type / "klines" / f"{sym}.parquet",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def load_ensemble_strategies(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    ens = raw.get("ensemble", {})
    if ens and ens.get("enabled", False):
        return ens.get("strategies", {})
    return {}


def compute_rolling_sharpe(returns: pd.Series, window_hours: int = 480) -> pd.Series:
    """Compute rolling Sharpe ratio (annualized from hourly)."""
    roll_mean = returns.rolling(window_hours).mean()
    roll_std = returns.rolling(window_hours).std()
    return np.where(roll_std > 0, roll_mean / roll_std * np.sqrt(8760), 0.0)


def compute_turnover(positions: pd.Series) -> float:
    """Compute avg daily turnover (position changes / day)."""
    changes = positions.diff().abs().sum()
    n_days = len(positions) / 24.0
    return changes / max(n_days, 1.0)


def evaluate_scale_decision(metrics: dict, rules: dict) -> str:
    """Evaluate scale-up/down/hard-stop rules and return decision."""
    # Hard stop (any trigger)
    hard_stop = rules.get("hard_stop", {})
    for cond in hard_stop.get("conditions", []):
        val = metrics.get(cond["metric"])
        if val is None:
            continue
        if cond["condition"] == "gt" and val > cond["threshold"]:
            return "PAUSE"
        if cond["condition"] == "lt" and val < cond["threshold"]:
            return "PAUSE"

    # Scale down (any trigger)
    scale_down = rules.get("scale_down", {})
    for cond in scale_down.get("conditions", []):
        val = metrics.get(cond["metric"])
        if val is None:
            continue
        if cond["condition"] == "gt" and val > cond["threshold"]:
            return "REDUCE"
        if cond["condition"] == "lt" and val < cond["threshold"]:
            return "REDUCE"

    # Scale up (all conditions met)
    scale_up = rules.get("scale_up", {})
    all_met = True
    for cond in scale_up.get("conditions", []):
        val = metrics.get(cond["metric"])
        if val is None:
            all_met = False
            break
        if cond["condition"] == "gt" and not (val > cond["threshold"]):
            all_met = False
            break
        if cond["condition"] == "lt" and not (val < cond["threshold"]):
            all_met = False
            break
        if cond["condition"] == "lte" and not (val <= cond["threshold"]):
            all_met = False
            break

    if all_met and scale_up.get("conditions"):
        return "SCALE_UP"

    return "KEEP"


def generate_report(
    mode: str = "daily",
    lookback_days: int | None = None,
    config_path: str = CONFIG_PATH,
    rules_path: str = RULES_PATH,
) -> dict:
    """
    Generate a production operations report.

    Args:
        mode: "daily" or "weekly"
        lookback_days: override lookback (default: 1 for daily, 7 for weekly)
        config_path: frozen strategy config
        rules_path: scale rules config

    Returns:
        Report dict with all metrics.
    """
    from qtrade.config import load_config
    from qtrade.backtest.run_backtest import run_symbol_backtest

    if lookback_days is None:
        lookback_days = 1 if mode == "daily" else 7

    cfg = load_config(config_path)
    rules = load_scale_rules(rules_path)
    ensemble_strats = load_ensemble_strategies(config_path)
    market_type = cfg.market_type_str

    symbols = cfg.market.symbols
    weights = {s: rules["portfolio"]["weights"].get(s, 1.0 / len(symbols)) for s in symbols}

    # Determine report period (use last N+30 days for warmup, report on last N)
    warmup_days = 60
    report_start_bar = -(lookback_days * 24)  # last N days in bars

    sleeve_results = {}
    sleeve_equities = {}
    sleeve_positions = {}

    for sym in symbols:
        data_path = resolve_data_path(cfg, sym)
        if not data_path.exists():
            print(f"  ⚠️  Data not found for {sym}: {data_path}")
            continue

        bt_cfg = cfg.to_backtest_dict(symbol=sym)

        strat_name = cfg.strategy.name
        if sym in ensemble_strats:
            strat_name = ensemble_strats[sym]["name"]
            bt_cfg["strategy_params"] = ensemble_strats[sym].get("params", {})

        try:
            res = run_symbol_backtest(sym, data_path, bt_cfg, strategy_name=strat_name,
                                      data_dir=cfg.data_dir)
            sleeve_results[sym] = res
            sleeve_equities[sym] = res.equity()
        except Exception as e:
            print(f"  ❌ Backtest failed for {sym}: {e}")

    if len(sleeve_equities) < len(symbols):
        print("  ⚠️  Not all sleeves available, partial report")

    # Align and build portfolio
    all_eq = list(sleeve_equities.values())
    min_start = max(eq.index[0] for eq in all_eq)
    max_end = min(eq.index[-1] for eq in all_eq)

    normalized = {}
    for sym in sleeve_equities:
        eq = sleeve_equities[sym].loc[min_start:max_end]
        normalized[sym] = eq / eq.iloc[0]

    active_symbols = list(sleeve_equities.keys())
    portfolio_norm = sum(normalized[s] * weights.get(s, 1.0 / len(active_symbols)) for s in active_symbols)
    portfolio_equity = portfolio_norm * cfg.backtest.initial_cash

    # Full period metrics
    full_returns = portfolio_equity.pct_change().fillna(0)
    full_sharpe = float(full_returns.mean() / full_returns.std() * np.sqrt(8760)) if full_returns.std() > 0 else 0.0

    # Report period metrics (last N days)
    n_bars = abs(report_start_bar)
    period_equity = portfolio_equity.iloc[-n_bars:] if len(portfolio_equity) > n_bars else portfolio_equity
    period_returns = period_equity.pct_change().fillna(0)

    period_return = float(period_equity.iloc[-1] / period_equity.iloc[0] - 1) if len(period_equity) > 1 else 0.0
    period_sharpe = float(period_returns.mean() / period_returns.std() * np.sqrt(8760)) if period_returns.std() > 0 else 0.0

    # Rolling 20D Sharpe (latest value)
    rolling_sr = compute_rolling_sharpe(full_returns, window_hours=min(480, len(full_returns)))
    latest_rolling_sr = float(rolling_sr[-1]) if len(rolling_sr) > 0 else 0.0

    # MDD
    peak = portfolio_equity.expanding().max()
    dd = (portfolio_equity - peak) / peak
    running_mdd = abs(float(dd.iloc[-1]))
    max_mdd = abs(float(dd.min()))

    # Period MDD
    period_peak = period_equity.expanding().max()
    period_dd = (period_equity - period_peak) / period_peak
    period_mdd = abs(float(period_dd.min()))

    # 30D return
    n_30d = min(30 * 24, len(portfolio_equity))
    ret_30d = float(portfolio_equity.iloc[-1] / portfolio_equity.iloc[-n_30d] - 1) if n_30d > 24 else 0.0

    # Sleeve metrics
    sleeve_metrics = {}
    for sym in active_symbols:
        eq = sleeve_equities[sym].loc[min_start:max_end]
        s_ret = float(eq.iloc[-1] / eq.iloc[0] - 1)
        s_returns = eq.pct_change().fillna(0)
        s_sr = float(s_returns.mean() / s_returns.std() * np.sqrt(8760)) if s_returns.std() > 0 else 0.0
        s_peak = eq.expanding().max()
        s_dd = (eq - s_peak) / s_peak
        s_mdd = abs(float(s_dd.min()))

        # 14D contribution
        n_14d = min(14 * 24, len(eq))
        s_14d_ret = float(eq.iloc[-1] / eq.iloc[-n_14d] - 1) if n_14d > 24 else 0.0

        sleeve_metrics[sym] = {
            "strategy": ensemble_strats.get(sym, {}).get("name", cfg.strategy.name),
            "total_return_pct": s_ret * 100,
            "sharpe": s_sr,
            "max_dd_pct": s_mdd * 100,
            "contribution_14d_pct": s_14d_ret * 100,
        }

    # Cost estimates (from backtest assumptions)
    assumptions = rules.get("backtest_assumptions", {})
    estimated_fee_bps = assumptions.get("fee_bps", 5)
    estimated_slip_bps = assumptions.get("slippage_bps", 3)
    estimated_fr_8h = assumptions.get("funding_rate_8h", 0.0001)

    # Aggregate metrics for decision engine
    decision_metrics = {
        "rolling_sharpe_20d": latest_rolling_sr,
        "running_mdd_pct": running_mdd * 100,
        "return_30d_pct": ret_30d * 100,
        "slippage_ratio": 1.0,  # placeholder; real = from live fill data
        "funding_ratio": 1.0,   # placeholder
        "flatten_event_count_30d": 0,  # placeholder; from risk_guard DB
        "worst_sleeve_14d_return_pct": min(s["contribution_14d_pct"] for s in sleeve_metrics.values()) if sleeve_metrics else 0,
    }

    decision = evaluate_scale_decision(decision_metrics, rules)

    # Risk events (placeholder — from risk_guard DB in production)
    risk_events = {
        "warning_count": 0,
        "reduce_count": 0,
        "flatten_count": 0,
    }

    report = {
        "meta": {
            "report_type": mode,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "config": config_path,
            "lookback_days": lookback_days,
            "data_range": f"{min_start} → {max_end}",
            "report_period": f"last {lookback_days} day(s)",
        },
        "portfolio": {
            "total_return_pct": float(portfolio_equity.iloc[-1] / portfolio_equity.iloc[0] - 1) * 100,
            "period_return_pct": period_return * 100,
            "rolling_sharpe_20d": latest_rolling_sr,
            "full_sharpe": full_sharpe,
            "max_dd_pct": max_mdd * 100,
            "running_dd_pct": running_mdd * 100,
            "period_mdd_pct": period_mdd * 100,
            "return_30d_pct": ret_30d * 100,
        },
        "sleeves": sleeve_metrics,
        "execution": {
            "assumed_fee_bps": estimated_fee_bps,
            "assumed_slippage_bps": estimated_slip_bps,
            "assumed_funding_8h": estimated_fr_8h,
            "realized_slippage_bps": "N/A (backtest mode)",
            "funding_drift_ratio": "N/A (backtest mode)",
            "note": "Real execution metrics available after live deployment",
        },
        "risk_events": risk_events,
        "decision": {
            "recommendation": decision,
            "decision_metrics": decision_metrics,
        },
    }

    return report


def format_markdown(report: dict) -> str:
    """Format report dict as Markdown."""
    mode = report["meta"]["report_type"].upper()
    lines = [
        f"# Production {mode} Report",
        f"",
        f"**Generated:** {report['meta']['generated_at']}",
        f"**Config:** `{report['meta']['config']}`",
        f"**Period:** {report['meta']['report_period']}",
        f"**Data Range:** {report['meta']['data_range']}",
        f"",
        f"---",
        f"",
        f"## Portfolio Summary",
        f"",
        f"| Metric | Value |",
        f"|--------|:-----:|",
        f"| Total Return | {report['portfolio']['total_return_pct']:.2f}% |",
        f"| Period Return | {report['portfolio']['period_return_pct']:.2f}% |",
        f"| 20D Rolling Sharpe | {report['portfolio']['rolling_sharpe_20d']:.2f} |",
        f"| Full Period Sharpe | {report['portfolio']['full_sharpe']:.2f} |",
        f"| Max Drawdown | {report['portfolio']['max_dd_pct']:.2f}% |",
        f"| Running Drawdown | {report['portfolio']['running_dd_pct']:.2f}% |",
        f"| 30D Return | {report['portfolio']['return_30d_pct']:.2f}% |",
        f"",
        f"## Sleeve Contribution",
        f"",
        f"| Sleeve | Strategy | Return | Sharpe | MaxDD | 14D Contrib |",
        f"|--------|----------|:------:|:------:|:-----:|:-----------:|",
    ]

    for sym, sm in report["sleeves"].items():
        sign = "✅" if sm["contribution_14d_pct"] > 0 else "❌"
        lines.append(
            f"| {sym} | {sm['strategy']} | {sm['total_return_pct']:.2f}% | "
            f"{sm['sharpe']:.2f} | {sm['max_dd_pct']:.2f}% | "
            f"{sign} {sm['contribution_14d_pct']:+.2f}% |"
        )

    lines.extend([
        f"",
        f"## Execution",
        f"",
        f"| Item | Assumed | Realized |",
        f"|------|:------:|:--------:|",
        f"| Fee (bps) | {report['execution']['assumed_fee_bps']} | {report['execution']['realized_slippage_bps']} |",
        f"| Slippage (bps) | {report['execution']['assumed_slippage_bps']} | {report['execution']['realized_slippage_bps']} |",
        f"| Funding (8h) | {report['execution']['assumed_funding_8h']:.4%} | {report['execution']['funding_drift_ratio']} |",
        f"",
        f"> {report['execution']['note']}",
        f"",
        f"## Risk Events (Period)",
        f"",
        f"| Level | Count |",
        f"|-------|:-----:|",
        f"| WARNING | {report['risk_events']['warning_count']} |",
        f"| REDUCE | {report['risk_events']['reduce_count']} |",
        f"| FLATTEN | {report['risk_events']['flatten_count']} |",
        f"",
        f"## Decision",
        f"",
    ])

    decision = report["decision"]["recommendation"]
    emoji = {"KEEP": "🟢", "SCALE_UP": "🔵", "REDUCE": "🟡", "PAUSE": "🔴"}.get(decision, "⚪")
    lines.append(f"**Recommendation: {emoji} {decision}**")
    lines.append(f"")

    dm = report["decision"]["decision_metrics"]
    lines.extend([
        f"| Metric | Value |",
        f"|--------|:-----:|",
        f"| 20D Rolling Sharpe | {dm['rolling_sharpe_20d']:.2f} |",
        f"| Running MDD | {dm['running_mdd_pct']:.2f}% |",
        f"| 30D Return | {dm['return_30d_pct']:.2f}% |",
        f"| Worst Sleeve 14D | {dm['worst_sleeve_14d_return_pct']:.2f}% |",
        f"",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Production Report Generator")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--daily", action="store_true", help="Generate daily report")
    mode_group.add_argument("--weekly", action="store_true", help="Generate weekly report")

    parser.add_argument("--lookback", type=int, default=None, help="Override lookback days")
    parser.add_argument("--config", type=str, default=CONFIG_PATH, help="Strategy config")
    parser.add_argument("--rules", type=str, default=RULES_PATH, help="Scale rules config")
    parser.add_argument("--json-only", action="store_true", help="Only output JSON")

    args = parser.parse_args()

    mode = "daily" if args.daily else "weekly"
    print(f"📊 Generating {mode} report...")

    report = generate_report(
        mode=mode,
        lookback_days=args.lookback,
        config_path=args.config,
        rules_path=args.rules,
    )

    # Save outputs
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPORT_DIR / mode / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = out_dir / f"{mode}_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"  📄 JSON: {json_path}")

    # Markdown
    if not args.json_only:
        md_content = format_markdown(report)
        md_path = out_dir / f"{mode}_report.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"  📝 Markdown: {md_path}")

        # Print to console
        print(f"\n{'='*70}")
        print(md_content)

    # Print decision
    decision = report["decision"]["recommendation"]
    emoji = {"KEEP": "🟢", "SCALE_UP": "🔵", "REDUCE": "🟡", "PAUSE": "🔴"}.get(decision, "⚪")
    print(f"\n  {emoji} Decision: {decision}")


if __name__ == "__main__":
    main()
