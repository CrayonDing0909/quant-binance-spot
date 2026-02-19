#!/usr/bin/env python3
"""
Risk Guard — 自動風控監控與 Kill Switch

功能：
  1. 讀取最近績效（回測 replay 或實盤 SQLite/equity 檔）
  2. 計算風控指標（20D rolling Sharpe, MDD, slippage, funding, sleeve contrib）
  3. 根據 YAML 規則輸出風控決策
  4. 產生 JSON 決策檔 + 人類可讀報告

模式：
  --dry-run     讀取實盤資料，只輸出建議（不下單）
  --replay      用回測 equity 模擬風控規則在歷史壓力期的觸發

使用方式：
  # Dry-run（實盤監控模式）
  python scripts/risk_guard.py --config config/risk_guard_alt_ensemble.yaml --dry-run

  # Replay（歷史壓力期驗證）
  python scripts/risk_guard.py --config config/risk_guard_alt_ensemble.yaml --replay

  # 自訂 replay 區間
  python scripts/risk_guard.py --config config/risk_guard_alt_ensemble.yaml --replay \\
      --replay-start 2022-07-01 --replay-end 2023-01-01

  # 指定 equity 檔案（dry-run 模式）
  python scripts/risk_guard.py --config config/risk_guard_alt_ensemble.yaml --dry-run \\
      --equity-dir reports/futures/live/
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ═══════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════

class RiskDecision:
    """A single risk decision from a rule evaluation."""

    NO_ACTION = "NO_ACTION"
    WARNING = "WARNING"
    REDUCE_RISK_50 = "REDUCE_RISK_50"
    DISABLE_SLEEVE = "DISABLE_SLEEVE"
    FLATTEN_ALL = "FLATTEN_ALL"

    # Severity ordering for tie-breaking
    SEVERITY_ORDER = {
        NO_ACTION: 0,
        WARNING: 1,
        REDUCE_RISK_50: 2,
        DISABLE_SLEEVE: 3,
        FLATTEN_ALL: 4,
    }

    def __init__(
        self,
        action: str,
        rule_name: str,
        description: str,
        severity: str,
        metric_name: str,
        metric_value: float,
        threshold: float,
        triggered: bool,
        timestamp: str | None = None,
    ):
        self.action = action
        self.rule_name = rule_name
        self.description = description
        self.severity = severity
        self.metric_name = metric_name
        self.metric_value = metric_value
        self.threshold = threshold
        self.triggered = triggered
        self.timestamp = timestamp or datetime.utcnow().isoformat()

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "rule_name": self.rule_name,
            "description": self.description,
            "severity": self.severity,
            "metric_name": self.metric_name,
            "metric_value": round(self.metric_value, 4) if isinstance(self.metric_value, float) else self.metric_value,
            "threshold": self.threshold,
            "triggered": self.triggered,
            "timestamp": self.timestamp,
        }

    @property
    def severity_level(self) -> int:
        base = self.action.split(":")[0] if ":" in self.action else self.action
        return self.SEVERITY_ORDER.get(base, 0)


# ═══════════════════════════════════════════════════════════════
# Metric Calculators
# ═══════════════════════════════════════════════════════════════

def compute_metrics(
    portfolio_equity: pd.Series,
    sleeve_equities: dict[str, pd.Series] | None = None,
    backtest_assumptions: dict | None = None,
) -> dict:
    """
    Compute all risk metrics from equity curve(s).

    Returns dict with metric names as keys, values as current metric values.
    """
    if portfolio_equity is None or len(portfolio_equity) < 2:
        return {}

    metrics = {}
    returns = portfolio_equity.pct_change().fillna(0)

    # ── 20D Rolling Sharpe ──
    window = min(20 * 24, len(returns))  # 20 days × 24 hours, or max available
    if window >= 48:  # at least 2 days
        rolling_ret = returns.rolling(window)
        rolling_mean = rolling_ret.mean()
        rolling_std = rolling_ret.std()
        rolling_sr = np.where(
            rolling_std > 0,
            rolling_mean / rolling_std * np.sqrt(8760),
            0,
        )
        metrics["rolling_sharpe_20d"] = float(rolling_sr[-1]) if len(rolling_sr) > 0 else 0.0
        # Store recent history for lookback checks
        metrics["_rolling_sharpe_20d_history"] = pd.Series(rolling_sr, index=returns.index)
    else:
        metrics["rolling_sharpe_20d"] = 0.0

    # ── Running MDD ──
    peak = portfolio_equity.expanding().max()
    drawdown = (portfolio_equity - peak) / peak
    metrics["running_mdd_pct"] = abs(float(drawdown.iloc[-1])) * 100
    metrics["_max_mdd_pct"] = abs(float(drawdown.min())) * 100

    # ── 30D Return ──
    lookback_30d = min(30 * 24, len(portfolio_equity))
    if lookback_30d > 24:
        ret_30d = (portfolio_equity.iloc[-1] / portfolio_equity.iloc[-lookback_30d] - 1) * 100
        metrics["return_30d_pct"] = float(ret_30d)
    else:
        metrics["return_30d_pct"] = 0.0

    # ── Total Return ──
    metrics["total_return_pct"] = (portfolio_equity.iloc[-1] / portfolio_equity.iloc[0] - 1) * 100

    # ── Full Period Sharpe ──
    if returns.std() > 0:
        metrics["full_period_sharpe"] = float(returns.mean() / returns.std() * np.sqrt(8760))
    else:
        metrics["full_period_sharpe"] = 0.0

    # ── Realized Slippage (placeholder — needs fill data) ──
    # In production, this comes from comparing signal price vs fill price.
    # For replay, we estimate from position turnover × assumed slippage.
    metrics["realized_slippage_bps"] = 0.0  # Must be filled externally

    # ── Funding Drag (placeholder) ──
    metrics["funding_drag_annualized_pct"] = 0.0  # Must be filled externally

    # ── Total Cost Ratio (placeholder) ──
    metrics["total_cost_ratio"] = 1.0  # Must be filled externally

    # ── Sleeve Metrics ──
    if sleeve_equities:
        for sym, seq in sleeve_equities.items():
            if seq is None or len(seq) < 2:
                continue
            s_peak = seq.expanding().max()
            s_dd = (seq - s_peak) / s_peak
            metrics[f"sleeve_mdd_pct_{sym}"] = abs(float(s_dd.min())) * 100
            s_ret = (seq.iloc[-1] / seq.iloc[0] - 1) * 100
            metrics[f"sleeve_return_pct_{sym}"] = float(s_ret)

    return metrics


# ═══════════════════════════════════════════════════════════════
# Rule Evaluator
# ═══════════════════════════════════════════════════════════════

def evaluate_rules(
    rules: list[dict],
    metrics: dict,
    metrics_history: dict | None = None,
) -> list[RiskDecision]:
    """Evaluate all rules against current metrics."""
    decisions = []

    for rule in rules:
        name = rule["name"]
        metric_name = rule["metric"]
        condition = rule["condition"]
        threshold = rule["threshold"]
        lookback_days = rule.get("lookback_days", 1)
        action = rule["action"]
        severity = rule.get("severity", "MEDIUM")
        desc = rule.get("description", "")
        symbol = rule.get("symbol")

        # Resolve metric — handle sleeve-specific metrics
        if symbol and metric_name == "sleeve_mdd_pct":
            resolved_metric = f"sleeve_mdd_pct_{symbol}"
        else:
            resolved_metric = metric_name

        current_value = metrics.get(resolved_metric, None)
        if current_value is None:
            # Metric not available
            decisions.append(RiskDecision(
                action="NO_ACTION",
                rule_name=name,
                description=f"{desc} (metric unavailable: {resolved_metric})",
                severity="LOW",
                metric_name=resolved_metric,
                metric_value=float("nan"),
                threshold=threshold,
                triggered=False,
            ))
            continue

        # Check condition
        if condition == "lt":
            triggered = current_value < threshold
        elif condition == "gt":
            triggered = current_value > threshold
        elif condition == "lte":
            triggered = current_value <= threshold
        elif condition == "gte":
            triggered = current_value >= threshold
        else:
            triggered = False

        # Lookback check: for lookback_days > 1, check consecutive triggers
        if triggered and lookback_days > 1 and metrics_history:
            history_key = f"_{resolved_metric}_history"
            hist = metrics_history.get(history_key) or metrics.get(history_key)
            if hist is not None and isinstance(hist, pd.Series):
                # Check last N days (N * 24 bars for 1h)
                n_bars = lookback_days * 24
                recent = hist.iloc[-n_bars:] if len(hist) >= n_bars else hist
                if condition == "lt":
                    consecutive = (recent < threshold).all()
                elif condition == "gt":
                    consecutive = (recent > threshold).all()
                else:
                    consecutive = True
                triggered = bool(consecutive)

        effective_action = action if triggered else "NO_ACTION"
        decisions.append(RiskDecision(
            action=effective_action,
            rule_name=name,
            description=desc,
            severity=severity if triggered else "LOW",
            metric_name=resolved_metric,
            metric_value=float(current_value),
            threshold=threshold,
            triggered=triggered,
        ))

    return decisions


def aggregate_decision(decisions: list[RiskDecision]) -> str:
    """Pick the most severe triggered action."""
    triggered = [d for d in decisions if d.triggered]
    if not triggered:
        return "NO_ACTION"
    return max(triggered, key=lambda d: d.severity_level).action


# ═══════════════════════════════════════════════════════════════
# Report Generator
# ═══════════════════════════════════════════════════════════════

def generate_report(
    metrics: dict,
    decisions: list[RiskDecision],
    final_action: str,
    mode: str = "dry-run",
) -> str:
    """Generate human-readable risk guard report."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"  RISK GUARD REPORT — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"  Mode: {mode.upper()}")
    lines.append("=" * 70)

    # Final decision
    action_emoji = {
        "NO_ACTION": "🟢",
        "WARNING": "🟡",
        "REDUCE_RISK_50": "🟠",
        "FLATTEN_ALL": "🔴",
    }
    emoji = action_emoji.get(final_action.split(":")[0], "⚪")
    lines.append(f"\n  ▶ FINAL DECISION: {emoji} {final_action}")

    # Metrics summary
    lines.append(f"\n  ── Key Metrics ──")
    display_metrics = {
        "rolling_sharpe_20d": "20D Rolling Sharpe",
        "running_mdd_pct": "Running MDD (%)",
        "_max_mdd_pct": "Max MDD (%)",
        "return_30d_pct": "30D Return (%)",
        "total_return_pct": "Total Return (%)",
        "full_period_sharpe": "Full Period Sharpe",
        "realized_slippage_bps": "Realized Slippage (bps)",
        "funding_drag_annualized_pct": "Funding Drag Ann. (%)",
    }
    for key, label in display_metrics.items():
        val = metrics.get(key)
        if val is not None and not isinstance(val, pd.Series):
            lines.append(f"    {label:30s}: {val:+.3f}")

    # Sleeve metrics
    sleeve_keys = [k for k in metrics if k.startswith("sleeve_")]
    if sleeve_keys:
        lines.append(f"\n  ── Sleeve Metrics ──")
        for k in sorted(sleeve_keys):
            label = k.replace("sleeve_", "").replace("_", " ").title()
            lines.append(f"    {label:30s}: {metrics[k]:+.2f}")

    # Rule evaluations
    lines.append(f"\n  ── Rule Evaluations ──")
    for d in sorted(decisions, key=lambda x: -x.severity_level):
        status = "🔴 TRIGGERED" if d.triggered else "✅ OK"
        lines.append(f"    [{d.severity:8s}] {d.rule_name:30s} {status}")
        if d.triggered:
            lines.append(f"              → {d.metric_name}={d.metric_value:.3f} "
                         f"{'<' if 'lt' in str(d.threshold) else '>'} {d.threshold} "
                         f"→ {d.action}")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# Replay Mode — Backtest Equity Replay
# ═══════════════════════════════════════════════════════════════

def run_replay(
    config: dict,
    replay_start: str | None = None,
    replay_end: str | None = None,
) -> dict:
    """
    Replay risk guard rules on historical backtest equity curves.

    Runs the ensemble backtest for the stress period, then walks through
    the equity bar-by-bar, evaluating rules at each point.
    """
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from qtrade.config import load_config
    from qtrade.backtest.run_backtest import run_symbol_backtest
    from qtrade.data.storage import load_klines

    replay_cfg = config.get("replay", {})
    strat_config_path = replay_cfg.get("strategy_config", "config/futures_ensemble_nw_tsmom.yaml")
    start = replay_start or replay_cfg.get("stress_period_start", "2022-07-01")
    end = replay_end or replay_cfg.get("stress_period_end", "2023-01-01")

    print(f"\n  🎬 Replay Mode: {start} → {end}")
    print(f"  Strategy config: {strat_config_path}")

    cfg = load_config(strat_config_path)
    market_type = cfg.market_type_str

    portfolio_cfg = config.get("portfolio", {})
    symbols = portfolio_cfg.get("symbols", ["ETHUSDT", "SOLUSDT"])
    weights = portfolio_cfg.get("weights", {"ETHUSDT": 0.54, "SOLUSDT": 0.46})
    strategies = portfolio_cfg.get("strategies", {})

    rules = config.get("rules", [])

    # Load ensemble strategies from the strategy config
    ens = getattr(cfg, '_raw', {}).get('ensemble', {}) if hasattr(cfg, '_raw') else {}
    ens_strats = ens.get('strategies', {})

    # Run per-symbol backtests
    sleeve_equities = {}
    for sym in symbols:
        data_path = cfg.data_dir / "binance" / market_type / cfg.market.interval / f"{sym}.parquet"
        if not data_path.exists():
            print(f"    ⚠️ Missing data for {sym}: {data_path}")
            continue

        # Determine strategy for this symbol
        strat_name = strategies.get(sym)
        strat_params = {}

        # Try ensemble config first
        if ens_strats and sym in ens_strats:
            strat_name = strat_name or ens_strats[sym]["name"]
            strat_params = ens_strats[sym].get("params", {})

        bt_cfg = cfg.to_backtest_dict(symbol=sym)
        bt_cfg["start"] = start
        bt_cfg["end"] = end
        if strat_name:
            bt_cfg["strategy_name"] = strat_name
        if strat_params:
            bt_cfg["strategy_params"] = strat_params

        # For the consensus config: debounce ON → disagree_weight=0
        if "disagree_weight" in bt_cfg.get("strategy_params", {}):
            bt_cfg["strategy_params"]["disagree_weight"] = 0.0

        try:
            res = run_symbol_backtest(sym, data_path, bt_cfg,
                                      strategy_name=strat_name,
                                      data_dir=cfg.data_dir)
            sleeve_equities[sym] = res.equity()
            print(f"    {sym}: {len(res.equity())} bars, "
                  f"ret={res.total_return_pct():+.1f}%")
        except Exception as e:
            print(f"    ❌ {sym} replay failed: {e}")

    if not sleeve_equities:
        return {"error": "No sleeve data available for replay"}

    # Combine into portfolio equity
    aligned = {}
    min_start = max(eq.index[0] for eq in sleeve_equities.values())
    max_end = min(eq.index[-1] for eq in sleeve_equities.values())
    for sym, eq in sleeve_equities.items():
        aligned[sym] = eq.loc[min_start:max_end]

    norm = {s: eq / eq.iloc[0] for s, eq in aligned.items()}
    port_norm = sum(norm[s] * weights.get(s, 0) for s in norm)
    portfolio_equity = port_norm * 10000  # Normalize to $10000

    # Walk through bar-by-bar and evaluate rules
    print(f"\n  📊 Replay: {len(portfolio_equity)} bars")
    print(f"  Evaluating {len(rules)} rules at each point...\n")

    trigger_timeline = []
    check_interval = 24  # Check every 24 bars (daily)

    for i in range(480, len(portfolio_equity), check_interval):  # Start after 20D warmup
        eq_slice = portfolio_equity.iloc[:i + 1]
        sleeve_slices = {s: aligned[s].iloc[:i + 1] for s in aligned}

        m = compute_metrics(eq_slice, sleeve_slices)
        decs = evaluate_rules(rules, m)
        final = aggregate_decision(decs)

        if final != "NO_ACTION":
            ts = portfolio_equity.index[i]
            triggered_rules = [d for d in decs if d.triggered]
            trigger_timeline.append({
                "timestamp": str(ts),
                "bar_index": i,
                "action": final,
                "triggered_rules": [d.rule_name for d in triggered_rules],
                "metrics": {
                    "rolling_sharpe_20d": m.get("rolling_sharpe_20d", 0),
                    "running_mdd_pct": m.get("running_mdd_pct", 0),
                    "return_30d_pct": m.get("return_30d_pct", 0),
                    "portfolio_value": float(eq_slice.iloc[-1]),
                },
            })

    # Compute MDD reduction from hypothetical flattening
    first_flatten = None
    for t in trigger_timeline:
        if t["action"] == "FLATTEN_ALL":
            first_flatten = t
            break

    mdd_without_guard = abs(float(
        ((portfolio_equity - portfolio_equity.expanding().max()) /
         portfolio_equity.expanding().max()).min()
    )) * 100

    mdd_with_guard = mdd_without_guard
    if first_flatten:
        flatten_idx = first_flatten["bar_index"]
        eq_before = portfolio_equity.iloc[:flatten_idx + 1]
        mdd_before_flatten = abs(float(
            ((eq_before - eq_before.expanding().max()) /
             eq_before.expanding().max()).min()
        )) * 100
        # After flatten, equity is flat (no further drawdown)
        mdd_with_guard = mdd_before_flatten

    # Summary
    print(f"  ═══ Replay Summary ═══")
    print(f"  Period: {start} → {end}")
    print(f"  Total bars: {len(portfolio_equity)}")
    print(f"  Total return: {(portfolio_equity.iloc[-1] / portfolio_equity.iloc[0] - 1) * 100:+.2f}%")
    print(f"  MDD without guard: {mdd_without_guard:.2f}%")
    print(f"  MDD with guard:    {mdd_with_guard:.2f}%")
    print(f"  MDD reduction:     {mdd_without_guard - mdd_with_guard:.2f}%")
    print(f"\n  Trigger Timeline ({len(trigger_timeline)} events):")

    for t in trigger_timeline[:20]:  # Show first 20
        emoji = "🔴" if "FLATTEN" in t["action"] else "🟡" if "WARNING" in t["action"] else "🟠"
        sr = t["metrics"].get("rolling_sharpe_20d", 0)
        mdd = t["metrics"].get("running_mdd_pct", 0)
        print(f"    {emoji} {t['timestamp'][:19]} → {t['action']:20s} "
              f"(SR={sr:+.2f}, MDD={mdd:.1f}%, rules={t['triggered_rules']})")

    if len(trigger_timeline) > 20:
        print(f"    ... and {len(trigger_timeline) - 20} more events")

    return {
        "period": f"{start} → {end}",
        "total_bars": len(portfolio_equity),
        "total_return_pct": round((portfolio_equity.iloc[-1] / portfolio_equity.iloc[0] - 1) * 100, 2),
        "mdd_without_guard": round(mdd_without_guard, 2),
        "mdd_with_guard": round(mdd_with_guard, 2),
        "mdd_reduction": round(mdd_without_guard - mdd_with_guard, 2),
        "n_triggers": len(trigger_timeline),
        "first_flatten": first_flatten,
        "trigger_timeline": trigger_timeline[:50],  # Limit for JSON
    }


# ═══════════════════════════════════════════════════════════════
# Dry-Run Mode — Live Data Check
# ═══════════════════════════════════════════════════════════════

def run_dry_run(
    config: dict,
    equity_dir: str | None = None,
) -> dict:
    """
    Dry-run mode: read recent equity data and evaluate rules.

    In production, reads from SQLite trading DB or equity CSV.
    For now, falls back to the most recent backtest equity.
    """
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    portfolio_cfg = config.get("portfolio", {})
    symbols = portfolio_cfg.get("symbols", ["ETHUSDT", "SOLUSDT"])
    weights = portfolio_cfg.get("weights", {})
    rules = config.get("rules", [])

    # Try to load equity from live DB or backtest output
    sleeve_equities = {}

    # Attempt 1: Load from SQLite trading DB
    if equity_dir:
        eq_path = Path(equity_dir)
        for sym in symbols:
            csv = eq_path / f"equity_{sym}.csv"
            if csv.exists():
                df = pd.read_csv(csv, index_col=0, parse_dates=True)
                if "equity" in df.columns:
                    sleeve_equities[sym] = df["equity"]

    # Attempt 2: Fallback — run a quick backtest for the last 60 days
    if not sleeve_equities:
        print("  ⚠️ No live equity data found. Falling back to recent backtest...")
        try:
            from qtrade.config import load_config
            from qtrade.backtest.run_backtest import run_symbol_backtest

            strat_cfg_path = config.get("replay", {}).get(
                "strategy_config", "config/futures_ensemble_nw_tsmom.yaml"
            )
            cfg = load_config(strat_cfg_path)
            market_type = cfg.market_type_str

            ens = getattr(cfg, '_raw', {}).get('ensemble', {}) if hasattr(cfg, '_raw') else {}
            ens_strats = ens.get('strategies', {})

            for sym in symbols:
                data_path = (cfg.data_dir / "binance" / market_type /
                             cfg.market.interval / f"{sym}.parquet")
                if not data_path.exists():
                    continue

                bt_cfg = cfg.to_backtest_dict(symbol=sym)
                # Only last ~90 days
                bt_cfg["start"] = "2025-11-01"

                strat_name = None
                strat_params = {}
                if ens_strats and sym in ens_strats:
                    strat_name = ens_strats[sym]["name"]
                    strat_params = ens_strats[sym].get("params", {})
                    bt_cfg["strategy_name"] = strat_name
                    bt_cfg["strategy_params"] = strat_params

                # Debounce ON
                if "disagree_weight" in bt_cfg.get("strategy_params", {}):
                    bt_cfg["strategy_params"]["disagree_weight"] = 0.0

                res = run_symbol_backtest(sym, data_path, bt_cfg,
                                          strategy_name=strat_name,
                                          data_dir=cfg.data_dir)
                sleeve_equities[sym] = res.equity()

        except Exception as e:
            print(f"    ❌ Backtest fallback failed: {e}")

    if not sleeve_equities:
        return {"error": "No equity data available", "action": "NO_ACTION"}

    # Combine portfolio equity
    aligned = {}
    min_start = max(eq.index[0] for eq in sleeve_equities.values())
    max_end = min(eq.index[-1] for eq in sleeve_equities.values())
    for sym, eq in sleeve_equities.items():
        aligned[sym] = eq.loc[min_start:max_end]

    norm = {s: eq / eq.iloc[0] for s, eq in aligned.items()}
    port_norm = sum(norm[s] * weights.get(s, 0) for s in norm)
    portfolio_equity = port_norm * 10000

    # Compute metrics
    metrics = compute_metrics(portfolio_equity, aligned)

    # Evaluate rules
    decisions = evaluate_rules(rules, metrics)
    final_action = aggregate_decision(decisions)

    # Generate report
    report_text = generate_report(metrics, decisions, final_action, mode="dry-run")
    print(report_text)

    return {
        "mode": "dry-run",
        "action": final_action,
        "metrics": {k: v for k, v in metrics.items() if not k.startswith("_")
                    and not isinstance(v, pd.Series)},
        "decisions": [d.to_dict() for d in decisions],
        "triggered": [d.to_dict() for d in decisions if d.triggered],
    }


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Risk Guard — automated risk monitoring & kill switch"
    )
    parser.add_argument(
        "--config", "-c",
        default="config/risk_guard_alt_ensemble.yaml",
        help="Path to risk guard YAML config",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Dry-run mode: read data, output decisions (no actions)")
    parser.add_argument("--replay", action="store_true",
                        help="Replay mode: simulate rules on historical data")
    parser.add_argument("--replay-start", default=None,
                        help="Override replay start date (YYYY-MM-DD)")
    parser.add_argument("--replay-end", default=None,
                        help="Override replay end date (YYYY-MM-DD)")
    parser.add_argument("--equity-dir", default=None,
                        help="Directory containing equity CSV files (dry-run mode)")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Risk Guard — Alt Ensemble (ETH+SOL TSMOM)              ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Config: {args.config}")
    print(f"  Rules:  {len(config.get('rules', []))}")

    if args.replay:
        result = run_replay(config, args.replay_start, args.replay_end)
    elif args.dry_run:
        result = run_dry_run(config, args.equity_dir)
    else:
        print("\n  ⚠️ No mode specified. Use --dry-run or --replay.")
        print("  Example:")
        print("    python scripts/risk_guard.py -c config/risk_guard_alt_ensemble.yaml --replay")
        print("    python scripts/risk_guard.py -c config/risk_guard_alt_ensemble.yaml --dry-run")
        return

    # Save results
    out_cfg = config.get("output", {})
    out_dir = Path(out_cfg.get("report_dir", "reports/risk_guard"))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_str = "replay" if args.replay else "dry_run"
    out_path = out_dir / f"{mode_str}_{ts}"
    out_path.mkdir(parents=True, exist_ok=True)

    # Clean result for JSON (remove non-serializable)
    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()
                    if not isinstance(v, (pd.Series, pd.DataFrame))}
        elif isinstance(obj, list):
            return [_clean(i) for i in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, pd.Timestamp):
            return str(obj)
        return obj

    json_path = out_path / "decision.json"
    with open(json_path, "w") as f:
        json.dump(_clean(result), f, indent=2, default=str)

    print(f"\n  ✅ Decision saved: {json_path}")

    # Also save human-readable report if in dry-run mode
    if args.dry_run and "metrics" in result:
        metrics = {k: v for k, v in result.get("metrics", {}).items()}
        decisions = [RiskDecision(**d) for d in result.get("decisions", [])]
        final = result.get("action", "NO_ACTION")
        report = generate_report(metrics, decisions, final, "dry-run")
        report_path = out_path / "report.txt"
        with open(report_path, "w") as f:
            f.write(report)
        print(f"  ✅ Report saved:   {report_path}")


if __name__ == "__main__":
    main()
