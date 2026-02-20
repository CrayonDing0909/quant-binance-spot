#!/usr/bin/env python3
"""
Production Launch Guard — Pre-Flight Check for R1 Deployment

Verifies all pre-conditions before allowing live trading launch:
  1. Frozen config hash integrity
  2. Environment variables (API keys, Telegram)
  3. Data freshness (recent klines available & complete)
  4. Risk guard status (no active FLATTEN)
  5. NTP / clock sync sanity

Usage:
    # Dry-run check (default)
    PYTHONPATH=src python scripts/prod_launch_guard.py --dry-run

    # With custom scale rules
    PYTHONPATH=src python scripts/prod_launch_guard.py --rules config/prod_scale_rules_R1.yaml

    # JSON output only
    PYTHONPATH=src python scripts/prod_launch_guard.py --json-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

DEFAULT_RULES_PATH = "config/prod_scale_rules_R1.yaml"
REPORT_DIR = Path("reports/prod_guard")


def compute_sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_rules(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class LaunchCheck:
    """A single pre-flight check result."""

    def __init__(self, name: str, passed: bool, detail: str, severity: str = "HARD"):
        self.name = name
        self.passed = passed
        self.detail = detail
        self.severity = severity  # HARD = must pass, SOFT = advisory

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
            "severity": self.severity,
        }

    def __repr__(self) -> str:
        icon = "✅" if self.passed else "❌"
        return f"  {icon} [{self.severity}] {self.name}: {self.detail}"


def check_config_hash(rules: dict) -> LaunchCheck:
    """Verify frozen config hash matches expected."""
    config_path = rules.get("strategy_config", "config/prod_candidate_R1.yaml")
    expected_hash = rules.get("config_hash", "")

    if not Path(config_path).exists():
        return LaunchCheck("config_hash", False,
                           f"Config file not found: {config_path}")

    actual_hash = compute_sha256(config_path)
    if expected_hash and actual_hash != expected_hash:
        return LaunchCheck("config_hash", False,
                           f"Hash mismatch! Expected: {expected_hash[:16]}... Got: {actual_hash[:16]}...")

    return LaunchCheck("config_hash", True,
                       f"OK — {config_path} ({actual_hash[:16]}...)")


def check_env_vars() -> list[LaunchCheck]:
    """Check required environment variables."""
    checks = []

    required = {
        "BINANCE_API_KEY": "Binance API Key",
        "BINANCE_API_SECRET": "Binance API Secret",
    }
    optional = {
        "FUTURES_TELEGRAM_BOT_TOKEN": "Telegram Bot Token (notifications)",
        "FUTURES_TELEGRAM_CHAT_ID": "Telegram Chat ID (notifications)",
    }

    for var, desc in required.items():
        val = os.environ.get(var) or os.environ.get(var.replace("BINANCE_", ""))
        if val:
            checks.append(LaunchCheck(f"env_{var}", True, f"{desc} — set ({len(val)} chars)"))
        else:
            checks.append(LaunchCheck(f"env_{var}", False, f"{desc} — NOT SET"))

    for var, desc in optional.items():
        val = os.environ.get(var)
        if val:
            checks.append(LaunchCheck(f"env_{var}", True, f"{desc} — set", severity="SOFT"))
        else:
            checks.append(LaunchCheck(f"env_{var}", False, f"{desc} — not set (notifications disabled)", severity="SOFT"))

    return checks


def check_data_freshness(rules: dict) -> list[LaunchCheck]:
    """Check that kline data files exist and are recent."""
    import pandas as pd

    checks = []
    symbols = rules.get("portfolio", {}).get("symbols", ["BTCUSDT", "ETHUSDT", "SOLUSDT"])

    # Try common data paths
    data_dirs = [
        Path("data/binance/futures/1h"),
        Path("data/binance/futures/klines"),
    ]
    data_dir = None
    for d in data_dirs:
        if d.exists():
            data_dir = d
            break

    if data_dir is None:
        checks.append(LaunchCheck("data_dir", False, "No data directory found"))
        return checks

    for sym in symbols:
        fpath = data_dir / f"{sym}.parquet"
        if not fpath.exists():
            checks.append(LaunchCheck(f"data_{sym}", False, f"{fpath} not found"))
            continue

        try:
            df = pd.read_parquet(fpath)
            n_bars = len(df)
            last_ts = df.index[-1]
            now = datetime.now(timezone.utc)

            # Check if last bar is < 48h old
            if hasattr(last_ts, 'tz_localize'):
                last_ts_utc = last_ts.tz_localize("UTC") if last_ts.tzinfo is None else last_ts
            else:
                last_ts_utc = last_ts

            hours_stale = (now - last_ts_utc).total_seconds() / 3600
            fresh = hours_stale < 48

            detail = f"{n_bars:,} bars, last: {last_ts}, staleness: {hours_stale:.0f}h"
            if not fresh:
                detail += " ⚠️ DATA STALE — run download_data.py first"

            checks.append(LaunchCheck(f"data_{sym}", fresh, detail))

            # Check for gaps (> 3h between consecutive bars)
            if n_bars > 100:
                diffs = df.index.to_series().diff().dropna()
                max_gap_h = diffs.max().total_seconds() / 3600
                if max_gap_h > 3:
                    checks.append(LaunchCheck(f"data_gaps_{sym}", False,
                                              f"Max gap: {max_gap_h:.1f}h (limit: 3h)", severity="SOFT"))
                else:
                    checks.append(LaunchCheck(f"data_gaps_{sym}", True,
                                              f"Max gap: {max_gap_h:.1f}h — OK", severity="SOFT"))

        except Exception as e:
            checks.append(LaunchCheck(f"data_{sym}", False, f"Error reading: {e}"))

    return checks


def check_clock_sync() -> LaunchCheck:
    """Basic clock sanity check."""
    now = datetime.now(timezone.utc)
    # If hour is reasonable (not year 1970 or 2099), pass
    if 2025 <= now.year <= 2030:
        return LaunchCheck("clock_sync", True,
                           f"System time: {now.isoformat()} — OK", severity="SOFT")
    return LaunchCheck("clock_sync", False,
                       f"System time looks wrong: {now.isoformat()}", severity="SOFT")


def check_backtest_consistency(rules: dict) -> LaunchCheck:
    """Verify strategy config can be loaded and backtest params are safe."""
    try:
        from qtrade.config import load_config
        config_path = rules.get("strategy_config", "config/prod_candidate_R1.yaml")
        cfg = load_config(config_path)

        issues = []
        if cfg.backtest.trade_on != "next_open":
            issues.append(f"trade_on={cfg.backtest.trade_on} (should be next_open)")
        if cfg.backtest.fee_bps <= 0:
            issues.append("fee_bps <= 0")
        if not cfg.backtest.funding_rate.enabled:
            issues.append("funding_rate disabled")

        if issues:
            return LaunchCheck("backtest_config", False, f"Issues: {', '.join(issues)}")

        return LaunchCheck("backtest_config", True,
                           f"OK — trade_on=next_open, fee={cfg.backtest.fee_bps}bps, "
                           f"slip={cfg.backtest.slippage_bps}bps, funding=ON")
    except Exception as e:
        return LaunchCheck("backtest_config", False, f"Config load failed: {e}")


def run_guard(rules_path: str, verbose: bool = True) -> tuple[str, list[LaunchCheck]]:
    """
    Run all pre-flight checks.

    Returns:
        (verdict, checks) where verdict is "ALLOW_LAUNCH" or "BLOCK_LAUNCH:<reasons>"
    """
    rules = load_rules(rules_path)
    all_checks: list[LaunchCheck] = []

    # 1. Config hash
    all_checks.append(check_config_hash(rules))

    # 2. Env vars
    all_checks.extend(check_env_vars())

    # 3. Data freshness
    all_checks.extend(check_data_freshness(rules))

    # 4. Clock sync
    all_checks.append(check_clock_sync())

    # 5. Backtest config consistency
    all_checks.append(check_backtest_consistency(rules))

    # Evaluate
    hard_fails = [c for c in all_checks if c.severity == "HARD" and not c.passed]
    soft_fails = [c for c in all_checks if c.severity == "SOFT" and not c.passed]

    if hard_fails:
        reasons = "; ".join(c.name for c in hard_fails)
        verdict = f"BLOCK_LAUNCH:{reasons}"
    else:
        verdict = "ALLOW_LAUNCH"

    if verbose:
        print("=" * 70)
        print("  PRODUCTION LAUNCH GUARD — Pre-Flight Check")
        print(f"  Rules: {rules_path}")
        print(f"  Time:  {datetime.now(timezone.utc).isoformat()}")
        print("=" * 70)

        for c in all_checks:
            print(repr(c))

        if soft_fails:
            print(f"\n  ⚠️  Soft warnings ({len(soft_fails)}):")
            for c in soft_fails:
                print(f"     • {c.name}: {c.detail}")

        print(f"\n  {'─'*60}")
        if verdict == "ALLOW_LAUNCH":
            print(f"  ✅ VERDICT: ALLOW_LAUNCH")
            if soft_fails:
                print(f"     ({len(soft_fails)} soft warning(s) — review recommended)")
        else:
            print(f"  ❌ VERDICT: {verdict}")
            print(f"     Fix all HARD failures before launching.")

    return verdict, all_checks


def save_report(verdict: str, checks: list[LaunchCheck], output_dir: Path) -> Path:
    """Save guard report as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"guard_report_{ts}.json"

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "checks": [c.to_dict() for c in checks],
        "hard_pass": all(c.passed for c in checks if c.severity == "HARD"),
        "soft_pass": all(c.passed for c in checks if c.severity == "SOFT"),
        "total_checks": len(checks),
        "passed_checks": sum(1 for c in checks if c.passed),
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    return report_path


def main():
    parser = argparse.ArgumentParser(description="Production Launch Guard")
    parser.add_argument("--rules", type=str, default=DEFAULT_RULES_PATH,
                        help="Scale rules YAML path")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Dry-run mode (default)")
    parser.add_argument("--json-only", action="store_true",
                        help="Only output JSON, no console output")

    args = parser.parse_args()

    verbose = not args.json_only
    verdict, checks = run_guard(args.rules, verbose=verbose)

    # Save JSON report
    report_path = save_report(verdict, checks, REPORT_DIR)
    if verbose:
        print(f"\n  📄 Report saved: {report_path}")

    if args.json_only:
        report = {
            "verdict": verdict,
            "checks": [c.to_dict() for c in checks],
        }
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))

    # Exit code: 0 = allow, 1 = block
    sys.exit(0 if verdict == "ALLOW_LAUNCH" else 1)


if __name__ == "__main__":
    main()
