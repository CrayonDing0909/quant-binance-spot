#!/usr/bin/env python3
"""
Auto-generate docs/CLI_REFERENCE.md by scanning scripts/ and config/.

Usage:
    PYTHONPATH=src python scripts/gen_cli_reference.py

This replaces the old manually-maintained CLI_REFERENCE.md.
Run this whenever scripts or configs are added/removed.
"""
from __future__ import annotations

import ast
import re
import sys
from datetime import date, timezone, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# 1. Script scanning
# ---------------------------------------------------------------------------

# Categorize scripts by prefix/purpose
SCRIPT_CATEGORIES: dict[str, tuple[str, list[str]]] = {
    "core": (
        "Core Workflow (by typical execution order)",
        [
            "download_data.py",
            "download_oi_data.py",
            "run_backtest.py",
            "run_portfolio_backtest.py",
            "run_walk_forward.py",
            "run_cpcv.py",
            "validate.py",
            "validate_live_consistency.py",
            "prod_launch_guard.py",
            "run_live.py",
            "run_websocket.py",
        ],
    ),
    "optimize": (
        "Optimization & Research",
        [
            "optimize_params.py",
            "run_hyperopt.py",
            "run_experiment_matrix.py",
            "run_funding_basis_research.py",
            "run_mr_research.py",
            "run_oi_bb_rv_research.py",
            "validate_overlay_falsification.py",
            "build_universe.py",
            "run_symbol_governance_review.py",
        ],
    ),
    "ops": (
        "Operations & Monitoring",
        [
            "run_telegram_bot.py",
            "query_db.py",
            "health_check.py",
            "daily_report.py",
            "prod_report.py",
            "monitor_alpha_decay.py",
            "risk_guard.py",
        ],
    ),
    "infra": (
        "Infrastructure",
        [
            "deploy_oracle.sh",
            "setup_cron.sh",
            "setup_swap.sh",
        ],
    ),
}


def _extract_py_description(path: Path) -> str:
    """Extract description from a Python script.

    Priority:
    1. argparse ArgumentParser description=
    2. Module docstring first line
    3. "(no description)"
    """
    text = path.read_text(encoding="utf-8", errors="replace")

    # Try argparse description (simple string, not f-string)
    m = re.search(r'description=["\']([^"\']+)["\']', text)
    if m:
        return m.group(1).strip()

    # Try module docstring
    try:
        tree = ast.parse(text)
        docstring = ast.get_docstring(tree)
        if docstring:
            first_line = docstring.strip().split("\n")[0]
            return first_line
    except SyntaxError:
        pass

    return "(no description)"


def _extract_sh_description(path: Path) -> str:
    """Extract description from a shell script comment header."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in lines[1:10]:  # skip shebang, look at next lines
        stripped = line.lstrip("# ").strip()
        # Skip empty lines and separator lines (═══, ===, ---, ***)
        if not stripped:
            continue
        if all(c in "═=─-*~" for c in stripped):
            continue
        return stripped
    return "(no description)"


def _extract_description(path: Path) -> str:
    if path.suffix == ".py":
        return _extract_py_description(path)
    elif path.suffix == ".sh":
        return _extract_sh_description(path)
    return "(no description)"


# ---------------------------------------------------------------------------
# 2. Config scanning
# ---------------------------------------------------------------------------

def _classify_config(name: str) -> str:
    """Classify config file by prefix."""
    if name.startswith("prod_live"):
        return "production"
    if name.startswith("prod_candidate"):
        return "production"
    if name.startswith("prod_scale"):
        return "production"
    if name.startswith("risk_guard"):
        return "production"
    if name.startswith("futures_"):
        return "strategy"
    if name.startswith("research_"):
        return "research"
    if name == "validation.yaml":
        return "utility"
    return "other"


CONFIG_CATEGORY_LABELS = {
    "production": "Production (active on Oracle Cloud)",
    "strategy": "Strategy Definitions",
    "research": "Research (active experiments)",
    "utility": "Utility",
    "other": "Other",
}


# ---------------------------------------------------------------------------
# 3. Source module map (static, compact)
# ---------------------------------------------------------------------------

MODULE_MAP = """\
src/qtrade/
├── config.py              ← AppConfig dataclass, load_config()
├── strategy/              ← Strategy implementations
│   ├── base.py            ← StrategyContext (market_type, direction, signal_delay)
│   ├── tsmom_strategy.py  ← Active production strategy (TSMOM EMA)
│   └── exit_rules.py      ← SL/TP/Adaptive SL
├── backtest/
│   ├── run_backtest.py    ← BacktestResult dataclass, run_symbol_backtest()
│   ├── costs.py           ← Funding Rate + Volume Slippage cost model
│   └── metrics.py         ← Performance metrics + Long/Short analysis
├── live/
│   ├── base_runner.py     ← BaseRunner ABC (14 shared safety mechanisms)
│   ├── runner.py          ← LiveRunner (Polling mode)
│   ├── websocket_runner.py← WebSocketRunner (Event-driven, recommended)
│   └── signal_generator.py← SignalResult dataclass
├── validation/            ← WFA, DSR, PBO, CPCV, IC monitor
├── data/                  ← Multi-source: Binance/yfinance/ccxt
├── risk/                  ← Position sizing, Kelly, Monte Carlo
├── monitor/               ← Health check, Telegram, notifier
└── utils/                 ← Logging, security, time tools"""


# ---------------------------------------------------------------------------
# 4. Render
# ---------------------------------------------------------------------------

PROD_CONFIG = "prod_live_R3C_E3.yaml"


def _read_production_summary() -> str:
    """Read current production config and summarize key fields."""
    cfg_path = PROJECT_ROOT / "config" / PROD_CONFIG
    if not cfg_path.exists():
        return f"({PROD_CONFIG} not found)"

    text = cfg_path.read_text(encoding="utf-8", errors="replace")
    text_lines = text.splitlines()

    # Extract symbols list
    symbols: list[str] = []
    in_symbols = False
    for line in text_lines:
        stripped = line.strip()
        if stripped.startswith("symbols:"):
            in_symbols = True
            continue
        if in_symbols:
            if stripped.startswith("- "):
                symbols.append(stripped.lstrip("- ").strip())
            else:
                break

    # Extract simple key-value fields
    def _find(key: str) -> str:
        for line in text_lines:
            s = line.strip()
            if s.startswith(f"{key}:"):
                val = s.split(":", 1)[1].strip().strip('"').strip("'")
                if val:
                    return val
        return "?"

    strategy_name = _find("name")
    interval = _find("interval")
    leverage = _find("leverage")
    margin_type = _find("margin_type")
    direction = _find("direction")

    lines = [
        f"config: {PROD_CONFIG}",
        f"strategy: {strategy_name}",
        f"symbols: [{', '.join(symbols)}] ({len(symbols)} coins)",
        f"interval: {interval}",
        f"leverage: {leverage}x {margin_type}",
        f"direction: {direction}",
    ]
    return "\n".join(lines)


def generate() -> str:
    today = date.today().isoformat()
    parts: list[str] = []

    # Header
    parts.append(f"# Project Map & CLI Reference\n")
    parts.append(f"> **Auto-generated**: {today} by `scripts/gen_cli_reference.py`")
    parts.append(f"> **Production config**: `config/{PROD_CONFIG}`")
    parts.append(f"> **Strategy template**: `config/futures_tsmom.yaml` (TSMOM EMA base definition)")
    parts.append(f">")
    parts.append(f"> Re-generate: `PYTHONPATH=src python scripts/gen_cli_reference.py`")
    parts.append("")

    # -------------------------------------------------------------------
    # Scripts section
    # -------------------------------------------------------------------
    parts.append("---\n")
    parts.append("## Scripts\n")

    scripts_dir = PROJECT_ROOT / "scripts"
    all_categorized: set[str] = set()
    for _cat, (label, names) in SCRIPT_CATEGORIES.items():
        all_categorized.update(names)

    for _cat, (label, names) in SCRIPT_CATEGORIES.items():
        parts.append(f"### {label}\n")
        parts.append("| Script | Description |")
        parts.append("|--------|-------------|")
        for name in names:
            path = scripts_dir / name
            if path.exists():
                desc = _extract_description(path)
                parts.append(f"| `{name}` | {desc} |")
            # Skip non-existent scripts silently (no ghost entries)
        parts.append("")

    # Uncategorized scripts
    uncategorized = []
    for p in sorted(scripts_dir.glob("*.py")) + sorted(scripts_dir.glob("*.sh")):
        if p.name not in all_categorized and p.name != "gen_cli_reference.py":
            uncategorized.append(p)

    if uncategorized:
        parts.append("### Other\n")
        parts.append("| Script | Description |")
        parts.append("|--------|-------------|")
        for p in uncategorized:
            desc = _extract_description(p)
            parts.append(f"| `{p.name}` | {desc} |")
        parts.append("")

    # Archive note
    archive_dir = scripts_dir / "archive"
    if archive_dir.is_dir():
        count = sum(1 for _ in archive_dir.glob("*.py")) + sum(
            1 for _ in archive_dir.glob("*.sh")
        )
        if count:
            parts.append(
                f"> **Archive**: {count} completed research/migration scripts "
                f"in `scripts/archive/`. These are preserved for reference but "
                f"no longer part of the active workflow.\n"
            )

    # -------------------------------------------------------------------
    # Config section
    # -------------------------------------------------------------------
    parts.append("---\n")
    parts.append("## Configs\n")

    config_dir = PROJECT_ROOT / "config"
    configs_by_cat: dict[str, list[str]] = {}
    for p in sorted(config_dir.glob("*.yaml")):
        cat = _classify_config(p.name)
        configs_by_cat.setdefault(cat, []).append(p.name)

    for cat in ("production", "strategy", "research", "utility", "other"):
        names = configs_by_cat.get(cat, [])
        if not names:
            continue
        label = CONFIG_CATEGORY_LABELS[cat]
        parts.append(f"### {label}\n")
        parts.append("| Config | File |")
        parts.append("|--------|------|")
        for name in names:
            parts.append(f"| `{name}` | `config/{name}` |")
        parts.append("")

    # Archive note for configs
    config_archive = config_dir / "archive"
    if config_archive.is_dir():
        count = sum(1 for _ in config_archive.glob("*.yaml"))
        if count:
            parts.append(
                f"> **Archive**: {count} deprecated/completed research configs "
                f"in `config/archive/`. Preserved for git history reference.\n"
            )

    # -------------------------------------------------------------------
    # Module map
    # -------------------------------------------------------------------
    parts.append("---\n")
    parts.append("## Source Module Map\n")
    parts.append("```")
    parts.append(MODULE_MAP)
    parts.append("```\n")

    # -------------------------------------------------------------------
    # Production summary
    # -------------------------------------------------------------------
    parts.append("---\n")
    parts.append("## Current Production\n")
    parts.append("```yaml")
    parts.append(_read_production_summary())
    parts.append("```\n")

    # -------------------------------------------------------------------
    # Quick commands
    # -------------------------------------------------------------------
    parts.append("---\n")
    parts.append("## Quick Commands\n")
    pc = PROD_CONFIG
    parts.append("```bash")
    parts.append("# Backtest (production config)")
    parts.append(
        f"PYTHONPATH=src python scripts/run_backtest.py -c config/{pc}"
    )
    parts.append("")
    parts.append("# Walk-Forward validation")
    parts.append(
        f"PYTHONPATH=src python scripts/run_walk_forward.py -c config/{pc} --splits 6"
    )
    parts.append("")
    parts.append("# Full validation pipeline")
    parts.append(
        f"PYTHONPATH=src python scripts/validate.py -c config/{pc} --quick"
    )
    parts.append("")
    parts.append("# Download data")
    parts.append(
        f"PYTHONPATH=src python scripts/download_data.py -c config/{pc}"
    )
    parts.append("")
    parts.append("# Live trading (WebSocket, recommended)")
    parts.append(
        f"PYTHONPATH=src python scripts/run_websocket.py -c config/{pc} --real"
    )
    parts.append("")
    parts.append("# Dry-run test")
    parts.append(
        f"PYTHONPATH=src python scripts/run_live.py -c config/{pc} --real --dry-run --once"
    )
    parts.append("")
    parts.append("# Query trading DB")
    parts.append(
        f"PYTHONPATH=src python scripts/query_db.py -c config/{pc} summary"
    )
    parts.append("")
    parts.append("# Re-generate this file")
    parts.append("PYTHONPATH=src python scripts/gen_cli_reference.py")
    parts.append("```\n")

    # -------------------------------------------------------------------
    # Docs index
    # -------------------------------------------------------------------
    parts.append("---\n")
    parts.append("## Documentation Index\n")
    docs_dir = PROJECT_ROOT / "docs"
    parts.append("| Doc | Description |")
    parts.append("|-----|-------------|")
    for p in sorted(docs_dir.glob("*.md")):
        first_line = ""
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = line.strip().lstrip("#").strip()
            if stripped:
                first_line = stripped
                break
        parts.append(f"| [`{p.name}`](docs/{p.name}) | {first_line} |")

    archive_docs = docs_dir / "archive"
    if archive_docs.is_dir():
        count = sum(1 for _ in archive_docs.glob("*.md"))
        if count:
            parts.append("")
            parts.append(
                f"> **Archive**: {count} historical docs in `docs/archive/`."
            )
    parts.append("")

    return "\n".join(parts)


def main() -> None:
    output = generate()
    out_path = PROJECT_ROOT / "docs" / "CLI_REFERENCE.md"
    out_path.write_text(output, encoding="utf-8")
    print(f"Generated {out_path} ({len(output.splitlines())} lines)", file=sys.stderr)


if __name__ == "__main__":
    main()
