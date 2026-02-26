#!/usr/bin/env python3
"""
Auto-generate docs/DATA_STRATEGY_CATALOG.md by scanning src/qtrade/data/ and strategy/.

Usage:
    PYTHONPATH=src python scripts/gen_data_strategy_catalog.py

This provides a single source of truth for:
  - All data modules (K-line, derivatives, onchain, utility)
  - All registered strategies (with status detection from config)
  - Data download scripts

Run this whenever data modules or strategies are added/removed.
"""
from __future__ import annotations

import ast
import re
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DATA = PROJECT_ROOT / "src" / "qtrade" / "data"
SRC_STRATEGY = PROJECT_ROOT / "src" / "qtrade" / "strategy"
CONFIG_DIR = PROJECT_ROOT / "config"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# ---------------------------------------------------------------------------
# 1. Data module scanning
# ---------------------------------------------------------------------------

# Manual category assignment (stable, rarely changes)
DATA_MODULE_CATEGORIES: dict[str, str] = {
    "klines.py": "K 線",
    "binance_client.py": "K 線",
    "binance_futures_client.py": "K 線",
    "binance_vision.py": "K 線",
    "yfinance_client.py": "K 線",
    "ccxt_client.py": "K 線",
    "storage.py": "K 線",
    "open_interest.py": "衍生品",
    "funding_rate.py": "衍生品",
    "long_short_ratio.py": "衍生品",
    "taker_volume.py": "衍生品",
    "liquidation.py": "衍生品",
    "_derivatives_common.py": "衍生品",
    "onchain.py": "鏈上",
    "order_book.py": "即時",
    "multi_tf_loader.py": "工具",
    "quality.py": "工具",
}

CATEGORY_ORDER = ["K 線", "衍生品", "鏈上", "即時", "工具"]

# Modules to skip in the catalog
SKIP_MODULES = {"__init__.py"}


def _extract_module_docstring(path: Path) -> str:
    """Extract first line of module docstring."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        ds = ast.get_docstring(tree)
        if ds:
            return ds.strip().split("\n")[0]
    except SyntaxError:
        pass
    return "(no description)"


def _extract_public_exports(path: Path) -> list[str]:
    """Extract public class names and top-level function names (no underscore prefix)."""
    exports: list[str] = []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return exports
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            exports.append(node.name)
        elif isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            exports.append(f"`{node.name}()`")
    return exports


def scan_data_modules() -> list[dict]:
    """Scan src/qtrade/data/ and return module metadata."""
    modules: list[dict] = []
    for p in sorted(SRC_DATA.glob("*.py")):
        if p.name in SKIP_MODULES or p.name == "__pycache__":
            continue
        cat = DATA_MODULE_CATEGORIES.get(p.name, "其他")
        desc = _extract_module_docstring(p)
        exports = _extract_public_exports(p)
        modules.append({
            "file": p.name,
            "category": cat,
            "description": desc,
            "exports": exports,
        })
    return modules


# ---------------------------------------------------------------------------
# 2. Strategy scanning
# ---------------------------------------------------------------------------

def _extract_registered_names(path: Path) -> list[tuple[str, bool]]:
    """Extract (strategy_name, auto_delay) from @register_strategy decorators."""
    text = path.read_text(encoding="utf-8", errors="replace")
    results: list[tuple[str, bool]] = []
    for m in re.finditer(
        r'@register_strategy\(\s*"([^"]+)"(?:\s*,\s*auto_delay\s*=\s*(True|False))?\s*\)',
        text,
    ):
        name = m.group(1)
        auto_delay = m.group(2) != "False" if m.group(2) else True
        results.append((name, auto_delay))
    return results


def _detect_strategy_status(name: str, prod_configs: dict[str, list[str]]) -> str:
    """Detect strategy status from production configs."""
    for cfg_name, strategies in prod_configs.items():
        if name in strategies:
            if "candidate" in cfg_name:
                return "candidate"
            elif "prod_live" in cfg_name:
                return "production"
    return "implemented"


def _scan_prod_configs() -> dict[str, list[str]]:
    """Scan config/prod_*.yaml to find which strategies are in production."""
    result: dict[str, list[str]] = {}
    for p in sorted(CONFIG_DIR.glob("prod_*.yaml")):
        text = p.read_text(encoding="utf-8", errors="replace")
        # Extract strategy name from 'name:' field under 'strategy:' section
        names: list[str] = []
        # Simple heuristic: find all name: values that match known patterns
        for m in re.finditer(r'^\s+name:\s*(\S+)', text, re.MULTILINE):
            val = m.group(1).strip().strip('"').strip("'")
            if val and val not in ("cross", "isolated"):  # skip margin_type values
                names.append(val)
        # Also look for strategy names in routing/tier configs
        for m in re.finditer(r'strategy:\s*(\S+)', text):
            val = m.group(1).strip().strip('"').strip("'")
            if val and val not in ("cross", "isolated", "both", "long_only"):
                names.append(val)
        result[p.name] = list(set(names))
    return result


def scan_strategies() -> list[dict]:
    """Scan src/qtrade/strategy/ and return strategy metadata."""
    prod_configs = _scan_prod_configs()
    strategies: list[dict] = []
    for p in sorted(SRC_STRATEGY.glob("*.py")):
        if p.name.startswith("_") or p.name in ("__init__.py", "base.py", "filters.py", "exit_rules.py"):
            continue
        registered = _extract_registered_names(p)
        if not registered:
            continue
        desc = _extract_module_docstring(p)
        for name, auto_delay in registered:
            status = _detect_strategy_status(name, prod_configs)
            strategies.append({
                "name": name,
                "file": p.name,
                "description": desc,
                "auto_delay": auto_delay,
                "status": status,
            })
    return strategies


# ---------------------------------------------------------------------------
# 3. Download script scanning
# ---------------------------------------------------------------------------

DOWNLOAD_SCRIPTS = [
    ("download_data.py", "K 線 / FR / OI", "`-c`, `--interval`, `--funding-rate`, `--oi`, `--derivatives`"),
    ("download_oi_data.py", "Open Interest", "`--symbols`, `--provider`"),
    ("fetch_derivatives_data.py", "LSR / Taker Vol / CVD", "`--symbols`, `--metrics`, `--source`, `--coverage`"),
    ("fetch_liquidation_data.py", "清算數據", "`--symbols`, `--source`"),
    ("fetch_onchain_data.py", "鏈上 (DeFi Llama)", "`--source`, `--stablecoins`, `--yields`, `--protocols`"),
]


# ---------------------------------------------------------------------------
# 4. Render
# ---------------------------------------------------------------------------

STATUS_EMOJI = {
    "production": "🟢 生產中",
    "candidate": "🟡 候選/Paper",
    "implemented": "⚪ 已實作",
}


def generate() -> str:
    today = date.today().isoformat()
    parts: list[str] = []

    # Header
    parts.append("# Data & Strategy Catalog\n")
    parts.append(f"> **Auto-generated**: {today} by `scripts/gen_data_strategy_catalog.py`")
    parts.append(f"> **Do NOT edit by hand** — regenerate after adding data modules or strategies.")
    parts.append(f">")
    parts.append(f"> ```bash")
    parts.append(f"> PYTHONPATH=src python scripts/gen_data_strategy_catalog.py")
    parts.append(f"> ```")
    parts.append("")

    # -------------------------------------------------------------------
    # Data Modules
    # -------------------------------------------------------------------
    parts.append("---\n")
    parts.append("## Data Modules\n")
    parts.append("| Category | Module | Description | Key Exports |")
    parts.append("|----------|--------|-------------|-------------|")

    modules = scan_data_modules()
    for cat in CATEGORY_ORDER:
        cat_modules = [m for m in modules if m["category"] == cat]
        for m in cat_modules:
            exports_str = ", ".join(m["exports"][:5])  # limit to 5 exports
            if len(m["exports"]) > 5:
                exports_str += ", ..."
            parts.append(
                f"| **{m['category']}** | `{m['file']}` | {m['description']} | {exports_str} |"
            )

    # Any uncategorized
    other = [m for m in modules if m["category"] not in CATEGORY_ORDER]
    for m in other:
        exports_str = ", ".join(m["exports"][:5])
        parts.append(
            f"| **其他** | `{m['file']}` | {m['description']} | {exports_str} |"
        )
    parts.append("")

    # -------------------------------------------------------------------
    # Registered Strategies
    # -------------------------------------------------------------------
    parts.append("---\n")
    parts.append("## Registered Strategies\n")
    parts.append("| Status | Strategy Name | File | Description |")
    parts.append("|--------|---------------|------|-------------|")

    strategies = scan_strategies()
    # Sort: production first, then candidate, then implemented
    status_order = {"production": 0, "candidate": 1, "implemented": 2}
    strategies.sort(key=lambda s: (status_order.get(s["status"], 9), s["name"]))

    for s in strategies:
        emoji = STATUS_EMOJI.get(s["status"], "⚪")
        parts.append(
            f"| {emoji} | `{s['name']}` | `{s['file']}` | {s['description']} |"
        )
    parts.append("")
    parts.append(f"> **Total**: {len(strategies)} registered strategies")
    parts.append("")

    # -------------------------------------------------------------------
    # Download Scripts
    # -------------------------------------------------------------------
    parts.append("---\n")
    parts.append("## Download Scripts\n")
    parts.append("| Script | Data Type | Key Flags |")
    parts.append("|--------|-----------|-----------|")

    for script, dtype, flags in DOWNLOAD_SCRIPTS:
        path = SCRIPTS_DIR / script
        if path.exists():
            parts.append(f"| `{script}` | {dtype} | {flags} |")
    parts.append("")

    # -------------------------------------------------------------------
    # Data Storage Paths
    # -------------------------------------------------------------------
    parts.append("---\n")
    parts.append("## Data Storage Paths\n")
    parts.append("```")
    parts.append("data/")
    parts.append("├── binance/")
    parts.append("│   ├── futures/1h/              ← 主力 K 線（生產用）")
    parts.append("│   ├── futures/5m/              ← 微結構研究用")
    parts.append("│   ├── futures/15m/             ← overlay 研究用")
    parts.append("│   ├── futures/4h/              ← HTF 趨勢用")
    parts.append("│   ├── futures/1d/              ← 日線 regime 用")
    parts.append("│   ├── futures/open_interest/   ← OI（binance/coinglass/merged）")
    parts.append("│   ├── futures/derivatives/     ← 衍生品指標")
    parts.append("│   │   ├── lsr/                 ← Long/Short Ratio")
    parts.append("│   │   ├── top_lsr_account/     ← 大戶 L/S (帳戶)")
    parts.append("│   │   ├── top_lsr_position/    ← 大戶 L/S (持倉)")
    parts.append("│   │   ├── taker_vol_ratio/     ← Taker Buy/Sell Ratio")
    parts.append("│   │   └── cvd/                 ← Cumulative Volume Delta")
    parts.append("│   └── futures/liquidation/     ← 清算數據")
    parts.append("├── onchain/")
    parts.append("│   └── defillama/               ← DeFi Llama 鏈上數據")
    parts.append("└── ...")
    parts.append("```\n")

    return "\n".join(parts)


def main() -> None:
    output = generate()
    out_path = PROJECT_ROOT / "docs" / "DATA_STRATEGY_CATALOG.md"
    out_path.write_text(output, encoding="utf-8")
    print(
        f"Generated {out_path} ({len(output.splitlines())} lines)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
