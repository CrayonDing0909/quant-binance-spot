# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Dashboard

```bash
# Launch human oversight dashboard
PYTHONPATH=src streamlit run scripts/dashboard.py
```

Shows: agent task status, research pipeline, live equity curve, recent trades, pending human decisions.

## gstack

Use `/browse` from gstack for all web browsing. Never use `mcp__claude-in-chrome__*` tools.

Available skills: `/office-hours`, `/plan-ceo-review`, `/plan-eng-review`, `/review`, `/investigate`, `/cso`, `/retro`, `/careful`, `/guard`, `/freeze`, `/unfreeze`, `/browse`, `/ship`, `/codex`, `/gstack-upgrade`

If skills aren't working: `cd .claude/skills/gstack && PATH="$HOME/.bun/bin:$PATH" ./setup`

Skip for this project: `/qa`, `/design-*`, `/plan-design-review`, `/canary`, `/benchmark`, `/land-and-deploy` (no web UI, Oracle Cloud deploy)

## Strategy Development Skills

These are interactive workflow skills for the full strategy lifecycle:

| Skill | When to use |
|---|---|
| `/check-direction` | **Before** starting research — scan failure registry and dead ends |
| `/research` | Full research workflow: idea → EDA → handoff gates → PR |
| `/implement-strategy` | After research GO — implementation checklist |
| `/validate-strategy` | After implementation — full validation pipeline (11 steps) |
| `/risk-check` | Pre-launch audit or weekly/monthly portfolio review |

Typical flow: `/check-direction` → `/research` → `/implement-strategy` → `/validate-strategy` → `/risk-check`

## Hooks

Pre-commit hook automatically runs `test_code_safety_guard.py` + `test_resample_shift_guard.py` before every commit. If tests fail, the commit is blocked.

## Strategy Development — GitHub PR Workflow

Every research cycle should have its own branch and PR:

```bash
git checkout -b research/<strategy-name>-$(date +%Y%m%d)
# develop: config/research_*.yaml + scripts/research_*.py + docs/research/*.md
git push origin HEAD
gh pr create --label research --title "research: <strategy name>" \
  --body "## Hypothesis\n...\n## Verdict\nGO / KEEP_BASELINE / FAIL"
```

- PR description = strategy proposal (agents can `gh pr view <num>` to recover context)
- Verdict goes in PR comment: `GO` → merge, `FAIL` → close (history preserved)
- Never rely on agent memory for research decisions — the PR is the permanent record

## Commands

```bash
# Environment
python3.11 -m venv .venv && source .venv/bin/activate
python3.11 -m pip install -e .

# All scripts follow this pattern
PYTHONPATH=src python scripts/<script>.py -c config/<config>.yaml

# Tests
python -m pytest tests/ -x -q --tb=short                    # all tests
python -m pytest tests/test_code_safety_guard.py -x -q       # safety guards only
python -m pytest tests/test_resample_shift_guard.py -x -q    # resample look-ahead guard
python -m pytest tests/test_specific.py::test_name -x -q     # single test

# Backtest / Validate
PYTHONPATH=src python scripts/run_backtest.py -c config/prod_candidate_simplified.yaml
PYTHONPATH=src python scripts/validate.py -c config/prod_candidate_simplified.yaml
PYTHONPATH=src python scripts/validate.py -c config/research_<name>.yaml --quick

# Sync live trading DB from Oracle Cloud (for dashboard)
bash scripts/sync_oracle_db.sh

# Auto-generated docs (never edit by hand — regenerate after adding/removing scripts or configs)
PYTHONPATH=src python scripts/gen_cli_reference.py
PYTHONPATH=src python scripts/gen_data_strategy_catalog.py
```

## Architecture

The system is a Python 3.11, vectorbt-based quant research and live-trading platform targeting Binance Futures.

**Core library** lives in `src/qtrade/`:
- `config.py` — `AppConfig` dataclass, loaded via `load_config(path)`. Use `cfg.to_backtest_dict()` to pass to backtests — never manually assemble dicts.
- `strategy/base.py` — `StrategyContext` dataclass. Every strategy receives `(df, ctx: StrategyContext, params: dict)` and returns `pd.Series` of positions in `[-1, 1]`.
- `backtest/run_backtest.py` — `BacktestResult` dataclass; returns `pf`, `pf_bh`, `stats`, `df`, `pos`, cost outputs, and `adjusted_stats`.
- `live/base_runner.py` — `BaseRunner` ABC with 14 safety mechanisms (circuit breaker, SL/TP, orphan cleanup, etc.). All new runner types must inherit it.
- `live/runner.py` — `LiveRunner` (polling/cron). `live/websocket_runner.py` — `WebSocketRunner` (event-driven).
- `live/signal_generator.py` — `SignalResult` dataclass for all signal outputs.
- `validation/` — Prado methods (WFA, CPCV, DSR), factor orthogonality, embargo enforcement.
- `data/` — Kline ingestion, aggregated trades, funding rates, LSR data.

**Agent workflow** (7 roles defined in `.cursor/agents/`):
```
Orchestrator → Portfolio Strategist → Alpha Researcher → Quant Developer
→ Quant Researcher → Risk Manager → DevOps → Production
```
Use `AGENTS.md` for which agent to use when. Use `/start-research`, `/resume-task`, or `/diagnose-strategy` as entry points.

**Research continuity** lives in `tasks/active/*.yaml`. The system is foreground-autonomous within a chat; use `/resume-task` to continue across chat boundaries.

## Critical Rules

### Look-Ahead Bias (enforced by CI)

- Signal at `close[i]` → execute at `open[i+1]`. Always `price=df["open"]` in `vbt.Portfolio.from_orders()`.
- `StrategyContext(...)` MUST include an explicit `signal_delay=` with a comment (1 for backtest, 0 for live).
- HTF resample MUST use `causal_resample_align()` from `qtrade.strategy.filters` — never `resample → reindex(ffill)` directly without a `.shift(1)`.

### Code Safety (enforced by `test_code_safety_guard.py`)

- Use `copy.deepcopy()` for any dict containing nested dicts (`cfg`, `base_cfg`, `overlay_params`).
- Never `.pop()` or inject keys into a caller's dict — use `.get()` and build new dicts.
- Critical trading paths (`_apply_position_sizing`, `execute_target_position`, `_process_signal`, circuit breakers) must log + alert on exception, never `except Exception: pass`.
- Config loading: use `.get(key)` with no default for fields that have dataclass defaults — never hardcode fallback values that differ from the dataclass.

### Strategy Output Convention

- Output positions in `[-1, 1]`. Do not clip inside the strategy function — `clip_positions_by_direction()` handles this downstream.
- `to_vbt_direction()` and `clip_positions_by_direction()` are shared utilities — never inline direction logic.

### Production Config Discipline

- Research: always use `config/research_*.yaml`. Never modify `config/prod_*` during research.
- Backtests must enable both `funding_rate.enabled: true` and `slippage_model.enabled: true`.
- Every backtest report must explicitly state: `Overlay: ON (mode=...)` or `Overlay: OFF`.

### Research Standards

- New research scripts must run factor orthogonality check via `marginal_information_ratio()`. R² > 0.50 → HARD STOP.
- New research scripts must enforce data embargo via `enforce_temporal_embargo()`. Config is in `config/validation.yaml` under `data_embargo`. Never modify embargo settings mid-research.
- On-chain data requires ≥70% coverage before use; document provider and latency.

### Hygiene

- `docs/CLI_REFERENCE.md` and `docs/DATA_STRATEGY_CATALOG.md` are auto-generated — never edit by hand.
- All `docs/` living docs must have a `> **Last updated**: YYYY-MM-DD` header; update it when you change code they describe.
- When research is complete: move `config/research_*.yaml` → `config/archive/` and one-off scripts → `scripts/archive/`, then regenerate `CLI_REFERENCE.md`.
- Do not delete files — move to `archive/` to keep git history navigable.
- Verify file paths exist before referencing them in docs or agent output.

### Live Trading

- Never use `--real` in tests or development — always `--paper` or `--dry-run`.
- Never write direct Binance API calls outside broker classes.
- Never restart the live runner without checking current positions first.
- `_apply_position_sizing` clips to `[-1, 1]`, not `[0, 1]`. Spot negative clip happens in `runner.run_once()`, not in position sizing.
- Production: `scripts/start_live.sh` on Oracle Cloud (canonical startup, `@reboot` cron).
- Active config: `prod_candidate_simplified.yaml` (6 symbols: BTC, ETH, SOL, DOGE, AVAX, LINK).

### Python Standards

- Absolute imports only: `from qtrade.config import AppConfig` — never relative imports.
- Logging in `src/qtrade/`: use `from qtrade.utils.log import get_logger` — never `import logging` or `print()`.
- `print()` only allowed in `scripts/` for CLI output.
- All strategies use `@register_strategy` decorator; signature `(df, ctx: StrategyContext, params: dict) -> pd.Series`.
- Type hints on all function signatures. Use `from __future__ import annotations`.
- Max line length: 120.

### Session-End Checklist

After completing tasks, check if you need to regenerate or update:

| You changed | Must update |
|---|---|
| `scripts/` files | `PYTHONPATH=src python scripts/gen_cli_reference.py` |
| `src/qtrade/data/` or `strategy/` modules | `PYTHONPATH=src python scripts/gen_data_strategy_catalog.py` |
| Strategy status (promoted/retired) | `.cursor/rules/project-overview.mdc` + `.cursor/agents/devops.md` |
| Any alpha research | `docs/ALPHA_RESEARCH_MAP.md` |
| New module/feature | `.cursor/skills/dev/feature-ownership-registry.md` |

Then append a summary to `.cursor/rules/recent-changes.mdc`.

## Key Reference Docs

- `docs/CURSOR_WORKFLOW.md` — agent workflow, slash commands, research orchestration
- `docs/CLI_REFERENCE.md` — all scripts and configs (auto-generated)
- `docs/STRATEGY_DEV_PLAYBOOK_R2_1.md` — full methodology, anti-bias guardrails, validation stack
- `docs/ALPHA_RESEARCH_MAP.md` — verified directions, dead ends, research frontier
- `docs/ACTIVE_BUY_ARCHITECTURE.md` — cross-asset product boundary (signal core here; watchlist/notify/broker in separate app repo)
