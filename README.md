# quant-binance-spot

Binance crypto quant research, backtesting, validation, and live-trading system.

這個 repo 的現況已經不是早期的單一策略教學專案，而是以：

- `TSMOM + HTF Filter + LSR overlay` 為核心的 futures-first 組合策略
- research -> backtest -> validation -> risk -> deploy 的治理流程
- Cursor multi-agent / orchestrated research workflow

為主的工作系統。

## Current Snapshot

- Active candidate: `config/prod_candidate_simplified.yaml`
- Strategy: `HTF Filter v2 + LSR (Simplified v2)` meta-blend
- Symbols: `BTC`, `ETH`, `SOL`, `DOGE`, `AVAX`, `LINK`
- Market: Binance Futures
- Deployment: Oracle Cloud
- Observation window: `2026-03-04` to `2026-03-18`

## Two Ways To Read This Repo

### External Visitor

如果你只是想快速理解這個 repo 在做什麼，先看這 3 份：

- `docs/R3C_STRATEGY_OVERVIEW.md`: 當前策略脈絡與演化
- `docs/ALPHA_RESEARCH_MAP.md`: 已驗證方向、dead ends、研究前沿
- `docs/STRATEGY_DEV_PLAYBOOK_R2_1.md`: 方法學與 anti-bias guardrails

### Internal Operator

如果你是要真的操作這個 repo，不要把根 `README.md` 當成操作手冊。直接從這兩份開始：

- `docs/CURSOR_WORKFLOW.md`: agent workflow、slash commands、研究編排入口
- `docs/CLI_REFERENCE.md`: scripts / configs / 常用執行入口總表

補充：

- `docs/DATA_STRATEGY_CATALOG.md`: `src/qtrade/data/` 與 `src/qtrade/strategy/` 模組總覽
- `docs/ORCHESTRATION_MVP.md`: research orchestration spec
- `tasks/README.md`: task manifest 與 research ledger 說明

## Quick Start

### Environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python3.11 -m pip install -e .
```

Most scripts are run as:

```bash
PYTHONPATH=src python scripts/<script>.py -c config/<config>.yaml
```

### Common Commands

Backtest:

```bash
PYTHONPATH=src python scripts/run_backtest.py -c config/prod_candidate_simplified.yaml
```

Validation:

```bash
PYTHONPATH=src python scripts/validate.py -c config/prod_candidate_simplified.yaml
```

Live / websocket:

```bash
PYTHONPATH=src python scripts/run_live.py -c config/prod_candidate_simplified.yaml
PYTHONPATH=src python scripts/run_websocket.py -c config/prod_candidate_simplified.yaml
```

Tests:

```bash
python -m pytest tests/ -x -q --tb=short
```

## Repo Map

```text
quant-binance-spot/
├── config/              active strategy / prod / risk / validation configs
├── config/archive/      archived research configs
├── data/                market, derivatives, and alt-data caches
├── docs/                workflow, playbooks, references, research docs
├── reports/             backtest / validation / research outputs
├── scripts/             backtest, research, validation, ops, deploy scripts
├── src/qtrade/          core library
├── tasks/               orchestrated research task manifests
└── tests/               regression and safety guards
```

## Guardrails

This repo is intentionally hardened against common quant failure modes.

- No look-ahead: signals on `close[i]`, execution on `open[i+1]`
- Backtests should use `price=df["open"]`
- Backtest `StrategyContext(...)` should explicitly set `signal_delay=1`
- HTF resample flows must be causally aligned
- Do not shallow-copy nested config dicts
- Do not mutate caller-owned param dicts
- Do not silently swallow exceptions on critical trading paths

Full rules live in:

- `docs/STRATEGY_DEV_PLAYBOOK_R2_1.md`
- `docs/ALPHA_RESEARCH_MAP.md`
- `docs/STRATEGY_PORTFOLIO_GOVERNANCE.md`

## Important Notes

- `docs/CLI_REFERENCE.md` and `docs/DATA_STRATEGY_CATALOG.md` are auto-generated; do not edit by hand
- Research work should use `config/research_*.yaml`, not `prod_*`
- Completed research assets are archived in `config/archive/` and `scripts/archive/`
- The active operating model is futures-first; some older spot-era artifacts remain for history
