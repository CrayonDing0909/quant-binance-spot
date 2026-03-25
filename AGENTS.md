# Agent Operating Guide

This file is the shortest operator guide for driving the repo's agent workflow.

## Default Rule

- New research or unclear direction: use `@orchestrator` or `/start-research`
- Existing strategy feels broken but root cause unclear: use `@portfolio-strategist` or `/diagnose-strategy`
- Resume paused research: use `/resume-task`
- Direct specialist work only after the research packet clearly says what to do next

## Important Mental Model

- The system is **foreground-autonomous**, not a background daemon
- During one active chat, the orchestrator should run as far as it can
- After the chat ends, continuity lives in `tasks/active/*.yaml`
- To continue later, reference the `task_id` or use `/resume-task`

## Which Agent to Use

| Need | Default agent |
|------|----------------|
| Research intake / routing / resume | `@orchestrator` |
| Baseline weakness / complement thesis | `@portfolio-strategist` |
| EDA / alpha research / proposal | `@alpha-researcher` |
| Implementation / configs / backtests / `.cursor` updates | `@quant-developer` |
| Validation verdict | `@quant-researcher` |
| Risk / launch approval | `@risk-manager` |
| Deploy / Oracle / cron / tmux | `@devops` |

## New Strategy Lifecycle

1. `/start-research`
2. Portfolio framing
3. Alpha research and falsification
4. Quant developer implementation in `config/research_*.yaml`
5. Quant researcher validation
6. Risk review
7. Freeze and deploy

Reference files:
- `docs/CURSOR_WORKFLOW.md`
- `docs/ORCHESTRATION_MVP.md`
- `.cursor/skills/dev/new-strategy-lifecycle.md`

## What a Good Handoff Should Include

Every completed research packet should make the next step obvious:

- `Current status`
- `Key decision`
- `Next recommended action`
- `Recommended agent`
- `Primary files / artifacts`
- `Open risks / blockers`
