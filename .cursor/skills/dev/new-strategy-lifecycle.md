---
description: Canonical lifecycle for new strategy development from idea intake to deploy. Use when starting a new alpha, deciding the next specialist handoff, or asking how to move from research to production.
globs:
alwaysApply: false
---
# Skill: New Strategy Lifecycle

> Single reference for turning an idea into a production-ready strategy.

## When to Use

Use this skill when:
- a new alpha / filter / overlay / portfolio layer idea appears
- an existing strategy feels broken but you cannot tell whether the issue is signal / entry / exit / sizing
- you are unsure which agent should take the next step
- you need to move from research into implementation / validation / deploy
- you want to resume a strategy-development task after a chat break

## Default Path

1. **Intake via Orchestrator**
   - Start with `/start-research`
   - Goal: normalize the idea, create `tasks/active/*.yaml`, and route the work

0. **If root cause is unclear, diagnose first**
   - Use `/diagnose-strategy`
   - Goal: lock one primary experiment family and decide whether you are in `Loop A: Alpha Existence` or `Loop B: Trade Expression`

2. **Portfolio framing**
   - Use `@portfolio-strategist` logic to define:
     - baseline weakness
     - gap
     - archetype
     - integration mode (`Filter` / `Overlay` / `Standalone` / `Portfolio Layer`)
     - success / kill criteria

3. **Alpha research**
   - Use `@alpha-researcher` logic
   - Follow `.cursor/skills/alpha/portfolio-research-protocol.md`
   - Before GO, enforce `.cursor/skills/alpha/handoff-gates.md`
   - Output must include:
     - `Current status`
     - `Key decision`
     - `Next recommended action`
     - `Recommended agent`
     - `Primary files / artifacts`

4. **Implementation**
   - Hand off to `@quant-developer`
   - Keep experiments in `config/research_*.yaml`
   - Do **not** modify `prod_*` during research
   - If adding a critical module, update `.cursor/skills/dev/feature-ownership-registry.md`

5. **Validation**
   - Hand off to `@quant-researcher`
   - Run the full validation stack from `docs/STRATEGY_DEV_PLAYBOOK_R2_1.md`
   - Expect verdict: `GO_NEXT`, `NEED_MORE_WORK`, `KEEP_BASELINE`, or `FAIL`

6. **Risk / launch review**
   - Hand off to `@risk-manager`
   - Only proceed if the strategy is explicitly approved

7. **Freeze and deploy**
   - `@quant-developer`: freeze research config into `prod_*`
   - `@devops`: deploy to Oracle Cloud

## Agent Routing Cheat Sheet

| Situation | Default agent |
|----------|----------------|
| New idea, unclear path | `@orchestrator` |
| Need gap / archetype / integration mode | `@portfolio-strategist` |
| Need EDA / proposal / falsification | `@alpha-researcher` |
| Need code / config / backtest implementation | `@quant-developer` |
| Need full validation verdict | `@quant-researcher` |
| Need launch approval / risk action | `@risk-manager` |
| Need deploy / cron / tmux / Oracle work | `@devops` |

## Continuation Rules

- Inside one active chat, the orchestrator should advance foreground-autonomously as far as possible.
- After chat ends, continuity lives in `tasks/active/*.yaml`.
- To continue, use `/resume-task` or explicitly reference the `task_id`.
- Do not assume a background worker exists.

## Required References

- Workflow overview: `docs/CURSOR_WORKFLOW.md`
- Orchestration spec: `docs/ORCHESTRATION_MVP.md`
- Strategy playbook: `docs/STRATEGY_DEV_PLAYBOOK_R2_1.md`
- Portfolio research protocol: `.cursor/skills/alpha/portfolio-research-protocol.md`
- Systematic experiment design: `.cursor/skills/alpha/systematic-experiment-design.md`
- Handoff gates: `.cursor/skills/alpha/handoff-gates.md`
- Feature ownership registry: `.cursor/skills/dev/feature-ownership-registry.md`
