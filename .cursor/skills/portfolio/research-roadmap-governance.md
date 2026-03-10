---
description: Maintain a ranked backlog of complementary strategy theses tied to baseline weaknesses
globs:
alwaysApply: false
---
# Skill: Research Roadmap Governance

> Loaded by Portfolio Strategist when prioritizing complementary strategy work.
> **Last updated**: 2026-03-10

## Purpose

Convert strategy research from a stream of ideas into a managed backlog tied to explicit portfolio gaps.

## Backlog Rules

Each roadmap item must include:

- `Target gap`
- `Target regime`
- `Candidate archetype`
- `Integration mode`
- `Priority`
- `Owner for next step`
- `Kill criteria`

If any of the above is missing, the item is not ready for execution.

## Priority Order

Sort roadmap items by:

1. Severity of Baseline weakness
2. Expected diversification value
3. Likelihood of creating a second return engine
4. Data readiness
5. Developer cost

Do not sort only by novelty or literature appeal.

## State Machine

Use this life cycle for roadmap items:

`diagnosed_gap -> thesis_defined -> alpha_research -> developer_backtest -> validation -> accepted_or_closed`

If an item fails, record **why** it failed:

- wrong regime target
- weak alpha
- too correlated
- too costly
- over-filter only

## Owner Routing

| Next Step Needed | Owner |
|------------------|-------|
| Need exploratory EDA | Alpha Researcher |
| Need formal backtest / ablation | Quant Developer |
| Need truth audit / statistical verdict | Quant Researcher |
| Need capital / deployment decision | Risk Manager |

## Governance Principle

A closed direction is a success if it saves future time.

Documenting "do not revisit this unless X changes" is valuable output.

## Review Cadence

- Re-rank the roadmap after each meaningful research outcome
- Revisit priorities after major regime shifts or observation-period reviews
- Remove vague backlog items that have no clear target gap

## Success Criterion

This skill is complete when the next 2-3 research directions are justified by a portfolio roadmap, not by whoever has the newest idea.
