---
description: Map baseline weaknesses to complementary strategy archetypes and integration modes
globs:
alwaysApply: false
---
# Skill: Complementary Strategy Design

> Loaded by Portfolio Strategist when converting a weakness diagnosis into a concrete research thesis.
> **Last updated**: 2026-03-10

## Core Rule

Do not ask "what alpha is interesting?"

Ask:

> "What kind of strategy would make the portfolio suffer less in the exact regime where Baseline is weak?"

## Mapping Weakness To Strategy Type

| Diagnosed Weakness | Preferred Strategy Archetype | Default Integration Mode |
|-------------------|------------------------------|--------------------------|
| Sideways chop whipsaw | Mean reversion | Standalone or portfolio layer |
| Panic overshoot / liquidation | Event-driven reversal | Standalone satellite |
| Trend signal too permissive | Regime filter | Filter |
| Exit too slow after exhaustion | Reversal / exhaustion overlay | Overlay |
| Broad regime mismatch | Risk-on / risk-off allocator | Portfolio layer |

## Integration Choice

Choose exactly one primary mode first:

1. **Filter**: blocks low-quality baseline trades
2. **Overlay**: scales an existing position
3. **Standalone**: generates its own entry / exit logic
4. **Portfolio Layer**: adjusts risk across strategies

Default preference order:

- If the new idea is truly independent and targets a different market behavior, prefer `Standalone`
- If it only helps avoid bad baseline entries, prefer `Filter`
- If it changes conviction but not direction, prefer `Overlay`
- If it acts across all strategies/regimes, prefer `Portfolio Layer`

## Anti-Pattern

Do not automatically wrap every idea as a TSMOM enhancement.

Bad framing:

> "Can we add one more microstructure filter to TSMOM?"

Better framing:

> "Does this idea deserve to exist as an independent anti-chop strategy?"

## Thesis Template

Every complementary strategy thesis must specify:

- `Target gap`
- `Target regime`
- `Expected role`
- `Integration mode`
- `Why Baseline cannot solve this internally`
- `Kill criteria`

Example:

> Target gap: sideways chop drawdown  
> Target regime: low-trend + high-vol  
> Expected role: absorb whipsaw periods where TSMOM loses edge  
> Integration mode: standalone satellite  
> Why Baseline cannot solve this internally: additional filters reduce time-in-market but do not monetize mean reversion  
> Kill criteria: corr with Baseline > 0.3 or no MDD improvement in chop

## Preferred Research Order

When Baseline is trend-following, prioritize:

1. Independent anti-chop strategies
2. Event-driven reversal
3. Portfolio-level allocators
4. Internal filters / overlays

Reason:
Filters often reduce damage, but they rarely create a true second return engine.

## Exit Condition

This skill is complete when the proposed strategy is framed as a **portfolio complement**, not a random feature add-on.
