---
description: Evaluate new strategy ideas by their portfolio contribution, not only standalone metrics
globs:
alwaysApply: false
---
# Skill: Marginal Contribution Framework

> Loaded by Portfolio Strategist when deciding whether a candidate deserves developer time.
> **Last updated**: 2026-03-10

## Core Principle

The default unit of judgment is **portfolio contribution**, not standalone beauty.

A strategy can be worth researching if:

- standalone SR is only moderate,
- but it materially reduces drawdown in the exact regime where Baseline struggles,
- and correlation with the existing portfolio is low enough.

## Minimum Questions

Before handoff, estimate:

1. Expected correlation with Baseline
2. Expected drawdown relief in target regime
3. Whether the candidate adds a second return engine or only suppresses trades
4. Whether it should be measured against portfolio MDD improvement, SR improvement, or regime balance

## Evaluation Ladder

Use this order:

1. **Problem fit**: does the candidate attack the diagnosed weakness?
2. **Behavior fit**: is the return profile meaningfully different from Baseline?
3. **Integration fit**: should it be blend / replace / independent?
4. **Expected marginal value**: what portfolio metric is likely to improve?

If step 1 fails, stop immediately.

## Useful Heuristics

| Signal Type | Main Value | Main Risk |
|------------|------------|-----------|
| Filter | Fewer bad trades | Over-filter / lost TIM |
| Overlay | Better sizing | Weak signal just dilutes edge |
| Standalone | True diversification | Operational complexity |
| Portfolio Layer | Lower portfolio volatility | Too slow / too generic |

## Suggested Acceptance Bar Before Developer Handoff

At strategy-design stage, prefer candidates that plausibly satisfy:

- expected corr with Baseline `< 0.30` for standalone ideas
- clear regime-specific benefit statement
- not purely dependent on making TSMOM "slightly less bad"
- likely to improve one of:
  - regime drawdown
  - portfolio MDD
  - correlation structure

## Reject These Early

- "Interesting IC" with no explanation of portfolio role
- Another filter on top of an already over-gated baseline
- A candidate that helps everywhere and nowhere
- A candidate whose only edge is theoretical orthogonality

## Standard Output

Write one short table before handoff:

| Item | Answer |
|------|--------|
| Target weakness | ... |
| Expected role | ... |
| Integration mode | ... |
| Expected corr with Baseline | ... |
| Main metric to improve | ... |
| Why worth developer time | ... |

## Success Criterion

This skill is complete when a candidate is justified by **marginal contribution logic**, not by curiosity alone.
