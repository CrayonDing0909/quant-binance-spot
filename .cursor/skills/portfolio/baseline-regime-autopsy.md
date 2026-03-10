---
description: Decompose the baseline strategy by regime and failure mode before starting new complementary strategy research
globs:
alwaysApply: false
---
# Skill: Baseline Regime Autopsy

> Loaded by Portfolio Strategist before proposing any complementary strategy.
> **Last updated**: 2026-03-10

## Purpose

Before researching a new strategy, first diagnose exactly where `A (Baseline)` is weak.

Do not jump from "drawdown feels bad" to "let's add a new factor."

## Required Questions

Every autopsy must answer:

1. Which regimes create the largest drawdowns?
2. Are losses caused by wrong direction, over-trading, delayed exits, or dead time in chop?
3. Is the pain concentrated in a few symbols or portfolio-wide?
4. Is the weakness best addressed by a new directional strategy, a filter, or a portfolio allocator?

## Regime Grid

At minimum, split baseline performance across:

- Trend vs Chop
- High-vol vs Low-vol
- Risk-on vs Risk-off
- Crash / panic event windows
- Symbol-level splits for the live universe

If one regime explains most pain, write it explicitly:

> "Baseline weakness is primarily **chop + high-vol**, not generic underperformance."

## Failure Mode Taxonomy

Classify losses into one of these buckets:

| Failure Mode | Typical Symptom | Likely Complement Type |
|-------------|-----------------|------------------------|
| Trend false positives | Frequent flips, small repeated losses | Regime filter / portfolio layer |
| Late reversal response | Large give-back after trend ends | Mean reversion satellite / exit overlay |
| Event shock | Sudden concentrated drawdown | Event-driven reversal |
| Carry collapse | Direction okay but carry leg hurts | Carry replacement / better funding input |
| Symbol concentration | A few symbols dominate pain | Symbol governance / satellite by subset |

## Output Template

Produce a short memo with:

### 1. Weakness Summary
- Top 1-2 painful regimes
- Worst symbols
- Main failure mode

### 2. Candidate Complement Classes
- Best-fit archetype
- Why it fits the diagnosed weakness
- Why simpler alternatives are insufficient

### 3. Research Routing
- `Alpha Researcher`: what to test in EDA
- `Quant Developer`: what kind of backtest will be needed later
- `Quant Researcher`: what evidence is required for acceptance

## Hard Stops

Do not route research forward if:

- The weakness is not clearly identified
- The proposed strategy type does not match the diagnosed failure mode
- The only rationale is "low correlation" without explaining what pain it offsets

## Success Criterion

This skill is complete when the team can say:

> "We are not researching a random idea. We are researching a strategy specifically to reduce `Baseline` pain in regime X."
