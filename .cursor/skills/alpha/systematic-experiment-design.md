---
description: Strategy diagnosis card, experiment-family framework, Alpha Existence vs Trade Expression loop
globs:
alwaysApply: false
---
# Skill: Systematic Experiment Design

> Load this skill when a strategy discussion starts turning into "maybe tweak TP/SL, maybe add HTF, maybe change weighting."
> The goal is to convert ad hoc iteration into a controlled research loop.

## When to Use

Use this skill when:
- improving an existing strategy without a clear diagnosis
- deciding whether a hypothesis failed because the **signal** is wrong or because the **trade expression** is wrong
- discussing TP / SL / HTF resonance / recency weighting / regime filters
- asking an agent to design the next research cycle

## Core Rule

**One research cycle = one primary experiment family.**

Do not mix:
- signal-definition changes
- entry-timing changes
- exit-design changes
- position-sizing changes
- portfolio-role changes

in the same conclusion.

If more than one family is changing, split the work into multiple cycles.

## Step 1 — Write a Strategy Diagnosis Card

Before any new experiment, write this card:

```markdown
## Strategy Diagnosis Card

- Baseline / current strategy:
- Exact pain point:
- Pain regime:
- Current archetype:
- Target role of the next change:
- Primary experiment family:
- Economic mechanism:
- What stays fixed:
- Pass rule:
- Kill rule:
- What result would prove the current hypothesis wrong:
```

### Required meanings

- **Exact pain point**: not "Sharpe not high enough", but "2024 chop regime produced repeated false breakouts" or "standalone MR has unacceptable MDD despite positive expectancy"
- **Target role**: `Filter`, `Overlay`, `Standalone`, `Portfolio Layer`, or `Replacement`
- **Primary experiment family**: exactly one of the five families below
- **What stays fixed**: symbol universe, timeframe, cost assumptions, benchmark config, and all non-target strategy components

## Step 2 — Classify the Experiment Family

| Family | Question it answers | Typical examples | Hold constant |
|--------|----------------------|------------------|---------------|
| **Signal mechanism** | Does the alpha source exist in this form? | recency weighting, normalization, crowding definition, alternative state label | entries, exits, sizing |
| **Entry timing** | Can we enter the same edge at better moments? | threshold persistence, band touch, HTF resonance, cooldown logic | signal definition, exits, sizing |
| **Exit design** | How should an already-proven edge be monetized? | NW center TP, ATR SL, time stop, trailing exit | signal definition, entry logic, sizing |
| **Position sizing** | How aggressively should we size proven setups? | conviction weighting, regime scaling, overlay multiplier | direction logic, entry/exit rules |
| **Portfolio role** | How should the idea live in the stack? | standalone vs overlay, replacement vs blend, long-only variant | signal definition, execution assumptions |

## Step 3 — Separate the Two Loops

### Loop A: Alpha Existence

Use this loop when testing **signal mechanism**.

Questions:
- Is the raw mechanism causal and real?
- Does it survive unconditional and conditional IC checks?
- Does it have enough event frequency to matter?
- Is it new information, or just the same factor in different clothing?

Typical evidence:
- causal IC
- year-by-year IC
- pure vs confounded IC
- cross-symbol breadth
- regime split
- event count / observations per year
- orthogonality pre-screen

**Hard rule**: If Loop A fails, do not start discussing TP/SL or finer entry timing.

### Loop B: Trade Expression

Use this loop after Loop A passes.

Questions:
- What entry condition expresses the same edge best?
- What TP/SL structure monetizes the edge best?
- What holding-period / sizing scheme improves implementation without changing the alpha source?

Typical evidence:
- expectancy
- average win / average loss
- win rate
- max drawdown
- cost drag
- hold time
- trade count / year
- delay sensitivity

**Hard rule**: Loop B comparisons must always be measured against the best accepted Loop A baseline.

## Step 4 — Use Family-Specific Success Metrics

### Signal mechanism

Primary metrics:
- causal IC
- conditional IC
- event frequency
- same-sign yearly consistency
- residual IC after orthogonality screen

Good question:
> "Does recency weighting improve causal IC without collapsing frequency?"

Bad question:
> "What if we also move TP closer while changing weighting?"

### Entry timing

Primary metrics:
- false-entry rate
- conditional expectancy at triggered bars
- trade count change
- delay robustness

Good question:
> "Does 4h resonance improve entry quality on the same signal?"

### Exit design

Primary metrics:
- expectancy
- drawdown
- average holding period
- cost-adjusted return
- tail loss / stop efficiency

Good question:
> "For the same entry logic, is NW center TP better than opposite-extreme TP?"

### Position sizing

Primary metrics:
- portfolio Sharpe delta
- MDD delta
- concentration
- turnover impact

Good question:
> "Should outside-band setups carry 1.5x size as an overlay multiplier?"

### Portfolio role

Primary metrics:
- correlation to baseline
- marginal SR contribution
- overlay ablation
- replacement viability

Good question:
> "Is this better as a standalone leg, or only as an overlay on the current baseline?"

## Step 5 — Build an Experiment Matrix

Each row must be one controlled hypothesis.

```markdown
| question | family | changed_component | held_constant | economic_reason | metric | baseline | pass_rule | kill_rule | next_action |
|----------|--------|-------------------|---------------|-----------------|--------|----------|-----------|-----------|-------------|
```

Example:

```markdown
| Does NW center TP reduce MDD without killing expectancy? | exit_design | TP target | signal definition, entry logic, sizing | MR edge should realize toward conditional mean | expectancy, MDD, hold_h | baseline TP=opposite LSR extreme | MDD improves >15% with expectancy drop <10% | expectancy <= 0 after costs | keep best exit and hand to developer |
```

## Mechanical TP / SL Research Order

For MR strategies:

1. **Structural TP**
   - mean / VWAP / NW center
   - opposite band
   - volatility-normalized mean target

2. **Structural SL**
   - ATR stop
   - band expansion failure
   - max adverse excursion threshold
   - time stop

3. **Trade-off review**
   - expectancy
   - MDD
   - hold time
   - cost drag
   - left-tail behavior

Do not compare TP/SL variants until the alpha source is accepted.

## Anti-Patterns

- "Let's change pctrank weighting, TP, SL, and HTF filter together"
- "The raw signal is weak, but maybe a smarter exit will save it"
- "The event quality improved, but trade count collapsed from 120/year to 3/year"
- "This looks better as an overlay, but let's still force it into standalone"
- "We do not know what is held constant"

## Required Output for a Good Research Turn

At minimum, produce:

1. **Diagnosis Card**
2. **Selected experiment family**
3. **Loop type** (`Alpha Existence` or `Trade Expression`)
4. **Experiment matrix** with 1-3 rows
5. **Decision**
   - continue current family
   - freeze this family and move to next family
   - shelve / archive

## Default Prompt Pattern

When talking to an agent, use this structure:

```markdown
Baseline pain:
Experiment family:
Loop type:
Hypothesis:
What stays fixed:
Pass rule:
Kill rule:
Deliverable:
Relevant files:
```

Example:

```markdown
Baseline pain: BTC LSR standalone has positive expectancy but MDD too high.
Experiment family: exit_design
Loop type: Trade Expression
Hypothesis: NW center TP can reduce drawdown without destroying expectancy.
What stays fixed: BTC-only, 1h, same LSR pctrank signal, same cost model, same entry thresholds.
Pass rule: MDD improves >15% with expectancy drop <10%.
Kill rule: expectancy <= 0 after costs or trades/year drops below 30.
Deliverable: EDA notebook + 3-row exit comparison matrix + recommended next step.
Relevant files: src/qtrade/strategy/lsr_contrarian_strategy.py, src/qtrade/strategy/nwkl_strategy.py
```
