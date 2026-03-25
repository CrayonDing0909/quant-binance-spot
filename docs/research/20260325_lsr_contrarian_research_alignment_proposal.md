# Strategy Proposal
## LSR Contrarian Research Alignment

- Date: `2026-03-25`
- Research owner: `Alpha Researcher`
- Task: `research_20260324_105700_lsr_contrarian_standalone_revisit`
- Current status: `ACTIVE — research record re-anchored after accepted Loop A and Loop B baseline`
- Research verdict: `continue_alpha_research_no_handoff`
- Evidence tier: `accepted_loop_a_plus_loop_b_baseline`
- Handoff decision: `Do not hand off to Quant Developer yet`

## 0. Archetype Classification
- Archetype: `MR (Mean Reversion) — crowding-driven contrarian`
- Return profile: `low WR / high R:R / positive expectancy`
- Expected win rate: `~35% to 40%`
- Expected R:R: `~2.5 to 3.0`
- Estimated trades/yr: `~120 to 125` on the accepted baseline
- Cost sensitivity: `Moderate`
- Primary kill criteria applied: `net expectancy <= 0`, `trades/yr < 30`, or `2025-2026 IC worse than EW-only baseline`

## 0.5 Portfolio Context
- Target gap: `standalone or satellite BTC mean-reversion leg with low overlap to the current trend stack`
- Expected integration target: `Standalone candidate with overlay fallback`
- Historical research:
  - `notebooks/research/20260324_lsr_nw_btc_eda.ipynb`
  - `notebooks/research/20260325_lsr_alpha_existence.ipynb`
  - `notebooks/research/20260325_lsr_combined_signal_eda.ipynb`
- Why this direction is still worth continuing:
  - The NW hard-gate branch was shelved, but the full LSR direction is not dead.
  - Loop A accepted a cleaner signal definition.
  - The accepted combined baseline restored both IC quality and viable trade frequency.

## 1. Accepted Evidence

### What is accepted
- `signal_mechanism` / Loop A:
  - `EW pctrank(hl=84)` is better than baseline `pctrank(168)`.
  - `ADX(14) > 25` is a valid regime gate for this direction.
  - The LSR signal is not just lagged price reversal in disguise.
- `Loop B baseline`:
  - Accepted baseline is:
    - `EW pctrank(hl=84)`
    - `ADX(14) > 25`
    - contrarian direction unchanged
    - `v2 swing exits` unchanged
  - Combined 2025-2026 IC: `-0.0411`
  - EW-only 2025-2026 IC: `-0.0296`
  - Estimated trades/year: `123.9`
  - Gross expectancy: `+0.565%`
  - Net expectancy after RT=0.12%: `+0.445%`

### What is not accepted
- `NW hard-gate standalone` as the primary entry filter:
  - It improved conditional IC, but collapsed frequency too far.
- `TP / SL superiority`:
  - Current exits were held fixed in the accepted baseline.
  - No clean exit-design falsification has been run yet.
- `portfolio role`:
  - This direction may still fit better as `standalone`, `long-only satellite`, or `overlay`.
  - That decision is still open.

## 2. Strategy Diagnosis Card

- Baseline / current strategy:
  - `src/qtrade/strategy/lsr_contrarian_strategy.py`
  - v2 swing mode with opposite-extreme TP, ATR stop, time stop
- Exact pain point:
  - The LSR direction now has an accepted baseline, but its best trade expression is still unclear because prior work mixed regime ideas, gating ideas, and execution variants.
- Pain regime:
  - Signal quality is uneven across sides and across candidate gating styles.
  - The worst prior failure mode was frequency collapse when a hard gate looked too selective.
- Current archetype:
  - `MR crowding-driven contrarian`
- Target role of the next change:
  - `Standalone candidate` first, with `overlay fallback` if later role tests reject standalone
- Primary experiment family:
  - `entry_timing`
- Economic mechanism:
  - Retail crowding extremes should mean-revert more reliably when the market is already in the accepted ADX-trending state; the next question is which entry trigger captures that same edge with fewer false starts.
- What stays fixed:
  - BTC-only
  - 1h
  - contrarian direction
  - `EW pctrank(hl=84)` signal definition
  - `ADX(14) > 25` regime filter
  - same cost model (`RT=0.12%`)
  - same `v2 swing` TP / SL framework
- Pass rule:
  - Improve conditional expectancy or false-entry quality without reducing trade frequency below `80/yr` and without degrading the accepted baseline IC profile.
- Kill rule:
  - Trade count collapses materially, or IC / expectancy falls below the accepted baseline, or the candidate only looks better because it silently changes another family.
- What result would prove the current hypothesis wrong:
  - If clean entry-timing variants cannot improve quality while holding the accepted signal and exits fixed, then the bottleneck is probably not entry timing.

## 3. Why The Next Cycle Is Entry Timing

The next cycle is locked to `entry_timing` because:
- Loop A has already passed.
- The accepted baseline already combines the chosen signal definition and regime gate.
- The current unresolved question is now: `how do we enter the same accepted edge better?`

The next cycle is explicitly **not**:
- `signal_mechanism`: already accepted for the current baseline
- `exit_design`: must wait until the best accepted entry expression is fixed
- `position_sizing`: premature before role and expression are clear
- `portfolio_role`: important, but should be decided after entry and exit are cleaner

## 4. Cycle 1 — Entry Timing Matrix

Research cycle declaration:
- Baseline pain: accepted baseline works, but best trigger expression is still unknown
- Primary experiment family: `entry_timing`
- Loop type: `Loop B — Trade Expression`
- What stays fixed: signal definition, exits, direction, cost model, symbol, timeframe
- Pass rule: conditional expectancy improvement with viable trade count
- Kill rule: hidden family mixing or frequency collapse

| question | family | changed_component | held_constant | economic_reason | metric | baseline | pass_rule | kill_rule | next_action |
|----------|--------|-------------------|---------------|-----------------|--------|----------|-----------|-----------|-------------|
| Does `4h` resonance improve entry quality on the accepted baseline? | `entry_timing` | `HTF resonance gate` | `EW pctrank(hl=84)`, `ADX>25`, v2 exits, BTC 1h, costs | same crowding edge may be more reliable when higher timeframe direction confirms local exhaustion | conditional expectancy, false-entry rate, trades/yr, delay robustness | accepted baseline with no HTF gate | expectancy improves and trades/yr stays `>= 80` | IC or net expectancy falls below accepted baseline, or trades/yr drops `< 80` | keep only if it wins without changing exits |
| Does threshold persistence (`2-3` consecutive extreme bars) reduce noise without killing density? | `entry_timing` | `threshold persistence` | same as above | crowded extremes may be more reliable when they persist briefly instead of flashing once | conditional expectancy, entry bar quality, trades/yr | single-bar trigger | expectancy improves and trade count loss stays `< 25%` | frequency collapses or signal delay erases edge | keep only the best persistence rule |
| Does a fixed cooldown after exit improve re-entry quality? | `entry_timing` | `cooldown / re-entry spacing` | same as above | mean-reversion setups may cluster and overtrade the same move; a cooldown may reduce redundant entries | expectancy, repeated-loss rate, trades/yr | no cooldown | repeated-loss rate falls and expectancy stays positive | little quality gain with meaningful density loss | keep only if it reduces churn cleanly |

Entry-cycle notes:
- `long-only` is intentionally **not** in this matrix because it is a `portfolio_role` question, not a pure entry-timing question.
- Any candidate that modifies TP / SL at the same time is invalid for this cycle.

## 5. Cycle 2 — Exit Design Matrix

This cycle only starts **after** Cycle 1 selects an accepted entry baseline.

Research cycle declaration:
- Baseline pain: accepted signal exists, but the best monetization structure is not proven
- Primary experiment family: `exit_design`
- Loop type: `Loop B — Trade Expression`
- What stays fixed: accepted signal, accepted entry timing, direction, cost model, symbol, timeframe
- Pass rule: better expectancy / MDD / hold-time trade-off on the same accepted entry
- Kill rule: any variant that only wins by changing signal or entry logic

| question | family | changed_component | held_constant | economic_reason | metric | baseline | pass_rule | kill_rule | next_action |
|----------|--------|-------------------|---------------|-----------------|--------|----------|-----------|-----------|-------------|
| For the same accepted entry, is `NW center TP` better than `opposite-extreme TP`? | `exit_design` | `structural TP` | accepted signal and entry, ATR stop, time stop, costs | MR paths may realize toward a local mean before reaching the opposite sentiment extreme | gross expectancy, net expectancy, MDD, hold_h | v2 opposite-extreme TP | MDD improves meaningfully without net expectancy turning negative | density survives but expectancy collapses | keep only if the trade-off is objectively better |
| Is `ATR 2.5` or `ATR 3.0` the better structural stop? | `exit_design` | `structural SL` | accepted signal and entry, TP, time stop, costs | the current stop may be too tight or too loose for BTC 1h volatility | left-tail loss, avg loss, expectancy, stop efficiency | current ATR stop | better tail control with no fatal expectancy damage | one side improves only because hold time explodes or expectancy turns negative | keep only the best stop multiplier |
| Is `120h` or `168h` the better time stop for the same entry/exit structure? | `exit_design` | `time stop` | accepted signal and entry, TP, ATR stop, costs | if the edge decays faster than assumed, a shorter time stop may reduce dead capital; if not, a longer stop may monetize more of the swing | expectancy, hold_h, cost drag, dead-trade share | current v2 time stop | hold time improves or expectancy improves without hurting tail behavior | no gain, higher drag, or worsened left tail | keep only if the time-stop change adds clear value |

Exit-cycle rule:
- Follow the required order: `Structural TP -> Structural SL -> Trade-off review`.

## 6. Cycle 3 — Portfolio Role Matrix

This cycle decides how the direction should live in the stack.

Research cycle declaration:
- Baseline pain: even with an accepted standalone candidate, the best stack role is still unclear
- Primary experiment family: `portfolio_role`
- Loop type: `Loop B — Trade Expression`
- What stays fixed: accepted signal, accepted entry, accepted exits, costs, symbol, timeframe
- Pass rule: the chosen role must improve implementability without hiding behind mixed-family changes
- Kill rule: declaring a role win without role-specific validation

| question | family | changed_component | held_constant | economic_reason | metric | baseline | pass_rule | kill_rule | next_action |
|----------|--------|-------------------|---------------|-----------------|--------|----------|-----------|-----------|-------------|
| Is the accepted baseline still strong enough as a BTC-only standalone leg? | `portfolio_role` | `stack role = standalone` | accepted signal, entry, exits, costs | if the edge has enough independent quality and density, it can justify standalone implementation | net expectancy, trades/yr, correlation to production, validation readiness | current accepted baseline | net positive, viable density, and acceptable correlation | correlation too high or standalone case weaker than an overlay alternative | hand off only if standalone remains clearly justified |
| Is `long-only satellite` a better role than symmetric standalone? | `portfolio_role` | `stack role = long-only satellite` | accepted signal, entry, exits, costs | prior evidence suggests the long side may be structurally cleaner than the short side | net expectancy, trades/yr, side-specific robustness, marginal role fit | symmetric standalone | long-only improves robustness without becoming too sparse | trade density collapses or gains are only sample noise | keep alive only if role-level evidence is better |
| Is this better as an overlay / filter on the existing stack than as its own leg? | `portfolio_role` | `stack role = overlay/filter` | accepted signal, entry, exits, costs | some crowding edges are more valuable as conviction or regime modifiers than as independent direction engines | overlay ablation, marginal SR, opportunity loss, implementation cost | standalone candidate | overlay improves net SR or reduces pain with lower complexity | overlay adds no portfolio value | if overlay wins, open a separate position-sizing cycle |

Role-cycle outcome rule:
- If `overlay/filter` wins, open a new `position_sizing` cycle after this one.
- If `standalone` wins cleanly, prepare a Quant Developer handoff.
- If `long-only satellite` wins, validate it as a separate role rather than folding it back into entry-timing claims.

## 7. Position Sizing Is Deferred

`position_sizing` is intentionally deferred until after the role decision.

Why:
- Sizing is only meaningful after the stack role is chosen.
- If the direction becomes an overlay, sizing questions become `conviction scaling`.
- If the direction stays standalone, sizing becomes a later implementation-control question.

If reopened later as its own cycle, the first candidates should be:
- `ADX conviction multiplier`
- `outside-band or high-conviction size boost`
- `side-asymmetric sizing` only if role work justifies it

## 8. Validation Gates For Future Promotion

Before developer handoff, the accepted branch should satisfy:
- one accepted baseline per family
- no mixed-family conclusion
- net expectancy remains positive after costs
- viable trade frequency for the chosen role
- role-specific validation is complete:
  - standalone: correlation and validation-readiness
  - overlay: overlay ablation and marginal SR
  - long-only satellite: density and side-specific persistence

## 9. Bottom Line

The correct next move is **not** to ask all of these at once:
- Should we change weighting?
- Should we add HTF resonance?
- Should we move TP?
- Should we widen SL?
- Should we go long-only?

The correct next move is:
1. Keep the accepted signal baseline fixed.
2. Run one clean `entry_timing` cycle.
3. Then run one clean `exit_design` cycle.
4. Then decide `portfolio_role`.

## 10. Recommended Next Agent
- Recommended next agent: `@alpha-researcher`
- Recommended prompt:

```markdown
@alpha-researcher

Run the next LSR research cycle using `docs/research/20260325_lsr_contrarian_research_alignment_proposal.md`.

Lock the cycle to:
- Primary experiment family: `entry_timing`
- Loop type: `Loop B — Trade Expression`
- Baseline: BTC-only, 1h, `EW pctrank(hl=84) + ADX(14)>25`, contrarian direction, same RT cost model, same v2 swing exits

Only test the three rows in the Entry Timing Matrix.
Do not change signal definition, TP/SL, sizing, or portfolio role in this cycle.

Deliverable:
- Notebook with row-by-row comparison
- pass/kill verdict for each row
- one accepted entry baseline or explicit FAIL
```
