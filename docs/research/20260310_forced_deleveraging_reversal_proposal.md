## Strategy Proposal V2
### Forced Deleveraging Reversal Satellite

- Date: `2026-03-10`
- Research owner: `Alpha Researcher`
- Baseline: `config/prod_candidate_simplified.yaml`
- Research window: `2022-01-01` onward (common OI + LSR availability window)
- Research verdict: `SHELVED`
- Evidence tier: `proxy-valid / state-invalid`
- Handoff decision: `Do not hand off to Quant Developer yet`

## Hypothesis
Forced deleveraging can create a reflexive rebound window that is structurally different from the baseline 1h trend engine: the baseline is often flat or short when liquidation pressure exhausts, so a separate long-only event leg could harvest panic-to-rebound and sharp reversal payoffs.

## Mechanism
- Forced sellers are crowded longs being unwound into stressed liquidity.
- The rebound becomes tradable only after the cascade ends and price stops making new lows.
- The edge should appear in bars where the baseline is flat or short, otherwise it is just trend continuation wearing an event label.

## Archetype
`Event-Driven Reversal`

This is explicitly:
- Not a trend filter
- Not a confirmatory overlay
- Not a portfolio throttle

## Market Regime Target
- `panic -> rebound`
- `sharp reversal`
- `false breakdown / reclaim`
- `high-vol no-trend` only if the trigger can avoid broad dip-buying

## Data Dependencies
| Dataset | Path / Source | Coverage | Verdict |
|---|---|---:|---|
| 1h futures klines | `data/binance/futures/1h/*.parquet` | 100% for research symbols | usable |
| 5m futures klines | `data/binance/futures/5m/*.parquet` | available | usable |
| OI merged | `data/binance/futures/open_interest/merged/*.parquet` | 100.0% across the 2022+ research window | usable |
| Historical liquidation-state | local `liquidation/*.parquet` / Coinglass | unavailable | **hard blocker** |
| Historical LSR for full baseline replay | `data/binance/futures/derivatives/lsr/*.parquet` | 98.7%-100.0% alignment after vision backfill | usable for 2022+ replay |

## Data Quality Verdict
This research result is `proxy-valid` on the 2022+ common-data window, but still **not promotion-valid** because true historical liquidation-state remains unavailable.

## Reopen Framework
### Current State
- `proxy-valid`: YES
- `state-valid`: NO
- `handoff-ready`: NO
- Current label: `proxy-valid / state-invalid`

### Interpretation
- The thesis has survived the common-data-window audit.
- The thesis has **not** survived a true liquidation-state audit, because that audit cannot yet be run.
- Reopening this direction is justified, but only under a staged framework instead of immediate Quant Dev handoff.

### Phase A. Proxy Validation
- Status: `PASS`
- What is already true:
  - `cascade_end` survives with `no_htf_gate`
  - TIM is in the second-leg target zone
  - correlation to baseline is low
  - OI coverage and LSR replay are both restored on the 2022+ window

### Phase B. State Validation
- Status: `BLOCKED`
- Required to unlock:
  - historical liquidation-state data for all target symbols
  - rerun the same bucket / HTF / entry / exit framework using state tags instead of OI proxy tags
  - confirm that `cascade_end` or a better successor still survives without HTF hard gating

### Phase C. Quant Dev Handoff
- Status: `BLOCKED`
- Required to unlock:
  - `state-valid` evidence
  - `panic_liquidation` no longer weak, or `false_breakdown` becomes meaningfully solved
  - concentration remains healthy
  - the module still behaves like a second leg, not an event-colored overlay

## Research Summary
- Best proxy candidate: `cascade_end` with `no_htf_gate`
- Mean TIM: `12.2%`
- Mean baseline correlation: `-0.07`
- Mean trade return: `+0.65%`
- Symbols improved: `6/6`
- Baseline flat-or-wrong share at entry: `70.4%`
- Wrong-way share: `32.8%`
- Orthogonality: average `R^2 = 0.006`, average residual IC `-0.031`

## Bucket-Level Findings
- `crash_rebound`: strongest candidate bucket. Aggregate candidate 24h return `+4.98%` vs baseline `+0.09%`.
- `sharp_reversal`: also strong. Aggregate candidate 24h return `+4.62%` vs baseline `+0.05%`.
- `panic_liquidation`: weak / mixed. Aggregate candidate 24h return `-2.26%` vs baseline `+0.14%`.
- `false_breakdown_reclaim`: broad proxy definitions find many events, but the thesis-consistent `cascade_end` trigger covers almost none of them.
- `high-vol no-trend`: edge is near zero; this target bucket is not convincingly solved yet.

## Step Verdicts
### Step 1. Independent archetype?
Partially yes.

Why:
- `cascade_end` survives without HTF hard gating.
- Correlation to baseline is low and negative.
- More than half of entries occur while the baseline is flat or wrong.

Why not enough:
- The evidence is still proxy-based, not liquidation-state based.
- Promotion still lacks true state-based validation.

### Step 2. Event map
Old `oi_liq_bounce` covers:
- `panic_liquidation`: about `20.7%`
- `crash_rebound`: about `21.3%`
- `sharp_reversal`: about `3.7%`
- `false_breakdown_reclaim`: about `1.3%`

Interpretation:
- The legacy strategy mostly sees obvious panic bars.
- It largely misses the more interesting reclaim-style reversals.

### Step 3. Definition layer comparison
- `Proxy` (`event_fires_immediately`): tradable but too close to raw OI/price shock detection.
- `State-based` (`cascade_end`): best fit to the thesis; density enters the satellite zone and independence remains strong.
- `Tradeable trigger` (`cascade_end_reclaim`): same as `cascade_end` in current proxy implementation; no extra improvement yet.

Conclusion:
- The best surviving unit is `cascade_end reversal`.
- It is still only an OI-state proxy, not true liquidation-state.

### Step 4. HTF dependency
- `no_htf_gate` wins for the best thesis-consistent trigger.
- `soft_htf_veto` and `hard_htf_gate` reduce density too far and raise overlap with the baseline.

Interpretation:
- This is the most encouraging result in the study.
- The candidate does not need bull-trend permission to survive.

### Step 5. Entry confirmation
Ranking:
1. `cascade_end`
2. `cascade_end_reclaim`
3. `event_fires_immediately`
4. `false_breakdown_reclaim`
5. `cascade_end_absorption`

Interpretation:
- Immediate entry is too early.
- False-breakdown reclaim is too broad and turns into dip-buying beta.
- The event becomes most tradable only after the cascade visibly stops.

### Step 6. Exit redesign
Path analysis:
- 12h fixed hold: `-0.03%`
- 24h fixed hold: `+0.47%`
- 48h fixed hold: `+0.59%`
- event-decay exit: `+0.08%`
- reclaim-failure exit: `+0.28%`
- staged TP: `+0.22%`

Interpretation:
- Alpha is not just a one-bar squeeze.
- The path wants more than 12h, but the incremental gain from 24h to 48h is modest.
- If reopened later, start from `24h fixed hold`, not early profit-taking.

### Step 7. Symbol eligibility
Current 2022+ proxy results say all 6 baseline symbols can show positive edge and pass the coverage gate.

If the thesis is reopened after better data:
- First-wave symbols to re-test: `SOLUSDT`, `DOGEUSDT`, `AVAXUSDT`, `ETHUSDT`
- Second-wave: `BTCUSDT`, `LINKUSDT`

Reason:
- Mid-beta names show stronger payoff without higher correlation.

## Independence Test
- Mean return correlation vs baseline: `-0.07`
- Average baseline flat-or-wrong share: `70.4%`
- Average wrong-way share: `32.8%`

Interpretation:
- The proxy signal is genuinely complementary.
- This is the strongest argument for keeping the thesis alive in dormant form.

## Density Test
- Best thesis-consistent trigger TIM: `12.2%`
- Trade count: `1053` total

Interpretation:
- Density is finally in the portfolio-meaningful zone.
- This is no longer the old `4%-5% TIM` problem.

## Concentration Test
- Max symbol contribution share: `20.8%`
- Top-5 trades share: `4.1%`
- Year share spread: `2022 33.6%`, `2023 21.0%`, `2024 24.5%`, `2025 20.9%`

Interpretation:
- Concentration is healthy.
- The thesis is not carried by one symbol or one event cluster.

## Primary Failure Mode
- Research still relies on OI-state proxy instead of true liquidation-state.
- `panic_liquidation` remains weak and `false_breakdown` is still not truly solved.
- Without real liquidation-state history, the current edge can still be an event-colored overlay module rather than a validated second leg.

## Why This Is Not a Quant Developer Handoff Yet
- The best result is still `proxy-valid`, not `state-valid`.
- `panic_liquidation` remains weak and `false_breakdown` is not truly solved.
- Engineering time would still be spent on a thesis that cannot pass the state-based evidence gate.

## Promotion Criteria If Reopened
- Historical liquidation-state data available for all target symbols
- Best trigger still survives with:
  - corr < 0.15 preferred, < 0.25 maximum
  - TIM between 8% and 15%
  - no HTF hard gate dependence
  - positive edge in `panic -> rebound` and `sharp_reversal`
  - at least 1 reclaim-style definition clearly better than pure proxy

## Rollback / Kill Criteria
- Reopen study only if data blockers are fixed.
- Kill permanently if, after true liquidation-state data is added, `cascade_end` no longer survives without HTF hard gating.

## Final Recommendation
`SHELVED`

Do not hand off to Quant Developer now.

Carry this direction forward only as:
`proxy-valid / state-invalid reopen candidate`

If only one v2 direction is preserved for future reopening, keep:
`cascade_end reversal`

This is still just an overlay thesis in disguise.
