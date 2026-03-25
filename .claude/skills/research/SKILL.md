---
description: "Full strategy research workflow — from idea to handoff. Use when starting new alpha research, exploring a new signal direction, or resuming a paused research task."
---

# /research — Strategy Research Workflow

You are now running an interactive research workflow. Guide the user through each phase, asking for decisions at key gates. Do NOT skip phases.

## Phase 0: Pre-Flight — Kill Known Dead Ends

Before ANY work, check the failure registry:

1. Read `.cursor/skills/alpha/failure-registry.md` — scan the Registry table
2. Read `docs/ALPHA_RESEARCH_MAP.md` — check Closed Directions
3. Ask: "Does this new direction overlap with any known failures?"

**If overlap detected**: Show the user the relevant failure row (Lesson + Prevention Rule + Wasted Resources). Ask if they want to proceed anyway or pivot.

## Phase 1: Economic Intuition Pre-Check (MANDATORY)

Before computing ANY IC or running ANY EDA, the user must answer 3 questions:

| # | Question |
|---|----------|
| Q1 | **Who is the counterparty?** — "This signal profits because [specific agent] systematically mis-prices [specific risk/information]." |
| Q2 | **Why doesn't it get arbitraged away?** — Structural reason: capacity constraint, latency, regulatory barrier, behavioral bias |
| Q3 | **Under what market regime does this mechanism break?** |

**Hard rule**: If Q1 has no coherent answer → do not proceed. Statistical mining without economic intuition has ~15% CPCV pass rate vs ~45% for economically-motivated factors.

Show anti-pattern examples:
- Bad: "I'll compute IC for 50 features and pick the best one"
- Bad: "Entropy should work because information theory"
- Good: "OI drop + price crash = forced liquidation → smart money buys the dip"

## Phase 2: Archetype Classification

Classify the strategy into one of 7 archetypes:

| Archetype | Core Mechanism | Key Metric |
|---|---|---|
| Momentum / Trend | Serial correlation exploitation | IC(lag=1), trend duration |
| Mean Reversion | Deviation from equilibrium | Gross PnL per trade (NOT just IC) |
| Carry | Premium collection | Premium stability across symbols and years |
| Breakout / Volatility | Regime transition capture | Hit rate, timing precision |
| Microstructure | Order flow / liquidity | Tick-level IC, latency requirements |
| Regime Filter | Quality gate for existing signals | Overlap ratio with existing HTF filter |
| Cross-Asset | Relative value / spread | Pair correlation stability |

**Kill criteria** (immediate FAIL):
- Mean Reversion: gross PnL/trade negative → FAIL (IC alone is misleading for MR)
- Carry: premium flips sign in any 2-year window → FAIL
- Regime Filter: >50% bar overlap with existing HTF filter → FAIL (redundant)
- Cross-Asset: avg pairwise crypto corr > 0.5 → cross-sectional ranking won't work

## Phase 3: Factor Orthogonality Check (G0 Gate)

Run this BEFORE deep EDA:

```python
from qtrade.validation.factor_orthogonality import marginal_information_ratio

result = marginal_information_ratio(
    candidate=candidate_signal,
    existing_signals={"tsmom": tsmom, "htf": htf, "lsr": lsr},
    forward_returns=fwd_returns,
)
print(f"R²={result.r_squared:.3f}, Residual IC={result.residual_ic:.4f}")
```

- R² > 0.50 → **HARD STOP**. This is the same factor in different clothing. Project wasted ~16h on 4 redundant filter directions before adding this gate.
- R² < 0.50 AND residual IC > 0.005 → proceed

## Phase 4: Causal IC Computation

All IC must use `signal[t]` vs `return[t+1]`:

```python
# CORRECT: causal IC
ic = signal.shift(1).corr(forward_return)

# WRONG: look-ahead
ic = signal.corr(forward_return)
```

**Extra lag for resampled data** (4h/1d signals on 1h data):
```python
signal_4h = compute_signal(df_4h)
signal_4h_causal = signal_4h.shift(1)  # resampling lag
signal_1h = signal_4h_causal.reindex(df_1h.index, method='ffill')
ic = signal_1h.shift(1).corr(return_1h)  # execution delay
```

**Artifact detection** (ALL must pass):
- A1: IC same-sign in ≥3 calendar years
- A2: Pre/post-shift IC difference < 50%
- A3: Same-sign IC in ≥6/8 symbols
- A4: ≥500 observations per symbol
- A5: Strongest raw IC > 0.01 — if below, **stop exploring variants immediately**

## Phase 5: Research Config Setup

```bash
# Create research branch
git checkout -b research/<strategy-name>-$(date +%Y%m%d)

# Research config (NEVER modify prod_*)
config/research_<name>.yaml

# Research script
scripts/research_<name>.py
```

Research scripts MUST use embargo:
```python
from qtrade.validation.embargo import load_embargo_config, enforce_temporal_embargo
embargo = load_embargo_config()
df = enforce_temporal_embargo(df, embargo)
```

## Phase 6: Handoff Gate Check (7/7 Required for GO)

| Gate | Criteria |
|---|---|
| G0 | Factor Orthogonality: R² < 0.50 against production signals |
| G1 | Causal IC > 0.01 in ≥6/8 symbols |
| G2 | IC consistent across ≥3 calendar years |
| G3 | Pure signal IC reported separately from combined IC |
| G4 | Structural prerequisites passed (cross-corr, TF redundancy) |
| G5 | P(success) × SR improvement > 0.05 |
| G6 | Can articulate what NEW information this provides |

**Decision rules**:
- 7/7 pass → GO, create PR with handoff prompt
- G0 fail → HARD STOP, do not handoff
- Any other fail → options: more EDA, FAIL direction, or downgrade to WEAK GO

## Phase 7: Handoff

Create a GitHub PR with:
```bash
gh pr create --label research --title "research: <strategy name>" --body "..."
```

PR body must include:
1. G0-G6 pass/fail record
2. Strategy proposal (hypothesis, mechanism, data deps)
3. Preliminary IC / Sharpe estimates (causal)
4. Integration mode: Filter / Overlay / Standalone / Portfolio Layer
5. Known risks and limitations

## Phase 8: Update Records

After completing research (GO or FAIL):
1. Update `docs/ALPHA_RESEARCH_MAP.md`
2. If FAIL: add row to `.cursor/skills/alpha/failure-registry.md`
3. Update trial registry in `config/validation.yaml`
4. Append to `.cursor/rules/recent-changes.mdc`
