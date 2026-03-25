---
description: Handoff quality gates (G1-G6), handoff prompt requirements, Next Steps output format
globs:
alwaysApply: false
---
# Skill: Handoff Quality Gates & Output Format

> Loaded by Alpha Researcher before making GO/FAIL decisions.

## 7 Hard Quality Gates (Must ALL Pass Before GO)

| # | Gate | Pass Criteria | Failure Case Reference |
|---|------|-------------|----------------------|
| **G0** | **Factor Orthogonality** | `marginal_information_ratio()` R² < 0.50 against ALL existing production signals; `check_latent_factor_loading()` max loading < 0.70. **Must run `from qtrade.validation.factor_orthogonality import marginal_information_ratio` and report R² + residual IC** | OI/On-chain/Macro/VPIN all standalone > HTF baseline but stacked → over-filter. All were essentially "same factor wearing different clothes" (regime filtering). If R² > 0.50 → FAIL (redundant). **Project lost ~16h developer time on 4 redundant filter ablations** |
| G1 | **Causal IC Strength** | Causal IC (with shift) > 0.01, at least 6/8 symbols | CVD: IC 0.019→0.001 after proper shift |
| G2 | **IC Annual Consistency** | IC consistent across >= 3 calendar years (no sign flip in any 2-year window) | CVD: 2022 negative→2026 positive |
| G3 | **Confounding Factor Isolation** | Pure signal IC computed and recorded; improvement not from existing components | 4h TF: +1.53 SR came from HTF filter, not 4h signal |
| G4 | **Structural Prerequisites** | Step 3.5 all passed (cross-correlation, TF redundancy, etc.) | XSMOM: crypto cross-corr ~0.7 killed cross-sectional strategy |
| G5 | **Resource ROI Estimate** | `P(success) × SR improvement > 0.05` (rough estimate) | Taker Vol: 14 variants for IC=-0.006 |
| G6 | **New Information Clarity** | Can write: "This signal provides **what information that the current production pipeline cannot capture**?" | 4h TF: highly redundant with 1h+HTF pipeline (corr=0.79) |

### G0 Implementation

```python
from qtrade.validation.factor_orthogonality import marginal_information_ratio, check_latent_factor_loading

# 1. Marginal Information Ratio (mandatory)
result = marginal_information_ratio(
    candidate=candidate_signal,
    existing_signals={"tsmom": tsmom, "htf": htf, "lsr": lsr},
    forward_returns=fwd_returns,
)
print(f"R²={result.r_squared:.3f}, Residual IC={result.residual_ic:.4f}")
# PASS: R² < 0.50 AND residual_ic > 0.005

# 2. Latent Factor Loading (recommended)
loading = check_latent_factor_loading(
    candidate=candidate_signal,
    existing_signals={"tsmom": tsmom, "htf": htf, "lsr": lsr},
)
print(f"Max loading: {loading.max_loading:.3f} on {loading.max_loading_pc}")
# PASS: max_loading < 0.70
```

### Decision Rules

- **7/7 pass** → GO, write handoff prompt
- **6/7 pass**, failed item is G5 → Can GO but must flag risk in handoff prompt
- **G0 fail** → HARD STOP. Factor is redundant. Do NOT handoff. Options: (a) compute residual signal and test residual IC instead, (b) abandon direction
- **Any other failure** → No handoff. Options:
  - Do more EDA to fix failed item
  - FAIL this direction, add to Closed Directions
  - Downgrade to WEAK GO (overlay/filter level only, no standalone)

### Reopen Framework Classification

If the direction is not a handoff candidate but still worth preserving, classify it using the shared reopen framework:

- `proxy-valid / state-invalid`
  - proxy/common-window evidence is good
  - but true state data or true state validation is missing
- `state-valid / handoff-blocked`
  - state evidence is good
  - but bucket quality / concentration / overlap still blocks handoff

This classification is **supplemental** to the final verdict.

Example:
- Final verdict: `SHELVED`
- Evidence tier: `proxy-valid / state-invalid`

Do not collapse all non-handoff directions into a vague `interesting but incomplete`.

## Experiment Hygiene (Mandatory Before Evaluating G0-G6)

Before deciding GO / FAIL, the research packet must clearly state:

1. **Primary experiment family**
   - `signal_mechanism`
   - `entry_timing`
   - `exit_design`
   - `position_sizing`
   - `portfolio_role`

2. **Loop type**
   - `Loop A: Alpha Existence`
   - `Loop B: Trade Expression`

3. **Held-constant baseline**
   - what was intentionally not changed in this cycle

### Hard Rules

- If **Loop A** failed, do **not** hand off a Loop B change. No amount of TP/SL tuning rescues a non-existent alpha.
- If multiple experiment families were changed together, do not write a confident handoff. Split the result into separate cycles or downgrade confidence.
- Any research conclusion must answer:
  - Was the hypothesis wrong?
  - Or was the trade expression wrong?

### Minimum language expected in a good packet

```markdown
- Primary experiment family: exit_design
- Loop type: Loop B (Trade Expression)
- What stayed fixed: BTC-only, 1h, same LSR pctrank signal, same cost model, same entries
- Decision: signal accepted; exit variant X beats baseline on MDD without killing expectancy
```

## Handoff Prompt Requirements

After passing quality gates, handoff to **Quant Developer** must include:

1. Strategy Proposal full path (all fields filled)
2. Exploratory Notebook path
3. **G1-G6 itemized pass/fail record**
4. Data dependencies clearly marked (Developer needs to confirm data availability)
5. Preliminary IC / Sharpe estimates (must be causal IC)
6. Expected integration mode (Filter / Overlay / Standalone)
7. Known risks and limitations

## Next Steps Output Format

**Every research session must end with a "Next Steps" block:**

```markdown
---
## Next Steps (pick one)

| Option | Agent | Prompt | When to pick |
|--------|-------|--------|-------------|
| A | `@quant-developer` | "Alpha Researcher completed <strategy> proposal (GO_NEXT). Implement based on: [key params]..." | Positive results, proceed to implementation |
| B | `@alpha-researcher` | "<direction> hypothesis failed. Explore <alternative> instead, initial leads: [summary]..." | Current direction failed but alternatives exist |
| C | (none) | Move `config/research_*.yaml` to `config/archive/`, research ends | Dead end, no viable direction |
```

### Rules
- **Option A** prompt must include: strategy name, signal definition summary, key data requirements, preliminary performance numbers
- **Option B** prompt must explain: why current direction failed, initial leads for alternative
- **Option C** only when all exploration directions are FAIL
- Multiple candidates → provide A1, A2 for Orchestrator to prioritize
