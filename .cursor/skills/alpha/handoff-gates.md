---
description: Handoff quality gates (G1-G6), handoff prompt requirements, Next Steps output format
globs:
alwaysApply: false
---
# Skill: Handoff Quality Gates & Output Format

> Loaded by Alpha Researcher before making GO/FAIL decisions.

## 6 Hard Quality Gates (Must ALL Pass Before GO)

| # | Gate | Pass Criteria | Failure Case Reference |
|---|------|-------------|----------------------|
| G1 | **Causal IC Strength** | Causal IC (with shift) > 0.01, at least 6/8 symbols | CVD: IC 0.019→0.001 after proper shift |
| G2 | **IC Annual Consistency** | IC consistent across >= 3 calendar years (no sign flip in any 2-year window) | CVD: 2022 negative→2026 positive |
| G3 | **Confounding Factor Isolation** | Pure signal IC computed and recorded; improvement not from existing components | 4h TF: +1.53 SR came from HTF filter, not 4h signal |
| G4 | **Structural Prerequisites** | Step 3.5 all passed (cross-correlation, TF redundancy, etc.) | XSMOM: crypto cross-corr ~0.7 killed cross-sectional strategy |
| G5 | **Resource ROI Estimate** | `P(success) × SR improvement > 0.05` (rough estimate) | Taker Vol: 14 variants for IC=-0.006 |
| G6 | **New Information Clarity** | Can write: "This signal provides **what information that the current production pipeline cannot capture**?" | 4h TF: highly redundant with 1h+HTF pipeline (corr=0.79) |

### Decision Rules

- **6/6 pass** → GO, write handoff prompt
- **5/6 pass**, failed item is G5 → Can GO but must flag risk in handoff prompt
- **Any other failure** → No handoff. Options:
  - Do more EDA to fix failed item
  - FAIL this direction, add to Closed Directions
  - Downgrade to WEAK GO (overlay/filter level only, no standalone)

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
