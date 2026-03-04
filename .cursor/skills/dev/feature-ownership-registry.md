---
description: Central registry of all critical modules — ownership, config, review schedule
globs:
alwaysApply: false
---
# Feature Ownership Registry

> Central directory of every critical module in the system.
> Every module MUST have an owner, config location, and action protocol.
> **Last updated**: 2026-03-04

## How to Use This File

- **Quant Developer**: When creating a new module, add it here before considering the task complete.
- **Quant Researcher**: When running validation, verify all gates have methodology owners.
- **Risk Manager**: During periodic review, check for ownerless modules and flag them.

---

## Module Registry

### Validation Gates

| Module | Owner (Methodology) | Implementer | Action on FAIL | Governance Spec | Config Location | Review Schedule | Last Calibrated |
|--------|---------------------|-------------|----------------|-----------------|-----------------|-----------------|-----------------|
| Alpha Decay (IC Monitor) | Quant Researcher | Quant Developer | Risk: WARNING / REDUCE / FLATTEN | `skills/validation/alpha-decay-governance.md` | `validation.yaml` → `alpha_decay` | Weekly (cron) | 2026-03-04 |
| Walk-Forward Analysis | Quant Researcher | Quant Developer | Researcher re-evaluates OOS threshold | — | `validation.yaml` → `walk_forward` + `thresholds.max_sharpe_degradation` | Per validation run | 2026-02-28 |
| CPCV / PBO | Quant Researcher | Quant Developer | Researcher flags overfitting | — | `validation.yaml` → `prado_methods.pbo` + `prado_methods.cpcv` | Per validation run | 2026-02-28 |
| Deflated Sharpe Ratio | Quant Researcher | Quant Developer | Researcher: check trial count accuracy | — | `validation.yaml` → `prado_methods.deflated_sharpe` + `trial_registry` | After each research direction concludes | 2026-03-02 |
| Cost Stress Test | Quant Researcher | Quant Developer | Researcher: evaluate cost sensitivity | — | `validation.yaml` → `cost_stress` | Per validation run | 2026-03-02 |
| Delay Stress Test | Quant Researcher | Quant Developer | Researcher: evaluate timing fragility | — | `validation.yaml` → `delay_stress` | Per validation run | 2026-03-02 |
| Holdout OOS Test | Quant Researcher | Quant Developer | Researcher: check IS/OOS degradation | — | `validation.yaml` → `holdout_oos` | Per validation run | 2026-03-02 |
| Red Flags | Quant Researcher | Quant Developer | Researcher: deep investigation required | — | Hardcoded in `src/qtrade/validation/red_flags.py` | Per validation run | 2026-03-02 |
| Regime Analysis | Quant Researcher | Quant Developer | Researcher: check bear/bull/sideways split | — | `validation.yaml` → `market_regimes` | Per validation run | 2026-03-02 |
| Data Embargo | Quant Researcher | Quant Developer | Block research run if embargo violated | — | `validation.yaml` → `data_embargo` | Monthly (update cutoff) | 2026-03-02 |

### Infrastructure / Cost Models

| Module | Owner (Methodology) | Implementer | Action on Issue | Governance Spec | Config Location | Review Schedule | Last Calibrated |
|--------|---------------------|-------------|-----------------|-----------------|-----------------|-----------------|-----------------|
| Slippage Model (Volume-Based) | Quant Researcher | Quant Developer | Researcher: recalibrate coefficients | — | `prod_candidate_simplified.yaml` → `backtest.slippage_model` | After leverage/symbol change | 2026-03-04 |
| Funding Rate Model | Quant Developer | Quant Developer | DevOps: check data freshness | — | `prod_candidate_simplified.yaml` → `backtest.funding_rate` | Monthly (data pipeline) | 2026-02-28 |
| HTF Filter (4h + Daily) | Alpha Researcher | Quant Developer | Researcher: check filter still effective | — | `prod_candidate_simplified.yaml` → `strategy.params.htf_filter` | After regime shift | 2026-02-28 |
| Symbol Governance | Quant Researcher | Quant Developer | Risk: add/remove/reweight symbols | — | `prod_candidate_simplified.yaml` → `market.symbols` + `portfolio.allocation` | Monthly IC scan | 2026-03-04 |

### Operations

| Module | Owner (Methodology) | Implementer | Action on Issue | Governance Spec | Config Location | Review Schedule | Last Calibrated |
|--------|---------------------|-------------|-----------------|-----------------|-----------------|-----------------|-----------------|
| Circuit Breaker | Risk Manager | Quant Developer | Risk: trigger halt | — | `prod_candidate_simplified.yaml` → `risk` | Monthly | 2026-02-28 |
| Position Sizing | Risk Manager | Quant Developer | Risk: verify Kelly fraction | — | `prod_candidate_simplified.yaml` → `position_sizing` | After strategy change | 2026-02-28 |

---

## Governance Template

When adding a new critical module, fill in this template and add a row to the appropriate table above.

### Minimum (for simple modules)

Add one row to the registry table with all columns filled. This is sufficient for modules with straightforward pass/fail logic (e.g., a new validation gate with a single threshold).

### Full Governance Spec (for complex modules)

If the module has multiple thresholds, a non-obvious methodology, or requires calibration — create a dedicated governance skill file modeled on `alpha-decay-governance.md`:

```markdown
# Skill: [Module Name] Governance

> Single source of truth for [Module Name] methodology.
> **Last updated**: YYYY-MM-DD

## 1. Ownership
| Role | Responsibility | When |
|------|---------------|------|
| **[Owner]** (OWNER) | Define methodology, thresholds, calibration | ... |
| **[Implementer]** | Implement in code + tests | ... |
| **[Actor]** | Act on results | ... |

## 2. Methodology Spec
[How the module works — formula, inputs, outputs]

## 3. Thresholds
[What values are used, why, and where they live in config]

## 4. Calibration Protocol
[When and how to recalibrate thresholds]

## 5. Escalation Path
[What happens on FAIL — who does what]

## 6. Anti-Patterns
[Common mistakes to avoid]
```

### Checklist Before Marking a New Module Complete

- [ ] Row added to Feature Ownership Registry
- [ ] Owner explicitly assigned (not "everyone" — one role owns methodology)
- [ ] Config location documented (no hardcoded thresholds without a config reference)
- [ ] Action on FAIL defined (what happens when this module reports a problem)
- [ ] Review schedule set (how often thresholds get re-checked)
- [ ] If complex: dedicated governance skill file created

---

## Anti-Patterns

- **"Everyone owns it"** = nobody owns it. One role must be the methodology OWNER.
- **Thresholds set once, never recalibrated** — every threshold needs a review schedule.
- **FAIL with no action protocol** — if a gate can FAIL, someone must know what to do next.
- **Owner not in agent .md** — if it is not in the agent file, the agent will not know about it.
