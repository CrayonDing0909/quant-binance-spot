---
description: Portfolio-aware research protocol, signal integration decision tree, priority scoring system
globs:
alwaysApply: false
---
# Skill: Portfolio-Aware Research Protocol

> Loaded by Alpha Researcher before starting any new research direction.

## Research Pre-Check (Step 0 — Before Archetype Classification)

**Must complete these 5 steps before starting new research:**

1. **Read Alpha Research Map** — Open [docs/ALPHA_RESEARCH_MAP.md](mdc:docs/ALPHA_RESEARCH_MAP.md)
   - Check coverage map: which dimensions are covered? which are blank?
   - Check data-signal taxonomy: has this data-signal combo been tested? results?
   - Check research frontier ranking: what are the top-ranked directions?

2. **Specify Target Gap** — In Notebook and Proposal, explicitly write:
   > "This research fills **[item #X: gap name]** in the Alpha Coverage Map"
   
   If your direction is not in a coverage gap, explain why it's worth revisiting.

3. **Estimate Marginal SR Contribution** (rough is fine):
   - Expected correlation with production portfolio < ? (corr < 0.3 for standalone threshold)
   - Expected standalone SR > ?
   - Expected portfolio SR improvement > ?

4. **Check Data-Signal Taxonomy** — Confirm:
   - Has this data source already had signals tested?
   - If yes, what was the failure reason? How is this research different?
   - If in "Closed Directions" list, must explain why it can be revived.

5. **Declare Integration Target** — Before starting:
   > "This signal is expected to be used as **[Filter / Overlay / Standalone / Portfolio Layer]**"

### "Don't Research" Is a Valid Option

If the highest-scored candidate in the frontier ranking < 3.0, and all coverage gaps face data quality or alpha uncertainty issues, **"no new research this cycle" is valid**.

Focus instead on: improving existing overlay params, enhancing data coverage, updating the research map.

## Signal Integration Decision Tree

```
New alpha signal discovered
    │
    ├── Is it a regime/filter signal (binary: tradeable/not)?
    │       Yes ──→ 【Filter / Regime Gate】
    │              Example: htf_trend_filter, adx_regime, vol_pause
    │              Validation: IC improvement > 5%, trade opportunity loss < 30%
    │              Cost: Minimal (fewer trades = lower cost)
    │
    ├── Is it a continuous scaling signal (amplify/reduce existing position)?
    │       Yes ──→ 【Overlay (confirmatory or exit)】
    │              Example: lsr_confirmatory, oi_vol, derivatives_micro
    │              Validation: overlay ablation 2×2, net SR improvement > 0
    │              Cost: Low-Medium (no direction change, just magnitude)
    │
    ├── Is it an independent directional signal (own entry/exit logic)?
    │       Yes ──→ Check correlation with existing portfolio
    │              corr < 0.3 ──→ 【Standalone satellite strategy】
    │                              Validation: full V1-V12 validation stack
    │              corr >= 0.3 ──→ 【Blend into meta_blend or abandon】
    │                              Validation: marginal SR test + blend sweep
    │
    └── Is it a portfolio-level signal (risk-on/risk-off, affects all strategies)?
            Yes ──→ 【Portfolio Layer】
                   Validation: no MDD increase, long lookback
                   Cost: Very low (low-frequency adjustment)
```

### Integration Level Output Requirements

| Level | Deliverable | Recipient | Key Validation |
|-------|-----------|----------|---------------|
| **Filter** | Filter logic + IC comparison | Quant Developer (add to `filters.py`) | IC improvement > 5%, trade frequency loss < 30% |
| **Overlay** | Overlay logic + params | Quant Developer (add to `overlays/`) | 2×2 ablation, net SR > 0 |
| **Standalone** | Full Strategy Proposal | Quant Developer (new strategy + config) | Full V1-V12, corr < 0.3 |
| **Portfolio Layer** | Regime signal + scaling logic | Quant Developer (`low_freq_portfolio`) | No MDD increase |

### Researcher Judgment Guide

> **Top Principle**: When in doubt, don't handoff. More EDA is cheaper than a failed formal backtest.
> One premature GO wastes 2-4h developer time and erodes team trust.
> Historical handoff success rate ~20% — the only way to improve it is to raise the GO threshold.

- **Prefer Overlay**: Lowest cost, fastest validation, least risk. If a signal can work as overlay, try overlay before standalone.
- **Standalone is High Investment**: Needs full V1-V12, new config, new data pipeline. Only if corr < 0.3 and SR > 1.0.
- **Filter is Invisible Assist**: Doesn't directly generate alpha but reduces false signals and costs. HTF Filter was a success case (+0.485 SR).
- **"Don't research" is high-quality output**: If all directions fail quality thresholds, documenting why and improving existing pipeline is more valuable than a weak handoff.

## Priority Scoring System (5 Factors)

| Factor | Weight | 1 (Low) | 3 (Medium) | 5 (High) |
|--------|--------|---------|-----------|----------|
| **Marginal Diversification** | 30% | corr > 0.5 | corr 0.2-0.5 | corr < 0.1 |
| **Data Quality & Availability** | 20% | No data, needs new API | Data exists but coverage < 70% | Full coverage, already downloaded |
| **Expected Alpha Strength** | 20% | IC < 0.01 or known FAIL | IC 0.01-0.03 or initial EDA positive | IC > 0.03 and cross-symbol robust |
| **Implementation Complexity** | 15% | New strategy + data pipeline + live integration | New overlay or existing strategy modification | Already implemented, just needs backtest |
| **Academic/Empirical Support** | 15% | No literature, pure intuition | Traditional market literature exists | Strong literature + crypto-specific research |

**Score = 0.30×Diversification + 0.20×Data + 0.20×Alpha + 0.15×Complexity + 0.15×Literature**

### Thresholds
- **< 2.5**: Don't start deep research. Record reason, mark "not feasible".
- **2.5-3.0**: Initial EDA (1-2h), decide based on results.
- **3.0-4.0**: Standard research flow (full Notebook + Proposal).
- **> 4.0**: High priority, prioritize over other directions.

> **Full frontier ranking table** maintained in [docs/ALPHA_RESEARCH_MAP.md](mdc:docs/ALPHA_RESEARCH_MAP.md) Section 3.
