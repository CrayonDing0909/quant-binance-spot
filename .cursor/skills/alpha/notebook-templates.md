---
description: Notebook structure templates (standard 7-section + Multi-TF 7+2), Strategy Proposal template
globs:
alwaysApply: false
---
# Skill: Research Templates (Notebook + Strategy Proposal)

> Loaded by Alpha Researcher when creating notebooks or proposals.

## Notebook Structure (Standard 7-Section)

```
notebooks/research/
├── YYYYMMDD_<topic>_exploration.ipynb  ← Primary exploration
├── YYYYMMDD_<topic>_feature_eng.ipynb  ← Feature engineering experiments
└── YYYYMMDD_<topic>_signal_proto.ipynb ← Preliminary signal prototype
```

### Required Sections

0. **Portfolio Context** (mandatory — before any analysis):
   - Target gap in Alpha Coverage Map
   - Expected integration mode (Filter / Overlay / Standalone / Portfolio Layer)
   - Data-signal combos being explored (cross-reference data-signal taxonomy)
   - Prior attempts? If yes, link previous Notebook and explain difference.
   - 5-factor priority score (rough estimate).
1. **Hypothesis**: Clear, falsifiable hypothesis
2. **Data Description**: Data sources, time range, frequency, coverage
3. **EDA**: Exploratory analysis (distributions, correlations, time-series features)
4. **Feature Engineering**: Factor construction process
5. **Preliminary Signal** (must include sub-sections):
   - 5a. **Causal IC Table**: All-symbol IC (must use `shift(1)` causal version); show with/without shift side-by-side
   - 5b. **Annual IC Stability**: IC decomposed by calendar year; flag any sign-flipping years
   - 5c. **Confounding Factor Isolation** (if applicable): Pure signal IC vs signal+existing component IC
   - 5d. **Handoff Quality Gate Checklist**: G1-G6 pass/fail assessment (even at EDA stage)
6. **Limitations**: Known limitations, potential biases, data defects
7. **Conclusion**: Whether to proceed to Phase 2

> **Validation boundary (strict):**
> - **Can do**: IC analysis, signal-grouped returns, Rank IC, signal autocorrelation, simple long-short grouping (pandas manual calculation in Notebook)
> - **Should NOT do**: `vbt.Portfolio.from_orders()` full backtest, cost models, Sharpe/MDD/CAGR final metrics, calling `scripts/run_backtest.py`

## Multi-TF / Derivatives Notebook Template (7+2)

When research involves multi-timeframe or derivatives data, extend to 7+2:

```
Sections 0-7: Same as standard
── Extended sections ──
8. Multi-TF Alignment Analysis
   - Per-TF signal direction alignment %
   - Returns comparison: aligned vs not aligned
   - Marginal improvement from best TF combination
9. Cost Impact Assessment
   - Turnover estimation (by TF)
   - Cost erosion vs marginal alpha breakeven analysis
   - Does it pass `net edge after 2× cost > 0` gate?
```

## Strategy Proposal Template

File location: `docs/research/YYYYMMDD_<strategy_name>_proposal.md`

```markdown
# Strategy Proposal: <Strategy Name>

## 0. Archetype Classification
- **Archetype**: [Trend / MR / Carry / Volatility / Event / Hybrid]
- **Return Profile**: [Positive skew / Negative skew / Symmetric / Bimodal]
- **Expected Win Rate**: [XX%]
- **Expected R:R**: [X.X]
- **Estimated Trades/yr**: [XXX]
- **Cost Sensitivity**: [Low / Moderate / Critical]
- **Primary Kill Criteria Applied**: [describe which check was run]

## 0.5 Portfolio Context
- **Target Gap (Alpha Coverage Map)**: [which gap in ALPHA_RESEARCH_MAP.md? cite item #]
- **Expected Correlation with Production**: [X.XX] (< 0.3 for standalone)
- **Integration Target**: [Filter / Overlay / Standalone / Portfolio Layer]
- **Priority Score (5 factors)**: [X.X/5.0]
- **Historical research on this data-signal combo**: [cite taxonomy entry or "none (first study)"]

## 1. Hypothesis
## 2. Mechanism
## 3. Market Regime Target
## 4. Expected Edge Source
## 5. Data Dependencies
## 6. Primary Risk / Failure Mode
## 7. Data Requirements & Coverage Check
## 8. Ablation Plan
## 9. Validation Gates
## 10. Promotion Criteria
## 11. Rollback Criteria
## 12. Evidence
## 13. Blend Configuration (if applicable)
```

## Blend Research Guidelines (Multi-Strategy)

When exploring strategy combinations:

1. **Signal Correlation**: Calculate signal/return correlation (< 0.5 for blend value)
2. **Per-symbol Performance Diff**: Identify complementary symbols for per-symbol routing
3. **Combination Methods**:
   - **Linear weighting**: `final = w_A * signal_A + w_B * signal_B` (simple but signals may cancel)
   - **Confirmatory**: `signal_B` scales `signal_A` (preserves main direction, adjusts magnitude)
   - **Regime-switch**: Choose strategy based on market state (needs ADX etc.)
   
   ⚠️ **Linear weighting trap**: If two strategies have opposite directions, linear weighting cancels out. Prefer Confirmatory or Per-symbol routing.

### Handoff for Blend Strategies Must Include:
- Recommended blend weights (or per-symbol weight matrix)
- Strategy implementation names (`@register_strategy` names)
- Known `auto_delay` settings (which sub-strategies use `auto_delay=False`)
