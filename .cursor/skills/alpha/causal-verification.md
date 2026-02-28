---
description: EDA causal IC verification protocol, artifact detection checklist
globs:
alwaysApply: false
---
# Skill: EDA Causal Verification Protocol

> Loaded by Alpha Researcher whenever computing IC or making GO/FAIL decisions.
> **Every IC calculation error propagates to handoff decisions, wasting 2-4h of developer time.**

## Causal IC Computation Rules (Mandatory)

1. **Signal-Return Alignment**: All IC must use `signal[t]` vs `return[t+1]`.
   ```python
   # Correct: causal IC
   ic = signal.shift(1).corr(forward_return)  # signal at t, return at t+1
   
   # Wrong: contains look-ahead
   ic = signal.corr(forward_return)  # signal and return same period = look-ahead
   ```

2. **Resample Signal Extra Lag**: If signal uses resampled data (4h/1d from 1h), must add extra `shift(1)` after resample to compensate resampling lag (consistent with production `_apply_htf_filter()` logic).
   ```python
   signal_4h = compute_signal(df_4h)
   signal_4h_causal = signal_4h.shift(1)      # delay 1 HTF bar (resampling lag)
   signal_1h = signal_4h_causal.reindex(df_1h.index, method='ffill')
   ic = signal_1h.shift(1).corr(return_1h)    # another shift(1) for 1h execution delay
   ```

3. **IC Cross-Validation**: Any IC > 0.03 or counter-intuitive result must be verified with **2 independent methods**:
   - Rank IC (Spearman) vs Pearson IC
   - Different lookback windows (at least 2)
   - In-sample (first 70%) vs Out-of-sample (last 30%)
   - If methods differ by > 50%, use the lower value

4. **Confounding Factor Isolation**: When new signal interacts with existing components, **must** report two IC sets:
   - **Pure signal IC**: without any existing filter/overlay
   - **Signal + existing component IC**: with existing filters
   - If improvement mostly comes from existing component (pure IC < 30% of combined IC), conclude "improvement from filter, not new signal"

## Artifact Detection Checklist (Must ALL Pass Before Declaring Alpha)

| # | Check | Pass Criteria | Failure Means |
|---|-------|-------------|--------------|
| A1 | IC cross-year consistency | At least 3 calendar years with same-sign IC | Signal is regime-dependent, unreliable |
| A2 | Causal shift impact | Pre/post-shift IC difference < 50% | Signal has look-ahead contamination |
| A3 | Cross-symbol consistency | At least 6/8 symbols with same-sign IC | Signal only works on specific symbols, not general alpha |
| A4 | Sample size | >= 500 observations per symbol | Statistically unreliable |
| A5 | Base signal strength | Strongest raw signal IC > 0.01 | Alpha source too weak, not worth exploring variants |

> **A5 Hard Rule**: If strongest raw signal IC < 0.01, **immediately stop exploring variants**.
> Lesson (Taker Vol): 14 variants explored for IC=-0.006; when base signal is weak, variant exploration has diminishing returns.
