---
description: Full validation pipeline (11 steps), commands, meta_blend extra checks
globs:
alwaysApply: false
---
# Skill: Validation Pipeline

> Loaded by Quant Researcher when running full validation.

## Pipeline (Sequential, All Steps Required)

## Research-to-Validation Boundary

Before full validation starts, verify what kind of experiment is being promoted:

### Loop A: Alpha Existence

This loop proves the raw mechanism exists.

Expected evidence before backtest-heavy work:
- causal IC / conditional IC
- year-by-year sign stability
- event frequency / trades per year
- pure vs confounded evidence
- orthogonality pre-screen

If Loop A is weak, do **not** spend validation time on TP/SL or fine entry-timing variants.

### Loop B: Trade Expression

This loop assumes the alpha source is already accepted and asks how to express it.

Allowed families:
- `entry_timing`
- `exit_design`
- `position_sizing`
- `portfolio_role`

Expected evidence before full validation:
- comparison against the accepted Loop A baseline
- explicit `what stayed fixed`
- clear statement of which single family changed
- improvement on the target metric without hiding a collapse in trade count or cost robustness

### Hard Rule

Validation should reject packets that say "strategy improved" but cannot tell whether:
- the raw signal improved, or
- only the execution layer changed

That is a research hygiene failure, not just a reporting issue.

### 1. Causality Check
- Signal generated at `close[i]`, executed at `open[i+1]`?
- `signal_delay` correctly set?
- Truncation invariance: different date ranges produce consistent results?

### 2. Backtest Integrity
```bash
PYTHONPATH=src python scripts/run_backtest.py -c config/<cfg>.yaml
```
- `price=df['open']` confirmed
- Cost model enabled: funding_rate + slippage
- Yearly decomposition: any single year carrying all the performance?

### 3. Walk-Forward Analysis
```bash
PYTHONPATH=src python scripts/run_walk_forward.py -c config/<cfg>.yaml --splits 6
```
- OOS Sharpe > 0.3 (for momentum)?
- IS vs OOS performance gap?

### 4. CPCV Cross-Validation
```bash
PYTHONPATH=src python scripts/run_cpcv.py -c config/<cfg>.yaml --splits 6 --test-splits 2
```
- PBO (Probability of Backtest Overfitting) < 0.5?

### 5. Statistical Tests
```bash
PYTHONPATH=src python scripts/validate.py -c config/<cfg>.yaml --full
```
- DSR (Deflated Sharpe Ratio): significant after multiple comparisons?
- Bootstrap confidence interval

### 6. Cost Stress Test
- 1.5× and 2.0× cost multiplier: still profitable?

### 7. Delay Stress Test
- +1 bar extra delay, performance drop?
- Drop > 50% → timing-sensitive (fragile)

### 8. Overlay Consistency Check (Stage D.5)
- Developer completed overlay ablation (naked vs overlay)?
- Overlay mode + params consistent across config, backtest path, live path?
- If component claims to replace overlay → require 3-way ablation data

### 9. Pre-Deploy Consistency Check (MANDATORY — **post-deployment**)

> **Note**: This check requires live trade records, which only exist after deployment.
> It is NOT a pre-deployment gate — it becomes mandatory **after the observation period warmup** (14+ days of live data).
> During pre-deployment validation, this step is skipped (`consistency.enabled: false` in `validation.yaml`).

```bash
# Run after 14+ days of live data:
PYTHONPATH=src python scripts/validate_live_consistency.py -c config/prod_candidate_simplified.yaml -v
```
- Backtest/live path consistency: config passthrough, strategy context, signal consistency, overlay consistency
- **Must PASS during observation period review** — if FAIL, investigate signal mismatch before promoting to production

### 10. Alpha Decay Check (V10)

> **Governance**: `.cursor/skills/validation/alpha-decay-governance.md`
> **Owner**: Quant Researcher defines methodology + thresholds; Developer implements; Risk acts.

```bash
# Via validate.py (integrated gate):
PYTHONPATH=src python scripts/validate.py -c config/<cfg>.yaml -v config/validation.yaml --only alpha_decay

# Standalone monitoring:
PYTHONPATH=src python scripts/monitor_alpha_decay.py -c config/<cfg>.yaml
```

Three-layer gate (all must PASS):
- **Gate A (quality)**: avg recent IC >= `recent_ic_min` (default 0.005 for TSMOM)
- **Gate B (stability)**: avg decay <= `max_decay_pct` (with small-denominator guard when |historical_ic| < 0.01)
- **Gate C (confidence)**: critical alerts <= `max_critical_alerts`

On FAIL:
1. Researcher re-analyzes: real decay or threshold miscalibration?
2. If miscalibration → recalibrate thresholds (see governance spec Section 5)
3. If real decay → Risk Manager executes action protocol (WARNING / REDUCE / FLATTEN)

### 11. Factor Orthogonality Audit (V11 — NEW)

> **Tool**: `src/qtrade/validation/factor_orthogonality.py`
> **Script**: `scripts/analyze_factor_geometry.py`
> **Owner**: Quant Researcher runs audit; Alpha Researcher uses in EDA pre-screen

```bash
# Full factor geometry audit
PYTHONPATH=src python scripts/analyze_factor_geometry.py -c config/prod_candidate_simplified.yaml

# Per-candidate marginal information ratio (in research scripts)
from qtrade.validation.factor_orthogonality import marginal_information_ratio
```

Three checks:
- **Correlation Matrix**: No pair of production signals should have |corr| > 0.50
- **PCA Effective Factors**: n_effective_factors >= n_signals × 0.60 (at least 60% independent)
- **Marginal Info**: Any new candidate signal must have R² < 0.50 against existing signals (G0 gate)

On FAIL:
1. If correlation matrix shows high pairs: investigate if one signal subsumes the other → ablation to decide which to keep
2. If PCA effective factors too few: production may have structural redundancy → recommend simplification
3. If candidate R² > 0.50: do not proceed to ablation — this is the same factor in different form

**Historical Context**: OI/On-chain/Macro/VPIN all passed traditional G1-G6 gates but were structurally redundant with HTF filter. This audit would have caught them at the R² stage, saving ~16h of developer ablation time.

### 12. Regime-Stratified CPCV (V12 — NEW)

> **Tool**: `regime_stratified_cpcv()` in `prado_methods.py`

```bash
# Integrated into validate.py (coming soon)
# Or standalone:
from qtrade.validation.prado_methods import regime_stratified_cpcv
result = regime_stratified_cpcv(symbol, data_path, cfg, strategy_name, btc_prices)
```

- Standard CPCV + regime breakdown of OOS performance
- Checks: `regime_concentration_pct < 0.70` (single regime should not contribute >70% of OOS PnL)
- `is_regime_dependent = True` → strategy may be a regime proxy, not a robust alpha source
- Uses `auto_detect_regimes()` from `regime_analysis.py`

### 13. meta_blend Extra Validation

When auditing a `meta_blend` strategy, additionally check:

| Check | Description | Pass Criteria |
|-------|-------------|---------------|
| **Ablation Study** | Pure A, Pure B, A+B comparison | Blend Sharpe >= max(A, B) or MDD improved >20% |
| **Per-symbol IC** | IC contribution by sub-strategy per symbol | Blend IC better in >=60% symbols |
| **Signal direction conflict** | % time sub-strategies have opposing directions | Conflict < 40% |
| **auto_delay** | meta_blend must be `auto_delay=False` | Must be False (prevents double-delay) |
| **Per-symbol tier overfitting** | CPCV/PBO check per-symbol configs | PBO < 0.5 |
| **Weak symbol detection** | Symbols with WFA OOS+ < 50% or 2× cost unprofitable | Recommend remove or deweight |

**Common Issues**:
- **BTC Sharpe abnormally low** → almost certainly `auto_delay` double-delay (BTC's `breakout_vol_atr` has built-in delay)
- **Carry signal structural shorting** → BasisCarry may persistently short in bull market; verify confirmatory mode
- **Persistent negative IC symbols** → low-cap coins (XRP, LTC) carry unstable; suggest `tsmom_only` tier

## Family-Specific Validation Focus

| Experiment family | What validation should focus on |
|------------------|---------------------------------|
| `signal_mechanism` | causality, IC stability, orthogonality, symbol breadth, regime dependence |
| `entry_timing` | delay stress, false-entry reduction, conditional trade quality, trade count loss |
| `exit_design` | expectancy, MDD, hold time, tail loss, cost-adjusted edge |
| `position_sizing` | portfolio Sharpe delta, concentration, turnover, capital efficiency |
| `portfolio_role` | correlation, marginal SR, blend / overlay ablation, replacement viability |

## Validation Division of Labor

> **You are the sole owner of full validation.** Developer only runs `validate.py --quick`.
> WFA, CPCV, DSR, Cost Stress, Delay Stress — all independently executed by you.
> **Do not trust Developer's validation numbers** — rerun with original config and data.
