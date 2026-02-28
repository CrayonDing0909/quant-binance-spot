---
description: Full validation pipeline (11 steps), commands, meta_blend extra checks
globs:
alwaysApply: false
---
# Skill: Validation Pipeline

> Loaded by Quant Researcher when running full validation.

## Pipeline (Sequential, All Steps Required)

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

### 9. Pre-Deploy Consistency Check (MANDATORY gate)
```bash
PYTHONPATH=src python scripts/validate_live_consistency.py -c config/research_<name>.yaml
```
- Backtest/live path consistency: config passthrough, strategy context, signal consistency, overlay consistency
- **Must PASS before GO_NEXT — no exceptions**

### 10. Alpha Decay Monitoring
```bash
PYTHONPATH=src python scripts/monitor_alpha_decay.py -c config/<cfg>.yaml
```
- Rolling IC stable?
- Annual IC persistently declining?

### 11. meta_blend Extra Validation

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

## Validation Division of Labor

> **You are the sole owner of full validation.** Developer only runs `validate.py --quick`.
> WFA, CPCV, DSR, Cost Stress, Delay Stress — all independently executed by you.
> **Do not trust Developer's validation numbers** — rerun with original config and data.
