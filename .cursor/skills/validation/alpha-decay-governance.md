---
description: Alpha Decay methodology spec, ownership, thresholds, calibration protocol
globs:
alwaysApply: false
---
# Skill: Alpha Decay Governance

> Single source of truth for Alpha Decay methodology. All agents MUST follow this spec.
> **Last updated**: 2026-03-04

## 1. Ownership

| Role | Responsibility | When |
|------|---------------|------|
| **Quant Researcher** (OWNER) | Define methodology, formula, thresholds, calibration | After every strategy change or threshold recalibration |
| **Quant Developer** | Implement in `ic_monitor.py` + `validate.py` + tests | When Researcher updates spec |
| **Risk Manager** | Act on results (hold / reduce / flatten) | Weekly review + on FAIL |

**One sentence**: Researcher defines, Developer implements, Risk executes.

## 2. IC Formula Spec

### Forward Return

```python
# IC monitoring uses close-to-close forward return with look-ahead shift.
# This is acceptable because IC measures signal QUALITY, not trade PnL.
# Backtest PnL uses open prices — different concern.
forward_returns = prices.pct_change(forward_bars).shift(-forward_bars)
```

- `forward_bars` default: 24 (1h data = 24h horizon)
- Price series: `df["close"]` (for IC quality measurement)
- Signal filtering: only bars where `signal != 0` (active positions)

### Spearman Rank Correlation

IC = `spearmanr(signal, forward_return)` — robust to outliers, no linearity assumption.

## 3. Decay Formula (with Small-Denominator Guard)

The naive ratio `1 - recent_ic / historical_ic` explodes when `historical_ic` is near 0.

### Rule

```python
min_ic_denominator = 0.01  # configurable

if abs(historical_ic) < min_ic_denominator:
    # Small denominator: use absolute difference instead of ratio
    ic_decay_pct = abs(historical_ic - recent_ic)
    # Interpret: raw IC difference, not percentage
else:
    # Normal case: ratio-based decay
    ic_decay_pct = 1.0 - (recent_ic / historical_ic)
```

This prevents +240%, +1500% nonsense decay values.

## 4. Three-Layer Gate (PASS/FAIL)

All three gates must PASS for the overall alpha_decay check to PASS.

### Gate A — Quality (recent IC floor)

- Metric: `avg_recent_ic` across all symbols
- Threshold: `>= 0.005` (TSMOM strategies have structurally low IC, typically 0.005-0.015)
- Rationale: IC below 0.005 means signal is indistinguishable from noise

### Gate B — Stability (decay magnitude)

- Metric: `avg_decay_pct` (with small-denominator guard applied)
- Threshold: `<= 0.60` (60% decay from historical)
- When small-denominator guard activates: threshold is `<= 0.02` (absolute IC difference)

### Gate C — Confidence (alert count)

- Metric: total `critical_alerts` across all symbols
- Threshold: `<= 2` (allow up to 2 symbols to have transient issues)
- Critical alert triggers: IC near zero + decay above threshold

### PASS/FAIL Output

```
PASS: All 3 gates satisfied — alpha is healthy
FAIL: At least 1 gate failed — escalate per Section 6
```

## 5. Calibration Protocol

Thresholds MUST be re-calibrated when:
1. Production strategy changes (new config freeze)
2. Alpha decay FAIL persists for 2+ consecutive weekly checks
3. Market regime shift detected (bear/bull transition)

### Calibration Steps (Researcher)

1. Run `monitor_alpha_decay.py` on full history
2. Compute IC distribution: mean, std, 10th percentile
3. Set `recent_ic_min` = max(IC_10th_percentile, 0.003) — never below noise floor
4. Set `max_decay_pct` based on historical IC variability
5. Document rationale in commit message
6. Hand off to Developer for implementation

## 6. Escalation Path

```
Alpha Decay FAIL
  |
  v
Researcher: Re-analyze (is this real decay or threshold miscalibration?)
  |
  +-- Threshold miscalibration --> Recalibrate (Section 5) --> Developer implements
  |
  +-- Real decay detected --> Risk Manager decides action:
        |
        +-- Mild (1 gate failed, borderline) --> WARNING, increase monitoring frequency
        +-- Moderate (2 gates failed) --> REDUCE position size by 50%
        +-- Severe (all 3 gates failed, persistent) --> FLATTEN, strategy review
```

## 7. Implementation Files

| File | What it does |
|------|-------------|
| `src/qtrade/validation/ic_monitor.py` | `RollingICMonitor` — core IC calculation + alerts |
| `scripts/monitor_alpha_decay.py` | Standalone monitoring script (cron-compatible) |
| `scripts/validate.py` | `run_alpha_decay_check()` — validation gate integration |
| `config/validation.yaml` | `alpha_decay` section — all thresholds |
| `scripts/cron_alpha_decay.sh` | Weekly cron job |

## 8. Anti-Patterns (DO NOT)

- DO NOT set `recent_ic_min > 0.02` for momentum strategies (IC is structurally low)
- DO NOT use ratio-based decay when `|historical_ic| < 0.01` (use absolute diff)
- DO NOT skip calibration after strategy change
- DO NOT treat alpha decay FAIL as "just a warning" — it requires explicit Researcher analysis
- DO NOT hardcode thresholds in code — all thresholds come from `validation.yaml`
