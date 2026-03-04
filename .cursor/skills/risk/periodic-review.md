---
description: Weekly and monthly periodic portfolio risk review procedures
globs:
alwaysApply: false
---
# Skill: Periodic Risk Review

> Loaded by Risk Manager for weekly quick checks and monthly deep reviews.

## Weekly Quick Check

```bash
source .venv/bin/activate

# 1. Risk Guard dry-run
PYTHONPATH=src python scripts/risk_guard.py \
  --config config/risk_guard_alt_ensemble.yaml --dry-run

# 2. Health check
PYTHONPATH=src python scripts/health_check.py \
  -c config/prod_candidate_simplified.yaml --real --notify

# 3. Alpha Decay scan (automated via cron — check latest report)
#    Cron: scripts/cron_alpha_decay.sh runs weekly Sunday 00:00 UTC
#    Reports: reports/alpha_decay/
#    Governance: .cursor/skills/validation/alpha-decay-governance.md
ls -lt reports/alpha_decay/ | head -5
cat reports/alpha_decay/$(ls -t reports/alpha_decay/ | head -1)
# If alpha_decay gate FAIL:
#   Mild (1 gate failed)   → WARNING, increase monitoring to daily
#   Moderate (2 gates)     → REDUCE position size by 50%
#   Severe (3 gates, 2wk)  → FLATTEN, trigger strategy review with Researcher

# 4. Trade review (run for each live strategy)
PYTHONPATH=src python scripts/trade_review.py \
  -c config/prod_candidate_simplified.yaml --days 7 --with-replay

# 5. Backtest↔Live consistency replay (run for each live strategy)
PYTHONPATH=src python scripts/validate_live_consistency.py \
  -c config/prod_candidate_simplified.yaml -v
```

### Weekly Warning Thresholds

| Metric | Warning Line | Action |
|--------|-------------|--------|
| 20D Rolling Sharpe | < 0 (2 consecutive weeks) | Check if strategy is still effective |
| Current Drawdown | > 50% of circuit breaker | Prepare to reduce position |
| Runner exceptions | > 3/week | Notify DevOps |
| Signal flip frequency | Abnormally high | Possible signal noise increase |
| Consistency check FAIL | Any FAIL item | Notify Developer; overlay inconsistency needs immediate fix |
| consistency_rate | < 95% | Mark WARNING, notify Researcher for IC analysis |
| Trade review live/backtest deviation | live SR / backtest SR < 0.30 | Trigger strategy review (see STRATEGY_PORTFOLIO_GOVERNANCE.md R2) |
| Consecutive losses | > 5 trades and not consolidation | Mark WARNING, trigger alpha decay scan |

## Monthly Deep Review

On top of weekly checks, additionally:

```bash
# 1. Full Monte Carlo re-run with latest data
PYTHONPATH=src python -c "
from qtrade.risk.monte_carlo import run_monte_carlo_simulation
# Load latest backtest results and run MC simulation
"

# 2. Correlation matrix refresh (90-day window)
PYTHONPATH=src python -c "
from qtrade.risk.portfolio_risk import calculate_correlation_matrix
corr = calculate_correlation_matrix(returns_dict)
print(corr)
"

# 3. Kelly Fraction re-calibration (using last 6 months live trades)
PYTHONPATH=src python scripts/query_db.py -c config/prod_candidate_simplified.yaml trades

# 4. Production report
PYTHONPATH=src python scripts/prod_report.py
```

### Monthly Extra Checks

- Correlation significantly changed? (pairwise corr shifts in bull/bear markets)
- Kelly fraction needs adjustment? (win rate or payoff ratio changed)
- Circuit breaker threshold needs adjustment?
- Alpha Decay IC persistently declining? (check Gate A/B/C per governance spec)

> **Division of labor**: Risk Manager leads monthly review (MC + correlation + Kelly calibration).
> If alpha decay is detected (specific symbol IC declining, rolling SR degrading), hand off to Quant Researcher for deep IC analysis.
> Avoid duplicate work — you calculate correlation once; Researcher only intervenes when deep analysis is needed.
>
> **Alpha Decay ownership**: Researcher OWNS methodology (formula, thresholds). Risk OWNS action (reduce/flatten). Developer implements.
> See `.cursor/skills/validation/alpha-decay-governance.md` for full spec.

## Architecture Gap Scan (Monthly)

> **Purpose**: Catch ownerless modules, stale thresholds, and misconfigured gates before they cause production incidents.
> **Registry**: `.cursor/skills/dev/feature-ownership-registry.md`

### Procedure

Open the Feature Ownership Registry and check **every row** against these conditions:

| Condition | Severity | Action |
|-----------|----------|--------|
| Module has no Owner or Owner is "TBD" | **HIGH** | Escalate to user — assign owner before next review |
| `Last Calibrated` is >60 days ago | **MEDIUM** | Notify the Owner to recalibrate; if no response, escalate to user |
| A threshold looks unreasonable and no calibration rationale exists | **MEDIUM** | Notify Researcher to validate; document finding |
| A validation gate always PASS across all symbols and all periods | **LOW** | Likely too lenient — flag for Researcher review |
| A validation gate always FAIL across all symbols | **HIGH** | Likely misconfigured (alpha decay incident pattern) — escalate immediately |
| New module exists in code but is not in the registry | **HIGH** | Notify Developer to add it; block next deploy until registered |

### Output Format

In the monthly report, add a section:

```
## Architecture Gap Scan
- ARCHITECTURE_GAP [HIGH]: <module> has no owner — assign before next review
- ARCHITECTURE_GAP [MEDIUM]: <module> last calibrated 75 days ago — notify owner
- No gaps found ✓
```

### Escalation

- **HIGH** gaps: Tag the user directly — these are architecture decisions no agent can make alone.
- **MEDIUM** gaps: Notify the responsible Owner; if unresolved after 1 week, escalate to HIGH.
- **LOW** gaps: Log in report for awareness; address in next relevant session.