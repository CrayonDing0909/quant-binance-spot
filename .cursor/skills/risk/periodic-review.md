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
  -c config/prod_candidate_htf_lsr.yaml --real --notify

# 3. Alpha Decay scan
PYTHONPATH=src python scripts/monitor_alpha_decay.py \
  -c config/prod_candidate_htf_lsr.yaml

# 4. Trade review (run for each live strategy)
PYTHONPATH=src python scripts/trade_review.py \
  -c config/prod_candidate_htf_lsr.yaml --days 7 --with-replay

# 5. Backtest↔Live consistency replay (run for each live strategy)
PYTHONPATH=src python scripts/validate_live_consistency.py \
  -c config/prod_candidate_htf_lsr.yaml -v
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
PYTHONPATH=src python scripts/query_db.py -c config/prod_candidate_htf_lsr.yaml trades

# 4. Production report
PYTHONPATH=src python scripts/prod_report.py
```

### Monthly Extra Checks

- Correlation significantly changed? (pairwise corr shifts in bull/bear markets)
- Kelly fraction needs adjustment? (win rate or payoff ratio changed)
- Circuit breaker threshold needs adjustment?
- Alpha Decay IC persistently declining?

> **Division of labor**: Risk Manager leads monthly review (MC + correlation + Kelly calibration).
> If alpha decay is detected (specific symbol IC declining, rolling SR degrading), hand off to Quant Researcher for deep IC analysis.
> Avoid duplicate work — you calculate correlation once; Researcher only intervenes when deep analysis is needed.
