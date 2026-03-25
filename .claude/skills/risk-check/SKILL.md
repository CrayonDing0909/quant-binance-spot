---
description: "Pre-launch risk audit and periodic portfolio review. Use before deploying a new strategy, or for weekly/monthly health checks."
---

# /risk-check — Risk Audit Workflow

Ask the user: **"Pre-launch audit (new strategy) or periodic review (existing portfolio)?"**

---

## Mode A: Pre-Launch Audit

Run these 6 steps before any strategy goes live with real capital.

### Step 1: Monte Carlo Stress Test

4 scenarios, each 1000 simulations:
- Normal market (historical vol)
- 2× volatility regime
- Flash crash (-15% in 1 hour)
- Funding rate spike (3× historical)

Check: max simulated drawdown < circuit breaker threshold × 0.7 (safety margin)

### Step 2: Kelly Fraction Validation

- Compute Kelly fraction from backtest results
- Production leverage must be ≤ half-Kelly
- If strategy uses 3× leverage: verify Kelly supports ≥ 6×

### Step 3: Portfolio Risk Assessment

- Correlation with existing production strategies (want < 0.5)
- Combined portfolio max drawdown estimate
- Marginal Sharpe contribution (positive?)
- Capital allocation recommendation

### Step 4: Risk Limits Check

| Limit | Threshold |
|---|---|
| Max position per symbol | ≤ 40% of capital |
| Max drawdown circuit breaker | Set in config |
| Min trade interval | No infinite loop risk |
| Orphan order cleanup | Enabled in BaseRunner |
| SL/TP mechanism | Configured and tested |

### Step 5: Production Config Audit

- [ ] Config is frozen (not research_*)
- [ ] `funding_rate.enabled: true`
- [ ] `slippage_model.enabled: true`
- [ ] `signal_delay=0` for live (with comment)
- [ ] Symbols match intended universe
- [ ] Leverage matches risk allocation

### Step 6: Observation Period Contract

Define before launch:
- Observation window: minimum 14 days
- Success criteria: live PnL within 2σ of backtest expectation
- Kill criteria: drawdown > X% or Sharpe < Y
- Capital ramp: 25% → 50% → 100% over 3 stages

**Verdict**: `APPROVED` (with conditions) or `BLOCKED` (with specific issues)

---

## Mode B: Periodic Review

### Weekly Quick Check (5 minutes)

```bash
# Health check
PYTHONPATH=src python scripts/health_check.py -c config/prod_candidate_simplified.yaml --real --notify

# Trade review
PYTHONPATH=src python scripts/trade_review.py -c config/prod_candidate_simplified.yaml --days 7

# DB query
PYTHONPATH=src python scripts/query_db.py -c config/prod_candidate_simplified.yaml summary
```

| Check | Warning threshold |
|---|---|
| Weekly PnL | < -3% |
| Max single-day DD | > 5% |
| Trade count anomaly | < 50% or > 200% of expected |
| Signal replay match | < 95% consistency |
| Circuit breaker status | Any trigger in last 7 days |

### Monthly Deep Review

1. **Alpha decay scan**: rolling IC trend, compare to initial validation IC
2. **Factor geometry audit**: `scripts/analyze_factor_geometry.py` — check for creeping redundancy
3. **Symbol governance review**: any symbol consistently underperforming?
4. **Cost model accuracy**: actual vs modeled slippage/funding

```bash
# Signal replay verification
PYTHONPATH=src python scripts/verify_signal_replay.py -c config/prod_candidate_simplified.yaml --days 30 --notify

# Live consistency check (after 14+ days)
PYTHONPATH=src python scripts/validate_live_consistency.py -c config/prod_candidate_simplified.yaml -v
```

### Observation Period Gate (for new strategies)

After the observation window:
- [ ] Live PnL within 2σ of backtest expectation
- [ ] No circuit breaker triggers
- [ ] Signal replay consistency > 95%
- [ ] No unexpected position-side errors

**PASS** → promote to full capital. **FAIL** → investigate before proceeding.
