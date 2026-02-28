---
description: Risk audit report format, judgment criteria, action items YAML, handoff protocols, Next Steps output
globs:
alwaysApply: false
---
# Skill: Report Format & Handoff Protocol

> Loaded by Risk Manager when writing audit reports or making verdicts.

## Judgment Criteria

### Pre-Launch Verdict

| Verdict | Condition | Follow-up |
|---------|-----------|-----------|
| `APPROVED` | All 5 steps pass | Hand to DevOps for deployment |
| `CONDITIONAL` | Mostly pass with caveats | Deploy with conditions (e.g. reduce position, shorter observation) |
| `REJECTED` | Any critical step fails | Return to Quant Developer with specific issues |

### Periodic Verdict

| Verdict | Condition | Follow-up |
|---------|-----------|-----------|
| `HEALTHY` | All metrics normal | Continue running |
| `WARNING` | Approaching warning lines | Increase monitoring frequency, prepare contingency |
| `REDUCE` | Risk metrics deteriorating | Notify DevOps to reduce positions |
| `FLATTEN` | Severe risk event | Notify DevOps to flatten immediately |

## Pre-Launch Audit Report Template

```markdown
# Risk Audit Report — <Strategy Name>

## Audit Type: Pre-Launch / Weekly / Monthly
## Date: YYYY-MM-DD
## Auditor: Risk Manager Agent

### 1. Monte Carlo Summary
| Scenario | Metric | Value | Pass/Fail |
|----------|--------|-------|-----------|
| MC1 | 5th pct CAGR | ... | ... |
| MC2 | 95th pct MDD | ... | ... |
| MC3 | Median Sharpe | ... | ... |
| MC4 | Sharpe decay | ... | ... |

### 2. Position Sizing
- Full Kelly: ...%
- Config position size: ...%
- Kelly utilization: ...% (should be <= 25%)

### 3. Portfolio Risk
- Portfolio VaR (95%): ...%
- Max pairwise correlation: ...
- Diversification ratio: ...

### 4. Risk Limits
- Max drawdown config: ...%
- Current drawdown: ...%
- Headroom: ...%

### 5. Launch Guard
- Config hash: PASS/FAIL
- Env vars: PASS/FAIL
- Data freshness: PASS/FAIL
- Risk guard status: PASS/FAIL

### Verdict: APPROVED / CONDITIONAL / REJECTED
### Reason: <detailed reason>
### Conditions (if CONDITIONAL): <conditions>
### Next Review Date: YYYY-MM-DD
```

## Periodic Review Action Items YAML

```yaml
# ── ACTION ITEMS ──
verdict: WARNING  # HEALTHY / WARNING / REDUCE / FLATTEN

action_items:
  - id: 1
    severity: WARNING    # WARNING / CRITICAL
    category: concentration  # concentration / alpha_decay / correlation / drawdown / leverage
    symbols: [BTCUSDT]
    description: "BTC effective allocation 35.4% exceeds 30% threshold"
    current_value: "weight: 0.8900"
    suggested_value: "weight: 0.4450"
    next_agent: quant-developer
    next_action: "Run BTC 1x vs 2x backtest comparison"

next_review_date: YYYY-MM-DD
```

> Action Items are structured suggestions for orchestrator, not auto-executed commands.

## Handoff Protocol

### Receive (from Quant Researcher)
- `GO_NEXT` verdict + full backtest report → Start Pre-Launch Audit

### Receive (from Orchestrator — /risk-review)
- Weekly routine → Output Periodic Verdict + Action Items

### Receive (from Quant Developer — /risk-action results)
- Comparison backtest results → Final verdict: APPROVED (deploy new config) / REJECTED (keep current)

### Send (to DevOps)
- `APPROVED`: Risk Audit Report + config path + deployment recommendation

### Return (to Quant Developer)
- `REJECTED`: Specific failure items + suggested fix direction

## Next Steps Output

### APPROVED:
| Option | Agent | Prompt |
|--------|-------|--------|
| A | `@devops` | "APPROVED. Deploy <strategy>. Config: `config/...yaml`, Risk Audit: [path]" |
| B | `@devops` | "APPROVED, paper trading 1 week first. Config: `config/...yaml` --paper" |

### CONDITIONAL:
| Option | Agent | Prompt |
|--------|-------|--------|
| A | `@quant-developer` | "CONDITIONAL. Conditions: [...]. Adjust config and resubmit" |
| B | `@devops` | "CONDITIONAL deploy with reduced params: [adjustments]" |

### REJECTED:
| Option | Agent | Prompt |
|--------|-------|--------|
| A | `@quant-developer` | "REJECTED. Failures: [...]. Suggested fixes: [...]" |
| B | `@alpha-researcher` | "Strategy <name> risk-failed. Fundamental risk: [...]. Reassess design" |
