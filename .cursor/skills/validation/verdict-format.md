---
description: Verdict criteria, report output format, handoff protocol, Next Steps output
globs:
alwaysApply: false
---
# Skill: Verdict & Report Format

> Loaded by Quant Researcher when making judgments and writing reports.

## Judgment Criteria

> **Important**: GO_NEXT requires Pipeline Steps 8 (Overlay Consistency) and 9 (Pre-Deploy Consistency) to ALL PASS. Even if all statistical gates pass, consistency failure blocks GO_NEXT.

| Verdict | Condition |
|---------|-----------|
| `GO_NEXT` | All gates pass (incl. consistency), OOS Sharpe > 0.3, profitable under cost stress |
| `KEEP_BASELINE` | Candidate not better than baseline, maintain status quo |
| `NEED_MORE_WORK` | Potential but some gates failed, needs improvement |
| `FAIL` | False alpha or fundamental problem, reject hypothesis |

## Report Output Format

Every review must produce:

1. **Change Summary**: What changed, what's the goal
2. **Metrics Table**: Baseline vs Candidate (Return, Sharpe, MDD, Calmar, Trades)
3. **Yearly Table**: Annual performance breakdown
4. **WFA Summary**: Walk-forward IS/OOS Sharpe
5. **Cost Stress Table**: 1×, 1.5×, 2.0× cost Sharpe
6. **Falsification Matrix**: Gates pass/fail
7. **Final Verdict**: `GO_NEXT` / `KEEP_BASELINE` / `NEED_MORE_WORK` / `FAIL` + reason
8. **Evidence Paths**: All referenced report file paths

## Handoff Protocol

### Receive (from Quant Developer)
- BacktestResult + report
- Confirm config is `config/research_*.yaml` (not production)
- Start validation pipeline

### Send (to Risk Manager)
- `GO_NEXT`: Full validation report + BacktestResult + config path

### Return (to Quant Developer)
- `NEED_MORE_WORK`: Specific items to improve

### Return (to Alpha Researcher)
- If hypothesis itself is flawed (not implementation)

## Next Steps Output

### GO_NEXT:
| Option | Agent | Prompt |
|--------|-------|--------|
| A | `@risk-manager` | "GO_NEXT for <strategy>. Config: `config/research_<name>.yaml`, report: [path], OOS Sharpe=X, MDD=Y%, DSR p=Z" |
| B | `@quant-developer` | "GO_NEXT but suggest improving <items> before risk audit" |

### NEED_MORE_WORK:
| Option | Agent | Prompt |
|--------|-------|--------|
| A | `@quant-developer` | "NEED_MORE_WORK. Failed gates: [...]. Please fix: [suggestions]" |
| B | `@alpha-researcher` | "Strategy <name> failed, root cause may be hypothesis: [analysis]" |

### FAIL:
| Option | Agent | Prompt |
|--------|-------|--------|
| A | `@alpha-researcher` | "FAIL: [fundamental issue]. Explore alternative: [leads if any]" |
| B | (none) | Archive `config/research_*.yaml` → `config/archive/`, research ends |

### Rules
- Next Steps prompt must include: verdict, config path, failed gates (if any), key metrics
- GO_NEXT → Option A is default; NEED_MORE_WORK/FAIL → choose Developer or Researcher based on root cause
