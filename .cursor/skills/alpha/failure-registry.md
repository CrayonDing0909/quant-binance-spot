---
description: Historical failure post-mortems and prevention rules for alpha research
globs:
alwaysApply: false
---
# Skill: Failure Post-Mortem Registry

> Loaded by Alpha Researcher at the start of any new research direction.
> **Scan this table before starting — don't repeat known failures.**

## Registry

| Failure Case | Date | Lesson | Prevention Rule | Wasted Resources |
|-------------|------|--------|----------------|-----------------|
| **4h TF Optimization** | 2026-02-27 | When testing new signal with existing filter, improvement may come from filter's look-ahead bias, not new signal | Always test **pure signal IC** first, then combined; report both. If pure IC < 30% of combined IC, attribute to filter | Quant Dev ~4h |
| **XSMOM Cross-Sectional Momentum** | 2026-02-27 | Crypto high cross-correlation (~0.7) structurally kills cross-sectional ranking; equity-effective cross-sectional strategies don't transfer to crypto | Run `df.pct_change().corr()` before cross-sectional research; avg pairwise corr > 0.5 → direct FAIL | Quant Dev ~3h |
| **CVD IC Bias** | 2026-02-27 | IC calculation bugs are silent and fatal: initial IC=+0.019, strict calculation IC=+0.001 (19× difference) | Any surprising IC must be cross-validated with at least 2 alternative methods | Misleading conclusions |
| **Taker Vol Over-Exploration** | 2026-02-27 | When base signal is too weak (IC=-0.006), exploring 14 variants has diminishing returns | If strongest raw signal IC < 0.01, stop exploring variants, declare alpha source too weak | Alpha Research ~6h |
| **BB Mean Reversion** | 2026-02-25 | IC positive (+0.02~0.05) but gross PnL all negative (PF 0.83-0.88) because IC cannot capture payoff asymmetry | MR strategies must first simulate gross PnL/trade with TP/SL, not just IC | Alpha Research ~2h |
| **FR Carry** | 2026-02-25 | Funding Rate unstable across coins (SOL/BNB 2yr FR < 0), not a reliable carry source | Carry strategies need premium verified positive across all target symbols in all 2-year windows | Alpha Research ~2h |

## Maintenance Rules

- After each confirmed FAIL, **must** add a row to this table
- Format: Failure Case | Date | Lesson | Prevention Rule | Wasted Resources
- This table is permanent — don't delete old entries (unless root cause is overturned)
