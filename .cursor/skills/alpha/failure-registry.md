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
| **OI Regime Filter (stacking)** | 2026-02-28 | OI standalone SR=4.12 > HTF=3.86, but stacking HTF+OI=4.04 (incremental +4.66% < 5%). Two regime filters on same signal both gate low-conviction bars → over-filter (5/8 symbols SR dropped). Redundancy: OI and HTF both remove similar weak-signal bars | When stacking two regime filters, check overlap ratio first. If filter A and B gate >50% of the same bars, stacking won't help. Test replacement (A→B) before addition (A+B) | Quant Dev ~3h |
| **Entropy Regime (PE/SE/ApEn)** | 2026-03-02 | Crypto 1h price entropy has zero predictive power (all IC < 0.01). Unlike VP (vol proxy, corr=0.71), entropy IS independent from vol (corr < 0.2), but simply contains no useful information for 1h forward returns. 1h crypto returns are too noisy for ordinal/distributional entropy to capture regime shifts | Before researching "complexity" or "information-theoretic" signals on crypto 1h, verify base IC > 0.01 first. Confounding-first design saved time (30min vs hours) | Alpha ~30min |
| **Orderflow Composite Standalone (taker_vol_ratio proxy)** | 2026-03-02 | taker_vol_ratio 是小時級 buy/sell 聚合比率，IC(24h)=+0.003 太弱。即使策略邏輯正確（contrarian mode 8/8 positive pre-cost SR）且與 TSMOM 幾乎完全正交（corr=-0.023），alpha 仍不夠覆蓋交易成本（年化成本 ~143% >> alpha ~5.7%）。**根因是數據解析度而非策略邏輯**：taker_vol_ratio 在小時聚合時丟失逐筆交易的 informed/noise trading 信息。Tick-level OFI (Cont 2014) 數據已在 aggTrades pipeline 中可用，理論上信息量更大 | Proxy 數據（aggregated ratio）用於建立獨立策略前，必須先驗證 raw IC > 0.01。如果小時級 proxy IC < 0.005，不要嘗試構建策略，改用 tick-level 數據。corr(TSMOM) 低不等於有足夠 alpha | Alpha+Dev ~6h |

## Maintenance Rules

- After each confirmed FAIL, **must** add a row to this table
- Format: Failure Case | Date | Lesson | Prevention Rule | Wasted Resources
- This table is permanent — don't delete old entries (unless root cause is overturned)
- **Trial Registry**: 每次新研究方向 FAIL/PASS/KEEP_BASELINE 後，**必須**更新 `config/validation.yaml` 的 `trial_registry` section：
  1. 新增一行到 `directions` 列表（id, description, estimated_variants, status, date）
  2. 更新 `cumulative_n_trials` = Σ 所有 directions 的 estimated_variants
  3. 這是 DSR（Deflated Sharpe Ratio）多重測試修正的基礎，遺漏會導致 DSR 低估過擬合風險