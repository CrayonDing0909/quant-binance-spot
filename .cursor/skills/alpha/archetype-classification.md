---
description: Strategy archetype classification, per-archetype analysis methods, kill criteria, cost framework
globs:
alwaysApply: false
---
# Skill: Strategy Archetype Classification & Cost Framework

> Loaded by Alpha Researcher when starting any new research direction.

## Step 1 — Classify the Archetype

| Archetype | Core Mechanism | Return Distribution | Key Metrics (beyond Sharpe) |
|-----------|---------------|--------------------|-----------------------------|
| Trend Following | Momentum persistence | Positive skew, low WR (30-45%), high R:R | Tail Ratio, Avg Win/Loss, Time in Market |
| Mean Reversion | Price overextension correction | Negative skew, high WR (55-70%), low R:R | Avg holding time, Turnover/yr, Gross PnL/trade, Cost erosion % |
| Carry / Yield | Structural premium harvesting | Concentrated, occasional large drawdowns | FR stability, Crowding index, Regime shift sensitivity |
| Volatility | Vol regime transitions | Bimodal | Squeeze frequency, Direction accuracy, Conditional return |
| Event-Driven | Discrete market events | Sparse but large returns | Hit rate, Signal rarity, FPR, Event timing precision |
| Multi-TF Resonance | Multi-timeframe signal confluence | Like Trend but higher WR | TF alignment %, Signal confirmation %, vs single-TF improvement |
| Microstructure | Order flow / Taker behavior | Depends on parent strategy (overlay) | Entry timing improvement, Slippage reduction, CVD divergence hit rate |

## Step 2 — Apply Archetype-Specific Analysis

Each archetype requires specialized analysis beyond the standard 7-section Notebook:

- **Trend**: IC decay across lookbacks, persistence half-life, regime decomposition (trending vs ranging)
- **MR**: **Gross PnL/trade simulation with TP/SL**, turnover estimation, cost breakeven analysis, return distribution skew check
- **Carry**: FR regime stability, crowding risk proxy, drawdown clustering analysis
- **Volatility**: Conditional returns (squeeze vs non-squeeze), direction accuracy, squeeze duration distribution
- **Event**: Event frequency, FPR analysis, signal-to-event time lag
- **Multi-TF**: TF alignment frequency, single vs multi-TF signal quality comparison (IC/hit rate), HTF filter effect (false signal reduction % vs capture loss %)
- **Microstructure**: Taker imbalance + CVD IC decay, overlay before/after execution quality comparison, cost savings vs delay risk

## Step 3 — Kill Criteria (Early Termination)

If triggered, **stop analysis immediately**:

| Archetype | Kill Condition |
|-----------|---------------|
| Trend | All lookbacks (24h-720h) IC < 0.01 |
| **MR** | **Gross PnL/trade < 0 (before costs) — cannot be fixed by parameter optimization** |
| Carry | Any 2-year window with negative average premium |
| Volatility | Post-squeeze direction accuracy < 52% |
| Event | False positive rate > 80% |
| Multi-TF | Multi-TF aligned signal IC not better than single TF (improvement < 5%) |
| Microstructure | 5m/15m signal IC zeroes out at 2× slippage (net edge < 0) |

> **Crypto meta-insight**: Crypto returns have **positive skew + fat tails**. This structurally favors trend following (captures right tail) and punishes mean reversion (killed by right-tail adverse positions). Any MR strategy must account for this asymmetry.

## Step 3.5 — Structural Prerequisites (Must Pass Before Deep Analysis)

| Strategy Type | Prerequisite | How to Check | If Fails → |
|--------------|-------------|-------------|-----------|
| **Cross-sectional** | Universe avg pairwise return correlation < 0.5 | `df.pct_change().corr()` off-diagonal mean | **Direct FAIL** (crypto almost always > 0.5) |
| **TF replacement** (e.g. 1h→4h) | New TF must show IC improvement **after removing existing pipeline contribution** | Test pure new-TF signal IC, then with existing filters; pure IC improvement > 0.005 | FAIL or downgrade to "marginal improvement" |
| **Resample signals** | IC must use lagged version; pre/post-lag IC delta < 50% | Compare `signal.corr(ret)` vs `signal.shift(1).corr(ret)` | Signal has look-ahead contamination |
| **Cross-market transplant** | Must verify crypto market structural differences | Confirm crypto has similar microstructure | Cannot assume literature conclusions apply |

## Cost Sensitivity Framework

| Archetype | Typical Trades/yr | Cost Sensitivity | Early Cost Check |
|-----------|:-:|:-:|------|
| Trend | 50-200 | Low | Confirm edge > 2× round-trip cost |
| **MR** | **300-1500** | **Fatal** | **Must simulate gross PnL/trade with TP/SL first** |
| Carry | 10-50 | Negligible | Focus on FR stability |
| Volatility | 30-100 | Moderate | Confirm squeeze frequency supports expected trade volume |
| Event | 20-100 | Low-Medium | Confirm event frequency in sample period |
| Multi-TF | Same as parent | Same as parent | Extra TF alignment filter should not significantly increase turnover |
| Microstructure (5m/15m overlay) | 200-500 | **High** | **5m turnover ≈ 12× 1h; net edge after 2× cost > 0** |

### High-Frequency Cost Warning

> ⚠️ **TF vs Cost rule of thumb**:
> - **1h**: round-trip ~0.12% (fees 0.04%×2 + slippage 0.04%)
> - **15m**: if turnover 4× → annualized cost ~4× → need Sharpe > 2.0 to cover
> - **5m**: if turnover 12× → annualized cost ~12× → very few strategies pass
> - **1m**: unless market making or stat arb, costs won't be covered
>
> **Rule**: For any < 1h strategy, first estimate `annual_turnover × round_trip_cost`.
> If > 50% of gross Sharpe, direct FAIL.

> **MR Iron Rule**: Always simulate gross PnL with explicit TP/SL first.
> `win_rate × avg_win - loss_rate × avg_loss` must be positive **before** costs.
> If gross expectancy is negative, no parameter tuning can save it.
