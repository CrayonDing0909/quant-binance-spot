---
description: "Quant signal knowledge base — optimal parameters, academic evidence, and known pitfalls for crypto derivatives/on-chain signals. Use when designing new strategies, choosing signal parameters, or evaluating research directions."
---

# /quant-signals-reference — Signal Construction Knowledge Base

When the user is researching or implementing a new strategy that involves any of these signals, provide the evidence-based optimal parameters and known pitfalls.

## How to Use

1. Read the relevant reference file(s) from `references/` below
2. Compare the user's proposed parameters against the academic/practitioner consensus
3. Flag any known pitfalls or decay risks
4. Suggest parameter ranges based on the literature

## Available References

| Signal | File | When to Consult |
|---|---|---|
| Coinbase Premium | `references/coinbase_premium.md` | Cross-exchange spread, institutional flow |
| Funding Rate | `references/funding_rate.md` | FR carry, contrarian reversal |
| Open Interest | `references/open_interest.md` | OI regime, liquidation cascade, 4-quadrant |
| Long/Short Ratio | `references/lsr_contrarian.md` | Retail sentiment, crowding |
| Stablecoin Flow | `references/stablecoin_flow.md` | On-chain liquidity, capital rotation |
| Multi-Factor | `references/multi_factor_composite.md` | Signal combination, weighting, portfolio |
| **BTC vs ETH** | `references/btc_vs_eth_structure.md` | **MUST READ** when applying any signal cross-asset |

## IMPORTANT: Parameters Are Context-Dependent

Reference files contain parameters from academic/practitioner research, but these are **starting points, not gospel**:
- Parameters were optimized for specific strategy combinations, timeframes, and assets
- Different factor combinations → different optimal parameters
- What matters is the **methodology** for finding parameters, not the numbers themselves

Use reference values as:
- **Initial search range** (not point estimates)
- **Sanity checks** (if your optimal is wildly different, investigate why)
- **Structural insights** (e.g., "FR contrarian > FR carry" is durable; "z-score ±2.0" is context-dependent)

## Parameter Optimization Without Overfitting

This is the most valuable part of this skill. When tuning parameters:

### 1. Ablation Before Optimization
- Test the signal ON vs OFF first (is it adding value at ALL?)
- 3-way ablation: A(baseline), B(new signal only), C(baseline + new)
- If C < A, the signal is redundant regardless of parameters

### 2. Coarse-to-Fine Search
- Start with 3-5 values per parameter (wide range)
- Pick the **plateau region**, not the peak
- If performance is sensitive to ±10% parameter change → overfitted

### 3. Walk-Forward Validation (WFA)
- ≥5 expanding or rolling windows
- Parameter chosen in-sample, tested out-of-sample
- If IS SR >> OOS SR (>2x gap) → overfitted

### 4. Parameter Sensitivity Test
- Vary each parameter ±20% from chosen value
- If SR drops >30% → fragile, likely overfitted
- Good parameters sit on a **plateau**, not a spike

### 5. Cost Stress Test
- Test at 1.5x and 2.0x fee/slippage
- If strategy dies at 2x cost → edge is too thin

### 6. Known Anti-Patterns (This Codebase)
- **Gate stacking degrades**: OI+HTF, Macro+HTF all degraded -5% to -10% when stacked
- **IC ≠ alpha**: Volume Profile IC=0.024 but was confounded with vol regime
- **Carry is crowded**: FR carry SR fell from 6.45 (2020-24) to negative (2025)
- **Stablecoin losing edge for BTC**: institutional capital now enters via ETF, not USDT
- **Replacement > Stacking**: if new signal standalone > old, replace entirely, don't layer

### 7. Red Flags That Scream Overfit
- SR only positive in 1 out of 4 years
- Optimal param is at the edge of search range (not middle)
- Adding the 7th factor "helps" (beyond 6 → diminishing returns per Unravel Finance)
- Win rate > 60% on hourly crypto (suspicious — crypto is noisy)
