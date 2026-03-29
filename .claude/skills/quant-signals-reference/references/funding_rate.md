# Funding Rate Signal

## Two Modes

### A. Carry (harvesting positive FR)
- Long spot + short perp when annualized FR > 5%
- **WARNING: Crowded and decaying** — SR fell from 6.45 (2020-24) to negative (2025)
- Source: He et al. (arXiv:2212.06888), Schmeling et al. 2023

### B. Contrarian (fading extremes) — RECOMMENDED
- Z-score of 8h funding rate over rolling window
- Entry when z < -2.0 (extreme negative = overcrowded shorts → go long)
- Entry when z > +2.0 (extreme positive = overcrowded longs → go short)
- Exit at z reversion to ±0.5

## Key Thresholds
| Level | Meaning |
|---|---|
| +0.01% to +0.05%/8h | Bullish bias, possible long crowding |
| -0.01% to -0.03%/8h | Bearish pressure, short crowding |
| > +0.10%/8h (~100% ann.) | **Extreme** long leverage, high liquidation risk |
| < -0.10%/8h | Extreme short squeeze risk |

## Optimal Parameters
| Parameter | Value |
|---|---|
| Z-score lookback | 168 bars (7 days at 1h) |
| Entry threshold | z ±2.0 |
| Exit threshold | z ±0.5 |

## Known Pitfalls
- FR carry is **crowded** post-2024 (ETF participants, basis traders)
- Contrarian signal more durable than carry
- FR alone is insufficient — combine with OI for higher probability
- ScienceDirect 2025: cross-exchange arb 115.9% / 6mo, MDD 1.92%, HODL corr=0

## Sources
- He et al. (arXiv:2212.06888) — Fundamentals of Perpetual Futures
- BIS Working Paper 1087
- ScienceDirect 2025 — Funding rate arbitrage
