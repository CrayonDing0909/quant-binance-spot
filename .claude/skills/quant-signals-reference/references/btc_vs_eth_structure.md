# BTC vs ETH — Signal Portability Warning

## Rule: Never blindly port signals across BTC/ETH

Same signal, same parameters → structurally different results. This is NOT random variance.

## Quick Reference: Which Signals Port, Which Don't

| Signal | BTC reliability | ETH reliability | Why different |
|---|---|---|---|
| FR contrarian | High (clean extremes, fast reversion) | **Low** (staking hedge noise, 87.5% at floor) | ETH perps structurally shorted by staking hedgers |
| OI 4-quadrant | Moderate-High | Moderate | ETH OI fragmented across L2 perps |
| LSR contrarian | Moderate | Moderate-Low | More retail noise on ETH |
| CB Premium | BTC-only (Coinbase BTC volume >> ETH) | Weak | Institutional flow signal = primarily BTC |
| Stablecoin | Degraded (ETF bypass) | **More degraded** | DeFi mechanical flows confound |
| TSMOM | Works both | Works both | Price momentum is asset-agnostic |

## When Designing Cross-Asset Strategies

1. **Always backtest per-symbol** — never assume BTC results extend to ETH
2. **Year-by-year decomposition** — check if ETH alpha is concentrated in one period
3. **Check for structural breaks** — ETH had Merge (2022.09); BTC had ETF (2024.01)
4. **ETH signals need longer lookback** — ETH processes information slower (Guo et al.)
5. **ETH whipsaws are 5x more expensive** — <12h trades: BTC -0.40% vs ETH -1.96%

## Recommended Per-Symbol Adjustments

| Parameter | BTC | ETH |
|---|---|---|
| Rebalance frequency | 4h (faster signal) | 24h (filter noise) |
| FR weight | 1.0 | 0.3 (staking hedge pollutes) |
| CB Premium weight | 1.0 | 0.3 (weak signal on ETH) |
| Stablecoin weight | 0.3 | 0.1 (DeFi noise) |
| Min hold time | None needed | Consider 12h+ filter |
| Regime filter | Optional | **Mandatory** (50MA + momentum) |

## Key Academic Sources
- Guo et al. 2024 — Cross-crypto return predictability (asymmetric)
- BitMEX Q3 2025 — FR structure by exchange/asset
- Glassnode/Keyrock — On-chain velocity BTC 0.61% vs ETH 1.3%
- BIS WP 1087 — Arbitrage efficiency gap
- Full details: `docs/research/20260328_btc_eth_signal_divergence.md`
