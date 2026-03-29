# Stablecoin Flow Signal

## Signal Construction
- Net USDT/USDC inflow to exchange wallets (24h rolling)
- 30d momentum of stablecoin market cap (DeFi Llama)
- Z-score over 90-day window

## Interpretation
| Flow | Signal |
|---|---|
| Rising exchange inflows | Dry powder building → bullish |
| Falling / outflows | Demand buffer shrinking → bearish |
| New highs during downtrend | "非常特殊又明显的指标" — very special signal (华山论剑) |
| Large single-day inflow ($2B+) | Often coincides with breakout |

## Academic Evidence
- BDC Consulting: USDT supply model → **229% ROI** vs buy-and-hold
- arXiv:2603.23480 (2026.03): GARCH-Copula-XGBoost, stablecoin vol improves BTC vol forecast
- Correlation of stablecoin supply growth to BTC price: **95.24%**
- Lead time: days to weeks

## CRITICAL WARNING: Signal Degradation for BTC
- CryptoQuant CEO (2025): "Most new BTC liquidity now enters via MSTR and ETFs, not stablecoin routes"
- Bot contamination: 77-80% of on-chain stablecoin transactions are bot-driven (2024)
- **Recommendation**: Weight 0.3 (not 1.0) for BTC strategies; may remain stronger for altcoins

## Optimal Parameters
| Parameter | Value |
|---|---|
| Momentum window | 30 days |
| Z-score lookback | 90 days (×24 for 1h bars) |
| Causality | shift(1) day before ffill to 1h |
| Weight in composite | 0.3 (degraded for BTC) |

## Sources
- BDC Consulting — USDT supply study
- arXiv:2603.23480 — Stablecoin volatility forecasting
- CryptoQuant — Stablecoin exchange flows
- 华山论剑 (@huashanlunjians)
