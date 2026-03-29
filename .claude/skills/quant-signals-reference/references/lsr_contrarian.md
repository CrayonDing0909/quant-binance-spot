# Long/Short Ratio (LSR) — Contrarian Signal

## Signal Construction
- Account L/S Ratio = net long accounts / total accounts with open positions
- Top Trader Account LSR > retail LSR (filters noise)
- **Multi-σ extremes only** — minor shifts are noise

## Key Thresholds
| Level | Signal |
|---|---|
| >70% long | Extreme bullish crowding → contrarian short |
| <40% long (>60% short) | Extreme bearish crowding → contrarian long |
| ~50% | Neutral / consolidation |
| LSR 6.03 | Historical extreme bullish → preceded 20% correction |
| LSR 0.44 | Historical extreme bearish → preceded 20% rebound |

## Codebase-Validated Parameters (LSR Contrarian v3)
| Parameter | Value | Evidence |
|---|---|---|
| EW pctrank halflife | 84 bars | Better 2025-2026 IC than rolling |
| Entry percentile | 0.85 (extreme) | Research cycles 1-3 |
| ADX gate | >25 | Filters choppy markets |
| Persist | ≥2 bars | Reduces single-bar noise |
| Cooldown | 24h | Prevents re-entry losses |
| TP mode | Midpoint (pr=0.50) | MDD -30.68% → -19.56% |
| SL | ATR 3.0x | Optimal across 2.0-4.0 sweep |
| Time stop | 168h | Prevents zombie holdings |

## Known Pitfalls
- Institutional hedging distorts LSR for BTC (CryptoQuant CEO Ki Young Ju)
- MVRV (realized value) achieves 82% accuracy vs LSR for BTC
- LSR more reliable for **altcoins** with lower institutional participation
- Top LSR (大戶) IC ≈ 0 — no information value
- retail_vs_top divergence: IC strongest but 2026 IC flipped + 7.8x turnover

## Sources
- AInvest contrarian signal analysis (2025, 2026)
- CoinGlass L/S ratio data
- Gate.com L/S ratio explainer
- Codebase: LSR Contrarian v3 research cycles
