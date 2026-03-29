# Open Interest (OI) Signal

## 4-Quadrant Framework (Gate.com: 60-70% accuracy)

| Price | OI | Signal | Score |
|---|---|---|---|
| ↑ Rising | ↑ Rising | Strong uptrend confirmed (new longs) | +1.0 |
| ↑ Rising | ↓ Falling | Short squeeze (not new demand) | +0.3 |
| ↓ Falling | ↑ Rising | Strong downtrend (new shorts) | -1.0 |
| ↓ Falling | ↓ Falling | Liquidation flush → contrarian bullish | +0.5 |

## OI Z-Score (extreme leverage detection)
- Lookback: **50 bars** (Glassnode standard)
- Z > 2.0 = extreme leverage → mean reversion setup
- Combine with FR > 0.10%/8h for highest probability

## Liquidation Cascade Pattern
1. OI surges to high (both sides entering)
2. Price moves adversely → one side starts liquidating
3. OI collapses 20-30% from peak → cascade complete
4. **Contrarian entry**: after cascade completion (OI stabilized at lower level)
5. Expected hold: 24-72 hours for rebound

## Key Statistics
- Oct 2025: $19B OI evaporated in 36h
- Feb 2026: $2.5B BTC liquidations
- OI vs volatility R² = 0.36 (significant but not standalone)

## Known Pitfalls
- OI alone is noisy — must combine with FR and price direction
- 4h timeframe is best balance of signal quality vs noise
- OI standalone was FAIL in this codebase (incremental +4.66% < 5% threshold when stacked with HTF)
- But OI standalone SR=4.12 > HTF SR=3.86 — replacement potential exists

## Sources
- Gate.com derivatives signals guide (2026)
- Glassnode LPOC methodology
- Amberdata liquidation cascade guide
