# Coinbase Premium Index (CPI)

## Signal Construction
- `premium = (coinbase_price - binance_price) / binance_price`
- Smoothing: SMA-50 or EMA (CryptoQuant/TradingView consensus)
- Z-score: 60-day rolling window for stability

## Optimal Parameters
| Parameter | Value | Source |
|---|---|---|
| Z-score lookback | 60 days | CryptoQuant, practitioner consensus |
| Z-score entry | ±2.0 | Quant standard |
| Z-score stop | ±3.5 | Practitioner |
| Z-score exit | ±0.5 | Mean reversion target |
| Preferred timeframe | 4h candles | TradingView BIGTAKER |
| Session filter | US hours 09:30-16:00 EST | Optional, improves SNR |

## Key Thresholds
- Persistent positive (3+ days after negative streak) → reversal signal
- -167.8 bps → institutional demand materially reversed (CryptoQuant 2026)
- 36-40 day negative streaks historically preceded sharp recoveries

## Usage
- **Best as regime indicator**, not standalone trigger
- Negative CPI for prolonged period = absence of institutional sponsorship → red flag for longs
- Combine with OI + FR for higher confidence

## Sources
- CryptoQuant BTC CPI chart
- CoinGlass CPI
- TradingView BIGTAKER analysis
- CoinTelegraph institutional signal article (2026)
