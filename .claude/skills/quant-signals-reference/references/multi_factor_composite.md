# Multi-Factor Composite Strategy

## Academic Benchmarks

| Framework | Method | SR | Source |
|---|---|---|---|
| ACM ICGAIB 2025 | IC-weighted Z-scores, 4-6 factors | **2.5** | ACM proceedings |
| Unravel Finance | Cross-sectional 6-factor, inverse vol weighting | **~2.5** | Blog research |
| CF Benchmarks | Sentiment-gated basis (MACD z + 90d return z → CDF) | **1.52** | Research blog |

## Key Design Principles

### Factor Combination
- **4-6 uncorrelated signals** arithmetic average is robust baseline
- Beyond 6 factors: diminishing returns (Unravel Finance finding)
- IC-weighted > equal-weighted (ACM 2025)
- Inverse volatility weighting handles crypto vol dispersion

### Z-Score Windows
- **30-day** lookback for momentum z-scores
- **90-day** for medium-term trend z-scores
- Walk-forward validation essential (regime shifts in crypto are rapid)

### Entry/Exit
- Z ±1.0 entry/exit band (ACM 2025 multi-factor)
- Sentiment > 0.8 (extreme greed) → enter carry trade (CF Benchmarks)
- Sentiment < 0.2 (extreme fear) → exit

### Funding Rate as Factor
- Best used as **regime signal** (gating other signals), not primary directional trigger
- Open-interest-weighted composite across exchanges avoids exchange-specific distortion

## Anti-Patterns from This Codebase
- **Gate stacking kills alpha**: OI+HTF (-5%), Macro+HTF (-10%), On-chain+HTF over-filter
- **Pattern**: each gate reduces time-in-market → stacking → TIM collapses
- **Solution**: Use signals as composite score inputs, NOT cascading gates
- **Replacement > Stacking**: if new signal is better standalone, replace the old one entirely

## Practical Design for This Codebase
```
score = Σ (weight_i × z_score_i) / Σ weight_i

where:
  - z_score_i ∈ [-1, +1] (clipped)
  - weight_i = IC-informed (or equal if IC similar)
  - entry: score > threshold (0.25-0.35)
  - exit: score reversal OR mechanical SL/TP/time
```

## Sources
- ACM 2025 multi-factor ETH paper
- Unravel Finance cross-sectional alpha
- CF Benchmarks sentiment-basis research
- arXiv:2510.14435 crypto asset class review
