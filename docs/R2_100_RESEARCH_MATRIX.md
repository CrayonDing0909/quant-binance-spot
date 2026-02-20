# R2-100 Research Matrix

Goal: push annualized return materially higher (toward 100%) with explicit acceptance of higher drawdown.

Important: keep `config/prod_candidate_R1.yaml` as production baseline. All R2 tests run in research branch/configs only.

## Why R1 plateaued

- Increasing leverage/weight alone (Agg A/B/C) raised drawdown faster than alpha.
- 2025-like chop regime is the main failure mode.
- Current sleeves are mostly trend beta; need new orthogonal alpha and better regime routing.

## Hard Rules (must keep)

- No look-ahead (`trade_on=next_open`, no future leakage)
- Costs always on (fee/slippage/funding), plus `cost_mult=1.5` stress
- Report all negative runs (no cherry-picking)

## Experiment Matrix (6 runs)

### E1 — Universe Expansion (Top-N liquid futures)
- Hypothesis: more symbols improve opportunity set and reduce single-sleeve concentration.
- Change:
  - Expand symbols from 3 to 15-30.
  - Keep current BTC breakout + TSMOM logic per symbol family.
  - Add liquidity filter (ADV/volume threshold).
- Pass hint:
  - CAGR uplift >= +50% vs R1, while `cost_mult=1.5` Sharpe > 0.5.

### E2 — Regime Router (Trend/Chop switch)
- Hypothesis: avoid 2025 chop bleed by reducing exposure or disabling breakout in chop.
- Change:
  - Add market regime classifier (e.g., ADX + realized vol compression).
  - Trend regime: breakout + tsmom fully enabled.
  - Chop regime: reduce gross exposure 40-70% and disable weak sleeve.
- Pass hint:
  - 2025 yearly return improves by >= +10pp vs current aggressive profile.

### E3 — BTC Breakout 2.0 (quality filter)
- Hypothesis: avoid fake breakouts with structure confirmation.
- Change:
  - Keep ATR channel core.
  - Add one confirmation: volume expansion or higher-timeframe trend agreement.
  - Optional re-entry cooldown after false-break event.
- Pass hint:
  - BTC sleeve Sharpe > 1.0 with turnover not exploding.

### E4 — Add Orthogonal Sleeve (non-trend)
- Hypothesis: trend-only stack is too correlated; need uncorrelated alpha.
- Change:
  - Add one sleeve from: basis/funding carry, relative-value spread, or intraday mean-reversion micro sleeve.
  - Cap sleeve risk budget (10-20%).
- Pass hint:
  - Portfolio Sharpe +0.15 uplift with same or lower Calmar decay in stress run.

### E5 — Dynamic Risk Budget (not fixed 34/33/33)
- Hypothesis: fixed weights underuse strong sleeves and overfund weak regimes.
- Change:
  - Rolling sleeve scoring by OOS-like stats window (Sharpe, drawdown, stability).
  - Map score -> weight bounds.
- Pass hint:
  - CAGR uplift >= +20% and MDD increase <= +10pp vs comparable aggressive config.

### E6 — Execution Robustness for high-turnover configs
- Hypothesis: aggressive configs die from friction.
- Change:
  - Compare maker-first timeout profiles (5s/10s/20s).
  - Simulate adverse slippage and funding spikes.
- Pass hint:
  - `cost_mult=1.5` to `2.0` degradation remains bounded; strategy stays profitable.

## Acceptance Gates (R2 candidate)

Candidate can proceed only if all pass:

1. Holdout CAGR >= 35% (stretch: >= 50%)
2. `cost_mult=1.5` Sharpe >= 0.7
3. Worst-year return >= -20%
4. At least 2 independent sleeves contribute positive return on holdout
5. No anti-lookahead or execution consistency regressions

## Run Order (recommended)

1. E1 + E2 first (largest expected impact)
2. E3 and E4
3. E5
4. E6 final hardening

## Standard Output Required per Experiment

- `portfolio_stats.json` (1.0x and 1.5x)
- year-by-year table (2022, 2023, 2024, 2025, 2026 YTD)
- sleeve contribution table
- walk-forward summary
- one-paragraph honest risk statement

## Claude Prompt Stub

Use this if needed:

"Run the R2-100 matrix experiments E1-E6 in `docs/R2_100_RESEARCH_MATRIX.md` without changing production config. Keep anti-lookahead safeguards, run strict cost backtests at 1.0x and 1.5x, and produce evidence paths for every result."
