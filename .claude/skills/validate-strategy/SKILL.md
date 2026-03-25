---
description: "Run the full validation pipeline on a strategy. Use after implementation is complete, before risk review. Sequential — do not skip steps."
---

# /validate-strategy — Validation Pipeline Workflow

You are the Quant Researcher role. You independently validate strategies — do NOT trust the developer's numbers. Rerun everything yourself.

## Step 0: Determine What's Being Validated

Ask the user: "Which config are you validating?" Then classify:

- **Loop A (Alpha Existence)**: raw signal mechanism — focus on IC, causality, symbol breadth
- **Loop B (Trade Expression)**: entry/exit/sizing on accepted alpha — focus on cost, delay fragility

**Hard rule**: If the research packet can't distinguish whether the signal improved vs only the execution layer changed → reject the packet back to research. This is a hygiene failure.

## Step 1: Causality Check

```bash
PYTHONPATH=src python scripts/validate.py -c config/<cfg>.yaml --quick
```

Verify:
- [ ] Signal at `close[i]` → execute at `open[i+1]`
- [ ] `signal_delay=1` explicitly set in backtest context
- [ ] `price=df['open']` in vbt.Portfolio.from_orders()
- [ ] HTF resample uses `causal_resample_align()` (no raw `reindex(ffill)` without `.shift(1)`)
- [ ] Truncation invariance: run on 2 different date ranges, results consistent

## Step 2: Full Backtest with Costs

```bash
PYTHONPATH=src python scripts/run_backtest.py -c config/<cfg>.yaml
```

Checklist:
- [ ] `funding_rate.enabled: true`
- [ ] `slippage_model.enabled: true`
- [ ] Report includes: Return, CAGR, Sharpe, MaxDD, Calmar
- [ ] Yearly decomposition — flag if single year carries >60% of total return
- [ ] Overlay status explicitly stated: `ON (mode=...)` or `OFF`

## Step 3: Walk-Forward Analysis

```bash
PYTHONPATH=src python scripts/run_walk_forward.py -c config/<cfg>.yaml --splits 6
```

- [ ] OOS Sharpe > 0.3 (for momentum strategies)
- [ ] IS vs OOS gap: report ratio
- [ ] No split with catastrophic OOS loss

## Step 4: CPCV Cross-Validation

```bash
PYTHONPATH=src python scripts/run_cpcv.py -c config/<cfg>.yaml --splits 6 --test-splits 2
```

- [ ] PBO (Probability of Backtest Overfitting) < 0.5
- [ ] OOS distribution: median > 0

## Step 5: DSR (Deflated Sharpe Ratio)

```bash
PYTHONPATH=src python scripts/validate.py -c config/<cfg>.yaml --full
```

- [ ] DSR significant after multiple comparison correction
- [ ] `trial_registry` in `config/validation.yaml` is up-to-date (cumulative_n_trials)

## Step 6: Cost Stress Test

- [ ] 1.5× cost multiplier: still profitable?
- [ ] 2.0× cost multiplier: still profitable?

## Step 7: Delay Stress Test

- [ ] +1 bar extra delay: measure performance drop
- [ ] Drop > 50% → strategy is timing-fragile, flag as HIGH RISK

## Step 8: Overlay Ablation (if prod uses overlay)

- [ ] naked vs naked comparison (signal quality)
- [ ] overlay vs overlay comparison (deployment config)
- [ ] Delta Sharpe and Delta MDD per-symbol

## Step 9: Factor Orthogonality Audit

```bash
PYTHONPATH=src python scripts/analyze_factor_geometry.py -c config/<cfg>.yaml
```

- [ ] No production signal pair has |corr| > 0.50
- [ ] PCA effective factors >= n_signals × 0.60
- [ ] New candidate R² < 0.50 against existing signals

## Step 10: Regime-Stratified CPCV

```python
from qtrade.validation.prado_methods import regime_stratified_cpcv
result = regime_stratified_cpcv(symbol, data_path, cfg, strategy_name, btc_prices)
```

- [ ] `regime_concentration_pct < 0.70` (no single regime >70% of OOS PnL)
- [ ] If regime-dependent → flag, may be a regime proxy not robust alpha

## Step 11: meta_blend Extra Checks (if applicable)

- [ ] Ablation: Pure A, Pure B, A+B — blend Sharpe >= max(A,B) or MDD improved >20%
- [ ] `auto_delay=False` (prevents double-delay; BTC breakout_vol_atr has built-in delay)
- [ ] Signal direction conflict < 40%
- [ ] Weak symbol detection: WFA OOS+ < 50% → recommend remove

## Verdict

Only 4 verdicts allowed:

| Verdict | Criteria | Next step |
|---|---|---|
| **GO_NEXT** | All steps pass, robust across regimes | → Risk review via `/risk-check` |
| **KEEP_BASELINE** | New strategy ≈ baseline, no improvement | → Close PR, keep current prod |
| **NEED_MORE_WORK** | Promising but specific gaps | → Back to research with gap list |
| **FAIL** | Fundamental problems | → Close PR, add to failure registry |

Create a validation report with all checklist items and post as PR comment.
