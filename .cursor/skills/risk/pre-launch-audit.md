---
description: Pre-launch risk audit steps (MC, Kelly, Portfolio Risk, Risk Limits, Launch Guard, meta_blend checks)
globs:
alwaysApply: false
---
# Skill: Pre-Launch Risk Audit

> Loaded by Risk Manager when performing pre-launch audit after Quant Researcher's GO_NEXT verdict.

## Step 1: Monte Carlo Stress Test

```bash
source .venv/bin/activate
PYTHONPATH=src python scripts/run_r3c_monte_carlo.py --n-sims 1000
```

Four stress scenarios:

| Scenario | Description | Pass Criteria |
|----------|-------------|---------------|
| MC1: Return Bootstrap | Block resampling (preserves autocorrelation) | 5th percentile CAGR > 0 |
| MC2: Trade-Order Shuffle | Randomize trade sequence | 95th percentile MDD < 2× baseline MDD |
| MC3: Cost Perturbation | Fees/slippage/funding ±20% random perturbation | Median Sharpe > 0.3 |
| MC4: Execution Jitter | 0-1 bar random delay | Sharpe decay < 30% |

## Step 2: Kelly Fraction Validation

```bash
PYTHONPATH=src python -c "
from qtrade.risk.position_sizing import KellyPositionSizer
sizer = KellyPositionSizer(
    win_rate=<backtest_win_rate>,
    avg_win=<backtest_avg_win>,
    avg_loss=<backtest_avg_loss>,
    kelly_fraction=0.25  # Conservative: 1/4 Kelly
)
print(f'Full Kelly: {sizer.kelly_pct / sizer.kelly_fraction:.2%}')
print(f'Quarter Kelly: {sizer.kelly_pct:.2%}')
"
```

Checks:
- Position sizing <= Quarter Kelly?
- If Volatility Targeting, target vol reasonable (15-25% annualized)?
- Max single-symbol position <= config `max_single_position_pct`?

## Step 3: Portfolio Risk Assessment

```python
from qtrade.risk.portfolio_risk import PortfolioRiskManager

rm = PortfolioRiskManager(
    max_portfolio_var=0.05,    # Daily VaR 5%
    max_correlation=0.8,       # Max allowed correlation
    diversification_threshold=0.3,
)
passed, metrics = rm.check_risk_limits(returns_dict, weights)
```

Checks:
- Portfolio VaR (95%) <= 5%?
- Max pairwise correlation <= 0.8? (high corr = false diversification)
- Diversification ratio >= 0.3?

## Step 4: Risk Limits Check

```python
from qtrade.risk.risk_limits import RiskLimits

limits = RiskLimits(
    max_position_pct=1.0,
    max_drawdown_pct=0.65,  # Current production circuit breaker
    max_leverage=5.0,
    max_single_position_pct=0.5,
)
```

Cross-reference with config:
- `max_drawdown_pct` matches `AppConfig.risk.circuit_breaker_pct`?
- `max_leverage` matches Binance account setting?
- New symbols → need to reduce single-symbol max position?

## Step 5: Production Launch Guard

```bash
PYTHONPATH=src python scripts/prod_launch_guard.py --dry-run
```

Checks: config hash integrity, env vars (API Key, Telegram Token), data freshness, Risk Guard status, NTP sync.

## Step 5b: meta_blend Extra Checks

When auditing a `meta_blend` strategy:

| Check | Description | Pass Criteria |
|-------|-------------|---------------|
| **auto_delay consistency** | meta_blend must have `auto_delay=False` to avoid double-delay | Confirm `@register_strategy("meta_blend", auto_delay=False)` |
| **Sub-strategy signal conflict** | Some symbols may have opposing sub-strategy directions | Net signal exposure > 20% (not near zero long-term) |
| **Concentration Risk** | HHI and BTC+ETH combined weight | HHI < 0.2, top-2 combined < 40% |
| **Sub-strategy data deps** | `tsmom_carry_v2` needs FR + OI | Confirm Oracle Cloud has periodic download |
| **Ablation validation** | Pure A, Pure B, A+B comparison | Blend Sharpe >= max(Pure A, Pure B) or MDD significantly improved |

> **Double-delay is the most common fatal issue**: BTC Sharpe once dropped from 1.18 to 0.50 due to this. Always verify `auto_delay` setting.
