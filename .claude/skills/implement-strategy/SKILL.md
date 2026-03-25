---
description: "Strategy implementation checklist — from research handoff to working backtest. Covers code patterns, config setup, causal alignment, and common Binance gotchas."
---

# /implement-strategy — Implementation Checklist

You are the Quant Developer. Implement the strategy from the research handoff while following all safety rules.

## Step 1: Receive Handoff

Read the PR or task manifest. Confirm you have:
- [ ] Strategy proposal with hypothesis and mechanism
- [ ] G0-G6 gate results from research
- [ ] Integration mode: Filter / Overlay / Standalone / Portfolio Layer
- [ ] Data dependencies list
- [ ] Preliminary IC / Sharpe estimates

## Step 2: Strategy Implementation

### File setup
```bash
# Branch (if not already on one)
git checkout -b research/<strategy-name>-$(date +%Y%m%d)

# Strategy file
src/qtrade/strategy/<strategy_name>_strategy.py

# Config
config/research_<name>.yaml
```

### Mandatory pattern
```python
@register_strategy("my_strategy")
def generate_positions(df: pd.DataFrame, ctx: StrategyContext, params: dict) -> pd.Series:
    # 1. Compute indicators (import from qtrade.indicators, don't reimplement)
    # 2. Generate raw signal in [-1, 1]
    # 3. Return — framework handles signal_delay + direction clip
    return pos
```

### Rules
- [ ] Use `@register_strategy` decorator
- [ ] Signature: `(df, ctx: StrategyContext, params: dict) -> pd.Series`
- [ ] Return raw positions `[-1, 1]` — do NOT clip in strategy
- [ ] Do NOT call `.shift()` or `.clip()` — framework handles via `signal_delay`
- [ ] Use `ctx.is_futures`, `ctx.can_long`, `ctx.can_short` for conditional logic
- [ ] Logging: `from qtrade.utils.log import get_logger`, never `print()` in src/

### HTF resample — CRITICAL
```python
# CORRECT: use causal_resample_align
from qtrade.strategy.filters import causal_resample_align
trend_1h = causal_resample_align(df, "4h", compute_fn, df.index)

# WRONG: direct resample + reindex = LOOK-AHEAD
htf = compute(df_4h)
trend = htf.reindex(df.index, method="ffill")  # 3-hour look-ahead!
```

## Step 3: Config Setup

```yaml
# config/research_<name>.yaml
name: "<strategy_name>"
market_type: "futures"
direction: "both"
signal_delay: 1  # backtest: trade on next open
symbols: [BTC, ETH, SOL, DOGE, AVAX, LINK]

strategy:
  name: "<strategy_name>"
  params:
    # ... strategy-specific params

funding_rate:
  enabled: true
slippage_model:
  enabled: true
```

Rules:
- [ ] NEVER modify `config/prod_*` during research
- [ ] Use `cfg.to_backtest_dict()` — never manually assemble dicts
- [ ] Use `copy.deepcopy()` for any nested config dicts
- [ ] Never `.pop()` from caller's dict — use `.get()` and build new dicts

## Step 4: meta_blend Integration (if multi-strategy)

If adding to the meta_blend mixer:
- [ ] `auto_delay: false` in config — CRITICAL (prevents double-delay)
- [ ] Per-symbol routing configured
- [ ] Direction conflict check: sub-strategies opposing < 40% of time

**Known incident**: BTC Sharpe was abnormally low due to `auto_delay` double-delay with `breakout_vol_atr` built-in delay.

## Step 5: Binance API Gotchas (if touching live code)

- [ ] Hedge Mode: `positionSide` must match (`LONG`/`SHORT`), not just `side`
- [ ] SL/TP: use `STOP_MARKET`/`TAKE_PROFIT_MARKET` with `stopPrice`, not limit orders
- [ ] Order response: set `newOrderRespType=RESULT` for immediate fill info
- [ ] Backtest → live gap: expect 3-5× performance discount in live

## Step 6: Quick Validation

```bash
# Run safety guards
python -m pytest tests/test_code_safety_guard.py tests/test_resample_shift_guard.py -x -q

# Quick validate (causality + basic sanity)
PYTHONPATH=src python scripts/validate.py -c config/research_<name>.yaml --quick

# Backtest
PYTHONPATH=src python scripts/run_backtest.py -c config/research_<name>.yaml
```

Do NOT run full validation (WFA, CPCV, DSR) — that's the Quant Researcher's job via `/validate-strategy`.

## Step 7: Handoff to Validation

Push branch, update PR with:
- Implementation summary
- Backtest results (with costs enabled)
- Overlay status: ON or OFF
- Any data pipeline changes needed

Then tell the user: "Implementation complete. Run `/validate-strategy` for full validation."
