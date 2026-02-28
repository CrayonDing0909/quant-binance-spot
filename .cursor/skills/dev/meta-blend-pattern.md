---
description: meta_blend multi-strategy mixer pattern, per-symbol routing, auto_delay gotcha
globs:
alwaysApply: false
---
# Skill: meta_blend Pattern

> Loaded by Quant Developer when working with multi-strategy blends.

## When to Use meta_blend

- Already have production strategy, want to add new strategy to same account (ONE_WAY conflict avoidance)
- Some symbols perform better in Strategy A, others in Strategy B → per-symbol routing
- Two strategies with low correlation → blend reduces MDD

## Development Steps

1. **Phase 1 — Weight Optimization**: Sweep weights with `scripts/research_strategy_blend.py`
   ```bash
   PYTHONPATH=src python scripts/research_strategy_blend.py
   ```
2. **Phase 2 — Configure meta_blend**: Define `sub_strategies` and per-symbol overrides in YAML
3. **Phase 3 — Validate**: Backtest + WFA + cost stress + ablation (Pure A / Pure B / A+B)
4. **Phase 4 — Production Candidate**: Create `config/prod_candidate_meta_blend.yaml`

## ⚠️ Critical: auto_delay=False

- `meta_blend` MUST register with `@register_strategy("meta_blend", auto_delay=False)`
- Reason: sub-strategies called via `get_strategy()` handle their own delay and direction clip
- If meta_blend also applies `auto_delay=True`, sub-strategies with `auto_delay=False` (like `breakout_vol_atr`, built-in delay) get **double-delayed** — signal shifts one extra bar
- **Real incident**: BTC Sharpe dropped from 1.18 to 0.50 due to double-delay

## Per-Symbol Routing Example (YAML)

```yaml
strategy:
  name: "meta_blend"
  params:
    sub_strategies:                    # Default sub-strategy combo (most symbols)
      - name: "tsmom_carry_v2"
        weight: 1.0
        params: {tier: "default", ...}
  symbol_overrides:
    BTCUSDT:                          # BTC uses different combo
      sub_strategies:
        - name: "breakout_vol_atr"
          weight: 0.30
          params: {...}
        - name: "tsmom_carry_v2"
          weight: 0.70
          params: {tier: "btc_enhanced", ...}
```

## Reference Files

- Strategy implementation: `src/qtrade/strategy/meta_blend_strategy.py`
- Research config: `config/research_meta_blend.yaml`
- Production candidate: `config/prod_candidate_meta_blend.yaml`
- Weight optimization script: `scripts/research_strategy_blend.py`
