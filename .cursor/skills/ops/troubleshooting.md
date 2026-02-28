---
description: Troubleshooting SOPs and Next Steps output format for DevOps
globs:
alwaysApply: false
---
# Skill: Troubleshooting SOPs & Output Format

> Loaded by DevOps for incident response and post-task output.

## Troubleshooting SOP

1. **Runner not moving**: `tmux attach -t meta_blend_live` check log — usually API rate limit or network
2. **Position mismatch**: `query_db.py summary` compare with actual Binance positions
3. **SL/TP not placed**: Check `algo_orders_cache` — possible price calculation error or API change (algo order 404)
4. **Circuit breaker triggered**: Check `max_drawdown_pct` setting (currently 40%), confirm real loss vs API data delay
5. **OOM (Out of Memory)**: Confirm Swap is configured (`free -h`), mandatory for 1GB RAM. See resource limits skill.
6. **Algo Order 404**: Binance API may have changed; latest fix uses STOP_MARKET first, fallback STOP+price

## Next Steps Output Format

### After Deployment

```markdown
---
## Next Steps (pick one)

| Option | Agent | Prompt | When to pick |
|--------|-------|--------|-------------|
| A | `@devops` | "Deployment complete, run health check to confirm runner" | Standard flow |
| B | `@risk-manager` | "New strategy <name> deployed. Schedule /risk-review next week" | New strategy launch |
```

### After Troubleshooting

```markdown
---
## Next Steps (pick one)

| Option | Agent | Prompt | When to pick |
|--------|-------|--------|-------------|
| A | (none) | Issue resolved, no follow-up needed | Simple fix |
| B | `@quant-developer` | "Found <issue>, code fix needed at [location]" | Root cause in code |
| C | `@risk-manager` | "<event> occurred, recommend ad-hoc risk check" | Event may affect positions |
```

### Rules
- After deployment → **always** suggest health check (Option A is default)
- New strategy first launch → suggest paper trading observation or first risk check
- If troubleshooting involves fund safety → must suggest Risk Manager intervention
