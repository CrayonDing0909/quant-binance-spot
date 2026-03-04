# R3C Symbol Governance Spec

Status: Draft v1.0  
Owner: Quant Engineering  
Scope: `config/prod_candidate_R3C_universe.yaml` (19-symbol futures universe)

## 1) Goal

Add a production-safe **Negative Edge Filter** for symbol-level governance:
- Detect symbols with persistent negative expected value (negative edge / negative EV)
- De-weight first, then quarantine only when degradation persists
- Keep strategy logic unchanged (signal generation and execution path must remain intact)

This spec is designed so Claude Code can implement directly.

## 2) Naming (canonical terms)

- **Negative Edge Filter (NEF)**: core mechanism for identifying low/negative edge symbols
- **De-allocation**: soft weight reduction (not hard removal)
- **Kill List / Quarantine**: temporary hard disable for a symbol
- **Dynamic Universe**: weekly/monthly re-evaluated tradable symbol set

Preferred public name in code/docs:
`Negative Edge Filter with Kill-List/Quarantine`

## 3) Design Principles

1. **Safety first**: no immediate hard delete on first bad week
2. **Execution-aware**: include slippage/latency consistency, not just backtest-style PnL
3. **Reversible**: quarantine is temporary; symbols can re-enter
4. **Config-driven**: thresholds live in YAML, with safe defaults
5. **Non-invasive**: do not change strategy signal functions

## 4) Data Inputs (minimum required)

Per symbol, weekly (rolling 4 weeks by default):
- `net_pnl` (after fees/slippage/funding)
- `turnover`
- `returns_series` (for Sharpe)
- `max_drawdown_pct`
- `realized_slippage_bps`
- `model_slippage_bps`
- `signal_execution_consistency_pct`
- `missed_signals_pct`
- `trade_count`

## 5) Core Metrics

For each symbol `s`:
- `edge_sharpe_4w(s)` = Sharpe on 4-week net returns
- `edge_per_turnover_4w(s)` = `net_pnl / max(turnover, eps)`
- `slippage_ratio_4w(s)` = `realized_slippage_bps / max(model_slippage_bps, eps)`
- `consistency_4w(s)` = signal-exec consistency
- `missed_4w(s)` = missed signal %
- `dd_4w(s)` = rolling max drawdown %

## 6) State Machine

Each symbol has state:
- `active`
- `deweighted`
- `quarantined`

Transitions:

1) `active -> deweighted` if any trigger holds in current review:
- `edge_sharpe_4w < 0.3`
- OR `edge_per_turnover_4w < 0`
- OR `slippage_ratio_4w > 1.5`
- OR `consistency_4w < 99.0`

2) `deweighted -> quarantined` if persistent degradation:
- `edge_sharpe_4w < 0` for 2 consecutive reviews
- AND (`slippage_ratio_4w > 1.8` OR `missed_4w > 5.0`)
- OR `dd_4w > 25.0`

3) `deweighted -> active` recovery:
- `edge_sharpe_4w >= 0.8`
- AND `edge_per_turnover_4w > 0`
- AND `slippage_ratio_4w <= 1.2`
- for 2 consecutive reviews

4) `quarantined -> deweighted` recovery gate:
- quarantine minimum duration met (default 14 days)
- `edge_sharpe_4w >= 0.5`
- `slippage_ratio_4w <= 1.3`
- `consistency_4w >= 99.5`

## 7) Weight Policy

Base weights come from current universe config.

Multipliers by state:
- `active`: `1.00`
- `deweighted`: `0.50` (step-down can be 0.8 -> 0.65 -> 0.5 if desired)
- `quarantined`: `0.00`

Then re-normalize across non-quarantined symbols with constraints:
- `min_weight = 0.03`
- `max_weight = 0.20`
- optional `cash_reserve` preserved

## 8) Governance Schedule

- Review frequency: weekly (UTC Monday 00:00)
- Warmup requirement: at least 14 days live data before first action
- Apply changes in one batch after review
- Keep full audit logs

## 9) YAML Config Contract (to add)

Add under top-level `live`:

```yaml
live:
  symbol_governance:
    enabled: true
    review_frequency: "weekly"
    warmup_days: 14
    quarantine_min_days: 14

    thresholds:
      edge_sharpe_deweight: 0.3
      edge_sharpe_quarantine: 0.0
      edge_sharpe_recover_active: 0.8
      edge_sharpe_recover_from_quarantine: 0.5
      edge_per_turnover_min: 0.0
      slippage_ratio_deweight: 1.5
      slippage_ratio_quarantine: 1.8
      slippage_ratio_recover_active: 1.2
      slippage_ratio_recover_from_quarantine: 1.3
      consistency_min: 99.0
      consistency_recover: 99.5
      missed_signals_quarantine: 5.0
      dd_quarantine_pct: 25.0

    weights:
      active_multiplier: 1.0
      deweight_multiplier: 0.5
      quarantine_multiplier: 0.0
      min_weight: 0.03
      max_weight: 0.20
```

## 10) File/Module Plan

Recommended new files:
- `src/qtrade/live/symbol_governance.py`
- `src/qtrade/live/symbol_metrics_store.py`
- `scripts/run_symbol_governance_review.py`
- `reports/symbol_governance/latest_decisions.json`
- `reports/symbol_governance/history.jsonl`

Likely integrations:
- `src/qtrade/live/websocket_runner.py` (or base runner): expose realized execution stats
- `src/qtrade/config.py`: parse `live.symbol_governance`
- `src/qtrade/monitor/telegram_bot.py`: add `/universe_status` and `/symbol_health <symbol>`

## 11) Telegram UX

Add commands:
- `/universe_status`  
  Returns active/deweighted/quarantined lists + effective weights
- `/symbol_health BTCUSDT`  
  Returns latest 4-week metrics, current state, next action condition

Alerting:
- Notify on state transition only (avoid spam)
- Cooldown for repeated warnings

## 12) Acceptance Criteria

1. Governance never changes strategy signal logic
2. Weekly review produces deterministic decision artifact JSON
3. Same inputs produce same state transitions
4. Quarantine and recovery paths both tested
5. Telegram commands return current state and metrics
6. If governance disabled, behavior is identical to current production

## 13) Minimal Test Plan

Unit tests:
- transition rules (`active -> deweighted -> quarantined -> deweighted -> active`)
- weight normalization with min/max constraints
- recovery guardrails and quarantine minimum days

Integration tests:
- weekly review script generates valid artifact
- effective weights applied without breaking live execution
- telegram commands read governance artifacts correctly

## 14) Direct Prompt For Claude Code

Use this prompt as-is:

```text
Implement the spec in docs/R3C_SYMBOL_GOVERNANCE_SPEC.md.

Hard constraints:
1) Do not modify strategy signal logic.
2) Add config-driven symbol governance under live.symbol_governance.
3) Implement state machine (active/deweighted/quarantined) and weekly review pipeline.
4) Persist artifacts to reports/symbol_governance/latest_decisions.json and history.jsonl.
5) Add Telegram commands /universe_status and /symbol_health <symbol>.
6) Keep backward compatibility when governance is disabled.

Deliverables:
- Code changes
- Config parsing updates
- Example YAML snippet added to prod_candidate_R3C_universe.yaml (or sample config)
- Unit tests for transition and weight normalization
- A runnable review command: scripts/run_symbol_governance_review.py

Validation:
- Show a dry-run example with generated decision artifact
- Show that disabled governance preserves existing behavior
```

