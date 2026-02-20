# Production Launch SOP — R1 Ensemble

**Version:** 1.0  
**Date:** 2026-02-20  
**Owner:** Quant Team  
**Config:** `config/prod_candidate_R1.yaml`  
**Config Hash (SHA256):** `b84a4c70f92e503be23739f1e5d8375e1bfdc2ef19b5f7a06b71bbcd8d1b453e`

---

## Strategy Overview

| Sleeve | Symbol | Strategy | Key Parameters |
|--------|--------|----------|---------------|
| BTC | BTCUSDT | `breakout_vol_atr` | channel=336, time_stop=72h, risk×0.7 |
| ETH | ETHUSDT | `tsmom_multi_ema` | lookbacks=[72,168,336,720] |
| SOL | SOLUSDT | `tsmom_ema` | lookback=168 |

**Weights:** BTC 34% / ETH 33% / SOL 33%  
**Market:** Futures, ISOLATED, 3x leverage  
**Anti-lookahead:** `trade_on=next_open`, costs enabled

---

## 1. Pre-Launch Checklist

### 1.1 Environment

- [ ] **OS & Runtime**: Python 3.11 on Oracle Cloud / deployment server
- [ ] **Virtual Environment**: `source .venv/bin/activate` confirmed
- [ ] **Dependencies**: `pip install -r requirements.txt` — no missing packages
- [ ] **NTP Sync**: `timedatectl` shows synced, or `chronyc tracking` offset < 100ms

### 1.2 API & Secrets

| Variable | Purpose | Check |
|----------|---------|-------|
| `BINANCE_API_KEY` | Exchange access | Set, non-empty |
| `BINANCE_API_SECRET` | Exchange access | Set, non-empty |
| `FUTURES_TELEGRAM_BOT_TOKEN` | Notifications | Set (optional but recommended) |
| `FUTURES_TELEGRAM_CHAT_ID` | Notifications | Set (optional but recommended) |

- [ ] Binance API permissions: **Futures trading** enabled, **Withdrawal** disabled
- [ ] IP whitelist configured on Binance (deployment server IP only)
- [ ] API key tested: `PYTHONPATH=src python scripts/run_live.py -c config/prod_candidate_R1.yaml --check`

### 1.3 Config Integrity

```bash
# Verify frozen config hash
shasum -a 256 config/prod_candidate_R1.yaml
# Expected: b84a4c70f92e503be23739f1e5d8375e1bfdc2ef19b5f7a06b71bbcd8d1b453e
```

- [ ] Hash matches
- [ ] `trade_on: next_open`
- [ ] `fee_bps: 5`, `slippage_bps: 3`
- [ ] `funding_rate.enabled: true`
- [ ] `ensemble.enabled: true` with correct weights

### 1.4 Data Freshness

```bash
PYTHONPATH=src python scripts/download_data.py -c config/prod_candidate_R1.yaml
```

- [ ] All 3 symbols (BTC/ETH/SOL) data downloaded
- [ ] Last bar < 2h old
- [ ] No gaps > 3h in recent 30 days

### 1.5 Launch Guard (Automated)

```bash
PYTHONPATH=src python scripts/prod_launch_guard.py --rules config/prod_scale_rules_R1.yaml
```

- [ ] Output: `ALLOW_LAUNCH`
- [ ] All HARD checks passed
- [ ] Review any SOFT warnings

### 1.6 Dry-Run Verification

```bash
# Paper mode — verify signals generate correctly
PYTHONPATH=src python scripts/run_live.py -c config/prod_candidate_R1.yaml --paper --once

# Real mode dry-run — verify broker connection without placing orders
PYTHONPATH=src python scripts/run_live.py -c config/prod_candidate_R1.yaml --real --dry-run --once
```

- [ ] Paper signals generated for all 3 symbols
- [ ] Dry-run shows position targets (not `NaN` or all-zero)
- [ ] SL/TP levels computed correctly
- [ ] Telegram notification received (if configured)

---

## 2. Launch Steps (Staged Rollout)

### Stage 0: Paper Trading (Pre-Requisite)

**Status:** Already completed during Layer 3 validation.

**Duration:** 30 days (minimum)  
**Capital:** $0 (simulated)

**Pass Criteria:**
- 30D Paper Sharpe > -0.3
- 30D Paper MDD < 12%
- Signal consistency > 98% (paper vs. backtest replay)

**Evidence:** Paper trading logs in `reports/futures/live/paper_state.json`

---

### Stage 1: $5,000 (Week 1–2)

**Entry Conditions:**
- [ ] Stage 0 passed
- [ ] `prod_launch_guard.py` → `ALLOW_LAUNCH`
- [ ] At least 1 team member available for monitoring during first 24h

**Launch Command:**
```bash
# tmux session for persistence
tmux new -s prod_r1

# Activate & start
cd /path/to/quant-binance-spot
source .venv/bin/activate

# WebSocket mode (recommended for production)
PYTHONPATH=src python scripts/run_websocket.py \
  -c config/prod_candidate_R1.yaml \
  --real \
  --telegram-commands
```

**Daily Monitoring:**
```bash
# Daily report
PYTHONPATH=src python scripts/prod_report.py --daily

# Risk guard check
PYTHONPATH=src python scripts/risk_guard.py --config config/prod_scale_rules_R1.yaml --dry-run
```

**Exit to Stage 2 (ALL required):**
- [ ] 14D rolling Sharpe > 0
- [ ] Running MDD < 10%
- [ ] Realized slippage ≤ assumed × 1.3
- [ ] No FLATTEN events

**Abort → Stage 0 (ANY triggers):**
- Running MDD > 12%
- 14D Sharpe < -0.5
- API connectivity issues > 3 times in 24h

---

### Stage 2: $10,000 (Week 3–4)

**Entry Conditions:**
- [ ] Stage 1 all exit criteria met
- [ ] `prod_launch_guard.py` → `ALLOW_LAUNCH`
- [ ] Weekly report reviewed

**Capital Adjustment:**
- Adjust USDT balance in Binance Futures wallet to $10,000
- Runner will auto-recalculate position sizes based on account balance

**Exit to Stage 3 (ALL required):**
- [ ] 14D rolling Sharpe > 0.3
- [ ] Running MDD < 10%
- [ ] 3/3 sleeves with positive 14D contribution
- [ ] Realized costs within 1.3× of assumptions

**Abort → Reduce to $5k (ANY triggers):**
- Running MDD > 10%
- 14D Sharpe < 0
- Any single sleeve 14D contribution < -5%

---

### Stage 3: $15,000+ (Week 5+)

**Entry Conditions:**
- [ ] Stage 2 all exit criteria met
- [ ] Monthly review completed

**Ongoing:**
- Daily monitoring via `prod_report.py --daily`
- Weekly review via `prod_report.py --weekly`
- Scale-up/down per `config/prod_scale_rules_R1.yaml` rules
- Monthly strategy review (compare live vs. backtest replay)

---

## 3. Daily Operations

### 3.1 Daily Check Commands

```bash
# 1. Verify bot is running
tmux ls  # Should show prod_r1 session

# 2. Generate daily report
PYTHONPATH=src python scripts/prod_report.py --daily \
  --config config/prod_candidate_R1.yaml \
  --rules config/prod_scale_rules_R1.yaml

# 3. Run risk guard
PYTHONPATH=src python scripts/risk_guard.py \
  --config config/prod_scale_rules_R1.yaml --dry-run

# 4. Check positions (quick)
PYTHONPATH=src python scripts/run_live.py \
  -c config/prod_candidate_R1.yaml --status
```

### 3.2 Metric Thresholds & Actions

| Metric | Green (No Action) | Yellow (Review) | Red (Reduce/Flatten) |
|--------|:--:|:--:|:--:|
| 20D Rolling Sharpe | > 0.5 | 0 ~ 0.5 | < 0 |
| Running MDD | < 8% | 8% ~ 10% | > 10% |
| 30D Return | > 0% | -5% ~ 0% | < -5% |
| Realized Slippage | ≤ 1.2× | 1.2× ~ 1.5× | > 1.5× |
| Funding Drag (annualized) | < 1% | 1% ~ 2% | > 2% |
| Sleeve 14D Contribution | All positive | 1 negative | ≥2 negative |

### 3.3 Action Matrix

| Decision | Trigger | Action |
|----------|---------|--------|
| **KEEP** | All green | Continue as-is |
| **SCALE_UP** | See §Scale-Up rules, 10 consecutive days | Increase capital 50% (manual approval) |
| **REDUCE** | Any yellow > 3 days or any red | Reduce position sizes by 50% |
| **PAUSE** | Any hard stop trigger | Full flatten → go to Incident Response |

---

## 4. Scale Rules Reference

Full rules in `config/prod_scale_rules_R1.yaml`.

### Scale-Up (ALL conditions, 10 consecutive trading days)

| Condition | Threshold |
|-----------|-----------|
| 20D Rolling Sharpe | > 0.8 |
| Running MDD | < 8% |
| Slippage | ≤ 1.2× assumed |
| Funding Drift | ≤ 1.5× |
| FLATTEN events (30D) | = 0 |

### Scale-Down (ANY triggers)

| Condition | Threshold |
|-----------|-----------|
| 20D Rolling Sharpe | < 0 |
| Running MDD | > 10% |
| Slippage > 1.5× | sustained 3 days |
| Worst sleeve 14D | < -3% |

### Hard Stop (ANY triggers → immediate flatten)

| Condition | Threshold |
|-----------|-----------|
| Running MDD | > 12% |
| 30D Return | < -10% |
| risk_guard | FLATTEN_ALL |

---

## 5. Incident Response

### 5.1 Kill Switch Trigger

**When risk_guard issues `FLATTEN_ALL` or Hard Stop conditions are met:**

**T+0 (Immediate, within 5 minutes):**
1. `risk_guard` auto-sends Telegram alert
2. Verify the bot received the signal:
   ```bash
   # Check latest risk guard decision
   cat reports/risk_guard/latest_decision.json | python -m json.tool
   ```
3. If auto-flatten did NOT execute, manually flatten:
   ```bash
   # Emergency: close all positions via Binance web UI
   # OR use CLI if available:
   PYTHONPATH=src python scripts/run_live.py -c config/prod_candidate_R1.yaml --status
   ```

**T+5 min (Confirm Flat):**
1. Verify all positions are flat on Binance:
   - Futures positions: all 0
   - Open orders: all cancelled
2. Record account balance

**T+15 min (Assess):**
1. Download trading DB:
   ```bash
   cp reports/futures/live/trading.db reports/incidents/$(date +%Y%m%d_%H%M)_trading.db
   ```
2. Screenshot exchange position page
3. Generate incident report:
   ```bash
   PYTHONPATH=src python scripts/prod_report.py --daily
   ```

**T+30 min (Communicate):**
1. Send incident summary to team channel
2. Update incident log

### 5.2 Restart Conditions

After a kill switch trigger, the strategy may only be restarted when ALL conditions are met:

- [ ] Root cause identified and documented (Post-Mortem completed)
- [ ] If code bug: fix deployed and tested
- [ ] If market regime: wait for metrics to normalize (20D Sharpe > 0 for 5 consecutive days)
- [ ] `prod_launch_guard.py` → `ALLOW_LAUNCH`
- [ ] Team sign-off (at least 2 people)
- [ ] Restart at previous stage's capital level (not current stage)

### 5.3 Rollback Procedure

If the issue is config-related:
```bash
# The frozen config should NOT have been modified
shasum -a 256 config/prod_candidate_R1.yaml
# If hash doesn't match, restore from git:
git checkout config/prod_candidate_R1.yaml
```

If the issue is code-related:
```bash
# Identify last known good commit
git log --oneline -10
# Rollback
git checkout <good_commit_hash> -- src/qtrade/
# Re-run tests
python -m pytest tests/ -x -q --tb=short
```

---

## 6. Post-Mortem Template

Copy this template for each incident:

```markdown
# Post-Mortem: [INCIDENT_ID] [Date]

## Summary
- **Trigger:** [What rule was triggered]
- **Duration:** [Start time → Resolution time]
- **Impact:** [P&L impact, capital at risk]
- **Severity:** [CRITICAL / HIGH / MEDIUM]

## Timeline
| Time (UTC) | Event |
|------------|-------|
| HH:MM | [First anomaly detected] |
| HH:MM | [Kill switch triggered] |
| HH:MM | [Positions flattened] |
| HH:MM | [Root cause identified] |
| HH:MM | [Resolution deployed] |

## Root Cause
[Detailed description]

## What Went Well
- [e.g., Auto-flatten worked within 30s]

## What Went Wrong
- [e.g., Telegram alert delayed by 5min]

## Action Items
| Item | Owner | Due Date | Status |
|------|-------|----------|--------|
| [Fix X] | [Name] | [Date] | [ ] |

## Metrics Before/After
| Metric | Before Incident | After Recovery |
|--------|:--:|:--:|
| Account Balance | | |
| Running MDD | | |
| 20D Sharpe | | |

## Restart Decision
- [ ] Root cause fixed
- [ ] Tests passed
- [ ] Launch guard: ALLOW_LAUNCH
- [ ] Team sign-off: [Name 1], [Name 2]
- [ ] Restart stage: [Stage N at $Xk]
```

---

## 7. Weekly Review Checklist

Every week (ideally Monday morning):

```bash
# Generate weekly report
PYTHONPATH=src python scripts/prod_report.py --weekly
```

- [ ] Review weekly report metrics
- [ ] Compare live P&L vs. backtest replay for same period
- [ ] Check each sleeve's contribution trend
- [ ] Review execution quality (slippage, fill rates)
- [ ] Review funding cost accumulation
- [ ] Check scale-up/down eligibility
- [ ] Update incident log (if any incidents)
- [ ] Archive reports: `reports/prod_reports/weekly/`

---

## 8. Monthly Strategy Review

Every month:

- [ ] Run full backtest replay for the live period:
  ```bash
  PYTHONPATH=src python scripts/run_portfolio_backtest.py \
    -c config/prod_candidate_R1.yaml \
    --symbols BTCUSDT ETHUSDT SOLUSDT \
    --weights 0.34 0.33 0.33
  ```
- [ ] Compare live vs. replay: Return, Sharpe, MDD, trade count
- [ ] Acceptable deviation: Sharpe within ±0.3, MDD within ±3%
- [ ] Review funding rate regime changes
- [ ] Review if strategy assumptions still hold (volatility regime, correlation)
- [ ] Decision: Continue / Adjust capital / Pause for re-evaluation

---

## Appendix A: File Reference

| File | Purpose |
|------|---------|
| `config/prod_candidate_R1.yaml` | Frozen strategy config (DO NOT MODIFY) |
| `config/prod_scale_rules_R1.yaml` | Scale-up/down/stop rules |
| `scripts/prod_launch_guard.py` | Pre-flight check script |
| `scripts/prod_report.py` | Daily/weekly report generator |
| `scripts/risk_guard.py` | Risk monitoring & kill switch |
| `scripts/run_live.py` | Live trading launcher (polling) |
| `scripts/run_websocket.py` | Live trading launcher (WebSocket, recommended) |
| `scripts/run_portfolio_backtest.py` | Portfolio backtesting |
| `scripts/validate_live_consistency.py` | Live vs. backtest consistency check |

## Appendix B: Quick Reference Commands

```bash
# ── Pre-Launch ──
shasum -a 256 config/prod_candidate_R1.yaml              # Verify config hash
PYTHONPATH=src python scripts/prod_launch_guard.py        # Pre-flight check
PYTHONPATH=src python scripts/run_live.py -c config/prod_candidate_R1.yaml --check  # API check
PYTHONPATH=src python scripts/run_live.py -c config/prod_candidate_R1.yaml --real --dry-run --once  # Signal check

# ── Launch ──
tmux new -s prod_r1
PYTHONPATH=src python scripts/run_websocket.py -c config/prod_candidate_R1.yaml --real --telegram-commands

# ── Daily Ops ──
PYTHONPATH=src python scripts/prod_report.py --daily      # Daily report
PYTHONPATH=src python scripts/risk_guard.py --config config/prod_scale_rules_R1.yaml --dry-run  # Risk check
PYTHONPATH=src python scripts/run_live.py -c config/prod_candidate_R1.yaml --status  # Position check

# ── Weekly ──
PYTHONPATH=src python scripts/prod_report.py --weekly     # Weekly report

# ── Emergency ──
tmux kill-session -t prod_r1                              # Kill bot
# Then manually close all positions on Binance web UI
```
