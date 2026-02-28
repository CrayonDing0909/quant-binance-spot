---
description: Step-by-step deployment, restart, config upgrade, and emergency rollback procedures for Oracle Cloud
globs:
alwaysApply: false
---
# Skill: Deployment Procedures

> Loaded by DevOps when deploying, restarting, or rolling back runners on Oracle Cloud.

## Deploy / Restart WebSocket Runner

```bash
# 1. SSH
ssh -i ~/.ssh/oracle-trading-bot.key ubuntu@140.83.57.255

# 2. Setup Swap (first time only)
bash scripts/setup_swap.sh

# 3. Start with tmux (auto-restart loop)
tmux kill-session -t meta_blend_live 2>/dev/null
tmux new -d -s meta_blend_live 'while true; do
  cd ~/quant-binance-spot && source .venv/bin/activate && git pull &&
  PYTHONPATH=src python scripts/run_websocket.py -c config/prod_candidate_simplified.yaml --real;
  echo "Runner exited, restarting in 10s..."; sleep 10;
done'

# 4. Verify startup
sleep 10 && tmux capture-pane -t meta_blend_live -p | tail -20
```

## Update Deployment (Add Coins / Change Params)

```bash
ssh -i ~/.ssh/oracle-trading-bot.key ubuntu@140.83.57.255
cd ~/quant-binance-spot
git stash && git pull

# Download new data if needed
source .venv/bin/activate
PYTHONPATH=src python scripts/download_data.py -c config/prod_candidate_simplified.yaml
PYTHONPATH=src python scripts/download_data.py -c config/prod_candidate_simplified.yaml --funding-rate

# Restart runner
tmux send-keys -t meta_blend_live C-c
sleep 15 && tmux capture-pane -t meta_blend_live -p | tail -20
```

## Strategy Upgrade (Config Replacement)

Same tmux session, just change the config in the while-true loop:

```bash
ssh -i ~/.ssh/oracle-trading-bot.key ubuntu@140.83.57.255
cd ~/quant-binance-spot && source .venv/bin/activate && git pull

# Download any additional data needed
PYTHONPATH=src python scripts/download_data.py -c config/<NEW_CONFIG>.yaml
PYTHONPATH=src python scripts/download_data.py -c config/<NEW_CONFIG>.yaml --derivatives

# Restart (while-true loop auto-restarts with new config after git pull)
tmux send-keys -t meta_blend_live C-c
sleep 15 && tmux capture-pane -t meta_blend_live -p | tail -20
```

If tmux session needs full rebuild:
```bash
tmux kill-session -t meta_blend_live 2>/dev/null
tmux new -d -s meta_blend_live 'while true; do
  cd ~/quant-binance-spot && source .venv/bin/activate && git pull &&
  PYTHONPATH=src python scripts/run_websocket.py -c config/<NEW_CONFIG>.yaml --real;
  echo "Runner exited, restarting in 10s..."; sleep 10;
done'
```

## Extra Data Requirements for Current Strategy

| Data | Purpose | Download Command |
|------|---------|-----------------|
| 1h Klines | Primary + HTF resample (4h/1d) | `download_data.py -c <cfg>` |
| Funding Rate | FR carry signal | `download_data.py -c <cfg> --funding-rate` |
| Open Interest | OI overlay (vol_pause) | `download_data.py --oi` |
| **LSR** | **LSR confirmatory overlay** | `download_data.py -c <cfg> --derivatives` |

HTF Filter 4h/1d data is generated via `_resample_ohlcv()` from 1h — no extra kline download needed.

## Emergency Rollback

1. `tmux attach -t meta_blend_live` → Ctrl+C
2. `git log --oneline -5` to identify target commit
3. `git checkout <commit>` to revert
4. Restart runner (same as above)

**Config rollback**: Change tmux while-loop to use the fallback config (see current architecture in core agent doc).
