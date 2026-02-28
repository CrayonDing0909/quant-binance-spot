---
description: Unified multi-strategy Telegram Bot deployment and management
globs:
alwaysApply: false
---
# Skill: Telegram Bot Deployment

> Loaded by DevOps when deploying, updating, or troubleshooting the Telegram Bot.

## Architecture

The unified Telegram Bot is an **independent process** that directly queries Binance API for account state and reads signal snapshots (`last_signals.json`) from each strategy Runner.

```
┌─────────────────────────┐
│  meta_blend_live (tmux) │
│  WebSocketRunner        │
│  → writes last_signals  │
│    .json to reports/    │
└─────────────────────────┘
            │
            ▼
   ┌────────────────────┐
   │  tg_bot (tmux)     │
   │  run_telegram_bot   │
   │  MultiStrategyBot  │
   │  ← reads signals   │
   │  ← queries Binance │
   └────────────────────┘
```

- **Runner doesn't start command bot** — only keeps `TelegramNotifier` for trade push notifications
- **No Bot Token conflict** — only one process does long-polling
- **Multi-strategy support** — one Bot sees all strategy states
- **Paper strategy auto-detection** — config filename containing `paper` or `oi_liq_bounce` strategy name → auto-labeled 🧪 Paper
- **Exclusive position attribution** — shared symbols (BTC/ETH/SOL etc.) attributed only to Real strategy; Paper strategy shows placeholder

## Deploy Command

```bash
ssh -i ~/.ssh/oracle-trading-bot.key ubuntu@140.83.57.255
cd ~/quant-binance-spot && source .venv/bin/activate && git pull

tmux new -d -s tg_bot 'while true; do
  cd ~/quant-binance-spot && source .venv/bin/activate &&
  PYTHONPATH=src python scripts/run_telegram_bot.py \
    -c config/prod_candidate_htf_lsr.yaml \
    --real;
  echo "TG Bot exited, restarting in 10s..."; sleep 10;
done'

sleep 5 && tmux capture-pane -t tg_bot -p | tail -10
```

## Environment Variables

`.env` needs:
```
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

## Adding New Strategy

When a new strategy goes live, just:
1. Add `-c config/new_strategy.yaml` to the `tg_bot` tmux command
2. Restart tg_bot session

## Available Commands

| Command | Function |
|---------|----------|
| `/dashboard` | Global dashboard: account, strategies, positions overview |
| `/status` | Account status + per-strategy Runner status |
| `/positions` | Futures positions (exclusive strategy attribution) |
| `/trades` | Recent trades (merged fills, smart decimals, strategy labels) |
| `/pnl` | Realized PnL (multi-period switching) |
| `/signals` | Per-strategy latest signal snapshot |
| `/risk` | Risk metrics (margin, funding rate, etc.) |
| `/health` | Strategy health (signal freshness, heartbeat) |
| `/help` | Command menu (inline buttons) |

Daily summary auto-sent at UTC 00:05.

## Notes

- `--telegram-commands` flag in `run_live.py` is deprecated (prints redirect message)
- WebSocketRunner auto-writes `last_signals.json` for Bot to read
- Bot can start without `-c` params (only `/ping`, `/help` available)
