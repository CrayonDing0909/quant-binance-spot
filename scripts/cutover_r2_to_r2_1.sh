#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# R2 → R2.1 Production Cutover Script
# ═══════════════════════════════════════════════════════════════
# Execute on production server:
#   bash scripts/cutover_r2_to_r2_1.sh
#
# Prerequisites:
#   - config/prod_candidate_R2_1.yaml exists
#   - config/prod_scale_rules_R2_1.yaml exists
#   - .env has BINANCE_API_KEY, BINANCE_API_SECRET
#   - .env has FUTURES_TELEGRAM_BOT_TOKEN, FUTURES_TELEGRAM_CHAT_ID
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

PROJ_DIR="/home/ubuntu/quant-binance-spot"
CONFIG="config/prod_candidate_R2_1.yaml"
RULES="config/prod_scale_rules_R2_1.yaml"
OLD_SESSION="r2_prod"
NEW_SESSION="r2_1_prod"
TG_SESSION="r2_1_tg"
LOG_FILE="logs/cutover_$(date +%Y%m%d_%H%M%S).log"

cd "$PROJ_DIR"
source .venv/bin/activate
mkdir -p logs

echo "═══════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
echo "  R2 → R2.1 Production Cutover" | tee -a "$LOG_FILE"
echo "  Time: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════" | tee -a "$LOG_FILE"

# ──────────────────────────────────────────────────
# Step 1: Verify config exists and hash matches
# ──────────────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "📋 Step 1: Config Verification" | tee -a "$LOG_FILE"

if [ ! -f "$CONFIG" ]; then
    echo "❌ FATAL: $CONFIG not found!" | tee -a "$LOG_FILE"
    exit 1
fi

ACTUAL_HASH=$(shasum -a 256 "$CONFIG" | awk '{print $1}')
EXPECTED_HASH=$(grep "config_hash:" "$RULES" | awk '{print $2}' | tr -d '"')

echo "  Config: $CONFIG" | tee -a "$LOG_FILE"
echo "  Expected hash: $EXPECTED_HASH" | tee -a "$LOG_FILE"
echo "  Actual hash:   $ACTUAL_HASH" | tee -a "$LOG_FILE"

if [ "$ACTUAL_HASH" != "$EXPECTED_HASH" ]; then
    echo "❌ FATAL: Config hash mismatch!" | tee -a "$LOG_FILE"
    exit 1
fi
echo "✅ Config hash verified" | tee -a "$LOG_FILE"

# ──────────────────────────────────────────────────
# Step 2: Pre-flight guard
# ──────────────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "📋 Step 2: Pre-flight Guard" | tee -a "$LOG_FILE"

PYTHONPATH=src python scripts/prod_launch_guard.py --rules "$RULES" 2>&1 | tee -a "$LOG_FILE"
GUARD_EXIT=${PIPESTATUS[0]}

if [ "$GUARD_EXIT" -ne 0 ]; then
    echo "❌ FATAL: Launch guard BLOCKED. Fix all HARD failures." | tee -a "$LOG_FILE"
    exit 1
fi
echo "✅ Launch guard: ALLOW_LAUNCH" | tee -a "$LOG_FILE"

# ──────────────────────────────────────────────────
# Step 3: Data freshness
# ──────────────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "📋 Step 3: Data Download" | tee -a "$LOG_FILE"

PYTHONPATH=src python scripts/download_data.py -c "$CONFIG" 2>&1 | tee -a "$LOG_FILE"
PYTHONPATH=src python scripts/download_data.py -c "$CONFIG" --funding-rate 2>&1 | tee -a "$LOG_FILE"
echo "✅ Data refreshed" | tee -a "$LOG_FILE"

# ──────────────────────────────────────────────────
# Step 4: Dry-run verification
# ──────────────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "📋 Step 4: Dry-run Verification" | tee -a "$LOG_FILE"

PYTHONPATH=src python scripts/run_live.py -c "$CONFIG" --real --dry-run --once 2>&1 | tee -a "$LOG_FILE"
DRY_EXIT=${PIPESTATUS[0]}

if [ "$DRY_EXIT" -ne 0 ]; then
    echo "❌ FATAL: Dry-run failed!" | tee -a "$LOG_FILE"
    exit 1
fi
echo "✅ Dry-run passed" | tee -a "$LOG_FILE"

# ──────────────────────────────────────────────────
# Step 5: Stop old session
# ──────────────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "📋 Step 5: Stop R2 Production Session" | tee -a "$LOG_FILE"

if tmux has-session -t "$OLD_SESSION" 2>/dev/null; then
    echo "  Sending Ctrl-C to $OLD_SESSION..." | tee -a "$LOG_FILE"
    tmux send-keys -t "$OLD_SESSION" C-c
    sleep 3
    echo "  Killing session $OLD_SESSION..." | tee -a "$LOG_FILE"
    tmux kill-session -t "$OLD_SESSION"
    echo "✅ $OLD_SESSION stopped" | tee -a "$LOG_FILE"
else
    echo "⚠️  $OLD_SESSION not found (already stopped)" | tee -a "$LOG_FILE"
fi

# ──────────────────────────────────────────────────
# Step 6: Start new session
# ──────────────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "📋 Step 6: Start R2.1 Production Session" | tee -a "$LOG_FILE"

# Kill any existing R2.1 session first
if tmux has-session -t "$NEW_SESSION" 2>/dev/null; then
    echo "  Cleaning up existing $NEW_SESSION..." | tee -a "$LOG_FILE"
    tmux kill-session -t "$NEW_SESSION"
fi

tmux new -d -s "$NEW_SESSION"
tmux send-keys -t "$NEW_SESSION" \
    "cd $PROJ_DIR && source .venv/bin/activate && PYTHONPATH=src python scripts/run_websocket.py -c $CONFIG --real" Enter

echo "  Waiting 15s for startup..." | tee -a "$LOG_FILE"
sleep 15

echo "  --- R2.1 Session Output (last 50 lines) ---" | tee -a "$LOG_FILE"
tmux capture-pane -pt "$NEW_SESSION" | tail -n 50 | tee -a "$LOG_FILE"

# Verify startup markers
STARTUP_LOG=$(tmux capture-pane -pt "$NEW_SESSION" 2>/dev/null || echo "")

WS_STARTED=false
ENSEMBLE_OK=false
WS_CONNECTED=false

if echo "$STARTUP_LOG" | grep -q "WebSocket Runner 啟動"; then
    WS_STARTED=true
fi
if echo "$STARTUP_LOG" | grep -q "Ensemble 模式啟用"; then
    ENSEMBLE_OK=true
fi
if echo "$STARTUP_LOG" | grep -q "WebSocket 已連線\|等待 K 線事件"; then
    WS_CONNECTED=true
fi

echo "" | tee -a "$LOG_FILE"
echo "  WebSocket Started:  $WS_STARTED" | tee -a "$LOG_FILE"
echo "  Ensemble Mode:      $ENSEMBLE_OK" | tee -a "$LOG_FILE"
echo "  WebSocket Connected: $WS_CONNECTED" | tee -a "$LOG_FILE"

# ──────────────────────────────────────────────────
# Step 7: Verify no duplicate sessions
# ──────────────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "📋 Step 7: Session Check" | tee -a "$LOG_FILE"
echo "  Active tmux sessions:" | tee -a "$LOG_FILE"
tmux ls 2>&1 | tee -a "$LOG_FILE"

if tmux has-session -t "$OLD_SESSION" 2>/dev/null; then
    echo "❌ WARNING: $OLD_SESSION still exists! Manual intervention needed." | tee -a "$LOG_FILE"
else
    echo "✅ No duplicate trading sessions" | tee -a "$LOG_FILE"
fi

# ──────────────────────────────────────────────────
# Step 8: Telegram Sync
# ──────────────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "📋 Step 8: Telegram Sync" | tee -a "$LOG_FILE"

# Check env vars
TG_TOKEN="${FUTURES_TELEGRAM_BOT_TOKEN:-}"
TG_CHAT="${FUTURES_TELEGRAM_CHAT_ID:-}"

if [ -n "$TG_TOKEN" ] && [ -n "$TG_CHAT" ]; then
    echo "  ✅ FUTURES_TELEGRAM_BOT_TOKEN: set (${#TG_TOKEN} chars)" | tee -a "$LOG_FILE"
    echo "  ✅ FUTURES_TELEGRAM_CHAT_ID: set ($TG_CHAT)" | tee -a "$LOG_FILE"
else
    echo "  ❌ Telegram env vars missing" | tee -a "$LOG_FILE"
    echo "  FUTURES_TELEGRAM_BOT_TOKEN: ${TG_TOKEN:+set}${TG_TOKEN:-NOT SET}" | tee -a "$LOG_FILE"
    echo "  FUTURES_TELEGRAM_CHAT_ID: ${TG_CHAT:-NOT SET}" | tee -a "$LOG_FILE"
fi

# Check if Telegram bot is in the main process log
if echo "$STARTUP_LOG" | grep -q "Telegram 通知已啟用"; then
    echo "  ✅ Telegram notifications enabled in main process" | tee -a "$LOG_FILE"
else
    echo "  ⚠️  Telegram notifications not detected in main process log" | tee -a "$LOG_FILE"
fi

if echo "$STARTUP_LOG" | grep -q "Telegram Bot 已啟動\|Telegram 命令 Bot"; then
    echo "  ✅ Telegram command bot running in main process" | tee -a "$LOG_FILE"
else
    echo "  ⚠️  Telegram command bot not detected — starting backup..." | tee -a "$LOG_FILE"
    if tmux has-session -t "$TG_SESSION" 2>/dev/null; then
        tmux kill-session -t "$TG_SESSION"
    fi
    tmux new -d -s "$TG_SESSION"
    tmux send-keys -t "$TG_SESSION" \
        "cd $PROJ_DIR && source .venv/bin/activate && PYTHONPATH=src python scripts/run_telegram_bot.py -c $CONFIG --real" Enter
    echo "  ✅ Backup Telegram bot started in $TG_SESSION" | tee -a "$LOG_FILE"
fi

# ──────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
echo "  CUTOVER COMPLETE" | tee -a "$LOG_FILE"
echo "  Config: $CONFIG" | tee -a "$LOG_FILE"
echo "  Hash:   $ACTUAL_HASH" | tee -a "$LOG_FILE"
echo "  Session: $NEW_SESSION" | tee -a "$LOG_FILE"
echo "  Log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "📋 Next Steps:" | tee -a "$LOG_FILE"
echo "  1. Wait 15 min, then run health check:" | tee -a "$LOG_FILE"
echo "     tmux capture-pane -pt $NEW_SESSION | tail -n 300" | tee -a "$LOG_FILE"
echo "  2. Test Telegram commands: /ping /help /health /risk /signals /positions /status /stats" | tee -a "$LOG_FILE"
echo "  3. Monitor for 1st hour" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "🔄 Rollback (if needed):" | tee -a "$LOG_FILE"
echo "  tmux send-keys -t $NEW_SESSION C-c && sleep 3 && tmux kill-session -t $NEW_SESSION" | tee -a "$LOG_FILE"
echo "  tmux new -d -s $OLD_SESSION" | tee -a "$LOG_FILE"
echo "  tmux send-keys -t $OLD_SESSION 'cd $PROJ_DIR && source .venv/bin/activate && PYTHONPATH=src python scripts/run_websocket.py -c config/prod_candidate_R2.yaml --real' Enter" | tee -a "$LOG_FILE"
