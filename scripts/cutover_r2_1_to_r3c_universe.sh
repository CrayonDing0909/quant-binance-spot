#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# R2.1 → R3C Universe Production Cutover Script
# ═══════════════════════════════════════════════════════════════
# Execute on production server:
#   bash scripts/cutover_r2_1_to_r3c_universe.sh
#
# Rollback to R2.1:
#   bash scripts/cutover_r2_1_to_r3c_universe.sh --rollback
#
# Prerequisites:
#   - config/prod_candidate_R3C_universe.yaml exists
#   - config/prod_scale_rules_R3C_universe.yaml exists
#   - .env has BINANCE_API_KEY, BINANCE_API_SECRET
#   - .env has FUTURES_TELEGRAM_BOT_TOKEN, FUTURES_TELEGRAM_CHAT_ID
#   - All 19 symbols have 1h + 5m data downloaded
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

PROJ_DIR="/home/ubuntu/quant-binance-spot"
NEW_CONFIG="config/prod_candidate_R3C_universe.yaml"
OLD_CONFIG="config/prod_candidate_R2_1.yaml"
SCALE_RULES="config/prod_scale_rules_R3C_universe.yaml"
OLD_SESSION="r2_1_prod"
NEW_SESSION="r3c_prod"
TG_SESSION="r3c_tg"
LOG_FILE="logs/cutover_r3c_$(date +%Y%m%d_%H%M%S).log"

cd "$PROJ_DIR"
source .venv/bin/activate
mkdir -p logs

# ──────────────────────────────────────────────────
# Handle --rollback flag
# ──────────────────────────────────────────────────
if [ "${1:-}" = "--rollback" ]; then
    echo "═══════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
    echo "  ROLLBACK: R3C Universe → R2.1" | tee -a "$LOG_FILE"
    echo "  Time: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG_FILE"
    echo "═══════════════════════════════════════════════════════" | tee -a "$LOG_FILE"

    # Stop R3C session
    if tmux has-session -t "$NEW_SESSION" 2>/dev/null; then
        echo "  Sending Ctrl-C to $NEW_SESSION..." | tee -a "$LOG_FILE"
        tmux send-keys -t "$NEW_SESSION" C-c
        sleep 3
        tmux kill-session -t "$NEW_SESSION"
        echo "  ✅ $NEW_SESSION stopped" | tee -a "$LOG_FILE"
    fi

    # Stop R3C telegram session
    if tmux has-session -t "$TG_SESSION" 2>/dev/null; then
        tmux kill-session -t "$TG_SESSION"
        echo "  ✅ $TG_SESSION stopped" | tee -a "$LOG_FILE"
    fi

    # Start R2.1 session
    tmux new -d -s "$OLD_SESSION"
    tmux send-keys -t "$OLD_SESSION" \
        "cd $PROJ_DIR && source .venv/bin/activate && PYTHONPATH=src python scripts/run_websocket.py -c $OLD_CONFIG --real" Enter
    echo "  ✅ R2.1 session restarted in $OLD_SESSION" | tee -a "$LOG_FILE"

    echo "" | tee -a "$LOG_FILE"
    echo "  ROLLBACK COMPLETE" | tee -a "$LOG_FILE"
    echo "  Verify: tmux capture-pane -pt $OLD_SESSION | tail -n 50" | tee -a "$LOG_FILE"
    exit 0
fi

# ──────────────────────────────────────────────────
# Main Cutover Flow
# ──────────────────────────────────────────────────

echo "═══════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
echo "  R2.1 → R3C Universe Production Cutover" | tee -a "$LOG_FILE"
echo "  Time: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG_FILE"
echo "  19-symbol universe with micro accel overlay" | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════" | tee -a "$LOG_FILE"

# ──────────────────────────────────────────────────
# Step 1: Verify configs exist
# ──────────────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "📋 Step 1: Config Verification" | tee -a "$LOG_FILE"

if [ ! -f "$NEW_CONFIG" ]; then
    echo "❌ FATAL: $NEW_CONFIG not found!" | tee -a "$LOG_FILE"
    exit 1
fi
if [ ! -f "$SCALE_RULES" ]; then
    echo "❌ FATAL: $SCALE_RULES not found!" | tee -a "$LOG_FILE"
    exit 1
fi
if [ ! -f "$OLD_CONFIG" ]; then
    echo "⚠️  WARNING: $OLD_CONFIG not found (rollback config)" | tee -a "$LOG_FILE"
fi

CONFIG_HASH=$(shasum -a 256 "$NEW_CONFIG" | awk '{print $1}')
echo "  Config: $NEW_CONFIG" | tee -a "$LOG_FILE"
echo "  Hash:   $CONFIG_HASH" | tee -a "$LOG_FILE"
echo "✅ Config verified" | tee -a "$LOG_FILE"

# ──────────────────────────────────────────────────
# Step 2: Verify data availability for all 19 symbols
# ──────────────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "📋 Step 2: Data Availability Check" | tee -a "$LOG_FILE"

SYMBOLS=(BTCUSDT ETHUSDT SOLUSDT BNBUSDT XRPUSDT DOGEUSDT ADAUSDT AVAXUSDT LINKUSDT DOTUSDT LTCUSDT NEARUSDT APTUSDT ARBUSDT OPUSDT FILUSDT ATOMUSDT UNIUSDT AAVEUSDT)
DATA_OK=true

for SYM in "${SYMBOLS[@]}"; do
    # Check 1h data
    H1_FILE="data/binance/futures/1h/${SYM}.parquet"
    if [ ! -f "$H1_FILE" ]; then
        echo "  ❌ Missing 1h: $SYM" | tee -a "$LOG_FILE"
        DATA_OK=false
    fi
    # Check 5m data
    M5_FILE="data/binance/futures/5m/${SYM}.parquet"
    if [ ! -f "$M5_FILE" ]; then
        echo "  ⚠️  Missing 5m: $SYM (overlay will use 1h fallback)" | tee -a "$LOG_FILE"
    fi
done

if [ "$DATA_OK" = false ]; then
    echo "❌ FATAL: Missing critical data files!" | tee -a "$LOG_FILE"
    echo "  Run: PYTHONPATH=src python scripts/download_data.py -c $NEW_CONFIG" | tee -a "$LOG_FILE"
    exit 1
fi
echo "✅ All 19 symbols have 1h data" | tee -a "$LOG_FILE"

# ──────────────────────────────────────────────────
# Step 3: Data freshness — download latest
# ──────────────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "📋 Step 3: Data Download (incremental)" | tee -a "$LOG_FILE"

PYTHONPATH=src python scripts/download_data.py -c "$NEW_CONFIG" 2>&1 | tee -a "$LOG_FILE"
PYTHONPATH=src python scripts/download_data.py -c "$NEW_CONFIG" --funding-rate 2>&1 | tee -a "$LOG_FILE"

# Also download 5m data for micro accel overlay
for SYM in "${SYMBOLS[@]}"; do
    PYTHONPATH=src python scripts/download_data.py -c "$NEW_CONFIG" --symbol "$SYM" --interval 5m 2>&1 | tee -a "$LOG_FILE" || true
done

echo "✅ Data refreshed" | tee -a "$LOG_FILE"

# ──────────────────────────────────────────────────
# Step 4: Dry-run verification
# ──────────────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "📋 Step 4: Dry-run Verification" | tee -a "$LOG_FILE"

PYTHONPATH=src python scripts/run_live.py -c "$NEW_CONFIG" --real --dry-run --once 2>&1 | tee -a "$LOG_FILE"
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
echo "📋 Step 5: Stop R2.1 Production Session" | tee -a "$LOG_FILE"

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
echo "📋 Step 6: Start R3C Universe Production Session" | tee -a "$LOG_FILE"

# Kill any existing session first
if tmux has-session -t "$NEW_SESSION" 2>/dev/null; then
    echo "  Cleaning up existing $NEW_SESSION..." | tee -a "$LOG_FILE"
    tmux kill-session -t "$NEW_SESSION"
fi

tmux new -d -s "$NEW_SESSION"
tmux send-keys -t "$NEW_SESSION" \
    "cd $PROJ_DIR && source .venv/bin/activate && PYTHONPATH=src python scripts/run_websocket.py -c $NEW_CONFIG --real" Enter

echo "  Waiting 20s for startup (19 symbols need more time)..." | tee -a "$LOG_FILE"
sleep 20

echo "  --- R3C Session Output (last 80 lines) ---" | tee -a "$LOG_FILE"
tmux capture-pane -pt "$NEW_SESSION" | tail -n 80 | tee -a "$LOG_FILE"

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

TG_TOKEN="${FUTURES_TELEGRAM_BOT_TOKEN:-}"
TG_CHAT="${FUTURES_TELEGRAM_CHAT_ID:-}"

if [ -n "$TG_TOKEN" ] && [ -n "$TG_CHAT" ]; then
    echo "  ✅ FUTURES_TELEGRAM_BOT_TOKEN: set (${#TG_TOKEN} chars)" | tee -a "$LOG_FILE"
    echo "  ✅ FUTURES_TELEGRAM_CHAT_ID: set ($TG_CHAT)" | tee -a "$LOG_FILE"
else
    echo "  ❌ Telegram env vars missing" | tee -a "$LOG_FILE"
fi

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
        "cd $PROJ_DIR && source .venv/bin/activate && PYTHONPATH=src python scripts/run_telegram_bot.py -c $NEW_CONFIG --real" Enter
    echo "  ✅ Backup Telegram bot started in $TG_SESSION" | tee -a "$LOG_FILE"
fi

# ──────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
echo "  CUTOVER COMPLETE" | tee -a "$LOG_FILE"
echo "  Config: $NEW_CONFIG" | tee -a "$LOG_FILE"
echo "  Hash:   $CONFIG_HASH" | tee -a "$LOG_FILE"
echo "  Session: $NEW_SESSION" | tee -a "$LOG_FILE"
echo "  Symbols: 19" | tee -a "$LOG_FILE"
echo "  Log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "📋 Next Steps:" | tee -a "$LOG_FILE"
echo "  1. Wait 15 min, then run health check:" | tee -a "$LOG_FILE"
echo "     bash scripts/healthcheck_r3c_universe.sh" | tee -a "$LOG_FILE"
echo "  2. Test Telegram: /ping /help /health /risk /signals /positions /status /stats" | tee -a "$LOG_FILE"
echo "  3. Monitor for 1st hour, check all 19 symbols active" | tee -a "$LOG_FILE"
echo "  4. Follow scale rules in $SCALE_RULES" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "🔄 Rollback (if needed):" | tee -a "$LOG_FILE"
echo "  bash scripts/cutover_r2_1_to_r3c_universe.sh --rollback" | tee -a "$LOG_FILE"
