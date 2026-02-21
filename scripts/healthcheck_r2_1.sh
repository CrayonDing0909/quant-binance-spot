#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# R2.1 Post-Launch Health Check (15 minute)
# ═══════════════════════════════════════════════════════════════
# Execute after cutover:
#   bash scripts/healthcheck_r2_1.sh
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

PROJ_DIR="/home/ubuntu/quant-binance-spot"
SESSION="r2_1_prod"
TG_SESSION="r2_1_tg"
LOG_FILE="logs/healthcheck_$(date +%Y%m%d_%H%M%S).log"

cd "$PROJ_DIR"
source .venv/bin/activate
mkdir -p logs

echo "═══════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
echo "  R2.1 Post-Launch Health Check" | tee -a "$LOG_FILE"
echo "  Time: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════" | tee -a "$LOG_FILE"

# ──────────────────────────────────────────────────
# 1. Session alive check
# ──────────────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "📋 1. Session Status" | tee -a "$LOG_FILE"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "  ✅ $SESSION is running" | tee -a "$LOG_FILE"
else
    echo "  ❌ $SESSION NOT FOUND — CRITICAL" | tee -a "$LOG_FILE"
    exit 1
fi

echo "" | tee -a "$LOG_FILE"
echo "  Active tmux sessions:" | tee -a "$LOG_FILE"
tmux ls 2>&1 | tee -a "$LOG_FILE"

# ──────────────────────────────────────────────────
# 2. Error check
# ──────────────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "📋 2. Error/Warning Analysis (last 300 lines)" | tee -a "$LOG_FILE"

CAPTURE=$(tmux capture-pane -pt "$SESSION" 2>/dev/null || echo "")

ERROR_COUNT=$(echo "$CAPTURE" | grep -ci "error\|traceback\|exception\|fatal" || true)
WARNING_COUNT=$(echo "$CAPTURE" | grep -ci "warning\|⚠️" || true)

echo "  Errors detected: $ERROR_COUNT" | tee -a "$LOG_FILE"
echo "  Warnings detected: $WARNING_COUNT" | tee -a "$LOG_FILE"

if [ "$ERROR_COUNT" -gt 0 ]; then
    echo "" | tee -a "$LOG_FILE"
    echo "  --- Error lines ---" | tee -a "$LOG_FILE"
    echo "$CAPTURE" | grep -i "error\|traceback\|exception\|fatal" | tail -20 | tee -a "$LOG_FILE"
fi

# ──────────────────────────────────────────────────
# 3. K-line event continuity
# ──────────────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "📋 3. K-line Event Continuity" | tee -a "$LOG_FILE"

KLINE_EVENTS=$(echo "$CAPTURE" | grep -c "K線\|kline\|📊\|signal=" || true)
echo "  K-line related events: $KLINE_EVENTS" | tee -a "$LOG_FILE"

if [ "$KLINE_EVENTS" -gt 0 ]; then
    echo "  ✅ K-line events detected" | tee -a "$LOG_FILE"
    echo "  Last 5 K-line events:" | tee -a "$LOG_FILE"
    echo "$CAPTURE" | grep "📊\|signal=" | tail -5 | tee -a "$LOG_FILE"
else
    echo "  ⚠️  No K-line events detected yet (may be between bar intervals)" | tee -a "$LOG_FILE"
fi

# ──────────────────────────────────────────────────
# 4. Signal/Trade DB check
# ──────────────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "📋 4. Database Check" | tee -a "$LOG_FILE"

DB_PATH="reports/futures/tsmom_ema/live/trading.db"
if [ -f "$DB_PATH" ]; then
    echo "  ✅ Trading DB exists: $DB_PATH" | tee -a "$LOG_FILE"
    DB_SIZE=$(ls -lh "$DB_PATH" | awk '{print $5}')
    echo "  DB size: $DB_SIZE" | tee -a "$LOG_FILE"

    # Check recent signals
    RECENT_SIGNALS=$(python3 -c "
import sqlite3, datetime
conn = sqlite3.connect('$DB_PATH')
try:
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM signals WHERE timestamp > datetime(\"now\", \"-1 hour\")')
    print(f'Recent signals (1h): {c.fetchone()[0]}')
except:
    print('signals table not found or empty')
try:
    c.execute('SELECT COUNT(*) FROM trades WHERE timestamp > datetime(\"now\", \"-1 hour\")')
    print(f'Recent trades (1h): {c.fetchone()[0]}')
except:
    print('trades table not found or empty')
conn.close()
" 2>/dev/null || echo "  Could not query DB")
    echo "  $RECENT_SIGNALS" | tee -a "$LOG_FILE"
else
    echo "  ⚠️  Trading DB not found at $DB_PATH" | tee -a "$LOG_FILE"
fi

# ──────────────────────────────────────────────────
# 5. Telegram check
# ──────────────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "📋 5. Telegram Status" | tee -a "$LOG_FILE"

if echo "$CAPTURE" | grep -q "Telegram 通知已啟用"; then
    echo "  ✅ Telegram notifications enabled" | tee -a "$LOG_FILE"
else
    echo "  ⚠️  Telegram notification status unknown" | tee -a "$LOG_FILE"
fi

if echo "$CAPTURE" | grep -q "Telegram Bot 已啟動\|Telegram 命令 Bot"; then
    echo "  ✅ Telegram command bot running (main process)" | tee -a "$LOG_FILE"
elif tmux has-session -t "$TG_SESSION" 2>/dev/null; then
    echo "  ✅ Telegram command bot running (backup session: $TG_SESSION)" | tee -a "$LOG_FILE"
else
    echo "  ❌ Telegram command bot NOT detected" | tee -a "$LOG_FILE"
fi

# ──────────────────────────────────────────────────
# 6. Memory/Process check
# ──────────────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "📋 6. Process Health" | tee -a "$LOG_FILE"

PYTHON_PROCS=$(ps aux | grep "run_websocket.py" | grep -v grep | wc -l || true)
echo "  WebSocket processes running: $PYTHON_PROCS" | tee -a "$LOG_FILE"

if [ "$PYTHON_PROCS" -eq 1 ]; then
    echo "  ✅ Exactly 1 WebSocket process" | tee -a "$LOG_FILE"
    ps aux | grep "run_websocket.py" | grep -v grep | tee -a "$LOG_FILE"
elif [ "$PYTHON_PROCS" -eq 0 ]; then
    echo "  ❌ No WebSocket process found — CRITICAL" | tee -a "$LOG_FILE"
else
    echo "  ⚠️  Multiple WebSocket processes — check for duplicates" | tee -a "$LOG_FILE"
    ps aux | grep "run_websocket.py" | grep -v grep | tee -a "$LOG_FILE"
fi

# ──────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════" | tee -a "$LOG_FILE"
echo "  HEALTH CHECK SUMMARY" | tee -a "$LOG_FILE"
echo "═══════════════════════════════════════════════════════" | tee -a "$LOG_FILE"

FAIL_COUNT=0

if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "  ❌ Session not running" | tee -a "$LOG_FILE"
    FAIL_COUNT=$((FAIL_COUNT + 1))
else
    echo "  ✅ Session running" | tee -a "$LOG_FILE"
fi

if [ "$ERROR_COUNT" -gt 5 ]; then
    echo "  ❌ Too many errors ($ERROR_COUNT)" | tee -a "$LOG_FILE"
    FAIL_COUNT=$((FAIL_COUNT + 1))
else
    echo "  ✅ Error count acceptable ($ERROR_COUNT)" | tee -a "$LOG_FILE"
fi

echo "  📊 K-line events: $KLINE_EVENTS" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

if [ "$FAIL_COUNT" -eq 0 ]; then
    echo "  ✅ HEALTH: OK — System stable" | tee -a "$LOG_FILE"
else
    echo "  ❌ HEALTH: ISSUES DETECTED ($FAIL_COUNT critical)" | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"
echo "  📄 Log saved: $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "  📋 Full session log:" | tee -a "$LOG_FILE"
echo "     tmux capture-pane -pt $SESSION | tail -n 300" | tee -a "$LOG_FILE"
