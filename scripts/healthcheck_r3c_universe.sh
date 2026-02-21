#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# Health Check Script — R3C Universe Production
# ═══════════════════════════════════════════════════════════════
# Usage:
#   bash scripts/healthcheck_r3c_universe.sh
#
# Run this every 15-60 minutes after cutover, then hourly for
# the first 24 hours, then daily afterward.
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

PROJ_DIR="/home/ubuntu/quant-binance-spot"
CONFIG="config/prod_candidate_R3C_universe.yaml"
SESSION="r3c_univ_prod"
TG_SESSION="r3c_univ_tg"

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'  # No Color

pass() { echo -e "${GREEN}✅ PASS${NC}: $1"; }
warn() { echo -e "${YELLOW}⚠️  WARN${NC}: $1"; }
fail() { echo -e "${RED}❌ FAIL${NC}: $1"; }

cd "$PROJ_DIR" 2>/dev/null || cd "$(dirname "$0")/.."

echo "═══════════════════════════════════════════════════════"
echo "  R3C Universe Health Check"
echo "  Time: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "═══════════════════════════════════════════════════════"
echo ""

ISSUES=0

# ──────────────────────────────────────────────────
# Check 1: Trading session alive
# ──────────────────────────────────────────────────
echo "📋 Check 1: Trading Session"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    pass "Session $SESSION is active"
    
    # Get last 200 lines of output
    SESSION_LOG=$(tmux capture-pane -pt "$SESSION" -S -200 2>/dev/null || echo "")
    
    # Check for error patterns
    ERROR_COUNT=$(echo "$SESSION_LOG" | grep -ci "error\|exception\|traceback\|fatal" || true)
    if [ "$ERROR_COUNT" -gt 0 ]; then
        warn "Found $ERROR_COUNT error-like lines in session log"
        ISSUES=$((ISSUES + 1))
    else
        pass "No errors in recent session log"
    fi
    
    # Check for heartbeat (K-line events)
    if echo "$SESSION_LOG" | grep -q "K 線\|kline\|signal\|position"; then
        pass "Recent trading activity detected"
    else
        warn "No recent trading activity in visible log"
        ISSUES=$((ISSUES + 1))
    fi
else
    fail "Session $SESSION NOT FOUND!"
    ISSUES=$((ISSUES + 1))
fi

echo ""

# ──────────────────────────────────────────────────
# Check 2: Telegram session
# ──────────────────────────────────────────────────
echo "📋 Check 2: Telegram Bot"

if tmux has-session -t "$TG_SESSION" 2>/dev/null; then
    pass "Telegram session $TG_SESSION is active"
else
    # Check if TG bot is running in main session
    if echo "${SESSION_LOG:-}" | grep -q "Telegram Bot 已啟動\|Telegram 命令 Bot"; then
        pass "Telegram bot running in main session"
    else
        warn "No Telegram bot session found"
        ISSUES=$((ISSUES + 1))
    fi
fi

echo ""

# ──────────────────────────────────────────────────
# Check 3: Data freshness (1h files)
# ──────────────────────────────────────────────────
echo "📋 Check 3: Data Freshness"

SYMBOLS=(BTCUSDT ETHUSDT SOLUSDT BNBUSDT XRPUSDT DOGEUSDT ADAUSDT AVAXUSDT LINKUSDT DOTUSDT LTCUSDT NEARUSDT APTUSDT ARBUSDT OPUSDT FILUSDT ATOMUSDT UNIUSDT AAVEUSDT)
STALE_COUNT=0
MISSING_COUNT=0
NOW=$(date +%s)

for SYM in "${SYMBOLS[@]}"; do
    H1_FILE="data/binance/futures/klines/${SYM}_1h.parquet"
    if [ -f "$H1_FILE" ]; then
        FILE_AGE=$(( NOW - $(stat -c %Y "$H1_FILE" 2>/dev/null || stat -f %m "$H1_FILE" 2>/dev/null || echo 0) ))
        # Warn if older than 2 hours
        if [ "$FILE_AGE" -gt 7200 ]; then
            STALE_COUNT=$((STALE_COUNT + 1))
        fi
    else
        MISSING_COUNT=$((MISSING_COUNT + 1))
    fi
done

if [ "$MISSING_COUNT" -gt 0 ]; then
    fail "$MISSING_COUNT symbols missing 1h data!"
    ISSUES=$((ISSUES + 1))
else
    pass "All 19 symbols have 1h data"
fi

if [ "$STALE_COUNT" -gt 3 ]; then
    warn "$STALE_COUNT symbols have stale 1h data (>2h old)"
    ISSUES=$((ISSUES + 1))
elif [ "$STALE_COUNT" -gt 0 ]; then
    warn "$STALE_COUNT symbols slightly stale (>2h old)"
else
    pass "All 1h data files are fresh"
fi

echo ""

# ──────────────────────────────────────────────────
# Check 4: Config file integrity
# ──────────────────────────────────────────────────
echo "📋 Check 4: Config Integrity"

if [ -f "$CONFIG" ]; then
    CONFIG_HASH=$(shasum -a 256 "$CONFIG" | awk '{print $1}')
    SYMBOL_COUNT=$(grep -c "USDT" "$CONFIG" | head -1 || echo "0")
    pass "Config exists (hash: ${CONFIG_HASH:0:16}...)"
    
    # Verify it has 19 symbols in market section
    MARKET_SYMBOLS=$(grep -A 25 "^market:" "$CONFIG" | grep "USDT" | wc -l | tr -d ' ')
    if [ "$MARKET_SYMBOLS" -ge 19 ]; then
        pass "Config has $MARKET_SYMBOLS symbols"
    else
        warn "Config has $MARKET_SYMBOLS symbols (expected 19)"
        ISSUES=$((ISSUES + 1))
    fi
else
    fail "Config file $CONFIG not found!"
    ISSUES=$((ISSUES + 1))
fi

echo ""

# ──────────────────────────────────────────────────
# Check 5: Disk space
# ──────────────────────────────────────────────────
echo "📋 Check 5: Disk Space"

DISK_PCT=$(df "$PROJ_DIR" | tail -1 | awk '{print $5}' | tr -d '%')
if [ "$DISK_PCT" -gt 90 ]; then
    fail "Disk usage $DISK_PCT% — CRITICAL!"
    ISSUES=$((ISSUES + 1))
elif [ "$DISK_PCT" -gt 80 ]; then
    warn "Disk usage $DISK_PCT%"
    ISSUES=$((ISSUES + 1))
else
    pass "Disk usage $DISK_PCT%"
fi

echo ""

# ──────────────────────────────────────────────────
# Check 6: No duplicate trading sessions
# ──────────────────────────────────────────────────
echo "📋 Check 6: Session Conflicts"

TRADING_SESSIONS=$(tmux ls 2>/dev/null | grep -c "prod\|live\|trading" || true)
if [ "$TRADING_SESSIONS" -gt 2 ]; then
    warn "Found $TRADING_SESSIONS production-like sessions — verify no duplicates"
    tmux ls 2>/dev/null | grep "prod\|live\|trading" || true
    ISSUES=$((ISSUES + 1))
else
    pass "No duplicate trading sessions"
fi

echo ""

# ──────────────────────────────────────────────────
# Check 7: Log file growth
# ──────────────────────────────────────────────────
echo "📋 Check 7: Log Files"

if [ -d "logs" ]; then
    LOG_SIZE=$(du -sm logs/ 2>/dev/null | awk '{print $1}')
    if [ "${LOG_SIZE:-0}" -gt 1000 ]; then
        warn "Logs directory is ${LOG_SIZE}MB — consider rotation"
        ISSUES=$((ISSUES + 1))
    else
        pass "Logs directory: ${LOG_SIZE:-0}MB"
    fi
else
    pass "No logs directory (OK for fresh deployment)"
fi

echo ""

# ──────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════"
if [ "$ISSUES" -eq 0 ]; then
    echo -e "  ${GREEN}ALL CHECKS PASSED${NC} — R3C Universe is healthy"
else
    echo -e "  ${YELLOW}$ISSUES ISSUE(S) FOUND${NC} — review warnings above"
fi
echo "  Active sessions:"
tmux ls 2>/dev/null || echo "  (no tmux sessions)"
echo "═══════════════════════════════════════════════════════"

exit "$ISSUES"
