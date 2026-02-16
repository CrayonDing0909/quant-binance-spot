#!/bin/bash
# ═══════════════════════════════════════════════════════════
# Alpha Decay 定期監控 — Cron 排程腳本
#
# 建議排程：每月 1 號和 15 號 UTC 00:00 執行
#
# 安裝 cron:
#   crontab -e
#   0 0 1,15 * * /Users/dylanting/Documents/spot_bot/quant-binance-spot/scripts/cron_alpha_monitor.sh
#
# 或每週一執行:
#   0 0 * * 1 /Users/dylanting/Documents/spot_bot/quant-binance-spot/scripts/cron_alpha_monitor.sh
# ═══════════════════════════════════════════════════════════

set -e

# 專案路徑
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# 啟動虛擬環境
source .venv/bin/activate
export PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH"

# 載入環境變數（Telegram token 等）
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

# 時間戳
TIMESTAMP=$(date -u +"%Y%m%d_%H%M%S")
LOG_DIR="$PROJECT_DIR/reports/alpha_decay/logs"
mkdir -p "$LOG_DIR"

echo "[$TIMESTAMP] Starting Alpha Decay Monitor..." >> "$LOG_DIR/cron.log"

# 執行監控（靜默模式 + Telegram 通知 + JSON 輸出）
python scripts/monitor_alpha_decay.py \
    -c config/futures_rsi_adx_atr.yaml \
    --notify \
    --output-dir reports/alpha_decay \
    --quiet \
    >> "$LOG_DIR/cron.log" 2>&1

EXIT_CODE=$?

echo "[$TIMESTAMP] Finished with exit code: $EXIT_CODE" >> "$LOG_DIR/cron.log"

# 清理 30 天以上的舊報告
find "$PROJECT_DIR/reports/alpha_decay" -name "ic_report_*.json" -mtime +30 -delete 2>/dev/null || true

exit $EXIT_CODE
