#!/bin/bash
# ============================================================
# DCA 定投 + 社群自動發文 — Cron 設定
#
# 兩個排程（台灣時間，Asia/Taipei）：
#   23:00  買入 DCA（逐時 XIRR 差距是雜訊等級；23:00 為當前資料的最高，僅供參考）
#          ⚠️ 預設「不帶 --execute」= dry-run，不會真的下單。
#          確認無誤要正式上線時，才把買入那行的 RUN_FLAGS 改成 "--execute"。
#   18:00  發文（預設用 Telegram 把可貼到 Threads 的文案推到手機）
#          缺 token 時會自動降級成 dry-run，所以現在就能安裝。
#          想改用官方 API 直接發，把 --send-telegram 換成 --post-threads（並設好 token）。
#
# 用法：
#   ./scripts/setup_dca_cron.sh              # 預覽（預設）
#   ./scripts/setup_dca_cron.sh --install    # 安裝（請在 Oracle VM 上執行）
#   ./scripts/setup_dca_cron.sh --remove     # 移除
#   ./scripts/setup_dca_cron.sh --show       # 顯示目前設定
#
# 注意：
#   - CRON_TZ=Asia/Taipei 需要 Linux (Vixie cron, 例如 Ubuntu/Oracle Cloud)；
#     macOS cron 不支援 CRON_TZ，請改用下方註解的 UTC 時間版本。
#   - 此腳本只「附加」自己的區塊（marker 不同），不會動到 setup_cron.sh 的排程。
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_ROOT/logs"
CONFIG_FILE="config/dca.yaml"
CRON_MARKER="# QUANT-DCA-SOCIAL"

# 買入旗標：dry-run 用空字串；要正式下單改成 "--execute"（同時 config 的 dry_run 也要設 false）
RUN_FLAGS=""

# 偵測 python（優先 .venv）
if [ -d "$PROJECT_ROOT/.venv" ]; then
    PYTHON_CMD="$PROJECT_ROOT/.venv/bin/python"
    ACTIVATE="source $PROJECT_ROOT/.venv/bin/activate && "
else
    PYTHON_CMD="python3"
    ACTIVATE=""
fi

generate_cron_config() {
    cat << EOF
$CRON_MARKER - START
# ============================================================
# DCA + 社群發文 — 自動產生於 $(date '+%Y-%m-%d %H:%M:%S')
# 專案: $PROJECT_ROOT
# ============================================================
SHELL=/bin/bash
PATH=/usr/local/bin:/usr/bin:/bin
# 讓下方時間以台灣時間解讀（Linux Vixie cron 支援；macOS 不支援）
CRON_TZ=Asia/Taipei

# 1. 買入 DCA — 每日台灣時間 23:00（預設 dry-run，不帶 --execute）
0 23 * * * cd $PROJECT_ROOT && ${ACTIVATE}python scripts/run_dca.py -c $CONFIG_FILE $RUN_FLAGS >> $LOG_DIR/dca_buy.log 2>&1

# 2. 社群發文 — 每日台灣時間 18:00（Telegram 推文案到手機；要直接發 Threads 改 --post-threads）
0 18 * * * cd $PROJECT_ROOT && ${ACTIVATE}python scripts/post_dca_social.py -c $CONFIG_FILE --send-telegram >> $LOG_DIR/dca_social.log 2>&1

# ── 若系統 cron 不支援 CRON_TZ（例如 macOS），改用 UTC 時間（= 台灣時間 -8）：
#   買入 23:00 台灣 = 15:00 UTC: 0 15 * * *
#   發文 18:00 台灣 = 10:00 UTC: 0 10 * * *
$CRON_MARKER - END
EOF
}

install_cron() {
    mkdir -p "$LOG_DIR"
    local existing
    existing=$(crontab -l 2>/dev/null | sed "/$CRON_MARKER - START/,/$CRON_MARKER - END/d" || true)
    { echo "$existing"; echo ""; generate_cron_config; } | crontab -
    echo "✅ 已安裝 DCA + 社群發文 cron。目前設定："
    crontab -l 2>/dev/null | sed -n "/$CRON_MARKER - START/,/$CRON_MARKER - END/p"
}

remove_cron() {
    local remaining
    remaining=$(crontab -l 2>/dev/null | sed "/$CRON_MARKER - START/,/$CRON_MARKER - END/d" || true)
    if [ -z "$(echo "$remaining" | tr -d '[:space:]')" ]; then
        crontab -r 2>/dev/null || true
    else
        echo "$remaining" | crontab -
    fi
    echo "✅ 已移除 DCA + 社群發文 cron。"
}

show_cron() {
    if crontab -l 2>/dev/null | grep -q "$CRON_MARKER"; then
        crontab -l 2>/dev/null | sed -n "/$CRON_MARKER - START/,/$CRON_MARKER - END/p"
    else
        echo "⚠️  尚未安裝 DCA + 社群發文 cron。"
    fi
}

case "${1:-}" in
    --install|-i) install_cron ;;
    --remove|-r)  remove_cron ;;
    --show|-s)    show_cron ;;
    --preview|-p|"") echo "（預覽，未安裝。加 --install 才會寫入 crontab）"; echo ""; generate_cron_config ;;
    --help|-h) sed -n '2,32p' "${BASH_SOURCE[0]}" ;;
    *) echo "未知選項：$1（用 --help 看說明）"; exit 1 ;;
esac
