#!/bin/bash
# ============================================================
# Quant Trading Bot - Cron Jobs 自動設定腳本
# 
# 支援環境：
#   - macOS (本地開發)
#   - Ubuntu/Oracle Cloud (生產環境)
#
# 使用方法：
#   chmod +x scripts/setup_cron.sh
#   ./scripts/setup_cron.sh              # 互動模式
#   ./scripts/setup_cron.sh --install    # 直接安裝
#   ./scripts/setup_cron.sh --remove     # 移除 cron jobs
#   ./scripts/setup_cron.sh --show       # 顯示目前設定
# ============================================================

set -e

# ── 顏色定義 ──────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ── 偵測專案路徑 ──────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ── 偵測作業系統 ──────────────────────────────────────
detect_os() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [ -f /etc/oracle-cloud-agent/agent.yml ] || grep -q "Oracle" /etc/os-release 2>/dev/null; then
            echo "oracle"
        else
            echo "linux"
        fi
    else
        echo "unknown"
    fi
}

OS_TYPE=$(detect_os)

# ── 偵測 Python 環境 ──────────────────────────────────
detect_python() {
    if [ -d "$PROJECT_ROOT/.venv" ]; then
        echo "$PROJECT_ROOT/.venv/bin/python"
    elif command -v python3 &> /dev/null; then
        echo "python3"
    elif command -v python &> /dev/null; then
        echo "python"
    else
        echo ""
    fi
}

PYTHON_CMD=$(detect_python)

# ── 設定變數 ──────────────────────────────────────────
CONFIG_FILE="config/rsi_adx_atr.yaml"
LOG_DIR="$PROJECT_ROOT/logs"
CRON_MARKER="# QUANT-TRADING-BOT"

# ── 函數：列印訊息 ────────────────────────────────────
info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warn() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
}

header() {
    echo ""
    echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
}

# ── 函數：檢查環境 ────────────────────────────────────
check_environment() {
    header "環境檢查"
    
    echo -e "  作業系統: ${GREEN}$OS_TYPE${NC}"
    echo -e "  專案路徑: ${GREEN}$PROJECT_ROOT${NC}"
    
    if [ -z "$PYTHON_CMD" ]; then
        error "找不到 Python！請先安裝 Python 3.9+"
        exit 1
    fi
    echo -e "  Python:   ${GREEN}$PYTHON_CMD${NC}"
    
    if [ ! -f "$PROJECT_ROOT/$CONFIG_FILE" ]; then
        error "找不到配置檔: $CONFIG_FILE"
        exit 1
    fi
    echo -e "  配置檔:   ${GREEN}$CONFIG_FILE${NC}"
    
    if [ ! -f "$PROJECT_ROOT/.env" ]; then
        warn ".env 檔案不存在，Telegram 通知可能無法使用"
    else
        echo -e "  .env:     ${GREEN}已存在${NC}"
    fi
    
    success "環境檢查通過"
}

# ── 函數：建立目錄 ────────────────────────────────────
create_directories() {
    info "建立必要目錄..."
    
    mkdir -p "$LOG_DIR"
    mkdir -p "$PROJECT_ROOT/reports/live"
    
    success "目錄建立完成"
}

# ── 函數：生成 Cron 設定 ──────────────────────────────
generate_cron_config() {
    local activate_cmd=""
    
    if [ -d "$PROJECT_ROOT/.venv" ]; then
        activate_cmd="source $PROJECT_ROOT/.venv/bin/activate && "
    fi
    
    cat << EOF
$CRON_MARKER - START
# ============================================================
# Quant Trading Bot - 自動產生於 $(date '+%Y-%m-%d %H:%M:%S')
# 環境: $OS_TYPE
# 專案: $PROJECT_ROOT
# ============================================================

# 環境變數
SHELL=/bin/bash
PATH=/usr/local/bin:/usr/bin:/bin

# 1. 即時交易 - 每小時執行（1h K線收盤後）
0 * * * * cd $PROJECT_ROOT && ${activate_cmd}python scripts/run_live.py -c $CONFIG_FILE --paper --once >> $LOG_DIR/live.log 2>&1

# 2. 每日報表 - UTC 00:05（台灣時間 08:05）
5 0 * * * cd $PROJECT_ROOT && ${activate_cmd}python scripts/daily_report.py -c $CONFIG_FILE >> $LOG_DIR/daily_report.log 2>&1

# 3. 健康檢查 - 每 30 分鐘
*/30 * * * * cd $PROJECT_ROOT && ${activate_cmd}python scripts/health_check.py -c $CONFIG_FILE --notify >> $LOG_DIR/health.log 2>&1

# 4. 策略驗證 - 每週日 UTC 01:00
0 1 * * 0 cd $PROJECT_ROOT && ${activate_cmd}python scripts/validate.py -c $CONFIG_FILE --quick >> $LOG_DIR/validation.log 2>&1

# 5. Log 清理 - 每週一 04:00（保留 7 天）
0 4 * * 1 find $LOG_DIR -name "*.log" -mtime +7 -delete

$CRON_MARKER - END
EOF
}

# ── 函數：顯示 Cron 設定 ──────────────────────────────
show_cron_config() {
    header "Cron Jobs 設定預覽"
    generate_cron_config
    echo ""
}

# ── 函數：清除 Python 快取 ────────────────────────────
clean_pycache() {
    info "清除 Python 快取（.pyc / __pycache__）..."
    find "$PROJECT_ROOT" -name "*.pyc" -delete 2>/dev/null || true
    find "$PROJECT_ROOT" -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    success "Python 快取已清除"
}

# ── 函數：安裝 Cron Jobs ──────────────────────────────
install_cron() {
    header "安裝 Cron Jobs"
    
    # 清除 .pyc 快取，避免舊版本編譯檔導致新功能不生效
    clean_pycache
    
    # 備份現有 crontab
    local backup_file="/tmp/crontab_backup_$(date +%Y%m%d_%H%M%S)"
    crontab -l > "$backup_file" 2>/dev/null || true
    info "已備份現有 crontab 到: $backup_file"
    
    # 移除舊的 bot cron 設定
    local existing_cron=$(crontab -l 2>/dev/null | grep -v "$CRON_MARKER" | grep -v "^#.*Quant Trading Bot" || true)
    
    # 合併新設定
    local new_cron=$(echo "$existing_cron"; echo ""; generate_cron_config)
    
    # 安裝
    echo "$new_cron" | crontab -
    
    success "Cron Jobs 安裝完成！"
    
    echo ""
    info "已安裝的排程："
    echo "  • 即時交易:   每小時執行"
    echo "  • 每日報表:   每天 UTC 00:05"
    echo "  • 健康檢查:   每 30 分鐘"
    echo "  • 一致性驗證: 每週日 UTC 01:00"
    echo "  • Log 清理:   每週一 04:00"
}

# ── 函數：移除 Cron Jobs ──────────────────────────────
remove_cron() {
    header "移除 Cron Jobs"
    
    # 移除 bot 相關的 cron 設定
    local remaining_cron=$(crontab -l 2>/dev/null | sed "/$CRON_MARKER - START/,/$CRON_MARKER - END/d" || true)
    
    if [ -z "$remaining_cron" ]; then
        crontab -r 2>/dev/null || true
    else
        echo "$remaining_cron" | crontab -
    fi
    
    success "Cron Jobs 已移除"
}

# ── 函數：顯示目前設定 ────────────────────────────────
show_current() {
    header "目前 Cron 設定"
    
    if crontab -l 2>/dev/null | grep -q "$CRON_MARKER"; then
        crontab -l 2>/dev/null | sed -n "/$CRON_MARKER - START/,/$CRON_MARKER - END/p"
    else
        warn "尚未安裝 Quant Trading Bot 的 Cron Jobs"
    fi
}

# ── 函數：測試執行 ────────────────────────────────────
test_run() {
    header "測試執行"
    
    info "測試 run_live.py --once..."
    cd "$PROJECT_ROOT"
    
    if [ -d "$PROJECT_ROOT/.venv" ]; then
        source "$PROJECT_ROOT/.venv/bin/activate"
    fi
    
    python scripts/run_live.py -c "$CONFIG_FILE" --paper --once
    
    success "測試完成"
}

# ── 函數：互動模式 ────────────────────────────────────
interactive_mode() {
    header "Quant Trading Bot - Cron 設定工具"
    
    check_environment
    
    echo ""
    echo "請選擇操作："
    echo "  1) 預覽 Cron 設定"
    echo "  2) 安裝 Cron Jobs"
    echo "  3) 移除 Cron Jobs"
    echo "  4) 顯示目前設定"
    echo "  5) 測試執行"
    echo "  6) 退出"
    echo ""
    
    read -p "請輸入選項 [1-6]: " choice
    
    case $choice in
        1) show_cron_config ;;
        2) 
            create_directories
            install_cron 
            ;;
        3) remove_cron ;;
        4) show_current ;;
        5) test_run ;;
        6) 
            info "再見！"
            exit 0 
            ;;
        *)
            error "無效選項"
            exit 1
            ;;
    esac
}

# ── 主程式 ────────────────────────────────────────────
main() {
    case "${1:-}" in
        --install|-i)
            check_environment
            create_directories
            install_cron
            ;;
        --remove|-r)
            remove_cron
            ;;
        --show|-s)
            show_current
            ;;
        --preview|-p)
            check_environment
            show_cron_config
            ;;
        --update|-u)
            header "更新後清理（git pull 後使用）"
            clean_pycache
            info "建議重新啟動正在運行的 bot 進程"
            ;;
        --test|-t)
            check_environment
            test_run
            ;;
        --help|-h)
            echo "使用方法: $0 [選項]"
            echo ""
            echo "選項:"
            echo "  --install, -i   安裝 Cron Jobs"
            echo "  --remove, -r    移除 Cron Jobs"
            echo "  --show, -s      顯示目前設定"
            echo "  --preview, -p   預覽 Cron 設定"
            echo "  --update, -u    git pull 後清除 .pyc 快取"
            echo "  --test, -t      測試執行"
            echo "  --help, -h      顯示此說明"
            echo ""
            echo "不帶參數執行將進入互動模式"
            ;;
        *)
            interactive_mode
            ;;
    esac
}

main "$@"
