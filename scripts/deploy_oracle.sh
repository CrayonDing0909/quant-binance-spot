#!/bin/bash
# ════════════════════════════════════════════════════════════════════════════
# Oracle Cloud 一鍵部署腳本
# 
# 部署兩套策略：
#   1. NWKL v3.1   → systemd 服務（24/7 每小時監控 ETHUSDT 1H）
#   2. Dual-Momentum → systemd 定時器（每週一 UTC 00:00 = UTC+8 08:00）
#
# 使用方法：
#   # 首次部署（完整安裝）
#   bash scripts/deploy_oracle.sh --install
#
#   # 更新代碼後重新部署
#   bash scripts/deploy_oracle.sh --update
#
#   # 查看所有服務狀態
#   bash scripts/deploy_oracle.sh --status
#
#   # 查看即時日誌
#   bash scripts/deploy_oracle.sh --logs
#
#   # 停止所有服務
#   bash scripts/deploy_oracle.sh --stop
#
#   # 移除所有服務
#   bash scripts/deploy_oracle.sh --remove
#
# 前提條件：
#   - Ubuntu 22.04+ (Oracle Cloud)
#   - Python 3.10+ (系統已安裝)
#   - 專案已 clone 到 ~/quant-binance-spot/
#   - .env 已配置 Telegram 和 Binance API 憑證
#
# Author: Quantitative Research Engineer
# Date:   2026-02-19
# ════════════════════════════════════════════════════════════════════════════

set -e

# ── 顏色 ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ── 路徑 ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_ROOT/.venv"
LOG_DIR="$PROJECT_ROOT/logs"
SYSTEMD_DIR="/etc/systemd/system"

# ── 偵測 Python ──────────────────────────────────────────────────────────
detect_python() {
    if [ -f "$VENV_DIR/bin/python" ]; then
        echo "$VENV_DIR/bin/python"
    elif [ -f "$VENV_DIR/bin/python3" ]; then
        echo "$VENV_DIR/bin/python3"
    elif command -v python3.11 &> /dev/null; then
        echo "python3.11"
    elif command -v python3 &> /dev/null; then
        echo "python3"
    else
        echo ""
    fi
}

# ── 工具函數 ──────────────────────────────────────────────────────────────
info()    { echo -e "${BLUE}ℹ️  $1${NC}"; }
success() { echo -e "${GREEN}✅ $1${NC}"; }
warn()    { echo -e "${YELLOW}⚠️  $1${NC}"; }
error()   { echo -e "${RED}❌ $1${NC}"; }
header()  {
    echo ""
    echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
}

# ═══════════════════════════════════════════════════════════════════════════
# 環境檢查
# ═══════════════════════════════════════════════════════════════════════════
check_environment() {
    header "環境檢查"

    # 作業系統
    if [[ "$OSTYPE" != "linux-gnu"* ]]; then
        error "此腳本僅支援 Linux (Oracle Cloud Ubuntu)"
        exit 1
    fi
    echo -e "  作業系統:   ${GREEN}$(lsb_release -d 2>/dev/null | cut -f2 || echo "Linux")${NC}"
    echo -e "  主機名稱:   ${GREEN}$(hostname)${NC}"
    echo -e "  專案路徑:   ${GREEN}$PROJECT_ROOT${NC}"

    # Python
    PYTHON_CMD=$(detect_python)
    if [ -z "$PYTHON_CMD" ]; then
        error "找不到 Python！請先安裝 Python 3.10+"
        exit 1
    fi
    local py_ver=$($PYTHON_CMD --version 2>&1)
    echo -e "  Python:     ${GREEN}$py_ver ($PYTHON_CMD)${NC}"

    # Venv
    if [ -d "$VENV_DIR" ]; then
        echo -e "  虛擬環境:   ${GREEN}$VENV_DIR${NC}"
    else
        warn "虛擬環境不存在，將自動建立"
    fi

    # .env
    if [ -f "$PROJECT_ROOT/.env" ]; then
        echo -e "  .env:       ${GREEN}已存在${NC}"
        # 檢查關鍵環境變數
        if grep -q "TELEGRAM_BOT_TOKEN" "$PROJECT_ROOT/.env" 2>/dev/null || \
           grep -q "DM_TELEGRAM_BOT_TOKEN" "$PROJECT_ROOT/.env" 2>/dev/null; then
            echo -e "  Telegram:   ${GREEN}已配置${NC}"
        else
            warn "Telegram Token 未在 .env 中設定"
        fi
    else
        warn ".env 不存在 — Telegram 通知將無法使用"
        warn "請建立 .env 並填入以下內容："
        echo "    TELEGRAM_BOT_TOKEN=your_bot_token"
        echo "    TELEGRAM_CHAT_ID=your_chat_id"
        echo "    DM_TELEGRAM_BOT_TOKEN=your_bot_token"
        echo "    DM_TELEGRAM_CHAT_ID=your_chat_id"
    fi

    # 磁碟空間
    local disk_free=$(df -h / | awk 'NR==2{print $4}')
    echo -e "  可用磁碟:   ${GREEN}$disk_free${NC}"

    # 記憶體
    local mem_free=$(free -h | awk '/Mem:/{print $7}')
    echo -e "  可用記憶體: ${GREEN}$mem_free${NC}"

    success "環境檢查通過"
}

# ═══════════════════════════════════════════════════════════════════════════
# Python 環境設置
# ═══════════════════════════════════════════════════════════════════════════
setup_python() {
    header "Python 環境設置"

    # 建立 venv（如果不存在）
    if [ ! -d "$VENV_DIR" ]; then
        info "建立虛擬環境..."
        python3 -m venv "$VENV_DIR"
        success "虛擬環境建立完成"
    fi

    # 啟用 venv
    source "$VENV_DIR/bin/activate"

    # 更新 pip
    info "更新 pip..."
    pip install --upgrade pip -q

    # 安裝依賴
    info "安裝 Python 依賴..."
    pip install -r "$PROJECT_ROOT/requirements.txt" -q 2>&1 | tail -3

    # 確認 yfinance（Dual-Momentum 需要）
    pip install yfinance -q 2>&1 | tail -1

    # 安裝專案本身
    pip install -e "$PROJECT_ROOT" -q 2>&1 | tail -1

    success "Python 環境就緒"

    # 更新 PYTHON_CMD
    PYTHON_CMD="$VENV_DIR/bin/python"
}

# ═══════════════════════════════════════════════════════════════════════════
# 建立目錄
# ═══════════════════════════════════════════════════════════════════════════
create_directories() {
    info "建立必要目錄..."
    mkdir -p "$LOG_DIR"
    mkdir -p "$PROJECT_ROOT/reports/dual_momentum"
    mkdir -p "$PROJECT_ROOT/reports/futures/nwkl/live"
    mkdir -p "$PROJECT_ROOT/data"
    success "目錄就緒"
}

# ═══════════════════════════════════════════════════════════════════════════
# systemd 服務：NWKL v3.1（24/7 每小時監控）
# ═══════════════════════════════════════════════════════════════════════════
install_nwkl_service() {
    header "安裝 NWKL v3.1 服務"

    local SERVICE_FILE="$SYSTEMD_DIR/qtrade-nwkl.service"

    info "建立 systemd 服務: qtrade-nwkl.service"

    sudo tee "$SERVICE_FILE" > /dev/null << EOF
[Unit]
Description=QTrade NWKL v3.1 - Nadaraya-Watson Kernel Regression Strategy
Documentation=https://github.com/your-repo/quant-binance-spot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$(whoami)
Group=$(id -gn)
WorkingDirectory=$PROJECT_ROOT
Environment=PYTHONPATH=$PROJECT_ROOT/src
EnvironmentFile=$PROJECT_ROOT/.env

# ── 啟動命令 ──
# Paper Trading（安全模式，先觀察信號）
ExecStart=$VENV_DIR/bin/python scripts/run_live.py -c config/futures_nwkl.yaml --paper --telegram-commands

# ── 正式交易（取消下面的註解，並註解上面的 Paper 模式）──
# ExecStart=$VENV_DIR/bin/python scripts/run_live.py -c config/futures_nwkl.yaml --real --telegram-commands

# ── 重啟策略 ──
Restart=always
RestartSec=30
StartLimitIntervalSec=600
StartLimitBurst=5

# ── 記憶體限制（防止 OOM） ──
MemoryMax=1G
MemoryHigh=768M

# ── 日誌 ──
StandardOutput=append:$LOG_DIR/nwkl.log
StandardError=append:$LOG_DIR/nwkl-error.log

# ── 安全強化 ──
NoNewPrivileges=true
ProtectSystem=strict
ReadWritePaths=$PROJECT_ROOT
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    sudo systemctl enable qtrade-nwkl.service

    success "NWKL 服務已安裝（已啟用開機自啟）"
    info "模式: Paper Trading（安全模式）"
    info "切換正式交易: sudo nano $SERVICE_FILE"
}

# ═══════════════════════════════════════════════════════════════════════════
# systemd 定時器：Dual-Momentum（每週一 UTC 00:00）
# ═══════════════════════════════════════════════════════════════════════════
install_dm_timer() {
    header "安裝 Dual-Momentum 每週定時器"

    # ── Service（執行一次就結束） ──
    local SERVICE_FILE="$SYSTEMD_DIR/qtrade-dm-weekly.service"

    info "建立 systemd 服務: qtrade-dm-weekly.service"

    sudo tee "$SERVICE_FILE" > /dev/null << EOF
[Unit]
Description=QTrade Dual-Momentum - Weekly Signal Generator
Documentation=https://github.com/your-repo/quant-binance-spot
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User=$(whoami)
Group=$(id -gn)
WorkingDirectory=$PROJECT_ROOT
Environment=PYTHONPATH=$PROJECT_ROOT/src
EnvironmentFile=$PROJECT_ROOT/.env

ExecStart=$VENV_DIR/bin/python scripts/cron_dual_momentum.py

# ── 超時（防止 yfinance 卡住） ──
TimeoutStartSec=300

# ── 日誌 ──
StandardOutput=append:$LOG_DIR/dual_momentum.log
StandardError=append:$LOG_DIR/dual_momentum-error.log

# ── 安全 ──
NoNewPrivileges=true
ProtectSystem=strict
ReadWritePaths=$PROJECT_ROOT
PrivateTmp=true
EOF

    # ── Timer（每週一 UTC 00:00 = UTC+8 08:00） ──
    local TIMER_FILE="$SYSTEMD_DIR/qtrade-dm-weekly.timer"

    info "建立 systemd 定時器: qtrade-dm-weekly.timer"

    sudo tee "$TIMER_FILE" > /dev/null << EOF
[Unit]
Description=QTrade Dual-Momentum - Weekly Timer (Mon 00:00 UTC)

[Timer]
# 每週一 UTC 00:00（= 台灣時間 08:00）
OnCalendar=Mon *-*-* 00:00:00
# 如果錯過（例如機器關機），啟動後補執行
Persistent=true
# 隨機延遲 0~5 分鐘（避免 API 擁擠）
RandomizedDelaySec=300

[Install]
WantedBy=timers.target
EOF

    sudo systemctl daemon-reload
    sudo systemctl enable qtrade-dm-weekly.timer

    success "Dual-Momentum 定時器已安裝"
    info "排程: 每週一 UTC 00:00 (UTC+8 08:00)"
}

# ═══════════════════════════════════════════════════════════════════════════
# Log Rotation（日誌輪替）
# ═══════════════════════════════════════════════════════════════════════════
install_logrotate() {
    header "配置日誌輪替"

    sudo tee /etc/logrotate.d/qtrade > /dev/null << EOF
$LOG_DIR/*.log {
    daily
    rotate 14
    compress
    delaycompress
    missingok
    notifempty
    create 0644 $(whoami) $(id -gn)
    dateext
    dateformat -%Y%m%d
}
EOF

    success "日誌輪替已配置（保留 14 天）"
}

# ═══════════════════════════════════════════════════════════════════════════
# 健康檢查 Cron（每 6 小時）
# ═══════════════════════════════════════════════════════════════════════════
install_health_cron() {
    header "配置健康檢查"

    # 建立一個簡單的健康檢查腳本
    cat > "$PROJECT_ROOT/scripts/_health_watchdog.sh" << 'WATCHDOG_EOF'
#!/bin/bash
# NWKL 服務健康檢查
PROJECT_ROOT="$(dirname "$(dirname "$(readlink -f "$0")")")"
LOG="$PROJECT_ROOT/logs/watchdog.log"

echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] Health check started" >> "$LOG"

# 檢查 NWKL 服務狀態
if systemctl is-active --quiet qtrade-nwkl.service; then
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] ✅ NWKL service: running" >> "$LOG"
else
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] ❌ NWKL service: stopped — restarting..." >> "$LOG"
    sudo systemctl restart qtrade-nwkl.service
    sleep 5
    if systemctl is-active --quiet qtrade-nwkl.service; then
        echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] ✅ NWKL service: restarted successfully" >> "$LOG"
    else
        echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] ❌ NWKL service: restart FAILED" >> "$LOG"
    fi
fi

# 檢查 DM timer
if systemctl is-active --quiet qtrade-dm-weekly.timer; then
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] ✅ DM timer: active" >> "$LOG"
else
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] ⚠️  DM timer: inactive" >> "$LOG"
fi

# 檢查磁碟空間
DISK_PCT=$(df / | awk 'NR==2{print int($5)}')
if [ "$DISK_PCT" -gt 90 ]; then
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] ⚠️  Disk usage: ${DISK_PCT}% — consider cleanup" >> "$LOG"
fi
WATCHDOG_EOF

    chmod +x "$PROJECT_ROOT/scripts/_health_watchdog.sh"

    # 安裝 cron（每 6 小時 + 開機後 5 分鐘）
    local cron_line="0 */6 * * * $PROJECT_ROOT/scripts/_health_watchdog.sh"
    local existing=$(crontab -l 2>/dev/null || true)

    if echo "$existing" | grep -q "_health_watchdog.sh"; then
        info "健康檢查 cron 已存在，跳過"
    else
        (echo "$existing"; echo ""; echo "# QTrade 健康檢查 - 每 6 小時"; echo "$cron_line") | crontab -
        success "健康檢查 cron 已安裝（每 6 小時）"
    fi
}

# ═══════════════════════════════════════════════════════════════════════════
# 完整安裝
# ═══════════════════════════════════════════════════════════════════════════
do_install() {
    header "🚀 Oracle Cloud 完整部署"
    echo -e "  ${BOLD}策略 1: NWKL v3.1${NC}  (ETHUSDT 1H, 24/7 Paper Trading)"
    echo -e "  ${BOLD}策略 2: Dual-Momentum${NC} (BTC/ETH/SOL/BNB, 每週一)"
    echo ""

    check_environment
    setup_python
    create_directories
    install_nwkl_service
    install_dm_timer
    install_logrotate
    install_health_cron

    header "🎉 部署完成！"
    echo ""
    echo -e "  ${BOLD}服務控制：${NC}"
    echo ""
    echo -e "  ${GREEN}# ── NWKL（24/7 策略）──${NC}"
    echo "  sudo systemctl start qtrade-nwkl       # 啟動"
    echo "  sudo systemctl stop qtrade-nwkl        # 停止"
    echo "  sudo systemctl restart qtrade-nwkl     # 重啟"
    echo "  sudo journalctl -u qtrade-nwkl -f      # 即時日誌"
    echo ""
    echo -e "  ${GREEN}# ── Dual-Momentum（每週定時器）──${NC}"
    echo "  sudo systemctl start qtrade-dm-weekly.timer   # 啟動定時器"
    echo "  sudo systemctl start qtrade-dm-weekly.service # 手動觸發一次"
    echo "  systemctl list-timers qtrade-dm*              # 查看下次執行時間"
    echo ""
    echo -e "  ${GREEN}# ── 一鍵操作 ──${NC}"
    echo "  bash scripts/deploy_oracle.sh --status  # 查看所有狀態"
    echo "  bash scripts/deploy_oracle.sh --logs    # 即時日誌"
    echo "  bash scripts/deploy_oracle.sh --stop    # 停止所有"
    echo ""
    echo -e "  ${YELLOW}⚠️  重要：${NC}"
    echo "  1. NWKL 預設為 Paper Trading 模式"
    echo "     觀察 1-2 週確認信號正確後，再切換正式交易"
    echo "  2. 切換正式交易："
    echo "     sudo nano /etc/systemd/system/qtrade-nwkl.service"
    echo "     取消 --real 那行的註解，註解 --paper 那行"
    echo "     sudo systemctl daemon-reload && sudo systemctl restart qtrade-nwkl"
    echo ""

    # 提示啟動
    echo -e "  ${BOLD}現在啟動服務？${NC}"
    echo "  sudo systemctl start qtrade-nwkl"
    echo "  sudo systemctl start qtrade-dm-weekly.timer"
    echo ""
}

# ═══════════════════════════════════════════════════════════════════════════
# 更新代碼後重新部署
# ═══════════════════════════════════════════════════════════════════════════
do_update() {
    header "🔄 更新部署"

    info "拉取最新代碼..."
    cd "$PROJECT_ROOT"
    git pull

    info "清除 Python 快取..."
    find "$PROJECT_ROOT" -name "*.pyc" -delete 2>/dev/null || true
    find "$PROJECT_ROOT" -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

    info "更新 Python 依賴..."
    source "$VENV_DIR/bin/activate"
    pip install -r "$PROJECT_ROOT/requirements.txt" -q 2>&1 | tail -3
    pip install -e "$PROJECT_ROOT" -q 2>&1 | tail -1

    info "重新載入 systemd..."
    sudo systemctl daemon-reload

    info "重啟服務..."
    if systemctl is-active --quiet qtrade-nwkl.service; then
        sudo systemctl restart qtrade-nwkl.service
        success "NWKL 服務已重啟"
    else
        info "NWKL 服務未運行，跳過重啟"
    fi

    success "更新完成！"
}

# ═══════════════════════════════════════════════════════════════════════════
# 狀態面板
# ═══════════════════════════════════════════════════════════════════════════
do_status() {
    header "📊 服務狀態面板"

    echo ""
    echo -e "  ${BOLD}━━━ NWKL v3.1 (24/7) ━━━${NC}"
    if systemctl is-active --quiet qtrade-nwkl.service 2>/dev/null; then
        echo -e "  狀態:   ${GREEN}🟢 運行中${NC}"
        local pid=$(systemctl show qtrade-nwkl.service --property MainPID --value 2>/dev/null)
        local mem=$(systemctl show qtrade-nwkl.service --property MemoryCurrent --value 2>/dev/null)
        local uptime=$(systemctl show qtrade-nwkl.service --property ActiveEnterTimestamp --value 2>/dev/null)
        [ -n "$pid" ] && [ "$pid" != "0" ] && echo "  PID:    $pid"
        [ -n "$mem" ] && [ "$mem" != "[not set]" ] && echo "  記憶體: $(numfmt --to=iec $mem 2>/dev/null || echo $mem)"
        [ -n "$uptime" ] && echo "  啟動:   $uptime"
    elif systemctl is-enabled --quiet qtrade-nwkl.service 2>/dev/null; then
        echo -e "  狀態:   ${YELLOW}🟡 已安裝但未啟動${NC}"
    else
        echo -e "  狀態:   ${RED}🔴 未安裝${NC}"
    fi

    echo ""
    echo -e "  ${BOLD}━━━ Dual-Momentum (Weekly) ━━━${NC}"
    if systemctl is-active --quiet qtrade-dm-weekly.timer 2>/dev/null; then
        echo -e "  狀態:   ${GREEN}🟢 定時器運行中${NC}"
        local next_run=$(systemctl show qtrade-dm-weekly.timer --property NextElapseUSecRealtime --value 2>/dev/null)
        [ -n "$next_run" ] && echo "  下次:   $next_run"
        local last_run=$(systemctl show qtrade-dm-weekly.timer --property LastTriggerUSec --value 2>/dev/null)
        [ -n "$last_run" ] && [ "$last_run" != "n/a" ] && echo "  上次:   $last_run"
    elif systemctl is-enabled --quiet qtrade-dm-weekly.timer 2>/dev/null; then
        echo -e "  狀態:   ${YELLOW}🟡 已安裝但未啟動${NC}"
    else
        echo -e "  狀態:   ${RED}🔴 未安裝${NC}"
    fi

    echo ""
    echo -e "  ${BOLD}━━━ 系統資源 ━━━${NC}"
    echo "  CPU 負載:   $(uptime | awk -F'average:' '{print $2}' | xargs)"
    echo "  記憶體:     $(free -h | awk '/Mem:/{printf "%s / %s (%s available)", $3, $2, $7}')"
    echo "  磁碟:       $(df -h / | awk 'NR==2{printf "%s / %s (%s available)", $3, $2, $4}')"

    # 最近的日誌
    echo ""
    echo -e "  ${BOLD}━━━ 最近日誌 ━━━${NC}"

    if [ -f "$LOG_DIR/nwkl.log" ]; then
        echo -e "  ${CYAN}[NWKL 最後 3 行]${NC}"
        tail -3 "$LOG_DIR/nwkl.log" 2>/dev/null | sed 's/^/    /'
    fi

    if [ -f "$LOG_DIR/dual_momentum.log" ]; then
        echo -e "  ${CYAN}[DM 最後 3 行]${NC}"
        tail -3 "$LOG_DIR/dual_momentum.log" 2>/dev/null | sed 's/^/    /'
    fi

    # 信號歷史
    local sig_file="$PROJECT_ROOT/reports/dual_momentum/signal_history.csv"
    if [ -f "$sig_file" ]; then
        echo ""
        echo -e "  ${BOLD}━━━ Dual-Momentum 信號歷史 ━━━${NC}"
        tail -5 "$sig_file" | column -t -s',' | sed 's/^/    /'
    fi

    echo ""
}

# ═══════════════════════════════════════════════════════════════════════════
# 即時日誌
# ═══════════════════════════════════════════════════════════════════════════
do_logs() {
    header "📋 即時日誌（Ctrl+C 退出）"

    echo "選擇要查看的日誌："
    echo "  1) NWKL 服務日誌"
    echo "  2) Dual-Momentum 日誌"
    echo "  3) 全部（journalctl）"
    echo "  4) Watchdog 健康檢查"
    echo ""

    read -p "選擇 [1-4]: " choice

    case $choice in
        1)
            if [ -f "$LOG_DIR/nwkl.log" ]; then
                tail -f "$LOG_DIR/nwkl.log"
            else
                sudo journalctl -u qtrade-nwkl -f
            fi
            ;;
        2)
            if [ -f "$LOG_DIR/dual_momentum.log" ]; then
                tail -f "$LOG_DIR/dual_momentum.log"
            else
                sudo journalctl -u qtrade-dm-weekly -f
            fi
            ;;
        3) sudo journalctl -u "qtrade-*" -f ;;
        4) tail -f "$LOG_DIR/watchdog.log" ;;
        *) error "無效選項" ;;
    esac
}

# ═══════════════════════════════════════════════════════════════════════════
# 停止所有服務
# ═══════════════════════════════════════════════════════════════════════════
do_stop() {
    header "⏹️  停止所有服務"

    if systemctl is-active --quiet qtrade-nwkl.service 2>/dev/null; then
        sudo systemctl stop qtrade-nwkl.service
        success "NWKL 服務已停止"
    else
        info "NWKL 服務未運行"
    fi

    if systemctl is-active --quiet qtrade-dm-weekly.timer 2>/dev/null; then
        sudo systemctl stop qtrade-dm-weekly.timer
        success "DM 定時器已停止"
    else
        info "DM 定時器未運行"
    fi

    success "所有服務已停止"
}

# ═══════════════════════════════════════════════════════════════════════════
# 啟動所有服務
# ═══════════════════════════════════════════════════════════════════════════
do_start() {
    header "▶️  啟動所有服務"

    sudo systemctl start qtrade-nwkl.service
    sleep 2
    if systemctl is-active --quiet qtrade-nwkl.service; then
        success "NWKL 服務已啟動"
    else
        error "NWKL 服務啟動失敗"
        sudo journalctl -u qtrade-nwkl --no-pager -n 10
    fi

    sudo systemctl start qtrade-dm-weekly.timer
    if systemctl is-active --quiet qtrade-dm-weekly.timer; then
        success "DM 定時器已啟動"
        echo "  下次執行: $(systemctl show qtrade-dm-weekly.timer --property NextElapseUSecRealtime --value 2>/dev/null)"
    else
        error "DM 定時器啟動失敗"
    fi

    success "所有服務已啟動"
}

# ═══════════════════════════════════════════════════════════════════════════
# 移除所有服務
# ═══════════════════════════════════════════════════════════════════════════
do_remove() {
    header "🗑️  移除所有服務"

    warn "這將移除所有 QTrade systemd 服務和定時器"
    read -p "確定嗎？(y/N): " confirm
    if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
        info "已取消"
        return
    fi

    # 停止
    sudo systemctl stop qtrade-nwkl.service 2>/dev/null || true
    sudo systemctl stop qtrade-dm-weekly.timer 2>/dev/null || true
    sudo systemctl stop qtrade-dm-weekly.service 2>/dev/null || true

    # 停用
    sudo systemctl disable qtrade-nwkl.service 2>/dev/null || true
    sudo systemctl disable qtrade-dm-weekly.timer 2>/dev/null || true

    # 刪除
    sudo rm -f "$SYSTEMD_DIR/qtrade-nwkl.service"
    sudo rm -f "$SYSTEMD_DIR/qtrade-dm-weekly.service"
    sudo rm -f "$SYSTEMD_DIR/qtrade-dm-weekly.timer"
    sudo rm -f /etc/logrotate.d/qtrade

    sudo systemctl daemon-reload

    # 移除 cron
    local existing=$(crontab -l 2>/dev/null | grep -v "_health_watchdog.sh" | grep -v "^$" || true)
    echo "$existing" | crontab -

    success "所有 QTrade 服務已移除"
}

# ═══════════════════════════════════════════════════════════════════════════
# 手動觸發 Dual-Momentum
# ═══════════════════════════════════════════════════════════════════════════
do_trigger_dm() {
    header "🔄 手動觸發 Dual-Momentum 信號"

    if systemctl is-enabled --quiet qtrade-dm-weekly.service 2>/dev/null; then
        sudo systemctl start qtrade-dm-weekly.service
        info "已觸發，查看日誌..."
        sleep 10
        sudo journalctl -u qtrade-dm-weekly --no-pager -n 30
    else
        info "服務未安裝，直接執行腳本..."
        cd "$PROJECT_ROOT"
        source "$VENV_DIR/bin/activate"
        PYTHONPATH=src python scripts/cron_dual_momentum.py
    fi
}

# ═══════════════════════════════════════════════════════════════════════════
# 主程式
# ═══════════════════════════════════════════════════════════════════════════
main() {
    case "${1:-}" in
        --install|-i)
            do_install
            ;;
        --update|-u)
            do_update
            ;;
        --status|-s)
            do_status
            ;;
        --start)
            do_start
            ;;
        --stop)
            do_stop
            ;;
        --logs|-l)
            do_logs
            ;;
        --remove|-r)
            do_remove
            ;;
        --trigger-dm)
            do_trigger_dm
            ;;
        --help|-h)
            echo ""
            echo "Oracle Cloud 部署腳本 — QTrade 策略系統"
            echo ""
            echo "使用方法: bash scripts/deploy_oracle.sh [選項]"
            echo ""
            echo "選項:"
            echo "  --install, -i    首次完整安裝"
            echo "  --update,  -u    git pull 後更新部署"
            echo "  --status,  -s    查看服務狀態面板"
            echo "  --start          啟動所有服務"
            echo "  --stop           停止所有服務"
            echo "  --logs,    -l    查看即時日誌"
            echo "  --remove,  -r    移除所有服務"
            echo "  --trigger-dm     手動觸發 Dual-Momentum"
            echo "  --help,    -h    顯示此說明"
            echo ""
            echo "部署架構:"
            echo "  ┌─ qtrade-nwkl.service ────────────────────┐"
            echo "  │  NWKL v3.1 ETHUSDT 1H                   │"
            echo "  │  24/7 Paper Trading → Telegram 通知       │"
            echo "  │  systemd 自動重啟 + 記憶體限制             │"
            echo "  └──────────────────────────────────────────┘"
            echo "  ┌─ qtrade-dm-weekly.timer ──────────────────┐"
            echo "  │  Dual-Momentum BTC/ETH/SOL/BNB           │"
            echo "  │  每週一 UTC 00:00 (UTC+8 08:00)          │"
            echo "  │  生成信號 → Telegram → 手動調倉            │"
            echo "  └──────────────────────────────────────────┘"
            echo ""
            ;;
        *)
            echo ""
            echo -e "${BOLD}Oracle Cloud 部署腳本 — QTrade${NC}"
            echo ""
            echo "快速開始:"
            echo "  bash scripts/deploy_oracle.sh --install   # 首次安裝"
            echo "  bash scripts/deploy_oracle.sh --status    # 查看狀態"
            echo "  bash scripts/deploy_oracle.sh --help      # 完整說明"
            echo ""
            ;;
    esac
}

main "$@"
