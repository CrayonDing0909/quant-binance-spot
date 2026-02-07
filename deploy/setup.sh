#!/bin/bash
# ============================================================
# QTrade 云端一键部署脚本
#
# 支持: Google Cloud / Oracle Cloud / 任何 Ubuntu 实例
#
# 前置条件:
#   1. SSH 连接到云端实例
#   2. 已上传项目代码（git clone 或 scp）
#
# 使用:
#   chmod +x deploy/setup.sh
#   ./deploy/setup.sh
# ============================================================

set -euo pipefail

echo "============================================"
echo "  QTrade 云端部署脚本"
echo "============================================"

# ── 1. 安装 Docker ──
if ! command -v docker &> /dev/null; then
    echo ""
    echo "📦 安装 Docker..."
    sudo apt-get update
    sudo apt-get install -y ca-certificates curl gnupg
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    sudo usermod -aG docker $USER
    echo "✅ Docker 安装完成"
    echo "⚠️  请运行 'newgrp docker' 或重新登录以使用 docker"
else
    echo "✅ Docker 已安装: $(docker --version)"
fi

# ── 2. 低内存优化（e2-micro 1GB RAM）──
# 创建 swap 文件防止 OOM
if [ ! -f /swapfile ]; then
    echo ""
    echo "💾 创建 2GB swap 文件（防止低内存 OOM）..."
    sudo fallocate -l 2G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
    echo "✅ Swap 已启用"
else
    echo "✅ Swap 已存在"
fi

# ── 3. 检查 .env ──
if [ ! -f .env ]; then
    echo ""
    echo "⚠️  .env 文件不存在，正在创建模板..."
    cp .env.example .env
    echo ""
    echo "请编辑 .env 填入你的配置:"
    echo "  nano .env"
    echo ""
    echo "必填项:"
    echo "  TELEGRAM_BOT_TOKEN=你的Bot Token"
    echo "  TELEGRAM_CHAT_ID=你的Chat ID"
    echo ""
    echo "编辑完成后重新运行此脚本。"
    exit 1
fi
echo "✅ .env 文件已存在"

# ── 4. 构建镜像 ──
echo ""
echo "🔨 构建 Docker 镜像..."
docker compose build

# ── 5. 启动 Paper Trading ──
echo ""
echo "🚀 启动 Paper Trading..."
docker compose up -d paper-trading

# ── 6. 设置 cron 定时报表 ──
echo ""
echo "⏰ 设置每日报表 cron..."
CRON_CMD="5 0 * * * cd $(pwd) && docker compose run --rm daily-report >> /tmp/qtrade-report.log 2>&1"
(crontab -l 2>/dev/null | grep -v "qtrade-report" ; echo "$CRON_CMD") | crontab -
echo "✅ 每日 UTC 00:05 自动发送绩效报表"

# ── 7. 验证 ──
echo ""
sleep 3
echo "============================================"
echo "  ✅ 部署完成！"
echo "============================================"
echo ""
docker compose ps
echo ""
echo "  常用命令:"
echo "  查看日志:     docker compose logs -f paper-trading"
echo "  手动发报表:   docker compose run --rm daily-report"
echo "  停止:         docker compose down"
echo "  重启:         docker compose restart paper-trading"
echo ""
