#!/bin/bash
# 自動建立 Swap (虛擬記憶體) 腳本
# 適用於 Oracle Cloud Ubuntu 22.04 / 24.04

set -e

echo "🔍 檢查系統 Swap 狀態..."
if [ $(swapon --show | wc -l) -gt 0 ]; then
    echo "✅ 系統已有 Swap，跳過配置。"
    free -h
    exit 0
fi

echo "📦 正在建立 2GB Swap (虛擬記憶體)..."
# 使用 fallocate 快速分配
sudo fallocate -l 2G /swapfile
# 設定權限 (重要)
sudo chmod 600 /swapfile
# 格式化為 Swap
sudo mkswap /swapfile
# 啟用 Swap
sudo swapon /swapfile

echo "📝 寫入 fstab 確保重開機後生效..."
# 備份 fstab
sudo cp /etc/fstab /etc/fstab.bak
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

echo "⚙️  優化 Swap 使用傾向 (Swappiness)..."
# 設為 10 代表盡量用實體 RAM，真的不夠才用 Swap (避免拖慢效能)
sudo sysctl vm.swappiness=10
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf

echo "✅ Swap 配置完成！"
echo "------------------------------------------------"
free -h
echo "------------------------------------------------"
