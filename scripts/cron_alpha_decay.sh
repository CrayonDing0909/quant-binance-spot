#!/bin/bash
# Alpha Decay 自動化監控 — 每週 cron 執行
# crontab 設定: 0 0 * * 0  /path/to/scripts/cron_alpha_decay.sh
#
# 輸出 JSON 報告到 reports/alpha_decay/
# 有警報時自動發送 Telegram 通知

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_DIR"
source .venv/bin/activate

PYTHONPATH=src python scripts/monitor_alpha_decay.py \
  -c config/prod_candidate_simplified.yaml \
  --notify --quiet \
  --output-dir reports/alpha_decay
