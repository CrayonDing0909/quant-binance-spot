#!/bin/bash
# Sync trading.db from Oracle Cloud to local for dashboard use.
# Run: bash scripts/sync_oracle_db.sh

ORACLE_HOST="${ORACLE_HOST:-140.83.57.255}"
ORACLE_USER="${ORACLE_USER:-ubuntu}"
ORACLE_KEY="${ORACLE_KEY:-$HOME/.ssh/oracle-trading-bot.key}"
REMOTE_REPO="${REMOTE_REPO:-quant-binance-spot}"   # relative to home, no leading ~/
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)/reports"

echo "Syncing trading DBs from ${ORACLE_USER}@${ORACLE_HOST}:~/${REMOTE_REPO}/reports ..."

rsync -avz --progress \
  -e "ssh -i ${ORACLE_KEY} -o StrictHostKeyChecking=no" \
  --include="*/" \
  --include="trading.db" \
  --exclude="*" \
  "${ORACLE_USER}@${ORACLE_HOST}:${REMOTE_REPO}/reports/" \
  "${LOCAL_DIR}/"

echo "Done. Restart dashboard to pick up new data."
