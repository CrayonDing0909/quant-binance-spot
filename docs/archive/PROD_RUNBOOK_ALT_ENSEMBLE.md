# Production Runbook — Alt Ensemble (ETH+SOL TSMOM)

> **Status**: Conditional Go (2026-02-19 audit)
> **Prerequisite**: 30 天 paper trade 通過後方可進 real

---

## 1. 策略概述

| 欄位 | 值 |
|------|---|
| 組合名稱 | Alt Ensemble (ETH+SOL TSMOM, debounce ON) |
| ETHUSDT | `tsmom_multi_ema` (lookbacks=[72,168,336,720], disagree_weight=0.0) |
| SOLUSDT | `tsmom_ema` (lookback=168, disagree_weight=0.0) |
| BTCUSDT | Cash (零曝險) |
| 權重 | ETH 54% / SOL 46% |
| 市場 | Futures, 1h bar, ISOLATED margin |
| 執行 | trade_on=next_open, signal_delay=1 |
| 成本 | fee=5bps, slippage=3bps, funding_rate enabled |
| 風控配置 | `config/risk_guard_alt_ensemble.yaml` |
| 策略配置 | `config/futures_alt_ensemble.yaml` |

### 驗收指標（Audit 2026-02-19）

| 指標 | 值 | 通過 |
|------|---|-----|
| Nested WF avg OOS Sharpe | 0.701 | ❌ (< 0.8) |
| Final Holdout Sharpe (12m) | 1.116 | ✅ |
| Final Holdout MDD | 7.6% | ✅ |
| 1.5x cost Sharpe | 0.632 | ✅ |
| Positive sleeves (holdout) | 2/2 | ✅ |

---

## 2. Daily Checklist（每日 08:00 UTC 執行）

執行命令：
```bash
python scripts/risk_guard.py -c config/risk_guard_alt_ensemble.yaml --dry-run
```

### 2.1 組合層指標

| # | 指標 | 資料來源 | 計算方式 | 正常區間 | 異常動作 |
|---|------|---------|---------|---------|---------|
| 1 | **Daily PnL (%)** | SQLite `daily_equity` / paper state | `(equity_t - equity_t-1) / equity_t-1 × 100` | [-3%, +3%] | 超過 ±5% → 人工審查 |
| 2 | **20D Rolling Sharpe** | equity curve | `√8760 × mean(20d_ret) / std(20d_ret)` | [0, +3] | < 0 連續 3 天 → WARNING; < -0.5 → FLATTEN |
| 3 | **Running MDD (%)** | equity curve | `(peak - current) / peak × 100` | [0%, 8%] | > 12% → FLATTEN_ALL |
| 4 | **30D Return (%)** | equity curve | `(eq_now / eq_30d_ago - 1) × 100` | [-5%, +10%] | < -10% → FLATTEN_ALL |
| 5 | **Daily Turnover (%)** | trade fills | `Σ|ΔPosition| / Σ|Weight| × 100` | [0%, 15%] | > 30% → 檢查信號異常 |
| 6 | **Realized Slippage (bps)** | fill price vs signal price | `(fill - signal) / signal × 10000` | [0, 3.5] | > 4.5 連續 5 天 → REDUCE_RISK_50 |
| 7 | **Funding Drag (bps/day)** | funding settlements | `Σfunding_cost / equity × 10000 / days` | [0, 0.8] | 年化 > 2% → WARNING |
| 8 | **Signal Consistency** | paper signal vs backtest replay | `mismatch_count / total_signals × 100` | [0%, 2%] | > 5% → 停止 paper, 審查 |

### 2.2 Sleeve 層指標

| # | 指標 | ETHUSDT 正常區間 | SOLUSDT 正常區間 | 異常動作 |
|---|------|----------------|----------------|---------|
| 9 | **Sleeve MDD (%)** | [0%, 12%] | [0%, 12%] | > 15% → DISABLE_SLEEVE |
| 10 | **Sleeve Contribution** | ETH ≥ -2% (30d) | SOL ≥ -2% (30d) | 單 sleeve 拖累 > 5% → 人工評估 |
| 11 | **Position State** | ∈ {-1, 0, +1} | ∈ {-1, 0, +1} | 不在範圍 → 立即檢查 broker |
| 12 | **Trade Count (24h)** | [0, 3] | [0, 3] | > 5 → 信號過度震盪, 檢查 |

---

## 3. Weekly Checklist（每週一 00:00 UTC 執行）

### 3.1 權重偏離檢查

```
目標權重: ETH 54% / SOL 46%
偏離容忍度: ±5%
再平衡觸發: 任一 sleeve 實際權重偏離 > 5%
```

**再平衡流程**：
1. 計算當前各 sleeve 市值佔比
2. 若偏離 > 5%：
   a. 計算需要調整的目標 delta
   b. 使用 limit order（timeout 10s → fallback market）
   c. 記錄再平衡成本到日誌
3. 若偏離 ≤ 5%：不操作，記錄 "SKIP"

### 3.2 模型健康度回顧

| 項目 | 方法 | 閾值 |
|------|-----|-----|
| 7D Rolling Sharpe | `√8760 × mean(7d_ret) / std(7d_ret)` | > -1.0 |
| Drawdown State | 當前 DD 是否在收窄 | 持續加深 > 7 天 → WARNING |
| Cost Drift | 實際成本 vs 回測假設 | 差異 < 30% |
| Funding Rate Trend | 近 7 天平均 funding rate | 年化 < 3% |

### 3.3 週報模板

```
=== Alt Ensemble Weekly Report ===
Week: YYYY-MM-DD → YYYY-MM-DD
PnL: +X.XX% (YTD: +X.XX%)
Sharpe (20D): X.XX
MDD: X.XX%
Trades: N (ETH: M, SOL: K)
Turnover: X.X%
Costs: fee X.XX + slip X.XX + funding X.XX = total X.XX bps/trade
Weight Drift: ETH X.X% / SOL X.X% (rebalanced: Y/N)
Kill Switch: NO_ACTION / WARNING
```

---

## 4. Incident Playbook（異常處置）

### 4.1 嚴重度分級

| Level | 條件 | 自動動作 | 人工介入 |
|-------|------|---------|---------|
| **CRITICAL** | MDD > 12% 或 SR < -0.5 或 30D < -10% | `FLATTEN_ALL` | 立即通知 → 暫停策略 → 審查 |
| **HIGH** | Slippage > 1.5x 連續 5D 或 sleeve MDD > 15% | `REDUCE_RISK_50` / `DISABLE_SLEEVE` | 4h 內人工確認 |
| **MEDIUM** | SR < 0 連續 3D 或 funding > 2% | `WARNING` (Telegram) | 24h 內人工審查 |
| **LOW** | 指標接近閾值但未超 | 記錄日誌 | 週報中提及 |

### 4.2 FLATTEN_ALL 處置流程

```
1. [自動] Risk Guard 發出 FLATTEN_ALL 建議
2. [自動] Telegram 通知到群組
3. [人工] 30 分鐘內確認（若無人確認則自動執行 — 僅 real mode）
4. [執行] 對所有 sleeve 發出 target_pct=0 指令
5. [執行] 取消所有掛單（SL/TP）
6. [記錄] 寫入 incident log：
   - 觸發時間、規則、指標值
   - 平倉成交紀錄
   - 當時市場狀態
7. [回顧] 24h 內完成 post-mortem：
   - 是否誤觸發？
   - 策略是否結構性失效？
   - 恢復條件
```

### 4.3 REDUCE_RISK_50 處置流程

```
1. [自動] 發出建議
2. [人工] 4h 內確認
3. [執行] 將所有 sleeve 的 position_pct 降至 0.5
4. [監控] 24h 後重評估：
   - 若指標恢復 → 逐步恢復至 1.0
   - 若持續惡化 → 升級為 FLATTEN_ALL
```

### 4.4 DISABLE_SLEEVE 處置流程

```
1. [自動] 發出建議（指定 symbol）
2. [人工] 4h 內確認
3. [執行] 關閉指定 sleeve（target_pct=0）
4. [執行] 將權重分配給剩餘 sleeve（或保持 cash）
5. [監控] 7 天後重評估：
   - 若該 sleeve 單獨回測近 30 天 Sharpe > 0 → 考慮恢復
   - 否則 → 維持關閉
```

### 4.5 回滾條件

策略重啟的前提條件（**全部**滿足）：
1. 觸發原因已理解並有明確解釋
2. 20D rolling Sharpe 回到 > 0
3. 回測近 60 天 Sharpe > 0.3
4. MDD 收窄至 < 8%
5. 人工審查簽核

---

## 5. 30 天 Paper Gate

### 5.1 啟動流程

```bash
# 1. 啟動 paper trade
python scripts/run_live.py -c config/futures_ensemble_nw_tsmom.yaml --paper

# 2. 每日跑 risk guard
python scripts/risk_guard.py -c config/risk_guard_alt_ensemble.yaml --dry-run

# 3. 每日記錄到 spreadsheet/log（自動化推薦）
```

### 5.2 PASS 標準

| 指標 | 通過條件 | 如何計算 |
|------|---------|---------|
| 30D Sharpe | > -0.3 | `√252 × mean(daily_ret) / std(daily_ret)` |
| Max MDD | < 12% | Running MDD over 30 days |
| Slippage Ratio | ≤ 1.3× 回測假設 | `avg_realized_slip / 3bps` |
| Funding Ratio | ≤ 1.5× 回測假設 | `avg_funding_rate / 0.01%` |
| Signal Consistency | ≥ 98% | Paper 信號 vs backtest replay 一致性 |
| Kill Switch Fires | ≤ 1 次 WARNING | 30 天內不應有 CRITICAL 觸發 |

### 5.3 FAIL 處置

若未通過 Paper Gate：
1. **停止 paper trade**
2. **診斷原因**：
   - Slippage 過高 → 改用 limit order / 降低 participation rate
   - Signal 不一致 → 檢查 kline cache / 數據源穩定性
   - Sharpe 太低 → 回到策略研究（重新評估 sleeve 組成或 debounce 參數）
3. **修復後重跑 30 天 paper**（不可縮短）

---

## 6. 基礎設施要求

### 6.1 運行環境

| 項目 | 要求 |
|------|-----|
| Python | 3.11+ |
| 系統 | Oracle Cloud / 任意 VPS |
| 記憶體 | ≥ 1GB（含 kline cache） |
| 磁碟 | ≥ 500MB（SQLite + logs） |
| 網路 | 穩定連接 Binance API |
| 執行方式 | tmux session / systemd |

### 6.2 Telegram 通知

必須配置以下環境變數：
```bash
export FUTURES_TELEGRAM_BOT_TOKEN="..."
export FUTURES_TELEGRAM_CHAT_ID="..."
```

### 6.3 監控 Cron

```bash
# 每小時跑 risk guard（在 crontab 中）
0 * * * * cd /path/to/quant-binance-spot && python scripts/risk_guard.py -c config/risk_guard_alt_ensemble.yaml --dry-run >> logs/risk_guard.log 2>&1
```

---

## 7. 緊急聯絡與升級路徑

| 情境 | 動作 | 時限 |
|------|-----|-----|
| CRITICAL 觸發 | Telegram 通知 + 自動平倉 | 即時 |
| HIGH 觸發 | Telegram 通知 | 4h 內人工確認 |
| MEDIUM 觸發 | Telegram 通知 | 24h 內人工審查 |
| 系統 crash / 斷線 | systemd 自動重啟 + Telegram | 5 分鐘內 |
| API 連線失敗 | 重試 3 次 → Telegram 警報 | 15 分鐘內 |

---

## 附錄 A：常用命令

```bash
# 啟動 paper trade（WebSocket 模式，推薦）
python scripts/run_websocket.py -c config/futures_alt_ensemble.yaml --paper

# 啟動 paper trade（Polling 模式）
python scripts/run_live.py -c config/futures_alt_ensemble.yaml --paper

# 每日 risk guard check
python scripts/risk_guard.py -c config/risk_guard_alt_ensemble.yaml --dry-run

# Replay 壓力測試
python scripts/risk_guard.py -c config/risk_guard_alt_ensemble.yaml --replay

# 自訂 replay 區間
python scripts/risk_guard.py -c config/risk_guard_alt_ensemble.yaml --replay \
    --replay-start 2022-07-01 --replay-end 2023-01-01

# 單幣回測驗證
python scripts/run_backtest.py -c config/futures_alt_ensemble.yaml --symbol ETHUSDT
python scripts/run_backtest.py -c config/futures_alt_ensemble.yaml --symbol SOLUSDT

# 組合回測
python scripts/run_portfolio_backtest.py -c config/futures_alt_ensemble.yaml

# 全測試
python -m pytest tests/ -x -q --tb=short
```

---

## 附錄 B：版本歷史

| 日期 | 版本 | 變更 |
|------|-----|-----|
| 2026-02-19 | v1.0 | 初版 — Conditional Go audit 完成 |
