# Cursor Agent 工作流指南

> **Last updated**: 2026-02-25 (新增 overlay ablation + 一致性檢查到上線 checklist)

> 日常開發時的 prompt 參考。5 個 Agent、什麼時候用誰、怎麼下 prompt。
> Slash commands 可用 `/` 快速觸發常用流程，見下方「Slash Commands」段落。

---

## 你的角色

你是 **Orchestrator（指揮者）**。Agent 之間不會自動溝通 — 你是中間人。
你的工作就是：看結果 → 判斷下一步 → 叫對的 Agent 做事。

---

## 速查：我想做什麼 → 叫誰？

| 我想... | Agent | Prompt 範例 |
|---------|-------|-------------|
| 探索新策略想法 | `@alpha-researcher` | `幫我研究 Funding Rate 反向策略的可行性` |
| 看鏈上數據有沒有 alpha | `@alpha-researcher` | `分析 BTCUSDT Open Interest 變化與價格的 IC` |
| 整理研究成提案 | `@alpha-researcher` | `把上面的研究整理成 Strategy Proposal` |
| 實作策略程式碼 | `@quant-developer` | `根據 docs/research/xxx_proposal.md 實作策略` |
| 跑回測 | `@quant-developer` | `用 config/research_xxx.yaml 跑回測 + --quick 驗證` |
| 審查回測結果 | `@quant-researcher` | `審查 reports/research/xxx/ 的回測結果` |
| 上線前風控審查 | `@risk-manager` | `對 xxx 策略做 pre-launch audit` |
| 每週風控檢查 | `@risk-manager` | `跑這週的風控快速檢查` |
| 部署最新改動到線上 | `@devops` | `幫我部署最新改動到 Oracle Cloud` |
| 部署到 Oracle Cloud | `@devops` | `部署 xxx 策略到生產環境` |
| 排查線上問題 | `@devops` | `Runner 好像掛了，幫我排查` |
| 加新幣 / 下載數據 | `@devops` | `幫 config 加入 AAVEUSDT 並下載數據` |
| 修 bug / 重構程式碼 | `@quant-developer` | `重構 xxx 模組，把 print 改成 logger` |
| 跑測試 | 任何 Agent | `跑一下 pytest 確認沒壞東西` |

---

## Slash Commands（快捷指令）

常用的 prompt 已做成 slash command，存放在 `.cursor/commands/`。
在 Chat 輸入框打 `/` 就會看到可用指令，選擇後會自動填入完整 prompt。

| Command | Agent | 用途 |
|---------|-------|------|
| `/deploy` | `@devops` | 部署最新改動到 Oracle Cloud（git push → pull → 重啟） |
| `/healthcheck` | `@devops` | 完整健康檢查（runner 狀態、持倉、系統資源） |
| `/backtest` | `@quant-developer` | 跑回測 + `--quick` 驗證（需填入 config 路徑） |
| `/config-freeze` | `@quant-developer` | 凍結研究配置為生產配置（Risk Manager APPROVED 後） |
| `/risk-review` | `@risk-manager` | 每週風控快速檢查 |
| `/risk-action` | `@quant-developer` | 根據 risk-review WARNING 產出改善方案並跑對比回測 |
| `/research-audit` | `@quant-researcher` | 審查回測結果（需填入報告路徑） |

> **帶有 `<填入...>` 的指令**：選擇 command 後，把 placeholder 替換成實際路徑再送出。

### 新增自訂 Command

如果有新的常用流程想做成 command：

1. 在 `.cursor/commands/` 建立 `<command-name>.md`
2. 檔案內容就是完整 prompt（包含 `@agent` tag）
3. 重啟 Cursor 或重新載入後即可使用 `/<command-name>`

---

## 三種工作模式

### Mode A: 完整流程（新策略從 0 到上線）

適合：全新策略開發，從研究到部署。

```
Step 1: @alpha-researcher
        「我想研究 [主題]，幫我建 Notebook 做 EDA」
        → 產出: notebooks/research/YYYYMMDD_xxx_exploration.ipynb

Step 2: @alpha-researcher
        「IC 看起來不錯，整理成 Strategy Proposal」
        → 產出: docs/research/YYYYMMDD_xxx_proposal.md

Step 3: @quant-developer
        「根據 docs/research/xxx_proposal.md 實作策略，
         建 config/research_xxx.yaml，跑回測 + --quick 驗證」
        → 產出: src/qtrade/strategy/xxx_strategy.py
                config/research_xxx.yaml
                reports/research/xxx/

Step 4: @quant-researcher
        「審查 reports/research/xxx/ 的回測結果，跑完整驗證 pipeline」
        → 產出: GO_NEXT / NEED_MORE_WORK / FAIL
        （Researcher 獨立重跑 WFA/CPCV/DSR/Cost Stress/Delay Stress）

Step 5: @risk-manager  (只有 GO_NEXT 才到這步)
        「對 xxx 策略做 pre-launch audit，
         config: config/research_xxx.yaml」
        → 產出: APPROVED / CONDITIONAL / REJECTED

Step 6: @quant-developer  (只有 APPROVED 才到這步)
        「凍結 config/research_xxx.yaml → config/prod_live_xxx.yaml」
        → 產出: config/prod_live_xxx.yaml（凍結的生產配置）

Step 7: @devops
        「部署 xxx 策略，配置: config/prod_live_xxx.yaml」
        → 產出: 線上運行
```

### Mode B: 快速迭代（改進現有策略）

適合：調參數、加 filter、修 bug。跳過 Alpha Researcher。

```
Step 1: @quant-developer
        「在 tsmom_ema 策略加一個 volatility filter，
         用 config/research_tsmom_vol.yaml 跑回測」

Step 2: @quant-researcher
        「審查回測結果，對比 baseline」

Step 3: @risk-manager  (if GO_NEXT)
        「做 pre-launch audit」

Step 4: @quant-developer  (if APPROVED)
        「凍結研究配置為生產配置 config/prod_live_*.yaml」

Step 5: @devops
        「部署已凍結的 config/prod_live_*.yaml 並重啟 runner」

Step 6: @devops
        「幫我部署最新改動到 Oracle Cloud」
        → Agent 自動：git push → SSH pull → 重啟 runner
```

### Mode C: 日常維運

適合：每天/每週的例行檢查。

```
每日:
  @devops  「跑健康檢查和每日報表」

每週:
  Step 1: /risk-review → @risk-manager 產出風控報告
  Step 2: 如果有 WARNING → /risk-action → @quant-developer 跑改善方案 + 對比回測
  Step 3: 把對比結果交給 @risk-manager 做最終判決
  Step 4: 如果 APPROVED → /config-freeze → @quant-developer 凍結新 config
  Step 5: @devops 部署凍結後的 config

每月:
  @risk-manager  「跑月度深度風控審查（Monte Carlo + 相關性 + Kelly 校準）」

有問題時:
  @devops  「Runner 異常 / 倉位不對 / 連線問題，排查」
```

---

## 部署同步（Local → Oracle Cloud）

**最簡單的方式**：直接叫 DevOps agent 幫你做：

```
@devops 幫我部署最新改動到 Oracle Cloud
```

Agent 會自動執行 git push → SSH pull → 判斷是否重啟 runner，並在 commit 前和重啟前跟你確認。

以下是手動流程（供參考或 agent 不可用時使用）。

程式碼透過 GitHub 同步：本機 push → Oracle Cloud pull。

```
┌──────────┐    git push    ┌──────────┐    git pull    ┌──────────────┐
│  Local   │ ────────────→  │  GitHub  │ ←──────────── │ Oracle Cloud │
│ (Cursor) │                │   Repo   │                │  (trading)   │
└──────────┘                └──────────┘                └──────────────┘
```

### 完整流程

```bash
# ── 1. 本機：commit + push ──
git add -A
git commit -m "feat: 改動描述"
git push

# ── 2. SSH 到 Oracle Cloud ──
ssh -i ~/.ssh/oracle-trading-bot.key ubuntu@140.83.57.255

# ── 3. Pull 最新 code ──
cd ~/quant-binance-spot
git pull

# ── 4. 判斷是否需要重啟 Runner ──
# 如果改了 config / 策略邏輯 / 依賴 → 需要重啟
# 如果只改了 docs / tests / scripts → 不需要（下次 runner 自動重啟時會 git pull）
```

### 需要重啟 Runner 時

```bash
# 停舊的
tmux attach -t r3c_e3_live   # 進入 tmux
# Ctrl+C 停止 runner
# Ctrl+B, d 離開 tmux

# runner 會自動重啟（tmux 裡的 while true 循環）
# 等 10 秒後確認
sleep 10 && tmux capture-pane -t r3c_e3_live -p | tail -20
```

### 什麼改動需要重啟？

| 改動類型 | 需要重啟？ | 原因 |
|---------|-----------|------|
| `config/prod_live_*.yaml` | **是** | Runner 啟動時讀取 config |
| `src/qtrade/strategy/` | **是** | 策略邏輯變更需重新載入 |
| `src/qtrade/live/` | **是** | Runner 核心邏輯 |
| `src/qtrade/data/` | **是** | 數據處理邏輯 |
| `docs/` / `tests/` | 否 | 不影響運行中的 runner |
| `scripts/` | 否 | Runner 不引用其他 script |
| `.cursor/` | 否 | 只影響本機開發 |

### 快速一鍵部署（Copy-Paste）

```bash
# 本機一行搞定
git add -A && git commit -m "fix: 改動描述" && git push

# Oracle Cloud 一行搞定（不重啟 runner，等自動重啟）
ssh -i ~/.ssh/oracle-trading-bot.key ubuntu@140.83.57.255 \
  "cd ~/quant-binance-spot && git pull"

# Oracle Cloud 一行搞定（強制重啟 runner）
ssh -i ~/.ssh/oracle-trading-bot.key ubuntu@140.83.57.255 \
  "cd ~/quant-binance-spot && git pull && tmux send-keys -t r3c_e3_live C-c"
```

> **注意**：tmux 裡的 `while true; do git pull && run_websocket.py; done` 循環會在 runner 停止後自動 pull 最新 code 並重啟。所以 `Ctrl+C` 停止 runner 就等於「重啟」。

---

## 讀取 Agent 輸出 — Next Steps 機制

每個 Agent 完成任務後，都會在輸出最後附上一個 **「Next Steps (pick one)」** 表格。
這讓你（Orchestrator）不需要記住完整的流程，只需要看建議選項、選一個、貼上 Prompt。

### 格式

```markdown
---
## Next Steps (pick one)

| Option | Agent | Prompt | When to pick |
|--------|-------|--------|-------------|
| A | `@next-agent` | "就緒可用的 prompt 文字..." | 什麼情況選這個 |
| B | `@other-agent` | "另一個可用的 prompt..." | 什麼情況選這個 |
```

### 使用方式

1. **看完 Agent 的報告結果**
2. **捲到最後的 Next Steps 表格**
3. **選一個 Option**，複製它的 Prompt 欄位
4. **新開 Chat，貼上 Prompt，必要時微調**（如填入路徑、加上你的判斷）
5. **送出**

### 關鍵原則

- 建議是 **advisory（建議性的）**，你永遠可以選不在表上的動作
- 如果你不確定選哪個，通常 **Option A 是預設推薦**
- 每個 Prompt 裡會包含足夠的上下文（路徑、數字、判決），下一個 Agent 不需要回頭翻
- 如果表格沒有出現，提醒 Agent：「請附上 Next Steps 選項」

### 各 Agent 的典型 Next Steps

| Agent 完成後 | 常見選項 |
|-------------|---------|
| Alpha Researcher | A: 交給 Developer 實作 · B: 轉換研究方向 · C: 歸檔停止 |
| Quant Developer | A: 提交 Researcher 審查 · B: 自我迭代改進 · C: 退回 Researcher |
| Quant Researcher | GO→ Risk Manager · NMW→ Developer · FAIL→ Researcher 或歸檔 |
| Risk Manager | APPROVED→ Developer (config freeze) → DevOps · CONDITIONAL→ Developer · REJECTED→ Developer |
| DevOps | A: 跑健康檢查 · B: 排定風控觀察 |

---

## Prompt 技巧

### 1. 指定角色就夠了

不需要重複解釋系統架構 — Agent 會自動讀取 `.cursor/rules/` 和自己的角色定義。

```
# 好 — 簡潔
@quant-developer 根據 proposal 實作 funding_reversal 策略

# 不需要 — 過度描述
@quant-developer 我們用的是 vectorbt 回測框架，策略要用 @register_strategy，
信號範圍 [-1,1]，要避免 look-ahead bias... 幫我實作策略
```

### 2. 提供具體路徑

Agent 讀檔案比猜測快。給路徑 > 給描述。

```
# 好
@quant-researcher 審查 reports/research/funding_reversal/20260224_120000/ 的結果

# 差
@quant-researcher 審查一下我剛跑的那個策略回測
```

### 3. 一次一個任務

不要在一個 prompt 裡讓同一個 Agent 做多件不相關的事。

```
# 好 — 分兩個 prompt
@quant-developer 實作 funding_reversal 策略
（等完成後）
@quant-developer 跑回測

# 差 — 太多事混在一起
@quant-developer 實作策略、跑回測、優化參數、然後幫我部署
```

### 4. 退回時附理由

當 Researcher 或 Risk Manager 退回時，把理由帶給下一個 Agent。

```
@quant-developer
Researcher 判定 NEED_MORE_WORK，原因：
1. WFA OOS Sharpe < 0.3 (只有 0.18)
2. 2024 年度報酬為負
請調整策略參數或加 regime filter 改善
```

### 5. 用 Plan Mode 做大型任務

按 `Shift+Tab` 進入 Plan Mode，讓 Agent 先分析 → 提問 → 擬定計畫 → 你確認後再執行。
適合用在：大重構、新模組設計、多檔案改動。

---

## 文件產出路徑總覽

| Phase | 產出 | 路徑 |
|-------|------|------|
| 研究探索 | Notebook | `notebooks/research/YYYYMMDD_*.ipynb` |
| 策略提案 | Proposal | `docs/research/YYYYMMDD_*_proposal.md` |
| 研究配置 | YAML | `config/research_*.yaml` |
| 策略程式碼 | Python | `src/qtrade/strategy/*_strategy.py` |
| 回測報告 | Report | `reports/research/<topic>/<timestamp>/` |
| 風控報告 | Audit | `reports/risk_audit/<timestamp>/` |
| 策略模板 | YAML | `config/futures_*.yaml` (可複用的策略定義) |
| 生產配置 | YAML | `config/prod_live_*.yaml` (由 Developer 凍結) |
| 實盤日誌 | Log | `logs/websocket.log` |

---

## 檢查清單：上線前必做

- [ ] **Validation Matrix 全部 PASS**（見 Playbook Section 7，V1-V12）
- [ ] **Overlay ablation 完成**（見 Playbook Stage D.5），報告中明確標註 overlay ON/OFF
- [ ] **`validate_live_consistency.py` 通過**（含 overlay 一致性檢查）
- [ ] Quant Researcher 判定 `GO_NEXT`（含一致性 gate）
- [ ] Risk Manager 判定 `APPROVED`
- [ ] Quant Developer 已凍結生產配置（`config/prod_live_xxx.yaml`）且不與研究配置混用
- [ ] 數據已下載（K 線 + Funding Rate + 策略所需額外數據如 OI）
- [ ] `prod_launch_guard.py` 通過
- [ ] Paper trading 至少跑 1 週（可選但建議）
- [ ] Telegram 通知已配置

---

## Agent 定義檔位置

| Agent | 檔案 |
|-------|------|
| Alpha Researcher | `.cursor/agents/alpha-researcher.md` |
| Quant Developer | `.cursor/agents/quant-developer.md` |
| Quant Researcher | `.cursor/agents/quant-researcher.md` |
| Risk Manager | `.cursor/agents/risk-manager.md` |
| DevOps | `.cursor/agents/devops.md` |

## Rules 定義檔位置

| Rule | 檔案 | 適用範圍 |
|------|------|----------|
| Project Overview | `.cursor/rules/project-overview.mdc` | 全局（alwaysApply） |
| Anti-Bias | `.cursor/rules/anti-bias.mdc` | 全局（alwaysApply） |
| Python Standards | `.cursor/rules/python-standards.mdc` | 所有 `*.py` |
| Strategy Dev | `.cursor/rules/strategy-dev.mdc` | `src/qtrade/strategy/**` |
| Backtest | `.cursor/rules/backtest.mdc` | backtest 相關檔案 |
| Live Trading | `.cursor/rules/live-trading.mdc` | live 相關檔案 |
| Research Methodology | `.cursor/rules/research-methodology.mdc` | research 相關檔案 |
