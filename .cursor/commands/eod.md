End-of-Day 文件同步檢查。依序執行以下步驟，每步自動判斷是否需要動作。

---

## Step 1：自動掃描本次改動

執行 `git diff --name-only` 和 `git diff --staged --name-only` 列出所有未提交 + 已暫存的改動檔案。
若本次 session 已有 commit，也用 `git log --oneline -5` 確認最近幾筆 commit 涉及的檔案。

將改動檔案分類：
- `scripts/` — 腳本
- `config/` — 配置
- `src/qtrade/` — 程式碼
- `docs/` — 文件
- `.cursor/` — rules / commands / agents

列出摘要，格式：
```
📂 本次改動：
  scripts/   → (列出檔名)
  src/       → (列出檔名)
  ...
```

---

## Step 2：CLI_REFERENCE.md 判斷

如果 `scripts/` 或 `config/` 有任何改動（新增/修改/刪除），執行：
```bash
cd /Users/dylanting/Documents/spot_bot/quant-binance-spot
source .venv/bin/activate
PYTHONPATH=src python scripts/gen_cli_reference.py
```
如果沒有改動，輸出「✅ scripts/ 和 config/ 無改動，跳過 CLI_REFERENCE 重新生成」。

---

## Step 3：Living Docs 檢查

根據以下對照表，比對 Step 1 的改動，判斷哪些 living docs 需要更新：

| 改動區域 | 需要檢查的 doc |
|---------|--------------|
| 策略程式碼 (`src/qtrade/strategy/`) | `docs/R3C_STRATEGY_OVERVIEW.md` |
| 驗證/回測流程 (`src/qtrade/validation/`, `src/qtrade/backtest/`) | `docs/STRATEGY_DEV_PLAYBOOK_R2_1.md` |
| Symbol governance (`src/qtrade/live/symbol_governance.py`) | `docs/R3C_SYMBOL_GOVERNANCE_SPEC.md` |
| 策略組合工具 (`scripts/compare_strategies.py`, `scripts/generate_blend_config.py`) | `docs/STRATEGY_PORTFOLIO_GOVERNANCE.md` |
| Agent 定義 (`.cursor/agents/`) 或 handoff 流程 | `docs/CURSOR_WORKFLOW.md` |
| 新增 `docs/` 下的 living doc | `.cursor/rules/hygiene.mdc` 的 Living docs 表 |
| config 結構或命名規則變更 | `.cursor/rules/hygiene.mdc` 的 Config Naming Convention 表 |
| 部署/監控相關 (`src/qtrade/live/`, `src/qtrade/monitor/`) | `.cursor/agents/devops.md` |
| 策略狀態變更（上線、退役、候選） | `.cursor/rules/project-overview.mdc` |

對每個需要更新的 doc：
1. 讀取該 doc 的當前內容
2. 判斷是否確實需要更新（可能改動很小不影響 doc）
3. 如果需要，更新內容並更新 `Last updated` 日期
4. 如果不確定，列出建議並詢問使用者

如果沒有 doc 需要更新，輸出「✅ 所有 living docs 已是最新狀態」。

---

## Step 4：Lint 檢查

對本次改動的所有 `.py` 檔案跑 lint（使用 read_lints 工具）。
如果有新引入的錯誤，修復或提醒使用者。
如果無錯誤，輸出「✅ Lint 檢查通過」。

---

## Step 5：更新 recent-changes.mdc

讀取 `.cursor/rules/recent-changes.mdc`，執行：

1. **追加今日記錄**：在 `---` 分隔線之後，以今天日期為標題，寫入本次 session 的變更摘要（中文，簡潔）。如果今天已有條目，追加到現有條目下方（不要重複日期標題）。
2. **清理舊條目**：刪除超過 30 天的條目（根據 `## YYYY-MM-DD` 標題判斷）。

格式範例：
```markdown
## 2026-02-25

### 某功能名稱
- 變更摘要 1（`相關檔案路徑`）
- 變更摘要 2（`相關檔案路徑`）
```

---

## Step 6：最終確認

輸出所有執行結果的摘要：
```
🏁 EOD 檢查完成：
  ✅/⚠️ CLI_REFERENCE: (結果)
  ✅/⚠️ Living docs: (更新了哪些 / 無需更新)
  ✅/⚠️ Lint: (結果)
  ✅/⚠️ Recent changes: (已更新 / 已是最新)
```
