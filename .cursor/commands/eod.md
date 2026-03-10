End-of-Day 文件同步檢查。依序執行以下步驟，每步自動判斷是否需要動作。

---

## Step 1：自動掃描本次改動

### 1a. 以 workspace 實際改動為主，不以 git 歷史推測為主

改動來源的優先順序：

1. **本次 chat 明確編輯 / 建立 / 刪除的檔案**
2. **當前 worktree 的已儲存改動**：`git diff --name-only` + `git diff --staged --name-only`
3. **只有在需要補充背景時**，才看 `git log --name-only --oneline -5`

原則：

- **不要**因為某個檔案出現在最近 commit 就自動算成本次 session 改動
- **不要**把最近 5 筆 commit 全部視為本次工作範圍
- 如果 repo 本來就有 unrelated dirty files，而你能辨識本次 chat 實際碰過哪些檔案，**只追本次 chat touched files**
- 如果無法分辨哪些 dirty files 屬於本次 session，要明確標記 `⚠️ worktree 內含可能非本次 session 的既有改動`

### 1b. 掃描方式

先用 `git diff --name-only` 和 `git diff --staged --name-only` 列出所有未提交 + 已暫存的改動檔案。
若需要補充 commit context，可用 `git log --name-only --oneline -5`，但只能拿來幫助理解，不可擴大本次工作範圍。

將改動檔案分類：
- `scripts/` — 腳本
- `config/` — 配置
- `src/qtrade/` — 程式碼
- `docs/` — 文件
- `.cursor/` — rules / commands / agents
- `tasks/` — research task manifests / templates
- repo root — 例如 `README.md`

如果本次 chat 中已有明確 touched files，輸出時優先列這些檔案；其餘 worktree 改動若看起來像既有未處理改動，另外標成 `可能非本次 session`。

列出摘要，格式：
```
📂 本次改動：
  scripts/   → (列出檔名)
  src/       → (列出檔名)
  ...
```

如果有疑似非本次 session 的改動，額外輸出：
```
⚠️ 可能非本次 session 的既有改動：
  ...
```

---

## Step 2：自動生成文件判斷

### 2a. CLI_REFERENCE.md
如果 `scripts/` 或 `config/` 有任何改動（新增/修改/刪除），執行：
```bash
cd /Users/dylanting/Documents/spot_bot/quant-binance-spot
source .venv/bin/activate
PYTHONPATH=src python scripts/gen_cli_reference.py
```
如果沒有改動，輸出「✅ scripts/ 和 config/ 無改動，跳過 CLI_REFERENCE 重新生成」。

### 2b. DATA_STRATEGY_CATALOG.md
如果符合以下任一條件，執行：
- `src/qtrade/data/` 有新增/修改/刪除模組
- `src/qtrade/strategy/` 有新增/修改/刪除模組
- `config/prod_*.yaml` 有新增/修改/刪除

執行：
```bash
PYTHONPATH=src python scripts/gen_data_strategy_catalog.py
```
如果沒有改動，輸出「✅ data/、strategy/、prod config 無改動，跳過 DATA_STRATEGY_CATALOG 重新生成」。

---

## Step 3：Living Docs 檢查

根據以下對照表，比對 Step 1 的改動，判斷哪些 living docs 需要更新：

| 改動區域 | 需要檢查的 doc |
|---------|--------------|
| 策略程式碼 (`src/qtrade/strategy/`) | `docs/R3C_STRATEGY_OVERVIEW.md` |
| 驗證/回測流程 (`src/qtrade/validation/`, `src/qtrade/backtest/`) | `docs/STRATEGY_DEV_PLAYBOOK_R2_1.md` |
| Symbol governance / feature ownership | `.cursor/skills/dev/feature-ownership-registry.md` |
| 策略組合工具 (`scripts/compare_strategies.py`, `scripts/generate_blend_config.py`) | `docs/STRATEGY_PORTFOLIO_GOVERNANCE.md` |
| Agent 定義 (`.cursor/agents/`)、commands (`.cursor/commands/`)、或 handoff / orchestration / task schema (`tasks/`) | `docs/CURSOR_WORKFLOW.md`、`docs/ORCHESTRATION_MVP.md` |
| 新增 `docs/` 下的 living doc | `.cursor/rules/hygiene.mdc` 的 Living docs 表 |
| config 結構或命名規則變更 | `.cursor/rules/hygiene.mdc` 的 Config Naming Convention 表 |
| 部署/監控相關 (`src/qtrade/live/`, `src/qtrade/monitor/`) | `.cursor/agents/devops.md` |
| 策略狀態變更（上線、退役、候選） | `docs/R3C_STRATEGY_OVERVIEW.md`、`.cursor/rules/project-overview.mdc`、`.cursor/agents/devops.md` |
| 數據模組新增/修改 (`src/qtrade/data/`) | Step 2b 已自動重新生成 `DATA_STRATEGY_CATALOG.md`（如 Step 2b 已執行則無需額外動作） |
| 策略新增/修改 (`src/qtrade/strategy/`) | Step 2b 已自動重新生成 `DATA_STRATEGY_CATALOG.md`（如 Step 2b 已執行則無需額外動作） |

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
如果本次沒有 `.py` 改動，輸出「✅ 本次無 Python 改動，跳過 lint」。
如果無錯誤，輸出「✅ Lint 檢查通過」。

---

## Step 5：更新 recent-changes.mdc

讀取 `.cursor/rules/recent-changes.mdc`，執行：

1. **追加今日記錄**：在 `---` 分隔線之後，以今天日期為標題，寫入本次 session 的變更摘要（中文，簡潔）。如果今天已有條目，追加到現有條目下方（不要重複日期標題）。
2. **維持單一 rolling log 區塊**：如果檔案內出現重複的 frontmatter / header / 舊版附錄，清成單一有效區塊後再追加。
3. **清理舊條目**：刪除超過 7 天的條目（根據 `## YYYY-MM-DD` 標題判斷），與 repo 規則保持一致。

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
  ✅/⚠️ DATA_STRATEGY_CATALOG: (結果)
  ✅/⚠️ Living docs: (更新了哪些 / 無需更新)
  ✅/⚠️ Lint: (結果)
  ✅/⚠️ Recent changes: (已更新 / 已是最新)
```
