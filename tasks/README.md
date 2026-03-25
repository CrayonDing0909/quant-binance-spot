# Research Task Manifests

這個目錄存放 Research Orchestration 使用的 task manifests。
它們是 **system of record**，用來保存狀態、決策、影響範圍、與回退資訊。

## 結構

- `active/`：目前仍在進行或等待批准的任務
- `_templates/`：官方 YAML 模板
- `pilots/`：用真實研究主題做的 replay / smoke test 樣本

## 命名

- 檔名：`<timestamp>_<slug>.yaml`
- `task_id`：`research_<timestamp>_<slug>`

## 使用方式

1. 用 `/start-research` 建立新 manifest，並在 active chat 內前景續跑後續 stages
2. 用 `/task-status` 查狀態
3. 用 `/approve-stage` 做 gate 決策
4. 用 `/resume-task` 續跑卡住、暫停、或 chat 中斷後未完成的任務

欄位定義以 `docs/ORCHESTRATION_MVP.md` 和 `_templates/research_task_manifest.yaml` 為準。

## 設計原則

- 平常由 `@orchestrator` 自動推進流程，你不需要手動編輯 manifest
- 自動推進的正確語意是：**active chat 內 foreground-autonomous 執行**，不是背景 daemon
- manifest 保留完整 task ledger，適合事後追：
  - 做了哪些判斷
  - 影響哪些檔案
  - 跑了哪些測試
  - 如果改壞了要怎麼退
- `tasks/active/` 是主要工作區；必要時可保留最近完成任務，供審計與回溯

## Execution Model

- `foreground_autonomous`：目前已支援。`/start-research` 應在同一個 active chat invocation 內盡量跑完 `intake -> strategist -> alpha_research -> stop_or_handoff`。
- `background_queue`：目前**未支援**。如果 chat 結束，task 只會保留在 manifest 裡等待之後 `/resume-task` 或新的使用者訊息續跑。
