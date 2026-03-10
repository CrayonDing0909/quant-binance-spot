@orchestrator 啟動新的 research orchestration task

請把以下目標當作新的研究任務：
- Goal: `<填入一句話研究目標>`
- Optional context: `<填入相關背景、路徑、限制條件>`

執行方式：
1. 依 `tasks/_templates/research_task_manifest.yaml` 在 `tasks/active/` 建立新的 task manifest。
2. 先做 intake；若目標已足夠明確，不要先問泛問題，直接進 strategist stage。
3. 依 `docs/ORCHESTRATION_MVP.md` 更新 stage / status / heartbeat / review / rollback 欄位。
4. 預設自動串 `strategist -> alpha_research -> stop_or_handoff`；只有 `confidence_level = low`、硬 blocker、或高風險決策時才停在 `awaiting_approval`。
5. task manifest 是 system of record：請填入 `confidence_level`、`review_required`、`code_audit`、`rollback`、`final_packet`。
6. 除非需要 review，否則不要每個內部 stage 都回報給我。
7. 回覆只需包含：
   - Task ID
   - Current stage
   - Status
   - What finished
   - What is running
   - What needs approval
   - Next step
