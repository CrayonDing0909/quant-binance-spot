@orchestrator 啟動新的 research orchestration task

請把以下目標當作新的研究任務：
- Goal: `<填入一句話研究目標>`
- Optional context: `<填入相關背景、路徑、限制條件>`

執行方式：
1. 依 `tasks/_templates/research_task_manifest.yaml` 在 `tasks/active/` 建立新的 task manifest。
2. 先做 intake；若目標已足夠明確，不要先問泛問題，直接進 strategist stage。
3. 依 `docs/ORCHESTRATION_MVP.md` 更新 stage / status / heartbeat / review / rollback 欄位。
4. **不要在 manifest 建立後就停下**。預設應在同一個 active chat invocation 內繼續執行 `strategist -> alpha_research -> stop_or_handoff`。
5. 只有以下情況才提前停在使用者可見邊界：`confidence_level = low`、硬 blocker、需要 approval、或 final packet 已完成。
6. task manifest 是 system of record：請填入 `confidence_level`、`review_required`、`code_audit`、`rollback`、`final_packet`。
7. 除非需要 review / blocker / final packet，否則不要每個內部 stage 都回報給我。
8. 若同一 invocation 內已跑到 `stop_or_handoff`，回覆應直接反映 final packet，而不是只回報 manifest 已建立。
9. 回覆只需包含：
   - Task ID
   - Current stage
   - Status
   - What finished
   - What is running
   - What needs approval
   - Next step
   - Recommended agent
   - Primary files / artifacts

補充規則：
- 若結果已進入 `stop_or_handoff`，請用 **handoff packet** 形式回覆，而不是只說 manifest 已建立。
- handoff packet 至少應包含：`Current status`、`Key decision`、`Next recommended action`、`Recommended agent`、`Primary files / artifacts`、`Open risks / blockers`。
