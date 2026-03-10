@orchestrator 查詢 research task 狀態

請讀取以下 task manifest：
- Task ID or path: `<填入 task_id 或 tasks/...yaml 路徑>`

執行方式：
1. 讀取對應 manifest。
2. 回報目前的 `current_stage`、`status`、`owner_agent`、`last_heartbeat_at`、`confidence_level`。
3. 若狀態為 `blocked` 或 `stalled`，說清楚原因與解除方式。
4. 若狀態為 `awaiting_approval`，說清楚正在等我批准什麼，以及為什麼不是自動前進。
5. 額外回報 `rollback.target` 與 `rollback.steps` 摘要，讓我知道若改壞該怎麼退。
6. 最後輸出：
   - Task ID
   - Current stage
   - Status
   - Confidence level
   - Last heartbeat
   - Blocked reason (if any)
   - What needs approval
   - Rollback summary
   - Next step
