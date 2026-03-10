@orchestrator 續跑 research task

請讓以下研究任務恢復前進：
- Task ID or path: `<填入 task_id 或 tasks/...yaml 路徑>`
- Resume reason / new input: `<填入補充資訊、解除 blocker 的條件、或新的 scope>`

執行方式：
1. 讀取 task manifest。
2. 若狀態為 `paused` / `blocked` / `stalled`，先更新 reason 與 `last_heartbeat_at`。
3. 判斷應該原地續跑、回到上一 stage，或 reroute 到別的 stage。
4. 若 blocker 已解除，預設自動前進；只有信心下降到 `low` 時才重新要求 review。
5. 更新 manifest 的 `current_stage`、`status`、`confidence_level`、`next_recommended_action`。
6. 回覆只需包含：
   - Task ID
   - Current stage
   - Status
   - What changed
   - What is running
   - What needs approval
   - Next step
