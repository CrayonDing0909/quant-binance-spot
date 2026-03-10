@orchestrator 批准或拒絕目前 stage gate

請處理以下研究任務的 gate：
- Task ID or path: `<填入 task_id 或 tasks/...yaml 路徑>`
- Decision: `<approve / reject / narrow_scope / broaden_scope / cancel>`
- Optional note: `<填入理由或限制>`

執行方式：
1. 讀取 task manifest，確認目前 `review_required = true` 或 `approval.required = true`。
2. 把本次決策寫回 manifest（包含 `resolved_at` 與 note）。
3. 若 decision 是 `approve`，自動把任務推進到下一個 stage。
4. 若 decision 是 `reject` 或 `cancel`，把任務標為 `completed` 或 `cancelled`，並給出收尾建議。
5. 若 decision 是 scope 變更，更新 goal / next action / blockers 後再續跑。
6. 補上 `final_packet` 與 `rollback` 摘要，讓 task 在結束時可追溯。
7. 回覆只需包含：
   - Task ID
   - Decision
   - Current stage
   - Status
   - What finished
   - What is running
   - Next step
