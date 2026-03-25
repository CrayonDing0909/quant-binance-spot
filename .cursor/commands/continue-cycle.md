@orchestrator 繼續下一個 research cycle

請讀取目前 running 的 research task 並產出下一個 cycle 的完整 prompt：
- Task ID or path: `<填入 task_id，或留空讓 orchestrator 自動找 running task>`

執行方式：
1. 掃描 `tasks/active/*.yaml`，找到 `status: running` 的 task。
2. 讀取 `next_recommended_action` 和 `artifacts.proposal`。
3. 從 proposal 中找到下一個 cycle 的 Experiment Matrix（根據已完成的 cycle 判斷）。
4. 從最近完成的 notebook 中提取 accepted baseline 數字。
5. 產出一段完整的 prompt，包含：
   - `@agent` tag（通常是 `@alpha-researcher`）
   - Research Cycle Declaration（baseline pain, family, loop type, what stays fixed）
   - Task description（引用 proposal section）
   - Accepted baseline numbers（從上一輪 notebook 提取）
   - Experiment matrix rows
   - Pass / Kill rules
   - Relevant files
   - Deliverable
6. 以 code block 形式輸出這段 prompt。
7. 問使用者：「要我直接執行這段 prompt，還是你要複製到新 chat？」

補充規則：
- 若沒有 running task，回報並建議用 `/task-status` 查看。
- 若 `next_recommended_action` 指向非研究工作（如 developer handoff），說明並建議正確的 agent。
- 若 proposal 中的下一個 cycle matrix 不存在（所有 cycle 已完成），建議進入 developer handoff 或 portfolio role 決策。
