@portfolio-strategist 幫我先做策略診斷，不要直接跳到調參數或實作

請針對以下策略 / 方向做 **Strategy Diagnosis Card + Experiment Matrix**：

- Strategy / baseline: `<填入策略名、config 或報告路徑>`
- Current pain: `<填入目前觀察到的問題，例如 MDD 太高 / 某年失效 / 頻率太低 / 震盪期連續打臉>`
- Optional context: `<填入已知假說、候選指標、相關檔案>`

輸出要求：
1. 先明確診斷 `baseline pain` 與 `pain regime`
2. 判斷這輪應該鎖定哪一個 **primary experiment family**：
   - `signal_mechanism`
   - `entry_timing`
   - `exit_design`
   - `position_sizing`
   - `portfolio_role`
3. 明確標註這輪是：
   - `Loop A: Alpha Existence`
   - 或 `Loop B: Trade Expression`
4. 寫出一張 **Strategy Diagnosis Card**：
   - baseline pain
   - current archetype
   - target role
   - primary experiment family
   - economic mechanism
   - what stays fixed
   - pass rule
   - kill rule
5. 產出 1-3 列 **Experiment Matrix**
   - `question`
   - `family`
   - `changed_component`
   - `held_constant`
   - `economic_reason`
   - `metric`
   - `baseline`
   - `pass_rule`
   - `kill_rule`
   - `next_action`
6. 若你判斷現在還不該研究 TP/SL 或 HTF，請直接說明原因
7. 最後附上：
   - `Recommended next agent`
   - `Recommended prompt`
   - `Why no new agent is needed`（若適用）

硬規則：
- 不要把 signal / entry / exit / sizing 混在同一輪結論
- 若 raw alpha 還沒被證明，禁止直接討論 trade expression 優化
- 若方向本質上只是現有策略的同質因子，請明說不值得研究
