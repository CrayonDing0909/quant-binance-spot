@quant-developer 凍結研究配置為生產配置

Risk Manager 已判定 APPROVED，請執行 config freeze：
- 研究配置: `config/research_<填入配置名>.yaml`
- 產出: `config/prod_live_<填入名稱>.yaml`

凍結步驟：
1. 從研究配置複製為 `config/prod_live_<name>.yaml`
2. 移除實驗性註解，確認所有參數為最終值
3. 確認 `notification`、`risk.circuit_breaker_pct`、`risk.max_drawdown_pct` 已正確設定
4. 列出 research vs prod 的關鍵差異
5. **掃描舊配置引用**：如果此次凍結是要**替換**現有生產策略，執行以下掃描並更新所有過時引用：
   ```bash
   # 把 <舊配置名> 替換為被替換的配置檔名（不含路徑前綴），例如 prod_live_R3C_E3
   grep -r "<舊配置名>" .cursor/ docs/ --include="*.md" --include="*.mdc" -l
   ```
   對列出的每個檔案，將舊配置名更新為新的生產配置名。
   同時檢查 tmux session 名稱引用是否也需要更新。

完成後我會把凍結的 config 交給 @devops 部署。
