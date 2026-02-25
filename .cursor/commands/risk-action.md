@quant-developer 根據 Risk Review 的 WARNING 產出改善方案

上一次 /risk-review（由 @risk-manager 產出）發現以下 WARNING：
<貼入 WARNING 列表，例如：>
- ⚠️ Alpha decay in: BTCUSDT, ETHUSDT, XRPUSDT, DOGEUSDT, ADAUSDT, LINKUSDT
- ⚠️ High avg correlation 0.681
- ⚠️ BTC concentration 35.4% after 2x weight

請針對每個 WARNING 執行以下流程：

1. **分析影響範圍**：哪些幣種 / 參數受影響？嚴重程度？
2. **提出 config 變更方案**（至少一個保守方案 + 一個積極方案）：
   - 保守方案：最小改動（例如只降 BTC 權重回 1x）
   - 積極方案：多項調整（例如降 BTC 權重 + deweight 衰退幣種 + 降 multiplier）
3. **建立研究配置**：`config/research_risk_mitigation_<date>.yaml`（不要動 prod config）
4. **跑對比回測**：
   - Baseline: 現有 config/prod_candidate_meta_blend.yaml
   - Option A: 保守方案
   - Option B: 積極方案
5. **產出比較表**：Total Return, Sharpe, Max DD, 7D/30D Return, HHI, Avg Corr

完成後我會把結果交給 @risk-manager 做最終判決，再決定是否部署。
