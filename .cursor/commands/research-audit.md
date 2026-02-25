@quant-researcher 審查回測結果

請針對以下回測報告做完整驗證（你必須獨立重跑，不信任 Developer 的數字）：
- 研究配置: `config/research_<填入配置名>.yaml`
- 報告路徑: `reports/research/<填入路徑>/`

完整驗證 Pipeline（按順序執行）：
1. Look-ahead bias 檢查（signal delay、price 用 open）
2. Cost model 是否開啟（funding rate + slippage）
3. 年度報酬一致性（無單年撐全部的情況）
4. Long/Short breakdown 對稱性
5. WFA OOS Sharpe >= 0.3 gate（>= 5 splits）
6. CPCV 交叉驗證
7. DSR 統計檢驗（Deflated Sharpe Ratio）
8. Cost Stress Test（1.5x, 2.0x multiplier）
9. Delay Stress Test（+1 bar 延遲）

最後給出判定：GO_NEXT / NEED_MORE_WORK / FAIL，附理由和各 gate 的 PASS/FAIL 結果。
