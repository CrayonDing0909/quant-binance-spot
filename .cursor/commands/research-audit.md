@quant-researcher 審查回測結果

請針對以下回測報告做完整驗證：
- 報告路徑: `reports/research/<填入路徑>/`

驗證項目：
1. Look-ahead bias 檢查（signal delay、price 用 open）
2. Cost model 是否開啟（funding rate + slippage）
3. WFA OOS Sharpe >= 0.3 gate
4. 年度報酬一致性（無單年撐全部的情況）
5. DSR 統計檢驗
6. Long/Short breakdown 對稱性

最後給出判定：GO_NEXT / NEED_MORE_WORK / FAIL，附理由。
