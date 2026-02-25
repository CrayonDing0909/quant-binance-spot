@quant-developer 幫我跑回測

請用以下配置跑回測 + `--quick` 驗證：
- Config: `config/research_<填入配置名>.yaml`
- 包含完整 cost model（funding rate + slippage）
- 產出 yearly breakdown、long/short 分析
- 跑 `validate.py --quick` 確認基本健全
- 報告存到 `reports/research/<topic>/<timestamp>/`

跑完後給我 summary：Total Return, Sharpe, Max DD, Calmar。
（WFA/CPCV/Cost Stress 等完整驗證會交給 @quant-researcher 獨立執行）
