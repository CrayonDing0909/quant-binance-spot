@quant-developer 幫我跑回測

請用以下配置跑回測 + Walk-Forward Analysis：
- Config: `config/research_<填入配置名>.yaml`
- 包含完整 cost model（funding rate + slippage）
- 產出 yearly breakdown、long/short 分析、WFA OOS 統計
- 報告存到 `reports/research/<topic>/<timestamp>/`

跑完後給我 summary：Total Return, Sharpe, Max DD, WFA OOS Sharpe。
