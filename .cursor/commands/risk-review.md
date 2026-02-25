@risk-manager 跑這週的風控快速檢查

請針對目前的生產組合做以下檢查：

**主策略**: config/prod_candidate_meta_blend.yaml (Meta-Blend 8-Symbol)
**候選策略**: config/prod_live_oi_liq_bounce.yaml (OI Liq Bounce, Paper Trading)

1. 累計 PnL 和 drawdown 趨勢
2. 近 7 天 Sharpe 和 win rate
3. Alpha decay 指標（IC 和 turnover）
4. 各幣種相關性變化
5. 倉位集中度
6. 交易復盤（`trade_review.py --days 7`）— 勝率/PnL 是否偏離回測預期

如有觸發任何警戒線，請標示 WARNING 並給出建議。
