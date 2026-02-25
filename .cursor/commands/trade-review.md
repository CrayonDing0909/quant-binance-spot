@risk-manager 跑本週的交易復盤

請針對目前的生產策略執行交易復盤分析：

## 自動執行步驟

### Step 1: 交易復盤報告（每個在線策略都跑）
```bash
source .venv/bin/activate

# Meta-Blend（主策略）
PYTHONPATH=src python scripts/trade_review.py -c config/prod_candidate_meta_blend.yaml --days 7 --with-replay

# OI Liq Bounce（Paper Trading，如在線）
PYTHONPATH=src python scripts/trade_review.py -c config/prod_live_oi_liq_bounce.yaml --days 7
```

### Step 2: 診斷分析

根據復盤報告，逐項檢查：

1. **勝率偏離**：live win rate vs backtest win rate，偏離 > 15% 需標註 WARNING
2. **PnL 偏離**：live PnL vs backtest replay PnL，偏離 > 30% 需標註 WARNING
3. **信號執行一致性**：方向不一致 > 5% 需標註 WARNING
4. **市場環境**：如果多數幣種處於盤整（ADX < 20），TSMOM 虧損是預期內的正常行為
5. **連續虧損**：如果連續虧損 > 5 筆，啟動 alpha decay 快速掃描

### Step 3: 決策建議

根據分析結果，給出以下其中一個判定：

| 判定 | 條件 | 行動 |
|------|------|------|
| ✅ NORMAL | 所有指標正常，虧損在預期範圍內 | 繼續觀察 |
| ⚠️ WATCH | 1-2 個 WARNING，但市場環境可解釋 | 下週重點監控 |
| 🔶 INVESTIGATE | 3+ 個 WARNING 或 PnL 嚴重偏離 | 啟動 alpha decay 監控，通知 Quant Researcher |
| 🔴 ACTION | 持續 2+ 週 INVESTIGATE | 啟動策略審查流程（參見 STRATEGY_PORTFOLIO_GOVERNANCE.md） |
