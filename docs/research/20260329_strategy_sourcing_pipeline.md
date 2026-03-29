---
tags: [research, strategy-sourcing, pipeline, MOC]
date: 2026-03-29
status: active
related:
  - "[[20260328_huashan_external_research]]"
  - "[[ALPHA_RESEARCH_MAP]]"
---

> **Last updated**: 2026-03-29

# Strategy Sourcing Pipeline — 可行策略方向總覽

> 系統性搜集的可行策略方向，按優先級排序。
> 每個方向經過 feasibility check 後才列入。

## Tier 1: 免費數據 + 現有架構直接可做

### A. WorldQuant 101 Formulaic Alphas
- **來源**: Kakushadze 2016 (arXiv:1601.00991)
- **內容**: 101 個 price-volume 公式，不需基本面數據
- **特點**: 平均持有 0.6-6.4 天，平均 pairwise corr 15.9%
- **Python 實作**: github.com/yli188/WorldQuant_alpha101_code
- **Crypto 適用性**: HIGH — 所有 price-volume alpha 直接可轉換
- **最有潛力的**: #9(趨勢/反轉切換), #12(成交量方向反轉), #43(成交量暴增×反轉), #55(%K-成交量相關), #101(日內 body ratio)
- **狀態**: `GO — blend tested`
- **IC Scan 結果 (20260329)**: wq012 IC=+0.011, wq028 IC=+0.008, wq055 IC=-0.013
- **Backtest**: Long-only 1h: BTC SR=0.45, ETH SR=0.74. TSMOM corr=-0.004 (零相關!)
- **Blend**: TSMOM 90% + WQ 10% → **SR 3.77→3.90 (+0.13), MDD -3.9%→-3.5%** ← 第一個能提升 prod SR 的策略
- **15m 版本**: FAIL — alpha 在 15m 太 noisy (SR=-0.34)。這些是 daily-origin alphas，1h 是最佳頻率
- **下一步**: `/validate-strategy` 驗證中

### B. Deribit 25Δ Options Skew
- **來源**: Deribit API (免費), Glassnode, Amberdata
- **信號**: `skew = IV(25Δ call) - IV(25Δ put)`，z-score > 1.5 = bullish
- **證據**: 75-80% risk reversal 配置在 BTC backtest 中 Sharpe 最佳
- **數據**: Deribit REST API（免費，1h 可用）
- **Crypto 適用性**: HIGH — BTC/ETH only（SOL 期權太薄）
- **狀態**: `PENDING DATA FETCH`

### C. Time-of-Day Signal Weighting
- **來源**: ACR Journal 2025, 38 交易所 1940 pairs 研究
- **信號**: 歐美重疊時段 (13:00-16:30 UTC) 回報高於平均
- **實作**: 現有信號 × time_weight（歐美時段 1.5x，亞洲 0.7x）
- **成本**: 零
- **狀態**: `READY TO IMPLEMENT`

### D. Weekend Momentum
- **來源**: ACR Journal 2025
- **信號**: BTC 週末 daily return 0.0023 vs 工作日 0.0012 (2x)
- **實作**: `long if (day_of_week in [5,6]) and (close > ema(close, 7))`
- **成本**: 零
- **狀態**: `READY TO IMPLEMENT`

## Tier 2: 需要新數據源（低成本）

### E. Liquidation Heatmap Entry/Exit
- **來源**: CoinGlass, Hyblock Capital
- **信號**: 價格接近密集清算區 → 預期 stop hunt → 反轉
- **成本**: CoinGlass Pro ~$30/月
- **證據**: >70% win rate (backtested on BitMEX since 2019)
- **狀態**: `PENDING DATA SOURCE`

### F. Whale Wallet Tracking
- **來源**: Nansen, CryptoQuant, Arkham Intelligence
- **信號**: 鯨魚淨流出交易所 72h → 累積 → bullish
- **成本**: CryptoQuant API ~$30/月
- **狀態**: `PENDING DATA SOURCE`

### G. Crypto Pair Spread MR
- **來源**: ALPHA_RESEARCH_MAP #25 (WEAK GO)
- **信號**: BTC/ETH, BTC/SOL log-spread z-score mean reversion
- **IC**: 0.035-0.057
- **狀態**: `HANDOFF → @quant-developer`（已在 backlog）

## Tier 3: 需要新基礎建設

### H. HFT / Market Making
- **需要**: $70K 建置 + $50K 本金 + $6K/月 + sub-50ms 延遲
- **狀態**: `NOT FEASIBLE` 目前

### I. MEV Detection
- **需要**: DEX 執行環境，不適用 CEX
- **狀態**: `NOT APPLICABLE`

## 學術/競賽來源索引

| 來源 | 類型 | URL | 備註 |
|---|---|---|---|
| WorldQuant 101 Alphas | Paper + Code | arXiv:1601.00991 | 101 個公式 |
| WorldQuant BRAIN IQC | Competition | worldquant.com/brain/iqc | 不公佈獲勝公式 |
| Kaggle G-Research Crypto | Competition | gresearch.com | Hull MA + LightGBM 獲勝 |
| Kaggle DRW Crypto | Competition | github.com/coderback/drw-crypto-market-prediction | Ridge regression |
| Numerai Signals | Tournament | docs.numer.ai | 需去除 Barra factors |
| Quantopian (已關) | Platform | — | PEAD factor 最成功 |
| ACR Journal 2025 | Paper | Weekend momentum effect | BTC 週末 2x 日回報 |
| Quantpedia | Database | quantpedia.com | BTC overnight sessions |
| Palazzi 2025 | Paper | J. Futures Markets | 10 pair cointegration |
| Springer Fin. Innovation 2025 | Paper | Triple Barrier + CUSUM | Dollar/volume bars |

## Anti-Pattern 提醒

- "月賺 30%" = 詐騙（年化 2000%+，無經審計紀錄）
- 84-97% 散戶第一年虧錢
- Backtest SR 和 Live SR corr < 0.05
- 排行榜 = 純倖存者偏差（10-25x 槓桿，MDD 30-70%）

## 迭代流程

```
1. 從本文件選一個 PENDING 方向
2. /check-direction → 確認不是 dead end
3. /quant-signals-reference → 查最佳參數 + anti-overfit
4. 跑 IC scan / EDA
5. 更新本文件狀態
6. GO → /implement-strategy → /validate-strategy
```
