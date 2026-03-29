---
tags: [research, signals, btc-vs-eth, market-structure, literature-review]
date: 2026-03-28
status: active
related:
  - "[[20260328_huashan_external_research]]"
  - "[[ALPHA_RESEARCH_MAP]]"
---

> **Last updated**: 2026-03-28

# BTC vs ETH 信號差異 — 為什麼同樣策略表現天差地別？

> 背景: 华山 Convergence v2.1 在 BTC SR=1.34 / MDD=-21%，但 ETH SR=0.26 / MDD=-44%

## 核心結論

BTC 和 ETH 已經是**結構性不同的資產**。同一個 composite signal 不能用相同參數跨幣種套用。差異不是隨機雜訊，而是 6 個可識別的結構性原因。

## 6 大結構性原因

### 1. 資訊處理速度不對稱
- BTC 消化 common shocks 最快；ETH 延遲反應
- BTC lagged returns 能預測 ETH（資訊外溢），反向不成立
- **影響**: BTC 上的信號是「及時」的，ETH 上可能是「過時」或「反轉」的
- Source: Guo, Sang, Tu & Wang (2024), *J. Economic Dynamics and Control*

### 2. 機構 vs 散戶組成
- BTC: ~30% 機構持有（ETF $25B inflow in 2025 + 企業 treasury）
- ETH: 機構佔比較低，DeFi-native 散戶佔比更高
- 機構操作在多年 horizon，不對日內噪音反應 → BTC noise floor 更低
- Source: NBER WP 31317, ainvest (2025)

### 3. Funding Rate 結構性差異
- ETH FR 在 0.01% floor 達 **87.5%** 的時間（BitMEX 數據）vs BTC 78.2%
- ETH 永續合約被 staking delta-hedge 結構性做空（long spot ETH + short perp 賺 staking yield）
- 這不是看空信號，是機械性避險 → FR 逆向信號在 ETH 上系統性誤判
- ETH FR std dev 比 BTC 高 35%（Hyperliquid: 0.0131% vs 0.0097%）→ 更 noisy
- Source: BitMEX Q3 2025 Derivatives Report

### 4. 鏈上速度混淆
- BTC daily on-chain turnover: 0.61% → 鏈上信號 = 真實交易意圖
- ETH daily on-chain turnover: **1.3%** → 混雜 staking、LST 再平衡、L2 bridge、DeFi 機械操作
- ~16% ETH supply 在 liquid staking + collateral 結構中，持續非方向性移動
- **影響**: 穩定幣流入/鏈上活躍度在 ETH 上包含大量非 alpha 噪音
- Source: Glassnode/Keyrock joint research

### 5. 套利效率差距
- BTC basis arbitrage 更快、更深（更多機構資本 → FR 極端快速回歸）
- ETH 套利較慢，FR 極端可能持續更久 OR 不規則回歸
- 結果: BTC 上 FR contrarian = 可靠 mean-reversion；ETH 上 = 不確定
- Source: BIS Working Paper 1087

### 6. Post-Merge 結構斷裂（2022.09）
- ETH 從 PoW → PoS，volatility 結構改變
- 任何跨 Merge 日期的模型都包含結構性斷裂
- BTC 無等效事件
- **影響**: 2022 前後的 ETH 參數不穩定

## Trade-Level 驗證

| 指標 | BTC | ETH |
|---|---|---|
| 短期交易（<12h）avg PnL | -0.40% | **-1.96%**（5x 更慘） |
| Win rate | 41.2% | 37.0% |
| Avg loss | -1.65% | **-2.41%**（46% 更深） |
| 2025-Q1 return | +2.3% | **-31.8%** |
| 2022-Q2 (LUNA) return | +11.5% | **-17.7%** |
| 最長連敗 | 10 trades | **11 trades**（×2 次） |

ETH 災難集中在兩個季度：2022-Q2（-17.7%）和 2025-Q1（-31.8%），合計 -50%，壓垮整個策略。

## 設計啟示

### 不該做的
- 不要用相同參數跨 BTC/ETH 套用
- 不要信任 ETH 上的 FR contrarian signal（staking hedge 污染）
- 不要在 ETH 結構性下跌期依賴 convergence score（false bottom 陷阱）

### 該做的
- **Per-symbol 參數**: BTC 4h rebalance, ETH daily rebalance（或更慢）
- **ETH 加 regime filter**: 價格 < 50日均線 + 動量下降 → 暫停或降倉
- **ETH 降低穩定幣/FR 權重**: 這兩個信號在 ETH 上被結構性噪音污染
- **最低持倉時間**: 過濾 <12h 的 whipsaw trades（ETH 上尤其致命）
- **或者: BTC-only** — ETH 用其他更適合的策略

## 學術文獻

1. Guo, Sang, Tu & Wang (2024) — Cross-Cryptocurrency Return Predictability, *JEDC*
2. "Seesaw Effect" — Negative lead-lag at intraday level, *J. Empirical Finance*
3. BIS Working Paper 1087 — Crypto carry trade structure
4. BitMEX Q3 2025 — Derivatives market report (FR structure by exchange)
5. Easley et al. (Cornell, 2024) — VPIN in crypto markets
6. Fulgur Ventures — Bitcoin funding rates and price predictability
7. Glassnode/Keyrock — Bitcoin and Ethereum as competing store-of-value
8. Bitwise Research — ETH/BTC fundamental mispricing indicator (R²=0.799)
9. NBER WP 31317 — Are Cryptos Different? Evidence from Retail Trading
10. Wiley Financial Review — Price Discovery in Bitcoin ETF Market
