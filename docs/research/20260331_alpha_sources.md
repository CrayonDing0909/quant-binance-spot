---
tags: [research, alpha, pipeline, active]
date: 2026-03-31
status: active
related:
  - "[[20260329_strategy_sourcing_pipeline]]"
  - "[[20260330_polymarket_bot_report]]"
---

> **Last updated**: 2026-03-31

# Alpha 來源掃描 — 2026-03 搜集結果

## Tier 1: 馬上能做

### A. Hyperliquid HLP Vault（被動收入）
- **做什麼**: 存 USDC 到 Hyperliquid 的 HLP vault，當市場的 counterparty
- **怎麼賺**: 賺 maker spread + 清算收入（大戶爆倉時你分到錢）
- **證據**: 2026/2/1 $700M BTC 爆倉 → vault +5.8%（單日），年化穩定 13.4%
- **最低**: 100 USDC
- **風險**: 大戶賺錢時 vault 虧（你是對手方）；4 天提款鎖定
- **來源**: KuCoin research, CoinDesk

### B. Stablecoin Velocity（新鏈上信號）
- **做什麼**: 追蹤 stablecoin volume/mcap ratio 的變化
- **怎麼賺**: velocity 上升 = 資金活躍 = risk-on regime → 做多
- **數據**: DeFiLlama API（免費）
- **狀態**: 2026 新信號，需要 EDA 驗證
- **來源**: BeInCrypto on-chain signals report

### C. Funding Rate 極端 Mean Reversion（已驗證）
- **做什麼**: FR 極端正 → 做空，FR 極端負 → 做多
- **怎麼賺**: 68-72% WR on extreme signals（學術驗證）
- **數據**: 已有（`data/binance/futures/funding_rate/`）
- **差異 vs 之前**: 之前當 overlay，這次做 **standalone event-driven**（只在極端時交易）
- **來源**: arXiv:2212.06888, ScienceDirect 2025, BIS WP 1087

## Tier 2: 需要新數據源

### D. Polymarket 運動/天氣套利
- PM odds vs 專業模型（NOAA 天氣、Pinnacle 體育賠率）
- 需要: The Odds API ~$50/月

### E. Smart Money 錢包跟單
- 5+ 大戶同時進場 → 跟進
- 需要: Nansen $150/月 或 Dune 免費

### F. Yield-Bearing Stablecoin 供給
- 供給翻倍 = institutional risk-on 建倉
- 數據: DeFiLlama 免費

## Dead Ends（不再嘗試）

| 方向 | 為什麼 | 日期 |
|---|---|---|
| TSMOM 1h crypto | IC≈0.02，實盤全虧 | 2026-02 |
| PM contrarian <$0.35 | PM 極端時 85-92% 準確 | 2026-03-30 |
| FR carry (always-on) | 2025+ SR 負值，已被套完 | 2026-03-28 |
| PM latency arb | Fee 1.56% > spread | 2026-03 |
| Grid bot (無 regime filter) | 趨勢突破時爆虧 | 已知 |
