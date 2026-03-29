---
tags: [research, strategy, liquidation, event-driven]
date: 2026-03-29
status: active
related:
  - "[[20260329_strategy_sourcing_pipeline]]"
  - "[[ALPHA_RESEARCH_MAP]]"
---

> **Last updated**: 2026-03-29

# Liquidation Cascade Mean-Reversion v2

## 為什麼這有效（因果機制，不是 pattern mining）

清算是**被迫賣出**：
1. 價格下跌 → 槓桿交易者 margin 不足
2. 交易所自動平倉 → 大量市價賣單湧入
3. 賣壓進一步壓低價格 → 觸發更多清算（cascade）
4. Cascade 結束後 → 價格超跌 → 均值回歸

**這不是預測方向。這是在確認事件發生後，賺取超跌的回彈。**

## 數據驗證 (2022-2026)

### BTC — OI 24h 下跌 >10%

| 年份 | Events | Avg 24h return | Win Rate |
|---|---|---|---|
| 2022 | 402 | -0.97% | 44% ❌ |
| 2023 | 252 | +0.36% | 59% |
| 2024 | 212 | **+1.72%** | **82%** ✅ |
| 2025 | 88 | **+1.64%** | **66%** ✅ |
| 2026 | 42 | **+2.36%** | 50% |

### OI 下跌 >15% (更強的信號)

| | BTC avg 72h | ETH avg 72h |
|---|---|---|
| All years | **+1.90%** | **+2.66%** |

### 2022 為什麼是負的？
2022 是持續熊市（BTC 從 $48K 跌到 $16K），清算是趨勢的一部分不是超跌。
→ 需要 regime filter: 只在非極端熊市做。

## 策略設計

```
TRIGGER: OI 24h change < -10%
CONFIRM: OI 已停止下跌（1h OI change > -1%）
ENTRY:   做多 (market buy)
TP:      +3%
SL:      -2%
TIME:    72h max hold
REGIME:  BTC 30d return > -20% (排除持續熊市)

頻率: ~88 events/年 (2025 BTC), ~每4天一次
TIM:   ~10% (大部分時間不在場內 → 不付 funding)
```

## vs 舊策略的差異

| | 舊 TSMOM | 新 Liquidation Cascade |
|---|---|---|
| 核心假設 | 「趨勢會持續」 | 「被迫賣出造成超跌」 |
| IC | ~0.02 (弱) | N/A (event-driven) |
| 因果機制 | 無（統計相關） | 有（清算 → 超跌 → 反彈） |
| 在場時間 | ~80% | ~10% |
| Funding 成本 | 巨大 | 幾乎沒有 |
| 濾器數量 | 6+ 層 | 1 (regime filter) |
| Overfit 風險 | 高 (PBO 47.5%) | 低（規則極簡） |
