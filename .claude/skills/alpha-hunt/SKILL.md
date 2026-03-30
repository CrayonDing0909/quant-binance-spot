---
description: "系統性策略發想與評估 — 從 0 找到可交易的 alpha。適用於不知道該做什麼策略的時候。"
---

# /alpha-hunt — 從 0 到策略

當你不知道該做什麼策略、或現有策略失效需要找新方向時，用這個流程。

## Step 1: 選擇 Edge 來源類型

每個能賺錢的策略都有一個 edge 來源。先選一個你有能力利用的：

| Edge 類型 | 說明 | 例子 | 我們能做嗎？ |
|---|---|---|---|
| **資訊優勢** | 比別人更早知道某件事 | 新聞 NLP、社群情緒、insider tip | ⚠️ 需要數據源 |
| **結構性低效** | 市場結構本身造成的持續偏差 | Funding rate 溢價、跨交易所價差、PM maker rebate | ✅ 可程式化 |
| **行為偏差** | 人的心理造成的定價錯誤 | 恐慌後的超跌反彈、crowd overreaction | ✅ 有歷史數據可驗證 |
| **技術優勢** | 比別人更快或更有效率 | HFT、co-location、低延遲 | ❌ 需要 $70K+ 基礎建設 |
| **跨市場套利** | 同一資產在不同地方價格不同 | Binance vs Coinbase、spot vs perp | ⚠️ 大部分已被套完 |

**行動**: 選一個類型，然後往下走。

## Step 2: 腦暴具體策略

### 方法 A: 從已知信號出發

用 `/quant-signals-reference` 查看已有的信號庫：
- Funding Rate（contrarian）
- OI（4-quadrant）
- LSR（crowd sentiment）
- CB Premium（institutional flow）
- Stablecoin（liquidity）

問自己：「這些信號有沒有用在新的場景上？」
- 同一個信號用在不同幣種？
- 同一個信號用在不同時間框架？
- 兩個信號的組合？

### 方法 B: 從外部靈感出發

去這些地方找靈感：
1. **arXiv** — `/arxiv search crypto trading` 看最近 6 個月的論文
2. **Reddit r/algotrading** — 搜 "crypto bot" 看別人在做什麼
3. **Quantpedia.com** — 免費策略摘要
4. **GitHub trending** — 搜 "crypto trading" 看 star > 100 的 repo
5. **Twitter/X** — 搜 "crypto alpha", "on-chain signal"
6. **你的朋友** — 直接問有在賺錢的人

### 方法 C: 從市場異常出發

觀察市場有沒有不合理的地方：
- 某個時段回報率特別高？（seasonality）
- 某個事件發生後總是反應過度？（event-driven）
- 某個幣種的衍生品定價偏離現貨？（basis trade）
- 預測市場的定價偏離你的估計？（PM mispricing）

## Step 3: 快速可行性檢查

回答這 5 個問題。任何一個答案是「不行」→ 換方向。

1. **數據**: 能免費拿到嗎？歷史至少 6 個月？
2. **學術支持**: 有論文或可靠來源說這有效嗎？（用 `/arxiv` 或 `/semantic-scholar`）
3. **Edge 大小**: 粗估 > 交易成本嗎？（PM maker 0%、Binance ~5bps）
4. **競爭**: 有多少人在做同樣的事？（如果 GitHub 有 10 個 repo 在做 → 可能已擁擠）
5. **Dead ends**: 用 `/check-direction` 確認不是已知的失敗方向

## Step 4: 回測

按照 `/quant-signals-reference` 的 anti-overfit 方法論：

1. 先跑 **random baseline** — 必須 ≈ 50% WR / PnL ≈ 0，否則回測有 bug
2. 跑你的策略 — 看 WR 和 PnL
3. 畫 **calibration chart** — 你的預測 vs 實際，偏離 = edge
4. 看 **cumulative PnL curve** — 應該穩定上升，不是一兩筆暴擊撐起來的
5. 做 **parameter sensitivity** — 改 ±20% 參數，結果不能崩

## Step 5: Dry Run → Real

**絕對不要跳過 dry run。** 我們已經虧了 $14 因為跳過。

1. Dry run **至少 50 筆結算**
2. 用 `scripts/analyze_pm_dryrun.py` 分析結果
3. PnL > 0 → 切 real mode，每筆 $1 起步
4. 跑 1 週 real → 如果正 → 逐步加碼
5. 任何時候 daily loss limit 觸發 → 停下來分析

## 已知的 Dead Ends（不要再踩）

| 方向 | 為什麼失敗 | 日期 |
|---|---|---|
| TSMOM 1h crypto | IC ≈ 0.02 扣成本後無 edge，實盤全虧 | 2026-02 |
| PM contrarian < $0.35 | PM 在極端時 85-92% 準確，做反向 = 送錢 | 2026-03-30 |
| FR carry (always-on) | 2025+ SR 負值，已被機構套完 | 2026-03-28 |
| 直接轉 USDC 到 PM 地址 | 困在 proxy wallet，$9 無法取回 | 2026-03-30 |

## 有潛力但未完全驗證的方向

| 方向 | 狀態 | 下一步 |
|---|---|---|
| PM Momentum ETH t=5 <$0.65 | 回測 +$35.4 on 319 trades | Dry run 驗證中 |
| PM Contrarian ETH $0.35-$0.45 London | 回測 +$11.8 on 63 trades | Dry run 驗證中 |
| PM Market Making | 研究中，可能是真正的 edge | 需要在 London/NY 時段監控 order book |
| WQ Alpha 101 (012+028+055) | 回測 TSMOM corr=-0.004, blend +SR | Validated but TSMOM 已失效 |
| Liquidation cascade + PM | 數據支持 (66-82% WR on bounce) | 需要結合 PM daily market |
