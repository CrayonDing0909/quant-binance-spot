---
tags: [postmortem, tsmom, dead-end, lessons]
date: 2026-03-31
status: closed
---

# TSMOM Post-Mortem — 為什麼回測 SR=3.85 實盤全虧

> **Verdict**: TSMOM on crypto 1h is dead. SR=3.85 was overfit. Never revisit.

---

## 1. Executive Summary

TSMOM Carry V2 在回測中顯示 SR=3.85，但實盤 27 天 82 筆交易幾乎全虧。TradingView 視覺化確認 equity curve 是向下的。根本原因：1h crypto 市場結構性不適合 TSMOM，SR=3.85 是 2543 次試驗的倖存者偏差 + 多層 filter 疊加的 overfit 產物。

---

## 2. Timeline & Cost

| 日期 | 事件 | 時間 |
|---|---|---|
| 2026-02 初 | 開始 TSMOM 研究 | — |
| 2026-02-24 | BTC enhanced tier (720h lookback) | 研究 |
| 2026-02-26 | HTF Filter v2 (C_4h+daily_hard) | 研究 |
| 2026-02-27 | LSR overlay 研究 | 研究 |
| 2026-02-28 | OI regime, on-chain, macro filter 研究 | 研究 |
| 2026-03-04 | Symbol trimming (移除 ADA, BNB) | 優化 |
| 2026-03-24 | 上線 prod (prod_candidate_simplified.yaml) | 部署 |
| 2026-03-28 | 收到策略審查報告 — 27 天全虧 | 💀 |
| 2026-03-31 | TradingView 確認失敗 | 驗屍 |

**總時間**: ~5 週研究 + 4 週實盤
**機會成本**: 這 5 週可以探索 10+ 個其他方向
**實際虧損**: $151.77（實盤交易虧損）

---

## 3. Root Cause Analysis

### 3.1 SR=3.85 從第一天就該是紅旗

學術基準：
- Moskowitz, Ooi & Pedersen (2012): TSMOM 跨 58 個期貨品種，**月頻**，SR ≈ 1.0
- 在 **daily** 頻率上，TSMOM SR 通常 0.5-1.5
- 在 **1h** 頻率上，合理的 SR 上限大約 1.0-2.0（扣成本後）

**SR=3.85 是學術基準的 4 倍。在 2543 次累計試驗後得到這個數字，幾乎可以確定是 overfit。**

**規則**: 以後 crypto 1h 回測 SR > 2.0 → 自動觸發「overfit until proven otherwise」，必須先通過 DSR 才能繼續。

### 3.2 Crypto 1h 為什麼結構性不適合 TSMOM

| 原因 | 說明 |
|---|---|
| **Mean reversion 主導** | 1h 級別，crypto 的 autocorrelation 接近零甚至為負。動量在日內快速消散，mean reversion 力量更強 |
| **微結構噪音** | Funding rate 每 8h 結算、清算瀑布、交易所特定 wicks — 這些不是趨勢信號，是噪音 |
| **擁擠** | 每個散戶 algo 都在跑某種版本的 1h momentum — edge 早已被套利掉 |
| **Regime 不穩定** | Crypto 的動量半衰期遠短於傳統資產。2022 的參數在 2024 就失效了 |
| **交易成本** | 1h 頻率，即使 2bps slippage + funding = 吃掉大部分 edge |
| **6 幣 vs 58 個期貨** | 學術 TSMOM 靠跨 58 個品種分散。我們只有 6 個高度相關的 crypto — 分散化是幻覺 |

### 3.3 Overfit 怎麼潛入的

**a) 參數太多**

| 層 | 自由參數數量 |
|---|---|
| TSMOM core | 4 (lookback, vol_target, ema_fast, ema_slow) |
| EMA alignment | 2 (agree_weight, disagree_weight) |
| Basis signal | 3 (ema_fast, ema_slow, tanh_scale) |
| FR signal | 1 (rolling_window) |
| BasisCarry weights | 2 (fr_weight, basis_weight) |
| Confirmatory mode | 4 (agree_scale, disagree_scale, neutral_scale, smoothing) |
| HTF Filter | 8 (4h ema fast/slow, adx period/threshold, 4 weight levels) |
| Turnover control | 3 (composite_ema, position_step, min_change_threshold) |
| Per-symbol overrides | ~10 per symbol |
| **Total** | **~45+ 自由參數** |

45 個參數在 ~45,000 個 1h bars（5 年 × 8760h）上優化 = 每 1000 bars 一個參數。**嚴重不足。** 健康的比率是每個參數至少 10,000 bars。

**b) 「再加一個 filter 就好」陷阱**

```
TSMOM signal (IC=0.02, weak)
  → 加 EMA alignment → "改善了"
  → 加 Basis confirmation → "更好了"
  → 加 HTF Filter → "IC +212%!"
  → 加 OI regime filter → "8/8 symbols improved!"
  → 加 vol_pause overlay → "MDD 降了!"
  → 加 LSR overlay → "again improved!"

每加一層 = 更多的回測優化表面
每一層在回測中都「有效」但都是在同一批數據上調的
實盤中，這些 filter 的行為跟回測不同，因為市場微結構已經改變
```

**c) PBO = 47.5%**

我們自己的 CPCV 測試算出 PBO = 47.5% — 這意味著**接近一半的機率是 overfit**。我們「通過」了因為 threshold 設在 50%，但 47.5% 幾乎就是在說「拋硬幣」。

**d) 倖存者偏差**

- 2543 個累計研究試驗 → 選出最好的組合
- 24 個研究方向只有 5 個「成功」（80% 失敗率）
- 6 個幣種從 8 個中篩選出來（移除了 IC 差的 ADA, BNB）

### 3.4 流程失敗點

| 失敗 | 應該怎麼做 |
|---|---|
| SR=3.85 沒有自動觸發紅旗 | 設 SR > 2.0 = 自動 DSR 檢查 |
| PBO 47.5% 被當成「通過」 | 改 threshold 為 40% 或用更嚴格的 CPCV |
| 每週都在「再調一個參數」 | 設 2 週 time-box，到期必須 GO/FAIL |
| 實盤虧了 27 天才停 | 設 2 週 kill switch（14 天虧損 → 自動停） |
| 沒有先在 TradingView 上視覺化驗證 | 每個策略上線前必須先在 TV 上看 equity curve |
| Filter stacking 沒有自動 ablation 檢查 | 每加一個 filter 必須跑 3-way ablation |

### 3.5 「2543 次試驗」的數學

假設：
- 真實 SR = 0（策略沒有 alpha）
- 試了 2543 個變體
- 最好的一個 SR = 3.85

Bailey & López de Prado (2014) 的 Deflated Sharpe Ratio：
```
DSR = SR × √(T) / √(1 + skew × SR / 3 + kurtosis × SR² / 4)
     adjusted for: e × (1 - γ) × cumulative_trials

With 2543 trials, expected maximum SR from pure noise ≈ 3.0-4.0
→ SR=3.85 完全在「隨機最好結果」的範圍內
```

---

## 4. Failure Registry Entry

```
| TSMOM Carry V2 on Crypto 1H | 2026-02 to 2026-03 |
| Backtest SR=3.85, live 27 days all losses. Root cause: 1H crypto is
| structurally hostile to TSMOM (mean-reversion dominant, microstructure
| noise, 6 correlated assets ≠ 58 futures), plus 45+ params overfit on
| 5yr data with PBO=47.5%. TradingView visual confirmed equity curve
| goes down. Never revisit TSMOM standalone on crypto intraday. |
| Time wasted: 5 weeks research + 4 weeks live = ~$152 real loss |
| Salvageable: vol_scaling, HTF trend filter (as input, not standalone) |
```

---

## 5. Salvageable Components

| 組件 | 可以回收用在哪 | 在哪裡 |
|---|---|---|
| Vol scaling `(vol_target / realized_vol).clip(0.1, 2.0)` | 任何策略的 position sizing | `tsmom_carry_v2_strategy.py:104` |
| HTF 4h+daily trend | 作為 regime gate 給其他策略（不是 standalone） | `tsmom_carry_v2_strategy.py:1127` |
| Turnover control (EMA + quantize + deadzone) | 任何頻繁交易的策略 | `tsmom_carry_v2_strategy.py:296` |
| Causal resample 1h → 4h/1d | HTF 數據處理的標準方法 | `filters.py:causal_resample_align` |
| Pine Script TV replica | 快速驗證新策略用 | `scripts/tradingview/tsmom_carry_v2_default.pine` |

---

## 6. Process Improvements（具體規則變更）

### Rule 1: SR 紅旗

```
Crypto 1h 回測 SR > 2.0 → 自動標記 SUSPECT
必須先通過 Deflated Sharpe Ratio (DSR) 才能繼續
用 cumulative_n_trials 計算（不是當次試驗數）
```

### Rule 2: TradingView 視覺化驗證

```
任何策略上線前，必須先轉成 Pine Script 在 TV 上看 equity curve
如果 equity curve 視覺上就不行 → 不需要更多回測，直接 FAIL
```

### Rule 3: Time-box

```
單一策略方向最多 2 週
到期必須 GO (with evidence) 或 FAIL (with post-mortem)
不允許「再試一下」
```

### Rule 4: Filter Stacking 限制

```
每加一個 filter/overlay 必須跑 3-way ablation：
  A: 現有 baseline
  B: 新 filter 替代
  C: 現有 + 新 filter 疊加
如果 C < max(A, B) → 不加
```

### Rule 5: Kill Switch

```
實盤 14 天後強制 review
如果 cumulative PnL < 0 → 停止交易，做 post-mortem
不允許「再等等看」
```

### Rule 6: 參數/數據 比率

```
自由參數數 / 回測 bars 數 > 1/5000 → 自動標記 overfit 風險
45 params / 45000 bars = 1/1000 → 🔴 高風險
```
