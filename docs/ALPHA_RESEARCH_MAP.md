> **Last updated**: 2026-02-27

# Alpha 研究地圖 (Alpha Research Map)

Alpha Researcher **開始任何新研究前必讀**的結構化知識庫。
包含三大區塊：Alpha 覆蓋地圖、數據-信號分類圖譜、研究前沿排序表。

> **維護方式**：Alpha Researcher 每次研究 session 結束後必須更新本文件。
> 其他 agent 發現新的數據來源或策略狀態變更時，也應同步更新。

---

## 1. Alpha 覆蓋地圖

當前生產組合已捕捉的 alpha 維度、覆蓋品質、以及尚未填補的缺口。

**生產策略**：HTF Filter v2 + LSR — Meta-Blend 8-Symbol + HTF Filter + LSR Confirmatory Overlay, 3x leverage
**Satellite 策略**：OI Liquidation Bounce v4.2 — 5-Symbol long-only, 1x leverage (Paper Trading 中)

| # | Alpha 維度 | 信號來源 | 生產策略/組件 | 覆蓋品質 | 缺口/機會 |
|---|-----------|---------|-------------|---------|----------|
| 1 | 時序動量 (TSMOM) | Price EMA cross | `tsmom_carry_v2` — 主策略核心 | ★★★★★ 強（8/8 幣種） | 已飽和，難再改善 |
| 2 | Carry / 收益率 | FR proxy（價格估算） | `tsmom_carry_v2` carry 腿 | ★★★☆☆ 中（proxy，非真實 FR） | 真實 FR 數據可能改善 carry 腿品質 |
| 3 | HTF 趨勢確認 | 4h EMA + Daily ADX regime | HTF Filter v2（`_apply_htf_filter`） | ★★★★★ 強（8/8 改善） | 已飽和 |
| 4 | 散戶擁擠（LSR） | LSR percentile rank | LSR Confirmatory Overlay（boost/reduce） | ★★★★☆ 強（overlay） | standalone potential 受限於 2026 IC 翻轉 |
| 5 | OI 確認層 | OI 24h pct_change | LSR Overlay v2（`oi_confirm`） | ★★★☆☆ 中（overlay 組件） | OI 作為獨立 regime 信號未探索 |
| 6 | FR 擁擠確認 | FR pctrank + LSR 同向 | LSR Overlay v2（`fr_confirm`） | ★★★☆☆ 中（overlay 組件） | — |
| 7 | OI 事件驅動 | OI drop + price drop → bounce | `oi_liq_bounce` v4.2 | ★★★★☆ 強（satellite） | 僅做多、低曝險率（4.2%）、歷史數據有限 |
| 8 | 波動率 regime | ATR percentile | `vol_pause` overlay | ★★★☆☆ 中（僅退出用） | 方向性波動率信號未探索 |
| 9 | 截面動量 (XSMOM) | 相對強弱排名 | `xsmom`（已實作，未上線） | ☆☆☆☆☆ 未驗證 | 需正式回測 + corr 分析 |
| 10 | 微結構/訂單流 | Taker vol, CVD | `derivatives_micro_overlay`（已實作） | ☆☆☆☆☆ 未驗證 | 成本可行性待確認 |
| 11 | 鏈上 regime | TVL, 穩定幣流量 | `onchain.py`（僅數據模組） | ☆☆☆☆☆ 無信號 | 需建構 risk-on/off 信號 |
| 12 | 清算瀑布精確化 | 清算數據 | `oi_liq_bounce`（部分使用） | ★★☆☆☆ 弱（CoinGlass 歷史有限） | 更豐富數據源 + 更精確入場 |
| 13 | 多 TF 共振（獨立策略）| 多 TF 信號一致 | `multi_tf_resonance`（已實作） | ☆☆☆☆☆ 未驗證 | HTF Filter 已覆蓋部分功能 |
| 14 | Order Book 不平衡 | Depth imbalance | `order_book.py`（僅數據模組） | ☆☆☆☆☆ 無信號 | 無歷史數據，需 live 收集 |

### 維度覆蓋摘要

- **已充分覆蓋（★★★★+）**：時序動量、HTF 趨勢、散戶擁擠、OI 事件
- **部分覆蓋（★★-★★★）**：Carry、OI 確認、波動率、清算
- **未覆蓋（空白缺口）**：截面動量、微結構/訂單流、鏈上 regime、Order Book

---

## 2. 數據-信號分類圖譜

結構化對映：**數據源 → 可衍生信號類型 → 是否已測試 → 結果 → 當前用途**

### 2A. K 線衍生信號

| 數據源 | 信號類型 | 已測試？ | 結果 | 當前用途 |
|--------|---------|---------|------|---------|
| 1h K 線 | EMA cross momentum | ✅ 是 | SR=2.87 (portfolio) | `tsmom_carry_v2` 核心 |
| 1h K 線 | RSI + ADX + ATR | ✅ 是 | 早期策略，已被 TSMOM 取代 | `rsi_adx_atr`（retired） |
| 1h K 線 | Bollinger Band MR | ✅ 是 (20260225) | ❌ FAIL — 8/8 gross PnL < 0 | 無（MR 在加密無效） |
| 1h K 線 | Breakout + Vol expansion | ✅ 是 | BTC 專用，SR 適中 | `breakout_vol_atr`（BTC 30% blend） |
| 1h K 線 | NW Envelope regime | ✅ 是 | 已實作但未進生產 | `nw_envelope_regime`（archived） |
| 4h K 線 | EMA 趨勢過濾 | ✅ 是 (20260226) | +0.485 SR 改善 | HTF Filter v2（4h 趨勢腿） |
| Daily K 線 | ADX regime 判斷 | ✅ 是 (20260226) | HTF Filter 組件 | HTF Filter v2（daily regime 腿） |
| 5m/15m K 線 | 微結構入場時機 | ⚠️ 部分（EDA） | 高成本風險（12× turnover） | 無 |
| 1h K 線 | 截面相對強弱 | ⚠️ 部分（已實作） | 需正式回測 | `xsmom`（未上線） |

### 2B. 衍生品信號

| 數據源 | 信號類型 | 已測試？ | 結果 | 當前用途 |
|--------|---------|---------|------|---------|
| LSR（散戶） | Percentile 逆向交易 | ✅ 是 (20260226) | SR=1.39 standalone（含成本） | Overlay（confirmatory） |
| LSR（散戶） | Confirmatory scale（boost/reduce） | ✅ 是 (20260226-27) | +0.13 SR overlay 改善 | LSR Confirmatory Overlay |
| LSR（散戶 vs 大戶） | 散戶/大戶背離 | ✅ 是 (20260227) | IC=-0.041 最強，但 2026 IC 翻轉 + 換手率 7.8x | 僅研究，風險過高 |
| Top LSR（大戶帳戶） | 大戶方向追隨 | ✅ 是 (20260227) | IC 弱（-0.002），不如散戶 LSR | 無價值 |
| Top LSR（大戶持倉） | 大戶持倉追隨 | ✅ 是 (20260227) | IC 接近 0 | 無價值 |
| Taker Vol Ratio | 買賣不平衡動量 | ⚠️ 部分（EDA 20260227） | IC 有潛力，需獨立深入分析 | 無 |
| CVD | 價格-CVD 背離 | ⚠️ 部分（EDA 20260226） | 概念可行，需更多樣本 | `cvd_divergence`（已實作未驗證） |
| CVD | CVD momentum overlay | ⚠️ 部分（EDA 20260227） | IC 中等，待深入 | 無 |
| OI | Drop + bounce 事件 | ✅ 是（v4.2 完整驗證） | SR=2.49, corr≈0.01 | `oi_liq_bounce`（paper trading） |
| OI | Rising 確認（24h pct_change） | ✅ 是 (20260227) | overlay 組件，邊際改善小 | LSR Overlay v2 `oi_confirm` |
| OI | Regime 指標（high/low OI） | ❌ 否 | — | — |
| OI | Crowding 逆向 | ⚠️ 部分（EDA 20260227） | Cross-symbol crowding 因果修正後無效 | 無 |
| Funding Rate | 直接 carry 策略 | ✅ 是 (20260225) | ❌ FAIL — portfolio SR=-0.63（SOL/BNB FR < 0） | 無（standalone 不可行） |
| Funding Rate | Proxy carry（價格估算） | ✅ 是 | 作為 tsmom 輔助因子有效 | `tsmom_carry_v2` carry 腿 |
| Funding Rate | FR pctrank 擁擠確認 | ✅ 是 (20260227) | overlay 組件 | LSR Overlay v2 `fr_confirm` |
| Funding Rate | FR + LSR 雙重擁擠 | ✅ 是 (20260227) | 邊際改善 +0.013 SR（小但正面） | LSR Overlay v2 D mode |
| 清算數據 | 瀑布後反彈 | ✅ 是 | OI Liq Bounce 核心 | `oi_liq_bounce` |
| 清算數據 | 精確入場時機 | ❌ 否 | — | 歷史數據不足 |

### 2C. 鏈上/另類數據信號

| 數據源 | 信號類型 | 已測試？ | 結果 | 當前用途 |
|--------|---------|---------|------|---------|
| DeFi Llama TVL | Risk-on/off regime | ❌ 否 | — | — |
| 穩定幣市值 | 流動性 regime | ❌ 否 | — | — |
| DeFi Llama Yields | 跨市場套利 | ❌ 否 | — | — |
| Order Book Depth | Bid/Ask 不平衡 | ❌ 否（模組已建） | — | 無歷史數據 |

### 2D. 組合/系統層級信號

| 數據源 | 信號類型 | 已測試？ | 結果 | 當前用途 |
|--------|---------|---------|------|---------|
| 多 TF alignment | 信號共振過濾 | ✅ 是 (20260226) | HTF Filter 已覆蓋 | HTF Filter v2 |
| ATR percentile | Vol pause 退出 | ✅ 是 | +1.1 SR 改善（ablation） | `vol_pause` overlay |
| Cross-symbol corr | 擁擠/風險偵測 | ⚠️ 部分 (20260227) | 因果修正後無效（Δ SR=+0.01） | 無 |
| Risk regime 綜合 | Risk-on/off 組合縮放 | ❌ 否 | `low_freq_portfolio`（已實作未驗證） | — |

---

## 3. 研究前沿排序表

按 5 因子評分系統排序的候選研究方向。

### 評分標準

| 因子 | 權重 | 1 分（低） | 5 分（高） |
|------|------|----------|----------|
| 邊際分散化 | 30% | corr > 0.5 with existing | corr < 0.1 |
| 數據品質與可得性 | 20% | 無數據、需新來源 | 完整覆蓋、已下載 |
| 預期 alpha 強度 | 20% | IC < 0.01 或已知 FAIL | IC > 0.03 |
| 實作複雜度 | 15% | 新策略 + 新管線 | 簡單 overlay 在現有策略上 |
| 學術/實證支持 | 15% | 無文獻 | 強文獻 + 加密專屬研究 |

**門檻**：總分 < 2.5 不啟動深入研究。最高候選分數 < 3.0 時，「本週期不研究」是合理選項。

### 當前排序（2026-02-27）

| # | 研究方向 | 目標缺口 | 整合模式 | 分散化 | 數據 | Alpha | 複雜度 | 文獻 | **總分** | 備註 |
|---|---------|---------|---------|:------:|:----:|:-----:|:------:|:----:|:--------:|------|
| 1 | Taker Vol 不平衡 overlay | 微結構/訂單流 | Overlay | 4 | 4 | 3 | 4 | 3 | **3.6** | EDA 顯示 IC 有潛力，成本需驗證 |
| 2 | 鏈上 regime overlay（TVL/穩定幣） | 鏈上 regime | Overlay/Filter | 5 | 3 | 2 | 4 | 2 | **3.4** | 與現有策略正交，但 alpha 不確定 |
| 3 | OI regime（high/low OI 環境分類） | OI 確認 | Filter | 3 | 5 | 3 | 5 | 2 | **3.4** | 數據已有，簡單實作 |
| 4 | 截面動量 (XSMOM) 上線驗證 | 截面動量 | Standalone/Blend | 3 | 5 | 3 | 5 | 4 | **3.7** | 已實作，需正式 corr + 回測 |
| 5 | CVD divergence 深入研究 | 微結構 | Overlay | 4 | 4 | 2 | 3 | 3 | **3.2** | EDA 初步，需更完整驗證 |
| 6 | 真實 FR carry 改進 | Carry 品質 | 策略內部升級 | 1 | 5 | 2 | 5 | 3 | **2.7** | corr 高（改善同一策略），但簡單 |
| 7 | Order Book depth 不平衡 | 訂單流 | Overlay | 5 | 1 | 3 | 2 | 3 | **3.0** | 無歷史數據是致命問題 |
| 8 | retail_vs_top LSR standalone | 散戶擁擠 | Standalone | 4 | 4 | 3 | 3 | 2 | **3.3** | 2026 IC 翻轉 + 高換手率待解決 |
| 9 | Cross-symbol 擁擠偵測 | 系統風險 | Filter | 3 | 4 | 1 | 4 | 2 | **2.6** | 因果修正後幾乎無效 |
| 10 | 5m/15m 微結構入場 overlay | 執行改善 | Overlay | 2 | 3 | 2 | 2 | 3 | **2.3** | 成本侵蝕太大，低於門檻 |

### 建議下一步研究（Top 3）

1. **XSMOM 上線驗證**（#4, 3.7 分）— 已實作，只需跑正式回測 + 與生產組合 corr 分析。成本最低的研究。
2. **Taker Vol 不平衡 overlay**（#1, 3.6 分）— EDA 有初步信號，作為 overlay 成本影響小。
3. **鏈上 regime overlay**（#2, 3.4 分）/ **OI regime filter**（#3, 3.4 分）— 兩者並列，前者分散化更好但 alpha 不確定，後者數據更成熟。

---

## 4. 已關閉的研究方向（Dead Ends）

記錄已測試但確認無效的方向，避免重複研究。

| 方向 | 測試日期 | 失敗原因 | Notebook/報告 | 可復活條件 |
|------|---------|---------|-------------|-----------|
| Pure BB Mean Reversion | 2026-02-25 | 8/8 幣種 gross PnL < 0 | Alpha Researcher archetype 分析 | 加密市場正偏態結構性改變（極不可能） |
| FR 直接 Carry | 2026-02-25 | SOL/BNB 2yr FR < 0, portfolio SR=-0.63 | Alpha Researcher archetype 分析 | FR regime 穩定化 + 擁擠指數成熟 |
| Vol Squeeze 獨立策略 | 2026-02-25 | 4/8 通過但 edge 勉強覆蓋成本 | `notebooks/research/archive/20260224_vol_squeeze_exploration.ipynb` | 成本大幅降低或新的方向判斷指標 |
| OI Spike 策略 | 2026-02-25 | Hit rate ≈ 50%（無 alpha） | Alpha Researcher archetype 分析 | — |
| Cross-symbol crowding filter | 2026-02-27 | 因果修正後 Δ SR=+0.01（無效） | `notebooks/research/20260227_lsr_full_alpha_exploration.ipynb` | 更好的因果處理方法 |
| Top LSR（大戶）方向追隨 | 2026-02-27 | IC ≈ 0（無信息量） | `notebooks/research/20260227_lsr_full_alpha_exploration.ipynb` | — |
| retail_vs_top LSR standalone | 2026-02-27 | IC 最強但 2026 IC 翻轉 + 換手率 7.8x | `notebooks/research/20260227_lsr_full_alpha_exploration.ipynb` | 解決 IC 翻轉問題 + 降低換手率 |

---

## 5. 更新紀錄

| 日期 | 更新內容 | 更新者 |
|------|---------|--------|
| 2026-02-27 | 初版建立：覆蓋地圖 14 維度、數據-信號圖譜 40+ 條目、研究前沿 10 方向 | Quant Developer（從歷史研究記錄彙整） |
