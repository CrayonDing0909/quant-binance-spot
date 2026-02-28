> **Last updated**: 2026-02-28 (On-Chain Regime Filter ablation + validation → **KEEP_BASELINE**: B(On-chain only) SR=4.00 vs A(HTF) SR=3.80, Δ=+5.3% borderline, 全 gate PASS but 觀察期內不替換)

# Alpha 研究地圖 (Alpha Research Map)

Alpha Researcher **開始任何新研究前必讀**的結構化知識庫。
包含三大區塊：Alpha 覆蓋地圖、數據-信號分類圖譜、研究前沿排序表。

> **維護方式**：Alpha Researcher 每次研究 session 結束後必須更新本文件。
> 其他 agent 發現新的數據來源或策略狀態變更時，也應同步更新。

---

## 1. Alpha 覆蓋地圖

當前生產組合已捕捉的 alpha 維度、覆蓋品質、以及尚未填補的缺口。

**生產策略**：HTF Filter v2 + LSR — Meta-Blend 8-Symbol + HTF Filter + LSR Confirmatory Overlay, 3x leverage
**Satellite 策略**：無（OI Liq Bounce v4.2 已 SHELVED 2026-02-27，insight 轉為 oi_cascade overlay）

| # | Alpha 維度 | 信號來源 | 生產策略/組件 | 覆蓋品質 | 缺口/機會 |
|---|-----------|---------|-------------|---------|----------|
| 1 | 時序動量 (TSMOM) | Price EMA cross | `tsmom_carry_v2` — 主策略核心 | ★★★★★ 強（8/8 幣種） | 已飽和，難再改善 |
| 2 | Carry / 收益率 | FR proxy（價格估算） | `tsmom_carry_v2` carry 腿 | ★★★☆☆ 中（proxy，非真實 FR） | 真實 FR 數據可能改善 carry 腿品質 |
| 3 | HTF 趨勢確認 | 4h EMA + Daily ADX regime | HTF Filter v2（`_apply_htf_filter`） | ★★★★★ 強（8/8 改善） | 已飽和 |
| 4 | 散戶擁擠（LSR） | LSR percentile rank | LSR Confirmatory Overlay（boost/reduce） | ★★★★☆ 強（overlay） | standalone potential 受限於 2026 IC 翻轉 |
| 5 | OI 確認層 | OI 24h pct_change | LSR Overlay v2（`oi_confirm`） | ★★★☆☆ 中（overlay 組件） | OI regime filter **WEAK GO** (20260228): IC=0.006 弱但 F5 Δ SR +0.317, 8/8 improved |
| 16 | OI Regime Gate | OI pctrank_720 level filter | ablation 完成 → **FAIL** | ★★☆☆☆ 弱 → FAIL | Ablation: A(HTF)=3.86, B(OI)=4.12, C(HTF+OI)=4.04。Incremental SR +4.66% < 5% → FAIL。OI standalone 強但與 HTF 疊加 over-filter |
| 6 | FR 擁擠確認 | FR pctrank + LSR 同向 | LSR Overlay v2（`fr_confirm`） | ★★★☆☆ 中（overlay 組件） | — |
| 7 | OI 事件驅動 | OI drop + price drop → bounce | `oi_cascade` overlay + `oi_liq_bounce`（SHELVED） | ★★★☆☆ 中（overlay 研究中） | 獨立策略效益低（TIM=4.2%），已轉 overlay；BTC 空頭抵消需調優 |
| 8 | 波動率 regime | ATR percentile | `vol_pause` overlay | ★★★☆☆ 中（僅退出用） | 方向性波動率信號未探索 |
| 9 | 截面動量 (XSMOM) | 相對強弱排名 | `xsmom`（已實作，**FAIL**） | ❌ 無效（SR=-0.50, 6 variants 全負） | 加密截面動量不存在（高相關性 + rank-invariant residual） |
| 10 | 微結構/訂單流 | Taker vol, CVD | `derivatives_micro_overlay`（已實作） | ★★☆☆☆ 弱（WEAK GO） | TVR IC=-0.006(弱但獨立), CVD 不穩定, 建議作為 LSR overlay 第4確認因子 |
| 11 | 鏈上 regime | TVL/穩定幣 momentum | EDA GO → Ablation+Validation → **KEEP_BASELINE** (20260228) | ★★★★☆ 強 | IC=0.065, B(On-chain only) SR=4.00 vs A(HTF) SR=3.80, Δ=+5.3% borderline。WFA 8/8 PASS, CPCV PBO max 0.13, DSR 2.28。Code preserved, 觀察期後(3/14)重評估 |
| 12 | 清算瀑布精確化 | 清算數據 | `oi_liq_bounce`（部分使用） | ★★☆☆☆ 弱（CoinGlass 歷史有限） | 更豐富數據源 + 更精確入場 |
| 13 | 多 TF 共振（獨立策略）| 多 TF 信號一致 | `multi_tf_resonance`（已實作） | ☆☆☆☆☆ 未驗證 | HTF Filter 已覆蓋部分功能 |
| 14 | Order Book 不平衡 | Depth imbalance | `order_book.py`（僅數據模組） | ☆☆☆☆☆ 無信號 | 無歷史數據，需 live 收集 |
| 15 | TF 優化（4h 替換 1h） | 4h TSMOM vs 1h+HTF | EDA + 正式回測完成 (20260227) | ❌ **CLOSED** — 修正 look-ahead 後 Δ SR=+0.20, PBO 52-67% | HTF fix 後邊際消失，4h Pure SR 3.97 vs baseline 3.77，之前 +1.53 SR 來自 look-ahead bias |

### 維度覆蓋摘要

- **已充分覆蓋（★★★★+）**：時序動量、HTF 趨勢、散戶擁擠、OI 事件
- **部分覆蓋（★★-★★★）**：Carry、OI 確認、波動率、清算
- **已確認無效**：截面動量 (XSMOM)、TF 優化（4h 替換 1h，修正 look-ahead 後 Δ SR 僅 +0.20, PBO 偏高）
- **已測試 WEAK GO（★★）**：微結構/訂單流（TVR 獨立但 IC 弱）
- **已測試 FAIL**：OI Regime Gate（standalone SR=4.12 > HTF=3.86，但 incremental +4.66% < 5%，與 HTF 疊加 over-filter）
- **已測試 KEEP_BASELINE（★★★★）**：鏈上 regime（On-chain only SR=4.00 > HTF SR=3.80, 全 gate PASS，但增量 borderline +5.3%，觀察期內不替換。Code preserved 供 3/14 後重評估）
- **未覆蓋（空白缺口）**：Order Book

---

## 2. 數據-信號分類圖譜

結構化對映：**數據源 → 可衍生信號類型 → 是否已測試 → 結果 → 當前用途**

### 2A. K 線衍生信號

| 數據源 | 信號類型 | 已測試？ | 結果 | 當前用途 |
|--------|---------|---------|------|---------|
| 1h K 線 | EMA cross momentum | ✅ 是 | SR=2.87 (portfolio) | `tsmom_carry_v2` 核心 |
| 1h K 線 | RSI + ADX + ATR | ✅ 是 | 早期策略，已被 TSMOM 取代 | `rsi_adx_atr`（retired） |
| 1h K 線 | Bollinger Band MR | ✅ 是 (20260225) | ❌ FAIL — 8/8 gross PnL < 0 | 無（MR 在加密無效） |
| 1h K 線 | Breakout + Vol expansion | ✅ 是 (20260228 ablation) | **負貢獻** Δ SR=-0.03，20 params | ~~`breakout_vol_atr`~~ **REMOVED** |
| 1h K 線 | NW Envelope regime | ✅ 是 | 已實作但未進生產 | `nw_envelope_regime`（archived） |
| 4h K 線 | EMA 趨勢過濾 | ✅ 是 (20260226) | +0.485 SR 改善 | HTF Filter v2（4h 趨勢腿） |
| Daily K 線 | ADX regime 判斷 | ✅ 是 (20260226) | HTF Filter 組件 | HTF Filter v2（daily regime 腿） |
| 5m/15m K 線 | 微結構入場時機 | ⚠️ 部分（EDA） | 高成本風險（12× turnover） | 無 |
| 1h K 線 | 截面相對強弱 | ✅ 是 (20260227) | ❌ FAIL — avg SR=-0.50, 6 variants 全負 | `xsmom`（FAIL，已關閉） |
| 4h K 線 | TSMOM TF 替換（1h→4h） | ✅ 是 (20260227 EDA+正式回測) | ❌ **CLOSED** — 修正 HTF look-ahead 後 4h Pure SR 3.97 vs baseline 3.77 (Δ=+0.20), PBO 52-67%。之前 +1.53 SR 來自 look-ahead bias | 無（已關閉） |

### 2B. 衍生品信號

| 數據源 | 信號類型 | 已測試？ | 結果 | 當前用途 |
|--------|---------|---------|------|---------|
| LSR（散戶） | Percentile 逆向交易 | ✅ 是 (20260226) | SR=1.39 standalone（含成本） | Overlay（confirmatory） |
| LSR（散戶） | Confirmatory scale（boost/reduce） | ✅ 是 (20260226-27) | +0.13 SR overlay 改善 | LSR Confirmatory Overlay |
| LSR（散戶 vs 大戶） | 散戶/大戶背離 | ✅ 是 (20260227) | IC=-0.041 最強，但 2026 IC 翻轉 + 換手率 7.8x | 僅研究，風險過高 |
| Top LSR（大戶帳戶） | 大戶方向追隨 | ✅ 是 (20260227) | IC 弱（-0.002），不如散戶 LSR | 無價值 |
| Top LSR（大戶持倉） | 大戶持倉追隨 | ✅ 是 (20260227) | IC 接近 0 | 無價值 |
| Taker Vol Ratio | TVR pctrank 逆向 overlay | ✅ 是 (20260227 EDA) | IC=-0.006(弱逆向), corr(LSR)=-0.013(獨立), Smooth24+85pctile Δ SR=+0.155(2.1x turnover) | **WEAK GO**: 建議作為 LSR overlay v2 第4確認因子 |
| Taker Vol Ratio | Raw TVR overlay | ✅ 是 (20260227 EDA) | autocorr≈0.01(iid noise), raw pctrank→6x turnover(overfitted) | ❌ 必須先 24h SMA 平滑 |
| CVD | CVD direction momentum | ✅ 是 (20260227 EDA) | IC=+0.001(不可復現), 直接 overlay Δ SR=-0.251(HURTS performance) | ❌ CVD 非動量信號，與初步 EDA 矛盾 |
| CVD | CVD raw/pctrank 逆向 | ✅ 是 (20260227 EDA) | IC=-0.013~-0.023(逆向), 但 IC 年度翻轉(2022 負→2026 正) | ❌ 不穩定，不建議使用 |
| CVD | 價格-CVD 背離 | ✅ 是 (20260227 EDA) | Δ SR=+0.053(邊際), turnover 2.6x | ❌ 不值得複雜度 |
| OI | Drop + bounce 事件 | ✅ 是（v4.2 完整驗證） | SR=2.49 standalone, +0.11 SR as overlay | `oi_liq_bounce`（SHELVED）→ `oi_cascade` overlay |
| OI | Rising 確認（24h pct_change） | ✅ 是 (20260227) | overlay 組件，邊際改善小 | LSR Overlay v2 `oi_confirm` |
| OI | Regime 指標（pctrank level filter） | ✅ 是 (20260228 EDA + ablation) | **FAIL (incremental)**: Standalone SR=4.12 > HTF=3.86 (+6.7%), 但 HTF+OI SR=4.04 incremental 僅 +4.66% < 5%。OI 與 HTF 部分冗餘，疊加 over-filter (5/8 symbols SR 下降) | **FAIL** — 不加入生產。Code preserved in `filters.py` |
| OI | Crowding 逆向 | ⚠️ 部分（EDA 20260227） | Cross-symbol crowding 因果修正後無效 | 無 |
| Funding Rate | 直接 carry 策略 | ✅ 是 (20260225) | ❌ FAIL — portfolio SR=-0.63（SOL/BNB FR < 0） | 無（standalone 不可行） |
| Funding Rate | Proxy carry（價格估算） | ✅ 是 | 作為 tsmom 輔助因子有效 | `tsmom_carry_v2` carry 腿 |
| Funding Rate | FR pctrank 擁擠確認 | ✅ 是 (20260227) | overlay 組件 | LSR Overlay v2 `fr_confirm` |
| Funding Rate | FR + LSR 雙重擁擠 | ✅ 是 (20260227) | 邊際改善 +0.013 SR（小但正面） | LSR Overlay v2 D mode |
| 清算數據 | 瀑布後反彈 | ✅ 是 | OI Liq Bounce 核心 → overlay 轉化 | `oi_cascade` overlay（+0.11 SR, 5/8 symbols） |
| 清算數據 | 精確入場時機 | ❌ 否 | — | 歷史數據不足 |

### 2C. 鏈上/另類數據信號

| 數據源 | 信號類型 | 已測試？ | 結果 | 當前用途 |
|--------|---------|---------|------|---------|
| DeFi Llama TVL | TVL momentum regime filter | ✅ 是 (20260228 EDA) | **GO**: tvl_mom_30d IC=0.065, 8/8 same sign, A1 5+/2-, quintile spread +4.35 | → Handoff Quant Dev as Filter |
| DeFi Llama TVL | TVL/SC ratio momentum | ✅ 是 (20260228 EDA) | **GO**: monotonic quintile spread +4.69, 8/8 improved at P30 Δ SR +0.41 | → Handoff Quant Dev as Filter |
| 穩定幣市值 | SC momentum regime | ✅ 是 (20260228 EDA) | **GO**: sc_mom_30d IC=0.053, 8/8 same sign, A1 5+/2- | → Handoff as secondary indicator |
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

### 當前排序（2026-02-28）

| # | 研究方向 | 目標缺口 | 整合模式 | 分散化 | 數據 | Alpha | 複雜度 | 文獻 | **總分** | 備註 |
|---|---------|---------|---------|:------:|:----:|:-----:|:------:|:----:|:--------:|------|
| ~~1~~ | ~~鏈上 regime overlay（TVL/穩定幣）~~ | — | — | — | — | — | — | — | ~~3.4~~ | **KEEP_BASELINE (20260228)**: IC=0.065, B(On-chain) SR=4.00 > A(HTF) SR=3.80, 全 gate PASS 但 Δ=+5.3% borderline。觀察期後(3/14)可重評估 |
| ~~2~~ | ~~OI regime（high/low OI 環境分類）~~ | — | — | — | — | — | — | — | ~~3.4~~ | **FAIL (20260228)**: Ablation incremental +4.66% < 5%。→ Dead Ends |
| 3 | retail_vs_top LSR standalone | 散戶擁擠 | Standalone | 4 | 4 | 3 | 3 | 2 | **3.3** | 2026 IC 翻轉 + 高換手率待解決 |
| 12 | OI 替代 HTF（架構級變更） | HTF 趨勢確認 | Filter（替換） | 2 | 5 | 4 | 2 | 2 | **3.0** | **BACKLOG**: OI standalone SR=4.12 > HTF=3.86 (+6.7%)，但為架構級替換需獨立 WFA+CPCV 全流程驗證。風險高，非緊急 |
| 4 | 真實 FR carry 改進 | Carry 品質 | 策略內部升級 | 1 | 5 | 2 | 5 | 3 | **2.7** | corr 高（改善同一策略），但簡單 |
| 5 | Order Book depth 不平衡 | 訂單流 | Overlay | 5 | 1 | 3 | 2 | 3 | **3.0** | 無歷史數據是致命問題 |
| 6 | Cross-symbol 擁擠偵測 | 系統風險 | Filter | 3 | 4 | 1 | 4 | 2 | **2.6** | 因果修正後幾乎無效 |
| 7 | 5m/15m 微結構入場 overlay | 執行改善 | Overlay | 2 | 3 | 2 | 2 | 3 | **2.3** | 成本侵蝕太大，低於門檻 |
| ~~8~~ | ~~4h TSMOM TF 替換~~ | — | — | — | — | — | — | — | ~~3.6~~ | **CLOSED (20260227)**: 修正 HTF look-ahead 後 Δ SR 僅 +0.20, PBO 52-67%。已移至 Dead Ends |
| ~~9~~ | ~~截面動量 (XSMOM)~~ | — | — | — | — | — | — | — | ~~3.7~~ | **FAIL (20260227)**: avg SR=-0.50，已移至 Dead Ends |
| ~~10~~ | ~~Taker Vol 不平衡 overlay~~ | — | — | — | — | — | — | — | ~~3.6~~ | **WEAK GO (20260227)**: IC弱(-0.006)但獨立, Δ SR+0.155, 建議作第4確認因子→Quant Dev |
| ~~11~~ | ~~CVD divergence/momentum~~ | — | — | — | — | — | — | — | ~~3.2~~ | **FAIL (20260227)**: CVD momentum 傷害 TSMOM(Δ SR=-0.25), IC 年度翻轉, 背離信號邊際 |

### 建議下一步研究（Top 2）

1. **retail_vs_top LSR standalone**（#3, 3.3 分）— IC 最強但需解決 2026 翻轉 + 換手率。
2. **真實 FR carry 改進**（#4, 2.7 分）— 低分散化但實作簡單。

### 未來 Backlog（非緊急）

- **OI 替代 HTF**（#12, 3.0 分）— OI standalone SR=4.12 > HTF=3.86，作為 HTF 的完全替代品有潛力。但這是架構級變更（移除已驗證的 HTF filter），需要獨立的 WFA+CPCV+DSR 全流程驗證 + 生產切換計劃。維持生產穩定優先，待更充分的驗證動機出現再啟動。

### 已完成 Ablation + Validation 的方向

- **On-Chain Regime Filter**（#1, KEEP_BASELINE）→ **統計驗證全 PASS 但增量 borderline**
  - **Ablation 結果**: A(HTF)=SR 3.80, B(On-chain)=SR 4.00, C(HTF+On-chain)=SR 3.88
  - **B standalone**: SR +5.3%, MDD -3.49% (best), Calmar 10.46
  - **Validation**: WFA 8/8 PASS (avg deg -3.2%, 5/8 OOS>IS), CPCV PBO max 0.13, DSR 2.28 p<0.001
  - **Verdict**: KEEP_BASELINE — 增量 borderline(+5.3%), 觀察期(→3/14)內不替換, 2/8 symbols 退化(SOL/LINK)
  - **保留決策**: Code preserved (`onchain_regime_filter()` in `filters.py`), configs 保留
  - **重評估條件**: 3/14 觀察期結束後如果 production HTF SR 衰退，可重啟 On-chain 替換流程
  - **Notebook**: `notebooks/research/20260228_onchain_regime_overlay_eda.ipynb`

- ~~**OI Regime Filter**（#2, FAIL）~~ → Ablation 結果: incremental SR +4.66% < 5% threshold。
  OI standalone (SR=4.12) 實際上比 HTF (3.86) 更強，但疊加 HTF+OI (4.04) 造成 over-filter (5/8 symbols SR 下降)。
  **不加入生產。** Code preserved in `filters.py`。
  Configs archived: `config/archive/research_oi_ablation_*.yaml`
- **Taker Vol (Smooth24) overlay** → 作為 LSR overlay 第4確認因子（`tvr_confirm_enabled`）。
  預期邊際改善 Δ SR ≈ +0.05~0.10（含成本）。低風險低收益，開發者決定是否值得實作。

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
| XSMOM 截面動量 | 2026-02-27 | 8/8 幣種 avg SR=-0.50, 6 variants 全負。residual 去 BTC 是 rank-invariant（數學等價無殘差）。blend TSMOM 也只稀釋。corr=-0.11 但負 alpha 無意義 | `config/research_xsmom.yaml`, 回測報告 `reports/futures/xsmom/` | 加密截面動量因子結構性改變（極不可能） |
| CVD momentum overlay | 2026-02-27 | CVD direction 直接 overlay 傷害 TSMOM（Δ SR=-0.251, 0/8 improved）。初步 EDA 的 IC=+0.019 不可復現（嚴格計算後 IC=+0.001）。CVD 是逆向信號而非動量信號。IC 年度翻轉（2022 負→2026 正） | `notebooks/research/20260227_taker_vol_overlay_eda.ipynb` | CVD IC 穩定化（極不可能） |
| Price-CVD divergence | 2026-02-27 | Δ SR=+0.053（邊際），turnover 2.6x baseline。IC=-0.010 且不穩定 | `notebooks/research/20260227_taker_vol_overlay_eda.ipynb` | — |
| 4h TSMOM TF 替換（1h→4h） | 2026-02-27 | 修正 HTF filter look-ahead 後，4h Pure SR 3.97 vs baseline 3.77（Δ=+0.20 僅邊際）。4h+HTF PBO 52-67%（偏高）。之前顯示的巨大改善（+1.53 SR）完全來自 HTF filter look-ahead bias。corr=0.79 高冗餘 | EDA: `scripts/archive/research_4h_tsmom_eda.py`; configs: `config/archive/research_4h_tsmom_*.yaml` | 發現新的低相關 4h 信號構造方式（極不可能，結構性冗餘） |
| BTC breakout_vol_atr blend | 2026-02-28 | Tier Ablation 5-config 研究。BTC btc_enhanced only SR=2.02 vs +breakout SR=1.99 → **breakout 是負貢獻**（Δ=-0.03）。20 params 換來負 alpha，明確 overfitting 產物。移除後 Portfolio SR 3.85 > 3.77 | `config/research_simplified_prod_candidate.yaml`, ablation configs B/C/D | — (結構性無效) |
| Tier routing phantom（tsmom_heavy ≠ default）| 2026-02-28 | confirmatory 模式下 `w_tsmom`/`w_basis_carry` 是 dead params。Config B (all default) = Config C (all tsmom_heavy)，SR 完全相同 3.57。5/8 symbols 的 tier routing 是幻象複雜度 | Ablation B vs C comparison | 改為 additive mode（但 additive 已證實較差） |

---

## 5. 更新紀錄

| 日期 | 更新內容 | 更新者 |
|------|---------|--------|
| 2026-02-27 | 初版建立：覆蓋地圖 14 維度、數據-信號圖譜 40+ 條目、研究前沿 10 方向 | Quant Developer（從歷史研究記錄彙整） |
| 2026-02-27 | XSMOM 正式回測 FAIL：avg SR=-0.50, 6 variants 全負。移至 Dead Ends。研究前沿重新排序 | Quant Developer |
| 2026-02-27 | Taker Vol overlay 深入 EDA: TVR IC=-0.006(弱逆向,獨立), CVD momentum FAIL(Δ SR=-0.25), smooth24 TVR overlay Δ SR=+0.155。WEAK GO: 建議作 LSR overlay 第4確認因子。CVD direction/divergence 移至 Dead Ends | Alpha Researcher |
| 2026-02-27 | 4h TSMOM TF Optimization EDA: IC Δ=+0.0045(6/8), gross SR 0/8 better, corr(prod,4h)=0.787, cost -4.42pp/yr。🟡 不適合 standalone 但成本節省值得正式回測 → Handoff Quant Dev | Alpha Researcher |
| 2026-02-27 | **4h TF 維度 CLOSED**: 正式回測修正 HTF look-ahead 後 Δ SR 僅 +0.20（4h Pure 3.97 vs baseline 3.77），PBO 52-67% 偏高。之前 +1.53 SR 完全來自 bias。歸檔 3 configs + EDA script → Dead Ends | Alpha Researcher |
| 2026-02-28 | **Tier Ablation 完成 + Config E 簡化候選**: 5-config ablation (A=prod, B=all default, C=all tsmom_heavy, D=BTC no breakout, E=simplified)。發現: (1) B=C 證實 w_tsmom dead param, (2) BTC breakout 是負貢獻 (SR -0.03), (3) Config E (SR 3.85) 反超 prod (SR 3.77), params -47%。BTC 720h lookback 價值 +0.38 SR。建議 Config E → 正式 validation → 替換生產 | Alpha Researcher |
| 2026-02-28 | **OI Regime Filter EDA (WEAK GO)**: 13 indicators, 8 symbols, 2022-2026。所有 IC 負值（higher OI → lower ret）。最強 IC=-0.006 < 0.01 (A5 WARN)，但 quintile spread -1.31 Sharpe（強條件效應）。F5(pctrank_720>0.3) Δ SR +0.317, 8/8 improved, freq loss 29.8%。方向交互: Long+FallingOI SR=1.50(BEST) vs Short+FallingOI SR=0.01(DEAD)。G1 FAIL, G3 PARTIAL, 其餘 PASS → WEAK GO Filter handoff | Alpha Researcher |
| 2026-02-28 | **OI Regime Filter Ablation (FAIL)**: 3-way ablation A(HTF)=3.86, B(OI)=4.12, C(HTF+OI)=4.04。Incremental SR +4.66% < 5% threshold → FAIL。Key findings: (1) OI standalone (4.12) 實際上比 HTF (3.86) 更強 (+6.74%), (2) 但 HTF+OI 疊加造成 over-filter, 5/8 symbols SR 下降, (3) C 的 MDD (-3.14%) 最佳但 return 最低。OI 與 HTF 部分冗餘（兩者都過濾低conviction信號）。Code preserved, configs archived | Quant Developer |
| 2026-02-28 | **OI 替代 HTF → BACKLOG #12**: 用戶決策維持生產穩定（Option C），OI 替代 HTF 為架構級變更，記為未來 backlog（需獨立 WFA+CPCV 全流程驗證）。研究前沿 #2 標記 FAIL，新增 #12 BACKLOG 項目 | Alpha Researcher |
| 2026-02-28 | **On-Chain Regime Overlay EDA (GO)**: 18 indicators (TVL/穩定幣), 8 symbols, 2020-2026。tvl_mom_30d IC=0.065 (>10× OI), A1-A5 全 PASS (8/8 same sign, 5+/2-)。tvl_sc_ratio_mom_30d quintile spread +4.69 (monotonic!)。Filter ≥P30: 8/8 improved, avg Δ SR=+0.409。Risk-On/Off: 8/8 Risk-On better, avg Δ=+1.454。G6: avg |corr|=0.302 (partially independent)。**6/6 G gates PASS → GO** → Handoff Quant Dev with mandatory ablation | Alpha Researcher |
| 2026-02-28 | **On-Chain Regime Filter Ablation + Validation (KEEP_BASELINE)**: lookback bug 修正(365→720)。3-way ablation: A(HTF)=3.80, B(On-chain)=4.00(+5.3%), C(HTF+On-chain)=3.88。B 全 validation PASS: WFA 8/8 (avg deg -3.2%, 5/8 OOS>IS), CPCV PBO max 0.13, DSR 2.28 p<0.001。prado_methods.py CPCV bug 修正(probability_of_backtest_overfitting→_simplified_pbo_estimate)。**Verdict: KEEP_BASELINE** — 增量 borderline, 觀察期內不替換, code preserved 供 3/14 後重評估 | Quant Developer + Quant Researcher |