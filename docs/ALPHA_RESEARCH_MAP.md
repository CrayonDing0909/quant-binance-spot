> **Last updated**: 2026-03-05 (Production 6-Symbol deploy 2026-03-04, observation → 2026-03-18. ADA/BNB removed (no IC), ETH simplified, vol-based slippage enabled. Portfolio SR=3.17 realistic cost)

# Alpha 研究地圖 (Alpha Research Map)

Alpha Researcher **開始任何新研究前必讀**的結構化知識庫。
包含三大區塊：Alpha 覆蓋地圖、數據-信號分類圖譜、研究前沿排序表。

> **維護方式**：Alpha Researcher 每次研究 session 結束後必須更新本文件。
> 其他 agent 發現新的數據來源或策略狀態變更時，也應同步更新。

---

## 1. Alpha 覆蓋地圖

當前生產組合已捕捉的 alpha 維度、覆蓋品質、以及尚未填補的缺口。

**生產策略**：HTF Filter v2 + LSR (Simplified v2) — Meta-Blend **6-Symbol** (BTC, ETH, SOL, DOGE, AVAX, LINK) + HTF Filter + LSR Confirmatory Overlay, 3x leverage
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
| 17 | Volume Profile（va_width_pct filter）| 1h OHLCV 構建 VP | ✅ EDA + Ablation → **FAIL** | ❌ 無效 | va_width_pct IC=0.024 但本質是 vol regime proxy (corr=0.71)。Ablation: VP gate SR=3.23 > Vol gate SR=2.92 (7/8 wins) 但兩者都劣於 baseline SR=3.62 (-10.9%/-19.5%)。既有 HTF filter 已充分捕捉 vol regime。方向關閉 |
| 18 | VPIN regime filter | aggTrades → volume clock | ✅ EDA+Ablation → **KEEP_BASELINE** (20260302) | ★★★☆☆ 中 | **KEEP_BASELINE**: B(VPIN only) SR 3.98 +3.5% 7/8✅ (歷史第二強 standalone), C(HTF+VPIN) SR 3.80 -1.2% 4/8✅。正交性(corr=0.025)減輕 over-filter(-1.2% vs macro -9.8%) 但仍無法完全克服。Code preserved |
| 19 | Macro Cross-Market Regime | GLD/VIX/DXY daily (yfinance) | EDA GO → Ablation **KEEP_BASELINE** (20260301) | ★★★★☆ 強 | **KEEP_BASELINE**: B(Macro only) SR=2.525 > A(HTF) SR=2.413 (+4.6%, 7/8✅), C(HTF+Macro) SR=2.177 (-9.8%, 0/8✅)。Standalone borderline (+4.6% < 5%), stacking severely degrades。同 OI/On-chain 的 over-filter 模式。Code preserved, 觀察期後(3/14)可評估 HTF 替換 |
| 20 | Entropy Regime (PE/SE/ApEn) | 1h OHLCV close (已有數據) | ✅ EDA → **FAIL** (20260302) | ❌ 無效 | 7/7 entropy 全通過 confounding (|corr(vol)|<0.2)，但所有 IC<0.01 (A5 FAIL)。最強 pe_720 avg IC=-0.0024。Binary filter 3/8 improved。加密 1h 序列 entropy 無預測力。方向關閉 |
| 21 | Orderflow Composite Standalone | taker_vol_ratio → OFI+VPIN+CVD | ✅ EDA+Backtest → **FAIL** (20260302) | ❌ 無效 | **首次獨立策略嘗試（非 filter/overlay）**。corr(TSMOM)=-0.023（極低，最佳分散化候選）。但 IC(24h)=+0.003 太弱，pre-cost SR=0.378, MDD=-55.5%。Post-cost SR 深度負值（年交易成本 ~143% >> 回報 ~5.7%）。taker_vol_ratio 1h proxy 不含足夠 alpha 支撐獨立策略 |
| 22 | Tick-level Avg Trade Size | aggTrades → total_vol/num_trades | ✅ EDA → **WEAK GO** (20260305) | ★★★☆☆ 中 | **avg_trade_size IC=-0.030** (7/8 same sign, A5 PASS)。corr(TSMOM)=0.04, corr(ATR)=0.25（正交）。A1 4/7 FAIL（2022/2025/2026 弱正）。Alpha 不在 OFI 方向而在 trade SIZE（whale distribution）。OFI 衍生特徵全 FAIL (IC<0.01) → Handoff Quant Dev ablation |

### 維度覆蓋摘要

- **已充分覆蓋（★★★★+）**：時序動量、HTF 趨勢、散戶擁擠、OI 事件
- **部分覆蓋（★★-★★★）**：Carry、OI 確認、波動率、清算
- **已確認無效**：截面動量 (XSMOM)、TF 優化（4h 替換 1h，修正 look-ahead 後 Δ SR 僅 +0.20, PBO 偏高）
- **已測試 WEAK GO（★★）**：微結構/訂單流（TVR 獨立但 IC 弱）
- **已測試 FAIL**：OI Regime Gate（standalone SR=4.12 > HTF=3.86，但 incremental +4.66% < 5%，與 HTF 疊加 over-filter）、Volume Profile va_width_pct（IC=0.024 但 corr(vol)=0.71, ablation VP gate SR=3.23 < baseline SR=3.62, vol regime proxy 無增量）
- **已測試 KEEP_BASELINE（★★★★）**：鏈上 regime（On-chain only SR=4.00 > HTF SR=3.80, 全 gate PASS，但增量 borderline +5.3%，觀察期內不替換）、Macro Cross-Market Regime（B(Macro) SR=2.525 > A(HTF) SR=2.413 +4.6% 7/8✅，但 C(HTF+Macro) SR=2.177 -9.8%。Standalone borderline, stacking over-filters）
- **已確認 VP 全線無效**：傳統 VP S/R（POC dist, VA pos, VP skew）IC<0.01。唯一有效信號 va_width_pct 本質是 vol regime proxy，ablation 確認無超越 vol 的增量。整個 VP 研究方向關閉
- **已確認 Entropy 無效（★☆☆☆☆）**：PE/SE/ApEn 3 種 entropy × 3 lookbacks = 9 指標。Confounding OK (|corr(vol)|<0.2)，但所有 IC<0.01。加密 1h 序列太接近 random walk，entropy 微小波動不含有用信息
- **已確認 Orderflow Composite 獨立策略無效（★☆☆☆☆）**：taker_vol_ratio 1h proxy → OFI+VPIN+CVD 組合。corr(TSMOM)=-0.023（歷來最佳分散化），但 IC=+0.003 太弱，pre-cost SR=0.378，交易成本遠超 alpha。taker_vol_ratio proxy 解析度不足
- **Tick OFI direction 確認無效，但 avg_trade_size WEAK GO（★★★）**：tick-level OFI IC=-0.011（僅 2x proxy），所有 OFI 衍生特徵 IC<0.01。但 avg_trade_size IC=-0.030（5x proxy），corr(TSMOM)=0.04（完全正交），7/8 same sign → 鯨魚交易行為是真正的 microstructure alpha
- **已測試 KEEP_BASELINE（★★★）**：VPIN regime（B(VPIN only) SR 3.98 +3.5% **7/8**✅ — 歷史第二強 standalone。C(HTF+VPIN) SR 3.80 -1.2% 4/8 — 正交性減輕 over-filter 但仍無法克服。Code preserved, pipeline 已建）
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
| Taker Vol Ratio | **Orderflow Composite Standalone** (OFI+VPIN+CVD) | ✅ 是 (20260302 EDA+BT) | ❌ **FAIL**: composite IC(24h)=+0.003, **corr(TSMOM)=-0.023**(極低!), SR=0.378(pre-cost), MDD=-55.5%, post-cost 深度虧損 | **FAIL** — 1h taker_vol_ratio proxy alpha 不足。Code preserved: `orderflow_composite_strategy.py` |
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
| 1h OHLCV | Volume Profile（POC dist/VA pos/VP skew） | ✅ 是 (20260228 EDA) | ❌ **IC<0.01**: POC dist IC=-0.004, VA pos IC=+0.006, VP skew IC=+0.002。傳統 VP S/R 在 1h 框架下無預測力 | 無 |
| 1h OHLCV | Volume Profile va_width_pct (VA 寬度) | ✅ 是 (EDA 20260228 + Ablation 20260301) | ❌ **FAIL**: IC=+0.024 但 corr(vol)=0.71。Ablation: VP gate SR=3.23 > Vol gate SR=2.92 (7/8 wins) 但兩者都劣於 baseline SR=3.62。VP 是 vol proxy，無增量 alpha | 方向關閉 |
| 1h OHLCV close | Entropy regime (PE/SE/ApEn) | ✅ 是 (20260302 EDA) | ❌ **FAIL**: 全部 IC<0.01, confounding OK (|corr(vol)|<0.2) 但無預測力 | 方向關閉 |
| aggTrades (tick) | VPIN (Volume-Sync Probability Informed Trading) | ✅ 是 (EDA+Ablation 20260302) | **KEEP_BASELINE**: EDA IC=0.005, corr(ATR)=0.025 (完全正交!)。Ablation: B(VPIN only) SR 3.98 +3.5% 7/8✅, C(HTF+VPIN) SR 3.80 -1.2% 4/8。正交性減輕 over-filter(-1.2% vs macro -9.8%) 但仍不足 | Pipeline built: `agg_trades.py`. Code preserved (`vpin_regime_filter()` in `filters.py`) |
| aggTrades (tick) | OFI (Order Flow Imbalance) — tick-level 版 | ✅ 是 (20260305 EDA) | ❌ **FAIL**: 所有 OFI 衍生特徵 IC<0.01 | Cont et al. 2014。tick-level OFI IC=-0.011 (2x proxy -0.006) 但仍 <0.01。ofi_mom/cum/zscore 全 FAIL。**Alpha 不在 flow direction 而在 trade size** |
| aggTrades (tick) | avg_trade_size (whale detection) | ✅ 是 (20260305 EDA) | **WEAK GO**: IC=-0.030, 7/8 same sign, corr(TSMOM)=0.04 | **最強 aggTrades 特徵**。Higher avg trade size → lower future returns (smart money distribution)。A1 4/7 FAIL。→ Handoff Quant Dev ablation |
| GLD daily (yfinance) | GLD_mom_60d risk-off filter | ✅ 是 (20260301 EDA) | **GO**: IC=-0.049, 8/8 same sign, 5/5 gates, corr(BTC_mom)=0.125, corr(TVL)=0.034 | → Handoff Quant Dev as Filter |
| VIX daily (yfinance) | VIX_mom_30d contrarian recovery | ✅ 是 (20260301 EDA) | **GO**: IC=+0.043, **8/8 years consistent**, corr(GLD)=0.053(正交!) | → Combined with GLD as Macro Regime Filter |
| DXY daily (yfinance) | DXY_mom_90d dollar strength | ✅ 是 (20260301 EDA) | **WEAK GO**: IC=-0.040, 5/8 years consistent, corr(BTC_mom)=0.33 偏高 | 候補（DXY 與 BTC momentum 較冗餘） |
| GLD+VIX combined | Macro risk-off composite z-score | ✅ 是 (EDA 20260301 + Ablation 20260301) | **KEEP_BASELINE**: EDA IC=+0.056, 8/8 positive。Ablation: B(Macro)=2.525 > A(HTF)=2.413 (+4.6%, 7/8✅), C(HTF+Macro)=2.177 (-9.8%, 0/8)。Standalone borderline, stacking over-filters | Code preserved (`macro_regime_filter()` in `filters.py`) |

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

### 當前排序（2026-03-05）

| # | 研究方向 | 目標缺口 | 整合模式 | 分散化 | 數據 | Alpha | 複雜度 | 文獻 | **總分** | 備註 |
|---|---------|---------|---------|:------:|:----:|:-----:|:------:|:----:|:--------:|------|
| ~~15~~ | ~~Macro Cross-Market Regime (GLD+VIX)~~ | — | — | — | — | — | — | — | ~~4.3~~ | **KEEP_BASELINE (20260301)**: Ablation B(Macro only)=2.525 > A(Baseline)=2.413 (+4.6%, 7/8✅), C(HTF+Macro)=2.177 (-9.8%, 0/8✅)。Standalone borderline (+4.6% < 5%), stacking severely degrades (-9.8%)。同 OI/On-chain 的 over-filter 模式 |
| ~~1~~ | ~~鏈上 regime overlay（TVL/穩定幣）~~ | — | — | — | — | — | — | — | ~~3.4~~ | **KEEP_BASELINE (20260228)**: IC=0.065, B(On-chain) SR=4.00 > A(HTF) SR=3.80, 全 gate PASS 但 Δ=+5.3% borderline。觀察期後(3/14)可重評估 |
| ~~2~~ | ~~OI regime（high/low OI 環境分類）~~ | — | — | — | — | — | — | — | ~~3.4~~ | **FAIL (20260228)**: Ablation incremental +4.66% < 5%。→ Dead Ends |
| 3 | retail_vs_top LSR standalone | 散戶擁擠 | Standalone | 4 | 4 | 3 | 3 | 2 | **3.3** | 2026 IC 翻轉 + 高換手率待解決 |
| 12 | OI 替代 HTF（架構級變更） | HTF 趨勢確認 | Filter（替換） | 2 | 5 | 4 | 2 | 2 | **3.0** | **BACKLOG**: OI standalone SR=4.12 > HTF=3.86 (+6.7%)，但為架構級替換需獨立 WFA+CPCV 全流程驗證。風險高，非緊急 |
| 4 | 真實 FR carry 改進 | Carry 品質 | 策略內部升級 | 1 | 5 | 2 | 5 | 3 | **2.7** | corr 高（改善同一策略），但簡單 |
| 5 | Order Book depth 不平衡 | 訂單流 | Overlay | 5 | 1 | 3 | 2 | 3 | **3.0** | 無歷史數據是致命問題 |
| 6 | Cross-symbol 擁擠偵測 | 系統風險 | Filter | 3 | 4 | 1 | 4 | 2 | **2.6** | 因果修正後幾乎無效 |
| ~~13~~ | ~~Volume Profile POC/VA filter~~ | — | — | — | — | — | — | — | ~~3.6~~ | **FAIL (20260301 ablation)**: VP gate SR=3.23 > Vol gate SR=2.92 (7/8 wins) 但兩者都劣於 baseline SR=3.62。VP 是 vol regime proxy 無增量。→ Dead Ends |
| ~~14~~ | ~~VPIN regime indicator~~ | — | — | — | — | — | — | — | ~~3.3~~ | **KEEP_BASELINE (20260302)**: B(VPIN only) SR 3.98 +3.5% **7/8**✅（歷史第二強 standalone）, C(HTF+VPIN) SR 3.80 **-1.2%** 4/8（仍 over-filter）。正交性減輕 stacking 退化(-1.2% vs macro -9.8%) 但無法消除。Code preserved, 數據管線已建 |
| 7 | 5m/15m 微結構入場 overlay | 執行改善 | Overlay | 2 | 3 | 2 | 2 | 3 | **2.3** | 成本侵蝕太大，低於門檻 |
| ~~8~~ | ~~4h TSMOM TF 替換~~ | — | — | — | — | — | — | — | ~~3.6~~ | **CLOSED (20260227)**: 修正 HTF look-ahead 後 Δ SR 僅 +0.20, PBO 52-67%。已移至 Dead Ends |
| ~~9~~ | ~~截面動量 (XSMOM)~~ | — | — | — | — | — | — | — | ~~3.7~~ | **FAIL (20260227)**: avg SR=-0.50，已移至 Dead Ends |
| ~~10~~ | ~~Taker Vol 不平衡 overlay~~ | — | — | — | — | — | — | — | ~~3.6~~ | **WEAK GO (20260227)**: IC弱(-0.006)但獨立, Δ SR+0.155, 建議作第4確認因子→Quant Dev |
| ~~11~~ | ~~CVD divergence/momentum~~ | — | — | — | — | — | — | — | ~~3.2~~ | **FAIL (20260227)**: CVD momentum 傷害 TSMOM(Δ SR=-0.25), IC 年度翻轉, 背離信號邊際 |
| ~~16~~ | ~~Orderflow Composite Standalone (taker_vol_ratio)~~ | — | — | — | — | — | — | — | ~~3.8~~ | **FAIL (20260302)**: 首次獨立策略研究（OFI+VPIN+CVD）。corr(TSMOM)=-0.023（極低 → 分散化 5/5），但 IC=+0.003 太弱（alpha 1/5）。Pre-cost SR=0.378, MDD=-55.5%。Post-cost 深度負值。taker_vol_ratio 1h proxy 解析度不足 |

### 建議下一步研究（Top 3）

1. **avg_trade_size Filter/Overlay Ablation**（#22 WEAK GO → Quant Dev）— IC=-0.030（歷來 aggTrades 最強），corr(TSMOM)=0.04（幾乎完全正交），7/8 same sign。Alpha 在 trade SIZE 而非 OFI direction。**關鍵發現：tick OFI IC 僅微幅提升（-0.011 vs proxy -0.006），但 avg_trade_size IC 是 5x（-0.030）**。A1 4/7 是唯一弱點。需 ablation: A(HTF only), B(avg_trade_size only), C(HTF+avg_trade_size)。
2. **Filter Replacement 框架**（觀察期後 →2026-03-18）— 所有 4 個 filter (OI/On-chain/Macro/VPIN) + avg_trade_size 疊加 HTF 都退化，但 standalone 均優於 HTF baseline。問題是 gate math（多重 gate 降低 time-in-market）。**下一步：用最強的 standalone signal 完全替換 HTF**，而非疊加。
3. **retail_vs_top LSR standalone**（#3, 3.3 分）— IC 最強但需解決 2026 翻轉 + 換手率。

> **關鍵洞察（20260305 Tick OFI EDA）**: 經過 6 輪 microstructure 研究（proxy OFI, VPIN, Entropy, Orderflow Composite, VP, **tick OFI**），最終發現 alpha 不在 order flow **方向** (OFI/CVD) 而在 order flow **結構** (avg_trade_size)。tick-level OFI 僅比 proxy 好 2x (IC -0.011 vs -0.006)，仍不足 A5 門檻。但 avg_trade_size（total_vol/num_trades = 鯨魚偵測）IC=-0.030，是所有 microstructure 信號中最強且與 TSMOM 完全正交。
>
> OI (+6.7%)、On-chain (+5.3%)、Macro (+4.6%) 三者 standalone 均優於 HTF，但均未跨 5% threshold，且疊加會 over-filter。這暗示 HTF filter 的功能可能被更精確的單一信號替代，但需要 "filter replacement" 而非 "filter stacking" 的研究框架。觀察期(→3/18)後可統一評估。

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

- **Macro Cross-Market Regime Filter**（#15, KEEP_BASELINE）→ **Standalone borderline, stacking degrades**
  - **Ablation 結果**: A(Baseline/HTF)=SR 2.413, B(Macro only)=SR 2.525(+4.6%), C(HTF+Macro)=SR 2.177(-9.8%)
  - **B standalone**: 7/8 symbols improved (only BNB slightly worse), +4.6% but < 5% threshold
  - **C stacking**: 0/8 symbols improved, severe over-filter (-9.8%)
  - **Pattern**: 同 OI (+4.66% incremental FAIL) 和 On-chain (+5.3% borderline) — 三者 standalone 都比 HTF 好，但疊加都退化
  - **Verdict**: KEEP_BASELINE — standalone borderline, stacking unusable
  - **保留決策**: Code preserved (`macro_regime_filter()` in `filters.py`), data cached (`data/macro/`)
  - **重評估條件**: 3/14 觀察期後如果 HTF SR 衰退，可與 OI/On-chain 一起評估 "filter replacement" 方案
  - **Script**: `scripts/research_macro_regime_ablation.py`
  - **Report**: `reports/research/macro_regime_ablation/ablation_results.json`

- **VPIN Regime Filter**（#14, KEEP_BASELINE）→ **正交性減輕 over-filter 但不足以消除**
  - **EDA 結果**: corr(ATR)=0.025（歷來最佳 confounding）, IC=0.005 (<0.01), Q1 SR=1.85 vs Q5 SR=0.87
  - **Ablation 結果**: A(HTF only)=SR 3.846, B(VPIN only)=SR 3.980(+3.5%), C(HTF+VPIN)=SR 3.798(-1.2%)
  - **B standalone**: 7/8 symbols improved (only BNB slightly worse -0.026), Return +500% vs A +391%
  - **C stacking**: 4/8 improved, 正交性確實減輕退化(-1.2% vs Macro -9.8%, OI over-filter), 但仍為負
  - **關鍵發現**: 即使 VPIN ⊥ vol (corr=0.025)，stacking 仍然不 work → 問題是 gate math（多重 gate 降低 time-in-market）而非 signal redundancy
  - **歷史比較**: VPIN standalone +3.5% 是 4 個 filter 中最弱（OI +6.7% > On-chain +5.3% > Macro +4.6% > VPIN +3.5%），但 stacking 退化最小（VPIN -1.2% vs Macro -9.8%）
  - **Verdict**: KEEP_BASELINE — standalone < 5%, stacking 無效
  - **保留決策**: Code preserved (`vpin_regime_filter()` in `filters.py`), pipeline built (`agg_trades.py`)
  - **Script**: `scripts/research_vpin_ablation.py`
  - **Report**: `reports/research/vpin_ablation/ablation_results.json`

- ~~**Volume Profile va_width_pct filter**（#13, FAIL）~~ → Ablation: VP gate SR=3.23 > Vol gate SR=2.92 (7/8 wins)，
  但兩者都劣於 baseline SR=3.62 (-10.9%/-19.5%)。VP 本質是 vol regime proxy (corr=0.71)，
  既有 HTF filter 已充分處理 vol regime。**方向關閉。**
  Ablation report: `reports/research/vp_vs_vol_ablation/ablation_results.json`

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
| Volume Profile va_width_pct filter | 2026-03-01 | EDA: IC=0.024 (8/8 same sign, A1-A5 全 PASS), 但 corr(realized_vol)=0.71。Ablation: VP gate SR=3.23 > Vol gate SR=2.92 (7/8 wins) 但兩者**劣於 baseline SR=3.62** (-10.9%/-19.5%)。VP 含些微超越 vol 的信息但無生產增量。傳統 VP S/R (POC/VA pos/VP skew) IC<0.01 全 FAIL | EDA: `notebooks/research/20260228_volume_profile_eda.ipynb`; Ablation: `reports/research/vp_vs_vol_ablation/` | VP 構造性超越 TSMOM+HTF 的 vol regime 處理（極不可能） |
| Entropy Regime Indicator (PE/SE/ApEn) | 2026-03-02 | 3 entropy measures × 3 lookbacks = 9 indicators。**Confounding OK** (|corr(vol)|<0.2, 與 VP corr=0.71 不同)。但**所有 IC<0.01** (A5 FAIL): pe_720 IC=-0.0024, pe_168 IC=-0.0021, se_720 IC=+0.0005。Binary filter 3/8 improved (WEAK)。加密 1h 收益序列的 entropy 太接近 random walk, ordinal patterns 和 return distribution 的微小波動不含有用信息。Entropy 是獨立維度（非 vol proxy）但無 alpha | Script: `scripts/archive/research_entropy_regime_eda.py`; Report: `reports/research/entropy_regime_eda/` | 加密市場結構性改變使 1h entropy regime 成為有效信號（極不可能） |
| Orderflow Composite Standalone (taker_vol_ratio proxy) | 2026-03-02 | **首次獨立策略嘗試**（OFI+VPIN+CVD 組合，非 filter/overlay）。taker_vol_ratio 1h proxy IC(24h)=+0.003（太弱）。contrarian 好於 momentum（8/8 positive SR），但 pre-cost SR=0.378, MDD=-55.5%。Post-cost 深度虧損：threshold positioning 降 turnover 63% 後仍年化交易成本 ~143% >> 年化回報 ~5.7%。**正面發現：corr(TSMOM)=-0.023 是歷來所有候選中最低**，證實微結構維度與 TSMOM 幾乎完全正交。失敗根因是數據解析度：小時級聚合 ratio 丟失大量 tick-level 信息 | Script: `scripts/archive/research_orderflow_composite_eda.py`; Config: `config/archive/research_orderflow_composite.yaml`; Report: `reports/research/orderflow_composite/`; Strategy: `src/qtrade/strategy/orderflow_composite_strategy.py` (preserved) | tick-level aggTrades OFI（數據已就緒，未測）可能更強；或更高頻 (5m/15m) 信號 + 降頻入場 |

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
| 2026-02-28 | **Order Flow / AMT / Volume Profile 文獻調研**: 新增覆蓋地圖 #17 (Volume Profile POC/VA filter) + #18 (VPIN regime filter)。新增數據-信號圖譜 3 條目 (VP/VPIN/OFI)。研究前沿新增 #13 (VP, 3.6分, 最高候選) + #14 (VPIN, 3.3分)。`RESEARCH_LITERATURE.md` Microstructure 區塊大幅擴充（AMT 經典 4 本 + OFI/VPIN 學術論文 6 篇 + 加密專屬 3 篇 + 教科書 2 本）| Alpha Researcher |
| 2026-02-28 | **Volume Profile EDA (WEAK GO)**: 5 VP indicators × 3 lookbacks × 8 symbols。**va_width_pct IC=0.024 (8/8 same sign, A1-A5 全 PASS, quintile spread +3.84)**。但 G3 混淆因子分析: corr(realized_vol)=0.80, VP 本質是 vol regime proxy。傳統 VP S/R 信號 (POC dist, VA pos, VP skew) IC 全<0.01 FAIL。Filter 效果: Above VA Long SR=2.14 >> Inside VA 0.64。**Verdict: WEAK GO** — va_width_pct 需 ablation vs simple vol filter | Alpha Researcher |
| 2026-03-01 | **VP va_width_pct vs Vol Ablation (FAIL)**: 3-way ablation A(baseline)=SR 3.62, B(VP gate)=SR 3.23(-10.9%), C(vol gate)=SR 2.92(-19.5%)。VP > Vol 在 7/8 symbols，但兩者都劣於 baseline → VP 無生產增量。corr(VP,vol)=0.71。VP 是 vol regime proxy，既有 HTF filter 已充分覆蓋。方向關閉，移至 Dead Ends。Research frontier #13 FAIL | Quant Developer |
| 2026-03-01 | **Macro Cross-Market Regime EDA (GO)**: 26 indicators (VIX/SPY/QQQ/DXY/US10Y/GLD × momentum/pctrank), 8 symbols, 2019-2026。**GLD_mom_60d IC=-0.049** (gold rally=risk-off→crypto weak, 8/8 same sign, 6/8yr, **corr(BTC_mom)=0.125 corr(TVL)=0.034 最獨立**)。**VIX_mom_30d IC=+0.043** (8/8 years consistent! contrarian fear recovery)。GLD⊥VIX(corr=0.053)→**Combined IC=0.056**(>individual)。Binary filter block bottom 20%: **8/8 improved Δ SR +0.20~0.61**, Risk-On avg SR=1.04 vs Risk-Off avg SR=-1.24。5/5 quality gates PASS。5-factor score **4.3**（最高分候選）。→ **GO, Handoff Quant Dev mandatory ablation** | Alpha Researcher |
|| 2026-03-01 | **Macro Cross-Market Regime Ablation (KEEP_BASELINE)**: 3-way ablation A(HTF baseline)=SR 2.413, B(Macro only)=SR 2.525(+4.6%, 7/8✅), C(HTF+Macro)=SR 2.177(-9.8%, 0/8✅)。B standalone borderline (+4.6% < 5%)。C stacking severely degrades。同 OI/On-chain 的 over-filter 模式。Code preserved (`macro_regime_filter()` in `filters.py`), 觀察期後(3/14)可評估 filter replacement | Quant Developer |
| 2026-03-02 | **Entropy Regime Indicator EDA (FAIL)**: 3 entropy measures (PE/SE/ApEn) × 3 lookbacks (24/168/720h), 8 symbols, 2020-2026。**Confounding 全通過** (7/7 survive, max |corr(vol)|=0.16 — 與 VP corr=0.71 完全不同)。但 **所有 IC<0.01 (A5 FAIL)**：最強 pe_720 avg IC=-0.0024, 7/8 same sign(A3 OK)。Year-by-year: 5/7 consistent(A1 OK), shift impact -4.3%(A2 OK)。Quintile spread Q1-Q5=+0.21(方向正確但極弱, 0/8 monotonic)。Binary filter(block P80): 3/8 improved。**結論：Entropy ≠ vol proxy（好消息），但 entropy 對 crypto 1h return 根本沒預測力（壞消息）。方向關閉** | Alpha Researcher |
| 2026-03-02 | **VPIN Regime EDA (WEAK GO)**: aggTrades pipeline 建完 + 8 symbols 2020-2026。**Confounding OUTSTANDING**: corr(VPIN, ATR_pctrank)=0.025 (歷來最佳)。IC=0.005 (<0.01 A5 FAIL), 7/8 same sign (A3 PASS), vpin_delta_6 6/7yr (A1 PASS)。Quintile Q1 SR=1.85 vs Q5 SR=0.87。Filter P70: ΔSR +0.12, 6/8✅。VPIN IC < ATR IC，但完全正交。**5/6 gates PASS → WEAK GO**。正交性使 HTF+VPIN 疊加可能不重蹈 over-filter 覆轍 → Handoff Quant Dev ablation | Alpha Researcher |
| 2026-03-02 | **VPIN Regime Filter Ablation (KEEP_BASELINE)**: 3-way ablation A(HTF only)=SR 3.846, B(VPIN only)=SR 3.980(+3.5%, **7/8**✅), C(HTF+VPIN)=SR 3.798(-1.2%, 4/8)。**正交性假設部分驗證**：VPIN ⊥ vol 確實減輕 stacking 退化（-1.2% vs macro -9.8%），但仍無法完全消除。**所有 4 個 filter (OI/On-chain/Macro/VPIN) standalone 均優於 HTF 但 stacking 均退化** → 問題是 gate math（多重 gate 降低 time-in-market）而非 signal redundancy。Code preserved, pipeline built。Configs archived | Quant Developer |
| 2026-03-02 | **Orderflow Composite Standalone Strategy (FAIL)**: 首次嘗試獨立策略（非 filter/overlay）。taker_vol_ratio 1h proxy → OFI+VPIN+CVD 3-signal composite。**正面發現：corr(TSMOM)=-0.023**（歷來所有研究中最低，證實微結構 ⊥ TSMOM）。**負面：IC(24h)=+0.003 太弱**（A5 FAIL），pre-cost SR=0.378，MDD=-55.5%。Contrarian mode 8/8 positive SR（fade the flow 方向正確）。Threshold positioning 降 turnover 63%（~5,500 trades/sym/4yr）仍不足（年化成本 ~143% >> alpha ~5.7%）。**根因：taker_vol_ratio 小時級聚合丟失 tick 信息**，非策略邏輯問題。Strategy code preserved (`orderflow_composite_strategy.py`)，script+config archived。研究前沿更新：tick-level OFI（數據已就緒）替代 proxy 版為下一步 | Alpha Researcher |
| 2026-03-05 | **Tick-level OFI EDA (#22 WEAK GO)**: 8 symbols, aggTrades 2020-2026。**關鍵發現：alpha 不在 OFI direction 而在 trade SIZE**。avg_trade_size IC=-0.030 (7/8 same sign, corr(TSMOM)=0.04, corr(ATR)=0.25)。所有 OFI 衍生特徵 FAIL (IC<0.01): tick OFI IC=-0.011 (僅 2x proxy -0.006), ofi_mom/cum/zscore 全弱。avg_trade_size A1 4/7 FAIL（唯一弱點）。6-Symbol 部署更新 + FutureWarning fix | Alpha Researcher |