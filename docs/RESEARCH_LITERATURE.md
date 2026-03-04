> **Last updated**: 2026-03-02

# 量化策略學術文獻參考庫

本文件是 Alpha Researcher 的文獻知識庫，按策略原型分類。
每個原型包含經典文獻和加密貨幣專屬研究。

> **維護方式**：任何 agent 在研究過程中發現有價值的新文獻，應在 session 結束前將其加入本檔案對應的原型分類下。
> 格式：`- 作者 (年份) "標題" — 一句話摘要`

---

## Trend Following（趨勢追蹤）

### 經典文獻
- Moskowitz, Ooi & Pedersen (2012) "Time Series Momentum" — TSMOM 跨資產類別有效，正偏態收益分布
- Baltas & Kosowski (2013) "Momentum Strategies in Futures Markets" — lookback 敏感度分析、信號速度與衰減
- Lempérière et al. (2014) "Two Centuries of Trend Following" — 趨勢 alpha 持久但單資產 Sharpe 低（需組合）
- Hurst, Ooi & Pedersen (2017) "A Century of Evidence on Trend-Following Investing" — 趨勢在所有資產類別和時期均有效

### 加密貨幣專屬
- Bianchi & Babiak (2022) "Cryptocurrencies as an Asset Class? An Empirical Assessment" — 加密資產的動量效應顯著
- Liu & Tsyvinski (2021) "Risks and Returns of Cryptocurrency" — 加密市場動量因子有效，但 3 週後衰減

### 關鍵洞察
- 加密貨幣收益分布具有**正偏態 + 肥尾**，結構性有利於趨勢追蹤（捕捉右尾）
- 單資產 TSMOM 的 Sharpe 通常 < 1.0，多幣種組合可提升至 1.5+
- lookback window 對加密貨幣的最佳範圍通常在 48h-336h（比傳統資產更短）

---

## Mean Reversion（均值回歸）

### 經典文獻
- Brock, Lakonishok & LeBaron (1992) "Simple Technical Trading Rules and the Stochastic Properties of Stock Returns" — BB 策略在股票市場的早期驗證
- Connors & Alvarez (2009) "Short-Term Trading Strategies That Work" — RSI MR，70%+ 勝率但需嚴格倉位管理
- Avellaneda & Lee (2010) "Statistical Arbitrage in the U.S. Equities Market" — Ornstein-Uhlenbeck 框架，高換手率是敵人

### 加密貨幣專屬
- Caporale & Plastun (2020) "Momentum and Contrarian Effects in the Cryptocurrency Market" — 加密短期反轉存在但微弱；正偏態扼殺 MR
- Makarov & Schoar (2020) "Trading and Arbitrage in Cryptocurrency Markets" — 跨交易所 MR 存在但手續費和延遲吃掉 edge

### 關鍵洞察
- **MR 鐵律**：先用帶 TP/SL 的模擬計算每筆 gross PnL，IC 為正不代表策略可行
- 加密市場正偏態 + 肥尾結構性懲罰 MR（被右尾反向持倉擊殺）
- 高換手率（300-1500 次/年）使成本成為致命問題
- 經驗：2026-02 對 8 幣種 BB MR 回測，100% gross PnL 為負

---

## Carry / Yield（收益率策略）

### 經典文獻
- Koijen, Moskowitz, Pedersen & Vrugt (2018) "Carry" — carry 作為跨資產統一風險因子
- Lustig, Roussanov & Verdelhan (2011) "Common Risk Factors in Currency Markets" — 貨幣 carry trade 的風險分解

### 加密貨幣專屬
- Funding rate carry 具有 regime 依賴性；擁擠的 carry 交易反轉劇烈
- 經驗：2026-02 研究顯示 SOL/BNB 2 年期 FR 均值 < 0，portfolio SR = -0.63

### 關鍵洞察
- Funding rate 在牛市（正 FR 穩定）表現最好，在 regime shift 時迅速反轉
- 需要「擁擠指數」來偵測 carry trade 過度擁擠的風險
- 單純 carry 的 Sharpe 低，通常作為趨勢策略的輔助因子（如 tsmom_carry_v2）

---

## Volatility（波動率策略）

### 經典文獻
- Mandelbrot (1963) "The Variation of Certain Speculative Prices" — 波動率叢集現象
- Bollerslev (1986) "Generalized Autoregressive Conditional Heteroskedasticity" — GARCH 波動率模型
- Cont (2001) "Empirical Properties of Asset Returns" — 波動率叢集的實證特徵

### 加密貨幣專屬
- ATR 百分位 squeeze 偵測在加密市場有效；vol regime 轉換比傳統市場更快
- 經驗：2026-02 研究顯示 Vol Squeeze 4/8 幣種通過，但 edge 勉強覆蓋成本

### 關鍵洞察
- 加密波動率 regime 轉換週期 ~2-4 週（傳統市場 1-3 個月）
- Squeeze → Expansion 方向的預測是核心難點，準確率需 > 52% 才有 edge
- 適合作為 overlay（如 vol_pause）而非獨立策略

---

## Event-Driven（事件驅動）

### 經典文獻
- 尚待補充（歡迎 agent 在研究過程中添加）

### 加密貨幣專屬
- OI 清算反彈（OI Liq Bounce）是本系統目前最成功的 event-driven 策略
- 清算瀑布後的反彈在加密市場特別顯著（槓桿出清 → 價格超調 → 反彈）
- 經驗：2026-02 OI Liq Bounce v4.2 SR=2.49, MDD=-1.3%, 與主策略相關性 ≈ 0.01

### 關鍵洞察
- 事件驅動策略的核心挑戰：假陽性率必須 < 80%
- 信號稀有性（time-in-market 低）是特色也是限制——資金效率低但分散化價值高
- 清算數據的歷史覆蓋有限（Binance ~7d live / CoinGlass 有歷史但需 API key）

---

## Multi-TF Resonance（多時間框架共振）

### 經典文獻
- 尚待補充（多 TF 交叉驗證的學術研究較少，多為實務經驗）

### 加密貨幣專屬
- 多 TF 一致性可提升信號品質，但改善幅度需 > 5%（否則不值得複雜度增加）
- 典型組合：1h 信號 + 4h 趨勢 + 1d regime

### 關鍵洞察
- TF alignment 過濾會減少信號頻率，需權衡「更好的信號」vs「更少的交易機會」
- 終止條件：如果多 TF 信號 IC 不優於單一 TF（改善 < 5%），應放棄

---

## Microstructure / Order Flow / Auction Market Theory（市場微結構 / 訂單流 / 拍賣市場理論）

### 經典文獻 — 理論基礎

#### 市場微結構核心模型
- Kyle (1985) "Continuous Auctions and Insider Trading" — 資訊不對稱下的連續拍賣模型，定義了 market impact 的線性框架（lambda 參數），奠定 order flow 理論的數學基礎
- Glosten & Milgrom (1985) "Bid, Ask, and Transaction Prices in a Specialist Market with Heterogeneously Informed Traders" — 序貫交易模型：每筆交易帶有信息，market maker 從 order flow 推斷 informed trading，解釋了 bid-ask spread 的信息成分
- Easley & O'Hara (1992) "Time and the Process of Security Price Adjustment" — PIN (Probability of Informed Trading) 模型，用交易到達率推斷知情交易比例；VPIN 的理論前身
- Hasbrouck (1991) "Measuring the Information Content of Stock Trades" — 用 VAR 模型量化每筆交易的「信息衝擊」，區分永久性 vs 暫時性價格影響

#### Order Flow Imbalance (OFI) 與 Price Impact
- ⭐ Cont, Kukanov & Stoikov (2014) "The Price Impact of Order Book Events", Quantitative Finance, 14(1), 109-126 — **最重要的現代 OFI 論文**。定義了 Order Flow Imbalance 指標，實證證明 OFI 對短期價格變動有線性預測力。量化 orderflow 策略的數學基礎
- Bouchaud, Farmer & Lillo (2009) "How Markets Slowly Digest Changes in Supply and Demand" — Order flow 的長記憶性（long memory）研究，limit order book 的統計特性（冪律分布、自相關結構）
- Cont (2001) "Empirical Properties of Asset Returns" — 金融資產收益的程式化事實（stylized facts），包含波動率叢集、肥尾、volume-volatility 相關等

#### VPIN 與 Informed Trading 偵測
- ⭐ Easley, López de Prado & O'Hara (2012) "Flow Toxicity and Liquidity in a High-Frequency World", Review of Financial Studies, 25(5) — VPIN (Volume-Synchronized Probability of Informed Trading) 指標，用 volume clock 替代 time clock 偵測 informed trading。2010 Flash Crash 前 VPIN 飆升的經典實證

#### Auction Market Theory (AMT) 與 Market/Volume Profile
- Steidlmayer & Koy (1986) "Markets and Market Logic" — AMT 的開山之作。CBOT 交易員 Steidlmayer 開發的 Market Profile：價格在「公允價值」附近形成鐘形分布，偏離→回歸
- Dalton, Jones & Dalton (1993) "Mind Over Markets" — Market Profile 實戰聖經。系統教授 TPO (Time-Price Opportunity) profile：trending day, normal day, double distribution day 等市場結構分類
- Dalton (2007) "Markets in Profile" — Mind Over Markets 續作，加入電子交易時代的市場結構適應
- Steidlmayer & Hawkins (2003) "Steidlmayer on Markets" — 更新版 AMT 框架，融入電子市場理解

#### 教科書
- Cartea, Jaimungal & Penalva (2015) "Algorithmic and High-Frequency Trading" (Cambridge Univ Press) — HFT / market making / order flow 系統性教科書，涵蓋最優執行、做市模型、order book 動態
- O'Hara (1995) "Market Microstructure Theory" — 市場微結構理論的經典教科書

### 實務書籍（非學術但有操作價值）
- Trader Dale (2018) "Volume Profile: The Insider's Guide to Trading" — 實務導向的 Volume Profile 教程：POC (Point of Control)、Value Area (VA)、HVN/LVN 作為支撐/阻力
- 開源證券《市場微觀結構研究系列》（25+ 篇系列報告）— 掛單方向長期記憶性、訂單流失衡因子、撤單行為等；國內最系統的 orderflow 量化研究系列

### 加密貨幣專屬
- Makarov & Schoar (2020) "Trading and Arbitrage in Cryptocurrency Markets" — 跨交易所 orderflow 動態、套利效率
- Alexander & Heck (2020) "A Critical Investigation of Cryptocurrency Data and Analysis" — 加密市場微結構特殊性：24/7 交易、碎片化流動性、wash trading 問題
- Dyhrberg, Foley & Svec (2018) "How Investible is Bitcoin? Analyzing the Liquidity and Transaction Costs of Bitcoin Markets" — BTC 市場的流動性結構和 market microstructure
- Taker Buy/Sell Imbalance 和 CVD（Cumulative Volume Delta）是加密市場主要的微結構信號
- 5m/15m 級別的微結構信號換手率極高（~12× 1h 策略），成本壓力巨大

### 關鍵洞察

**理論框架（三層結構）**：
| 層次 | 概念 | 核心問題 |
|------|------|---------|
| 理論基礎 | Auction Market Theory (AMT) | 市場如何透過買賣雙方的拍賣機制發現「公允價格」？ |
| 可視化工具 | Market Profile / Volume Profile | 如何把成交量在「價格維度」上展開，找出高/低成交區？ |
| 執行層 | Order Flow (Footprint / Delta / OFI) | 在微觀層面，誰在主動買？誰在主動賣？力量如何演變？ |

**核心概念**：
- **Volume Profile**：POC (Point of Control) = 成交量最大的價格；Value Area (VA) = 包含 70% 成交量的價格區間；HVN (High Volume Node) = 支撐/阻力，LVN (Low Volume Node) = 價格快速穿越區
- **OFI (Order Flow Imbalance)**：衡量主動買入 vs 主動賣出的淨差值，Cont et al. 2014 證實對短期價格有線性預測力
- **VPIN**：用 volume clock 偵測 informed trading 的密度，可作為 regime indicator（高 VPIN = 高毒性 = 高風險）
- **CVD (Cumulative Volume Delta)**：主動買賣的累積差值，追蹤買賣方力量的長期變化趨勢

**對本系統的適用性評估**：
- 微結構信號更適合作為 **overlay**（改善入場時機）而非獨立策略
- 終止條件：5m/15m 信號的 IC 在 2× slippage 下歸零（net edge < 0）即 FAIL
- 成本是最大敵人：年化成本可能是 1h 策略的 4-12 倍
- **Volume Profile 是最可行的研究方向**：可用現有 1h OHLCV 近似構建，POC/VA 作為 HTF filter 的增強，不需要 tick data
- **VPIN 是次可行方向**：需要 aggTrades 數據（Binance Vision 可下載），但信號是 daily 級別，與我們的 1h 框架相容
- **OFI 短期預測可行性最低**：需要 tick data + order book snapshot，數據量極大且與 1h 策略框架衝突

**數據限制**：
- Order Book depth 無歷史數據（僅 live WebSocket stream），`order_book.py` 已建基礎設施但無回測資料
- aggTrades（逐筆成交）有歷史（Binance Vision），但 1 天 BTC ≈ 數百 MB，處理成本高
- 經驗：2026-02 TVR EDA 顯示 IC=-0.006（弱逆向但獨立），CVD 不穩定（IC 年度翻轉）

---

## 跨原型通用文獻

### 加密市場總論
- Bouri et al. (2019) "On the Hedge and Safe Haven Properties of Bitcoin" — BTC 作為避風港的有限性
- Corbet et al. (2019) "Cryptocurrencies as a Financial Asset: A Systematic Analysis" — 加密資產的統計特性總覽

### 資訊理論 / Entropy
- Bandt & Pompe (2002) "Permutation Entropy: A Natural Complexity Measure for Time Series" — Permutation Entropy 原始論文，用 ordinal pattern 衡量時序隨機性，快速且穩健
- Pincus (1991) "Approximate Entropy as a Complexity Measure" — Approximate Entropy 原始論文，衡量時序的自我相似性/規則性
- Richman & Moorman (2000) "Physiological Time-Series Analysis Using Approximate Entropy and Sample Entropy" — Sample Entropy 改進 ApEn 的偏差問題
- Risso (2008) "The Informational Efficiency and the Financial Crashes" — 用 Shannon entropy 偵測金融市場的結構性變化
- Zunino et al. (2009) "Forbidden Patterns, Permutation Entropy and Stock Market Inefficiency" — PE 應用於金融市場效率檢測

> **實測結論 (2026-03-02)**：PE/SE/ApEn 3 種 entropy 應用於 8 crypto symbols 1h OHLCV，所有 IC < 0.01。加密 1h 序列太接近 random walk，entropy 不含有用交易信息。Entropy ≠ vol proxy（好消息），但也 ≠ alpha（壞消息）。

### 量化方法論
- López de Prado (2018) "Advances in Financial Machine Learning" — 金融 ML 的去偏方法（PBO、CPCV、DSR）
- Bailey et al. (2014) "The Deflated Sharpe Ratio" — 多重測試校正
- Harvey et al. (2016) "...and the Cross-Section of Expected Returns" — p-hacking 與過擬合警告

### 組合管理
- Markowitz (1952) "Portfolio Selection" — 均值-方差優化
- Kelly (1956) "A New Interpretation of Information Rate" — Kelly criterion 資金管理
- DeMiguel, Garlappi & Uppal (2009) "Optimal Versus Naive Diversification" — 1/N 組合在小樣本下的優勢

---

## 如何貢獻新文獻

任何 agent 在研究過程中發現有價值的文獻，請按以下步驟添加：

1. **判斷原型分類**：將文獻歸入對應的策略原型區塊
2. **區分來源**：放入「經典文獻」或「加密貨幣專屬」子分類
3. **格式統一**：`- 作者 (年份) "標題" — 一句話實用摘要`
4. **更新日期**：修改文件頂部的 `Last updated` 日期
5. **關鍵洞察**：如果文獻提供了重要的實踐啟示，也更新「關鍵洞察」區塊

> **品質標準**：只收錄對策略開發有直接指導意義的文獻。純理論文獻除非提供可操作的洞察，否則不納入。
