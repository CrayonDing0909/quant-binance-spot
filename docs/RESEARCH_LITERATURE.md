> **Last updated**: 2026-02-26

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

## Microstructure（市場微結構）

### 經典文獻
- Kyle (1985) "Continuous Auctions and Insider Trading" — 資訊不對稱與價格衝擊
- Hasbrouck (1991) "Measuring the Information Content of Stock Trades" — 訂單流信息含量

### 加密貨幣專屬
- Taker Buy/Sell Imbalance 和 CVD（Cumulative Volume Delta）是加密市場主要的微結構信號
- 5m/15m 級別的微結構信號換手率極高（~12× 1h 策略），成本壓力巨大

### 關鍵洞察
- 微結構信號更適合作為 overlay（改善入場時機）而非獨立策略
- 終止條件：5m/15m 信號的 IC 在 2× slippage 下歸零（net edge < 0）即 FAIL
- 成本是最大敵人：年化成本可能是 1h 策略的 4-12 倍

---

## 跨原型通用文獻

### 加密市場總論
- Bouri et al. (2019) "On the Hedge and Safe Haven Properties of Bitcoin" — BTC 作為避風港的有限性
- Corbet et al. (2019) "Cryptocurrencies as a Financial Asset: A Systematic Analysis" — 加密資產的統計特性總覽

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
