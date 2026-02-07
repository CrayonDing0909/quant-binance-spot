# 主流交易策略參考指南

本文檔列出主流交易策略的參考資源，幫助你學習和實現各種交易策略。

## 📚 目錄

1. [專案中已有的策略](#專案中已有的策略)
2. [主流策略分類](#主流策略分類)
3. [學習資源](#學習資源)
4. [開源策略庫](#開源策略庫)
5. [書籍推薦](#書籍推薦)
6. [線上課程與社群](#線上課程與社群)

---

## 專案中已有的策略

你的專案中已經實現了以下策略，可以作為參考：

### 1. **RSI 策略** (`rsi_strategy.py`)
- **類型**：均值回歸策略
- **邏輯**：RSI < 30 買入，RSI > 70 賣出
- **變體**：`rsi_momentum` - 基於 RSI 動量變化

### 2. **EMA 交叉策略** (`ema_cross.py`)
- **類型**：趨勢跟隨策略
- **邏輯**：快速 EMA 上穿慢速 EMA 時買入

### 3. **狀態管理策略** (`example_stateful_strategy.py`)
- **類型**：帶止損止盈的 EMA 策略
- **特點**：展示如何實現止損和止盈邏輯

### 4. **SMC 策略** (`smc_strategy.py`)
- **類型**：價格行為策略
- **邏輯**：基於 Smart Money Concepts（訂單塊、流動性池等）

### 查看策略代碼
```bash
# 查看所有策略文件
ls src/qtrade/strategy/

# 閱讀策略實現
cat src/qtrade/strategy/rsi_strategy.py
cat src/qtrade/strategy/ema_cross.py
```

---

## 主流策略分類

### 1. 趨勢跟隨策略 (Trend Following)

**核心思想**：跟隨市場趨勢，上漲時買入，下跌時賣出

#### 常見策略：
- **移動平均線交叉** (MA/EMA Cross)
  - 短期均線上穿長期均線 → 買入
  - 短期均線下穿長期均線 → 賣出
- **MACD 策略**
  - MACD 線上穿信號線 → 買入
  - MACD 線下穿信號線 → 賣出
- **ADX 趨勢強度策略**
  - ADX > 25 且 +DI > -DI → 買入
- **唐奇安通道突破** (Donchian Channel)
  - 價格突破上軌 → 買入
  - 價格跌破下軌 → 賣出

**參考資源**：
- [Investopedia: Trend Following](https://www.investopedia.com/terms/t/trending-market.asp)
- [TradingView: Moving Average Strategies](https://www.tradingview.com/scripts/moving-average/)

### 2. 均值回歸策略 (Mean Reversion)

**核心思想**：價格偏離均值後會回歸，在極端位置反向交易

#### 常見策略：
- **RSI 超買超賣**
  - RSI < 30（超賣）→ 買入
  - RSI > 70（超買）→ 賣出
- **布林帶策略** (Bollinger Bands)
  - 價格觸及下軌 → 買入
  - 價格觸及上軌 → 賣出
- **隨機指標** (Stochastic Oscillator)
  - %K < 20 → 買入
  - %K > 80 → 賣出
- **威廉指標** (Williams %R)
  - %R < -80 → 買入
  - %R > -20 → 賣出

**參考資源**：
- [Investopedia: Mean Reversion](https://www.investopedia.com/terms/m/meanreversion.asp)
- [QuantConnect: Mean Reversion Strategies](https://www.quantconnect.com/learning/articles/introduction-to-mean-reversion-strategies)

### 3. 突破策略 (Breakout)

**核心思想**：價格突破關鍵位置時入場

#### 常見策略：
- **支撐阻力突破**
  - 突破阻力位 → 買入
  - 跌破支撐位 → 賣出
- **布林帶突破**
  - 突破上軌 → 買入（趨勢延續）
- **成交量突破**
  - 價格突破 + 成交量放大 → 買入
- **波動率突破** (Volatility Breakout)
  - 基於 ATR 的突破策略

**參考資源**：
- [Investopedia: Breakout Trading](https://www.investopedia.com/terms/b/breakout.asp)
- [TradingView: Breakout Strategies](https://www.tradingview.com/scripts/breakout/)

### 4. 動量策略 (Momentum)

**核心思想**：跟隨價格動量，強者恆強

#### 常見策略：
- **價格動量**
  - 過去 N 天收益率 > 閾值 → 買入
- **相對強度** (Relative Strength)
  - 比較不同資產的表現
- **動量指標組合**
  - RSI + MACD + 成交量

**參考資源**：
- [Investopedia: Momentum Investing](https://www.investopedia.com/terms/m/momentum.asp)
- [QuantStart: Momentum Strategies](https://www.quantstart.com/articles/Momentum-Strategies/)

### 5. 價格行為策略 (Price Action)

**核心思想**：基於價格行為和市場結構，不依賴技術指標

#### 常見策略：
- **Smart Money Concepts (SMC)**
  - 訂單塊 (Order Blocks)
  - 流動性池 (Liquidity Pools)
  - 市場結構 (Market Structure)
- **供需區域** (Supply & Demand Zones)
- **價格模式識別**
  - 頭肩頂/底
  - 雙頂/雙底
  - 三角形整理

**參考資源**：
- [TradingView: Price Action Trading](https://www.tradingview.com/scripts/price-action/)
- [Babypips: Price Action Trading](https://www.babypips.com/learn/forex/price-action-trading)

### 6. 套利策略 (Arbitrage)

**核心思想**：利用價格差異獲利

#### 常見策略：
- **跨交易所套利**
  - 同一資產在不同交易所的價差
- **三角套利**
  - 利用不同交易對之間的價差
- **統計套利**
  - 配對交易 (Pairs Trading)

**參考資源**：
- [Investopedia: Arbitrage](https://www.investopedia.com/terms/a/arbitrage.asp)
- [QuantStart: Pairs Trading](https://www.quantstart.com/articles/Pairs-Trading-Strategy/)

### 7. 多因子策略 (Multi-Factor)

**核心思想**：結合多個指標或因子

#### 常見策略：
- **技術指標組合**
  - RSI + MACD + 成交量
- **基本面 + 技術面**
  - 結合財務數據和技術指標
- **機器學習策略**
  - 使用 ML 模型預測價格

**參考資源**：
- [QuantConnect: Multi-Factor Models](https://www.quantconnect.com/learning/articles/introduction-to-multi-factor-models)

---

## 學習資源

### 1. 技術指標百科

1. **Investopedia** (https://www.investopedia.com/)
   - 最全面的金融和交易知識庫
   - 每個指標都有詳細說明和示例
   - **推薦**：搜索 "RSI", "MACD", "Bollinger Bands" 等

2. **TradingView** (https://www.tradingview.com/)
   - 免費圖表平台
   - 數千個策略腳本（Pine Script）
   - 可以查看和學習別人的策略實現
   - **推薦**：搜索策略名稱，查看公開腳本

3. **QuantConnect** (https://www.quantconnect.com/)
   - 量化交易平台
   - 大量策略示例和教程
   - 支持 Python 和 C#
   - **推薦**：查看 "Algorithm Library"

4. **QuantStart** (https://www.quantstart.com/)
   - 量化交易教育網站
   - 免費教程和文章
   - **推薦**：查看 "Trading Strategies" 分類

### 2. 策略實現參考

1. **Backtrader 策略庫**
   - GitHub: https://github.com/mementum/backtrader
   - 查看 `samples/` 目錄中的策略示例

2. **Zipline 策略示例**
   - GitHub: https://github.com/quantopian/zipline
   - Quantopian 的策略庫（已關閉，但代碼仍可參考）

3. **Freqtrade 策略庫**
   - GitHub: https://github.com/freqtrade/freqtrade-strategies
   - 大量加密貨幣交易策略

4. **Python for Finance**
   - GitHub: https://github.com/yhilpisch/py4fi
   - 包含多個策略實現示例

---

## 開源策略庫

### 1. **GitHub 策略庫**

#### Freqtrade Strategies
- **鏈接**：https://github.com/freqtrade/freqtrade-strategies
- **內容**：數百個加密貨幣交易策略
- **語言**：Python
- **特點**：可直接使用，包含回測結果

#### Awesome Quant
- **鏈接**：https://github.com/wilsonfreitas/awesome-quant
- **內容**：量化交易資源大全
- **包含**：策略、工具、數據源等

#### QuantConnect Algorithms
- **鏈接**：https://github.com/QuantConnect/Lean
- **內容**：QuantConnect 的開源算法庫
- **語言**：Python, C#
- **特點**：生產級別的策略實現

### 2. **策略模板庫**

#### Backtrader Samples
- **鏈接**：https://github.com/mementum/backtrader/tree/master/samples
- **內容**：Backtrader 框架的策略示例
- **包含**：各種技術指標策略

#### TradingGym
- **鏈接**：https://github.com/notadamking/tradinggym
- **內容**：強化學習交易策略
- **特點**：使用 RL 進行策略優化

---

## 書籍推薦

### 入門級

1. **《Python金融大數據分析》** (Yves Hilpisch)
   - 涵蓋 Python 在金融中的應用
   - 包含策略實現示例

2. **《量化交易：如何建立自己的算法交易》** (Ernest Chan)
   - 策略開發的實用指南
   - 包含均值回歸、動量等策略

### 進階級

3. **《算法交易：制勝策略與原理》** (Ernest Chan)
   - 深入講解各種策略原理
   - 包含統計套利、配對交易等

4. **《量化投資：策略與技術》** (丁鵬)
   - 中文量化投資教材
   - 涵蓋多種策略類型

5. **《Advances in Financial Machine Learning》** (Marcos López de Prado)
   - 機器學習在量化交易中的應用
   - 適合進階學習者

### 技術指標專著

6. **《技術分析全書》** (John J. Murphy)
   - 技術分析的經典教材
   - 涵蓋所有主流技術指標

7. **《日本蠟燭圖技術》** (Steve Nison)
   - K 線圖和價格模式
   - 價格行為交易必讀

---

## 線上課程與社群

### 1. 線上課程

1. **Coursera - Financial Engineering and Risk Management**
   - 提供：多所大學的金融工程課程
   - 語言：英文

2. **Udemy - Algorithmic Trading**
   - 提供：實用的算法交易課程
   - 語言：英文
   - 價格：付費（常有折扣）

3. **QuantInsti - EPAT**
   - 提供：專業量化交易課程
   - 語言：英文
   - 價格：付費

### 2. 社群與論壇

1. **Reddit**
   - r/algotrading - 算法交易討論
   - r/quant - 量化交易討論
   - r/StockMarket - 股票市場討論

2. **Stack Overflow**
   - 標籤：`algorithmic-trading`, `quantitative-finance`
   - 技術問題解答

3. **QuantConnect Forum**
   - https://www.quantconnect.com/forum
   - 策略討論和分享

4. **TradingView Community**
   - https://www.tradingview.com/scripts/
   - 策略腳本分享和討論

### 3. 中文資源

1. **聚寬 (JoinQuant)**
   - https://www.joinquant.com/
   - 中文量化交易平台
   - 大量策略示例和教程

2. **米筐 (RiceQuant)**
   - https://www.ricequant.com/
   - 中文量化平台
   - 策略庫和回測工具

3. **掘金量化**
   - https://www.myquant.cn/
   - 中文量化社區
   - 策略分享和討論

---

## 如何學習新策略

### 步驟 1: 理解策略原理
1. 閱讀策略說明（Investopedia、書籍等）
2. 理解策略的市場假設
3. 了解策略的適用場景

### 步驟 2: 查看現有實現
1. 在 TradingView 搜索策略名稱
2. 查看 Pine Script 實現
3. 在 GitHub 搜索策略代碼

### 步驟 3: 在專案中實現
1. 參考專案中現有策略的結構
2. 使用 `@register_strategy` 註冊策略
3. 實現 `generate_positions` 函數

### 步驟 4: 回測和優化
1. 運行回測查看表現
2. 優化參數
3. 驗證策略穩定性

### 步驟 5: 改進策略
1. 添加過濾條件
2. 結合多個指標
3. 添加風險管理（止損、止盈）

---

## 快速查找策略

### 按策略類型查找

| 策略類型 | 推薦資源 | 關鍵字搜索 |
|---------|---------|-----------|
| 趨勢跟隨 | Investopedia, TradingView | "trend following", "moving average cross" |
| 均值回歸 | QuantConnect, Investopedia | "mean reversion", "RSI strategy" |
| 突破策略 | TradingView, Investopedia | "breakout strategy", "support resistance" |
| 動量策略 | QuantStart, Investopedia | "momentum strategy", "relative strength" |
| 價格行為 | TradingView, Babypips | "price action", "SMC strategy" |
| 套利策略 | QuantStart, Investopedia | "arbitrage", "pairs trading" |

### 按指標查找

| 指標 | 策略類型 | 推薦資源 |
|-----|---------|---------|
| RSI | 均值回歸 | Investopedia: "RSI Strategy" |
| MACD | 趨勢跟隨 | TradingView: "MACD Crossover" |
| 布林帶 | 均值回歸/突破 | Investopedia: "Bollinger Bands Strategy" |
| EMA/SMA | 趨勢跟隨 | TradingView: "Moving Average Cross" |
| ADX | 趨勢跟隨 | Investopedia: "ADX Strategy" |
| Stochastic | 均值回歸 | Investopedia: "Stochastic Oscillator" |

---

## 實用工具

### 1. 策略回測平台

- **Backtrader** (你正在使用的框架)
- **QuantConnect** - 雲端回測平台
- **Zipline** - 開源回測框架
- **Freqtrade** - 加密貨幣交易框架

### 2. 數據源

- **Binance API** - 加密貨幣數據（你已在使用）
- **Yahoo Finance** - 股票數據
- **Alpha Vantage** - 免費金融數據 API
- **Quandl** - 金融數據平台

### 3. 技術指標庫

- **TA-Lib** - 技術分析庫（C/C++，有 Python 綁定）
- **Pandas-TA** - Python 技術指標庫
- **你專案中的 `qtrade.indicators`** - 已實現的指標庫

---

## 總結

### 學習路徑建議

1. **初學者**：
   - 先學習專案中已有的策略（RSI、EMA）
   - 閱讀 `QUICK_START_GUIDE.md`
   - 在 Investopedia 學習技術指標基礎

2. **進階學習**：
   - 在 TradingView 查看策略實現
   - 閱讀相關書籍
   - 在 GitHub 查看開源策略

3. **實戰開發**：
   - 參考專案中的策略模板
   - 實現新策略並回測
   - 優化和驗證策略

### 重要提醒

⚠️ **策略學習注意事項**：

1. **不要盲目複製策略**：理解原理後再實現
2. **回測不等於實盤**：實盤表現可能不同
3. **避免過度優化**：使用驗證腳本檢測過擬合
4. **風險管理第一**：策略再好也要有止損
5. **持續學習**：市場在變化，策略也需要更新

---

## 下一步行動

1. ✅ 查看專案中已有的策略實現
2. ✅ 閱讀 `QUICK_START_GUIDE.md` 了解如何開發策略
3. ✅ 在 Investopedia 或 TradingView 學習感興趣的策略
4. ✅ 參考策略模板實現新策略
5. ✅ 回測和優化策略

祝你交易順利！🎉

