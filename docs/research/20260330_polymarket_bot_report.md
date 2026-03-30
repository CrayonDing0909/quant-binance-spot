---
tags: [report, polymarket, strategy, active]
date: 2026-03-30
status: active
---

# Polymarket 15 分鐘自動交易 Bot — 策略報告 v2

> 寫給有基本程式能力但沒有量化交易經驗的人。
> 更新於 2026-03-30 晚：加入真實 PM odds 回測結果。

---

## 一、我們在做什麼？

Polymarket 每 15 分鐘開一個新市場：「BTC 接下來 15 分鐘會漲還是跌？」

你可以買「漲」或「跌」的股份。押對 = 每股值 $1；押錯 = $0。

**股份價格反映市場共識。** 如果大家覺得 80% 會漲，「漲」的股份賣 $0.80。

我們的 bot：**找到 Polymarket 定價偏離 Binance 實際走勢的時刻，買入被低估的那邊。**

---

## 二、策略演進（三個版本，前兩個失敗了）

### v1：純 Contrarian（買便宜邊）
- 邏輯：永遠買 Polymarket 上便宜的那邊
- 結果：**6.5% WR，慘敗**
- 問題：便宜有便宜的道理，大部分時候市場是對的

### v2：Krajekis TA（RSI/MACD/EMA 技術分析）
- 邏輯：用技術指標預測 15 分鐘方向
- 結果：修正回測後 **56.6% WR**，但 PnL 微正（+$213 on 17K trades）
- 問題：TA 在 15 分鐘級別預測力很弱

### v3（當前）：Odds Divergence（Polymarket vs Binance 偏差）
- 邏輯：不預測方向，找 Polymarket 定價錯誤
- 來源：朋友的建議 —「實盤漲跌幅跟賭盤預測偏差過大時進場」
- 狀態：**有初步真實數據回測支持**

---

## 三、v3 策略 — 怎麼運作？

### 核心邏輯

```
每 15 分鐘窗口，在第 5-7 分鐘時（sweet spot）：

1. 看 Polymarket：Up 股份賣多少？（例如 $0.25）
2. 看 Binance：BTC 實際上漲了還是跌了？

如果 Polymarket 說「只有 25% 機率會漲」
但 Binance 顯示價格其實沒怎麼跌（或已經反彈）
→ 偏差！買 Up（因為 PM 低估了漲的機率）
```

### 為什麼不直接看方向？

因為**方向不重要，賠率才重要**：

```
假設：買 Up @ $0.25，WR 只有 35%

贏（35% 機率）：花 $0.25 拿 $1.00 → 淨賺 $0.75
輸（65% 機率）：損失 $0.25

期望值 = 35% × $0.75 - 65% × $0.25 = +$0.10（正的！）

即使猜錯比猜對多，只要賠率夠好就賺。
```

### 進場規則

1. **只在 PM odds 外 30%-70% 時進場**（PDF rule 1）
2. **不做明顯的單邊行情**（PDF rule 2）— 如果 BTC 已漲 >0.5%，趨勢太明顯，跳過
3. **Sweet spot 時機**：窗口開始後 5-10 分鐘才看（讓價格結構先形成）

---

## 四、用真實 Polymarket 數據的回測結果

### 數據來源
- 過去 72 小時的 100 個 BTC 15m 市場
- 使用 Polymarket CLOB API 的 `/prices-history` 拿到每分鐘的真實 odds
- **不是模擬 odds，是 PM 上的實際交易價格**

### 結果

| 策略 | 進場時間 | 交易數 | Win Rate | PnL |
|---|---|---|---|---|
| **Contrarian @ t=5min** | 5分鐘 | 100 | 35% | **+15.7** ✅ |
| **Contrarian @ t=7min** | 7分鐘 | 100 | 31% | **+20.3** ✅ |
| Momentum @ t=0 | 開盤 | 100 | 60% | +10.4 |
| Momentum @ t=10min | 10分鐘 | 100 | 83% | +3.5 |
| Divergence @ t=5min | 5分鐘 | 42 | 21% | **+11.3** ✅ |
| Contrarian @ t=0 | 開盤 | 100 | 40% | -13.2 ❌ |
| Contrarian @ t=10min | 10分鐘 | 100 | 17% | -25.8 ❌ |

### 解讀

1. **WR 低不代表虧錢** — Contrarian @ t=7min 只有 31% WR 但 PnL +20.3
2. **時機關鍵** — t=0 和 t=10 的 contrarian 都虧，只有 t=5-7 是正的
3. **Momentum 很高 WR 但 PnL 低** — 因為買的是貴邊，贏少虧多
4. **100 筆樣本太少** — 需要更多數據驗證（正在抓歷史 snapshot）

### 為什麼 t=5-7min 是最佳時機？

```
t=0:  方向不明，Contrarian 亂猜 → 虧
t=5:  方向初現，PM 開始偏離但還不極端 → Contrarian 有 edge
t=7:  PM 已偏離，但結算還有 8 分鐘可能反轉 → 最佳 Contrarian 時機
t=10: 方向已定，PM 幾乎確定對 → Contrarian 太晚 → 虧
```

---

## 五、系統架構

```
Oracle Cloud (大阪, 日本 IP)
  │
  ├── polymarket_15m (tmux session) — v3 Odds Divergence
  │   │
  │   │  每 30 秒循環：
  │   ├── 1. Gamma API → 找到當前 15m 市場
  │   ├── 2. 看 PM 當前 Up/Down 報價
  │   ├── 3. 看 Binance BTC 實際價格 vs 窗口開盤價
  │   ├── 4. 計算偏差 → 偏差夠大就下注
  │   └── 5. Limit Order (maker = 0% fee) + Telegram 通知
  │
  └── polymarket_bot — 日頻清算反彈策略（獨立）

MetaMask 錢包 (Polygon)
  ├── Polymarket CLOB: $8.75 可交易
  └── EOA: $1.00 備用
```

### 代碼結構

```
src/qtrade/polymarket/
├── odds_divergence.py     ← v3 策略：PM odds vs Binance 偏差
├── market_discovery.py    ← 自動發現 15m 市場
├── binance_feed.py        ← Binance 價格 + TA 指標
├── krajekis_strategy.py   ← v2 策略（保留作參考）
├── runner_15m.py          ← 主循環
├── client.py              ← PM API 下單
├── oi_monitor.py          ← 日頻清算偵測
└── runner.py              ← 日頻循環
```

---

## 六、風險

### 已知風險

1. **樣本太少** — 100 筆真實 PM odds 回測不夠，需要數千筆
2. **流動性風險** — PM 掛單簿薄，$1 的單沒問題，大單會滑價
3. **你朋友的警告** — PM 數據是「上次成交價」不是「可執行掛單價」，套利價格可能一下就沒了
4. **策略可能不穩定** — 72 小時的 edge 可能只是偶然

### 最壞情況

$8.75 CLOB 餘額，每筆 $1。如果連輸 8 筆 → 虧完。
但 Contrarian 連輸 8 筆的機率 = 0.65^8 = 3.2%（如果 WR=35%）。

### 目前的保護

- 每筆 $1（固定）
- 每天最多虧 3 筆就停
- 只在 PM odds 外 30-70% 才交易（不是每個窗口都做）

---

## 七、vs 之前 TSMOM 的根本差異

| | 舊 TSMOM | 新 Polymarket Bot |
|---|---|---|
| **做什麼** | 預測 BTC 7 天方向 | 在 PM 找 15 分鐘定價錯誤 |
| **怎麼賺** | 趨勢持續 → 賺差價 | PM 定價偏差 → 賺賠率 |
| **Edge** | IC ≈ 0.02（噪音） | PM overreaction（行為偏差） |
| **WR** | 回測 ~50%，實盤全虧 | WR 31-35% 但 PnL 正 |
| **最大虧損** | 帳戶 100%（有槓桿） | 每筆 $1（固定） |
| **回測** | 回測好看但實盤崩 | 用真實 PM odds 回測 |

---

## 八、下一步

1. **等歷史 snapshot 下載完** → 用數千筆真實 PM odds 做完整回測
2. **觀察 bot 實盤交易** → 看真實 WR 和 PnL
3. **如果驗證正 EV** → 加倉到 $5/筆
4. **如果不行** → 回到分析，或改成 momentum 策略（83% WR 但低 PnL）

---

## 九、朋友的關鍵建議（原文）

> 「先抓實盤漲跌幅％跟賭盤預測勝率％這兩個數據來看，當兩者偏差過大時，就嘗試進場」

> 「我反而不會刻意挑高RR的條件，高RR通常伴隨低勝率，需要更大量的樣本才能驗證EV」

> 「polymarket給的data是上次成交價，不是掛單價，所以一個套利價格出現時，他可能倉位很小，一下就沒了」

**策略規則（PDF）：**
1. 只在 chance 30%-70% 外時，與市場做反向
2. 不做明顯的單邊行情
3. 8:00 UTC+8 前值得關注（每日開盤）
