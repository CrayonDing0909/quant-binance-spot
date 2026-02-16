# 新手完整教學：從策略發想到實盤

本教學將帶你從零開始，完整地開發一個交易策略，包括策略發想、實現、回測、優化和驗證。

**支援現貨 (Spot) 和合約 (Futures) 交易！** 🟢 🔴

## 📋 目錄

1. [專案功能概覽](#專案功能概覽)
2. [第一步：策略發想](#第一步策略發想)
3. [第二步：建立策略](#第二步建立策略)
4. [第三步：回測策略](#第三步回測策略)
5. [第四步：優化參數](#第四步優化參數)
6. [第五步：驗證策略](#第五步驗證策略)
7. [第六步：風險管理](#第六步風險管理)
8. [第七步：即時交易](#第七步即時交易)（含 Telegram Bot、雲端部署）
9. [第八步：監控與維運](#第八步監控與維運)
10. [合約交易教學](#合約交易教學)（含 SL/TP 去重、倉位分配、API 降級）
11. [增量 K 線快取](#增量-k-線快取) ⭐ v2.8（解決實盤 vs 回測數據不一致）
12. [多數據源與長期歷史數據](#多數據源與長期歷史數據)
13. [組合回測](#組合回測)
14. [RSI Exit 策略配置](#rsi-exit-策略配置) ⭐ NEW（經驗證的最佳配置）
15. [策略升級：Dynamic RSI + Funding Filter](#策略升級dynamic-rsi--funding-filter) ⭐ v3.0
16. [執行優化：Maker 優先下單](#執行優化maker-優先下單) ⭐ v3.0
17. [WebSocket 事件驅動](#websocket-事件驅動) ⭐ v3.0（延遲 <1 秒）
18. [SQLite 交易資料庫](#sqlite-交易資料庫) ⭐ v3.0
19. [波動率過濾 + HTF 軟趨勢過濾](#波動率過濾--htf-軟趨勢過濾) ⭐ v3.1
20. [波動率目標倉位管理](#波動率目標倉位管理) ⭐ v3.1
21. [Alpha Decay 監控](#alpha-decay-監控) ⭐ v3.1
22. [策略組合 Ensemble](#策略組合-ensemble) ⭐ v3.1
23. [完整範例：RSI 策略](#完整範例rsi策略)

---

## ⚡ 快速理解：回測如何運作？

**重要概念**：回測腳本透過讀取設定檔來確定使用哪個策略。

### 工作流程

```
1. 你建立策略程式碼 → 用 @register_strategy("策略名稱") 註冊
                    ↓
2. 你在 config/base.yaml 中設定 → strategy.name = "策略名稱"
                    ↓
3. 你執行 python scripts/run_backtest.py
                    ↓
4. 腳本讀取設定檔 → 找到策略名稱
                    ↓
5. 腳本根據名稱找到策略函式 → 使用設定的參數執行
                    ↓
6. 產生回測結果和報告
```

### 關鍵點

- ✅ **策略名稱必須一致**：程式碼中的 `@register_strategy("my_strategy")` 和設定檔中的 `strategy.name: "my_strategy"` 必須完全一致
- ✅ **參數在設定檔中**：策略的參數在 `config/base.yaml` 的 `strategy.params` 中設定
- ✅ **切換策略很簡單**：只需要修改設定檔，無需修改程式碼

### 範例

**程式碼中註冊策略**：
```python
@register_strategy("my_rsi_strategy")  # ← 策略名稱
def generate_positions(df, ctx, params):
    # 策略邏輯
    ...
```

**設定檔中指定策略**：
```yaml
strategy:
  name: "my_rsi_strategy"  # ← 必須與上面的名稱一致
  params:
    period: 14
```

**執行回測**：
```bash
python scripts/run_backtest.py  # 自動讀取設定，使用 my_rsi_strategy
```

---

## 專案功能概覽

這個專案提供了完整的量化交易策略開發工具，**支援現貨和合約交易**：

### 🎯 核心功能

1. **策略開發**
   - 快速建立策略範本
   - 統一的指標庫（RSI、MACD、布林帶等）
   - 策略註冊系統
   - **Dynamic RSI**（Rolling Percentile 自適應閾值，對抗 Alpha Decay）⭐ v3.0
   - **Funding Rate 過濾器**（獨立因子，過濾擁擠交易）⭐ v3.0
   - **波動率過濾器**（ATR/Price 低於閾值時不開倉，過濾低波動磨耗）⭐ v3.1
   - **HTF 軟趨勢過濾**（多時間框架連續權重，非二元閘門）⭐ v3.1
   - **策略組合 Ensemble**（RSI+MACD 低相關配對，信號平均）⭐ v3.1

2. **資料管理**
   - 自動下載幣安資料（現貨/合約）
   - **多數據源支援** ⭐ NEW（Yahoo Finance、CCXT、Binance Vision）
   - **長期歷史數據**（BTC 2014 年起）
   - 資料品質檢查
   - 資料清洗

3. **回測系統**
   - 向量化回測（快速）
   - 支援手續費和滑點
   - 自動產生報告和圖表
   - **支援做空模擬（合約）** ⭐ NEW
   - **組合回測**（多幣種權重配置）⭐ NEW

4. **策略優化**
   - 參數網格搜尋
   - 多指標優化
   - **綜合回測**（市場階段 + 倉位管理 + 出場策略）⭐ NEW

5. **策略驗證**
   - 過擬合檢測
   - 滾動視窗驗證
   - 參數敏感性分析
   - **Kelly 公式驗證** ⭐ NEW
   - **Live/Backtest 一致性驗證** ⭐ NEW
   - **Pre-Deploy 一致性檢查** ⭐ NEW（上架前必跑）
   - **Alpha Decay 監控**（Rolling IC + 年度 IC 衰退偵測 + Telegram 警報）⭐ v3.1

6. **風險管理**
   - 倉位管理
   - 風險限制
   - 組合風險管理
   - **Kelly 倉位計算** ⭐ NEW
   - **波動率目標倉位**（目標年化波動率，自動調整持倉大小）⭐ v3.1

7. **即時交易** ⭐ NEW
   - Paper Trading（模擬交易）
   - Real Trading（真實交易）
   - **自動 SL/TP 掛單**（標準 API + Algo Order 自動降級）⭐ NEW
   - **SL/TP 去重 + 持久化快取**（防止重複掛單）⭐ NEW
   - **SL/TP 自動補掛**（reconciliation）⭐ NEW
   - **SL/TP 觸發冷卻機制**（cooldown）⭐ NEW
   - **多幣種倉位分配** + 防震盪機制 ⭐ NEW
   - **增量 K 線快取**（解決實盤 vs 回測數據不一致）⭐ v2.8
   - **方向切換確認機制**（可選 2-tick 確認）⭐ v2.8
   - **支援現貨 🟢 和合約 🔴** ⭐ NEW
   - **Maker 優先下單**（省一半手續費：0.04% → 0.02%）⭐ v3.0
   - **WebSocket 事件驅動**（延遲 <1 秒，Oracle Cloud 1GB RAM 可跑）⭐ v3.0
   - **SQLite 交易資料庫**（結構化交易紀錄 + CLI 查詢）⭐ v3.0

8. **Telegram Bot 互動** ⭐ NEW
   - **命令選單提示**（輸入 `/` 直接選）⭐ NEW
   - 帳戶狀態、持倉、交易記錄查詢
   - **SL/TP 摘要 + 預估盈虧**（`/status`）⭐ NEW
   - 即時信號生成（`/signals`）+ **信號快取**（與 cron 信號一致）⭐ v2.8
   - 今日盈虧明細（`/pnl`）⭐ NEW
   - 交易統計（`/stats`）⭐ NEW

9. **監控與維運** ⭐ NEW
   - 系統健康檢查
   - 每日績效報表
   - 心跳監控
   - **雲端部署教學**（Oracle Cloud）⭐ NEW

---

## 第一步：策略發想

### 什麼是交易策略？

交易策略就是一套規則，告訴你在什麼時候買入、什麼時候賣出。

### 常見的策略思路

#### 1. 趨勢跟蹤策略
**思路**：跟隨市場趨勢，上漲時買入，下跌時賣出
- **範例**：移動平均線交叉策略
  - 當短期均線上穿長期均線 → 買入訊號
  - 當短期均線下穿長期均線 → 賣出訊號

#### 2. 均值回歸策略
**思路**：價格偏離均值後會回歸
- **範例**：RSI 超買超賣策略
  - RSI < 30（超賣）→ 買入
  - RSI > 70（超買）→ 賣出

#### 3. 突破策略
**思路**：價格突破關鍵位置時入場
- **範例**：布林帶突破
  - 價格突破上軌 → 買入
  - 價格跌破下軌 → 賣出

### 策略發想步驟

1. **觀察市場**：看看價格走勢有什麼規律
2. **形成假設**：比如「RSI 低於 30 時，價格可能會反彈」
3. **設計規則**：把假設轉化為具體的買賣規則
4. **測試驗證**：用歷史資料回測，看是否有效

### 範例：我們要開發的策略

**策略名稱**：RSI 超買超賣策略

**策略思路**：
- RSI（相對強弱指標）衡量價格動量
- RSI < 30：市場超賣，可能反彈 → 買入
- RSI > 70：市場超買，可能回調 → 賣出

**具體規則**：
1. 計算 RSI（週期 14）
2. 當 RSI < 30 時，買入（持倉 100%）
3. 當 RSI > 70 時，賣出（持倉 0%）
4. 其他情況保持當前持倉

---

## 第二步：建立策略

### 方法一：使用範本產生器（推薦新手）

專案提供了策略範本產生器，可以快速建立策略檔案：

```bash
# 建立 RSI 策略
python scripts/create_strategy.py --name my_rsi_strategy --type rsi
```

這會自動建立策略檔案，你只需要修改參數即可。

### 方法二：手動建立策略

#### 2.1 建立策略檔案

在 `src/qtrade/strategy/` 目錄下建立新檔案，例如 `my_rsi_strategy.py`：

```python
"""
我的 RSI 策略

策略邏輯：
- RSI < 30：買入
- RSI > 70：賣出
"""
from __future__ import annotations
import pandas as pd
from .base import StrategyContext
from . import register_strategy
from ..indicators import calculate_rsi


@register_strategy("my_rsi_strategy")
def generate_positions(df: pd.DataFrame, ctx: StrategyContext, params: dict) -> pd.Series:
    """
    產生持倉訊號
    
    Args:
        df: K 線資料（包含 open, high, low, close, volume）
        ctx: 策略上下文（包含交易對資訊）
        params: 策略參數
    
    Returns:
        持倉比例序列 [0, 1]，0 表示空倉，1 表示滿倉
    """
    # 1. 取得參數（如果沒有提供，使用預設值）
    period = params.get("period", 14)      # RSI 週期，預設 14
    oversold = params.get("oversold", 30)   # 超賣閾值，預設 30
    overbought = params.get("overbought", 70)  # 超買閾值，預設 70
    
    # 2. 取得收盤價
    close = df["close"]
    
    # 3. 計算 RSI
    rsi = calculate_rsi(close, period=period)
    
    # 4. 產生訊號
    # RSI < 30：買入訊號（持倉 = 1）
    # RSI > 70：賣出訊號（持倉 = 0）
    # 其他情況：保持當前狀態（這裡簡化處理，實際可以用更複雜的邏輯）
    
    pos = pd.Series(0.0, index=df.index)  # 初始化為空倉
    
    # 買入條件
    buy_signal = rsi < oversold
    pos[buy_signal] = 1.0
    
    # 賣出條件
    sell_signal = rsi > overbought
    pos[sell_signal] = 0.0
    
    # 5. 重要：避免未來資訊洩漏
    # 在 t 時刻的訊號，應該在 t+1 時刻執行
    pos = pos.shift(1).fillna(0.0)
    
    return pos
```

#### 2.2 註冊策略

在 `src/qtrade/strategy/__init__.py` 中匯入你的策略：

```python
from . import my_rsi_strategy  # noqa: E402
```

#### 2.3 設定策略參數

在 `config/base.yaml` 中設定策略：

```yaml
strategy:
  name: "my_rsi_strategy"  # 策略名稱（必須與 @register_strategy 中的名稱一致）
  params:
    period: 14        # RSI 週期
    oversold: 30      # 超賣閾值
    overbought: 70    # 超買閾值
```

### 策略開發要點

1. **避免未來資訊洩漏**：必須使用 `shift(1)` 將訊號向後移動
2. **回傳值**：持倉比例必須在 [0, 1] 範圍內
3. **使用指標庫**：不要重複實作指標，使用 `qtrade.indicators` 中的函式

---

## 第三步：回測策略

### 3.1 準備資料

首先需要下載歷史資料：

```bash
# 下載 BTCUSDT 的 1 小時 K 線資料
python scripts/download_data.py --symbol BTCUSDT --interval 1h --start 2022-01-01
```

### 3.2 設定策略（重要！）

**在執行回測之前，必須先設定要使用的策略。**

編輯 `config/base.yaml` 檔案：

```yaml
strategy:
  name: "my_rsi_strategy"  # 這裡指定要使用的策略名稱
  params:
    period: 14
    oversold: 30
    overbought: 70
```

**重要說明**：
- `strategy.name` 必須與你在程式碼中 `@register_strategy("策略名稱")` 的名稱一致
- `strategy.params` 是策略的參數，不同策略需要的參數不同
- 修改設定後，直接執行回測即可，無需修改程式碼

### 3.3 執行回測

#### 方法一：使用設定檔（推薦新手）

```bash
# 執行回測（會自動讀取 config/base.yaml 中的策略設定）
python scripts/run_backtest.py
```

#### 方法二：使用命令列參數（推薦進階用戶）

```bash
# 直接指定策略（覆蓋設定檔中的策略）
python scripts/run_backtest.py -s my_rsi_strategy

# 指定設定檔
python scripts/run_backtest.py -c config/my_rsi.yaml

# 只回測指定交易對
python scripts/run_backtest.py -s my_rsi_strategy --symbol BTCUSDT
```

**回測腳本會**：
1. 讀取設定檔（或使用命令列參數）
2. 找到對應的策略函式
3. 使用設定的參數執行回測
4. 產生報告和圖表

**輸出目錄組織**：

報告會自動按 **市場類型 → 策略名稱 → 執行類型 → 時間戳** 分類：

```
reports/
├── spot/                          ← 現貨
│   └── rsi_adx_atr/              ← 策略名稱
│       ├── backtest/             ← 回測結果
│       │   └── 20260213_114901/  ← 時間戳
│       ├── portfolio/            ← 組合回測
│       ├── validation/           ← 驗證報告
│       └── live/                 ← Paper/Real 狀態
│
└── futures/                       ← 合約
    └── rsi_adx_atr/
        ├── backtest/
        ├── portfolio/
        ├── validation/
        └── live/
```

路徑規則：`reports/{market_type}/{strategy}/{run_type}/{timestamp}/`

> 💡 **不用手動設定路徑**！系統會自動從 config 的 `market_type` 和 `strategy.name` 組合出正確的目錄。YAML 裡的 `output.report_dir` 只需要設成 `"./reports"`（或完全不寫，預設就是）。

### 3.4 查看結果

回測完成後，會在 `reports/{market_type}/{strategy}/backtest/{timestamp}/` 目錄產生：

1. **統計報告** (`stats_BTCUSDT.csv`)
   - 總收益率
   - 最大回撤
   - 夏普比率
   - 勝率
   - 總交易次數

2. **資金曲線圖** (`equity_curve_BTCUSDT.png`)
   - 價格走勢
   - 買賣訊號
   - 資金曲線
   - 回撤圖

### 3.5 理解回測結果

#### 關鍵指標說明

- **總收益率 (Total Return)**：策略的總收益百分比
  - 正數 = 盈利
  - 負數 = 虧損

- **最大回撤 (Max Drawdown)**：從最高點到最低點的最大跌幅
  - 越小越好
  - 超過 50% 通常風險較大

- **夏普比率 (Sharpe Ratio)**：風險調整後的收益
  - > 1.0：不錯
  - > 2.0：很好
  - < 0：策略表現差

- **勝率 (Win Rate)**：盈利交易佔總交易的比例
  - 通常 50% 以上較好

- **總交易次數 (Total Trades)**：交易頻率
  - 太少：可能錯過機會
  - 太多：手續費成本高

### 3.6 分析資金曲線

查看 `equity_curve_BTCUSDT.png`：

1. **價格和訊號圖**：
   - 綠點：買入訊號
   - 紅點：賣出訊號
   - 檢查訊號是否合理

2. **資金曲線圖**：
   - 是否穩定上升？
   - 是否有大幅回撤？
   - 是否長期虧損？

3. **回撤圖**：
   - 回撤是否在可接受範圍內？

---

## 第四步：優化參數

如果回測結果不理想，可以嘗試優化參數。

### 4.1 手動測試不同參數

修改 `config/base.yaml`：

```yaml
strategy:
  name: "my_rsi_strategy"
  params:
    period: 14        # 試試 10, 14, 20
    oversold: 30      # 試試 25, 30, 35
    overbought: 70    # 試試 65, 70, 75
```

然後重新執行回測，比較結果。

### 4.2 使用自動優化工具（推薦）

專案提供了參數優化工具，可以自動測試多個參數組合：

```bash
# 優化 RSI 策略參數
python scripts/optimize_params.py --strategy my_rsi_strategy --metric "Sharpe Ratio"
```

這會：
1. 測試所有參數組合
2. 找到表現最好的參數
3. 產生優化報告

### 4.3 綜合回測工具 ⭐ NEW

使用 `comprehensive_backtest.py` 可以一次測試多個維度：

```bash
# 完整測試（市場階段 + 倉位管理 + 出場策略 + 策略參數）
python scripts/comprehensive_backtest.py --symbol BTCUSDT

# 只測試特定維度
python scripts/comprehensive_backtest.py --symbol BTCUSDT --test position_sizing exit_strategy

# Futures 模式
python scripts/comprehensive_backtest.py --symbol BTCUSDT --market-type futures
```

**測試維度**：
| 維度 | 測試項目 |
|------|---------|
| market_regime | 牛市、熊市、震盪市、高/低波動 |
| position_sizing | 固定倉位、Kelly 公式（全/半/四分之一） |
| exit_strategy | ATR SL/TP、Trailing Stop、RSI Exit |
| strategy_params | 預設、積極、保守 |

### 4.4 優化指標選擇

可以選擇不同的優化指標：

- `Total Return [%]`：總收益率
- `Sharpe Ratio`：夏普比率（推薦）
- `Max Drawdown [%]`：最大回撤（越小越好）

```bash
# 優化夏普比率
python scripts/optimize_params.py --strategy my_rsi_strategy --metric "Sharpe Ratio"

# 優化總收益率
python scripts/optimize_params.py --strategy my_rsi_strategy --metric "Total Return [%]"
```

### 4.4 查看優化結果

優化完成後，會產生 `reports/{market_type}/{strategy}/optimize/optimization_BTCUSDT.csv`，包含：
- 所有參數組合
- 對應的表現指標
- 最佳參數組合

### ⚠️ 避免過度優化

**重要**：不要過度優化參數！

- 過度優化會導致過擬合
- 在歷史資料上表現好，不代表未來也會好
- 應該選擇在多個時間視窗都表現穩定的參數

---

## 第五步：驗證策略

在實盤前，必須驗證策略是否過擬合。

### 5.1 統一驗證入口

所有驗證功能已整合到 `validate.py`，**跑完會直接在終端用白話告訴你每項結果**，不用自己看數字查標準：

```bash
# 執行標準驗證套件（Walk-Forward + Monte Carlo + Cross-Asset + Kelly）
# ⭐ 推薦：新手直接跑這個，不用加任何參數
python scripts/validate.py -c config/rsi_adx_atr_rsi_exit.yaml

# 快速驗證（跳過耗時測試）
python scripts/validate.py -c config/rsi_adx_atr_rsi_exit.yaml --quick

# 完整驗證（包括一致性檢查）
python scripts/validate.py -c config/rsi_adx_atr_rsi_exit.yaml --full

# 只執行特定驗證
python scripts/validate.py -c config/rsi_adx_atr_rsi_exit.yaml --only walk_forward
python scripts/validate.py -c config/rsi_adx_atr_rsi_exit.yaml --only walk_forward,monte_carlo
```

> 💡 **不用背指標**！跑完之後終端會自動顯示每項測試的「白話翻譯 + 標準 + 判斷」，YAML 報告裡也有 `meaning` 欄位解釋。詳見 [§5.6 理解驗證結果](#56-理解驗證結果)。

**可用的驗證類型**：

| 驗證類型 | 用途（白話） | 大約耗時 |
|----------|-------------|---------|
| `walk_forward` | 策略在新數據上還行嗎？ | ~30s |
| `monte_carlo` | 最壞情況會虧多少？ | ~10s |
| `loao` | 去掉一個幣還行嗎？ | ~20s |
| `regime` | 高/低波動市場都行嗎？ | ~10s |
| `dsr` | Sharpe 是真的還是碰巧？ | ~5s |
| `pbo` | 過擬合機率多高？ | ~5s |
| `kelly` | 每次該投多少？ | ~15s |
| `consistency` | 實盤跟回測一致嗎？ | ~30s |

### 5.2 Walk-Forward Analysis

滾動視窗驗證會自動執行，將資料分成多個訓練/測試視窗：
- 在訓練集上優化參數
- 在測試集上驗證表現

**如何判斷過擬合**：
- ✅ 測試集表現接近訓練集 → 策略穩定
- ❌ 測試集表現遠差於訓練集 → 可能過擬合

### 5.3 Kelly 公式驗證

驗證策略是否適合使用 Kelly 倉位管理：

```bash
# 只執行 Kelly 驗證
python scripts/validate.py -c config/rsi_adx_atr.yaml --only kelly
```

**驗證內容**：
- 檢查統計穩定性（勝率、盈虧比是否穩定）
- 比較不同 Kelly fraction 的回測表現
- 給出是否使用 Kelly 的建議

**範例輸出**：
```
======================================================================
  💰 Kelly Formula Validation
======================================================================

  BTCUSDT:
    勝率: 55.2% (120/217)
    盈虧比: 1.45
    Full Kelly: 12.3%
    穩定性 (CV): 0.35
    推薦倉位: 25% Kelly
    原因: Calmar ratio 最優 (2.15)
    ✅ 推薦使用 25% Kelly = 3.1% 倉位
```

### 5.4 Live/Backtest 一致性驗證

驗證即時交易訊號與回測是否一致（檢測 look-ahead bias）：

```bash
# 執行一致性驗證（需要先運行 Paper Trading 一段時間）
python scripts/validate.py -c config/rsi_adx_atr.yaml --only consistency

# 完整驗證（包含一致性）
python scripts/validate.py -c config/rsi_adx_atr.yaml --full
```

**建議排程（cron）**：
```bash
# 每週日 00:00 執行一致性驗證
0 0 * * 0 cd /path/to/quant-binance-spot && python scripts/validate.py -c config/rsi_adx_atr.yaml --only consistency
```

### 5.5 ⭐ Pre-Deploy 一致性檢查（上架前必跑）

**每次部署策略到實盤之前，務必跑這個腳本！** 它會自動比對回測和實盤的所有路徑，13 項全過才放心上線。

```bash
# 標準檢查（建議每次部署前必跑）
python scripts/validate_live_consistency.py -c config/futures_rsi_adx_atr.yaml

# 詳細模式（顯示每個項目的細節）
python scripts/validate_live_consistency.py -c config/futures_rsi_adx_atr.yaml -v

# 只檢查特定項目
python scripts/validate_live_consistency.py -c config/futures_rsi_adx_atr.yaml --only signal,sltp
```

**檢查清單（13 項）**：

| # | 檢查項 | 說明 |
|---|--------|------|
| 1 | `config_passthrough` | YAML → backtest_dict → live 參數傳遞鏈完整 |
| 2 | `strategy_context` | StrategyContext (market_type, direction) 一致 |
| 3 | `strategy_function` | 回測和實盤使用同一個策略函數 |
| 4 | `signal_consistency` | 相同數據產生相同信號 |
| 5 | `signal_clip` | 信號 clip 邏輯一致（spot [0,1] / futures [-1,1]）|
| 6 | `entry_price` | close[N-1] vs open[N] 差距 < 0.5% |
| 7 | `sltp_formula` | SL/TP 方向公式正確（多倉 SL 在下方等）|
| 8 | `sltp_price_base` | SL/TP 基準價內部一致 |
| 9 | `position_sizing` | 倉位計算鏈路正確 |
| 10 | `fee_match` | 手續費設定 = 交易所費率 |
| 11 | `date_filter` | start/end 日期正確套用 |
| 12 | `cooldown` | 冷卻期設定一致 |
| 13 | `funding_rate_warning` | 合約資金費率未建模提醒 |

**結果解讀**：
- ✅ `PASS` — 完全一致
- ⚠️ `WARN` — 可接受差異（需了解影響）
- 🔴 `FAIL` — 不一致，必須修復後才能上線

**範例輸出**：
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📋 回測↔實盤一致性檢查報告
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✅ [PASS] config_passthrough    YAML → backtest_dict() 鏈路完整
  ✅ [PASS] strategy_context      market=futures, dir=both
  ✅ [PASS] signal_consistency    最後信號一致: 0.00
  ✅ [PASS] fee_match             fee=4bps = Binance Taker
  ⚠️  [WARN] funding_rate_warning  Futures 5x — 回測不計入 funding
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  結果: 12 PASS / 1 WARN / 0 FAIL (共 13 項)
  ⚠️  有警告項目，建議了解後再部署。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**可用的 `--only` 分組**：
| 分組 | 包含項目 |
|------|----------|
| `params` | config_passthrough, strategy_context |
| `strategy` | strategy_function |
| `signal` | signal_consistency, signal_clip |
| `entry` | entry_price |
| `sltp` | sltp_formula, sltp_price_base |
| `sizing` | position_sizing |
| `fee` | fee_match |
| `date` | date_filter |
| `cooldown` | cooldown |
| `funding` | funding_rate_warning |

### 5.6 理解驗證結果

#### 終端輸出

跑完驗證後，終端會直接顯示**白話總結**，告訴你每項測試做了什麼、數值多少、標準是什麼：

```
========================================================================
  📋 Validation Summary — 策略驗證總結
========================================================================

  每項測試檢查策略的不同面向，幫助你判斷策略是否可以上線。
  ✅ PASS = 通過   ⚠️ CHECK = 需注意   ❌ FAIL = 不建議上線

  ─────────────────────────────────────────────────────────────────
  ✅ PASS  Walk-Forward（前瞻驗證）
         測試方法: 用歷史訓練 → 在新數據上驗證，模擬真實使用場景
         績效衰退: -9.3%（標準: < 50% 為佳）
           BTCUSDT: 訓練 SR=3.79 → 測試 SR=4.20 (衰退 -10.7%)
           ETHUSDT: 訓練 SR=4.95 → 測試 SR=5.35 (衰退 -7.9%)

  ✅ PASS  Monte Carlo（壓力測試）
         測試方法: 隨機模擬 10000 種市場情境，看最差情況虧多少
         平均 VaR 95%: 0.62%（意思：95% 的情況下單日虧損 < 此值）

  ✅ PASS  DSR（校正 Sharpe Ratio）
         測試方法: 把回測裡「調了很多參數」的因素扣除，看 Sharpe 是否仍顯著
         校正 SR: 4.77, p-value: 0.0000（標準: p < 0.05）

  ✅ PASS  PBO（過擬合機率）
         測試方法: 用排列組合計算「回測好但實盤差」的機率
         過擬合機率: 30.0%（標準: < 50%）

  ✅ PASS  Kelly（最佳倉位驗證）
         測試方法: 用歷史勝率+盈虧比算最佳倉位，並檢查穩定性
           BTCUSDT: 勝率=54.1%, 盈虧比=1.55, 建議=25% Kelly = 6.1%
           ETHUSDT: 勝率=56.0%, 盈虧比=1.89, 建議=25% Kelly = 8.2%

  ─────────────────────────────────────────────────────────────────
  🎉 Overall: ✅ 策略驗證通過 — 可以考慮上線！

  💡 提示：驗證通過≠保證賺錢，只是表示策略在統計上有合理性。
           實盤會受滑價、funding rate、流動性等因素影響。
========================================================================
```

> 💡 不需要記住每個指標的意思，終端輸出已經幫你寫好了。

#### YAML 報告

報告會保存在 `reports/{market_type}/{strategy}/validation/{timestamp}/validation_summary.yaml`。

**新版 YAML 每個項目都附有 `threshold`（判斷標準）和 `meaning`（白話解釋）**，打開就看得懂：

```yaml
timestamp: '2026-02-13T12:01:49'
tests:
  walk_forward:
    passed: true
    avg_degradation: -9.3%
    threshold: < 50%               # ← 判斷標準
    meaning: 訓練期→測試期的績效衰退幅度，越低代表策略越穩健
    per_symbol:
      BTCUSDT:
        train_sharpe: 3.79
        test_sharpe: 4.20
        degradation: -10.7%
        splits_completed: 5        # ← 5 折全部成功
      ETHUSDT:
        train_sharpe: 4.95
        test_sharpe: 5.35
        degradation: -7.9%
        splits_completed: 5

  monte_carlo:
    passed: true
    avg_var_95: 0.62%
    threshold: 日 VaR 95% < 30%
    meaning: 模擬 10000 次隨機情境，估計最差情況的單日虧損

  dsr:
    passed: true
    deflated_sharpe: 4.77
    p_value: 0.0
    threshold: p-value < 0.05 (統計顯著)
    meaning: 考慮了『試了很多參數才找到這個結果』的情況後，Sharpe 是否仍然顯著？

  pbo:
    passed: true
    pbo_pct: 30.0%
    threshold: PBO < 50%
    meaning: 用交叉驗證估計策略是『真的好』還是『碰巧好』的機率

  kelly:
    passed: true
    suitable_assets: 2/2
    threshold: 所有幣種都適合使用 Kelly
    meaning: 根據歷史勝率和盈虧比，計算最佳倉位大小並檢驗其穩定性
    per_symbol:
      BTCUSDT: { win_rate: 54.1%, win_loss_ratio: 1.55, recommended: '25% Kelly = 6.1%' }
      ETHUSDT: { win_rate: 56.0%, win_loss_ratio: 1.89, recommended: '25% Kelly = 8.2%' }

overall_passed: true
```

#### 產出的報告文件

```
reports/{market_type}/{strategy}/validation/{timestamp}/
├── validation_summary.yaml   # 總結（上面的 YAML）
├── walk_forward_BTCUSDT.csv  # Walk-Forward 每折詳細數據
├── walk_forward_ETHUSDT.csv
├── regime_results.csv        # 高/低波動市場表現比較
├── kelly_BTCUSDT.txt         # Kelly 詳細分析報告
├── kelly_ETHUSDT.txt
└── kelly_summary.csv         # Kelly 倉位建議彙總
```

#### 快速判斷表

| 測試項目 | 在看什麼？ | ✅ 好 | ❌ 差 |
|----------|-----------|-------|-------|
| Walk-Forward | 策略在新數據上還行嗎？ | 衰退 < 30% | 衰退 > 50% |
| Monte Carlo | 最壞情況虧多少？ | VaR 95% 合理 | VaR 極端 |
| DSR | Sharpe 是真的嗎？ | p-value < 0.05 | p > 0.05 |
| PBO | 是不是碰巧調出來的？ | PBO < 50% | PBO > 50% |
| Kelly | 每次該下多大注？ | 適合 + 穩定 | 不適合 |

#### PBO（過擬合機率）白話解釋

PBO 衡量「訓練表現最好的策略在測試時表現差」的機率：

```
簡單比喻：
你考了 20 次模擬考（= 20 組參數），選了分數最高的去考真的。
PBO 就是在問：「模擬考第一名，真正考試會不會考砸？」

PBO = 30% → 只有 30% 的機會考砸，策略蠻穩的
PBO = 70% → 70% 的機會考砸，很可能是運氣好
```

**PBO 高但策略仍可用的情況**：
- 所有 Test Sharpe 都 > 0（仍然盈利）
- Walk-Forward 衰退 < 30%
- Paper Trading 表現與回測一致

#### 如果驗證沒過怎麼辦？

1. **簡化策略** — 參數越少，越不容易過擬合
2. **減少參數數量** — 只留最關鍵的 3-5 個
3. **使用更保守的參數** — 不要追求最高回測績效
4. **增加歷史資料** — 數據越長，統計越可靠
5. **先跑 Paper Trading** — 用真實市場驗證 2 週以上

---

## 第六步：風險管理

在實盤前，應該新增風險管理。

### 6.1 資料品質檢查

專案會自動檢查資料品質，但你可以手動驗證：

```python
from qtrade.data import validate_data_quality

report = validate_data_quality(df)
if not report.is_valid:
    print("資料有問題，需要清洗")
```

### 6.2 設定風險限制

在 `config/base.yaml` 中設定：

```yaml
risk:
  max_position_pct: 1.0      # 最大倉位 100%
  max_drawdown_pct: 0.5     # 最大回撤限制 50%
  max_leverage: 1.0          # 最大槓桿（現貨為 1.0）
```

### 6.3 倉位管理

可以使用不同的倉位管理方法：

```python
from qtrade.risk import FixedPositionSizer, KellyPositionSizer

# 固定倉位（80%）
sizer = FixedPositionSizer(position_pct=0.8)

# Kelly 公式（更科學，但需要歷史資料）
sizer = KellyPositionSizer(
    win_rate=0.6,      # 勝率 60%
    avg_win=0.05,      # 平均盈利 5%
    avg_loss=0.03,     # 平均虧損 3%
    kelly_fraction=0.5  # 使用 50% Kelly（保守）
)
```

### 6.4 Kelly 倉位設定 ⭐ NEW

如果 Kelly 驗證通過，可以在設定檔中啟用：

```yaml
position_sizing:
  method: "kelly"
  kelly_fraction: 0.25  # Quarter Kelly（保守推薦）
  min_trades_for_kelly: 30  # 至少 30 筆交易才啟用
```

**Kelly Fraction 建議**：
- `0.25`（Quarter Kelly）：最保守，推薦新手
- `0.5`（Half Kelly）：中等風險
- `1.0`（Full Kelly）：最激進，不推薦

詳細參數說明請參考 `config/futures_rsi_adx_atr.yaml` 中的註解。

---

## 第七步：即時交易 ⭐ NEW

驗證通過後，可以開始即時交易。

### 7.1 Paper Trading（模擬交易）

**推薦先用 Paper Trading 觀察策略表現至少 1-2 週**。

```bash
# 啟動 Paper Trading（推薦使用經驗證的 RSI Exit 配置）
python scripts/run_live.py -c config/rsi_adx_atr_rsi_exit.yaml --paper

# 只交易 BTCUSDT
python scripts/run_live.py -c config/rsi_adx_atr_rsi_exit.yaml --paper --symbol BTCUSDT

# 立即執行一次（不等待 K 線收盤）
python scripts/run_live.py -c config/rsi_adx_atr_rsi_exit.yaml --paper --once

# 查看帳戶狀態
python scripts/run_live.py -c config/rsi_adx_atr_rsi_exit.yaml --status
```

**Paper Trading 特點**：
- 不需要 API Key
- 模擬帳戶，不會虧真錢
- 狀態儲存在 `reports/{market_type}/{strategy}/live/paper_state.json`

### 7.2 Telegram 通知設定

在 `.env` 中設定以下變數即可自動啟用通知：

```bash
TELEGRAM_BOT_TOKEN=123456:ABC-DEF
TELEGRAM_CHAT_ID=987654321
```

**如何取得**：
1. 在 Telegram 搜尋 `@BotFather`，建立新 Bot 取得 Token
2. 搜尋 `@userinfobot`，取得你的 Chat ID

### 7.2.1 Telegram Command Bot ⭐ NEW

除了被動通知，還支援**互動式指令**查詢帳戶狀態。Bot 啟動時會自動向 Telegram 註冊命令選單，輸入 `/` 即可看到命令提示直接點選。

#### 啟動方式

Telegram Bot 是一個**常駐服務**，與 cron 定時交易搭配使用：
- **cron** 負責每小時跑一次交易（`run_live.py --once`）
- **Telegram Bot** 負責 24/7 接收你的查詢命令

```bash
# 啟動 Telegram Bot（需要 Binance API Key 才能查詢帳戶）
python scripts/run_telegram_bot.py -c config/futures_rsi_adx_atr.yaml --real

# 背景運行（正式環境，SSH 斷線也不會停）
nohup python scripts/run_telegram_bot.py -c config/futures_rsi_adx_atr.yaml --real > logs/telegram_bot.log 2>&1 &

# 純測試（不連 Binance，只測 /ping /help）
python scripts/run_telegram_bot.py
```

> 💡 **說明**：`--real` 不代表會下單。Telegram Bot 永遠是 **dry-run 模式**（只查詢、不下單），`--real` 只是讓它連接 Binance API 來讀取帳戶資訊。

#### 可用指令

| 指令 | 說明 | 需要 Broker |
|------|------|:-----------:|
| `/help` | 查看所有指令 | ❌ |
| `/ping` | 測試 Bot 連線 | ❌ |
| `/status` | 帳戶總覽 + 持倉 + SL/TP 摘要 | ✅ |
| `/balance` | 查看帳戶餘額 | ✅ |
| `/positions` | 當前持倉詳細資訊（含 SL/TP 價格與預估盈虧）| ✅ |
| `/trades` | 最近 10 筆成交記錄 | ✅ |
| `/pnl` | 今日盈虧（已實現 + 資金費率 + 手續費）| ✅ |
| `/signals` | 交易信號（優先讀 cron 快照，確保一致）| ✅ |
| `/stats` | 交易統計（勝率、累積 PnL）| ✅ |

#### 指令輸出範例

**`/status`** — 一眼看清帳戶狀態：
```
💼 帳戶狀態
💰 總權益: $1,472.36
💵 可用餘額: $1,219.80
📈 未實現盈虧: $+12.07

📋 持倉 (1)
🟢 ETHUSDT [SHORT] +12.03
   🛡️ SL: $2,050.00 (-$48.20)
   🎯 TP: $1,780.00 (+$45.60)
```

**`/pnl`** — 今日盈虧明細：
```
📈 今日盈虧 (02-12 UTC)
💰 總計: $+87.55
✅ 已實現: $+89.96
🧑‍💼 未實現: $+0.00
💸 手續費: $-2.38
🔄 資金費率: $-0.02
```

**通知內容範例**（自動推播）：
```
🟢 開倉 BTCUSDT
方向: LONG
數量: 0.015 BTC
入場價: $67,500
止損: $66,000 (-2.2%, -$22.50)
止盈: $72,000 (+6.7%, +$67.50)
```

### 7.3 Real Trading（真實交易）

⚠️ **警告：真實交易會使用真金白銀，請謹慎操作！**

**前置準備**：
1. 在 Binance 建立 API Key
2. 設定環境變數

```bash
# .env
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
```

**檢查連線**：
```bash
python scripts/run_live.py -c config/rsi_adx_atr.yaml --check
```

**Dry-run 測試**（不下單，只看訊號）：
```bash
python scripts/run_live.py -c config/rsi_adx_atr.yaml --real --dry-run --once
```

**真實交易**：
```bash
python scripts/run_live.py -c config/rsi_adx_atr.yaml --real --once
```

### 7.4 使用 Cron 自動執行

```bash
# 編輯 crontab
crontab -e

# 每小時第 5 分鐘執行（配合 1h K 線）
# ⭐ 建議在 :05 而非 :00 執行，讓 K 線數據穩定
5 * * * * cd /path/to/quant-binance-spot && python scripts/run_live.py -c config/rsi_adx_atr_rsi_exit.yaml --paper --once >> logs/live.log 2>&1

# 每 4 小時執行（配合 4h K 線）
5 */4 * * * cd /path/to/quant-binance-spot && python scripts/run_live.py -c config/rsi_adx_atr_rsi_exit.yaml --paper --once >> logs/live.log 2>&1
```

**重要**：
- `--once`：執行一次後立即退出，**cron 必須加這個參數**
- 不加 `--once` 會進入 daemon 模式，持續等待下一根 K 線
- 設在 `:05` 而非 `:00`：讓 K 線數據穩定，避免假信號
- **v2.8**：如果 `live.kline_cache: true`，cron 自動使用增量快取（首次會建立種子，後續只拉增量）

### 7.5 部署到雲端伺服器（Oracle Cloud）⭐ NEW

建議使用 Oracle Cloud 免費方案跑 Bot，24 小時不間斷。

#### 7.5.1 首次部署

```bash
# 1. SSH 連進伺服器
ssh ubuntu@<你的 Oracle Cloud IP>

# 2. 安裝 Python 環境
sudo apt update && sudo apt install -y python3.11 python3.11-venv git

# 3. 拉專案
git clone https://github.com/<你的帳號>/quant-binance-spot.git
cd quant-binance-spot

# 4. 建立虛擬環境並安裝依賴
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 5. 設定環境變數
cp .env.example .env   # 或手動建立
nano .env              # 填入 API Key、Telegram Token 等

# 6. 下載所有幣種的 K 線 + Funding Rate
PYTHONPATH=src python scripts/download_data.py -c config/futures_rsi_adx_atr.yaml
PYTHONPATH=src python scripts/download_data.py -c config/futures_rsi_adx_atr.yaml --funding-rate

# 7. 用 tmux 啟動 WebSocket Runner（推薦）
mkdir -p logs
tmux new -d -s trading "cd ~/quant-binance-spot && source .venv/bin/activate && PYTHONPATH=src python scripts/run_websocket.py -c config/futures_rsi_adx_atr.yaml --real 2>&1 | tee logs/websocket.log"

# 8. 設定 cron（輔助任務，不含交易 — 交易由 WebSocket 負責）
crontab -e
# 詳見下方 cron 設定範例
```

#### 7.5.2 更新代碼 & 重啟

當你在本機改完 code 並 `git push` 後，在伺服器上執行：

```bash
# SSH 連進伺服器
ssh ubuntu@<你的 Oracle Cloud IP>

# 進入專案
cd ~/quant-binance-spot
source .venv/bin/activate

# 拉最新代碼
git pull

# ⭐ 關鍵：清除 Python 快取（避免舊 .pyc 導致新功能不生效）
./scripts/setup_cron.sh --update
# 或手動：
# find . -name "*.pyc" -delete && find . -name "__pycache__" -exec rm -rf {} +

# 找到舊的 bot 進程並停掉
ps aux | grep run_telegram_bot
kill <PID>

# 重新啟動
nohup python scripts/run_telegram_bot.py -c config/futures_rsi_adx_atr.yaml --real > logs/telegram_bot.log 2>&1 &

# 確認有跑起來
tail -f logs/telegram_bot.log
# 看到「✅ 已註冊 X 個命令到 Telegram 選單」就代表成功
# 按 Ctrl+C 退出 tail（bot 繼續在背景跑）
```

> ⚠️ **重要：`git pull` 後必須清除 `.pyc` 快取！** Python 會優先讀取已編譯的 `.pyc` 而非最新的 `.py` 原始碼。如果不清除，新功能（如 `kline_cache`）可能**完全不生效**，且不會報錯。用 `./scripts/setup_cron.sh --update` 一鍵完成。

> 💡 **小技巧**：cron 交易不需要重啟，`git pull` + 清 `.pyc` 後下一輪 cron 自動用新代碼。只有 Telegram Bot 常駐服務需要手動重啟。

> ⚠️ **v2.8 注意**：K 線快取（`kline_cache/`）會在 `git pull` 後保留。如果你改了策略邏輯，建議刪除快取讓它重新建立：`rm -rf reports/futures/*/live/kline_cache/`

#### 7.5.3 查看 Log

```bash
# 查看 Telegram Bot log
tail -100 logs/telegram_bot.log

# 查看交易 log（cron 輸出）
tail -100 logs/live.log

# 即時追蹤
tail -f logs/telegram_bot.log
```

---

## 第八步：監控與維運 ⭐ NEW

### 8.1 系統健康檢查

```bash
# 執行檢查並輸出結果
python scripts/health_check.py

# 只在異常時發送 Telegram 通知
python scripts/health_check.py --notify

# 總是發送通知（包括正常時）
python scripts/health_check.py --notify --notify-on-ok

# 指定設定檔
python scripts/health_check.py --config config/rsi_adx_atr.yaml

# 檢查真實交易狀態（而非 Paper Trading）⭐ NEW
python scripts/health_check.py --config config/rsi_adx_atr.yaml --real

# 輸出 JSON 格式
python scripts/health_check.py --json
```

**檢查項目**：
- 📁 磁碟空間使用率
- 💾 記憶體使用率
- 📊 交易狀態檔案是否過期（Paper 或 Real）
- 🔄 系統執行狀態

**建議 cron 設定**：
```bash
# 每 30 分鐘檢查一次，異常時通知
*/30 * * * * cd /path/to/quant-binance-spot && python scripts/health_check.py --notify >> logs/health.log 2>&1
```

### 8.2 每日績效報表

```bash
# 手動執行
python scripts/daily_report.py -c config/rsi_adx_atr.yaml

# 只列印不發送 Telegram
python scripts/daily_report.py -c config/rsi_adx_atr.yaml --print-only
```

**報表內容**：
- 📊 帳戶權益、收益率、最大回撤
- 📋 當前持倉明細
- 📈 今日交易記錄
- 📉 過去 7 天收益趨勢

**建議 cron 設定**：
```bash
# 每天 UTC 00:05 執行
5 0 * * * cd /path/to/quant-binance-spot && python scripts/daily_report.py -c config/rsi_adx_atr.yaml >> logs/daily_report.log 2>&1
```

### 8.3 完整生產環境設定範例

一台雲端伺服器上，你需要跑這些東西：

| 元件 | 執行方式 | 說明 |
|------|---------|------|
| **交易引擎** | tmux 常駐 | WebSocket Runner（即時觸發，延遲 <1s） |
| **Telegram Bot** | WebSocket Runner 內建 | 自動啟用，不需額外啟動 |
| **輔助任務** | cron 定時 | 報表、健康檢查、數據更新、Alpha 監控 |

#### 交易引擎（tmux WebSocket）

```bash
# 啟動（或重啟）
tmux kill-session -t trading 2>/dev/null
tmux new -d -s trading "cd ~/quant-binance-spot && source .venv/bin/activate && PYTHONPATH=src python scripts/run_websocket.py -c config/futures_rsi_adx_atr.yaml --real 2>&1 | tee logs/websocket.log"
```

#### Cron Jobs 設定（輔助任務）

```bash
# 編輯 crontab
crontab -e

# ⚠️ 交易由 WebSocket Runner 負責，cron 裡的 run_live.py 要註解掉！

# === 數據更新 ===
# 每 8 小時下載 Funding Rate
10 */8 * * * cd ~/quant-binance-spot && source .venv/bin/activate && python scripts/download_data.py -c config/futures_rsi_adx_atr.yaml --funding-rate >> logs/download_data.log 2>&1

# === 監控 ===
# 每 30 分鐘健康檢查
*/30 * * * * cd ~/quant-binance-spot && source .venv/bin/activate && python scripts/health_check.py -c config/futures_rsi_adx_atr.yaml --real --notify >> logs/health.log 2>&1

# === 報表 ===
# 每天 00:05 UTC 發送日報
5 0 * * * cd ~/quant-binance-spot && source .venv/bin/activate && python scripts/daily_report.py -c config/futures_rsi_adx_atr.yaml >> logs/daily_report.log 2>&1

# === 驗證 ===
# 每週日 01:00 UTC 一致性驗證
0 1 * * 0 cd ~/quant-binance-spot && source .venv/bin/activate && python scripts/validate.py -c config/futures_rsi_adx_atr.yaml --quick >> logs/validation.log 2>&1

# === Alpha Decay 監控 ===
# 每週日 01:00 UTC 執行 IC 分析 + Telegram 通知
0 1 * * 0 cd ~/quant-binance-spot && source .venv/bin/activate && bash scripts/cron_alpha_monitor.sh >> logs/alpha_monitor.log 2>&1

# === Log 清理 ===
# 每週一 04:00，保留 7 天
0 4 * * 1 find ~/quant-binance-spot/logs -name "*.log" -mtime +7 -delete
```

#### Telegram Bot 常駐

```bash
# 背景啟動（SSH 斷線也不會停）
cd ~/quant-binance-spot && source .venv/bin/activate
nohup python scripts/run_telegram_bot.py -c config/futures_rsi_adx_atr.yaml --real > logs/telegram_bot.log 2>&1 &

# 確認啟動成功（應看到「增量 K 線快取已啟用」和「已註冊 X 個命令」）
sleep 3 && tail -5 logs/telegram_bot.log
```

**⚠️ 常見錯誤**：
- ❌ cron 忘記加 `--once`，導致多個 daemon 同時運行
- ❌ cron 設在 `:00` 整點，K 線數據可能還不穩定
- ❌ cron 忘記 `source .venv/bin/activate`，找不到 Python 套件
- ❌ `git pull` 後忘記重啟 Telegram Bot（cron 會自動用新代碼，但 Bot 不會）
- ❌ `git pull` 後沒清 `.pyc` 快取（新功能可能不生效，見 [Q13](#q13-git-pull-後-kline_cache-沒有生效-new)）
- ❌ 改了策略邏輯後沒刪 K 線快取（舊快取的歷史數據可能與新邏輯不匹配）

---

## 合約交易教學 ⭐ NEW

本專案支援現貨 (Spot) 和合約 (Futures) 交易，使用相同的開發流程。

### 現貨 vs 合約

| 功能 | 現貨 🟢 | 合約 🔴 |
|------|---------|---------|
| 做多 | ✅ | ✅ |
| 做空 | ❌ | ✅ |
| 槓桿 | 1x | 1x-125x |
| 信號範圍 | [0, 1] | [-1, 1] |
| 風險 | 較低 | 較高 |

### 10.1 配置合約策略

建立合約策略配置檔（例如 `config/futures_rsi_adx_atr.yaml`）：

```yaml
# 合約策略配置
market:
  symbols: ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
  interval: "1h"
  start: "2022-01-01"
  end: null
  market_type: "futures"  # ⭐ 關鍵：設定為 futures

# 合約專屬配置
futures:
  leverage: 5             # 槓桿倍數（建議 1-5 倍）
  margin_type: "ISOLATED" # ISOLATED（逐倉）或 CROSSED（全倉）
  position_mode: "ONE_WAY" # ONE_WAY（單向）或 HEDGE（雙向）

strategy:
  name: "rsi_adx_atr"
  params:
    rsi_period: 10
    oversold: 30           # RSI < 30 → 做多
    overbought: 70         # RSI > 70 → 做空（合約專用）
    min_adx: 15
    # ...其他參數

backtest:
  initial_cash: 10000
  fee_bps: 4              # 合約手續費較低
  slippage_bps: 3
  trade_on: "next_open"

# 風控（合約建議更保守）
risk:
  max_drawdown_pct: 0.15  # 15% 熔斷

# 實盤專屬配置 (v2.8)
live:
  kline_cache: true           # 增量 K 線快取（策略從 bar 0 跑到最新，與回測一致）
  flip_confirmation: false    # 方向切換 2-tick 確認（快取模式下不需要）

# Telegram 通知（可與現貨使用不同 Bot）
notification:
  telegram_bot_token: ${FUTURES_TELEGRAM_BOT_TOKEN}
  telegram_chat_id: ${FUTURES_TELEGRAM_CHAT_ID}
  prefix: "🔴 [FUTURES]"
  enabled: true

output:
  report_dir: "./reports"   # 路徑會自動組合為 reports/futures/rsi_adx_atr/...
```

> 💡 **`live` 區段是 v2.8 新增的**。詳見 [§11 增量 K 線快取](#增量-k-線快取)。

### 10.2 策略做空信號

策略可以輸出 [-1, 1] 範圍的信號：
- `1.0`：滿倉做多
- `0.0`：空倉
- `-1.0`：滿倉做空

```python
@register_strategy("my_futures_strategy")
def generate_positions(df: pd.DataFrame, ctx: StrategyContext, params: dict) -> pd.Series:
    rsi = calculate_rsi(df["close"], period=14)
    
    pos = pd.Series(0.0, index=df.index)
    
    # 做多條件
    pos[rsi < 30] = 1.0
    
    # 做空條件（僅合約有效）
    if ctx.supports_short:  # ⭐ 檢查是否支援做空
        pos[rsi > 70] = -1.0
    
    return pos.shift(1).fillna(0.0)
```

**注意**：現貨模式下，負數信號會自動 clip 到 0，策略程式碼不需要特別處理。

### 10.3 下載合約資料

```bash
# 下載合約 K 線資料
python scripts/download_data.py -c config/futures_rsi_adx_atr.yaml

# 資料會儲存到 data/binance/futures/1h/
```

### 10.4 回測合約策略

```bash
# 執行合約回測（支援做空模擬）
python scripts/run_backtest.py -c config/futures_rsi_adx_atr.yaml
```

### 10.5 Paper Trading（合約）

```bash
# 啟動合約 Paper Trading
python scripts/run_live.py -c config/futures_rsi_adx_atr.yaml --paper --once

# 查看狀態
python scripts/run_live.py -c config/futures_rsi_adx_atr.yaml --status
```

輸出範例：
```
🔴 Paper Trading [FUTURES]
   槓桿: 3x

──────────────────────────────────────────────────
  BTCUSDT: SHORT 50%, price=67404.28, RSI=75.5, ADX=34.48
  ETHUSDT: LONG 30%, price=1974.45, RSI=25.2, ADX=25.0

==================================================
  Paper Trading 帳戶摘要 🔴 [FUTURES (3x)]
==================================================
  初始資金:   $10,000.00
  當前現金:   $5,000.00
  總權益:     $10,500.00
  總收益:     +5.00%
  交易筆數:   5
  BTCUSDT [SHORT]: 0.100000 @ 68000.00 (PnL: +$600.00 📈)
==================================================
```

### 10.6 測試合約功能

執行測試腳本驗證功能：

```bash
# 執行合約功能測試
python scripts/test_futures_manual.py
```

### 10.7 合約 Cron 設定

現貨和合約可以同時運行，互不干擾：

```bash
# === 現貨策略（Paper Trading）===
5 * * * * cd /opt/qtrade && python scripts/run_live.py -c config/rsi_adx_atr.yaml --paper --once >> logs/spot_live.log 2>&1

# === 合約策略（Real Trading）===
5 * * * * cd /opt/qtrade && python scripts/run_live.py -c config/futures_rsi_adx_atr.yaml --real --once >> logs/futures_live.log 2>&1
```

**注意**：
- 設在 `:05` 而非 `:00`，讓 K 線數據穩定
- `--once` 是必須的，否則會進入 daemon 模式
- 現貨用 `--paper`，合約確認穩定後用 `--real`

### 10.8 自動止損止盈 (SL/TP) ⭐ NEW

合約交易支援**自動掛止損止盈單**，不需手動監控。

#### 工作原理

```
開倉 → 自動計算 SL/TP 價格（基於 ATR）
     → 透過 Binance Algo Order API 掛條件單
     → 每次 cron 執行時檢查並補掛遺漏的單（reconciliation）
```

**條件單 API 自動選擇**

Binance 有兩種條件單 API，**系統會自動選擇可用的那個**，你不用管：

| API | 端點 | 說明 |
|-----|------|------|
| 標準條件單 | `POST /fapi/v1/order` | `STOP_MARKET`、`TAKE_PROFIT_MARKET`，**優先使用** |
| Algo Order | `POST /fapi/v1/algo/futures/newOrderVp` | 當標準 API 回傳 `-4120` 時**自動降級** |

> 💡 **白話**：有些交易對（如 ETHUSDT）在某些帳戶上不支援標準條件單，Binance 會回傳 `-4120` 錯誤。系統偵測到後會自動改用 Algo Order API，**全自動，不需要任何設定**。如果你在 log 中看到 `-4120` 錯誤，不用擔心。

#### 配置範例

```yaml
strategy:
  params:
    stop_loss_atr: 1.5    # 止損 = 入場價 ± 1.5×ATR
    take_profit_atr: 4.0  # 止盈 = 入場價 ± 4.0×ATR
```

| 方向 | 止損 (SL) | 止盈 (TP) |
|------|-----------|-----------|
| LONG | 入場價 - 1.5×ATR | 入場價 + 4.0×ATR |
| SHORT | 入場價 + 1.5×ATR | 入場價 - 4.0×ATR |

#### SL/TP 自動補掛（Reconciliation）

每次 cron 執行時，系統會自動檢查並修復 SL/TP：

| 狀況 | 系統行為 |
|------|---------|
| 有持倉但沒有 SL | 自動補掛 SL |
| 有持倉但沒有 TP | 自動補掛 TP |
| 平倉了但還有 SL/TP | 自動取消 + 清理快取 |
| SL/TP 被手動取消 | 下次 cron 自動補掛 |
| SL/TP 被交易所觸發 | 偵測到無持倉 → 清理快取 |

> 💡 簡單說：**你不用管 SL/TP**，系統每小時自動幫你確認、補掛、清理。

#### SL/TP 去重機制 ⭐ NEW

系統會**自動檢查是否已有相同的 SL/TP 掛單**，避免重複下單：

1. **觸發價比對**：新 SL/TP 與現有掛單的觸發價差距 < 0.2% → 跳過
2. **方向比對**：只比對相同持倉方向的掛單
3. **跨 API 比對**：同時檢查標準 API 和 Algo Order API 的掛單

**常見場景**：
- cron 每小時跑一次 → ATR 微調導致 SL/TP 價格小幅變動 → 差距 < 0.2%，**不會重複掛單**
- bot 重啟後 → 讀取快取 → 知道已經掛了哪些單 → **不會重複掛單**

#### SL/TP 持久化快取 ⭐ NEW

SL/TP 訂單資訊保存在 `reports/{market_type}/{strategy}/live/algo_orders_cache.json`，**即使 bot 重啟也不會遺失**。

完整生命週期：

```
開倉 → 掛 SL + TP → 寫入快取檔
  ↓
每次 cron → 讀快取 → 去重檢查 → 補掛缺失的
  ↓
平倉 → 取消 SL + TP → 清理快取
  ↓
SL/TP 被觸發（交易所端） → 下次 cron 偵測無持倉 → 自動清理快取
```

> ⚠️ **注意**：如果你在 Binance 手動取消了 SL/TP，下次 cron 會自動重新掛上（因為系統偵測到「有持倉但無 SL/TP」）。如果你真的不想要 SL/TP，需要在 YAML 中把 `stop_loss_atr` 和 `take_profit_atr` 設成 `null`。

#### Telegram 通知範例

開倉時自動推播：
```
🟢 開倉 BTCUSDT
方向: LONG
入場價: $67,500
止損: $66,000 (-2.2%, -$22.50)  ← 預估虧損
止盈: $72,000 (+6.7%, +$67.50)  ← 預估盈利
```

用 `/status` 命令也能看到每個持倉的 SL/TP 和預估盈虧：
```
📋 持倉 (1)
🟢 ETHUSDT [SHORT] +12.03
   🛡️ SL: $2,050.00 (-$48.20)
   🎯 TP: $1,780.00 (+$45.60)
```

#### SL/TP 觸發後的冷卻機制

當 SL/TP 被觸發後，系統有**冷卻機制**（cooldown）防止立即反向開倉：

```yaml
strategy:
  params:
    cooldown_bars: 4  # SL/TP 觸發後等待 4 根 K 線才能重新開倉
```

**注意**：
- 平倉時會自動取消 **SL 和 TP**，並清理快取
- 如果 SL/TP 被取消，下次 cron 會自動補掛
- Paper Trading 不會真的掛單，只是模擬
- 去重機制容差 0.2%：ATR 微調不會產生重複單

### 10.9 熔斷機制 (Circuit Breaker) ⭐ NEW

當回撤超過設定閾值時，系統會自動停止交易：

```yaml
risk:
  max_drawdown_pct: 0.65  # 65% 回撤觸發熔斷（5x 槓桿建議）
```

**工作原理**：
1. 每次 cron 執行時計算當前回撤
2. 如果回撤 > `max_drawdown_pct`，停止開新倉
3. 發送 Telegram 警告通知
4. 可手動重置或等待回撤恢復

**槓桿與熔斷建議**：
| 槓桿 | 建議熔斷線 | 說明 |
|------|-----------|------|
| 1x | 30% | 現貨保守 |
| 3x | 45% | 低槓桿 |
| 5x | 65% | 中槓桿 |
| 10x+ | 不建議 | 風險極高 |

### 10.9.1 合約風險提醒 ⚠️

1. **槓桿風險**：高槓桿會放大盈虧，建議新手用 1-3 倍
2. **強平風險**：價格劇烈波動可能觸發強制平倉
3. **資金費率**：合約持倉需要支付/收取資金費率（回測不含此成本）
4. **建議**：
   - 先用 Paper Trading 測試至少 2 週
   - 合約 max_drawdown_pct 設定更保守（如 10-15%）
   - 新手先用 `allocation: 0.35`，熟悉後再加大
   - 總曝險 > 100% 前先看懂 [§10.12 多幣種倉位分配](#1012-多幣種倉位分配--new)

### 10.10 Binance 帳戶模式說明

Binance Futures 有兩種持倉模式：

| 模式 | 說明 | 特點 |
|------|------|------|
| **One-Way Mode** | 同一交易對只能單向持倉 | 做多再做空會互相抵消 |
| **Hedge Mode** | 可同時持有多空倉位 | 需要指定 `positionSide` |

**如何查看/切換**：
- Binance App → Futures → 設定 ⚙️ → Position Mode

**注意**：如果你的帳戶是 **Hedge Mode**，本專案已自動支援，無需額外設定。

### 10.11 回測數字的現實預期

⚠️ **回測報酬不等於實際報酬！**

| 影響因素 | 回測假設 | 實際情況 |
|----------|----------|----------|
| 滑價 | 0.03% | 可能 0.1-0.5% |
| 市場衝擊 | 無 | 大單會影響價格 |
| 流動性 | 隨時成交 | 可能無法成交 |
| 資金容量 | 無限 | 有上限 |

**經驗法則**：
```
實際報酬 ≈ 回測報酬 ÷ 3~5

回測年化 400% → 實際預期 80-130%
回測年化 100% → 實際預期 20-35%
```

**回測的正確用法**：
- ✅ 比較不同策略的優劣
- ✅ 驗證策略邏輯是否合理
- ❌ 直接當作收益預測
- ❌ 追求最高回測報酬

### 10.12 多幣種倉位分配 ⭐ NEW

#### 倉位大小由什麼決定？

很多人以為「加大槓桿 = 加大倉位」，**這是錯的**。

| 參數 | 作用 | 在哪裡設定 |
|------|------|-----------|
| `portfolio.allocation` | 每個幣分配多少 % 權益 | `config/*.yaml` |
| `portfolio.cash_reserve` | 保留多少 % 現金不動 | `config/*.yaml` |
| `futures.leverage` | 只影響保證金需求 | `config/*.yaml` |

```
⚠️ 槓桿 ≠ 倉位大小！

5x 槓桿、分配 100% 權益 → 倉位 = $1,000，保證金 = $200
10x 槓桿、分配 100% 權益 → 倉位 = $1,000，保證金 = $100
                                    ↑ 倉位不變    ↑ 保證金變少
```

想要加大倉位？調 `allocation`，不是調 `leverage`。

#### 配置範例

```yaml
# 多幣種倉位分配
portfolio:
  cash_reserve: 0           # 不保留現金
  allocation:
    BTCUSDT: 1.00           # 100% 權益分配給 BTC
    ETHUSDT: 1.00           # 100% 權益分配給 ETH
    SOLUSDT: 1.00           # 100% 權益分配給 SOL
    # 總曝險 300%（三幣各 100%），5x 槓桿下保證金佔 60%
```

**計算公式**：

```
某幣種倉位金額 = 總權益 × allocation × |信號強度|

例如：
  總權益 = $1,000
  BTCUSDT allocation = 1.00
  信號 = -100%（做空）
  → 倉位名義價值 = $1,000 × 1.00 × 100% = $1,000
  → 5x 槓桿下，保證金需求 = $1,000 / 5 = $200
```

#### 常見配置

| 風格 | cash_reserve | allocation | 總曝險 | MDD 預估 |
|------|-------------|------------|--------|----------|
| 保守 | 0.30 | 自動平均（不寫 allocation） | ~70% | ~10% |
| 平衡 | 0.10 | BTC 0.45 + ETH 0.45 | ~90% | ~15% |
| 激進 | 0.00 | BTC 1.00 + ETH 1.00 + SOL 1.00 | 300% | ~38% |

> ⚠️ **總曝險 > 100%** 代表你在用槓桿放大倉位。MDD 也會等比放大。300% 曝險下 5x 槓桿保證金佔 60%，回撤最大到 38%（離 65% 熔斷線還有 27% 緩衝）。

**沒寫 `allocation` 時**：系統自動平均分配。2 個幣 + 30% 現金 = 每個幣 35%。

#### 多幣種防震盪（Anti-Rebalancing）⭐ NEW

同時交易多個幣時，A 幣的 PnL 波動會改變總權益，導致 B 幣的「當前倉位 %」飄移。
系統有**防護機制**，避免因為微小波動觸發不必要的交易和手續費：

```
                  填充率 = |當前倉位%| / |目標倉位%|

目標: SHORT 100%
當前: SHORT -95%（因為其他幣 PnL 波動）
填充率 = 95% ≥ 80% → ✅ 跳過（視為正常波動）

目標: SHORT 100%
當前: SHORT -32%（剛開倉，還沒到位）
填充率 = 32% < 80% → ⚡ 執行加倉
```

> 💡 **80% 門檻**：倉位已達目標的 80% 以上 → 不動。低於 80% → 加倉到目標。這避免了每小時因為 PnL 微調而不斷加減倉。

---

## 增量 K 線快取 ⭐ v2.8

### 問題：回測 vs 實盤的數據差異

| | 回測 | 實盤（舊版） |
|---|------|-------------|
| 數據 | 完整歷史，從第 1 bar 到最後 | 每次只看最近 300 bar |
| 策略行為 | 狀態機從 bar 0 開始走 | 每次從 bar N-300 重新開始 |
| 穩定性 | 確定性結果 | 窗口偏移 1 bar 就可能導致信號翻轉 |

**為什麼會不一致？**

策略（如 RSI + ADX）本質是「狀態機」：每一根 K 線的計算依賴前面所有 bar 的歷史。
回測從 bar 0 跑到最後，結果是確定的。
但實盤每次只拉最近 300 bar，**窗口起點每小時偏移 1 bar**，就像讓狀態機從不同的初始狀態開始走——即使只差 1 bar，後續 RSI/ADX 的值可能走向完全不同的路徑。

```
回測:   [bar 0] [bar 1] [bar 2] ... [bar 8000] → 信號 SHORT
實盤 T: [bar 7700] ... [bar 8000] → 信號 SHORT ✅
實盤 T+1: [bar 7701] ... [bar 8001] → 信號 LONG ❌ ← 起點偏移 1 bar！
```

### 解決方案：增量 K 線快取

**原理**：首次啟動拉取 300 根種子 K 線，後續每次 cron 只拉「新的 K 線」並 append 到快取。策略始終從 bar 0 跑到最新 bar，與回測行為一致。

```
首次 cron：
  拉取 300 bar 種子 → 存入 kline_cache/BTCUSDT.parquet
  策略跑 [bar 0 ~ bar 300] → 信號 SHORT

第 2 次 cron：
  增量拉 +1 bar → append 到 parquet
  策略跑 [bar 0 ~ bar 301] → 信號 SHORT ✅ 穩定

第 1000 次 cron：
  快取已有 1300 bar
  策略跑 [bar 0 ~ bar 1300] → 與回測行為一致 ✅
```

### 11.1 配置方式

在 YAML 中新增 `live` 區段：

```yaml
# 實盤專屬配置
live:
  kline_cache: true           # 啟用增量 K 線快取（推薦）
  flip_confirmation: false    # 方向切換 2-tick 確認（快取模式下不需要）
```

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `kline_cache` | `true` | 啟用增量快取，策略從 bar 0 跑到最新 bar |
| `flip_confirmation` | `false` | 方向翻轉需連續 2 次 cron 確認才執行 |

**建議組合**：

| 情境 | `kline_cache` | `flip_confirmation` | 說明 |
|------|:---:|:---:|------|
| 正式環境（推薦）| `true` | `false` | 快取解決根本問題，不需額外確認 |
| 快取 + 保守 | `true` | `true` | 雙重保險，但會延遲 1 tick 才翻轉 |
| 舊版相容 | `false` | `true` | 滑動窗口 + 確認機制防止頻繁翻轉 |
| 不建議 | `false` | `false` | 容易因數據偏移導致頻繁翻轉 |

### 11.2 快取儲存

快取檔案位於 `reports/{market_type}/{strategy}/live/kline_cache/`：

```
reports/futures/rsi_adx_atr/live/
├── kline_cache/
│   ├── BTCUSDT.parquet    # BTC K 線快取
│   └── ETHUSDT.parquet    # ETH K 線快取
├── last_signals.json      # 信號快照（供 Telegram Bot 讀取）
├── signal_state.json      # 信號方向狀態（供翻轉確認用）
└── algo_orders_cache.json # SL/TP 掛單快取
```

**空間佔用**：
- 1h K 線 × 1 年 ≈ 8,760 bar × ~50 bytes ≈ **430 KB**（可忽略）
- 快取只增不減，不會被截斷

### 11.3 工作流程

```
[首次 cron]
  IncrementalKlineCache.get_klines("BTCUSDT")
  → 拉取 300 根種子 K 線
  → 存入 kline_cache/BTCUSDT.parquet
  → 傳給 generate_signal()

[後續每次 cron]
  IncrementalKlineCache.get_klines("BTCUSDT")
  → 載入 parquet (記憶體快取加速)
  → 只拉最後一根之後的新 K 線 (增量)
  → append + 去重 + 丟棄未收盤
  → 存回 parquet
  → 傳給 generate_signal()

[Telegram /signals 命令]
  → 優先讀取 last_signals.json（與 cron 信號一致）
  → 若無快照，fallback 到即時生成（也使用快取數據）
```

### 11.4 方向切換確認（Flip Confirmation）

當 `flip_confirmation: true` 時，方向翻轉（如 SHORT → LONG）需要連續 2 次 cron 確認：

```
T=0: 信號 SHORT → 正常執行
T=1: 信號 LONG  → ⏸️ 第 1 次翻轉，暫不執行，等下一 tick 確認
T=2: 信號 LONG  → ✅ 連續 2 次 LONG，確認翻轉，執行買入
```

```
T=0: 信號 SHORT → 正常執行
T=1: 信號 LONG  → ⏸️ 第 1 次翻轉
T=2: 信號 SHORT → ❌ 翻回來了，取消翻轉，維持 SHORT
```

> 💡 **什麼時候需要？** 如果你沒有啟用 `kline_cache`（滑動窗口模式），建議開啟 `flip_confirmation` 來避免數據微小差異導致的頻繁多空翻轉。如果已啟用快取，通常不需要。

### 11.5 注意事項

1. **首次 cron 稍慢**：需要拉取 300 根種子 K 線，之後每次只拉增量（通常 1 根）
2. **快取損壞**：如果 parquet 檔損壞，系統會自動重新初始化（等同首次啟動）
3. **刪除快取**：如果你想讓策略「從零開始」，刪除 `kline_cache/` 目錄即可
4. **不影響 cron 設定**：`run_live.py --real --once` 會自動偵測 `live.kline_cache` 設定
5. **Telegram Bot 共享快取**：Bot 的 `/signals` fallback 會使用同一份快取數據

---

## 多數據源與長期歷史數據 ⭐ NEW

專案支援從多個數據源下載歷史數據，讓你可以回測更長的時間範圍。

### 11.1 可用數據源

| 數據源 | 最早數據 | 支援週期 | 說明 |
|--------|----------|----------|------|
| **Binance** | 2017-08 | 1m-1M | 預設，最穩定 |
| **Yahoo Finance** | 2014-09 | 1d 為主 | BTC/ETH 日線數據 |
| **CCXT (Bitfinex)** | 2014-01 | 1h-1d | 早期 1h 數據 |
| **CCXT (Bitstamp)** | 2011-08 | 1d 為主 | BTC 最早數據 |
| **Binance Vision** | 2017-08 | 全部 | 批量下載，最快 |

### 11.2 下載長期歷史數據

```bash
# 查看可用數據源
python scripts/download_data.py --list-sources

# 下載 BTC 日線 (2014 年起，Yahoo Finance)
python scripts/download_data.py --source yfinance --symbol BTCUSDT --start 2014-09-17 --interval 1d --full

# 下載 BTC 1h (2014 年起，Bitfinex via CCXT)
python scripts/download_data.py --source ccxt --exchange bitfinex --symbol BTCUSDT --start 2014-01-01 --interval 1h --full

# 下載後會自動合併到本地 Parquet 文件
```

### 11.3 數據合併邏輯

- 從不同數據源下載的數據會自動合併到同一個 Parquet 文件
- 合併時以時間戳為主鍵，優先保留較新的數據
- 例如：Bitfinex (2014-2017) + Binance (2017-now) → 完整 12+ 年數據

### 11.4 數據存儲

```
data/binance/
├── spot/
│   ├── 1h/
│   │   ├── BTCUSDT.parquet  # 包含所有來源的合併數據
│   │   └── ETHUSDT.parquet
│   └── 1d/
│       └── BTCUSDT.parquet  # 11+ 年日線數據
└── futures/
    └── 1h/
```

**存儲效率**：
- 11 年日線數據：~227 KB
- 12 年 1h 數據：~30 MB

---

## 組合回測 ⭐ NEW

支援多幣種組合回測，評估投資組合的整體表現。

### 12.1 基本用法

```bash
# 等權重組合 (BTC 50% + ETH 50%)
python scripts/run_portfolio_backtest.py -c config/rsi_adx_atr.yaml --symbols BTCUSDT ETHUSDT

# 自訂權重 (BTC 70% + ETH 30%)
python scripts/run_portfolio_backtest.py -c config/rsi_adx_atr.yaml --symbols BTCUSDT ETHUSDT --weights 0.7 0.3

# 三幣種組合
python scripts/run_portfolio_backtest.py -c config/rsi_adx_atr.yaml --symbols BTCUSDT ETHUSDT SOLUSDT --weights 0.5 0.3 0.2
```

### 12.2 組合回測結果範例

```
📊 組合配置:
   BTCUSDT: 50.0%
   ETHUSDT: 50.0%

📅 共同時間範圍: 2017-08-17 → 2026-02-09
  BTCUSDT: 回報 41764.97%, MDD -17.56%
  ETHUSDT: 回報 1065444.70%, MDD -26.80%

======================================================================
  組合回測結果: BTCUSDT + ETHUSDT
======================================================================

指標                                           組合策略        組合 Buy&Hold
----------------------------------------------------------------------
Total Return [%]                        553604.83            1036.55
Annualized Return [%]                      176.53              33.22
Max Drawdown [%]                            22.80              87.47
Sharpe Ratio                                 4.30               0.76
```

### 12.3 組合優化建議

| 配置 | 特點 | 適合 |
|------|------|------|
| BTC 70% + ETH 30% | Sharpe 最高，回撤最低 | 穩健型 |
| BTC 50% + ETH 50% | 平衡收益與風險 | 平衡型 |
| BTC 30% + ETH 70% | 收益最高，風險較大 | 進取型 |

### 12.4 輸出文件

組合回測會生成以下文件：

```
reports/{market_type}/{strategy}/portfolio/{timestamp}/
├── portfolio_equity_curve.png   # 組合資金曲線圖
├── portfolio_equity.csv         # 組合淨值數據
└── portfolio_stats.json         # 組合統計指標
```

例如：`reports/futures/rsi_adx_atr/portfolio/20260213_114917/`

---

## RSI Exit 策略配置 ⭐ NEW

這是經過完整驗證的最佳策略配置，建議新手直接使用。

### 13.1 策略特點

**RSI Exit** 與標準 ATR TP 的區別：

| 特性 | 標準配置 | RSI Exit |
|------|----------|----------|
| 止損 | ATR-based | ATR-based |
| 止盈 | 固定 ATR 倍數 | **RSI overbought** |
| 優點 | 明確目標 | 讓利潤奔跑 |
| 適合 | 震盪市 | **趨勢市** |

### 13.2 優化後的最佳參數

```yaml
# config/rsi_adx_atr_rsi_exit.yaml
strategy:
  name: "rsi_adx_atr"
  params:
    rsi_period: 10       # 更短週期，更快反應
    oversold: 30         # 更早入場
    overbought: 75       # 延後出場，讓利潤跑
    min_adx: 15          # 降低趨勢門檻
    adx_period: 14
    stop_loss_atr: 1.5   # 更緊止損
    take_profit_atr: null  # ⭐ 關鍵：null = RSI 出場
    atr_period: 14
    cooldown_bars: 4
```

### 13.3 驗證結果（2026-02-13）

**Walk-Forward 驗證**：
| 幣種 | Train Sharpe | Test Sharpe | 績效衰退 | 5-Fold |
|------|-------------|-------------|----------|--------|
| BTCUSDT | 3.00 | **3.43** | -14.5% | 5/5 ✅ |
| ETHUSDT | 3.89 | **4.07** | -4.6% | 5/5 ✅ |

**完整驗證摘要**：
```
Walk-Forward: ✅ PASS (平均衰退 -9.5%)
Monte Carlo:  ✅ PASS (VaR 95%: 0.60%)
DSR:          ✅ PASS (校正 SR: 3.75)
PBO:          ✅ PASS (20%)
Kelly:        ✅ PASS (2/2)
```

### 13.4 回測表現（2023-2026）

| 幣種 | 策略收益 | Sharpe | Max DD | vs Buy&Hold |
|------|----------|--------|--------|-------------|
| BTCUSDT | +1,044% | 4.03 | -10.3% | 3.4x |
| ETHUSDT | +1,663% | 3.72 | -10.4% | 23x |
| SOLUSDT | +13,976% | 4.18 | -17.9% | - |

### 13.5 快速開始

```bash
# 1. 下載數據
python scripts/download_data.py -c config/rsi_adx_atr_rsi_exit.yaml

# 2. 執行回測
python scripts/run_backtest.py -c config/rsi_adx_atr_rsi_exit.yaml

# 3. 驗證策略
python scripts/validate.py -c config/rsi_adx_atr_rsi_exit.yaml

# 4. Paper Trading
python scripts/run_live.py -c config/rsi_adx_atr_rsi_exit.yaml --paper --once
```

### 13.6 Kelly 倉位建議

| 幣種 | 勝率 | 盈虧比 | 建議倉位 |
|------|------|--------|----------|
| BTCUSDT | 52.7% | 2.02 | **7.3%** (Quarter Kelly) |
| ETHUSDT | 56.1% | 1.71 | **7.6%** (Quarter Kelly) |

如果你有 4 個幣種 + 20% 現金儲備，每幣約 20% 權重，實際倉位 = 20% × 7.5% ≈ **1.5%** 總資金風險。

---

## 策略升級：Dynamic RSI + Funding Filter ⭐ v3.0

### 15.1 問題：Alpha Decay

研究發現 RSI 的預測能力（IC）在衰減：
- 2023: IC = +0.065
- 2024: IC = +0.041（下降 37%）
- 2025: IC = +0.025（下降 62%）
- 2026: IC = +0.018（下降 72%）

固定閾值（30/70）無法適應市場結構變化。

### 15.2 解決方案：Dynamic RSI（Rolling Percentile）

用**滾動百分位**取代固定閾值，讓策略自動適應市場：

```yaml
strategy:
  params:
    rsi_mode: "dynamic"       # 啟用動態 RSI
    rsi_lookback_days: 14     # 滾動窗口 14 天
    rsi_quantile_low: 0.10    # 10th percentile = 超賣
    rsi_quantile_high: 0.90   # 90th percentile = 超買
```

**原理**：
- 不再用固定 30/70，而是取過去 14 天 RSI 的 10%/90% 分位
- 市場震盪時閾值收窄（更敏感），趨勢時閾值放寬（更耐心）
- BTC 2026 年 IC 從 -0.002（Static）恢復到 +0.015（Dynamic）

### 15.3 Funding Rate 過濾器

Funding Rate 是**真正的獨立因子**（與 RSI/ADX 相關性 < 0.1），用來過濾擁擠交易：

```yaml
strategy:
  params:
    use_funding_filter: true
    fr_max_pos: 0.0002    # 0.02%: funding > 此值時不做多（多頭太擁擠）
    fr_max_neg: -0.0002   # -0.02%: funding < 此值時不做空（空頭太擁擠）
```

**效果**：過濾極端 funding rate 的信號後，BTC 做多勝率提升約 5%。

**配套 cron（每 8 小時更新 Funding Rate 數據）**：
```bash
10 */8 * * * cd ~/quant-binance-spot && source .venv/bin/activate && python scripts/download_data.py -c config/futures_rsi_adx_atr.yaml --funding-rate >> logs/download_data.log 2>&1
```

### 15.4 風控參數優化

透過 `scan_risk_params.py` 網格掃描 SL × Cooldown 的熱力圖，找到最佳平衡點：

| 參數 | 舊值 | 新值 | 原因 |
|------|------|------|------|
| `stop_loss_atr` | 2.5 | **2.0** | 更快止損，保護本金 |
| `cooldown_bars` | 5 | **3** | 縮短冷卻期，不錯過機會 |

```bash
# 自行掃描
python scripts/scan_risk_params.py -c config/futures_rsi_adx_atr.yaml
```

---

## 執行優化：Maker 優先下單 ⭐ v3.0

### 16.1 問題：Taker 手續費高

Binance Futures 手續費：
- Taker（市價單）：**0.04%**
- Maker（限價單）：**0.02%**（甚至可能有返佣）

每筆交易省 0.02%，一年下來交易頻繁的策略可省 **2-4% 年化收益**。

### 16.2 Smart Order 機制

```
策略觸發下單
    ↓
掛限價單 (Maker) @ 最佳買/賣價
    ↓
等待 10 秒
    ↓
┌─ 已成交？ → ✅ 完成（省了 0.02%）
└─ 未成交？ → 撤單 → 改市價單 (Taker) → 確保成交
```

### 16.3 配置

```yaml
live:
  prefer_limit_order: true    # 啟用 Maker 優先
  limit_order_timeout_s: 10   # 等待秒數（超時改市價單）
```

> 💡 不需要改策略代碼，系統在 Broker 層自動處理。Paper Trading 不受影響。

---

## WebSocket 事件驅動 ⭐ v3.0

### 17.1 Polling vs WebSocket

| 特性 | Polling（cron） | WebSocket |
|------|----------------|-----------|
| 延遲 | ~5 分鐘 | **< 1 秒** |
| 觸發方式 | 定時（每小時第 5 分） | K 線收盤瞬間 |
| 記憶體 | ~200MB（每次載入 vectorbt） | **~50MB**（只保留 300 bar） |
| 適合 | 低頻策略、多策略 | 對延遲敏感的策略 |

### 17.2 啟動方式

```bash
# 1. 配置 Swap（Oracle Cloud 1GB RAM 必備，只需跑一次）
bash scripts/setup_swap.sh

# 2. 用 tmux 背景啟動
tmux new -d -s trading "cd ~/quant-binance-spot && source .venv/bin/activate && PYTHONPATH=src python scripts/run_websocket.py -c config/futures_rsi_adx_atr.yaml --real 2>&1 | tee logs/websocket.log"

# 3. 檢查狀態
sleep 5 && cat logs/websocket.log
```

### 17.3 日常管理

```bash
# 查看即時 log
tmux attach -t trading
# (Ctrl+B 然後 D 離開，程式繼續跑)

# 不進 tmux，直接看最近 log
tail -50 logs/websocket.log

# 重啟
tmux kill-session -t trading
tmux new -d -s trading "cd ~/quant-binance-spot && source .venv/bin/activate && PYTHONPATH=src python scripts/run_websocket.py -c config/futures_rsi_adx_atr.yaml --real 2>&1 | tee logs/websocket.log"
```

### 17.4 注意事項

- ⚠️ **WebSocket 和 Cron 不可同時使用**。用 WebSocket 時要把 cron 裡的 `run_live.py` 註解掉
- ✅ Funding Rate 下載的 cron 要保留（WebSocket 不負責下載數據）
- ✅ 健康檢查、日報、Telegram Bot 等 cron 不受影響
- ✅ 使用相同的 Broker、策略、信號生成邏輯，行為與 Polling 模式一致

### 17.5 架構（Oracle Cloud 1GB RAM 可跑）

```
WebSocket Runner
├── 連線 Binance Futures WebSocket
│   └── 訂閱 btcusdt@kline_1h, ethusdt@kline_1h, solusdt@kline_1h
├── Rolling DataFrame（每幣 365 根 K 線，~1MB）
├── K 線收盤事件
│   ├── 更新 DataFrame
│   ├── 產生信號（generate_signal）
│   ├── 執行交易（broker.execute）
│   └── 寫入 SQLite + Telegram 通知
└── 記憶體：~50MB（不載入 vectorbt）
```

---

## SQLite 交易資料庫 ⭐ v3.0

### 18.1 為什麼需要？

之前的交易紀錄散落在 log 文件和 JSON 中，難以分析。現在用 SQLite 結構化儲存：

| 表 | 內容 | 用途 |
|----|------|------|
| `trades` | 每筆交易明細（價格、數量、PnL、手續費、是否 Maker） | 績效分析 |
| `signals` | 每次信號快照（RSI、ADX、目標倉位） | 策略診斷 |
| `daily_equity` | 每日權益快照 | 資金曲線 |

### 18.2 查詢工具

```bash
# 績效總覽
python scripts/query_db.py -c config/futures_rsi_adx_atr.yaml summary

# 最近 20 筆交易
python scripts/query_db.py -c config/futures_rsi_adx_atr.yaml trades --limit 20

# 信號紀錄
python scripts/query_db.py -c config/futures_rsi_adx_atr.yaml signals --limit 10

# 匯出 CSV
python scripts/query_db.py -c config/futures_rsi_adx_atr.yaml export

# 壓縮資料庫（清理空間）
python scripts/query_db.py -c config/futures_rsi_adx_atr.yaml compact
```

### 18.3 範例輸出

```
============================================================
  📊 交易績效總覽
============================================================
  總交易筆數:     47
  勝利:           28
  虧損:           19
  勝率:           59.6%
  總 PnL:         $+312.50
  平均 PnL:       $+6.65
  最佳交易:       $+89.30
  最差交易:       $-45.20
  總手續費:       $18.80
------------------------------------------------------------
  Maker 成交比例: 72.3%
  Maker 省下費用: $6.42
============================================================
```

### 18.4 資料庫位置

```
reports/futures/rsi_adx_atr/live/trading.db
```

資料庫會在**首次實盤交易時自動建立**，不需要手動初始化。

---

## 波動率過濾 + HTF 軟趨勢過濾 ⭐ v3.1

### 19.1 波動率過濾器（Volatility Filter）

**問題**：低波動期間（震盪市）頻繁交易，手續費吃掉利潤。

**解決方案**：當 ATR/Price 低於閾值時，不開新倉：

```yaml
strategy:
  params:
    min_atr_ratio: 0.005         # ATR/Price < 0.5% 時不開倉
    vol_filter_mode: "absolute"  # "absolute"（固定閾值）或 "percentile"（百分位）
```

**模式說明**：

| 模式 | 邏輯 | 適合 |
|------|------|------|
| `absolute` | ATR/Price < `min_atr_ratio` → 不開倉 | 通用，推薦 |
| `percentile` | ATR 排名 < `min_percentile`% → 不開倉 | 自適應 |

**效果**：回測顯示過濾低波動後 Sharpe 明顯提升，因為減少了無效交易和手續費消耗。

### 19.2 HTF 軟趨勢過濾（Soft Trend Filter）

**問題**：舊版 HTF 過濾是二元閘門（順趨勢全過、逆趨勢全擋），導致漏掉逆趨勢的好機會。

**解決方案**：用**連續權重**取代二元閘門：

```yaml
strategy:
  params:
    htf_interval: "4h"            # 高時間框架
    htf_mode: "soft"              # "soft"（推薦）或 "hard"（二元閘門）
    htf_ema_fast: 20              # 快速 EMA
    htf_ema_slow: 50              # 慢速 EMA
    htf_align_weight: 1.0         # 順趨勢：全倉
    htf_counter_weight: 0.5       # 逆趨勢：半倉
    htf_neutral_weight: 0.75      # 無趨勢：75%
```

**權重邏輯**：

| 情境 | 舉例 | 權重 | 效果 |
|------|------|:----:|------|
| 順趨勢 | 做多 + HTF 上升 | 1.0 | 全倉開入 |
| 逆趨勢 | 做多 + HTF 下降 | 0.5 | 半倉開入 |
| 無趨勢 | HTF 持平 | 0.75 | 75% 倉位 |

> 💡 **soft 模式的優點**：不會完全錯過逆趨勢的交易機會，而是降低曝險。已有持倉不受影響，只調整新開倉的倉位大小。

**HTF 層級建議**：

| 當前週期 | HTF 建議 |
|----------|---------|
| 15m | 1h |
| 1h | 4h |
| 4h | 1d |

### 19.3 狀態機修復

v3.1 修復了策略狀態機的一個重要問題：**平倉後直接反手**。

**舊行為**（有問題）：
```
持有多倉 → 空頭出場信號觸發 → 立即開空倉
```

**新行為**（修復後）：
```
持有多倉 → 出場信號觸發 → 回到 Flat（空倉）→ cooldown → 新的入場信號 → 開倉
```

這個修復確保每次出場後都經過 cooldown 冷卻期，避免頻繁反手造成的滑點和手續費損失。

---

## 波動率目標倉位管理 ⭐ v3.1

### 20.1 問題：固定倉位的缺陷

固定倉位（`method: "fixed"`）不考慮市場波動率：
- 高波動期：倉位不變，風險過大
- 低波動期：倉位不變，資金效率低

### 20.2 解決方案：波動率目標（Volatility Targeting）

根據**目標年化波動率**自動調整倉位大小：

```yaml
position_sizing:
  method: "volatility"       # 啟用波動率目標
  target_volatility: 1.00    # 目標年化波動率 100%（全倉位）
  vol_lookback: 168          # 計算波動率的回看期（168 bars = 7 天 @ 1h）
```

**原理**：
```
調整因子 = target_volatility / current_volatility

如果當前年化波動率 = 200%，目標 = 100%
→ 調整因子 = 100% / 200% = 0.5
→ 原始信號 1.0 → 調整後 0.5（半倉）

如果當前年化波動率 = 50%，目標 = 100%
→ 調整因子 = 100% / 50% = 2.0（上限 cap）
→ 原始信號 1.0 → 調整後 1.0（全倉，不超過槓桿上限）
```

**調整因子上下限**：`[0.1, 2.0]`，避免極端倉位。

### 20.3 三種倉位管理比較

| 方法 | 特點 | 適合 |
|------|------|------|
| `fixed` | 固定比例，簡單 | 新手、小資金 |
| `kelly` | 基於勝率/盈虧比，數學最優 | 穩定策略、充足歷史 |
| `volatility` | 基於市場波動率，動態調整 | ⭐ **推薦**，平衡風險 |

### 20.4 不同時間框架的 vol_lookback

| 時間框架 | vol_lookback | 等效天數 |
|----------|:-----------:|:--------:|
| 15m | 672 | 7 天 |
| 1h | 168 | 7 天 |
| 4h | 42 | 7 天 |
| 1d | 20 | 20 天 |

---

## Alpha Decay 監控 ⭐ v3.1

### 21.1 什麼是 Alpha Decay？

策略的預測能力（Alpha）會隨時間衰減。我們用 **Information Coefficient（IC）** 來量化：

| 年份 | RSI IC | 衰退 |
|------|:------:|:----:|
| 2023 | +0.065 | — |
| 2024 | +0.041 | -37% |
| 2025 | +0.025 | -62% |
| 2026 | +0.018 | -72% |

當 IC 接近 0，策略信號不再有預測價值。

### 21.2 IC 監控工具

```bash
# 執行 Alpha Decay 監控
python scripts/monitor_alpha_decay.py -c config/futures_rsi_adx_atr.yaml

# 靜默模式 + Telegram 通知（用於 cron）
python scripts/monitor_alpha_decay.py -c config/futures_rsi_adx_atr.yaml --quiet --notify

# 指定前瞻期和窗口
python scripts/monitor_alpha_decay.py -c config/futures_rsi_adx_atr.yaml --forward-bars 24 --window-days 180
```

**輸出範例**：
```
════════════════════════════════════════════════════════════
  BTCUSDT  IC Analysis
════════════════════════════════════════════════════════════
  📊 有效信號: 12,345 筆 (68.5% 活躍)

  ── 全局 IC ──
  Overall IC:     +0.0312  (p=0.0001)
  Average IC:     +0.0285
  IC IR:          0.450

  ── Alpha Decay 偵測 ──
  Historical IC:  +0.0450
  Recent IC:      +0.0180
  IC 衰退:        -60%  🔴

  ── 年度 IC ──
  2023: +0.0650  🟢 ██████████████
  2024: +0.0410  🟢 ████████
  2025: +0.0250  🟡 █████
  2026: +0.0180  🟡 ███

  ✅ 無警報，信號品質正常
```

### 21.3 自動排程

建議每週跑一次 IC 監控：

```bash
# crontab -e
# 每週日 01:00 UTC 執行 Alpha Decay 監控
0 1 * * 0 cd ~/quant-binance-spot && source .venv/bin/activate && bash scripts/cron_alpha_monitor.sh >> logs/alpha_monitor.log 2>&1
```

### 21.4 警報級別

| 級別 | 觸發條件 | 建議行動 |
|------|---------|---------|
| 🚨 CRITICAL | IC 衰退 > 50% 或 Recent IC ≈ 0 | 考慮停止策略或切換時間框架 |
| ⚠️ WARNING | IC IR < 0.3 或 p-value > 0.05 | 監控，準備應急方案 |
| ℹ️ INFO | IC 下降但仍顯著 | 持續觀察 |

### 21.5 JSON 報告

每次執行會保存 JSON 報告到 `reports/alpha_decay/`：

```bash
ls reports/alpha_decay/
# ic_report_20260216_120000.json
```

可用於長期追蹤 IC 趨勢。

---

## 策略組合 Ensemble ⭐ v3.1

### 22.1 為什麼要組合策略？

單一策略容易受 Alpha Decay 影響。組合**低相關**策略可以：
- 🟢 平滑資金曲線（降低波動）
- 🟢 降低 Max Drawdown
- 🟢 提升 Sharpe Ratio（分散化效應）

### 22.2 策略相關性分析

先用研究腳本找出低相關配對：

```bash
python scripts/research_strategy_correlation.py -c config/futures_rsi_adx_atr.yaml --symbol BTCUSDT
```

**輸出**：
```
信號方向相關性矩陣 (Pearson)
                    rsi_adx_atr  macd_momentum  bb_mean_reversion
rsi_adx_atr              1.000          0.150             0.420
macd_momentum            0.150          1.000             0.380
bb_mean_reversion        0.420          0.380             1.000

低相關配對 (|corr| < 0.3):
  rsi_adx_atr × macd_momentum: corr=0.150 ✅
```

### 22.3 使用 Ensemble 策略

```bash
# 回測 Ensemble 策略
python scripts/run_backtest.py -c config/futures_ensemble.yaml

# Walk-Forward 驗證
python scripts/run_walk_forward.py -c config/futures_ensemble.yaml --splits 6
```

**配置檔**（`config/futures_ensemble.yaml`）：

```yaml
strategy:
  name: "ensemble_rsi_macd"
  params:
    strategy1_name: "rsi_adx_atr"
    strategy2_name: "macd_momentum"
    weight1: 0.5              # RSI 策略權重 50%
    weight2: 0.5              # MACD 策略權重 50%
    # ... 共用過濾器參數（與主策略相同）
```

### 22.4 組合邏輯

```
RSI 信號:  +1.0 (做多)
MACD 信號: +0.5 (弱多)
→ 組合信號 = 0.5 × 1.0 + 0.5 × 0.5 = +0.75

RSI 信號:  +1.0 (做多)
MACD 信號: -1.0 (做空)
→ 組合信號 = 0.5 × 1.0 + 0.5 × (-1.0) = 0.0（衝突 → 不開倉）
```

> 💡 **自然避險**：當兩個策略信號衝突時，組合信號趨近 0（不交易），自動避開不確定的行情。

### 22.5 可用的多時間框架配置

| 配置檔 | 時間框架 | HTF | 用途 |
|--------|:--------:|:---:|------|
| `futures_rsi_adx_atr.yaml` | 1h | 4h | ⭐ 主力 |
| `futures_rsi_adx_atr_15m.yaml` | 15m | 1h | 高頻研究 |
| `futures_rsi_adx_atr_4h.yaml` | 4h | 1d | 低頻研究 |
| `futures_ensemble.yaml` | 1h | 4h | 組合策略 |

---

## 完整範例：RSI 策略

讓我們完整地開發一個 RSI 策略。

### 步驟 1：建立策略檔案

建立 `src/qtrade/strategy/my_rsi_strategy.py`：

```python
"""
RSI 超買超賣策略

策略邏輯：
- RSI < 30：買入
- RSI > 70：賣出
"""
from __future__ import annotations
import pandas as pd
from .base import StrategyContext
from . import register_strategy
from ..indicators import calculate_rsi


@register_strategy("my_rsi_strategy")
def generate_positions(df: pd.DataFrame, ctx: StrategyContext, params: dict) -> pd.Series:
    period = params.get("period", 14)
    oversold = params.get("oversold", 30)
    overbought = params.get("overbought", 70)
    
    close = df["close"]
    rsi = calculate_rsi(close, period=period)
    
    pos = pd.Series(0.0, index=df.index)
    pos[rsi < oversold] = 1.0  # 超賣買入
    pos[rsi > overbought] = 0.0  # 超買賣出
    
    # 避免未來資訊洩漏
    pos = pos.shift(1).fillna(0.0)
    
    return pos
```

### 步驟 2：註冊策略

在 `src/qtrade/strategy/__init__.py` 中新增：

```python
from . import my_rsi_strategy  # noqa: E402
```

### 步驟 3：設定策略（⭐ 重要！）

**這是關鍵步驟**：回測腳本會讀取設定檔來確定使用哪個策略。

編輯 `config/base.yaml` 檔案：

```yaml
strategy:
  name: "my_rsi_strategy"  # ⚠️ 必須與 @register_strategy("my_rsi_strategy") 中的名稱完全一致
  params:
    period: 14
    oversold: 30
    overbought: 70
```

**運作原理**：
1. 當你執行 `python scripts/run_backtest.py` 時
2. 腳本會讀取 `config/base.yaml` 檔案
3. 根據 `strategy.name` 找到對應的策略函式（透過 `@register_strategy` 註冊的）
4. 使用 `strategy.params` 中的參數執行策略
5. 對設定中的所有交易對進行回測

**如何切換策略？**
只需要修改設定檔，無需修改程式碼：
```yaml
strategy:
  name: "ema_cross"  # 切換到 EMA 交叉策略
  params:
    fast: 20
    slow: 60
```

**如果策略名稱錯誤會怎樣？**
會報錯並顯示所有可用的策略：
```
ValueError: Strategy 'wrong_name' not found. 
Available: ['ema_cross', 'rsi', 'my_rsi_strategy', ...]
```

### 步驟 4：下載資料

```bash
python scripts/download_data.py --symbol BTCUSDT --interval 1h --start 2022-01-01
```

### 步驟 5：執行回測

```bash
python scripts/run_backtest.py
```

### 步驟 6：查看結果

查看 `reports/spot/{strategy}/backtest/{timestamp}/` 目錄下的 `stats_BTCUSDT.csv` 和 `equity_curve_BTCUSDT.png`

### 步驟 7：優化參數

```bash
python scripts/optimize_params.py --strategy my_rsi_strategy --metric "Sharpe Ratio"
```

### 步驟 8：驗證策略

```bash
# 標準驗證套件
python scripts/validate.py -c config/rsi_adx_atr.yaml

# 只執行 Kelly 驗證
python scripts/validate.py -c config/rsi_adx_atr.yaml --only kelly
```

### 步驟 9：Paper Trading

```bash
# 啟動模擬交易
python scripts/run_live.py -c config/base.yaml --paper --once

# 查看狀態
python scripts/run_live.py -c config/base.yaml --status
```

### 步驟 10：如果結果好，可以實盤

如果驗證通過且 Paper Trading 表現良好，可以考慮實盤交易。

---

## 常見問題

### Q1: 策略總是虧損怎麼辦？

**可能原因**：
1. 策略邏輯有問題
2. 參數不合適
3. 市場環境不適合該策略

**解決方法**：
1. 檢查訊號是否合理（查看資金曲線圖）
2. 嘗試不同參數
3. 嘗試不同交易對
4. 簡化策略邏輯

### Q2: 回測結果很好，但驗證失敗？

**這是過擬合的典型表現**。

**解決方法**：
1. 簡化策略
2. 減少參數
3. 使用更保守的參數
4. 增加更多歷史資料

### Q3: 如何選擇優化指標？

**推薦順序**：
1. **夏普比率**：綜合考慮收益和風險
2. **總收益率**：如果風險可控
3. **最大回撤**：如果風險承受能力低

### Q4: 策略在某個交易對上表現好，另一個不好？

**這是正常的**，不同資產有不同的特性。

**解決方法**：
1. 為不同資產使用不同參數
2. 使用策略組合
3. 只交易適合該策略的資產

### Q5: Live 和 Backtest 訊號不一致？ ⭐ NEW

**可能原因**：
1. Look-ahead bias（使用了未來資訊）
2. 資料對齊問題
3. 實作錯誤
4. **滑動窗口數據差異**（v2.8 前最常見的原因）→ 見 [Q10](#q10-實盤信號跟回測不一致頻繁翻轉-v28)

**解決方法**：
```bash
# 執行一致性驗證找出問題
python scripts/validate.py -c config/rsi_adx_atr.yaml --only consistency

# v2.8: 確認增量快取已啟用
grep "kline_cache" config/futures_rsi_adx_atr.yaml
```

### Q6: Kelly 建議不使用怎麼辦？ ⭐ NEW

**可能原因**：
1. 勝率太低（< 40%）
2. 盈虧比太差（< 1.0）
3. 交易次數太少（< 30 筆）
4. 統計不穩定

**解決方法**：
1. 暫時使用固定倉位
2. 累積更多交易資料後再評估
3. 優化策略提高勝率和盈虧比

### Q7: 槓桿加大但倉位沒變大？ ⭐ NEW

**這是正確行為！** 槓桿只影響保證金需求，不影響倉位名義價值。

```
5x 槓桿、倉位 $1,000 → 保證金 = $200
10x 槓桿、倉位 $1,000 → 保證金 = $100  ← 保證金變少，倉位不變
```

如果你想加大倉位，調整 `portfolio.allocation`：

```yaml
portfolio:
  allocation:
    BTCUSDT: 0.50  # 50% 權益 → 改成 1.00 → 100% 權益，倉位翻倍
```

詳見 [§10.12 多幣種倉位分配](#1012-多幣種倉位分配--new)。

### Q8: 為什麼出現重複的 SL/TP 掛單？ ⭐ NEW

**舊版本已知問題，已修復。** 原因是 bot 重啟後遺失了掛單記錄，每次 cron 都重新掛一組。

**修復後的機制**：
1. SL/TP 記錄持久化到 `algo_orders_cache.json`（重啟不丟失）
2. 掛單前檢查是否已存在相近觸發價的掛單（0.2% 容差）
3. 平倉時自動取消所有 SL + TP 並清理快取

如果你在 Binance 看到多餘的舊掛單，手動取消即可，下次 cron 會根據當前持倉自動補掛正確的。

### Q9: 信號顯示 100% 但實際倉位只有 32%？ ⭐ NEW

**可能原因**：
1. **剛開倉**，下一輪 cron 會加倉到目標（填充率 < 80% 才會加倉）
2. **`allocation` 設定太低**，倉位不會超過分配比例

**解決方法**：
1. 等下一輪 cron 執行（每小時第 5 分鐘）
2. 檢查 `config.yaml` 的 `portfolio.allocation` 設定：

```yaml
# 如果你想讓信號 100% 對應全部權益
portfolio:
  cash_reserve: 0
  allocation:
    BTCUSDT: 1.00  # 100% 權益
    ETHUSDT: 1.00
    SOLUSDT: 1.00
```

### Q10: 實盤信號跟回測不一致（頻繁翻轉）？ ⭐ v2.8

**原因**：策略是「狀態機」，計算結果依賴起始 bar。回測用完整歷史，實盤舊版每次只拉最近 300 bar（滑動窗口），窗口偏移 1 bar 就可能讓信號翻轉。

**解決方法**：啟用增量 K 線快取（v2.8 預設開啟）：

```yaml
live:
  kline_cache: true         # 快取數據，策略從 bar 0 跑到最新 bar
  flip_confirmation: false  # 數據穩定後不需要確認
```

詳見 [§11 增量 K 線快取](#增量-k-線快取)。

### Q11: `/signals` 和 cron 信號不一致？ ⭐ v2.8

**原因**：舊版 `/signals` 命令會獨立拉取 300 bar 數據即時計算，與 cron 拉取的數據可能有微小差異（偏移 1 bar），導致信號不同。

**修復後的機制**：
1. cron 每次執行後，會將信號快照保存到 `last_signals.json`
2. `/signals` 命令優先讀取這個快照，**保證與 cron 信號完全一致**
3. 快照顯示格式為「⏱ X 分鐘前」，表示是快取的
4. 若快照不存在或超過 2 小時，才 fallback 到即時生成（也會使用增量快取數據）

### Q12: ETH 的 SL 掛不上但 BTC 可以？ ⭐ NEW

**原因**：Binance 對不同交易對的條件單 API 支援不同。部分帳戶的 ETHUSDT 標準 API 會回傳 `-4120` 錯誤。

**系統已自動處理**：偵測到 `-4120` 後自動降級到 Algo Order API。如果你在 log 中看到：

```
⚠️ ETHUSDT: 標準 Order (STOP_MARKET) 失敗... -4120
✅ ETHUSDT: 條件單已掛 via Algo Order API
```

這是正常的，SL/TP 已成功掛上。只要最後看到 `✅ 條件單已掛` 就沒問題。

**另外**：如果看到 `algoOrder/openOrders` 的 `404` 警告，這也是無害的——表示你的帳戶不支援 Algo Order 查詢 API，系統會自動回退到標準掛單 + 本地快取。

### Q13: `git pull` 後 kline_cache 沒有生效？ ⭐ NEW

**症狀**：更新代碼到 v2.8 後，log 裡看不到 `📦 增量 K 線快取已啟用`，且 `kline_cache/` 目錄不存在。

**原因**：Python 的 `.pyc` 編譯快取。`git pull` 只更新 `.py` 原始碼，但 Python 會優先讀取 `__pycache__/` 裡的舊版 `.pyc`，導致新功能（如 kline_cache 初始化）**完全不被執行**。

**解決方法**：

```bash
# 方法一：用部署腳本（推薦）
./scripts/setup_cron.sh --update

# 方法二：手動清除
find . -name "*.pyc" -delete
find . -name "__pycache__" -exec rm -rf {} +
```

**驗證**：

```bash
# 清除後手動觸發一次，確認快取生效
python scripts/run_live.py -c config/futures_rsi_adx_atr.yaml --real --dry-run --once 2>&1 | grep "增量"
# 應看到：📦 增量 K 線快取已啟用

# 下一次 cron 執行後檢查快取檔
ls -la reports/futures/rsi_adx_atr/live/kline_cache/
# 應看到 BTCUSDT.parquet、ETHUSDT.parquet 和 SOLUSDT.parquet
```

> ⚠️ **黃金法則**：每次 `git pull` 後都執行 `./scripts/setup_cron.sh --update`，養成習慣就不會踩坑。

### Q14: 出現不明交易（bot log 裡沒有記錄）？ ⭐ NEW

**症狀**：在 Binance 交易記錄中看到某個時間的交易，但 bot 的所有 log 文件都沒有對應記錄。

**排查步驟**：

```bash
# 1. 確認所有 log 裡都沒有記錄
grep -r "09:44" /home/ubuntu/quant-binance-spot/logs/ 2>/dev/null

# 2. 確認沒有常駐進程在跑
ps aux | grep run_live | grep -v grep

# 3. 用 API 查交易的 orderId 和 origType
python -c "
from qtrade.live.binance_futures_broker import BinanceFuturesBroker
from datetime import datetime
b = BinanceFuturesBroker(dry_run=True)
for s in ['BTCUSDT', 'ETHUSDT']:
    trades = b.http.signed_get('/fapi/v1/userTrades', {'symbol': s, 'limit': 20})
    for t in trades:
        ts = datetime.fromtimestamp(int(t['time'])/1000)
        oid = t['orderId']
        order = b.http.signed_get('/fapi/v1/order', {'symbol': s, 'orderId': oid})
        print(f'{s} {ts} {t[\"side\"]} origType={order.get(\"origType\")} clientOrderId={order.get(\"clientOrderId\")}')
"
```

**判讀結果**：

| `origType` | `clientOrderId` 特徵 | 來源 |
|------------|---------------------|------|
| `STOP_MARKET` | 系統生成 | ✅ SL 掛單自動觸發 |
| `TAKE_PROFIT_MARKET` | 系統生成 | ✅ TP 掛單自動觸發 |
| `MARKET` | 隨機字串 | ⚠️ 手動操作（Binance App/Web） |
| `MARKET` | 有特定前綴 | ⚠️ 其他 API/Bot |

如果確認不是自己操作的，建議到 Binance App → API Management 檢查是否有其他 API Key。

---

## 下一步

1. **專案地圖 & 指令速查**：查看 `CLI_REFERENCE.md`
2. **策略參數調整**：編輯 `config/futures_rsi_adx_atr.yaml`
3. **回測驗證**：使用 `scripts/run_backtest.py` 和 `scripts/validate.py`

---

## 總結

完整的策略開發流程：

1. ✅ **策略發想**：觀察市場，形成假設
2. ✅ **建立策略**：編寫程式碼，實現邏輯
3. ✅ **回測**：用歷史資料測試
4. ✅ **優化**：找到最佳參數
5. ✅ **驗證**：確保不過擬合
6. ✅ **Kelly 驗證**：評估倉位管理方式
7. ✅ **風險管理**：新增風險控制
8. ✅ **Paper Trading**：模擬交易觀察
9. ✅ **一致性驗證**：確保 Live/Backtest 一致
10. ✅ **Pre-Deploy 檢查**：`validate_live_consistency.py` 全過再上線
11. ✅ **實盤**：如果一切順利（`live.kline_cache: true` 確保數據一致）
12. ✅ **監控**：健康檢查和日報

記住：**量化交易不是一夜暴富，而是持續改進的過程**。

祝你交易順利！🎉
