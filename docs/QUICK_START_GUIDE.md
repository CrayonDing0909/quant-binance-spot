# 新手完整教學：從策略發想到實盤

本教學將帶你從零開始，完整地開發一個交易策略，包括策略發想、實現、回測、優化和驗證。

## 📋 目錄

1. [專案功能概覽](#專案功能概覽)
2. [第一步：策略發想](#第一步策略發想)
3. [第二步：建立策略](#第二步建立策略)
4. [第三步：回測策略](#第三步回測策略)
5. [第四步：優化參數](#第四步優化參數)
6. [第五步：驗證策略](#第五步驗證策略)
7. [第六步：風險管理](#第六步風險管理)
8. [第七步：即時交易](#第七步即時交易)
9. [第八步：監控與維運](#第八步監控與維運)
10. [完整範例：RSI 策略](#完整範例rsi策略)

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

這個專案提供了完整的量化交易策略開發工具：

### 🎯 核心功能

1. **策略開發**
   - 快速建立策略範本
   - 統一的指標庫（RSI、MACD、布林帶等）
   - 策略註冊系統

2. **資料管理**
   - 自動下載幣安資料
   - 資料品質檢查
   - 資料清洗

3. **回測系統**
   - 向量化回測（快速）
   - 支援手續費和滑點
   - 自動產生報告和圖表

4. **策略優化**
   - 參數網格搜尋
   - 多指標優化

5. **策略驗證**
   - 過擬合檢測
   - 滾動視窗驗證
   - 參數敏感性分析
   - **Kelly 公式驗證** ⭐ NEW
   - **Live/Backtest 一致性驗證** ⭐ NEW

6. **風險管理**
   - 倉位管理
   - 風險限制
   - 組合風險管理
   - **Kelly 倉位計算** ⭐ NEW

7. **即時交易** ⭐ NEW
   - Paper Trading（模擬交易）
   - Real Trading（真實交易）
   - Telegram 通知

8. **監控與維運** ⭐ NEW
   - 系統健康檢查
   - 每日績效報表
   - 心跳監控

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
4. 產生報告和圖表到 `reports/{strategy_name}/` 目錄

**輸出目錄組織**：
- 預設情況下，輸出會按策略名稱自動分類
- 例如：`reports/my_rsi_strategy/stats_BTCUSDT.csv`
- 這樣可以避免不同策略的結果互相覆蓋

### 3.4 查看結果

回測完成後，會在 `reports/{strategy_name}/` 目錄產生：

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

### 4.3 優化指標選擇

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

優化完成後，會產生 `reports/optimization_BTCUSDT.csv`，包含：
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

所有驗證功能已整合到 `validate.py`：

```bash
# 執行標準驗證套件（Walk-Forward + Monte Carlo + Cross-Asset + Kelly）
python scripts/validate.py -c config/rsi_adx_atr.yaml

# 快速驗證（跳過耗時測試）
python scripts/validate.py -c config/rsi_adx_atr.yaml --quick

# 完整驗證（包括一致性檢查）
python scripts/validate.py -c config/rsi_adx_atr.yaml --full

# 只執行特定驗證
python scripts/validate.py -c config/rsi_adx_atr.yaml --only walk_forward
python scripts/validate.py -c config/rsi_adx_atr.yaml --only walk_forward,monte_carlo
```

**可用的驗證類型**：
- `walk_forward` - 滾動視窗驗證
- `monte_carlo` - Monte Carlo 風險分析
- `loao` - Leave-One-Asset-Out 穩健性測試
- `regime` - 市場狀態驗證
- `dsr` - Deflated Sharpe Ratio（校正 Sharpe）
- `pbo` - Probability of Backtest Overfitting
- `kelly` - Kelly 公式驗證
- `consistency` - Live/Backtest 一致性驗證

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

### 5.5 理解驗證結果

驗證完成後會生成摘要報告（`reports/{strategy}/validation_{timestamp}/validation_summary.yaml`）：

```
======================================================================
  📋 Validation Summary
======================================================================
  Walk-Forward: ✅ PASS (平均衰退 15.2%)
  Monte Carlo: ✅ PASS (平均 VaR 95%: 0.58%)
  Cross-Asset: ✅ PASS (moderate)
  DSR: ✅ PASS (校正 SR: 1.2500)
  PBO: ✅ PASS (12.5%)
  Kelly: ✅ PASS (適合: 4/4)
----------------------------------------------------------------------
  Overall: ✅ 策略驗證通過
======================================================================
```

**好的結果**：
- Walk-Forward 衰退 < 30%
- PBO < 50%
- DSR p-value < 0.05
- Kelly 適合使用
- 一致性驗證 > 95%

**不好的結果**：
- Walk-Forward 衰退 > 50%
- PBO > 50%（可能過擬合）
- DSR 不顯著
- Kelly 不適合（統計不穩定）
- 一致性驗證 < 90%（可能有 look-ahead bias）

**PBO (Probability of Backtest Overfitting) 解釋**：

PBO 衡量「訓練表現最好的策略在測試時表現差」的機率：
- 收集所有 Walk-Forward 的 Train/Test Sharpe
- 找出 Train Sharpe 最高的那個 Split
- 看它的 Test Sharpe 在所有 Test 中排第幾

```
例如：如果 Train 最佳的 Split，
      其 Test Sharpe 排在 18/20（倒數第二）
      → PBO = 18/20 = 90%
      → 訓練表現好 ≠ 測試表現好
      → 可能過擬合
```

**PBO 高但策略仍可用的情況**：
- 所有 Test Sharpe 都 > 0（仍然盈利）
- Walk-Forward 衰退 < 30%
- Paper Trading 表現與回測一致

如果驗證結果不好，需要：
1. 簡化策略
2. 減少參數數量
3. 使用更保守的參數
4. 增加樣本外測試資料

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

詳細文件：[風險管理指南](RISK_MANAGEMENT.md)

---

## 第七步：即時交易 ⭐ NEW

驗證通過後，可以開始即時交易。

### 7.1 Paper Trading（模擬交易）

**推薦先用 Paper Trading 觀察策略表現至少 1-2 週**。

```bash
# 啟動 Paper Trading
python scripts/run_live.py -c config/rsi_adx_atr.yaml --paper

# 只交易 BTCUSDT
python scripts/run_live.py -c config/rsi_adx_atr.yaml --paper --symbol BTCUSDT

# 立即執行一次（不等待 K 線收盤）
python scripts/run_live.py -c config/rsi_adx_atr.yaml --paper --once

# 查看帳戶狀態
python scripts/run_live.py -c config/rsi_adx_atr.yaml --status
```

**Paper Trading 特點**：
- 不需要 API Key
- 模擬帳戶，不會虧真錢
- 狀態儲存在 `reports/live/{strategy_name}/paper_state.json`

### 7.2 Telegram 通知設定

在 `.env` 中設定以下變數即可自動啟用通知：

```bash
TELEGRAM_BOT_TOKEN=123456:ABC-DEF
TELEGRAM_CHAT_ID=987654321
```

**如何取得**：
1. 在 Telegram 搜尋 `@BotFather`，建立新 Bot 取得 Token
2. 搜尋 `@userinfobot`，取得你的 Chat ID

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

# 每小時整點執行（配合 1h K 線）
0 * * * * cd /path/to/quant-binance-spot && python scripts/run_live.py -c config/rsi_adx_atr.yaml --paper --once >> logs/live.log 2>&1

# 每 4 小時執行（配合 4h K 線）
0 */4 * * * cd /path/to/quant-binance-spot && python scripts/run_live.py -c config/rsi_adx_atr.yaml --paper --once >> logs/live.log 2>&1
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

# 輸出 JSON 格式
python scripts/health_check.py --json
```

**檢查項目**：
- 📁 磁碟空間使用率
- 💾 記憶體使用率
- 📊 交易狀態檔案是否過期
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

### 8.3 完整 Cron 設定範例

```bash
# 編輯 crontab
crontab -e

# === 交易執行 ===
# 每小時執行 Paper Trading（配合 1h K 線）
0 * * * * cd /opt/qtrade && python scripts/run_live.py -c config/rsi_adx_atr.yaml --paper --once >> logs/live.log 2>&1

# === 監控 ===
# 每 30 分鐘健康檢查
*/30 * * * * cd /opt/qtrade && python scripts/health_check.py --notify >> logs/health.log 2>&1

# === 報表 ===
# 每天 00:05 UTC 發送日報
5 0 * * * cd /opt/qtrade && python scripts/daily_report.py -c config/rsi_adx_atr.yaml >> logs/daily_report.log 2>&1

# === 驗證 ===
# 每週日 00:00 執行一致性驗證
0 0 * * 0 cd /opt/qtrade && python scripts/validate.py -c config/rsi_adx_atr.yaml --only consistency >> logs/consistency.log 2>&1
```

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

查看 `reports/stats_BTCUSDT.csv` 和 `reports/equity_curve_BTCUSDT.png`

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

**解決方法**：
```bash
# 執行一致性驗證找出問題
python scripts/validate.py -c config/rsi_adx_atr.yaml --only consistency
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

---

## 下一步

1. **學習更多策略**：查看 `STRATEGY_DEVELOPMENT.md`
2. **了解風險管理**：查看 `RISK_MANAGEMENT.md`
3. **學習策略組合**：查看 `STRATEGY_PORTFOLIO.md`
4. **資料品質**：查看 `DATA_QUALITY.md`

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
10. ✅ **實盤**：如果一切順利
11. ✅ **監控**：健康檢查和日報

記住：**量化交易不是一夜暴富，而是持續改進的過程**。

祝你交易順利！🎉
