# 策略專業化升級計畫

> 目標：將 RSI+ADX+ATR 策略從「業餘級別」升級到「半專業級別」
> 
> 基於診斷發現的 5 個核心問題，分 3 個 Prompt 逐步實施。
> 每個 Prompt 獨立可測試，完成後可 review 再進入下一步。

---

## ⚠️ 重要研究發現（2026-02-15 更新）

> 以下實證分析在 Prompt 1 規劃完成後進行，發現原始多因子設計存在**根本性缺陷**。
> 基於這些發現，Prompt 1 的方向需要調整。

### 研究 A：因子相關性分析

**方法**：對 BTC (56,442 bars) / ETH (54,532 bars) 的 1h K 線數據，計算 6 個候選因子的 Spearman 相關矩陣。

**結果**：

| 因子對 | BTC r | ETH r | 判定 |
|-------|-------|-------|------|
| RSI vs BB | **+0.854** | **+0.858** | ⛔ 幾乎相同的因子 |
| RSI vs MACD | **-0.655** | **-0.666** | ⛔ 冗餘（反向） |
| RSI vs OBV | **-0.547** | **-0.564** | ⛔ 冗餘 |
| RSI vs MOM20 | **-0.533** | **-0.533** | ⛔ 冗餘 |
| MACD vs BB | **-0.651** | **-0.665** | ⛔ 冗餘 |
| BB vs OBV | **-0.630** | **-0.648** | ⛔ 冗餘 |
| BB vs MOM20 | **-0.689** | **-0.692** | ⛔ 冗餘 |
| OBV vs MOM20 | **+0.503** | **+0.520** | ⛔ 冗餘 |
| EMA vs RSI | -0.257 | -0.258 | ✅ 獨立 |
| EMA vs MACD | -0.088 | -0.069 | ✅ 獨立 |

**因子（因子定義）**：
- RSI = `(50 - RSI(10)) / 50` — 均值回歸方向
- EMA = `sign(EMA(20) - EMA(50))` — 趨勢方向
- MACD = `sign(MACD_Histogram)` — 動量方向
- BB = `1 - 2 * BollingerBand_%B` — 均值回歸方向
- OBV = `sign(OBV - EMA(OBV, 20))` — 成交量方向
- MOM20 = `sign(close.pct_change(20))` — 20 bar 動量

**結論**：RSI、BB、MACD、OBV、MOM20 全部高度相關（|r| > 0.5），**本質上是同一個因子的不同包裝**。EMA 是唯一真正獨立的因子。

### 研究 B：PCA 主成分分析

**方法**：對 5 個因子做 PCA，看有效獨立維度數量。

| 主成分 | 方差解釋 | 累積 | 含義 |
|-------|---------|------|------|
| **PC1** | **50.3%** | 50.3% | 「短期超買超賣」維度（RSI/BB/MACD/OBV 共同載入） |
| **PC2** | **31.5%** | 81.7% | 「長期趨勢」維度（EMA 載入） |
| PC3 | 14.0% | 95.7% | 殘差 |
| PC4 | 3.5% | 99.2% | 噪音 |
| PC5 | 0.8% | 100% | 噪音 |

**結論**：5 個因子只有 **~2 個有效獨立維度**。超過 50% 的信息集中在 PC1（均值回歸）。

### 研究 C：IC (Information Coefficient) 分析

**方法**：計算每個因子值與下一期收益的 Spearman Rank Correlation。
- `|IC| > 0.02` → 有預測力
- `|IC| > 0.05` → 強預測力

**結果（1h forward return）**：

| 因子 | BTC IC | ETH IC | 判定 |
|------|--------|--------|------|
| BB (均值回歸) | **+0.048** | **+0.042** | ✅ 最強（但和 RSI 冗餘） |
| RSI (均值回歸) | **+0.042** | **+0.035** | ✅ 有效（和 BB 同源） |
| OBV (成交量) | -0.029 | -0.030 | 弱 + 冗餘 |
| MOM20 (動量) | -0.027 | -0.023 | 弱 + 冗餘 |
| MACD (動量) | -0.025 | -0.022 | 弱 + 冗餘 |
| **EMA (趨勢)** | **-0.005** | **-0.003** | ❌ **零預測力** |

**關鍵發現 1 — IC 符號矛盾**：

- RSI、BB 的 IC 是 **正的**（+0.04）→ 均值回歸在 1h 有效（超賣 → 漲回來）
- MACD、OBV、MOM20 的 IC 是 **負的**（-0.025）→ 動量在 1h 是**反指標**（漲多 → 下跌）

這代表**在 1h 時間框架上，均值回歸和趨勢/動量方向相反**。
把它們加權組合 = 左手打右手，互相抵消有效信號。

**關鍵發現 2 — EMA 趨勢無用**：

EMA 趨勢因子 IC ≈ -0.005 / -0.003，完全沒有預測力。
在 Prompt 1 原始計畫中給它 25% 權重 = **稀釋 1/4 的有效信號**。

### 研究 D：Alpha Decay（IC 逐年衰減）

**方法**：分年度計算 RSI / BB 的 IC。

| 年份 | RSI IC | BB IC | 趨勢 |
|------|--------|-------|------|
| 2020 | +0.043 | +0.040 | 正常 |
| 2021 | +0.036 | +0.046 | 正常 |
| 2022 | +0.046 | +0.061 | 較強 |
| **2023** | **+0.065** | **+0.065** | **⭐ 峰值** |
| 2024 | +0.034 | +0.049 | 衰退開始 |
| 2025 | +0.031 | +0.033 | 持續衰退 |
| **2026** | **+0.018** | **+0.003** | **⚠️ 接近歸零** |

**結論**：RSI 的預測力從 2023 峰值（+0.065）衰減到 2026 年（+0.018），**衰減 72%**。
BB 更嚴重，2026 年 IC 僅 +0.003，幾乎完全失效。
這意味著市場正在套利掉這些公開指標的 alpha。

### 研究 E：學術文獻佐證

1. **Moskowitz, Ooi & Pedersen (2012)** *"Time Series Momentum"*, Journal of Financial Economics
   - 趨勢跟蹤在**月度頻率**（1-12 個月動量）被證實有效
   - 但**沒有**證據支持**小時級別**有效 → 與我們的 EMA IC ≈ 0 一致

2. **Lo, Mamaysky & Wang (2000)** *"Foundations of Technical Analysis"*
   - 技術指標在考慮交易成本後 alpha 大幅縮水
   - 有效的指標會隨時間衰減（被市場學習後套利掉）

3. **Fama-French 多因子模型的啟示**：
   - 真正的多因子 = 每個因子來自**不同的風險溢價來源**
   - RSI、MACD、BB、OBV 全部來自同一個價格序列 → **不是真正的多因子**
   - 真正的獨立因子應來自不同數據源（衍生品結構、鏈上數據、跨資產）

4. **加密市場特有的因子研究**：
   - **Funding Rate**：衍生品市場結構信號，不來自價格，被多篇論文驗證有效
   - **跨資產動量**：BTC vs ETH 相對強弱，信息來源不同
   - **波動率結構**：期限結構、realized vs implied volatility

### 研究 F：對原始 Prompt 1 五因子計畫的影響

| # | 原計畫因子 | 權重 | IC | 獨立性 | 結論 |
|---|-----------|------|-----|--------|------|
| 1 | RSI 均值回歸 | 20% | +0.042 ✅ | — | ✅ 保留，但和 BB 是同一因子 |
| 2 | 趨勢 (EMA+ADX) | 25% | **-0.005** ❌ | 獨立 | ❌ **IC ≈ 0，不應做預測因子** |
| 3 | MACD 動量 | 25% | -0.025 | r=-0.66 冗餘 | ❌ **和 RSI 反向冗餘** |
| 4 | 波動率 Regime | 15% | N/A（過濾器） | 獨立 | ⚡ 作為過濾器合理，非預測因子 |
| 5 | OBV 成交量 | 15% | -0.029 | r=-0.55 冗餘 | ❌ **和 RSI 冗餘** |

**原計畫的根本問題**：
1. **假多樣化**：5 個因子只有 ~1.5 個獨立信號源
2. **趨勢因子浪費權重**：EMA 無預測力卻佔 25%
3. **因子互相打架**：均值回歸 (IC > 0) 和動量 (IC < 0) 在 1h 上方向相反
4. **沒有真正的外部信息**：全部因子源自 price + volume，無獨立數據

---

## 現狀盤點

### 已有基礎設施（不需從零開始）

| 模組 | 路徑 | 狀態 | 說明 |
|------|------|------|------|
| Funding Rate 模型 | `backtest/costs.py` | ✅ 已實現 | `compute_funding_costs()` + `adjust_equity_for_funding()` |
| Volume Slippage 模型 | `backtest/costs.py` | ✅ 已實現 | `compute_volume_slippage()` (Square-Root Impact) |
| Walk-Forward 驗證 | `validation/walk_forward.py` | ⚠️ 有框架 | 需要修 bug + 整合到 script |
| Deflated Sharpe Ratio | `validation/prado_methods.py` | ⚠️ 有框架 | 需要整合到回測流程 |
| PBO (過擬合機率) | `validation/prado_methods.py` | ⚠️ 有框架 | 需要搭配參數掃描使用 |
| CPCV | `validation/prado_methods.py` | ⚠️ 有框架 | 需要整合到 script |
| 多因子策略 | `strategy/multi_factor.py` | ✅ 已實現 | 4 因子 (Trend/Momentum/MeanRev/Volume) |
| 多因子 config | `config/futures_multi_factor.yaml` | ✅ 已存在 | 需要調優 |
| Funding Rate 數據下載 | `data/funding_rate.py` | ✅ 已實現 | 可下載歷史 funding rate |

### 診斷發現的核心問題

| # | 問題 | 嚴重度 | 解決方案 |
|---|------|--------|---------|
| 1 | 止損太窄 (1.5 ATR) + 冷卻太短 (1 bar) → 反覆割肉 | 🔴 最高 | Prompt 1 |
| 2 | ADX 過濾器與 RSI 均值回歸邏輯矛盾 | 🔴 高 | Prompt 1 |
| 3 | 🆕 因子假多樣化（5 因子 ≈ 1.5 個獨立信號） | 🔴 高 | Prompt 1（已修正方向） |
| 4 | 🆕 Alpha 衰減：RSI IC 從 2023 到 2026 衰減 72% | 🟡 中 | Prompt 1 + 監控 |
| 5 | 回測未啟用 funding rate 成本扣除 | 🟡 中 | Prompt 3 |
| 6 | 無 Walk-Forward / DSR 驗證，可能 overfitting | 🟡 中 | Prompt 2 |

---

## 修正後的執行順序

基於研究發現，**先搞清楚真實 edge 再改策略**：

```
Prompt 2 (Walk-Forward 驗證) ← 原本排第二，改為最先
  │  理由：先確認現有策略扣除 overfitting 後還剩多少 alpha
  │
  ▼ Review → 如果 Sharpe 衰退 > 50%，策略可能不值得改

Prompt 3 (成本模型) ← 原本排第三，改為第二
  │  理由：扣除 funding rate 後看真實利潤
  │
  ▼ Review → 如果扣成本後 Sharpe < 1.0，需要根本性改變

Prompt 1 (策略修正) ← 原本排第一，改為最後
  │  理由：在知道 true edge 之後，才知道該改什麼
  │  方向：不再做「5 因子加權」，改為「風控修復 + 真正獨立因子」
  │
  ▼ Review → 最終驗收
```

---

## Prompt 1（修正版）：風控修復 + 真正獨立因子

### 核心思路變更

~~原計畫：建立 5 因子加權策略~~

**新計畫**：
1. 保留 RSI 作為唯一的均值回歸預測因子（IC 最高）
2. 把其他指標降級為**過濾器**（不產生信號，只決定「開不開倉」）
3. 修復風控（SL/cooldown）
4. 引入**真正獨立**的信號源（funding rate、跨資產）

### 方案 A：最小修改（低風險，推薦先做）

修改現有 `rsi_adx_atr` 策略的參數和過濾邏輯，不建立新策略：

```yaml
# config/futures_rsi_adx_atr_v2.yaml
strategy:
  name: "rsi_adx_atr"
  params:
    # RSI（唯一的預測因子）
    rsi_period: 10
    oversold: 30
    overbought: 80
    
    # ADX → 降級為「低波動過濾器」
    # 注意：ADX 不應同時要求「有趨勢」又做「均值回歸」
    # 新角色：ADX < 40 時才用 RSI 均值回歸（避免趨勢太強時被碾壓）
    min_adx: 10            # 降低門檻，減少過濾掉的交易
    adx_period: 14
    
    # 風控修復（核心改善）
    stop_loss_atr: 2.5     # 原 1.5 → 2.5（業界標準 2-3x）
    take_profit_atr: null   # 不設固定止盈
    atr_period: 14
    cooldown_bars: 5        # 原 1 → 5（止損後等 5 小時）
```

**預期效果**：
- 虧損重入率：60% → ~30%（cooldown 5h 效果）
- 止損被掃率下降（SL 2.5x 更寬）
- 不引入新的參數 → 不增加 overfitting 風險

### 方案 B：引入真正獨立因子（中等風險）

在方案 A 基礎上，增加來自**不同數據源**的因子：

**候選 1：Funding Rate 因子**
```
來源：Binance Futures 每 8h 結算的 funding rate
獨立性：不來自價格序列，與 RSI 理論相關性 ≈ 0
邏輯：
  - funding_rate > +0.05% → 市場過度看多 → 不做多 / 可做空
  - funding_rate < -0.05% → 市場過度看空 → 不做空 / 可做多
  - 正常範圍 → 不影響
角色：過濾器（不直接產生信號）
數據：已有 download_funding_rates() 可下載
```

**候選 2：BTC-ETH 相對強弱**
```
來源：跨資產價格比
獨立性：不是單一價格序列的衍生
邏輯：
  - ETH/BTC ratio 上升 → ETH 做多偏好增加
  - ETH/BTC ratio 下降 → ETH 做空偏好增加
角色：ETH 的方向偏移因子
限制：只適用於 ETH，BTC 需要其他因子
```

**候選 3：時間季節性因子**
```
來源：歷史統計規律
獨立性：和價格完全無關
邏輯：
  - 統計分析每小時 / 每星期幾的平均收益
  - 某些時段（如美股開盤前後）有統計顯著的偏移
  - 在不利時段降低曝險
角色：過濾器
需要：先做實證分析確認是否存在
```

**候選 4：波動率 Regime 切換**
```
來源：ATR 的百分位數（價格衍生，但衡量不同維度）
獨立性：中等（和 RSI 相關性低，因為衡量的是波動率而非方向）
邏輯：
  - ATR > 95th percentile → 高波動，減少倉位 50%
  - ATR < 25th percentile → 低波動，跳過（不值得交易）
  - 正常 → 全倉位
角色：倉位縮放器（不改變方向）
```

### 方案 C：根本性反思（最高風險，但可能最有價值）

如果 Prompt 2 和 Prompt 3 顯示扣成本後 Sharpe < 1.0：

1. **承認 1h RSI 的 alpha 有限且在衰減**
2. **轉向執行層面的 edge**：
   - 更好的進出場時機（limit order vs market order）
   - 減少 funding 支出（在 funding 結算前平倉）
   - 智慧倉位管理（Kelly criterion）
3. **或轉向不同時間框架**：
   - 趨勢跟蹤在 daily/weekly 上有學術支持（Moskowitz 2012）
   - 1h 均值回歸可能 → 4h/daily 趨勢跟蹤

### Step 1.1：風控參數掃描

**不改策略邏輯**，只掃描風控參數：

```bash
# 掃描 stop_loss_atr × cooldown_bars 的組合
python scripts/scan_risk_params.py -c config/futures_rsi_adx_atr.yaml \
  --sl-range 1.5,2.0,2.5,3.0 \
  --cooldown-range 1,3,5,8,12
```

### Step 1.2：Funding Rate 過濾器（如果方案 B 通過 review）

在 `strategy/filters.py` 新增：

```python
def funding_rate_filter(
    df: pd.DataFrame,
    positions: pd.Series,
    funding_rates: pd.Series,
    max_positive_rate: float = 0.0005,  # 0.05%
    max_negative_rate: float = -0.0005,
) -> pd.Series:
    """
    Funding rate 過濾器：
    - funding > max_positive 時，屏蔽做多信號
    - funding < max_negative 時，屏蔽做空信號
    """
    filtered = positions.copy()
    # 正 funding 太高 → 不做多（要付 funding）
    filtered[(funding_rates > max_positive_rate) & (positions > 0)] = 0.0
    # 負 funding 太高 → 不做空（要付 funding）
    filtered[(funding_rates < max_negative_rate) & (positions < 0)] = 0.0
    return filtered
```

### 驗收標準

- [ ] 方案 A（風控修復）回測通過：虧損重入率 < 40%
- [ ] Walk-Forward 各年度 Sharpe 方差 < 舊策略
- [ ] （可選）Funding Rate 過濾器能減少 10%+ 的年化成本
- [ ] 長短分析（Long/Short split）兩邊都有正 PF

---

## Prompt 2：Walk-Forward 驗證 + Deflated Sharpe

### 目標
確保策略不是 overfitting，用嚴格的統計方法驗證 alpha 的真實性。

### 不做什麼
- ❌ 不改策略邏輯（Prompt 1 已完成）
- ❌ 不改回測引擎
- ❌ 不做參數優化（先驗證，後優化）

### 具體步驟

#### Step 2.1：修復 `walk_forward.py` 的已知問題

現有問題：
1. `run_symbol_backtest` 呼叫方式可能不兼容最新 API（需確認 `strategy_name` 參數）
2. Walk-Forward 切割數據時，應該用完整數據跑策略（warmup），然後只取測試區間的績效
3. 臨時 parquet 文件路徑可能衝突

**修復計畫**：
```python
# 修改 walk_forward_analysis():
# 1. 修正 run_symbol_backtest 呼叫（傳入 strategy_name 而非 cfg.get）
# 2. 確保 warmup：策略在完整歷史上運行，但只取 test 區間的交易
# 3. 新增 strategy_name 參數
```

#### Step 2.2：建立 `scripts/run_walk_forward.py` 腳本

**路徑**: `scripts/run_walk_forward.py`

功能：
```bash
python scripts/run_walk_forward.py -c config/futures_rsi_adx_atr.yaml --splits 6
```

輸出：
```
=== Walk-Forward Analysis ===
Split 1: train 2019-09→2020-12 | test 2021-01→2022-03
  train: +287.3% (SR 3.34) | test: +1695.4% (SR 5.27)
Split 2: train 2019-09→2022-03 | test 2022-04→2023-06
  train: +3247.1% (SR 4.62) | test: +496.5% (SR 4.07)
...

=== 過擬合風險評估 ===
平均 Train Sharpe: 4.12
平均 Test Sharpe:  3.45
Sharpe 衰退率: 16.3%
OOS 一致性: 83.7% → ✅ 可接受（< 30% 衰退）

=== Deflated Sharpe Ratio ===
觀察 Sharpe: 3.60
測試組合數: 31
修正後 Sharpe: 2.09
p-value: 0.0001
結論: ✅ 統計顯著（修正後仍 > 1.5）
```

#### Step 2.3：整合 Deflated Sharpe 到回測報告

在 `scripts/run_backtest.py` 的輸出末尾，自動顯示 DSR：

```python
# 如果用戶指定了 --n-trials（表示做過參數掃描）
if args.n_trials:
    from qtrade.validation.prado_methods import deflated_sharpe_ratio
    dsr = deflated_sharpe_ratio(
        observed_sharpe=sharpe,
        n_trials=args.n_trials,
        n_observations=len(pf.returns()),
        skewness=pf.returns().skew(),
        kurtosis=pf.returns().kurtosis() + 3,
    )
    print(f"  Deflated Sharpe: {dsr.deflated_sharpe:.2f} (p={dsr.p_value:.4f})")
```

#### Step 2.4：建立 CPCV 驗證腳本

**路徑**: `scripts/run_cpcv.py`

```bash
python scripts/run_cpcv.py -c config/futures_rsi_adx_atr.yaml --splits 6 --test-splits 2
```

這會用現有的 `combinatorial_purged_cv()` 函數，執行嚴格的交叉驗證。

### 驗收標準

- [ ] Walk-Forward 所有 test splits Sharpe > 0（沒有任何年虧錢）
- [ ] Sharpe 衰退率 < 30%
- [ ] Deflated Sharpe > 1.5（修正 selection bias 後仍正向）
- [ ] CPCV PBO < 0.5（過擬合機率 < 50%）

---

## Prompt 3：完整成本模型

### 目標
把 funding rate 正式納入回測 Sharpe 計算，看扣除所有成本後策略是否仍然有利可圖。

### 不做什麼
- ❌ 不改策略邏輯
- ❌ 不改驗證框架
- ❌ 不做新的參數優化

### 具體步驟

#### Step 3.1：確認 funding rate 數據

```bash
# 下載歷史 funding rate
python scripts/download_data.py -c config/futures_rsi_adx_atr.yaml --funding-rate
```

檢查數據：
```bash
ls -la data/binance/futures/funding_rate/
# 應該有 BTCUSDT.parquet, ETHUSDT.parquet
```

#### Step 3.2：在回測腳本中強制顯示成本影響

修改 `scripts/run_backtest.py`：在 Futures 回測結束時，**自動**計算並顯示 funding 成本影響。

目前 config 已有 `funding_rate.enabled: true`，但回測報告沒有醒目顯示成本對比。

**新增輸出**：
```
=== 💰 成本影響分析 ===

                     扣 Funding 前    扣 Funding 後
Total Return [%]:      16,456.1%       14,823.7%
Sharpe Ratio:             3.60           3.21
Max Drawdown [%]:        23.3%          24.8%
Annualized Return:       173.5%         158.2%

Funding 成本摘要:
  總成本: $1,632.40 (16.3% of initial)
  年化成本: 2.4%
  結算次數: 6,570
  平均 8h rate: 0.0098%
```

#### Step 3.3：修正回測 Sharpe 為「扣除成本後」

目前 `run_symbol_backtest()` 回傳的 `adjusted_stats` 已經有扣除 funding 後的指標，但回測報告和 Walk-Forward 驗證都用的是原始 Sharpe。

**修改**：
- 在 `run_backtest.py` 的報告中，如果 `adjusted_stats` 存在，以 `adjusted_stats` 的 Sharpe 為主
- 在 Walk-Forward 中，使用 adjusted Sharpe 計算衰退率

#### Step 3.4：建立成本敏感性分析

**新增到** `scripts/run_backtest.py`（加 `--cost-sensitivity` flag）：

```bash
python scripts/run_backtest.py -c config/futures_rsi_adx_atr.yaml --cost-sensitivity
```

測試不同成本假設下的 Sharpe：
```
=== 成本敏感性 ===
Funding Rate     Fee (bps)    Slippage (bps)    Sharpe
0.005%/8h        4            3                 3.45
0.010%/8h        4            3                 3.21   ← 基準
0.020%/8h        4            3                 2.74
0.010%/8h        8            5                 2.89
0.010%/8h        4            10                2.55

→ 策略對 slippage 最敏感（每增加 1bps，Sharpe 降 0.09）
→ 策略對 funding rate 中等敏感（rate 翻倍，Sharpe 降 0.47）
```

### 驗收標準

- [ ] Funding rate 數據成功下載並對齊
- [ ] 回測報告顯示「扣除成本前/後」對比
- [ ] 扣除所有成本後 Sharpe > 1.5（仍然有正 edge）
- [ ] 成本敏感性分析完成，了解策略的成本容忍度

---

## 預期最終結果

| 指標 | 改善前 (rsi_adx_atr) | 改善後 (風控修復 + 驗證 + 成本) |
|------|---------------------|-------------------------------|
| Sharpe (名義) | 3.60 / 4.03 | ~2.5-3.5（更保守但更真實）|
| Sharpe (扣成本) | 未計算 | ~2.0-2.5 |
| Sharpe (Deflated) | ~2.09 / 2.52 | ~1.8-2.2 |
| MDD | 23% / 24% | <20% |
| 虧損重入率 | 60-62% | <35% |
| Walk-Forward 衰退 | 未測 | <30% |
| PBO (過擬合機率) | 未測 | <50% |
| 年間 Sharpe 穩定性 | 6x 差異 (-1.14~6.07) | <3x 差異 |
| ETH 2020 | -58.9%, Sharpe -1.14 | >-20%（至少大幅改善）|

---

## 風險提示

1. **Alpha 正在衰減**：RSI IC 從 2023 到 2026 衰減 72%，策略的有效窗口可能有限
2. **Walk-Forward 可能揭示更多問題**：如果策略的 OOS 表現嚴重衰退，可能需要根本性改變
3. **成本模型可能殺死策略**：如果扣除 funding 後 Sharpe < 1.0，策略可能不值得跑
4. **不要過度優化新策略的參數**：保持預設參數，用 Walk-Forward 驗證其穩健性
5. **真正的多因子需要外部數據**：純價格衍生的因子無法提供真正的多樣化

---

## 完成後的下一步（未來的 Prompt）

- **P4: 真正獨立因子整合**：Funding Rate 過濾器 + 跨資產信號
- **P5: 自適應參數**：用 rolling window 動態調整 RSI 閾值（應對 alpha decay）
- **P6: 策略 ensemble**：同時跑多個策略，信號投票
- **P7: 風險預算**：根據 ATR 動態調整倉位大小（Kelly / Risk Parity）
- **P8: 線上監控**：Rolling IC + Alpha decay 自動偵測，IC < 0.01 時自動停策略
- **P9: 時間框架遷移**：如果 1h alpha 持續衰減，考慮遷移到 4h/daily 趨勢跟蹤
