# CLI 指令集快速參考

> ⚠️ **重要**：執行任何指令前，先啟動虛擬環境：
> ```bash
> cd /Users/dylanting/Documents/spot_bot/quant-binance-spot
> source .venv/bin/activate
> ```

---

## 📋 指令總覽

| 指令 | 用途 | 常用範例 |
|------|------|----------|
| `run_backtest.py` | 策略回測 | `python scripts/run_backtest.py -c config/futures_rsi_adx_atr.yaml` |
| `validate.py` | 策略驗證 | `python scripts/validate.py -c config/rsi_adx_atr.yaml --quick` |
| `download_data.py` | 下載數據 | `python scripts/download_data.py -c config/rsi_adx_atr.yaml` |
| `run_live.py` | 實盤/模擬交易 | `python scripts/run_live.py -c config/rsi_adx_atr.yaml --paper` |
| `optimize_params.py` | 參數優化 | `python scripts/optimize_params.py --strategy rsi_adx_atr` |
| `create_strategy.py` | 建立新策略 | `python scripts/create_strategy.py --name my_strategy` |
| `health_check.py` | 系統健康檢查 | `python scripts/health_check.py --notify` |
| `daily_report.py` | 每日報告 | `python scripts/daily_report.py` |

---

## 🔥 最常用指令

### 現貨 (Spot) 完整流程

```bash
# 1. 下載數據
python scripts/download_data.py -c config/rsi_adx_atr.yaml

# 2. 回測
python scripts/run_backtest.py -c config/rsi_adx_atr.yaml

# 3. 驗證（快速）
python scripts/validate.py -c config/rsi_adx_atr.yaml --quick

# 4. 模擬交易
python scripts/run_live.py -c config/rsi_adx_atr.yaml --paper
```

### 合約 (Futures) 完整流程

```bash
# 1. 下載數據
python scripts/download_data.py -c config/futures_rsi_adx_atr.yaml

# 2. 回測（多空都做）
python scripts/run_backtest.py -c config/futures_rsi_adx_atr.yaml --direction both

# 3. 回測（只做多）
python scripts/run_backtest.py -c config/futures_rsi_adx_atr.yaml --direction long_only

# 4. 回測（只做空）
python scripts/run_backtest.py -c config/futures_rsi_adx_atr.yaml --direction short_only

# 5. 驗證
python scripts/validate.py -c config/futures_rsi_adx_atr.yaml --quick

# 6. 模擬交易
python scripts/run_live.py -c config/futures_rsi_adx_atr.yaml --paper
```

---

## 📊 回測 (run_backtest.py)

```bash
python scripts/run_backtest.py [OPTIONS]
```

| 參數 | 說明 | 範例 |
|------|------|------|
| `-c, --config` | 配置檔路徑 | `-c config/rsi_adx_atr.yaml` |
| `-s, --strategy` | 策略名稱（覆蓋配置） | `-s rsi` |
| `--symbol` | 指定交易對 | `--symbol BTCUSDT` |
| `--output-dir` | 輸出目錄 | `--output-dir reports/test` |
| `-t, --timestamp` | 加時間戳（預設啟用） | `-t` |
| `--no-timestamp` | 不加時間戳（會覆蓋） | `--no-timestamp` |
| `-d, --direction` | 交易方向 | `-d both` / `-d long_only` / `-d short_only` |

**範例：**
```bash
# 基本回測
python scripts/run_backtest.py -c config/rsi_adx_atr.yaml

# 只回測 BTCUSDT
python scripts/run_backtest.py -c config/rsi_adx_atr.yaml --symbol BTCUSDT

# 合約做多做空
python scripts/run_backtest.py -c config/futures_rsi_adx_atr.yaml -d both

# 不加時間戳（覆蓋舊報告）
python scripts/run_backtest.py -c config/rsi_adx_atr.yaml --no-timestamp
```

---

## ✅ 驗證 (validate.py)

```bash
python scripts/validate.py -c CONFIG [OPTIONS]
```

| 參數 | 說明 | 範例 |
|------|------|------|
| `-c, --config` | 配置檔路徑（必要） | `-c config/rsi_adx_atr.yaml` |
| `-v, --validation-config` | 驗證配置檔 | `-v config/validation.yaml` |
| `--quick` | 快速模式（基本驗證） | `--quick` |
| `--full` | 完整模式（所有驗證） | `--full` |
| `--only` | 只執行指定驗證 | `--only walk_forward,monte_carlo` |
| `-o, --output` | 輸出目錄 | `-o reports/validation` |

**可用驗證項目：**
- `walk_forward` - Walk-Forward 分析
- `monte_carlo` - 蒙地卡羅模擬
- `loao` - Leave-One-Asset-Out
- `regime` - 市場狀態分析
- `dsr` - Deflated Sharpe Ratio
- `pbo` - Probability of Backtest Overfitting
- `kelly` - Kelly Criterion
- `consistency` - 一致性檢查

**範例：**
```bash
# 快速驗證
python scripts/validate.py -c config/rsi_adx_atr.yaml --quick

# 完整驗證
python scripts/validate.py -c config/rsi_adx_atr.yaml --full

# 只執行特定驗證
python scripts/validate.py -c config/rsi_adx_atr.yaml --only walk_forward,kelly
```

---

## 📥 數據下載 (download_data.py)

```bash
python scripts/download_data.py [OPTIONS]
```

| 參數 | 說明 | 範例 |
|------|------|------|
| `-c, --config` | 配置檔路徑 | `-c config/rsi_adx_atr.yaml` |
| `--symbol` | 只下載指定交易對 | `--symbol BTCUSDT` |
| `--full` | 強制全量下載 | `--full` |
| `--status` | 只顯示狀態 | `--status` |

**範例：**
```bash
# 下載配置檔中的所有交易對
python scripts/download_data.py -c config/rsi_adx_atr.yaml

# 只下載 BTCUSDT
python scripts/download_data.py -c config/rsi_adx_atr.yaml --symbol BTCUSDT

# 查看本地數據狀態
python scripts/download_data.py -c config/rsi_adx_atr.yaml --status

# 強制重新下載
python scripts/download_data.py -c config/rsi_adx_atr.yaml --full
```

---

## 🚀 實盤/模擬交易 (run_live.py)

```bash
python scripts/run_live.py -c CONFIG [OPTIONS]
```

| 參數 | 說明 | 範例 |
|------|------|------|
| `-c, --config` | 配置檔路徑 | `-c config/rsi_adx_atr.yaml` |
| `-s, --strategy` | 策略名稱 | `-s rsi_adx_atr` |
| `--symbol` | 只交易指定交易對 | `--symbol BTCUSDT` |
| `--paper` | Paper Trading（預設） | `--paper` |
| `--real` | 真實交易（需 API Key） | `--real` |
| `--status` | 查看帳戶狀態 | `--status` |
| `--check` | 檢查 API 連線 | `--check` |
| `--once` | 只執行一次 | `--once` |
| `--dry-run` | 真實模式但不下單 | `--dry-run` |

**範例：**
```bash
# 模擬交易
python scripts/run_live.py -c config/rsi_adx_atr.yaml --paper

# 查看模擬帳戶狀態
python scripts/run_live.py -c config/rsi_adx_atr.yaml --status

# 檢查 API 連線
python scripts/run_live.py -c config/rsi_adx_atr.yaml --check

# 真實交易（測試模式，不實際下單）
python scripts/run_live.py -c config/rsi_adx_atr.yaml --real --dry-run

# 真實交易（會實際下單！）
python scripts/run_live.py -c config/rsi_adx_atr.yaml --real
```

---

## 🔧 參數優化 (optimize_params.py)

```bash
python scripts/optimize_params.py --strategy STRATEGY [OPTIONS]
```

| 參數 | 說明 | 範例 |
|------|------|------|
| `--strategy` | 策略名稱（必要） | `--strategy rsi_adx_atr` |
| `--method` | 優化方法 | `--method grid` |
| `--metric` | 優化目標 | `--metric "Sharpe Ratio"` |
| `--config` | 配置檔路徑 | `--config config/rsi_adx_atr.yaml` |
| `--symbol` | 指定交易對 | `--symbol BTCUSDT` |

**範例：**
```bash
# 基本參數優化
python scripts/optimize_params.py --strategy rsi_adx_atr

# 優化 Sharpe Ratio
python scripts/optimize_params.py --strategy rsi_adx_atr --metric "Sharpe Ratio"

# 只優化 BTCUSDT
python scripts/optimize_params.py --strategy rsi_adx_atr --symbol BTCUSDT
```

---

## 🏗️ 建立新策略 (create_strategy.py)

```bash
python scripts/create_strategy.py --name NAME [OPTIONS]
```

| 參數 | 說明 | 範例 |
|------|------|------|
| `--name` | 策略名稱（必要） | `--name my_awesome_strategy` |
| `--type` | 策略類型 | `--type rsi` / `--type custom` |

**範例：**
```bash
# 建立自訂策略
python scripts/create_strategy.py --name my_strategy --type custom

# 建立 RSI 類型策略
python scripts/create_strategy.py --name my_rsi --type rsi
```

---

## 🏥 健康檢查 (health_check.py)

```bash
python scripts/health_check.py [OPTIONS]
```

| 參數 | 說明 | 範例 |
|------|------|------|
| `-c, --config` | 配置檔路徑 | `-c config/rsi_adx_atr.yaml` |
| `--notify` | 異常時發送通知 | `--notify` |
| `--notify-on-ok` | 正常也發送通知 | `--notify-on-ok` |
| `--json` | JSON 格式輸出 | `--json` |

**範例：**
```bash
# 基本健康檢查
python scripts/health_check.py

# 檢查並通知
python scripts/health_check.py --notify
```

---

## 🧪 測試腳本

```bash
# Futures API 連線測試（不需要 API Key）
python scripts/test_futures_connection.py

# Futures Broker 測試（需要 API Key）
python scripts/test_futures_broker.py

# Futures 風控測試
python scripts/test_futures_risk.py --funding-only
```

---

## 📁 配置檔參考

| 檔案 | 用途 |
|------|------|
| `config/base.yaml` | 基礎配置 |
| `config/rsi_adx_atr.yaml` | RSI+ADX+ATR 現貨策略 |
| `config/futures_rsi_adx_atr.yaml` | RSI+ADX+ATR 合約策略 |
| `config/validation.yaml` | 驗證配置 |

---

## 💡 小技巧

### 查看任何指令的幫助
```bash
python scripts/SCRIPT_NAME.py --help
```

### 批次回測多個策略
```bash
for config in config/*.yaml; do
    python scripts/run_backtest.py -c "$config"
done
```

### 快速檢查 API 連線
```bash
python scripts/test_futures_connection.py  # 不需要 API Key
```

### 設置環境變數（交易用）
```bash
export BINANCE_API_KEY=your_key
export BINANCE_API_SECRET=your_secret
```
