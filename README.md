# quant-binance-spot

使用 vectorbt 進行研究/回測，並支援 Binance 現貨即時交易。

## 專案架構

```
quant-binance-spot/
├── config/              # 配置檔
│   ├── base.yaml        # 基礎配置
│   ├── rsi_adx_atr.yaml # RSI+ADX+ATR 策略（推薦）
│   └── dev.yaml         # 開發環境配置
├── data/                # 資料儲存
│   └── binance/spot/    # Binance 現貨資料
├── docs/                # 文件
│   ├── QUICK_START_GUIDE.md   # 新手完整教學 ⭐
│   ├── PROJECT_FEATURES.md    # 專案功能說明
│   ├── RISK_MANAGEMENT.md     # 風險管理指南
│   └── ...
├── reports/             # 回測報告和圖表
│   └── live/            # 即時交易狀態
├── scripts/             # 腳本
│   ├── download_data.py       # 下載資料
│   ├── run_backtest.py        # 運行回測
│   ├── run_live.py            # 即時交易（Paper/Real）
│   ├── health_check.py        # 系統健康檢查
│   ├── daily_report.py        # 每日績效報表
│   ├── run_consistency_check.py # Live/Backtest 一致性驗證
│   ├── validate_strategy.py   # 策略驗證（過擬合檢測）
│   ├── validate_kelly.py      # Kelly Criterion 驗證
│   ├── optimize_params.py     # 參數優化工具
│   ├── create_strategy.py     # 策略模板生成器
│   └── setup_cron.sh          # Cron Jobs 自動設定
└── src/qtrade/
    ├── backtest/        # 回測模組
    ├── indicators/      # 指標庫（RSI, EMA, MACD, ADX, ATR...）
    ├── strategy/        # 策略模組
    ├── live/            # 即時交易模組
    │   ├── runner.py          # 交易主循環
    │   ├── paper_broker.py    # Paper Trading 模擬
    │   ├── signal_generator.py # 信號生成器
    │   └── trading_state.py   # 狀態持久化
    ├── monitor/         # 監控模組
    │   ├── health.py          # 健康檢查
    │   └── notifier.py        # Telegram 通知
    ├── validation/      # 驗證模組
    ├── risk/            # 風險管理
    └── config.py        # 配置載入
```

## 快速開始

```bash
# 建立虛擬環境
python -m venv .venv
source .venv/bin/activate
pip install -e .

# 設定環境變數（Telegram 通知）
cat > .env << EOF
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
EOF

# 下載資料
python scripts/download_data.py

# 運行回測
python scripts/run_backtest.py -c config/rsi_adx_atr.yaml

# 驗證策略（過擬合檢測）
python scripts/validate_strategy.py -c config/rsi_adx_atr.yaml
```

## 即時交易

### Paper Trading（模擬交易）

```bash
# 單次執行（測試用）
python scripts/run_live.py -c config/rsi_adx_atr.yaml --paper --once

# 持續運行
python scripts/run_live.py -c config/rsi_adx_atr.yaml --paper
```

### 自動化排程（Cron Jobs）

```bash
# 自動設定 Cron Jobs（支援 macOS / Linux / Oracle Cloud）
chmod +x scripts/setup_cron.sh
./scripts/setup_cron.sh --install

# 查看目前設定
./scripts/setup_cron.sh --show

# 移除
./scripts/setup_cron.sh --remove
```

預設排程：
- 每小時整點：執行交易信號檢查
- 每 30 分鐘：健康檢查（異常時 Telegram 通知）
- 每天 08:05（台灣時間）：每日績效報表
- 每週日：Live/Backtest 一致性驗證

### 健康檢查

```bash
# 執行檢查
python scripts/health_check.py -c config/rsi_adx_atr.yaml

# 異常時發送 Telegram 通知
python scripts/health_check.py -c config/rsi_adx_atr.yaml --notify

# 正常時也發送通知
python scripts/health_check.py -c config/rsi_adx_atr.yaml --notify --notify-on-ok
```

## 配置說明

### config/rsi_adx_atr.yaml（範例）

```yaml
market:
  symbols: ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
  interval: "1h"
  start: "2022-01-01"
  end: null

backtest:
  initial_cash: 10000
  fee_bps: 6           # 手續費（基點，6 = 0.06%）
  slippage_bps: 5      # 滑點
  trade_on: "next_open"
  validate_data: true
  clean_data: true

strategy:
  name: "rsi_adx_atr"
  params:
    rsi_period: 10
    overbought: 65
    min_adx: 15
    adx_period: 14
    stop_loss_atr: 1.5
    take_profit_atr: 4.0
    atr_period: 14
    cooldown_bars: 1
  # 幣種專屬參數覆寫
  symbol_overrides:
    BTCUSDT:
      oversold: 30
    ETHUSDT:
      oversold: 35

# 風控
risk:
  max_drawdown_pct: 0.30  # 最大回撤 30% 觸發熔斷

# 倉位計算
position_sizing:
  method: "fixed"         # fixed / kelly / volatility
  position_pct: 1.0
  kelly_fraction: 0.25    # Quarter Kelly（保守）
  min_trades_for_kelly: 20

# 多幣種倉位分配
portfolio:
  cash_reserve: 0.20      # 保留 20% 現金
  allocation: null        # null = 自動等權分配
  # 手動分配範例：
  # allocation:
  #   BTCUSDT: 0.30
  #   ETHUSDT: 0.30
  #   BNBUSDT: 0.20
  #   SOLUSDT: 0.20

output:
  report_dir: "./reports"
```

### 自動倉位分配

設定 `allocation: null` 時，系統會自動計算：

```
每幣權重 = (1 - cash_reserve) / 幣種數量
```

| 幣種數 | 現金保留 | 每幣權重 |
|--------|---------|---------|
| 4 個 | 20% | 20% |
| 5 個 | 20% | 16% |
| 6 個 | 20% | 13.3% |

新增幣種只需修改 `market.symbols`，倉位會自動重新分配！

## 策略類型

### RSI + ADX + ATR（推薦）

```yaml
strategy:
  name: "rsi_adx_atr"
  params:
    rsi_period: 10
    oversold: 30
    min_adx: 15
    stop_loss_atr: 1.5
    take_profit_atr: 4.0
```

### EMA 交叉策略

```yaml
strategy:
  name: "ema_cross"
  params:
    fast: 20
    slow: 60
```

### RSI 策略

```yaml
strategy:
  name: "rsi"
  params:
    period: 14
    oversold: 30
    overbought: 70
```

## 開發新策略

### 使用模板生成器（推薦）

```bash
python scripts/create_strategy.py --name my_strategy --type rsi
```

### 手動建立

在 `src/qtrade/strategy/` 目錄下建立新的策略檔案：

```python
from __future__ import annotations
import pandas as pd
from .base import StrategyContext
from . import register_strategy


@register_strategy("my_strategy")
def generate_positions(df: pd.DataFrame, ctx: StrategyContext, params: dict) -> pd.Series:
    """
    生成持倉信號
    
    Returns:
        pd.Series: 持倉比例 [0, 1]，1.0 = 滿倉，0.0 = 空倉
    """
    from qtrade.indicators import calculate_ema
    
    close = df["close"]
    ema_fast = calculate_ema(close, params.get("fast", 20))
    ema_slow = calculate_ema(close, params.get("slow", 60))
    
    signal = (ema_fast > ema_slow).astype(float)
    pos = signal.shift(1).fillna(0.0)  # 避免未來資訊洩露
    
    return pos.clip(0.0, 1.0)
```

## 驗證策略（過擬合檢測）

### Walk-Forward Analysis

```bash
python scripts/validate_strategy.py -c config/rsi_adx_atr.yaml
```

**判斷過擬合：**
- 訓練集收益率 >> 測試集收益率（下降 >30%）
- 測試集回撤明顯增加

### 參數敏感性分析

```bash
python scripts/optimize_params.py --strategy rsi_adx_atr
```

**判斷過擬合：**
- 參數微小變化導致收益率大幅波動
- 收益率標準差 > 50%

## 風險管理

```python
from qtrade.risk import FixedPositionSizer, KellyPositionSizer

# 固定倉位
sizer = FixedPositionSizer(position_pct=0.8)
size = sizer.calculate_size(signal=1.0, equity=10000, price=50000)

# Kelly Criterion
kelly = KellyPositionSizer(kelly_fraction=0.25)
```

詳細文件：[風險管理指南](docs/RISK_MANAGEMENT.md)

## 📚 相關文件

### 🚀 新手必讀
- **[快速開始指南](docs/QUICK_START_GUIDE.md)** ⭐⭐⭐ - 從策略發想到實現的完整教學
- **[專案功能說明](docs/PROJECT_FEATURES.md)** ⭐⭐ - 專案提供的所有功能詳細說明

### 📖 詳細文件
- [命令列使用指南](docs/COMMAND_LINE_USAGE.md)
- [風險管理指南](docs/RISK_MANAGEMENT.md)
- [資料品質檢查指南](docs/DATA_QUALITY.md)
- [策略組合指南](docs/STRATEGY_PORTFOLIO.md)
- [交易策略參考](docs/TRADING_STRATEGIES_REFERENCE.md)

## 注意事項

1. **避免未來資訊洩露**：信號必須使用 `shift(1)` 向後移動
2. **測試策略**：運行 `tests/test_strategy_no_lookahead.py` 確保沒有未來資訊洩露
3. **參數選擇**：使用驗證腳本測試參數穩定性，避免過度優化
4. **Telegram 通知**：設定 `.env` 檔案以接收交易和健康檢查通知
