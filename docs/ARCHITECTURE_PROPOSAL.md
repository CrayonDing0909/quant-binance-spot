# 專案架構重構建議

## 目前問題

1. **驗證代碼分散** - validation 相關程式碼散落在 backtest/ 和 validation/ 兩個地方
2. **Scripts 功能重疊** - 有太多驗證相關的腳本，命名不一致
3. **Config 碎片化** - 每種驗證測試都有獨立的 config 文件

## 建議的新架構

```
quant-binance-spot/
│
├── config/
│   ├── strategies/              # 策略配置（主要）
│   │   ├── rsi_adx_atr.yaml
│   │   ├── ema_cross.yaml
│   │   └── smc_basic.yaml
│   │
│   ├── validation.yaml          # 統一驗證配置（合併所有驗證設定）
│   └── settings.yaml            # 全局設定（API、通知等）
│
├── scripts/
│   ├── backtest.py              # 主回測入口
│   ├── validate.py              # 統一驗證入口（整合所有驗證）
│   ├── live.py                  # 實盤入口
│   ├── download.py              # 數據下載
│   └── report.py                # 報告生成
│
├── src/qtrade/
│   ├── core/                    # 核心模組
│   │   ├── config.py
│   │   └── types.py
│   │
│   ├── data/                    # 數據層（不變）
│   │   ├── binance_client.py
│   │   ├── klines.py
│   │   └── storage.py
│   │
│   ├── indicators/              # 指標層（不變）
│   │   └── ...
│   │
│   ├── strategy/                # 策略層（不變）
│   │   └── ...
│   │
│   ├── backtest/                # 回測層
│   │   ├── engine.py            # 回測引擎
│   │   ├── metrics.py           # 績效指標
│   │   └── plotting.py          # 繪圖
│   │
│   ├── validation/              # 驗證層（整合）
│   │   ├── __init__.py
│   │   ├── oos_testing.py       # OOS 測試
│   │   ├── walk_forward.py      # Walk-Forward
│   │   ├── monte_carlo.py       # Monte Carlo
│   │   ├── cross_asset.py       # Cross-Asset
│   │   ├── prado_methods.py     # DSR, PBO, CPCV
│   │   └── consistency.py       # 一致性檢查
│   │
│   ├── risk/                    # 風險層（不變）
│   │   └── ...
│   │
│   ├── live/                    # 實盤層（不變）
│   │   └── ...
│   │
│   └── utils/                   # 工具層
│       └── ...
│
├── tests/                       # 測試
│   └── ...
│
├── reports/                     # 報告（自動生成）
│   └── {strategy}/{timestamp}/  # 每次運行一個目錄
│       ├── backtest/
│       ├── validation/
│       └── summary.json
│
└── data/                        # 數據
    └── ...
```

## 統一驗證配置 (validation.yaml)

```yaml
# config/validation.yaml
# 所有驗證設定集中在這裡

defaults:
  train_ratio: 0.6
  val_ratio: 0.2
  test_ratio: 0.2

oos_testing:
  enabled: true
  periods:
    - name: "full_history"
      train_start: "2017-08-17"
      train_end: "2023-12-31"
      test_start: "2024-01-01"
    - name: "recent"
      train_start: "2022-01-01"
      train_end: "2024-12-31"
      test_start: "2025-01-01"

walk_forward:
  enabled: true
  n_splits: 5

monte_carlo:
  enabled: true
  n_simulations: 10000
  confidence_levels: [0.95, 0.99]

cross_asset:
  enabled: true
  run_loao: true
  run_correlation_stratified: true
  run_regime_validation: true

prado_methods:
  deflated_sharpe:
    enabled: true
    n_trials: 729  # Grid search 組合數
  pbo:
    enabled: true
    threshold: 0.5
  cpcv:
    enabled: false  # 預設關閉（耗時）
    n_splits: 6
    n_test_splits: 2

market_regimes:
  - name: "bear_market_2022"
    start: "2022-01-01"
    end: "2023-01-01"
    description: "2022 熊市"
  - name: "bull_market_2021"
    start: "2021-01-01"
    end: "2022-01-01"
    description: "2021 牛市"
```

## 統一驗證腳本 (validate.py)

```python
# scripts/validate.py
"""
統一驗證入口

使用方式:
    # 執行所有驗證
    python scripts/validate.py -c config/strategies/rsi_adx_atr.yaml

    # 只執行特定驗證
    python scripts/validate.py -c config/strategies/rsi_adx_atr.yaml --only oos,walk_forward

    # 快速驗證（跳過耗時的測試）
    python scripts/validate.py -c config/strategies/rsi_adx_atr.yaml --quick

    # 完整驗證（包括 CPCV）
    python scripts/validate.py -c config/strategies/rsi_adx_atr.yaml --full
"""
```

## 統一報告結構

每次運行會生成：

```
reports/rsi_adx_atr/20260209_183000/
├── backtest/
│   ├── equity_curve_BTCUSDT.png
│   ├── stats_BTCUSDT.csv
│   └── trades_BTCUSDT.csv
│
├── validation/
│   ├── oos_testing.csv
│   ├── walk_forward.csv
│   ├── monte_carlo.csv
│   ├── cross_asset.csv
│   ├── deflated_sharpe.csv
│   └── pbo.csv
│
├── run_info.json          # 運行資訊
└── summary.md             # 人類可讀的總結報告
```

## 重構步驟

### Phase 1: 整合驗證模組
1. 建立 `src/qtrade/validation/` 統一目錄
2. 移動所有驗證代碼到該目錄
3. 建立統一的 `ValidationPipeline` 類

### Phase 2: 整合腳本
1. 建立 `scripts/validate.py` 統一入口
2. 刪除重複的驗證腳本
3. 保留舊腳本作為 deprecated

### Phase 3: 整合配置
1. 建立 `config/validation.yaml`
2. 移除碎片化的驗證配置
3. 更新文檔

## 優先級

- **高**: 整合驗證模組（減少代碼重複）
- **中**: 整合腳本（改善使用體驗）
- **低**: 整合配置（維護成本）

## 是否立即執行？

這個重構會影響較多文件，建議：
1. 先完成當前功能開發
2. 確認策略穩定後再重構
3. 或者漸進式重構（先整合最常用的部分）
