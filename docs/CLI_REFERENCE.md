# 📍 專案地圖 & 指令速查

> **最後更新**: 2026-02-16 | **主力配置**: `config/futures_rsi_adx_atr.yaml`
>
> 這份文件是整個專案的「儀表板」。其他文件太長不想看？只看這份。

---

## 🎯 我想做什麼？

| 我想... | 指令 |
|---------|------|
| **回測策略** | `python scripts/run_backtest.py -c config/futures_rsi_adx_atr.yaml` |
| **看回測加上成本後的真實績效** | 同上（config 已設 `funding_rate.enabled: true`，自動顯示前/後對比）|
| **用 DSR 校正 Sharpe** | `python scripts/run_backtest.py -c config/futures_rsi_adx_atr.yaml --n-trials 31` |
| **Walk-Forward 驗證** | `python scripts/run_walk_forward.py -c config/futures_rsi_adx_atr.yaml --splits 6` |
| **CPCV 交叉驗證** | `python scripts/run_cpcv.py -c config/futures_rsi_adx_atr.yaml --splits 6 --test-splits 2` |
| **成本敏感性分析** | `python scripts/run_cost_sensitivity.py -c config/futures_rsi_adx_atr.yaml` |
| **一站式驗證** | `python scripts/validate.py -c config/futures_rsi_adx_atr.yaml --quick` |
| **Pre-Deploy 檢查** | `python scripts/validate_live_consistency.py -c config/futures_rsi_adx_atr.yaml` |
| **下載數據** | `python scripts/download_data.py -c config/futures_rsi_adx_atr.yaml` |
| **下載 Funding Rate** | `python scripts/download_data.py -c config/futures_rsi_adx_atr.yaml --funding-rate` |
| **參數掃描（SL × Cooldown）** | `python scripts/scan_risk_params.py -c config/futures_rsi_adx_atr.yaml` |
| **參數掃描（overbought）** | `python scripts/scan_overbought.py -c config/futures_rsi_adx_atr.yaml` |
| **Hyperopt 優化** | `python scripts/run_hyperopt.py -c config/futures_rsi_adx_atr.yaml` |
| **組合回測** | `python scripts/run_portfolio_backtest.py -c config/futures_rsi_adx_atr.yaml` |
| **實盤（cron 模式）** | `python scripts/run_live.py -c config/futures_rsi_adx_atr.yaml --real --once` |
| **實盤（WebSocket 模式）** | `python scripts/run_websocket.py -c config/futures_rsi_adx_atr.yaml --real` ⭐ |
| **Dry-run 測試** | `python scripts/run_live.py -c config/futures_rsi_adx_atr.yaml --real --dry-run --once` |
| **查詢交易資料庫** | `python scripts/query_db.py -c config/futures_rsi_adx_atr.yaml summary` |
| **Telegram Bot** | `python scripts/run_telegram_bot.py -c config/futures_rsi_adx_atr.yaml --real` |
| **健康檢查** | `python scripts/health_check.py -c config/futures_rsi_adx_atr.yaml --real --notify` |
| **每日報表** | `python scripts/daily_report.py -c config/futures_rsi_adx_atr.yaml` |
| **建立新策略** | `python scripts/create_strategy.py --name my_strategy --type custom` |
| **Oracle 更新部署** | `git pull && ./scripts/setup_cron.sh --update` |
| **Oracle 配置 Swap** | `bash scripts/setup_swap.sh` |
| **Alpha Decay 監控** | `python scripts/monitor_alpha_decay.py -c config/futures_rsi_adx_atr.yaml` |
| **策略相關性分析** | `python scripts/research_strategy_correlation.py -c config/futures_rsi_adx_atr.yaml` |

---

## 📂 所有腳本一覽

### 核心流程（按順序）

| # | 腳本 | 用途 | 關鍵參數 |
|---|------|------|----------|
| 1 | `download_data.py` | 下載 K 線 / Funding Rate | `-c`, `--funding-rate`, `--full` |
| 2 | `run_backtest.py` | 回測（含成本模型） | `-c`, `--symbol`, `-d both/long_only/short_only`, `--n-trials` |
| 3 | `run_walk_forward.py` | Walk-Forward 驗證 | `-c`, `--splits`, `--n-trials` |
| 4 | `run_cpcv.py` | CPCV 交叉驗證 | `-c`, `--splits`, `--test-splits` |
| 5 | `run_cost_sensitivity.py` | 成本敏感性分析 | `-c`, `--symbol` |
| 6 | `validate.py` | 一站式驗證（WFA/MC/DSR/PBO/Kelly） | `-c`, `--quick`, `--full`, `--only` |
| 7 | `validate_live_consistency.py` | Pre-Deploy 13 項檢查 | `-c`, `-v`, `--only` |
| 8 | `run_live.py` | 實盤 / Paper（Polling 模式） | `-c`, `--real/--paper`, `--once`, `--dry-run` |
| 9 | `run_websocket.py` ⭐ | 實盤（WebSocket 事件驅動） | `-c`, `--real/--paper` |

### 優化 & 分析

| 腳本 | 用途 |
|------|------|
| `optimize_params.py` | 網格搜尋參數優化 |
| `run_hyperopt.py` | Bayesian 超參數優化 |
| `scan_overbought.py` | 掃描 overbought 最佳值 |
| `scan_risk_params.py` ⭐ | SL × Cooldown 網格掃描（熱力圖） |
| `comprehensive_backtest.py` | 多維度綜合回測（regime / exit / sizing） |
| `run_portfolio_backtest.py` | 多幣種組合回測 |

### 運維 & 監控

| 腳本 | 用途 |
|------|------|
| `run_telegram_bot.py` | Telegram 互動 Bot（常駐服務） |
| `query_db.py` ⭐ | SQLite 交易資料庫查詢（summary / trades / signals / equity / export） |
| `health_check.py` | 系統健康檢查（cron 每 30 分鐘） |
| `daily_report.py` | 每日績效報表 |
| `monitor_alpha_decay.py` ⭐ | Alpha Decay 監控（IC 分析 + Telegram 通知） |
| `cron_alpha_monitor.sh` ⭐ | Alpha Decay 監控排程腳本 |
| `setup_cron.sh` | 自動設定 cron + 清 `.pyc`（`--update`） |
| `setup_swap.sh` | Oracle Cloud Swap 配置（1GB RAM 機器必備） |
| `setup_secrets.py` | 設定 API Key / Telegram Token |

### 研究 & 分析

| 腳本 | 用途 |
|------|------|
| `research_dynamic_rsi.py` ⭐ | Static vs Dynamic RSI 對比研究 |
| `research_funding_filter.py` ⭐ | Funding Rate 過濾效果分析 |
| `research_strategy_correlation.py` ⭐ | 策略相關性矩陣 + Ensemble 推薦 |

### 測試 & 開發

| 腳本 | 用途 |
|------|------|
| `create_strategy.py` | 策略範本產生器 |
| `test_futures_connection.py` | 合約 API 連線測試（不需 Key） |
| `test_futures_broker.py` | Broker 功能測試（需 Key） |
| `test_futures_manual.py` | 手動合約功能測試 |
| `test_futures_risk.py` | 風控功能測試 |

---

## ⚙️ 配置檔清單

### 🔴 生產主力

| 配置檔 | 用途 | Oracle 部署 |
|--------|------|:-----------:|
| `futures_rsi_adx_atr.yaml` | **合約 RSI+ADX+ATR v3.1（BTC+ETH+SOL 三幣）** | ✅ |

### 📊 回測 / 研究用

| 配置檔 | 用途 |
|--------|------|
| `rsi_adx_atr.yaml` | 現貨版本 |
| `rsi_adx_atr_rsi_exit.yaml` | RSI Exit 變體（TP=null） |
| `futures_rsi_adx_atr_15m.yaml` | 15m 時間框架（HTF=1h） |
| `futures_rsi_adx_atr_4h.yaml` | 4h 時間框架（HTF=1d） |
| `futures_ensemble.yaml` | RSI+MACD 組合策略 |
| `futures_full_history.yaml` | 長期歷史回測 |
| `rsi_adx_atr_full_history.yaml` | 現貨長期歷史 |

### 📁 範例 / 實驗（可忽略）

| 配置檔 | 說明 |
|--------|------|
| `base.yaml` | 基礎範本 |
| `dev.yaml` | 開發用 |
| `futures_multi_factor.yaml` | 多因子實驗（已廢棄方向） |
| `futures_bb_mean_reversion.yaml` | BB 策略實驗 |
| `futures_macd_momentum.yaml` | MACD 策略實驗 |
| `futures_rsi_adx_atr_enhanced.yaml` | Enhanced 變體 |
| `rsi_adx_atr_enhanced.yaml` | Enhanced 現貨版 |
| `rsi_adx_atr_1d.yaml` | 日線回測 |
| `my_strategy_example.yaml` | 教學範例 |
| `rsi_example.yaml` | RSI 教學範例 |
| `smc_example.yaml` | SMC 教學範例 |
| `stock_rsi_adx_atr.yaml` | 股票回測 |
| `validation.yaml` | 驗證專用配置 |

---

## 🧩 原始碼模組地圖

```
src/qtrade/
├── config.py              ← 統一配置管理（AppConfig, load_config）
├── strategy/              ← 策略庫
│   ├── rsi_adx_atr_strategy.py  ← ⭐ 主力策略（Dynamic RSI + Funding + Vol Filter + HTF Soft）
│   ├── ensemble_strategy.py     ← RSI+MACD 組合策略
│   ├── base.py                  ← StrategyContext
│   ├── exit_rules.py            ← SL/TP/RSI Exit 邏輯
│   ├── filters.py               ← ⭐ 過濾器（Funding Rate / 波動率 / HTF 軟趨勢）
│   ├── multi_factor.py          ← 多因子（實驗）
│   ├── bb_mean_reversion.py     ← BB（實驗）
│   ├── macd_momentum.py         ← MACD（實驗）
│   └── ...其他範例
├── indicators/            ← 技術指標（RSI, ADX, ATR, BB, MACD, EMA, OBV...）
├── backtest/
│   ├── run_backtest.py    ← 回測引擎 (run_symbol_backtest + ⭐ Volatility Targeting)
│   ├── costs.py           ← 成本模型（Funding Rate + Volume Slippage）
│   ├── metrics.py         ← 績效指標 + Long/Short 分析
│   ├── plotting.py        ← 繪圖
│   └── hyperopt_engine.py ← Bayesian 優化
├── validation/
│   ├── walk_forward.py    ← Walk-Forward Analysis + Summary
│   ├── prado_methods.py   ← DSR, PBO, CPCV
│   ├── ic_monitor.py      ← ⭐ Alpha Decay 監控（Rolling IC + 年度 IC + 警報）
│   ├── consistency.py     ← Live/Backtest 一致性
│   └── cross_asset.py     ← 跨資產驗證
├── live/
│   ├── runner.py          ← 實盤 Runner（Polling 模式 LiveRunner）
│   ├── websocket_runner.py ← ⭐ 實盤 Runner（WebSocket 事件驅動）
│   ├── signal_generator.py ← 信號生成
│   ├── binance_futures_broker.py ← Binance 合約 Broker（含 Maker 優先下單）
│   ├── trading_db.py      ← ⭐ SQLite 交易資料庫
│   ├── kline_cache.py     ← ⭐ 增量 K 線快取
│   └── trading_state.py   ← 交易狀態持久化
├── data/
│   ├── funding_rate.py    ← Funding Rate 下載/對齊
│   ├── storage.py         ← Parquet 存取
│   └── ...多數據源客戶端
├── risk/                  ← 風險管理 (position sizing, Kelly, Monte Carlo)
├── monitor/               ← 健康檢查、通知、Telegram Bot
└── utils/                 ← 日誌、安全、時間工具
```

---

## 📚 文件索引

| 文件 | 行數 | 該看嗎？ | 內容 |
|------|:----:|:--------:|------|
| **CLI_REFERENCE.md** | ~320 | ⭐ **必看** | 你現在在看的這份（專案地圖） |
| **PROFESSIONAL_UPGRADE_PLAN.md** | 566 | ⭐ **必看** | 策略升級計畫 + 因子研究 + P1/P2/P3 詳情 |
| QUICK_START_GUIDE.md | ~3000 | 📖 查閱 | 完整教學（新手 → 部署 → FAQ），當百科全書查 |
| RISK_MANAGEMENT.md | — | 📖 查閱 | 風控詳細說明 |
| TRADING_STRATEGIES_REFERENCE.md | — | 📖 查閱 | 策略開發參考 |
| DATA_QUALITY.md | — | 📖 查閱 | 數據品質說明 |
| STRATEGY_PORTFOLIO.md | — | 📖 查閱 | 組合策略說明 |

---

## 📊 reports/ 輸出結構

```
reports/{market_type}/{strategy}/{run_type}/{timestamp}/
```

| run_type | 內容 |
|----------|------|
| `backtest/` | 回測報告 (stats, equity curve, trades CSV) |
| `portfolio/` | 組合回測 |
| `validation/` | 驗證報告 (walk_forward, cost_sensitivity) |
| `live/` | 交易狀態 + kline_cache + algo_orders_cache |

---

## 🚧 當前專案狀態 (2026-02-16)

### ✅ 已完成

| 項目 | 內容 | 狀態 |
|------|------|------|
| **Prompt 2** | Walk-Forward + DSR + CPCV 驗證框架 | ✅ 完成 |
| **Prompt 3** | 完整成本模型（Funding Rate + Volume Slippage + Sensitivity） | ✅ 完成 |
| **P1 方案 A** | 風控參數優化：SL 2.5→2.0, CD 5→3（熱力圖掃描） | ✅ 完成 |
| **P1 方案 B** | Funding Rate 過濾器（獨立因子，過濾擁擠交易） | ✅ 完成 |
| **P4** | Dynamic RSI（Rolling Percentile 自適應閾值，對抗 Alpha Decay） | ✅ 完成 |
| **執行優化** | Maker 優先下單（Taker 0.04% → Maker 0.02%，省一半手續費） | ✅ 完成 |
| **WebSocket** | 事件驅動 Runner（延遲 5min → <1s，Oracle Cloud 1GB RAM 可跑） | ✅ 完成 |
| **SQLite DB** | 結構化交易資料庫（trades / signals / daily_equity）+ CLI 查詢 | ✅ 完成 |
| **波動率過濾器** | ATR/Price < 0.005 時不開倉，過濾低波動磨耗 | ✅ 完成 |
| **HTF 軟趨勢過濾** | 4h EMA 連續權重（順趨勢 100% / 逆趨勢 50% / 無趨勢 75%） | ✅ 完成 |
| **波動率目標倉位** | `target_volatility: 1.00`（100% 年化波動率目標） | ✅ 完成 |
| **狀態機修復** | 平倉後不直接反手，強制回 Flat + cooldown 再入場 | ✅ 完成 |
| **Alpha Decay 監控** | Rolling IC + 年度 IC + Telegram 警報（`monitor_alpha_decay.py`） | ✅ 完成 |
| **P5 Ensemble** | RSI+MACD 組合策略（低相關 corr=0.15，Sharpe 提升） | ✅ 完成 |
| **P6 時間框架** | 15m / 4h 配置檔已建立，供研究用 | ✅ 完成 |
| **三幣組合** | 加入 SOLUSDT（低相關 corr=0.21，Sharpe +34%，MDD -39%） | ✅ 完成 |
| **驗證工具審計** | 10 個驗證工具全面審計 + Bug 修復（Bootstrap Sharpe、成本模型） | ✅ 完成 |

### 🔲 待做

| 項目 | 內容 | 優先級 | 說明 |
|------|------|:------:|------|
| **Ensemble 實盤** | 組合策略實盤驗證 | 🟡 中 | 需累積 Paper Trading 數據 |
| **15m/4h 回測** | 不同時間框架績效比較 | 🔵 低 | 配置已備，需下載對應數據 |

### ⚠️ 已知風險

- **Alpha 衰減**: RSI IC 從 2023 (+0.065) → 2026 (+0.018)，衰減 72%（已用 Dynamic RSI + IC 監控緩解）
- **因子假多樣化**: RSI/BB/MACD/OBV 相關 |r| > 0.5（本質同一因子，Ensemble 僅用低相關配對）
- 詳見 `PROFESSIONAL_UPGRADE_PLAN.md` 研究 A~F

---

## 🏗️ 當前 Oracle 部署配置

```
交易對:    BTCUSDT, ETHUSDT, SOLUSDT（三幣）
策略:      rsi_adx_atr v3.1（Dynamic RSI + Funding Filter + Vol Filter + HTF Soft）
倉位分配:  各 100%（總曝險 300%）
槓桿:      5x ISOLATED（保證金佔用 60%）
波動率目標: 100%（target_volatility: 1.00）
執行模式:  WebSocket 事件驅動（tmux session: trading）
熔斷線:    65%（歷史 MDD 38.3%，緩衝 26.7%）
```

---

## 🚀 Oracle Cloud 部署方式

### 方式 A：WebSocket 事件驅動（推薦，延遲 <1 秒）

```bash
# 1. 配置 Swap（1GB RAM 機器必備，只需跑一次）
bash scripts/setup_swap.sh

# 2. 用 tmux 啟動 WebSocket Runner
tmux new -d -s trading "cd ~/quant-binance-spot && source .venv/bin/activate && PYTHONPATH=src python scripts/run_websocket.py -c config/futures_rsi_adx_atr.yaml --real 2>&1 | tee logs/websocket.log"

# 3. （可選）設定 Alpha Decay 監控 cron
# crontab -e 加入：
# 0 1 * * 0 cd ~/quant-binance-spot && source .venv/bin/activate && bash scripts/cron_alpha_monitor.sh >> logs/alpha_monitor.log 2>&1

# 4. 查看 log
tmux attach -t trading       # 進入 tmux（Ctrl+B D 離開）
tail -50 logs/websocket.log  # 不進 tmux 也能看

# 5. 重啟
tmux kill-session -t trading
tmux new -d -s trading "..."  # 同上
```

### 方式 B：Cron 定時（傳統，延遲 ~5 分鐘）

```bash
# crontab -e
5 * * * * cd ~/quant-binance-spot && source .venv/bin/activate && python scripts/run_live.py -c config/futures_rsi_adx_atr.yaml --real --once >> logs/futures_live.log 2>&1
```

> ⚠️ **兩種方式不可同時使用**。用 WebSocket 時要把 cron 裡的 `run_live.py` 註解掉。

### 更新部署（加幣 / 改參數）

```bash
# 在本機改好 config，commit + push 後：
ssh -i ~/.ssh/oracle-trading-bot.key ubuntu@140.83.57.255
cd ~/quant-binance-spot
git stash && git pull   # stash 本地改動再拉

# 如果加了新幣，下載 K 線 + Funding Rate
source .venv/bin/activate
PYTHONPATH=src python scripts/download_data.py -c config/futures_rsi_adx_atr.yaml
PYTHONPATH=src python scripts/download_data.py -c config/futures_rsi_adx_atr.yaml --funding-rate

# 重啟 runner
tmux attach -t trading   # Ctrl+C 停舊的
PYTHONPATH=src python scripts/run_websocket.py -c config/futures_rsi_adx_atr.yaml --real
# Ctrl+B, d 離開（或直接關 SSH 視窗）
```

---

## 💡 快速提示

```bash
# 任何腳本的幫助
python scripts/<script>.py --help

# Oracle 部署後更新
ssh ubuntu@<IP>
cd ~/quant-binance-spot && git pull && ./scripts/setup_cron.sh --update

# 查看實盤 log
tail -100 /home/ubuntu/quant-binance-spot/logs/websocket.log    # WebSocket 模式
tail -100 /home/ubuntu/quant-binance-spot/logs/futures_live.log  # Cron 模式

# 查詢交易資料庫
python scripts/query_db.py -c config/futures_rsi_adx_atr.yaml summary
python scripts/query_db.py -c config/futures_rsi_adx_atr.yaml trades --limit 20
python scripts/query_db.py -c config/futures_rsi_adx_atr.yaml export  # 匯出 CSV

# 查看當前持倉
python -c "
from qtrade.live.binance_futures_broker import BinanceFuturesBroker
b = BinanceFuturesBroker(dry_run=True)
for p in b.get_positions():
    print(f'{p.symbol} [{p.position_side}]: qty={p.qty:+.6f} pnl=\${p.unrealized_pnl:+,.2f}')
"
```
