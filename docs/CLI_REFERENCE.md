# Project Map & CLI Reference

> **Auto-generated**: 2026-03-25 by `scripts/gen_cli_reference.py`
> **Production config**: `config/prod_live_R3C_E3.yaml`
> **Strategy template**: `config/futures_tsmom.yaml` (TSMOM EMA base definition)
>
> Re-generate: `PYTHONPATH=src python scripts/gen_cli_reference.py`

---

## Scripts

### Core Workflow (by typical execution order)

| Script | Description |
|--------|-------------|
| `download_data.py` | 多數據源 K 線數據下載工具 |
| `download_oi_data.py` | Download OI historical data |
| `run_backtest.py` | 運行策略回測 |
| `run_portfolio_backtest.py` | 組合回測（v3.0 — 統一成本模型 + Ensemble 支援） |
| `run_walk_forward.py` | Run Walk-Forward Analysis |
| `run_cpcv.py` | CPCV 驗證 (López de Prado) |
| `validate.py` | 統一策略驗證工具 |
| `validate_live_consistency.py` | 回測↔實盤一致性驗證（Pre-Deploy Checklist） |
| `prod_launch_guard.py` | Production Launch Guard |
| `run_live.py` | 即時交易 |
| `run_websocket.py` | WebSocket Live Trading Bot |

### Optimization & Research

| Script | Description |
|--------|-------------|
| `optimize_params.py` | 🧬 Hyperopt Parameter Optimizer v2 |
| `run_hyperopt.py` | Hyperopt Parameter Optimization |
| `run_experiment_matrix.py` | NW Strategy Experiment Matrix |
| `run_funding_basis_research.py` | Funding / Basis Alpha Research — Phase 1 |
| `run_mr_research.py` | MR Research Matrix — Phase MR-1 |
| `run_oi_bb_rv_research.py` | OI-BB-RV Research Pipeline |
| `validate_overlay_falsification.py` | R2.1 Vol-Only Overlay Falsification Validator |
| `build_universe.py` | Build Universe for R3 Track A |
| `run_symbol_governance_review.py` | Run Symbol Governance Weekly Review |

### Operations & Monitoring

| Script | Description |
|--------|-------------|
| `run_telegram_bot.py` | Telegram Bot 統一常駐服務（支援多策略） |
| `query_db.py` | 交易資料庫查詢工具 |
| `health_check.py` | 系統健康檢查 |
| `daily_report.py` | Paper Trading 每日績效報表 |
| `prod_report.py` | Production Report Generator |
| `monitor_alpha_decay.py` | Alpha Decay 監控 |
| `risk_guard.py` | Risk Guard — automated risk monitoring & kill switch |

### Infrastructure

| Script | Description |
|--------|-------------|
| `deploy_oracle.sh` | Oracle Cloud 一鍵部署腳本 |
| `setup_cron.sh` | Quant Trading Bot - Cron Jobs 自動設定腳本 |
| `setup_swap.sh` | 自動建立 Swap (虛擬記憶體) 腳本 |

### Other

| Script | Description |
|--------|-------------|
| `analyze_factor_geometry.py` | 因子幾何審計工具 — 偵測因子冗餘與潛在重疊 |
| `cleanup_data.py` | 數據清理工具 — 釋放磁碟空間 |
| `compare_strategies.py` | 策略組合比較工具 — 邊際 Sharpe 分析 + 最佳權重配置 |
| `compute_governance_metrics.py` | Compute governance metrics from trading.db |
| `download_aggtrades_data.py` | Download aggTrades from Binance Vision and compute VPIN/CVD/OFI metrics |
| `fetch_derivatives_data.py` | Binance 衍生品數據下載工具（LSR, Taker Vol, CVD） |
| `fetch_liquidation_data.py` | 清算/爆倉數據下載工具 |
| `fetch_onchain_data.py` | 鏈上數據探索工具（DeFi Llama / CryptoQuant / Glassnode） |
| `gen_data_strategy_catalog.py` | Auto-generate docs/DATA_STRATEGY_CATALOG.md by scanning src/qtrade/data/ and strategy/. |
| `generate_blend_config.py` | 從比較報告或手動指定生成 meta_blend YAML 配置 |
| `research_avg_trade_size_ablation.py` | avg_trade_size Overlay/Filter Ablation — Quant Developer (#22 Handoff) |
| `research_forced_deleveraging_reversal.py` | Forced deleveraging reversal EDA |
| `research_low_freq_portfolio_layer.py` | Evaluate low_freq_portfolio as a portfolio exposure layer over the frozen production baseline. |
| `research_tick_ofi_eda.py` | Tick-level OFI EDA — Alpha Research #22 |
| `scan_active_buy_candidates.py` | Scan weekly active-buy candidates and export a contract for a separate app repo. |
| `trade_review.py` | 交易復盤工具 — 診斷信號與執行差異 |
| `verify_signal_replay.py` | 信號重播驗證 — 比對回測信號與實盤 SQLite 記錄 |
| `cron_alpha_decay.sh` | Alpha Decay 自動化監控 — 每週 cron 執行 |

> **Archive**: 38 completed research/migration scripts in `scripts/archive/`. These are preserved for reference but no longer part of the active workflow.

---

## Configs

### Production (active on Oracle Cloud)

| Config | File |
|--------|------|
| `prod_candidate_R3C_universe.yaml` | `config/prod_candidate_R3C_universe.yaml` |
| `prod_candidate_htf_filter.yaml` | `config/prod_candidate_htf_filter.yaml` |
| `prod_candidate_htf_lsr.yaml` | `config/prod_candidate_htf_lsr.yaml` |
| `prod_candidate_meta_blend.yaml` | `config/prod_candidate_meta_blend.yaml` |
| `prod_candidate_simplified.yaml` | `config/prod_candidate_simplified.yaml` |
| `prod_live_R3C_E3.yaml` | `config/prod_live_R3C_E3.yaml` |
| `prod_live_oi_liq_bounce.yaml` | `config/prod_live_oi_liq_bounce.yaml` |
| `prod_scale_rules_R3C_universe.yaml` | `config/prod_scale_rules_R3C_universe.yaml` |
| `risk_guard_alt_ensemble.yaml` | `config/risk_guard_alt_ensemble.yaml` |

### Strategy Definitions

| Config | File |
|--------|------|
| `futures_alt_ensemble.yaml` | `config/futures_alt_ensemble.yaml` |
| `futures_breakout_vol.yaml` | `config/futures_breakout_vol.yaml` |
| `futures_ensemble_nw_tsmom.yaml` | `config/futures_ensemble_nw_tsmom.yaml` |
| `futures_funding_carry.yaml` | `config/futures_funding_carry.yaml` |
| `futures_multi_strat_ensemble.yaml` | `config/futures_multi_strat_ensemble.yaml` |
| `futures_nwkl.yaml` | `config/futures_nwkl.yaml` |
| `futures_tsmom.yaml` | `config/futures_tsmom.yaml` |

### Research (active experiments)

| Config | File |
|--------|------|
| `research_active_buy_scan.yaml` | `config/research_active_buy_scan.yaml` |
| `research_low_freq_portfolio_layer.yaml` | `config/research_low_freq_portfolio_layer.yaml` |
| `research_lsr_contrarian_v3.yaml` | `config/research_lsr_contrarian_v3.yaml` |
| `research_macd_rsi_htf_spot_entry.yaml` | `config/research_macd_rsi_htf_spot_entry.yaml` |

### Utility

| Config | File |
|--------|------|
| `validation.yaml` | `config/validation.yaml` |

### Other

| Config | File |
|--------|------|
| `dual_momentum.yaml` | `config/dual_momentum.yaml` |

> **Archive**: 85 deprecated/completed research configs in `config/archive/`. Preserved for git history reference.

---

## Source Module Map

```
src/qtrade/
├── config.py              ← AppConfig dataclass, load_config()
├── strategy/              ← Strategy implementations
│   ├── base.py            ← StrategyContext (market_type, direction, signal_delay)
│   ├── tsmom_strategy.py  ← Active production strategy (TSMOM EMA)
│   └── exit_rules.py      ← SL/TP/Adaptive SL
├── backtest/
│   ├── run_backtest.py    ← BacktestResult dataclass, run_symbol_backtest()
│   ├── costs.py           ← Funding Rate + Volume Slippage cost model
│   └── metrics.py         ← Performance metrics + Long/Short analysis
├── live/
│   ├── base_runner.py     ← BaseRunner ABC (14 shared safety mechanisms)
│   ├── runner.py          ← LiveRunner (Polling mode)
│   ├── websocket_runner.py← WebSocketRunner (Event-driven, recommended)
│   └── signal_generator.py← SignalResult dataclass
├── validation/            ← WFA, DSR, PBO, CPCV, IC monitor
├── data/                  ← Multi-source: Binance/yfinance/ccxt
├── risk/                  ← Position sizing, Kelly, Monte Carlo
├── monitor/               ← Health check, Telegram, notifier
└── utils/                 ← Logging, security, time tools
```

---

## Current Production

```yaml
config: prod_live_R3C_E3.yaml
strategy: tsmom_ema
symbols: [BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT, DOGEUSDT, ADAUSDT, AVAXUSDT, LINKUSDT, LTCUSDT] (10 coins)
interval: 1h
leverage: 3x ISOLATED
direction: both
```

---

## Quick Commands

```bash
# Backtest (production config)
PYTHONPATH=src python scripts/run_backtest.py -c config/prod_live_R3C_E3.yaml

# Walk-Forward validation
PYTHONPATH=src python scripts/run_walk_forward.py -c config/prod_live_R3C_E3.yaml --splits 6

# Full validation pipeline
PYTHONPATH=src python scripts/validate.py -c config/prod_live_R3C_E3.yaml --quick

# Download data
PYTHONPATH=src python scripts/download_data.py -c config/prod_live_R3C_E3.yaml

# Live trading (WebSocket, recommended)
PYTHONPATH=src python scripts/run_websocket.py -c config/prod_live_R3C_E3.yaml --real

# Dry-run test
PYTHONPATH=src python scripts/run_live.py -c config/prod_live_R3C_E3.yaml --real --dry-run --once

# Query trading DB
PYTHONPATH=src python scripts/query_db.py -c config/prod_live_R3C_E3.yaml summary

# Re-generate this file
PYTHONPATH=src python scripts/gen_cli_reference.py
```

---

## Documentation Index

| Doc | Description |
|-----|-------------|
| [`ACTIVE_BUY_APP_MVP.md`](docs/ACTIVE_BUY_APP_MVP.md) | > **Last updated**: 2026-03-12 |
| [`ACTIVE_BUY_ARCHITECTURE.md`](docs/ACTIVE_BUY_ARCHITECTURE.md) | > **Last updated**: 2026-03-12 |
| [`ALPHA_RESEARCH_MAP.md`](docs/ALPHA_RESEARCH_MAP.md) | > **Last updated**: 2026-03-25 (LSR contrarian standalone revisit ALL 3 CYCLES COMPLETE. Cycle 3 portfolio_role: long-only satellite chosen (net exp +0.645%, SR 1.52, corr=0.256 to prod). Symmetric standalone is robustness fallback. HANDOFF READY → @quant-developer for v3 implementation) |
| [`CLI_REFERENCE.md`](docs/CLI_REFERENCE.md) | Project Map & CLI Reference |
| [`CURSOR_WORKFLOW.md`](docs/CURSOR_WORKFLOW.md) | Cursor Agent 工作流指南 |
| [`DATA_STRATEGY_CATALOG.md`](docs/DATA_STRATEGY_CATALOG.md) | Data & Strategy Catalog |
| [`ORCHESTRATION_MVP.md`](docs/ORCHESTRATION_MVP.md) | Research Orchestration MVP |
| [`R3C_STRATEGY_OVERVIEW.md`](docs/R3C_STRATEGY_OVERVIEW.md) | R3C Universe 策略總覽（小白友善版） |
| [`RESEARCH_LITERATURE.md`](docs/RESEARCH_LITERATURE.md) | > **Last updated**: 2026-03-02 |
| [`STRATEGY_DEV_PLAYBOOK_R2_1.md`](docs/STRATEGY_DEV_PLAYBOOK_R2_1.md) | Strategy Development Playbook (R2.1) |
| [`STRATEGY_PORTFOLIO_GOVERNANCE.md`](docs/STRATEGY_PORTFOLIO_GOVERNANCE.md) | 策略組合治理規範 (Strategy Portfolio Governance) |

> **Archive**: 5 historical docs in `docs/archive/`.
