---
description: Data download commands, storage paths, cron jobs, watchdog configuration
globs:
alwaysApply: false
---
# Skill: Data Management

> Loaded by DevOps for data pipeline management on Oracle Cloud.

## Data Download Commands

```bash
source .venv/bin/activate

# HTF Filter v2 + LSR (klines + FR)
PYTHONPATH=src python scripts/download_data.py -c config/prod_candidate_htf_lsr.yaml
PYTHONPATH=src python scripts/download_data.py -c config/prod_candidate_htf_lsr.yaml --funding-rate
PYTHONPATH=src python scripts/download_data.py -c config/prod_candidate_htf_lsr.yaml --derivatives

# R3C (retained for rollback)
PYTHONPATH=src python scripts/download_data.py -c config/prod_live_R3C_E3.yaml

# OI (binance_vision + binance API, merge)
PYTHONPATH=src python scripts/download_oi_data.py --provider binance_vision --symbols BTCUSDT ETHUSDT SOLUSDT DOGEUSDT AVAXUSDT
PYTHONPATH=src python scripts/download_oi_data.py --provider binance --symbols BTCUSDT ETHUSDT SOLUSDT DOGEUSDT AVAXUSDT
```

## Data Storage Paths

```
data/binance/futures/1h/{SYMBOL}.parquet              ← Klines (primary TF)
data/binance/futures/4h/{SYMBOL}.parquet              ← Klines (auxiliary)
data/binance/futures/1d/{SYMBOL}.parquet              ← Klines (auxiliary)
data/binance/futures/funding_rate/{SYMBOL}.parquet     ← Funding Rate
data/binance/futures/open_interest/merged/{SYMBOL}.parquet      ← OI (merged)
data/binance/futures/open_interest/binance_vision/{SYMBOL}.parquet ← OI (vision)
data/binance/futures/open_interest/binance/{SYMBOL}.parquet      ← OI (API)
data/binance/futures/derivatives/lsr/{SYMBOL}.parquet             ← Long/Short Ratio
data/binance/futures/derivatives/top_lsr_account/{SYMBOL}.parquet ← Top Trader LSR (account)
data/binance/futures/derivatives/top_lsr_position/{SYMBOL}.parquet ← Top Trader LSR (position)
data/binance/futures/derivatives/taker_vol_ratio/{SYMBOL}.parquet ← Taker Buy/Sell Ratio
data/binance/futures/derivatives/cvd/{SYMBOL}.parquet             ← CVD
data/binance/futures/liquidation/{SYMBOL}.parquet                 ← Liquidation data
```

## Cron Jobs (Oracle Cloud, UTC timezone)

```
# HTF Filter v2 + LSR Kline + FR (every 6h)
10 */6 * * * download_data.py -c config/prod_candidate_htf_lsr.yaml

# R3C Kline + FR (retained for rollback, every 6h)
15 */6 * * * download_data.py -c config/prod_live_R3C_E3.yaml

# OI binance_vision (daily at 02:30 UTC)
30 2 * * * download_oi_data.py --provider binance_vision --symbols ...

# OI binance API (every 2h at :45)
45 */2 * * * download_oi_data.py --provider binance --symbols ...

# Derivatives (LSR + Taker Vol + CVD) — daily at 03:00 UTC
# ⚠️ LSR data is REQUIRED for prod_candidate_htf_lsr.yaml overlay
0 3 * * * download_data.py -c config/prod_candidate_htf_lsr.yaml --derivatives
```

## Multi-Runner Watchdog

Output directories are isolated by strategy name:
```
reports/live_watchdog/{strategy_name}/
  ├── latest_status.json
  ├── history.jsonl
  └── watchdog.pid
```

Check specific strategy watchdog:
```bash
cat reports/live_watchdog/R3C_E3/latest_status.json | python3 -m json.tool
```

## Data Management Responsibilities

- **Research stage**: Alpha Researcher explores new sources, initial download, coverage assessment
- **Backtest/validation**: Quant Developer downloads per config
- **Production/persistence (DevOps)**: Cron periodic updates, data quality monitoring, ensure new strategy data sources are in cron
