# Data & Strategy Catalog

> **Auto-generated**: 2026-02-28 by `scripts/gen_data_strategy_catalog.py`
> **Do NOT edit by hand** — regenerate after adding data modules or strategies.
>
> ```bash
> PYTHONPATH=src python scripts/gen_data_strategy_catalog.py
> ```

---

## Data Modules

| Category | Module | Description | Key Exports |
|----------|--------|-------------|-------------|
| **K 線** | `binance_client.py` | (no description) | BinanceHTTP |
| **K 線** | `binance_futures_client.py` | Binance Futures HTTP Client | BinanceFuturesHTTP |
| **K 線** | `binance_vision.py` | Binance Data Vision - 官方歷史數據批量下載 | `download_binance_vision_klines()`, `get_available_symbols()`, `check_data_availability()` |
| **K 線** | `ccxt_client.py` | CCXT 多交易所數據源 - 統一 API 訪問 100+ 交易所 | `convert_symbol()`, `list_available_exchanges()`, `fetch_ccxt_klines()`, `check_exchange_symbol()`, `get_earliest_data_timestamp()` |
| **K 線** | `klines.py` | (no description) | `fetch_klines()` |
| **K 線** | `storage.py` | (no description) | `save_klines()`, `load_klines()`, `get_local_data_range()`, `merge_klines()` |
| **K 線** | `yfinance_client.py` | Yahoo Finance 數據源 - 提供長期歷史加密貨幣數據 | `convert_symbol()`, `fetch_yfinance_klines()`, `get_yfinance_data_range()` |
| **衍生品** | `_derivatives_common.py` | 衍生品數據通用下載/載入模組 | `fetch_vision_single_metric()`, `fetch_api_single_metric()` |
| **衍生品** | `funding_rate.py` | Binance Futures Funding Rate 歷史資料下載與快取 | `download_funding_rates()`, `save_funding_rates()`, `load_funding_rates()`, `get_funding_rate_path()`, `align_funding_to_klines()` |
| **衍生品** | `liquidation.py` | 清算/爆倉數據模組 | `load_liquidation()`, `get_liquidation_path()`, `align_liquidation_to_klines()` |
| **衍生品** | `long_short_ratio.py` | Long/Short Ratio 數據模組 | `download_lsr()`, `save_lsr()`, `load_lsr()`, `get_lsr_path()`, `align_lsr_to_klines()`, ... |
| **衍生品** | `open_interest.py` | Open Interest (OI) 歷史資料下載、快取與對齊 | OIProvider, BinanceOIProvider, BinanceVisionOIProvider, CoinglassOIProvider, `get_oi_provider()`, ... |
| **衍生品** | `taker_volume.py` | Taker Buy/Sell Volume 數據模組 — 包含 CVD 衍生計算 | `download_taker_volume()`, `compute_cvd()`, `save_taker_volume()`, `save_cvd()`, `load_taker_volume()`, ... |
| **鏈上** | `onchain.py` | 鏈上數據模組 | `save_onchain()`, `load_onchain()`, `get_onchain_path()`, `align_onchain_to_klines()`, `compute_onchain_coverage()` |
| **即時** | `order_book.py` | Order Book Depth 數據模組（Phase 4C） | OrderBookSnapshot, `compute_imbalance()`, `compute_depth_profile()`, OrderBookCache, `parse_depth_message()` |
| **工具** | `multi_tf_loader.py` | 多時間框架數據載入器 (Multi-TF Loader) | MultiTFLoader |
| **工具** | `quality.py` | 數據質量檢查模組 | DataQualityIssue, DataQualityReport, DataQualityChecker, `validate_data_quality()`, `clean_data()` |

---

## Registered Strategies

| Status | Strategy Name | File | Description |
|--------|---------------|------|-------------|
| 🟢 生產中 | `oi_liq_bounce` | `oi_liq_bounce_strategy.py` | OI Liquidation Bounce 策略 (v4.2) |
| 🟡 候選/Paper | `breakout_vol_atr` | `breakout_vol_strategy.py` | Breakout + Volatility Expansion 策略 |
| 🟡 候選/Paper | `meta_blend` | `meta_blend_strategy.py` | Meta-Blend Strategy — 多策略信號混合器 |
| 🟡 候選/Paper | `tsmom_ema` | `tsmom_strategy.py` | TSMOM（Time-Series Momentum）策略 |
| 🟡 候選/Paper | `tsmom_multi_ema` | `tsmom_strategy.py` | TSMOM（Time-Series Momentum）策略 |
| ⚪ 已實作 | `breakout_vol` | `breakout_vol_strategy.py` | Breakout + Volatility Expansion 策略 |
| ⚪ 已實作 | `crowding_contrarian` | `derivatives_enhanced_strategy.py` | 衍生品增強策略 — Derivatives-Enhanced Strategies |
| ⚪ 已實作 | `cvd_divergence` | `derivatives_enhanced_strategy.py` | 衍生品增強策略 — Derivatives-Enhanced Strategies |
| ⚪ 已實作 | `funding_carry` | `funding_carry_strategy.py` | Funding Rate Carry 策略 |
| ⚪ 已實作 | `funding_carry_xs` | `funding_carry_strategy.py` | Funding Rate Carry 策略 |
| ⚪ 已實作 | `liq_cascade_v2` | `derivatives_enhanced_strategy.py` | 衍生品增強策略 — Derivatives-Enhanced Strategies |
| ⚪ 已實作 | `low_freq_portfolio` | `low_freq_portfolio_strategy.py` | Low-Frequency Portfolio Layer — 日/週級別的組合管理策略 |
| ⚪ 已實作 | `lsr_contrarian` | `lsr_contrarian_strategy.py` | LSR Contrarian 策略 — 散戶多空比極端值逆向交易 |
| ⚪ 已實作 | `mean_revert_liquidity` | `mean_revert_liquidity.py` | Mean Reversion — Liquidity Sweep Strategy |
| ⚪ 已實作 | `mr_bollinger` | `mr_microstructure_strategy.py` | Mean Reversion 微觀結構策略（研究版 — Phase MR-1） |
| ⚪ 已實作 | `mr_rsi_htf` | `mr_microstructure_strategy.py` | Mean Reversion 微觀結構策略（研究版 — Phase MR-1） |
| ⚪ 已實作 | `mr_zscore` | `mr_microstructure_strategy.py` | Mean Reversion 微觀結構策略（研究版 — Phase MR-1） |
| ⚪ 已實作 | `multi_tf_resonance` | `multi_tf_resonance_strategy.py` | Multi-TF Resonance — 多時間框架共振策略 |
| ⚪ 已實作 | `nw_envelope_regime` | `nw_envelope_regime_strategy.py` | NW Envelope + Regime Filter + MTF Gating 策略（Futures 多空） |
| ⚪ 已實作 | `nwkl` | `nwkl_strategy.py` | NWKL v3.1（Nadaraya-Watson Kernel Regression + Lorentzian Distance k-NN Classifier） |
| ⚪ 已實作 | `oi_bb_rv` | `oi_bb_rv_strategy.py` | OI + Bollinger Band Breakout + Realized Volatility Filter (OI-BB-RV) |
| ⚪ 已實作 | `rsi_adx_atr` | `rsi_adx_atr_strategy.py` | RSI + ADX + ATR 組合策略 |
| ⚪ 已實作 | `rsi_adx_atr_trailing` | `rsi_adx_atr_strategy.py` | RSI + ADX + ATR 組合策略 |
| ⚪ 已實作 | `tsmom` | `tsmom_strategy.py` | TSMOM（Time-Series Momentum）策略 |
| ⚪ 已實作 | `tsmom_carry_v2` | `tsmom_carry_v2_strategy.py` | TSMOM + Carry V2 — Per-Symbol 多因子複合策略 |
| ⚪ 已實作 | `tsmom_mr_composite` | `tsmom_mr_composite_strategy.py` | TSMOM + Mean-Reversion Composite 策略 |
| ⚪ 已實作 | `tsmom_multi` | `tsmom_strategy.py` | TSMOM（Time-Series Momentum）策略 |
| ⚪ 已實作 | `x_model_weekend` | `x_model_weekend.py` | X-Model — ERL→IRL Weekend Liquidity Sweep Strategy v8 (ICT/SMC) |
| ⚪ 已實作 | `xsmom` | `xsmom_strategy.py` | XSMOM（Cross-Sectional Momentum）策略 — 改進版 |
| ⚪ 已實作 | `xsmom_tsmom` | `xsmom_strategy.py` | XSMOM（Cross-Sectional Momentum）策略 — 改進版 |

> **Total**: 30 registered strategies

---

## Download Scripts

| Script | Data Type | Key Flags |
|--------|-----------|-----------|
| `download_data.py` | K 線 / FR / OI | `-c`, `--interval`, `--funding-rate`, `--oi`, `--derivatives` |
| `download_oi_data.py` | Open Interest | `--symbols`, `--provider` |
| `fetch_derivatives_data.py` | LSR / Taker Vol / CVD | `--symbols`, `--metrics`, `--source`, `--coverage` |
| `fetch_liquidation_data.py` | 清算數據 | `--symbols`, `--source` |
| `fetch_onchain_data.py` | 鏈上 (DeFi Llama) | `--source`, `--stablecoins`, `--yields`, `--protocols` |

---

## Data Storage Paths

```
data/
├── binance/
│   ├── futures/1h/              ← 主力 K 線（生產用）
│   ├── futures/5m/              ← 微結構研究用
│   ├── futures/15m/             ← overlay 研究用
│   ├── futures/4h/              ← HTF 趨勢用
│   ├── futures/1d/              ← 日線 regime 用
│   ├── futures/open_interest/   ← OI（binance/coinglass/merged）
│   ├── futures/derivatives/     ← 衍生品指標
│   │   ├── lsr/                 ← Long/Short Ratio
│   │   ├── top_lsr_account/     ← 大戶 L/S (帳戶)
│   │   ├── top_lsr_position/    ← 大戶 L/S (持倉)
│   │   ├── taker_vol_ratio/     ← Taker Buy/Sell Ratio
│   │   └── cvd/                 ← Cumulative Volume Delta
│   └── futures/liquidation/     ← 清算數據
├── onchain/
│   └── defillama/               ← DeFi Llama 鏈上數據
└── ...
```
