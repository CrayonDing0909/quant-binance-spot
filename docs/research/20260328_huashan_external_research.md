---
tags: [research, signals, huashan, literature-review]
date: 2026-03-28
status: active
strategy: "[[huashan_convergence]]"
related:
  - "[[ALPHA_RESEARCH_MAP]]"
  - "[[STRATEGY_DEV_PLAYBOOK_R2_1]]"
  - "[[20260325_lsr_contrarian_research_alignment_proposal]]"
---

> **Last updated**: 2026-03-28

# 华山论剑策略 — 外部研究參考文獻整理

> 相關策略: `src/qtrade/strategy/huashan_convergence_strategy.py`
> 相關 config: `config/research_huashan_convergence.yaml`
> 來源頻道: [[华山论剑]] (@huashanlunjians)

## 1. Coinbase Premium Index (CPI)

| 來源 | 發現 |
|---|---|
| CryptoQuant BTC CPI | SMA-50 平滑；z-score entry ±2.0, stop ±3.5, exit ±0.5 |
| TradingView BIGTAKER | 4h timeframe 最佳；US session filter (09:30-16:00 EST) 提升品質 |
| CoinTelegraph 2026 | -167.8 bps = 機構需求反轉；連續 3+ 天正值 after 負值 = reversal signal |
| 實務共識 | 60 天 z-score lookback；regime indicator 而非 standalone trigger |

## 2. Funding Rate

| 來源 | 發現 |
|---|---|
| He et al. (arXiv:2212.06888) | Carry SR=1.8(散戶成本)~3.5(做市商成本) |
| Schmeling et al. 2023 (arXiv:2510.14435) | Carry SR=6.45(2020-24)→4.06(2024+)→**2025 負值**，carry 已擁擠 |
| BIS WP 1087 | 平均 carry 6-8% p.a.，常超過 20% |
| ScienceDirect 2025 | 跨所套利 6 個月 115.9% return, MDD 1.92%, HODL corr=0 |
| 量化共識 | **Contrarian 更持久**：z-score ±2.0 entry; FR >0.10%/8h = extreme |

**關鍵門檻**:
- +0.01%~+0.05%/8h = bullish bias, 可能 long crowding
- < -0.03%/8h = short crowding, reversal watch
- > +0.10%/8h (~100% ann.) = extreme long leverage, 高清算風險
- < -0.10%/8h = extreme short squeeze risk

## 3. Open Interest (OI)

| 來源 | 發現 |
|---|---|
| Gate.com 2026 | 4-quadrant (Price dir × OI dir) 60-70% accuracy |
| Glassnode LPOC | OI trend × Price trend composite; 90th-pctile crossing 0.01 = systemic risk |
| Amberdata | OI 下降 20-30% from peak = cascade 完成, contrarian 進場 |
| Navnoor Bawa 2025 | Oct 2025: $19B OI 在 36h 蒸發; WLFI volume 提前 5h spike |
| 量化共識 | OI Z-score > 2.0 = extreme; **50-bar lookback**; 與 FR 組合最佳 |

**4-Quadrant 框架**:
```
Price ↑ + OI ↑ = 強勢上漲確認 (新多頭進場)       → bullish
Price ↑ + OI ↓ = 空頭回補 (非新需求)             → neutral/cautious
Price ↓ + OI ↑ = 強勢下跌 (新空頭進場)           → bearish
Price ↓ + OI ↓ = 多頭清算/恐慌平倉              → contrarian bullish
```

## 4. Long/Short Ratio (LSR)

| 來源 | 發現 |
|---|---|
| AInvest 2025 | >70% long = extreme bullish crowding; <40% long = extreme bearish |
| CoinGlass | LSR 6.03(bullish extreme) 和 0.44(bearish extreme) preceded 20% corrections/rebounds |
| Gate.com | Top Trader Account LSR > retail LSR 更有效 |
| CryptoQuant CEO Ki Young Ju | 機構避險使 LSR 失真; realized metrics (MVRV) 在 BTC 上準確率 82% |
| 量化共識 | **Multi-σ extremes 才有效**；5min~4h snapshots；altcoin > BTC |

## 5. Stablecoin Flow

| 來源 | 發現 |
|---|---|
| BDC Consulting | USDT supply model: **229% ROI** vs buy-and-hold |
| arXiv:2603.23480 (2026.03) | GARCH-Copula-XGBoost: stablecoin upside vol + volume 改善 BTC vol forecast |
| CryptoQuant 2026.03 | $2.2B USDT single-day inflow → BTC breakout above $74K |
| CryptoQuant CEO 2025 | "新 BTC 流動性走 MSTR/ETF，非穩定幣" → **BTC 信號衰退** |
| 量化共識 | Leading indicator lag = **days to weeks**；bot 占 77-80% 需過濾 |

## 6. Multi-Factor Composite

| 來源 | 方法 | SR |
|---|---|---|
| ACM ICGAIB 2025 | IC-weighted Z-scores of 4-6 factors, z ±1.0 entry | **2.5** |
| Unravel Finance | Cross-sectional 6-factor, inverse vol weighting | **~2.5** |
| CF Benchmarks | Sentiment-gated basis (MACD z + 90d return z → CDF) | **1.52** |
| 跨研究共識 | **4-6 個不相關因子 arithmetic average 是 robust baseline**；>6 個邊際遞減 |

## 7. 本 Codebase 已驗證的優化模式

| 模式 | 證據 | 效果 |
|---|---|---|
| Midpoint TP | LSR v3 Cycle 2 | MDD -30.68% → -19.56% (11pp 改善) |
| ATR 3.0 SL | LSR v3 sweep 2.0-4.0 | P5 loss protection 最佳平衡 |
| 168h Time Stop | LSR v3 | 防止 zombie holding |
| ADX > 25 gate | LSR v3 | 過濾震盪假信號 |
| Persist ≥ 2 bars | LSR v3 Cycle 1 | 減少 single-bar spike noise |
| Cooldown 24h | LSR v3 | 防止止損後立即重入場 |
| EW pctrank (hl=84) | LSR v3 | 2025-2026 IC 更佳 |
| **Anti-pattern: gate stacking** | OI+HTF, Macro+HTF | **每次疊加都退化 -5%~-10%** |

## 參考文獻完整列表

1. He, Manela, Ross & von Wachter — "Fundamentals of Perpetual Futures" (arXiv:2212.06888)
2. Schmeling, Schrimpf & Todorov 2023 — Crypto carry (cited in arXiv:2510.14435)
3. BIS Working Paper No. 1087 — Crypto derivatives market structure
4. ScienceDirect 2025 — Funding rate arbitrage across exchanges
5. arXiv:2603.23480 (2026.03) — GARCH-Copula-XGBoost stablecoin volatility forecasting
6. BDC Consulting — USDT supply as BTC price predictor
7. ACM ICGAIB 2025 — Multi-factor ETH strategy (IC-weighted Z-scores)
8. Unravel Finance — Cross-sectional alpha factors in crypto
9. CF Benchmarks — Sentiment-gated Bitcoin basis strategy
10. Glassnode — Leverage Position Openings and Closures (LPOC)
11. CryptoQuant — Coinbase Premium Index, stablecoin exchange flows
12. Gate.com — Derivatives market signals guide (2026)
13. Amberdata — Liquidation cascade anticipation
