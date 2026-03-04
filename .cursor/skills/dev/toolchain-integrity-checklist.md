# Toolchain Integrity Checklist

> **Last updated**: 2026-02-28

在依賴任何工具的輸出做決策之前，先做以下快速健全性檢查。
本 checklist 的目標是防止「工具本身有 bug → 得出錯誤結論」的情況。

---

## 1. 使用前確認（Pre-Flight）

### 回測引擎 (`run_symbol_backtest`)
- [ ] `price=df['open']`（不是 close）
- [ ] `data_dir` 有傳入（否則 funding rate 會缺失，Sharpe 偏高）
- [ ] `cfg` 來自 `AppConfig.to_backtest_dict()`（不是手動建構）
- [ ] Futures 模式下 `funding_rate.enabled: true` 和 `slippage_model.enabled: true`

### 驗證工具 (`validate.py`)
- [ ] 所有子函數（WFA、MC、LOAO、Kelly、CPCV）都收到 `data_dir`
- [ ] PBO 使用 `combinatorial_purged_cv()`（不是已棄用的 `_simplified_pbo_estimate`）
- [ ] WFA splits 數量合理（建議 ≥ 5，數據 < 2000 bars 會自動縮減）

### 數據路徑
- [ ] 使用 `cfg.resolve_kline_path(symbol)` 或 `cfg.resolve_kline_paths()`
- [ ] **不要**手動拼接 `"klines"` 子目錄（歷史遺留，已統一為 interval 目錄）
- [ ] 確認 parquet 文件存在後再傳給回測

### 策略配置
- [ ] `meta_blend` 策略的 `auto_delay=False`（防止 double-delay）
- [ ] overlay 參數用 `deepcopy`（防止 cross-contamination）
- [ ] HTF resample 後有 `shift(1)` 或使用 `causal_resample_align()`

---

## 2. 使用後驗證（Post-Flight）

### 回測結果
- [ ] 檢查 `adjusted_stats` 而非 `stats`（adjusted 含 funding + slippage）
- [ ] 確認 trade count > 0（空結果可能是配置錯誤）
- [ ] Sharpe 超過 5.0 → 高度懷疑 bug（look-ahead、costs off）

### 驗證結果
- [ ] WFA 衰退率 > 80% → 可能過擬合，但也檢查數據是否足夠
- [ ] PBO 由 CPCV 產出 → 確認 `n_combinations` ≥ 20（否則統計意義不足）
- [ ] MC VaR 結果 → 確認是基於含成本的收益率

---

## 3. 常見陷阱速查

| 症狀 | 根因 | 修復 |
|------|------|------|
| Sharpe 突然變高 | `data_dir` 未傳入 → 無 funding rate | 確認 `data_dir=cfg.data_dir` |
| PBO 異常高（>50%） | 使用了簡化版 PBO | 改用 `combinatorial_purged_cv()` |
| 數據載入失敗 | 路徑中有 `"klines"` | 使用 `cfg.resolve_kline_path()` |
| WFA 結果全 NaN | 數據太短 | 減少 `n_splits` 或延長數據 |
| Overlay 效果消失 | overlay_params cross-contamination | `to_backtest_dict()` 已用 deepcopy |
| HTF filter 績效虛高 | resample 後沒 shift(1) | 使用 `causal_resample_align()` |

---

## 4. Config Mutation Prevention

### Dict 複製
- [ ] Config dict（含嵌套 overlay/funding_rate/slippage_model）使用 `copy.deepcopy()`
- [ ] **不要**用 `.copy()` 複製含嵌套結構的 dict（只複製第一層）
- [ ] 在迴圈中使用 config dict 時（如 WFA grid search、多 symbol 回測），每次迭代都要 deepcopy

### Params 傳遞
- [ ] 不用 `params.pop()` — 會修改 caller 的 dict
- [ ] 用 `params.get()` 讀取，需傳遞時建立新 dict
- [ ] `_data_dir` 等注入型 key 在複製後的 dict 上操作，不修改原始 dict

### Defaults 一致性
- [ ] `load_config()` 中 `.get(key)` 不帶第二個參數（讓 dataclass default 生效）
- [ ] 新增 config field 時，默認值只設在 dataclass definition 上

---

## 5. IC 監控 Pre-Flight（Alpha Decay）

> Governance: `.cursor/skills/validation/alpha-decay-governance.md`

### ic_monitor.py
- [ ] Forward return 使用 `pct_change(forward_bars).shift(-forward_bars)` — 這是 IC 品質量測，非 PnL
- [ ] Decay 公式有小分母保護：`|historical_ic| < min_ic_denominator` 時用絕對差
- [ ] `min_ic_denominator` 來自 `validation.yaml`，不硬編碼

### validation.yaml alpha_decay section
- [ ] `recent_ic_min` 不超過 0.02（TSMOM 結構性低 IC）
- [ ] `max_critical_alerts` 容許至少 1-2 個幣種暫時性問題
- [ ] 門檻改動需有 Researcher 校準依據（不能 Developer 自行定）

### 常見陷阱

| 症狀 | 根因 | 修復 |
|------|------|------|
| Decay +240%/+1500% | 小分母爆炸：historical IC 近 0 | `min_ic_denominator` guard |
| 全幣種 FAIL | `recent_ic_min` 太高 | 降至 0.005（TSMOM 常態） |
| Gate 形同虛設 | `max_critical_alerts` 太寬鬆 | 保持 <= 2，配合 3-layer gate |

---

## 6. 自動化檢查

以下測試會在 CI 中自動驗證上述規則：

- `tests/test_data_dir_propagation.py` — data_dir 在所有驗證路徑中傳播
- `tests/test_validation_pipeline.py` — DSR/PBO 基本正確性
- `tests/test_resample_shift_guard.py` — HTF resample 必須 shift(1)
- `tests/test_overlay_data_isolation.py` — overlay params 隔離
- `tests/test_code_safety_guard.py` — 5 類反模式自動掃描（淺複製、params.pop、signal_delay、silent except、直接 vbt 呼叫）
