# 策略組合治理規範 (Strategy Portfolio Governance)

> **Last updated**: 2026-02-25 (審查排程分工更新)

本文件定義多策略組合的治理規則：新策略如何納入、何時替換、如何決定 meta_blend vs. 獨立 Runner。
與 `R3C_SYMBOL_GOVERNANCE_SPEC.md`（幣種層級治理）互補，本文件聚焦**策略層級**治理。

---

## 1) 核心原則

1. **邊際貢獻優先**：新策略必須提升組合 Sharpe，而非只看單策略績效
2. **低相關性 > 高 Sharpe**：一個 SR=1.0 但與現有組合相關性 0.05 的策略，比 SR=2.0 但相關性 0.7 的策略更有價值
3. **漸進式納入**：新策略先 paper trading，再小倉位，最後全倉
4. **可逆性**：退場流程必須可逆，策略可重新啟用
5. **數據驅動**：所有決策必須附帶量化證據，禁止「感覺不錯就上」

---

## 2) 策略納入標準（Addition Criteria）

新策略必須**全部通過**以下門檻才能納入組合：

| # | 門檻 | 閾值 | 說明 |
|---|------|------|------|
| A1 | 驗證棧全通過 | DSR, WFA, MC, Risk Audit 全 PASS | 基本品質保證 |
| A2 | 策略相關性 | 與現有組合 corr < 0.30 | 確保真正分散化 |
| A3 | 邊際 Sharpe | 加入後組合 SR 提升 > 0 | 有正面貢獻 |
| A4 | 交易樣本量 | 每幣種 >= 30 筆交易 | 統計顯著性 |
| A5 | 年度穩定性 | 無任何年份虧損 > -5% | 避免極端集中風險 |

### 驗證流程

```
scripts/compare_strategies.py
  --existing <現有組合 config>
  --candidate <新策略 config>
  →  輸出: 相關性矩陣、邊際 SR、最佳權重、冗餘警告
  →  判定: ADD / SKIP / NEED_MORE_WORK
```

---

## 3) 策略替換標準（Replacement Criteria）

任一條件觸發 → 啟動審查流程：

| # | 觸發條件 | 閾值 | 說明 |
|---|---------|------|------|
| R1 | 實盤 Alpha 衰退 | 滾動 13 週 SR < 0，連續 2 次 | 策略失效跡象 |
| R2 | 實盤/回測偏離 | live_SR / backtest_SR < 0.30 | 過擬合或市場結構改變 |
| R3 | 新策略全面優於 | 相同幣種，新策略 SR 更高且 corr < 0.3 | 升級替換 |
| R4 | 成本惡化 | 實際滑點 / 模型滑點 > 1.5 持續 4 週 | 執行環境惡化 |

### 審查流程

1. 觸發條件觸發 → Risk Manager 發起審查
2. Quant Researcher 重跑驗證（含最近數據）
3. 決策：`KEEP` / `REDUCE_WEIGHT` / `RETIRE`

---

## 4) 策略生命週期（State Machine）

```
                    ┌──────────────────────────────────────┐
                    │                                      │
                    ▼                                      │
  research → candidate → paper_trading → active → reduced → retired → archived
                                            │        ▲
                                            │        │
                                            └────────┘
                                           (recovery)
```

| 狀態 | 權重乘數 | 說明 |
|------|---------|------|
| `research` | 0.00 | 開發中，未通過驗證 |
| `candidate` | 0.00 | 通過驗證，待 paper trading |
| `paper_trading` | 0.00 | Paper 模式，不影響實際資金 |
| `active` | 1.00 | 正式上線 |
| `reduced` | 0.50 | 觸發 R1/R4，降低權重觀察 |
| `retired` | 0.00 | 確認失效，停止交易 |
| `archived` | — | Config 移至 `config/archive/` |

### 狀態轉換條件

- `candidate → paper_trading`：Risk Manager APPROVED
- `paper_trading → active`：Paper 期 >= 14 天 + 無異常
- `active → reduced`：觸發 R1 或 R4
- `reduced → active`：滾動 4 週 SR > 0.5 + 執行一致性恢復
- `reduced → retired`：reduced 狀態持續 8 週且未恢復
- `retired → archived`：確認不再需要，移至 archive

---

## 5) meta_blend vs. 獨立 Runner 決策

### 決策流程圖

```
新策略通過風控審計
    │
    ├── 跟現有策略交易相同幣種？
    │       │
    │       ├── 是 → 方向相同 (both vs long_only)？
    │       │       ├── 是 → 加入 meta_blend（信號層級混合）
    │       │       └── 不同 → 獨立 Runner（需子帳號或 HEDGE_MODE）
    │       │
    │       └── 否 → 策略收益率相關性 < 0.3？
    │               ├── 是 → 獨立 Runner（最大化分散效益）
    │               └── 否 → 加入 meta_blend（避免冗餘曝險）
```

### 技術約束

| 模式 | 條件 | 帳號需求 |
|------|------|---------|
| **meta_blend** | 同帳號、ONE_WAY mode | 單一帳號即可 |
| **獨立 Runner** | 不同方向或完全不同幣種 | 需 Binance 子帳號或 HEDGE_MODE |

### 當前策略分類

| 策略 | 方向 | 重疊幣種 | 建議模式 |
|------|------|---------|---------|
| R3C | both (多空) | — | 基準 |
| meta_blend | both (多空) | 8/10 與 R3C 重疊 | **替換** R3C |
| oi_liq_bounce | long_only | 5 個與 R3C 重疊 | **獨立 Runner**（子帳號）或 **meta_blend 包裝** |

---

## 6) 組合權重配置

### 策略層級權重（跨策略配置）

當多個策略同時運行時，需配置「策略層級權重」：

```yaml
# 範例：策略組合配置
strategy_portfolio:
  strategies:
    - name: "meta_blend"
      config: "config/prod_live_meta_blend.yaml"
      weight: 0.70       # 佔總資金 70%
      account: "main"
    - name: "oi_liq_bounce"
      config: "config/prod_live_oi_liq_bounce.yaml"
      weight: 0.30       # 佔總資金 30%
      account: "sub_1"   # 子帳號
```

### 權重決定方法

1. **等權重**：最簡單，適合初期 2-3 個策略
2. **反波動率**：低波動策略分配更多資金（`vol_parity`）
3. **最佳化**：均值-方差最佳化，最大化組合 SR（`scripts/compare_strategies.py` 會計算）
4. **Kelly**：基於 Kelly Fraction 分配（保守用 Quarter Kelly）

### 約束條件

- 單策略最大權重：60%
- 單策略最小權重：10%
- 總槓桿上限：配合 Risk Manager 設定
- 每次調整幅度不超過 ±20%

---

## 7) 定期審查排程

| 頻率 | 審查內容 | 主導人 | 協作 |
|------|---------|--------|------|
| **每週** | 各策略滾動 SR、MDD、交易數 | Risk Manager | — |
| **每月** | MC 重跑 + 相關性矩陣 + Kelly 校準 + 邊際貢獻 | Risk Manager | 如發現 alpha decay 異常，交 Quant Researcher 做深度 IC 分析 |
| **每季** | 全組合 re-optimization、策略淘汰評估 | Risk Manager (召集) | 全團隊參與 |

> **分工原則**：月度審查由 Risk Manager 主導（執行 MC + 相關性 + Kelly），涵蓋組合風險面向。
> 如果月度審查發現 alpha decay 跡象（如 IC 持續下降、滾動 SR 惡化），
> Risk Manager 將相關幣種/策略交給 Quant Researcher 做深入 IC 分析和 alpha 失效判定。
> 避免 Risk Manager 和 Researcher 重複計算相關性矩陣。

### 審查輸出

每次審查必須產出：

1. 各策略績效摘要（SR、MDD、交易數、曝險率）
2. 跨策略相關性矩陣（與上次比較，標記顯著變化）
3. 邊際貢獻分析（移除每個策略後組合 SR 的變化）
4. 建議動作：`MAINTAIN` / `REBALANCE` / `ADD_CANDIDATE` / `REDUCE` / `RETIRE`

---

## 8) 工具鏈

| 工具 | 用途 | 觸發時機 |
|------|------|---------|
| `scripts/compare_strategies.py` | 新策略評估 + 邊際 Sharpe | 新策略完成風控審計後 |
| `scripts/run_portfolio_backtest.py --multi-strategy` | 多策略組合回測 | 權重調整前 |
| `scripts/run_portfolio_backtest.py` | 單策略組合回測 | 日常驗證 |
| Risk Manager `/risk-review` | 定期風控審查 | 每週/每月 |

---

## 9) 反偏差規則（與 Playbook R2.1 一致）

- 禁止事後刪除表現差的策略而不遵循替換流程
- 禁止 cherry-pick 表現好的時間窗口來支持納入決策
- 所有比較必須使用相同時間範圍
- 邊際 Sharpe 測試必須含成本（不可用 simple mode）
- 新策略的回測期必須涵蓋現有策略的回測期

---

## 10) 常見問題 (FAQ)

**Q: 新策略只交易 2 個幣種，但 SR 很高，該納入嗎？**
A: 需通過 A2（相關性）和 A3（邊際 SR）。如果只有 2 個幣種，統計顯著性可能不足（A4）。建議先在 paper trading 中驗證。

**Q: 現有策略表現下滑，但新策略還沒準備好，怎麼辦？**
A: 進入 `reduced` 狀態（權重 ×0.5），不急於替換。等新策略完成驗證棧再替換。

**Q: meta_blend 裡的子策略權重該怎麼決定？**
A: 使用 `scripts/compare_strategies.py` 的最佳化輸出，或用 `research_strategy_blend.py` 做 per-symbol 權重掃描。

**Q: 兩個策略相關性突然升高怎麼辦？**
A: 月度審查會偵測到。如果 corr > 0.5 持續 4 週，觸發審查：考慮降低其中一個的權重或合併進 meta_blend。
