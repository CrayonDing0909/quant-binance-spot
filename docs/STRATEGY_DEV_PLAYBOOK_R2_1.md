# Strategy Development Playbook (R2.1)

> **Last updated**: 2026-02-25 (新增 Stage D.5 Overlay Ablation、驗證矩陣、一致性檢查)

This playbook captures the end-to-end process used to evolve from R1 -> R2 -> R2.1.
Use this as the default template for all future strategy research and production rollout.

## 1) Core Principles

- Remove false alpha first (look-ahead, execution leakage, data leakage).
- Prefer isolated, testable improvements over large multi-factor jumps.
- Require cost-aware, out-of-sample evidence before promotion.
- Treat failed experiments as useful information, not waste.
- Never overwrite production config during research.

## 2) Standard Lifecycle

### Stage A: Baseline Freeze

1. Freeze current production candidate (`prod_candidate_*.yaml`).
2. Record config hash and report path.
3. Define baseline KPIs:
   - Total Return
   - CAGR
   - Sharpe
   - MaxDD
   - Calmar
   - Trades / Flips

### Stage B: Hypothesis and Scope

For each new idea, write:

- Hypothesis (why this should work)
- Mechanism (what market behavior it captures)
- Failure mode (when it should fail)
- Scope (entry/exit/overlay/data/execution)

Only one major hypothesis per experiment batch.

### Stage C: Minimal Implementation

1. Add new code as opt-in (default disabled).
2. Keep backward compatibility.
3. Add dedicated config files under `config/research_*.yaml`.
4. Keep production config unchanged.

### Stage D: Validation Stack (must pass in order)

1. **Causality checks**
   - Truncation invariance
   - Online vs offline consistency
   - Execution parity (`t` signal, `t+1 open` execution)
2. **Backtest checks**
   - Full-period (strict costs)
   - Cost stress (1.5x, 2.0x)
   - Year-by-year decomposition
3. **Generalization checks**
   - Walk-forward (>= 5 splits)
   - Holdout period
4. **Delay fragility**
   - +1 bar delay stress

If any critical check fails, do not promote.

### Stage D.5: Overlay Ablation (必做)

如果生產 pipeline 有套用 overlay（`vol_pause`、`oi_vol` 等），**所有準備上生產的策略**必須跑 overlay ablation。

#### 情境 A：標準 overlay ablation（2×2）

當策略搭配現有的生產 overlay 時：

| # | 組合 | 說明 |
|---|------|------|
| 1 | 策略裸跑（無 overlay） | 純信號品質基準 |
| 2 | 策略 + overlay（與生產配置一致） | 部署環境基準 |

報告必須包含：
- Delta Sharpe（overlay ON vs OFF）
- Delta MDD
- 逐幣種分解（哪些幣種受益、哪些不受影響）

#### 情境 B：替代聲明 ablation（3-way）

如果新組件**聲稱可以取代 overlay**（例如：「carry confirmation 可取代 vol_pause」），必須跑 3-way：

| # | 組合 | 說明 |
|---|------|------|
| 1 | 策略裸跑 | 基準 |
| 2 | 策略 + 新組件（無 overlay） | 測試替代效果 |
| 3 | 策略 + 新組件 + overlay | 測試互補性 |

**沒有 3-way 數據的替代聲明，直接駁回。**

> 工具參考：`scripts/research_overlay_4way.py` 可做 R3C × MetaBlend 的 4-way 比較。
> 新策略可參考此腳本結構，或手動跑 `run_backtest.py` 配合 overlay config 切換。

### Stage E: Ablation and Attribution

When multiple components are present, run:

- component A only
- component B only
- A + B

Do not claim improvements without attribution.

> **注意**：Stage E 的通用 ablation 與 Stage D.5 的 overlay ablation 是互補的。
> Stage D.5 聚焦 overlay ON/OFF；Stage E 聚焦策略內部組件（如 carry vs TSMOM）。

### Stage F: Decision

Use only these verdicts:

- `GO_NEXT` (promote to next stage)
- `KEEP_BASELINE` (no promotion)
- `NEED_MORE_WORK` (promising but incomplete)
- `FAIL` (reject hypothesis)

### Stage G: Rollout

1. Paper gate (>= 7-14 days)
2. Small capital ramp (25% -> 50% -> 100%)
3. Hard rollback conditions

## 3) Required Reports Per Experiment

Each experiment must output:

1. Change summary (files + purpose)
2. Metrics table (baseline vs candidate)
3. Yearly table
4. Walk-forward summary
5. Cost stress table
6. Falsification matrix
7. Final verdict and next action
8. Evidence paths

## 4) Anti-Bias Rules

- No cherry-picking winning windows.
- No post-hoc strategy deletion without pre-defined selection rule.
- No mixed timestamps in final comparison table (single run timestamp only).
- No hidden defaults: all active knobs must be explicit in config.

## 5) Data Quality Gate

Before any signal uses a dataset:

- Coverage threshold (default >= 70%, target >= 90%).
- Time alignment documented (timezone, bar alignment).
- Missing value policy documented (max forward-fill gap).
- Provider/source documented.

If coverage gate fails, result must be marked `INVALID`.

## 6) Overlay-Specific Rules

For overlays (exit/risk controls):

- Validate overlay and base strategy both independently.
- Ensure overlay is included in walk-forward path.
- Track both trades and flips (not trades only).
- Run delay stress because overlays are often timing-sensitive.
- **Must pass Stage D.5 overlay ablation** before production promotion.
- Overlay ON/OFF 狀態必須在回測報告中明確標註（見 `backtest.mdc` Overlay Audit Rules）。

## 7) Validation Matrix (唯一事實來源)

新策略上生產前，以下所有項目必須全部通過。這張表是跨 agent 的統一檢查清單。

| # | 測試 | 負責人 | 工具/指令 | 通過標準 |
|---|------|--------|-----------|----------|
| V1 | 完整回測（生產一致，overlay ON） | Developer | `run_backtest.py -c config/research_*.yaml` | Sharpe > 0.3，成本已啟用 |
| V2 | Overlay ablation（裸跑 vs overlay） | Developer | 手動或 `research_overlay_4way.py` | 報告 Delta Sharpe / MDD |
| V3 | `validate --quick`（因果 + 基本檢查） | Developer | `validate.py --quick` | 全 PASS |
| V4 | 部署前一致性（含 overlay 一致性） | Researcher | `validate_live_consistency.py` | 全 PASS |
| V5 | Walk-Forward（>= 5 splits） | Researcher | `run_walk_forward.py --splits 6` | OOS Sharpe > 0.3 |
| V6 | CPCV 交叉驗證 | Researcher | `run_cpcv.py --splits 6 --test-splits 2` | PBO < 0.5 |
| V7 | 成本壓力測試（1.5x、2.0x） | Researcher | 手動乘數 | 2x 下仍盈利 |
| V8 | 延遲壓力測試（+1 bar） | Researcher | 手動延遲 | SR 衰減 < 50% |
| V9 | DSR 統計檢定 | Researcher | `validate.py --full` | p-value < 0.05 |
| V10 | Monte Carlo（MC1-MC4） | Risk Manager | MC 腳本 | 4 項全 PASS |
| V11 | 組合風險評估 | Risk Manager | 風險腳本 | VaR < 5%, corr < 0.8 |
| V12 | Production Launch Guard | Risk Manager | `prod_launch_guard.py --dry-run` | ALLOW_LAUNCH |

### 分工原則

- **Developer**（V1-V3）：確保回測正確、overlay ablation 完成、基本因果檢查通過
- **Researcher**（V4-V9）：獨立重跑完整驗證，不信任 Developer 的數字
- **Risk Manager**（V10-V12）：壓力測試和部署門檻

### 跳過條件

- V2（Overlay ablation）：僅當策略**不使用任何 overlay** 時可跳過
- V4（部署前一致性）：所有策略必做，不可跳過
- V6（CPCV）：交易次數 < 100 時可改用 bootstrap CI

## 8) Production Promotion Checklist

Before switching live:

1. Config frozen and hashed.
2. Launch guard updated for new config hash/path.
3. **Validation Matrix (Section 7) 全部 PASS**。
4. **Overlay ablation 完成**（Stage D.5），overlay ON/OFF 已在報告中標註。
5. **`validate_live_consistency.py` 通過** — 回測/實盤路徑一致。
6. Dry-run once in real mode.
7. Single active live tmux session confirmed.
8. Telegram notification + command bot validated.
9. Post-launch 15-minute health check passed.

## 9) Recommended Directory Conventions

- Strategy code: `src/qtrade/strategy/`
- Overlay code: `src/qtrade/strategy/overlays/`
- Research scripts: `scripts/run_*_research.py`
- Research configs: `config/research_*.yaml`
- Reports:
  - `reports/<topic>/<timestamp>/...`
  - Include machine-readable JSON summary

## 10) Reusable Experiment Template

Copy and fill this block for each new strategy:

- Hypothesis:
- Market regime target:
- Expected edge source:
- Primary risk:
- Data dependencies:
- Ablation plan:
- Validation gates:
- Promotion criteria:
- Rollback criteria:

## 11) R2.1 Key Lessons (to preserve)

- Fixing look-ahead comes before optimization.
- OI with weak coverage can create false conclusions.
- Vol-only overlay can improve metrics, but timing fragility must be explicitly tested.
- "High return" is insufficient without falsification and execution realism.

