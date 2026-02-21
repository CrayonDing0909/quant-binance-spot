
# Strategy Development Playbook (R2.1)

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

### Stage E: Ablation and Attribution

When multiple components are present, run:

- component A only
- component B only
- A + B

Do not claim improvements without attribution.

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

## 7) Production Promotion Checklist

Before switching live:

1. Config frozen and hashed.
2. Launch guard updated for new config hash/path.
3. Dry-run once in real mode.
4. Single active live tmux session confirmed.
5. Telegram notification + command bot validated.
6. Post-launch 15-minute health check passed.

## 8) Recommended Directory Conventions

- Strategy code: `src/qtrade/strategy/`
- Overlay code: `src/qtrade/strategy/overlays/`
- Research scripts: `scripts/run_*_research.py`
- Research configs: `config/research_*.yaml`
- Reports:
  - `reports/<topic>/<timestamp>/...`
  - Include machine-readable JSON summary

## 9) Reusable Experiment Template

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

## 10) R2.1 Key Lessons (to preserve)

- Fixing look-ahead comes before optimization.
- OI with weak coverage can create false conclusions.
- Vol-only overlay can improve metrics, but timing fragility must be explicitly tested.
- "High return" is insufficient without falsification and execution realism.

