---
description: "Pre-research sanity check — scan failure registry and known dead ends before investing time. Use BEFORE starting any new research direction."
---

# /check-direction — Is This Direction Worth Pursuing?

Before spending hours on research, check if this direction (or something structurally similar) has already been tried and failed.

## Step 1: Scan Failure Registry

Read `.cursor/skills/alpha/failure-registry.md` and check for overlap with the proposed direction.

Known dead ends (summary — read the file for full details):

| Direction | Why it failed | Time wasted |
|---|---|---|
| 4h TF Optimization | Improvement came from HTF filter look-ahead, not signal | 4h |
| XSMOM Cross-Sectional | Crypto pairwise corr ~0.7 kills ranking strategies | 3h |
| CVD IC Bias | IC inflated 19× by calculation bug | — |
| Taker Vol Variants | Base IC=-0.006, 14 variants explored for nothing | 6h |
| BB Mean Reversion (1h) | IC positive but gross PnL negative (PF 0.83) | 2h |
| FR Carry | Funding rate unstable across coins (SOL/BNB 2yr FR < 0) | 2h |
| OI/On-chain/Macro/VPIN Filters | All redundant with HTF filter (same factor, different dress) | 16h total |
| Entropy Regime | Zero predictive power on 1h crypto (IC < 0.01) | 30min |
| Orderflow Composite Standalone | Hourly taker_vol_ratio IC=0.003, alpha << cost | 6h |

**Pattern to watch**: If the proposed direction is a "Regime Filter" targeting "low-conviction bars" → very likely redundant with existing HTF filter. Test REPLACEMENT before ADDITION.

## Step 2: Check ALPHA_RESEARCH_MAP

Read `docs/ALPHA_RESEARCH_MAP.md`:
- Check "Closed Directions" — is this direction already closed?
- Check "Research Frontier" — is this direction already queued with higher priority?
- Check "Verified Directions" — does this overlap with something already in production?

## Step 3: Quick Structural Checks

Run these before any EDA:

```python
# Cross-correlation check (kills cross-sectional strategies)
corr_matrix = df.pct_change().corr()
avg_pairwise = corr_matrix.values[np.triu_indices_from(corr_matrix, k=1)].mean()
if avg_pairwise > 0.5:
    print("WARNING: avg pairwise corr > 0.5 — cross-sectional ranking won't work")

# Base signal IC quick check
ic = signal.shift(1).corr(forward_return)
if abs(ic) < 0.01:
    print("WARNING: base IC < 0.01 — not worth exploring variants")
```

## Step 4: Economic Intuition Gate

Ask the user to answer:
1. **Who is the counterparty?**
2. **Why doesn't it get arbitraged away?**
3. **When does it break?**

If Q1 has no answer → recommend NOT proceeding.

## Output

After all checks, give one of:

| Verdict | Meaning |
|---|---|
| **PROCEED** | No overlap with known failures, economic intuition is sound |
| **CAUTION** | Partial overlap — proceed but monitor specific risk |
| **STOP** | Structurally identical to a known failure — pivot or abandon |
