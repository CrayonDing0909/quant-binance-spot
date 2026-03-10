---
description: Shared reopen framework for research directions that are promising at proxy level but blocked before developer handoff
globs:
alwaysApply: false
---
# Skill: Reopen Framework

> Loaded by Alpha Researcher when a direction is promising enough to preserve, but not ready for Quant Developer handoff.

## When To Use

Use this framework when the direction is:
- better than a clean `FAIL`
- not strong enough for `GO_NEXT`
- blocked by data quality, state validation, or unresolved bucket gaps

Typical examples:
- proxy signal survives, but true state labels are missing
- common-window evidence is good, but full evidence stack is incomplete
- independence is good, but critical target buckets are still weak

## Core Principle

Separate **evidence tier** from **final verdict**.

- Final verdict answers: "Do we hand off now?"
- Evidence tier answers: "How far did this thesis survive before it got blocked?"

This avoids collapsing all promising-but-blocked directions into the same vague `SHELVED`.

## Standard Evidence Tiers

Use exactly one of these labels:

1. `not-proxy-valid`
2. `proxy-valid / state-invalid`
3. `state-valid / handoff-blocked`
4. `handoff-ready`

### Meanings

#### `not-proxy-valid`
- The idea does not survive even the proxy/common-data-window test.
- Treat as `FAIL` unless there is a very specific alternative hypothesis.

#### `proxy-valid / state-invalid`
- The idea survives with proxy definitions.
- Data gates on the active research window are good enough.
- But the thesis has not passed the true state-based evidence layer.
- This is the standard label for dormant reopen candidates.

#### `state-valid / handoff-blocked`
- The state-based thesis survives.
- But some second-leg requirements still fail:
  - unresolved target buckets
  - concentration too high
  - overlap too high
  - density outside portfolio-useful zone

#### `handoff-ready`
- Evidence stack is complete enough for Quant Developer handoff.
- This does not replace the normal GO/FAIL gates; it means the research side is complete.

## Standard Phase Model

Always structure reopen candidates into 3 phases:

### Phase A. Proxy Validation

Goal:
- Keep the thesis alive on a common-data window with causality intact.

Typical gates:
- coverage gate passes on the active window
- replay baseline dependencies are available on the same window
- best trigger survives without hidden dependency
- density is in the intended range
- correlation remains acceptably low

### Phase B. State Validation

Goal:
- Replace proxy tags with true state labels or true state measurements.

Typical gates:
- true historical state data exists
- same trigger family still survives after replacing proxy labels
- bucket-level edge is still visible after the state upgrade

### Phase C. Developer Handoff Gate

Goal:
- Confirm this is worth engineering, not just intellectually interesting.

Typical gates:
- target buckets are materially improved
- concentration is healthy
- correlation is still low enough
- thesis behaves like a second leg, not a disguised overlay/filter

## Required Machine-Readable Output

If a research script produces JSON, add:

```json
{
  "reopen_framework": {
    "current_state": "proxy_valid_state_invalid",
    "status_banner": "proxy-valid / state-invalid",
    "proxy_valid": true,
    "state_valid": false,
    "handoff_ready": false,
    "current_blockers": [
      "missing_true_liquidation_state_history"
    ],
    "phase_status": {
      "phase_a_proxy_validation": "pass",
      "phase_b_state_validation": "blocked",
      "phase_c_handoff_gate": "blocked"
    }
  }
}
```

## Required Notebook / Proposal Language

When a direction is preserved but blocked, explicitly write:

- current evidence tier
- what already passed
- what is still blocked
- what exact data or proof would reopen the direction

Do not write vague conclusions like:
- "interesting but incomplete"
- "maybe revisit later"

Write instead:
- `proxy-valid / state-invalid`
- `state-valid / handoff-blocked`

## Decision Rules

- If Phase A fails: usually `FAIL`
- If Phase A passes but Phase B is blocked: `SHELVED` + `proxy-valid / state-invalid`
- If Phase B passes but Phase C is blocked: `NEED_MORE_WORK` or `SHELVED`, with `state-valid / handoff-blocked`
- If all three phases pass: handoff may proceed

## Final Output Pattern

For blocked-but-preserved directions, end with:

```markdown
Verdict: SHELVED
Evidence tier: proxy-valid / state-invalid
Reopen trigger: historical liquidation-state becomes available
Recommended dormant direction: cascade_end reversal
```
