"""
Symbol Governance — Negative Edge Filter with Kill-List / Quarantine

Spec: docs/R3C_SYMBOL_GOVERNANCE_SPEC.md

State machine:
    active  ─(degrade)──▶  deweighted  ─(persist)──▶  quarantined
    active  ◀──(recover)─  deweighted  ◀──(gate)───   quarantined

This module is *purely additive*: it only adjusts effective weights and
never touches strategy signal generation or order execution logic.

When governance is **disabled**, all functions short-circuit and return
the original base weights unchanged (backward compatible).
"""
from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..config import SymbolGovernanceConfig
from .symbol_metrics_store import SymbolMetrics
from ..utils.log import get_logger

logger = get_logger("symbol_governance")


# ═══════════════════════════════════════════════════════════════
#  Enums & Data Models
# ═══════════════════════════════════════════════════════════════

class SymbolState(str, Enum):
    ACTIVE = "active"
    DEWEIGHTED = "deweighted"
    QUARANTINED = "quarantined"


@dataclass
class SymbolRecord:
    """Persistent per-symbol governance state."""
    symbol: str
    state: SymbolState = SymbolState.ACTIVE
    state_since: str = ""                   # ISO timestamp
    consecutive_negative_sharpe: int = 0    # for quarantine trigger
    consecutive_recovery_reviews: int = 0   # for recovery gate
    quarantine_start: Optional[str] = None  # ISO timestamp
    base_weight: float = 0.0
    effective_weight: float = 0.0
    last_review: Optional[str] = None
    reason_codes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["state"] = self.state.value
        return d

    @classmethod
    def from_dict(cls, raw: dict) -> "SymbolRecord":
        raw = dict(raw)
        raw["state"] = SymbolState(raw.get("state", "active"))
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})


@dataclass
class GovernanceDecision:
    """Single review decision artifact."""
    timestamp: str
    review_date: str
    symbols: Dict[str, dict]         # symbol → per-symbol detail
    summary: dict                    # counts by state
    config_snapshot: dict            # thresholds used

    def to_dict(self) -> dict:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════
#  Core Engine
# ═══════════════════════════════════════════════════════════════

class SymbolGovernanceEngine:
    """
    Deterministic governance engine.

    Usage:
        engine = SymbolGovernanceEngine(gov_cfg, base_weights)
        engine.load_state(previous_records)
        decision = engine.run_review(metrics_by_symbol, review_time)
        engine.persist(artifacts_dir)
    """

    def __init__(
        self,
        gov_cfg: SymbolGovernanceConfig,
        base_weights: Dict[str, float],
    ):
        self.cfg = gov_cfg
        self.t = gov_cfg.thresholds
        self.w = gov_cfg.weights
        self.base_weights = dict(base_weights)

        # Current state per symbol
        self.records: Dict[str, SymbolRecord] = {}
        for sym, bw in base_weights.items():
            self.records[sym] = SymbolRecord(
                symbol=sym,
                state=SymbolState.ACTIVE,
                state_since=datetime.now(timezone.utc).isoformat(),
                base_weight=bw,
                effective_weight=bw,
            )

    # ── State persistence ──────────────────────────────────────

    def load_state(self, records: List[dict] | None) -> None:
        """Load previous state from artifact (latest_decisions.json)."""
        if not records:
            return
        for raw in records:
            sym = raw.get("symbol", "")
            if sym in self.records:
                self.records[sym] = SymbolRecord.from_dict(raw)
                # Preserve base_weight from config (authoritative source)
                self.records[sym].base_weight = self.base_weights.get(sym, 0.0)

    def get_records_list(self) -> List[dict]:
        return [r.to_dict() for r in self.records.values()]

    # ── Main review entry point ────────────────────────────────

    def run_review(
        self,
        metrics: Dict[str, SymbolMetrics],
        review_time: Optional[datetime] = None,
        dry_run: bool = False,
    ) -> GovernanceDecision:
        """
        Execute one review cycle. Deterministic: same inputs → same outputs.

        Args:
            metrics: per-symbol SymbolMetrics (already .compute()'d)
            review_time: optional override (for reproducibility)
            dry_run: if True, do not mutate internal state

        Returns:
            GovernanceDecision artifact
        """
        now = review_time or datetime.now(timezone.utc)
        now_iso = now.isoformat()

        # Work on copies if dry_run
        working = {
            sym: SymbolRecord.from_dict(rec.to_dict())
            for sym, rec in self.records.items()
        } if dry_run else self.records

        per_symbol_detail: Dict[str, dict] = {}

        for sym, rec in working.items():
            m = metrics.get(sym)
            if m is None:
                per_symbol_detail[sym] = {
                    "previous_state": rec.state.value,
                    "new_state": rec.state.value,
                    "reason_codes": ["no_metrics"],
                    "metrics": {},
                    "base_weight": rec.base_weight,
                    "effective_weight": rec.effective_weight,
                }
                continue

            prev_state = rec.state
            reason_codes: List[str] = []

            # Evaluate transitions
            new_state = self._evaluate_transition(rec, m, now, reason_codes)
            self._update_record(rec, new_state, now_iso, reason_codes)

            per_symbol_detail[sym] = {
                "previous_state": prev_state.value,
                "new_state": new_state.value,
                "reason_codes": list(reason_codes),
                "metrics": m.to_dict(),
                "base_weight": rec.base_weight,
                "effective_weight": 0.0,  # filled after renormalization
            }

        # Compute effective weights
        eff = self._compute_effective_weights(working)
        for sym, w in eff.items():
            working[sym].effective_weight = w
            if sym in per_symbol_detail:
                per_symbol_detail[sym]["effective_weight"] = w

        # Summary
        counts = {"active": 0, "deweighted": 0, "quarantined": 0}
        for rec in working.values():
            counts[rec.state.value] = counts.get(rec.state.value, 0) + 1

        decision = GovernanceDecision(
            timestamp=now_iso,
            review_date=now.strftime("%Y-%m-%d"),
            symbols=per_symbol_detail,
            summary=counts,
            config_snapshot={
                "thresholds": asdict(self.t),
                "weights": asdict(self.w),
                "quarantine_min_days": self.cfg.quarantine_min_days,
                "warmup_days": self.cfg.warmup_days,
            },
        )

        if not dry_run:
            for sym, rec in working.items():
                rec.last_review = now_iso

        return decision

    # ── Transition logic (spec §6) ─────────────────────────────

    def _evaluate_transition(
        self,
        rec: SymbolRecord,
        m: SymbolMetrics,
        now: datetime,
        reasons: List[str],
    ) -> SymbolState:
        t = self.t
        state = rec.state

        if state == SymbolState.ACTIVE:
            return self._eval_active(rec, m, reasons)
        elif state == SymbolState.DEWEIGHTED:
            return self._eval_deweighted(rec, m, now, reasons)
        elif state == SymbolState.QUARANTINED:
            return self._eval_quarantined(rec, m, now, reasons)
        return state

    def _eval_active(
        self, rec: SymbolRecord, m: SymbolMetrics, reasons: List[str]
    ) -> SymbolState:
        """active → deweighted if any trigger fires."""
        t = self.t
        triggered = False

        if m.edge_sharpe_4w < t.edge_sharpe_deweight:
            reasons.append(f"sharpe={m.edge_sharpe_4w:.2f}<{t.edge_sharpe_deweight}")
            triggered = True
        if m.edge_per_turnover_4w < t.edge_per_turnover_min:
            reasons.append(f"edge_per_to={m.edge_per_turnover_4w:.4f}<{t.edge_per_turnover_min}")
            triggered = True
        if m.slippage_ratio_4w > t.slippage_ratio_deweight:
            reasons.append(f"slip_ratio={m.slippage_ratio_4w:.2f}>{t.slippage_ratio_deweight}")
            triggered = True
        if m.consistency_4w < t.consistency_min:
            reasons.append(f"consistency={m.consistency_4w:.1f}<{t.consistency_min}")
            triggered = True

        if triggered:
            # Reset recovery counter on demotion
            rec.consecutive_recovery_reviews = 0
            return SymbolState.DEWEIGHTED
        return SymbolState.ACTIVE

    def _eval_deweighted(
        self,
        rec: SymbolRecord,
        m: SymbolMetrics,
        now: datetime,
        reasons: List[str],
    ) -> SymbolState:
        """
        deweighted → quarantined (persistent degradation)
        deweighted → active (recovery)
        """
        t = self.t

        # ── Check quarantine escalation ──
        if m.edge_sharpe_4w < t.edge_sharpe_quarantine:
            rec.consecutive_negative_sharpe += 1
        else:
            rec.consecutive_negative_sharpe = 0

        quarantine_trigger = False
        # Path 1: persistent negative sharpe + slippage/missed
        if rec.consecutive_negative_sharpe >= self.cfg.consecutive_reviews_for_quarantine:
            if (m.slippage_ratio_4w > t.slippage_ratio_quarantine
                    or m.missed_4w > t.missed_signals_quarantine):
                reasons.append(
                    f"persist_neg_sharpe({rec.consecutive_negative_sharpe}x)"
                    f"+slip_or_miss"
                )
                quarantine_trigger = True

        # Path 2: extreme drawdown
        if m.dd_4w > t.dd_quarantine_pct:
            reasons.append(f"dd={m.dd_4w:.1f}>{t.dd_quarantine_pct}")
            quarantine_trigger = True

        if quarantine_trigger:
            rec.consecutive_recovery_reviews = 0
            return SymbolState.QUARANTINED

        # ── Check recovery to active ──
        recovery = (
            m.edge_sharpe_4w >= t.edge_sharpe_recover_active
            and m.edge_per_turnover_4w > t.edge_per_turnover_min
            and m.slippage_ratio_4w <= t.slippage_ratio_recover_active
        )
        if recovery:
            rec.consecutive_recovery_reviews += 1
            if rec.consecutive_recovery_reviews >= self.cfg.consecutive_reviews_for_recovery:
                reasons.append(
                    f"recovery({rec.consecutive_recovery_reviews}x):"
                    f"sharpe={m.edge_sharpe_4w:.2f},slip={m.slippage_ratio_4w:.2f}"
                )
                return SymbolState.ACTIVE
        else:
            rec.consecutive_recovery_reviews = 0

        return SymbolState.DEWEIGHTED

    def _eval_quarantined(
        self,
        rec: SymbolRecord,
        m: SymbolMetrics,
        now: datetime,
        reasons: List[str],
    ) -> SymbolState:
        """quarantined → deweighted (recovery gate)."""
        t = self.t

        # Check minimum quarantine duration
        if rec.quarantine_start:
            q_start = datetime.fromisoformat(rec.quarantine_start)
            if q_start.tzinfo is None:
                q_start = q_start.replace(tzinfo=timezone.utc)
            days_in_q = (now - q_start).total_seconds() / 86400
            if days_in_q < self.cfg.quarantine_min_days:
                reasons.append(
                    f"quarantine_min_not_met({days_in_q:.1f}/{self.cfg.quarantine_min_days}d)"
                )
                return SymbolState.QUARANTINED

        # Recovery gate
        gate_pass = (
            m.edge_sharpe_4w >= t.edge_sharpe_recover_from_quarantine
            and m.slippage_ratio_4w <= t.slippage_ratio_recover_from_quarantine
            and m.consistency_4w >= t.consistency_recover
        )
        if gate_pass:
            reasons.append(
                f"quarantine_recovery_gate:"
                f"sharpe={m.edge_sharpe_4w:.2f},slip={m.slippage_ratio_4w:.2f},"
                f"cons={m.consistency_4w:.1f}"
            )
            rec.consecutive_negative_sharpe = 0
            rec.consecutive_recovery_reviews = 0
            return SymbolState.DEWEIGHTED

        return SymbolState.QUARANTINED

    # ── Record update helper ───────────────────────────────────

    @staticmethod
    def _update_record(
        rec: SymbolRecord,
        new_state: SymbolState,
        now_iso: str,
        reasons: List[str],
    ) -> None:
        if new_state != rec.state:
            rec.state = new_state
            rec.state_since = now_iso
            if new_state == SymbolState.QUARANTINED:
                rec.quarantine_start = now_iso
            elif new_state == SymbolState.ACTIVE:
                rec.quarantine_start = None
                rec.consecutive_negative_sharpe = 0
                rec.consecutive_recovery_reviews = 0
        rec.reason_codes = list(reasons)

    # ── Weight policy (spec §7) ────────────────────────────────

    def _compute_effective_weights(
        self,
        records: Dict[str, SymbolRecord],
    ) -> Dict[str, float]:
        """
        Compute effective weights with multiplier + min/max + renormalization.

        Quarantined symbols get 0.0 weight.
        Remaining budget is redistributed across non-quarantined symbols.
        """
        multipliers = {
            SymbolState.ACTIVE: self.w.active_multiplier,
            SymbolState.DEWEIGHTED: self.w.deweight_multiplier,
            SymbolState.QUARANTINED: self.w.quarantine_multiplier,
        }

        # Step 1: Apply multiplier
        raw: Dict[str, float] = {}
        for sym, rec in records.items():
            raw[sym] = rec.base_weight * multipliers[rec.state]

        # Step 2: Separate zero-weight (quarantined) from non-zero
        nonzero_syms = [sym for sym, w in raw.items() if w > 0]
        result: Dict[str, float] = {sym: 0.0 for sym in raw}

        if not nonzero_syms:
            # All symbols quarantined (or multiplier=0) → all-zero is the safe
            # conservative stance. Do NOT fall back to equal weight.
            logger.warning(
                "⚠️  All symbols have zero weight after governance multiplier "
                "(all quarantined?) — returning all-zero weights"
            )
            return {sym: 0.0 for sym in raw}

        # Step 3: Renormalize + clip using pinning algorithm
        #   Pin symbols that hit min/max bounds, redistribute rest.
        #   Hard constraint: sum(weights) == 1.0
        #   Soft constraint: min_weight <= w <= max_weight (relaxed if infeasible)
        budget = 1.0
        free = list(nonzero_syms)
        pinned: Dict[str, float] = {}

        for _ in range(len(nonzero_syms) + 1):
            if not free:
                break
            free_total = sum(raw[s] for s in free)
            if free_total <= 0:
                for s in free:
                    pinned[s] = budget / len(free)
                free = []
                break

            new_pinned = False
            for s in list(free):
                w = (raw[s] / free_total) * budget
                if w < self.w.min_weight:
                    pinned[s] = self.w.min_weight
                    budget -= self.w.min_weight
                    free.remove(s)
                    new_pinned = True
                elif w > self.w.max_weight:
                    pinned[s] = self.w.max_weight
                    budget -= self.w.max_weight
                    free.remove(s)
                    new_pinned = True
            if not new_pinned:
                break

        # Distribute remaining budget among free symbols proportionally
        if free:
            free_total = sum(raw[s] for s in free)
            for s in free:
                pinned[s] = (raw[s] / free_total) * budget if free_total > 0 else budget / len(free)
        elif budget > 1e-9:
            # All symbols pinned but budget remains (max_weight too tight).
            # Distribute surplus proportionally across all non-zero to preserve sum=1.
            all_pinned = [s for s in nonzero_syms if s in pinned]
            if all_pinned:
                total_pinned = sum(pinned[s] for s in all_pinned)
                if total_pinned > 0:
                    for s in all_pinned:
                        pinned[s] += (pinned[s] / total_pinned) * budget

        for sym in raw:
            result[sym] = pinned.get(sym, 0.0)

        return result

    # ── Effective weights public API ───────────────────────────

    def get_effective_weights(self) -> Dict[str, float]:
        """Return current effective weights (call after run_review)."""
        return {sym: rec.effective_weight for sym, rec in self.records.items()}

    # ── Persistence ────────────────────────────────────────────

    def persist(
        self,
        decision: GovernanceDecision,
        artifacts_dir: str | Path | None = None,
    ) -> Tuple[Path, Path]:
        """
        Write artifacts:
          - latest_decisions.json (overwrite)
          - history.jsonl (append)

        Returns (latest_path, history_path)
        """
        base = Path(artifacts_dir or self.cfg.artifacts_dir)
        base.mkdir(parents=True, exist_ok=True)

        latest_path = base / "latest_decisions.json"
        history_path = base / "history.jsonl"

        # latest_decisions.json — includes full records for state reload
        payload = {
            **decision.to_dict(),
            "records": self.get_records_list(),
        }
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        # history.jsonl — one line per review
        with open(history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(decision.to_dict(), ensure_ascii=False) + "\n")

        logger.info(
            f"Governance artifacts written → {latest_path} / {history_path}"
        )
        return latest_path, history_path


# ═══════════════════════════════════════════════════════════════
#  Public Helpers
# ═══════════════════════════════════════════════════════════════

def load_latest_decisions(artifacts_dir: str | Path) -> Optional[dict]:
    """Read the latest governance artifact (if exists)."""
    p = Path(artifacts_dir) / "latest_decisions.json"
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_governance_weights(
    base_weights: Dict[str, float],
    gov_cfg: SymbolGovernanceConfig,
    artifacts_dir: str | Path | None = None,
) -> Dict[str, float]:
    """
    Convenience: load latest decisions and return effective weights.

    If governance is disabled or no artifact exists, returns base_weights unchanged.
    This is the single integration point for BaseRunner.
    """
    if not gov_cfg.enabled:
        return dict(base_weights)

    adir = artifacts_dir or gov_cfg.artifacts_dir
    latest = load_latest_decisions(adir)
    if not latest:
        return dict(base_weights)

    # Rebuild effective weights from artifact
    symbols_data = latest.get("symbols", {})
    result = {}
    for sym, bw in base_weights.items():
        sym_info = symbols_data.get(sym)
        if sym_info:
            result[sym] = sym_info.get("effective_weight", bw)
        else:
            result[sym] = bw
    return result
