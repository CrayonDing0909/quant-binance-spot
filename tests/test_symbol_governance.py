"""
Symbol Governance — Unit Tests

Coverage:
  1. State machine transitions (active → deweighted → quarantined → deweighted → active)
  2. Quarantine minimum days enforcement
  3. Weight scaling + min/max + renormalization
  4. Disabled governance preserves original weights (smoke test)
  5. Determinism (same inputs → same outputs)
  6. Recovery requires consecutive reviews
  7. Artifact persistence
"""
from __future__ import annotations

import json
import math
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from qtrade.config import (
    SymbolGovernanceConfig,
    SymbolGovernanceThresholds,
    SymbolGovernanceWeights,
)
from qtrade.live.symbol_governance import (
    SymbolGovernanceEngine,
    SymbolState,
    SymbolRecord,
    GovernanceDecision,
    load_latest_decisions,
    apply_governance_weights,
)
from qtrade.live.symbol_metrics_store import SymbolMetrics


# ═══════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def default_gov_cfg() -> SymbolGovernanceConfig:
    """Default governance config with spec thresholds."""
    return SymbolGovernanceConfig(
        enabled=True,
        review_frequency="weekly",
        warmup_days=14,
        quarantine_min_days=14,
        consecutive_reviews_for_quarantine=2,
        consecutive_reviews_for_recovery=2,
        thresholds=SymbolGovernanceThresholds(),
        weights=SymbolGovernanceWeights(),
    )


@pytest.fixture
def base_weights() -> dict[str, float]:
    return {
        "BTCUSDT": 0.30,
        "ETHUSDT": 0.30,
        "SOLUSDT": 0.20,
        "DOGEUSDT": 0.20,
    }


def _healthy_metrics(symbol: str) -> SymbolMetrics:
    """Return a clearly healthy metric set (all pass thresholds)."""
    m = SymbolMetrics(
        symbol=symbol,
        net_pnl=100.0,
        turnover=10000.0,
        returns_series=[0.002] * 672,
        max_drawdown_pct=2.0,
        realized_slippage_bps=2.0,
        model_slippage_bps=3.0,
        signal_execution_consistency_pct=100.0,
        missed_signals_pct=0.0,
        trade_count=50,
    )
    return m.compute()


def _degraded_metrics(symbol: str) -> SymbolMetrics:
    """Return metrics that trigger active → deweighted."""
    m = SymbolMetrics(
        symbol=symbol,
        net_pnl=-50.0,
        turnover=10000.0,
        returns_series=[-0.001] * 672,
        max_drawdown_pct=5.0,
        realized_slippage_bps=5.0,
        model_slippage_bps=3.0,
        signal_execution_consistency_pct=98.5,
        missed_signals_pct=1.0,
        trade_count=30,
    )
    return m.compute()


def _severely_degraded_metrics(symbol: str) -> SymbolMetrics:
    """Return metrics that trigger deweighted → quarantined (path: persistent negative Sharpe + slip)."""
    m = SymbolMetrics(
        symbol=symbol,
        net_pnl=-200.0,
        turnover=5000.0,
        returns_series=[-0.003] * 672,
        max_drawdown_pct=10.0,
        realized_slippage_bps=6.0,
        model_slippage_bps=3.0,
        signal_execution_consistency_pct=97.0,
        missed_signals_pct=6.0,
        trade_count=20,
    )
    return m.compute()


def _extreme_dd_metrics(symbol: str) -> SymbolMetrics:
    """Metrics with extreme drawdown (>25%) → quarantine via DD path."""
    m = SymbolMetrics(
        symbol=symbol,
        net_pnl=-500.0,
        turnover=10000.0,
        returns_series=[-0.005] * 672,
        max_drawdown_pct=30.0,
        realized_slippage_bps=3.0,
        model_slippage_bps=3.0,
        signal_execution_consistency_pct=99.5,
        missed_signals_pct=0.0,
        trade_count=40,
    )
    return m.compute()


def _recovery_metrics(symbol: str) -> SymbolMetrics:
    """Metrics that satisfy deweighted → active recovery."""
    m = SymbolMetrics(
        symbol=symbol,
        net_pnl=200.0,
        turnover=20000.0,
        returns_series=[0.003] * 672,
        max_drawdown_pct=1.0,
        realized_slippage_bps=3.0,
        model_slippage_bps=3.0,
        signal_execution_consistency_pct=100.0,
        missed_signals_pct=0.0,
        trade_count=60,
    )
    return m.compute()


def _quarantine_recovery_metrics(symbol: str) -> SymbolMetrics:
    """Metrics that satisfy quarantined → deweighted recovery gate."""
    m = SymbolMetrics(
        symbol=symbol,
        net_pnl=150.0,
        turnover=15000.0,
        returns_series=[0.002] * 672,
        max_drawdown_pct=2.0,
        realized_slippage_bps=3.5,
        model_slippage_bps=3.0,
        signal_execution_consistency_pct=99.8,
        missed_signals_pct=0.0,
        trade_count=55,
    )
    return m.compute()


# ═══════════════════════════════════════════════════════════════
#  1. State Machine Transitions
# ═══════════════════════════════════════════════════════════════

class TestStateTransitions:
    """Full lifecycle: active → deweighted → quarantined → deweighted → active"""

    def test_active_stays_active_with_healthy_metrics(self, default_gov_cfg, base_weights):
        engine = SymbolGovernanceEngine(default_gov_cfg, base_weights)
        metrics = {sym: _healthy_metrics(sym) for sym in base_weights}
        decision = engine.run_review(metrics)

        for sym in base_weights:
            assert decision.symbols[sym]["new_state"] == "active"

    def test_active_to_deweighted(self, default_gov_cfg, base_weights):
        engine = SymbolGovernanceEngine(default_gov_cfg, base_weights)
        metrics = {sym: _healthy_metrics(sym) for sym in base_weights}
        # Degrade one symbol
        metrics["BTCUSDT"] = _degraded_metrics("BTCUSDT")

        decision = engine.run_review(metrics)

        assert decision.symbols["BTCUSDT"]["new_state"] == "deweighted"
        assert decision.symbols["ETHUSDT"]["new_state"] == "active"

    def test_deweighted_to_quarantined_persistent(self, default_gov_cfg, base_weights):
        """Requires consecutive_reviews_for_quarantine (2) reviews of negative Sharpe
        *while in deweighted state*. So 3 total reviews: 1 to enter deweighted + 2 more."""
        engine = SymbolGovernanceEngine(default_gov_cfg, base_weights)
        now = datetime(2026, 1, 6, tzinfo=timezone.utc)

        # Review 1: degrade → deweighted
        metrics = {sym: _healthy_metrics(sym) for sym in base_weights}
        metrics["BTCUSDT"] = _severely_degraded_metrics("BTCUSDT")
        engine.run_review(metrics, review_time=now)
        assert engine.records["BTCUSDT"].state == SymbolState.DEWEIGHTED

        # Review 2: still severe (1st consecutive negative sharpe while deweighted)
        now += timedelta(weeks=1)
        engine.run_review(metrics, review_time=now)
        assert engine.records["BTCUSDT"].state == SymbolState.DEWEIGHTED

        # Review 3: still severe (2nd consecutive) → quarantined
        now += timedelta(weeks=1)
        engine.run_review(metrics, review_time=now)
        assert engine.records["BTCUSDT"].state == SymbolState.QUARANTINED

    def test_deweighted_to_quarantined_extreme_dd(self, default_gov_cfg, base_weights):
        """Extreme drawdown triggers immediate quarantine from deweighted."""
        engine = SymbolGovernanceEngine(default_gov_cfg, base_weights)
        now = datetime(2026, 1, 6, tzinfo=timezone.utc)

        # First: degrade to deweighted
        metrics = {sym: _healthy_metrics(sym) for sym in base_weights}
        metrics["DOGEUSDT"] = _degraded_metrics("DOGEUSDT")
        engine.run_review(metrics, review_time=now)
        assert engine.records["DOGEUSDT"].state == SymbolState.DEWEIGHTED

        # Then: extreme DD → quarantined
        now += timedelta(weeks=1)
        metrics["DOGEUSDT"] = _extreme_dd_metrics("DOGEUSDT")
        engine.run_review(metrics, review_time=now)
        assert engine.records["DOGEUSDT"].state == SymbolState.QUARANTINED

    def test_deweighted_recovery_to_active(self, default_gov_cfg, base_weights):
        """Recovery requires consecutive_reviews_for_recovery (2) reviews."""
        engine = SymbolGovernanceEngine(default_gov_cfg, base_weights)
        now = datetime(2026, 1, 6, tzinfo=timezone.utc)

        # Degrade
        metrics = {sym: _healthy_metrics(sym) for sym in base_weights}
        metrics["ETHUSDT"] = _degraded_metrics("ETHUSDT")
        engine.run_review(metrics, review_time=now)
        assert engine.records["ETHUSDT"].state == SymbolState.DEWEIGHTED

        # Recovery review 1
        now += timedelta(weeks=1)
        metrics["ETHUSDT"] = _recovery_metrics("ETHUSDT")
        engine.run_review(metrics, review_time=now)
        # Still deweighted after only 1 recovery review
        assert engine.records["ETHUSDT"].state == SymbolState.DEWEIGHTED

        # Recovery review 2
        now += timedelta(weeks=1)
        engine.run_review(metrics, review_time=now)
        # Now recovered
        assert engine.records["ETHUSDT"].state == SymbolState.ACTIVE

    def test_quarantined_recovery_to_deweighted(self, default_gov_cfg, base_weights):
        """Quarantined → deweighted (after min days met + gate pass)."""
        engine = SymbolGovernanceEngine(default_gov_cfg, base_weights)
        now = datetime(2026, 1, 6, tzinfo=timezone.utc)

        # Get to quarantine (via DD path)
        metrics = {sym: _healthy_metrics(sym) for sym in base_weights}
        metrics["SOLUSDT"] = _degraded_metrics("SOLUSDT")
        engine.run_review(metrics, review_time=now)
        now += timedelta(weeks=1)
        metrics["SOLUSDT"] = _extreme_dd_metrics("SOLUSDT")
        engine.run_review(metrics, review_time=now)
        assert engine.records["SOLUSDT"].state == SymbolState.QUARANTINED

        # Wait 15 days (> 14 min) and provide good metrics
        now += timedelta(days=15)
        metrics["SOLUSDT"] = _quarantine_recovery_metrics("SOLUSDT")
        engine.run_review(metrics, review_time=now)
        assert engine.records["SOLUSDT"].state == SymbolState.DEWEIGHTED

    def test_full_lifecycle(self, default_gov_cfg, base_weights):
        """active → deweighted → quarantined → deweighted → active"""
        engine = SymbolGovernanceEngine(default_gov_cfg, base_weights)
        sym = "BTCUSDT"
        now = datetime(2026, 1, 6, tzinfo=timezone.utc)

        def review(m, t):
            metrics = {s: _healthy_metrics(s) for s in base_weights}
            metrics[sym] = m
            return engine.run_review(metrics, review_time=t)

        # 1. active → deweighted
        review(_degraded_metrics(sym), now)
        assert engine.records[sym].state == SymbolState.DEWEIGHTED

        # 2. deweighted → quarantined (need 2 severe reviews while deweighted)
        now += timedelta(weeks=1)
        review(_severely_degraded_metrics(sym), now)  # consecutive_negative_sharpe=1
        now += timedelta(weeks=1)
        review(_severely_degraded_metrics(sym), now)  # consecutive_negative_sharpe=2 → quarantine
        assert engine.records[sym].state == SymbolState.QUARANTINED

        # 3. quarantined → deweighted (after min days)
        now += timedelta(days=15)
        review(_quarantine_recovery_metrics(sym), now)
        assert engine.records[sym].state == SymbolState.DEWEIGHTED

        # 4. deweighted → active (2 consecutive recovery reviews)
        now += timedelta(weeks=1)
        review(_recovery_metrics(sym), now)
        now += timedelta(weeks=1)
        review(_recovery_metrics(sym), now)
        assert engine.records[sym].state == SymbolState.ACTIVE


# ═══════════════════════════════════════════════════════════════
#  2. Quarantine Minimum Days
# ═══════════════════════════════════════════════════════════════

class TestQuarantineMinDays:

    def test_cannot_exit_quarantine_before_min_days(self, default_gov_cfg, base_weights):
        engine = SymbolGovernanceEngine(default_gov_cfg, base_weights)
        sym = "BTCUSDT"
        now = datetime(2026, 1, 6, tzinfo=timezone.utc)

        # Get into quarantine (degrade first → deweighted, then extreme DD → quarantine)
        metrics = {s: _healthy_metrics(s) for s in base_weights}
        metrics[sym] = _degraded_metrics(sym)
        engine.run_review(metrics, review_time=now)  # → deweighted
        now += timedelta(weeks=1)
        metrics[sym] = _extreme_dd_metrics(sym)
        engine.run_review(metrics, review_time=now)  # → quarantined
        assert engine.records[sym].state == SymbolState.QUARANTINED

        # Try to exit after only 7 days (< 14 min)
        now += timedelta(days=7)
        metrics[sym] = _quarantine_recovery_metrics(sym)
        decision = engine.run_review(metrics, review_time=now)
        assert engine.records[sym].state == SymbolState.QUARANTINED
        assert any("quarantine_min_not_met" in r for r in
                    decision.symbols[sym].get("reason_codes", []))


# ═══════════════════════════════════════════════════════════════
#  3. Weight Normalization
# ═══════════════════════════════════════════════════════════════

class TestWeightNormalization:

    def test_all_active_weights_sum_to_one(self, default_gov_cfg, base_weights):
        engine = SymbolGovernanceEngine(default_gov_cfg, base_weights)
        metrics = {sym: _healthy_metrics(sym) for sym in base_weights}
        engine.run_review(metrics)

        eff = engine.get_effective_weights()
        total = sum(eff.values())
        assert abs(total - 1.0) < 1e-6, f"Weights sum to {total}, expected 1.0"

    def test_quarantined_gets_zero_weight(self, default_gov_cfg, base_weights):
        engine = SymbolGovernanceEngine(default_gov_cfg, base_weights)
        sym = "DOGEUSDT"
        now = datetime(2026, 1, 6, tzinfo=timezone.utc)

        # Get DOGEUSDT quarantined
        metrics = {s: _healthy_metrics(s) for s in base_weights}
        metrics[sym] = _degraded_metrics(sym)
        engine.run_review(metrics, review_time=now)
        now += timedelta(weeks=1)
        metrics[sym] = _extreme_dd_metrics(sym)
        engine.run_review(metrics, review_time=now)
        assert engine.records[sym].state == SymbolState.QUARANTINED

        eff = engine.get_effective_weights()
        assert eff[sym] == 0.0

        # Remaining weights still sum to 1.0
        remaining = sum(w for s, w in eff.items() if s != sym)
        assert abs(remaining - 1.0) < 1e-6

    def test_deweighted_has_lower_effective_weight(self):
        """Deweighted symbol gets lower effective weight than same symbol when active.
        Use wider max_weight to avoid 4-symbol clipping artefact."""
        gov_cfg = SymbolGovernanceConfig(
            enabled=True,
            weights=SymbolGovernanceWeights(
                active_multiplier=1.0,
                deweight_multiplier=0.5,
                quarantine_multiplier=0.0,
                min_weight=0.01,
                max_weight=0.50,  # wider max so renormalization doesn't erase diff
            ),
        )
        weights = {
            "AAAUSDT": 0.10,
            "BBBUSDT": 0.10,
            "CCCUSDT": 0.10,
            "DDDUSDT": 0.10,
            "EEEUSDT": 0.10,
            "FFFUSDT": 0.10,
            "GGGUSDT": 0.10,
            "HHHUSDT": 0.10,
            "IIIUSDT": 0.10,
            "JJJUSDT": 0.10,
        }

        engine = SymbolGovernanceEngine(gov_cfg, weights)
        metrics = {sym: _healthy_metrics(sym) for sym in weights}

        # All active first
        engine.run_review(metrics)
        eff_active = engine.get_effective_weights()
        aaa_active = eff_active["AAAUSDT"]

        # Deweight AAAUSDT
        metrics["AAAUSDT"] = _degraded_metrics("AAAUSDT")
        engine.run_review(metrics)
        eff_deweighted = engine.get_effective_weights()
        aaa_deweighted = eff_deweighted["AAAUSDT"]

        assert aaa_deweighted < aaa_active

    def test_min_weight_enforced(self, default_gov_cfg):
        """With many symbols and one deweighted, min_weight floor should hold."""
        symbols = {f"SYM{i}USDT": 1.0 / 20 for i in range(20)}
        engine = SymbolGovernanceEngine(default_gov_cfg, symbols)
        metrics = {sym: _healthy_metrics(sym) for sym in symbols}
        engine.run_review(metrics)

        eff = engine.get_effective_weights()
        for sym, w in eff.items():
            if w > 0:
                assert w >= default_gov_cfg.weights.min_weight - 1e-6

    def test_all_quarantined_returns_zero_weights(self, default_gov_cfg, base_weights):
        """When ALL symbols are quarantined, weights must be all-zero (not equal)."""
        engine = SymbolGovernanceEngine(default_gov_cfg, base_weights)
        now = datetime(2026, 1, 6, tzinfo=timezone.utc)

        # Quarantine every symbol via extreme DD path:
        #   Step 1: degrade all → deweighted
        #   Step 2: extreme DD all → quarantined
        metrics = {sym: _degraded_metrics(sym) for sym in base_weights}
        engine.run_review(metrics, review_time=now)
        for sym in base_weights:
            assert engine.records[sym].state == SymbolState.DEWEIGHTED

        now += timedelta(weeks=1)
        metrics = {sym: _extreme_dd_metrics(sym) for sym in base_weights}
        engine.run_review(metrics, review_time=now)
        for sym in base_weights:
            assert engine.records[sym].state == SymbolState.QUARANTINED

        eff = engine.get_effective_weights()
        for sym, w in eff.items():
            assert w == 0.0, f"{sym} should be 0.0 but got {w}"
        assert sum(eff.values()) == 0.0

    def test_max_weight_enforced(self, default_gov_cfg):
        """No symbol should exceed max_weight (with enough symbols to be feasible)."""
        # Use 6 symbols: 6 * 0.20 = 1.20 > 1.0 → max_weight is constraining
        symbols = {
            "SYM1USDT": 0.30,
            "SYM2USDT": 0.25,
            "SYM3USDT": 0.15,
            "SYM4USDT": 0.10,
            "SYM5USDT": 0.10,
            "SYM6USDT": 0.10,
        }
        engine = SymbolGovernanceEngine(default_gov_cfg, symbols)
        metrics = {sym: _healthy_metrics(sym) for sym in symbols}
        engine.run_review(metrics)

        eff = engine.get_effective_weights()
        for sym, w in eff.items():
            assert w <= default_gov_cfg.weights.max_weight + 1e-6


# ═══════════════════════════════════════════════════════════════
#  4. Disabled Governance (Backward Compatibility)
# ═══════════════════════════════════════════════════════════════

class TestDisabledGovernance:

    def test_apply_governance_weights_disabled(self, base_weights):
        """When disabled, apply_governance_weights returns base_weights unchanged."""
        disabled_cfg = SymbolGovernanceConfig(enabled=False)
        result = apply_governance_weights(base_weights, disabled_cfg)
        assert result == base_weights

    def test_apply_governance_weights_no_artifact(self, base_weights, tmp_path):
        """When enabled but no artifact exists, returns base_weights."""
        enabled_cfg = SymbolGovernanceConfig(
            enabled=True, artifacts_dir=str(tmp_path / "nonexistent")
        )
        result = apply_governance_weights(base_weights, enabled_cfg, str(tmp_path / "nonexistent"))
        assert result == base_weights


# ═══════════════════════════════════════════════════════════════
#  5. Determinism
# ═══════════════════════════════════════════════════════════════

class TestDeterminism:

    def test_same_inputs_same_outputs(self, default_gov_cfg, base_weights):
        now = datetime(2026, 2, 17, tzinfo=timezone.utc)
        metrics = {sym: _healthy_metrics(sym) for sym in base_weights}
        metrics["BTCUSDT"] = _degraded_metrics("BTCUSDT")

        engine1 = SymbolGovernanceEngine(default_gov_cfg, base_weights)
        d1 = engine1.run_review(metrics, review_time=now, dry_run=True)

        engine2 = SymbolGovernanceEngine(default_gov_cfg, base_weights)
        d2 = engine2.run_review(metrics, review_time=now, dry_run=True)

        assert d1.to_dict() == d2.to_dict()

    def test_dry_run_does_not_mutate_state(self, default_gov_cfg, base_weights):
        engine = SymbolGovernanceEngine(default_gov_cfg, base_weights)
        metrics = {sym: _healthy_metrics(sym) for sym in base_weights}
        metrics["BTCUSDT"] = _degraded_metrics("BTCUSDT")

        # State before
        state_before = engine.get_records_list()

        # Dry run
        engine.run_review(metrics, dry_run=True)

        # State should be unchanged
        state_after = engine.get_records_list()
        assert state_before == state_after


# ═══════════════════════════════════════════════════════════════
#  6. Artifact Persistence
# ═══════════════════════════════════════════════════════════════

class TestArtifactPersistence:

    def test_persist_creates_files(self, default_gov_cfg, base_weights, tmp_path):
        engine = SymbolGovernanceEngine(default_gov_cfg, base_weights)
        metrics = {sym: _healthy_metrics(sym) for sym in base_weights}
        decision = engine.run_review(metrics)

        latest_path, history_path = engine.persist(decision, tmp_path)

        assert latest_path.exists()
        assert history_path.exists()

        # Validate JSON structure
        with open(latest_path) as f:
            data = json.load(f)
        assert "timestamp" in data
        assert "summary" in data
        assert "records" in data

        # History should have exactly one line
        with open(history_path) as f:
            lines = f.readlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert "timestamp" in entry

    def test_load_latest_decisions(self, default_gov_cfg, base_weights, tmp_path):
        engine = SymbolGovernanceEngine(default_gov_cfg, base_weights)
        metrics = {sym: _healthy_metrics(sym) for sym in base_weights}
        decision = engine.run_review(metrics)
        engine.persist(decision, tmp_path)

        loaded = load_latest_decisions(tmp_path)
        assert loaded is not None
        assert loaded["summary"]["active"] == len(base_weights)

    def test_state_reload_preserves_records(self, default_gov_cfg, base_weights, tmp_path):
        """Persist → reload → same state."""
        engine = SymbolGovernanceEngine(default_gov_cfg, base_weights)
        metrics = {sym: _healthy_metrics(sym) for sym in base_weights}
        metrics["ETHUSDT"] = _degraded_metrics("ETHUSDT")
        decision = engine.run_review(metrics)
        engine.persist(decision, tmp_path)

        # New engine, load state
        engine2 = SymbolGovernanceEngine(default_gov_cfg, base_weights)
        loaded = load_latest_decisions(tmp_path)
        engine2.load_state(loaded.get("records"))

        assert engine2.records["ETHUSDT"].state == SymbolState.DEWEIGHTED
        assert engine2.records["BTCUSDT"].state == SymbolState.ACTIVE


# ═══════════════════════════════════════════════════════════════
#  7. Metrics Computation
# ═══════════════════════════════════════════════════════════════

class TestMetricsComputation:

    def test_sharpe_positive_returns(self):
        m = SymbolMetrics(
            symbol="BTCUSDT",
            net_pnl=100,
            turnover=10000,
            returns_series=[0.001] * 100,
            model_slippage_bps=3.0,
            realized_slippage_bps=2.0,
        ).compute()
        # Constant positive returns → very high Sharpe
        assert m.edge_sharpe_4w > 10

    def test_sharpe_negative_returns(self):
        m = SymbolMetrics(
            symbol="BTCUSDT",
            net_pnl=-100,
            turnover=10000,
            returns_series=[-0.002] * 100,
            model_slippage_bps=3.0,
            realized_slippage_bps=5.0,
        ).compute()
        assert m.edge_sharpe_4w < 0

    def test_edge_per_turnover(self):
        m = SymbolMetrics(
            symbol="BTCUSDT",
            net_pnl=50,
            turnover=100000,
        ).compute()
        assert abs(m.edge_per_turnover_4w - 50 / 100000) < 1e-10

    def test_slippage_ratio(self):
        m = SymbolMetrics(
            symbol="BTCUSDT",
            realized_slippage_bps=6.0,
            model_slippage_bps=3.0,
        ).compute()
        assert abs(m.slippage_ratio_4w - 2.0) < 1e-10


# ═══════════════════════════════════════════════════════════════
#  8. Recovery Counter Reset
# ═══════════════════════════════════════════════════════════════

class TestRecoveryCounterReset:

    def test_recovery_counter_resets_on_bad_review(self, default_gov_cfg, base_weights):
        """If one recovery review is good but next is bad, counter resets."""
        engine = SymbolGovernanceEngine(default_gov_cfg, base_weights)
        sym = "ETHUSDT"
        now = datetime(2026, 1, 6, tzinfo=timezone.utc)

        # Degrade
        metrics = {s: _healthy_metrics(s) for s in base_weights}
        metrics[sym] = _degraded_metrics(sym)
        engine.run_review(metrics, review_time=now)
        assert engine.records[sym].state == SymbolState.DEWEIGHTED

        # Recovery review 1 (good)
        now += timedelta(weeks=1)
        metrics[sym] = _recovery_metrics(sym)
        engine.run_review(metrics, review_time=now)
        assert engine.records[sym].consecutive_recovery_reviews == 1

        # Bad review (counter should reset)
        now += timedelta(weeks=1)
        metrics[sym] = _degraded_metrics(sym)
        engine.run_review(metrics, review_time=now)
        assert engine.records[sym].consecutive_recovery_reviews == 0

        # Still deweighted
        assert engine.records[sym].state == SymbolState.DEWEIGHTED
