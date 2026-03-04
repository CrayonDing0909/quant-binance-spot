from __future__ import annotations

import numpy as np
import pandas as pd

from qtrade.validation.ic_monitor import RollingICMonitor


def _make_price_series(n: int = 3000, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0002, 0.01, n)
    price = 100 * (1 + pd.Series(rets)).cumprod()
    price.index = pd.date_range("2022-01-01", periods=n, freq="h")
    return price


class TestRollingICMonitor:
    def test_compute_returns_non_empty_report(self):
        price = _make_price_series()
        monitor = RollingICMonitor(window=24 * 60, forward_bars=24, recent_days=60, interval="1h")

        fwd = price.pct_change(24).shift(-24)
        signal = np.sign(fwd).fillna(0.0)

        report = monitor.compute(signal, price)

        assert report.signal_count > 100
        assert -1.0 <= report.overall_ic <= 1.0
        assert report.ic_std >= 0.0
        assert isinstance(report.yearly_ic, dict)

    def test_decay_alert_triggers_when_recent_ic_collapses(self):
        price = _make_price_series(n=4500, seed=123)
        monitor = RollingICMonitor(
            window=24 * 90,
            forward_bars=24,
            recent_days=45,
            decay_threshold=0.3,
            interval="1h",
        )

        fwd = price.pct_change(24).shift(-24)
        signal = np.sign(fwd).fillna(0.0)

        # Make recent signal intentionally anti-correlated to force decay.
        recent_bars = 24 * 45
        signal.iloc[-recent_bars:] = -signal.iloc[-recent_bars:]

        report = monitor.compute(signal, price)
        alerts = monitor.check_alerts(report)

        assert report.is_decaying
        assert any(a.metric == "ic_decay_pct" and a.severity == "critical" for a in alerts)

    def test_insufficient_observations_returns_empty_report(self):
        idx = pd.date_range("2024-01-01", periods=50, freq="h")
        price = pd.Series(np.linspace(100, 110, 50), index=idx)
        signal = pd.Series(np.where(np.arange(50) % 2 == 0, 1.0, 0.0), index=idx)

        monitor = RollingICMonitor(window=24 * 10, forward_bars=24, min_observations=200, interval="1h")
        report = monitor.compute(signal, price)

        assert report.signal_count == 0
        assert report.overall_ic == 0.0
        assert report.overall_ic_pvalue == 1.0

    def test_small_denominator_guard_prevents_ratio_explosion(self):
        """When historical IC is near zero, decay should use absolute diff, not ratio."""
        price = _make_price_series(n=4000, seed=99)
        monitor = RollingICMonitor(
            window=24 * 60,
            forward_bars=24,
            recent_days=60,
            decay_threshold=0.5,
            interval="1h",
            min_ic_denominator=0.01,
        )

        # Create a nearly random signal → historical IC ≈ 0
        rng = np.random.default_rng(99)
        signal = pd.Series(
            rng.choice([-1.0, 0.0, 1.0], size=len(price), p=[0.3, 0.4, 0.3]),
            index=price.index,
        )

        report = monitor.compute(signal, price)

        # With near-zero historical IC, the old ratio formula would produce
        # absurd values like +240% or +1500%.  The guard should keep
        # ic_decay_pct as a small absolute number (< 1.0).
        assert abs(report.ic_decay_pct) < 1.0, (
            f"Small-denominator guard failed: ic_decay_pct={report.ic_decay_pct}, "
            f"historical_ic={report.historical_ic}, recent_ic={report.recent_ic}"
        )

    def test_min_ic_denominator_parameter_accepted(self):
        """Verify the new min_ic_denominator parameter is accepted."""
        monitor = RollingICMonitor(min_ic_denominator=0.05)
        assert monitor.min_ic_denominator == 0.05

        monitor2 = RollingICMonitor()  # default
        assert monitor2.min_ic_denominator == 0.01
