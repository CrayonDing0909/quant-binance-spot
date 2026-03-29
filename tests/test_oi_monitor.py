"""Unit tests for OI liquidation cascade detection."""
from __future__ import annotations

import pandas as pd
import pytest

from qtrade.polymarket.oi_monitor import (
    CascadeSignal,
    compute_oi_change,
    compute_price_change,
    detect_cascade,
)


def _make_oi_series(values: list[float], freq: str = "1h") -> pd.Series:
    """Helper to create OI series."""
    idx = pd.date_range("2025-01-01", periods=len(values), freq=freq, tz="UTC")
    return pd.Series(values, index=idx)


def _make_price_series(values: list[float], freq: str = "1h") -> pd.Series:
    """Helper to create price series."""
    idx = pd.date_range("2025-01-01", periods=len(values), freq=freq, tz="UTC")
    return pd.Series(values, index=idx)


class TestComputeOiChange:
    def test_no_change(self):
        oi = _make_oi_series([100.0] * 30)
        change_24h, change_1h = compute_oi_change(oi)
        assert change_24h == 0.0
        assert change_1h == 0.0

    def test_10_percent_drop(self):
        values = [100.0] * 25 + [90.0]
        oi = _make_oi_series(values)
        change_24h, change_1h = compute_oi_change(oi, lookback_24h=24)
        assert abs(change_24h - (-0.10)) < 0.01

    def test_too_short_series(self):
        oi = _make_oi_series([100.0, 90.0])
        change_24h, change_1h = compute_oi_change(oi)
        assert change_24h == 0.0


class TestComputePriceChange:
    def test_no_change(self):
        prices = _make_price_series([50000.0] * 800)
        change = compute_price_change(prices)
        assert change == 0.0

    def test_20_percent_drop(self):
        values = [50000.0] * 721 + [40000.0]
        prices = _make_price_series(values)
        change = compute_price_change(prices, lookback=720)
        assert abs(change - (-0.20)) < 0.01


class TestDetectCascade:
    def test_no_cascade_normal_market(self):
        oi = _make_oi_series([100.0] * 30)
        prices = _make_price_series([50000.0] * 800)
        signal = detect_cascade("BTCUSDT", oi, prices)
        assert signal is None

    def test_cascade_detected(self):
        # OI drops 12% over 24h, stabilized in last hour
        values = [100.0] * 24 + [88.0, 88.0]
        oi = _make_oi_series(values)
        prices = _make_price_series([50000.0] * 800)
        signal = detect_cascade("BTCUSDT", oi, prices)
        assert signal is not None
        assert signal.direction == "long"
        assert signal.confidence == "normal"
        assert signal.is_valid

    def test_cascade_high_confidence(self):
        # OI drops 18%
        values = [100.0] * 24 + [82.0, 82.0]
        oi = _make_oi_series(values)
        prices = _make_price_series([50000.0] * 800)
        signal = detect_cascade("BTCUSDT", oi, prices)
        assert signal is not None
        assert signal.confidence == "high"

    def test_cascade_rejected_bear_market(self):
        # OI drops but we're in extreme bear (-25% in 30d)
        values = [100.0] * 24 + [88.0, 88.0]
        oi = _make_oi_series(values)
        prices = _make_price_series([50000.0] * 720 + [37500.0] * 80)
        signal = detect_cascade("BTCUSDT", oi, prices)
        # Signal is created but is_valid should be False
        assert signal is None or not signal.is_valid

    def test_cascade_rejected_still_falling(self):
        # OI still falling (1h change < -1%)
        values = [100.0] * 24 + [89.0, 87.0]  # still dropping
        oi = _make_oi_series(values)
        prices = _make_price_series([50000.0] * 800)
        signal = detect_cascade("BTCUSDT", oi, prices)
        if signal is not None:
            # OI 1h change is ~-2.2%, which is < -1% → should be invalid
            assert not signal.is_valid


class TestCascadeSignal:
    def test_is_valid_all_conditions_met(self):
        sig = CascadeSignal(
            symbol="BTCUSDT",
            oi_change_24h=-0.12,
            oi_change_1h=0.001,
            price_change_30d=-0.05,
            current_price=85000,
            timestamp=pd.Timestamp("2025-03-01", tz="UTC"),
            direction="long",
            confidence="normal",
        )
        assert sig.is_valid

    def test_is_invalid_oi_still_falling(self):
        sig = CascadeSignal(
            symbol="BTCUSDT",
            oi_change_24h=-0.12,
            oi_change_1h=-0.02,  # still falling
            price_change_30d=-0.05,
            current_price=85000,
            timestamp=pd.Timestamp("2025-03-01", tz="UTC"),
            direction="long",
            confidence="normal",
        )
        assert not sig.is_valid

    def test_is_invalid_bear_market(self):
        sig = CascadeSignal(
            symbol="BTCUSDT",
            oi_change_24h=-0.12,
            oi_change_1h=0.001,
            price_change_30d=-0.25,  # extreme bear
            current_price=40000,
            timestamp=pd.Timestamp("2025-03-01", tz="UTC"),
            direction="long",
            confidence="normal",
        )
        assert not sig.is_valid
