"""Unit tests for Krajekis 5-layer strategy."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from qtrade.polymarket.binance_feed import TASignals
from qtrade.polymarket.krajekis_strategy import (
    determine_session,
    classify_volatility,
    determine_direction,
    evaluate_15m_window,
)
from qtrade.polymarket.market_discovery import Market15m


def _make_ta(**overrides) -> TASignals:
    """Create TASignals with sensible defaults."""
    defaults = dict(
        close=85000, vwap=84900, ema_21=85100, ema_50=84800,
        rsi_14=55, macd_hist=10, macd_hist_prev=5, atr_14=50,
        volume_ratio=1.1,
    )
    defaults.update(overrides)
    return TASignals(**defaults)


def _make_market(**overrides) -> Market15m:
    """Create Market15m with defaults. End date is 7 minutes from now."""
    from datetime import timedelta
    end = datetime.now(timezone.utc) + timedelta(minutes=7)
    defaults = dict(
        slug="btc-updown-15m-123", question="BTC Up?",
        end_date=end.isoformat(),
        token_id_up="token_up_123", token_id_down="token_down_123",
        price_up=0.50, price_down=0.50,
        accepting_orders=True, condition_id="cond_123",
    )
    defaults.update(overrides)
    return Market15m(**defaults)


class TestDetermineSession:
    def test_asia(self):
        dt = datetime(2026, 3, 30, 3, 0, tzinfo=timezone.utc)
        assert determine_session(dt) == "asia"

    def test_london_kill(self):
        dt = datetime(2026, 3, 30, 8, 0, tzinfo=timezone.utc)
        assert determine_session(dt) == "london_kill"

    def test_ny_open(self):
        dt = datetime(2026, 3, 30, 14, 0, tzinfo=timezone.utc)
        assert determine_session(dt) == "ny_open"

    def test_weekend(self):
        dt = datetime(2026, 3, 28, 14, 0, tzinfo=timezone.utc)  # Saturday
        assert determine_session(dt) == "weekend"


class TestClassifyVolatility:
    def test_low(self):
        assert classify_volatility(30, low_threshold=40, high_threshold=100) == "low"

    def test_high(self):
        assert classify_volatility(150, low_threshold=40, high_threshold=100) == "high"

    def test_medium(self):
        assert classify_volatility(70, low_threshold=40, high_threshold=100) == "medium"


class TestDetermineDirection:
    def test_trend_bullish_ny(self):
        ta = _make_ta(
            ema_21=85100, ema_50=84800, close=85200, vwap=84900,
            macd_hist=15, macd_hist_prev=10, volume_ratio=1.3,
        )
        assert determine_direction(ta, "ny_open") == "up"

    def test_trend_bearish_ny(self):
        ta = _make_ta(
            ema_21=84500, ema_50=85000, close=84300, vwap=84800,
            macd_hist=-15, macd_hist_prev=-10, volume_ratio=1.3,
        )
        assert determine_direction(ta, "ny_open") == "down"

    def test_no_signal_ny_conflicting(self):
        ta = _make_ta(
            ema_21=85000, ema_50=85000, close=85000, vwap=85000,
            macd_hist=1, macd_hist_prev=2, volume_ratio=0.8,
        )
        assert determine_direction(ta, "ny_open") is None

    def test_mean_reversion_overbought_asia(self):
        ta = _make_ta(
            rsi_14=75, macd_hist=5, macd_hist_prev=10,  # exhaustion
            close=85500, vwap=85000,  # far above VWAP
        )
        assert determine_direction(ta, "asia") == "down"

    def test_mean_reversion_oversold_asia(self):
        ta = _make_ta(
            rsi_14=25, macd_hist=-5, macd_hist_prev=-10,  # exhaustion
            close=84000, vwap=84500,  # far below VWAP
        )
        assert determine_direction(ta, "asia") == "up"

    def test_london_kill_sweep_reversal(self):
        ta = _make_ta(
            rsi_14=28, macd_hist=3, macd_hist_prev=1,  # reversing up after oversold
        )
        assert determine_direction(ta, "london_kill") == "up"


class TestEvaluate15mWindow:
    def test_trend_follow_low_vol(self):
        """Low vol + NY trend → buy expensive side."""
        market = _make_market(price_up=0.75, price_down=0.25)
        ta = _make_ta(
            ema_21=85100, ema_50=84800, close=85200, vwap=84900,
            macd_hist=15, macd_hist_prev=10, volume_ratio=1.3,
        )
        setup = evaluate_15m_window(
            market, ta, session="ny_open", vol_regime="low",
            sweet_spot_start=900, sweet_spot_end=60,
        )
        assert setup is not None
        assert setup.side == "up"
        assert setup.scenario == "trend_follow"
        assert setup.price_target == 0.75

    def test_mean_reversion_high_vol(self):
        """High vol + Asia oversold → buy cheap UP side."""
        market = _make_market(price_up=0.15, price_down=0.85)
        ta = _make_ta(
            rsi_14=25, macd_hist=-2, macd_hist_prev=-5,
            close=84000, vwap=84500,
        )
        setup = evaluate_15m_window(
            market, ta, session="asia", vol_regime="high",
            sweet_spot_start=900, sweet_spot_end=60,
        )
        assert setup is not None
        assert setup.side == "up"
        assert setup.scenario == "mean_reversion"
        assert setup.price_target == 0.15

    def test_skip_outside_sweet_spot(self):
        """Too early in window → no trade."""
        from datetime import timedelta
        far_end = datetime.now(timezone.utc) + timedelta(minutes=14)
        market = _make_market(end_date=far_end.isoformat())
        ta = _make_ta()
        setup = evaluate_15m_window(
            market, ta, session="ny_open", vol_regime="low",
            sweet_spot_start=600, sweet_spot_end=120,
        )
        # With 15 min remaining (900s), outside sweet spot (600s)
        assert setup is None

    def test_skip_no_direction(self):
        """Conflicting TA → no trade."""
        market = _make_market(price_up=0.50, price_down=0.50)
        ta = _make_ta(
            ema_21=85000, ema_50=85000, close=85000, vwap=85000,
            macd_hist=0, macd_hist_prev=0, rsi_14=50,
        )
        setup = evaluate_15m_window(
            market, ta, session="asia", vol_regime="low",
            sweet_spot_start=900, sweet_spot_end=60,
        )
        assert setup is None

    def test_confidence_scaling(self):
        """Multiple aligned signals → high confidence → larger bet."""
        market = _make_market(price_up=0.80, price_down=0.20)
        ta = _make_ta(
            ema_21=85200, ema_50=84800, close=85300, vwap=84900,
            macd_hist=20, macd_hist_prev=10, rsi_14=40,
            volume_ratio=1.5,
        )
        setup = evaluate_15m_window(
            market, ta, session="ny_open", vol_regime="low",
            bet_size=1.0, sweet_spot_start=900, sweet_spot_end=60,
        )
        assert setup is not None
        assert setup.confidence == "high"
        assert setup.size_usdc == 1.5  # 1.0 × 1.5
