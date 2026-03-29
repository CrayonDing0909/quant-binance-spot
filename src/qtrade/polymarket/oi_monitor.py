"""
OI Liquidation Cascade Monitor — Pure functions for detection.

All functions are stateless and take data in, return signals out.
No API calls, no side effects — easy to unit test.

Evidence (2022-2026 backtest):
    OI 24h drop >10% → BTC avg 24h return +1.72% (2024), +1.64% (2025)
    Win rate: 66-82% depending on year
    Causal mechanism: forced liquidation → oversold → mean reversion
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CascadeSignal:
    """Detected liquidation cascade event."""

    symbol: str
    oi_change_24h: float      # e.g. -0.12 = -12%
    oi_change_1h: float       # recent 1h change (for "stopped falling" check)
    price_change_30d: float   # regime filter
    current_price: float
    timestamp: pd.Timestamp
    direction: str            # "long" (buy after crash) or "short" (sell after pump)
    confidence: str           # "high" (>15% drop) or "normal" (>10% drop)

    @property
    def is_valid(self) -> bool:
        """Signal passes all filters."""
        return (
            abs(self.oi_change_24h) >= 0.10
            and self.oi_change_1h > -0.01   # OI stopped falling
            and self.price_change_30d > -0.20  # not extreme bear market
        )


def compute_oi_change(
    oi_series: pd.Series,
    lookback_24h: int = 24,
    lookback_1h: int = 1,
) -> tuple[float, float]:
    """
    Compute OI percentage change over 24h and 1h.

    Args:
        oi_series: Hourly OI values (index = datetime, values = OI)
        lookback_24h: Bars for 24h change (default 24 for 1h candles)
        lookback_1h: Bars for 1h change

    Returns:
        (oi_change_24h, oi_change_1h) as fractions (e.g. -0.12 = -12%)
    """
    if len(oi_series) < lookback_24h + 1:
        return 0.0, 0.0

    current = oi_series.iloc[-1]
    past_24h = oi_series.iloc[-(lookback_24h + 1)]
    past_1h = oi_series.iloc[-(lookback_1h + 1)]

    if past_24h == 0 or past_1h == 0:
        return 0.0, 0.0

    change_24h = (current - past_24h) / past_24h
    change_1h = (current - past_1h) / past_1h

    return float(change_24h), float(change_1h)


def compute_price_change(
    close_series: pd.Series,
    lookback: int = 720,
) -> float:
    """Compute price change over lookback period (default 30 days at 1h)."""
    if len(close_series) < lookback + 1:
        return 0.0

    current = close_series.iloc[-1]
    past = close_series.iloc[-(lookback + 1)]

    if past == 0:
        return 0.0

    return float((current - past) / past)


def detect_cascade(
    symbol: str,
    oi_series: pd.Series,
    close_series: pd.Series,
    oi_drop_threshold: float = -0.10,
    oi_stabilize_threshold: float = -0.01,
    bear_market_threshold: float = -0.20,
) -> CascadeSignal | None:
    """
    Detect a liquidation cascade event.

    Args:
        symbol: e.g. "BTCUSDT"
        oi_series: Hourly OI values
        close_series: Hourly close prices
        oi_drop_threshold: OI 24h drop threshold (default -10%)
        oi_stabilize_threshold: OI 1h change must be above this (stopped falling)
        bear_market_threshold: 30d price change must be above this (regime filter)

    Returns:
        CascadeSignal if cascade detected, None otherwise
    """
    oi_change_24h, oi_change_1h = compute_oi_change(oi_series)
    price_change_30d = compute_price_change(close_series)

    # No cascade
    if oi_change_24h > oi_drop_threshold:
        return None

    confidence = "high" if oi_change_24h < -0.15 else "normal"

    # OI dropped = longs liquidated → price overshot down → buy the bounce
    direction = "long"

    signal = CascadeSignal(
        symbol=symbol,
        oi_change_24h=oi_change_24h,
        oi_change_1h=oi_change_1h,
        price_change_30d=price_change_30d,
        current_price=float(close_series.iloc[-1]),
        timestamp=close_series.index[-1],
        direction=direction,
        confidence=confidence,
    )

    if signal.is_valid:
        return signal

    return None
