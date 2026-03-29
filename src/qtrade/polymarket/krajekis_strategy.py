"""
Krajekis 5-Layer Strategy for Polymarket 15-Minute Markets.

Source: @krajekis (via @lqp2021), organized in
    docs/research/polymarket_btc_5_15min_strategy.md

5 Layers:
    1. Session → determines strategy bias (trend vs mean-reversion)
    2. Volatility → ATR classifies low/high → determines price target
    3. Price Structure → VWAP/EMA/RSI/MACD → determines direction
    4. Entry Timing → sweet spot at 5-10 min remaining
    5. Risk Management → daily loss limit

All functions are pure (no side effects) — easy to unit test.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from qtrade.polymarket.binance_feed import TASignals
from qtrade.polymarket.market_discovery import Market15m

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TradeSetup:
    """A complete trade setup from the 5-layer analysis."""

    side: str               # "up" or "down"
    token_id: str           # Polymarket CLOB token ID
    price_target: float     # what price to place limit order at
    size_usdc: float        # bet amount
    scenario: str           # "trend_follow" or "mean_reversion"
    confidence: str         # "high", "normal", "low"
    reason: str             # human-readable explanation
    session: str
    vol_regime: str


# ═══════════════════════════════════════════════════════════
#  Layer 1: Session Logic
# ═══════════════════════════════════════════════════════════

def determine_session(utc_now: datetime | None = None) -> str:
    """
    Classify current time into trading session.

    All sessions defined in UTC (converted from EST in the original doc).
    Sessions can overlap — we return the highest-priority one.

    Returns: "asia", "london_kill", "ny_open", "london_close", "weekend"
    """
    if utc_now is None:
        utc_now = datetime.now(timezone.utc)

    # Weekend check first
    if utc_now.weekday() >= 5:  # Saturday=5, Sunday=6
        return "weekend"

    hour = utc_now.hour

    # NY Open/Overlap: 12:00-16:00 UTC (07:00-11:00 EST) — highest priority
    if 12 <= hour < 16:
        return "ny_open"

    # London Kill Zone: 07:00-10:00 UTC (02:00-05:00 EST)
    if 7 <= hour < 10:
        return "london_kill"

    # London Close: 15:00-17:00 UTC (10:00-12:00 EST)
    if 15 <= hour < 17:
        return "london_close"

    # Asia: 00:00-08:00 UTC (19:00-03:00 EST)
    if hour < 8:
        return "asia"

    # Default (off-hours)
    return "asia"


# Session → strategy bias mapping
SESSION_BIAS = {
    "asia":         "mean_reversion",   # range-bound, low vol, fade extremes
    "london_kill":  "wait_for_sweep",   # wait for stop hunt, then follow reversal
    "ny_open":      "trend_follow",     # max volatility, follow confirmed breakouts
    "london_close": "mean_reversion",   # trend exhaustion, fade extensions
    "weekend":      "mean_reversion",   # thin liquidity, prefer VWAP reversion
}


# ═══════════════════════════════════════════════════════════
#  Layer 2: Volatility Regime
# ═══════════════════════════════════════════════════════════

def classify_volatility(
    atr_14: float,
    low_threshold: float = 40.0,
    high_threshold: float = 100.0,
) -> str:
    """
    Classify volatility regime based on 15-minute ATR.

    Returns: "low", "medium", "high"

    Low vol → buy expensive side (70-95¢), high win rate
    High vol → buy cheap side (5-25¢), low win rate but big payoff
    """
    if atr_14 <= low_threshold:
        return "low"
    elif atr_14 >= high_threshold:
        return "high"
    return "medium"


# ═══════════════════════════════════════════════════════════
#  Layer 3: Direction (Price Structure)
# ═══════════════════════════════════════════════════════════

def determine_direction(ta: TASignals, session: str) -> str | None:
    """
    Use TA indicators to determine UP or DOWN bias.

    Rules (from krajekis doc):
        Trend follow (NY open, strong breakout):
            - EMA21 > EMA50 + price > VWAP + MACD expanding → UP
            - EMA21 < EMA50 + price < VWAP + MACD expanding → DOWN

        Mean reversion (Asia, London close, weekend):
            - Price far above VWAP + RSI > 70 + MACD shrinking → DOWN
            - Price far below VWAP + RSI < 30 + MACD shrinking → UP

    Returns: "up", "down", or None (no clear signal)
    """
    bias = SESSION_BIAS.get(session, "mean_reversion")

    if bias == "trend_follow":
        # Strong trend confirmation needed
        if (ta.trend_bullish
                and ta.macd_expanding_bull
                and ta.volume_ratio > 1.0):
            return "up"
        if (ta.trend_bearish
                and ta.macd_expanding_bear
                and ta.volume_ratio > 1.0):
            return "down"
        return None

    elif bias == "mean_reversion":
        # Fade extremes
        if (ta.rsi_overbought
                and ta.macd_exhaustion
                and ta.vwap_distance_pct > 0.1):
            return "down"
        if (ta.rsi_oversold
                and ta.macd_exhaustion
                and ta.vwap_distance_pct < -0.1):
            return "up"

        # Also: simple VWAP reversion if price is extended
        if ta.vwap_distance_pct > 0.3 and not ta.macd_expanding_bull:
            return "down"
        if ta.vwap_distance_pct < -0.3 and not ta.macd_expanding_bear:
            return "up"

        return None

    elif bias == "wait_for_sweep":
        # London Kill Zone: wait for reversal confirmation
        # Only enter if we see a completed sweep (RSI extreme + reversal)
        if ta.rsi_oversold and ta.macd_hist > ta.macd_hist_prev:
            return "up"   # swept lows, now reversing up
        if ta.rsi_overbought and ta.macd_hist < ta.macd_hist_prev:
            return "down"  # swept highs, now reversing down
        return None

    return None


# ═══════════════════════════════════════════════════════════
#  Layer 4 + 5: Entry Timing + Risk → Full Evaluation
# ═══════════════════════════════════════════════════════════

def evaluate_15m_window(
    market: Market15m,
    ta: TASignals,
    session: str,
    vol_regime: str,
    bet_size: float = 1.0,
    # Timing config
    sweet_spot_start: float = 600.0,   # 10 minutes
    sweet_spot_end: float = 120.0,     # 2 minutes
    # Pricing config
    trend_min_price: float = 0.65,
    trend_max_price: float = 0.95,
    mr_min_price: float = 0.02,
    mr_max_price: float = 0.25,
) -> TradeSetup | None:
    """
    Full 5-layer evaluation for a 15-minute market window.

    Pure function — takes all data in, returns decision out.

    Args:
        market: Current Polymarket 15m market
        ta: Computed TA signals from Binance
        session: Current trading session
        vol_regime: "low", "medium", "high"
        bet_size: USDC per trade

    Returns:
        TradeSetup or None if no trade
    """
    if not market.accepting_orders:
        return None

    # ── Layer 4: Timing ──
    remaining = market.time_remaining_seconds()
    if remaining > sweet_spot_start or remaining < sweet_spot_end:
        return None  # outside sweet spot

    # ── Layer 3: Direction ──
    direction = determine_direction(ta, session)
    if direction is None:
        return None

    # ── Layer 2: Pricing based on volatility ──
    if vol_regime == "low" or vol_regime == "medium":
        # Low/medium vol → trend follow → buy the EXPENSIVE side
        scenario = "trend_follow"
        if direction == "up":
            price_target = market.price_up
            token_id = market.token_id_up
        else:
            price_target = market.price_down
            token_id = market.token_id_down

        # Check if price is in acceptable range for trend follow
        if price_target < trend_min_price or price_target > trend_max_price:
            return None

    else:
        # High vol → mean reversion → buy the CHEAP side
        scenario = "mean_reversion"
        if direction == "up":
            price_target = market.price_up
            token_id = market.token_id_up
        else:
            price_target = market.price_down
            token_id = market.token_id_down

        # Check if price is in acceptable range for mean reversion
        if price_target < mr_min_price or price_target > mr_max_price:
            return None

    if not token_id:
        return None

    # ── Confidence ──
    # Higher confidence if multiple indicators align
    signals_aligned = sum([
        ta.trend_bullish if direction == "up" else ta.trend_bearish,
        ta.macd_expanding_bull if direction == "up" else ta.macd_expanding_bear,
        (ta.rsi_14 < 50) if direction == "up" else (ta.rsi_14 > 50),
        ta.volume_ratio > 1.2,
    ])

    if signals_aligned >= 3:
        confidence = "high"
    elif signals_aligned >= 2:
        confidence = "normal"
    else:
        confidence = "low"

    # Scale bet size by confidence
    if confidence == "high":
        adjusted_size = bet_size * 1.5
    elif confidence == "low":
        adjusted_size = bet_size * 0.5
    else:
        adjusted_size = bet_size

    odds = 1.0 / price_target if price_target > 0 else 0
    reason = (
        f"[{session}|{vol_regime}vol|{scenario}] "
        f"{direction.upper()} @ ${price_target:.3f} ({odds:.1f}:1) "
        f"| RSI={ta.rsi_14:.0f} MACD={'↑' if ta.macd_hist > 0 else '↓'} "
        f"VWAP {ta.vwap_distance_pct:+.2f}% "
        f"| {remaining:.0f}s left | conf={confidence}"
    )

    return TradeSetup(
        side=direction,
        token_id=token_id,
        price_target=round(price_target, 4),
        size_usdc=round(adjusted_size, 2),
        scenario=scenario,
        confidence=confidence,
        reason=reason,
        session=session,
        vol_regime=vol_regime,
    )
