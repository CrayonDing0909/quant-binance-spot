"""
Mean Reversion Contrarian Strategy for 15-minute Polymarket markets.

Core logic (pure function, easy to test):
    1. Get current Polymarket odds for Up/Down
    2. If one side is cheap (< max_price threshold):
       → Someone is panicking, crowd overreacting
       → Bet on the cheap side (contrarian)
    3. Use LIMIT ORDERS (maker = 0% fee)

Edge sources:
    - Fee asymmetry: 0% as maker, near-zero at extreme odds even as taker
    - Crowd overreaction: after big moves, crowd extrapolates but price reverts
    - Asymmetric payoff: risk 1U to win 2-25U at extreme odds

Evidence:
    - Friend's backtest: 51-52% WR on 15min → 100U→370U (SOL) over 2025
    - Extreme events (odds 22:1) on 2026/2/6 drove +41.73% in 9 days
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from qtrade.polymarket.market_discovery import Market15m

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BetSignal:
    """A signal to place a bet on Polymarket."""

    market_slug: str
    side: str           # "up" or "down"
    token_id: str
    price: float        # share price we'd buy at
    odds: float         # implied payout ratio
    size_usdc: float    # how much to bet
    reason: str
    confidence: str     # "normal" or "high"


def evaluate_market(
    market: Market15m,
    bet_size: float = 1.0,
    max_price: float = 0.45,
    min_price: float = 0.02,
    min_time_remaining: float = 120.0,
) -> BetSignal | None:
    """
    Evaluate a 15-minute market for contrarian entry.

    Logic:
        - If "Up" is cheap (< max_price) → buy Up (crowd thinks "Down")
        - If "Down" is cheap (< max_price) → buy Down (crowd thinks "Up")
        - If both sides near 50/50 → skip (no edge, high fee)
        - If either side < min_price → skip (suspicious, likely near resolution)

    Args:
        market: Current 15-minute market
        bet_size: USDC per bet
        max_price: Maximum share price to buy (lower = more contrarian)
        min_price: Minimum share price (skip if too extreme — near resolution)
        min_time_remaining: Don't enter if <N seconds to market close

    Returns:
        BetSignal or None if no trade
    """
    if not market.accepting_orders:
        return None

    remaining = market.time_remaining_seconds()
    if remaining < min_time_remaining:
        return None

    # Find the cheaper side
    if market.price_up < market.price_down:
        cheap_side = "up"
        cheap_price = market.price_up
        cheap_token = market.token_id_up
        cheap_odds = market.odds_up
    else:
        cheap_side = "down"
        cheap_price = market.price_down
        cheap_token = market.token_id_down
        cheap_odds = market.odds_down

    # Skip if not cheap enough (no edge at 50/50)
    if cheap_price > max_price:
        return None

    # Skip if suspiciously cheap (near resolution, price already decided)
    if cheap_price < min_price:
        return None

    # Skip if no token ID
    if not cheap_token:
        return None

    confidence = "high" if cheap_price < 0.20 else "normal"

    # Scale bet size on very extreme odds
    adjusted_size = bet_size
    if confidence == "high" and cheap_price < 0.15:
        adjusted_size = bet_size * 1.5  # more on extreme contrarian

    return BetSignal(
        market_slug=market.slug,
        side=cheap_side,
        token_id=cheap_token,
        price=cheap_price,
        odds=cheap_odds,
        size_usdc=round(adjusted_size, 2),
        reason=f"Contrarian: {cheap_side} @ ${cheap_price:.3f} ({cheap_odds:.1f}:1), {remaining:.0f}s left",
        confidence=confidence,
    )


def should_exit(
    entry_price: float,
    current_price: float,
    take_profit: float = 0.10,
    stop_loss: float = 0.05,
) -> str | None:
    """
    Check if we should exit a position.

    Returns:
        "tp" for take profit, "sl" for stop loss, None to hold
    """
    pnl = current_price - entry_price

    if pnl >= take_profit:
        return "tp"
    if pnl <= -stop_loss:
        return "sl"

    return None
