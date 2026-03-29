"""
Signal Engine — Decides whether and how to bet on Polymarket.

Pure function: (CascadeSignal, MarketInfo, Config) → BetDecision | None

No side effects, no API calls — easy to test with mock data.
"""
from __future__ import annotations

from dataclasses import dataclass

from qtrade.polymarket.oi_monitor import CascadeSignal
from qtrade.polymarket.client import MarketInfo


@dataclass(frozen=True)
class BetDecision:
    """A decision to place a bet."""

    market_slug: str
    direction: str            # "up" or "down"
    token_id: str             # Polymarket CLOB token ID to buy
    amount_usdc: float        # How much to bet
    odds: float               # Implied payout ratio
    reason: str               # Human-readable explanation
    signal: CascadeSignal     # The triggering signal
    market: MarketInfo        # The market we're betting on


def evaluate_bet(
    signal: CascadeSignal,
    market: MarketInfo,
    bet_amount: float = 1.0,
    min_odds: float = 2.0,
    max_odds: float = 50.0,
    max_price: float = 0.50,
) -> BetDecision | None:
    """
    Evaluate whether to bet on Polymarket given a cascade signal.

    Logic:
        1. Cascade detected (OI dropped >10%) → expect bounce → bet "Up"
        2. Only bet if the market odds are attractive (price < max_price)
        3. Skip if odds are too extreme (potential manipulation)

    Args:
        signal: Detected liquidation cascade
        market: Current Polymarket daily market
        bet_amount: USDC per bet (default 1.0)
        min_odds: Minimum acceptable odds (default 2:1)
        max_odds: Maximum odds before we suspect manipulation (default 50:1)
        max_price: Maximum share price to buy (default 0.50 = need >50% implied prob)

    Returns:
        BetDecision or None if we shouldn't bet
    """
    if not market.active:
        return None

    # Determine which side to bet based on cascade direction
    if signal.direction == "long":
        # OI crashed → expect bounce → bet UP
        target_price = market.price_up
        target_odds = market.odds_up
        target_token = market.token_id_up
        direction = "up"
    else:
        # OI surged → expect drop → bet DOWN
        target_price = market.price_down
        target_odds = market.odds_down
        target_token = market.token_id_down
        direction = "down"

    # Filter: price too high (not attractive enough)
    if target_price > max_price:
        return None

    # Filter: odds too low
    if target_odds < min_odds:
        return None

    # Filter: odds suspiciously high (thin market, possible manipulation)
    if target_odds > max_odds:
        return None

    # Filter: no token ID (market data incomplete)
    if not target_token:
        return None

    # Scale bet amount by confidence
    adjusted_amount = bet_amount
    if signal.confidence == "high":
        adjusted_amount = bet_amount * 1.5  # 50% more on strong signals

    reason = (
        f"OI {signal.oi_change_24h:+.1%} on {signal.symbol} "
        f"→ bet {direction} @ {target_odds:.1f}:1 odds "
        f"(confidence: {signal.confidence})"
    )

    return BetDecision(
        market_slug=market.slug,
        direction=direction,
        token_id=target_token,
        amount_usdc=round(adjusted_amount, 2),
        odds=target_odds,
        reason=reason,
        signal=signal,
        market=market,
    )
