"""
Odds Divergence Strategy — Polymarket vs Binance Reality.

Core idea (from friend's insight):
    "先抓實盤漲跌幅％跟賭盤預測勝率％這兩個數據來看，
     當兩者偏差過大時，就嘗試進場"

Edge: Polymarket odds lag or overreact vs Binance price reality.
    When Polymarket says 75% chance of DOWN but Binance shows
    the drop has already stabilized → buy UP (fade Polymarket).

Rules (from friend's PDF):
    1. Only enter when Polymarket chance is outside 30%-70% range
    2. Don't trade obvious one-sided trends
    3. Pay attention to 00:00 UTC (8:00 UTC+8) daily open
    4. Don't chase extreme RR — moderate RR with higher WR is better

NOT a TA strategy. Uses only:
    - Polymarket odds (from market discovery)
    - Binance price displacement (from recent klines)
    - Simple trend filter (is it one-sided?)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from qtrade.polymarket.market_discovery import Market15m

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DivergenceSignal:
    """A detected divergence between Polymarket odds and Binance reality."""

    market_slug: str
    side: str               # "up" or "down" — which side to BET on
    token_id: str
    polymarket_odds: float  # what Polymarket thinks (e.g. 0.25 for Up)
    fair_odds: float        # what Binance displacement suggests
    divergence: float       # fair_odds - polymarket_odds (positive = underpriced)
    price_target: float     # share price to buy at
    size_usdc: float
    reason: str


def compute_fair_probability(
    price_at_entry: float,
    window_open: float,
    window_elapsed_pct: float,
) -> float:
    """
    Estimate the "fair" probability that the window closes UP,
    based on current Binance price displacement.

    Logic:
        - If price is above window open → more likely to close up
        - The further into the window we are, the more predictive
          current displacement is (less time for reversal)
        - At sweet spot (67% through), momentum has 82.5% WR per backtest

    Args:
        price_at_entry: current Binance price
        window_open: price at window start
        window_elapsed_pct: 0.0 (start) to 1.0 (end)

    Returns:
        fair_p_up: estimated probability of UP [0.05, 0.95]
    """
    if window_open == 0:
        return 0.50

    displacement_pct = (price_at_entry - window_open) / window_open * 100

    # Base: 50/50 + displacement effect
    # Calibrated from backtest: at sweet spot (67% elapsed),
    # momentum WR = 82.5% when displacement is clear
    # Scale the confidence by how far into the window we are
    confidence = 0.3 + 0.5 * window_elapsed_pct  # 0.3 at start, 0.8 at end

    # Map displacement to probability shift
    # ±0.1% displacement ≈ ±10% probability at full confidence
    shift = displacement_pct * 100 * confidence  # e.g. 0.05% * 100 * 0.7 = 3.5

    fair_p_up = 0.50 + shift / 100
    fair_p_up = max(0.05, min(0.95, fair_p_up))

    return fair_p_up


def detect_divergence(
    market: Market15m,
    binance_price: float,
    window_open: float,
    bet_size: float = 1.0,
    min_divergence: float = 0.15,
    min_remaining: float = 120.0,
    max_remaining: float = 600.0,
    odds_range_low: float = 0.30,
    odds_range_high: float = 0.70,
    trend_filter_pct: float = 0.5,
) -> DivergenceSignal | None:
    """
    Detect divergence between Polymarket odds and Binance reality.

    Logic:
        1. Check Polymarket odds are outside 30%-70% (rule 1 from PDF)
        2. Compute "fair" probability from Binance displacement
        3. If divergence > threshold → fade Polymarket
        4. Filter out obvious one-sided trends (rule 2)

    Args:
        market: Current Polymarket 15m market
        binance_price: Current Binance BTC price
        window_open: Price at window start (for displacement calc)
        bet_size: USDC per bet
        min_divergence: minimum gap between fair odds and market odds
        min_remaining: don't enter with <N seconds left
        max_remaining: don't enter too early
        odds_range_low/high: only trade when market odds outside this range (rule 1)
        trend_filter_pct: skip if displacement > this % (obvious trend, rule 2)

    Returns:
        DivergenceSignal or None
    """
    if not market.accepting_orders:
        return None

    remaining = market.time_remaining_seconds()
    if remaining < min_remaining or remaining > max_remaining:
        return None

    # Rule 2: Don't trade obvious one-sided moves
    if window_open > 0:
        displacement_pct = abs(binance_price - window_open) / window_open * 100
        if displacement_pct > trend_filter_pct:
            return None  # too one-sided, trend is clear

    # Rule 1: Only trade when odds outside 30%-70%
    up_outside = market.price_up < odds_range_low or market.price_up > odds_range_high
    down_outside = market.price_down < odds_range_low or market.price_down > odds_range_high
    if not (up_outside or down_outside):
        return None  # odds too balanced, no edge

    # Compute fair probability from Binance
    total_window = 900.0  # 15 minutes
    elapsed_pct = 1.0 - remaining / total_window
    fair_p_up = compute_fair_probability(binance_price, window_open, elapsed_pct)
    fair_p_down = 1.0 - fair_p_up

    # Check divergence for each side
    # Divergence = fair_odds - polymarket_odds
    # Positive divergence = Polymarket underprices this side = opportunity
    div_up = fair_p_up - market.price_up
    div_down = fair_p_down - market.price_down

    # Pick the side with larger positive divergence
    if div_up > div_down and div_up >= min_divergence:
        side = "up"
        token_id = market.token_id_up
        price_target = market.price_up
        divergence = div_up
    elif div_down >= min_divergence:
        side = "down"
        token_id = market.token_id_down
        price_target = market.price_down
        divergence = div_down
    else:
        return None  # no sufficient divergence

    if not token_id:
        return None

    odds = 1.0 / price_target if price_target > 0 else 0

    reason = (
        f"Divergence: PM={price_target:.2f} vs fair={fair_p_up if side == 'up' else fair_p_down:.2f} "
        f"(gap={divergence:+.2f}) | "
        f"Binance {'+' if binance_price > window_open else ''}"
        f"{(binance_price - window_open) / window_open * 100:.3f}% from open | "
        f"{remaining:.0f}s left"
    )

    return DivergenceSignal(
        market_slug=market.slug,
        side=side,
        token_id=token_id,
        polymarket_odds=price_target,
        fair_odds=fair_p_up if side == "up" else fair_p_down,
        divergence=divergence,
        price_target=price_target,
        size_usdc=bet_size,
        reason=reason,
    )
