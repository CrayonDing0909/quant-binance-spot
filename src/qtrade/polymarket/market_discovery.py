"""
Market Discovery — Find active 15-minute Polymarket markets.

Adapted from discountry/polymarket-trading-bot (MIT License).
Uses Gamma API to discover current 15-minute Up/Down markets.

Slug pattern: {coin}-updown-15m-{unix_timestamp}
  e.g. "btc-updown-15m-1766671200"

Each 15-minute window is a separate market with its own token IDs.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)

GAMMA_API = "https://gamma-api.polymarket.com"

COIN_SLUGS = {
    "BTC": "btc-updown-15m",
    "ETH": "eth-updown-15m",
    "SOL": "sol-updown-15m",
    "XRP": "xrp-updown-15m",
}


@dataclass(frozen=True)
class Market15m:
    """A 15-minute Up/Down market."""

    slug: str
    question: str
    end_date: str
    token_id_up: str
    token_id_down: str
    price_up: float
    price_down: float
    accepting_orders: bool
    condition_id: str = ""

    @property
    def odds_up(self) -> float:
        return 1.0 / self.price_up if self.price_up > 0 else 0.0

    @property
    def odds_down(self) -> float:
        return 1.0 / self.price_down if self.price_down > 0 else 0.0

    def time_remaining_seconds(self) -> float:
        """Seconds until market closes."""
        if not self.end_date:
            return -1
        try:
            end = datetime.fromisoformat(self.end_date.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            return max(0, (end - now).total_seconds())
        except Exception:
            return -1


def _parse_market(data: dict[str, Any]) -> Market15m | None:
    """Parse Gamma API market response into Market15m."""
    try:
        prices = data.get("outcomePrices", "[]")
        if isinstance(prices, str):
            prices = json.loads(prices)

        clob_ids = data.get("clobTokenIds", "[]")
        if isinstance(clob_ids, str):
            clob_ids = json.loads(clob_ids)

        return Market15m(
            slug=data.get("slug", ""),
            question=data.get("question", ""),
            end_date=data.get("endDate", ""),
            token_id_up=clob_ids[0] if len(clob_ids) > 0 else "",
            token_id_down=clob_ids[1] if len(clob_ids) > 1 else "",
            price_up=float(prices[0]) if len(prices) > 0 else 0.5,
            price_down=float(prices[1]) if len(prices) > 1 else 0.5,
            accepting_orders=data.get("acceptingOrders", False),
            condition_id=data.get("conditionId", ""),
        )
    except Exception as e:
        logger.error(f"Failed to parse market: {e}")
        return None


def get_market_by_slug(slug: str) -> Market15m | None:
    """Fetch a specific market by slug."""
    try:
        resp = httpx.get(f"{GAMMA_API}/markets/slug/{slug}", timeout=10)
        if resp.status_code == 200:
            return _parse_market(resp.json())
        return None
    except Exception as e:
        logger.error(f"Failed to fetch market {slug}: {e}")
        return None


def find_current_15m_market(coin: str) -> Market15m | None:
    """
    Find the currently active 15-minute market for a coin.

    Args:
        coin: "BTC", "ETH", "SOL", or "XRP"

    Returns:
        Market15m or None if no active market found
    """
    coin = coin.upper()
    if coin not in COIN_SLUGS:
        raise ValueError(f"Unsupported coin: {coin}. Use: {list(COIN_SLUGS.keys())}")

    prefix = COIN_SLUGS[coin]
    now = datetime.now(timezone.utc)

    # Round to current 15-minute window
    minute = (now.minute // 15) * 15
    current_window = now.replace(minute=minute, second=0, microsecond=0)
    current_ts = int(current_window.timestamp())

    # Try current, next, and previous windows
    for offset in [0, 900, -900]:
        ts = current_ts + offset
        slug = f"{prefix}-{ts}"
        market = get_market_by_slug(slug)
        if market and market.accepting_orders:
            logger.info(f"Found active market: {slug} (remaining: {market.time_remaining_seconds():.0f}s)")
            return market

    logger.warning(f"No active 15m market found for {coin}")
    return None
