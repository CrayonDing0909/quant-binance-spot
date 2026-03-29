"""
Polymarket API Client — Handles all Polymarket interaction.

Responsibilities:
    - Fetch current market odds for daily BTC/ETH/SOL up-or-down
    - Place bets (buy YES/NO shares)
    - Check wallet USDC balance
    - Approve USDC spending (one-time)

Uses py-clob-client (official Polymarket Python SDK).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx
from web3 import Web3

logger = logging.getLogger(__name__)

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_HOST = "https://clob.polymarket.com"
POLYGON_CHAIN_ID = 137

# Native USDC on Polygon (not bridged)
USDC_ADDRESS = "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359"
USDC_ABI = [
    {
        "inputs": [{"name": "account", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [
            {"name": "spender", "type": "address"},
            {"name": "amount", "type": "uint256"},
        ],
        "name": "approve",
        "outputs": [{"name": "", "type": "bool"}],
        "stateMutability": "nonpayable",
        "type": "function",
    },
]


@dataclass(frozen=True)
class MarketInfo:
    """Polymarket daily binary market info."""

    slug: str
    title: str
    condition_id: str
    token_id_up: str
    token_id_down: str
    price_up: float      # 0.0 - 1.0
    price_down: float     # 0.0 - 1.0
    volume: float
    end_date: str
    active: bool

    @property
    def odds_up(self) -> float:
        """Implied payout if UP wins (e.g. 0.05 price → 20:1 odds)."""
        return 1.0 / self.price_up if self.price_up > 0 else 0.0

    @property
    def odds_down(self) -> float:
        """Implied payout if DOWN wins."""
        return 1.0 / self.price_down if self.price_down > 0 else 0.0


def fetch_daily_market(
    symbol: str = "bitcoin",
    date_str: str | None = None,
) -> MarketInfo | None:
    """
    Fetch daily "up or down" market from Polymarket.

    Args:
        symbol: "bitcoin", "ethereum", or "solana"
        date_str: Date string like "march-30-2026". Auto-generates if None.

    Returns:
        MarketInfo or None if market not found.
    """
    if date_str is None:
        now = datetime.now(timezone.utc)
        month = now.strftime("%B").lower()
        day = now.day
        year = now.year
        date_str = f"{month}-{day}-{year}"

    slug = f"{symbol}-up-or-down-on-{date_str}"

    try:
        resp = httpx.get(
            f"{GAMMA_API}/events",
            params={"slug": slug},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        if not data:
            logger.warning(f"Market not found: {slug}")
            return None

        event = data[0]
        markets = event.get("markets", [])

        if len(markets) < 1:
            logger.warning(f"No sub-markets in: {slug}")
            return None

        # The first market contains up/down outcomes
        market = markets[0]
        outcome_prices = market.get("outcomePrices", "[]")

        if isinstance(outcome_prices, str):
            import json
            prices = json.loads(outcome_prices)
        else:
            prices = outcome_prices

        # Get token IDs for trading
        clob_ids = market.get("clobTokenIds", "[]")
        if isinstance(clob_ids, str):
            import json
            clob_ids = json.loads(clob_ids)

        price_up = float(prices[0]) if len(prices) > 0 else 0.0
        price_down = float(prices[1]) if len(prices) > 1 else 0.0
        token_up = clob_ids[0] if len(clob_ids) > 0 else ""
        token_down = clob_ids[1] if len(clob_ids) > 1 else ""

        return MarketInfo(
            slug=slug,
            title=event.get("title", slug),
            condition_id=market.get("conditionId", ""),
            token_id_up=token_up,
            token_id_down=token_down,
            price_up=price_up,
            price_down=price_down,
            volume=float(event.get("volume", 0)),
            end_date=market.get("endDate", ""),
            active=market.get("active", False),
        )

    except Exception as e:
        logger.error(f"Failed to fetch market {slug}: {e}")
        return None


def get_usdc_balance(wallet_address: str, rpc_url: str = "https://polygon.drpc.org") -> float:
    """Get USDC balance on Polygon for a wallet address."""
    try:
        w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 10}))
        usdc = w3.eth.contract(
            address=Web3.to_checksum_address(USDC_ADDRESS),
            abi=USDC_ABI,
        )
        balance = usdc.functions.balanceOf(
            Web3.to_checksum_address(wallet_address)
        ).call()
        return balance / 1e6  # USDC has 6 decimals
    except Exception as e:
        logger.error(f"Failed to get USDC balance: {e}")
        return 0.0


def place_market_order(
    wallet_key: str,
    token_id: str,
    amount_usdc: float,
    side: str = "BUY",
) -> dict | None:
    """
    Place a market order on Polymarket CLOB.

    Args:
        wallet_key: Polygon wallet private key
        token_id: The CLOB token ID for the outcome to buy
        amount_usdc: Amount in USDC to bet
        side: "BUY" to buy shares

    Returns:
        Order result dict or None on failure
    """
    try:
        from py_clob_client.client import ClobClient
        from py_clob_client.clob_types import MarketOrderArgs, OrderType

        client = ClobClient(
            CLOB_HOST,
            key=wallet_key,
            chain_id=POLYGON_CHAIN_ID,
        )

        # Derive or create API credentials
        creds = client.create_or_derive_api_creds()
        client.set_api_creds(creds)

        # Build market order
        order_args = MarketOrderArgs(
            token_id=token_id,
            amount=amount_usdc,
        )

        # Create and sign the order
        signed_order = client.create_market_order(order_args)
        result = client.post_order(signed_order, OrderType.FOK)

        logger.info(f"Order placed: {result}")
        return result

    except Exception as e:
        logger.error(f"Failed to place order: {e}")
        return None
