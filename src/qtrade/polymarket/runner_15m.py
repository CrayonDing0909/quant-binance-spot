"""
15-Minute Polymarket Bot Runner.

Architecture:
    Every 30 seconds:
        1. Find current active 15-minute market
        2. Check current odds
        3. If contrarian signal → place limit order (maker = 0% fee)
        4. Monitor open positions for TP/SL
        5. When market expires → auto-discover next market

Uses polling (not WebSocket) for simplicity in v1.
Can upgrade to WebSocket later for faster execution.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from qtrade.polymarket.market_discovery import find_current_15m_market, Market15m
from qtrade.polymarket.mean_reversion import evaluate_market, BetSignal

logger = logging.getLogger(__name__)


def send_telegram(message: str, bot_token: str, chat_id: str) -> None:
    """Send Telegram notification."""
    if not bot_token or not chat_id:
        return
    try:
        import httpx
        httpx.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"},
            timeout=10,
        )
    except Exception as e:
        logger.error(f"Telegram failed: {e}")


def place_limit_order(
    wallet_key: str,
    token_id: str,
    price: float,
    size_usdc: float,
    api_key: str = "",
    api_secret: str = "",
    api_passphrase: str = "",
) -> dict | None:
    """
    Place a LIMIT order (maker = 0% fee).

    Unlike market orders, limit orders rest in the book until filled.
    As maker, we pay 0% fee and earn rebates.
    """
    try:
        from py_clob_client.client import ClobClient
        from py_clob_client.clob_types import OrderArgs, OrderType, ApiCreds

        client = ClobClient(
            "https://clob.polymarket.com",
            key=wallet_key,
            chain_id=137,
        )

        # Set API creds if available
        if api_key and api_secret and api_passphrase:
            creds = ApiCreds(
                api_key=api_key,
                api_secret=api_secret,
                api_passphrase=api_passphrase,
            )
            client.set_api_creds(creds)
        else:
            creds = client.create_or_derive_api_creds()
            client.set_api_creds(creds)

        # Calculate shares: size_usdc / price = number of shares
        # Each share pays $1 if we win
        shares = size_usdc / price

        order_args = OrderArgs(
            token_id=token_id,
            price=price,
            size=round(shares, 2),
        )

        signed_order = client.create_order(order_args)
        result = client.post_order(signed_order, OrderType.GTC)

        logger.info(f"Limit order placed: {result}")
        return result

    except Exception as e:
        logger.error(f"Order failed: {e}")
        return None


def run_once(config: dict, state: dict) -> dict:
    """
    Single check cycle for 15-minute strategy.

    Args:
        config: Bot configuration
        state: Mutable state (positions, cooldowns)

    Returns:
        Updated state
    """
    coins = config.get("coins", ["BTC"])
    bet_size = config.get("bet_size_usdc", 1.0)
    max_price = config.get("max_price", 0.45)
    min_price = config.get("min_price", 0.02)
    min_time_remaining = config.get("min_time_remaining", 120.0)
    max_positions = config.get("max_positions", 3)
    dry_run = config.get("dry_run", True)
    wallet_key = config.get("wallet_key", "")
    api_key = config.get("api_key", "")
    api_secret = config.get("api_secret", "")
    api_passphrase = config.get("api_passphrase", "")
    tg_token = config.get("telegram_bot_token", "")
    tg_chat = config.get("telegram_chat_id", "")

    now = datetime.now(timezone.utc)

    # Track positions
    positions = state.get("positions", [])
    active_positions = [p for p in positions if p.get("status") == "open"]

    if len(active_positions) >= max_positions:
        logger.debug(f"Max positions ({max_positions}) reached")
        return state

    for coin in coins:
        # Find current market
        market = find_current_15m_market(coin)
        if market is None:
            continue

        # Skip if we already have a position in this market
        if any(p.get("slug") == market.slug for p in active_positions):
            continue

        # Evaluate for entry
        signal = evaluate_market(
            market=market,
            bet_size=bet_size,
            max_price=max_price,
            min_price=min_price,
            min_time_remaining=min_time_remaining,
        )

        if signal is None:
            logger.debug(f"  {coin}: no signal (Up=${market.price_up:.3f}, Down=${market.price_down:.3f})")
            continue

        # Place order
        msg = (
            f"{'[DRY] ' if dry_run else ''}*15M BET*\n"
            f"Coin: {coin}\n"
            f"Side: {signal.side.upper()}\n"
            f"Price: ${signal.price:.3f} ({signal.odds:.1f}:1)\n"
            f"Size: {signal.size_usdc} USDC\n"
            f"{signal.reason}"
        )

        if dry_run:
            logger.info(f"DRY RUN: {signal.reason}")
        else:
            result = place_limit_order(
                wallet_key=wallet_key,
                token_id=signal.token_id,
                price=signal.price,
                size_usdc=signal.size_usdc,
                api_key=api_key,
                api_secret=api_secret,
                api_passphrase=api_passphrase,
            )
            if result:
                msg += f"\nOrder: OK"
                positions.append({
                    "slug": market.slug,
                    "coin": coin,
                    "side": signal.side,
                    "entry_price": signal.price,
                    "size": signal.size_usdc,
                    "odds": signal.odds,
                    "timestamp": now.isoformat(),
                    "status": "open",
                })
            else:
                msg += f"\nOrder: FAILED"

        send_telegram(msg, tg_token, tg_chat)

        # Log trade
        log_dir = Path(config.get("log_dir", "logs/polymarket"))
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"trades_15m_{now.strftime('%Y-%m-%d')}.jsonl"
        with open(log_file, "a") as f:
            entry = {
                "timestamp": now.isoformat(),
                "dry_run": dry_run,
                "coin": coin,
                "slug": market.slug,
                "side": signal.side,
                "price": signal.price,
                "odds": signal.odds,
                "size": signal.size_usdc,
                "confidence": signal.confidence,
                "time_remaining": market.time_remaining_seconds(),
            }
            f.write(json.dumps(entry) + "\n")

    state["positions"] = positions
    return state


def run_loop(config: dict) -> None:
    """Main loop — checks every poll_interval seconds."""
    poll_interval = config.get("poll_interval_seconds", 30)
    state: dict = {"positions": []}

    logger.info("=" * 50)
    logger.info("Polymarket 15-Minute Bot starting")
    logger.info(f"  Coins: {config.get('coins')}")
    logger.info(f"  Bet size: {config.get('bet_size_usdc')} USDC")
    logger.info(f"  Max price: {config.get('max_price')} (min odds: {1/config.get('max_price', 0.45):.1f}:1)")
    logger.info(f"  Dry run: {config.get('dry_run', True)}")
    logger.info(f"  Poll interval: {poll_interval}s")
    logger.info("=" * 50)

    while True:
        try:
            state = run_once(config, state)
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)

        time.sleep(poll_interval)
