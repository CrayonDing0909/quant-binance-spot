"""
Polymarket Bot Runner — Main loop.

Architecture:
    Every check_interval (default 1h):
        1. Fetch latest OI data from Binance
        2. Detect liquidation cascade (oi_monitor)
        3. If cascade → fetch Polymarket odds (client)
        4. Evaluate bet (signal engine)
        5. Place bet if decision is GO (client)
        6. Send Telegram notification
        7. Log everything

Stateful tracking:
    - Cooldown: don't bet on same symbol within cooldown_hours
    - Daily limit: max bets per day
    - Balance check: don't bet if USDC balance too low
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from qtrade.polymarket.oi_monitor import detect_cascade
from qtrade.polymarket.client import (
    fetch_daily_market,
    get_usdc_balance,
    place_market_order,
)
from qtrade.polymarket.signal import evaluate_bet, BetDecision

logger = logging.getLogger(__name__)

# Symbol → Polymarket market name mapping
SYMBOL_MARKET_MAP = {
    "BTCUSDT": "bitcoin",
    "ETHUSDT": "ethereum",
    "SOLUSDT": "solana",
}


def load_oi_data(symbol: str, data_dir: Path) -> pd.Series | None:
    """Load latest OI data from local parquet files."""
    from qtrade.data.open_interest import OI_PROVIDER_SEARCH_ORDER

    oi_base = data_dir / "binance" / "futures" / "open_interest"
    for provider in OI_PROVIDER_SEARCH_ORDER:
        path = oi_base / provider / f"{symbol}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            col = "sumOpenInterestValue" if "sumOpenInterestValue" in df.columns else df.columns[0]
            return df[col].astype(float)
    return None


def load_price_data(symbol: str, data_dir: Path) -> pd.Series | None:
    """Load latest price data from local parquet files."""
    path = data_dir / "binance" / "futures" / "1h" / f"{symbol}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return df["close"]


def fetch_live_oi(symbol: str) -> pd.Series | None:
    """Fetch recent OI from Binance API (live, not cached)."""
    try:
        import httpx

        # Binance Futures API: recent OI (500 bars max at 1h)
        resp = httpx.get(
            "https://fapi.binance.com/futures/data/openInterestHist",
            params={"symbol": symbol, "period": "1h", "limit": 500},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        if not data:
            return None

        records = []
        for r in data:
            ts = pd.Timestamp(r["timestamp"], unit="ms", tz="UTC")
            val = float(r["sumOpenInterestValue"])
            records.append({"timestamp": ts, "oi": val})

        df = pd.DataFrame(records).set_index("timestamp").sort_index()
        return df["oi"]

    except Exception as e:
        logger.error(f"Failed to fetch live OI for {symbol}: {e}")
        return None


def fetch_live_price(symbol: str) -> tuple[pd.Series | None, float]:
    """Fetch recent klines from Binance API."""
    try:
        import httpx

        resp = httpx.get(
            "https://fapi.binance.com/fapi/v1/klines",
            params={"symbol": symbol, "interval": "1h", "limit": 750},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        records = []
        for k in data:
            ts = pd.Timestamp(k[0], unit="ms", tz="UTC")
            close = float(k[4])
            records.append({"timestamp": ts, "close": close})

        df = pd.DataFrame(records).set_index("timestamp").sort_index()
        current_price = df["close"].iloc[-1]
        return df["close"], current_price

    except Exception as e:
        logger.error(f"Failed to fetch live price for {symbol}: {e}")
        return None, 0.0


def send_telegram(message: str, bot_token: str, chat_id: str) -> None:
    """Send a Telegram notification."""
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
        logger.error(f"Telegram notification failed: {e}")


def run_once(
    config: dict,
    state: dict,
) -> dict:
    """
    Run one check cycle. Returns updated state.

    Args:
        config: Bot configuration dict
        state: Mutable state dict (cooldowns, daily counts, etc.)

    Returns:
        Updated state dict
    """
    symbols = config.get("symbols", ["BTCUSDT"])
    bet_amount = config.get("bet_amount_usdc", 1.0)
    min_odds = config.get("min_odds", 2.0)
    max_odds = config.get("max_odds", 50.0)
    max_price = config.get("max_price", 0.50)
    cooldown_hours = config.get("cooldown_hours", 24)
    max_daily_bets = config.get("max_daily_bets", 3)
    wallet_address = config.get("wallet_address", "")
    wallet_key = config.get("wallet_key", "")
    dry_run = config.get("dry_run", True)
    tg_token = config.get("telegram_bot_token", "")
    tg_chat = config.get("telegram_chat_id", "")
    oi_drop_threshold = config.get("oi_drop_threshold", -0.10)

    now = datetime.now(timezone.utc)
    today = now.strftime("%Y-%m-%d")

    # Reset daily counter
    if state.get("last_date") != today:
        state["last_date"] = today
        state["daily_bets"] = 0

    # Check daily limit
    if state.get("daily_bets", 0) >= max_daily_bets:
        logger.info(f"Daily bet limit reached ({max_daily_bets})")
        return state

    # Check USDC balance
    if not dry_run:
        balance = get_usdc_balance(wallet_address)
        if balance < bet_amount:
            logger.warning(f"Insufficient USDC: {balance:.2f} < {bet_amount}")
            return state
        logger.info(f"USDC balance: {balance:.2f}")

    for symbol in symbols:
        logger.info(f"Checking {symbol}...")

        # Cooldown check
        last_bet_time = state.get(f"cooldown_{symbol}")
        if last_bet_time:
            hours_since = (now - pd.Timestamp(last_bet_time)).total_seconds() / 3600
            if hours_since < cooldown_hours:
                logger.info(f"  {symbol}: cooldown ({hours_since:.0f}h / {cooldown_hours}h)")
                continue

        # Fetch live data
        oi_series = fetch_live_oi(symbol)
        price_series, current_price = fetch_live_price(symbol)

        if oi_series is None or price_series is None:
            logger.warning(f"  {symbol}: failed to fetch data")
            continue

        # Detect cascade
        signal = detect_cascade(
            symbol=symbol,
            oi_series=oi_series,
            close_series=price_series,
            oi_drop_threshold=oi_drop_threshold,
        )

        if signal is None:
            logger.info(f"  {symbol}: no cascade detected")
            continue

        logger.info(
            f"  {symbol}: CASCADE DETECTED! "
            f"OI_24h={signal.oi_change_24h:+.1%}, "
            f"OI_1h={signal.oi_change_1h:+.2%}, "
            f"price_30d={signal.price_change_30d:+.1%}"
        )

        # Fetch Polymarket market
        pm_symbol = SYMBOL_MARKET_MAP.get(symbol)
        if pm_symbol is None:
            logger.warning(f"  {symbol}: no Polymarket mapping")
            continue

        market = fetch_daily_market(pm_symbol)
        if market is None:
            logger.warning(f"  {symbol}: no Polymarket daily market found")
            continue

        logger.info(
            f"  Polymarket: {market.title}, "
            f"Up={market.price_up:.3f} ({market.odds_up:.1f}:1), "
            f"Down={market.price_down:.3f} ({market.odds_down:.1f}:1)"
        )

        # Evaluate bet
        decision = evaluate_bet(
            signal=signal,
            market=market,
            bet_amount=bet_amount,
            min_odds=min_odds,
            max_odds=max_odds,
            max_price=max_price,
        )

        if decision is None:
            logger.info(f"  {symbol}: signal detected but odds not attractive enough")
            continue

        # Place bet or dry run
        msg = (
            f"{'[DRY RUN] ' if dry_run else ''}*CASCADE BET*\n"
            f"Symbol: {symbol}\n"
            f"Direction: {decision.direction.upper()}\n"
            f"Amount: {decision.amount_usdc} USDC\n"
            f"Odds: {decision.odds:.1f}:1\n"
            f"Reason: {decision.reason}\n"
            f"Market: {decision.market_slug}"
        )

        if dry_run:
            logger.info(f"  DRY RUN: would bet {decision.amount_usdc} USDC on {decision.direction}")
        else:
            result = place_market_order(
                wallet_key=wallet_key,
                token_id=decision.token_id,
                amount_usdc=decision.amount_usdc,
            )
            if result:
                msg += f"\nOrder: {json.dumps(result)[:200]}"
                state[f"cooldown_{symbol}"] = now.isoformat()
                state["daily_bets"] = state.get("daily_bets", 0) + 1
            else:
                msg += "\nOrder: FAILED"

        send_telegram(msg, tg_token, tg_chat)
        logger.info(f"  {symbol}: {msg}")

        # Log trade
        log_dir = Path(config.get("log_dir", "logs/polymarket"))
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"trades_{today}.jsonl"
        with open(log_file, "a") as f:
            entry = {
                "timestamp": now.isoformat(),
                "dry_run": dry_run,
                "decision": {
                    "direction": decision.direction,
                    "amount": decision.amount_usdc,
                    "odds": decision.odds,
                    "reason": decision.reason,
                    "market": decision.market_slug,
                },
                "signal": {
                    "symbol": signal.symbol,
                    "oi_change_24h": signal.oi_change_24h,
                    "price": signal.current_price,
                    "confidence": signal.confidence,
                },
            }
            f.write(json.dumps(entry) + "\n")

    return state


def run_loop(config: dict) -> None:
    """
    Main bot loop. Runs forever, checking every check_interval seconds.
    """
    check_interval = config.get("check_interval_seconds", 3600)  # default 1h
    state: dict = {}

    logger.info("=" * 50)
    logger.info("Polymarket Cascade Bot starting")
    logger.info(f"  Symbols: {config.get('symbols')}")
    logger.info(f"  Bet amount: {config.get('bet_amount_usdc')} USDC")
    logger.info(f"  Dry run: {config.get('dry_run', True)}")
    logger.info(f"  Check interval: {check_interval}s")
    logger.info("=" * 50)

    while True:
        try:
            state = run_once(config, state)
        except Exception as e:
            logger.error(f"Error in run_once: {e}", exc_info=True)

        logger.info(f"Sleeping {check_interval}s until next check...")
        time.sleep(check_interval)
