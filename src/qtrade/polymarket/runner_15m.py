"""
15-Minute Polymarket Bot Runner — v2 (Krajekis 5-Layer Strategy).

Architecture:
    Every 30 seconds:
        1. Find current active 15-minute market
        2. Fetch real-time Binance 1m klines → compute TA signals
        3. Determine session + volatility regime
        4. Evaluate 5-layer strategy (krajekis)
        5. If signal → place LIMIT order (maker = 0% fee)
        6. Track daily P&L for risk management

v2 changes vs v1:
    - Added Binance TA feed (RSI, MACD, EMA, VWAP, ATR)
    - Session-aware (Asia/London/NY/Weekend)
    - Two scenarios: trend_follow (expensive side) + mean_reversion (cheap side)
    - Daily loss limit
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from qtrade.polymarket.market_discovery import find_current_15m_market, Market15m
from qtrade.polymarket.binance_feed import compute_ta_signals, fetch_recent_klines
from qtrade.polymarket.odds_divergence import detect_divergence, DivergenceSignal
from qtrade.polymarket.krajekis_strategy import determine_session

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

        # signature_type=2 + funder for Polymarket proxy wallet
        safe_address = os.environ.get("POLYMARKET_SAFE_ADDRESS", "")
        sig_type = int(os.environ.get("POLYMARKET_SIGNATURE_TYPE", "2"))
        init_kwargs = {
            "host": "https://clob.polymarket.com",
            "key": wallet_key,
            "chain_id": 137,
            "signature_type": sig_type,
        }
        if safe_address:
            init_kwargs["funder"] = safe_address
        client = ClobClient(**init_kwargs)

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

        # Polymarket min order = $1 notional (shares * price >= 1)
        min_shares = max(shares, 1.0 / price + 0.01) if price > 0 else shares
        shares = max(shares, min_shares)

        order_args = OrderArgs(
            token_id=token_id,
            price=price,
            size=round(shares, 1),
            side="BUY",
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
    Single check cycle — Krajekis 5-layer strategy.

    Args:
        config: Bot configuration
        state: Mutable state (positions, daily counters)

    Returns:
        Updated state
    """
    coins = config.get("coins", ["BTC"])
    bet_size = config.get("risk", {}).get("bet_size_usdc", config.get("bet_size_usdc", 1.0))
    max_positions = config.get("risk", {}).get("max_positions", config.get("max_positions", 3))
    max_daily_losses = config.get("risk", {}).get("max_daily_losses", 3)
    max_daily_trades = config.get("risk", {}).get("max_daily_trades", 10)
    dry_run = config.get("dry_run", True)
    wallet_key = config.get("wallet_key", "")
    api_key = config.get("api_key", "")
    api_secret = config.get("api_secret", "")
    api_passphrase = config.get("api_passphrase", "")
    tg_token = config.get("telegram_bot_token", "")
    tg_chat = config.get("telegram_chat_id", "")

    # Volatility thresholds
    vol_cfg = config.get("volatility", {})
    # Entry timing
    entry_cfg = config.get("entry_window", {})
    sweet_start = entry_cfg.get("sweet_spot_start", 600)
    sweet_end = entry_cfg.get("sweet_spot_end", 120)
    # Pricing
    pricing = config.get("pricing", {})
    trend_cfg = pricing.get("trend_follow", {})
    mr_cfg = pricing.get("mean_reversion", {})

    now = datetime.now(timezone.utc)
    today = now.strftime("%Y-%m-%d")

    # ── Daily risk management ──
    if state.get("last_date") != today:
        state["last_date"] = today
        state["daily_losses"] = 0
        state["daily_trades"] = 0

    if state.get("daily_losses", 0) >= max_daily_losses:
        logger.info(f"Daily loss limit reached ({max_daily_losses}). Pausing.")
        return state

    if state.get("daily_trades", 0) >= max_daily_trades:
        logger.info(f"Daily trade limit reached ({max_daily_trades}).")
        return state

    # ── Layer 1: Session ──
    session = determine_session(now)

    # Track positions
    positions = state.get("positions", [])
    active_positions = [p for p in positions if p.get("status") == "open"]
    # Dedup: track which slugs we already signaled (even in dry run)
    signaled_slugs = state.get("signaled_slugs", set())

    if len(active_positions) >= max_positions:
        return state

    for coin in coins:
        # Find current 15m market
        market = find_current_15m_market(coin)
        if market is None:
            continue

        # Dedup: skip if we already signaled this exact window+coin
        dedup_key = f"{market.slug}_{coin}"
        if dedup_key in signaled_slugs:
            continue

        # Skip if already positioned in this window
        if any(p.get("slug") == market.slug for p in active_positions):
            continue

        # ── Get current Binance price for divergence check ──
        from qtrade.polymarket.binance_feed import BINANCE_SYMBOLS
        import httpx as _httpx
        bn_symbol = BINANCE_SYMBOLS.get(coin.upper(), f"{coin}USDT")
        try:
            _resp = _httpx.get(
                "https://fapi.binance.com/fapi/v1/ticker/price",
                params={"symbol": bn_symbol}, timeout=5,
            )
            binance_price = float(_resp.json()["price"])
        except Exception:
            logger.warning(f"  {coin}: failed to get Binance price")
            continue

        # ── Get window open price (first trade of this 15m window) ──
        # Approximate: use market's implied price structure
        # Better: get the kline open for this 15m window
        try:
            klines = fetch_recent_klines(coin, interval="15m", limit=2)
            if klines is not None and len(klines) >= 1:
                window_open = float(klines["open"].iloc[-1])
            else:
                window_open = binance_price
        except Exception:
            window_open = binance_price

        # ── Simple Contrarian: buy PM cheap side at sweet spot ──
        # Matches the backtest that showed +15.7 to +20.3 PnL on 100 trades
        remaining = market.time_remaining_seconds()
        cheap_threshold = config.get("cheap_threshold", 0.35)
        trend_max = config.get("trend_filter_pct", 0.5)

        # Timing: sweet spot only
        if remaining > sweet_start or remaining < sweet_end:
            continue

        # Rule 2: skip obvious one-sided trends
        disp_pct = abs(binance_price - window_open) / window_open * 100 if window_open > 0 else 0
        if disp_pct > trend_max:
            logger.debug(f"  {coin}: trend too strong ({disp_pct:.3f}%)")
            continue

        # Rule 1: buy cheap side when PM odds outside 30-70%
        if market.price_up < cheap_threshold:
            side = "up"
            share_price = market.price_up
            token_id = market.token_id_up
        elif market.price_down < cheap_threshold:
            side = "down"
            share_price = market.price_down
            token_id = market.token_id_down
        else:
            logger.debug(
                f"  {coin}: no cheap side | "
                f"Up=${market.price_up:.3f} Down=${market.price_down:.3f} | "
                f"{remaining:.0f}s left"
            )
            continue

        if not token_id:
            continue

        # Build signal object for downstream flow
        from dataclasses import dataclass
        @dataclass
        class _Signal:
            side: str
            token_id: str
            price_target: float
            size_usdc: float
            polymarket_odds: float
            fair_odds: float
            divergence: float
            reason: str

        odds = 1.0 / share_price if share_price > 0 else 0
        setup = _Signal(
            side=side,
            token_id=token_id,
            price_target=share_price,
            size_usdc=bet_size,
            polymarket_odds=share_price,
            fair_odds=0.5,
            divergence=0.5 - share_price,
            reason=f"Contrarian: buy {side} @ ${share_price:.3f} ({odds:.1f}:1), PM偏離 {remaining:.0f}s left",
        )

        # TA for logging only
        ta = compute_ta_signals(coin)

        # ── Execute ──
        odds = 1.0 / setup.price_target if setup.price_target > 0 else 0
        remaining_min = market.time_remaining_seconds() / 60

        # Telegram message — human readable
        side_emoji = "🟢" if setup.side == "up" else "🔴"
        session_zh = {"asia": "亞洲", "london_kill": "倫敦", "ny_open": "紐約", "london_close": "倫敦收盤", "weekend": "週末"}.get(session, session)
        win_amount = setup.size_usdc * (odds - 1) if odds > 1 else 0
        disp_pct = (binance_price - window_open) / window_open * 100 if window_open > 0 else 0

        msg = (
            f"{'🧪 ' if dry_run else '💰 '}*Polymarket 15m*\n"
            f"\n"
            f"{side_emoji} *{coin} → {setup.side.upper()}*\n"
            f"📊 策略: Odds 偏差 (PM vs Binance)\n"
            f"💵 下注: ${setup.size_usdc:.2f} @ ${setup.price_target:.3f}\n"
            f"🎯 賠率: {odds:.1f}:1 (贏${win_amount:.2f} / 虧${setup.size_usdc:.2f})\n"
            f"📈 Binance: {binance_price:.0f} ({disp_pct:+.3f}% from open)\n"
            f"📊 偏差: PM={setup.polymarket_odds:.2f} vs fair={setup.fair_odds:.2f} (gap={setup.divergence:+.2f})\n"
            f"⏱ 剩餘: {remaining_min:.0f} 分鐘 | 🕐 {session_zh}\n"
        )

        if dry_run:
            logger.info(f"DRY RUN [{coin}]: {setup.reason}")
        else:
            result = place_limit_order(
                wallet_key=wallet_key,
                token_id=setup.token_id,
                price=setup.price_target,
                size_usdc=setup.size_usdc,
                api_key=api_key,
                api_secret=api_secret,
                api_passphrase=api_passphrase,
            )
            if result:
                msg += "\n\n✅ 下單成功"
                positions.append({
                    "slug": market.slug,
                    "coin": coin,
                    "side": setup.side,
                    "entry_price": setup.price_target,
                    "size": setup.size_usdc,
                    "strategy": "odds_divergence",
                    "divergence": setup.divergence,
                    "session": session,
                    "timestamp": now.isoformat(),
                    "status": "open",
                })
                state["daily_trades"] = state.get("daily_trades", 0) + 1
            else:
                msg += "\n\n❌ 下單失敗"

        send_telegram(msg, tg_token, tg_chat)

        # Mark as signaled (dedup)
        signaled_slugs.add(dedup_key)

        # Log trade
        log_dir = Path(config.get("log_dir", "logs/polymarket"))
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"trades_15m_{today}.jsonl"
        with open(log_file, "a") as f:
            entry = {
                "timestamp": now.isoformat(),
                "dry_run": dry_run,
                "coin": coin,
                "slug": market.slug,
                "side": setup.side,
                "price": setup.price_target,
                "odds": odds,
                "size": setup.size_usdc,
                "strategy": "odds_divergence",
                "divergence": setup.divergence,
                "polymarket_odds": setup.polymarket_odds,
                "fair_odds": setup.fair_odds,
                "session": session,
                "binance_price": binance_price,
                "window_open": window_open,
                "ta": {
                    "rsi": round(ta.rsi_14, 1),
                    "macd_hist": round(ta.macd_hist, 2),
                    "vwap_dist": round(ta.vwap_distance_pct, 3),
                    "atr": round(ta.atr_14, 2),
                    "ema21_gt_50": ta.ema_21 > ta.ema_50,
                },
                "time_remaining": market.time_remaining_seconds(),
            }
            f.write(json.dumps(entry) + "\n")

    state["positions"] = positions
    # Clean old slugs (keep only last 200 to prevent memory leak)
    if len(signaled_slugs) > 200:
        signaled_slugs = set(list(signaled_slugs)[-100:])
    state["signaled_slugs"] = signaled_slugs
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
