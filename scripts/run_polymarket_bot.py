#!/usr/bin/env python3
"""
Polymarket Cascade Bot — Entry point.

Usage:
    # Dry run (default — no real bets)
    PYTHONPATH=src python scripts/run_polymarket_bot.py

    # Dry run with custom config
    PYTHONPATH=src python scripts/run_polymarket_bot.py -c config/polymarket_bot.yaml

    # Single check (no loop)
    PYTHONPATH=src python scripts/run_polymarket_bot.py --once

    # Real trading (BE CAREFUL)
    PYTHONPATH=src python scripts/run_polymarket_bot.py --real
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description="Polymarket Cascade Bot")
    parser.add_argument("-c", "--config", default="config/polymarket_bot.yaml")
    parser.add_argument("--once", action="store_true", help="Single check, no loop")
    parser.add_argument("--real", action="store_true", help="Real trading (not dry run)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Inject environment variables
    config["wallet_address"] = os.environ.get("POLYMARKET_WALLET_ADDRESS", "")
    config["wallet_key"] = os.environ.get("POLYMARKET_WALLET_KEY", "")
    config["telegram_bot_token"] = os.environ.get("TELEGRAM_BOT_TOKEN", config.get("telegram_bot_token", ""))
    config["telegram_chat_id"] = os.environ.get("TELEGRAM_CHAT_ID", config.get("telegram_chat_id", ""))

    # Override dry_run if --real
    if args.real:
        config["dry_run"] = False
        print("⚠️  REAL TRADING MODE — bets will use real USDC")
    else:
        config["dry_run"] = True
        print("🧪 DRY RUN MODE — no real bets will be placed")

    # Validate
    if not config["dry_run"] and not config["wallet_key"]:
        print("ERROR: POLYMARKET_WALLET_KEY not set in environment")
        sys.exit(1)

    from qtrade.polymarket.runner import run_once, run_loop

    if args.once:
        state = {}
        run_once(config, state)
    else:
        run_loop(config)


if __name__ == "__main__":
    main()
