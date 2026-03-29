#!/usr/bin/env python3
"""
Polymarket 15-Minute Contrarian Bot — Entry point.

Usage:
    # Dry run
    PYTHONPATH=src python scripts/run_polymarket_15m.py

    # Single check
    PYTHONPATH=src python scripts/run_polymarket_15m.py --once

    # Real trading
    PYTHONPATH=src python scripts/run_polymarket_15m.py --real
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description="Polymarket 15m Contrarian Bot")
    parser.add_argument("-c", "--config", default="config/polymarket_15m.yaml")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--real", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Inject env vars
    config["wallet_key"] = os.environ.get("POLYMARKET_WALLET_KEY", "")
    config["wallet_address"] = os.environ.get("POLYMARKET_WALLET_ADDRESS", "")
    config["api_key"] = os.environ.get("POLYMARKET_API_KEY", "")
    config["api_secret"] = os.environ.get("POLYMARKET_API_SECRET", "")
    config["api_passphrase"] = os.environ.get("POLYMARKET_API_PASSPHRASE", "")
    config["telegram_bot_token"] = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    config["telegram_chat_id"] = os.environ.get("TELEGRAM_CHAT_ID", "")

    if args.real:
        config["dry_run"] = False
        print("⚠️  REAL TRADING MODE")
    else:
        config["dry_run"] = True
        print("🧪 DRY RUN MODE")

    if not config["dry_run"] and not config["wallet_key"]:
        print("ERROR: POLYMARKET_WALLET_KEY not set")
        sys.exit(1)

    from qtrade.polymarket.runner_15m import run_once, run_loop

    if args.once:
        run_once(config, {"positions": []})
    else:
        run_loop(config)


if __name__ == "__main__":
    main()
