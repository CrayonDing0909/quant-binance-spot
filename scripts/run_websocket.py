#!/usr/bin/env python3
"""
WebSocket Live Trading 啟動腳本

Usage:
    # Paper Trading
    python scripts/run_websocket.py -c config/futures_rsi_adx_atr.yaml --paper

    # Real Trading
    python scripts/run_websocket.py -c config/futures_rsi_adx_atr.yaml --real

    # Real Trading (dry-run，不下單)
    python scripts/run_websocket.py -c config/futures_rsi_adx_atr.yaml --real --dry-run
"""
import sys
import argparse
import traceback
import logging
from pathlib import Path

# 確保 src 在 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qtrade.config import load_config
from qtrade.utils.log import get_logger

logger = get_logger("main_ws")


def main():
    parser = argparse.ArgumentParser(description="WebSocket Live Trading Bot")
    parser.add_argument("-c", "--config", required=True, help="配置檔案路徑")
    parser.add_argument("--paper", action="store_true", help="啟用 Paper Trading 模式")
    parser.add_argument("--real", action="store_true", help="啟用 Real Trading 模式")
    parser.add_argument("--dry-run", action="store_true", help="Real 模式下僅記錄不發送訂單")

    args = parser.parse_args()

    # 載入配置
    logger.info("📦 載入配置...")
    cfg = load_config(args.config)
    logger.info(f"   策略: {cfg.strategy.name}")
    logger.info(f"   交易對: {cfg.market.symbols}")
    logger.info(f"   市場: {cfg.market_type_str}")

    # 決定模式
    if args.real:
        mode = "real"
        if cfg.market_type_str != "futures":
            logger.error("❌ Real Trading 目前僅支援 Futures 模式")
            sys.exit(1)

        logger.info("🔧 初始化 Futures Broker...")
        from qtrade.live.binance_futures_broker import BinanceFuturesBroker
        broker = BinanceFuturesBroker(
            dry_run=args.dry_run,
            leverage=cfg.futures.leverage if cfg.futures else 1,
            margin_type=cfg.futures.margin_type if cfg.futures else "ISOLATED",
            state_dir=cfg.get_report_dir("live"),
            prefer_limit=cfg.live.prefer_limit_order,
            limit_timeout_s=cfg.live.limit_order_timeout_s,
        )
        logger.info("✅ Broker 已就緒")
    else:
        mode = "paper"
        logger.info("🔧 初始化 Paper Broker...")
        from qtrade.live.paper_broker import PaperBroker
        broker = PaperBroker(
            initial_cash=cfg.backtest.initial_cash,
            fee_bps=cfg.backtest.fee_bps,
            slippage_bps=cfg.backtest.slippage_bps,
            state_path=cfg.get_report_dir("live") / "paper_state.json",
            market_type=cfg.market_type_str,
            leverage=cfg.futures.leverage if cfg.futures else 1,
        )
        logger.info("✅ Paper Broker 已就緒")

    # 啟動 WebSocket Runner
    logger.info("🔧 初始化 WebSocket Runner...")
    from qtrade.live.websocket_runner import WebSocketRunner
    runner = WebSocketRunner(cfg, broker, mode=mode)

    logger.info("🚀 啟動中...")
    runner.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("👋 Bot 已停止")
    except Exception as e:
        logger.error(f"❌ 發生未預期錯誤: {e}")
        traceback.print_exc()
        sys.exit(1)
