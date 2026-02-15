#!/usr/bin/env python3
"""
WebSocket Live Trading 啟動腳本

Usage:
    python scripts/run_websocket.py -c config/futures_rsi_adx_atr.yaml --paper
    python scripts/run_websocket.py -c config/futures_rsi_adx_atr.yaml --real
"""
import sys
import argparse
from pathlib import Path

# 確保 src 在 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qtrade.config import load_config
from qtrade.utils.log import get_logger
from qtrade.live.websocket_runner import WebSocketRunner
from qtrade.live.paper_broker import PaperBroker
from qtrade.live.binance_futures_broker import BinanceFuturesBroker

logger = get_logger("main_ws")

def main():
    parser = argparse.ArgumentParser(description="WebSocket Live Trading Bot")
    parser.add_argument("-c", "--config", required=True, help="配置檔案路徑")
    parser.add_argument("--paper", action="store_true", help="啟用 Paper Trading 模式")
    parser.add_argument("--real", action="store_true", help="啟用 Real Trading 模式")
    parser.add_argument("--dry-run", action="store_true", help="Real 模式下僅記錄不發送訂單")
    
    args = parser.parse_args()

    # 載入配置
    cfg = load_config(args.config)
    # setup_logging(cfg.logging) # 用預設 logging

    # 決定模式
    if args.real:
        mode = "real"
        if not cfg.market_type_str == "futures":
            logger.error("❌ Real Trading 目前僅支援 Futures 模式")
            sys.exit(1)
            
        broker = BinanceFuturesBroker(
            dry_run=args.dry_run,
            leverage=cfg.futures.leverage if cfg.futures else 1,
            margin_type=cfg.futures.margin_type if cfg.futures else "ISOLATED",
            state_dir=cfg.get_report_dir("live"),
            prefer_limit=cfg.live.prefer_limit_order,
            limit_timeout_s=cfg.live.limit_order_timeout_s,
        )
    else:
        mode = "paper"
        broker = PaperBroker(
            initial_cash=10000.0,
            fee_rate=0.001,
            slippage=0.0005, # 模擬滑點
        )

    # 啟動 WebSocket Runner
    try:
        runner = WebSocketRunner(cfg, broker, mode=mode)
        runner.run()
    except KeyboardInterrupt:
        logger.info("👋 Bot 已停止")
    except Exception as e:
        logger.exception(f"❌ 發生未預期錯誤: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
