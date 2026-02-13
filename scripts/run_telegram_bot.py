#!/usr/bin/env python3
"""
Telegram Bot 常駐服務

與 cron --once 搭配使用：
    - cron 負責每小時跑交易（run_live.py --once）
    - 本腳本負責 24/7 接收 Telegram 命令（/positions, /signals 等）

使用方式：
    # 前景運行（測試）
    python scripts/run_telegram_bot.py -c config/futures_rsi_adx_atr.yaml --real

    # 背景運行（正式）
    nohup python scripts/run_telegram_bot.py -c config/futures_rsi_adx_atr.yaml --real >> logs/telegram_bot.log 2>&1 &

    # 獨立測試（無 broker，只測連線）
    python scripts/run_telegram_bot.py

環境變數（.env）：
    TELEGRAM_BOT_TOKEN=your_bot_token
    TELEGRAM_CHAT_ID=your_chat_id
"""
import argparse
import os
import sys
from pathlib import Path

# 添加專案路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from dotenv import load_dotenv

# 載入環境變數
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path)


def main():
    parser = argparse.ArgumentParser(description="Telegram Bot 常駐服務")
    parser.add_argument("-c", "--config", type=str, default=None,
                        help="配置檔路徑（例如 config/futures_rsi_adx_atr.yaml）")
    parser.add_argument("--real", action="store_true",
                        help="使用真實 Broker（需要 Binance API Key）")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run",
                        help="Broker 為 dry-run 模式（不下單，僅查詢）")
    args = parser.parse_args()

    print("=" * 60)
    print("🤖 Telegram Bot 常駐服務")
    print("=" * 60)

    broker = None
    runner = None
    state_manager = None

    if args.config:
        # ── 帶 config 模式：建立 broker + LiveRunner ──
        from qtrade.config import load_config
        from qtrade.monitor.notifier import TelegramNotifier

        cfg = load_config(args.config)
        market_type = cfg.market_type_str

        print(f"   配置: {args.config}")
        print(f"   市場: {market_type.upper()}")
        print(f"   交易對: {', '.join(cfg.market.symbols)}")

        if args.real:
            # 真實 Broker（dry-run=True：只查詢不下單）
            if market_type == "futures":
                from qtrade.live.binance_futures_broker import BinanceFuturesBroker
                leverage = cfg.futures.leverage if cfg.futures else 10
                margin_type = cfg.futures.margin_type if cfg.futures else "ISOLATED"
                broker = BinanceFuturesBroker(
                    dry_run=True,  # Bot 只查詢，永遠不下單
                    leverage=leverage,
                    margin_type=margin_type,
                    state_dir=cfg.get_report_dir("live"),
                )
            else:
                from qtrade.live.binance_spot_broker import BinanceSpotBroker
                broker = BinanceSpotBroker(dry_run=True)

            print(f"   Broker: {'Futures' if market_type == 'futures' else 'Spot'} (查詢模式)")
        else:
            print("   Broker: 無（僅限 /ping, /help, /price）")

        # 建立 LiveRunner（用於 /signals 命令）
        if broker:
            from qtrade.live.runner import LiveRunner
            notifier = TelegramNotifier.from_config(cfg.notification)
            runner = LiveRunner(
                cfg=cfg,
                broker=broker,
                mode="real" if args.real else "paper",
                notifier=notifier,
            )
            state_manager = runner.state_manager
            print(f"   LiveRunner: ✅（支援 /signals, /stats）")
    else:
        print("   ⚠️  未指定 config，僅支援基本命令（/ping, /help, /price）")

    print()
    print("   在 Telegram 發送 /help 查看可用命令")
    print("   按 Ctrl+C 停止")
    print("=" * 60)

    from qtrade.monitor.telegram_bot import TelegramCommandBot

    bot = TelegramCommandBot(
        live_runner=runner,
        broker=broker,
        state_manager=state_manager,
    )

    try:
        bot.run_polling()  # 阻塞式輪詢
    except KeyboardInterrupt:
        print("\n⛔ 收到停止信號")
    except ValueError as e:
        print(f"\n❌ 配置錯誤: {e}")
        print("\n請在 .env 文件中設置：")
        print("  TELEGRAM_BOT_TOKEN=your_bot_token")
        print("  TELEGRAM_CHAT_ID=your_chat_id")


if __name__ == "__main__":
    main()
