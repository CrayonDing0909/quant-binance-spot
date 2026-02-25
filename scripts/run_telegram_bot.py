#!/usr/bin/env python3
"""
Telegram Bot 統一常駐服務

獨立進程運行，直連 Binance API 查詢帳戶狀態，
讀取各策略 Runner 寫出的信號快照 (last_signals.json)。

解決問題：
    - 多個 tmux session 共用同一 Bot Token 導致訊息互搶
    - 無法跨策略查看全局狀態

使用方式：
    # 單策略
    PYTHONPATH=src python scripts/run_telegram_bot.py \
        -c config/prod_candidate_meta_blend.yaml --real

    # 多策略（推薦）
    PYTHONPATH=src python scripts/run_telegram_bot.py \
        -c config/prod_candidate_meta_blend.yaml \
        -c config/prod_live_oi_liq_bounce.yaml \
        --real

    # 獨立測試（無 broker，只測連線）
    PYTHONPATH=src python scripts/run_telegram_bot.py

環境變數（.env）：
    TELEGRAM_BOT_TOKEN=your_bot_token
    TELEGRAM_CHAT_ID=your_chat_id
"""
import argparse
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
    parser = argparse.ArgumentParser(
        description="Telegram Bot 統一常駐服務（支援多策略）"
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        action="append",
        default=None,
        help="配置檔路徑（可多次指定，例如 -c config/a.yaml -c config/b.yaml）",
    )
    parser.add_argument(
        "--real", action="store_true",
        help="使用真實 Broker（需要 Binance API Key）",
    )
    parser.add_argument(
        "--dry-run", action="store_true", dest="dry_run",
        help="Broker 為 dry-run 模式（不下單，僅查詢）",
    )
    args = parser.parse_args()

    config_paths = args.config or []

    print("=" * 60)
    print("🤖 Telegram Bot 統一常駐服務")
    print("=" * 60)

    if not config_paths:
        print("   ⚠️  未指定 config，僅支援基本命令（/ping, /help）")
        print("   用法: run_telegram_bot.py -c config/a.yaml [-c config/b.yaml] --real")
        print()
        print("   在 Telegram 發送 /help 查看可用命令")
        print("   按 Ctrl+C 停止")
        print("=" * 60)

        from qtrade.monitor.telegram_bot import TelegramBot
        bot = TelegramBot()
        try:
            bot._set_bot_commands()
            bot._running = True
            bot._poll_loop()
        except KeyboardInterrupt:
            print("\n⛔ 收到停止信號")
        return

    # ── 載入多個配置 ──
    from qtrade.config import load_config

    configs: list[tuple[str, object]] = []
    all_symbols: list[str] = []

    for cp in config_paths:
        cfg = load_config(cp)
        strategy_name = cfg.strategy.name
        configs.append((strategy_name, cfg))
        all_symbols.extend(cfg.market.symbols)
        print(f"   📄 {cp}")
        print(f"      策略: {strategy_name}")
        print(f"      交易對: {', '.join(cfg.market.symbols)}")

    print(f"\n   📊 共 {len(configs)} 個策略, {len(set(all_symbols))} 個交易對")

    # ── 建立 Broker（唯一，dry_run=True 只查詢）──
    broker = None
    if args.real:
        # 使用第一個 futures config 的槓桿/margin 設定
        first_futures_cfg = None
        for _, cfg in configs:
            if cfg.market_type_str == "futures":
                first_futures_cfg = cfg
                break

        if first_futures_cfg:
            from qtrade.live.binance_futures_broker import BinanceFuturesBroker
            leverage = first_futures_cfg.futures.leverage if first_futures_cfg.futures else 1
            margin_type = first_futures_cfg.futures.margin_type if first_futures_cfg.futures else "ISOLATED"
            broker = BinanceFuturesBroker(
                dry_run=True,  # Bot 永遠不下單
                leverage=leverage,
                margin_type=margin_type,
                state_dir=first_futures_cfg.get_report_dir("live"),
            )
            print(f"   🔗 Broker: Futures (查詢模式)")
        else:
            from qtrade.live.binance_spot_broker import BinanceSpotBroker
            broker = BinanceSpotBroker(dry_run=True)
            print(f"   🔗 Broker: Spot (查詢模式)")
    else:
        print("   🔗 Broker: 無（僅限基本指令）")

    # ── 告警配置（從第一個 config 讀取，或用預設值）──
    alert_config = {}
    for _, cfg in configs:
        raw_path = getattr(cfg, "_config_path", None)
        if raw_path:
            try:
                import yaml
                with open(raw_path) as f:
                    raw = yaml.safe_load(f)
                tg_alerts = raw.get("telegram", {}).get("alerts", {})
                if tg_alerts:
                    alert_config = tg_alerts
                    print(f"   🔔 告警: {tg_alerts}")
                    break
            except Exception:
                pass

    print()
    print("   在 Telegram 發送 /help 查看可用命令")
    print("   按 Ctrl+C 停止")
    print("=" * 60)

    # ── 啟動 MultiStrategyBot ──
    from qtrade.monitor.multi_strategy_bot import MultiStrategyBot

    bot = MultiStrategyBot(
        configs=configs,
        broker=broker,
        alert_config=alert_config,
    )

    try:
        bot.run_polling()
    except KeyboardInterrupt:
        print("\n⛔ 收到停止信號")
    except ValueError as e:
        print(f"\n❌ 配置錯誤: {e}")
        print("\n請在 .env 文件中設置：")
        print("  TELEGRAM_BOT_TOKEN=your_bot_token")
        print("  TELEGRAM_CHAT_ID=your_chat_id")


if __name__ == "__main__":
    main()
