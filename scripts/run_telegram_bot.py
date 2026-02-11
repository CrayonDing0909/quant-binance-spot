#!/usr/bin/env python3
"""
Telegram Bot 獨立運行腳本

使用方式：
    # 方式 1：獨立測試 Telegram 命令
    python scripts/run_telegram_bot.py
    
    # 方式 2：與 LiveRunner 整合（見下方範例）

環境變數設置（在 .env 文件中）：
    TELEGRAM_BOT_TOKEN=your_bot_token
    TELEGRAM_CHAT_ID=your_chat_id
    TELEGRAM_ADMIN_IDS=123456789,987654321  # 可選，限制哪些用戶可執行命令
"""
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


def run_standalone():
    """獨立運行 Telegram Bot（用於測試）"""
    from qtrade.monitor.telegram_bot import TelegramCommandBot
    
    print("=" * 60)
    print("🤖 Telegram Bot 獨立模式")
    print("=" * 60)
    print()
    print("此模式用於測試 Telegram 命令是否正常工作。")
    print("因為沒有連接 LiveRunner，部分命令會顯示模擬數據。")
    print()
    print("在 Telegram 中向你的 Bot 發送 /help 查看可用命令")
    print()
    print("按 Ctrl+C 停止")
    print("=" * 60)
    
    try:
        bot = TelegramCommandBot()
        bot.run_polling()
    except KeyboardInterrupt:
        print("\n⛔ 收到停止信號")
    except ValueError as e:
        print(f"\n❌ 配置錯誤: {e}")
        print("\n請在 .env 文件中設置：")
        print("  TELEGRAM_BOT_TOKEN=your_bot_token")
        print("  TELEGRAM_CHAT_ID=your_chat_id")


def example_with_live_runner():
    """
    與 LiveRunner 整合的範例（供參考）
    
    這是一個範例，展示如何在 run_live.py 中整合 Telegram Bot
    """
    from qtrade.config import load_config
    from qtrade.live.runner import LiveRunner
    from qtrade.live.paper_broker import PaperBroker
    from qtrade.monitor.telegram_bot import TelegramCommandBot
    
    # 載入配置
    cfg = load_config("config/futures_rsi_adx_atr.yaml")
    
    # 創建 Broker
    broker = PaperBroker(
        initial_cash=10000,
        fee_rate=0.001,
    )
    
    # 創建 LiveRunner
    runner = LiveRunner(
        cfg=cfg,
        broker=broker,
        mode="paper",
    )
    
    # 創建 Telegram Bot（與 runner 整合）
    telegram_bot = TelegramCommandBot(
        live_runner=runner,
        broker=broker,
        state_manager=runner.state_manager,
    )
    
    # 背景啟動 Telegram Bot
    telegram_bot.start_background()
    
    print("✅ Telegram Bot 已在背景運行")
    print("   你可以在 Telegram 中發送 /status 查看狀態")
    
    try:
        # 運行 LiveRunner（阻塞）
        runner.run()
    finally:
        # 停止 Telegram Bot
        telegram_bot.stop()


if __name__ == "__main__":
    # 檢查命令行參數
    if len(sys.argv) > 1 and sys.argv[1] == "--example":
        print("這是一個整合範例，請參考代碼修改你的 run_live.py")
        print()
        print("=" * 60)
        import inspect
        print(inspect.getsource(example_with_live_runner))
        print("=" * 60)
    else:
        run_standalone()
