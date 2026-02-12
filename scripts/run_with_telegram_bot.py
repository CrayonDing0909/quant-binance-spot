#!/usr/bin/env python3
"""
帶互動式 Telegram Bot 的交易腳本示例

這個腳本展示如何將互動式 Telegram Bot 整合到你的交易系統中。

功能：
    - 啟動交易 Bot 的同時啟動 Telegram 命令監聽
    - 支援透過 Telegram 查詢狀態、持倉、餘額等
    - 資源消耗極低（適合 Oracle Cloud 免費層）

使用方法：
    python scripts/run_with_telegram_bot.py

確保 .env 中有設置：
    TELEGRAM_BOT_TOKEN=xxxx:yyyyyyy
    TELEGRAM_CHAT_ID=123456789
"""
import sys
import time
from pathlib import Path

# 添加專案路徑
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qtrade.monitor import TelegramNotifier, TelegramBot


def main():
    print("=" * 60)
    print("🤖 Trading Bot with Telegram Commands")
    print("=" * 60)
    
    # ── 初始化 Notifier ──
    notifier = TelegramNotifier(prefix="🟢 [SPOT]")
    
    if not notifier.enabled:
        print("❌ Telegram 未配置，請在 .env 中設置：")
        print("   TELEGRAM_BOT_TOKEN=xxxx:yyyyyyy")
        print("   TELEGRAM_CHAT_ID=123456789")
        return
    
    # ── 初始化互動式 Bot ──
    # 注意：這裡 broker=None，你需要傳入你的 broker 實例
    # 例如：broker=BinanceSpotBroker(...)
    bot = TelegramBot(
        notifier=notifier,
        broker=None,  # TODO: 替換為你的 broker 實例
    )
    
    # 註冊自定義命令示例
    def cmd_custom(args, chat_id):
        return "🎉 這是自定義命令的回覆！"
    
    bot.register_command("custom", cmd_custom, "自定義命令示例")
    
    # ── 啟動 Bot ──
    bot.start()
    notifier.send("🚀 Trading Bot 已啟動！\n\n發送 /help 查看可用命令")
    
    print("✅ Telegram Bot 已啟動，等待命令...")
    print("   發送 /help 到你的 Telegram Bot 查看可用命令")
    print("   按 Ctrl+C 停止")
    
    try:
        # 主循環（你的交易邏輯放這裡）
        while True:
            # TODO: 在這裡放你的交易邏輯
            # 例如：
            # signal = strategy.generate_signal(data)
            # if signal:
            #     broker.execute_trade(signal)
            #     notifier.send_trade(...)
            
            time.sleep(60)  # 每分鐘檢查一次
            
    except KeyboardInterrupt:
        print("\n⏹️  正在停止...")
    finally:
        bot.stop()
        notifier.send("⛔ Trading Bot 已停止")
        print("✅ 已停止")


if __name__ == "__main__":
    main()
