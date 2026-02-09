"""
Telegram 通知模組

發送交易信號、帳戶摘要、錯誤告警到 Telegram。

設置步驟：
    1. 在 Telegram 搜索 @BotFather，創建 Bot，獲取 Token
    2. 在 Telegram 搜索 @userinfobot，獲取你的 Chat ID
    3. 在 .env 中設置：
        TELEGRAM_BOT_TOKEN=xxxx:yyyyyyy
        TELEGRAM_CHAT_ID=123456789

使用方法：
    notifier = TelegramNotifier()  # 自動讀取 .env
    notifier.send("Hello!")
    notifier.send_trade(symbol="BTCUSDT", side="BUY", ...)
"""
from __future__ import annotations
import os
import requests
from datetime import datetime, timezone

from ..utils.log import get_logger

logger = get_logger("telegram")


class TelegramNotifier:
    """Telegram Bot 通知器"""

    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
    ):
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self.enabled = bool(self.bot_token and self.chat_id)

        if not self.enabled:
            logger.warning(
                "⚠️  Telegram 通知未啟用（缺少 TELEGRAM_BOT_TOKEN 或 TELEGRAM_CHAT_ID）\n"
                "   設置方法：在 .env 中加入 TELEGRAM_BOT_TOKEN 和 TELEGRAM_CHAT_ID"
            )

    def send(self, text: str, parse_mode: str = "HTML") -> bool:
        """
        發送文字訊息

        Args:
            text: 訊息內容（支援 HTML 格式）
            parse_mode: "HTML" 或 "Markdown"

        Returns:
            是否發送成功
        """
        if not self.enabled:
            return False

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }

        try:
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code == 200:
                return True
            else:
                logger.error(f"Telegram 發送失敗: {resp.status_code} {resp.text}")
                return False
        except Exception as e:
            logger.error(f"Telegram 發送異常: {e}")
            return False

    # ── 預定義訊息模板 ────────────────────────────────

    def send_trade(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        reason: str = "",
        pnl: float | None = None,
        weight: float | None = None,
    ) -> bool:
        """發送交易通知"""
        emoji = "🟢" if side == "BUY" else "🔴"
        pnl_str = ""
        if pnl is not None:
            pnl_emoji = "📈" if pnl > 0 else "📉"
            pnl_str = f"\n  PnL: <b>{pnl:+.2f}</b> {pnl_emoji}"

        weight_str = f" ({weight:.0%})" if weight is not None else ""

        msg = (
            f"{emoji} <b>{side} {symbol}</b>{weight_str}\n"
            f"  數量: {qty:.6f}\n"
            f"  價格: ${price:,.2f}\n"
            f"  原因: {reason}"
            f"{pnl_str}"
        )
        return self.send(msg)

    def send_signal_summary(self, signals: list[dict], mode: str = "PAPER") -> bool:
        """發送信號摘要（每個 tick 結束後）"""
        now = datetime.now(timezone.utc).strftime("%m-%d %H:%M UTC")
        lines = [f"📊 <b>Signal Tick</b> [{mode}] @ {now}\n"]

        for sig in signals:
            ind = sig.get("indicators", {})
            signal_pct = sig["signal"]
            emoji = "🟢" if signal_pct > 0.5 else "⚪"
            lines.append(
                f"{emoji} <b>{sig['symbol']}</b>: "
                f"signal={signal_pct:.0%}, "
                f"${sig['price']:,.2f}\n"
                f"   RSI={ind.get('rsi', '?')} | "
                f"ADX={ind.get('adx', '?')} | "
                f"+DI={ind.get('plus_di', '?')} -DI={ind.get('minus_di', '?')}"
            )

        return self.send("\n".join(lines))

    def send_account_summary(
        self,
        initial_cash: float,
        equity: float,
        cash: float,
        positions: dict,
        trade_count: int,
        mode: str = "PAPER",
    ) -> bool:
        """發送帳戶摘要"""
        ret = (equity / initial_cash - 1) * 100
        emoji = "📈" if ret > 0 else "📉"

        lines = [
            f"💼 <b>Account [{mode}]</b> {emoji}\n",
            f"  初始: ${initial_cash:,.2f}",
            f"  權益: <b>${equity:,.2f}</b> ({ret:+.2f}%)",
            f"  現金: ${cash:,.2f}",
            f"  交易: {trade_count} 筆",
        ]

        if positions:
            lines.append("\n  持倉:")
            for sym, info in positions.items():
                lines.append(f"  • {sym}: {info['qty']:.6f} @ ${info['avg_entry']:,.2f}")

        return self.send("\n".join(lines))

    def send_error(self, error_msg: str) -> bool:
        """發送錯誤告警"""
        msg = f"🚨 <b>ERROR</b>\n\n{error_msg}"
        return self.send(msg)

    def send_startup(
        self,
        strategy: str,
        symbols: list[str],
        interval: str,
        mode: str,
        weights: dict[str, float] | None = None,
    ) -> bool:
        """發送啟動通知"""
        alloc = ""
        if weights:
            alloc = "\n  分配: " + ", ".join(f"{s}={w:.0%}" for s, w in weights.items())

        msg = (
            f"🚀 <b>Trading Bot 啟動</b> [{mode.upper()}]\n\n"
            f"  策略: {strategy}\n"
            f"  交易對: {', '.join(symbols)}\n"
            f"  週期: {interval}"
            f"{alloc}"
        )
        return self.send(msg)

    def send_shutdown(self, ticks: int, trades: int, hours: float) -> bool:
        """發送停止通知"""
        msg = (
            f"⛔ <b>Trading Bot 停止</b>\n\n"
            f"  運行: {hours:.1f}h\n"
            f"  Ticks: {ticks}\n"
            f"  交易: {trades} 筆"
        )
        return self.send(msg)
