"""
Telegram 通知模块

发送交易信号、账户摘要、错误告警到 Telegram。

设置步骤：
    1. 在 Telegram 搜索 @BotFather，创建 Bot，获取 Token
    2. 在 Telegram 搜索 @userinfobot，获取你的 Chat ID
    3. 在 .env 中设置：
        TELEGRAM_BOT_TOKEN=xxxx:yyyyyyy
        TELEGRAM_CHAT_ID=123456789

使用方法：
    notifier = TelegramNotifier()  # 自动读取 .env
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
                "⚠️  Telegram 通知未启用（缺少 TELEGRAM_BOT_TOKEN 或 TELEGRAM_CHAT_ID）\n"
                "   设置方法：在 .env 中加入 TELEGRAM_BOT_TOKEN 和 TELEGRAM_CHAT_ID"
            )

    def send(self, text: str, parse_mode: str = "HTML") -> bool:
        """
        发送文字消息

        Args:
            text: 消息内容（支持 HTML 格式）
            parse_mode: "HTML" 或 "Markdown"

        Returns:
            是否发送成功
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
                logger.error(f"Telegram 发送失败: {resp.status_code} {resp.text}")
                return False
        except Exception as e:
            logger.error(f"Telegram 发送异常: {e}")
            return False

    # ── 预定义消息模板 ────────────────────────────────

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
        """发送交易通知"""
        emoji = "🟢" if side == "BUY" else "🔴"
        pnl_str = ""
        if pnl is not None:
            pnl_emoji = "📈" if pnl > 0 else "📉"
            pnl_str = f"\n  PnL: <b>{pnl:+.2f}</b> {pnl_emoji}"

        weight_str = f" ({weight:.0%})" if weight is not None else ""

        msg = (
            f"{emoji} <b>{side} {symbol}</b>{weight_str}\n"
            f"  数量: {qty:.6f}\n"
            f"  价格: ${price:,.2f}\n"
            f"  原因: {reason}"
            f"{pnl_str}"
        )
        return self.send(msg)

    def send_signal_summary(self, signals: list[dict], mode: str = "PAPER") -> bool:
        """发送信号摘要（每个 tick 结束后）"""
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
        """发送账户摘要"""
        ret = (equity / initial_cash - 1) * 100
        emoji = "📈" if ret > 0 else "📉"

        lines = [
            f"💼 <b>Account [{mode}]</b> {emoji}\n",
            f"  初始: ${initial_cash:,.2f}",
            f"  权益: <b>${equity:,.2f}</b> ({ret:+.2f}%)",
            f"  现金: ${cash:,.2f}",
            f"  交易: {trade_count} 笔",
        ]

        if positions:
            lines.append("\n  持仓:")
            for sym, info in positions.items():
                lines.append(f"  • {sym}: {info['qty']:.6f} @ ${info['avg_entry']:,.2f}")

        return self.send("\n".join(lines))

    def send_error(self, error_msg: str) -> bool:
        """发送错误告警"""
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
        """发送启动通知"""
        alloc = ""
        if weights:
            alloc = "\n  分配: " + ", ".join(f"{s}={w:.0%}" for s, w in weights.items())

        msg = (
            f"🚀 <b>Trading Bot 启动</b> [{mode.upper()}]\n\n"
            f"  策略: {strategy}\n"
            f"  交易对: {', '.join(symbols)}\n"
            f"  周期: {interval}"
            f"{alloc}"
        )
        return self.send(msg)

    def send_shutdown(self, ticks: int, trades: int, hours: float) -> bool:
        """发送停止通知"""
        msg = (
            f"⛔ <b>Trading Bot 停止</b>\n\n"
            f"  运行: {hours:.1f}h\n"
            f"  Ticks: {ticks}\n"
            f"  交易: {trades} 笔"
        )
        return self.send(msg)

