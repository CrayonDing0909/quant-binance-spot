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

支援 Spot/Futures 分開通知：
    # 方法 1：直接傳參數
    spot_notifier = TelegramNotifier(
        bot_token=os.getenv("SPOT_TELEGRAM_BOT_TOKEN"),
        chat_id=os.getenv("SPOT_TELEGRAM_CHAT_ID"),
        prefix="🟢 [SPOT]"
    )
    
    # 方法 2：從 NotificationConfig 建立
    notifier = TelegramNotifier.from_config(cfg.notification)
"""
from __future__ import annotations
import os
import requests
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from ..utils.log import get_logger

if TYPE_CHECKING:
    from ..config import NotificationConfig

logger = get_logger("telegram")


class TelegramNotifier:
    """
    Telegram Bot 通知器
    
    支援：
    - 多 Bot（Spot / Futures 各用不同 Bot）
    - 訊息前綴（在同一個 Chat 區分來源）
    """

    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
        prefix: str = "",
        enabled: bool = True,
    ):
        """
        初始化 Telegram 通知器
        
        Args:
            bot_token: Bot Token（None = 從環境變數讀取）
            chat_id: Chat ID（None = 從環境變數讀取）
            prefix: 訊息前綴，例如 "🟢 [SPOT]" 或 "🔴 [FUTURES]"
            enabled: 是否啟用（可用於臨時停用）
        """
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self.prefix = prefix
        self._user_enabled = enabled
        
        # 實際啟用狀態：用戶啟用 + 有 token + 有 chat_id
        self.enabled = enabled and bool(self.bot_token and self.chat_id)

        if self._user_enabled and not self.enabled:
            logger.warning(
                "⚠️  Telegram 通知未啟用（缺少 TELEGRAM_BOT_TOKEN 或 TELEGRAM_CHAT_ID）\n"
                "   設置方法：在 .env 中加入 TELEGRAM_BOT_TOKEN 和 TELEGRAM_CHAT_ID"
            )
        elif self.enabled and self.prefix:
            logger.info(f"✅ Telegram 通知已啟用，前綴: {self.prefix}")

    @classmethod
    def from_config(cls, config: "NotificationConfig | None") -> "TelegramNotifier":
        """
        從 NotificationConfig 建立通知器
        
        Args:
            config: NotificationConfig 或 None（None = 使用預設環境變數）
            
        Returns:
            TelegramNotifier 實例
        """
        if config is None:
            return cls()
        return cls(
            bot_token=config.telegram_bot_token,
            chat_id=config.telegram_chat_id,
            prefix=config.prefix,
            enabled=config.enabled,
        )

    def _format_message(self, text: str) -> str:
        """加上前綴（如果有的話）"""
        if self.prefix:
            return f"{self.prefix}\n\n{text}"
        return text

    def send(self, text: str, parse_mode: str = "HTML", add_prefix: bool = True) -> bool:
        """
        發送文字訊息

        Args:
            text: 訊息內容（支援 HTML 格式）
            parse_mode: "HTML" 或 "Markdown"
            add_prefix: 是否加上前綴（預設 True）

        Returns:
            是否發送成功
        """
        if not self.enabled:
            return False

        # 加上前綴
        formatted_text = self._format_message(text) if add_prefix else text

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": formatted_text,
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
        leverage: int | None = None,
        liquidation_price: float | None = None,
        stop_loss_price: float | None = None,
        take_profit_price: float | None = None,
    ) -> bool:
        """
        發送交易通知
        
        Args:
            symbol: 交易對
            side: BUY / SELL / LONG / SHORT / CLOSE_LONG / CLOSE_SHORT
            qty: 數量
            price: 價格（入場均價）
            reason: 原因
            pnl: 盈虧
            weight: 倉位權重
            leverage: 槓桿（合約專用）
            liquidation_price: 強平價格（合約專用）
            stop_loss_price: 止損價格
            take_profit_price: 止盈價格
        """
        # 根據 side 決定 emoji 和標籤
        side_map = {
            "BUY": ("🟢", "BUY"),
            "SELL": ("🔴", "SELL"),
            "LONG": ("🟢", "LONG"),
            "SHORT": ("🔴", "SHORT"),
            "CLOSE_LONG": ("📤", "CLOSE LONG"),
            "CLOSE_SHORT": ("📥", "CLOSE SHORT"),
        }
        emoji, side_label = side_map.get(side.upper(), ("⚪", side))
        
        pnl_str = ""
        if pnl is not None:
            pnl_emoji = "📈" if pnl > 0 else "📉"
            pnl_str = f"\n  PnL: <b>{pnl:+.2f}</b> {pnl_emoji}"

        weight_str = f" ({weight:.0%})" if weight is not None else ""
        
        # 合約專屬資訊
        leverage_str = f" ({leverage}x)" if leverage and leverage > 1 else ""
        liq_str = ""
        if liquidation_price:
            liq_str = f"\n  🚨 強平價: ${liquidation_price:,.2f}"
        
        # 止損止盈（含預估盈虧）
        is_long = side.upper() in {"BUY", "LONG"}
        is_short = side.upper() in {"SELL", "SHORT"}

        sl_str = ""
        if stop_loss_price and stop_loss_price > 0:
            sl_pnl = self._estimate_pnl(
                entry=price, target=stop_loss_price,
                qty=qty, is_long=is_long, is_short=is_short,
            )
            sl_str = f"\n  🛡️ 止損: ${stop_loss_price:,.2f}"
            if sl_pnl is not None:
                sl_str += f" (<b>{sl_pnl:+.2f} USDT</b>)"

        tp_str = ""
        if take_profit_price and take_profit_price > 0:
            tp_pnl = self._estimate_pnl(
                entry=price, target=take_profit_price,
                qty=qty, is_long=is_long, is_short=is_short,
            )
            tp_str = f"\n  🎯 止盈: ${take_profit_price:,.2f}"
            if tp_pnl is not None:
                tp_str += f" (<b>{tp_pnl:+.2f} USDT</b>)"

        msg = (
            f"{emoji} <b>{side_label} {symbol}</b>{weight_str}{leverage_str}\n"
            f"  📍 入場: ${price:,.2f}\n"
            f"  📦 數量: {qty:.6f}"
            f"{sl_str}"
            f"{tp_str}"
            f"{liq_str}"
            f"\n  📝 原因: {reason}"
            f"{pnl_str}"
        )
        return self.send(msg)

    @staticmethod
    def _estimate_pnl(
        entry: float, target: float, qty: float,
        is_long: bool = False, is_short: bool = False,
    ) -> float | None:
        """
        估算 SL/TP 觸發時的盈虧

        Returns:
            預估 PnL (USDT)，無法判斷方向時回傳 None
        """
        if is_long:
            return (target - entry) * abs(qty)
        elif is_short:
            return (entry - target) * abs(qty)
        return None

    def send_signal_summary(
        self, 
        signals: list[dict], 
        mode: str = "PAPER",
        has_trade: bool = False,
    ) -> bool:
        """
        發送信號摘要（每個 tick 結束後）
        
        Args:
            signals: 信號列表
            mode: PAPER / REAL
            has_trade: 這次 tick 是否有交易
        """
        now = datetime.now(timezone.utc).strftime("%m-%d %H:%M UTC")
        
        # 交易狀態指示
        if has_trade:
            trade_status = "✅ <b>已下單</b>"
        else:
            trade_status = "💤 無交易（倉位不變）"
        
        lines = [f"📊 <b>Signal Tick</b> [{mode}] @ {now}\n{trade_status}\n"]

        for sig in signals:
            ind = sig.get("indicators", {})
            signal_pct = sig["signal"]
            
            # 支援做空信號：[-1, 1]
            # 🟢 = 做多 (> 0.5)，🔴 = 做空 (< -0.5)，⚪ = 空倉
            if signal_pct > 0.01:
                emoji = "🟢"
                signal_label = f"LONG {signal_pct:.0%}"
            elif signal_pct < -0.01:
                emoji = "🔴"
                signal_label = f"SHORT {abs(signal_pct):.0%}"
            else:
                emoji = "⚪"
                signal_label = "FLAT"
            
            # 指標行
            ind_parts = [
                f"RSI={ind.get('rsi', '?')}",
                f"ADX={ind.get('adx', '?')}",
                f"+DI={ind.get('plus_di', '?')} -DI={ind.get('minus_di', '?')}",
            ]
            if "er" in ind:
                ind_parts.append(f"ER={ind['er']}")
            ind_str = " | ".join(ind_parts)

            sig_lines = (
                f"{emoji} <b>{sig['symbol']}</b>: "
                f"{signal_label}, "
                f"${sig['price']:,.2f}\n"
                f"   {ind_str}"
            )
            
            # 附加持倉 + SL/TP 資訊（由 runner 注入）
            pos_info = sig.get("_position", {})
            pos_pct = pos_info.get("pct", 0)
            if abs(pos_pct) > 0.01:
                side = pos_info.get("side", "?")
                entry = pos_info.get("entry", 0)
                qty = pos_info.get("qty", 0)
                is_long = side == "LONG"
                
                pos_str = f"\n   📦 {side} {pos_pct:+.0%}"
                if entry > 0:
                    pos_str += f" @ ${entry:,.2f}"
                sig_lines += pos_str
                
                sl = pos_info.get("sl")
                tp = pos_info.get("tp")
                if sl and entry > 0 and qty > 0:
                    sl_pnl = self._estimate_pnl(entry, sl, qty, is_long=is_long, is_short=not is_long)
                    pnl_str = f" (<b>{sl_pnl:+.2f}</b>)" if sl_pnl is not None else ""
                    sig_lines += f"\n   🛡️ SL: ${sl:,.2f}{pnl_str}"
                if tp and entry > 0 and qty > 0:
                    tp_pnl = self._estimate_pnl(entry, tp, qty, is_long=is_long, is_short=not is_long)
                    pnl_str = f" (<b>{tp_pnl:+.2f}</b>)" if tp_pnl is not None else ""
                    sig_lines += f"\n   🎯 TP: ${tp:,.2f}{pnl_str}"
                if not sl and not tp:
                    sig_lines += "\n   ⚠️ 無 SL/TP 掛單"
            
            lines.append(sig_lines)

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
        market_type: str = "spot",
        leverage: int | None = None,
    ) -> bool:
        """
        發送啟動通知
        
        Args:
            strategy: 策略名稱
            symbols: 交易對列表
            interval: K 線週期
            mode: paper / real
            weights: 倉位分配
            market_type: spot / futures
            leverage: 槓桿倍數（合約專用）
        """
        alloc = ""
        if weights:
            alloc = "\n  分配: " + ", ".join(f"{s}={w:.0%}" for s, w in weights.items())
        
        # 市場類型標籤
        market_emoji = "🟢" if market_type == "spot" else "🔴"
        market_label = "SPOT" if market_type == "spot" else "FUTURES"
        
        # 合約槓桿
        leverage_str = f" ({leverage}x)" if leverage and leverage > 1 else ""

        msg = (
            f"🚀 <b>Trading Bot 啟動</b> [{mode.upper()}]\n\n"
            f"  {market_emoji} 市場: {market_label}{leverage_str}\n"
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
