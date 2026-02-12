"""
互動式 Telegram Bot

擴展 TelegramNotifier，支援接收用戶命令並回覆。

功能：
    - /status - 查詢帳戶狀態
    - /positions - 查看當前持倉
    - /trades [n] - 查看最近 n 筆交易
    - /balance - 查看餘額
    - /price <symbol> - 查詢價格
    - /pnl - 查看今日盈虧
    - /help - 顯示幫助

使用方法：
    from qtrade.monitor.telegram_bot import TelegramBot
    
    bot = TelegramBot(broker=broker, notifier=notifier)
    bot.start()  # 開始監聽命令（非阻塞）

資源消耗：
    - 記憶體：幾乎不增加（只是 HTTP 長輪詢）
    - CPU：極低（每秒 1 次輪詢）
    - 網路：極低（每次約 1KB）
"""
from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable, Any
import requests

from ..utils.log import get_logger

if TYPE_CHECKING:
    from .notifier import TelegramNotifier

logger = get_logger("telegram_bot")


class TelegramBot:
    """
    互動式 Telegram Bot
    
    支援命令接收和回覆，不需要 Web UI，資源消耗極低。
    """
    
    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
        notifier: "TelegramNotifier | None" = None,
        broker: Any = None,
        poll_interval: float = 1.0,
        allowed_users: list[str] | None = None,
    ):
        """
        初始化 Telegram Bot
        
        Args:
            bot_token: Bot Token（None = 從環境變數讀取）
            chat_id: 允許的 Chat ID（None = 從環境變數讀取）
            notifier: TelegramNotifier 實例（用於發送訊息）
            broker: Broker 實例（用於查詢帳戶狀態）
            poll_interval: 輪詢間隔（秒）
            allowed_users: 允許使用的用戶 ID 列表（安全性）
        """
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self.notifier = notifier
        self.broker = broker
        self.poll_interval = poll_interval
        self.allowed_users = allowed_users or [self.chat_id]
        
        self._running = False
        self._thread: threading.Thread | None = None
        self._last_update_id = 0
        
        # 命令處理器
        self._commands: dict[str, Callable] = {}
        self._register_default_commands()
        
        # 狀態緩存（減少 API 調用）
        self._cache: dict[str, Any] = {}
        self._cache_ttl = 5  # 秒
        
        self.enabled = bool(self.bot_token and self.chat_id)
        if not self.enabled:
            logger.warning("⚠️  Telegram Bot 未啟用（缺少 BOT_TOKEN 或 CHAT_ID）")
    
    def _register_default_commands(self):
        """註冊預設命令"""
        self.register_command("start", self._cmd_start, "啟動 Bot")
        self.register_command("help", self._cmd_help, "顯示幫助")
        self.register_command("status", self._cmd_status, "帳戶狀態")
        self.register_command("balance", self._cmd_balance, "查看餘額")
        self.register_command("positions", self._cmd_positions, "當前持倉")
        self.register_command("trades", self._cmd_trades, "最近交易")
        self.register_command("price", self._cmd_price, "查詢價格")
        self.register_command("pnl", self._cmd_pnl, "今日盈虧")
        self.register_command("ping", self._cmd_ping, "測試連接")
    
    def register_command(self, name: str, handler: Callable, description: str = ""):
        """
        註冊自定義命令
        
        Args:
            name: 命令名稱（不含 /）
            handler: 處理函數，簽名：handler(args: list[str], chat_id: str) -> str
            description: 命令描述（用於 /help）
        """
        self._commands[name] = {"handler": handler, "description": description}
    
    # ══════════════════════════════════════════════════════════════
    # 核心方法
    # ══════════════════════════════════════════════════════════════
    
    def start(self):
        """啟動 Bot（非阻塞，背景執行）"""
        if not self.enabled:
            logger.warning("Telegram Bot 未啟用，跳過啟動")
            return
        
        if self._running:
            logger.warning("Telegram Bot 已在運行中")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info("🤖 Telegram Bot 已啟動，等待命令...")
    
    def stop(self):
        """停止 Bot"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("🛑 Telegram Bot 已停止")
    
    def _poll_loop(self):
        """長輪詢循環"""
        while self._running:
            try:
                updates = self._get_updates()
                for update in updates:
                    self._handle_update(update)
            except Exception as e:
                logger.error(f"輪詢錯誤: {e}")
            
            time.sleep(self.poll_interval)
    
    def _get_updates(self) -> list[dict]:
        """獲取新訊息"""
        url = f"https://api.telegram.org/bot{self.bot_token}/getUpdates"
        params = {
            "offset": self._last_update_id + 1,
            "timeout": 30,  # 長輪詢
            "allowed_updates": ["message"],
        }
        
        try:
            resp = requests.get(url, params=params, timeout=35)
            data = resp.json()
            
            if data.get("ok"):
                updates = data.get("result", [])
                if updates:
                    self._last_update_id = updates[-1]["update_id"]
                return updates
        except Exception as e:
            logger.error(f"獲取更新失敗: {e}")
        
        return []
    
    def _handle_update(self, update: dict):
        """處理單個更新"""
        message = update.get("message", {})
        text = message.get("text", "")
        chat_id = str(message.get("chat", {}).get("id", ""))
        user_id = str(message.get("from", {}).get("id", ""))
        
        # 安全檢查
        if chat_id not in self.allowed_users and user_id not in self.allowed_users:
            logger.warning(f"未授權的用戶嘗試訪問: {user_id}")
            self._send_message(chat_id, "⛔ 你沒有權限使用此 Bot")
            return
        
        # 解析命令
        if text.startswith("/"):
            parts = text[1:].split()
            command = parts[0].lower().split("@")[0]  # 移除 @botname
            args = parts[1:] if len(parts) > 1 else []
            
            self._execute_command(command, args, chat_id)
    
    def _execute_command(self, command: str, args: list[str], chat_id: str):
        """執行命令"""
        if command in self._commands:
            try:
                handler = self._commands[command]["handler"]
                response = handler(args, chat_id)
                if response:
                    self._send_message(chat_id, response)
            except Exception as e:
                logger.error(f"命令執行失敗 /{command}: {e}")
                self._send_message(chat_id, f"❌ 命令執行失敗: {e}")
        else:
            self._send_message(chat_id, f"❓ 未知命令: /{command}\n使用 /help 查看可用命令")
    
    def _send_message(self, chat_id: str, text: str, parse_mode: str = "HTML"):
        """發送訊息"""
        if self.notifier:
            self.notifier.send(text, parse_mode=parse_mode, add_prefix=False)
        else:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                "chat_id": chat_id,
                "text": text,
                "parse_mode": parse_mode,
            }
            try:
                requests.post(url, json=payload, timeout=10)
            except Exception as e:
                logger.error(f"發送訊息失敗: {e}")
    
    def _send_photo(self, chat_id: str, photo_path: str, caption: str = ""):
        """發送圖片"""
        url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"
        with open(photo_path, "rb") as f:
            files = {"photo": f}
            data = {"chat_id": chat_id, "caption": caption}
            try:
                requests.post(url, files=files, data=data, timeout=30)
            except Exception as e:
                logger.error(f"發送圖片失敗: {e}")
    
    # ══════════════════════════════════════════════════════════════
    # 預設命令處理器
    # ══════════════════════════════════════════════════════════════
    
    def _cmd_start(self, args: list[str], chat_id: str) -> str:
        return (
            "🤖 <b>Trading Bot 已啟動</b>\n\n"
            "可用命令：\n"
            "/status - 帳戶狀態\n"
            "/positions - 當前持倉\n"
            "/balance - 查看餘額\n"
            "/trades [n] - 最近交易\n"
            "/price <symbol> - 查詢價格\n"
            "/pnl - 今日盈虧\n"
            "/help - 詳細幫助"
        )
    
    def _cmd_help(self, args: list[str], chat_id: str) -> str:
        lines = ["📖 <b>命令列表</b>\n"]
        for name, info in self._commands.items():
            desc = info.get("description", "")
            lines.append(f"/{name} - {desc}")
        return "\n".join(lines)
    
    def _cmd_ping(self, args: list[str], chat_id: str) -> str:
        return "🏓 Pong! Bot 運行正常"
    
    def _cmd_status(self, args: list[str], chat_id: str) -> str:
        """帳戶狀態"""
        if not self.broker:
            return "⚠️ Broker 未連接"
        
        try:
            # 嘗試獲取帳戶資訊
            if hasattr(self.broker, "get_account_summary"):
                summary = self.broker.get_account_summary()
                return self._format_account_summary(summary)
            elif hasattr(self.broker, "account"):
                return self._format_account_summary(self.broker.account)
            else:
                return "⚠️ 無法獲取帳戶資訊"
        except Exception as e:
            return f"❌ 獲取狀態失敗: {e}"
    
    def _cmd_balance(self, args: list[str], chat_id: str) -> str:
        """查看餘額"""
        if not self.broker:
            return "⚠️ Broker 未連接"
        
        try:
            if hasattr(self.broker, "get_balance"):
                balance = self.broker.get_balance()
                return self._format_balance(balance)
            elif hasattr(self.broker, "balance"):
                return self._format_balance(self.broker.balance)
            else:
                return "⚠️ 無法獲取餘額"
        except Exception as e:
            return f"❌ 獲取餘額失敗: {e}"
    
    def _cmd_positions(self, args: list[str], chat_id: str) -> str:
        """當前持倉（含 SL/TP 掛單與預估盈虧）"""
        if not self.broker:
            return "⚠️ Broker 未連接"
        
        try:
            if hasattr(self.broker, "get_positions"):
                positions = self.broker.get_positions()
            elif hasattr(self.broker, "positions"):
                positions = self.broker.positions
            else:
                return "⚠️ 無法獲取持倉"
            
            if not positions:
                return "📭 目前沒有持倉"
            
            # 查詢每個 symbol 的 SL/TP 掛單
            sl_tp_map: dict[str, dict] = {}
            if hasattr(self.broker, "get_all_conditional_orders"):
                for pos in positions:
                    sym = pos.symbol if hasattr(pos, "symbol") else pos.get("symbol", "")
                    if not sym:
                        continue
                    try:
                        orders = self.broker.get_all_conditional_orders(sym)
                        sl_tp_map[sym] = {"sl": None, "tp": None}
                        for o in orders:
                            otype = o.get("type", "")
                            trigger = float(o.get("stopPrice", 0) or o.get("triggerPrice", 0) or 0)
                            if otype in {"STOP_MARKET", "STOP"} and trigger > 0:
                                sl_tp_map[sym]["sl"] = trigger
                            elif otype in {"TAKE_PROFIT_MARKET", "TAKE_PROFIT"} and trigger > 0:
                                sl_tp_map[sym]["tp"] = trigger
                    except Exception:
                        pass  # 查詢失敗不影響持倉顯示
            
            return self._format_positions(positions, sl_tp_map=sl_tp_map)
        except Exception as e:
            return f"❌ 獲取持倉失敗: {e}"
    
    def _cmd_trades(self, args: list[str], chat_id: str) -> str:
        """最近交易"""
        n = int(args[0]) if args else 5
        n = min(n, 20)  # 最多顯示 20 筆
        
        if not self.broker:
            return "⚠️ Broker 未連接"
        
        try:
            if hasattr(self.broker, "get_recent_trades"):
                trades = self.broker.get_recent_trades(n)
            elif hasattr(self.broker, "trade_history"):
                trades = list(self.broker.trade_history)[-n:]
            else:
                return "⚠️ 無法獲取交易記錄"
            
            if not trades:
                return "📭 沒有交易記錄"
            
            return self._format_trades(trades)
        except Exception as e:
            return f"❌ 獲取交易失敗: {e}"
    
    def _cmd_price(self, args: list[str], chat_id: str) -> str:
        """查詢價格"""
        if not args:
            return "❓ 請指定交易對，例如：/price BTCUSDT"
        
        symbol = args[0].upper()
        
        try:
            # 使用 Binance API 查詢價格
            url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
            resp = requests.get(url, timeout=5)
            data = resp.json()
            
            if "code" in data:
                return f"❌ 無效的交易對: {symbol}"
            
            price = float(data["lastPrice"])
            change_pct = float(data["priceChangePercent"])
            high = float(data["highPrice"])
            low = float(data["lowPrice"])
            volume = float(data["volume"])
            
            emoji = "📈" if change_pct > 0 else "📉"
            
            return (
                f"{emoji} <b>{symbol}</b>\n\n"
                f"💰 價格: <b>${price:,.2f}</b>\n"
                f"📊 24h 漲跌: {change_pct:+.2f}%\n"
                f"🔺 最高: ${high:,.2f}\n"
                f"🔻 最低: ${low:,.2f}\n"
                f"📦 成交量: {volume:,.0f}"
            )
        except Exception as e:
            return f"❌ 查詢價格失敗: {e}"
    
    def _cmd_pnl(self, args: list[str], chat_id: str) -> str:
        """今日盈虧"""
        if not self.broker:
            return "⚠️ Broker 未連接"
        
        try:
            if hasattr(self.broker, "get_daily_pnl"):
                pnl = self.broker.get_daily_pnl()
                return self._format_pnl(pnl)
            else:
                return "⚠️ 無法獲取盈虧資訊"
        except Exception as e:
            return f"❌ 獲取盈虧失敗: {e}"
    
    # ══════════════════════════════════════════════════════════════
    # 格式化輔助方法
    # ══════════════════════════════════════════════════════════════
    
    def _format_account_summary(self, summary: dict) -> str:
        """格式化帳戶摘要"""
        equity = summary.get("equity", 0)
        cash = summary.get("cash", summary.get("available", 0))
        positions_value = summary.get("positions_value", 0)
        total_pnl = summary.get("total_pnl", summary.get("unrealized_pnl", 0))
        
        pnl_emoji = "📈" if total_pnl >= 0 else "📉"
        
        return (
            f"💼 <b>帳戶狀態</b>\n\n"
            f"💰 總權益: <b>${equity:,.2f}</b>\n"
            f"💵 可用餘額: ${cash:,.2f}\n"
            f"📊 持倉價值: ${positions_value:,.2f}\n"
            f"{pnl_emoji} 未實現盈虧: ${total_pnl:+,.2f}"
        )
    
    def _format_balance(self, balance: dict | float) -> str:
        """格式化餘額"""
        if isinstance(balance, (int, float)):
            return f"💰 餘額: <b>${balance:,.2f}</b>"
        
        lines = ["💰 <b>餘額明細</b>\n"]
        for asset, amount in balance.items():
            if amount > 0:
                lines.append(f"• {asset}: {amount:,.8f}")
        return "\n".join(lines)
    
    def _format_positions(
        self, positions: dict | list, sl_tp_map: dict | None = None,
    ) -> str:
        """格式化持倉（含 SL/TP 掛單與預估盈虧）"""
        lines = ["📊 <b>當前持倉</b>\n"]
        sl_tp_map = sl_tp_map or {}
        
        if isinstance(positions, dict):
            positions = [{"symbol": k, **v} for k, v in positions.items()]
        
        for pos in positions:
            # 支援 dataclass (FuturesPosition) 和 dict
            if hasattr(pos, "symbol"):
                symbol = pos.symbol
                qty = pos.qty
                entry = pos.entry_price
                pnl = pos.unrealized_pnl
                mark = getattr(pos, "mark_price", 0)
                lev = getattr(pos, "leverage", 0)
            else:
                symbol = pos.get("symbol", "?")
                qty = pos.get("qty", pos.get("quantity", 0))
                entry = pos.get("avg_entry", pos.get("entry_price", 0))
                pnl = pos.get("unrealized_pnl", 0)
                mark = pos.get("mark_price", 0)
                lev = pos.get("leverage", 0)
            
            # 計算 PnL%
            notional = abs(qty * entry) if entry else 0
            pnl_pct = (pnl / notional * 100) if notional > 0 else 0
            
            is_long = qty > 0
            side_label = "LONG" if is_long else "SHORT"
            emoji = "🟢" if pnl >= 0 else "🔴"
            
            pos_lines = [
                f"{emoji} <b>{symbol}</b> [{side_label}]",
                f"   數量: {abs(qty):.6f}",
                f"   入場: ${entry:,.2f}",
            ]
            if mark > 0:
                pos_lines.append(f"   現價: ${mark:,.2f}")
            if lev and lev > 1:
                pos_lines.append(f"   槓桿: {lev}x")
            pos_lines.append(f"   盈虧: <b>${pnl:+,.2f}</b> ({pnl_pct:+.2f}%)")
            
            # SL/TP 掛單資訊
            sl_tp = sl_tp_map.get(symbol, {})
            sl_price = sl_tp.get("sl") if sl_tp else None
            tp_price = sl_tp.get("tp") if sl_tp else None
            
            if sl_price:
                sl_pnl = self._calc_pnl(entry, sl_price, abs(qty), is_long)
                pnl_str = f" (<b>{sl_pnl:+.2f}</b>)" if sl_pnl is not None else ""
                pos_lines.append(f"   🛡️ SL: ${sl_price:,.2f}{pnl_str}")
            if tp_price:
                tp_pnl = self._calc_pnl(entry, tp_price, abs(qty), is_long)
                pnl_str = f" (<b>{tp_pnl:+.2f}</b>)" if tp_pnl is not None else ""
                pos_lines.append(f"   🎯 TP: ${tp_price:,.2f}{pnl_str}")
            
            if not sl_price and not tp_price:
                pos_lines.append("   ⚠️ 無 SL/TP 掛單")
            
            lines.append("\n".join(pos_lines))
        
        return "\n\n".join(lines)
    
    @staticmethod
    def _calc_pnl(
        entry: float, target: float, qty: float, is_long: bool,
    ) -> float | None:
        """估算觸發 SL/TP 時的盈虧 (USDT)"""
        if entry <= 0 or qty <= 0:
            return None
        if is_long:
            return (target - entry) * qty
        else:
            return (entry - target) * qty
    
    def _format_trades(self, trades: list) -> str:
        """格式化交易記錄"""
        lines = ["📜 <b>最近交易</b>\n"]
        
        for trade in trades:
            symbol = trade.get("symbol", "?")
            side = trade.get("side", "?")
            qty = trade.get("qty", trade.get("quantity", 0))
            price = trade.get("price", 0)
            time_str = trade.get("time", "")
            
            emoji = "🟢" if side.upper() in ["BUY", "LONG"] else "🔴"
            lines.append(
                f"{emoji} {side} {symbol} @ ${price:,.2f} x {qty:.6f}"
            )
        
        return "\n".join(lines)
    
    def _format_pnl(self, pnl: dict) -> str:
        """格式化盈虧"""
        today = pnl.get("today", 0)
        realized = pnl.get("realized", 0)
        unrealized = pnl.get("unrealized", 0)
        
        emoji = "📈" if today >= 0 else "📉"
        
        return (
            f"{emoji} <b>今日盈虧</b>\n\n"
            f"💰 今日總計: <b>${today:+,.2f}</b>\n"
            f"✅ 已實現: ${realized:+,.2f}\n"
            f"⏳ 未實現: ${unrealized:+,.2f}"
        )


# ══════════════════════════════════════════════════════════════
# TelegramCommandBot — 整合 LiveRunner 的進階版
# ══════════════════════════════════════════════════════════════

class TelegramCommandBot(TelegramBot):
    """
    進階 Telegram Bot，整合 LiveRunner 和 TradingStateManager。

    額外支援：
        /signals - 即時生成交易信號
        /stats   - 查看交易統計（勝率、PnL 等）

    使用方式：
        bot = TelegramCommandBot(live_runner=runner, broker=broker)
        bot.start_background()  # 非阻塞
    """

    def __init__(
        self,
        live_runner: Any = None,
        broker: Any = None,
        state_manager: Any = None,
        notifier: "TelegramNotifier | None" = None,
        **kwargs,
    ):
        # 嘗試從 live_runner 推斷缺少的參數
        if live_runner and not broker:
            broker = getattr(live_runner, "broker", None)
        if live_runner and not notifier:
            notifier = getattr(live_runner, "notifier", None)
        if live_runner and not state_manager:
            state_manager = getattr(live_runner, "state_manager", None)

        super().__init__(broker=broker, notifier=notifier, **kwargs)

        self.live_runner = live_runner
        self.state_manager = state_manager

        # 註冊額外命令
        self.register_command("signals", self._cmd_signals, "即時信號")
        self.register_command("stats", self._cmd_stats, "交易統計")

    # ── 別名方法，與 run_live.py 期望的介面一致 ──

    def start_background(self):
        """啟動 Bot（非阻塞，背景執行）— start() 的別名"""
        self.start()

    def run_polling(self):
        """阻塞式輪詢（用於獨立運行模式）"""
        if not self.enabled:
            raise ValueError(
                "Telegram Bot 未啟用（缺少 BOT_TOKEN 或 CHAT_ID）"
            )
        self._running = True
        logger.info("🤖 Telegram Bot 已啟動（阻塞模式），等待命令...")
        try:
            self._poll_loop()
        except KeyboardInterrupt:
            logger.info("🛑 收到停止信號")
        finally:
            self._running = False
            logger.info("🛑 Telegram Bot 已停止")

    # ── /signals ──

    def _cmd_signals(self, args: list[str], chat_id: str) -> str:
        """即時生成交易信號"""
        if not self.live_runner:
            return "⚠️ LiveRunner 未連接，無法生成信號"

        try:
            from ..live.signal_generator import generate_signal

            runner = self.live_runner
            cfg = runner.cfg
            symbols = cfg.market.symbols
            strategy_name = cfg.strategy.name
            interval = cfg.market.interval
            market_type = cfg.market_type_str
            direction = cfg.direction
            params = dict(cfg.strategy.params) if cfg.strategy.params else {}

            lines = ["📡 <b>最新信號</b>\n"]

            for symbol in symbols:
                try:
                    sig = generate_signal(
                        symbol=symbol,
                        strategy_name=strategy_name,
                        params=params,
                        interval=interval,
                        market_type=market_type,
                        direction=direction,
                    )
                    signal_pct = sig["signal"]
                    price = sig["price"]
                    ind = sig.get("indicators", {})

                    if signal_pct > 0.5:
                        emoji = "🟢"
                        label = f"LONG {signal_pct:.0%}"
                    elif signal_pct < -0.5:
                        emoji = "🔴"
                        label = f"SHORT {abs(signal_pct):.0%}"
                    else:
                        emoji = "⚪"
                        label = f"FLAT {signal_pct:.0%}"

                    lines.append(
                        f"{emoji} <b>{symbol}</b>: {label} @ ${price:,.2f}\n"
                        f"   RSI={ind.get('rsi', '?')} | ADX={ind.get('adx', '?')}"
                    )
                except Exception as e:
                    lines.append(f"❌ {symbol}: {e}")

            return "\n".join(lines)
        except Exception as e:
            return f"❌ 信號生成失敗: {e}"

    # ── /stats ──

    def _cmd_stats(self, args: list[str], chat_id: str) -> str:
        """交易統計"""
        if not self.state_manager:
            return "⚠️ 交易狀態管理器未連接"

        try:
            state = self.state_manager.state
            trades = state.trades or []

            if not trades:
                return "📊 <b>交易統計</b>\n\n📭 尚無交易記錄"

            total = len(trades)
            wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
            losses = sum(1 for t in trades if t.get("pnl", 0) < 0)
            win_rate = (wins / total * 100) if total > 0 else 0
            total_pnl = sum(t.get("pnl", 0) for t in trades)
            total_fee = sum(t.get("fee", 0) for t in trades)

            return (
                f"📊 <b>交易統計</b>\n\n"
                f"📝 總交易: {total} 筆\n"
                f"✅ 獲勝: {wins} 筆\n"
                f"❌ 虧損: {losses} 筆\n"
                f"🎯 勝率: {win_rate:.1f}%\n"
                f"💰 累積 PnL: <b>${total_pnl:+,.2f}</b>\n"
                f"💸 總手續費: ${total_fee:,.2f}"
            )
        except Exception as e:
            return f"❌ 獲取統計失敗: {e}"


# ══════════════════════════════════════════════════════════════
# 快捷函數
# ══════════════════════════════════════════════════════════════

def create_bot(
    broker: Any = None,
    notifier: "TelegramNotifier | None" = None,
) -> TelegramBot:
    """
    創建 Telegram Bot 的快捷函數
    
    Args:
        broker: Broker 實例
        notifier: TelegramNotifier 實例
    
    Returns:
        TelegramBot 實例
    """
    return TelegramBot(broker=broker, notifier=notifier)
