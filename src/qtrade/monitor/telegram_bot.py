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
        self.register_command("help", self._cmd_help, "顯示幫助")
        self.register_command("status", self._cmd_status, "帳戶狀態（含 SL/TP）")
        self.register_command("balance", self._cmd_balance, "查看餘額")
        self.register_command("positions", self._cmd_positions, "當前持倉（詳細）")
        self.register_command("trades", self._cmd_trades, "最近交易")
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
        
        # 向 Telegram 註冊命令選單（輸入 / 時會出現提示）
        self._set_bot_commands()
        
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info("🤖 Telegram Bot 已啟動，等待命令...")
    
    def _set_bot_commands(self):
        """向 Telegram 註冊命令選單，讓使用者輸入 / 時看到命令提示"""
        try:
            commands = [
                {"command": name, "description": info["description"] or name}
                for name, info in self._commands.items()
            ]
            url = f"https://api.telegram.org/bot{self.bot_token}/setMyCommands"
            resp = requests.post(url, json={"commands": commands}, timeout=10)
            if resp.status_code == 200 and resp.json().get("ok"):
                logger.info(f"✅ 已註冊 {len(commands)} 個命令到 Telegram 選單")
            else:
                logger.warning(f"⚠️ 註冊命令選單失敗: {resp.text}")
        except Exception as e:
            logger.warning(f"⚠️ 註冊命令選單異常: {e}")
    
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
            "allowed_updates": ["message", "callback_query"],
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
        """處理單個更新（支援 message 和 callback_query）"""
        # ── 按鈕回調 (Inline Keyboard) ──
        callback = update.get("callback_query")
        if callback:
            cb_id = callback.get("id", "")
            data = callback.get("data", "")
            chat_id = str(callback.get("message", {}).get("chat", {}).get("id", ""))
            user_id = str(callback.get("from", {}).get("id", ""))

            # 安全檢查
            if chat_id not in self.allowed_users and user_id not in self.allowed_users:
                self._answer_callback(cb_id, "⛔ 無權限")
                return

            # 回應 Telegram（消除按鈕上的 loading 動畫）
            self._answer_callback(cb_id)

            # 解析 callback data 為命令
            if data.startswith("/"):
                parts = data[1:].split()
                command = parts[0].lower().split("@")[0]
                args = parts[1:] if len(parts) > 1 else []
                self._execute_command(command, args, chat_id)
            return

        # ── 一般訊息 ──
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
    
    def _send_message(
        self,
        chat_id: str,
        text: str,
        parse_mode: str = "HTML",
        reply_markup: dict | None = None,
    ):
        """發送訊息（可附帶 Inline Keyboard）"""
        if self.notifier and not reply_markup:
            self.notifier.send(text, parse_mode=parse_mode, add_prefix=False)
        else:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                "chat_id": chat_id or self.chat_id,
                "text": text,
                "parse_mode": parse_mode,
            }
            if reply_markup:
                payload["reply_markup"] = reply_markup
            try:
                requests.post(url, json=payload, timeout=10)
            except Exception as e:
                logger.error(f"發送訊息失敗: {e}")

    def _answer_callback(self, callback_query_id: str, text: str = ""):
        """回應 callback_query（消除按鈕 loading 動畫）"""
        url = f"https://api.telegram.org/bot{self.bot_token}/answerCallbackQuery"
        payload = {"callback_query_id": callback_query_id}
        if text:
            payload["text"] = text
        try:
            requests.post(url, json=payload, timeout=5)
        except Exception:
            pass
    
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
    
    def _cmd_help(self, args: list[str], chat_id: str) -> str:
        lines = ["📖 <b>命令列表</b>\n"]
        for name, info in self._commands.items():
            desc = info.get("description", "")
            lines.append(f"/{name} - {desc}")
        return "\n".join(lines)
    
    def _cmd_ping(self, args: list[str], chat_id: str) -> str:
        return "🏓 Pong! Bot 運行正常"
    
    def _cmd_status(self, args: list[str], chat_id: str) -> str:
        """帳戶狀態（含每個持倉的 SL/TP 摘要）"""
        if not self.broker:
            return "⚠️ Broker 未連接"
        
        try:
            # Futures broker: get_account_info() → raw Binance dict
            if hasattr(self.broker, "get_account_info"):
                info = self.broker.get_account_info()
                if not info:
                    return "⚠️ 無法獲取帳戶資訊（API 回傳空）"
                equity = float(info.get("totalWalletBalance", 0)) + float(info.get("totalUnrealizedProfit", 0))
                available = float(info.get("availableBalance", 0))
                unrealized = float(info.get("totalUnrealizedProfit", 0))
                
                positions = []
                if hasattr(self.broker, "get_positions"):
                    positions = self.broker.get_positions()
                
                pnl_emoji = "📈" if unrealized >= 0 else "📉"
                
                lines = [
                    f"💼 <b>帳戶狀態</b>\n",
                    f"💰 總權益: <b>${equity:,.2f}</b>",
                    f"💵 可用餘額: ${available:,.2f}",
                    f"{pnl_emoji} 未實現盈虧: ${unrealized:+,.2f}",
                ]
                
                # 每個持倉的摘要 + SL/TP
                if positions:
                    lines.append(f"\n📋 <b>持倉 ({len(positions)})</b>")
                    for pos in positions:
                        sym = pos.symbol if hasattr(pos, "symbol") else pos.get("symbol", "?")
                        qty = pos.qty if hasattr(pos, "qty") else pos.get("qty", 0)
                        entry = pos.entry_price if hasattr(pos, "entry_price") else pos.get("entry_price", 0)
                        pnl = pos.unrealized_pnl if hasattr(pos, "unrealized_pnl") else pos.get("unrealized_pnl", 0)
                        is_long = qty > 0
                        side = "LONG" if is_long else "SHORT"
                        emoji = "🟢" if pnl >= 0 else "🔴"
                        
                        lines.append(f"{emoji} <b>{sym}</b> [{side}] ${pnl:+,.2f}")
                        
                        # 查詢 SL/TP
                        if hasattr(self.broker, "get_all_conditional_orders"):
                            try:
                                orders = self.broker.get_all_conditional_orders(sym)
                                sl_price, tp_price = None, None
                                for o in orders:
                                    trigger = float(o.get("stopPrice", 0) or o.get("triggerPrice", 0) or 0)
                                    if trigger <= 0:
                                        continue
                                    otype = o.get("type", "")
                                    if otype in {"STOP_MARKET", "STOP"}:
                                        sl_price = trigger
                                    elif otype in {"TAKE_PROFIT_MARKET", "TAKE_PROFIT"}:
                                        tp_price = trigger
                                    elif entry > 0:
                                        # Algo orders fallback
                                        if is_long:
                                            if trigger < entry:
                                                sl_price = trigger
                                            else:
                                                tp_price = trigger
                                        else:
                                            if trigger > entry:
                                                sl_price = trigger
                                            else:
                                                tp_price = trigger
                                
                                sl_str, tp_str = "", ""
                                if sl_price:
                                    sl_pnl = self._calc_pnl(entry, sl_price, abs(qty), is_long)
                                    sl_str = f"   🛡️ SL: ${sl_price:,.2f}"
                                    if sl_pnl is not None:
                                        sl_str += f" ({sl_pnl:+.2f})"
                                    lines.append(sl_str)
                                if tp_price:
                                    tp_pnl = self._calc_pnl(entry, tp_price, abs(qty), is_long)
                                    tp_str = f"   🎯 TP: ${tp_price:,.2f}"
                                    if tp_pnl is not None:
                                        tp_str += f" ({tp_pnl:+.2f})"
                                    lines.append(tp_str)
                                if not sl_price and not tp_price:
                                    lines.append("   ⚠️ 無 SL/TP")
                            except Exception:
                                lines.append("   ⚠️ SL/TP 查詢失敗")
                else:
                    lines.append("\n📭 無持倉")
                
                return "\n".join(lines)
            # Paper broker
            elif hasattr(self.broker, "account"):
                return self._format_account_summary(vars(self.broker.account) if hasattr(self.broker.account, '__dict__') else self.broker.account)
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
                    entry = pos.entry_price if hasattr(pos, "entry_price") else pos.get("entry_price", 0)
                    is_long = (pos.qty if hasattr(pos, "qty") else pos.get("qty", 0)) > 0
                    if not sym:
                        continue
                    try:
                        orders = self.broker.get_all_conditional_orders(sym)
                        sl_tp_map[sym] = {"sl": None, "tp": None}
                        for o in orders:
                            trigger = float(o.get("stopPrice", 0) or o.get("triggerPrice", 0) or 0)
                            if trigger <= 0:
                                continue
                            otype = o.get("type", "")
                            
                            # 優先用 type 欄位判斷
                            if otype in {"STOP_MARKET", "STOP"}:
                                sl_tp_map[sym]["sl"] = trigger
                            elif otype in {"TAKE_PROFIT_MARKET", "TAKE_PROFIT"}:
                                sl_tp_map[sym]["tp"] = trigger
                            elif entry > 0:
                                # Algo orders 可能沒有 type 欄位
                                # 用觸發價 vs 入場價推斷 SL/TP
                                if is_long:
                                    # LONG: SL < entry, TP > entry
                                    if trigger < entry:
                                        sl_tp_map[sym]["sl"] = trigger
                                    else:
                                        sl_tp_map[sym]["tp"] = trigger
                                else:
                                    # SHORT: SL > entry, TP < entry
                                    if trigger > entry:
                                        sl_tp_map[sym]["sl"] = trigger
                                    else:
                                        sl_tp_map[sym]["tp"] = trigger
                    except Exception as e:
                        logger.debug(f"查詢 {sym} SL/TP 失敗: {e}")
            
            return self._format_positions(positions, sl_tp_map=sl_tp_map)
        except Exception as e:
            return f"❌ 獲取持倉失敗: {e}"
    
    def _cmd_trades(self, args: list[str], chat_id: str) -> str:
        """最近交易"""
        n = int(args[0]) if args else 10
        n = min(n, 20)  # 最多顯示 20 筆
        
        if not self.broker:
            return "⚠️ Broker 未連接"
        
        try:
            # 方式 1: Futures broker — get_trade_history()
            if hasattr(self.broker, "get_trade_history"):
                trades = self.broker.get_trade_history(limit=n)
                if trades:
                    return self._format_exchange_trades(trades)
            
            # 方式 2: Paper broker
            if hasattr(self.broker, "trade_history"):
                trades = list(self.broker.trade_history)[-n:]
                if trades:
                    return self._format_trades(trades)
            
            return "📭 沒有交易記錄"
        except Exception as e:
            return f"❌ 獲取交易失敗: {e}"
    
    def _fetch_income_paginated(self, start_dt, end_dt) -> list[dict]:
        """
        分頁查詢 Binance 收益歷史
        
        Binance /fapi/v1/income 限制 startTime~endTime 最多 7 天，
        超過就要分段查詢再合併。
        """
        from datetime import timedelta
        
        all_incomes: list[dict] = []
        window_start = start_dt
        
        while window_start < end_dt:
            window_end = min(window_start + timedelta(days=7), end_dt)
            start_ms = int(window_start.timestamp() * 1000)
            end_ms = int(window_end.timestamp() * 1000)
            
            chunk = self.broker.get_income_history(
                limit=1000, start_time=start_ms, end_time=end_ms
            )
            all_incomes.extend(chunk)
            window_start = window_end
        
        return all_incomes

    def _cmd_pnl(self, args: list[str], chat_id: str) -> str:
        """
        盈虧查詢，支援時間範圍
        
        /pnl        → 今日
        /pnl 7d     → 最近 7 天
        /pnl 30d    → 最近 30 天
        /pnl all    → 全部（最近 90 天）
        """
        if not self.broker:
            return "⚠️ Broker 未連接"
        
        try:
            if not hasattr(self.broker, "get_income_history"):
                return "⚠️ 無法獲取盈虧資訊"

            from datetime import timedelta
            now = datetime.now(timezone.utc)

            # 解析時間範圍
            period = (args[0].lower() if args else "").strip()
            if period == "7d":
                start_dt = now - timedelta(days=7)
                label = "最近 7 天"
            elif period == "30d":
                start_dt = now - timedelta(days=30)
                label = "最近 30 天"
            elif period == "all":
                start_dt = now - timedelta(days=90)
                label = "全部 (90天)"
            else:
                start_dt = now.replace(hour=0, minute=0, second=0, microsecond=0)
                label = f"今日 ({now.strftime('%m-%d')} UTC)"

            # Binance API 限制 startTime~endTime 最多 7 天
            # 超過 7 天的查詢需要分段
            incomes = self._fetch_income_paginated(start_dt, now)

            realized = sum(i["income"] for i in incomes if i["income_type"] == "REALIZED_PNL")
            commission = sum(i["income"] for i in incomes if i["income_type"] == "COMMISSION")
            funding = sum(i["income"] for i in incomes if i["income_type"] == "FUNDING_FEE")
            transfer = sum(i["income"] for i in incomes if i["income_type"] == "TRANSFER")

            # 未實現盈虧
            unrealized = 0.0
            if hasattr(self.broker, "get_positions"):
                for pos in self.broker.get_positions():
                    unrealized += pos.unrealized_pnl if hasattr(pos, "unrealized_pnl") else 0

            total = realized + commission + funding + unrealized
            emoji = "📈" if total >= 0 else "📉"

            # 統計交易筆數
            trade_count = sum(1 for i in incomes if i["income_type"] == "REALIZED_PNL")

            lines = [
                f"{emoji} <b>盈虧 — {label}</b>\n",
                f"💰 總計: <b>${total:+,.2f}</b>",
                f"✅ 已實現: ${realized:+,.2f}  ({trade_count} 筆)",
                f"⏳ 未實現: ${unrealized:+,.2f}",
                f"💸 手續費: ${commission:+,.2f}",
                f"🔄 資金費率: ${funding:+,.2f}",
            ]
            if abs(transfer) > 0.001:
                lines.append(f"💱 轉帳: ${transfer:+,.2f}")

            # 用 Inline Keyboard 讓用戶可以直接點按鈕切換時間範圍
            buttons = {
                "inline_keyboard": [[
                    {"text": "📅 今日", "callback_data": "/pnl"},
                    {"text": "📅 7天", "callback_data": "/pnl 7d"},
                    {"text": "📅 30天", "callback_data": "/pnl 30d"},
                    {"text": "📅 全部", "callback_data": "/pnl all"},
                ]]
            }
            self._send_message(chat_id, "\n".join(lines), reply_markup=buttons)
            return ""  # 已直接發送，不需要再回傳
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
        """格式化交易記錄（Paper Broker 格式）"""
        lines = ["📜 <b>最近交易</b>\n"]
        
        for trade in trades:
            symbol = trade.get("symbol", "?")
            side = trade.get("side", "?")
            qty = trade.get("qty", trade.get("quantity", 0))
            price = trade.get("price", 0)
            
            emoji = "🟢" if side.upper() in ["BUY", "LONG"] else "🔴"
            lines.append(
                f"{emoji} {side} {symbol} @ ${price:,.2f} x {qty:.6f}"
            )
        
        return "\n".join(lines)
    
    def _format_exchange_trades(self, trades: list[dict]) -> str:
        """格式化交易所交易記錄（Binance Futures 格式）"""
        lines = ["📜 <b>最近交易</b>\n"]
        
        for t in trades:
            symbol = t.get("symbol", "?")
            side = t.get("side", "?")
            pos_side = t.get("position_side", "")
            qty = t.get("qty", 0)
            price = t.get("price", 0)
            pnl = t.get("realized_pnl", 0)
            commission = t.get("commission", 0)
            ts = t.get("time", 0)
            
            # 格式化時間
            if isinstance(ts, (int, float)) and ts > 1e12:
                dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
                time_str = dt.strftime("%m-%d %H:%M")
            else:
                time_str = str(ts)[:16] if ts else "?"
            
            emoji = "🟢" if side == "BUY" else "🔴"
            pnl_str = f" 📈 ${pnl:+,.2f}" if abs(pnl) > 0.001 else ""
            fee_str = f" (fee: ${commission:.4f})" if commission > 0 else ""
            
            lines.append(
                f"{emoji} {time_str} {side}/{pos_side} {symbol}\n"
                f"   {qty:.6f} @ ${price:,.2f}{pnl_str}{fee_str}"
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
        self.register_command("risk", self._cmd_risk, "風險總覽")
        self.register_command("health", self._cmd_health, "系統狀態")

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
        self._set_bot_commands()
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
        """
        顯示最新交易信號

        優先讀取 cron (run_once) 產生的 last_signals.json，
        確保 /signals 顯示的信號與實際交易決策一致。
        若快照不存在或過時 (>2h)，才即時重新生成。
        """
        if not self.live_runner:
            return "⚠️ LiveRunner 未連接，無法生成信號"

        try:
            import json as _json
            from datetime import datetime as _dt, timezone as _tz

            cfg = self.live_runner.cfg
            sig_path = cfg.get_report_dir("live") / "last_signals.json"

            cached_signals = None
            cache_age_str = ""

            # ── 嘗試讀取 cron 產生的信號快照 ──
            if sig_path.exists():
                try:
                    with open(sig_path, "r") as f:
                        payload = _json.load(f)
                    gen_time = _dt.fromisoformat(payload["generated_at"])
                    age_sec = (_dt.now(_tz.utc) - gen_time).total_seconds()

                    if age_sec < 7200:  # 2 小時內有效
                        cached_signals = payload.get("signals", [])
                        mins = int(age_sec // 60)
                        cache_age_str = f"⏱ {mins} 分鐘前 ({gen_time.strftime('%H:%M UTC')})"
                except Exception:
                    pass  # 解析失敗，走 fallback

            # ── 有快照 → 使用快照信號 + 即時持倉/SL/TP ──
            if cached_signals:
                lines = [f"📡 <b>最新信號</b>  {cache_age_str}\n"]
                for sig in cached_signals:
                    sig_line = self._format_signal_line(sig)
                    lines.append(sig_line)
                return "\n".join(lines)

            # ── 無快照 → 即時生成（fallback）──
            from ..live.signal_generator import generate_signal

            # 嘗試使用 LiveRunner 的增量快取（如果可用）
            kline_cache = getattr(self.live_runner, "_kline_cache", None)

            lines = ["📡 <b>最新信號</b>  ⚡ 即時\n"]
            for symbol in cfg.market.symbols:
                try:
                    symbol_params = cfg.strategy.get_params(symbol)
                    cached_df = None
                    if kline_cache is not None:
                        cached_df = kline_cache.get_klines(symbol)
                        if cached_df is not None and len(cached_df) < 50:
                            cached_df = None
                    sig = generate_signal(
                        symbol=symbol,
                        strategy_name=cfg.strategy.name,
                        params=symbol_params,
                        interval=cfg.market.interval,
                        market_type=cfg.market_type_str,
                        direction=cfg.direction,
                        df=cached_df,
                    )
                    sig_line = self._format_signal_line(sig)
                    lines.append(sig_line)
                except Exception as e:
                    lines.append(f"❌ {symbol}: {e}")

            return "\n".join(lines)
        except Exception as e:
            return f"❌ 信號生成失敗: {e}"

    def _format_signal_line(self, sig) -> str:
        """格式化單個幣種的信號 + 即時持倉/SL/TP（支援 SignalResult 和 dict）"""
        if hasattr(sig, "signal"):  # SignalResult
            signal_pct = sig.signal
            price = sig.price
            symbol = sig.symbol
            ind = sig.indicators
        else:  # dict (from JSON cache)
            signal_pct = sig.get("signal", 0)
            price = sig.get("price", 0)
            symbol = sig.get("symbol", "?")
            ind = sig.get("indicators", {})

        if signal_pct > 0.5:
            emoji, label = "🟢", f"LONG {signal_pct:.0%}"
        elif signal_pct < -0.5:
            emoji, label = "🔴", f"SHORT {abs(signal_pct):.0%}"
        else:
            emoji, label = "⚪", f"FLAT {signal_pct:.0%}"

        sig_line = (
            f"{emoji} <b>{symbol}</b>: {label} @ ${price:,.2f}\n"
            f"   RSI={ind.get('rsi', '?')} | ADX={ind.get('adx', '?')}"
        )

        # 附加即時持倉 + SL/TP（從 Binance 查詢，非快照）
        if self.broker and hasattr(self.broker, "get_position"):
            try:
                pos_obj = self.broker.get_position(symbol)
                if pos_obj and abs(pos_obj.qty) > 1e-10:
                    side = "LONG" if pos_obj.qty > 0 else "SHORT"
                    is_long = pos_obj.qty > 0
                    entry = pos_obj.entry_price
                    qty = abs(pos_obj.qty)
                    pnl = pos_obj.unrealized_pnl
                    pnl_emoji = "📈" if pnl >= 0 else "📉"
                    sig_line += (
                        f"\n   📦 {side} {qty:.6g} @ ${entry:,.2f}"
                        f"  {pnl_emoji} ${pnl:+,.2f}"
                    )
                    # 查詢 SL/TP
                    if hasattr(self.broker, "get_all_conditional_orders"):
                        orders = self.broker.get_all_conditional_orders(symbol)
                        sl_price, tp_price = None, None
                        pos_side_str = "LONG" if is_long else "SHORT"
                        for o in orders:
                            # v2.7.1: 只顯示與當前持倉同方向的 SL/TP
                            o_ps = o.get("positionSide", "")
                            if o_ps and o_ps != pos_side_str and o_ps != "BOTH":
                                continue
                            trigger = float(
                                o.get("stopPrice", 0)
                                or o.get("triggerPrice", 0)
                                or 0
                            )
                            if trigger <= 0:
                                continue
                            otype = o.get("type", "")
                            if otype in {"STOP_MARKET", "STOP"}:
                                sl_price = trigger
                            elif otype in {"TAKE_PROFIT_MARKET", "TAKE_PROFIT"}:
                                tp_price = trigger
                            elif entry > 0:
                                if is_long:
                                    sl_price = trigger if trigger < entry else sl_price
                                    tp_price = trigger if trigger >= entry else tp_price
                                else:
                                    sl_price = trigger if trigger > entry else sl_price
                                    tp_price = trigger if trigger <= entry else tp_price
                        if sl_price:
                            sl_pnl = self._calc_pnl(entry, sl_price, qty, is_long)
                            pnl_str = f" ({sl_pnl:+.2f})" if sl_pnl is not None else ""
                            sig_line += f"\n   🛡️ SL: ${sl_price:,.2f}{pnl_str}"
                        if tp_price:
                            tp_pnl = self._calc_pnl(entry, tp_price, qty, is_long)
                            pnl_str = f" ({tp_pnl:+.2f})" if tp_pnl is not None else ""
                            sig_line += f"\n   🎯 TP: ${tp_price:,.2f}{pnl_str}"
                        if not sl_price and not tp_price:
                            sig_line += "\n   ⚠️ 無 SL/TP 掛單"
            except Exception:
                pass  # 查詢失敗不影響信號顯示

        return sig_line

    # ── /risk ──

    def _cmd_risk(self, args: list[str], chat_id: str) -> str:
        """
        風險總覽：保證金使用、曝險、Drawdown、熔斷狀態
        """
        if not self.broker:
            return "⚠️ Broker 未連接"

        try:
            lines = ["🛡️ <b>風險總覽</b>\n"]

            # ── 帳戶資訊 ──
            if hasattr(self.broker, "get_account_info"):
                info = self.broker.get_account_info()
                if info:
                    equity = float(info.get("totalWalletBalance", 0)) + float(
                        info.get("totalUnrealizedProfit", 0)
                    )
                    available = float(info.get("availableBalance", 0))
                    margin_balance = float(info.get("totalMarginBalance", 0))
                    init_margin = float(info.get("totalInitialMargin", 0))
                    maint_margin = float(info.get("totalMaintMargin", 0))

                    # 保證金使用率
                    margin_ratio = (
                        (init_margin / margin_balance * 100) if margin_balance > 0 else 0
                    )
                    maint_ratio = (
                        (maint_margin / margin_balance * 100) if margin_balance > 0 else 0
                    )

                    lines.extend([
                        f"💰 權益: <b>${equity:,.2f}</b>",
                        f"💵 可用: ${available:,.2f}",
                        f"📊 保證金使用: {margin_ratio:.1f}%",
                        f"🔒 維持保證金: {maint_ratio:.1f}%",
                    ])

                    if margin_ratio >= 80:
                        lines.append("⚠️ <b>保證金使用率偏高！</b>")

            # ── 逐幣曝險 ──
            positions = []
            if hasattr(self.broker, "get_positions"):
                positions = self.broker.get_positions()
            if positions:
                lines.append(f"\n📋 <b>曝險明細 ({len(positions)} 倉)</b>")
                total_notional = 0.0
                for pos in positions:
                    sym = pos.symbol if hasattr(pos, "symbol") else pos.get("symbol", "?")
                    qty = pos.qty if hasattr(pos, "qty") else pos.get("qty", 0)
                    mark = getattr(pos, "mark_price", 0) or 0
                    liq = getattr(pos, "liquidation_price", 0) or 0
                    lev = getattr(pos, "leverage", 0) or 0
                    entry = pos.entry_price if hasattr(pos, "entry_price") else pos.get("entry_price", 0)
                    pnl = pos.unrealized_pnl if hasattr(pos, "unrealized_pnl") else pos.get("unrealized_pnl", 0)
                    notional = abs(qty * mark) if mark > 0 else abs(qty * entry)
                    total_notional += notional
                    side = "LONG" if qty > 0 else "SHORT"
                    pnl_emoji = "📈" if pnl >= 0 else "📉"

                    pos_line = (
                        f"  {pnl_emoji} <b>{sym}</b> [{side}] {lev}x\n"
                        f"     名義: ${notional:,.0f}  PnL: ${pnl:+,.2f}"
                    )
                    if liq > 0:
                        if mark > 0:
                            dist = abs(mark - liq) / mark * 100
                            pos_line += f"\n     強平: ${liq:,.2f} (距 {dist:.1f}%)"
                        else:
                            pos_line += f"\n     強平: ${liq:,.2f}"
                    lines.append(pos_line)

                lines.append(f"\n💎 總名義曝險: <b>${total_notional:,.0f}</b>")
            else:
                lines.append("\n📭 無持倉")

            # ── Drawdown / 熔斷 ──
            runner = self.live_runner
            if runner:
                cb_triggered = getattr(runner, "_circuit_breaker_triggered", False)
                init_eq = getattr(runner, "_initial_equity", None)

                if cb_triggered:
                    lines.append("\n🚨 <b>熔斷已觸發！已停止交易</b>")
                elif init_eq and init_eq > 0:
                    equity_now = 0.0
                    if hasattr(self.broker, "get_equity"):
                        equity_now = self.broker.get_equity()
                    if equity_now > 0:
                        dd = (1 - equity_now / init_eq) * 100
                        threshold = (runner.max_drawdown_pct or 0) * 100
                        bar_len = 10
                        filled = min(int(dd / max(threshold, 1) * bar_len), bar_len)
                        bar = "█" * filled + "░" * (bar_len - filled)
                        lines.append(
                            f"\n📉 Drawdown: <b>{dd:+.2f}%</b> / {threshold:.0f}%"
                            f"\n   [{bar}]"
                        )
                        lines.append(
                            f"   基準: ${init_eq:,.2f} → 現: ${equity_now:,.2f}"
                        )
                else:
                    lines.append("\n✅ 熔斷: 未觸發")

            return "\n".join(lines)
        except Exception as e:
            return f"❌ 風險查詢失敗: {e}"

    # ── /health ──

    def _cmd_health(self, args: list[str], chat_id: str) -> str:
        """
        系統健康：記憶體、磁碟、信號新鮮度、WebSocket 狀態、運行時間
        """
        import shutil

        lines = ["🏥 <b>系統健康</b>\n"]

        # ── 記憶體 ──
        try:
            try:
                import psutil
                mem = psutil.virtual_memory()
                mem_pct = mem.percent
                mem_avail = mem.available / (1024 ** 3)
                lines.append(f"🧠 記憶體: {mem_pct:.1f}% 使用 ({mem_avail:.2f} GB 可用)")
            except ImportError:
                # Linux fallback
                with open("/proc/meminfo") as f:
                    info = {}
                    for line in f:
                        parts = line.split()
                        if len(parts) >= 2:
                            info[parts[0].rstrip(":")] = int(parts[1])
                    total = info.get("MemTotal", 0)
                    avail = info.get("MemAvailable", info.get("MemFree", 0))
                    if total > 0:
                        pct = (1 - avail / total) * 100
                        lines.append(f"🧠 記憶體: {pct:.1f}% 使用 ({avail / 1048576:.2f} GB 可用)")
        except Exception:
            lines.append("🧠 記憶體: 無法檢測")

        # ── 磁碟 ──
        try:
            usage = shutil.disk_usage("/")
            disk_pct = usage.used / usage.total * 100
            free_gb = usage.free / (1024 ** 3)
            lines.append(f"💾 磁碟: {disk_pct:.1f}% 使用 ({free_gb:.1f} GB 可用)")
        except Exception:
            lines.append("💾 磁碟: 無法檢測")

        # ── WebSocket Runner 狀態 ──
        runner = self.live_runner
        if runner:
            # 運行時間
            start_time = getattr(runner, "start_time", None)
            if start_time:
                uptime_sec = time.time() - start_time
                hours = int(uptime_sec // 3600)
                mins = int((uptime_sec % 3600) // 60)
                lines.append(f"⏱ 運行時間: {hours}h {mins}m")

            # K 線 tick 數
            ticks = getattr(runner, "_tick_count", 0)
            trades = getattr(runner, "trade_count", 0)
            lines.append(f"📊 K 線處理: {ticks} 次 | 交易: {trades} 筆")

            # WebSocket 心跳
            last_ws = getattr(runner, "_last_ws_message_time", 0)
            if last_ws > 0:
                ws_age = time.time() - last_ws
                if ws_age < 60:
                    ws_status = f"✅ {ws_age:.0f}s 前"
                elif ws_age < 300:
                    ws_status = f"⚠️ {ws_age / 60:.0f}m 前"
                else:
                    ws_status = f"🚨 {ws_age / 60:.0f}m 前（可能斷線）"
                lines.append(f"📡 WebSocket: {ws_status}")

            # 信號新鮮度
            sig_path = getattr(runner, "_signal_state_path", None)
            if sig_path and sig_path.exists():
                sig_age = time.time() - sig_path.stat().st_mtime
                if sig_age < 3600:
                    lines.append(f"📝 信號: {sig_age / 60:.0f}m 前更新")
                else:
                    lines.append(f"📝 信號: ⚠️ {sig_age / 3600:.1f}h 前更新")

            # 熔斷狀態
            cb = getattr(runner, "_circuit_breaker_triggered", False)
            lines.append(f"🔒 熔斷: {'🚨 已觸發' if cb else '✅ 正常'}")

        else:
            lines.append("⚠️ Runner 未連接")

        return "\n".join(lines)

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
            # pnl 可能是 None（開倉時不計算 pnl），需要安全處理
            wins = sum(1 for t in trades if (t.get("pnl") or 0) > 0)
            losses = sum(1 for t in trades if (t.get("pnl") or 0) < 0)
            flat = total - wins - losses  # pnl=0 或 pnl=None 的交易
            trades_with_pnl = wins + losses
            win_rate = (wins / trades_with_pnl * 100) if trades_with_pnl > 0 else 0
            total_pnl = sum(t.get("pnl") or 0 for t in trades)
            total_fee = sum(t.get("fee") or 0 for t in trades)

            # 額外統計：使用 state 自帶的累積值（更準確）
            cum_pnl = state.cumulative_pnl if hasattr(state, "cumulative_pnl") else total_pnl
            max_dd = getattr(state, "max_drawdown_pct", 0)

            lines = [
                f"📊 <b>交易統計</b>\n",
                f"📝 總交易: {total} 筆",
                f"✅ 獲勝: {wins} 筆",
                f"❌ 虧損: {losses} 筆",
            ]
            if flat > 0:
                lines.append(f"⚪ 持平/開倉: {flat} 筆")
            lines.extend([
                f"🎯 勝率: {win_rate:.1f}%（{trades_with_pnl} 筆有 PnL）",
                f"💰 累積 PnL: <b>${cum_pnl:+,.2f}</b>",
                f"💸 總手續費: ${total_fee:,.2f}",
            ])
            if max_dd > 0:
                lines.append(f"📉 最大回撤: {max_dd:.2f}%")

            return "\n".join(lines)
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
