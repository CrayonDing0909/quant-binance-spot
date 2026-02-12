"""
Telegram Bot 命令處理模組

支援雙向互動：
- 接收命令（/status, /balance, /trades 等）
- 發送通知（交易、告警）

使用方法：
    # 方式 1：獨立運行
    bot = TelegramCommandBot()
    bot.run_polling()  # 阻塞運行
    
    # 方式 2：與 LiveRunner 整合（背景執行）
    bot = TelegramCommandBot(live_runner=runner, broker=broker)
    bot.start_background()
    # ... 主程式邏輯 ...
    bot.stop()

設置步驟：
    1. 在 Telegram 搜索 @BotFather，創建 Bot，獲取 Token
    2. 在 .env 中設置 TELEGRAM_BOT_TOKEN 和 TELEGRAM_CHAT_ID
    3. 可選：設置 TELEGRAM_ADMIN_IDS 限制哪些用戶可以執行命令
"""
from __future__ import annotations

import os
import asyncio
import threading
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional, Any

from ..utils.log import get_logger

logger = get_logger("telegram_bot")

# 延遲導入，避免沒安裝 python-telegram-bot 時報錯
try:
    from telegram import Update, Bot
    from telegram.ext import (
        Application,
        CommandHandler,
        ContextTypes,
        MessageHandler,
        filters,
    )
    TELEGRAM_BOT_AVAILABLE = True
except ImportError:
    TELEGRAM_BOT_AVAILABLE = False
    logger.warning(
        "⚠️  python-telegram-bot 未安裝，Telegram 命令功能不可用\n"
        "   安裝: pip install python-telegram-bot"
    )

if TYPE_CHECKING:
    from ..live.runner import LiveRunner
    from ..live.paper_broker import PaperBroker


class TelegramCommandBot:
    """
    Telegram 命令處理 Bot
    
    支援的命令：
        /status  - 顯示當前持倉和權益
        /balance - 顯示帳戶餘額
        /trades  - 顯示最近交易紀錄
        /signals - 顯示最新信號
        /stop    - 停止交易 Bot（需確認）
        /help    - 顯示幫助
    """
    
    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
        admin_ids: list[int] | None = None,
        live_runner: Optional["LiveRunner"] = None,
        broker: Any = None,
        state_manager: Any = None,
    ):
        """
        初始化 Telegram 命令 Bot
        
        Args:
            bot_token: Bot Token（None = 從環境變數讀取）
            chat_id: 預設 Chat ID（限制回覆對象）
            admin_ids: 管理員 user_id 列表（None = 不限制）
            live_runner: LiveRunner 實例（用於獲取狀態）
            broker: Broker 實例（PaperBroker 或 Real Broker）
            state_manager: TradingStateManager 實例
        """
        if not TELEGRAM_BOT_AVAILABLE:
            raise ImportError(
                "python-telegram-bot 未安裝，請執行: pip install python-telegram-bot"
            )
        
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        
        # 管理員 ID 列表（可從環境變數讀取，逗號分隔）
        if admin_ids:
            self.admin_ids = admin_ids
        else:
            admin_str = os.getenv("TELEGRAM_ADMIN_IDS", "")
            self.admin_ids = [int(x.strip()) for x in admin_str.split(",") if x.strip()]
        
        self.live_runner = live_runner
        self.broker = broker
        self.state_manager = state_manager
        
        # 內部狀態
        self._app: Optional[Application] = None
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._running = False
        
        # 最新信號快取（由 LiveRunner 更新）
        self._last_signals: list[dict] = []
        
        if not self.bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN 未設置")
        
        logger.info(f"✅ Telegram Bot 初始化完成")
        if self.admin_ids:
            logger.info(f"   管理員 ID: {self.admin_ids}")
    
    def _is_authorized(self, user_id: int) -> bool:
        """檢查用戶是否有權限執行命令"""
        if not self.admin_ids:
            return True  # 沒設置 admin_ids = 不限制
        return user_id in self.admin_ids
    
    async def _unauthorized_response(self, update: Update) -> None:
        """未授權回覆"""
        await update.message.reply_text(
            "⛔ 你沒有權限執行此命令\n"
            f"你的 User ID: {update.effective_user.id}"
        )
    
    # ══════════════════════════════════════════════════════════════════════════
    # 命令處理器
    # ══════════════════════════════════════════════════════════════════════════
    
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """/start - 歡迎訊息"""
        user = update.effective_user
        await update.message.reply_text(
            f"👋 Hi {user.first_name}!\n\n"
            f"我是交易 Bot 助手，你可以用以下命令查詢狀態：\n\n"
            f"/status - 查看當前持倉\n"
            f"/balance - 查看帳戶餘額\n"
            f"/trades - 查看最近交易\n"
            f"/signals - 查看最新信號\n"
            f"/help - 顯示幫助\n\n"
            f"你的 User ID: <code>{user.id}</code>",
            parse_mode="HTML",
        )
    
    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """/help - 顯示幫助"""
        await update.message.reply_text(
            "📖 <b>命令列表</b>\n\n"
            "<b>查詢類：</b>\n"
            "/status - 顯示當前持倉和權益\n"
            "/balance - 顯示帳戶餘額\n"
            "/trades - 顯示最近 10 筆交易\n"
            "/signals - 顯示最新信號\n"
            "/stats - 顯示交易統計\n\n"
            "<b>控制類：</b>\n"
            "/stop - 停止交易 Bot（需確認）\n\n"
            "<b>其他：</b>\n"
            "/ping - 檢查 Bot 是否在線\n"
            "/id - 顯示你的 User ID",
            parse_mode="HTML",
        )
    
    async def cmd_ping(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """/ping - 檢查 Bot 是否在線"""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        
        if self.live_runner and self.live_runner.is_running:
            status_emoji = "🟢"
            status_text = "Daemon 模式運行中"
        elif self.live_runner:
            status_emoji = "🔴"
            status_text = "Daemon 已停止"
        else:
            # 獨立模式 - Cron 控制
            status_emoji = "🟢"
            status_text = "Cron 模式（每小時執行）"
        
        await update.message.reply_text(
            f"🏓 Pong!\n\n"
            f"⏰ 時間: {now}\n"
            f"{status_emoji} Trading Bot: {status_text}\n"
            f"📡 Telegram Bot: 在線"
        )
    
    async def cmd_id(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """/id - 顯示 User ID"""
        user = update.effective_user
        await update.message.reply_text(
            f"👤 <b>你的資訊</b>\n\n"
            f"User ID: <code>{user.id}</code>\n"
            f"Username: @{user.username or 'N/A'}\n"
            f"Name: {user.first_name} {user.last_name or ''}",
            parse_mode="HTML",
        )
    
    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """/status - 顯示當前持倉和權益"""
        if not self._is_authorized(update.effective_user.id):
            await self._unauthorized_response(update)
            return
        
        try:
            status = self._get_status()
            await update.message.reply_text(status, parse_mode="HTML")
        except Exception as e:
            logger.error(f"獲取狀態失敗: {e}")
            await update.message.reply_text(f"❌ 獲取狀態失敗: {e}")
    
    async def cmd_balance(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """/balance - 顯示帳戶餘額"""
        if not self._is_authorized(update.effective_user.id):
            await self._unauthorized_response(update)
            return
        
        try:
            balance = self._get_balance()
            await update.message.reply_text(balance, parse_mode="HTML")
        except Exception as e:
            logger.error(f"獲取餘額失敗: {e}")
            await update.message.reply_text(f"❌ 獲取餘額失敗: {e}")
    
    async def cmd_trades(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """/trades - 顯示最近交易"""
        if not self._is_authorized(update.effective_user.id):
            await self._unauthorized_response(update)
            return
        
        try:
            trades = self._get_recent_trades()
            await update.message.reply_text(trades, parse_mode="HTML")
        except Exception as e:
            logger.error(f"獲取交易紀錄失敗: {e}")
            await update.message.reply_text(f"❌ 獲取交易紀錄失敗: {e}")
    
    async def cmd_signals(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """/signals - 顯示最新信號"""
        if not self._is_authorized(update.effective_user.id):
            await self._unauthorized_response(update)
            return
        
        try:
            signals = self._get_signals()
            await update.message.reply_text(signals, parse_mode="HTML")
        except Exception as e:
            logger.error(f"獲取信號失敗: {e}")
            await update.message.reply_text(f"❌ 獲取信號失敗: {e}")
    
    async def cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """/stats - 顯示交易統計"""
        if not self._is_authorized(update.effective_user.id):
            await self._unauthorized_response(update)
            return
        
        try:
            stats = self._get_stats()
            await update.message.reply_text(stats, parse_mode="HTML")
        except Exception as e:
            logger.error(f"獲取統計失敗: {e}")
            await update.message.reply_text(f"❌ 獲取統計失敗: {e}")
    
    async def cmd_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """/stop - 停止交易 Bot"""
        if not self._is_authorized(update.effective_user.id):
            await self._unauthorized_response(update)
            return
        
        if not self.live_runner:
            # 獨立模式 - Cron 控制
            await update.message.reply_text(
                "⚠️ <b>Cron 模式</b>\n\n"
                "Trading Bot 由 Cron 控制，無法透過 Telegram 停止。\n\n"
                "如需停止，請 SSH 到伺服器執行：\n"
                "<code>crontab -e</code>\n"
                "然後註解或刪除相關行。",
                parse_mode="HTML",
            )
            return
        
        if not self.live_runner.is_running:
            await update.message.reply_text("⚠️ Trading Bot 目前沒有運行")
            return
        
        # 需要確認
        await update.message.reply_text(
            "⚠️ <b>確認停止交易 Bot?</b>\n\n"
            "發送 /confirm_stop 確認停止\n"
            "發送其他任何訊息取消",
            parse_mode="HTML",
        )
    
    async def cmd_confirm_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """/confirm_stop - 確認停止"""
        if not self._is_authorized(update.effective_user.id):
            await self._unauthorized_response(update)
            return
        
        if self.live_runner and self.live_runner.is_running:
            self.live_runner.stop()
            await update.message.reply_text("⛔ Trading Bot 正在停止...")
            logger.warning(f"Trading Bot 被 Telegram 命令停止 (user: {update.effective_user.id})")
        else:
            await update.message.reply_text("⚠️ Trading Bot 已經停止")
    
    # ══════════════════════════════════════════════════════════════════════════
    # 資料獲取方法（整合 broker / state_manager / 直接查詢 Binance）
    # ══════════════════════════════════════════════════════════════════════════
    
    def _get_futures_broker(self):
        """獲取或創建 Futures Broker（獨立模式用）"""
        if self.broker:
            return self.broker
        
        # 獨立模式：嘗試創建 BinanceFuturesBroker
        try:
            from ..live.binance_futures_broker import BinanceFuturesBroker
            return BinanceFuturesBroker(dry_run=True)  # dry_run 只查詢不下單
        except Exception as e:
            logger.warning(f"無法創建 Futures Broker: {e}")
            return None
    
    def _get_status(self) -> str:
        """獲取當前狀態"""
        lines = ["💼 <b>交易狀態</b>\n"]
        
        # 模式
        if self.live_runner:
            mode = self.live_runner.mode.upper()
            market = self.live_runner.market_type.upper()
            strategy = self.live_runner.strategy_name
            running = "🟢 運行中" if self.live_runner.is_running else "🔴 已停止"
            
            lines.append(f"📊 模式: {mode} ({market})")
            lines.append(f"📈 策略: {strategy}")
            lines.append(f"⚡ 狀態: {running}")
            lines.append(f"🔄 Ticks: {self.live_runner.tick_count}")
            lines.append(f"📝 交易: {self.live_runner.trade_count} 筆")
            lines.append("")
        else:
            # 獨立模式：顯示 Cron 運行狀態
            lines.append("📊 模式: REAL (FUTURES) via Cron")
            lines.append("⚡ 狀態: 🟢 Cron 每小時執行")
            lines.append("")
        
        # 持倉
        positions = self._get_positions()
        if positions:
            lines.append("<b>📦 持倉：</b>")
            for sym, info in positions.items():
                qty = info.get("qty", 0)
                entry = info.get("avg_entry", 0)
                side = info.get("side", "LONG" if qty > 0 else "SHORT")
                unrealized_pnl = info.get("unrealized_pnl", 0)
                
                # 計算市值
                mark_price = info.get("mark_price", entry)
                value = abs(qty) * mark_price
                
                # PnL emoji
                pnl_emoji = "📈" if unrealized_pnl > 0 else "📉"
                
                lines.append(
                    f"  • {sym} [{side}]: {abs(qty):.4f}\n"
                    f"    入場: ${entry:,.2f} | 標記: ${mark_price:,.2f}\n"
                    f"    {pnl_emoji} 未實現 PnL: ${unrealized_pnl:+,.2f}"
                )
        else:
            lines.append("📦 持倉：無")
        
        return "\n".join(lines)
    
    def _get_balance(self) -> str:
        """獲取帳戶餘額"""
        lines = ["💰 <b>帳戶餘額</b>\n"]
        
        from ..live.paper_broker import PaperBroker
        
        # 獲取 broker（可能是傳入的或獨立創建的）
        broker = self.broker or self._get_futures_broker()
        
        if isinstance(broker, PaperBroker):
            # Paper 模式
            account = broker.account
            
            # 計算權益需要當前價格
            prices = {}
            for sym, pos in account.positions.items():
                if pos.is_open:
                    # 嘗試獲取價格
                    try:
                        from ..live.signal_generator import fetch_recent_klines
                        df = fetch_recent_klines(sym, self.live_runner.interval if self.live_runner else "1h", 5)
                        prices[sym] = float(df["close"].iloc[-1])
                    except Exception:
                        prices[sym] = pos.avg_entry  # fallback
            
            equity = broker.get_equity(prices)
            ret = (equity / account.initial_cash - 1) * 100
            ret_emoji = "📈" if ret > 0 else "📉"
            
            lines.append(f"💵 現金: ${account.cash:,.2f}")
            lines.append(f"💎 權益: ${equity:,.2f}")
            lines.append(f"📊 初始: ${account.initial_cash:,.2f}")
            lines.append(f"{ret_emoji} 報酬: {ret:+.2f}%")
            
        elif broker and hasattr(broker, "get_balance"):
            # Real 模式（Futures）
            try:
                # Futures broker 用 get_balance() 不帶參數
                if hasattr(broker, "get_equity"):
                    balance = broker.get_balance()
                    equity = broker.get_equity()
                    
                    lines.append(f"💵 可用餘額: ${balance:,.2f}")
                    lines.append(f"💎 帳戶權益: ${equity:,.2f}")
                    
                    # 顯示未實現盈虧
                    positions = broker.get_positions()
                    total_pnl = sum(p.unrealized_pnl for p in positions if p and abs(p.qty) > 1e-8)
                    if total_pnl != 0:
                        pnl_emoji = "📈" if total_pnl > 0 else "📉"
                        lines.append(f"{pnl_emoji} 未實現 PnL: ${total_pnl:+,.2f}")
                else:
                    # Spot broker
                    usdt = broker.get_balance("USDT")
                    lines.append(f"💵 USDT: ${usdt:,.2f}")
                    
                    # 計算總權益
                    total = usdt
                    for sym in (self.live_runner.symbols if self.live_runner else []):
                        qty = broker.get_position(sym)
                        if qty > 0:
                            price = broker.get_price(sym)
                            total += qty * price
                    
                    lines.append(f"💎 總權益: ${total:,.2f}")
            except Exception as e:
                lines.append(f"❌ 查詢失敗: {e}")
        else:
            lines.append("⚠️ 無法獲取餘額資訊（請確認 API Key 已設置）")
        
        return "\n".join(lines)
    
    def _get_recent_trades(self, limit: int = 10) -> str:
        """獲取最近交易"""
        lines = ["📜 <b>最近交易</b>\n"]
        
        trades = []
        
        # 從 state_manager 獲取
        if self.state_manager:
            trades = self.state_manager.state.trades[-limit:]
        # 從 PaperBroker 獲取
        elif self.broker:
            from ..live.paper_broker import PaperBroker
            if isinstance(self.broker, PaperBroker):
                trades = [
                    {
                        "timestamp": t.time.isoformat() if hasattr(t, "time") else "N/A",
                        "symbol": t.symbol,
                        "side": t.side,
                        "qty": t.qty,
                        "price": t.price,
                        "pnl": t.pnl,
                    }
                    for t in self.broker.account.trades[-limit:]
                ]
        else:
            # 獨立模式：優先從 Binance API 獲取
            trades = self._load_trades_from_binance(limit)
            
            # 如果 API 失敗，fallback 到 state 文件
            if not trades:
                trades = self._load_trades_from_state_file(limit)
        
        if not trades:
            lines.append("暫無交易紀錄")
            return "\n".join(lines)
        
        for t in trades[:limit]:  # 已經是最新的在前
            symbol = t.get("symbol", "?")
            side = t.get("side", "?")
            position_side = t.get("position_side", "")
            qty = t.get("qty", 0)
            price = t.get("price", 0)
            pnl = t.get("pnl") or t.get("realized_pnl")
            timestamp = t.get("timestamp") or t.get("time")
            
            # 格式化時間
            time_str = ""
            if timestamp:
                try:
                    if isinstance(timestamp, (int, float)):
                        # 毫秒時間戳
                        dt = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)
                    else:
                        dt = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
                    time_str = dt.strftime("%m-%d %H:%M") + " "
                except Exception:
                    pass
            
            # 判斷方向
            if position_side:
                side_label = f"{side}/{position_side}"
            else:
                side_label = side
            
            side_emoji = "🟢" if "BUY" in side.upper() else "🔴"
            pnl_str = ""
            if pnl is not None and pnl != 0:
                pnl_emoji = "📈" if pnl > 0 else "📉"
                pnl_str = f" {pnl_emoji} ${pnl:+.2f}"
            
            lines.append(f"{side_emoji} {time_str}{symbol} {side_label}\n   {qty:.4f} @ ${price:,.2f}{pnl_str}")
        
        return "\n".join(lines)
    
    def _load_trades_from_binance(self, limit: int = 10) -> list[dict]:
        """從 Binance API 獲取交易歷史"""
        broker = self._get_futures_broker()
        if not broker or not hasattr(broker, "get_trade_history"):
            return []
        
        try:
            # 獲取所有交易對的歷史
            trades = broker.get_trade_history(symbol=None, limit=limit * 2)
            return trades[:limit]
        except Exception as e:
            logger.warning(f"從 Binance 獲取交易歷史失敗: {e}")
            return []
    
    def _load_trades_from_state_file(self, limit: int = 10) -> list[dict]:
        """從 state 文件讀取交易紀錄"""
        import json
        from pathlib import Path
        
        # 獲取專案根目錄（支援絕對路徑）
        project_root = Path(__file__).parent.parent.parent.parent  # src/qtrade/monitor -> project root
        
        # 嘗試多個可能的 state 文件路徑
        possible_paths = [
            project_root / "reports/live/rsi_adx_atr/real_state.json",
            project_root / "reports/live/rsi_adx_atr_enhanced/real_state.json",
            project_root / "reports/live/futures_rsi_adx_atr/real_state.json",
            # 也嘗試相對路徑（以防工作目錄正確）
            Path("reports/live/rsi_adx_atr/real_state.json"),
        ]
        
        for path in possible_paths:
            if path.exists():
                try:
                    with open(path, encoding="utf-8") as f:
                        data = json.load(f)
                    trades = data.get("trades", [])[-limit:]
                    if trades:
                        logger.info(f"從 {path} 讀取到 {len(trades)} 筆交易")
                    return trades
                except Exception as e:
                    logger.warning(f"讀取 {path} 失敗: {e}")
        
        return []
    
    def _get_signals(self) -> str:
        """獲取最新信號"""
        lines = ["📡 <b>最新信號</b>\n"]
        
        # 如果有快取的信號，使用快取
        signals = self._last_signals
        
        # 獨立模式：實時生成信號
        if not signals and not self.live_runner:
            signals = self._generate_realtime_signals()
        
        if not signals:
            lines.append("暫無信號（等待下一個 Tick）")
            return "\n".join(lines)
        
        for sig in signals:
            symbol = sig.get("symbol", "?")
            signal = sig.get("signal", 0)
            price = sig.get("price", 0)
            ind = sig.get("indicators", {})
            
            # 信號方向
            if signal > 0.5:
                emoji = "🟢"
                label = f"LONG {signal:.0%}"
            elif signal < -0.5:
                emoji = "🔴"
                label = f"SHORT {abs(signal):.0%}"
            else:
                emoji = "⚪"
                label = f"FLAT {signal:.0%}"
            
            lines.append(f"{emoji} <b>{symbol}</b>: {label} @ ${price:,.2f}")
            
            # 指標
            ind_parts = []
            if "rsi" in ind:
                ind_parts.append(f"RSI={ind['rsi']}")
            if "adx" in ind:
                ind_parts.append(f"ADX={ind['adx']}")
            if ind_parts:
                lines.append(f"   {' | '.join(ind_parts)}")
        
        return "\n".join(lines)
    
    def _generate_realtime_signals(self) -> list[dict]:
        """實時生成信號（獨立模式用）"""
        signals = []
        
        # 預設交易對（Futures 雙向模式）
        symbols = ["BTCUSDT", "ETHUSDT"]
        strategy_name = "rsi_adx_atr"
        interval = "1h"
        market_type = "futures"
        direction = "both"
        
        try:
            from ..live.signal_generator import generate_signal
            
            for symbol in symbols:
                try:
                    sig = generate_signal(
                        symbol=symbol,
                        strategy_name=strategy_name,
                        params={},  # 使用預設參數
                        interval=interval,
                        market_type=market_type,
                        direction=direction,
                    )
                    signals.append(sig)
                except Exception as e:
                    logger.warning(f"生成 {symbol} 信號失敗: {e}")
        except ImportError as e:
            logger.warning(f"無法導入 signal_generator: {e}")
        
        return signals
    
    def _get_stats(self) -> str:
        """獲取交易統計"""
        lines = ["📊 <b>交易統計</b>\n"]
        
        stats = None
        
        if self.state_manager:
            stats = self.state_manager.get_trade_stats()
            state = self.state_manager.state
            
            lines.append(f"📝 總交易: {state.total_trades} 筆")
            lines.append(f"✅ 獲勝: {state.winning_trades} 筆")
            lines.append(f"❌ 虧損: {state.losing_trades} 筆")
            lines.append(f"🎯 勝率: {stats['win_rate']:.1%}")
            lines.append(f"💰 累積 PnL: ${state.cumulative_pnl:,.2f}")
            lines.append(f"📉 最大回撤: {state.max_drawdown_pct:.2f}%")
            
        elif self.live_runner:
            stats = self.live_runner._get_trade_stats()
            
            lines.append(f"📝 總交易: {stats.get('total_trades', 0)} 筆")
            lines.append(f"🎯 勝率: {stats.get('win_rate', 0):.1%}")
            lines.append(f"📈 平均獲利: ${stats.get('avg_win', 0):,.2f}")
            lines.append(f"📉 平均虧損: ${stats.get('avg_loss', 0):,.2f}")
        else:
            # 獨立模式：優先從 Binance API 計算
            stats = self._calculate_stats_from_binance()
            
            # Fallback 到 state 文件
            if not stats:
                stats = self._load_stats_from_state_file()
            
            if stats:
                lines.append(f"📝 總交易: {stats.get('total_trades', 0)} 筆")
                if stats.get('winning_trades') is not None:
                    lines.append(f"✅ 獲勝: {stats.get('winning_trades', 0)} 筆")
                    lines.append(f"❌ 虧損: {stats.get('losing_trades', 0)} 筆")
                win_rate = stats.get('win_rate', 0)
                if win_rate > 0:
                    lines.append(f"🎯 勝率: {win_rate:.1%}")
                lines.append(f"💰 累積 PnL: ${stats.get('cumulative_pnl', 0):,.2f}")
                if stats.get('commission'):
                    lines.append(f"💸 總手續費: ${stats.get('commission', 0):,.2f}")
            else:
                lines.append("⚠️ 暫無交易統計（尚未有交易紀錄）")
        
        return "\n".join(lines)
    
    def _calculate_stats_from_binance(self) -> dict | None:
        """從 Binance API 計算交易統計"""
        broker = self._get_futures_broker()
        if not broker:
            return None
        
        try:
            # 獲取收益歷史（已實現盈虧）
            if hasattr(broker, "get_income_history"):
                income = broker.get_income_history(income_type="REALIZED_PNL", limit=500)
                commission = broker.get_income_history(income_type="COMMISSION", limit=500)
                
                # 計算統計
                total_pnl = sum(i["income"] for i in income)
                total_commission = sum(abs(c["income"]) for c in commission)
                
                # 計算勝率
                wins = [i for i in income if i["income"] > 0]
                losses = [i for i in income if i["income"] < 0]
                total_trades = len(wins) + len(losses)
                win_rate = len(wins) / total_trades if total_trades > 0 else 0
                
                return {
                    "total_trades": total_trades,
                    "winning_trades": len(wins),
                    "losing_trades": len(losses),
                    "win_rate": win_rate,
                    "cumulative_pnl": total_pnl,
                    "commission": total_commission,
                }
            
            # Fallback: 從交易歷史計算
            trades = broker.get_trade_history(limit=500)
            if not trades:
                return None
            
            total_pnl = sum(t.get("realized_pnl", 0) for t in trades)
            total_commission = sum(t.get("commission", 0) for t in trades)
            
            return {
                "total_trades": len(trades),
                "cumulative_pnl": total_pnl,
                "commission": total_commission,
            }
            
        except Exception as e:
            logger.warning(f"從 Binance 計算統計失敗: {e}")
            return None
    
    def _load_stats_from_state_file(self) -> dict | None:
        """從 state 文件讀取統計資訊"""
        import json
        from pathlib import Path
        
        # 獲取專案根目錄（支援絕對路徑）
        project_root = Path(__file__).parent.parent.parent.parent  # src/qtrade/monitor -> project root
        
        # 嘗試多個可能的 state 文件路徑
        possible_paths = [
            project_root / "reports/live/rsi_adx_atr/real_state.json",
            project_root / "reports/live/rsi_adx_atr_enhanced/real_state.json",
            project_root / "reports/live/futures_rsi_adx_atr/real_state.json",
            # 也嘗試相對路徑
            Path("reports/live/rsi_adx_atr/real_state.json"),
        ]
        
        for path in possible_paths:
            if path.exists():
                try:
                    with open(path, encoding="utf-8") as f:
                        data = json.load(f)
                    
                    # 計算勝率
                    total = data.get("total_trades", 0)
                    winning = data.get("winning_trades", 0)
                    win_rate = winning / total if total > 0 else 0
                    
                    logger.info(f"從 {path} 讀取統計資訊")
                    return {
                        "total_trades": total,
                        "winning_trades": winning,
                        "losing_trades": data.get("losing_trades", 0),
                        "win_rate": win_rate,
                        "cumulative_pnl": data.get("cumulative_pnl", 0),
                        "max_drawdown_pct": data.get("max_drawdown_pct", 0),
                    }
                except Exception as e:
                    logger.warning(f"讀取 {path} 失敗: {e}")
        
        return None
    
    def _get_positions(self) -> dict:
        """獲取當前持倉"""
        positions = {}
        
        # 從 state_manager 獲取
        if self.state_manager:
            for sym, pos in self.state_manager.state.positions.items():
                if pos.get("qty", 0) > 1e-10:
                    positions[sym] = pos
            return positions
        
        # 獲取 broker（可能是傳入的或獨立創建的）
        broker = self.broker or self._get_futures_broker()
        
        if not broker:
            return positions
        
        from ..live.paper_broker import PaperBroker
        
        if isinstance(broker, PaperBroker):
            for sym, pos in broker.account.positions.items():
                if pos.is_open:
                    positions[sym] = {"qty": pos.qty, "avg_entry": pos.avg_entry}
        elif hasattr(broker, "get_positions"):
            # Futures broker - 使用 get_positions() 獲取所有持倉
            try:
                all_positions = broker.get_positions()
                for pos in all_positions:
                    if pos and abs(pos.qty) > 1e-8:
                        positions[pos.symbol] = {
                            "qty": pos.qty,
                            "avg_entry": pos.entry_price,
                            "mark_price": pos.mark_price,
                            "unrealized_pnl": pos.unrealized_pnl,
                            "side": "LONG" if pos.qty > 0 else "SHORT",
                        }
            except Exception as e:
                logger.warning(f"獲取持倉失敗: {e}")
        elif hasattr(broker, "get_position"):
            # Spot broker
            symbols = self.live_runner.symbols if self.live_runner else []
            for sym in symbols:
                try:
                    qty = broker.get_position(sym)
                    if qty > 1e-10:
                        price = broker.get_price(sym)
                        positions[sym] = {"qty": qty, "avg_entry": price}
                except Exception:
                    pass
        
        return positions
    
    def update_signals(self, signals: list[dict]) -> None:
        """更新最新信號（由 LiveRunner 呼叫）"""
        self._last_signals = signals
    
    # ══════════════════════════════════════════════════════════════════════════
    # 運行方法
    # ══════════════════════════════════════════════════════════════════════════
    
    def _build_app(self) -> Application:
        """建立 Telegram Application"""
        app = Application.builder().token(self.bot_token).build()
        
        # 註冊命令處理器
        app.add_handler(CommandHandler("start", self.cmd_start))
        app.add_handler(CommandHandler("help", self.cmd_help))
        app.add_handler(CommandHandler("ping", self.cmd_ping))
        app.add_handler(CommandHandler("id", self.cmd_id))
        app.add_handler(CommandHandler("status", self.cmd_status))
        app.add_handler(CommandHandler("balance", self.cmd_balance))
        app.add_handler(CommandHandler("trades", self.cmd_trades))
        app.add_handler(CommandHandler("signals", self.cmd_signals))
        app.add_handler(CommandHandler("stats", self.cmd_stats))
        app.add_handler(CommandHandler("stop", self.cmd_stop))
        app.add_handler(CommandHandler("confirm_stop", self.cmd_confirm_stop))
        
        return app
    
    def run_polling(self) -> None:
        """
        阻塞運行 Bot（Long Polling）
        
        適用於獨立運行或測試
        """
        self._app = self._build_app()
        self._running = True
        
        logger.info("🤖 Telegram Bot 開始 Polling...")
        
        try:
            self._app.run_polling(allowed_updates=Update.ALL_TYPES)
        except Exception as e:
            logger.error(f"Telegram Bot 運行錯誤: {e}")
        finally:
            self._running = False
    
    def start_background(self) -> None:
        """
        背景運行 Bot
        
        適用於與 LiveRunner 整合
        """
        if self._running:
            logger.warning("Telegram Bot 已在運行中")
            return
        
        def _run_in_thread():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            
            self._app = self._build_app()
            self._running = True
            
            logger.info("🤖 Telegram Bot 開始背景 Polling...")
            
            try:
                self._loop.run_until_complete(self._app.initialize())
                self._loop.run_until_complete(self._app.start())
                self._loop.run_until_complete(
                    self._app.updater.start_polling(allowed_updates=Update.ALL_TYPES)
                )
                
                # 保持運行直到 stop 被呼叫
                while self._running:
                    self._loop.run_until_complete(asyncio.sleep(1))
                    
            except Exception as e:
                logger.error(f"Telegram Bot 背景運行錯誤: {e}")
            finally:
                try:
                    self._loop.run_until_complete(self._app.updater.stop())
                    self._loop.run_until_complete(self._app.stop())
                    self._loop.run_until_complete(self._app.shutdown())
                except Exception:
                    pass
                self._running = False
                self._loop.close()
        
        self._thread = threading.Thread(target=_run_in_thread, daemon=True)
        self._thread.start()
        
        # 等待啟動
        import time
        time.sleep(1)
        
        if self._running:
            logger.info("✅ Telegram Bot 背景啟動成功")
        else:
            logger.error("❌ Telegram Bot 背景啟動失敗")
    
    def stop(self) -> None:
        """停止 Bot"""
        logger.info("⛔ 正在停止 Telegram Bot...")
        self._running = False
        
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        
        logger.info("✅ Telegram Bot 已停止")
    
    @property
    def is_running(self) -> bool:
        """Bot 是否正在運行"""
        return self._running
