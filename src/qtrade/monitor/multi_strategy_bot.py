"""
統一多策略 Telegram Bot

獨立進程運行，直連 Binance API 查詢帳戶狀態，
讀取各策略 Runner 寫出的信號快照 (last_signals.json)。

解決問題：
    - 多個 tmux session 共用同一 Bot Token 導致訊息互搶
    - 無法跨策略查看全局狀態

使用方式：
    PYTHONPATH=src python scripts/run_telegram_bot.py \
        -c config/prod_candidate_meta_blend.yaml \
        -c config/prod_live_oi_liq_bounce.yaml \
        --real
"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import requests

from ..utils.log import get_logger
from .telegram_bot import TelegramBot

if TYPE_CHECKING:
    from ..config import AppConfig

logger = get_logger("multi_strategy_bot")


class MultiStrategyBot(TelegramBot):
    """
    統一多策略 Telegram Bot

    - 帳戶級查詢（/dashboard, /status, /positions, /pnl, /balance, /risk）：
      直連 Binance API，能看到所有策略的持倉
    - 策略級查詢（/signals, /health）：
      讀取各策略 Runner 寫出的 last_signals.json
    """

    def __init__(
        self,
        configs: list[tuple[str, AppConfig]],
        broker: Any = None,
        alert_config: dict | None = None,
        **kwargs,
    ):
        """
        Args:
            configs: [(strategy_name, AppConfig), ...] 多策略配置
            broker: BinanceFuturesBroker(dry_run=True)
            alert_config: 告警配置 dict
        """
        super().__init__(broker=broker, **kwargs)
        self._configs = configs
        self._alert_cfg = alert_config or {}

        # 背景任務
        self._daily_last_date: str | None = None
        self._peak_equity: float = 0.0

        # 覆蓋預設命令
        self._commands = {}
        self._register_multi_commands()

    # ══════════════════════════════════════════════════════════════
    # 命令註冊
    # ══════════════════════════════════════════════════════════════

    def _register_multi_commands(self):
        self.register_command("help", self._cmd_help_multi, "📖 幫助選單")
        self.register_command("dashboard", self._cmd_dashboard, "📊 全局總覽")
        self.register_command("status", self._cmd_status_multi, "💼 帳戶狀態")
        self.register_command("signals", self._cmd_signals_multi, "📡 交易信號")
        self.register_command("pnl", self._cmd_pnl_multi, "💰 盈虧查詢")
        self.register_command("positions", self._cmd_positions_multi, "📋 持倉列表")
        self.register_command("health", self._cmd_health_multi, "🏥 系統健康")
        self.register_command("risk", self._cmd_risk_multi, "🛡️ 風險總覽")
        self.register_command("balance", self._cmd_balance, "💵 餘額")
        self.register_command("trades", self._cmd_trades, "📜 交易記錄")
        self.register_command("ping", self._cmd_ping, "🏓 測試")

    def start(self):
        """啟動 Bot + 背景任務"""
        super().start()
        if self.enabled:
            self._start_background_tasks()

    def _start_background_tasks(self):
        """啟動每日摘要 & 告警檢查"""
        t1 = threading.Thread(target=self._daily_summary_loop, daemon=True)
        t1.start()
        if self._alert_cfg:
            t2 = threading.Thread(target=self._alert_loop, daemon=True)
            t2.start()
            logger.info("🔔 告警檢查已啟動")

    # ══════════════════════════════════════════════════════════════
    # 輔助方法
    # ══════════════════════════════════════════════════════════════

    def _strategy_names(self) -> list[str]:
        return [n for n, _ in self._configs]

    def _get_cfg(self, name: str) -> AppConfig | None:
        for n, c in self._configs:
            if n == name:
                return c
        return None

    def _symbol_to_strategy(self, symbol: str) -> str:
        """根據 symbol 找到所屬策略名稱"""
        for name, cfg in self._configs:
            if symbol in cfg.market.symbols:
                return name
        return "unknown"

    def _read_signals(self, name: str, cfg: AppConfig) -> tuple[list | None, str]:
        """讀取某策略的 last_signals.json"""
        sig_path = cfg.get_report_dir("live") / "last_signals.json"
        if not sig_path.exists():
            return None, ""
        try:
            with open(sig_path) as f:
                payload = json.load(f)
            gen_at = payload.get("generated_at", "")
            signals = payload.get("signals", [])
            return signals, gen_at
        except Exception:
            return None, ""

    def _signal_age_str(self, gen_at: str) -> str:
        """計算信號新鮮度文字"""
        if not gen_at:
            return ""
        try:
            gen_time = datetime.fromisoformat(gen_at)
            age_sec = (datetime.now(timezone.utc) - gen_time).total_seconds()
            if age_sec < 60:
                return f"⏱ {int(age_sec)}s 前"
            elif age_sec < 3600:
                return f"⏱ {int(age_sec // 60)}m 前"
            else:
                return f"⏱ {age_sec / 3600:.1f}h 前"
        except Exception:
            return ""

    def _get_account_info(self) -> dict | None:
        if self.broker and hasattr(self.broker, "get_account_info"):
            return self.broker.get_account_info()
        return None

    def _get_positions(self) -> list:
        if self.broker and hasattr(self.broker, "get_positions"):
            return self.broker.get_positions()
        return []

    def _get_equity(self) -> float:
        if self.broker and hasattr(self.broker, "get_equity"):
            return self.broker.get_equity()
        return 0.0

    def _pos_attr(self, pos, attr: str, default=0):
        """安全取得 position 屬性（支援 dataclass / dict）"""
        return getattr(pos, attr, None) or (pos.get(attr, default) if isinstance(pos, dict) else default)

    # ══════════════════════════════════════════════════════════════
    # /help — 按鈕選單
    # ══════════════════════════════════════════════════════════════

    def _cmd_help_multi(self, args: list[str], chat_id: str) -> str:
        buttons = {
            "inline_keyboard": [
                [
                    {"text": "📊 總覽", "callback_data": "/dashboard"},
                    {"text": "💼 狀態", "callback_data": "/status"},
                ],
                [
                    {"text": "📡 信號", "callback_data": "/signals"},
                    {"text": "💰 盈虧", "callback_data": "/pnl"},
                ],
                [
                    {"text": "📋 持倉", "callback_data": "/positions"},
                    {"text": "🛡️ 風險", "callback_data": "/risk"},
                ],
                [
                    {"text": "🏥 健康", "callback_data": "/health"},
                    {"text": "💵 餘額", "callback_data": "/balance"},
                ],
                [
                    {"text": "📜 交易", "callback_data": "/trades"},
                    {"text": "🏓 Ping", "callback_data": "/ping"},
                ],
            ]
        }
        strategies = ", ".join(self._strategy_names())
        text = (
            "📖 <b>指令選單</b>\n\n"
            f"🔗 策略: {strategies}\n\n"
            "<b>📊 帳戶</b>\n"
            "/dashboard — 全局總覽\n"
            "/status — 帳戶狀態\n"
            "/balance — 餘額\n\n"
            "<b>📈 交易</b>\n"
            "/signals [策略名] — 交易信號\n"
            "/positions — 持倉詳情\n"
            "/pnl [7d|30d|all] — 盈虧\n"
            "/trades [n] — 交易記錄\n\n"
            "<b>⚙️ 系統</b>\n"
            "/health — 系統健康\n"
            "/risk — 風險總覽\n"
        )
        self._send_message(chat_id, text, reply_markup=buttons)
        return ""

    # ══════════════════════════════════════════════════════════════
    # /dashboard — 全局總覽
    # ══════════════════════════════════════════════════════════════

    def _cmd_dashboard(self, args: list[str], chat_id: str) -> str:
        if not self.broker:
            return "⚠️ Broker 未連接"

        try:
            info = self._get_account_info()
            if not info:
                return "⚠️ 無法取得帳戶資訊"

            equity = float(info.get("totalWalletBalance", 0)) + float(
                info.get("totalUnrealizedProfit", 0)
            )
            unrealized = float(info.get("totalUnrealizedProfit", 0))
            available = float(info.get("availableBalance", 0))
            pnl_emoji = "📈" if unrealized >= 0 else "📉"

            lines = [
                "📊 <b>Dashboard</b>\n",
                f"💰 總權益: <b>${equity:,.2f}</b>",
                f"💵 可用: ${available:,.2f}",
                f"{pnl_emoji} 未實現: ${unrealized:+,.2f}",
            ]

            # ── 各策略持倉摘要 ──
            positions = self._get_positions()
            strategy_positions: dict[str, list] = {n: [] for n in self._strategy_names()}
            for pos in positions:
                sym = self._pos_attr(pos, "symbol", "?")
                strategy = self._symbol_to_strategy(sym)
                if strategy in strategy_positions:
                    strategy_positions[strategy].append(pos)
                else:
                    strategy_positions.setdefault("其他", []).append(pos)

            for strat_name in self._strategy_names():
                strat_pos = strategy_positions.get(strat_name, [])
                strat_pnl = sum(self._pos_attr(p, "unrealized_pnl", 0) for p in strat_pos)
                strat_emoji = "📈" if strat_pnl >= 0 else "📉"
                count = len(strat_pos)
                lines.append(
                    f"\n<b>{'─' * 20}</b>"
                    f"\n🏷 <b>{strat_name}</b>  ({count} 倉) {strat_emoji} ${strat_pnl:+,.2f}"
                )
                if strat_pos:
                    for p in strat_pos:
                        sym = self._pos_attr(p, "symbol", "?")
                        qty = self._pos_attr(p, "qty", 0)
                        pnl = self._pos_attr(p, "unrealized_pnl", 0)
                        side = "L" if qty > 0 else "S"
                        e = "🟢" if pnl >= 0 else "🔴"
                        lines.append(f"  {e} {sym} [{side}] ${pnl:+,.2f}")
                else:
                    lines.append("  📭 無持倉")

            # ── 熔斷狀態 ──
            lines.append(f"\n<b>{'─' * 20}</b>")
            # 讀取各策略的 signal_state.json 看熔斷
            any_cb = False
            for name, cfg in self._configs:
                state_path = cfg.get_report_dir("live") / "signal_state.json"
                if state_path.exists():
                    try:
                        with open(state_path) as f:
                            state_data = json.load(f)
                        if state_data.get("circuit_breaker_triggered"):
                            lines.append(f"🚨 {name}: 熔斷已觸發！")
                            any_cb = True
                    except Exception:
                        pass
            if not any_cb:
                lines.append("✅ 熔斷: 全部正常")

            buttons = {
                "inline_keyboard": [
                    [
                        {"text": "📡 信號", "callback_data": "/signals"},
                        {"text": "📋 持倉詳情", "callback_data": "/positions"},
                    ],
                    [
                        {"text": "💰 盈虧", "callback_data": "/pnl"},
                        {"text": "🛡️ 風險", "callback_data": "/risk"},
                    ],
                ]
            }
            self._send_message(chat_id, "\n".join(lines), reply_markup=buttons)
            return ""
        except Exception as e:
            return f"❌ Dashboard 查詢失敗: {e}"

    # ══════════════════════════════════════════════════════════════
    # /status — 帳戶狀態（分區塊）
    # ══════════════════════════════════════════════════════════════

    def _cmd_status_multi(self, args: list[str], chat_id: str) -> str:
        if not self.broker:
            return "⚠️ Broker 未連接"

        try:
            info = self._get_account_info()
            if not info:
                return "⚠️ 無法取得帳戶資訊"

            equity = float(info.get("totalWalletBalance", 0)) + float(
                info.get("totalUnrealizedProfit", 0)
            )
            wallet = float(info.get("totalWalletBalance", 0))
            available = float(info.get("availableBalance", 0))
            unrealized = float(info.get("totalUnrealizedProfit", 0))
            margin_balance = float(info.get("totalMarginBalance", 0))
            init_margin = float(info.get("totalInitialMargin", 0))

            margin_ratio = (init_margin / margin_balance * 100) if margin_balance > 0 else 0
            pnl_emoji = "📈" if unrealized >= 0 else "📉"

            lines = [
                "💼 <b>帳戶狀態</b>\n",
                f"💰 總權益: <b>${equity:,.2f}</b>",
                f"💵 錢包: ${wallet:,.2f}",
                f"💵 可用: ${available:,.2f}",
                f"{pnl_emoji} 未實現 PnL: ${unrealized:+,.2f}",
                f"📊 保證金使用: {margin_ratio:.1f}%",
            ]

            if margin_ratio >= 80:
                lines.append("⚠️ <b>保證金偏高！</b>")

            # ── 按策略分組持倉 ──
            positions = self._get_positions()
            if positions:
                for name, cfg in self._configs:
                    strat_pos = [
                        p for p in positions
                        if self._pos_attr(p, "symbol", "") in cfg.market.symbols
                    ]
                    if not strat_pos:
                        continue
                    strat_pnl = sum(self._pos_attr(p, "unrealized_pnl", 0) for p in strat_pos)
                    e = "📈" if strat_pnl >= 0 else "📉"
                    lines.append(f"\n🏷 <b>{name}</b> {e} ${strat_pnl:+,.2f}")
                    for p in strat_pos:
                        sym = self._pos_attr(p, "symbol", "?")
                        qty = self._pos_attr(p, "qty", 0)
                        pnl = self._pos_attr(p, "unrealized_pnl", 0)
                        entry = self._pos_attr(p, "entry_price", 0)
                        side = "LONG" if qty > 0 else "SHORT"
                        pe = "🟢" if pnl >= 0 else "🔴"
                        lines.append(f"  {pe} {sym} [{side}] ${pnl:+,.2f}")
            else:
                lines.append("\n📭 無持倉")

            buttons = {
                "inline_keyboard": [[
                    {"text": "📋 持倉詳情", "callback_data": "/positions"},
                    {"text": "🛡️ 風險", "callback_data": "/risk"},
                ]]
            }
            self._send_message(chat_id, "\n".join(lines), reply_markup=buttons)
            return ""
        except Exception as e:
            return f"❌ 狀態查詢失敗: {e}"

    # ══════════════════════════════════════════════════════════════
    # /signals — 多策略信號
    # ══════════════════════════════════════════════════════════════

    def _cmd_signals_multi(self, args: list[str], chat_id: str) -> str:
        """
        /signals          → 顯示所有策略信號
        /signals meta_blend → 只顯示 meta_blend
        """
        filter_name = args[0] if args else None

        try:
            all_lines: list[str] = []
            has_any = False

            for name, cfg in self._configs:
                if filter_name and filter_name.lower() not in name.lower():
                    continue

                signals, gen_at = self._read_signals(name, cfg)
                age_str = self._signal_age_str(gen_at)

                if signals is None:
                    all_lines.append(f"\n🏷 <b>{name}</b>  ⚠️ 無信號快照")
                    continue

                has_any = True
                all_lines.append(f"\n🏷 <b>{name}</b>  {age_str}")

                for sig in signals:
                    signal_val = sig.get("signal", 0)
                    price = sig.get("price", 0)
                    symbol = sig.get("symbol", "?")
                    ind = sig.get("indicators", {})

                    if signal_val > 0.5:
                        emoji, label = "🟢", f"LONG {signal_val:.0%}"
                    elif signal_val < -0.5:
                        emoji, label = "🔴", f"SHORT {abs(signal_val):.0%}"
                    elif abs(signal_val) > 0.01:
                        emoji, label = "🟡", f"{'L' if signal_val > 0 else 'S'} {signal_val:.0%}"
                    else:
                        emoji, label = "⚪", "FLAT"

                    line = f"  {emoji} <b>{symbol}</b>: {label} @ ${price:,.2f}"

                    # 指標摘要（顯示前幾個可用的）
                    ind_parts = []
                    for k in ("rsi", "adx", "tsmom", "carry", "er"):
                        v = ind.get(k)
                        if v is not None:
                            ind_parts.append(f"{k.upper()}={v}")
                    if ind_parts:
                        line += f"\n    {' | '.join(ind_parts[:4])}"

                    # 如果有持倉資訊（_position 欄位）
                    pos_info = sig.get("_position")
                    if pos_info and abs(pos_info.get("pct", 0)) > 0.01:
                        side = pos_info.get("side", "?")
                        entry = pos_info.get("entry", 0)
                        sl = pos_info.get("sl")
                        tp = pos_info.get("tp")
                        line += f"\n    📦 {side} @ ${entry:,.2f}"
                        if sl:
                            line += f" | SL ${sl:,.2f}"
                        if tp:
                            line += f" | TP ${tp:,.2f}"

                    all_lines.append(line)

            if not all_lines:
                return "📡 無可用信號"

            header = "📡 <b>交易信號</b>"
            text = header + "\n" + "\n".join(all_lines)

            # 按鈕：按策略篩選
            strat_buttons = [
                {"text": f"📡 {n}", "callback_data": f"/signals {n}"}
                for n in self._strategy_names()
            ]
            # 加「全部」按鈕
            strat_buttons.insert(0, {"text": "📡 全部", "callback_data": "/signals"})
            buttons = {"inline_keyboard": [strat_buttons]}
            self._send_message(chat_id, text, reply_markup=buttons)
            return ""
        except Exception as e:
            return f"❌ 信號查詢失敗: {e}"

    # ══════════════════════════════════════════════════════════════
    # /pnl — 帳戶盈虧
    # ══════════════════════════════════════════════════════════════

    def _cmd_pnl_multi(self, args: list[str], chat_id: str) -> str:
        """
        /pnl        → 今日
        /pnl 7d     → 最近 7 天
        /pnl 30d    → 最近 30 天
        /pnl all    → 全部（90天）
        """
        if not self.broker:
            return "⚠️ Broker 未連接"

        try:
            if not hasattr(self.broker, "get_income_history"):
                return "⚠️ 無盈虧查詢功能"

            now = datetime.now(timezone.utc)
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

            incomes = self._fetch_income_paginated(start_dt, now)

            realized = sum(i["income"] for i in incomes if i["income_type"] == "REALIZED_PNL")
            commission = sum(i["income"] for i in incomes if i["income_type"] == "COMMISSION")
            funding = sum(i["income"] for i in incomes if i["income_type"] == "FUNDING_FEE")

            unrealized = 0.0
            positions = self._get_positions()
            for pos in positions:
                unrealized += self._pos_attr(pos, "unrealized_pnl", 0)

            total = realized + commission + funding + unrealized
            emoji = "📈" if total >= 0 else "📉"
            trade_count = sum(1 for i in incomes if i["income_type"] == "REALIZED_PNL")

            lines = [
                f"{emoji} <b>盈虧 — {label}</b>\n",
                f"💰 總計: <b>${total:+,.2f}</b>",
                f"✅ 已實現: ${realized:+,.2f}  ({trade_count} 筆)",
                f"⏳ 未實現: ${unrealized:+,.2f}",
                f"💸 手續費: ${commission:+,.2f}",
                f"🔄 資金費率: ${funding:+,.2f}",
            ]

            # 按策略拆分已實現 PnL
            if len(self._configs) > 1:
                lines.append(f"\n<b>{'─' * 20}</b>")
                all_symbols_map: dict[str, str] = {}
                for name, cfg in self._configs:
                    for sym in cfg.market.symbols:
                        all_symbols_map[sym] = name

                strat_pnl: dict[str, float] = {n: 0.0 for n in self._strategy_names()}
                for inc in incomes:
                    if inc["income_type"] == "REALIZED_PNL":
                        sym = inc.get("symbol", "")
                        sn = all_symbols_map.get(sym, "其他")
                        strat_pnl[sn] = strat_pnl.get(sn, 0) + inc["income"]

                for sn, pnl in strat_pnl.items():
                    e = "📈" if pnl >= 0 else "📉"
                    lines.append(f"  {e} {sn}: ${pnl:+,.2f}")

            buttons = {
                "inline_keyboard": [[
                    {"text": "📅 今日", "callback_data": "/pnl"},
                    {"text": "📅 7天", "callback_data": "/pnl 7d"},
                    {"text": "📅 30天", "callback_data": "/pnl 30d"},
                    {"text": "📅 全部", "callback_data": "/pnl all"},
                ]]
            }
            self._send_message(chat_id, "\n".join(lines), reply_markup=buttons)
            return ""
        except Exception as e:
            return f"❌ 盈虧查詢失敗: {e}"

    # ══════════════════════════════════════════════════════════════
    # /positions — 持倉列表（可展開詳情）
    # ══════════════════════════════════════════════════════════════

    def _cmd_positions_multi(self, args: list[str], chat_id: str) -> str:
        """
        /positions          → 精簡列表
        /positions BTCUSDT  → 展開單一幣種詳情 + SL/TP
        """
        if not self.broker:
            return "⚠️ Broker 未連接"

        try:
            positions = self._get_positions()
            if not positions:
                return "📭 目前沒有持倉"

            detail_symbol = args[0].upper() if args else None

            # 如果指定幣種，顯示詳情
            if detail_symbol:
                return self._format_position_detail(detail_symbol, positions)

            # 否則顯示精簡列表
            lines = ["📋 <b>持倉列表</b>\n"]
            detail_buttons = []

            for name, cfg in self._configs:
                strat_pos = [
                    p for p in positions
                    if self._pos_attr(p, "symbol", "") in cfg.market.symbols
                ]
                if not strat_pos:
                    continue
                strat_pnl = sum(self._pos_attr(p, "unrealized_pnl", 0) for p in strat_pos)
                e = "📈" if strat_pnl >= 0 else "📉"
                lines.append(f"🏷 <b>{name}</b> {e} ${strat_pnl:+,.2f}")

                for p in strat_pos:
                    sym = self._pos_attr(p, "symbol", "?")
                    qty = self._pos_attr(p, "qty", 0)
                    entry = self._pos_attr(p, "entry_price", 0)
                    pnl = self._pos_attr(p, "unrealized_pnl", 0)
                    mark = self._pos_attr(p, "mark_price", 0)
                    lev = self._pos_attr(p, "leverage", 0)
                    side = "LONG" if qty > 0 else "SHORT"
                    pe = "🟢" if pnl >= 0 else "🔴"

                    notional = abs(qty * mark) if mark > 0 else abs(qty * entry)
                    pnl_pct = (pnl / notional * 100) if notional > 0 else 0

                    lev_str = f" {lev}x" if lev and lev > 1 else ""
                    lines.append(
                        f"  {pe} <b>{sym}</b> [{side}]{lev_str}"
                        f"  ${pnl:+,.2f} ({pnl_pct:+.1f}%)"
                    )
                    detail_buttons.append(
                        {"text": f"🔍 {sym}", "callback_data": f"/positions {sym}"}
                    )
                lines.append("")

            # 展開按鈕（每行最多 3 個）
            button_rows = []
            for i in range(0, len(detail_buttons), 3):
                button_rows.append(detail_buttons[i : i + 3])
            buttons = {"inline_keyboard": button_rows} if button_rows else None
            self._send_message(chat_id, "\n".join(lines), reply_markup=buttons)
            return ""
        except Exception as e:
            return f"❌ 持倉查詢失敗: {e}"

    def _format_position_detail(self, symbol: str, positions: list) -> str:
        """顯示單一幣種的完整持倉 + SL/TP"""
        target = None
        for p in positions:
            if self._pos_attr(p, "symbol", "") == symbol:
                target = p
                break
        if not target:
            return f"📭 {symbol} 無持倉"

        qty = self._pos_attr(target, "qty", 0)
        entry = self._pos_attr(target, "entry_price", 0)
        pnl = self._pos_attr(target, "unrealized_pnl", 0)
        mark = self._pos_attr(target, "mark_price", 0)
        lev = self._pos_attr(target, "leverage", 0)
        liq = self._pos_attr(target, "liquidation_price", 0)
        is_long = qty > 0
        side = "LONG" if is_long else "SHORT"

        notional = abs(qty * mark) if mark > 0 else abs(qty * entry)
        pnl_pct = (pnl / notional * 100) if notional > 0 else 0
        pe = "🟢" if pnl >= 0 else "🔴"

        strategy = self._symbol_to_strategy(symbol)

        lines = [
            f"{pe} <b>{symbol}</b> [{side}] — {strategy}\n",
            f"📦 數量: {abs(qty):.6f}",
            f"📍 入場: ${entry:,.2f}",
        ]
        if mark > 0:
            lines.append(f"💹 現價: ${mark:,.2f}")
        if lev and lev > 1:
            lines.append(f"⚡ 槓桿: {lev}x")
        lines.append(f"📊 名義: ${notional:,.0f}")
        lines.append(f"💰 盈虧: <b>${pnl:+,.2f}</b> ({pnl_pct:+.2f}%)")

        if liq and liq > 0:
            if mark > 0:
                dist = abs(mark - liq) / mark * 100
                lines.append(f"🚨 強平: ${liq:,.2f} (距 {dist:.1f}%)")
            else:
                lines.append(f"🚨 強平: ${liq:,.2f}")

        # SL/TP 掛單
        if hasattr(self.broker, "get_all_conditional_orders"):
            try:
                orders = self.broker.get_all_conditional_orders(symbol)
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

                lines.append("")
                if sl_price:
                    sl_pnl = self._calc_pnl(entry, sl_price, abs(qty), is_long)
                    pnl_str = f" (<b>{sl_pnl:+.2f}</b>)" if sl_pnl is not None else ""
                    lines.append(f"🛡️ 止損: ${sl_price:,.2f}{pnl_str}")
                if tp_price:
                    tp_pnl = self._calc_pnl(entry, tp_price, abs(qty), is_long)
                    pnl_str = f" (<b>{tp_pnl:+.2f}</b>)" if tp_pnl is not None else ""
                    lines.append(f"🎯 止盈: ${tp_price:,.2f}{pnl_str}")
                if not sl_price and not tp_price:
                    lines.append("⚠️ 無 SL/TP 掛單")
            except Exception:
                lines.append("⚠️ SL/TP 查詢失敗")

        return "\n".join(lines)

    # ══════════════════════════════════════════════════════════════
    # /health — 紅黃綠摘要
    # ══════════════════════════════════════════════════════════════

    def _cmd_health_multi(self, args: list[str], chat_id: str) -> str:
        """
        /health         → 紅黃綠摘要
        /health detail  → 技術細節
        """
        import shutil

        show_detail = bool(args and args[0].lower() == "detail")

        # 收集各策略健康指標
        strategy_health: list[tuple[str, str, list[str]]] = []  # (name, status, details)

        for name, cfg in self._configs:
            status = "green"
            details: list[str] = []
            live_dir = cfg.get_report_dir("live")

            # 信號新鮮度
            sig_path = live_dir / "last_signals.json"
            if sig_path.exists():
                sig_age = time.time() - sig_path.stat().st_mtime
                if sig_age < 3600:
                    details.append(f"📝 信號: {sig_age / 60:.0f}m 前")
                elif sig_age < 7200:
                    details.append(f"📝 信號: ⚠️ {sig_age / 3600:.1f}h 前")
                    status = "yellow" if status == "green" else status
                else:
                    details.append(f"📝 信號: 🚨 {sig_age / 3600:.1f}h 前")
                    status = "red"
            else:
                # 嘗試 signal_state.json
                ss_path = live_dir / "signal_state.json"
                if ss_path.exists():
                    ss_age = time.time() - ss_path.stat().st_mtime
                    if ss_age < 7200:
                        details.append(f"📝 狀態: {ss_age / 60:.0f}m 前")
                    else:
                        details.append(f"📝 狀態: ⚠️ {ss_age / 3600:.1f}h 前")
                        status = "yellow" if status == "green" else status
                else:
                    details.append("📝 無信號檔案")
                    status = "yellow" if status == "green" else status

            # K 線快取新鮮度
            cache_dir = live_dir / "kline_cache"
            if cache_dir.exists():
                parquets = list(cache_dir.glob("*.parquet"))
                if parquets:
                    newest = max(p.stat().st_mtime for p in parquets)
                    cache_age = time.time() - newest
                    if cache_age < 7200:
                        details.append(f"📊 K線: {cache_age / 60:.0f}m 前")
                    else:
                        details.append(f"📊 K線: ⚠️ {cache_age / 3600:.1f}h 前")
                        status = "yellow" if status == "green" else status

            # Watchdog 狀態
            wd_path = live_dir.parent / "live_watchdog" / name if False else None
            # 嘗試讀取 watchdog latest_status
            wd_dir = Path("reports/live_watchdog") / name
            wd_latest = wd_dir / "latest_status.json"
            if wd_latest.exists():
                try:
                    with open(wd_latest) as f:
                        wd_data = json.load(f)
                    wd_status = wd_data.get("overall_status", "?")
                    if wd_status == "ok":
                        details.append("🩺 Watchdog: ✅")
                    elif wd_status == "warn":
                        details.append("🩺 Watchdog: ⚠️")
                        status = "yellow" if status == "green" else status
                    else:
                        details.append(f"🩺 Watchdog: 🚨 {wd_status}")
                        status = "red"
                except Exception:
                    pass

            strategy_health.append((name, status, details))

        # 彙總
        overall = "green"
        for _, s, _ in strategy_health:
            if s == "red":
                overall = "red"
            elif s == "yellow" and overall == "green":
                overall = "yellow"

        status_emoji = {"green": "🟢", "yellow": "🟡", "red": "🔴"}
        lines = [
            f"🏥 <b>系統健康</b>  {status_emoji[overall]} {overall.upper()}\n"
        ]

        for name, s, details in strategy_health:
            lines.append(f"{status_emoji[s]} <b>{name}</b>")
            if show_detail:
                for d in details:
                    lines.append(f"  {d}")

        # 系統資源
        lines.append(f"\n<b>{'─' * 20}</b>")
        try:
            try:
                import psutil
                mem = psutil.virtual_memory()
                lines.append(f"🧠 記憶體: {mem.percent:.0f}% ({mem.available / 1073741824:.1f} GB 可用)")
            except ImportError:
                try:
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
                            lines.append(f"🧠 記憶體: {pct:.0f}% ({avail / 1048576:.1f} GB 可用)")
                except Exception:
                    pass
        except Exception:
            pass

        try:
            usage = shutil.disk_usage("/")
            disk_pct = usage.used / usage.total * 100
            free_gb = usage.free / 1073741824
            lines.append(f"💾 磁碟: {disk_pct:.0f}% ({free_gb:.1f} GB 可用)")
        except Exception:
            pass

        if not show_detail:
            buttons = {
                "inline_keyboard": [[
                    {"text": "🔍 詳細資訊", "callback_data": "/health detail"},
                ]]
            }
        else:
            buttons = {
                "inline_keyboard": [[
                    {"text": "📊 總覽", "callback_data": "/dashboard"},
                    {"text": "🏥 摘要", "callback_data": "/health"},
                ]]
            }
        self._send_message(chat_id, "\n".join(lines), reply_markup=buttons)
        return ""

    # ══════════════════════════════════════════════════════════════
    # /risk — 風險總覽（按策略分組）
    # ══════════════════════════════════════════════════════════════

    def _cmd_risk_multi(self, args: list[str], chat_id: str) -> str:
        if not self.broker:
            return "⚠️ Broker 未連接"

        try:
            lines = ["🛡️ <b>風險總覽</b>\n"]

            info = self._get_account_info()
            if info:
                equity = float(info.get("totalWalletBalance", 0)) + float(
                    info.get("totalUnrealizedProfit", 0)
                )
                available = float(info.get("availableBalance", 0))
                margin_balance = float(info.get("totalMarginBalance", 0))
                init_margin = float(info.get("totalInitialMargin", 0))
                maint_margin = float(info.get("totalMaintMargin", 0))

                margin_ratio = (init_margin / margin_balance * 100) if margin_balance > 0 else 0
                maint_ratio = (maint_margin / margin_balance * 100) if margin_balance > 0 else 0

                lines.extend([
                    f"💰 權益: <b>${equity:,.2f}</b>",
                    f"💵 可用: ${available:,.2f}",
                    f"📊 保證金使用: {margin_ratio:.1f}%",
                    f"🔒 維持保證金: {maint_ratio:.1f}%",
                ])
                if margin_ratio >= 80:
                    lines.append("⚠️ <b>保證金使用率偏高！</b>")

            # 按策略分組曝險
            positions = self._get_positions()
            if positions:
                total_notional = 0.0
                for name, cfg in self._configs:
                    strat_pos = [
                        p for p in positions
                        if self._pos_attr(p, "symbol", "") in cfg.market.symbols
                    ]
                    if not strat_pos:
                        continue

                    strat_notional = 0.0
                    lines.append(f"\n🏷 <b>{name}</b>")
                    for p in strat_pos:
                        sym = self._pos_attr(p, "symbol", "?")
                        qty = self._pos_attr(p, "qty", 0)
                        mark = self._pos_attr(p, "mark_price", 0)
                        entry = self._pos_attr(p, "entry_price", 0)
                        liq = self._pos_attr(p, "liquidation_price", 0)
                        lev = self._pos_attr(p, "leverage", 0)
                        pnl = self._pos_attr(p, "unrealized_pnl", 0)
                        notional = abs(qty * mark) if mark > 0 else abs(qty * entry)
                        strat_notional += notional
                        total_notional += notional
                        side = "L" if qty > 0 else "S"
                        pe = "📈" if pnl >= 0 else "📉"

                        pos_line = f"  {pe} {sym} [{side}] {lev}x ${notional:,.0f}"
                        if liq and liq > 0 and mark > 0:
                            dist = abs(mark - liq) / mark * 100
                            pos_line += f" (強平距 {dist:.1f}%)"
                        lines.append(pos_line)
                    lines.append(f"  💎 小計: ${strat_notional:,.0f}")

                lines.append(f"\n💎 <b>總名義曝險: ${total_notional:,.0f}</b>")
            else:
                lines.append("\n📭 無持倉")

            buttons = {
                "inline_keyboard": [[
                    {"text": "📊 總覽", "callback_data": "/dashboard"},
                    {"text": "📋 持倉", "callback_data": "/positions"},
                ]]
            }
            self._send_message(chat_id, "\n".join(lines), reply_markup=buttons)
            return ""
        except Exception as e:
            return f"❌ 風險查詢失敗: {e}"

    # ══════════════════════════════════════════════════════════════
    # /balance, /trades — 複用 TelegramBot 基底
    # ══════════════════════════════════════════════════════════════

    # _cmd_balance 和 _cmd_trades 直接繼承自 TelegramBot

    # ══════════════════════════════════════════════════════════════
    # 每日自動摘要（UTC 00:05）
    # ══════════════════════════════════════════════════════════════

    def _daily_summary_loop(self):
        """背景線程：每天 UTC 00:05 推送前一天摘要"""
        logger.info("📅 每日摘要排程已啟動")
        while self._running:
            try:
                now = datetime.now(timezone.utc)
                today_str = now.strftime("%Y-%m-%d")

                # UTC 00:05 ~ 00:10 之間且今天還沒送過
                if now.hour == 0 and 5 <= now.minute <= 10 and self._daily_last_date != today_str:
                    self._send_daily_summary()
                    self._daily_last_date = today_str
            except Exception as e:
                logger.error(f"每日摘要異常: {e}")

            time.sleep(30)  # 每 30 秒檢查一次

    def _send_daily_summary(self):
        """生成並推送每日摘要"""
        if not self.broker or not self.chat_id:
            return

        try:
            now = datetime.now(timezone.utc)
            yesterday_start = (now - timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            yesterday_end = now.replace(hour=0, minute=0, second=0, microsecond=0)

            incomes = self._fetch_income_paginated(yesterday_start, yesterday_end)

            realized = sum(i["income"] for i in incomes if i["income_type"] == "REALIZED_PNL")
            commission = sum(i["income"] for i in incomes if i["income_type"] == "COMMISSION")
            funding = sum(i["income"] for i in incomes if i["income_type"] == "FUNDING_FEE")
            trade_count = sum(1 for i in incomes if i["income_type"] == "REALIZED_PNL")
            total = realized + commission + funding

            equity = self._get_equity()
            positions = self._get_positions()
            pos_count = len(positions)
            unrealized = sum(self._pos_attr(p, "unrealized_pnl", 0) for p in positions)

            emoji = "📈" if total >= 0 else "📉"
            date_str = yesterday_start.strftime("%Y-%m-%d")

            lines = [
                f"📅 <b>每日摘要 — {date_str}</b>\n",
                f"{emoji} 當日 PnL: <b>${total:+,.2f}</b>",
                f"  ✅ 已實現: ${realized:+,.2f} ({trade_count} 筆)",
                f"  💸 手續費: ${commission:+,.2f}",
                f"  🔄 資金費率: ${funding:+,.2f}",
                f"",
                f"💰 當前權益: ${equity:,.2f}",
                f"⏳ 未實現 PnL: ${unrealized:+,.2f}",
                f"📋 持倉數: {pos_count}",
            ]

            # 按策略拆分
            if len(self._configs) > 1:
                all_symbols_map: dict[str, str] = {}
                for name, cfg in self._configs:
                    for sym in cfg.market.symbols:
                        all_symbols_map[sym] = name

                strat_pnl: dict[str, float] = {}
                for inc in incomes:
                    if inc["income_type"] == "REALIZED_PNL":
                        sym = inc.get("symbol", "")
                        sn = all_symbols_map.get(sym, "其他")
                        strat_pnl[sn] = strat_pnl.get(sn, 0) + inc["income"]

                if strat_pnl:
                    lines.append(f"\n<b>{'─' * 20}</b>")
                    for sn, pnl in strat_pnl.items():
                        e = "📈" if pnl >= 0 else "📉"
                        lines.append(f"  {e} {sn}: ${pnl:+,.2f}")

            self._send_message(self.chat_id, "\n".join(lines))
            logger.info(f"📅 每日摘要已推送: {date_str}")
        except Exception as e:
            logger.error(f"每日摘要生成失敗: {e}")

    # ══════════════════════════════════════════════════════════════
    # 告警系統
    # ══════════════════════════════════════════════════════════════

    def _alert_loop(self):
        """背景線程：每 5 分鐘檢查告警條件"""
        logger.info("🔔 告警檢查已啟動")
        # 初始化時記錄峰值
        try:
            self._peak_equity = self._get_equity() or 0
        except Exception:
            pass

        while self._running:
            try:
                self._check_alerts()
            except Exception as e:
                logger.error(f"告警檢查異常: {e}")
            time.sleep(300)  # 每 5 分鐘

    def _check_alerts(self):
        """檢查所有告警條件"""
        if not self.broker or not self.chat_id:
            return

        # ── Drawdown 告警 ──
        dd_warn = self._alert_cfg.get("drawdown_warn_pct", 0)
        dd_crit = self._alert_cfg.get("drawdown_critical_pct", 0)

        if dd_warn or dd_crit:
            equity = self._get_equity()
            if equity and equity > self._peak_equity:
                self._peak_equity = equity

            if self._peak_equity > 0 and equity > 0:
                dd_pct = (1 - equity / self._peak_equity) * 100
                if dd_crit and dd_pct >= dd_crit:
                    self._send_message(
                        self.chat_id,
                        f"🚨🚨 <b>嚴重回撤告警</b>\n\n"
                        f"回撤: <b>{dd_pct:.1f}%</b> (閾值 {dd_crit}%)\n"
                        f"峰值: ${self._peak_equity:,.2f}\n"
                        f"當前: ${equity:,.2f}",
                    )
                elif dd_warn and dd_pct >= dd_warn:
                    self._send_message(
                        self.chat_id,
                        f"⚠️ <b>回撤警告</b>\n\n"
                        f"回撤: <b>{dd_pct:.1f}%</b> (閾值 {dd_warn}%)\n"
                        f"峰值: ${self._peak_equity:,.2f}\n"
                        f"當前: ${equity:,.2f}",
                    )

        # ── 大額交易通知 ──
        large_trade_usdt = self._alert_cfg.get("large_trade_usdt", 0)
        if large_trade_usdt and hasattr(self.broker, "get_income_history"):
            try:
                now = datetime.now(timezone.utc)
                start_ms = int((now - timedelta(minutes=6)).timestamp() * 1000)
                end_ms = int(now.timestamp() * 1000)
                recent = self.broker.get_income_history(
                    limit=50, start_time=start_ms, end_time=end_ms
                )
                for inc in recent:
                    if inc["income_type"] == "REALIZED_PNL":
                        pnl = abs(inc["income"])
                        if pnl >= large_trade_usdt:
                            sym = inc.get("symbol", "?")
                            val = inc["income"]
                            e = "📈" if val >= 0 else "📉"
                            self._send_message(
                                self.chat_id,
                                f"💎 <b>大額交易</b>\n\n"
                                f"{e} {sym}: ${val:+,.2f}",
                            )
            except Exception:
                pass
