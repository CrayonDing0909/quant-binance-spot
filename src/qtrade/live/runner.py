"""
Live Runner — 即時交易主循環 (v4.0)

v4.0: 繼承 BaseRunner，消除與 WebSocketRunner 的重複代碼
    - 所有安全機制由 BaseRunner 統一管理
    - 本類只負責 Polling 定時器 + run_once() 批次處理

功能：
    - 每根 K 線收盤後運行策略
    - 對比信號與當前倉位，決定交易
    - 支援 Paper Trading / Real Trading 模式切換
    - Telegram 通知（交易 + 定期摘要）
    - 支援動態倉位計算（Kelly / 波動率）
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, Optional

from ..config import AppConfig
from ..monitor.notifier import TelegramNotifier
from .signal_generator import generate_signal, SignalResult, PositionInfo
from .kline_cache import IncrementalKlineCache
from .paper_broker import PaperBroker
from .trading_state import TradingStateManager
from .trading_db import TradingDatabase
from .base_runner import BaseRunner
from ..utils.log import get_logger

live_logger = get_logger("live_runner")


class BrokerProtocol(Protocol):
    """Broker 通用介面，Paper / Spot / Futures broker 都實現此介面。

    所有 Broker 的 execute_target_position 簽名必須一致：
      - current_price: float | None = None（Futures 可自動取價）
      - stop_loss_price / take_profit_price: float | None = None
    """
    def execute_target_position(
        self, symbol: str, target_pct: float,
        current_price: float | None = None, reason: str = "",
        stop_loss_price: float | None = None,
        take_profit_price: float | None = None,
    ) -> object | None: ...

    def get_position_pct(self, symbol: str, current_price: float) -> float: ...


class LiveRunner(BaseRunner):
    """
    即時交易主循環 (v4.0)

    繼承 BaseRunner 取得所有安全機制，
    本類只負責 Polling 定時器和批次信號處理。

    Usage:
        runner = LiveRunner(cfg, broker, mode="paper")
        runner.run()  # 阻塞運行，每根 K 線觸發一次
    """

    def __init__(
        self,
        cfg: AppConfig,
        broker: BrokerProtocol,
        mode: str = "paper",
        notifier: TelegramNotifier | None = None,
        state_path: Optional[Path] = None,
    ):
        super().__init__(cfg, broker, mode, notifier)

        self.tick_count = 0

        # 狀態管理器（用於 Real Trading 持久化）
        self.state_manager: Optional[TradingStateManager] = None
        if state_path or mode == "real":
            default_state_path = cfg.get_report_dir("live") / f"{mode}_state.json"
            self.state_manager = TradingStateManager(
                state_path=state_path or default_state_path,
                strategy_name=self.strategy_name,
                symbols=self.symbols,
                interval=self.interval,
                mode=mode,
                encrypt=(mode == "real"),
            )

        # K 線快取（可選，由 config 控制）
        if cfg.live.kline_cache:
            cache_dir = cfg.get_report_dir("live") / "kline_cache"
            self._kline_cache = IncrementalKlineCache(
                cache_dir=cache_dir,
                interval=self.interval,
                seed_bars=300,
                market_type=self.market_type,
            )
            self._log.info("📦 增量 K 線快取已啟用")

    @property
    def _log(self):
        return live_logger

    def run_once(self) -> list[SignalResult]:
        """
        執行一次信號檢查 + 下單

        Returns:
            signals: 所有幣種的信號列表
        """
        if self._check_circuit_breaker():
            self._log.warning("⛔ 熔斷已觸發，跳過本次交易")
            return []

        self.tick_count += 1
        signals = []
        has_trade = False

        if self.state_manager:
            self.state_manager.increment_tick()

        for symbol in self.symbols:
            # Ensemble 路由：取得 symbol 專屬策略名與參數
            sym_strategy, params = self._get_strategy_for_symbol(symbol)
            direction = self.cfg.direction

            try:
                cached_df = None
                if self._kline_cache is not None:
                    cached_df = self._kline_cache.get_klines(symbol)
                    if cached_df is not None and len(cached_df) < 50:
                        self._log.warning(
                            f"⚠️  {symbol}: 快取數據不足 ({len(cached_df)} bar)，"
                            f"fallback 到 fetch_recent_klines"
                        )
                        cached_df = None

                sig = generate_signal(
                    symbol=symbol,
                    strategy_name=sym_strategy,
                    params=params,
                    interval=self.interval,
                    market_type=self.market_type,
                    direction=direction,
                    df=cached_df,
                    overlay_cfg=getattr(self.cfg, '_overlay_cfg', None),
                )
            except Exception as e:
                self._log.error(f"❌ {symbol} 信號生成失敗: {e}")
                self.notifier.send_error(f"{symbol} 信號生成失敗: {e}")
                if self.state_manager:
                    self.state_manager.log_error(f"{symbol} 信號生成失敗: {e}")
                continue

            signals.append(sig)

            # 使用 BaseRunner 的共享信號處理
            trade = self._process_signal(symbol, sig)

            if trade:
                has_trade = True

                # LR-specific: 記錄到狀態管理器
                if self.state_manager:
                    self.state_manager.log_trade(
                        symbol=symbol,
                        side=trade.side,
                        qty=trade.qty,
                        price=trade.price,
                        fee=getattr(trade, "fee", 0.0),
                        pnl=trade.pnl,
                        reason=getattr(trade, "reason", ""),
                        order_id=getattr(trade, "order_id", ""),
                    )
                    if isinstance(self.broker, PaperBroker):
                        pos = self.broker.get_position(symbol)
                        self.state_manager.update_position(symbol, pos.qty, pos.avg_entry)

            # 附加持倉 + SL/TP 資訊到 signal dict（供 Telegram 摘要使用）
            self._attach_position_info(symbol, sig)

        # 發送信號摘要到 Telegram
        if has_trade or self.tick_count <= 1 or self.tick_count % 6 == 0:
            self.notifier.send_signal_summary(
                signals,
                mode=self.mode.upper(),
                has_trade=has_trade,
            )

        self._save_last_signals(signals)

        if isinstance(self.broker, PaperBroker):
            self.broker.touch_state()

        # 定期重新計算 Kelly
        if self.cfg.position_sizing.method == "kelly" and self.tick_count % 24 == 0:
            self._init_position_sizer()

        return signals

    def _attach_position_info(self, symbol: str, sig: SignalResult) -> None:
        """附加持倉 + SL/TP 資訊到 SignalResult"""
        price = sig.price
        current_pct = 0.0
        try:
            current_pct = self.broker.get_position_pct(symbol, price)
        except Exception:
            pass

        pos_info = PositionInfo(pct=current_pct)

        if not isinstance(self.broker, PaperBroker) and hasattr(self.broker, "get_position"):
            try:
                pos_obj = self.broker.get_position(symbol)
                if pos_obj and abs(pos_obj.qty) > 1e-10:
                    live_pct = self.broker.get_position_pct(symbol, price)
                    pos_info = PositionInfo(
                        pct=live_pct,
                        entry=pos_obj.entry_price,
                        qty=abs(pos_obj.qty),
                        side="LONG" if pos_obj.qty > 0 else "SHORT",
                    )
                    if hasattr(self.broker, "get_all_conditional_orders"):
                        orders = self.broker.get_all_conditional_orders(symbol)
                        pos_side_str = "LONG" if pos_obj.qty > 0 else "SHORT"
                        for o in orders:
                            o_ps = o.get("positionSide", "")
                            if o_ps and o_ps != pos_side_str and o_ps != "BOTH":
                                continue
                            otype = o.get("type", "")
                            trigger = float(
                                o.get("stopPrice", 0) or o.get("triggerPrice", 0) or 0
                            )
                            if trigger <= 0:
                                continue
                            if otype in {"STOP_MARKET", "STOP"}:
                                pos_info.sl = trigger
                            elif otype in {"TAKE_PROFIT_MARKET", "TAKE_PROFIT"}:
                                pos_info.tp = trigger
                            elif pos_obj.entry_price > 0:
                                is_long = pos_obj.qty > 0
                                if is_long:
                                    if trigger < pos_obj.entry_price:
                                        pos_info.sl = trigger
                                    else:
                                        pos_info.tp = trigger
                                else:
                                    if trigger > pos_obj.entry_price:
                                        pos_info.sl = trigger
                                    else:
                                        pos_info.tp = trigger
            except Exception:
                pass

        sig.position_info = pos_info

    def _save_last_signals(self, signals: list[SignalResult]) -> None:
        """保存最新信號到 JSON，供 Telegram /signals 讀取"""
        try:
            sig_path = self.cfg.get_report_dir("live") / "last_signals.json"
            sig_path.parent.mkdir(parents=True, exist_ok=True)

            serializable = [sig.to_dict() for sig in signals]

            payload = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "mode": self.mode,
                "signals": serializable,
            }

            with open(sig_path, "w") as f:
                json.dump(payload, f, indent=2, default=str)
        except Exception as e:
            self._log.debug(f"  保存信號快照失敗: {e}")

    def run(self, max_ticks: int | None = None) -> None:
        """
        阻塞運行主循環

        每根 K 線收盤後觸發一次 run_once()。
        通過 Ctrl+C 停止。
        """
        self.is_running = True
        self.start_time = time.time()
        interval_seconds = self._interval_to_seconds(self.interval)

        alloc_str = ", ".join(f"{s}={w:.0%}" for s, w in self._weights.items())
        self._log.info("=" * 60)
        self._log.info(f"🚀 Live Trading 啟動 [{self.mode.upper()}]")
        self._log.info(f"   策略: {self.strategy_name}")
        self._log.info(f"   交易對: {', '.join(self.symbols)}")
        self._log.info(f"   倉位分配: {alloc_str}")
        self._log.info(f"   K線週期: {self.interval} ({interval_seconds}s)")
        self._log.info(
            f"   模式: {'📝 Paper Trading' if self.mode == 'paper' else '💰 Real Trading'}"
        )
        if self.max_drawdown_pct:
            self._log.info(f"   熔斷線: 回撤 ≥ {self.max_drawdown_pct:.0%} → 自動平倉停止")
        self._log.info(
            f"   K線快取: {'✅ 增量快取' if self._kline_cache else '❌ 滑動窗口 (300 bar)'}"
        )
        self._log.info(
            f"   翻轉確認: {'✅ 2-tick' if self.cfg.live.flip_confirmation else '❌ 直接執行'}"
        )
        self._log.info(f"   交易資料庫: {'✅ SQLite' if self.trading_db else '❌ 未啟用'}")
        self._log.info(f"   Telegram: {'✅ 已啟用' if self.notifier.enabled else '❌ 未啟用'}")
        self._log.info("=" * 60)

        leverage = self.cfg.futures.leverage if self.cfg.futures else None
        self.notifier.send_startup(
            strategy=self.strategy_name,
            symbols=self.symbols,
            interval=self.interval,
            mode=self.mode,
            weights=self._weights,
            market_type=self.market_type,
            leverage=leverage,
        )

        try:
            while self.is_running:
                wait = self._seconds_until_next_close(interval_seconds)
                if wait > 5:
                    self._log.info(f"⏳ 等待下一根 K 線收盤... ({wait:.0f}s)")
                    while wait > 0 and self.is_running:
                        time.sleep(min(wait, 10))
                        wait -= 10
                else:
                    time.sleep(max(wait, 1))

                if not self.is_running:
                    break

                time.sleep(3)

                now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                self._log.info(f"\n{'─'*50}")
                self._log.info(f"📍 Tick #{self.tick_count + 1} @ {now}")

                self.run_once()

                if self._circuit_breaker_triggered:
                    self._log.warning("🚨 熔斷觸發，主循環終止")
                    break

                if self.tick_count % 6 == 0:
                    self._send_periodic_summary()

                if max_ticks and self.tick_count >= max_ticks:
                    self._log.info(f"🏁 達到最大運行次數 ({max_ticks})，停止")
                    break

        except KeyboardInterrupt:
            self._log.info("\n⛔ 收到停止信號 (Ctrl+C)")
        finally:
            self.is_running = False
            elapsed = time.time() - (self.start_time or time.time())
            self._log.info(
                f"📊 運行統計: {self.tick_count} ticks, "
                f"{self.trade_count} trades, {elapsed/3600:.1f}h"
            )
            self.notifier.send_shutdown(self.tick_count, self.trade_count, elapsed / 3600)

    def stop(self) -> None:
        self.is_running = False

    @staticmethod
    def _interval_to_seconds(interval: str) -> int:
        mapping = {
            "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
            "1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600,
            "8h": 28800, "12h": 43200, "1d": 86400,
        }
        return mapping.get(interval, 3600)

    @staticmethod
    def _seconds_until_next_close(interval_seconds: int) -> float:
        now = time.time()
        next_close = (int(now / interval_seconds) + 1) * interval_seconds
        return max(next_close - now, 0)
