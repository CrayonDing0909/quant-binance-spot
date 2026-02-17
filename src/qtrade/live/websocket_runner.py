"""
WebSocket Runner — 輕量化事件驅動交易執行器 (v4.0)

v4.0: 繼承 BaseRunner，消除與 LiveRunner 的重複代碼
    - 所有安全機制（SL/TP 冷卻、補掛、熔斷等）由 BaseRunner 統一管理
    - 本類只負責 WebSocket 連線 + K 線事件迴圈

適用場景：
    - Oracle Cloud (1GB RAM) 等資源受限環境
    - 需要即時反應（K 線收盤 0 秒延遲）
    - 與 Polling Runner (cron) 共用同一個 IncrementalKlineCache
"""
import json
import time
import logging
import traceback
import pandas as pd
from typing import Dict, Any

from ..config import AppConfig
from ..utils.log import get_logger
from .signal_generator import generate_signal
from .kline_cache import IncrementalKlineCache
from .base_runner import BaseRunner

ws_logger = get_logger("ws_runner")

# 心跳超時（秒）
HEARTBEAT_TIMEOUT = 300

# interval → 分鐘 對照表
INTERVAL_MINUTES = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "6h": 360, "8h": 480,
    "12h": 720, "1d": 1440,
}


class WebSocketRunner(BaseRunner):
    """
    基於 WebSocket 的輕量化執行器 (v4.0)

    繼承 BaseRunner 取得所有安全機制，
    本類只負責 WS 連線管理和 K 線事件驅動。
    """

    def __init__(self, cfg: AppConfig, broker, mode: str = "paper", notifier=None):
        super().__init__(cfg, broker, mode, notifier)

        self._tick_count = 0
        self._last_ws_message_time: float = 0.0
        self._ws_client = None
        self._last_kline_ts: Dict[str, int] = {}
        self._last_summary_time: float = 0.0
        self._interval_minutes = INTERVAL_MINUTES.get(self.interval, 60)

        # K 線快取（BaseRunner 的 _kline_cache 由子類設定）
        cache_dir = cfg.get_report_dir("live") / "kline_cache"
        self._kline_cache = IncrementalKlineCache(
            cache_dir=cache_dir,
            interval=self.interval,
            seed_bars=300,
            market_type=self.market_type,
        )

        # 預熱 K 線快取
        self._init_kline_buffer()

    @property
    def _log(self):
        return ws_logger

    # ══════════════════════════════════════════════════════════
    #  K 線管理
    # ══════════════════════════════════════════════════════════

    def _init_kline_buffer(self):
        """使用 IncrementalKlineCache 預熱 K 線"""
        self._log.info("📥 正在預熱 K 線緩衝區...")
        for symbol in self.symbols:
            try:
                df = self._kline_cache.get_klines(symbol)
                if df is not None and len(df) > 0:
                    self._log.info(
                        f"  ✅ {symbol}: 已載入 {len(df)} 根 {self.market_type} K 線 "
                        f"({df.index[0].strftime('%Y-%m-%d')} ~ "
                        f"{df.index[-1].strftime('%m-%d %H:%M')})"
                    )
                else:
                    self._log.warning(f"  ⚠️  {symbol}: 無法載入 K 線數據")
            except Exception as e:
                self._log.error(f"  ❌ {symbol}: K 線載入失敗: {e}")
                self._log.error(traceback.format_exc())

    def _on_kline_event(self, msg: Dict[str, Any]):
        """WebSocket K 線事件回調"""
        try:
            if "k" not in msg:
                return

            k = msg["k"]
            symbol = k["s"]
            is_closed = k["x"]
            close_price = float(k["c"])

            if is_closed:
                ts = k["t"]
                if self._last_kline_ts.get(symbol) == ts:
                    return
                self._last_kline_ts[symbol] = ts

                self._log.info(f"🕯️  {symbol} K 線收盤: ${close_price:,.2f}")

                self._append_kline(symbol, k)

                self._tick_count += 1
                self._run_strategy_for_symbol(symbol)

                now = time.time()
                if now - self._last_summary_time > 6 * 3600:
                    self._last_summary_time = now
                    self._send_periodic_summary()

                if self.cfg.position_sizing.method == "kelly" and self._tick_count % 24 == 0:
                    self._init_position_sizer()

        except Exception as e:
            self._log.error(f"WebSocket 處理異常: {e}")
            self._log.error(traceback.format_exc())

    def _append_kline(self, symbol: str, k: Dict[str, Any]):
        """追加 K 線到 IncrementalKlineCache（含缺口偵測）"""
        try:
            new_time = pd.to_datetime(k["t"], unit="ms", utc=True)

            cached = self._kline_cache.get_cached(symbol)
            if cached is not None and len(cached) > 0:
                last_time = cached.index[-1]
                expected_gap = pd.Timedelta(minutes=self._interval_minutes)
                actual_gap = new_time - last_time

                if actual_gap > expected_gap * 2:
                    self._log.warning(
                        f"⚠️  {symbol}: 偵測到 K 線缺口 "
                        f"({last_time.strftime('%H:%M')} → {new_time.strftime('%H:%M')}, "
                        f"差距 {actual_gap})，HTTP 補齊中..."
                    )
                    self._kline_cache.fill_gap(symbol, last_time)

            new_row = pd.DataFrame([{
                "open": float(k["o"]),
                "high": float(k["h"]),
                "low": float(k["l"]),
                "close": float(k["c"]),
                "volume": float(k["v"]),
                "close_time": pd.to_datetime(k["T"], unit="ms", utc=True),
            }], index=pd.DatetimeIndex([new_time], name="open_time"))

            self._kline_cache.append_bar(symbol, new_row)
        except Exception as e:
            self._log.error(f"❌ {symbol} K 線追加失敗: {e}")
            self._log.error(traceback.format_exc())

    # ══════════════════════════════════════════════════════════
    #  策略執行
    # ══════════════════════════════════════════════════════════

    def _run_strategy_for_symbol(self, symbol: str):
        """針對單一幣種執行策略"""
        if self._check_circuit_breaker():
            self._log.warning("⛔ 熔斷已觸發，跳過交易")
            return

        self._log.info(f"⚡️ 觸發策略: {symbol}")

        df = self._kline_cache.get_cached(symbol)
        if df is None or len(df) < 50:
            self._log.warning(
                f"⚠️  {symbol} 數據不足 ({len(df) if df is not None else 0}/50)，跳過策略"
            )
            return

        params = self.cfg.strategy.get_params(symbol)
        direction = self.cfg.direction

        try:
            sig = generate_signal(
                symbol=symbol,
                strategy_name=self.strategy_name,
                params=params,
                interval=self.interval,
                market_type=self.market_type,
                direction=direction,
                df=df,
            )
        except Exception as e:
            self._log.error(f"❌ {symbol} 信號生成失敗: {e}")
            self._log.error(traceback.format_exc())
            return

        # 使用 BaseRunner 的共享信號處理
        trade = self._process_signal(symbol, sig)

        # 發送信號摘要
        if trade:
            try:
                self.notifier.send_signal_summary(
                    [sig], mode=f"WS_{self.mode.upper()}", has_trade=True,
                )
            except Exception:
                pass

    # ══════════════════════════════════════════════════════════
    #  WebSocket 管理 + 心跳監控
    # ══════════════════════════════════════════════════════════

    def run(self):
        """啟動 WebSocket 連接並保持運行"""
        self.start_time = time.time()
        self._last_summary_time = time.time()

        alloc_str = ", ".join(f"{s}={w:.0%}" for s, w in self._weights.items())
        self._log.info("=" * 60)
        self._log.info(f"🚀 WebSocket Runner 啟動 [{self.mode.upper()}]")
        self._log.info(f"   策略: {self.strategy_name}")
        self._log.info(f"   訂閱: {', '.join(self.symbols)} @ {self.interval}")
        self._log.info(f"   倉位分配: {alloc_str}")
        self._log.info(f"   市場: {self.market_type}")
        self._log.info(f"   倉位計算: {self.cfg.position_sizing.method}")
        self._log.info(f"   交易資料庫: {'✅ SQLite' if self.trading_db else '❌ 未啟用'}")
        self._log.info(f"   Telegram: {'✅ 已啟用' if self.notifier.enabled else '❌ 未啟用'}")
        cache_info = []
        for sym in self.symbols:
            n = self._kline_cache.get_bar_count(sym)
            cache_info.append(f"{sym}={n}")
        self._log.info(f"   K 線快取: {', '.join(cache_info)} (IncrementalKlineCache ✅)")
        self._log.info(f"   心跳超時: {HEARTBEAT_TIMEOUT}s")
        self._log.info("=" * 60)

        try:
            self.notifier.send_startup(
                strategy=f"{self.strategy_name} (WebSocket v4.0)",
                symbols=self.symbols,
                interval=self.interval,
                mode=self.mode,
                weights=self._weights,
                market_type=self.market_type,
            )
        except Exception as e:
            self._log.warning(f"啟動通知發送失敗: {e}")

        try:
            from binance.websocket.um_futures.websocket_client import UMFuturesWebsocketClient

            logging.getLogger("binance").setLevel(logging.WARNING)

            self._ws_client = UMFuturesWebsocketClient(
                on_message=self._on_message_handler,
            )

            for symbol in self.symbols:
                self._ws_client.kline(symbol=symbol.lower(), interval=self.interval, id=1)
                self._log.info(f"📡 訂閱串流: {symbol.lower()}@kline_{self.interval}")

        except Exception as e:
            self._log.error(f"❌ WebSocket 連線失敗: {e}")
            self._log.error(traceback.format_exc())
            raise

        self.is_running = True
        self._last_ws_message_time = time.time()
        self._log.info("✅ WebSocket 已連線，等待 K 線事件...")

        try:
            while self.is_running:
                try:
                    time.sleep(1)

                    if self._last_ws_message_time > 0:
                        elapsed = time.time() - self._last_ws_message_time
                        if elapsed > HEARTBEAT_TIMEOUT:
                            self._log.warning(
                                f"⚠️  WebSocket 已 {elapsed:.0f}s 未收到消息，可能斷線"
                            )
                            try:
                                self.notifier.send_error(
                                    f"⚠️  WebSocket 可能斷線 ({elapsed:.0f}s 無消息)\n"
                                    f"等待自動重連..."
                                )
                            except Exception:
                                pass
                            self._last_ws_message_time = time.time()
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    self._log.error(f"主迴圈異常（自動恢復）: {e}")
                    self._log.error(traceback.format_exc())
                    time.sleep(5)

        except KeyboardInterrupt:
            self._log.info("⛔ 收到 KeyboardInterrupt，停止 WebSocket...")
        finally:
            if self._ws_client:
                try:
                    self._ws_client.stop()
                except Exception:
                    pass
            hours = (time.time() - self.start_time) / 3600 if self.start_time else 0
            try:
                self.notifier.send_shutdown(0, self.trade_count, hours)
            except Exception:
                pass
            self._log.info(
                f"👋 WebSocket Runner 已停止 (運行 {hours:.1f}h, 交易 {self.trade_count} 筆)"
            )

    def _on_message_handler(self, _, msg):
        """轉發消息到處理函數"""
        self._last_ws_message_time = time.time()
        try:
            if isinstance(msg, str):
                msg = json.loads(msg)
            if isinstance(msg, dict) and msg.get("e") == "kline":
                self._on_kline_event(msg)
        except Exception as e:
            self._log.error(f"WS Message Error: {e}")
            self._log.error(traceback.format_exc())
