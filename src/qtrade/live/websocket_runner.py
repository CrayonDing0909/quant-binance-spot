"""
WebSocket Runner — 輕量化事件驅動交易執行器 (v4.1)

v4.1: 新增自動重連機制
    - WS 斷線後自動重建 client + 重新訂閱
    - 指數退避重連（10s → 20s → 40s ... 最大 300s）
    - on_close / on_error callback 主動偵測斷線
    - 重連計數暴露給 watchdog / TG

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
import threading
import pandas as pd
from typing import Dict, Any

from ..config import AppConfig
from ..utils.log import get_logger
from .signal_generator import generate_signal
from .kline_cache import IncrementalKlineCache
from .base_runner import BaseRunner

ws_logger = get_logger("ws_runner")

# 心跳超時（秒）— 超過此時間無 WS 消息即觸發重連
HEARTBEAT_TIMEOUT = 300

# 重連參數
RECONNECT_BASE_DELAY = 10       # 首次重連等待（秒）
RECONNECT_MAX_DELAY = 300       # 最大重連等待（秒）
RECONNECT_BACKOFF_FACTOR = 2    # 指數退避乘數
RECONNECT_CONSECUTIVE_FAIL_ALERT = 5  # 連續失敗 N 次後強制 TG 告警（無視 cooldown）

# interval → 分鐘 對照表
INTERVAL_MINUTES = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "6h": 360, "8h": 480,
    "12h": 720, "1d": 1440,
}


class WebSocketRunner(BaseRunner):
    """
    基於 WebSocket 的輕量化執行器 (v4.1)

    繼承 BaseRunner 取得所有安全機制，
    本類只負責 WS 連線管理、K 線事件驅動和自動重連。
    """

    def __init__(self, cfg: AppConfig, broker, mode: str = "paper", notifier=None):
        super().__init__(cfg, broker, mode, notifier)

        self._tick_count = 0
        self._started_at: float = 0.0
        self._last_ws_message_time: float = 0.0
        self._last_kline_event_time: float = 0.0
        self._last_main_loop_heartbeat: float = 0.0
        self._ws_ready: bool = False
        self._subscriptions_ready: bool = False
        self._ws_client = None
        self._last_kline_ts: Dict[str, int] = {}
        self._last_summary_time: float = 0.0
        self._interval_minutes = INTERVAL_MINUTES.get(self.interval, 60)
        self._ws_disconnect_alert_cooldown_sec: float = 1800.0
        self._last_ws_disconnect_alert_time: float = 0.0

        # 重連狀態
        self._reconnect_count: int = 0
        self._consecutive_failures: int = 0
        self._last_reconnect_time: float = 0.0
        self._reconnect_delay: float = RECONNECT_BASE_DELAY
        self._ws_needs_reconnect: bool = False
        self._reconnect_lock = threading.Lock()

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
                self._last_kline_event_time = time.time()

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

        # Ensemble 路由：取得 symbol 專屬策略名與參數
        sym_strategy, params = self._get_strategy_for_symbol(symbol)
        direction = self.cfg.direction

        try:
            sig = generate_signal(
                symbol=symbol,
                strategy_name=sym_strategy,
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
    #  WebSocket 管理 + 心跳監控 + 自動重連
    # ══════════════════════════════════════════════════════════

    def _create_ws_client(self):
        """建立 WS client 並訂閱所有 symbol（供初次連線與重連共用）"""
        from binance.websocket.um_futures.websocket_client import UMFuturesWebsocketClient

        logging.getLogger("binance").setLevel(logging.WARNING)

        client = UMFuturesWebsocketClient(
            on_message=self._on_message_handler,
            on_close=self._on_ws_close,
            on_error=self._on_ws_error,
        )

        for symbol in self.symbols:
            client.kline(symbol=symbol.lower(), interval=self.interval, id=1)
            self._log.info(f"📡 訂閱串流: {symbol.lower()}@kline_{self.interval}")

        return client

    def _stop_ws_client(self):
        """
        安全關閉舊的 WS client。

        使用 daemon thread + timeout 防止 join() 卡住主迴圈：
        BinanceSocketManager.stop() 內部呼叫 thread.join()，
        若底層 socket 處於半死狀態，join() 可能永遠不返回。
        """
        self._ws_ready = False
        self._subscriptions_ready = False
        old_client = self._ws_client
        self._ws_client = None
        if old_client is None:
            return

        def _do_stop():
            try:
                old_client.stop()
            except Exception:
                pass

        stopper = threading.Thread(target=_do_stop, daemon=True)
        stopper.start()
        stopper.join(timeout=5)
        if stopper.is_alive():
            self._log.warning(
                "⚠️  舊 WS client stop 超時（5s），已放棄等待（daemon thread 會自行回收）"
            )

    def _reconnect_ws(self) -> bool:
        """
        嘗試重建 WS 連線 + 重新訂閱。

        返回 True 表示重連成功，False 表示失敗（將在下次主迴圈迭代重試）。
        重連使用指數退避：10s → 20s → 40s ... 最大 300s，成功後重置。
        """
        with self._reconnect_lock:
            now = time.time()

            # 退避保護：距離上次重連嘗試不足 delay 秒則跳過
            if now - self._last_reconnect_time < self._reconnect_delay:
                return False

            self._last_reconnect_time = now
            self._reconnect_count += 1
            attempt = self._reconnect_count

            self._log.warning(
                f"🔄 WebSocket 重連中... (第 {attempt} 次, "
                f"連續失敗={self._consecutive_failures}, "
                f"delay={self._reconnect_delay:.0f}s)"
            )

            # 1) 停掉舊 client（有 5s timeout 防 hang）
            self._stop_ws_client()

            # 2) 建新 client
            try:
                self._ws_client = self._create_ws_client()
                self._ws_ready = True
                self._subscriptions_ready = True
                self._last_ws_message_time = time.time()
                self._ws_needs_reconnect = False

                # 重連成功 → 重置退避和連續失敗計數
                self._reconnect_delay = RECONNECT_BASE_DELAY
                self._consecutive_failures = 0
                self._log.info(
                    f"✅ WebSocket 重連成功 (第 {attempt} 次)"
                )

                # TG 通知
                try:
                    self.notifier.send(
                        f"🔄 <b>WebSocket 重連成功</b>\n"
                        f"第 {attempt} 次重連，已恢復正常。"
                    )
                except Exception:
                    pass
                return True

            except Exception as e:
                self._consecutive_failures += 1
                self._log.error(f"❌ WebSocket 重連失敗 (第 {attempt} 次): {e}")
                self._log.error(traceback.format_exc())

                # 退避加倍
                self._reconnect_delay = min(
                    self._reconnect_delay * RECONNECT_BACKOFF_FACTOR,
                    RECONNECT_MAX_DELAY,
                )

                # 連續失敗達門檻 → 強制 TG 告警（無視 cooldown）
                force_alert = (
                    self._consecutive_failures >= RECONNECT_CONSECUTIVE_FAIL_ALERT
                    and self._consecutive_failures % RECONNECT_CONSECUTIVE_FAIL_ALERT == 0
                )
                should_alert = force_alert or (
                    now - self._last_ws_disconnect_alert_time >= self._ws_disconnect_alert_cooldown_sec
                )

                if should_alert:
                    try:
                        self.notifier.send_error(
                            f"❌ WebSocket 重連失敗 (第 {attempt} 次, "
                            f"連續失敗 {self._consecutive_failures})\n"
                            f"錯誤: {e}\n"
                            f"下次重試: {self._reconnect_delay:.0f}s 後"
                        )
                        self._last_ws_disconnect_alert_time = now
                    except Exception:
                        pass
                return False

    def _on_ws_close(self, _):
        """WS 連線關閉回調 — 標記需要重連"""
        self._log.warning("⚠️  WebSocket on_close 觸發，標記需要重連")
        self._ws_needs_reconnect = True

    def _on_ws_error(self, _, error):
        """WS 錯誤回調 — 標記需要重連"""
        self._log.error(f"⚠️  WebSocket on_error: {error}")
        self._ws_needs_reconnect = True

    def run(self):
        """啟動 WebSocket 連接並保持運行"""
        self.start_time = time.time()
        self._started_at = self.start_time
        self._last_summary_time = time.time()
        self._last_main_loop_heartbeat = time.time()
        self._last_kline_event_time = 0.0
        self._ws_ready = False
        self._subscriptions_ready = False

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
        self._log.info(f"   重連退避: {RECONNECT_BASE_DELAY}s ~ {RECONNECT_MAX_DELAY}s")
        self._log.info("=" * 60)

        try:
            self.notifier.send_startup(
                strategy=f"{self.strategy_name} (WebSocket v4.1)",
                symbols=self.symbols,
                interval=self.interval,
                mode=self.mode,
                weights=self._weights,
                market_type=self.market_type,
            )
        except Exception as e:
            self._log.warning(f"啟動通知發送失敗: {e}")

        # 首次連線
        try:
            self._ws_client = self._create_ws_client()
            self._ws_ready = True
            self._subscriptions_ready = True
        except Exception as e:
            self._log.error(f"❌ WebSocket 初始連線失敗: {e}")
            self._log.error(traceback.format_exc())
            raise

        self.is_running = True
        self._last_ws_message_time = time.time()
        self._log.info("✅ WebSocket 已連線，等待 K 線事件...")

        try:
            while self.is_running:
                try:
                    time.sleep(1)
                    self._last_main_loop_heartbeat = time.time()

                    # 檢查是否需要重連（on_close/on_error 觸發 或 心跳超時）
                    needs_reconnect = self._ws_needs_reconnect
                    if not needs_reconnect and self._last_ws_message_time > 0:
                        elapsed = time.time() - self._last_ws_message_time
                        if elapsed > HEARTBEAT_TIMEOUT:
                            needs_reconnect = True
                            self._log.warning(
                                f"⚠️  WebSocket 已 {elapsed:.0f}s 未收到消息，觸發重連"
                            )

                    if needs_reconnect:
                        self._reconnect_ws()

                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    self._log.error(f"主迴圈異常（自動恢復）: {e}")
                    self._log.error(traceback.format_exc())
                    time.sleep(5)

        except KeyboardInterrupt:
            self._log.info("⛔ 收到 KeyboardInterrupt，停止 WebSocket...")
        finally:
            self._stop_ws_client()
            hours = (time.time() - self.start_time) / 3600 if self.start_time else 0
            try:
                self.notifier.send_shutdown(0, self.trade_count, hours)
            except Exception:
                pass
            self._log.info(
                f"👋 WebSocket Runner 已停止 "
                f"(運行 {hours:.1f}h, 交易 {self.trade_count} 筆, "
                f"重連 {self._reconnect_count} 次)"
            )

    def _on_message_handler(self, _, msg):
        """轉發消息到處理函數"""
        self._last_ws_message_time = time.time()
        self._last_main_loop_heartbeat = time.time()
        try:
            if isinstance(msg, str):
                msg = json.loads(msg)
            if isinstance(msg, dict) and msg.get("e") == "kline":
                self._on_kline_event(msg)
        except Exception as e:
            self._log.error(f"WS Message Error: {e}")
            self._log.error(traceback.format_exc())
