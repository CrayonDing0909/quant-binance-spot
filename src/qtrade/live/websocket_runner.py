"""
WebSocket Runner — 輕量化事件驅動交易執行器 (v4.3)

v4.3: 修正 Binance 速率限制導致的斷線
    - 用單一 SUBSCRIBE 訊息批次訂閱全部串流（取代逐一 .kline() 呼叫）
    - 舊做法：19 個 .kline() = 19 個 SUBSCRIBE 訊息 <100ms → 超過 5 msg/s → 被斷線
    - 新做法：1 個 SUBSCRIBE 含全部串流 → 不觸發限制

v4.2: 改善連線穩定性
    - 新連線後 2 分鐘內用 30s 快速心跳偵測死連線（取代 300s）
    - 初始連線也設定 grace period，避免首次不必要的重連
    - _stop_ws_client 同時清除 on_open callback

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
from datetime import datetime, timezone
from typing import Dict, Any

from ..config import AppConfig
from ..utils.log import get_logger
from .signal_generator import generate_signal, SignalResult
from .kline_cache import IncrementalKlineCache
from .base_runner import BaseRunner

ws_logger = get_logger("ws_runner")

# 心跳超時（秒）— 超過此時間無 WS 消息即觸發重連
HEARTBEAT_TIMEOUT = 300

# 快速心跳偵測 — 新連線後短暫使用更嚴格的超時
# 原因：binance-connector 的 create_connection() 是同步的，連線成功後
#        Binance 每 250ms 就會推送 kline 更新，若 30s 內沒收到任何消息
#        幾乎可以確定連線已死（伺服器主動斷開或訂閱失敗）
FAST_HEARTBEAT_TIMEOUT = 30     # 新連線後的快速偵測超時（秒）
FAST_HEARTBEAT_WINDOW = 120     # 快速偵測窗口長度（秒）

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
    基於 WebSocket 的輕量化執行器 (v4.3)

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
        self._reconnect_grace_until: float = 0.0  # 重連成功後的寬限期（忽略舊 callback 殘留）
        self._fast_heartbeat_until: float = 0.0   # 快速心跳偵測窗口結束時間

        # 最新信號快取（per-symbol），供 _save_last_signals 寫 JSON
        self._latest_signals: Dict[str, SignalResult] = {}

        # K 線快取（BaseRunner 的 _kline_cache 由子類設定）
        cache_dir = cfg.get_report_dir("live") / "kline_cache"
        self._kline_cache = IncrementalKlineCache(
            cache_dir=cache_dir,
            interval=self.interval,
            seed_bars=300,
            market_type=self.market_type,
        )

        # ── Multi-TF 輔助 K 線快取 (Phase 4A) ──
        # auxiliary_intervals 來自 MarketConfig（例如 ["4h", "1d"]）
        self._auxiliary_intervals: list[str] = getattr(cfg.market, "auxiliary_intervals", [])
        self._aux_kline_caches: Dict[str, IncrementalKlineCache] = {}
        self._aux_last_kline_ts: Dict[str, int] = {}  # key: f"{symbol}_{interval}"
        for aux_iv in self._auxiliary_intervals:
            if aux_iv != self.interval:
                aux_cache_dir = cache_dir / f"aux_{aux_iv}"
                self._aux_kline_caches[aux_iv] = IncrementalKlineCache(
                    cache_dir=aux_cache_dir,
                    interval=aux_iv,
                    seed_bars=100,  # 輔助 TF 只需較少 bars
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
        """使用 IncrementalKlineCache 預熱 K 線（含 auxiliary TFs）"""
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

        # 預熱 auxiliary TF caches
        for aux_iv, aux_cache in self._aux_kline_caches.items():
            for symbol in self.symbols:
                try:
                    df = aux_cache.get_klines(symbol)
                    n = len(df) if df is not None else 0
                    self._log.info(f"  📊 {symbol}@{aux_iv}: {n} bars (auxiliary)")
                except Exception as e:
                    self._log.debug(f"  {symbol}@{aux_iv}: aux cache failed: {e}")

    def _on_kline_event(self, msg: Dict[str, Any]):
        """WebSocket K 線事件回調（支援主 TF 和 auxiliary TF）"""
        try:
            if "k" not in msg:
                return

            k = msg["k"]
            symbol = k["s"]
            is_closed = k["x"]
            close_price = float(k["c"])
            event_interval = k.get("i", self.interval)

            if is_closed:
                ts = k["t"]

                # ── Auxiliary TF：只更新快取，不觸發策略 ──
                if event_interval != self.interval and event_interval in self._aux_kline_caches:
                    dedup_key = f"{symbol}_{event_interval}"
                    if self._aux_last_kline_ts.get(dedup_key) == ts:
                        return
                    self._aux_last_kline_ts[dedup_key] = ts
                    self._append_aux_kline(symbol, event_interval, k)
                    self._log.debug(
                        f"📊 {symbol}@{event_interval} aux K 線收盤: ${close_price:,.2f}"
                    )
                    return

                # ── 主 TF：正常處理 ──
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

    def _append_aux_kline(self, symbol: str, interval: str, k: Dict[str, Any]):
        """追加 auxiliary TF K 線到對應快取"""
        try:
            aux_cache = self._aux_kline_caches.get(interval)
            if aux_cache is None:
                return

            new_time = pd.to_datetime(k["t"], unit="ms", utc=True)
            new_row = pd.DataFrame([{
                "open": float(k["o"]),
                "high": float(k["h"]),
                "low": float(k["l"]),
                "close": float(k["c"]),
                "volume": float(k["v"]),
                "close_time": pd.to_datetime(k["T"], unit="ms", utc=True),
            }], index=pd.DatetimeIndex([new_time], name="open_time"))

            aux_cache.append_bar(symbol, new_row)
        except Exception as e:
            self._log.debug(f"  {symbol}@{interval} aux kline append failed: {e}")

    # ══════════════════════════════════════════════════════════
    #  策略執行
    # ══════════════════════════════════════════════════════════

    def _run_strategy_for_symbol(self, symbol: str):
        """針對單一幣種執行策略（含 multi-TF + derivatives 注入）"""
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

        # ── Phase 4A: 組裝 auxiliary_data (multi-TF) ──
        auxiliary_data: Dict[str, "pd.DataFrame"] = {}
        for aux_iv, aux_cache in self._aux_kline_caches.items():
            aux_df = aux_cache.get_cached(symbol)
            if aux_df is not None and len(aux_df) > 0:
                auxiliary_data[aux_iv] = aux_df

        # ── Phase 4A: 刷新 derivatives 快取（BaseRunner） ──
        self._maybe_refresh_derivatives_cache()
        derivatives_data = getattr(self, "_derivatives_cache", None) or {}

        # 將 auxiliary / derivatives 塞入 params，供 generate_signal → StrategyContext 使用
        if auxiliary_data:
            params = {**params, "_auxiliary_data": auxiliary_data}
        if derivatives_data:
            params = {**params, "_derivatives_data": derivatives_data}

        # ── Regime Gate: inject BTC reference data for portfolio-level scaling ──
        regime_gate_cfg = getattr(self.cfg, '_regime_gate_cfg', None)
        if regime_gate_cfg and regime_gate_cfg.get("enabled", False):
            ref_sym = regime_gate_cfg.get("reference_symbol", "BTCUSDT")
            ref_df = self._kline_cache.get_cached(ref_sym)
            if ref_df is not None and len(ref_df) > 50:
                params = {**params, "_regime_gate": regime_gate_cfg, "_regime_gate_ref_df": ref_df}

        try:
            sig = generate_signal(
                symbol=symbol,
                strategy_name=sym_strategy,
                params=params,
                interval=self.interval,
                market_type=self.market_type,
                direction=direction,
                df=df,
                overlay_cfg=getattr(self.cfg, '_overlay_cfg', None),
            )
        except Exception as e:
            self._log.error(f"❌ {symbol} 信號生成失敗: {e}")
            self._log.error(traceback.format_exc())
            return

        # ── IC Gate: apply live alpha decay scaling before execution ──
        sig = self._apply_ic_gate(symbol, sig)

        # 使用 BaseRunner 的共享信號處理
        trade = self._process_signal(symbol, sig)

        # 更新信號快取並寫 JSON（供統一 TG Bot 讀取）
        self._latest_signals[symbol] = sig
        self._save_last_signals()

        # 發送信號摘要
        if trade:
            try:
                self.notifier.send_signal_summary(
                    [sig], mode=f"WS_{self.mode.upper()}", has_trade=True,
                )
            except Exception:
                pass

    def _save_last_signals(self) -> None:
        """保存最新信號快照到 JSON，供統一 Telegram Bot (/signals) 讀取"""
        try:
            sig_path = self.cfg.get_report_dir("live") / "last_signals.json"
            sig_path.parent.mkdir(parents=True, exist_ok=True)

            serializable = [
                sig.to_dict() for sig in self._latest_signals.values()
            ]

            payload = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "mode": self.mode,
                "signals": serializable,
            }

            with open(sig_path, "w") as f:
                json.dump(payload, f, indent=2, default=str)
        except Exception as e:
            self._log.debug(f"  保存信號快照失敗: {e}")

    def _generate_startup_signals(self) -> None:
        """
        啟動時用快取 K 線為所有 symbol 生成一次信號快照。

        目的：消除 WebSocket 冷啟動盲區。
        Runner 重啟後最長需等到下一根 K 線收盤（最多 59 分鐘）才會寫
        last_signals.json。在此期間 Telegram Bot /signals 會讀到上一個
        runner 寫的過期檔案（可能含舊的 indicator 格式或過時的價格）。

        此方法在 WebSocket 連線成功後、主迴圈之前呼叫，立即刷新信號快照。
        注意：這裡只生成信號 + 寫 JSON，**不執行交易**（避免重啟即下單）。
        """
        self._log.info("📡 啟動信號快照：用快取 K 線生成初始信號...")
        generated = 0

        for symbol in self.symbols:
            try:
                df = self._kline_cache.get_cached(symbol)
                if df is None or len(df) < 50:
                    self._log.debug(
                        f"  {symbol}: 數據不足 ({len(df) if df is not None else 0}/50)，跳過"
                    )
                    continue

                sym_strategy, params = self._get_strategy_for_symbol(symbol)
                direction = self.cfg.direction

                # 組裝 auxiliary + derivatives（與 _run_strategy_for_symbol 一致）
                auxiliary_data: Dict[str, "pd.DataFrame"] = {}
                for aux_iv, aux_cache in self._aux_kline_caches.items():
                    aux_df = aux_cache.get_cached(symbol)
                    if aux_df is not None and len(aux_df) > 0:
                        auxiliary_data[aux_iv] = aux_df

                derivatives_data = getattr(self, "_derivatives_cache", None) or {}

                if auxiliary_data:
                    params = {**params, "_auxiliary_data": auxiliary_data}
                if derivatives_data:
                    params = {**params, "_derivatives_data": derivatives_data}

                sig = generate_signal(
                    symbol=symbol,
                    strategy_name=sym_strategy,
                    params=params,
                    interval=self.interval,
                    market_type=self.market_type,
                    direction=direction,
                    df=df,
                    overlay_cfg=getattr(self.cfg, '_overlay_cfg', None),
                )

                self._latest_signals[symbol] = sig
                generated += 1

            except Exception as e:
                self._log.warning(f"  {symbol} 啟動信號生成失敗: {e}")

        if generated > 0:
            self._save_last_signals()
            self._log.info(
                f"✅ 啟動信號快照完成: {generated}/{len(self.symbols)} symbols"
            )
        else:
            self._log.warning("⚠️  啟動信號快照: 無法為任何 symbol 生成信號")

    # ══════════════════════════════════════════════════════════
    #  WebSocket 管理 + 心跳監控 + 自動重連
    # ══════════════════════════════════════════════════════════

    def _create_ws_client(self):
        """
        建立 WS client 並訂閱所有 symbol（供初次連線與重連共用）

        ★ 關鍵修正 (v4.3)：用單一 SUBSCRIBE 訊息批次訂閱全部串流。
        舊做法（逐一 .kline()）會在 <100ms 內送出 N 個 SUBSCRIBE 訊息，
        超過 Binance WebSocket 5 msg/s 速率限制 → 被服務器強制斷線。
        """
        from binance.websocket.um_futures.websocket_client import UMFuturesWebsocketClient

        logging.getLogger("binance").setLevel(logging.WARNING)

        client = UMFuturesWebsocketClient(
            on_message=self._on_message_handler,
            on_close=self._on_ws_close,
            on_error=self._on_ws_error,
        )

        # ★ 批次訂閱：一個 SUBSCRIBE 訊息包含全部串流（避免觸發 5 msg/s 限制）
        streams = [
            f"{sym.lower()}@kline_{self.interval}" for sym in self.symbols
        ]
        # ★ Phase 4A: 加入 auxiliary TF 串流（僅訂閱，不觸發策略）
        for aux_iv in self._auxiliary_intervals:
            if aux_iv != self.interval:
                for sym in self.symbols:
                    streams.append(f"{sym.lower()}@kline_{aux_iv}")
        client.subscribe(streams, id=1)
        for s in streams:
            self._log.info(f"📡 訂閱串流: {s}")
        self._log.info(
            f"✅ 已批次訂閱 {len(streams)} 個串流（單一 SUBSCRIBE 訊息）"
        )

        return client

    def _stop_ws_client(self):
        """
        安全關閉舊的 WS client。

        關鍵：先解除舊 client 的 on_close/on_error/on_message callback，
        防止舊 client 的清理流程觸發新 client 的重連迴圈。
        再用 daemon thread + timeout 防止 join() 卡住主迴圈。
        """
        self._ws_ready = False
        self._subscriptions_ready = False
        old_client = self._ws_client
        self._ws_client = None
        if old_client is None:
            return

        # ★ 關鍵：解除舊 client 的所有 callback
        # BinanceSocketManager.read_data() 在收到 CLOSE frame 時會呼叫 self.on_close，
        # 若不解除，舊 client 的關閉事件會觸發 _on_ws_close → _ws_needs_reconnect = True
        # → 造成新 client 被誤判為斷線 → 無限重連迴圈
        try:
            sm = old_client.socket_manager
            sm.on_close = None
            sm.on_error = None
            sm.on_message = None
            sm.on_open = None
        except Exception:
            pass

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

                # 設定 30s 寬限期：忽略舊 client 殘留的 on_close/on_error callback
                self._reconnect_grace_until = time.time() + 30
                # 啟用快速心跳偵測：新連線後若 30s 無消息則立即重連
                self._fast_heartbeat_until = time.time() + FAST_HEARTBEAT_WINDOW

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
        """WS 連線關閉回調 — 標記需要重連（尊重 grace period）"""
        if time.time() < self._reconnect_grace_until:
            self._log.info(
                "ℹ️  WebSocket on_close 觸發，但在重連寬限期內，忽略（可能是舊 client 殘留）"
            )
            return
        self._log.warning("⚠️  WebSocket on_close 觸發，標記需要重連")
        self._ws_needs_reconnect = True

    def _on_ws_error(self, _, error):
        """WS 錯誤回調 — 標記需要重連（尊重 grace period）"""
        if time.time() < self._reconnect_grace_until:
            self._log.info(
                f"ℹ️  WebSocket on_error ({error})，但在重連寬限期內，忽略"
            )
            return
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
        self._log.info(f"   心跳超時: {HEARTBEAT_TIMEOUT}s (新連線快速偵測: {FAST_HEARTBEAT_TIMEOUT}s × {FAST_HEARTBEAT_WINDOW}s 窗口)")
        self._log.info(f"   重連退避: {RECONNECT_BASE_DELAY}s ~ {RECONNECT_MAX_DELAY}s")
        self._log.info("=" * 60)

        try:
            self.notifier.send_startup(
                strategy=f"{self.strategy_name} (WebSocket v4.3)",
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
            # ★ 首次連線也設 grace period（Binance 有時會立即發 CLOSE frame）
            self._reconnect_grace_until = time.time() + 30
            # ★ 啟用快速心跳偵測：若 30s 內無消息，立即重連（不等 300s）
            self._fast_heartbeat_until = time.time() + FAST_HEARTBEAT_WINDOW
        except Exception as e:
            self._log.error(f"❌ WebSocket 初始連線失敗: {e}")
            self._log.error(traceback.format_exc())
            raise

        self.is_running = True
        self._last_ws_message_time = time.time()
        self._log.info("✅ WebSocket 已連線，等待 K 線事件...")

        # ★ 啟動信號快照 — 用快取 K 線立即生成一次信號，消除冷啟動盲區
        # 重啟後 last_signals.json 最長可能有 59 分鐘的陳舊資料，
        # 這裡立即刷新，讓 Telegram Bot /signals 立刻顯示最新狀態
        self._generate_startup_signals()

        # Phase 4B: 啟動衍生品 API 後台輪詢線程
        self._start_derivatives_bg_refresh()

        try:
            while self.is_running:
                try:
                    time.sleep(1)
                    self._last_main_loop_heartbeat = time.time()

                    # 檢查是否需要重連（on_close/on_error 觸發 或 心跳超時）
                    needs_reconnect = self._ws_needs_reconnect
                    if not needs_reconnect and self._last_ws_message_time > 0:
                        now = time.time()
                        elapsed = now - self._last_ws_message_time

                        # 新連線後 2 分鐘內用 30s 快速偵測，之後用 300s 常規偵測
                        in_fast_window = now < self._fast_heartbeat_until
                        timeout = FAST_HEARTBEAT_TIMEOUT if in_fast_window else HEARTBEAT_TIMEOUT

                        if elapsed > timeout:
                            needs_reconnect = True
                            mode_label = "快速偵測" if in_fast_window else "心跳超時"
                            self._log.warning(
                                f"⚠️  WebSocket 已 {elapsed:.0f}s 未收到消息"
                                f"（{mode_label}, 閾值={timeout}s），觸發重連"
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
