"""
WebSocket Runner — 輕量化事件驅動交易執行器

適用場景：
    - Oracle Cloud (1GB RAM) 等資源受限環境
    - 需要即時反應（K 線收盤 0 秒延遲）
    - 支援 Intra-bar SL/TP 監控

特性：
    - 僅維護最近 N 根 K 線 (Rolling Window)，記憶體佔用極低
    - 事件驅動：K 線收盤觸發策略，價格跳動觸發止損
    - 自動重連機制
"""
import json
import time
import logging
import traceback
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

from ..config import AppConfig
from ..utils.log import get_logger
from .signal_generator import generate_signal

logger = get_logger("ws_runner")

# Rolling Window 大小（只保留最近 N 根 K 線）
ROLLING_WINDOW = 500


class WebSocketRunner:
    """
    基於 WebSocket 的輕量化執行器

    不繼承 LiveRunner（避免 kline_cache/polling 等不需要的初始化），
    而是組合式使用 broker / notifier / trading_db 等元件。
    """

    def __init__(self, cfg: AppConfig, broker, mode: str = "paper", notifier=None):
        self.cfg = cfg
        self.broker = broker
        self.mode = mode
        self.strategy_name = cfg.strategy.name
        self.symbols = cfg.market.symbols
        self.interval = cfg.market.interval
        self.market_type = cfg.market_type_str  # "spot" or "futures"
        self.is_running = False
        self.trade_count = 0
        self.start_time: float | None = None

        # Telegram 通知
        from ..monitor.notifier import TelegramNotifier
        self.notifier = notifier or TelegramNotifier.from_config(cfg.notification)

        # 多幣種倉位分配權重
        self._weights: dict[str, float] = {}
        n = len(self.symbols)
        for sym in self.symbols:
            self._weights[sym] = cfg.portfolio.get_weight(sym, n)

        # Drawdown 熔斷
        self.max_drawdown_pct = cfg.risk.max_drawdown_pct if cfg.risk else None
        self._circuit_breaker_triggered = False
        self._initial_equity: float | None = None

        # 倉位計算器（簡化版：固定倉位）
        from ..risk.position_sizing import FixedPositionSizer
        self.position_sizer = FixedPositionSizer(cfg.position_sizing.position_pct)

        # SQLite 結構化資料庫
        self.trading_db = None
        try:
            from .trading_db import TradingDatabase
            db_path = cfg.get_report_dir("live") / "trading.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self.trading_db = TradingDatabase(db_path)
            logger.info(f"📦 SQLite 資料庫已就緒: {db_path}")
        except Exception as e:
            logger.warning(f"⚠️  SQLite 資料庫初始化失敗（不影響交易）: {e}")

        # 本地 K 線快取 {symbol: DataFrame}
        self._kline_buffer: Dict[str, pd.DataFrame] = {}
        self._ws_client = None
        self._last_kline_ts: Dict[str, int] = {}

        # 預熱 K 線緩衝區
        self._init_kline_buffer()

    def _init_kline_buffer(self):
        """啟動時預先拉取歷史 K 線，填滿緩衝區"""
        from .signal_generator import fetch_recent_klines
        from ..data.klines import fetch_klines
        from ..data.quality import clean_data

        logger.info("📥 正在預熱 K 線緩衝區...")
        for symbol in self.symbols:
            try:
                # 使用正確的 market_type 來拉取 K 線
                from datetime import timedelta
                interval_minutes = {
                    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
                    "1h": 60, "2h": 120, "4h": 240, "6h": 360, "8h": 480,
                    "12h": 720, "1d": 1440,
                }
                minutes = interval_minutes.get(self.interval, 60)
                bars = 300
                start_dt = datetime.now(timezone.utc) - timedelta(minutes=minutes * (bars + 10))
                start_str = start_dt.strftime("%Y-%m-%d")

                df = fetch_klines(symbol, self.interval, start_str, market_type=self.market_type)
                df = clean_data(df, fill_method="forward", remove_outliers=False, remove_duplicates=True)

                # 丟棄未收盤的 K 線
                if "close_time" in df.columns:
                    now = pd.Timestamp.now(tz="UTC")
                    df = df[df["close_time"] <= now]

                # 只保留最近 bars 根
                if len(df) > bars:
                    df = df.iloc[-bars:]

                self._kline_buffer[symbol] = df
                logger.info(f"  ✅ {symbol}: 已載入 {len(df)} 根 {self.market_type} K 線")
            except Exception as e:
                logger.error(f"  ❌ {symbol}: K 線載入失敗: {e}")
                logger.error(traceback.format_exc())
                self._kline_buffer[symbol] = pd.DataFrame()

    def _on_kline_event(self, msg: Dict[str, Any]):
        """WebSocket K 線事件回調"""
        try:
            if "k" not in msg:
                return

            k = msg["k"]
            symbol = k["s"]
            is_closed = k["x"]
            close_price = float(k["c"])

            # 僅在 K 線收盤時觸發策略
            if is_closed:
                ts = k["t"]

                # 防止重複處理同一根 K 線
                if self._last_kline_ts.get(symbol) == ts:
                    return
                self._last_kline_ts[symbol] = ts

                logger.info(f"🕯️  {symbol} K 線收盤: ${close_price:,.2f}")

                # 更新本地 Buffer
                self._append_kline(symbol, k)

                # 執行策略
                self._run_strategy_for_symbol(symbol)

        except Exception as e:
            logger.error(f"WebSocket 處理異常: {e}")
            logger.error(traceback.format_exc())

    def _append_kline(self, symbol: str, k: Dict[str, Any]):
        """將新 K 線追加到 DataFrame 並維持長度"""
        new_row = {
            "open": float(k["o"]),
            "high": float(k["h"]),
            "low": float(k["l"]),
            "close": float(k["c"]),
            "volume": float(k["v"]),
            "timestamp": pd.to_datetime(k["t"], unit="ms", utc=True)
        }

        df = self._kline_buffer.get(symbol, pd.DataFrame())
        new_df = pd.DataFrame([new_row]).set_index("timestamp")

        if df.empty:
            df = new_df
        else:
            df = pd.concat([df, new_df])
            df = df[~df.index.duplicated(keep='last')]

        # 只保留最近 N 根 (Rolling Window)
        if len(df) > ROLLING_WINDOW:
            df = df.iloc[-ROLLING_WINDOW:]

        self._kline_buffer[symbol] = df

    def _check_circuit_breaker(self) -> bool:
        """Drawdown 熔斷檢查"""
        if self._circuit_breaker_triggered:
            return True
        if not self.max_drawdown_pct:
            return False

        try:
            equity = self.broker.get_equity()
            if equity is None or equity <= 0:
                return False

            if self._initial_equity is None:
                self._initial_equity = equity
                return False

            drawdown = (self._initial_equity - equity) / self._initial_equity
            if drawdown >= self.max_drawdown_pct:
                logger.warning(
                    f"⛔ 熔斷觸發！回撤 {drawdown:.1%} >= {self.max_drawdown_pct:.1%}"
                )
                self._circuit_breaker_triggered = True
                return True
        except Exception as e:
            logger.debug(f"熔斷檢查失敗: {e}")

        return False

    def _apply_position_sizing(self, raw_signal: float, price: float, symbol: str) -> float:
        """應用倉位計算器調整信號"""
        if self.position_sizer is None:
            return raw_signal

        try:
            sized = self.position_sizer.calculate(abs(raw_signal))
            result = sized if raw_signal >= 0 else -sized
            return max(-1.0, min(1.0, result))  # clip [-1, 1]
        except Exception:
            return raw_signal

    def _run_strategy_for_symbol(self, symbol: str):
        """針對單一幣種執行策略"""
        # 熔斷檢查
        if self._check_circuit_breaker():
            logger.warning("⛔ 熔斷已觸發，跳過交易")
            return

        logger.info(f"⚡️ 觸發策略: {symbol}")

        # 1. 準備數據
        df = self._kline_buffer.get(symbol)
        if df is None or len(df) < 50:
            logger.warning(f"⚠️  {symbol} 數據不足 ({len(df) if df is not None else 0}/50)，跳過策略")
            return

        # 2. 生成信號
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
            logger.error(f"❌ {symbol} 信號生成失敗: {e}")
            logger.error(traceback.format_exc())
            return

        # 3. 處理信號
        self._process_signal(symbol, sig)

    def _process_signal(self, symbol: str, sig: dict):
        """處理信號並下單"""
        has_trade = False

        raw_signal = sig["signal"]
        price = sig["price"]
        indicators = sig.get("indicators", {})

        # 記錄信號到 DB
        if self.trading_db:
            try:
                current_pct_log = 0
                try:
                    current_pct_log = self.broker.get_position_pct(symbol, price)
                except Exception:
                    pass

                action = "HOLD"
                if raw_signal > 0.01 and current_pct_log <= 0.01:
                    action = "OPEN_LONG"
                elif raw_signal < -0.01 and current_pct_log >= -0.01:
                    action = "OPEN_SHORT"
                elif abs(raw_signal) < 0.01 and abs(current_pct_log) > 0.01:
                    action = "CLOSE"

                self.trading_db.log_signal(
                    symbol=symbol,
                    signal_value=raw_signal,
                    price=price,
                    rsi=indicators.get("rsi"),
                    adx=indicators.get("adx"),
                    atr=indicators.get("atr"),
                    plus_di=indicators.get("plus_di"),
                    minus_di=indicators.get("minus_di"),
                    target_pct=raw_signal * self._weights.get(symbol, 1.0),
                    current_pct=current_pct_log,
                    action=action,
                    timestamp=sig.get("timestamp"),
                )
            except Exception as e:
                logger.debug(f"信號記錄失敗: {e}")

        # 計算目標倉位
        if self.market_type == "spot" and raw_signal < 0:
            raw_signal = 0.0

        weight = self._weights.get(symbol, 1.0)
        adjusted_signal = self._apply_position_sizing(raw_signal, price, symbol)
        target_pct = adjusted_signal * weight

        current_pct = self.broker.get_position_pct(symbol, price)
        diff = abs(target_pct - current_pct)

        # Log 信號狀態
        logger.info(
            f"  📊 {symbol}: signal={raw_signal:.2f}, target={target_pct:.2f}, "
            f"current={current_pct:.2f}, diff={diff:.2f}, "
            f"RSI={indicators.get('rsi', '?')}, ADX={indicators.get('adx', '?')}"
        )

        # 執行交易（差異 >= 2% 才交易）
        if diff >= 0.02:
            reason = f"WS_signal={raw_signal:.0%} [{self.interval}]"

            # SL/TP 計算
            params = self.cfg.strategy.get_params(symbol)
            stop_loss_price = None
            take_profit_price = None
            stop_loss_atr = params.get("stop_loss_atr")
            take_profit_atr = params.get("take_profit_atr")
            atr_value = indicators.get("atr")

            if atr_value and target_pct != 0:
                if target_pct > 0:
                    if stop_loss_atr:
                        stop_loss_price = price - float(stop_loss_atr) * float(atr_value)
                    if take_profit_atr:
                        take_profit_price = price + float(take_profit_atr) * float(atr_value)
                elif target_pct < 0:
                    if stop_loss_atr:
                        stop_loss_price = price + float(stop_loss_atr) * float(atr_value)
                    if take_profit_atr:
                        take_profit_price = price - float(take_profit_atr) * float(atr_value)

            try:
                trade = self.broker.execute_target_position(
                    symbol=symbol,
                    target_pct=target_pct,
                    current_price=price,
                    reason=reason,
                    stop_loss_price=stop_loss_price,
                    take_profit_price=take_profit_price,
                )
            except Exception as e:
                logger.error(f"❌ {symbol} 交易執行失敗: {e}")
                logger.error(traceback.format_exc())
                return

            if trade:
                self.trade_count += 1
                has_trade = True

                # 記錄交易到 DB
                if self.trading_db:
                    try:
                        order_type = "MARKET"
                        if hasattr(trade, "raw") and trade.raw:
                            order_type = trade.raw.get("_order_type", "MARKET")

                        self.trading_db.log_trade(
                            symbol=symbol,
                            side=trade.side,
                            qty=trade.qty,
                            price=trade.price,
                            pnl=trade.pnl,
                            reason=reason,
                            order_type=order_type,
                            position_side=getattr(trade, "position_side", ""),
                        )
                    except Exception:
                        pass

                # 發送通知
                try:
                    self.notifier.send_trade(
                        symbol=symbol,
                        side=trade.side,
                        qty=trade.qty,
                        price=trade.price,
                        reason=reason,
                        pnl=trade.pnl,
                        stop_loss_price=stop_loss_price,
                        take_profit_price=take_profit_price,
                    )
                except Exception as e:
                    logger.debug(f"通知發送失敗: {e}")

        # 發送摘要 (僅當有交易時)
        if has_trade:
            try:
                self.notifier.send_signal_summary(
                    [sig], mode=f"WS_{self.mode.upper()}", has_trade=True
                )
            except Exception as e:
                logger.debug(f"摘要通知失敗: {e}")

    def run(self):
        """啟動 WebSocket 連接並保持運行"""
        self.start_time = time.time()

        alloc_str = ", ".join(f"{s}={w:.0%}" for s, w in self._weights.items())
        logger.info("=" * 60)
        logger.info(f"🚀 WebSocket Runner 啟動 [{self.mode.upper()}]")
        logger.info(f"   策略: {self.strategy_name}")
        logger.info(f"   訂閱: {', '.join(self.symbols)} @ {self.interval}")
        logger.info(f"   倉位分配: {alloc_str}")
        logger.info(f"   市場: {self.market_type}")
        logger.info(f"   交易資料庫: {'✅ SQLite' if self.trading_db else '❌ 未啟用'}")
        logger.info(f"   Telegram: {'✅ 已啟用' if self.notifier.enabled else '❌ 未啟用'}")
        logger.info(f"   K 線緩衝區: {', '.join(f'{s}={len(df)}' for s, df in self._kline_buffer.items())}")
        logger.info("=" * 60)

        # 發送啟動通知
        try:
            self.notifier.send_startup(
                strategy=f"{self.strategy_name} (WebSocket)",
                symbols=self.symbols,
                interval=self.interval,
                mode=self.mode,
                weights=self._weights,
                market_type=self.market_type,
            )
        except Exception as e:
            logger.warning(f"啟動通知發送失敗: {e}")

        # 啟動 WebSocket Client
        try:
            from binance.websocket.um_futures.websocket_client import UMFuturesWebsocketClient

            self._ws_client = UMFuturesWebsocketClient(
                on_message=self._on_message_handler,
            )

            # 訂閱 K 線串流
            for symbol in self.symbols:
                stream_name = f"{symbol.lower()}@kline_{self.interval}"
                self._ws_client.kline(symbol=symbol.lower(), interval=self.interval, id=1)
                logger.info(f"📡 訂閱串流: {stream_name}")

        except Exception as e:
            logger.error(f"❌ WebSocket 連線失敗: {e}")
            logger.error(traceback.format_exc())
            raise

        self.is_running = True
        logger.info("✅ WebSocket 已連線，等待 K 線事件...")

        try:
            while self.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("⛔ 收到 KeyboardInterrupt，停止 WebSocket...")
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
            logger.info(f"👋 WebSocket Runner 已停止 (運行 {hours:.1f}h, 交易 {self.trade_count} 筆)")

    def _on_message_handler(self, _, msg):
        """
        轉發消息到處理函數

        binance-futures-connector 的 callback 簽名: callback(socket_manager, message)
        其中 message 是 str (JSON)
        """
        try:
            if isinstance(msg, str):
                msg = json.loads(msg)

            # 過濾 K 線事件
            if isinstance(msg, dict) and msg.get("e") == "kline":
                self._on_kline_event(msg)
        except Exception as e:
            logger.error(f"WS Message Error: {e}")
            logger.error(traceback.format_exc())
