"""
WebSocket Runner — 輕量化事件驅動交易執行器 (v3.1)

v3.0: 初版 — 基本 Rolling Window + 信號觸發
v3.1: 修復回測一致性 + 移植完整安全機制
    - IncrementalKlineCache 取代 Rolling Window（信號與回測 100% 一致）
    - SL/TP 冷卻 + 孤兒掛單清理（v2.4 + v2.7.1）
    - SL/TP 補掛機制（v2.5）
    - 方向錯誤 TP 偵測（v2.7）
    - 防不必要重平衡（v2.8）
    - 完整倉位計算器（volatility / kelly / fixed）
    - 方向切換確認（可選）
    - 信號狀態持久化
    - WebSocket 斷線心跳監控
    - 定期權益快照

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
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

from ..config import AppConfig
from ..utils.log import get_logger
from .signal_generator import generate_signal
from .kline_cache import IncrementalKlineCache
from .paper_broker import PaperBroker
from ..risk.position_sizing import (
    PositionSizer,
    FixedPositionSizer,
    KellyPositionSizer,
    VolatilityPositionSizer,
)

logger = get_logger("ws_runner")

# 心跳超時（秒）：超過此時間沒收到任何 WS 消息就視為斷線
HEARTBEAT_TIMEOUT = 300  # 5 分鐘

# interval → 分鐘 對照表
INTERVAL_MINUTES = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "6h": 360, "8h": 480,
    "12h": 720, "1d": 1440,
}


class WebSocketRunner:
    """
    基於 WebSocket 的輕量化執行器 (v3.1)

    不繼承 LiveRunner（避免 polling 相關初始化），
    但移植了 LiveRunner 的 **全部安全機制**：
        - SL/TP 冷卻 + 孤兒清理 (v2.4 + v2.7.1)
        - SL/TP 補掛 (v2.5)
        - 方向錯誤 TP 偵測 (v2.7)
        - 防不必要重平衡 (v2.8)
        - Drawdown 熔斷
        - 方向切換確認（可選）
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
        self._tick_count = 0  # K 線收盤次數

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

        # 倉位計算器（支持 fixed / kelly / volatility）
        self.position_sizer: Optional[PositionSizer] = None
        self._init_position_sizer()

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

        # v3.1: IncrementalKlineCache（取代 Rolling Window，確保回測一致性）
        cache_dir = cfg.get_report_dir("live") / "kline_cache"
        self._kline_cache = IncrementalKlineCache(
            cache_dir=cache_dir,
            interval=self.interval,
            seed_bars=300,
            market_type=self.market_type,
        )

        # 信號狀態持久化（方向切換確認用）
        self._signal_state_path = cfg.get_report_dir("live") / "signal_state.json"
        self._signal_state: dict[str, float] = self._load_signal_state()

        # 心跳追蹤
        self._last_ws_message_time: float = 0.0
        self._ws_client = None
        self._last_kline_ts: Dict[str, int] = {}

        # 定期任務計時
        self._last_summary_time: float = 0.0

        # interval 分鐘數（用於缺口偵測）
        self._interval_minutes = INTERVAL_MINUTES.get(self.interval, 60)

        # 預熱 K 線快取
        self._init_kline_buffer()

    # ══════════════════════════════════════════════════════════
    #  倉位計算器
    # ══════════════════════════════════════════════════════════

    def _init_position_sizer(self) -> None:
        """根據配置初始化倉位計算器（與 LiveRunner 邏輯一致）"""
        ps_cfg = self.cfg.position_sizing

        if ps_cfg.method == "kelly":
            stats = self._get_trade_stats()
            total_trades = stats.get("total_trades", 0)
            min_trades = getattr(ps_cfg, "min_trades_for_kelly", 30)

            if total_trades < min_trades:
                logger.info(
                    f"📊 倉位計算: 交易數 ({total_trades}) < 最小要求 ({min_trades})，暫用固定倉位"
                )
                self.position_sizer = FixedPositionSizer(ps_cfg.position_pct)
            else:
                try:
                    win_rate = getattr(ps_cfg, "win_rate", None) or stats.get("win_rate", 0.5)
                    avg_win = getattr(ps_cfg, "avg_win", None) or stats.get("avg_win", 1.0)
                    avg_loss = getattr(ps_cfg, "avg_loss", None) or stats.get("avg_loss", 1.0)
                    self.position_sizer = KellyPositionSizer(
                        win_rate=win_rate,
                        avg_win=avg_win,
                        avg_loss=avg_loss,
                        kelly_fraction=ps_cfg.kelly_fraction,
                    )
                    logger.info(
                        f"📊 倉位計算: Kelly (fraction={ps_cfg.kelly_fraction}, "
                        f"kelly_pct={self.position_sizer.kelly_pct:.1%})"
                    )
                except ValueError as e:
                    logger.warning(f"⚠️  Kelly 參數無效: {e}，改用固定倉位")
                    self.position_sizer = FixedPositionSizer(ps_cfg.position_pct)

        elif ps_cfg.method == "volatility":
            self.position_sizer = VolatilityPositionSizer(
                base_position_pct=ps_cfg.position_pct,
                target_volatility=ps_cfg.target_volatility,
                lookback=ps_cfg.vol_lookback,
            )
            logger.info(f"📊 倉位計算: 波動率目標 ({ps_cfg.target_volatility:.1%})")

        else:
            self.position_sizer = FixedPositionSizer(ps_cfg.position_pct)
            logger.info(f"📊 倉位計算: 固定 ({ps_cfg.position_pct:.0%})")

    def _get_trade_stats(self) -> dict:
        """從 TradingDB 取得交易統計（Kelly 用）"""
        if self.trading_db:
            try:
                summary = self.trading_db.get_performance_summary()
                return {
                    "win_rate": summary.get("win_rate", 0.5),
                    "avg_win": summary.get("avg_win_pnl", 1.0),
                    "avg_loss": abs(summary.get("avg_loss_pnl", 1.0)),
                    "total_trades": summary.get("total_trades", 0),
                }
            except Exception:
                pass
        return {"win_rate": 0.5, "avg_win": 1.0, "avg_loss": 1.0, "total_trades": 0}

    def _get_equity(self) -> float | None:
        """取得當前權益（Paper / Real 通用）"""
        try:
            if isinstance(self.broker, PaperBroker):
                prices = {}
                for sym in self.symbols:
                    df = self._kline_cache.get_cached(sym)
                    if df is not None and len(df) > 0:
                        prices[sym] = float(df["close"].iloc[-1])
                return self.broker.get_equity(prices)
            elif hasattr(self.broker, "get_equity"):
                return self.broker.get_equity()
        except Exception as e:
            logger.debug(f"取得權益失敗: {e}")
        return None

    def _apply_position_sizing(self, raw_signal: float, price: float, symbol: str) -> float:
        """
        應用倉位計算器調整信號（與 LiveRunner 邏輯一致）

        Args:
            raw_signal: 原始信號 [-1, 1]
            price: 當前價格
            symbol: 交易對

        Returns:
            調整後的信號 [-1, 1]
        """
        if self.position_sizer is None:
            return raw_signal

        try:
            # 獲取當前權益
            if isinstance(self.broker, PaperBroker):
                prices = {}
                for sym in self.symbols:
                    df = self._kline_cache.get_cached(sym)
                    if df is not None and len(df) > 0:
                        prices[sym] = float(df["close"].iloc[-1])
                equity = self.broker.get_equity(prices)
            elif hasattr(self.broker, "get_equity"):
                try:
                    equity = self.broker.get_equity()  # Futures
                except TypeError:
                    equity = self.broker.get_equity([symbol])  # Spot
            else:
                equity = 10000

            # 計算倉位大小
            position_size = self.position_sizer.calculate_size(
                signal=raw_signal,
                equity=equity,
                price=price,
            )

            # 轉換為倉位比例
            position_value = position_size * price
            adjusted_signal = position_value / equity if equity > 0 else raw_signal

            # 限制在 [-1, 1]
            return max(-1.0, min(1.0, adjusted_signal))
        except Exception:
            return raw_signal

    # ══════════════════════════════════════════════════════════
    #  K 線管理
    # ══════════════════════════════════════════════════════════

    def _init_kline_buffer(self):
        """使用 IncrementalKlineCache 預熱 K 線（與 Polling Runner 共用同一份快取）"""
        logger.info("📥 正在預熱 K 線緩衝區...")
        for symbol in self.symbols:
            try:
                df = self._kline_cache.get_klines(symbol)
                if df is not None and len(df) > 0:
                    logger.info(
                        f"  ✅ {symbol}: 已載入 {len(df)} 根 {self.market_type} K 線 "
                        f"({df.index[0].strftime('%Y-%m-%d')} ~ "
                        f"{df.index[-1].strftime('%m-%d %H:%M')})"
                    )
                else:
                    logger.warning(f"  ⚠️  {symbol}: 無法載入 K 線數據")
            except Exception as e:
                logger.error(f"  ❌ {symbol}: K 線載入失敗: {e}")
                logger.error(traceback.format_exc())

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

                # 追加到 IncrementalKlineCache
                self._append_kline(symbol, k)

                # 執行策略
                self._tick_count += 1
                self._run_strategy_for_symbol(symbol)

                # 定期任務
                now = time.time()
                if now - self._last_summary_time > 6 * 3600:  # 每 6 小時
                    self._last_summary_time = now
                    self._send_periodic_summary()

                # 定期重新計算 Kelly（每 24 tick ≈ 24h）
                if self.cfg.position_sizing.method == "kelly" and self._tick_count % 24 == 0:
                    self._init_position_sizer()

        except Exception as e:
            logger.error(f"WebSocket 處理異常: {e}")
            logger.error(traceback.format_exc())

    def _append_kline(self, symbol: str, k: Dict[str, Any]):
        """追加 K 線到 IncrementalKlineCache（含缺口偵測）"""
        try:
            new_time = pd.to_datetime(k["t"], unit="ms", utc=True)

            # 缺口偵測：如果新 K 線與快取最後一根相差超過 2 個 interval，
            # 代表 WS 斷線期間遺漏了 K 線，需要 HTTP 補齊
            cached = self._kline_cache.get_cached(symbol)
            if cached is not None and len(cached) > 0:
                last_time = cached.index[-1]
                expected_gap = pd.Timedelta(minutes=self._interval_minutes)
                actual_gap = new_time - last_time

                if actual_gap > expected_gap * 2:
                    logger.warning(
                        f"⚠️  {symbol}: 偵測到 K 線缺口 "
                        f"({last_time.strftime('%H:%M')} → {new_time.strftime('%H:%M')}, "
                        f"差距 {actual_gap})，HTTP 補齊中..."
                    )
                    self._kline_cache.fill_gap(symbol, last_time)

            # 追加新 K 線
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
            logger.error(f"❌ {symbol} K 線追加失敗: {e}")
            logger.error(traceback.format_exc())

    # ══════════════════════════════════════════════════════════
    #  策略執行
    # ══════════════════════════════════════════════════════════

    def _run_strategy_for_symbol(self, symbol: str):
        """針對單一幣種執行策略"""
        # 熔斷檢查
        if self._check_circuit_breaker():
            logger.warning("⛔ 熔斷已觸發，跳過交易")
            return

        logger.info(f"⚡️ 觸發策略: {symbol}")

        # 取得完整 K 線（IncrementalKlineCache 累積歷史，與回測一致）
        df = self._kline_cache.get_cached(symbol)
        if df is None or len(df) < 50:
            logger.warning(
                f"⚠️  {symbol} 數據不足 ({len(df) if df is not None else 0}/50)，跳過策略"
            )
            return

        # 生成信號
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

        # 處理信號（包含全部安全機制）
        self._process_signal(symbol, sig)

    def _process_signal(self, symbol: str, sig: dict):
        """
        處理信號並下單（移植 LiveRunner 全部安全機制）

        流程與 LiveRunner.run_once 完全一致：
        1. 記錄信號到 DB
        2. Spot clip
        3. 倉位計算（volatility / kelly / fixed）
        4. SL/TP 冷卻 + 孤兒掛單清理 (v2.4 + v2.7.1)
        5. 防不必要重平衡 (v2.8)
        6. 方向切換確認（可選）
        7. 執行交易 + SL/TP 計算
        8. SL/TP 補掛 (v2.5 + v2.7)
        9. Algo cache 清理
        """
        raw_signal = sig["signal"]
        price = sig["price"]
        indicators = sig.get("indicators", {})
        params = self.cfg.strategy.get_params(symbol)

        # ── 1. 記錄信號到 DB ──────────────────────────────
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

        # ── 2. Spot clip ─────────────────────────────────
        if self.market_type == "spot" and raw_signal < 0:
            logger.debug(f"  {symbol}: Spot 模式不支援做空，信號 {raw_signal:.0%} clip 到 0")
            raw_signal = 0.0

        # ── 3. 倉位計算 ─────────────────────────────────
        weight = self._weights.get(symbol, 1.0 / max(len(self.symbols), 1))
        if price <= 0:
            return

        adjusted_signal = self._apply_position_sizing(raw_signal, price, symbol)
        target_pct = adjusted_signal * weight

        current_pct = self.broker.get_position_pct(symbol, price)
        diff = abs(target_pct - current_pct)

        # ── 4. SL/TP 冷卻 + 孤兒掛單清理 (v2.4 + v2.7.1) ──
        if self._check_sl_tp_cooldown(symbol, current_pct, target_pct):
            # 冷卻中 → 仍然需要檢查 SL/TP 補掛
            actual_pct = current_pct
            if not isinstance(self.broker, PaperBroker) and hasattr(self.broker, "get_position_pct"):
                try:
                    actual_pct = self.broker.get_position_pct(symbol, price)
                except Exception:
                    pass
            self._ensure_sl_tp(symbol, sig, params, actual_pct)
            return

        # ── 5. 防不必要重平衡 (v2.8) ───────────────────
        if target_pct != 0 and current_pct != 0:
            same_direction = (
                (target_pct > 0 and current_pct > 0) or
                (target_pct < 0 and current_pct < 0)
            )
            if same_direction:
                fill_ratio = abs(current_pct) / abs(target_pct)
                if fill_ratio >= 0.80:  # 已達目標 80% → 跳過微調
                    diff = 0
                    logger.debug(
                        f"  {symbol}: 方向一致且倉位充足 "
                        f"({current_pct:+.1%} / {target_pct:+.1%} = {fill_ratio:.0%})，跳過"
                    )
                else:
                    logger.info(
                        f"  {symbol}: 方向一致但倉位不足 "
                        f"({current_pct:+.1%} / {target_pct:+.1%} = {fill_ratio:.0%})，需加倉"
                    )

        # ── 6. 方向切換確認（可選）────────────────────
        # 先取 previous signal（更新前）
        prev_signal = self._signal_state.get(symbol)

        is_direction_flip = (
            (target_pct > 0.01 and current_pct < -0.01) or   # SHORT → LONG
            (target_pct < -0.01 and current_pct > 0.01)      # LONG → SHORT
        )

        if is_direction_flip and self.cfg.live.flip_confirmation:
            if prev_signal is None:
                logger.info(f"  {symbol}: 方向切換 (首次啟動) → 直接執行")
            else:
                new_dir = 1 if target_pct > 0 else -1
                prev_dir = 1 if prev_signal > 0 else (-1 if prev_signal < 0 else 0)
                if prev_dir == new_dir:
                    logger.info(
                        f"✅ {symbol}: 方向切換已確認 "
                        f"(前次={prev_signal:+.0%}, 本次={raw_signal:+.0%})"
                    )
                else:
                    logger.warning(
                        f"⚠️  {symbol}: 方向切換待確認 "
                        f"(持倉={current_pct:+.0%} → 信號={raw_signal:+.0%}) "
                        f"— 維持原方向"
                    )
                    # 覆寫 target_pct 為維持原方向
                    if current_pct < 0:
                        target_pct = -1.0 * weight
                    else:
                        target_pct = 1.0 * weight
                    diff = abs(target_pct - current_pct)
        elif is_direction_flip:
            logger.info(
                f"🔄 {symbol}: 方向切換 ({current_pct:+.0%} → {raw_signal:+.0%}) — 直接執行"
            )

        # 更新信號狀態（在檢查之後）
        self._signal_state[symbol] = sig["signal"]
        self._save_signal_state(self._signal_state)

        # Log 信號狀態
        logger.info(
            f"  📊 {symbol}: signal={raw_signal:.2f}, target={target_pct:.2f}, "
            f"current={current_pct:.2f}, diff={diff:.2f}, "
            f"RSI={indicators.get('rsi', '?')}, ADX={indicators.get('adx', '?')}"
        )

        # ── 7. 執行交易 ─────────────────────────────────
        if diff >= 0.02:
            ps_method = self.cfg.position_sizing.method
            reason = f"WS_signal={raw_signal:.0%}×{weight:.0%}"
            if ps_method != "fixed":
                reason += f" [{ps_method}→{adjusted_signal:.0%}]"

            # SL/TP 價格計算
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

                if stop_loss_price or take_profit_price:
                    pos_side = "LONG" if target_pct > 0 else "SHORT"
                    sl_str = f"${stop_loss_price:,.2f}" if stop_loss_price else "N/A"
                    tp_str = f"${take_profit_price:,.2f}" if take_profit_price else "N/A"
                    logger.info(f"🛡️  {symbol} [{pos_side}] SL={sl_str}, TP={tp_str}")

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

                # 記錄交易到 DB
                if self.trading_db:
                    try:
                        order_type = "MARKET"
                        fee_rate = 0.0004  # default taker
                        if hasattr(trade, "raw") and trade.raw:
                            order_type = trade.raw.get("_order_type", "MARKET")
                            fee_rate = trade.raw.get("_fee_rate", 0.0004)
                        self.trading_db.log_trade(
                            symbol=symbol,
                            side=trade.side,
                            qty=trade.qty,
                            price=trade.price,
                            fee=getattr(trade, "fee", 0.0),
                            fee_rate=fee_rate,
                            pnl=trade.pnl,
                            reason=reason,
                            order_type=order_type,
                            order_id_hash=getattr(trade, "order_id", "")[:8],
                            position_side=getattr(trade, "position_side", ""),
                        )
                    except Exception as e:
                        logger.debug(f"  {symbol}: 交易寫入 DB 失敗: {e}")

                # Telegram 通知交易
                try:
                    leverage = self.cfg.futures.leverage if self.cfg.futures else None
                    self.notifier.send_trade(
                        symbol=symbol,
                        side=trade.side,
                        qty=trade.qty,
                        price=trade.price,
                        reason=reason,
                        pnl=trade.pnl,
                        weight=weight,
                        leverage=leverage if self.market_type == "futures" else None,
                        stop_loss_price=stop_loss_price,
                        take_profit_price=take_profit_price,
                    )
                except Exception as e:
                    logger.debug(f"通知發送失敗: {e}")

                # 發送信號摘要
                try:
                    self.notifier.send_signal_summary(
                        [sig], mode=f"WS_{self.mode.upper()}", has_trade=True
                    )
                except Exception:
                    pass
        else:
            logger.debug(
                f"  {symbol}: 倉位不變 (target={target_pct:.0%}, current={current_pct:.0%})"
            )

        # ── 8. SL/TP 補掛機制 (v2.5 + v2.7) ──────────
        # 交易後重新讀取實際倉位
        actual_pct = current_pct
        if not isinstance(self.broker, PaperBroker) and hasattr(self.broker, "get_position_pct"):
            try:
                actual_pct = self.broker.get_position_pct(symbol, price)
            except Exception:
                pass

        self._ensure_sl_tp(symbol, sig, params, actual_pct)

        # ── 9. Algo cache 清理 ────────────────────────
        if (
            abs(actual_pct) <= 0.01
            and not isinstance(self.broker, PaperBroker)
            and hasattr(self.broker, "_remove_algo_cache")
        ):
            self.broker._remove_algo_cache(symbol)

    # ══════════════════════════════════════════════════════════
    #  安全機制（移植自 LiveRunner）
    # ══════════════════════════════════════════════════════════

    def _check_circuit_breaker(self) -> bool:
        """Drawdown 熔斷檢查（與 LiveRunner 一致）"""
        if self._circuit_breaker_triggered:
            return True
        if not self.max_drawdown_pct:
            return False

        try:
            equity = self._get_equity()
            if equity is None or equity <= 0:
                return False

            if self._initial_equity is None:
                if isinstance(self.broker, PaperBroker):
                    self._initial_equity = self.broker.account.initial_cash
                else:
                    self._initial_equity = equity
                logger.info(f"📊 熔斷基準權益: ${self._initial_equity:,.2f}")
                return False

            drawdown = 1.0 - (equity / self._initial_equity)

            if drawdown >= self.max_drawdown_pct:
                self._circuit_breaker_triggered = True
                logger.warning(
                    f"🚨🚨🚨 CIRCUIT BREAKER 觸發！"
                    f"Drawdown={drawdown:.1%} >= {self.max_drawdown_pct:.0%} "
                    f"(權益 ${equity:,.2f} / 基準 ${self._initial_equity:,.2f})"
                )
                # 平掉所有倉位
                for sym in self.symbols:
                    try:
                        p = 0.0
                        if hasattr(self.broker, "get_price"):
                            p = self.broker.get_price(sym)
                        if p <= 0:
                            df = self._kline_cache.get_cached(sym)
                            if df is not None and len(df) > 0:
                                p = float(df["close"].iloc[-1])
                        pct = self.broker.get_position_pct(sym, p)
                        if abs(pct) > 0.01:
                            self.broker.execute_target_position(
                                symbol=sym, target_pct=0.0,
                                current_price=p, reason="CIRCUIT_BREAKER"
                            )
                            logger.warning(f"  🔴 強制平倉 {sym}")
                    except Exception as e:
                        logger.error(f"  ❌ 強制平倉 {sym} 失敗: {e}")

                self.notifier.send_error(
                    f"🚨 <b>CIRCUIT BREAKER 熔斷觸發!</b>\n\n"
                    f"  Drawdown: <b>{drawdown:.1%}</b> (閾值 {self.max_drawdown_pct:.0%})\n"
                    f"  ⚠️ 已強制平倉所有持倉"
                )
                return True

            # 接近熔斷線預警
            if drawdown >= self.max_drawdown_pct * 0.8:
                logger.warning(f"⚠️  Drawdown 預警: {drawdown:.1%}")

        except Exception as e:
            logger.debug(f"熔斷檢查失敗: {e}")
        return False

    def _check_sl_tp_cooldown(
        self, symbol: str, current_pct: float, target_pct: float
    ) -> bool:
        """
        SL/TP 冷卻檢查 + 孤兒掛單清理（移植自 LiveRunner v2.4 + v2.7.1）

        場景 A (v2.4): SL/TP 觸發 → 倉位歸零 + 掛單消失 → 冷卻等下根 bar
        場景 B (v2.7.1): SL 觸發 → 倉位歸零 + TP 殘留 → 先清掃孤兒再冷卻

        Returns:
            True = 應跳過本次開倉（冷卻中）
        """
        if not (
            abs(current_pct) < 0.01              # 目前幾乎無倉
            and abs(target_pct) > 0.02            # 策略要求開倉
            and not isinstance(self.broker, PaperBroker)
            and hasattr(self.broker, "get_open_orders")
            and hasattr(self.broker, "get_trade_history")
        ):
            return False

        try:
            # 合併 regular + algo orders 檢查 SL/TP
            if hasattr(self.broker, "get_all_conditional_orders"):
                cond_orders = self.broker.get_all_conditional_orders(symbol)
            else:
                cond_orders = self.broker.get_open_orders(symbol)
            sl_tp_types = {"STOP_MARKET", "TAKE_PROFIT_MARKET", "STOP", "TAKE_PROFIT"}
            has_sl_tp = any(o.get("type") in sl_tp_types for o in cond_orders)

            # v2.7.1: 空倉 + 有殘留 SL/TP → 孤兒掛單
            if has_sl_tp:
                orphan_detail = [
                    f"{o.get('type')}[{o.get('positionSide', '?')}] "
                    f"@ ${float(o.get('stopPrice', 0) or o.get('triggerPrice', 0) or 0):,.2f}"
                    for o in cond_orders if o.get("type") in sl_tp_types
                ]
                logger.warning(
                    f"🧹 {symbol}: 無持倉但有殘留掛單 {orphan_detail} → 取消孤兒 SL/TP"
                )
                if hasattr(self.broker, "cancel_all_open_orders"):
                    self.broker.cancel_all_open_orders(symbol)
                else:
                    self.broker.cancel_stop_loss(symbol)
                    self.broker.cancel_take_profit(symbol)
                if hasattr(self.broker, "_remove_algo_cache"):
                    self.broker._remove_algo_cache(symbol)
                has_sl_tp = False

            if not has_sl_tp:
                # 無 SL/TP 掛單（或剛清理完孤兒） → 可能 SL/TP 剛觸發
                recent_trades = self.broker.get_trade_history(symbol=symbol, limit=5)
                now_ms = int(time.time() * 1000)
                cooldown_ms = 10 * 60 * 1000  # 10 分鐘

                recently_closed = any(
                    now_ms - t.get("time", 0) < cooldown_ms
                    for t in (recent_trades or [])
                )
                if recently_closed:
                    logger.warning(
                        f"⚠️  {symbol}: 無持倉且無 SL/TP，但最近 10min 有成交 → "
                        f"疑似 SL/TP 觸發，跳過本次開倉（冷卻等下根 bar）"
                    )
                    return True
        except Exception as e:
            logger.debug(f"  {symbol}: SL/TP 冷卻檢查失敗: {e}（繼續正常流程）")
        return False

    def _ensure_sl_tp(self, symbol: str, sig: dict, params: dict, actual_pct: float):
        """
        SL/TP 補掛機制（移植自 LiveRunner v2.5 + v2.7）

        確保每個有持倉的幣種都有 SL/TP 保護。
        包含方向錯誤 TP 偵測（翻倉後舊 TP 未取消的場景）。
        """
        if isinstance(self.broker, PaperBroker):
            return
        if abs(actual_pct) <= 0.01:
            return
        if not hasattr(self.broker, "place_stop_loss"):
            return
        if not hasattr(self.broker, "get_open_orders"):
            return

        stop_loss_atr = params.get("stop_loss_atr")
        take_profit_atr = params.get("take_profit_atr")
        atr_value = sig.get("indicators", {}).get("atr")
        price = sig["price"]

        if not ((stop_loss_atr or take_profit_atr) and atr_value):
            return

        try:
            # 查詢條件掛單
            if hasattr(self.broker, "get_all_conditional_orders"):
                cond_orders = self.broker.get_all_conditional_orders(symbol)
            else:
                cond_orders = self.broker.get_open_orders(symbol)

            position_side = "LONG" if actual_pct > 0 else "SHORT"

            # 只看與當前持倉同方向的 SL/TP
            def _match_side(o: dict) -> bool:
                o_ps = o.get("positionSide", "")
                return not o_ps or o_ps == position_side or o_ps == "BOTH"

            has_sl = any(
                o.get("type") in {"STOP_MARKET", "STOP"} and _match_side(o)
                for o in cond_orders
            )
            has_tp = any(
                o.get("type") in {"TAKE_PROFIT_MARKET", "TAKE_PROFIT"} and _match_side(o)
                for o in cond_orders
            )

            # v2.7: 檢查方向錯誤 TP（翻倉後舊 TP 未取消）
            if has_tp and hasattr(self.broker, "get_position"):
                pos_check = self.broker.get_position(symbol)
                if pos_check and pos_check.entry_price > 0:
                    is_long = pos_check.qty > 0
                    for o in cond_orders:
                        otype = o.get("type", "")
                        if otype not in {"TAKE_PROFIT_MARKET", "TAKE_PROFIT"}:
                            continue
                        trigger = float(
                            o.get("stopPrice", 0) or o.get("triggerPrice", 0) or 0
                        )
                        if trigger <= 0:
                            continue
                        wrong_dir = (
                            (is_long and trigger < pos_check.entry_price * 0.99) or
                            (not is_long and trigger > pos_check.entry_price * 1.01)
                        )
                        if wrong_dir:
                            logger.warning(
                                f"🚨 {symbol}: 方向錯誤 TP "
                                f"${trigger:,.2f} "
                                f"({'LONG' if is_long else 'SHORT'} 倉 "
                                f"entry=${pos_check.entry_price:,.2f}) → 取消"
                            )
                            self.broker.cancel_take_profit(symbol)
                            has_tp = False
                            break

            # 補掛 SL
            if not has_sl and stop_loss_atr:
                if actual_pct > 0:
                    sl_price = price - float(stop_loss_atr) * float(atr_value)
                else:
                    sl_price = price + float(stop_loss_atr) * float(atr_value)
                logger.info(
                    f"🔄 {symbol}: 補掛止損單 SL=${sl_price:,.2f} [{position_side}]"
                )
                self.broker.place_stop_loss(
                    symbol=symbol, stop_price=sl_price,
                    position_side=position_side, reason="ensure_stop_loss",
                )

            # 補掛 TP
            if not has_tp and take_profit_atr:
                if actual_pct > 0:
                    tp_price = price + float(take_profit_atr) * float(atr_value)
                else:
                    tp_price = price - float(take_profit_atr) * float(atr_value)
                logger.info(
                    f"🔄 {symbol}: 補掛止盈單 TP=${tp_price:,.2f} [{position_side}]"
                )
                self.broker.place_take_profit(
                    symbol=symbol, take_profit_price=tp_price,
                    position_side=position_side, reason="ensure_take_profit",
                )

            if has_sl and (has_tp or not take_profit_atr):
                logger.debug(f"  {symbol}: SL/TP 掛單正常 ✓")

        except Exception as e:
            logger.warning(f"⚠️  {symbol}: SL/TP 補掛檢查失敗: {e}")

    # ══════════════════════════════════════════════════════════
    #  信號狀態持久化
    # ══════════════════════════════════════════════════════════

    def _load_signal_state(self) -> dict[str, float]:
        """載入上一次的信號方向"""
        try:
            if self._signal_state_path.exists():
                with open(self._signal_state_path) as f:
                    data = json.load(f)
                return data.get("signals", {})
        except Exception:
            pass
        return {}

    def _save_signal_state(self, signal_map: dict[str, float]) -> None:
        """保存信號方向到磁碟"""
        try:
            self._signal_state_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "signals": signal_map,
            }
            with open(self._signal_state_path, "w") as f:
                json.dump(payload, f, indent=2)
        except Exception:
            pass

    # ══════════════════════════════════════════════════════════
    #  定期任務
    # ══════════════════════════════════════════════════════════

    def _send_periodic_summary(self):
        """定期推送帳戶摘要（支援 Paper + Real 模式）"""
        try:
            if isinstance(self.broker, PaperBroker):
                prices = {}
                for sym in self.symbols:
                    df = self._kline_cache.get_cached(sym)
                    if df is not None and len(df) > 0:
                        prices[sym] = float(df["close"].iloc[-1])
                if prices:
                    equity = self.broker.get_equity(prices)
                    positions_info = {
                        sym: {"qty": p.qty, "avg_entry": p.avg_entry}
                        for sym, p in self.broker.account.positions.items()
                        if p.is_open
                    }
                    self.notifier.send_account_summary(
                        initial_cash=self.broker.account.initial_cash,
                        equity=equity,
                        cash=self.broker.account.cash,
                        positions=positions_info,
                        trade_count=len(self.broker.account.trades),
                        mode=f"WS_{self.mode.upper()}",
                    )
                    if self.trading_db:
                        try:
                            self.trading_db.log_daily_equity(
                                equity=equity,
                                cash=self.broker.account.cash,
                                pnl_day=equity - self.broker.account.initial_cash,
                                trade_count=len(self.broker.account.trades),
                                position_count=len(positions_info),
                            )
                        except Exception:
                            pass
            else:
                # Real 模式
                usdt = self.broker.get_balance("USDT")
                positions_info = {}
                total_value = usdt
                for sym in self.symbols:
                    pos = self.broker.get_position(sym)
                    if pos and pos.is_open:
                        p = self.broker.get_price(sym)
                        val = abs(pos.qty) * p
                        total_value += val
                        positions_info[sym] = {
                            "qty": pos.qty,
                            "avg_entry": pos.entry_price,
                            "side": "LONG" if pos.qty > 0 else "SHORT",
                        }

                logger.info(
                    f"\n{'='*50}\n"
                    f"  帳戶摘要 [WS_{self.mode.upper()}]\n"
                    f"{'='*50}\n"
                    f"  USDT: ${usdt:,.2f}\n"
                    f"  總權益: ${total_value:,.2f}\n"
                    f"{'='*50}"
                )

                self.notifier.send_account_summary(
                    initial_cash=0,
                    equity=total_value,
                    cash=usdt,
                    positions=positions_info,
                    trade_count=self.trade_count,
                    mode=f"WS_{self.mode.upper()}",
                )

                if self.trading_db:
                    try:
                        self.trading_db.log_daily_equity(
                            equity=total_value,
                            cash=usdt,
                            trade_count=self.trade_count,
                            position_count=len(positions_info),
                        )
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"⚠️  週期報告失敗: {e}")

    # ══════════════════════════════════════════════════════════
    #  WebSocket 管理 + 心跳監控
    # ══════════════════════════════════════════════════════════

    def run(self):
        """啟動 WebSocket 連接並保持運行（含心跳監控）"""
        self.start_time = time.time()
        self._last_summary_time = time.time()

        alloc_str = ", ".join(f"{s}={w:.0%}" for s, w in self._weights.items())
        logger.info("=" * 60)
        logger.info(f"🚀 WebSocket Runner 啟動 [{self.mode.upper()}]")
        logger.info(f"   策略: {self.strategy_name}")
        logger.info(f"   訂閱: {', '.join(self.symbols)} @ {self.interval}")
        logger.info(f"   倉位分配: {alloc_str}")
        logger.info(f"   市場: {self.market_type}")
        logger.info(f"   倉位計算: {self.cfg.position_sizing.method}")
        logger.info(f"   交易資料庫: {'✅ SQLite' if self.trading_db else '❌ 未啟用'}")
        logger.info(f"   Telegram: {'✅ 已啟用' if self.notifier.enabled else '❌ 未啟用'}")
        cache_info = []
        for sym in self.symbols:
            n = self._kline_cache.get_bar_count(sym)
            cache_info.append(f"{sym}={n}")
        logger.info(f"   K 線快取: {', '.join(cache_info)} (IncrementalKlineCache ✅)")
        logger.info(f"   心跳超時: {HEARTBEAT_TIMEOUT}s")
        logger.info("=" * 60)

        # 啟動通知
        try:
            self.notifier.send_startup(
                strategy=f"{self.strategy_name} (WebSocket v3.1)",
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

            # 降低 binance lib 內部的 debug 雜訊
            logging.getLogger("binance").setLevel(logging.WARNING)

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
        self._last_ws_message_time = time.time()
        logger.info("✅ WebSocket 已連線，等待 K 線事件...")

        try:
            while self.is_running:
                try:
                    time.sleep(1)

                    # 心跳監控
                    if self._last_ws_message_time > 0:
                        elapsed = time.time() - self._last_ws_message_time
                        if elapsed > HEARTBEAT_TIMEOUT:
                            logger.warning(
                                f"⚠️  WebSocket 已 {elapsed:.0f}s 未收到消息，可能斷線"
                            )
                            try:
                                self.notifier.send_error(
                                    f"⚠️  WebSocket 可能斷線 ({elapsed:.0f}s 無消息)\n"
                                    f"等待自動重連..."
                                )
                            except Exception:
                                pass
                            # Reset 避免重複告警
                            self._last_ws_message_time = time.time()
                except KeyboardInterrupt:
                    raise  # 交給外層處理
                except Exception as e:
                    logger.error(f"主迴圈異常（自動恢復）: {e}")
                    logger.error(traceback.format_exc())
                    time.sleep(5)

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
            logger.info(
                f"👋 WebSocket Runner 已停止 (運行 {hours:.1f}h, 交易 {self.trade_count} 筆)"
            )

    def _on_message_handler(self, _, msg):
        """
        轉發消息到處理函數

        binance-futures-connector 的 callback 簽名: callback(socket_manager, message)
        其中 message 是 str (JSON)
        """
        # 更新心跳時間戳（任何消息都算）
        self._last_ws_message_time = time.time()

        try:
            if isinstance(msg, str):
                msg = json.loads(msg)

            # 過濾 K 線事件
            if isinstance(msg, dict) and msg.get("e") == "kline":
                self._on_kline_event(msg)
        except Exception as e:
            logger.error(f"WS Message Error: {e}")
            logger.error(traceback.format_exc())
