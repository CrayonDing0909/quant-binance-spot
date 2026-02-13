"""
Live Runner — 即時交易主循環

功能：
    - 每根 K 線收盤後運行策略
    - 對比信號與當前倉位，決定交易
    - 支援 Paper Trading / Real Trading 模式切換
    - Telegram 通知（交易 + 定期摘要）
    - 日誌記錄 + 狀態報告
    - 支援動態倉位計算（Kelly / 波動率）
"""
from __future__ import annotations
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, Optional

from ..config import AppConfig
from ..utils.log import get_logger
from ..monitor.notifier import TelegramNotifier
from ..risk.position_sizing import (
    PositionSizer,
    FixedPositionSizer,
    KellyPositionSizer,
    VolatilityPositionSizer,
)
from .signal_generator import generate_signal
from .kline_cache import IncrementalKlineCache
from .paper_broker import PaperBroker
from .trading_state import TradingStateManager

logger = get_logger("live_runner")


class BrokerProtocol(Protocol):
    """Broker 通用介面，Paper 和 Real broker 都實現此介面"""
    def execute_target_position(
        self, symbol: str, target_pct: float, current_price: float, reason: str = "",
        stop_loss_price: float | None = None, take_profit_price: float | None = None
    ) -> object | None: ...

    def get_position_pct(self, symbol: str, current_price: float) -> float: ...


class LiveRunner:
    """
    即時交易主循環

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
        self.cfg = cfg
        self.broker = broker
        self.mode = mode
        # 使用配置中的通知設定，或預設的環境變數
        self.notifier = notifier or TelegramNotifier.from_config(cfg.notification)
        self.strategy_name = cfg.strategy.name
        self.symbols = cfg.market.symbols
        self.interval = cfg.market.interval
        self.market_type = cfg.market_type_str  # "spot" or "futures"
        self.is_running = False

        # 多幣種倉位分配權重
        self._weights: dict[str, float] = {}
        n = len(self.symbols)
        for sym in self.symbols:
            self._weights[sym] = cfg.portfolio.get_weight(sym, n)

        # Drawdown 熔斷（Paper + Real 模式都生效）
        self.max_drawdown_pct = cfg.risk.max_drawdown_pct if cfg.risk else None
        self._circuit_breaker_triggered = False
        self._initial_equity: float | None = None  # 首次 tick 時記錄基準權益

        # 運行統計
        self.tick_count = 0
        self.trade_count = 0
        self.start_time: float | None = None
        
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
        
        # 倉位計算器
        self.position_sizer: Optional[PositionSizer] = None
        self._init_position_sizer()
        
        # v2.7: 信號狀態持久化（防止滑動窗口導致的方向翻轉）
        self._signal_state_path = cfg.get_report_dir("live") / "signal_state.json"

        # v2.8: 增量 K 線快取（解決滑動窗口狀態機發散問題）
        self._kline_cache: IncrementalKlineCache | None = None
        if cfg.live.kline_cache:
            cache_dir = cfg.get_report_dir("live") / "kline_cache"
            self._kline_cache = IncrementalKlineCache(
                cache_dir=cache_dir,
                interval=self.interval,
                seed_bars=300,
                market_type=self.market_type,
            )
            logger.info("📦 增量 K 線快取已啟用")

    def _init_position_sizer(self) -> None:
        """
        根據配置初始化倉位計算器
        
        支援三種方法：
        - fixed: 固定倉位比例
        - kelly: 根據歷史交易統計動態調整
        - volatility: 根據波動率調整
        """
        ps_cfg = self.cfg.position_sizing
        
        if ps_cfg.method == "kelly":
            # 從歷史交易計算統計數據
            stats = self._get_trade_stats()
            
            # 檢查是否有足夠的交易數據
            total_trades = stats.get("total_trades", 0)
            if total_trades < ps_cfg.min_trades_for_kelly:
                logger.info(
                    f"📊 倉位計算: 交易數 ({total_trades}) < 最小要求 ({ps_cfg.min_trades_for_kelly})，"
                    f"暫用固定倉位"
                )
                self.position_sizer = FixedPositionSizer(ps_cfg.position_pct)
            else:
                win_rate = ps_cfg.win_rate or stats.get("win_rate", 0.5)
                avg_win = ps_cfg.avg_win or stats.get("avg_win", 1.0)
                avg_loss = ps_cfg.avg_loss or stats.get("avg_loss", 1.0)
                
                try:
                    self.position_sizer = KellyPositionSizer(
                        win_rate=win_rate,
                        avg_win=avg_win,
                        avg_loss=avg_loss,
                        kelly_fraction=ps_cfg.kelly_fraction,
                    )
                    logger.info(
                        f"📊 倉位計算: Kelly (fraction={ps_cfg.kelly_fraction}, "
                        f"win_rate={win_rate:.1%}, kelly_pct={self.position_sizer.kelly_pct:.1%})"
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
            logger.info(
                f"📊 倉位計算: 波動率目標 ({ps_cfg.target_volatility:.1%})"
            )
        else:
            # 預設固定倉位
            self.position_sizer = FixedPositionSizer(ps_cfg.position_pct)
            logger.info(f"📊 倉位計算: 固定 ({ps_cfg.position_pct:.0%})")
    
    def _get_trade_stats(self) -> dict:
        """
        從狀態管理器或 Paper Broker 獲取交易統計
        
        Returns:
            {"win_rate": float, "avg_win": float, "avg_loss": float, "total_trades": int}
        """
        # 優先從狀態管理器獲取
        if self.state_manager:
            stats = self.state_manager.get_trade_stats()
            stats["total_trades"] = self.state_manager.state.total_trades
            return stats
        
        # Paper Broker
        if isinstance(self.broker, PaperBroker):
            trades = self.broker.account.trades
            if not trades:
                return {"win_rate": 0.5, "avg_win": 1.0, "avg_loss": 1.0, "total_trades": 0}
            
            wins = [t for t in trades if t.pnl and t.pnl > 0]
            losses = [t for t in trades if t.pnl and t.pnl < 0]
            total = len(wins) + len(losses)
            
            return {
                "win_rate": len(wins) / total if total > 0 else 0.5,
                "avg_win": sum(t.pnl for t in wins) / len(wins) if wins else 1.0,
                "avg_loss": abs(sum(t.pnl for t in losses) / len(losses)) if losses else 1.0,
                "total_trades": len(trades),
            }
        
        return {"win_rate": 0.5, "avg_win": 1.0, "avg_loss": 1.0, "total_trades": 0}
    
    def _apply_position_sizing(self, raw_signal: float, price: float, symbol: str) -> float:
        """
        應用倉位計算器調整信號
        
        Args:
            raw_signal: 原始信號 [-1, 1]（Futures 可負；Spot 已在 run_once clip 到 [0,1]）
            price: 當前價格
            symbol: 交易對
            
        Returns:
            調整後的信號 [-1, 1]
        """
        if self.position_sizer is None:
            return raw_signal
        
        # 獲取當前權益
        if isinstance(self.broker, PaperBroker):
            equity = self.broker.get_equity({symbol: price})
        elif hasattr(self.broker, "get_equity"):
            # Futures broker: get_equity() 不需要參數
            # Spot broker: get_equity(symbols) 需要 symbols 列表
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
        
        # 限制在 [-1, 1]（Futures 可做空，Spot 的負信號已在 run_once 提前 clip）
        return max(-1.0, min(1.0, adjusted_signal))

    def _get_equity(self) -> float | None:
        """
        取得當前權益（Paper / Real 通用）

        Returns:
            當前權益 (USDT)，失敗時回傳 None
        """
        try:
            if isinstance(self.broker, PaperBroker):
                # Paper: 需要傳入當前價格
                prices: dict[str, float] = {}
                for sym in self.symbols:
                    pos = self.broker.get_position(sym)
                    if pos.is_open:
                        try:
                            from .signal_generator import fetch_recent_klines
                            df = fetch_recent_klines(sym, self.interval, 5)
                            prices[sym] = float(df["close"].iloc[-1])
                        except Exception:
                            return None  # 拿不到價格就不檢查
                return self.broker.get_equity(prices)
            else:
                # Real broker: 直接查 Binance API
                if hasattr(self.broker, "get_equity"):
                    return self.broker.get_equity()
            return None
        except Exception as e:
            logger.debug(f"取得權益失敗: {e}")
            return None

    def _check_circuit_breaker(self) -> bool:
        """
        Drawdown 熔斷檢查（Paper + Real 模式通用）

        基準權益 = 首次 tick 時的權益快照。
        如果當前權益低於 (1 - max_drawdown_pct) × 基準權益，
        平掉所有倉位並停止交易。

        Returns:
            True = 觸發熔斷，False = 正常
        """
        if self._circuit_breaker_triggered:
            return True
        if not self.max_drawdown_pct:
            return False

        equity = self._get_equity()
        if equity is None or equity <= 0:
            return False

        # 首次記錄基準權益
        if self._initial_equity is None:
            if isinstance(self.broker, PaperBroker):
                self._initial_equity = self.broker.account.initial_cash
            else:
                self._initial_equity = equity
            logger.info(f"📊 熔斷基準權益: ${self._initial_equity:,.2f}")

        initial = self._initial_equity
        drawdown = 1.0 - (equity / initial)

        if drawdown >= self.max_drawdown_pct:
            self._circuit_breaker_triggered = True
            logger.warning(
                f"🚨🚨🚨 CIRCUIT BREAKER 觸發！"
                f"Drawdown={drawdown:.1%} >= {self.max_drawdown_pct:.0%} "
                f"(權益 ${equity:,.2f} / 基準 ${initial:,.2f})"
            )

            # 平掉所有倉位
            for sym in self.symbols:
                try:
                    price = 0.0
                    if hasattr(self.broker, "get_price"):
                        price = self.broker.get_price(sym)
                    if price <= 0:
                        from .signal_generator import fetch_recent_klines
                        df = fetch_recent_klines(sym, self.interval, 5)
                        price = float(df["close"].iloc[-1])

                    current_pct = self.broker.get_position_pct(sym, price)
                    if abs(current_pct) > 0.01:
                        trade = self.broker.execute_target_position(
                            symbol=sym, target_pct=0.0, current_price=price,
                            reason="CIRCUIT_BREAKER"
                        )
                        if trade:
                            logger.warning(
                                f"  🔴 強制平倉 {sym}: {trade.qty:.6f} @ {trade.price:.2f}"
                            )
                except Exception as e:
                    logger.error(f"  ❌ 強制平倉 {sym} 失敗: {e}")

            # Telegram 告警
            self.notifier.send_error(
                f"🚨 <b>CIRCUIT BREAKER 熔斷觸發!</b>\n\n"
                f"  Drawdown: <b>{drawdown:.1%}</b> (閾值 {self.max_drawdown_pct:.0%})\n"
                f"  權益: ${equity:,.2f} → 基準: ${initial:,.2f}\n"
                f"  ⚠️ 已強制平倉所有持倉，交易停止\n\n"
                f"  請檢查策略後手動重啟"
            )
            return True

        # 接近熔斷線時預警（達到 80% 閾值）
        if drawdown >= self.max_drawdown_pct * 0.8:
            logger.warning(
                f"⚠️  Drawdown 預警: {drawdown:.1%} "
                f"(熔斷線 {self.max_drawdown_pct:.0%})"
            )

        return False

    # ── 信號狀態持久化（防止滑動窗口翻轉）──────────────

    def _load_signal_state(self) -> dict[str, float]:
        """
        載入上一次 cron 的信號方向。

        Returns:
            {symbol: signal_value}，例如 {"BTCUSDT": -1.0, "ETHUSDT": -1.0}
        """
        try:
            if self._signal_state_path.exists():
                with open(self._signal_state_path) as f:
                    data = json.load(f)
                return data.get("signals", {})
        except Exception as e:
            logger.debug(f"  載入信號狀態失敗: {e}")
        return {}

    def _save_signal_state(self, signal_map: dict[str, float]) -> None:
        """保存本次 cron 的信號方向到磁碟"""
        try:
            self._signal_state_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "signals": signal_map,
            }
            with open(self._signal_state_path, "w") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            logger.debug(f"  保存信號狀態失敗: {e}")

    def run_once(self) -> list[dict]:
        """
        執行一次信號檢查 + 下單

        Returns:
            signals: 所有幣種的信號列表
        """
        # 熔斷檢查
        if self._check_circuit_breaker():
            logger.warning("⛔ 熔斷已觸發，跳過本次交易")
            return []

        self.tick_count += 1
        signals = []
        has_trade = False
        
        # 更新狀態管理器
        if self.state_manager:
            self.state_manager.increment_tick()

        # v2.7: 載入上一次信號方向（用於方向切換確認）
        prev_signal_state = self._load_signal_state()
        new_signal_state: dict[str, float] = {}

        for symbol in self.symbols:
            params = self.cfg.strategy.get_params(symbol)

            # 生成信號（使用 AppConfig 集中屬性，確保 Futures 模式能做空）
            direction = self.cfg.direction

            try:
                # v2.8: 使用增量快取提供完整歷史，避免滑動窗口發散
                cached_df = None
                if self._kline_cache is not None:
                    cached_df = self._kline_cache.get_klines(symbol)
                    if cached_df is not None and len(cached_df) < 50:
                        logger.warning(
                            f"⚠️  {symbol}: 快取數據不足 ({len(cached_df)} bar)，"
                            f"fallback 到 fetch_recent_klines"
                        )
                        cached_df = None

                sig = generate_signal(
                    symbol=symbol,
                    strategy_name=self.strategy_name,
                    params=params,
                    interval=self.interval,
                    market_type=self.market_type,
                    direction=direction,
                    df=cached_df,  # None → generate_signal 內部自行拉 300 bar
                )
            except Exception as e:
                logger.error(f"❌ {symbol} 信號生成失敗: {e}")
                self.notifier.send_error(f"{symbol} 信號生成失敗: {e}")
                if self.state_manager:
                    self.state_manager.log_error(f"{symbol} 信號生成失敗: {e}")
                continue

            signals.append(sig)

            # 執行交易（信號 × 分配權重 × 倉位調整）
            raw_signal = sig["signal"]
            
            # Spot 模式：自動 clip 信號到 [0, 1]（不支援做空）
            # Futures 模式：保持 [-1, 1]
            if self.market_type == "spot" and raw_signal < 0:
                logger.debug(f"  {symbol}: Spot 模式不支援做空，信號 {raw_signal:.0%} clip 到 0")
                raw_signal = 0.0
            
            weight = self._weights.get(symbol, 1.0 / max(len(self.symbols), 1))
            price = sig["price"]
            if price <= 0:
                continue
            
            # 應用倉位計算器（如果啟用）
            adjusted_signal = self._apply_position_sizing(raw_signal, price, symbol)
            target_pct = adjusted_signal * weight

            current_pct = self.broker.get_position_pct(symbol, price)
            diff = abs(target_pct - current_pct)

            # v2.4+v2.7.1: SL/TP 冷卻檢查 + 孤兒掛單清理
            # 場景 A（v2.4）：SL/TP 觸發 → 倉位歸零 + 掛單消失 → 冷卻等下根 bar
            # 場景 B（v2.7.1）：SL 觸發 → 倉位歸零 + TP 殘留 → 先清掃孤兒再冷卻
            #   （Hedge Mode 下 SL 觸發平倉，但 TP 是獨立訂單不會自動取消）
            if (
                abs(current_pct) < 0.01              # 目前幾乎無倉
                and abs(target_pct) > 0.02            # 策略要求開倉
                and not isinstance(self.broker, PaperBroker)
                and hasattr(self.broker, "get_open_orders")
                and hasattr(self.broker, "get_trade_history")
            ):
                try:
                    # 合併 regular + algo orders 檢查 SL/TP
                    if hasattr(self.broker, "get_all_conditional_orders"):
                        cond_orders = self.broker.get_all_conditional_orders(symbol)
                    else:
                        cond_orders = self.broker.get_open_orders(symbol)
                    sl_tp_types = {"STOP_MARKET", "TAKE_PROFIT_MARKET", "STOP", "TAKE_PROFIT"}
                    has_sl_tp = any(o.get("type") in sl_tp_types for o in cond_orders)

                    # v2.7.1: 空倉 + 有殘留 SL/TP → 孤兒掛單
                    # 典型場景：SL 觸發平倉後，TP 殘留在交易所。
                    # 若不取消，開新倉位後舊 TP 可能干擾（同 positionSide）
                    # 或造成顯示混亂（不同 positionSide）。
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
                        has_sl_tp = False  # 已清理，視為無掛單

                    if not has_sl_tp:
                        # 無 SL/TP 掛單（或剛清理完孤兒） → 可能 SL/TP 剛觸發
                        # 檢查 10 分鐘窗口：SL/TP 通常在 bar 開頭觸發
                        recent_trades = self.broker.get_trade_history(symbol=symbol, limit=5)
                        now_ms = int(time.time() * 1000)
                        cooldown_ms = 10 * 60 * 1000  # 10 分鐘

                        recently_closed = any(
                            now_ms - t.get("time", 0) < cooldown_ms
                            for t in (recent_trades or [])
                        )
                        if recently_closed:
                            logger.warning(
                                f"⚠️  {symbol}: 無持倉且無 SL/TP 掛單，但最近 10min 有成交 → "
                                f"疑似 SL/TP 觸發，跳過本次開倉（冷卻等下根 bar）"
                            )
                            continue  # 跳到下一個 symbol
                except Exception as e:
                    logger.debug(f"  {symbol}: SL/TP 冷卻檢查失敗: {e}（繼續正常流程）")

            # === 防止權益波動導致的不必要重平衡 ===
            # 多幣種同時持倉時，一幣的 PnL 波動會改變 equity，
            # 導致另一幣的 current_pct 漂移（如 -100% → -103%），觸發不必要的微調。
            # 修正：方向一致且倉位已達目標的 80% 以上時，跳過重平衡。
            # 大幅差距（如 -32% → -100%）仍會正常執行加倉。
            if target_pct != 0 and current_pct != 0:
                same_direction = (
                    (target_pct > 0 and current_pct > 0) or
                    (target_pct < 0 and current_pct < 0)
                )
                if same_direction:
                    fill_ratio = abs(current_pct) / abs(target_pct)
                    if fill_ratio >= 0.80:  # 已達目標 80% 以上 → 跳過微調
                        diff = 0
                        logger.debug(
                            f"  {symbol}: 方向一致且倉位充足 "
                            f"({current_pct:+.1%} / {target_pct:+.1%} = {fill_ratio:.0%})，"
                            f"跳過重平衡"
                        )
                    else:
                        logger.info(
                            f"  {symbol}: 方向一致但倉位不足 "
                            f"({current_pct:+.1%} / {target_pct:+.1%} = {fill_ratio:.0%})，"
                            f"需要加倉"
                        )

            # v2.7→v2.8: 方向切換確認機制（可選）
            # kline_cache=True 時，數據穩定，不需要確認（預設關閉）
            # kline_cache=False 時，建議開啟，防止滑動窗口造成的頻繁翻轉
            is_direction_flip = (
                (target_pct > 0.01 and current_pct < -0.01) or   # SHORT → LONG
                (target_pct < -0.01 and current_pct > 0.01)      # LONG → SHORT
            )
            # 始終記錄本次原始信號（用於下一次確認判斷）
            new_signal_state[symbol] = sig["signal"]

            if is_direction_flip and self.cfg.live.flip_confirmation:
                prev_signal = prev_signal_state.get(symbol)
                if prev_signal is None:
                    # 首次運行 / 無狀態檔 → 直接執行（不阻擋首筆交易）
                    logger.info(
                        f"  {symbol}: 方向切換 (首次啟動，無前次信號) → 直接執行"
                    )
                else:
                    new_dir = 1 if target_pct > 0 else -1
                    prev_dir = 1 if prev_signal > 0 else (-1 if prev_signal < 0 else 0)

                    if prev_dir == new_dir:
                        # 前次信號也是同方向 → 已確認，執行
                        logger.info(
                            f"✅ {symbol}: 方向切換已確認 "
                            f"(前次={prev_signal:+.0%}, 本次={raw_signal:+.0%})"
                        )
                    else:
                        # 第一次出現新方向 → 保存但不執行
                        logger.warning(
                            f"⚠️  {symbol}: 方向切換待確認 "
                            f"(持倉={current_pct:+.0%} → 信號={raw_signal:+.0%}, "
                            f"前次信號={prev_signal:+.0%})"
                            f" — 保持原方向，下次確認後執行"
                        )
                        # 覆寫 target_pct 為維持原方向
                        if current_pct < 0:
                            target_pct = -1.0 * weight
                        else:
                            target_pct = 1.0 * weight
                        diff = abs(target_pct - current_pct)
                        # diff 通常 ≈ 0（方向一致且接近滿倉），不會觸發交易
            elif is_direction_flip:
                logger.info(
                    f"🔄 {symbol}: 方向切換 "
                    f"({current_pct:+.0%} → {raw_signal:+.0%}) — 直接執行"
                )

            if diff >= 0.02:
                ps_method = self.cfg.position_sizing.method
                reason = f"signal={raw_signal:.0%}×{weight:.0%}"
                if ps_method != "fixed":
                    reason += f" [{ps_method}→{adjusted_signal:.0%}]"
                
                # v2.3: 計算止損止盈價格（支援做多、做空、減倉後保護剩餘倉位）
                stop_loss_price = None
                take_profit_price = None
                stop_loss_atr = params.get("stop_loss_atr")
                take_profit_atr = params.get("take_profit_atr")
                atr_value = sig.get("indicators", {}).get("atr")
                
                if atr_value and target_pct != 0:
                    # 目標是多倉（不論是開多、加多、還是減倉後仍為多）
                    if target_pct > 0:
                        if stop_loss_atr:
                            stop_loss_price = price - float(stop_loss_atr) * float(atr_value)
                        if take_profit_atr:
                            take_profit_price = price + float(take_profit_atr) * float(atr_value)
                        if stop_loss_price or take_profit_price:
                            sl_str = f"${stop_loss_price:,.2f}" if stop_loss_price else "N/A"
                            tp_str = f"${take_profit_price:,.2f}" if take_profit_price else "N/A"
                            logger.info(f"🛡️  {symbol} [LONG] SL={sl_str}, TP={tp_str}")
                    # 目標是空倉（不論是開空、加空、還是減倉後仍為空）
                    elif target_pct < 0:
                        if stop_loss_atr:
                            stop_loss_price = price + float(stop_loss_atr) * float(atr_value)
                        if take_profit_atr:
                            take_profit_price = price - float(take_profit_atr) * float(atr_value)
                        if stop_loss_price or take_profit_price:
                            sl_str = f"${stop_loss_price:,.2f}" if stop_loss_price else "N/A"
                            tp_str = f"${take_profit_price:,.2f}" if take_profit_price else "N/A"
                            logger.info(f"🛡️  {symbol} [SHORT] SL={sl_str}, TP={tp_str}")
                    
                trade = self.broker.execute_target_position(
                    symbol=symbol,
                    target_pct=target_pct,
                    current_price=price,
                    reason=reason,
                    stop_loss_price=stop_loss_price,
                    take_profit_price=take_profit_price,
                )
                if trade:
                    self.trade_count += 1
                    has_trade = True
                    
                    # 記錄到狀態管理器
                    if self.state_manager:
                        self.state_manager.log_trade(
                            symbol=symbol,
                            side=trade.side,
                            qty=trade.qty,
                            price=trade.price,
                            fee=getattr(trade, "fee", 0.0),
                            pnl=trade.pnl,
                            reason=reason,
                            order_id=getattr(trade, "order_id", ""),
                        )
                        # 更新持倉
                        if isinstance(self.broker, PaperBroker):
                            pos = self.broker.get_position(symbol)
                            self.state_manager.update_position(symbol, pos.qty, pos.avg_entry)
                    
                    # Telegram 通知交易
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
            else:
                logger.debug(f"  {symbol}: 倉位不變 (target={target_pct:.0%}, current={current_pct:.0%})")

            # v2.5: SL/TP 補掛機制 — 確保有持倉就有 SL/TP 保護
            # 不論是否執行了交易，每次 cron 都檢查 SL/TP 是否存在
            # 場景：初次掛單 API 失敗、交易所清除掛單、手動取消等
            if (
                abs(current_pct) <= 0.01                        # 無持倉
                and not isinstance(self.broker, PaperBroker)
                and hasattr(self.broker, "_remove_algo_cache")
            ):
                # 清理殘留的 algo cache（SL/TP 被觸發後，cache 可能殘留）
                self.broker._remove_algo_cache(symbol)

            # v2.7: 重新讀取交易後的實際持倉（方向切換後 current_pct 可能已過時）
            actual_pct = current_pct
            if not isinstance(self.broker, PaperBroker) and hasattr(self.broker, "get_position_pct"):
                try:
                    actual_pct = self.broker.get_position_pct(symbol, price)
                except Exception:
                    pass  # 查詢失敗時用 pre-trade 值

            if (
                abs(actual_pct) > 0.01                            # 有持倉
                and not isinstance(self.broker, PaperBroker)      # 只對 Real broker
                and hasattr(self.broker, "place_stop_loss")
                and hasattr(self.broker, "get_open_orders")
            ):
                stop_loss_atr = params.get("stop_loss_atr")
                take_profit_atr = params.get("take_profit_atr")
                atr_value = sig.get("indicators", {}).get("atr")

                if (stop_loss_atr or take_profit_atr) and atr_value:
                    try:
                        # 合併 regular + algo orders 檢查
                        if hasattr(self.broker, "get_all_conditional_orders"):
                            cond_orders = self.broker.get_all_conditional_orders(symbol)
                        else:
                            cond_orders = self.broker.get_open_orders(symbol)
                        position_side = "LONG" if actual_pct > 0 else "SHORT"

                        # v2.7.1: 只看與當前持倉同方向的 SL/TP（Hedge Mode 下不同 positionSide 是獨立的）
                        def _match_side(o: dict) -> bool:
                            o_ps = o.get("positionSide", "")
                            return not o_ps or o_ps == position_side or o_ps == "BOTH"

                        has_sl = any(o.get("type") in {"STOP_MARKET", "STOP"} and _match_side(o) for o in cond_orders)
                        has_tp = any(o.get("type") in {"TAKE_PROFIT_MARKET", "TAKE_PROFIT"} and _match_side(o) for o in cond_orders)

                        # v2.7: 檢查殘留的方向錯誤 TP（翻倉後舊 TP 未取消）
                        # 例：LONG 持倉卻有 TP < entry → 觸發會虧損 → 必須取消
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
                                            f"🚨 {symbol}: 發現方向錯誤的 TP "
                                            f"${trigger:,.2f} "
                                            f"({'LONG' if is_long else 'SHORT'} 倉 "
                                            f"entry=${pos_check.entry_price:,.2f}) "
                                            f"— 自動取消"
                                        )
                                        self.broker.cancel_take_profit(symbol)
                                        has_tp = False
                                        break

                        if not has_sl and stop_loss_atr:
                            if actual_pct > 0:
                                sl_price = price - float(stop_loss_atr) * float(atr_value)
                            else:
                                sl_price = price + float(stop_loss_atr) * float(atr_value)
                            logger.info(f"🔄 {symbol}: 補掛止損單 SL=${sl_price:,.2f} [{position_side}]")
                            self.broker.place_stop_loss(
                                symbol=symbol, stop_price=sl_price,
                                position_side=position_side, reason="ensure_stop_loss",
                            )

                        if not has_tp and take_profit_atr:
                            if actual_pct > 0:
                                tp_price = price + float(take_profit_atr) * float(atr_value)
                            else:
                                tp_price = price - float(take_profit_atr) * float(atr_value)
                            logger.info(f"🔄 {symbol}: 補掛止盈單 TP=${tp_price:,.2f} [{position_side}]")
                            self.broker.place_take_profit(
                                symbol=symbol, take_profit_price=tp_price,
                                position_side=position_side, reason="ensure_take_profit",
                            )

                        if has_sl and (has_tp or not take_profit_atr):
                            logger.debug(f"  {symbol}: SL/TP 掛單正常 ✓")
                    except Exception as e:
                        logger.warning(f"⚠️  {symbol}: SL/TP 補掛檢查失敗: {e}")

            # 附加持倉 + SL/TP 資訊到 signal dict（供 Telegram 摘要使用）
            # 注意：需要查詢交易後的最新持倉，而非交易前的 current_pct
            if not isinstance(self.broker, PaperBroker) and hasattr(self.broker, "get_position"):
                try:
                    pos_obj = self.broker.get_position(symbol)
                    if pos_obj and abs(pos_obj.qty) > 1e-10:
                        live_pct = self.broker.get_position_pct(symbol, price)
                        sig["_position"] = {
                            "pct": live_pct,
                            "entry": pos_obj.entry_price,
                            "qty": abs(pos_obj.qty),
                            "side": "LONG" if pos_obj.qty > 0 else "SHORT",
                        }
                        # 查詢 SL/TP 掛單
                        if hasattr(self.broker, "get_all_conditional_orders"):
                            orders = self.broker.get_all_conditional_orders(symbol)
                            pos_side_str = "LONG" if pos_obj.qty > 0 else "SHORT"
                            for o in orders:
                                # v2.7.1: 只顯示與當前持倉同方向的 SL/TP
                                o_ps = o.get("positionSide", "")
                                if o_ps and o_ps != pos_side_str and o_ps != "BOTH":
                                    continue
                                otype = o.get("type", "")
                                trigger = float(o.get("stopPrice", 0) or o.get("triggerPrice", 0) or 0)
                                if trigger <= 0:
                                    continue
                                if otype in {"STOP_MARKET", "STOP"}:
                                    sig["_position"]["sl"] = trigger
                                elif otype in {"TAKE_PROFIT_MARKET", "TAKE_PROFIT"}:
                                    sig["_position"]["tp"] = trigger
                                elif pos_obj.entry_price > 0:
                                    # Algo orders fallback: 用觸發價 vs 入場價推斷
                                    is_long = pos_obj.qty > 0
                                    if is_long:
                                        if trigger < pos_obj.entry_price:
                                            sig["_position"]["sl"] = trigger
                                        else:
                                            sig["_position"]["tp"] = trigger
                                    else:
                                        if trigger > pos_obj.entry_price:
                                            sig["_position"]["sl"] = trigger
                                        else:
                                            sig["_position"]["tp"] = trigger
                    else:
                        sig["_position"] = {"pct": 0}  # 已平倉
                except Exception:
                    sig["_position"] = {"pct": current_pct}  # 查詢失敗用舊值
            else:
                sig["_position"] = {"pct": current_pct}

        # 發送信號摘要到 Telegram
        # --once 模式（cron）：每次都發，讓每小時都能看到信號狀態
        # 持續運行模式：有交易或每 6 tick 發送一次
        if has_trade or self.tick_count <= 1 or self.tick_count % 6 == 0:
            self.notifier.send_signal_summary(
                signals, 
                mode=self.mode.upper(),
                has_trade=has_trade,
            )
        
        # 保存信號快照（供 /signals 指令讀取，確保一致性）
        self._save_last_signals(signals)

        # v2.7: 保存信號方向（供下一次 cron 方向切換確認）
        self._save_signal_state(new_signal_state)

        # 每次 tick 都更新狀態檔時間戳（即使沒交易），讓健康檢查能偵測 cron 存活
        if isinstance(self.broker, PaperBroker):
            self.broker.touch_state()

        # 定期重新計算 Kelly（每 24 tick = 24 小時）
        if self.cfg.position_sizing.method == "kelly" and self.tick_count % 24 == 0:
            self._init_position_sizer()

        return signals

    def _save_last_signals(self, signals: list[dict]) -> None:
        """保存最新信號到 JSON，供 Telegram /signals 讀取"""
        try:
            sig_path = self.cfg.get_report_dir("live") / "last_signals.json"
            sig_path.parent.mkdir(parents=True, exist_ok=True)

            # 序列化信號（去掉不可 JSON 化的欄位）
            serializable = []
            for sig in signals:
                s = {
                    "symbol": sig.get("symbol"),
                    "signal": sig.get("signal"),
                    "price": sig.get("price"),
                    "timestamp": sig.get("timestamp"),
                    "strategy": sig.get("strategy"),
                    "indicators": sig.get("indicators", {}),
                    "_position": sig.get("_position", {}),
                    "_sltp": sig.get("_sltp", {}),
                }
                serializable.append(s)

            payload = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "mode": self.mode,
                "signals": serializable,
            }

            with open(sig_path, "w") as f:
                json.dump(payload, f, indent=2, default=str)

        except Exception as e:
            logger.debug(f"  保存信號快照失敗: {e}")

    def run(self, max_ticks: int | None = None) -> None:
        """
        阻塞運行主循環

        每根 K 線收盤後觸發一次 run_once()。
        通過 Ctrl+C 停止。

        Args:
            max_ticks: 最大運行次數（None = 無限）
        """
        self.is_running = True
        self.start_time = time.time()
        interval_seconds = self._interval_to_seconds(self.interval)

        alloc_str = ", ".join(f"{s}={w:.0%}" for s, w in self._weights.items())
        logger.info("=" * 60)
        logger.info(f"🚀 Live Trading 啟動 [{self.mode.upper()}]")
        logger.info(f"   策略: {self.strategy_name}")
        logger.info(f"   交易對: {', '.join(self.symbols)}")
        logger.info(f"   倉位分配: {alloc_str}")
        logger.info(f"   K線週期: {self.interval} ({interval_seconds}s)")
        logger.info(f"   模式: {'📝 Paper Trading' if self.mode == 'paper' else '💰 Real Trading'}")
        if self.max_drawdown_pct:
            logger.info(f"   熔斷線: 回撤 ≥ {self.max_drawdown_pct:.0%} → 自動平倉停止")
        logger.info(f"   K線快取: {'✅ 增量快取' if self._kline_cache else '❌ 滑動窗口 (300 bar)'}")
        logger.info(f"   翻轉確認: {'✅ 2-tick' if self.cfg.live.flip_confirmation else '❌ 直接執行'}")
        logger.info(f"   Telegram: {'✅ 已啟用' if self.notifier.enabled else '❌ 未啟用'}")
        logger.info("=" * 60)

        # 啟動通知
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
                # 計算到下一根 K 線收盤的等待時間
                wait = self._seconds_until_next_close(interval_seconds)
                if wait > 5:
                    logger.info(f"⏳ 等待下一根 K 線收盤... ({wait:.0f}s)")
                    # 分段 sleep，支援 Ctrl+C
                    while wait > 0 and self.is_running:
                        time.sleep(min(wait, 10))
                        wait -= 10
                else:
                    time.sleep(max(wait, 1))

                if not self.is_running:
                    break

                # 等幾秒確保 K 線數據已入庫
                time.sleep(3)

                now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                logger.info(f"\n{'─'*50}")
                logger.info(f"📍 Tick #{self.tick_count + 1} @ {now}")

                # 執行信號檢查
                self.run_once()

                # 熔斷觸發 → 停止循環
                if self._circuit_breaker_triggered:
                    logger.warning("🚨 熔斷觸發，主循環終止")
                    break

                # 定期列印 + 推送帳戶摘要（每 6 tick = 6 小時）
                if self.tick_count % 6 == 0:
                    self._send_periodic_summary()

                if max_ticks and self.tick_count >= max_ticks:
                    logger.info(f"🏁 達到最大運行次數 ({max_ticks})，停止")
                    break

        except KeyboardInterrupt:
            logger.info("\n⛔ 收到停止信號 (Ctrl+C)")
        finally:
            self.is_running = False
            elapsed = time.time() - (self.start_time or time.time())
            logger.info(f"📊 運行統計: {self.tick_count} ticks, "
                        f"{self.trade_count} trades, {elapsed/3600:.1f}h")
            # 停止通知
            self.notifier.send_shutdown(self.tick_count, self.trade_count, elapsed / 3600)

    def _send_periodic_summary(self) -> None:
        """定期推送帳戶摘要（支援 Paper + Real 模式）"""
        from .signal_generator import fetch_recent_klines

        if isinstance(self.broker, PaperBroker):
            # Paper 模式：從 K 線獲取價格計算權益
            prices = {}
            for sym in self.symbols:
                pos = self.broker.get_position(sym)
                if pos.is_open:
                    try:
                        df = fetch_recent_klines(sym, self.interval, 5)
                        prices[sym] = float(df["close"].iloc[-1])
                    except Exception:
                        pass
            if prices:
                summary = self.broker.summary(prices)
                logger.info(f"\n{summary}")

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
                    mode=self.mode.upper(),
                )
        else:
            # Real 模式：直接查 Binance API
            try:
                usdt = self.broker.get_balance("USDT")
                positions_info = {}
                total_value = usdt
                for sym in self.symbols:
                    pos = self.broker.get_position(sym)
                    if pos and pos.is_open:
                        price = self.broker.get_price(sym)
                        val = abs(pos.qty) * price
                        total_value += val
                        side = "LONG" if pos.qty > 0 else "SHORT"
                        positions_info[sym] = {
                            "qty": pos.qty,
                            "avg_entry": pos.entry_price,
                            "side": side,
                        }

                logger.info(
                    f"\n{'='*50}\n"
                    f"  Real Trading 帳戶摘要\n"
                    f"{'='*50}\n"
                    f"  USDT: ${usdt:,.2f}\n"
                    f"  總權益: ${total_value:,.2f}\n"
                    f"{'='*50}"
                )

                self.notifier.send_account_summary(
                    initial_cash=0,  # Real 模式沒有 initial_cash 概念
                    equity=total_value,
                    cash=usdt,
                    positions=positions_info,
                    trade_count=self.trade_count,
                    mode=self.mode.upper(),
                )
            except Exception as e:
                logger.warning(f"⚠️  獲取 Real 帳戶摘要失敗: {e}")

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
        # 下一個整週期時間
        next_close = (int(now / interval_seconds) + 1) * interval_seconds
        return max(next_close - now, 0)
