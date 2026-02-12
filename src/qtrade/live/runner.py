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

        # Drawdown 熔斷
        self.max_drawdown_pct = cfg.risk.max_drawdown_pct if cfg.risk else None
        self._circuit_breaker_triggered = False

        # 運行統計
        self.tick_count = 0
        self.trade_count = 0
        self.start_time: float | None = None
        
        # 狀態管理器（用於 Real Trading 持久化）
        self.state_manager: Optional[TradingStateManager] = None
        if state_path or mode == "real":
            default_state_path = Path(f"reports/live/{self.strategy_name}/{mode}_state.json")
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

    def _check_circuit_breaker(self) -> bool:
        """
        Drawdown 熔斷檢查

        如果當前權益低於 (1 - max_drawdown_pct) × 初始資金，
        平掉所有倉位並停止交易。

        Returns:
            True = 觸發熔斷，False = 正常
        """
        if self._circuit_breaker_triggered:
            return True
        if not self.max_drawdown_pct:
            return False
        # 熔斷只支援 Paper 模式（Real 模式靠手動管理）
        if not isinstance(self.broker, PaperBroker):
            return False

        # 獲取當前價格
        prices: dict[str, float] = {}
        open_positions = []
        for sym in self.symbols:
            pos = self.broker.get_position(sym)
            if pos.is_open:
                open_positions.append(sym)
                try:
                    from .signal_generator import fetch_recent_klines
                    df = fetch_recent_klines(sym, self.interval, 5)
                    prices[sym] = float(df["close"].iloc[-1])
                except Exception as e:
                    logger.warning(f"⚠️  獲取 {sym} 價格失敗: {e}")

        # 如果有持倉但抓不到價格，跳過熔斷檢查（避免假性觸發）
        if open_positions and len(prices) < len(open_positions):
            missing = set(open_positions) - set(prices.keys())
            logger.warning(
                f"⚠️  熔斷檢查跳過：無法獲取 {missing} 的價格，"
                f"無法準確計算權益"
            )
            return False

        equity = self.broker.get_equity(prices)
        initial = self.broker.account.initial_cash
        drawdown = 1.0 - (equity / initial)

        if drawdown >= self.max_drawdown_pct:
            self._circuit_breaker_triggered = True
            logger.warning(
                f"🚨🚨🚨 CIRCUIT BREAKER 觸發！"
                f"Drawdown={drawdown:.1%} >= {self.max_drawdown_pct:.0%} "
                f"(權益 ${equity:,.2f} / 初始 ${initial:,.2f})"
            )

            # 平掉所有倉位
            for sym, price in prices.items():
                pos = self.broker.get_position(sym)
                if pos.is_open:
                    trade = self.broker.execute_target_position(
                        symbol=sym, target_pct=0.0, current_price=price,
                        reason="CIRCUIT_BREAKER"
                    )
                    if trade:
                        logger.warning(f"  🔴 強制平倉 {sym}: {trade.qty:.6f} @ {trade.price:.2f}")

            # Telegram 告警
            self.notifier.send_error(
                f"🚨 <b>CIRCUIT BREAKER 熔斷觸發!</b>\n\n"
                f"  Drawdown: <b>{drawdown:.1%}</b> (閾值 {self.max_drawdown_pct:.0%})\n"
                f"  權益: ${equity:,.2f} → 初始: ${initial:,.2f}\n"
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

        for symbol in self.symbols:
            params = self.cfg.strategy.get_params(symbol)

            # 生成信號（使用 AppConfig 集中屬性，確保 Futures 模式能做空）
            direction = self.cfg.direction

            try:
                sig = generate_signal(
                    symbol=symbol,
                    strategy_name=self.strategy_name,
                    params=params,
                    interval=self.interval,
                    market_type=self.market_type,
                    direction=direction,
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

            # v2.4: SL/TP 冷卻檢查 — 防止交易所止損後立刻重新進場
            # 場景：bar 開頭幾分鐘 SL/TP 被觸發 → 倉位歸零 → cron 用舊 bar 信號重新開倉
            # 檢測邏輯：
            #   1. 策略要求開倉（target ≠ 0）但 broker 無持倉（current ≈ 0）
            #   2. 交易所沒有 SL/TP 掛單（已被消耗）
            #   3. 最近 10 分鐘內有成交紀錄（= SL/TP 剛被觸發）
            #   → 跳過本次開倉，等下根 bar 讓策略重新確認
            if (
                abs(current_pct) < 0.01              # 目前幾乎無倉
                and abs(target_pct) > 0.02            # 策略要求開倉
                and not isinstance(self.broker, PaperBroker)
                and hasattr(self.broker, "get_open_orders")
                and hasattr(self.broker, "get_trade_history")
            ):
                try:
                    open_orders = self.broker.get_open_orders(symbol)
                    sl_tp_types = {"STOP_MARKET", "TAKE_PROFIT_MARKET", "STOP", "TAKE_PROFIT"}
                    has_sl_tp = any(o.get("type") in sl_tp_types for o in open_orders)

                    if not has_sl_tp:
                        # 無 SL/TP 掛單 → 可能剛被觸發，查最近成交
                        # 只檢查 10 分鐘窗口：SL/TP 只會在 bar 開頭 (xx:00~xx:05) 觸發
                        # 而上一次 cron 的平倉在 ~55 分鐘前，不會誤判
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

        # 發送信號摘要到 Telegram
        # --once 模式（cron）：每次都發，讓每小時都能看到信號狀態
        # 持續運行模式：有交易或每 6 tick 發送一次
        if has_trade or self.tick_count <= 1 or self.tick_count % 6 == 0:
            self.notifier.send_signal_summary(
                signals, 
                mode=self.mode.upper(),
                has_trade=has_trade,
            )
        
        # 每次 tick 都更新狀態檔時間戳（即使沒交易），讓健康檢查能偵測 cron 存活
        if isinstance(self.broker, PaperBroker):
            self.broker.touch_state()

        # 定期重新計算 Kelly（每 24 tick = 24 小時）
        if self.cfg.position_sizing.method == "kelly" and self.tick_count % 24 == 0:
            self._init_position_sizer()

        return signals

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
