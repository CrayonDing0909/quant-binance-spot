"""
Live Runner — 即时交易主循环

功能：
    - 每根 K 线收盘后运行策略
    - 对比信号与当前仓位，决定交易
    - 支持 Paper Trading / Real Trading 模式切换
    - Telegram 通知（交易 + 定期摘要）
    - 日志记录 + 状态报告
"""
from __future__ import annotations
import time
from datetime import datetime, timezone
from typing import Protocol

from ..config import AppConfig
from ..utils.log import get_logger
from ..monitor.notifier import TelegramNotifier
from .signal_generator import generate_signal
from .paper_broker import PaperBroker

logger = get_logger("live_runner")


class BrokerProtocol(Protocol):
    """Broker 通用接口，Paper 和 Real broker 都实现此接口"""
    def execute_target_position(
        self, symbol: str, target_pct: float, current_price: float, reason: str = ""
    ) -> object | None: ...

    def get_position_pct(self, symbol: str, current_price: float) -> float: ...


class LiveRunner:
    """
    即时交易主循环

    Usage:
        runner = LiveRunner(cfg, broker, mode="paper")
        runner.run()  # 阻塞运行，每根 K 线触发一次
    """

    def __init__(
        self,
        cfg: AppConfig,
        broker: BrokerProtocol,
        mode: str = "paper",
        notifier: TelegramNotifier | None = None,
    ):
        self.cfg = cfg
        self.broker = broker
        self.mode = mode
        self.notifier = notifier or TelegramNotifier()
        self.strategy_name = cfg.strategy.name
        self.symbols = cfg.market.symbols
        self.interval = cfg.market.interval
        self.is_running = False

        # 多币种仓位分配权重
        self._weights: dict[str, float] = {}
        n = len(self.symbols)
        for sym in self.symbols:
            self._weights[sym] = cfg.portfolio.get_weight(sym, n)

        # Drawdown 熔断
        self.max_drawdown_pct = cfg.risk.max_drawdown_pct if cfg.risk else None
        self._circuit_breaker_triggered = False

        # 运行统计
        self.tick_count = 0
        self.trade_count = 0
        self.start_time: float | None = None

    def _check_circuit_breaker(self) -> bool:
        """
        Drawdown 熔断检查

        如果当前权益低于 (1 - max_drawdown_pct) × 初始资金，
        平掉所有仓位并停止交易。

        Returns:
            True = 触发熔断，False = 正常
        """
        if self._circuit_breaker_triggered:
            return True
        if not self.max_drawdown_pct:
            return False
        # 熔断只支持 Paper 模式（Real 模式靠手动管理）
        if not isinstance(self.broker, PaperBroker):
            return False

        # 获取当前价格
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
                    logger.warning(f"⚠️  获取 {sym} 价格失败: {e}")

        # 如果有持仓但抓不到价格，跳过熔断检查（避免假性触发）
        if open_positions and len(prices) < len(open_positions):
            missing = set(open_positions) - set(prices.keys())
            logger.warning(
                f"⚠️  熔断检查跳过：无法获取 {missing} 的价格，"
                f"无法准确计算权益"
            )
            return False

        equity = self.broker.get_equity(prices)
        initial = self.broker.account.initial_cash
        drawdown = 1.0 - (equity / initial)

        if drawdown >= self.max_drawdown_pct:
            self._circuit_breaker_triggered = True
            logger.warning(
                f"🚨🚨🚨 CIRCUIT BREAKER 触发！"
                f"Drawdown={drawdown:.1%} >= {self.max_drawdown_pct:.0%} "
                f"(权益 ${equity:,.2f} / 初始 ${initial:,.2f})"
            )

            # 平掉所有仓位
            for sym, price in prices.items():
                pos = self.broker.get_position(sym)
                if pos.is_open:
                    trade = self.broker.execute_target_position(
                        symbol=sym, target_pct=0.0, current_price=price,
                        reason="CIRCUIT_BREAKER"
                    )
                    if trade:
                        logger.warning(f"  🔴 强制平仓 {sym}: {trade.qty:.6f} @ {trade.price:.2f}")

            # Telegram 告警
            self.notifier.send_error(
                f"🚨 <b>CIRCUIT BREAKER 熔断触发!</b>\n\n"
                f"  Drawdown: <b>{drawdown:.1%}</b> (阈值 {self.max_drawdown_pct:.0%})\n"
                f"  权益: ${equity:,.2f} → 初始: ${initial:,.2f}\n"
                f"  ⚠️ 已强制平仓所有持仓，交易停止\n\n"
                f"  请检查策略后手动重启"
            )
            return True

        # 接近熔断线时预警（达到 80% 阈值）
        if drawdown >= self.max_drawdown_pct * 0.8:
            logger.warning(
                f"⚠️  Drawdown 预警: {drawdown:.1%} "
                f"(熔断线 {self.max_drawdown_pct:.0%})"
            )

        return False

    def run_once(self) -> list[dict]:
        """
        执行一次信号检查 + 下单

        Returns:
            signals: 所有币种的信号列表
        """
        # 熔断检查
        if self._check_circuit_breaker():
            logger.warning("⛔ 熔断已触发，跳过本次交易")
            return []

        self.tick_count += 1
        signals = []
        has_trade = False

        for symbol in self.symbols:
            params = self.cfg.strategy.get_params(symbol)

            # 生成信号
            try:
                sig = generate_signal(
                    symbol=symbol,
                    strategy_name=self.strategy_name,
                    params=params,
                    interval=self.interval,
                )
            except Exception as e:
                logger.error(f"❌ {symbol} 信号生成失败: {e}")
                self.notifier.send_error(f"{symbol} 信号生成失败: {e}")
                continue

            signals.append(sig)

            # 执行交易（信号 × 分配权重）
            raw_signal = sig["signal"]
            weight = self._weights.get(symbol, 1.0 / max(len(self.symbols), 1))
            target_pct = raw_signal * weight
            price = sig["price"]
            if price <= 0:
                continue

            current_pct = self.broker.get_position_pct(symbol, price)
            diff = abs(target_pct - current_pct)

            if diff >= 0.02:
                reason = f"signal={raw_signal:.0%}×{weight:.0%}"
                trade = self.broker.execute_target_position(
                    symbol=symbol,
                    target_pct=target_pct,
                    current_price=price,
                    reason=reason,
                )
                if trade:
                    self.trade_count += 1
                    has_trade = True
                    # Telegram 通知交易
                    self.notifier.send_trade(
                        symbol=symbol,
                        side=trade.side,
                        qty=trade.qty,
                        price=trade.price,
                        reason=reason,
                        pnl=trade.pnl,
                        weight=weight,
                    )
            else:
                logger.debug(f"  {symbol}: 仓位不变 (target={target_pct:.0%}, current={current_pct:.0%})")

        # 每个 tick 发送信号摘要（仅当有交易或每 6 tick）
        if has_trade or self.tick_count % 6 == 0:
            self.notifier.send_signal_summary(signals, mode=self.mode.upper())

        return signals

    def run(self, max_ticks: int | None = None) -> None:
        """
        阻塞运行主循环

        每根 K 线收盘后触发一次 run_once()。
        通过 Ctrl+C 停止。

        Args:
            max_ticks: 最大运行次数（None = 无限）
        """
        self.is_running = True
        self.start_time = time.time()
        interval_seconds = self._interval_to_seconds(self.interval)

        alloc_str = ", ".join(f"{s}={w:.0%}" for s, w in self._weights.items())
        logger.info("=" * 60)
        logger.info(f"🚀 Live Trading 启动 [{self.mode.upper()}]")
        logger.info(f"   策略: {self.strategy_name}")
        logger.info(f"   交易对: {', '.join(self.symbols)}")
        logger.info(f"   仓位分配: {alloc_str}")
        logger.info(f"   K线周期: {self.interval} ({interval_seconds}s)")
        logger.info(f"   模式: {'📝 Paper Trading' if self.mode == 'paper' else '💰 Real Trading'}")
        if self.max_drawdown_pct:
            logger.info(f"   熔断线: 回撤 ≥ {self.max_drawdown_pct:.0%} → 自动平仓停止")
        logger.info(f"   Telegram: {'✅ 已启用' if self.notifier.enabled else '❌ 未启用'}")
        logger.info("=" * 60)

        # 启动通知
        self.notifier.send_startup(
            strategy=self.strategy_name,
            symbols=self.symbols,
            interval=self.interval,
            mode=self.mode,
            weights=self._weights,
        )

        try:
            while self.is_running:
                # 计算到下一根 K 线收盘的等待时间
                wait = self._seconds_until_next_close(interval_seconds)
                if wait > 5:
                    logger.info(f"⏳ 等待下一根 K 线收盘... ({wait:.0f}s)")
                    # 分段 sleep，支持 Ctrl+C
                    while wait > 0 and self.is_running:
                        time.sleep(min(wait, 10))
                        wait -= 10
                else:
                    time.sleep(max(wait, 1))

                if not self.is_running:
                    break

                # 等几秒确保 K 线数据已入库
                time.sleep(3)

                now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                logger.info(f"\n{'─'*50}")
                logger.info(f"📍 Tick #{self.tick_count + 1} @ {now}")

                # 执行信号检查
                self.run_once()

                # 熔断触发 → 停止循环
                if self._circuit_breaker_triggered:
                    logger.warning("🚨 熔断触发，主循环终止")
                    break

                # 定期打印 + 推送账户摘要（每 6 tick = 6 小时）
                if self.tick_count % 6 == 0:
                    self._send_periodic_summary()

                if max_ticks and self.tick_count >= max_ticks:
                    logger.info(f"🏁 达到最大运行次数 ({max_ticks})，停止")
                    break

        except KeyboardInterrupt:
            logger.info("\n⛔ 收到停止信号 (Ctrl+C)")
        finally:
            self.is_running = False
            elapsed = time.time() - (self.start_time or time.time())
            logger.info(f"📊 运行统计: {self.tick_count} ticks, "
                        f"{self.trade_count} trades, {elapsed/3600:.1f}h")
            # 停止通知
            self.notifier.send_shutdown(self.tick_count, self.trade_count, elapsed / 3600)

    def _send_periodic_summary(self) -> None:
        """定期推送账户摘要（支持 Paper + Real 模式）"""
        from .signal_generator import fetch_recent_klines

        if isinstance(self.broker, PaperBroker):
            # Paper 模式：从 K 线获取价格计算权益
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
                    qty = self.broker.get_position(sym)
                    if qty > 0:
                        price = self.broker.get_price(sym)
                        val = qty * price
                        total_value += val
                        positions_info[sym] = {"qty": qty, "avg_entry": price}

                logger.info(
                    f"\n{'='*50}\n"
                    f"  Real Trading 账户摘要\n"
                    f"{'='*50}\n"
                    f"  USDT: ${usdt:,.2f}\n"
                    f"  总权益: ${total_value:,.2f}\n"
                    f"{'='*50}"
                )

                self.notifier.send_account_summary(
                    initial_cash=0,  # Real 模式没有 initial_cash 概念
                    equity=total_value,
                    cash=usdt,
                    positions=positions_info,
                    trade_count=self.trade_count,
                    mode=self.mode.upper(),
                )
            except Exception as e:
                logger.warning(f"⚠️  获取 Real 账户摘要失败: {e}")

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
        # 下一个整周期时间
        next_close = (int(now / interval_seconds) + 1) * interval_seconds
        return max(next_close - now, 0)
