"""
Paper Trading Broker — 模拟下单引擎

功能：
    - 追踪虚拟现金和持仓
    - 模拟市价单（含手续费 + 滑点）
    - 记录每笔交易
    - 持久化状态到 JSON（可断线恢复）
"""
from __future__ import annotations
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal

from ..utils.log import get_logger

logger = get_logger("paper_broker")


@dataclass
class TradeRecord:
    timestamp: float        # unix epoch
    symbol: str
    side: str               # BUY / SELL
    qty: float
    price: float
    fee: float
    value: float            # price * qty
    pnl: float | None       # 平仓时计算
    reason: str = ""        # 开仓 / 止损 / 止盈 / 信号


@dataclass
class SymbolPosition:
    symbol: str
    qty: float = 0.0
    avg_entry: float = 0.0
    unrealized_pnl: float = 0.0

    @property
    def is_open(self) -> bool:
        return self.qty > 1e-10


@dataclass
class PaperAccount:
    initial_cash: float = 10_000.0
    cash: float = 10_000.0
    positions: dict[str, SymbolPosition] = field(default_factory=dict)
    trades: list[TradeRecord] = field(default_factory=list)
    fee_bps: float = 6.0
    slippage_bps: float = 5.0

    @property
    def fee_pct(self) -> float:
        return self.fee_bps / 10_000

    @property
    def slippage_pct(self) -> float:
        return self.slippage_bps / 10_000


class PaperBroker:
    """Paper Trading 模拟下单引擎"""

    def __init__(
        self,
        initial_cash: float = 10_000.0,
        fee_bps: float = 6.0,
        slippage_bps: float = 5.0,
        state_path: Path | str | None = None,
    ):
        self.account = PaperAccount(
            initial_cash=initial_cash,
            cash=initial_cash,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
        )
        self.state_path = Path(state_path) if state_path else None

        # 尝试从文件恢复状态
        if self.state_path and self.state_path.exists():
            self._load_state()
            logger.info(f"📂 恢复 Paper Trading 状态: cash={self.account.cash:.2f}, "
                        f"持仓={len([p for p in self.account.positions.values() if p.is_open])} 个")

    # ── 公开接口 ──────────────────────────────────────────

    def get_equity(self, prices: dict[str, float]) -> float:
        """计算总权益 = 现金 + 持仓市值"""
        equity = self.account.cash
        for sym, pos in self.account.positions.items():
            if pos.is_open and sym in prices:
                equity += pos.qty * prices[sym]
        return equity

    def get_position(self, symbol: str) -> SymbolPosition:
        if symbol not in self.account.positions:
            self.account.positions[symbol] = SymbolPosition(symbol=symbol)
        return self.account.positions[symbol]

    def get_position_pct(self, symbol: str, current_price: float) -> float:
        """获取某币种持仓占总权益的比例 [0, 1]"""
        pos = self.get_position(symbol)
        if not pos.is_open or current_price <= 0:
            return 0.0
        equity = self.get_equity({symbol: current_price})
        if equity <= 0:
            return 0.0
        return (pos.qty * current_price) / equity

    def execute_target_position(
        self,
        symbol: str,
        target_pct: float,
        current_price: float,
        reason: str = "signal",
    ) -> TradeRecord | None:
        """
        执行目标仓位调整

        将持仓调整到 target_pct（占总权益比例）。
        如果当前仓位已接近目标（差距 < 2%），不执行。

        Returns:
            TradeRecord 如果执行了交易，否则 None
        """
        target_pct = max(0.0, min(1.0, target_pct))
        current_pct = self.get_position_pct(symbol, current_price)

        # 差距太小不交易
        diff = target_pct - current_pct
        if abs(diff) < 0.02:
            return None

        equity = self.get_equity({symbol: current_price})

        if diff > 0:
            # 需要买入
            buy_value = diff * equity
            return self._buy(symbol, buy_value, current_price, reason)
        else:
            # 需要卖出
            sell_value = abs(diff) * equity
            return self._sell(symbol, sell_value, current_price, reason)

    # ── 内部方法 ──────────────────────────────────────────

    def _buy(self, symbol: str, value: float, price: float, reason: str) -> TradeRecord | None:
        # 滑点：买入价格更高
        exec_price = price * (1 + self.account.slippage_pct)
        qty = value / exec_price
        fee = value * self.account.fee_pct
        total_cost = value + fee

        if total_cost > self.account.cash:
            # 调整到可用现金
            total_cost = self.account.cash
            value = total_cost / (1 + self.account.fee_pct)
            fee = total_cost - value
            qty = value / exec_price

        if qty < 1e-10:
            return None

        self.account.cash -= total_cost

        pos = self.get_position(symbol)
        if pos.is_open:
            # 加仓：更新均价
            total_qty = pos.qty + qty
            pos.avg_entry = (pos.avg_entry * pos.qty + exec_price * qty) / total_qty
            pos.qty = total_qty
        else:
            pos.qty = qty
            pos.avg_entry = exec_price

        trade = TradeRecord(
            timestamp=time.time(),
            symbol=symbol,
            side="BUY",
            qty=qty,
            price=exec_price,
            fee=fee,
            value=value,
            pnl=None,
            reason=reason,
        )
        self.account.trades.append(trade)
        self._save_state()

        logger.info(f"📗 BUY  {symbol}: {qty:.6f} @ {exec_price:.2f} "
                    f"(fee={fee:.2f}, reason={reason})")
        return trade

    def _sell(self, symbol: str, value: float, price: float, reason: str) -> TradeRecord | None:
        pos = self.get_position(symbol)
        if not pos.is_open:
            return None

        # 滑点：卖出价格更低
        exec_price = price * (1 - self.account.slippage_pct)
        qty = min(value / exec_price, pos.qty)  # 不能卖超过持仓

        if qty < 1e-10:
            return None

        sell_value = qty * exec_price
        fee = sell_value * self.account.fee_pct
        self.account.cash += sell_value - fee

        pnl = (exec_price - pos.avg_entry) * qty - fee

        pos.qty -= qty
        if pos.qty < 1e-10:
            pos.qty = 0.0
            pos.avg_entry = 0.0

        trade = TradeRecord(
            timestamp=time.time(),
            symbol=symbol,
            side="SELL",
            qty=qty,
            price=exec_price,
            fee=fee,
            value=sell_value,
            pnl=pnl,
            reason=reason,
        )
        self.account.trades.append(trade)
        self._save_state()

        emoji = "📈" if pnl and pnl > 0 else "📉"
        logger.info(f"📕 SELL {symbol}: {qty:.6f} @ {exec_price:.2f} "
                    f"(fee={fee:.2f}, pnl={pnl:+.2f} {emoji}, reason={reason})")
        return trade

    # ── 状态持久化 ────────────────────────────────────────

    def _save_state(self) -> None:
        if not self.state_path:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "initial_cash": self.account.initial_cash,
            "cash": self.account.cash,
            "fee_bps": self.account.fee_bps,
            "slippage_bps": self.account.slippage_bps,
            "positions": {
                sym: {"qty": p.qty, "avg_entry": p.avg_entry}
                for sym, p in self.account.positions.items()
                if p.is_open
            },
            "trades": [
                {
                    "timestamp": t.timestamp,
                    "symbol": t.symbol,
                    "side": t.side,
                    "qty": t.qty,
                    "price": t.price,
                    "fee": t.fee,
                    "value": t.value,
                    "pnl": t.pnl,
                    "reason": t.reason,
                }
                for t in self.account.trades
            ],
        }
        with open(self.state_path, "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self) -> None:
        if not self.state_path or not self.state_path.exists():
            return
        with open(self.state_path) as f:
            state = json.load(f)
        self.account.initial_cash = state["initial_cash"]
        self.account.cash = state["cash"]
        self.account.fee_bps = state.get("fee_bps", 6.0)
        self.account.slippage_bps = state.get("slippage_bps", 5.0)
        for sym, pdata in state.get("positions", {}).items():
            self.account.positions[sym] = SymbolPosition(
                symbol=sym, qty=pdata["qty"], avg_entry=pdata["avg_entry"]
            )
        for tdata in state.get("trades", []):
            self.account.trades.append(TradeRecord(**tdata))

    # ── 报告 ─────────────────────────────────────────────

    def summary(self, prices: dict[str, float]) -> str:
        equity = self.get_equity(prices)
        ret = (equity / self.account.initial_cash - 1) * 100
        lines = [
            "=" * 50,
            f"  Paper Trading 账户摘要",
            "=" * 50,
            f"  初始资金:   ${self.account.initial_cash:,.2f}",
            f"  当前现金:   ${self.account.cash:,.2f}",
            f"  总权益:     ${equity:,.2f}",
            f"  总收益:     {ret:+.2f}%",
            f"  交易笔数:   {len(self.account.trades)}",
        ]
        for sym, pos in self.account.positions.items():
            if pos.is_open:
                price = prices.get(sym, 0)
                pnl = (price - pos.avg_entry) * pos.qty if price > 0 else 0
                lines.append(f"  {sym}: {pos.qty:.6f} @ {pos.avg_entry:.2f} (PnL: {pnl:+.2f})")
        lines.append("=" * 50)
        return "\n".join(lines)

