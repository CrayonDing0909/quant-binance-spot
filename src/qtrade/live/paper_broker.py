"""
Paper Trading Broker — 模擬下單引擎

功能：
    - 追蹤虛擬現金和持倉
    - 模擬市價單（含手續費 + 滑點）
    - 記錄每筆交易
    - 持久化狀態到 JSON（可斷線恢復）
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
    pnl: float | None       # 平倉時計算
    reason: str = ""        # 開倉 / 止損 / 止盈 / 信號


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
    """Paper Trading 模擬下單引擎"""

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

        # 嘗試從檔案恢復狀態
        if self.state_path and self.state_path.exists():
            self._load_state()
            logger.info(f"📂 恢復 Paper Trading 狀態: cash={self.account.cash:.2f}, "
                        f"持倉={len([p for p in self.account.positions.values() if p.is_open])} 個")

    # ── 公開介面 ──────────────────────────────────────────

    def get_equity(self, prices: dict[str, float]) -> float:
        """計算總權益 = 現金 + 持倉市值"""
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
        """獲取某幣種持倉佔總權益的比例 [0, 1]"""
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
        stop_loss_price: float | None = None,  # v2.0: 介面對齊（Paper 模式不使用）
    ) -> TradeRecord | None:
        """
        執行目標倉位調整

        將持倉調整到 target_pct（佔總權益比例）。
        如果當前倉位已接近目標（差距 < 2%），不執行。
        
        Note:
            stop_loss_price 在 Paper 模式不使用（回測已模擬止損邏輯），
            僅用於與 BinanceSpotBroker 介面對齊。

        Returns:
            TradeRecord 如果執行了交易，否則 None
        """
        target_pct = max(0.0, min(1.0, target_pct))
        current_pct = self.get_position_pct(symbol, current_price)

        # 差距太小不交易
        diff = target_pct - current_pct
        if abs(diff) < 0.02:
            return None

        equity = self.get_equity({symbol: current_price})

        if diff > 0:
            # 需要買入
            buy_value = diff * equity
            return self._buy(symbol, buy_value, current_price, reason)
        else:
            # 需要賣出
            sell_value = abs(diff) * equity
            return self._sell(symbol, sell_value, current_price, reason)

    # ── 內部方法 ──────────────────────────────────────────

    def _buy(self, symbol: str, value: float, price: float, reason: str) -> TradeRecord | None:
        # 滑點：買入價格更高
        exec_price = price * (1 + self.account.slippage_pct)
        qty = value / exec_price
        fee = value * self.account.fee_pct
        total_cost = value + fee

        if total_cost > self.account.cash:
            # 調整到可用現金
            total_cost = self.account.cash
            value = total_cost / (1 + self.account.fee_pct)
            fee = total_cost - value
            qty = value / exec_price

        if qty < 1e-10:
            return None

        self.account.cash -= total_cost

        pos = self.get_position(symbol)
        if pos.is_open:
            # 加倉：更新均價
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

        # 滑點：賣出價格更低
        exec_price = price * (1 - self.account.slippage_pct)
        qty = min(value / exec_price, pos.qty)  # 不能賣超過持倉

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

    # ── 狀態持久化 ────────────────────────────────────────

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

    # ── 報告 ─────────────────────────────────────────────

    def summary(self, prices: dict[str, float]) -> str:
        equity = self.get_equity(prices)
        ret = (equity / self.account.initial_cash - 1) * 100
        lines = [
            "=" * 50,
            f"  Paper Trading 帳戶摘要",
            "=" * 50,
            f"  初始資金:   ${self.account.initial_cash:,.2f}",
            f"  當前現金:   ${self.account.cash:,.2f}",
            f"  總權益:     ${equity:,.2f}",
            f"  總收益:     {ret:+.2f}%",
            f"  交易筆數:   {len(self.account.trades)}",
        ]
        for sym, pos in self.account.positions.items():
            if pos.is_open:
                price = prices.get(sym, 0)
                pnl = (price - pos.avg_entry) * pos.qty if price > 0 else 0
                lines.append(f"  {sym}: {pos.qty:.6f} @ {pos.avg_entry:.2f} (PnL: {pnl:+.2f})")
        lines.append("=" * 50)
        return "\n".join(lines)
