"""
Paper Trading Broker — 模擬下單引擎

功能：
    - 追蹤虛擬現金和持倉
    - 模擬市價單（含手續費 + 滑點）
    - 記錄每筆交易
    - 持久化狀態到 JSON（可斷線恢復）
    - 支援做空（Futures 模式）

做空模擬機制：
    - qty > 0: 做多（LONG）
    - qty < 0: 做空（SHORT）
    - qty = 0: 空倉
    
    做空時：
    - 開空倉 = 借入資產賣出，收到現金
    - 平空倉 = 買回資產還回，支付現金
    - 盈虧 = (開倉價 - 平倉價) × 數量
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
    side: str               # BUY / SELL / LONG / SHORT / CLOSE_LONG / CLOSE_SHORT
    qty: float
    price: float
    fee: float
    value: float            # price * qty
    pnl: float | None       # 平倉時計算
    reason: str = ""        # 開倉 / 止損 / 止盈 / 信號


@dataclass
class SymbolPosition:
    """
    持倉資訊
    
    Attributes:
        qty: 持倉數量，正數 = 做多，負數 = 做空
        avg_entry: 平均開倉價格
    """
    symbol: str
    qty: float = 0.0
    avg_entry: float = 0.0
    unrealized_pnl: float = 0.0

    @property
    def is_open(self) -> bool:
        """是否有持倉（多或空）"""
        return abs(self.qty) > 1e-10

    @property
    def is_long(self) -> bool:
        """是否持有多倉"""
        return self.qty > 1e-10

    @property
    def is_short(self) -> bool:
        """是否持有空倉"""
        return self.qty < -1e-10

    @property
    def side(self) -> str:
        """倉位方向：LONG / SHORT / NONE"""
        if self.is_long:
            return "LONG"
        elif self.is_short:
            return "SHORT"
        return "NONE"


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
    """
    Paper Trading 模擬下單引擎
    
    支援：
    - Spot 模式：只能做多 [0, 1]
    - Futures 模式：可做多做空 [-1, 1]
    """

    def __init__(
        self,
        initial_cash: float = 10_000.0,
        fee_bps: float = 6.0,
        slippage_bps: float = 5.0,
        state_path: Path | str | None = None,
        market_type: str = "spot",
        leverage: int = 1,
    ):
        """
        Args:
            initial_cash: 初始資金
            fee_bps: 手續費（基點）
            slippage_bps: 滑點（基點）
            state_path: 狀態檔路徑
            market_type: "spot" 或 "futures"
            leverage: 槓桿倍數（僅 futures 有效）
        """
        self.account = PaperAccount(
            initial_cash=initial_cash,
            cash=initial_cash,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
        )
        self.state_path = Path(state_path) if state_path else None
        self.market_type = market_type
        self.leverage = leverage if market_type == "futures" else 1
        self.supports_short = (market_type == "futures")

        # 嘗試從檔案恢復狀態
        if self.state_path and self.state_path.exists():
            self._load_state()
            logger.info(f"📂 恢復 Paper Trading 狀態: cash={self.account.cash:.2f}, "
                        f"持倉={len([p for p in self.account.positions.values() if p.is_open])} 個")

    # ── 公開介面 ──────────────────────────────────────────

    def get_equity(self, prices: dict[str, float]) -> float:
        """
        計算總權益
        
        Spot: 權益 = 現金 + 持倉市值
        Futures: 權益 = 現金 + 保證金 + 未實現盈虧
            - 多倉：市值 = 數量 × 現價
            - 空倉：保證金 + 未實現盈虧
              - 保證金 = |數量| × 開倉價 / 槓桿
              - 未實現盈虧 = (開倉價 - 現價) × |數量|
        """
        equity = self.account.cash
        for sym, pos in self.account.positions.items():
            if pos.is_open and sym in prices:
                price = prices[sym]
                if pos.is_long:
                    # 多倉：加上市值
                    equity += pos.qty * price
                elif pos.is_short:
                    # 空倉：加上保證金（開倉時扣除的）+ 未實現盈虧
                    margin_locked = abs(pos.qty) * pos.avg_entry / self.leverage
                    unrealized_pnl = (pos.avg_entry - price) * abs(pos.qty)
                    equity += margin_locked + unrealized_pnl
        return equity

    def get_position(self, symbol: str) -> SymbolPosition:
        if symbol not in self.account.positions:
            self.account.positions[symbol] = SymbolPosition(symbol=symbol)
        return self.account.positions[symbol]

    def get_position_pct(self, symbol: str, current_price: float) -> float:
        """
        獲取某幣種持倉佔總權益的比例
        
        Returns:
            Spot: [0, 1]，0 = 空倉，1 = 滿倉做多
            Futures: [-1, 1]，-1 = 滿倉做空，0 = 空倉，1 = 滿倉做多
        """
        pos = self.get_position(symbol)
        if not pos.is_open or current_price <= 0:
            return 0.0
        equity = self.get_equity({symbol: current_price})
        if equity <= 0:
            return 0.0
        
        # 計算倉位價值佔權益的比例（保留正負號）
        position_value = pos.qty * current_price
        return position_value / equity

    def execute_target_position(
        self,
        symbol: str,
        target_pct: float,
        current_price: float | None = None,
        reason: str = "signal",
        stop_loss_price: float | None = None,  # v2.0: 介面對齊（Paper 模式不使用）
        take_profit_price: float | None = None,  # v2.0: 介面對齊（Paper 模式不使用）
    ) -> TradeRecord | None:
        """
        執行目標倉位調整

        將持倉調整到 target_pct（佔總權益比例）。
        如果當前倉位已接近目標（差距 < 2%），不執行。
        
        Args:
            target_pct: 目標倉位比例
                - Spot 模式: [0, 1]
                - Futures 模式: [-1, 1]，負數表示做空
            current_price: 當前價格
            reason: 交易原因
            stop_loss_price: 止損價格（Paper 模式不使用）

        Returns:
            TradeRecord 如果執行了交易，否則 None

        Raises:
            ValueError: 如果 current_price 為 None（Paper 模式必須提供價格）
        """
        if current_price is None:
            raise ValueError(
                f"PaperBroker.execute_target_position: {symbol} 必須提供 current_price"
            )

        # 根據市場類型限制 target_pct 範圍
        if self.supports_short:
            target_pct = max(-1.0, min(1.0, target_pct))
        else:
            target_pct = max(0.0, min(1.0, target_pct))
        
        current_pct = self.get_position_pct(symbol, current_price)

        # 差距太小不交易
        diff = target_pct - current_pct
        if abs(diff) < 0.02:
            return None

        equity = self.get_equity({symbol: current_price})
        pos = self.get_position(symbol)
        
        # 判斷交易類型
        if target_pct > current_pct:
            # 目標 > 當前：需要增加多倉或減少空倉
            if pos.is_short:
                # 有空倉，先平空
                close_value = min(abs(diff), abs(current_pct)) * equity
                return self._close_short(symbol, close_value, current_price, reason)
            else:
                # 開多或加多
                buy_value = diff * equity
                return self._open_long(symbol, buy_value, current_price, reason)
        else:
            # 目標 < 當前：需要減少多倉或增加空倉
            if pos.is_long:
                # 有多倉，先平多
                close_value = min(abs(diff), abs(current_pct)) * equity
                return self._close_long(symbol, close_value, current_price, reason)
            elif self.supports_short:
                # 開空或加空（僅 Futures）
                short_value = abs(diff) * equity
                return self._open_short(symbol, short_value, current_price, reason)
            else:
                # Spot 模式不支援做空
                return None

    # ── 內部方法：做多 ──────────────────────────────────────

    def _open_long(self, symbol: str, value: float, price: float, reason: str) -> TradeRecord | None:
        """開多倉 / 加多倉"""
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
        if pos.is_long:
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
            side="LONG" if self.supports_short else "BUY",
            qty=qty,
            price=exec_price,
            fee=fee,
            value=value,
            pnl=None,
            reason=reason,
        )
        self.account.trades.append(trade)
        self._save_state()

        logger.info(f"📗 LONG {symbol}: {qty:.6f} @ {exec_price:.2f} "
                    f"(fee={fee:.2f}, reason={reason})")
        return trade

    def _close_long(self, symbol: str, value: float, price: float, reason: str) -> TradeRecord | None:
        """平多倉"""
        pos = self.get_position(symbol)
        if not pos.is_long:
            return None

        # 滑點：賣出價格更低
        exec_price = price * (1 - self.account.slippage_pct)
        raw_qty = value / exec_price
        qty = min(raw_qty, pos.qty)  # 不能賣超過持倉
        # 浮點精度修正：如果要平的量 ≥ 98% 持倉，視為全平
        if raw_qty >= pos.qty * 0.98:
            qty = pos.qty

        if qty < 1e-10:
            return None

        sell_value = qty * exec_price
        fee = sell_value * self.account.fee_pct
        self.account.cash += sell_value - fee

        pnl = (exec_price - pos.avg_entry) * qty - fee

        pos.qty -= qty
        if pos.qty < 1e-8:
            pos.qty = 0.0
            pos.avg_entry = 0.0

        trade = TradeRecord(
            timestamp=time.time(),
            symbol=symbol,
            side="CLOSE_LONG" if self.supports_short else "SELL",
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
        logger.info(f"📕 CLOSE_LONG {symbol}: {qty:.6f} @ {exec_price:.2f} "
                    f"(fee={fee:.2f}, pnl={pnl:+.2f} {emoji}, reason={reason})")
        return trade

    # ── 內部方法：做空 ──────────────────────────────────────

    def _open_short(self, symbol: str, value: float, price: float, reason: str) -> TradeRecord | None:
        """
        開空倉 / 加空倉
        
        做空機制：
        1. 借入資產並賣出，收到現金
        2. 之後需要買回資產還回
        3. 盈虧 = (開倉價 - 平倉價) × 數量
        
        模擬方式：
        - 開空倉時，不扣現金（視為借入賣出）
        - 將 qty 設為負數表示空倉
        - 平倉時結算盈虧
        """
        if not self.supports_short:
            logger.warning(f"⚠️  {symbol}: Spot 模式不支援做空")
            return None

        # 滑點：賣出價格更低（開空是賣）
        exec_price = price * (1 - self.account.slippage_pct)
        qty = value / exec_price
        fee = value * self.account.fee_pct

        # 檢查是否有足夠保證金（簡化：用現金的一定比例作為保證金）
        margin_required = value / self.leverage
        if margin_required > self.account.cash:
            margin_required = self.account.cash
            value = margin_required * self.leverage
            qty = value / exec_price
            fee = value * self.account.fee_pct

        if qty < 1e-10:
            return None

        # 扣除保證金
        self.account.cash -= (margin_required + fee)

        pos = self.get_position(symbol)
        if pos.is_short:
            # 加空倉：更新均價
            total_qty = pos.qty - qty  # qty 是正數，pos.qty 是負數
            # 加權平均：(舊均價 × |舊數量| + 新價格 × 新數量) / |總數量|
            pos.avg_entry = (pos.avg_entry * abs(pos.qty) + exec_price * qty) / abs(total_qty)
            pos.qty = total_qty
        else:
            pos.qty = -qty  # 負數表示空倉
            pos.avg_entry = exec_price

        trade = TradeRecord(
            timestamp=time.time(),
            symbol=symbol,
            side="SHORT",
            qty=qty,
            price=exec_price,
            fee=fee,
            value=value,
            pnl=None,
            reason=reason,
        )
        self.account.trades.append(trade)
        self._save_state()

        logger.info(f"📕 SHORT {symbol}: {qty:.6f} @ {exec_price:.2f} "
                    f"(fee={fee:.2f}, margin={margin_required:.2f}, reason={reason})")
        return trade

    def _close_short(self, symbol: str, value: float, price: float, reason: str) -> TradeRecord | None:
        """
        平空倉
        
        平空機制：
        1. 買回資產還回
        2. 計算盈虧 = (開倉價 - 平倉價) × 數量
        3. 釋放保證金 + 盈虧
        """
        pos = self.get_position(symbol)
        if not pos.is_short:
            return None

        # 滑點：買入價格更高（平空是買）
        exec_price = price * (1 + self.account.slippage_pct)
        raw_qty = value / exec_price
        qty = min(raw_qty, abs(pos.qty))  # 不能平超過持倉
        # 浮點精度修正：如果要平的量 ≥ 98% 持倉，視為全平
        if raw_qty >= abs(pos.qty) * 0.98:
            qty = abs(pos.qty)

        if qty < 1e-10:
            return None

        buy_value = qty * exec_price
        fee = buy_value * self.account.fee_pct
        
        # 計算盈虧：(開倉價 - 平倉價) × 數量
        pnl = (pos.avg_entry - exec_price) * qty - fee
        
        # 釋放保證金 + 盈虧
        margin_release = (pos.avg_entry * qty) / self.leverage
        self.account.cash += margin_release + pnl

        pos.qty += qty  # qty 是正數，pos.qty 是負數，相加後絕對值變小
        if abs(pos.qty) < 1e-8:
            pos.qty = 0.0
            pos.avg_entry = 0.0

        trade = TradeRecord(
            timestamp=time.time(),
            symbol=symbol,
            side="CLOSE_SHORT",
            qty=qty,
            price=exec_price,
            fee=fee,
            value=buy_value,
            pnl=pnl,
            reason=reason,
        )
        self.account.trades.append(trade)
        self._save_state()

        emoji = "📈" if pnl and pnl > 0 else "📉"
        logger.info(f"📗 CLOSE_SHORT {symbol}: {qty:.6f} @ {exec_price:.2f} "
                    f"(fee={fee:.2f}, pnl={pnl:+.2f} {emoji}, reason={reason})")
        return trade

    # ── 相容舊介面 ──────────────────────────────────────────

    def _buy(self, symbol: str, value: float, price: float, reason: str) -> TradeRecord | None:
        """相容舊介面：等同於 _open_long"""
        return self._open_long(symbol, value, price, reason)

    def _sell(self, symbol: str, value: float, price: float, reason: str) -> TradeRecord | None:
        """相容舊介面：等同於 _close_long"""
        return self._close_long(symbol, value, price, reason)

    # ── 狀態持久化 ────────────────────────────────────────

    def touch_state(self) -> None:
        """更新狀態檔的修改時間（即使沒有交易也寫入），供健康檢查判斷 cron 存活"""
        self._save_state()

    def _save_state(self) -> None:
        if not self.state_path:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "initial_cash": self.account.initial_cash,
            "cash": self.account.cash,
            "fee_bps": self.account.fee_bps,
            "slippage_bps": self.account.slippage_bps,
            "market_type": self.market_type,
            "leverage": self.leverage,
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
        # 恢復 market_type 和 leverage（如果有）
        if "market_type" in state:
            self.market_type = state["market_type"]
            self.supports_short = (self.market_type == "futures")
        if "leverage" in state:
            self.leverage = state["leverage"]
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
        
        # 市場類型標籤
        market_emoji = "🟢" if self.market_type == "spot" else "🔴"
        market_label = "SPOT" if self.market_type == "spot" else f"FUTURES ({self.leverage}x)"
        
        lines = [
            "=" * 50,
            f"  Paper Trading 帳戶摘要 {market_emoji} [{market_label}]",
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
                if pos.is_long:
                    pnl = (price - pos.avg_entry) * pos.qty if price > 0 else 0
                    side_label = "LONG"
                else:  # is_short
                    pnl = (pos.avg_entry - price) * abs(pos.qty) if price > 0 else 0
                    side_label = "SHORT"
                
                pnl_emoji = "📈" if pnl > 0 else "📉"
                lines.append(
                    f"  {sym} [{side_label}]: {abs(pos.qty):.6f} @ {pos.avg_entry:.2f} "
                    f"(PnL: {pnl:+.2f} {pnl_emoji})"
                )
        lines.append("=" * 50)
        return "\n".join(lines)
