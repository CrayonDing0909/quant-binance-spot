"""
Binance Spot Broker — 真实下单引擎

通过 Binance REST API 执行真实交易。
需要设置环境变量：
    BINANCE_API_KEY
    BINANCE_API_SECRET
"""
from __future__ import annotations
from dataclasses import dataclass

from ..data.binance_client import BinanceHTTP
from ..utils.log import get_logger

logger = get_logger("binance_broker")


@dataclass
class OrderResult:
    order_id: str
    symbol: str
    side: str
    qty: float
    price: float
    status: str
    raw: dict


class BinanceSpotBroker:
    """
    Binance Spot 真实下单引擎

    仅支持市价单（MARKET），适合非高频策略。
    """

    def __init__(self):
        self.http = BinanceHTTP()
        if not self.http.api_key or not self.http.api_secret:
            raise RuntimeError(
                "❌ 需要设置环境变量 BINANCE_API_KEY 和 BINANCE_API_SECRET\n"
                "   请在 .env 文件中配置"
            )
        logger.info("✅ Binance Spot Broker 初始化完成")

    def get_balance(self, asset: str = "USDT") -> float:
        """查询指定资产余额"""
        try:
            data = self.http.signed_get("/api/v3/account", {})
            for balance in data.get("balances", []):
                if balance["asset"] == asset:
                    return float(balance["free"])
            return 0.0
        except Exception as e:
            logger.error(f"查询余额失败: {e}")
            return 0.0

    def get_position(self, symbol: str) -> float:
        """
        查询持仓数量

        Spot 没有 position 的概念，通过查询 base asset 余额实现。
        例如 BTCUSDT → 查询 BTC 余额
        """
        base_asset = symbol.replace("USDT", "").replace("BUSD", "")
        return self.get_balance(base_asset)

    def get_price(self, symbol: str) -> float:
        """查询最新价格"""
        try:
            data = self.http.get("/api/v3/ticker/price", {"symbol": symbol})
            return float(data["price"])
        except Exception as e:
            logger.error(f"查询价格失败: {e}")
            return 0.0

    def market_buy(self, symbol: str, quote_qty: float) -> OrderResult | None:
        """
        市价买入（按报价资产金额）

        Args:
            symbol: 交易对, e.g. "BTCUSDT"
            quote_qty: 买入金额 (USDT), e.g. 100.0
        """
        try:
            result = self.http.signed_post("/api/v3/order", {
                "symbol": symbol,
                "side": "BUY",
                "type": "MARKET",
                "quoteOrderQty": f"{quote_qty:.2f}",
            })
            order = OrderResult(
                order_id=str(result["orderId"]),
                symbol=symbol,
                side="BUY",
                qty=float(result.get("executedQty", 0)),
                price=float(result.get("cummulativeQuoteQty", 0)) / max(float(result.get("executedQty", 1)), 1e-10),
                status=result.get("status", "UNKNOWN"),
                raw=result,
            )
            logger.info(f"📗 REAL BUY  {symbol}: {order.qty:.6f} @ ~{order.price:.2f} "
                        f"(${quote_qty:.2f}, orderId={order.order_id})")
            return order
        except Exception as e:
            logger.error(f"❌ 买入失败 {symbol}: {e}")
            return None

    def market_sell(self, symbol: str, qty: float) -> OrderResult | None:
        """
        市价卖出（按数量）

        Args:
            symbol: 交易对
            qty: 卖出数量 (base asset)
        """
        try:
            result = self.http.signed_post("/api/v3/order", {
                "symbol": symbol,
                "side": "SELL",
                "type": "MARKET",
                "quantity": f"{qty:.8f}",
            })
            order = OrderResult(
                order_id=str(result["orderId"]),
                symbol=symbol,
                side="SELL",
                qty=float(result.get("executedQty", 0)),
                price=float(result.get("cummulativeQuoteQty", 0)) / max(float(result.get("executedQty", 1)), 1e-10),
                status=result.get("status", "UNKNOWN"),
                raw=result,
            )
            logger.info(f"📕 REAL SELL {symbol}: {order.qty:.6f} @ ~{order.price:.2f} "
                        f"(orderId={order.order_id})")
            return order
        except Exception as e:
            logger.error(f"❌ 卖出失败 {symbol}: {e}")
            return None

    def execute_target_position(
        self,
        symbol: str,
        target_pct: float,
        current_price: float | None = None,
        reason: str = "signal",
    ) -> OrderResult | None:
        """
        执行目标仓位调整

        与 PaperBroker 接口一致，方便切换。
        """
        if current_price is None:
            current_price = self.get_price(symbol)
        if current_price <= 0:
            logger.error(f"无法获取 {symbol} 价格")
            return None

        target_pct = max(0.0, min(1.0, target_pct))

        # 计算当前仓位
        usdt_balance = self.get_balance("USDT")
        position_qty = self.get_position(symbol)
        position_value = position_qty * current_price
        total_equity = usdt_balance + position_value

        if total_equity <= 0:
            logger.error("账户权益为 0")
            return None

        current_pct = position_value / total_equity
        diff = target_pct - current_pct

        if abs(diff) < 0.02:
            return None  # 差距太小

        if diff > 0:
            # 需要买入
            buy_amount = diff * total_equity
            buy_amount = min(buy_amount, usdt_balance * 0.99)  # 预留 1% 手续费
            if buy_amount < 10:  # Binance 最小下单金额
                return None
            return self.market_buy(symbol, buy_amount)
        else:
            # 需要卖出
            sell_value = abs(diff) * total_equity
            sell_qty = sell_value / current_price
            sell_qty = min(sell_qty, position_qty)
            if sell_qty * current_price < 10:
                return None
            return self.market_sell(symbol, sell_qty)
