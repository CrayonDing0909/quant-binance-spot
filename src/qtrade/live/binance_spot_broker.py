"""
Binance Spot Broker — 真实下单引擎

通过 Binance REST API 执行真实交易。
需要设置环境变量：
    BINANCE_API_KEY
    BINANCE_API_SECRET

功能：
    - 市价买入/卖出
    - 自动处理 LOT_SIZE (stepSize / minQty) 和 MIN_NOTIONAL
    - 多币种权益计算
    - dry-run 模式（只记录不下单）
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

from ..data.binance_client import BinanceHTTP
from ..utils.log import get_logger

logger = get_logger("binance_broker")


@dataclass
class OrderResult:
    """交易结果（与 PaperBroker.TradeRecord 字段对齐）"""
    order_id: str
    symbol: str
    side: str           # BUY / SELL
    qty: float
    price: float
    fee: float          # 估算手续费
    value: float        # qty * price
    pnl: float | None   # 卖出时估算 PnL
    status: str
    reason: str = ""
    raw: dict = field(default_factory=dict)


@dataclass
class SymbolFilter:
    """Binance 交易对的下单规则"""
    min_qty: float = 0.0
    max_qty: float = float("inf")
    step_size: float = 0.0
    min_notional: float = 10.0  # 最小下单金额
    tick_size: float = 0.0

    def round_qty(self, qty: float) -> float:
        """根据 stepSize 对齐数量"""
        if self.step_size <= 0:
            return qty
        # 用 floor 避免超出余额
        precision = max(0, -int(math.log10(self.step_size)))
        return math.floor(qty * 10**precision) / 10**precision

    def validate_qty(self, qty: float) -> tuple[bool, str]:
        """检查数量是否合规"""
        if qty < self.min_qty:
            return False, f"qty {qty} < minQty {self.min_qty}"
        if qty > self.max_qty:
            return False, f"qty {qty} > maxQty {self.max_qty}"
        return True, ""

    def validate_notional(self, qty: float, price: float) -> tuple[bool, str]:
        """检查下单金额是否满足最低要求"""
        notional = qty * price
        if notional < self.min_notional:
            return False, f"notional ${notional:.2f} < minNotional ${self.min_notional:.2f}"
        return True, ""


class BinanceSpotBroker:
    """
    Binance Spot 真实下单引擎

    仅支持市价单（MARKET），适合非高频策略。

    Args:
        dry_run: True = 只记录不下单（用于测试）
    """

    # data-api.binance.vision 是公开数据端点，不支持签名请求（交易/查余额）
    # 真实交易必须用 api.binance.com
    _DATA_ONLY_ENDPOINTS = [
        "data-api.binance.vision",
        "data-api.binance.com",
    ]

    def __init__(self, dry_run: bool = False):
        self.http = BinanceHTTP()
        self.dry_run = dry_run
        self._filters: dict[str, SymbolFilter] = {}  # 缓存
        self._avg_entries: dict[str, float] = {}  # 追踪买入均价（用于计算 PnL）

        if not self.http.api_key or not self.http.api_secret:
            raise RuntimeError(
                "❌ 需要设置环境变量 BINANCE_API_KEY 和 BINANCE_API_SECRET\n"
                "   请在 .env 文件中配置"
            )

        # 自动切换到支持签名请求的端点
        if any(ep in self.http.base_url for ep in self._DATA_ONLY_ENDPOINTS):
            old_url = self.http.base_url
            self.http.base_url = "https://api.binance.com"
            logger.warning(
                f"⚠️  自动切换 API 端点: {old_url} → {self.http.base_url}\n"
                f"   （data-api.binance.vision 不支持签名请求/交易）"
            )

        mode_str = "🧪 DRY-RUN（不会真的下单）" if dry_run else "💰 LIVE（真金白银！）"
        logger.info(f"✅ Binance Spot Broker 初始化完成 [{mode_str}]")

    # ── 交易对规则 ────────────────────────────────────────

    def _get_filter(self, symbol: str) -> SymbolFilter:
        """从 exchangeInfo 获取交易对的下单规则"""
        if symbol in self._filters:
            return self._filters[symbol]

        try:
            data = self.http.get("/api/v3/exchangeInfo", {"symbol": symbol})
            for sym_info in data.get("symbols", []):
                if sym_info["symbol"] == symbol:
                    sf = SymbolFilter()
                    for f in sym_info.get("filters", []):
                        if f["filterType"] == "LOT_SIZE":
                            sf.min_qty = float(f["minQty"])
                            sf.max_qty = float(f["maxQty"])
                            sf.step_size = float(f["stepSize"])
                        elif f["filterType"] == "NOTIONAL":
                            sf.min_notional = float(f.get("minNotional", 10))
                        elif f["filterType"] == "MIN_NOTIONAL":
                            sf.min_notional = float(f.get("minNotional", 10))
                        elif f["filterType"] == "PRICE_FILTER":
                            sf.tick_size = float(f.get("tickSize", 0))
                    self._filters[symbol] = sf
                    logger.debug(
                        f"📋 {symbol} 规则: minQty={sf.min_qty}, "
                        f"stepSize={sf.step_size}, minNotional=${sf.min_notional}"
                    )
                    return sf
        except Exception as e:
            logger.warning(f"⚠️  获取 {symbol} exchangeInfo 失败: {e}，使用默认值")

        sf = SymbolFilter()
        self._filters[symbol] = sf
        return sf

    # ── 查询接口 ──────────────────────────────────────────

    def get_balance(self, asset: str = "USDT") -> float:
        """查询指定资产余额（free）"""
        try:
            data = self.http.signed_get("/api/v3/account", {})
            for balance in data.get("balances", []):
                if balance["asset"] == asset:
                    return float(balance["free"])
            return 0.0
        except Exception as e:
            logger.error(f"查询余额失败: {e}")
            return 0.0

    def get_all_balances(self) -> dict[str, float]:
        """查询所有资产余额 (free > 0)"""
        try:
            data = self.http.signed_get("/api/v3/account", {})
            return {
                b["asset"]: float(b["free"])
                for b in data.get("balances", [])
                if float(b["free"]) > 0
            }
        except Exception as e:
            logger.error(f"查询余额失败: {e}")
            return {}

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

    def get_equity(self, symbols: list[str] | None = None) -> float:
        """
        计算总权益 = USDT 余额 + 所有持仓市值

        Args:
            symbols: 要计算的交易对列表。None = 只算 USDT
        """
        equity = self.get_balance("USDT")
        if symbols:
            for sym in symbols:
                qty = self.get_position(sym)
                if qty > 0:
                    price = self.get_price(sym)
                    equity += qty * price
        return equity

    def get_position_pct(self, symbol: str, current_price: float) -> float:
        """获取某币种持仓占总权益的比例 [0, 1]"""
        qty = self.get_position(symbol)
        if qty <= 0 or current_price <= 0:
            return 0.0
        position_value = qty * current_price
        # 简化计算：equity ≈ USDT + 当前币种市值
        equity = self.get_balance("USDT") + position_value
        if equity <= 0:
            return 0.0
        return position_value / equity

    # ── 下单接口 ──────────────────────────────────────────

    def market_buy(
        self, symbol: str, quote_qty: float, reason: str = ""
    ) -> OrderResult | None:
        """
        市价买入（按报价资产金额）

        Args:
            symbol: 交易对, e.g. "BTCUSDT"
            quote_qty: 买入金额 (USDT), e.g. 100.0
            reason: 下单原因
        """
        sf = self._get_filter(symbol)

        # 检查最小下单金额
        if quote_qty < sf.min_notional:
            logger.warning(
                f"⚠️  {symbol} 买入金额 ${quote_qty:.2f} "
                f"< 最小 ${sf.min_notional:.2f}，跳过"
            )
            return None

        if self.dry_run:
            price = self.get_price(symbol)
            est_qty = quote_qty / price if price > 0 else 0
            est_qty = sf.round_qty(est_qty)
            est_fee = quote_qty * 0.001  # 估算 0.1% 手续费
            logger.info(
                f"🧪 [DRY-RUN] BUY  {symbol}: ~{est_qty:.6f} @ ~${price:,.2f} "
                f"(${quote_qty:.2f}, reason={reason})"
            )
            return OrderResult(
                order_id="DRY-RUN",
                symbol=symbol,
                side="BUY",
                qty=est_qty,
                price=price,
                fee=est_fee,
                value=quote_qty,
                pnl=None,
                status="DRY_RUN",
                reason=reason,
            )

        try:
            result = self.http.signed_post("/api/v3/order", {
                "symbol": symbol,
                "side": "BUY",
                "type": "MARKET",
                "quoteOrderQty": f"{quote_qty:.2f}",
            })

            exec_qty = float(result.get("executedQty", 0))
            cum_quote = float(result.get("cummulativeQuoteQty", 0))
            avg_price = cum_quote / max(exec_qty, 1e-10)
            est_fee = cum_quote * 0.001  # VIP 0 = 0.1%

            # 追踪买入均价
            self._avg_entries[symbol] = avg_price

            order = OrderResult(
                order_id=str(result["orderId"]),
                symbol=symbol,
                side="BUY",
                qty=exec_qty,
                price=avg_price,
                fee=est_fee,
                value=cum_quote,
                pnl=None,
                status=result.get("status", "UNKNOWN"),
                reason=reason,
                raw=result,
            )
            logger.info(
                f"📗 REAL BUY  {symbol}: {order.qty:.6f} @ ~${order.price:,.2f} "
                f"(${quote_qty:.2f}, orderId={order.order_id})"
            )
            return order

        except Exception as e:
            logger.error(f"❌ 买入失败 {symbol}: {e}")
            return None

    def market_sell(
        self, symbol: str, qty: float, reason: str = ""
    ) -> OrderResult | None:
        """
        市价卖出（按数量）

        Args:
            symbol: 交易对
            qty: 卖出数量 (base asset)
            reason: 下单原因
        """
        sf = self._get_filter(symbol)
        qty = sf.round_qty(qty)

        # 检查数量合规性
        ok, msg = sf.validate_qty(qty)
        if not ok:
            logger.warning(f"⚠️  {symbol} 卖出数量不合规: {msg}")
            return None

        # 检查最小金额
        price = self.get_price(symbol)
        ok, msg = sf.validate_notional(qty, price)
        if not ok:
            logger.warning(f"⚠️  {symbol} 卖出金额不足: {msg}")
            return None

        # 估算 PnL
        avg_entry = self._avg_entries.get(symbol, 0)
        est_pnl = (price - avg_entry) * qty if avg_entry > 0 else None

        if self.dry_run:
            est_fee = qty * price * 0.001
            logger.info(
                f"🧪 [DRY-RUN] SELL {symbol}: {qty:.6f} @ ~${price:,.2f} "
                f"(reason={reason})"
            )
            return OrderResult(
                order_id="DRY-RUN",
                symbol=symbol,
                side="SELL",
                qty=qty,
                price=price,
                fee=est_fee,
                value=qty * price,
                pnl=est_pnl,
                status="DRY_RUN",
                reason=reason,
            )

        try:
            result = self.http.signed_post("/api/v3/order", {
                "symbol": symbol,
                "side": "SELL",
                "type": "MARKET",
                "quantity": f"{qty:.8f}",
            })

            exec_qty = float(result.get("executedQty", 0))
            cum_quote = float(result.get("cummulativeQuoteQty", 0))
            avg_price = cum_quote / max(exec_qty, 1e-10)
            est_fee = cum_quote * 0.001

            # 计算 PnL
            pnl = (avg_price - avg_entry) * exec_qty if avg_entry > 0 else None

            order = OrderResult(
                order_id=str(result["orderId"]),
                symbol=symbol,
                side="SELL",
                qty=exec_qty,
                price=avg_price,
                fee=est_fee,
                value=cum_quote,
                pnl=pnl,
                status=result.get("status", "UNKNOWN"),
                reason=reason,
                raw=result,
            )
            logger.info(
                f"📕 REAL SELL {symbol}: {order.qty:.6f} @ ~${order.price:,.2f} "
                f"(orderId={order.order_id})"
            )
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

        Args:
            symbol: 交易对
            target_pct: 目标仓位占权益比例 [0, 1]
            current_price: 当前价格（None 时自动查询）
            reason: 下单原因
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

        logger.info(
            f"📊 {symbol}: 当前={current_pct:.1%}, 目标={target_pct:.1%}, "
            f"差距={diff:+.1%}, 权益=${total_equity:,.2f}"
        )

        if abs(diff) < 0.02:
            logger.debug(f"  {symbol}: 差距 < 2%，跳过")
            return None  # 差距太小

        if diff > 0:
            # 需要买入
            buy_amount = diff * total_equity
            buy_amount = min(buy_amount, usdt_balance * 0.995)  # 预留手续费
            return self.market_buy(symbol, buy_amount, reason=reason)
        else:
            # 需要卖出
            sell_value = abs(diff) * total_equity
            sell_qty = sell_value / current_price
            sell_qty = min(sell_qty, position_qty)
            return self.market_sell(symbol, sell_qty, reason=reason)

    # ── 连接检查 ──────────────────────────────────────────

    def check_connection(self, symbols: list[str] | None = None) -> dict:
        """
        检查 Binance API 连接状态

        Returns:
            {
                "server_time": "2026-02-08 12:00:00 UTC",
                "api_permissions": {...},
                "usdt_balance": 100.0,
                "balances": {...},
                "prices": {...},
                "filters": {...},
            }
        """
        result = {}

        # 1. 服务器时间
        try:
            data = self.http.get("/api/v3/time")
            ts = datetime.fromtimestamp(data["serverTime"] / 1000, tz=timezone.utc)
            result["server_time"] = ts.strftime("%Y-%m-%d %H:%M:%S UTC")
            logger.info(f"✅ 服务器连接正常: {result['server_time']}")
        except Exception as e:
            result["server_time_error"] = str(e)
            logger.error(f"❌ 服务器连接失败: {e}")

        # 2. 账户信息
        try:
            account = self.http.signed_get("/api/v3/account", {})
            result["can_trade"] = account.get("canTrade", False)
            result["can_withdraw"] = account.get("canWithdraw", False)
            result["account_type"] = account.get("accountType", "UNKNOWN")

            balances = {}
            for b in account.get("balances", []):
                free = float(b["free"])
                locked = float(b["locked"])
                if free > 0 or locked > 0:
                    balances[b["asset"]] = {"free": free, "locked": locked}
            result["balances"] = balances
            result["usdt_balance"] = balances.get("USDT", {}).get("free", 0)

            logger.info(f"✅ 账户连接正常: canTrade={result['can_trade']}")
            logger.info(f"   USDT 余额: ${result['usdt_balance']:,.2f}")
            if balances:
                for asset, val in balances.items():
                    if asset != "USDT" and val["free"] > 0:
                        logger.info(f"   {asset}: {val['free']}")
        except Exception as e:
            result["account_error"] = str(e)
            logger.error(f"❌ 账户查询失败: {e}")

        # 3. 交易对价格 + 规则
        if symbols:
            prices = {}
            filters = {}
            for sym in symbols:
                try:
                    p = self.get_price(sym)
                    prices[sym] = p
                    logger.info(f"   {sym}: ${p:,.2f}")
                except Exception as e:
                    logger.warning(f"   {sym}: 获取价格失败 - {e}")
                try:
                    sf = self._get_filter(sym)
                    filters[sym] = {
                        "min_qty": sf.min_qty,
                        "step_size": sf.step_size,
                        "min_notional": sf.min_notional,
                    }
                except Exception:
                    pass
            result["prices"] = prices
            result["filters"] = filters

        return result
