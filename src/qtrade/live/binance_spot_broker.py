"""
Binance Spot Broker — 真實下單引擎

通過 Binance REST API 執行真實交易。
需要設置環境變數：
    BINANCE_API_KEY
    BINANCE_API_SECRET

功能：
    - 市價買入/賣出
    - 硬止損（STOP_LOSS_MARKET 預掛單）- v2.0 新增
    - 自動處理 LOT_SIZE (stepSize / minQty) 和 MIN_NOTIONAL
    - 多幣種權益計算
    - dry-run 模式（只記錄不下單）
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
    """交易結果（與 PaperBroker.TradeRecord 欄位對齊）"""
    order_id: str
    symbol: str
    side: str           # BUY / SELL
    qty: float
    price: float
    fee: float          # 估算手續費
    value: float        # qty * price
    pnl: float | None   # 賣出時估算 PnL
    status: str
    reason: str = ""
    raw: dict = field(default_factory=dict)


@dataclass
class SymbolFilter:
    """Binance 交易對的下單規則"""
    min_qty: float = 0.0
    max_qty: float = float("inf")
    step_size: float = 0.0
    min_notional: float = 10.0  # 最小下單金額
    tick_size: float = 0.0

    def round_qty(self, qty: float) -> float:
        """根據 stepSize 對齊數量"""
        if self.step_size <= 0:
            return qty
        # 用 floor 避免超出餘額
        precision = max(0, -int(math.log10(self.step_size)))
        return math.floor(qty * 10**precision) / 10**precision

    def validate_qty(self, qty: float) -> tuple[bool, str]:
        """檢查數量是否合規"""
        if qty < self.min_qty:
            return False, f"qty {qty} < minQty {self.min_qty}"
        if qty > self.max_qty:
            return False, f"qty {qty} > maxQty {self.max_qty}"
        return True, ""

    def validate_notional(self, qty: float, price: float) -> tuple[bool, str]:
        """檢查下單金額是否滿足最低要求"""
        notional = qty * price
        if notional < self.min_notional:
            return False, f"notional ${notional:.2f} < minNotional ${self.min_notional:.2f}"
        return True, ""


class BinanceSpotBroker:
    """
    Binance Spot 真實下單引擎

    僅支援市價單（MARKET），適合非高頻策略。

    Args:
        dry_run: True = 只記錄不下單（用於測試）
    """

    # data-api.binance.vision 是公開數據端點，不支援簽名請求（交易/查餘額）
    # 真實交易必須用 api.binance.com
    _DATA_ONLY_ENDPOINTS = [
        "data-api.binance.vision",
        "data-api.binance.com",
    ]

    def __init__(self, dry_run: bool = False):
        self.http = BinanceHTTP()
        self.dry_run = dry_run
        self._filters: dict[str, SymbolFilter] = {}  # 快取
        self._avg_entries: dict[str, float] = {}  # 追蹤買入均價（用於計算 PnL）
        self._active_stop_orders: dict[str, str] = {}  # symbol → orderId（活躍的止損單）

        if not self.http.api_key or not self.http.api_secret:
            raise RuntimeError(
                "❌ 需要設置環境變數 BINANCE_API_KEY 和 BINANCE_API_SECRET\n"
                "   請在 .env 檔案中配置"
            )

        # 自動切換到支援簽名請求的端點
        if any(ep in self.http.base_url for ep in self._DATA_ONLY_ENDPOINTS):
            old_url = self.http.base_url
            self.http.base_url = "https://api.binance.com"
            logger.warning(
                f"⚠️  自動切換 API 端點: {old_url} → {self.http.base_url}\n"
                f"   （data-api.binance.vision 不支援簽名請求/交易）"
            )

        mode_str = "🧪 DRY-RUN（不會真的下單）" if dry_run else "💰 LIVE（真金白銀！）"
        logger.info(f"✅ Binance Spot Broker 初始化完成 [{mode_str}]")

    # ── 交易對規則 ────────────────────────────────────────

    def _get_filter(self, symbol: str) -> SymbolFilter:
        """從 exchangeInfo 獲取交易對的下單規則"""
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
                        f"📋 {symbol} 規則: minQty={sf.min_qty}, "
                        f"stepSize={sf.step_size}, minNotional=${sf.min_notional}"
                    )
                    return sf
        except Exception as e:
            logger.warning(f"⚠️  獲取 {symbol} exchangeInfo 失敗: {e}，使用預設值")

        sf = SymbolFilter()
        self._filters[symbol] = sf
        return sf

    # ── 查詢介面 ──────────────────────────────────────────

    def get_balance(self, asset: str = "USDT") -> float:
        """查詢指定資產餘額（free）"""
        try:
            data = self.http.signed_get("/api/v3/account", {})
            for balance in data.get("balances", []):
                if balance["asset"] == asset:
                    return float(balance["free"])
            return 0.0
        except Exception as e:
            logger.error(f"查詢餘額失敗: {e}")
            return 0.0

    def get_all_balances(self) -> dict[str, float]:
        """查詢所有資產餘額 (free > 0)"""
        try:
            data = self.http.signed_get("/api/v3/account", {})
            return {
                b["asset"]: float(b["free"])
                for b in data.get("balances", [])
                if float(b["free"]) > 0
            }
        except Exception as e:
            logger.error(f"查詢餘額失敗: {e}")
            return {}

    def get_position(self, symbol: str) -> float:
        """
        查詢持倉數量

        Spot 沒有 position 的概念，通過查詢 base asset 餘額實現。
        例如 BTCUSDT → 查詢 BTC 餘額
        """
        base_asset = symbol.replace("USDT", "").replace("BUSD", "")
        return self.get_balance(base_asset)

    def get_price(self, symbol: str) -> float:
        """查詢最新價格"""
        try:
            data = self.http.get("/api/v3/ticker/price", {"symbol": symbol})
            return float(data["price"])
        except Exception as e:
            logger.error(f"查詢價格失敗: {e}")
            return 0.0

    def get_equity(self, symbols: list[str] | None = None) -> float:
        """
        計算總權益 = USDT 餘額 + 所有持倉市值

        Args:
            symbols: 要計算的交易對列表。None = 只算 USDT
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
        """獲取某幣種持倉佔總權益的比例 [0, 1]"""
        qty = self.get_position(symbol)
        if qty <= 0 or current_price <= 0:
            return 0.0
        position_value = qty * current_price
        # 簡化計算：equity ≈ USDT + 當前幣種市值
        equity = self.get_balance("USDT") + position_value
        if equity <= 0:
            return 0.0
        return position_value / equity

    # ── 下單介面 ──────────────────────────────────────────

    def market_buy(
        self, symbol: str, quote_qty: float, reason: str = ""
    ) -> OrderResult | None:
        """
        市價買入（按報價資產金額）

        Args:
            symbol: 交易對, e.g. "BTCUSDT"
            quote_qty: 買入金額 (USDT), e.g. 100.0
            reason: 下單原因
        """
        sf = self._get_filter(symbol)

        # 檢查最小下單金額
        if quote_qty < sf.min_notional:
            logger.warning(
                f"⚠️  {symbol} 買入金額 ${quote_qty:.2f} "
                f"< 最小 ${sf.min_notional:.2f}，跳過"
            )
            return None

        if self.dry_run:
            price = self.get_price(symbol)
            est_qty = quote_qty / price if price > 0 else 0
            est_qty = sf.round_qty(est_qty)
            est_fee = quote_qty * 0.001  # 估算 0.1% 手續費
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

            # 追蹤買入均價
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
            logger.error(f"❌ 買入失敗 {symbol}: {e}")
            return None

    def market_sell(
        self, symbol: str, qty: float, reason: str = ""
    ) -> OrderResult | None:
        """
        市價賣出（按數量）

        Args:
            symbol: 交易對
            qty: 賣出數量 (base asset)
            reason: 下單原因
        """
        sf = self._get_filter(symbol)
        qty = sf.round_qty(qty)

        # 檢查數量合規性
        ok, msg = sf.validate_qty(qty)
        if not ok:
            logger.warning(f"⚠️  {symbol} 賣出數量不合規: {msg}")
            return None

        # 檢查最小金額
        price = self.get_price(symbol)
        ok, msg = sf.validate_notional(qty, price)
        if not ok:
            logger.warning(f"⚠️  {symbol} 賣出金額不足: {msg}")
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

            # 計算 PnL
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
            logger.error(f"❌ 賣出失敗 {symbol}: {e}")
            return None

    # ── 硬止損（預掛單）────────────────────────────────────

    def place_stop_loss(
        self,
        symbol: str,
        qty: float,
        stop_price: float,
        reason: str = "stop_loss",
    ) -> OrderResult | None:
        """
        預掛止損單（STOP_LOSS_MARKET）
        
        當價格跌破 stop_price 時，交易所自動執行市價賣出。
        即使程式斷線、API 故障，止損單依然有效。
        
        Args:
            symbol: 交易對
            qty: 止損數量
            stop_price: 觸發價格
            reason: 原因
            
        Returns:
            OrderResult 或 None
        """
        sf = self._get_filter(symbol)
        qty = sf.round_qty(qty)
        
        # 檢查數量
        ok, msg = sf.validate_qty(qty)
        if not ok:
            logger.warning(f"⚠️  {symbol} 止損數量不合規: {msg}")
            return None
        
        # 止損價格精度處理
        if sf.tick_size > 0:
            precision = max(0, -int(math.log10(sf.tick_size)))
            stop_price = round(stop_price, precision)
        
        if self.dry_run:
            logger.info(
                f"🧪 [DRY-RUN] STOP_LOSS {symbol}: {qty:.6f} @ ${stop_price:,.2f} "
                f"(reason={reason})"
            )
            return OrderResult(
                order_id="DRY-RUN-SL",
                symbol=symbol,
                side="SELL",
                qty=qty,
                price=stop_price,
                fee=0,
                value=qty * stop_price,
                pnl=None,
                status="DRY_RUN",
                reason=reason,
            )
        
        try:
            # 先取消舊的止損單（如果有）
            self.cancel_stop_loss(symbol)
            
            result = self.http.signed_post("/api/v3/order", {
                "symbol": symbol,
                "side": "SELL",
                "type": "STOP_LOSS_LIMIT",  # Binance Spot 需要用 STOP_LOSS_LIMIT
                "timeInForce": "GTC",
                "quantity": f"{qty:.8f}",
                "stopPrice": f"{stop_price:.8f}",
                "price": f"{stop_price * 0.995:.8f}",  # 限價略低於觸發價，確保成交
            })
            
            order_id = str(result["orderId"])
            self._active_stop_orders[symbol] = order_id
            
            order = OrderResult(
                order_id=order_id,
                symbol=symbol,
                side="SELL",
                qty=qty,
                price=stop_price,
                fee=0,
                value=qty * stop_price,
                pnl=None,
                status=result.get("status", "NEW"),
                reason=reason,
                raw=result,
            )
            
            logger.info(
                f"🛡️  STOP_LOSS 已掛 {symbol}: {qty:.6f} @ ${stop_price:,.2f} "
                f"(orderId={order_id})"
            )
            return order
            
        except Exception as e:
            logger.error(f"❌ 掛止損單失敗 {symbol}: {e}")
            return None
    
    def cancel_stop_loss(self, symbol: str) -> bool:
        """
        取消止損單
        
        Args:
            symbol: 交易對
            
        Returns:
            是否成功取消
        """
        order_id = self._active_stop_orders.get(symbol)
        if not order_id:
            return True  # 沒有活躍止損單
        
        if self.dry_run:
            logger.info(f"🧪 [DRY-RUN] 取消止損 {symbol} orderId={order_id}")
            self._active_stop_orders.pop(symbol, None)
            return True
        
        try:
            self.http.signed_delete("/api/v3/order", {
                "symbol": symbol,
                "orderId": order_id,
            })
            self._active_stop_orders.pop(symbol, None)
            logger.info(f"🗑️  止損單已取消 {symbol} orderId={order_id}")
            return True
        except Exception as e:
            # 可能已經成交或已取消
            if "Unknown order" in str(e) or "UNKNOWN_ORDER" in str(e):
                self._active_stop_orders.pop(symbol, None)
                return True
            logger.warning(f"⚠️  取消止損單失敗 {symbol}: {e}")
            return False
    
    def get_active_stop_order(self, symbol: str) -> dict | None:
        """
        查詢活躍的止損單
        
        Returns:
            訂單資訊或 None
        """
        order_id = self._active_stop_orders.get(symbol)
        if not order_id:
            return None
        
        try:
            result = self.http.signed_get("/api/v3/order", {
                "symbol": symbol,
                "orderId": order_id,
            })
            
            status = result.get("status")
            if status in ("FILLED", "CANCELED", "EXPIRED", "REJECTED"):
                # 止損單已不活躍
                self._active_stop_orders.pop(symbol, None)
                if status == "FILLED":
                    logger.info(f"🛡️  止損單已成交 {symbol}")
                return None
            
            return result
        except Exception as e:
            logger.warning(f"⚠️  查詢止損單失敗 {symbol}: {e}")
            return None

    def execute_target_position(
        self,
        symbol: str,
        target_pct: float,
        current_price: float | None = None,
        reason: str = "signal",
        stop_loss_price: float | None = None,
    ) -> OrderResult | None:
        """
        執行目標倉位調整
        
        v2.0 新增：支援自動掛止損單

        與 PaperBroker 介面一致，方便切換。

        Args:
            symbol: 交易對
            target_pct: 目標倉位佔權益比例 [0, 1]
            current_price: 當前價格（None 時自動查詢）
            reason: 下單原因
            stop_loss_price: 止損價格（None = 不掛止損，有值 = 買入後自動掛 STOP_LOSS）
        """
        if current_price is None:
            current_price = self.get_price(symbol)
        if current_price <= 0:
            logger.error(f"無法獲取 {symbol} 價格")
            return None

        target_pct = max(0.0, min(1.0, target_pct))

        # 計算當前倉位
        usdt_balance = self.get_balance("USDT")
        position_qty = self.get_position(symbol)
        position_value = position_qty * current_price
        total_equity = usdt_balance + position_value

        if total_equity <= 0:
            logger.error("帳戶權益為 0")
            return None

        current_pct = position_value / total_equity
        diff = target_pct - current_pct

        logger.info(
            f"📊 {symbol}: 當前={current_pct:.1%}, 目標={target_pct:.1%}, "
            f"差距={diff:+.1%}, 權益=${total_equity:,.2f}"
        )

        if abs(diff) < 0.02:
            logger.debug(f"  {symbol}: 差距 < 2%，跳過")
            return None  # 差距太小

        if diff > 0:
            # 需要買入
            buy_amount = diff * total_equity
            buy_amount = min(buy_amount, usdt_balance * 0.995)  # 預留手續費
            result = self.market_buy(symbol, buy_amount, reason=reason)
            
            # v2.0: 買入成功後自動掛止損單
            if result and stop_loss_price and stop_loss_price > 0:
                # 查詢實際持倉（包含這次買入）
                total_qty = self.get_position(symbol)
                if total_qty > 0:
                    sl_result = self.place_stop_loss(
                        symbol=symbol,
                        qty=total_qty,
                        stop_price=stop_loss_price,
                        reason="auto_stop_loss",
                    )
                    if sl_result:
                        logger.info(
                            f"🛡️  自動掛止損: {symbol} @ ${stop_loss_price:,.2f} "
                            f"(qty={total_qty:.6f})"
                        )
            
            return result
        else:
            # 需要賣出
            # 賣出前先取消止損單（避免重複賣出）
            self.cancel_stop_loss(symbol)
            
            sell_value = abs(diff) * total_equity
            sell_qty = sell_value / current_price
            sell_qty = min(sell_qty, position_qty)
            return self.market_sell(symbol, sell_qty, reason=reason)

    # ── 連線檢查 ──────────────────────────────────────────

    def check_connection(self, symbols: list[str] | None = None) -> dict:
        """
        檢查 Binance API 連線狀態

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

        # 1. 伺服器時間
        try:
            data = self.http.get("/api/v3/time")
            ts = datetime.fromtimestamp(data["serverTime"] / 1000, tz=timezone.utc)
            result["server_time"] = ts.strftime("%Y-%m-%d %H:%M:%S UTC")
            logger.info(f"✅ 伺服器連線正常: {result['server_time']}")
        except Exception as e:
            result["server_time_error"] = str(e)
            logger.error(f"❌ 伺服器連線失敗: {e}")

        # 2. 帳戶資訊
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

            logger.info(f"✅ 帳戶連線正常: canTrade={result['can_trade']}")
            logger.info(f"   USDT 餘額: ${result['usdt_balance']:,.2f}")
            if balances:
                for asset, val in balances.items():
                    if asset != "USDT" and val["free"] > 0:
                        logger.info(f"   {asset}: {val['free']}")
        except Exception as e:
            result["account_error"] = str(e)
            logger.error(f"❌ 帳戶查詢失敗: {e}")

        # 3. 交易對價格 + 規則
        if symbols:
            prices = {}
            filters = {}
            for sym in symbols:
                try:
                    p = self.get_price(sym)
                    prices[sym] = p
                    logger.info(f"   {sym}: ${p:,.2f}")
                except Exception as e:
                    logger.warning(f"   {sym}: 獲取價格失敗 - {e}")
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
