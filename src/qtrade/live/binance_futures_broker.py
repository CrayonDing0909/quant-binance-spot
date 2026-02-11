"""
Binance Futures Broker — USDT-M 合約真實下單引擎

通過 Binance Futures REST API 執行真實交易。
需要設置環境變數：
    BINANCE_API_KEY
    BINANCE_API_SECRET

功能：
    - 市價開多/開空 (market_long, market_short)
    - 市價平倉 (market_close)
    - 限價開多/開空 (limit_long, limit_short)
    - 目標倉位執行 (execute_target_position)
    - 槓桿設定 (set_leverage)
    - 保證金類型設定 (set_margin_type)
    - 自動處理 LOT_SIZE (stepSize / minQty) 和 MIN_NOTIONAL
    - 訂單管理（查詢/取消訂單）
    - dry-run 模式（只記錄不下單）

注意事項：
    - 本模組使用 USDT-M 永續合約 (fapi.binance.com)
    - 預設使用逐倉模式 (ISOLATED)
    - 建議先用 dry_run=True 測試
"""
from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from ..data.binance_futures_client import BinanceFuturesHTTP
from ..utils.log import get_logger

logger = get_logger("binance_futures_broker")

# Binance Futures 手續費率（用於估算，實際以交易所為準）
# Maker: 0.02%, Taker: 0.04% (VIP 0)
# 使用 BNB 抵扣可降 10%
FEE_RATE_MAKER = 0.0002  # 0.02%
FEE_RATE_TAKER = 0.0004  # 0.04%


@dataclass
class FuturesOrderResult:
    """合約交易結果"""
    order_id: str
    symbol: str
    side: str           # BUY / SELL
    position_side: str  # LONG / SHORT / BOTH
    qty: float
    price: float
    fee: float          # 估算手續費
    value: float        # qty * price
    pnl: float | None   # 平倉時估算 PnL
    status: str
    reason: str = ""
    raw: dict = field(default_factory=dict)


@dataclass
class FuturesSymbolFilter:
    """Binance 合約交易對的下單規則"""
    min_qty: float = 0.0
    max_qty: float = float("inf")
    step_size: float = 0.0
    min_notional: float = 5.0  # 最小下單金額
    tick_size: float = 0.0
    price_precision: int = 2
    quantity_precision: int = 3

    def round_qty(self, qty: float) -> float:
        """根據 stepSize 對齊數量"""
        if self.step_size <= 0:
            return round(qty, self.quantity_precision)
        precision = max(0, -int(math.log10(self.step_size)))
        return math.floor(qty * 10**precision) / 10**precision

    def round_price(self, price: float) -> float:
        """根據 tickSize 對齊價格"""
        if self.tick_size <= 0:
            return round(price, self.price_precision)
        precision = max(0, -int(math.log10(self.tick_size)))
        return round(price, precision)

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


@dataclass
class FuturesPosition:
    """合約持倉資訊"""
    symbol: str
    position_side: str  # LONG / SHORT / BOTH
    qty: float          # 正數 = 多，負數 = 空
    entry_price: float
    unrealized_pnl: float
    leverage: int
    margin_type: str    # ISOLATED / CROSSED
    liquidation_price: float = 0.0

    @property
    def is_open(self) -> bool:
        return abs(self.qty) > 1e-10

    @property
    def notional(self) -> float:
        """名義價值"""
        return abs(self.qty * self.entry_price)


class BinanceFuturesBroker:
    """
    Binance USDT-M 合約真實下單引擎

    支援雙向持倉模式，可同時做多做空。
    
    下單方向對應：
        做多開倉: side=BUY,  positionSide=LONG
        做多平倉: side=SELL, positionSide=LONG
        做空開倉: side=SELL, positionSide=SHORT
        做空平倉: side=BUY,  positionSide=SHORT

    Args:
        dry_run: True = 只記錄不下單（用於測試）
        leverage: 預設槓桿倍數
        margin_type: 保證金類型 ("ISOLATED" / "CROSSED")
    """

    def __init__(
        self,
        dry_run: bool = False,
        leverage: int = 10,
        margin_type: Literal["ISOLATED", "CROSSED"] = "ISOLATED",
    ):
        self.http = BinanceFuturesHTTP()
        self.dry_run = dry_run
        self.default_leverage = leverage
        self.default_margin_type = margin_type
        
        self._filters: dict[str, FuturesSymbolFilter] = {}
        self._leverage_cache: dict[str, int] = {}
        self._margin_type_cache: dict[str, str] = {}

        if not self.http.api_key or not self.http.api_secret:
            raise RuntimeError(
                "❌ 需要設置環境變數 BINANCE_API_KEY 和 BINANCE_API_SECRET\n"
                "   請在 .env 檔案中配置"
            )

        mode_str = "🧪 DRY-RUN（不會真的下單）" if dry_run else "💰 LIVE（真金白銀！）"
        logger.info(
            f"✅ Binance Futures Broker 初始化完成 [{mode_str}]\n"
            f"   預設槓桿: {leverage}x, 保證金類型: {margin_type}"
        )

    # ── 交易對規則 ────────────────────────────────────────

    def _get_filter(self, symbol: str) -> FuturesSymbolFilter:
        """從 exchangeInfo 獲取交易對的下單規則"""
        if symbol in self._filters:
            return self._filters[symbol]

        try:
            data = self.http.get("/fapi/v1/exchangeInfo")
            for sym_info in data.get("symbols", []):
                if sym_info["symbol"] == symbol:
                    sf = FuturesSymbolFilter(
                        price_precision=sym_info.get("pricePrecision", 2),
                        quantity_precision=sym_info.get("quantityPrecision", 3),
                    )
                    for f in sym_info.get("filters", []):
                        if f["filterType"] == "LOT_SIZE":
                            sf.min_qty = float(f["minQty"])
                            sf.max_qty = float(f["maxQty"])
                            sf.step_size = float(f["stepSize"])
                        elif f["filterType"] == "MIN_NOTIONAL":
                            sf.min_notional = float(f.get("notional", 5))
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

        sf = FuturesSymbolFilter()
        self._filters[symbol] = sf
        return sf

    # ── 帳戶 / 槓桿 / 保證金 ──────────────────────────────

    def get_balance(self, asset: str = "USDT") -> float:
        """查詢可用餘額"""
        try:
            data = self.http.signed_get("/fapi/v2/balance", {})
            for b in data:
                if b["asset"] == asset:
                    return float(b["availableBalance"])
            return 0.0
        except Exception as e:
            logger.error(f"查詢餘額失敗: {e}")
            return 0.0

    def get_account_info(self) -> dict:
        """查詢帳戶資訊"""
        try:
            return self.http.signed_get("/fapi/v2/account", {})
        except Exception as e:
            logger.error(f"查詢帳戶失敗: {e}")
            return {}

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """設定槓桿倍數"""
        if self.dry_run:
            logger.info(f"🧪 [DRY-RUN] 設定 {symbol} 槓桿: {leverage}x")
            self._leverage_cache[symbol] = leverage
            return True

        try:
            self.http.signed_post("/fapi/v1/leverage", {
                "symbol": symbol,
                "leverage": leverage,
            })
            self._leverage_cache[symbol] = leverage
            logger.info(f"⚙️  {symbol} 槓桿已設定: {leverage}x")
            return True
        except Exception as e:
            if "No need to change" in str(e):
                self._leverage_cache[symbol] = leverage
                return True
            logger.error(f"❌ 設定槓桿失敗 {symbol}: {e}")
            return False

    def set_margin_type(self, symbol: str, margin_type: Literal["ISOLATED", "CROSSED"]) -> bool:
        """設定保證金類型"""
        if self.dry_run:
            logger.info(f"🧪 [DRY-RUN] 設定 {symbol} 保證金類型: {margin_type}")
            self._margin_type_cache[symbol] = margin_type
            return True

        try:
            self.http.signed_post("/fapi/v1/marginType", {
                "symbol": symbol,
                "marginType": margin_type,
            })
            self._margin_type_cache[symbol] = margin_type
            logger.info(f"⚙️  {symbol} 保證金類型已設定: {margin_type}")
            return True
        except Exception as e:
            if "No need to change" in str(e):
                self._margin_type_cache[symbol] = margin_type
                return True
            logger.error(f"❌ 設定保證金類型失敗 {symbol}: {e}")
            return False

    def ensure_trading_settings(self, symbol: str) -> None:
        """確保交易設定（槓桿、保證金類型）已正確設定"""
        if symbol not in self._leverage_cache:
            self.set_leverage(symbol, self.default_leverage)
        if symbol not in self._margin_type_cache:
            self.set_margin_type(symbol, self.default_margin_type)

    # ── 持倉查詢 ──────────────────────────────────────────

    def get_position(self, symbol: str) -> FuturesPosition | None:
        """
        查詢持倉（淨持倉模式）
        
        Returns:
            FuturesPosition 或 None（無持倉）
        """
        try:
            data = self.http.signed_get("/fapi/v2/positionRisk", {"symbol": symbol})
            for pos in data:
                qty = float(pos["positionAmt"])
                if abs(qty) > 1e-10:
                    return FuturesPosition(
                        symbol=symbol,
                        position_side=pos.get("positionSide", "BOTH"),
                        qty=qty,
                        entry_price=float(pos["entryPrice"]),
                        unrealized_pnl=float(pos["unRealizedProfit"]),
                        leverage=int(pos.get("leverage", self.default_leverage)),
                        margin_type=pos.get("marginType", "isolated").upper(),
                        liquidation_price=float(pos.get("liquidationPrice", 0)),
                    )
            return None
        except Exception as e:
            logger.error(f"查詢持倉失敗 {symbol}: {e}")
            return None

    def get_positions(self) -> list[FuturesPosition]:
        """查詢所有持倉"""
        try:
            data = self.http.signed_get("/fapi/v2/positionRisk", {})
            positions = []
            for pos in data:
                qty = float(pos["positionAmt"])
                if abs(qty) > 1e-10:
                    positions.append(FuturesPosition(
                        symbol=pos["symbol"],
                        position_side=pos.get("positionSide", "BOTH"),
                        qty=qty,
                        entry_price=float(pos["entryPrice"]),
                        unrealized_pnl=float(pos["unRealizedProfit"]),
                        leverage=int(pos.get("leverage", self.default_leverage)),
                        margin_type=pos.get("marginType", "isolated").upper(),
                        liquidation_price=float(pos.get("liquidationPrice", 0)),
                    ))
            return positions
        except Exception as e:
            logger.error(f"查詢持倉失敗: {e}")
            return []

    def get_position_pct(self, symbol: str, current_price: float) -> float:
        """
        獲取持倉佔權益比例 [-1, 1]
        
        Returns:
            正數 = 多倉，負數 = 空倉
        """
        pos = self.get_position(symbol)
        if not pos or not pos.is_open or current_price <= 0:
            return 0.0

        equity = self.get_equity()
        if equity <= 0:
            return 0.0

        # 名義價值 / 權益
        notional = pos.qty * current_price
        return notional / equity

    def get_price(self, symbol: str) -> float:
        """查詢最新標記價格"""
        try:
            data = self.http.get("/fapi/v1/premiumIndex", {"symbol": symbol})
            return float(data["markPrice"])
        except Exception as e:
            logger.error(f"查詢價格失敗 {symbol}: {e}")
            return 0.0

    def get_equity(self) -> float:
        """查詢帳戶總權益（錢包餘額 + 未實現盈虧）"""
        try:
            data = self.http.signed_get("/fapi/v2/account", {})
            return float(data.get("totalWalletBalance", 0)) + float(data.get("totalUnrealizedProfit", 0))
        except Exception as e:
            logger.error(f"查詢權益失敗: {e}")
            return 0.0

    # ── 市價單 ──────────────────────────────────────────────

    def market_long(
        self,
        symbol: str,
        qty: float | None = None,
        usdt_value: float | None = None,
        reason: str = "",
    ) -> FuturesOrderResult | None:
        """
        市價做多
        
        Args:
            symbol: 交易對
            qty: 數量（與 usdt_value 二選一）
            usdt_value: USDT 金額（會根據價格計算數量）
            reason: 下單原因
        """
        self.ensure_trading_settings(symbol)
        sf = self._get_filter(symbol)
        price = self.get_price(symbol)

        if usdt_value and not qty:
            qty = usdt_value / price if price > 0 else 0
        if not qty:
            return None

        qty = sf.round_qty(qty)
        ok, msg = sf.validate_qty(qty)
        if not ok:
            logger.warning(f"⚠️  {symbol} 做多數量不合規: {msg}")
            return None

        ok, msg = sf.validate_notional(qty, price)
        if not ok:
            logger.warning(f"⚠️  {symbol} 做多金額不足: {msg}")
            return None

        if self.dry_run:
            est_fee = qty * price * FEE_RATE_TAKER
            logger.info(
                f"🧪 [DRY-RUN] LONG {symbol}: {qty:.6f} @ ~${price:,.2f} "
                f"(reason={reason})"
            )
            return FuturesOrderResult(
                order_id="DRY-RUN",
                symbol=symbol,
                side="BUY",
                position_side="LONG",
                qty=qty,
                price=price,
                fee=est_fee,
                value=qty * price,
                pnl=None,
                status="DRY_RUN",
                reason=reason,
            )

        try:
            result = self.http.signed_post("/fapi/v1/order", {
                "symbol": symbol,
                "side": "BUY",
                "type": "MARKET",
                "quantity": f"{qty}",
            })

            exec_qty = float(result.get("executedQty", 0))
            avg_price = float(result.get("avgPrice", price))
            est_fee = exec_qty * avg_price * FEE_RATE_TAKER

            order = FuturesOrderResult(
                order_id=str(result["orderId"]),
                symbol=symbol,
                side="BUY",
                position_side="LONG",
                qty=exec_qty,
                price=avg_price,
                fee=est_fee,
                value=exec_qty * avg_price,
                pnl=None,
                status=result.get("status", "UNKNOWN"),
                reason=reason,
                raw=result,
            )
            logger.info(
                f"📗 LONG {symbol}: {order.qty:.6f} @ ${order.price:,.2f} "
                f"(orderId={order.order_id})"
            )
            return order

        except Exception as e:
            # 嘗試解析 Binance 錯誤詳情
            error_msg = str(e)
            try:
                import json
                if hasattr(e, 'response') and e.response is not None:
                    error_detail = e.response.json()
                    error_msg = f"{e} | Binance: {error_detail}"
            except Exception:
                pass
            logger.error(f"❌ 做多失敗 {symbol}: {error_msg}")
            return None

    def market_short(
        self,
        symbol: str,
        qty: float | None = None,
        usdt_value: float | None = None,
        reason: str = "",
    ) -> FuturesOrderResult | None:
        """
        市價做空
        
        Args:
            symbol: 交易對
            qty: 數量（與 usdt_value 二選一）
            usdt_value: USDT 金額（會根據價格計算數量）
            reason: 下單原因
        """
        self.ensure_trading_settings(symbol)
        sf = self._get_filter(symbol)
        price = self.get_price(symbol)

        if usdt_value and not qty:
            qty = usdt_value / price if price > 0 else 0
        if not qty:
            return None

        qty = sf.round_qty(qty)
        ok, msg = sf.validate_qty(qty)
        if not ok:
            logger.warning(f"⚠️  {symbol} 做空數量不合規: {msg}")
            return None

        ok, msg = sf.validate_notional(qty, price)
        if not ok:
            logger.warning(f"⚠️  {symbol} 做空金額不足: {msg}")
            return None

        if self.dry_run:
            est_fee = qty * price * FEE_RATE_TAKER
            logger.info(
                f"🧪 [DRY-RUN] SHORT {symbol}: {qty:.6f} @ ~${price:,.2f} "
                f"(reason={reason})"
            )
            return FuturesOrderResult(
                order_id="DRY-RUN",
                symbol=symbol,
                side="SELL",
                position_side="SHORT",
                qty=qty,
                price=price,
                fee=est_fee,
                value=qty * price,
                pnl=None,
                status="DRY_RUN",
                reason=reason,
            )

        try:
            result = self.http.signed_post("/fapi/v1/order", {
                "symbol": symbol,
                "side": "SELL",
                "type": "MARKET",
                "quantity": f"{qty}",
            })

            exec_qty = float(result.get("executedQty", 0))
            avg_price = float(result.get("avgPrice", price))
            est_fee = exec_qty * avg_price * FEE_RATE_TAKER

            order = FuturesOrderResult(
                order_id=str(result["orderId"]),
                symbol=symbol,
                side="SELL",
                position_side="SHORT",
                qty=exec_qty,
                price=avg_price,
                fee=est_fee,
                value=exec_qty * avg_price,
                pnl=None,
                status=result.get("status", "UNKNOWN"),
                reason=reason,
                raw=result,
            )
            logger.info(
                f"📕 SHORT {symbol}: {order.qty:.6f} @ ${order.price:,.2f} "
                f"(orderId={order.order_id})"
            )
            return order

        except Exception as e:
            logger.error(f"❌ 做空失敗 {symbol}: {e}")
            return None

    def market_close(
        self,
        symbol: str,
        qty: float | None = None,
        reason: str = "close",
    ) -> FuturesOrderResult | None:
        """
        市價平倉
        
        Args:
            symbol: 交易對
            qty: 平倉數量（None = 全部平倉）
            reason: 下單原因
        """
        pos = self.get_position(symbol)
        if not pos or not pos.is_open:
            logger.warning(f"⚠️  {symbol} 無持倉可平")
            return None

        close_qty = qty if qty else abs(pos.qty)
        close_qty = min(close_qty, abs(pos.qty))

        sf = self._get_filter(symbol)
        close_qty = sf.round_qty(close_qty)
        price = self.get_price(symbol)

        # 平多 = SELL，平空 = BUY
        side = "SELL" if pos.qty > 0 else "BUY"
        position_label = "CLOSE_LONG" if pos.qty > 0 else "CLOSE_SHORT"

        if self.dry_run:
            # 估算 PnL
            if pos.qty > 0:  # 多倉
                pnl = (price - pos.entry_price) * close_qty
            else:  # 空倉
                pnl = (pos.entry_price - price) * close_qty
            est_fee = close_qty * price * FEE_RATE_TAKER

            logger.info(
                f"🧪 [DRY-RUN] {position_label} {symbol}: {close_qty:.6f} @ ~${price:,.2f} "
                f"(pnl={pnl:+.2f}, reason={reason})"
            )
            return FuturesOrderResult(
                order_id="DRY-RUN",
                symbol=symbol,
                side=side,
                position_side=pos.position_side,
                qty=close_qty,
                price=price,
                fee=est_fee,
                value=close_qty * price,
                pnl=pnl,
                status="DRY_RUN",
                reason=reason,
            )

        try:
            result = self.http.signed_post("/fapi/v1/order", {
                "symbol": symbol,
                "side": side,
                "type": "MARKET",
                "quantity": f"{close_qty}",
                "reduceOnly": "true",
            })

            exec_qty = float(result.get("executedQty", 0))
            avg_price = float(result.get("avgPrice", price))
            est_fee = exec_qty * avg_price * FEE_RATE_TAKER

            # 計算 PnL
            if pos.qty > 0:  # 多倉
                pnl = (avg_price - pos.entry_price) * exec_qty
            else:  # 空倉
                pnl = (pos.entry_price - avg_price) * exec_qty

            order = FuturesOrderResult(
                order_id=str(result["orderId"]),
                symbol=symbol,
                side=side,
                position_side=pos.position_side,
                qty=exec_qty,
                price=avg_price,
                fee=est_fee,
                value=exec_qty * avg_price,
                pnl=pnl,
                status=result.get("status", "UNKNOWN"),
                reason=reason,
                raw=result,
            )

            emoji = "📈" if pnl > 0 else "📉"
            logger.info(
                f"{emoji} {position_label} {symbol}: {order.qty:.6f} @ ${order.price:,.2f} "
                f"(pnl={pnl:+.2f}, orderId={order.order_id})"
            )
            return order

        except Exception as e:
            logger.error(f"❌ 平倉失敗 {symbol}: {e}")
            return None

    def close_all_positions(self) -> list[FuturesOrderResult]:
        """平掉所有持倉"""
        results = []
        for pos in self.get_positions():
            r = self.market_close(pos.symbol, reason="close_all")
            if r:
                results.append(r)
        return results

    # ── 止損單 ──────────────────────────────────────────────

    def place_stop_loss(
        self,
        symbol: str,
        stop_price: float,
        position_side: str = "LONG",
        qty: float | None = None,
        reason: str = "stop_loss",
    ) -> FuturesOrderResult | None:
        """
        預掛止損單（STOP_MARKET）

        當價格觸及 stop_price 時，交易所自動執行平倉。
        即使程式斷線，止損單依然有效。

        Args:
            symbol: 交易對
            stop_price: 止損觸發價格
            position_side: "LONG" = 平多倉止損, "SHORT" = 平空倉止損
            qty: 止損數量（None = 全部平倉）
            reason: 原因

        Returns:
            FuturesOrderResult 或 None
        """
        sf = self._get_filter(symbol)
        
        # 止損價格精度處理
        if sf.tick_size > 0:
            import math
            precision = max(0, -int(math.log10(sf.tick_size)))
            stop_price = round(stop_price, precision)

        # 平多倉 = SELL, 平空倉 = BUY
        side = "SELL" if position_side == "LONG" else "BUY"
        
        if self.dry_run:
            logger.info(
                f"🧪 [DRY-RUN] 止損單 {symbol} [{position_side}]: "
                f"trigger @ ${stop_price:,.2f} (reason={reason})"
            )
            return FuturesOrderResult(
                order_id="DRY-RUN-SL",
                symbol=symbol,
                side=side,
                position_side=position_side,
                qty=qty or 0,
                price=stop_price,
                fee=0,
                value=0,
                pnl=None,
                status="DRY_RUN",
                reason=reason,
            )

        try:
            # 先取消舊的止損單
            self.cancel_stop_loss(symbol)

            params = {
                "symbol": symbol,
                "side": side,
                "type": "STOP_MARKET",
                "stopPrice": f"{stop_price}",
                "closePosition": "true",  # 全部平倉
            }
            
            # 如果指定數量，就不用 closePosition
            if qty:
                qty = sf.round_qty(qty)
                params["quantity"] = f"{qty}"
                del params["closePosition"]

            result = self.http.signed_post("/fapi/v1/order", params)

            order = FuturesOrderResult(
                order_id=str(result["orderId"]),
                symbol=symbol,
                side=side,
                position_side=position_side,
                qty=qty or 0,
                price=stop_price,
                fee=0,
                value=0,
                pnl=None,
                status=result.get("status", "NEW"),
                reason=reason,
                raw=result,
            )
            logger.info(
                f"🛡️  止損單已掛 {symbol} [{position_side}]: "
                f"trigger @ ${stop_price:,.2f} (orderId={order.order_id})"
            )
            return order

        except Exception as e:
            logger.error(f"❌ 掛止損單失敗 {symbol}: {e}")
            return None

    def cancel_stop_loss(self, symbol: str) -> bool:
        """取消該交易對的止損單（STOP_MARKET 類型）"""
        if self.dry_run:
            logger.debug(f"🧪 [DRY-RUN] 取消止損單 {symbol}")
            return True

        try:
            orders = self.get_open_orders(symbol)
            for order in orders:
                if order.get("type") == "STOP_MARKET":
                    self.cancel_order(symbol, str(order["orderId"]))
                    logger.info(f"🗑️  止損單已取消 {symbol} orderId={order['orderId']}")
            return True
        except Exception as e:
            logger.warning(f"⚠️  取消止損單失敗 {symbol}: {e}")
            return False

    def get_active_stop_order(self, symbol: str) -> dict | None:
        """查詢該交易對的止損單"""
        try:
            orders = self.get_open_orders(symbol)
            for order in orders:
                if order.get("type") == "STOP_MARKET":
                    return order
            return None
        except Exception:
            return None

    # ── 目標倉位執行 ────────────────────────────────────────

    def execute_target_position(
        self,
        symbol: str,
        target_pct: float,
        current_price: float | None = None,
        reason: str = "signal",
        stop_loss_price: float | None = None,
    ) -> FuturesOrderResult | None:
        """
        執行目標倉位調整
        
        將持倉調整到 target_pct（佔總權益比例）。
        開倉後會自動掛止損單（如果提供 stop_loss_price）。
        
        Args:
            symbol: 交易對
            target_pct: 目標倉位比例 [-1, 1]
                - 正數 = 做多
                - 負數 = 做空
                - 0 = 平倉
            current_price: 當前價格（None 時自動查詢）
            reason: 下單原因
            stop_loss_price: 止損價格（None = 不掛止損）
            
        Returns:
            FuturesOrderResult 或 None
        """
        if current_price is None:
            current_price = self.get_price(symbol)
        if current_price <= 0:
            logger.error(f"無法獲取 {symbol} 價格")
            return None

        target_pct = max(-1.0, min(1.0, target_pct))
        current_pct = self.get_position_pct(symbol, current_price)
        diff = target_pct - current_pct

        logger.info(
            f"📊 {symbol}: 當前={current_pct:+.1%}, 目標={target_pct:+.1%}, "
            f"差距={diff:+.1%}"
        )

        # 差距太小不交易
        if abs(diff) < 0.02:
            logger.debug(f"  {symbol}: 差距 < 2%，跳過")
            return None

        equity = self.get_equity()
        if equity <= 0:
            logger.error("帳戶權益為 0")
            return None

        pos = self.get_position(symbol)
        leverage = self._leverage_cache.get(symbol, self.default_leverage)

        # 計算需要變動的名義價值
        change_notional = abs(diff) * equity

        if target_pct == 0:
            # 目標是空倉 → 全部平倉
            # 平倉前先取消止損單
            self.cancel_stop_loss(symbol)
            return self.market_close(symbol, reason=reason)

        if diff > 0:
            # 需要增加多頭曝險
            if pos and pos.qty < 0:
                # 有空倉，先平空
                self.cancel_stop_loss(symbol)
                close_qty = min(change_notional / current_price, abs(pos.qty))
                return self.market_close(symbol, qty=close_qty, reason=f"{reason}_close_short")
            else:
                # 開多或加多
                qty = change_notional / current_price
                result = self.market_long(symbol, qty=qty, reason=reason)
                
                # 開倉成功後掛止損單
                if result and stop_loss_price and stop_loss_price > 0:
                    self.place_stop_loss(
                        symbol=symbol,
                        stop_price=stop_loss_price,
                        position_side="LONG",
                        reason="auto_stop_loss",
                    )
                return result
        else:
            # 需要增加空頭曝險（或減少多頭）
            if pos and pos.qty > 0:
                # 有多倉，先平多
                self.cancel_stop_loss(symbol)
                close_qty = min(change_notional / current_price, pos.qty)
                return self.market_close(symbol, qty=close_qty, reason=f"{reason}_close_long")
            else:
                # 開空或加空
                qty = change_notional / current_price
                result = self.market_short(symbol, qty=qty, reason=reason)
                
                # 開空成功後掛止損單（空倉止損價在上方）
                if result and stop_loss_price and stop_loss_price > 0:
                    self.place_stop_loss(
                        symbol=symbol,
                        stop_price=stop_loss_price,
                        position_side="SHORT",
                        reason="auto_stop_loss",
                    )
                return result

    # ── 訂單管理 ──────────────────────────────────────────

    def get_open_orders(self, symbol: str | None = None) -> list[dict]:
        """查詢未成交訂單"""
        try:
            params = {}
            if symbol:
                params["symbol"] = symbol
            result = self.http.signed_get("/fapi/v1/openOrders", params)
            return result if isinstance(result, list) else []
        except Exception as e:
            logger.error(f"❌ 查詢未成交訂單失敗: {e}")
            return []

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """取消訂單"""
        if self.dry_run:
            logger.info(f"🧪 [DRY-RUN] 取消訂單 {symbol} orderId={order_id}")
            return True

        try:
            self.http.signed_delete("/fapi/v1/order", {
                "symbol": symbol,
                "orderId": order_id,
            })
            logger.info(f"🗑️  訂單已取消 {symbol} orderId={order_id}")
            return True
        except Exception as e:
            if "Unknown order" in str(e) or "UNKNOWN_ORDER" in str(e):
                return True
            logger.warning(f"⚠️  取消訂單失敗 {symbol}: {e}")
            return False

    def cancel_all_orders(self, symbol: str) -> bool:
        """取消某交易對的所有未成交訂單"""
        if self.dry_run:
            logger.info(f"🧪 [DRY-RUN] 取消 {symbol} 所有訂單")
            return True

        try:
            self.http.signed_delete("/fapi/v1/allOpenOrders", {"symbol": symbol})
            logger.info(f"🗑️  已取消 {symbol} 所有訂單")
            return True
        except Exception as e:
            logger.warning(f"⚠️  取消所有訂單失敗 {symbol}: {e}")
            return False

    # ── 連線檢查 ──────────────────────────────────────────

    def check_connection(self, symbols: list[str] | None = None) -> dict:
        """
        檢查 Binance Futures API 連線狀態
        """
        result = {}

        # 1. 伺服器時間
        try:
            data = self.http.get("/fapi/v1/time")
            ts = datetime.fromtimestamp(data["serverTime"] / 1000, tz=timezone.utc)
            result["server_time"] = ts.strftime("%Y-%m-%d %H:%M:%S UTC")
            logger.info(f"✅ Futures API 連線正常: {result['server_time']}")
        except Exception as e:
            result["server_time_error"] = str(e)
            logger.error(f"❌ Futures API 連線失敗: {e}")

        # 2. 帳戶資訊
        try:
            account = self.http.signed_get("/fapi/v2/account", {})
            result["can_trade"] = account.get("canTrade", False)
            result["total_wallet_balance"] = float(account.get("totalWalletBalance", 0))
            result["total_unrealized_profit"] = float(account.get("totalUnrealizedProfit", 0))
            result["total_margin_balance"] = float(account.get("totalMarginBalance", 0))
            result["available_balance"] = float(account.get("availableBalance", 0))

            logger.info(f"✅ 帳戶連線正常: canTrade={result['can_trade']}")
            logger.info(f"   錢包餘額: ${result['total_wallet_balance']:,.2f}")
            logger.info(f"   可用餘額: ${result['available_balance']:,.2f}")
            logger.info(f"   未實現盈虧: ${result['total_unrealized_profit']:+,.2f}")
        except Exception as e:
            result["account_error"] = str(e)
            logger.error(f"❌ 帳戶查詢失敗: {e}")

        # 3. 交易對價格
        if symbols:
            prices = {}
            for sym in symbols:
                try:
                    p = self.get_price(sym)
                    prices[sym] = p
                    logger.info(f"   {sym}: ${p:,.2f}")
                except Exception as e:
                    logger.warning(f"   {sym}: 獲取價格失敗 - {e}")
            result["prices"] = prices

        # 4. 持倉
        positions = self.get_positions()
        if positions:
            result["positions"] = [
                {
                    "symbol": p.symbol,
                    "side": "LONG" if p.qty > 0 else "SHORT",
                    "qty": p.qty,
                    "entry_price": p.entry_price,
                    "unrealized_pnl": p.unrealized_pnl,
                    "leverage": p.leverage,
                }
                for p in positions
            ]
            for p in positions:
                side = "LONG" if p.qty > 0 else "SHORT"
                logger.info(
                    f"   {p.symbol} [{side}]: {abs(p.qty):.4f} @ {p.entry_price:.2f} "
                    f"(PnL: {p.unrealized_pnl:+.2f})"
                )

        return result
