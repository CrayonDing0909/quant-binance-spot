#!/usr/bin/env python3
"""
Fixed-amount Binance Spot DCA runner.

Examples:
    python scripts/run_dca.py -c config/dca.yaml
    python scripts/run_dca.py -c config/dca.yaml --execute
    python scripts/run_dca.py -c config/dca.yaml --execute --force
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import yaml
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qtrade.live.binance_spot_broker import BinanceSpotBroker, OrderResult
from qtrade.monitor.notifier import TelegramNotifier
from qtrade.utils.log import get_logger

logger = get_logger("dca")


@dataclass(frozen=True)
class DCAOrder:
    symbol: str
    quote_qty: float


@dataclass(frozen=True)
class DCAConfig:
    dry_run: bool
    schedule: str
    timezone: str
    state_path: Path
    quote_asset: str
    min_quote_balance_after: float
    max_total_quote_per_run: float
    max_total_quote_per_day: float
    redeem_flexible_before_order: bool
    redeem_extra_quote: float
    notify: bool
    orders: list[DCAOrder]


def load_dca_config(path: str) -> DCAConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    data = raw.get("dca", raw)

    orders = [
        DCAOrder(symbol=str(item["symbol"]).upper(), quote_qty=float(item["quote_qty"]))
        for item in data.get("orders", [])
    ]
    if not orders:
        raise ValueError("dca.orders must contain at least one order")
    if any(order.quote_qty <= 0 for order in orders):
        raise ValueError("all dca.orders quote_qty values must be > 0")

    return DCAConfig(
        dry_run=bool(data.get("dry_run", True)),
        schedule=str(data.get("schedule", "daily")).lower(),
        timezone=str(data.get("timezone", "Asia/Taipei")),
        state_path=Path(data.get("state_path", "reports/dca/state.json")),
        quote_asset=str(data.get("quote_asset", "USDT")).upper(),
        min_quote_balance_after=float(data.get("min_quote_balance_after", 0.0)),
        max_total_quote_per_run=float(data.get("max_total_quote_per_run", 0.0)),
        max_total_quote_per_day=float(data.get("max_total_quote_per_day", 0.0)),
        redeem_flexible_before_order=bool(data.get("redeem_flexible_before_order", False)),
        redeem_extra_quote=float(data.get("redeem_extra_quote", 0.0)),
        notify=bool(data.get("notify", True)),
        orders=orders,
    )


def period_key(now: datetime, schedule: str) -> str:
    if schedule == "daily":
        return now.strftime("%Y-%m-%d")
    if schedule == "weekly":
        year, week, _ = now.isocalendar()
        return f"{year}-W{week:02d}"
    if schedule == "monthly":
        return now.strftime("%Y-%m")
    raise ValueError("dca.schedule must be daily, weekly, or monthly")


def load_state(path: Path) -> dict:
    if not path.exists():
        return {"completed": [], "runs": []}
    with open(path, "r", encoding="utf-8") as f:
        state = json.load(f)
    state.setdefault("completed", [])
    state.setdefault("runs", [])
    return state


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def completed_key(period: str, symbol: str) -> str:
    return f"{period}:{symbol.upper()}"


def spent_today(state: dict, local_date: str) -> float:
    total = 0.0
    for run in state.get("runs", []):
        if run.get("date") == local_date and run.get("live") is True:
            total += float(run.get("quote_qty", 0.0))
    return total


def validate_limits(cfg: DCAConfig, state: dict, local_date: str) -> None:
    run_total = sum(order.quote_qty for order in cfg.orders)
    if cfg.max_total_quote_per_run > 0 and run_total > cfg.max_total_quote_per_run:
        raise ValueError(
            f"run total {run_total:.2f} {cfg.quote_asset} exceeds "
            f"max_total_quote_per_run {cfg.max_total_quote_per_run:.2f}"
        )

    day_total = spent_today(state, local_date) + run_total
    if cfg.max_total_quote_per_day > 0 and day_total > cfg.max_total_quote_per_day:
        raise ValueError(
            f"daily total {day_total:.2f} {cfg.quote_asset} exceeds "
            f"max_total_quote_per_day {cfg.max_total_quote_per_day:.2f}"
        )


def get_flexible_position(broker: BinanceSpotBroker, asset: str) -> dict | None:
    data = broker.http.signed_get(
        "/sapi/v1/simple-earn/flexible/position",
        {"asset": asset, "size": 100},
    )
    rows = data.get("rows", []) if isinstance(data, dict) else []
    for row in rows:
        if row.get("asset") == asset and row.get("productId"):
            return row
    return None


def redeem_flexible_quote(
    broker: BinanceSpotBroker,
    asset: str,
    amount: float,
    dry_run: bool,
) -> str | None:
    if amount <= 0:
        return None

    position = get_flexible_position(broker, asset)
    if not position:
        raise RuntimeError(f"No Simple Earn Flexible position found for {asset}")

    product_id = str(position["productId"])
    available = float(position.get("totalAmount", 0.0))
    if available < amount:
        raise RuntimeError(
            f"Simple Earn Flexible {asset} {available:.8f} is below redeem amount {amount:.8f}"
        )

    if dry_run:
        logger.info(
            "[DRY-RUN] Redeem Simple Earn Flexible %s %.8f to SPOT (productId=%s)",
            asset,
            amount,
            product_id,
        )
        return "DRY-RUN-REDEEM"

    result = broker.http.signed_post(
        "/sapi/v1/simple-earn/flexible/redeem",
        {
            "productId": product_id,
            "amount": f"{amount:.8f}",
            "destAccount": "SPOT",
        },
    )
    if not result.get("success"):
        raise RuntimeError(f"Flexible redeem failed: {result}")
    redeem_id = str(result.get("redeemId", "UNKNOWN"))
    logger.info("Redeemed Simple Earn Flexible %s %.8f to SPOT (redeemId=%s)", asset, amount, redeem_id)
    return redeem_id


def ensure_quote_balance(
    broker: BinanceSpotBroker,
    cfg: DCAConfig,
    required: float,
    dry_run: bool,
) -> str | None:
    balance = broker.get_balance(cfg.quote_asset)
    if balance >= required:
        return None
    if not cfg.redeem_flexible_before_order:
        raise RuntimeError(
            f"{cfg.quote_asset} balance {balance:.2f} is below required {required:.2f}"
        )

    redeem_amount = required - balance + max(cfg.redeem_extra_quote, 0.0)
    redeem_id = redeem_flexible_quote(broker, cfg.quote_asset, redeem_amount, dry_run)
    if dry_run:
        return redeem_id

    for _ in range(6):
        time.sleep(2)
        balance = broker.get_balance(cfg.quote_asset)
        if balance >= required:
            return redeem_id
    raise RuntimeError(
        f"{cfg.quote_asset} balance still below required {required:.2f} after redeem"
    )


def send_summary(notifier: TelegramNotifier, lines: list[str]) -> None:
    if notifier.enabled:
        notifier.send("<b>DCA run</b>\n" + "\n".join(lines))


def format_money(value: float) -> str:
    return f"{value:,.2f}U"


def report_day(state: dict) -> int:
    dates = {
        run.get("date")
        for run in state.get("runs", [])
        if run.get("live") is True and run.get("date")
    }
    return len(dates)


def generate_report(cfg: DCAConfig) -> str:
    broker = BinanceSpotBroker(dry_run=True)
    state = load_state(cfg.state_path)
    day = report_day(state)

    symbols = [order.symbol for order in cfg.orders]
    rows = []
    total_cost = 0.0
    total_value = 0.0

    for symbol in symbols:
        symbol_runs = [
            run for run in state.get("runs", [])
            if run.get("live") is True and run.get("symbol") == symbol
        ]
        invested = sum(float(run.get("quote_qty", 0.0)) for run in symbol_runs)
        qty = sum(float(run.get("qty", 0.0)) for run in symbol_runs)
        current_price = broker.get_price(symbol)
        current_value = qty * current_price
        pnl = current_value - invested
        pnl_pct = (pnl / invested * 100.0) if invested > 0 else 0.0
        avg_price = (invested / qty) if qty > 0 else 0.0

        total_cost += invested
        total_value += current_value
        rows.append({
            "symbol": symbol,
            "invested": invested,
            "qty": qty,
            "avg_price": avg_price,
            "current_price": current_price,
            "current_value": current_value,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
        })

    total_pnl = total_value - total_cost
    total_pnl_pct = (total_pnl / total_cost * 100.0) if total_cost > 0 else 0.0

    lines = [
        f"因為判斷為相對底部了, 所以持續購入BTC 與ETH Day{day}",
        "-",
        "定投報告",
        f"累積投入: {format_money(total_cost)}",
        f"目前市值: {format_money(total_value)}",
        f"未實現損益: {total_pnl:+,.2f}U ({total_pnl_pct:+.2f}%)",
        "",
        "持倉明細",
    ]
    for row in rows:
        lines.extend([
            f"{row['symbol']}:",
            f"  投入金額: {format_money(row['invested'])}",
            f"  持有數量: {row['qty']:.8f}",
            f"  平均成本: {format_money(row['avg_price'])}",
            f"  目前價格: {format_money(row['current_price'])}",
            f"  目前市值: {format_money(row['current_value'])}",
            f"  未實現損益: {row['pnl']:+,.2f}U ({row['pnl_pct']:+.2f}%)",
        ])

    return "\n".join(lines)


def run_dca(cfg: DCAConfig, execute: bool, force: bool, notify: bool) -> int:
    tz = ZoneInfo(cfg.timezone)
    now = datetime.now(tz)
    local_date = now.strftime("%Y-%m-%d")
    period = period_key(now, cfg.schedule)
    state = load_state(cfg.state_path)

    live = execute and not cfg.dry_run
    dry_run = not live
    validate_limits(cfg, state, local_date)

    broker = BinanceSpotBroker(dry_run=dry_run)
    notifier = TelegramNotifier() if notify and cfg.notify else None

    pending = [
        order for order in cfg.orders
        if force or completed_key(period, order.symbol) not in set(state.get("completed", []))
    ]
    if not pending:
        logger.info("No pending DCA orders for period %s", period)
        return 0

    total_quote = sum(order.quote_qty for order in pending)
    if live:
        required = total_quote + cfg.min_quote_balance_after
        redeem_id = ensure_quote_balance(broker, cfg, required, dry_run=False)
    elif cfg.redeem_flexible_before_order:
        redeem_id = ensure_quote_balance(
            broker,
            cfg,
            total_quote + cfg.min_quote_balance_after,
            dry_run=True,
        )
    else:
        redeem_id = None

    summary = [
        f"mode={'LIVE' if live else 'DRY_RUN'}",
        f"period={period}",
        f"total={total_quote:.2f} {cfg.quote_asset}",
    ]
    if redeem_id:
        summary.append(f"redeem_id={redeem_id}")

    exit_code = 0
    for order in pending:
        result: OrderResult | None = broker.market_buy(
            order.symbol,
            order.quote_qty,
            reason=f"DCA {period}",
        )
        if result is None:
            summary.append(f"FAIL {order.symbol} {order.quote_qty:.2f}")
            exit_code = 2
            continue

        summary.append(
            f"BUY {order.symbol} {result.value:.2f} @ ~{result.price:.8g} status={result.status}"
        )
        if live:
            key = completed_key(period, order.symbol)
            if key not in state["completed"]:
                state["completed"].append(key)
            state["runs"].append({
                "ts": now.isoformat(),
                "date": local_date,
                "period": period,
                "symbol": order.symbol,
                "quote_qty": float(result.value),
                "qty": float(result.qty),
                "price": float(result.price),
                "fee": float(result.fee),
                "order_id": result.order_id,
                "live": True,
            })
            save_state(cfg.state_path, state)

    for line in summary:
        logger.info(line)
    if notifier:
        send_summary(notifier, summary)

    return exit_code


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed-amount Binance Spot DCA")
    parser.add_argument("--config", "-c", default="config/dca.yaml")
    parser.add_argument("--execute", action="store_true", help="place live orders")
    parser.add_argument("--dry-run", action="store_true", help="force dry-run mode")
    parser.add_argument("--force", action="store_true", help="ignore duplicate protection")
    parser.add_argument("--notify", action="store_true", help="force Telegram notification")
    parser.add_argument("--no-notify", action="store_true", help="disable Telegram notification")
    parser.add_argument("--report", action="store_true", help="print DCA report and exit")
    args = parser.parse_args()

    load_dotenv()
    cfg = load_dca_config(args.config)
    if args.dry_run:
        cfg = DCAConfig(**{**cfg.__dict__, "dry_run": True})

    if args.report:
        print(generate_report(cfg))
        return

    notify = args.notify or (cfg.notify and not args.no_notify)
    try:
        raise SystemExit(run_dca(cfg, execute=args.execute, force=args.force, notify=notify))
    except Exception as exc:
        logger.error("DCA failed: %s", exc)
        if notify:
            notifier = TelegramNotifier()
            if notifier.enabled:
                notifier.send_error(f"DCA failed: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
