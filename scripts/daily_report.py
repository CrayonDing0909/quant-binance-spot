"""
每日績效報表 — 推送 Paper Trading 摘要到 Telegram

使用方法:
    # 手動執行
    PYTHONPATH=src python scripts/daily_report.py -c config/rsi_adx_atr.yaml

    # 配合 cron 每天 UTC 00:05 執行
    # crontab -e
    # 5 0 * * * cd /opt/qtrade && .venv/bin/python scripts/daily_report.py -c config/rsi_adx_atr.yaml

報表內容:
    📊 帳戶權益、收益率、最大回撤
    📋 當前持倉明細
    📈 今日交易記錄
    📉 過去 7 天收益趨勢
"""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

from qtrade.config import load_config
from qtrade.monitor.notifier import TelegramNotifier
from qtrade.live.signal_generator import fetch_recent_klines
from qtrade.utils.log import get_logger

logger = get_logger("daily_report")


def load_paper_state(state_path: Path) -> dict | None:
    """載入 Paper Trading 狀態"""
    if not state_path.exists():
        return None
    with open(state_path) as f:
        return json.load(f)


def get_current_prices(symbols: list[str], interval: str) -> dict[str, float]:
    """獲取當前價格"""
    prices = {}
    for sym in symbols:
        try:
            df = fetch_recent_klines(sym, interval, 5)
            prices[sym] = float(df["close"].iloc[-1])
        except Exception as e:
            logger.warning(f"無法獲取 {sym} 價格: {e}")
    return prices


def calculate_stats(state: dict, prices: dict[str, float]) -> dict:
    """計算績效統計"""
    initial_cash = state["initial_cash"]
    cash = state["cash"]

    # 持倉市值
    position_value = 0.0
    positions = {}
    for sym, pos in state.get("positions", {}).items():
        price = prices.get(sym, 0)
        qty = pos["qty"]
        entry = pos["avg_entry"]
        value = qty * price
        pnl = (price - entry) * qty
        pnl_pct = ((price / entry) - 1) * 100 if entry > 0 else 0
        position_value += value
        positions[sym] = {
            "qty": qty,
            "entry": entry,
            "price": price,
            "value": value,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
        }

    equity = cash + position_value
    total_return = (equity / initial_cash - 1) * 100
    drawdown = (1 - equity / initial_cash) * 100 if equity < initial_cash else 0

    # 交易統計
    trades = state.get("trades", [])
    total_trades = len(trades)
    wins = sum(1 for t in trades if t.get("pnl") and t["pnl"] > 0)
    losses = sum(1 for t in trades if t.get("pnl") and t["pnl"] < 0)
    total_pnl = sum(t.get("pnl", 0) or 0 for t in trades)
    total_fees = sum(t.get("fee", 0) for t in trades)
    win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0

    # 今日交易
    now = datetime.now(timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    today_ts = today_start.timestamp()
    today_trades = [t for t in trades if t["timestamp"] >= today_ts]

    # 最近 7 天交易
    week_ts = (now - timedelta(days=7)).timestamp()
    week_trades = [t for t in trades if t["timestamp"] >= week_ts]
    week_pnl = sum(t.get("pnl", 0) or 0 for t in week_trades)

    return {
        "initial_cash": initial_cash,
        "cash": cash,
        "equity": equity,
        "position_value": position_value,
        "total_return": total_return,
        "drawdown": drawdown,
        "positions": positions,
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "total_fees": total_fees,
        "today_trades": today_trades,
        "week_trades": week_trades,
        "week_pnl": week_pnl,
    }


def format_report(stats: dict, strategy_name: str) -> str:
    """格式化報表"""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    ret = stats["total_return"]
    emoji = "📈" if ret > 0 else "📉"

    lines = [
        f"📊 <b>Daily Report</b> [{strategy_name}]",
        f"    {now}",
        "",
        f"{'─' * 30}",
        f"  {emoji} 總收益: <b>{ret:+.2f}%</b>",
        f"  💰 權益: <b>${stats['equity']:,.2f}</b>",
        f"  💵 現金: ${stats['cash']:,.2f}",
        f"  📦 持倉: ${stats['position_value']:,.2f}",
    ]

    if stats["drawdown"] > 0:
        lines.append(f"  ⚠️ 回撤: -{stats['drawdown']:.2f}%")

    # 持倉明細
    if stats["positions"]:
        lines.append(f"\n{'─' * 30}")
        lines.append("  <b>持倉明細:</b>")
        for sym, p in stats["positions"].items():
            pnl_emoji = "🟢" if p["pnl"] > 0 else "🔴"
            lines.append(
                f"  {pnl_emoji} {sym}:\n"
                f"     {p['qty']:.6f} @ ${p['entry']:,.2f}\n"
                f"     現價 ${p['price']:,.2f} | "
                f"PnL: {p['pnl']:+.2f} ({p['pnl_pct']:+.1f}%)"
            )

    # 交易統計
    lines.append(f"\n{'─' * 30}")
    lines.append("  <b>交易統計:</b>")
    lines.append(f"  總交易: {stats['total_trades']} 筆")
    if stats["total_trades"] > 0:
        lines.append(f"  勝率: {stats['win_rate']:.1f}% ({stats['wins']}W / {stats['losses']}L)")
        lines.append(f"  總 PnL: ${stats['total_pnl']:+,.2f}")
        lines.append(f"  總手續費: ${stats['total_fees']:,.2f}")

    # 今日交易
    if stats["today_trades"]:
        lines.append(f"\n{'─' * 30}")
        lines.append(f"  <b>今日交易 ({len(stats['today_trades'])} 筆):</b>")
        for t in stats["today_trades"][-5:]:  # 最多顯示 5 筆
            ts = datetime.fromtimestamp(t["timestamp"], tz=timezone.utc).strftime("%H:%M")
            pnl_str = f" PnL={t['pnl']:+.2f}" if t.get("pnl") is not None else ""
            lines.append(
                f"  [{ts}] {'🟢' if t['side'] == 'BUY' else '🔴'} "
                f"{t['side']} {t['symbol']} "
                f"{t['qty']:.4f} @ ${t['price']:,.2f}{pnl_str}"
            )
    else:
        lines.append(f"\n  📭 今日無交易")

    # 7 天 PnL
    if stats["week_trades"]:
        week_emoji = "📈" if stats["week_pnl"] > 0 else "📉"
        lines.append(
            f"\n  {week_emoji} 近 7 天: {len(stats['week_trades'])} 筆, "
            f"PnL=${stats['week_pnl']:+,.2f}"
        )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper Trading 每日績效報表")
    parser.add_argument("-c", "--config", default="config/rsi_adx_atr.yaml")
    parser.add_argument("-s", "--strategy", default=None)
    parser.add_argument("--print-only", action="store_true", help="只列印不發送 Telegram")
    args = parser.parse_args()

    cfg = load_config(args.config)
    strategy_name = args.strategy or cfg.strategy.name

    # 載入狀態
    state_path = cfg.get_report_dir("live") / "paper_state.json"
    state = load_paper_state(state_path)

    if state is None:
        print(f"❌ Paper Trading 狀態檔不存在: {state_path}")
        print(f"   請先執行: PYTHONPATH=src python scripts/run_live.py -c {args.config} --paper --once")
        return

    # 獲取當前價格
    symbols = list(state.get("positions", {}).keys()) or cfg.market.symbols
    prices = get_current_prices(symbols, cfg.market.interval)

    if not prices:
        print("❌ 無法獲取任何幣種的當前價格")
        return

    # 計算統計
    stats = calculate_stats(state, prices)
    report = format_report(stats, strategy_name)

    # 輸出
    print(report.replace("<b>", "").replace("</b>", ""))

    if not args.print_only:
        notifier = TelegramNotifier()
        ok = notifier.send(report)
        if ok:
            print("\n✅ 報表已發送到 Telegram")
        else:
            print("\n⚠️  Telegram 發送失敗（請檢查 .env 中的 Token 和 Chat ID）")


if __name__ == "__main__":
    main()
