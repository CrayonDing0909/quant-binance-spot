#!/usr/bin/env python3
"""
交易資料庫查詢工具

使用方式:
    # 查看績效總覽
    python scripts/query_db.py -c config/futures_rsi_adx_atr.yaml summary

    # 查看最近交易
    python scripts/query_db.py -c config/futures_rsi_adx_atr.yaml trades --limit 20

    # 查看特定幣種最近 7 天交易
    python scripts/query_db.py -c config/futures_rsi_adx_atr.yaml trades --symbol BTCUSDT --days 7

    # 查看信號記錄
    python scripts/query_db.py -c config/futures_rsi_adx_atr.yaml signals --limit 10

    # 查看每日權益曲線
    python scripts/query_db.py -c config/futures_rsi_adx_atr.yaml equity --days 30

    # 匯出交易到 CSV
    python scripts/query_db.py -c config/futures_rsi_adx_atr.yaml export --output trades.csv

    # 清理舊資料（保留最近 365 天）
    python scripts/query_db.py -c config/futures_rsi_adx_atr.yaml compact --keep-days 365
"""
import sys
import argparse
from pathlib import Path

# 確保 src 在 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qtrade.config import load_config
from qtrade.live.trading_db import TradingDatabase


def cmd_summary(db: TradingDatabase, args):
    """績效總覽"""
    stats = db.get_performance_summary(days=args.days)

    print("=" * 60)
    print("  📊 交易績效總覽")
    if args.days:
        print(f"  （最近 {args.days} 天）")
    print("=" * 60)
    print(f"  總交易筆數:     {stats['total_trades']}")
    print(f"  勝利:           {stats['winning_trades']}")
    print(f"  虧損:           {stats['losing_trades']}")
    print(f"  勝率:           {stats['win_rate']:.1%}")
    print(f"  總 PnL:         ${stats['total_pnl']:+,.2f}")
    print(f"  平均 PnL:       ${stats['avg_pnl']:+,.2f}")
    print(f"  最佳交易:       ${stats['best_trade']:+,.2f}")
    print(f"  最差交易:       ${stats['worst_trade']:+,.2f}")
    print(f"  總手續費:       ${stats['total_fees']:,.2f}")
    print("-" * 60)
    print(f"  Maker 成交比例: {stats['maker_pct']:.1%}")
    print(f"  Maker 省下費用: ${stats['total_fee_savings']:,.2f}")
    print("=" * 60)


def cmd_trades(db: TradingDatabase, args):
    """交易記錄"""
    trades = db.get_trades(
        symbol=args.symbol,
        days=args.days,
        limit=args.limit,
    )

    if not trades:
        print("📭 沒有交易記錄")
        return

    print(f"\n📝 最近 {len(trades)} 筆交易:")
    print("-" * 100)
    print(f"{'時間':<20} {'幣種':<10} {'方向':<6} {'數量':>10} {'價格':>12} "
          f"{'PnL':>10} {'類型':<12} {'原因'}")
    print("-" * 100)

    for t in trades:
        ts = t["timestamp"][:19] if t["timestamp"] else ""
        pnl_str = f"${t['pnl']:+,.2f}" if t["pnl"] is not None else "  開倉"
        print(
            f"{ts:<20} {t['symbol']:<10} {t['side']:<6} "
            f"{t['qty']:>10.6f} ${t['price']:>10,.2f} "
            f"{pnl_str:>10} {t['order_type']:<12} {t['reason']}"
        )

    print("-" * 100)


def cmd_signals(db: TradingDatabase, args):
    """信號記錄"""
    signals = db.get_signals(
        symbol=args.symbol,
        days=args.days,
        limit=args.limit,
    )

    if not signals:
        print("📭 沒有信號記錄")
        return

    print(f"\n📡 最近 {len(signals)} 筆信號:")
    print("-" * 110)
    print(f"{'時間':<20} {'幣種':<10} {'信號':>6} {'價格':>12} "
          f"{'RSI':>6} {'ADX':>6} {'ATR':>8} {'動作':<12} {'目標%':>6} {'現在%':>6}")
    print("-" * 110)

    for s in signals:
        ts = s["timestamp"][:19] if s["timestamp"] else ""
        rsi = f"{s['rsi']:.1f}" if s["rsi"] is not None else "  -"
        adx = f"{s['adx']:.1f}" if s["adx"] is not None else "  -"
        atr = f"{s['atr']:.1f}" if s["atr"] is not None else "     -"
        tgt = f"{s['target_pct']:.0%}" if s["target_pct"] is not None else "  -"
        cur = f"{s['current_pct']:.0%}" if s["current_pct"] is not None else "  -"

        print(
            f"{ts:<20} {s['symbol']:<10} {s['signal_value']:>+5.0%} "
            f"${s['price']:>10,.2f} {rsi:>6} {adx:>6} {atr:>8} "
            f"{s['action']:<12} {tgt:>6} {cur:>6}"
        )

    print("-" * 110)


def cmd_equity(db: TradingDatabase, args):
    """每日權益"""
    equity = db.get_daily_equity(days=args.days or 30)

    if not equity:
        print("📭 沒有權益記錄")
        return

    print(f"\n💰 每日權益（最近 {args.days or 30} 天）:")
    print("-" * 80)
    print(f"{'日期':<12} {'權益':>12} {'現金':>12} {'日PnL':>10} "
          f"{'交易數':>6} {'持倉數':>6}")
    print("-" * 80)

    for e in equity:
        print(
            f"{e['date']:<12} ${e['equity']:>10,.2f} ${e['cash']:>10,.2f} "
            f"${e['pnl_day']:>+8,.2f} {e['trade_count']:>6} {e['position_count']:>6}"
        )

    print("-" * 80)


def cmd_export(db: TradingDatabase, args):
    """匯出 CSV"""
    output = args.output or "trades_export.csv"
    count = db.export_trades_csv(output)
    print(f"✅ 匯出 {count} 筆交易到 {output}")


def cmd_compact(db: TradingDatabase, args):
    """清理舊資料"""
    keep = args.keep_days or 365
    deleted = db.compact(keep_days=keep)
    print(f"🧹 清理完成，刪除了 {deleted} 筆舊信號記錄（保留最近 {keep} 天）")


def main():
    parser = argparse.ArgumentParser(description="交易資料庫查詢工具")
    parser.add_argument("-c", "--config", required=True, help="配置檔案路徑")
    parser.add_argument("--db", help="直接指定 DB 檔案路徑（覆蓋 config）")

    sub = parser.add_subparsers(dest="command", help="子命令")

    # summary
    p_sum = sub.add_parser("summary", help="績效總覽")
    p_sum.add_argument("--days", type=int, help="最近 N 天")

    # trades
    p_trades = sub.add_parser("trades", help="交易記錄")
    p_trades.add_argument("--symbol", help="過濾幣種")
    p_trades.add_argument("--days", type=int, help="最近 N 天")
    p_trades.add_argument("--limit", type=int, default=50, help="最大筆數")

    # signals
    p_sig = sub.add_parser("signals", help="信號記錄")
    p_sig.add_argument("--symbol", help="過濾幣種")
    p_sig.add_argument("--days", type=int, help="最近 N 天")
    p_sig.add_argument("--limit", type=int, default=20, help="最大筆數")

    # equity
    p_eq = sub.add_parser("equity", help="每日權益")
    p_eq.add_argument("--days", type=int, default=30, help="最近 N 天")

    # export
    p_exp = sub.add_parser("export", help="匯出交易到 CSV")
    p_exp.add_argument("--output", "-o", help="輸出檔案路徑")

    # compact
    p_compact = sub.add_parser("compact", help="清理舊資料")
    p_compact.add_argument("--keep-days", type=int, default=365, help="保留最近 N 天")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # 載入 DB
    if args.db:
        db_path = Path(args.db)
    else:
        cfg = load_config(args.config)
        db_path = cfg.get_report_dir("live") / "trading.db"

    if not db_path.exists():
        print(f"❌ 資料庫不存在: {db_path}")
        print("   資料庫會在首次實盤交易時自動建立。")
        return

    db = TradingDatabase(db_path)

    commands = {
        "summary": cmd_summary,
        "trades": cmd_trades,
        "signals": cmd_signals,
        "equity": cmd_equity,
        "export": cmd_export,
        "compact": cmd_compact,
    }
    commands[args.command](db, args)


if __name__ == "__main__":
    main()
