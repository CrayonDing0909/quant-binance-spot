"""
Alpha Decay 監控腳本

追蹤策略信號的 Information Coefficient (IC)，偵測 Alpha 衰退。
支援 Telegram 通知 + JSON 報告輸出。

使用方式:
    # 基本用法
    python scripts/monitor_alpha_decay.py -c config/futures_rsi_adx_atr.yaml

    # 指定幣對
    python scripts/monitor_alpha_decay.py -c config/futures_rsi_adx_atr.yaml --symbol BTCUSDT

    # 啟用 Telegram 通知（只發警報）
    python scripts/monitor_alpha_decay.py -c config/futures_rsi_adx_atr.yaml --notify

    # 調整前瞻期和窗口
    python scripts/monitor_alpha_decay.py -c config/futures_rsi_adx_atr.yaml --forward-bars 48 --window-days 90

    # 輸出 JSON 報告
    python scripts/monitor_alpha_decay.py -c config/futures_rsi_adx_atr.yaml --output-dir reports/alpha_decay

    # Cron 模式（靜默輸出，只在有警報時通知）
    python scripts/monitor_alpha_decay.py -c config/futures_rsi_adx_atr.yaml --notify --quiet
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qtrade.config import load_config
from qtrade.data.storage import load_klines
from qtrade.strategy.base import StrategyContext
from qtrade.strategy import get_strategy
from qtrade.validation.ic_monitor import RollingICMonitor


def _build_telegram_message(
    strategy_name: str,
    market_type: str,
    symbol_reports: list[dict],
) -> str:
    """組裝 Telegram 通知訊息（HTML 格式）"""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"📊 <b>Alpha Decay Monitor</b>",
        f"🕐 {now}",
        f"策略: <code>{strategy_name}</code> | 市場: {market_type}",
        "",
    ]

    has_alerts = False
    for sr in symbol_reports:
        sym = sr["symbol"]
        report = sr["report"]
        alerts = sr["alerts"]

        emoji = "🔴" if report.is_decaying else "🟢"
        lines.append(f"<b>{sym}</b> {emoji}")
        lines.append(
            f"  IC: {report.overall_ic:+.4f} (p={report.overall_ic_pvalue:.4f})"
        )
        lines.append(f"  Recent: {report.recent_ic:+.4f} | Hist: {report.historical_ic:+.4f}")
        lines.append(f"  Decay: {report.ic_decay_pct:+.0%} | IR: {report.ic_ir:.2f}")

        if alerts:
            has_alerts = True
            for a in alerts:
                tag = {"critical": "🚨", "warning": "⚠️", "info": "ℹ️"}.get(
                    a.severity, "❓"
                )
                lines.append(f"  {tag} {a.message}")
        else:
            lines.append("  ✅ 信號品質正常")
        lines.append("")

    return "\n".join(lines), has_alerts


def _save_json_report(
    output_dir: Path,
    strategy_name: str,
    symbol_reports: list[dict],
) -> Path:
    """儲存 JSON 報告"""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"ic_report_{timestamp}.json"

    data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "strategy": strategy_name,
        "symbols": {},
    }

    for sr in symbol_reports:
        sym = sr["symbol"]
        report = sr["report"]
        alerts = sr["alerts"]
        data["symbols"][sym] = {
            "overall_ic": round(report.overall_ic, 6),
            "overall_ic_pvalue": round(report.overall_ic_pvalue, 6),
            "avg_ic": round(report.avg_ic, 6),
            "ic_std": round(report.ic_std, 6),
            "ic_ir": round(report.ic_ir, 4),
            "recent_ic": round(report.recent_ic, 6),
            "historical_ic": round(report.historical_ic, 6),
            "ic_decay_pct": round(report.ic_decay_pct, 4),
            "is_decaying": report.is_decaying,
            "yearly_ic": report.yearly_ic,
            "signal_count": report.signal_count,
            "active_signal_pct": round(report.active_signal_pct, 4),
            "alerts": [
                {"severity": a.severity, "message": a.message}
                for a in alerts
            ],
        }

    def _json_default(obj):
        """處理 numpy 類型的 JSON 序列化"""
        import numpy as np
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_json_default)

    return report_path


def main():
    parser = argparse.ArgumentParser(
        description="Alpha Decay 監控",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-c", "--config", type=str, default="config/futures_rsi_adx_atr.yaml"
    )
    parser.add_argument("--symbol", type=str, default=None)
    parser.add_argument(
        "--forward-bars", type=int, default=24, help="前瞻期（bar 數）"
    )
    parser.add_argument(
        "--window-days", type=int, default=180, help="Rolling IC 窗口天數"
    )
    parser.add_argument(
        "--notify",
        action="store_true",
        help="啟用 Telegram 通知（使用 config 中的 notification 設定）",
    )
    parser.add_argument(
        "--notify-always",
        action="store_true",
        help="無論有無警報都發送通知（預設只有警報時通知）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="JSON 報告輸出目錄（不指定則不輸出 JSON）",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="靜默模式（不輸出到 stdout，適合 cron）",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    market_type = cfg.market_type_str
    symbols = [args.symbol] if args.symbol else cfg.market.symbols

    strategy_name = cfg.strategy.name
    strategy_func = get_strategy(strategy_name)

    bars_per_day = {
        "1m": 1440, "5m": 288, "15m": 96,
        "1h": 24, "4h": 6, "1d": 1,
    }
    bpd = bars_per_day.get(cfg.market.interval, 24)
    window = args.window_days * bpd

    monitor = RollingICMonitor(
        window=window,
        forward_bars=args.forward_bars,
        interval=cfg.market.interval,
    )

    if not args.quiet:
        print(f"📊 Alpha Decay Monitor")
        print(f"   策略: {strategy_name}")
        print(f"   市場: {market_type}")
        print(f"   前瞻期: {args.forward_bars} bars ({args.forward_bars / bpd:.1f} 天)")
        print(f"   IC 窗口: {args.window_days} 天 ({window} bars)")
        print()

    symbol_reports = []

    for sym in symbols:
        data_path = (
            cfg.data_dir
            / "binance"
            / market_type
            / cfg.market.interval
            / f"{sym}.parquet"
        )
        if not data_path.exists():
            if not args.quiet:
                print(f"⚠️  {sym}: 數據不存在，跳過")
            continue

        df = load_klines(data_path)
        params = cfg.strategy.get_params(sym)

        ctx = StrategyContext(
            symbol=sym,
            interval=cfg.market.interval,
            market_type=market_type,
            direction=cfg.direction,
        )

        # 計算策略信號
        signals = strategy_func(df, ctx, params)

        # 計算 IC
        report = monitor.compute(signals, df["close"])
        alerts = monitor.check_alerts(report)

        symbol_reports.append({
            "symbol": sym,
            "report": report,
            "alerts": alerts,
        })

        # 輸出報告到 stdout
        if not args.quiet:
            print(f"{'═' * 60}")
            print(f"  {sym}  IC Analysis")
            print(f"{'═' * 60}")
            print(
                f"  📅 數據範圍: {df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}"
            )
            print(
                f"  📊 有效信號: {report.signal_count:,} 筆 ({report.active_signal_pct:.1%} 活躍)"
            )
            print()
            print(f"  ── 全局 IC ──")
            print(
                f"  Overall IC:     {report.overall_ic:+.4f}  (p={report.overall_ic_pvalue:.4f})"
            )
            print(f"  Average IC:     {report.avg_ic:+.4f}")
            print(f"  IC Std:         {report.ic_std:.4f}")
            print(f"  IC IR:          {report.ic_ir:.3f}")
            print()
            print(f"  ── Alpha Decay 偵測 ──")
            print(f"  Historical IC:  {report.historical_ic:+.4f}")
            print(f"  Recent IC:      {report.recent_ic:+.4f}")
            decay_emoji = "🔴" if report.is_decaying else "🟢"
            print(f"  IC 衰退:        {report.ic_decay_pct:+.0%}  {decay_emoji}")
            print()
            print(f"  ── 年度 IC ──")
            for year, ic in sorted(report.yearly_ic.items()):
                bar = "█" * max(1, int(abs(ic) * 200))
                sign = "+" if ic > 0 else ""
                emoji = "🟢" if ic > 0.03 else "🟡" if ic > 0 else "🔴"
                print(f"  {year}: {sign}{ic:.4f}  {emoji} {bar}")
            print()

            # 警報
            if alerts:
                print(f"  ── 警報 ({len(alerts)}) ──")
                for alert in alerts:
                    emoji = {"critical": "🚨", "warning": "⚠️", "info": "ℹ️"}.get(
                        alert.severity, "❓"
                    )
                    print(f"  {emoji} [{alert.severity.upper()}] {alert.message}")
                print()
            else:
                print(f"  ✅ 無警報，信號品質正常")
                print()

    # ── JSON 報告 ──
    if args.output_dir:
        report_path = _save_json_report(
            Path(args.output_dir), strategy_name, symbol_reports
        )
        if not args.quiet:
            print(f"📁 JSON 報告已儲存: {report_path}")

    # ── Telegram 通知 ──
    if args.notify:
        try:
            from qtrade.monitor.notifier import TelegramNotifier

            notifier = TelegramNotifier.from_config(cfg.notification)
            message, has_alerts = _build_telegram_message(
                strategy_name, market_type, symbol_reports
            )

            # 預設只在有警報時發送，--notify-always 時總是發送
            if has_alerts or args.notify_always:
                success = notifier.send(message, parse_mode="HTML")
                if not args.quiet:
                    if success:
                        print("📨 Telegram 通知已發送")
                    else:
                        print("⚠️  Telegram 通知發送失敗（檢查 token/chat_id）")
            else:
                if not args.quiet:
                    print("📭 無警報，跳過 Telegram 通知（使用 --notify-always 強制發送）")
        except Exception as e:
            if not args.quiet:
                print(f"⚠️  Telegram 通知錯誤: {e}")


if __name__ == "__main__":
    main()
