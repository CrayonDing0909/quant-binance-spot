"""
即時交易啟動腳本

使用方法:
    # Paper Trading（預設，不需要 API Key）
    python scripts/run_live.py -c config/rsi_adx_atr.yaml --paper

    # Paper Trading - 只交易 BTCUSDT
    python scripts/run_live.py -c config/rsi_adx_atr.yaml --paper --symbol BTCUSDT

    # Paper Trading - 立即執行一次（不等待 K 線收盤）
    python scripts/run_live.py -c config/rsi_adx_atr.yaml --paper --once

    # Real Trading — dry-run 模式（不下單，只看信號和模擬結果）
    python scripts/run_live.py -c config/rsi_adx_atr.yaml --real --dry-run --once

    # Real Trading（需要 BINANCE_API_KEY + BINANCE_API_SECRET）
    python scripts/run_live.py -c config/rsi_adx_atr.yaml --real --once

    # 檢查 Binance API 連線
    python scripts/run_live.py -c config/rsi_adx_atr.yaml --check

    # 查看 Paper Trading 帳戶狀態
    python scripts/run_live.py -c config/rsi_adx_atr.yaml --status

Telegram 通知:
    在 .env 中設定以下變數即可自動啟用:
        TELEGRAM_BOT_TOKEN=123456:ABC-DEF
        TELEGRAM_CHAT_ID=987654321
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

from qtrade.config import load_config
from qtrade.live.paper_broker import PaperBroker
from qtrade.live.runner import LiveRunner
from qtrade.live.signal_generator import generate_signal
from qtrade.monitor.notifier import TelegramNotifier


# ── Heartbeat（心跳監控）──────────────────────
# 每 HEARTBEAT_INTERVAL_HOURS 小時發送一次 Telegram 心跳
# 用於確認 cron / VM 仍在正常運行
HEARTBEAT_INTERVAL_HOURS = 6
HEARTBEAT_FILE = Path.home() / ".trading_heartbeat"


def _maybe_send_heartbeat(notifier: TelegramNotifier, mode: str) -> None:
    """如果距離上次心跳已超過 N 小時，發送一次心跳通知"""
    if not notifier.enabled:
        return

    now = time.time()
    last_beat = 0.0

    if HEARTBEAT_FILE.exists():
        try:
            last_beat = float(HEARTBEAT_FILE.read_text().strip())
        except (ValueError, OSError):
            last_beat = 0.0

    elapsed_hours = (now - last_beat) / 3600
    if elapsed_hours >= HEARTBEAT_INTERVAL_HOURS:
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        notifier.send(
            f"💚 <b>心跳正常</b> [{mode.upper()}]\n"
            f"  🕐 {ts}\n"
            f"  ✅ Cron 執行正常，Bot 運行中"
        )
        try:
            HEARTBEAT_FILE.write_text(str(now))
        except OSError:
            pass


def cmd_run(args, cfg) -> None:
    """運行即時交易"""
    strategy_name = args.strategy or cfg.strategy.name
    symbols = [args.symbol] if args.symbol else cfg.market.symbols
    market_type = cfg.market.market_type.value  # "spot" or "futures"

    # 覆蓋 config 中的 symbols
    if args.symbol:
        cfg = cfg.__class__(
            market=cfg.market.__class__(
                symbols=symbols,
                interval=cfg.market.interval,
                start=cfg.market.start,
                end=cfg.market.end,
                market_type=cfg.market.market_type,
            ),
            backtest=cfg.backtest,
            strategy=cfg.strategy,
            output=cfg.output,
            portfolio=cfg.portfolio,
            data_dir=cfg.data_dir,
            futures=cfg.futures,
            notification=cfg.notification,
        )

    # 初始化 Telegram 通知（從配置或環境變數）
    notifier = TelegramNotifier.from_config(cfg.notification)
    
    # 市場類型標籤
    market_emoji = "🟢" if market_type == "spot" else "🔴"
    market_label = "SPOT" if market_type == "spot" else "FUTURES"
    leverage = cfg.futures.leverage if cfg.futures else 1

    if args.real:
        # ── Real Trading 模式 ──
        mode = "real"
        dry_run = getattr(args, "dry_run", False)

        from qtrade.live.binance_spot_broker import BinanceSpotBroker

        broker = BinanceSpotBroker(dry_run=dry_run)
        runner = LiveRunner(cfg=cfg, broker=broker, mode=mode, notifier=notifier)

        if dry_run:
            print("🧪 DRY-RUN 模式：所有下單指令只會記錄，不會真的執行")
            print()

        if args.once:
            signals = runner.run_once()
            print(f"\n{'─'*50}")
            for sig in signals:
                ind = sig["indicators"]
                print(f"  {sig['symbol']}: signal={sig['signal']:.0%}, "
                      f"price={sig['price']:.2f}, "
                      f"RSI={ind.get('rsi', '?')}, ADX={ind.get('adx', '?')}")

            # 列印帳戶餘額
            print(f"\n{'='*50}")
            print(f"  Real Trading 帳戶 {'[DRY-RUN]' if dry_run else ''}")
            print(f"{'='*50}")
            usdt = broker.get_balance("USDT")
            print(f"  USDT 餘額: ${usdt:,.2f}")
            for sym in symbols:
                qty = broker.get_position(sym)
                price = broker.get_price(sym)
                if qty > 0:
                    print(f"  {sym}: {qty:.6f} ≈ ${qty * price:,.2f}")
            equity = broker.get_equity(symbols)
            print(f"  總權益: ${equity:,.2f}")
            print(f"{'='*50}")

            # 心跳監控
            _maybe_send_heartbeat(notifier, mode)
        else:
            if not dry_run:
                print("⚠️  即將以真實交易模式持續運行！")
                print("    按 Ctrl+C 可隨時停止")
                print()
            runner.run(max_ticks=args.max_ticks)
    else:
        # ── Paper Trading 模式 ──
        mode = "paper"
        
        print(f"{market_emoji} Paper Trading [{market_label}]")
        if market_type == "futures":
            print(f"   槓桿: {leverage}x")
        print()

        state_dir = Path(cfg.output.report_dir) / "live" / strategy_name
        state_dir.mkdir(parents=True, exist_ok=True)

        broker = PaperBroker(
            initial_cash=cfg.backtest.initial_cash,
            fee_bps=cfg.backtest.fee_bps,
            slippage_bps=cfg.backtest.slippage_bps,
            state_path=state_dir / "paper_state.json",
            market_type=market_type,
            leverage=leverage,
        )

        runner = LiveRunner(cfg=cfg, broker=broker, mode=mode, notifier=notifier)

        if args.once:
            signals = runner.run_once()
            print(f"\n{'─'*50}")
            for sig in signals:
                ind = sig["indicators"]
                signal_val = sig['signal']
                # 支援做空信號顯示
                if signal_val > 0.5:
                    signal_str = f"LONG {signal_val:.0%}"
                elif signal_val < -0.5:
                    signal_str = f"SHORT {abs(signal_val):.0%}"
                else:
                    signal_str = f"FLAT {signal_val:.0%}"
                print(f"  {sig['symbol']}: {signal_str}, "
                      f"price={sig['price']:.2f}, "
                      f"RSI={ind.get('rsi', '?')}, ADX={ind.get('adx', '?')}")

            # 列印帳戶狀態
            prices = {s["symbol"]: s["price"] for s in signals if s["price"] > 0}
            print(f"\n{broker.summary(prices)}")

            # 心跳監控
            _maybe_send_heartbeat(notifier, mode)
        else:
            runner.run(max_ticks=args.max_ticks)


def cmd_check(args, cfg) -> None:
    """檢查 Binance API 連線"""
    from qtrade.live.binance_spot_broker import BinanceSpotBroker

    print("=" * 50)
    print("  🔍 Binance API 連線檢查")
    print("=" * 50)

    try:
        broker = BinanceSpotBroker(dry_run=True)
    except RuntimeError as e:
        print(f"\n{e}")
        return

    result = broker.check_connection(symbols=cfg.market.symbols)

    print()
    if "server_time" in result:
        print(f"  ✅ 伺服器時間: {result['server_time']}")
    else:
        print(f"  ❌ 伺服器連線失敗: {result.get('server_time_error', '未知錯誤')}")

    if "account_error" in result:
        print(f"  ❌ 帳戶連線失敗: {result['account_error']}")
    else:
        print(f"  ✅ 帳戶類型: {result.get('account_type', '?')}")
        print(f"  ✅ 可交易: {result.get('can_trade', '?')}")
        print(f"  💰 USDT 餘額: ${result.get('usdt_balance', 0):,.2f}")

        balances = result.get("balances", {})
        for asset, val in balances.items():
            if asset != "USDT" and val["free"] > 0:
                print(f"  💰 {asset}: {val['free']}")

    prices = result.get("prices", {})
    if prices:
        print()
        for sym, price in prices.items():
            print(f"  📊 {sym}: ${price:,.2f}")

    filters = result.get("filters", {})
    if filters:
        print()
        print("  📋 交易規則:")
        for sym, f in filters.items():
            print(f"    {sym}: minQty={f['min_qty']}, "
                  f"stepSize={f['step_size']}, "
                  f"minNotional=${f['min_notional']}")

    print()
    print("=" * 50)
    print("  ✅ 連線檢查完成")
    print()
    print("  下一步:")
    print("    # dry-run 測試（不下單）")
    print(f"    python scripts/run_live.py -c {args.config} --real --dry-run --once")
    print()
    print("    # 真實交易（真金白銀！）")
    print(f"    python scripts/run_live.py -c {args.config} --real --once")
    print("=" * 50)


def cmd_status(args, cfg) -> None:
    """查看 Paper Trading 帳戶狀態"""
    strategy_name = args.strategy or cfg.strategy.name
    state_path = Path(cfg.output.report_dir) / "live" / strategy_name / "paper_state.json"

    if not state_path.exists():
        print(f"❌ 找不到狀態檔案: {state_path}")
        print(f"   請先運行: python scripts/run_live.py -c {args.config} --paper --once")
        return

    with open(state_path) as f:
        state = json.load(f)

    # 市場類型
    market_type = state.get("market_type", "spot")
    leverage = state.get("leverage", 1)
    market_emoji = "🟢" if market_type == "spot" else "🔴"
    market_label = "SPOT" if market_type == "spot" else f"FUTURES ({leverage}x)"

    print("=" * 50)
    print(f"  Paper Trading 狀態 {market_emoji} [{market_label}]")
    print(f"  策略: {strategy_name}")
    print("=" * 50)
    print(f"  初始資金:  ${state['initial_cash']:,.2f}")
    print(f"  當前現金:  ${state['cash']:,.2f}")
    print(f"  持倉:")
    for sym, pos in state.get("positions", {}).items():
        qty = pos['qty']
        # 支援做空顯示
        if qty > 0:
            side_label = "LONG"
        elif qty < 0:
            side_label = "SHORT"
            qty = abs(qty)
        else:
            continue
        print(f"    {sym} [{side_label}]: {qty:.6f} @ {pos['avg_entry']:.2f}")
    print(f"  交易筆數:  {len(state.get('trades', []))}")

    # 最近 5 筆交易
    trades = state.get("trades", [])
    if trades:
        print(f"\n  最近交易:")
        for t in trades[-5:]:
            from datetime import datetime, timezone
            ts = datetime.fromtimestamp(t["timestamp"], tz=timezone.utc).strftime("%m-%d %H:%M")
            pnl_str = f" PnL={t['pnl']:+.2f}" if t.get("pnl") is not None else ""
            print(f"    [{ts}] {t['side']:12s} {t['symbol']} "
                  f"{t['qty']:.6f} @ {t['price']:.2f}{pnl_str}")
    print("=" * 50)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="即時交易",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-c", "--config", type=str, default="config/rsi_adx_atr.yaml",
                        help="配置檔路徑")
    parser.add_argument("-s", "--strategy", type=str, default=None,
                        help="策略名稱")
    parser.add_argument("--symbol", type=str, default=None,
                        help="只交易指定交易對")

    # 模式選擇
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--paper", action="store_true", default=True,
                            help="Paper Trading 模式（預設）")
    mode_group.add_argument("--real", action="store_true",
                            help="真實交易模式（需要 API Key）")
    mode_group.add_argument("--status", action="store_true",
                            help="查看 Paper Trading 帳戶狀態")
    mode_group.add_argument("--check", action="store_true",
                            help="檢查 Binance API 連線")

    # 運行選項
    parser.add_argument("--once", action="store_true",
                        help="只執行一次（不等待 K 線收盤）")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run",
                        help="Real 模式下不實際下單（測試用）")
    parser.add_argument("--max-ticks", type=int, default=None,
                        help="最大運行次數")

    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.status:
        cmd_status(args, cfg)
    elif args.check:
        cmd_check(args, cfg)
    else:
        cmd_run(args, cfg)


if __name__ == "__main__":
    main()
