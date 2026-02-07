"""
即时交易启动脚本

使用方法:
    # Paper Trading（默认，不需要 API Key）
    python scripts/run_live.py -c config/rsi_adx_atr.yaml --paper

    # Paper Trading - 只交易 BTCUSDT
    python scripts/run_live.py -c config/rsi_adx_atr.yaml --paper --symbol BTCUSDT

    # Paper Trading - 立即执行一次（不等待 K 线收盘）
    python scripts/run_live.py -c config/rsi_adx_atr.yaml --paper --once

    # Real Trading（需要 BINANCE_API_KEY + BINANCE_API_SECRET）
    python scripts/run_live.py -c config/rsi_adx_atr.yaml --real

    # 查看 Paper Trading 账户状态
    python scripts/run_live.py -c config/rsi_adx_atr.yaml --status

Telegram 通知:
    在 .env 中设置以下变量即可自动启用:
        TELEGRAM_BOT_TOKEN=123456:ABC-DEF
        TELEGRAM_CHAT_ID=987654321
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from qtrade.config import load_config
from qtrade.live.paper_broker import PaperBroker
from qtrade.live.runner import LiveRunner
from qtrade.live.signal_generator import generate_signal
from qtrade.monitor.notifier import TelegramNotifier


def cmd_run(args, cfg) -> None:
    """运行即时交易"""
    strategy_name = args.strategy or cfg.strategy.name
    symbols = [args.symbol] if args.symbol else cfg.market.symbols

    # 覆盖 config 中的 symbols
    if args.symbol:
        cfg = cfg.__class__(
            market=cfg.market.__class__(
                symbols=symbols,
                interval=cfg.market.interval,
                start=cfg.market.start,
                end=cfg.market.end,
            ),
            backtest=cfg.backtest,
            strategy=cfg.strategy,
            output=cfg.output,
            portfolio=cfg.portfolio,
            data_dir=cfg.data_dir,
        )

    if args.real:
        mode = "real"
        print("⚠️  真实交易模式尚未完全启用")
        print("   请先用 --paper 模式验证策略表现")
        print("   确认后再取消此限制")
        return
    else:
        mode = "paper"

    # 初始化 Paper Broker
    state_dir = Path(cfg.output.report_dir) / "live" / strategy_name
    state_dir.mkdir(parents=True, exist_ok=True)

    broker = PaperBroker(
        initial_cash=cfg.backtest.initial_cash,
        fee_bps=cfg.backtest.fee_bps,
        slippage_bps=cfg.backtest.slippage_bps,
        state_path=state_dir / "paper_state.json",
    )

    # 初始化 Telegram 通知
    notifier = TelegramNotifier()

    runner = LiveRunner(cfg=cfg, broker=broker, mode=mode, notifier=notifier)

    if args.once:
        # 立即执行一次
        signals = runner.run_once()
        print(f"\n{'─'*50}")
        for sig in signals:
            ind = sig["indicators"]
            print(f"  {sig['symbol']}: signal={sig['signal']:.0%}, "
                  f"price={sig['price']:.2f}, "
                  f"RSI={ind.get('rsi', '?')}, ADX={ind.get('adx', '?')}")

        # 打印账户状态
        prices = {s["symbol"]: s["price"] for s in signals if s["price"] > 0}
        print(f"\n{broker.summary(prices)}")
    else:
        # 持续运行
        runner.run(max_ticks=args.max_ticks)


def cmd_status(args, cfg) -> None:
    """查看 Paper Trading 账户状态"""
    strategy_name = args.strategy or cfg.strategy.name
    state_path = Path(cfg.output.report_dir) / "live" / strategy_name / "paper_state.json"

    if not state_path.exists():
        print(f"❌ 找不到状态文件: {state_path}")
        print(f"   请先运行: python scripts/run_live.py -c {args.config} --paper --once")
        return

    with open(state_path) as f:
        state = json.load(f)

    print("=" * 50)
    print(f"  Paper Trading 状态 [{strategy_name}]")
    print("=" * 50)
    print(f"  初始资金:  ${state['initial_cash']:,.2f}")
    print(f"  当前现金:  ${state['cash']:,.2f}")
    print(f"  持仓:")
    for sym, pos in state.get("positions", {}).items():
        print(f"    {sym}: {pos['qty']:.6f} @ {pos['avg_entry']:.2f}")
    print(f"  交易笔数:  {len(state.get('trades', []))}")

    # 最近 5 笔交易
    trades = state.get("trades", [])
    if trades:
        print(f"\n  最近交易:")
        for t in trades[-5:]:
            from datetime import datetime, timezone
            ts = datetime.fromtimestamp(t["timestamp"], tz=timezone.utc).strftime("%m-%d %H:%M")
            pnl_str = f" PnL={t['pnl']:+.2f}" if t.get("pnl") is not None else ""
            print(f"    [{ts}] {t['side']:4s} {t['symbol']} "
                  f"{t['qty']:.6f} @ {t['price']:.2f}{pnl_str}")
    print("=" * 50)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="即时交易",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-c", "--config", type=str, default="config/rsi_adx_atr.yaml",
                        help="配置文件路径")
    parser.add_argument("-s", "--strategy", type=str, default=None,
                        help="策略名称")
    parser.add_argument("--symbol", type=str, default=None,
                        help="只交易指定交易对")

    # 模式选择
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--paper", action="store_true", default=True,
                            help="Paper Trading 模式（默认）")
    mode_group.add_argument("--real", action="store_true",
                            help="真实交易模式（需要 API Key）")
    mode_group.add_argument("--status", action="store_true",
                            help="查看 Paper Trading 账户状态")

    # 运行选项
    parser.add_argument("--once", action="store_true",
                        help="只执行一次（不等待 K 线收盘）")
    parser.add_argument("--max-ticks", type=int, default=None,
                        help="最大运行次数")

    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.status:
        cmd_status(args, cfg)
    else:
        cmd_run(args, cfg)


if __name__ == "__main__":
    main()
