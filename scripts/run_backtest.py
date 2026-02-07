"""
回测脚本

支持命令行参数和配置文件两种方式。

使用方法:
    # 使用配置文件（默认）
    python scripts/run_backtest.py

    # 指定配置文件
    python scripts/run_backtest.py -c config/rsi.yaml

    # 指定策略（覆盖配置文件中的策略）
    python scripts/run_backtest.py -s rsi

    # 指定策略和配置文件
    python scripts/run_backtest.py -c config/rsi.yaml -s rsi

    # 指定交易对（只回测指定交易对）
    python scripts/run_backtest.py --symbol BTCUSDT
"""
from __future__ import annotations
import argparse
from pathlib import Path
from qtrade.config import load_config
from qtrade.backtest.run_backtest import run_symbol_backtest
from qtrade.backtest.metrics import full_report, trade_summary, trade_analysis
from qtrade.backtest.plotting import plot_backtest_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="运行策略回测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="config/base.yaml",
        help="配置文件路径（默认: config/base.yaml）"
    )
    parser.add_argument(
        "-s", "--strategy",
        type=str,
        default=None,
        help="策略名称（覆盖配置文件中的策略）"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="指定交易对（默认使用配置文件中的所有交易对）"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录（默认: reports/{strategy_name}）"
    )

    args = parser.parse_args()

    # 加载配置
    cfg = load_config(args.config)

    # 确定使用的策略
    strategy_name = args.strategy or cfg.strategy.name
    if not strategy_name:
        print("❌ 错误: 未指定策略名称")
        print("   请在配置文件中设置 strategy.name，或使用 -s/--strategy 参数")
        return

    # 确定输出目录
    if args.output_dir:
        report_dir = Path(args.output_dir)
    else:
        base_report_dir = Path(cfg.output.report_dir)
        report_dir = base_report_dir / strategy_name

    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"📊 策略: {strategy_name}")
    print(f"📁 输出目录: {report_dir}")

    # 确定交易对
    symbols = [args.symbol] if args.symbol else cfg.market.symbols

    for sym in symbols:
        # 准备回测配置（每个币种使用合并后的参数）
        bt_cfg = {
            "initial_cash": cfg.backtest.initial_cash,
            "fee_bps": cfg.backtest.fee_bps,
            "slippage_bps": cfg.backtest.slippage_bps,
            "strategy_params": cfg.strategy.get_params(sym),
            "strategy_name": strategy_name,
            "validate_data": cfg.backtest.validate_data,
            "clean_data_before": cfg.backtest.clean_data,
            "interval": cfg.market.interval,
        }
        data_path = cfg.data_dir / "binance" / "spot" / cfg.market.interval / f"{sym}.parquet"

        if not data_path.exists():
            print(f"⚠️  数据文件不存在: {data_path}")
            print(f"   请先运行: python scripts/download_data.py --symbol {sym}")
            continue

        print(f"\n{'='*60}")
        print(f"回测: {strategy_name} - {sym}")
        print(f"{'='*60}")

        res = run_symbol_backtest(sym, data_path, bt_cfg, strategy_name)
        pf = res["pf"]
        pf_bh = res["pf_bh"]

        # ── 1. 策略 vs Buy & Hold 对比报告 ──────────────
        report = full_report(pf, pf_bh, strategy_name)
        print(f"\n{'─'*50}")
        print(f"  {sym}  策略 vs Buy & Hold")
        print(f"{'─'*50}")
        print(report.to_string())

        stats_path = report_dir / f"stats_{sym}.csv"
        report.to_csv(stats_path)
        print(f"\n✅ 统计报告: {stats_path}")

        # ── 2. 交易摘要 ────────────────────────────────
        t_summary = trade_summary(pf)
        if not t_summary.empty:
            print(f"\n{'─'*50}")
            print(f"  交易摘要")
            print(f"{'─'*50}")
            print(t_summary.to_string())

            ts_path = report_dir / f"trade_summary_{sym}.csv"
            t_summary.to_csv(ts_path)
            print(f"\n✅ 交易摘要: {ts_path}")

        # ── 3. 逐笔交易记录 ────────────────────────────
        trades_df = trade_analysis(pf)
        if not trades_df.empty:
            trades_path = report_dir / f"trades_{sym}.csv"
            trades_df.to_csv(trades_path, index=False)
            print(f"✅ 逐笔交易: {trades_path}  ({len(trades_df)} 笔)")

        # ── 4. 资金曲线图（含 Buy & Hold）───────────────
        plot_path = report_dir / f"equity_curve_{sym}.png"
        plot_backtest_summary(
            pf, res["df"], res["pos"], sym, plot_path,
            pf_benchmark=pf_bh,
            strategy_name=strategy_name,
        )
        print(f"✅ 资金曲线图: {plot_path}")


if __name__ == "__main__":
    main()
