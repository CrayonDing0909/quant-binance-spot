"""
回測腳本

支援命令列參數和配置檔兩種方式。

使用方法:
    # 使用配置檔（預設）
    python scripts/run_backtest.py

    # 指定配置檔
    python scripts/run_backtest.py -c config/rsi.yaml

    # 指定策略（覆蓋配置檔中的策略）
    python scripts/run_backtest.py -s rsi

    # 指定策略和配置檔
    python scripts/run_backtest.py -c config/rsi.yaml -s rsi

    # 指定交易對（只回測指定交易對）
    python scripts/run_backtest.py --symbol BTCUSDT

    # 加上時間戳（不覆蓋舊報告）
    python scripts/run_backtest.py --timestamp
"""
from __future__ import annotations
import argparse
from datetime import datetime
from pathlib import Path
from qtrade.config import load_config
from qtrade.backtest.run_backtest import run_symbol_backtest
from qtrade.backtest.metrics import full_report, trade_summary, trade_analysis
from qtrade.backtest.plotting import plot_backtest_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="運行策略回測",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="config/base.yaml",
        help="配置檔路徑（預設: config/base.yaml）"
    )
    parser.add_argument(
        "-s", "--strategy",
        type=str,
        default=None,
        help="策略名稱（覆蓋配置檔中的策略）"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="指定交易對（預設使用配置檔中的所有交易對）"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="輸出目錄（預設: reports/{strategy_name}）"
    )
    parser.add_argument(
        "--timestamp", "-t",
        action="store_true",
        help="在輸出目錄加上時間戳，避免覆蓋舊報告"
    )

    args = parser.parse_args()

    # 載入配置
    cfg = load_config(args.config)

    # 確定使用的策略
    strategy_name = args.strategy or cfg.strategy.name
    if not strategy_name:
        print("❌ 錯誤: 未指定策略名稱")
        print("   請在配置檔中設定 strategy.name，或使用 -s/--strategy 參數")
        return

    # 確定輸出目錄
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        report_dir = Path(args.output_dir)
        if args.timestamp:
            report_dir = report_dir / timestamp_str
    else:
        base_report_dir = Path(cfg.output.report_dir)
        if args.timestamp:
            report_dir = base_report_dir / strategy_name / timestamp_str
        else:
            report_dir = base_report_dir / strategy_name

    report_dir.mkdir(parents=True, exist_ok=True)

    # 保存運行資訊
    run_info = {
        "timestamp": timestamp_str,
        "strategy": strategy_name,
        "config": args.config,
        "data_start": cfg.market.start,
        "data_end": cfg.market.end or "now",
        "symbols": cfg.market.symbols,
    }
    run_info_path = report_dir / "run_info.json"
    import json
    with open(run_info_path, "w") as f:
        json.dump(run_info, f, indent=2, ensure_ascii=False)

    print(f"📊 策略: {strategy_name}")
    print(f"📁 輸出目錄: {report_dir}")
    print(f"🕐 運行時間: {timestamp_str}")

    # 確定交易對
    symbols = [args.symbol] if args.symbol else cfg.market.symbols

    for sym in symbols:
        # 準備回測配置（每個幣種使用合併後的參數）
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
            print(f"⚠️  數據檔案不存在: {data_path}")
            print(f"   請先運行: python scripts/download_data.py --symbol {sym}")
            continue

        print(f"\n{'='*60}")
        print(f"回測: {strategy_name} - {sym}")
        print(f"{'='*60}")

        res = run_symbol_backtest(sym, data_path, bt_cfg, strategy_name)
        pf = res["pf"]
        pf_bh = res["pf_bh"]

        # ── 1. 策略 vs Buy & Hold 對比報告 ──────────────
        report = full_report(pf, pf_bh, strategy_name)
        print(f"\n{'─'*50}")
        print(f"  {sym}  策略 vs Buy & Hold")
        print(f"{'─'*50}")
        print(report.to_string())

        stats_path = report_dir / f"stats_{sym}.csv"
        report.to_csv(stats_path)
        print(f"\n✅ 統計報告: {stats_path}")

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

        # ── 3. 逐筆交易記錄 ────────────────────────────
        trades_df = trade_analysis(pf)
        if not trades_df.empty:
            trades_path = report_dir / f"trades_{sym}.csv"
            trades_df.to_csv(trades_path, index=False)
            print(f"✅ 逐筆交易: {trades_path}  ({len(trades_df)} 筆)")

        # ── 4. 資金曲線圖（含 Buy & Hold）───────────────
        plot_path = report_dir / f"equity_curve_{sym}.png"
        plot_backtest_summary(
            pf, res["df"], res["pos"], sym, plot_path,
            pf_benchmark=pf_bh,
            strategy_name=strategy_name,
        )
        print(f"✅ 資金曲線圖: {plot_path}")


if __name__ == "__main__":
    main()
