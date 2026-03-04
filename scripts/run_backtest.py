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

    # 預設帶時間戳，使用 --no-timestamp 可關閉
    python scripts/run_backtest.py --no-timestamp

    # 合約回測 - 指定交易方向
    python scripts/run_backtest.py -c config/futures_rsi_adx_atr.yaml --direction both
    python scripts/run_backtest.py -c config/futures_rsi_adx_atr.yaml --direction long_only
    python scripts/run_backtest.py -c config/futures_rsi_adx_atr.yaml --direction short_only
"""
from __future__ import annotations
import argparse
from datetime import datetime
from pathlib import Path
import yaml
from qtrade.config import load_config
from qtrade.backtest.run_backtest import run_symbol_backtest
from qtrade.backtest.metrics import full_report, trade_summary, trade_analysis, long_short_split_analysis
from qtrade.backtest.plotting import plot_backtest_summary
from qtrade.validation.prado_methods import deflated_sharpe_ratio
from qtrade.validation.red_flags import check_red_flags, print_red_flags


def _load_ensemble_strategy(config_path: str, symbol: str) -> tuple[str, dict] | None:
    """
    從 ensemble 配置中取得某 symbol 的策略名與參數

    Returns:
        (strategy_name, params) 或 None（無 ensemble 或該 symbol 不在 map 中）
    """
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    ens = raw.get("ensemble")
    if ens and ens.get("enabled", False):
        strategies = ens.get("strategies", {})
        if symbol in strategies:
            s = strategies[symbol]
            return s["name"], s.get("params", {})
    return None


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
        default=True,
        help="在輸出目錄加上時間戳（預設啟用）"
    )
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="不加時間戳（會覆蓋舊報告）"
    )
    parser.add_argument(
        "--direction", "-d",
        type=str,
        choices=["both", "long_only", "short_only"],
        default=None,
        help="交易方向（覆蓋配置檔）: both=多空都做, long_only=只做多, short_only=只做空"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="參數組合數（用於 Deflated Sharpe Ratio 校正多重測試偏差）"
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="⚡ 快速模式：關閉 FR/Slippage 成本模型（僅供快速迭代，結果不可信）"
    )

    args = parser.parse_args()

    # 載入配置
    cfg = load_config(args.config)
    market_type = cfg.market_type_str  # "spot" or "futures"

    # 確定使用的策略
    strategy_name = args.strategy or cfg.strategy.name
    if not strategy_name:
        print("❌ 錯誤: 未指定策略名稱")
        print("   請在配置檔中設定 strategy.name，或使用 -s/--strategy 參數")
        return
    
    # 交易方向（命令列參數優先 → config 自動判斷）
    direction = args.direction or cfg.direction
    
    # 市場類型標籤
    market_emoji = "🟢" if market_type == "spot" else "🔴"
    market_label = "SPOT" if market_type == "spot" else "FUTURES"
    
    # 交易方向標籤
    direction_labels = {
        "both": "📊 多空都做",
        "long_only": "📈 只做多",
        "short_only": "📉 只做空",
    }
    direction_label = direction_labels.get(direction, direction)

    # 確定輸出目錄
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    use_timestamp = not args.no_timestamp  # 預設帶時間戳
    
    if args.output_dir:
        report_dir = Path(args.output_dir)
        if use_timestamp:
            report_dir = report_dir / timestamp_str
    else:
        base_report_dir = cfg.get_report_dir("backtest")
        if use_timestamp:
            report_dir = base_report_dir / timestamp_str
        else:
            report_dir = base_report_dir

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
    print(f"{market_emoji} 市場: {market_label}")
    if market_type == "futures":
        print(f"{direction_label}")
    print(f"📁 輸出目錄: {report_dir}")
    print(f"🕐 運行時間: {timestamp_str}")

    # 確定交易對
    symbols = [args.symbol] if args.symbol else cfg.market.symbols

    for sym in symbols:
        # ── Ensemble: 檢查是否有 per-symbol 策略路由 ──
        sym_strategy_name = strategy_name
        bt_cfg = cfg.to_backtest_dict(symbol=sym)

        ensemble_override = _load_ensemble_strategy(args.config, sym)
        if ensemble_override:
            sym_strategy_name, sym_params = ensemble_override
            bt_cfg["strategy_params"] = sym_params
            print(f"🧩 Ensemble: {sym} → {sym_strategy_name}")

        # 命令列 --direction 覆蓋
        if args.direction:
            bt_cfg["direction"] = args.direction
        # Simple mode：關閉成本模型
        if args.simple:
            bt_cfg["funding_rate"] = {"enabled": False}
            bt_cfg["slippage_model"] = {"enabled": False}
        # 根據 market_type 選擇數據路徑
        data_path = cfg.data_dir / "binance" / market_type / cfg.market.interval / f"{sym}.parquet"

        if not data_path.exists():
            print(f"⚠️  數據檔案不存在: {data_path}")
            print(f"   請先運行: python scripts/download_data.py -c {args.config} --symbol {sym}")
            continue

        print(f"\n{'='*60}")
        print(f"回測: {sym_strategy_name} - {sym} {market_emoji} [{market_label}] {direction_label}")
        print(f"{'='*60}")

        # leverage 已在 to_backtest_dict 中設定
        res = run_symbol_backtest(
            sym, data_path, bt_cfg, sym_strategy_name,
            data_dir=cfg.data_dir,
        )
        pf = res.pf
        pf_bh = res.pf_bh
        
        # 顯示成本模型狀態
        print(f"💰 成本模型: {res.cost_summary()}")
        
        # 顯示實際回測資料範圍
        df = res.df
        print(f"📅 資料範圍: {df.index[0].strftime('%Y-%m-%d %H:%M')} → {df.index[-1].strftime('%Y-%m-%d %H:%M')} ({len(df):,} bars)")

        # ── 0. 成本模型摘要（如果啟用）────────────────
        if res.slippage_result:
            sr = res.slippage_result
            print(f"\n📊 滑點模型: avg={sr.avg_slippage_bps:.1f}bps, max={sr.max_slippage_bps:.1f}bps, 高衝擊bar={sr.high_impact_bars}")

        if res.funding_cost:
            fc = res.funding_cost
            if fc.total_cost >= 0:
                print(f"💰 Funding 支出: ${fc.total_cost:,.2f} ({fc.total_cost_pct*100:.2f}%), 年化={fc.annualized_cost_pct*100:.2f}%/yr, 結算={fc.n_settlements}次")
            else:
                print(f"💰 Funding 收入: ${abs(fc.total_cost):,.2f} ({abs(fc.total_cost_pct)*100:.2f}%), 年化={abs(fc.annualized_cost_pct)*100:.2f}%/yr, 結算={fc.n_settlements}次")
                print(f"   （策略淨持空時段多 → 在 rate>0 時收取 funding）")

        # ── 1. 策略 vs Buy & Hold 對比報告 ──────────────
        report = full_report(pf, pf_bh, strategy_name)
        print(f"\n{'─'*50}")
        print(f"  {sym}  策略 vs Buy & Hold")
        print(f"{'─'*50}")
        print(report.to_string())

        # 如果有 funding 調整，顯示調整後的核心指標
        if res.adjusted_stats:
            adj = res.adjusted_stats
            orig_stats = pf.stats()
            print(f"\n{'─'*50}")
            print(f"  {sym}  Funding Rate 調整後績效")
            print(f"{'─'*50}")
            print(f"  {'指標':<30} {'原始':>12} {'調整後':>12} {'差異':>12}")
            print(f"  {'-'*66}")
            for key in ["Total Return [%]", "Max Drawdown [%]", "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio"]:
                orig_val = orig_stats.get(key, adj.get(key, 0))
                adj_val = adj.get(key, 0)
                # orig_stats 的 key 可能格式不同
                if key in orig_stats:
                    orig_val = orig_stats[key]
                diff = adj_val - orig_val
                print(f"  {key:<30} {orig_val:>12.2f} {adj_val:>12.2f} {diff:>+12.2f}")

        stats_path = report_dir / f"stats_{sym}.csv"
        report.to_csv(stats_path)
        print(f"\n✅ 統計報告: {stats_path}")

        # 儲存調整後的統計
        if res.adjusted_stats:
            import pandas as _pd
            adj_path = report_dir / f"stats_funding_adjusted_{sym}.csv"
            _pd.Series(res.adjusted_stats).to_csv(adj_path)
            print(f"✅ Funding 調整報告: {adj_path}")

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

        # ── 3. Long / Short 分開統計（合約模式）────────
        if market_type == "futures" and direction == "both":
            ls_analysis = long_short_split_analysis(pf, res.pos)
            if ls_analysis["df"] is not None and not ls_analysis["df"].empty:
                print(f"\n{'─'*50}")
                print(f"  {sym}  Long / Short 分開統計")
                print(f"{'─'*50}")
                print(ls_analysis["summary"])
                print()
                print(ls_analysis["df"].to_string())

                ls_path = report_dir / f"long_short_split_{sym}.csv"
                ls_analysis["df"].to_csv(ls_path)
                print(f"\n✅ Long/Short 統計: {ls_path}")

        # ── 4. 逐筆交易記錄 ────────────────────────────
        trades_df = trade_analysis(pf)
        if not trades_df.empty:
            trades_path = report_dir / f"trades_{sym}.csv"
            trades_df.to_csv(trades_path, index=False)
            print(f"✅ 逐筆交易: {trades_path}  ({len(trades_df)} 筆)")

        # ── 5. 資金曲線圖（含 Buy & Hold）───────────────
        plot_path = report_dir / f"equity_curve_{sym}.png"
        plot_backtest_summary(
            pf, res.df, res.pos, sym, plot_path,
            pf_benchmark=pf_bh,
            strategy_name=strategy_name,
        )
        print(f"✅ 資金曲線圖: {plot_path}")

        # ── 6. Deflated Sharpe Ratio（選用）─────────────
        if args.n_trials:
            returns = pf.returns()
            observed_sharpe = pf.stats().get("Sharpe Ratio", 0)
            skewness = returns.skew() if len(returns) > 2 else 0
            kurtosis = (returns.kurtosis() + 3) if len(returns) > 3 else 3

            dsr = deflated_sharpe_ratio(
                observed_sharpe=observed_sharpe,
                n_trials=args.n_trials,
                n_observations=len(returns),
                skewness=skewness,
                kurtosis=kurtosis,
            )

            print(f"\n{'─'*50}")
            print(f"  {sym}  Deflated Sharpe Ratio")
            print(f"{'─'*50}")
            print(f"  觀察 Sharpe:       {dsr.observed_sharpe:.2f}")
            print(f"  預期最大 (luck):   {dsr.expected_max_sharpe:.2f}")
            print(f"  Deflated Sharpe:   {dsr.deflated_sharpe:.2f}")
            print(f"  p-value:           {dsr.p_value:.4f}")
            print(f"  n_trials:          {dsr.n_trials}")

            if dsr.is_significant:
                print(f"  ✅ 統計顯著 (DSR > 0, p < 0.05)")
            else:
                print(f"  ⚠️  未達顯著水準")

        # ── 7. Red Flag Check ─────────────────────────────
        red_flags = check_red_flags(pf.stats())
        print_red_flags(red_flags)


if __name__ == "__main__":
    main()
