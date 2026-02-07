"""
策略验证脚本 - 检测过拟合

使用方法:
    # 使用配置文件
    python scripts/validate_strategy.py

    # 指定配置文件和策略
    python scripts/validate_strategy.py -c config/rsi_adx_atr.yaml -s rsi_adx_atr

    # 只验证单个交易对
    python scripts/validate_strategy.py -c config/rsi_adx_atr.yaml -s rsi_adx_atr --symbol BTCUSDT

    # 跳过参数敏感性分析（只做 walk-forward）
    python scripts/validate_strategy.py -s rsi_adx_atr --skip-sensitivity
"""
from __future__ import annotations
import argparse
from pathlib import Path
from itertools import product
from qtrade.config import load_config
from qtrade.backtest.validation import (
    walk_forward_analysis,
    parameter_sensitivity_analysis,
)


# ── 各策略的验证用参数网格（只选最重要的 2-3 个参数）──
VALIDATION_PARAM_GRIDS = {
    "rsi_adx_atr": {
        "oversold": [30, 35, 40],
        "min_adx": [15, 20, 25],
        "stop_loss_atr": [1.5, 2.0, 2.5],
    },
    "rsi_adx_atr_trailing": {
        "oversold": [30, 35, 40],
        "min_adx": [15, 20, 25],
        "trailing_stop_atr": [2.0, 2.5, 3.0],
    },
    "ema_cross": {
        "fast": [15, 20, 25],
        "slow": [55, 60, 65],
    },
    "ema_cross_protected": {
        "fast": [15, 20, 25],
        "slow": [55, 60, 65],
        "min_adx": [20, 25, 30],
    },
    "rsi": {
        "period": [10, 14, 18],
        "oversold": [25, 30, 35],
        "overbought": [65, 70, 75],
    },
    "rsi_momentum": {
        "period": [12, 14, 16],
        "oversold": [25, 30, 35],
        "overbought": [65, 70, 75],
    },
}

MAX_SENSITIVITY_COMBOS = 100  # 参数组合上限


def _auto_param_grid(params: dict, max_params: int = 3) -> dict:
    """
    自动生成验证用参数网格（只选前 max_params 个数值参数）
    """
    grid = {}
    count = 0
    for key, val in params.items():
        if count >= max_params:
            break
        if isinstance(val, int) and val > 0:
            grid[key] = [int(val * 0.8), val, int(val * 1.2)]
            count += 1
        elif isinstance(val, float) and val > 0:
            grid[key] = [round(val * 0.8, 2), val, round(val * 1.2, 2)]
            count += 1
    return grid


def main() -> None:
    parser = argparse.ArgumentParser(
        description="验证策略（过拟合检测）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-c", "--config", type=str, default="config/base.yaml",
                        help="配置文件路径")
    parser.add_argument("-s", "--strategy", type=str, default=None,
                        help="策略名称")
    parser.add_argument("--symbol", type=str, default=None,
                        help="指定交易对（默认使用配置中的所有交易对）")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录")
    parser.add_argument("--skip-sensitivity", action="store_true",
                        help="跳过参数敏感性分析")
    parser.add_argument("--splits", type=int, default=5,
                        help="Walk-forward 分割数（默认 5）")

    args = parser.parse_args()

    cfg = load_config(args.config)

    strategy_name = args.strategy or cfg.strategy.name
    if not strategy_name:
        print("❌ 错误: 未指定策略名称")
        return

    if args.output_dir:
        report_dir = Path(args.output_dir)
    else:
        report_dir = Path(cfg.output.report_dir) / strategy_name

    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"📊 验证策略: {strategy_name}")
    print(f"📁 输出目录: {report_dir}")

    symbols = [args.symbol] if args.symbol else cfg.market.symbols

    print("=" * 60)
    print("策略过拟合验证")
    print("=" * 60)

    for sym in symbols:
        bt_cfg = {
            "initial_cash": cfg.backtest.initial_cash,
            "fee_bps": cfg.backtest.fee_bps,
            "slippage_bps": cfg.backtest.slippage_bps,
            "strategy_params": cfg.strategy.get_params(sym),
            "strategy_name": strategy_name,
        }
        print(f"\n{'='*60}")
        print(f"验证: {strategy_name} - {sym}")
        print(f"{'='*60}")

        data_path = cfg.data_dir / "binance" / "spot" / cfg.market.interval / f"{sym}.parquet"

        if not data_path.exists():
            print(f"⚠️  数据文件不存在: {data_path}")
            continue

        # ── 1. Walk-Forward Analysis ────────────────────
        print(f"\n[1] Walk-Forward Analysis ({args.splits} splits)...")
        wf_results = walk_forward_analysis(sym, data_path, bt_cfg, n_splits=args.splits)

        if len(wf_results) > 0:
            print("\n滚动窗口结果:")
            print(wf_results.to_string(index=False))

            wf_path = report_dir / f"walk_forward_{sym}.csv"
            wf_results.to_csv(wf_path, index=False)
            print(f"\n✅ 已保存: {wf_path}")

            avg_train_return = wf_results["train_return"].mean()
            avg_test_return = wf_results["test_return"].mean()
            avg_train_sharpe = wf_results["train_sharpe"].mean()
            avg_test_sharpe = wf_results["test_sharpe"].mean()

            print(f"\n  {'指标':<20} {'训练集':>10} {'测试集':>10}")
            print(f"  {'─'*42}")
            print(f"  {'平均收益率':.<20} {avg_train_return:>9.2f}% {avg_test_return:>9.2f}%")
            print(f"  {'平均夏普比率':.<20} {avg_train_sharpe:>10.2f} {avg_test_sharpe:>10.2f}")

            # 使用 Sharpe Ratio 做主要判定
            # （累积收益率因训练/测试期长度不同而不可直接比较）
            if avg_train_sharpe > 0:
                sharpe_drop = (avg_train_sharpe - avg_test_sharpe) / abs(avg_train_sharpe)
                if sharpe_drop > 0.5:
                    print(f"\n  ❌ 高风险: 测试集夏普比率下降 {sharpe_drop*100:.1f}%，很可能过拟合！")
                elif sharpe_drop > 0.3:
                    print(f"\n  ⚠️  警告: 测试集夏普比率下降 {sharpe_drop*100:.1f}%，可能存在过拟合")
                elif sharpe_drop > 0:
                    print(f"\n  ✅ 稳定: 测试集夏普比率下降 {sharpe_drop*100:.1f}%，在合理范围内")
                else:
                    print(f"\n  🟢 优秀: 测试集夏普比率优于训练集（{avg_test_sharpe:.2f} > {avg_train_sharpe:.2f}）！")

            # 检查测试集是否一致为正
            positive_tests = (wf_results["test_return"] > 0).sum()
            total_tests = len(wf_results)
            print(f"  📊 测试集为正收益: {positive_tests}/{total_tests} 个窗口")
        else:
            print("  ⚠️  Walk-forward 没有结果")

        # ── 2. 参数敏感性分析 ──────────────────────────
        if args.skip_sensitivity:
            print("\n[2] 参数敏感性分析: 已跳过 (--skip-sensitivity)")
            continue

        print("\n[2] 参数敏感性分析...")

        # 获取参数网格
        param_grid = VALIDATION_PARAM_GRIDS.get(strategy_name)
        if param_grid is None:
            param_grid = _auto_param_grid(cfg.strategy.params)

        if not param_grid:
            print("  ⚠️  无法生成参数网格，跳过")
            continue

        # 检查组合数
        n_combos = 1
        for v in param_grid.values():
            n_combos *= len(v)

        if n_combos > MAX_SENSITIVITY_COMBOS:
            print(f"  ⚠️  参数组合太多 ({n_combos})，自动限制到最重要的参数")
            # 只保留前 2 个参数
            keys = list(param_grid.keys())[:2]
            param_grid = {k: param_grid[k] for k in keys}
            n_combos = 1
            for v in param_grid.values():
                n_combos *= len(v)

        print(f"  参数: {list(param_grid.keys())}，共 {n_combos} 种组合")

        sens_results = parameter_sensitivity_analysis(sym, data_path, bt_cfg, param_grid)
        print("\n参数敏感性结果:")
        print(sens_results.to_string(index=False))

        sens_path = report_dir / f"parameter_sensitivity_{sym}.csv"
        sens_results.to_csv(sens_path, index=False)
        print(f"\n✅ 已保存: {sens_path}")

        if len(sens_results) > 1:
            return_std = sens_results["total_return"].std()
            return_mean = sens_results["total_return"].mean()
            sharpe_std = sens_results["sharpe_ratio"].std()
            sharpe_mean = sens_results["sharpe_ratio"].mean()

            print(f"\n  {'指标':<20} {'均值':>10} {'标准差':>10} {'变异系数':>10}")
            print(f"  {'─'*52}")
            cv_ret = return_std / abs(return_mean) * 100 if return_mean != 0 else float('inf')
            cv_sharpe = sharpe_std / abs(sharpe_mean) * 100 if sharpe_mean != 0 else float('inf')
            print(f"  {'收益率 [%]':.<20} {return_mean:>10.2f} {return_std:>10.2f} {cv_ret:>9.1f}%")
            print(f"  {'夏普比率':.<20} {sharpe_mean:>10.2f} {sharpe_std:>10.2f} {cv_sharpe:>9.1f}%")

            # 检查所有组合是否都为正收益
            all_positive = (sens_results["total_return"] > 0).all()
            positive_pct = (sens_results["total_return"] > 0).sum() / len(sens_results) * 100

            if all_positive:
                print(f"\n  🟢 优秀: 所有参数组合都获得正收益")
            elif positive_pct >= 80:
                print(f"\n  ✅ 稳健: {positive_pct:.0f}% 的参数组合获得正收益")
            elif positive_pct >= 50:
                print(f"\n  ⚠️  一般: 只有 {positive_pct:.0f}% 的参数组合获得正收益")
            else:
                print(f"\n  ❌ 不稳定: 只有 {positive_pct:.0f}% 的参数组合获得正收益")

        print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
