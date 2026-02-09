"""
策略驗證腳本 - 檢測過擬合

使用方法:
    # 使用配置檔
    python scripts/validate_strategy.py

    # 指定配置檔和策略
    python scripts/validate_strategy.py -c config/rsi_adx_atr.yaml -s rsi_adx_atr

    # 只驗證單個交易對
    python scripts/validate_strategy.py -c config/rsi_adx_atr.yaml -s rsi_adx_atr --symbol BTCUSDT

    # 跳過參數敏感性分析（只做 walk-forward）
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


# ── 各策略的驗證用參數網格（只選最重要的 2-3 個參數）──
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

MAX_SENSITIVITY_COMBOS = 100  # 參數組合上限


def _auto_param_grid(params: dict, max_params: int = 3) -> dict:
    """
    自動生成驗證用參數網格（只選前 max_params 個數值參數）
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
        description="驗證策略（過擬合檢測）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-c", "--config", type=str, default="config/base.yaml",
                        help="配置檔路徑")
    parser.add_argument("-s", "--strategy", type=str, default=None,
                        help="策略名稱")
    parser.add_argument("--symbol", type=str, default=None,
                        help="指定交易對（預設使用配置中的所有交易對）")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="輸出目錄")
    parser.add_argument("--skip-sensitivity", action="store_true",
                        help="跳過參數敏感性分析")
    parser.add_argument("--splits", type=int, default=5,
                        help="Walk-forward 分割數（預設 5）")

    args = parser.parse_args()

    cfg = load_config(args.config)

    strategy_name = args.strategy or cfg.strategy.name
    if not strategy_name:
        print("❌ 錯誤: 未指定策略名稱")
        return

    if args.output_dir:
        report_dir = Path(args.output_dir)
    else:
        report_dir = Path(cfg.output.report_dir) / strategy_name

    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"📊 驗證策略: {strategy_name}")
    print(f"📁 輸出目錄: {report_dir}")

    symbols = [args.symbol] if args.symbol else cfg.market.symbols

    print("=" * 60)
    print("策略過擬合驗證")
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
        print(f"驗證: {strategy_name} - {sym}")
        print(f"{'='*60}")

        data_path = cfg.data_dir / "binance" / "spot" / cfg.market.interval / f"{sym}.parquet"

        if not data_path.exists():
            print(f"⚠️  數據檔案不存在: {data_path}")
            continue

        # ── 1. Walk-Forward Analysis ────────────────────
        print(f"\n[1] Walk-Forward Analysis ({args.splits} splits)...")
        wf_results = walk_forward_analysis(sym, data_path, bt_cfg, n_splits=args.splits)

        if len(wf_results) > 0:
            print("\n滾動窗口結果:")
            print(wf_results.to_string(index=False))

            wf_path = report_dir / f"walk_forward_{sym}.csv"
            wf_results.to_csv(wf_path, index=False)
            print(f"\n✅ 已儲存: {wf_path}")

            avg_train_return = wf_results["train_return"].mean()
            avg_test_return = wf_results["test_return"].mean()
            avg_train_sharpe = wf_results["train_sharpe"].mean()
            avg_test_sharpe = wf_results["test_sharpe"].mean()

            print(f"\n  {'指標':<20} {'訓練集':>10} {'測試集':>10}")
            print(f"  {'─'*42}")
            print(f"  {'平均收益率':.<20} {avg_train_return:>9.2f}% {avg_test_return:>9.2f}%")
            print(f"  {'平均夏普比率':.<20} {avg_train_sharpe:>10.2f} {avg_test_sharpe:>10.2f}")

            # 使用 Sharpe Ratio 做主要判定
            # （累積收益率因訓練/測試期長度不同而不可直接比較）
            if avg_train_sharpe > 0:
                sharpe_drop = (avg_train_sharpe - avg_test_sharpe) / abs(avg_train_sharpe)
                if sharpe_drop > 0.5:
                    print(f"\n  ❌ 高風險: 測試集夏普比率下降 {sharpe_drop*100:.1f}%，很可能過擬合！")
                elif sharpe_drop > 0.3:
                    print(f"\n  ⚠️  警告: 測試集夏普比率下降 {sharpe_drop*100:.1f}%，可能存在過擬合")
                elif sharpe_drop > 0:
                    print(f"\n  ✅ 穩定: 測試集夏普比率下降 {sharpe_drop*100:.1f}%，在合理範圍內")
                else:
                    print(f"\n  🟢 優秀: 測試集夏普比率優於訓練集（{avg_test_sharpe:.2f} > {avg_train_sharpe:.2f}）！")

            # 檢查測試集是否一致為正
            positive_tests = (wf_results["test_return"] > 0).sum()
            total_tests = len(wf_results)
            print(f"  📊 測試集為正收益: {positive_tests}/{total_tests} 個窗口")
        else:
            print("  ⚠️  Walk-forward 沒有結果")

        # ── 2. 參數敏感性分析 ──────────────────────────
        if args.skip_sensitivity:
            print("\n[2] 參數敏感性分析: 已跳過 (--skip-sensitivity)")
            continue

        print("\n[2] 參數敏感性分析...")

        # 獲取參數網格
        param_grid = VALIDATION_PARAM_GRIDS.get(strategy_name)
        if param_grid is None:
            param_grid = _auto_param_grid(cfg.strategy.params)

        if not param_grid:
            print("  ⚠️  無法生成參數網格，跳過")
            continue

        # 檢查組合數
        n_combos = 1
        for v in param_grid.values():
            n_combos *= len(v)

        if n_combos > MAX_SENSITIVITY_COMBOS:
            print(f"  ⚠️  參數組合太多 ({n_combos})，自動限制到最重要的參數")
            # 只保留前 2 個參數
            keys = list(param_grid.keys())[:2]
            param_grid = {k: param_grid[k] for k in keys}
            n_combos = 1
            for v in param_grid.values():
                n_combos *= len(v)

        print(f"  參數: {list(param_grid.keys())}，共 {n_combos} 種組合")

        sens_results = parameter_sensitivity_analysis(sym, data_path, bt_cfg, param_grid)
        print("\n參數敏感性結果:")
        print(sens_results.to_string(index=False))

        sens_path = report_dir / f"parameter_sensitivity_{sym}.csv"
        sens_results.to_csv(sens_path, index=False)
        print(f"\n✅ 已儲存: {sens_path}")

        if len(sens_results) > 1:
            return_std = sens_results["total_return"].std()
            return_mean = sens_results["total_return"].mean()
            sharpe_std = sens_results["sharpe_ratio"].std()
            sharpe_mean = sens_results["sharpe_ratio"].mean()

            print(f"\n  {'指標':<20} {'均值':>10} {'標準差':>10} {'變異係數':>10}")
            print(f"  {'─'*52}")
            cv_ret = return_std / abs(return_mean) * 100 if return_mean != 0 else float('inf')
            cv_sharpe = sharpe_std / abs(sharpe_mean) * 100 if sharpe_mean != 0 else float('inf')
            print(f"  {'收益率 [%]':.<20} {return_mean:>10.2f} {return_std:>10.2f} {cv_ret:>9.1f}%")
            print(f"  {'夏普比率':.<20} {sharpe_mean:>10.2f} {sharpe_std:>10.2f} {cv_sharpe:>9.1f}%")

            # 檢查所有組合是否都為正收益
            all_positive = (sens_results["total_return"] > 0).all()
            positive_pct = (sens_results["total_return"] > 0).sum() / len(sens_results) * 100

            if all_positive:
                print(f"\n  🟢 優秀: 所有參數組合都獲得正收益")
            elif positive_pct >= 80:
                print(f"\n  ✅ 穩健: {positive_pct:.0f}% 的參數組合獲得正收益")
            elif positive_pct >= 50:
                print(f"\n  ⚠️  一般: 只有 {positive_pct:.0f}% 的參數組合獲得正收益")
            else:
                print(f"\n  ❌ 不穩定: 只有 {positive_pct:.0f}% 的參數組合獲得正收益")

        print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
