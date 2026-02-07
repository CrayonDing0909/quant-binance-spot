"""
策略验证脚本 - 检测过拟合

使用方法:
    # 使用配置文件
    python scripts/validate_strategy.py
    
    # 指定配置文件
    python scripts/validate_strategy.py -c config/rsi.yaml
    
    # 指定策略（覆盖配置文件中的策略）
    python scripts/validate_strategy.py -s rsi
"""
from __future__ import annotations
import argparse
from pathlib import Path
from qtrade.config import load_config
from qtrade.backtest.validation import (
    walk_forward_analysis,
    parameter_sensitivity_analysis,
    detect_overfitting,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="验证策略（过拟合检测）",
        formatter_class=argparse.RawDescriptionHelpFormatter
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
        "--output-dir",
        type=str,
        default=None,
        help="输出目录（默认: reports/{strategy_name}）"
    )
    
    args = parser.parse_args()
    
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
        # 按策略分类组织输出
        base_report_dir = Path(cfg.output.report_dir)
        report_dir = base_report_dir / strategy_name
    
    report_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📊 验证策略: {strategy_name}")
    print(f"📁 输出目录: {report_dir}")
    
    bt_cfg = {
        "initial_cash": cfg.backtest.initial_cash,
        "fee_bps": cfg.backtest.fee_bps,
        "slippage_bps": cfg.backtest.slippage_bps,
        "strategy_params": cfg.strategy.params,
        "strategy_name": strategy_name,
    }
    
    print("=" * 60)
    print("策略过拟合验证")
    print("=" * 60)
    
    for sym in cfg.market.symbols:
        print(f"\n{'='*60}")
        print(f"验证策略: {sym}")
        print(f"{'='*60}")
        
        data_path = cfg.data_dir / "binance" / "spot" / cfg.market.interval / f"{sym}.parquet"
        
        # 1. 滚动窗口验证
        print("\n[1] 滚动窗口验证 (Walk-Forward Analysis)...")
        wf_results = walk_forward_analysis(sym, data_path, bt_cfg, n_splits=5)
        
        if len(wf_results) > 0:
            print("\n滚动窗口结果:")
            print(wf_results.to_string(index=False))
            
            # 保存结果
            wf_path = report_dir / f"walk_forward_{sym}.csv"
            wf_results.to_csv(wf_path, index=False)
            print(f"\n✅ 滚动窗口结果已保存: {wf_path}")
            
            # 检测过拟合
            avg_train_return = wf_results["train_return"].mean()
            avg_test_return = wf_results["test_return"].mean()
            avg_train_sharpe = wf_results["train_sharpe"].mean()
            avg_test_sharpe = wf_results["test_sharpe"].mean()
            
            print(f"\n平均训练集收益率: {avg_train_return:.2f}%")
            print(f"平均测试集收益率: {avg_test_return:.2f}%")
            print(f"平均训练集夏普比率: {avg_train_sharpe:.2f}")
            print(f"平均测试集夏普比率: {avg_test_sharpe:.2f}")
            
            if avg_train_return > 0:
                return_drop = (avg_train_return - avg_test_return) / abs(avg_train_return)
                if return_drop > 0.3:
                    print(f"⚠️  警告: 测试集收益率下降 {return_drop*100:.1f}%，可能存在过拟合！")
                else:
                    print(f"✓ 测试集表现稳定，收益率下降 {return_drop*100:.1f}%")
        
        # 2. 参数敏感性分析
        print("\n[2] 参数敏感性分析...")
        # 根据策略类型设置参数网格
        if strategy_name == "ema_cross":
            param_grid = {
                "fast": [15, 20, 25],
                "slow": [55, 60, 65],
            }
        else:
            # 默认参数网格
            param_grid = {}
            for key, val in cfg.strategy.params.items():
                if isinstance(val, (int, float)):
                    param_grid[key] = [int(val * 0.8), val, int(val * 1.2)]
        
        if param_grid:
            sens_results = parameter_sensitivity_analysis(sym, data_path, bt_cfg, param_grid)
            print("\n参数敏感性结果:")
            print(sens_results.to_string(index=False))
            
            # 保存结果
            sens_path = report_dir / f"parameter_sensitivity_{sym}.csv"
            sens_results.to_csv(sens_path, index=False)
            print(f"\n✅ 参数敏感性结果已保存: {sens_path}")
            
            # 分析参数稳定性
            if len(sens_results) > 1:
                return_std = sens_results["total_return"].std()
                sharpe_std = sens_results["sharpe_ratio"].std()
                
                print(f"\n收益率标准差: {return_std:.2f}%")
                print(f"夏普比率标准差: {sharpe_std:.2f}")
                
                if return_std > 50:  # 收益率波动很大
                    print("⚠️  警告: 参数变化导致收益率波动很大，策略可能不稳定！")
                else:
                    print("✓ 策略对参数变化相对稳定")
        
        print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()

