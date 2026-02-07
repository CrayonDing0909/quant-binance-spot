"""
参数优化工具

使用网格搜索或随机搜索优化策略参数。

使用方法:
    python scripts/optimize_params.py --strategy rsi
    python scripts/optimize_params.py --strategy ema_cross --method grid
    python scripts/optimize_params.py --strategy rsi --metric sharpe
"""
from __future__ import annotations
import argparse
from pathlib import Path
from itertools import product
import pandas as pd
from qtrade.config import load_config
from qtrade.backtest.run_backtest import run_symbol_backtest


def grid_search(
    symbol: str,
    data_path: Path,
    base_cfg: dict,
    param_grid: dict,
    metric: str = "Total Return [%]"
) -> pd.DataFrame:
    """
    网格搜索优化参数
    
    Args:
        symbol: 交易对符号
        data_path: 数据路径
        base_cfg: 基础回测配置
        param_grid: 参数网格，例如 {"fast": [10, 20, 30], "slow": [50, 60, 70]}
        metric: 优化目标指标
    
    Returns:
        包含所有参数组合结果的 DataFrame
    """
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    results = []
    total_combinations = len(list(product(*param_values)))
    
    print(f"开始网格搜索，共 {total_combinations} 种参数组合...")
    
    for i, combo in enumerate(product(*param_values), 1):
        params = dict(zip(param_names, combo))
        cfg = base_cfg.copy()
        cfg["strategy_params"] = {**base_cfg["strategy_params"], **params}
        
        try:
            res = run_symbol_backtest(symbol, data_path, cfg, cfg.get("strategy_name"))
            stats = res["stats"]
            
            result = {name: val for name, val in zip(param_names, combo)}
            result.update({
                "total_return": stats.get("Total Return [%]", 0),
                "sharpe_ratio": stats.get("Sharpe Ratio", 0),
                "max_drawdown": stats.get("Max Drawdown [%]", 0),
                "win_rate": stats.get("Win Rate [%]", 0),
                "total_trades": stats.get("Total Trades", 0),
            })
            results.append(result)
            
            if i % 10 == 0:
                print(f"进度: {i}/{total_combinations} ({i/total_combinations*100:.1f}%)")
        except Exception as e:
            print(f"⚠️  参数组合 {combo} 失败: {e}")
            continue
    
    if not results:
        print("❌ 所有参数组合都失败了，无法生成结果")
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    
    # 按优化指标排序
    if metric in df.columns:
        df = df.sort_values(metric, ascending=False)
    elif "total_return" in df.columns:
        print(f"⚠️  指标 {metric} 不存在，按 total_return 排序")
        df = df.sort_values("total_return", ascending=False)
    else:
        print(f"⚠️  无法找到排序指标，返回原始结果")
    
    return df


def get_param_grid(strategy_name: str) -> dict:
    """
    根据策略名称获取默认参数网格
    
    如果没有找到预定义的参数网格，会尝试从配置文件中读取策略参数，
    并自动生成一个参数网格（在原始值附近变化 ±20%）。
    
    Args:
        strategy_name: 策略名称
    
    Returns:
        参数网格字典
    """
    grids = {
        "rsi": {
            "period": [10, 12, 14, 16, 18],
            "oversold": [25, 30, 35],
            "overbought": [65, 70, 75],
        },
        "ema_cross": {
            "fast": [10, 15, 20, 25, 30],
            "slow": [50, 60, 70, 80],
        },
        "rsi_momentum": {
            "period": [12, 14, 16],
            "oversold": [25, 30, 35],
            "overbought": [65, 70, 75],
            "exit_threshold": [45, 50, 55],
        },
        # 自定义策略的参数网格
        "my_rsi_strategy": {
            "period": [10, 12, 14, 16, 18],
            "oversold": [25, 30, 35],
            "overbought": [65, 70, 75],
        },
    }
    
    return grids.get(strategy_name, {})


def main() -> None:
    parser = argparse.ArgumentParser(description="优化策略参数")
    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        help="策略名称"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="grid",
        choices=["grid"],
        help="优化方法（目前只支持 grid）"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="Total Return [%]",
        help="优化目标指标（Total Return [%], Sharpe Ratio, 等）"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/base.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="指定交易对（默认使用配置中的所有交易对）"
    )
    
    args = parser.parse_args()
    
    # 加载配置
    cfg = load_config(args.config)
    
    # 验证策略是否存在
    from qtrade.strategy import get_strategy
    try:
        get_strategy(args.strategy)
    except ValueError as e:
        print(f"❌ 错误: {e}")
        print(f"\n💡 提示:")
        print(f"   1. 确保策略已创建并注册")
        print(f"   2. 检查策略名称是否正确")
        print(f"   3. 如果策略文件已创建，确保在 src/qtrade/strategy/__init__.py 中导入")
        return
    
    # 准备回测配置
    bt_cfg = {
        "initial_cash": cfg.backtest.initial_cash,
        "fee_bps": cfg.backtest.fee_bps,
        "slippage_bps": cfg.backtest.slippage_bps,
        "strategy_params": cfg.strategy.params,
        "strategy_name": args.strategy,
    }
    
    # 获取参数网格
    param_grid = get_param_grid(args.strategy)
    if not param_grid:
        # 尝试从配置文件中自动生成参数网格
        print(f"⚠️  策略 {args.strategy} 没有默认参数网格")
        print("尝试从配置文件中自动生成参数网格...")
        
        strategy_params = cfg.strategy.params
        if strategy_params:
            param_grid = {}
            for key, val in strategy_params.items():
                if isinstance(val, (int, float)):
                    # 在原始值附近生成参数网格（±20%）
                    base_val = float(val)
                    if base_val > 0:
                        param_grid[key] = [
                            int(base_val * 0.8) if isinstance(val, int) else base_val * 0.8,
                            int(base_val) if isinstance(val, int) else base_val,
                            int(base_val * 1.2) if isinstance(val, int) else base_val * 1.2,
                        ]
                    else:
                        # 对于负数或零，使用固定范围
                        param_grid[key] = [val]
                elif isinstance(val, list):
                    # 如果已经是列表，直接使用
                    param_grid[key] = val
            
            if param_grid:
                print(f"✅ 自动生成的参数网格: {param_grid}")
            else:
                print("❌ 无法自动生成参数网格")
                print("请手动在 get_param_grid 函数中添加参数网格，或修改配置文件")
                return
        else:
            print("❌ 配置文件中没有策略参数")
            print("请手动在 get_param_grid 函数中添加参数网格")
            return
    
    print(f"参数网格: {param_grid}")
    
    # 确定交易对
    symbols = [args.symbol] if args.symbol else cfg.market.symbols
    
    # 对每个交易对进行优化
    all_results = {}
    
    for sym in symbols:
        print(f"\n{'='*60}")
        print(f"优化策略: {args.strategy} - {sym}")
        print(f"{'='*60}")
        
        data_path = cfg.data_dir / "binance" / "spot" / cfg.market.interval / f"{sym}.parquet"
        
        if not data_path.exists():
            print(f"⚠️  数据文件不存在: {data_path}")
            continue
        
        # 执行优化
        if args.method == "grid":
            results = grid_search(sym, data_path, bt_cfg, param_grid, args.metric)
            
            if results.empty:
                print(f"⚠️  {sym} 优化失败，跳过")
                continue
        else:
            print(f"❌ 不支持的优化方法: {args.method}")
            return
        
        # 保存结果
        report_dir = Path(cfg.output.report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = report_dir / f"optimization_{args.strategy}_{sym}.csv"
        results.to_csv(output_file, index=False)
        print(f"\n✅ 优化结果已保存: {output_file}")
        
        # 显示最佳参数
        print(f"\n📊 最佳参数组合（按 {args.metric} 排序）:")
        print(results.head(10).to_string(index=False))
        
        all_results[sym] = results
    
    # 汇总结果
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("汇总结果")
        print(f"{'='*60}")
        
        for sym, results in all_results.items():
            best = results.iloc[0]
            print(f"\n{sym} 最佳参数:")
            for param in param_grid.keys():
                print(f"  {param}: {best[param]}")
            print(f"  {args.metric}: {best.get(args.metric.replace(' [%]', '').lower().replace(' ', '_'), 'N/A')}")


if __name__ == "__main__":
    main()

