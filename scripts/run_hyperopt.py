#!/usr/bin/env python3
"""
Hyperopt 參數優化示例腳本

使用方法：
    # 基礎優化（100 次迭代）
    python scripts/run_hyperopt.py
    
    # 指定參數
    python scripts/run_hyperopt.py --symbol ETHUSDT --trials 200 --objective sharpe_ratio
    
    # 並行優化
    python scripts/run_hyperopt.py --n-jobs 4 --trials 500

可用的優化目標：
    - sharpe_ratio: 夏普比率（風險調整後報酬，推薦）
    - sortino_ratio: 索提諾比率（只考慮下行風險）
    - total_return: 總報酬率
    - win_rate: 勝率
    - profit_factor: 盈虧比
    - max_drawdown: 最大回撤（越小越好）
    - calmar_ratio: 卡瑪比率
    - risk_adjusted: 複合目標（綜合多個指標）
"""
import argparse
import sys
from pathlib import Path

# 添加專案路徑
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qtrade.backtest.hyperopt_engine import (
    HyperoptEngine,
    ParamSpace,
    RSI_ADX_ATR_PARAM_SPACE,
)
from qtrade.data.storage import find_klines_file


def parse_args():
    parser = argparse.ArgumentParser(description="Hyperopt Parameter Optimization")
    parser.add_argument("--symbol", default="BTCUSDT", help="交易對")
    parser.add_argument("--strategy", default="rsi_adx_atr", help="策略名稱")
    parser.add_argument("--trials", type=int, default=100, help="優化迭代次數")
    parser.add_argument("--objective", default="sharpe_ratio", 
                       choices=["sharpe_ratio", "sortino_ratio", "total_return", 
                               "win_rate", "profit_factor", "max_drawdown", 
                               "calmar_ratio", "risk_adjusted"],
                       help="優化目標")
    parser.add_argument("--n-jobs", type=int, default=1, help="並行數（-1=所有CPU）")
    parser.add_argument("--market-type", default="spot", choices=["spot", "futures"])
    parser.add_argument("--direction", default="both", 
                       choices=["both", "long_only", "short_only"])
    parser.add_argument("--interval", default="1h", help="K 線週期")
    parser.add_argument("--no-plot", action="store_true", help="不顯示圖表")
    parser.add_argument("--seed", type=int, default=42, help="隨機種子")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("🎯 Hyperopt Parameter Optimization")
    print("=" * 60)
    print(f"Symbol: {args.symbol}")
    print(f"Strategy: {args.strategy}")
    print(f"Objective: {args.objective}")
    print(f"Trials: {args.trials}")
    print(f"Parallel Jobs: {args.n_jobs}")
    print("=" * 60)
    
    # ── 配置 ──
    base_cfg = {
        "initial_cash": 10000,
        "fee_bps": 10,      # 0.1%
        "slippage_bps": 5,  # 0.05%
        "interval": args.interval,
        "market_type": args.market_type,
        "direction": args.direction,
        "strategy_params": {},  # 會被 Hyperopt 覆蓋
    }
    
    # ── 尋找數據文件 ──
    data_dir = Path(__file__).parent.parent / "data"
    data_path = find_klines_file(data_dir, args.symbol, args.interval)
    
    if data_path is None:
        print(f"❌ 找不到 {args.symbol} {args.interval} 的數據文件")
        print(f"   請先運行: python scripts/download_data.py --symbol {args.symbol}")
        sys.exit(1)
    
    print(f"📊 數據文件: {data_path}")
    
    # ── 參數空間 ──
    if args.strategy == "rsi_adx_atr":
        param_space = RSI_ADX_ATR_PARAM_SPACE
    else:
        # 自定義參數空間示例
        param_space = {
            "fast_period": ParamSpace.integer("fast_period", 5, 20),
            "slow_period": ParamSpace.integer("slow_period", 20, 100),
        }
    
    print(f"🔧 參數空間: {list(param_space.keys())}")
    
    # ── 創建引擎並優化 ──
    engine = HyperoptEngine(
        strategy_name=args.strategy,
        data_path=data_path,
        base_cfg=base_cfg,
        param_space=param_space,
        symbol=args.symbol,
        market_type=args.market_type,
        direction=args.direction,
    )
    
    result = engine.optimize(
        n_trials=args.trials,
        objective=args.objective,
        n_jobs=args.n_jobs,
        seed=args.seed,
    )
    
    # ── 顯示結果 ──
    print("\n")
    print(result.summary())
    
    # ── Top 10 結果 ──
    print("\n📊 Top 10 Trials:")
    top_trials = engine.get_top_trials(10)
    print(top_trials[["number", "value", "user_attrs_total_return", 
                      "user_attrs_sharpe_ratio", "user_attrs_max_drawdown",
                      "user_attrs_total_trades"]].to_string())
    
    # ── 可視化 ──
    if not args.no_plot:
        print("\n📈 Generating plots...")
        try:
            engine.plot_optimization_history()
            engine.plot_param_importances()
            
            # 如果有足夠的參數，繪製等高線圖
            param_names = list(param_space.keys())
            if len(param_names) >= 2:
                engine.plot_contour(param_names[0], param_names[1])
        except Exception as e:
            print(f"⚠️  圖表生成失敗: {e}")
            print("   可能需要安裝: pip install plotly")
    
    # ── 保存結果 ──
    output_dir = Path(__file__).parent.parent / "reports" / "hyperopt"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存最佳參數
    import json
    best_params_file = output_dir / f"{args.symbol}_{args.strategy}_best_params.json"
    with open(best_params_file, "w") as f:
        json.dump({
            "symbol": args.symbol,
            "strategy": args.strategy,
            "objective": args.objective,
            "best_value": result.best_value,
            "best_params": result.best_params,
            "n_trials": args.trials,
        }, f, indent=2)
    print(f"\n✅ 最佳參數已保存: {best_params_file}")
    
    # 保存所有試驗結果
    trials_file = output_dir / f"{args.symbol}_{args.strategy}_trials.csv"
    result.all_trials.to_csv(trials_file, index=False)
    print(f"✅ 試驗結果已保存: {trials_file}")
    
    print("\n🎉 Hyperopt 完成！")
    print(f"\n💡 使用最佳參數運行回測:")
    print(f"   將以下參數複製到你的配置文件中:")
    print(f"   strategy_params: {json.dumps(result.best_params, indent=4)}")


if __name__ == "__main__":
    main()
