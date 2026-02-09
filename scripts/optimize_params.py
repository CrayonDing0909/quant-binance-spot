"""
參數優化工具

使用網格搜索或隨機搜索優化策略參數。

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
    網格搜索優化參數
    
    Args:
        symbol: 交易對符號
        data_path: 數據路徑
        base_cfg: 基礎回測配置
        param_grid: 參數網格，例如 {"fast": [10, 20, 30], "slow": [50, 60, 70]}
        metric: 優化目標指標
    
    Returns:
        包含所有參數組合結果的 DataFrame
    """
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    results = []
    total_combinations = len(list(product(*param_values)))
    
    print(f"開始網格搜索，共 {total_combinations} 種參數組合...")
    
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
                print(f"進度: {i}/{total_combinations} ({i/total_combinations*100:.1f}%)")
        except Exception as e:
            print(f"⚠️  參數組合 {combo} 失敗: {e}")
            continue
    
    if not results:
        print("❌ 所有參數組合都失敗了，無法生成結果")
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    
    # 按優化指標排序
    if metric in df.columns:
        df = df.sort_values(metric, ascending=False)
    elif "total_return" in df.columns:
        print(f"⚠️  指標 {metric} 不存在，按 total_return 排序")
        df = df.sort_values("total_return", ascending=False)
    else:
        print(f"⚠️  無法找到排序指標，返回原始結果")
    
    return df


def get_param_grid(strategy_name: str) -> dict:
    """
    根據策略名稱獲取預設參數網格
    
    如果沒有找到預定義的參數網格，會嘗試從配置檔中讀取策略參數，
    並自動生成一個參數網格（在原始值附近變化 ±20%）。
    
    Args:
        strategy_name: 策略名稱
    
    Returns:
        參數網格字典
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
        # 自定義策略的參數網格
        "my_rsi_strategy": {
            "period": [10, 12, 14, 16, 18],
            "oversold": [25, 30, 35],
            "overbought": [65, 70, 75],
        },
        # RSI + ADX + ATR 組合策略
        "rsi_adx_atr": {
            "rsi_period": [10, 14, 18],
            "oversold": [30, 35, 40],
            "overbought": [65, 70, 75],
            "min_adx": [15, 20, 25],
            "stop_loss_atr": [1.5, 2.0, 2.5],
            "take_profit_atr": [2.5, 3.0, 4.0],
        },
        "rsi_adx_atr_trailing": {
            "rsi_period": [10, 14, 18],
            "oversold": [30, 35, 40],
            "min_adx": [15, 20, 25],
            "stop_loss_atr": [1.5, 2.0, 2.5],
            "trailing_stop_atr": [2.0, 2.5, 3.0],
        },
        "ema_cross_protected": {
            "fast": [15, 20, 25],
            "slow": [50, 60, 70],
            "min_adx": [20, 25, 30],
            "stop_loss_atr": [1.5, 2.0, 2.5],
            "take_profit_atr": [2.5, 3.0, 4.0],
        },
    }
    
    return grids.get(strategy_name, {})


def main() -> None:
    parser = argparse.ArgumentParser(description="優化策略參數")
    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        help="策略名稱"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="grid",
        choices=["grid"],
        help="優化方法（目前只支援 grid）"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="Total Return [%]",
        help="優化目標指標（Total Return [%%], Sharpe Ratio, 等）"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/base.yaml",
        help="配置檔路徑"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="指定交易對（預設使用配置中的所有交易對）"
    )
    
    args = parser.parse_args()
    
    # 載入配置
    cfg = load_config(args.config)
    
    # 驗證策略是否存在
    from qtrade.strategy import get_strategy
    try:
        get_strategy(args.strategy)
    except ValueError as e:
        print(f"❌ 錯誤: {e}")
        print(f"\n💡 提示:")
        print(f"   1. 確保策略已建立並註冊")
        print(f"   2. 檢查策略名稱是否正確")
        print(f"   3. 如果策略檔案已建立，確保在 src/qtrade/strategy/__init__.py 中導入")
        return
    
    # 獲取參數網格
    param_grid = get_param_grid(args.strategy)
    if not param_grid:
        # 嘗試從配置檔中自動生成參數網格
        print(f"⚠️  策略 {args.strategy} 沒有預設參數網格")
        print("嘗試從配置檔中自動生成參數網格...")
        
        strategy_params = cfg.strategy.params
        if strategy_params:
            param_grid = {}
            for key, val in strategy_params.items():
                if isinstance(val, (int, float)):
                    base_val = float(val)
                    if base_val > 0:
                        param_grid[key] = [
                            int(base_val * 0.8) if isinstance(val, int) else base_val * 0.8,
                            int(base_val) if isinstance(val, int) else base_val,
                            int(base_val * 1.2) if isinstance(val, int) else base_val * 1.2,
                        ]
                    else:
                        param_grid[key] = [val]
                elif isinstance(val, list):
                    param_grid[key] = val
            
            if param_grid:
                print(f"✅ 自動生成的參數網格: {param_grid}")
            else:
                print("❌ 無法自動生成參數網格")
                return
        else:
            print("❌ 配置檔中沒有策略參數")
            return
    
    print(f"參數網格: {param_grid}")
    
    # 確定交易對
    symbols = [args.symbol] if args.symbol else cfg.market.symbols
    
    # 對每個交易對進行優化
    all_results = {}
    
    for sym in symbols:
        # 準備回測配置（每個幣種使用合併後的參數）
        bt_cfg = {
            "initial_cash": cfg.backtest.initial_cash,
            "fee_bps": cfg.backtest.fee_bps,
            "slippage_bps": cfg.backtest.slippage_bps,
            "strategy_params": cfg.strategy.get_params(sym),
            "strategy_name": args.strategy,
        }
        print(f"\n{'='*60}")
        print(f"優化策略: {args.strategy} - {sym}")
        print(f"{'='*60}")
        
        data_path = cfg.data_dir / "binance" / "spot" / cfg.market.interval / f"{sym}.parquet"
        
        if not data_path.exists():
            print(f"⚠️  數據檔案不存在: {data_path}")
            continue
        
        # 執行優化
        if args.method == "grid":
            results = grid_search(sym, data_path, bt_cfg, param_grid, args.metric)
            
            if results.empty:
                print(f"⚠️  {sym} 優化失敗，跳過")
                continue
        else:
            print(f"❌ 不支援的優化方法: {args.method}")
            return
        
        # 儲存結果
        report_dir = Path(cfg.output.report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = report_dir / f"optimization_{args.strategy}_{sym}.csv"
        results.to_csv(output_file, index=False)
        print(f"\n✅ 優化結果已儲存: {output_file}")
        
        # 顯示最佳參數
        print(f"\n📊 最佳參數組合（按 {args.metric} 排序）:")
        print(results.head(10).to_string(index=False))
        
        all_results[sym] = results
    
    # 匯總結果
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("匯總結果")
        print(f"{'='*60}")
        
        for sym, results in all_results.items():
            best = results.iloc[0]
            print(f"\n{sym} 最佳參數:")
            for param in param_grid.keys():
                print(f"  {param}: {best[param]}")
            print(f"  {args.metric}: {best.get(args.metric.replace(' [%]', '').lower().replace(' ', '_'), 'N/A')}")


if __name__ == "__main__":
    main()
