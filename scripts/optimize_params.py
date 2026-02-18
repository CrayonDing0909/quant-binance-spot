"""
參數優化工具 v2 — Optuna TPE / CMA-ES / Grid Search

支援：
  - TPE 貝葉斯優化（預設，最高效）
  - CMA-ES 進化策略（連續參數最佳）
  - Grid Search 網格搜索（窮舉，向後相容）
  - 多幣種聯合優化（跨資產魯棒性）
  - Train/Test OOS 驗證（防止過擬合）
  - Walk-Forward 滾動驗證
  - 參數重要性分析

使用方法:
    # 基本 TPE 優化（推薦）
    python scripts/optimize_params.py -c config/futures_rsi_adx_atr.yaml --method tpe --n-trials 200

    # CMA-ES（適合連續參數空間）
    python scripts/optimize_params.py -c config/futures_rsi_adx_atr.yaml --method cmaes --n-trials 150

    # 帶 OOS 驗證（70% train / 30% test）
    python scripts/optimize_params.py -c config/futures_rsi_adx_atr.yaml --method tpe --n-trials 200 --oos-ratio 0.3

    # Walk-Forward 驗證（5 折）
    python scripts/optimize_params.py -c config/futures_rsi_adx_atr.yaml --method tpe --walk-forward 5

    # 擴展參數空間（含 Dynamic RSI, Adaptive SL, HTF 等）
    python scripts/optimize_params.py -c config/futures_rsi_adx_atr.yaml --method tpe --space extended

    # 指定單幣種 + 自定義目標
    python scripts/optimize_params.py -c config/futures_rsi_adx_atr.yaml --symbol ETHUSDT --objective sharpe_dd

    # 向後相容：Grid Search
    python scripts/optimize_params.py -c config/futures_rsi_adx_atr.yaml --method grid
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path

import pandas as pd

from qtrade.config import load_config
from qtrade.backtest.run_backtest import run_symbol_backtest
from qtrade.backtest.hyperopt_engine import (
    HyperoptEngine,
    WalkForwardValidator,
    ParamSpace,
    ParamDef,
    OptimizationResult,
    OBJECTIVES,
    PREDEFINED_SPACES,
    RSI_ADX_ATR_PARAM_SPACE,
    RSI_ADX_ATR_EXTENDED_PARAM_SPACE,
    get_objective_fn,
    split_data_for_oos,
    cleanup_oos_files,
)


# ══════════════════════════════════════════════════════════════
# Grid Search（向後相容）
# ══════════════════════════════════════════════════════════════

def grid_search(
    symbol: str,
    data_path: Path,
    base_cfg: dict,
    param_grid: dict,
    metric: str = "Total Return [%]"
) -> pd.DataFrame:
    """
    網格搜索優化參數（向後相容）
    """
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    results = []
    total_combinations = 1
    for v in param_values:
        total_combinations *= len(v)
    
    print(f"開始網格搜索，共 {total_combinations} 種參數組合...")
    
    for i, combo in enumerate(product(*param_values), 1):
        params = dict(zip(param_names, combo))
        cfg = base_cfg.copy()
        cfg["strategy_params"] = {**base_cfg["strategy_params"], **params}
        
        try:
            res = run_symbol_backtest(symbol, data_path, cfg, cfg.get("strategy_name"))
            stats = res.stats
            
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
    
    return df


# ══════════════════════════════════════════════════════════════
# 參數空間：從策略名稱自動選擇
# ══════════════════════════════════════════════════════════════

# 向後相容：舊的 Grid 參數定義
GRID_PARAM_GRIDS = {
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
}


def get_param_space_for_strategy(strategy_name: str, space_type: str = "core") -> dict[str, ParamDef]:
    """
    根據策略名稱和空間類型，返回 Optuna ParamDef 字典
    
    Args:
        strategy_name: 策略名稱
        space_type: "core" 或 "extended"
    """
    if space_type == "extended" and strategy_name.startswith("rsi_adx_atr"):
        return RSI_ADX_ATR_EXTENDED_PARAM_SPACE.copy()
    
    key = strategy_name
    if key in PREDEFINED_SPACES:
        return PREDEFINED_SPACES[key].copy()
    
    # 嘗試用前綴匹配
    for prefix in ["rsi_adx_atr", "ema_cross"]:
        if strategy_name.startswith(prefix) and prefix in PREDEFINED_SPACES:
            return PREDEFINED_SPACES[prefix].copy()
    
    return {}


def auto_generate_param_space(strategy_params: dict) -> dict[str, ParamDef]:
    """
    從現有策略參數自動生成 Optuna 搜索空間（±30% 範圍）
    
    只處理數值型參數，跳過 bool、str、None 等。
    """
    space = {}
    for key, val in strategy_params.items():
        if isinstance(val, bool):
            space[key] = ParamSpace.categorical(key, [True, False])
        elif isinstance(val, int) and val > 0:
            low = max(1, int(val * 0.7))
            high = int(val * 1.3)
            space[key] = ParamSpace.integer(key, low, high)
        elif isinstance(val, float) and val > 0:
            low = round(val * 0.7, 4)
            high = round(val * 1.3, 4)
            space[key] = ParamSpace.float(key, low, high)
        # 跳過 None、str 等非數值型
    
    return space


# ══════════════════════════════════════════════════════════════
# 輸出格式化
# ══════════════════════════════════════════════════════════════

def print_banner(strategy: str, symbols: list[str], method: str, objective: str, n_trials: int, market_type: str, direction: str):
    """印出漂亮的 banner"""
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + "  🧬 Hyperopt Parameter Optimizer v2".ljust(58) + "║")
    print("╠" + "═" * 58 + "╣")
    print("║" + f"  Strategy:  {strategy}".ljust(58) + "║")
    print("║" + f"  Symbols:   {', '.join(symbols)}".ljust(58) + "║")
    print("║" + f"  Method:    {method.upper()}".ljust(58) + "║")
    print("║" + f"  Objective: {objective}".ljust(58) + "║")
    print("║" + f"  Trials:    {n_trials}".ljust(58) + "║")
    print("║" + f"  Market:    {market_type} / {direction}".ljust(58) + "║")
    print("╚" + "═" * 58 + "╝")
    print()


def print_top_results(engine: HyperoptEngine, n: int = 10):
    """印出前 N 名結果"""
    top = engine.get_top_trials(n)
    
    # 找出 params_ 開頭的欄位
    param_cols = [c for c in top.columns if c.startswith("params_")]
    display_cols = ["number", "value"] + param_cols
    
    # 添加 user_attrs 中的指標
    attr_cols = [c for c in top.columns if c.startswith("user_attrs_")]
    display_cols += attr_cols
    
    available = [c for c in display_cols if c in top.columns]
    
    print(f"\n📊 Top {min(n, len(top))} Results:")
    print("─" * 80)
    
    if not available:
        print(top.head(n).to_string(index=False))
    else:
        display = top[available].copy()
        # 簡化欄位名
        rename = {}
        for c in display.columns:
            if c.startswith("params_"):
                rename[c] = c.replace("params_", "")
            elif c.startswith("user_attrs_"):
                rename[c] = c.replace("user_attrs_", "")
        display = display.rename(columns=rename)
        print(display.to_string(index=False))


def print_oos_results(oos_stats: dict):
    """印出 OOS 驗證結果"""
    print(f"\n{'='*60}")
    print("🔍 Out-of-Sample (OOS) Validation")
    print(f"{'='*60}")
    
    if oos_stats.get("avg_objective") is not None:
        print(f"  Avg OOS Objective: {oos_stats['avg_objective']:.4f}")
    
    per_symbol = oos_stats.get("per_symbol", {})
    for symbol, stats in per_symbol.items():
        if "error" in stats:
            print(f"  {symbol}: ❌ {stats['error']}")
        else:
            print(f"  {symbol}:")
            print(f"    Return: {stats.get('total_return', 0):.2f}%")
            print(f"    Sharpe: {stats.get('sharpe_ratio', 0):.4f}")
            print(f"    MaxDD:  {stats.get('max_drawdown', 0):.2f}%")
            print(f"    Trades: {stats.get('total_trades', 0)}")


def print_walk_forward_results(wf_df: pd.DataFrame):
    """印出 Walk-Forward 結果"""
    print(f"\n{'='*60}")
    print("🔄 Walk-Forward Validation Results")
    print(f"{'='*60}")
    
    for _, row in wf_df.iterrows():
        overfit = row.get("overfit_ratio", 0)
        emoji = "✅" if 0.5 < overfit < 2.0 else "⚠️"
        print(f"  Fold {int(row['fold'])}: Train={row['train_objective']:.4f} → "
              f"Test={row['test_objective']:.4f} "
              f"(ratio {overfit:.2f}x) {emoji}")
    
    if not wf_df.empty:
        avg_test = wf_df["test_objective"].mean()
        positive_folds = (wf_df["test_objective"] > 0).sum()
        total_folds = len(wf_df)
        print(f"\n  Average Test Objective: {avg_test:.4f}")
        print(f"  Positive Test Folds:   {positive_folds}/{total_folds}")
        
        # 判定過擬合風險
        avg_ratio = wf_df["overfit_ratio"].mean()
        if avg_ratio > 3.0:
            print("  ❌ 過擬合風險: 高（Train >> Test）")
        elif avg_ratio > 1.5:
            print("  ⚠️  過擬合風險: 中等")
        else:
            print("  ✅ 過擬合風險: 低（Train ≈ Test）")


# ══════════════════════════════════════════════════════════════
# Main CLI
# ══════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="🧬 Hyperopt Parameter Optimizer v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # TPE 優化（推薦，最高效）
  %(prog)s -c config/futures_rsi_adx_atr.yaml --method tpe --n-trials 200

  # 帶 OOS 驗證
  %(prog)s -c config/futures_rsi_adx_atr.yaml --method tpe --n-trials 200 --oos-ratio 0.3

  # Walk-Forward 驗證
  %(prog)s -c config/futures_rsi_adx_atr.yaml --method tpe --walk-forward 5

  # 擴展參數空間
  %(prog)s -c config/futures_rsi_adx_atr.yaml --method tpe --space extended

Available objectives: %(objectives)s
        """ % {"objectives": ", ".join(OBJECTIVES.keys()), "prog": "python scripts/optimize_params.py"},
    )
    
    # ── 必要參數 ──
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="config/futures_rsi_adx_atr.yaml",
        help="配置檔路徑（預設: config/futures_rsi_adx_atr.yaml）"
    )
    
    # ── 搜索方法 ──
    parser.add_argument(
        "--method",
        type=str,
        default="tpe",
        choices=["tpe", "cmaes", "grid"],
        help="搜索算法: tpe（貝葉斯，推薦）, cmaes（進化策略）, grid（窮舉）"
    )
    
    # ── 優化目標 ──
    parser.add_argument(
        "--objective",
        type=str,
        default="sharpe_dd",
        help=f"優化目標函數（預設: sharpe_dd）。可選: {', '.join(OBJECTIVES.keys())}"
    )
    
    # ── 試驗次數 ──
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="優化迭代次數（預設: 100，建議 TPE 用 200+，Grid 自動計算）"
    )
    
    # ── 參數空間 ──
    parser.add_argument(
        "--space",
        type=str,
        default="core",
        choices=["core", "extended"],
        help="參數空間範圍: core（核心參數）, extended（含 Dynamic RSI, Adaptive SL, HTF）"
    )
    
    # ── 幣種 ──
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="指定單一交易對（預設使用配置中的所有交易對做聯合優化）"
    )
    
    # ── OOS 驗證 ──
    parser.add_argument(
        "--oos-ratio",
        type=float,
        default=0.0,
        help="OOS 測試集比例，例如 0.3 = 70%% 訓練 / 30%% 測試（預設: 0 = 不做 OOS）"
    )
    
    # ── Walk-Forward ──
    parser.add_argument(
        "--walk-forward",
        type=int,
        default=0,
        help="Walk-Forward 驗證的 fold 數（預設: 0 = 不做 WF）"
    )
    parser.add_argument(
        "--wf-trials",
        type=int,
        default=50,
        help="Walk-Forward 每個 fold 的試驗數（預設: 50）"
    )
    
    # ── 並行 ──
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="並行數（預設: 1）"
    )
    
    # ── 其他 ──
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="隨機種子"
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=10,
        help="最低交易次數（少於此數的參數組合會被懲罰，預設: 10）"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="超時秒數（預設: 無限制）"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="安靜模式（減少輸出）"
    )
    # 向後相容
    parser.add_argument("--strategy", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--metric", type=str, default=None, help=argparse.SUPPRESS)
    
    args = parser.parse_args()
    
    # ── 載入配置 ──
    cfg = load_config(args.config)
    strategy_name = args.strategy or cfg.strategy.name
    market_type = cfg.market_type_str
    direction = cfg.direction
    
    # 向後相容 --metric
    objective = args.metric or args.objective
    
    # ── 驗證策略 ──
    from qtrade.strategy import get_strategy
    try:
        get_strategy(strategy_name)
    except ValueError as e:
        print(f"❌ 錯誤: {e}")
        return
    
    # ── 確定交易對 & 數據路徑 ──
    symbols = [args.symbol] if args.symbol else cfg.market.symbols
    symbol_data: dict[str, Path] = {}
    
    for sym in symbols:
        data_path = cfg.data_dir / "binance" / market_type / cfg.market.interval / f"{sym}.parquet"
        if not data_path.exists():
            print(f"⚠️  數據檔案不存在: {data_path}")
            print(f"   請先運行: python scripts/download_data.py -c {args.config} --symbol {sym}")
            continue
        symbol_data[sym] = data_path
    
    if not symbol_data:
        print("❌ 沒有可用的數據檔案")
        return
    
    # ── 確定參數空間 ──
    param_space = get_param_space_for_strategy(strategy_name, args.space)
    
    if not param_space:
        # 嘗試從配置自動生成
        print(f"⚠️  策略 {strategy_name} 沒有預定義的參數空間，嘗試自動生成...")
        strategy_params = cfg.strategy.params
        if strategy_params:
            param_space = auto_generate_param_space(strategy_params)
        
        if not param_space:
            print("❌ 無法確定參數空間")
            return
    
    # ── 回測配置 ──
    # 單幣種模式：傳 symbol 讓 symbol_overrides 生效
    # 多幣種模式：不傳 symbol，用全局 base params
    if args.symbol and len(symbol_data) == 1:
        base_bt_cfg = cfg.to_backtest_dict(symbol=args.symbol)
    else:
        base_bt_cfg = cfg.to_backtest_dict()
    
    # ── 印出 Banner ──
    if not args.quiet:
        print_banner(
            strategy=strategy_name,
            symbols=list(symbol_data.keys()),
            method=args.method,
            objective=objective,
            n_trials=args.n_trials,
            market_type=market_type,
            direction=direction,
        )
        
        print(f"📐 Parameter Space ({args.space}, {len(param_space)} params):")
        for name, pdef in param_space.items():
            if pdef.param_type == "categorical":
                print(f"   {name}: {pdef.choices}")
            else:
                step_str = f", step={pdef.step}" if pdef.step else ""
                print(f"   {name}: [{pdef.low}, {pdef.high}]{step_str} ({pdef.param_type})")
        print()
    
    # ── 報告目錄 ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = cfg.get_report_dir("optimize") / timestamp
    report_dir.mkdir(parents=True, exist_ok=True)
    
    t0 = time.time()
    
    # ════════════════════════════════════════════════════════
    # 模式 1: Walk-Forward 驗證
    # ════════════════════════════════════════════════════════
    if args.walk_forward > 0:
        print(f"🔄 Walk-Forward Validation: {args.walk_forward} folds × {args.wf_trials} trials/fold")
        print(f"   Method: {args.method.upper()}, Objective: {objective}")
        
        wf = WalkForwardValidator(
            strategy_name=strategy_name,
            symbol_data=symbol_data,
            base_cfg=base_bt_cfg,
            param_space=param_space,
            market_type=market_type,
            direction=direction,
            data_dir=cfg.data_dir,
        )
        
        wf_df = wf.run(
            n_splits=args.walk_forward,
            n_trials_per_fold=args.wf_trials,
            objective=objective,
            method=args.method,
        )
        
        # 印出結果
        print_walk_forward_results(wf_df)
        
        # 儲存
        wf_df.to_csv(report_dir / "walk_forward_results.csv", index=False)
        print(f"\n💾 Walk-Forward 結果已儲存: {report_dir / 'walk_forward_results.csv'}")
        
        elapsed = time.time() - t0
        print(f"\n⏱️  總耗時: {elapsed:.1f}s")
        return
    
    # ════════════════════════════════════════════════════════
    # 模式 2: OOS 驗證（Train/Test 分割）
    # ════════════════════════════════════════════════════════
    train_data = symbol_data
    test_data = {}
    
    if args.oos_ratio > 0:
        train_ratio = 1.0 - args.oos_ratio
        print(f"📂 OOS Split: Train {train_ratio*100:.0f}% / Test {args.oos_ratio*100:.0f}%")
        
        train_data = {}
        test_data = {}
        
        for sym, path in symbol_data.items():
            train_path, test_path = split_data_for_oos(path, train_ratio)
            train_data[sym] = train_path
            test_data[sym] = test_path
    
    # ════════════════════════════════════════════════════════
    # 模式 3: 標準優化
    # ════════════════════════════════════════════════════════
    
    # ── 向後相容：Grid Search 走舊路徑 ──
    if args.method == "grid" and len(symbol_data) == 1 and not args.oos_ratio:
        sym = list(symbol_data.keys())[0]
        
        # 從 ParamSpace 轉換為 grid
        param_grid = {}
        for name, pdef in param_space.items():
            if pdef.param_type == "int":
                step = pdef.step or 1
                param_grid[name] = list(range(pdef.low, pdef.high + 1, step))
            elif pdef.param_type == "float":
                if pdef.step:
                    vals = []
                    v = pdef.low
                    while v <= pdef.high + 1e-9:
                        vals.append(round(v, 6))
                        v += pdef.step
                    param_grid[name] = vals
                else:
                    param_grid[name] = [
                        round(pdef.low + (pdef.high - pdef.low) * i / 4, 4)
                        for i in range(5)
                    ]
            elif pdef.param_type == "categorical":
                param_grid[name] = pdef.choices
        
        total = 1
        for v in param_grid.values():
            total *= len(v)
        print(f"🔍 Grid Search: {total} 種參數組合")
        
        results = grid_search(
            sym,
            symbol_data[sym],
            base_bt_cfg,
            param_grid,
            metric="sharpe_ratio" if objective == "sharpe_dd" else objective,
        )
        
        if not results.empty:
            output_file = report_dir / f"grid_search_{strategy_name}_{sym}.csv"
            results.to_csv(output_file, index=False)
            print(f"\n✅ 結果已儲存: {output_file}")
            print(f"\n📊 Top 10:")
            print(results.head(10).to_string(index=False))
        
        elapsed = time.time() - t0
        print(f"\n⏱️  總耗時: {elapsed:.1f}s")
        return
    
    # ── Optuna 優化（TPE / CMA-ES / Grid-via-Optuna）──
    engine = HyperoptEngine(
        strategy_name=strategy_name,
        symbol_data=train_data,
        base_cfg=base_bt_cfg,
        param_space=param_space,
        market_type=market_type,
        direction=direction,
        data_dir=cfg.data_dir,
        min_trades=args.min_trades,
    )
    
    result = engine.optimize(
        n_trials=args.n_trials,
        objective=objective,
        method=args.method,
        n_jobs=args.n_jobs,
        timeout=args.timeout,
        show_progress=not args.quiet,
        seed=args.seed,
    )
    
    # ── 印出結果 ──
    if not args.quiet:
        print_top_results(engine, n=15)
    
    # ── OOS 驗證 ──
    if test_data:
        print("\n🔍 Running OOS validation with best params...")
        oos_stats = engine.run_oos_validation(
            best_params=result.best_params,
            oos_data=test_data,
            objective_fn=objective,
        )
        print_oos_results(oos_stats)
        
        # 比較 Train vs Test
        if oos_stats.get("avg_objective") is not None:
            train_val = result.best_value
            test_val = oos_stats["avg_objective"]
            
            print(f"\n  Train Objective: {train_val:.4f}")
            print(f"  Test Objective:  {test_val:.4f}")
            
            if test_val <= 0:
                print("  ❌ 嚴重過擬合！OOS 表現為負，訓練結果無法泛化")
                print("     建議：減少參數空間 / 增加訓練數據 / 用 --walk-forward 驗證")
            elif test_val > 0:
                ratio = train_val / test_val
                print(f"  Train / Test Ratio: {ratio:.2f}x")
                if ratio > 3.0:
                    print("  ❌ 嚴重過擬合風險！考慮減少參數空間或增加訓練數據")
                elif ratio > 1.5:
                    print("  ⚠️  中度過擬合風險，建議用 --walk-forward 進一步驗證")
                else:
                    print("  ✅ Train ≈ Test，過擬合風險低")
        
        # 清理臨時檔案
        cleanup_oos_files(symbol_data)
    
    # ── 儲存結果 ──
    engine.save_results(report_dir)
    
    # 額外：儲存最佳參數為可直接貼到 YAML 的格式
    best_yaml_lines = ["# 🧬 Hyperopt Best Parameters", f"# Generated: {timestamp}", f"# Objective: {objective} = {result.best_value:.4f}", f"# Method: {args.method}", ""]
    best_yaml_lines.append("strategy:")
    best_yaml_lines.append(f"  name: \"{strategy_name}\"")
    best_yaml_lines.append("  params:")
    
    # 先放 base params，再覆蓋 best params
    merged_params = {**cfg.strategy.params, **result.best_params}
    for k, v in sorted(merged_params.items()):
        if v is None:
            best_yaml_lines.append(f"    {k}: null")
        elif isinstance(v, bool):
            best_yaml_lines.append(f"    {k}: {'true' if v else 'false'}")
        elif isinstance(v, str):
            best_yaml_lines.append(f"    {k}: \"{v}\"")
        elif isinstance(v, float):
            best_yaml_lines.append(f"    {k}: {v:.4f}".rstrip('0').rstrip('.'))
        else:
            best_yaml_lines.append(f"    {k}: {v}")
    
    yaml_path = report_dir / "best_params.yaml"
    yaml_path.write_text("\n".join(best_yaml_lines) + "\n")
    
    # ── 最終摘要 ──
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"✅ Optimization Complete!")
    print(f"   Method:    {args.method.upper()}")
    print(f"   Trials:    {len(result.all_trials)}")
    print(f"   Best obj:  {result.best_value:.4f}")
    print(f"   Time:      {elapsed:.1f}s ({elapsed/max(args.n_trials,1):.1f}s/trial)")
    print(f"   Results:   {report_dir}")
    print(f"   Best YAML: {yaml_path}")
    print(f"{'='*60}")
    
    # 印出最佳參數（方便直接複製）
    print(f"\n📋 Best Parameters (copy to config YAML):")
    for k, v in sorted(result.best_params.items()):
        if isinstance(v, float):
            print(f"    {k}: {v:.4f}".rstrip('0').rstrip('.'))
        else:
            print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
