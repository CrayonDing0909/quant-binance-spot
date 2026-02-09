from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from .run_backtest import run_symbol_backtest


def walk_forward_analysis(
    symbol: str,
    data_path: Path,
    cfg: dict,
    n_splits: int = 5,
) -> pd.DataFrame:
    """
    Expanding-window Walk-Forward 驗證

    將數據等分成 (n_splits + 1) 個區間：
      Split 1: train = 區間[0]，           test = 區間[1]
      Split 2: train = 區間[0:1]（累加），  test = 區間[2]
      ...
      Split N: train = 區間[0:N-1]，       test = 區間[N]

    訓練集持續擴大，測試集始終是「未見過」的下一段數據。
    這比固定比例更貼近實際：你用歷史數據訓練，然後在新數據上驗證。
    """
    from ..data.storage import load_klines

    df = load_klines(data_path)
    total_len = len(df)

    # 等分成 n_splits+1 個區間
    n_segments = n_splits + 1
    seg_len = total_len // n_segments
    if seg_len < 500:  # 至少 500 根 bar（1h × 500 ≈ 21 天）
        print(f"  ⚠️  數據太短，每段只有 {seg_len} 根 bar")
        n_segments = max(2, total_len // 500)
        n_splits = n_segments - 1
        seg_len = total_len // n_segments

    results = []

    for i in range(n_splits):
        train_end = seg_len * (i + 1)
        test_start = train_end
        test_end = min(seg_len * (i + 2), total_len)

        if test_end - test_start < 200:
            break

        period_train = f"{df.index[0].strftime('%Y-%m')} → {df.index[train_end-1].strftime('%Y-%m')}"
        period_test = f"{df.index[test_start].strftime('%Y-%m')} → {df.index[test_end-1].strftime('%Y-%m')}"
        print(f"  Split {i+1}/{n_splits}: train {period_train}  |  test {period_test}", end="")

        # 訓練集回測
        train_df = df.iloc[:train_end].copy()
        train_data_path = data_path.parent / f"{symbol}_train_{i}.parquet"
        train_df.to_parquet(train_data_path)

        try:
            train_res = run_symbol_backtest(symbol, train_data_path, cfg, cfg.get("strategy_name"))
            train_stats = train_res["stats"]
        except Exception as e:
            print(f" ❌ train failed: {e}")
            train_data_path.unlink(missing_ok=True)
            continue

        # 測試集回測
        test_df = df.iloc[test_start:test_end].copy()
        test_data_path = data_path.parent / f"{symbol}_test_{i}.parquet"
        test_df.to_parquet(test_data_path)

        try:
            test_res = run_symbol_backtest(symbol, test_data_path, cfg, cfg.get("strategy_name"))
            test_stats = test_res["stats"]
        except Exception as e:
            print(f" ❌ test failed: {e}")
            train_data_path.unlink(missing_ok=True)
            test_data_path.unlink(missing_ok=True)
            continue

        train_ret = train_stats.get("Total Return [%]", 0)
        test_ret = test_stats.get("Total Return [%]", 0)
        train_sharpe = train_stats.get("Sharpe Ratio", 0)
        test_sharpe = test_stats.get("Sharpe Ratio", 0)
        print(f"  → train: {train_ret:+.1f}% (SR {train_sharpe:.2f})"
              f"  test: {test_ret:+.1f}% (SR {test_sharpe:.2f})")

        results.append({
            "split": i + 1,
            "train_period": period_train,
            "test_period": period_test,
            "train_bars": train_end,
            "test_bars": test_end - test_start,
            "train_return": train_ret,
            "test_return": test_ret,
            "train_sharpe": train_sharpe,
            "test_sharpe": test_sharpe,
            "train_dd": train_stats.get("Max Drawdown [%]", 0),
            "test_dd": test_stats.get("Max Drawdown [%]", 0),
        })

        train_data_path.unlink(missing_ok=True)
        test_data_path.unlink(missing_ok=True)

    return pd.DataFrame(results)


def parameter_sensitivity_analysis(
    symbol: str,
    data_path: Path,
    base_cfg: dict,
    param_grid: Dict[str, List],
) -> pd.DataFrame:
    """
    參數敏感性分析 - 檢測過擬合
    """
    import itertools

    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combos = list(itertools.product(*param_values))
    total = len(combos)

    results = []

    for idx, combo in enumerate(combos, 1):
        params = dict(zip(param_names, combo))
        cfg = base_cfg.copy()
        cfg["strategy_params"] = {**base_cfg["strategy_params"], **params}

        try:
            res = run_symbol_backtest(symbol, data_path, cfg, cfg.get("strategy_name"))
            stats = res["stats"]
        except Exception as e:
            print(f"  ⚠️  {params} failed: {e}")
            continue

        result = {name: val for name, val in zip(param_names, combo)}
        result.update({
            "total_return": stats.get("Total Return [%]", 0),
            "sharpe_ratio": stats.get("Sharpe Ratio", 0),
            "max_drawdown": stats.get("Max Drawdown [%]", 0),
            "win_rate": stats.get("Win Rate [%]", 0),
            "total_trades": stats.get("Total Trades", 0),
        })
        results.append(result)

        if idx % 5 == 0 or idx == total:
            print(f"  進度: {idx}/{total} ({idx/total*100:.0f}%)")

    return pd.DataFrame(results)


def detect_overfitting(
    train_metrics: pd.Series,
    test_metrics: pd.Series,
    threshold: float = 0.3,
) -> Dict[str, bool]:
    """檢測過擬合指標"""
    warnings = {}

    train_return = train_metrics.get("Total Return [%]", 0)
    test_return = test_metrics.get("Total Return [%]", 0)
    if train_return > 0:
        return_drop = (train_return - test_return) / abs(train_return)
        warnings["return_drop"] = return_drop > threshold

    train_sharpe = train_metrics.get("Sharpe Ratio", 0)
    test_sharpe = test_metrics.get("Sharpe Ratio", 0)
    if train_sharpe > 0:
        sharpe_drop = (train_sharpe - test_sharpe) / abs(train_sharpe)
        warnings["sharpe_drop"] = sharpe_drop > threshold

    train_dd = abs(train_metrics.get("Max Drawdown [%]", 0))
    test_dd = abs(test_metrics.get("Max Drawdown [%]", 0))
    if train_dd > 0:
        dd_increase = (test_dd - train_dd) / train_dd
        warnings["drawdown_increase"] = dd_increase > threshold

    warnings["overfitting_risk"] = any(warnings.values())

    return warnings
