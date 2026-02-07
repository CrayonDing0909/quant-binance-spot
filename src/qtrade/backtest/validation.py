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
    train_ratio: float = 0.7,
    n_splits: int = 5,
) -> pd.DataFrame:
    """
    滚动窗口验证 - 检测过拟合
    
    将数据分成多个训练/测试窗口，观察策略在样本外表现
    
    Args:
        symbol: 交易对符号
        data_path: 数据路径
        cfg: 回测配置
        train_ratio: 训练集比例
        n_splits: 分割数量
    
    Returns:
        包含每个窗口结果的 DataFrame
    """
    from ..data.storage import load_klines
    
    df = load_klines(data_path)
    total_len = len(df)
    train_len = int(total_len * train_ratio)
    test_len = total_len - train_len
    
    results = []
    
    for i in range(n_splits):
        # 计算窗口位置
        start_idx = int(i * (total_len - train_len - test_len) / max(1, n_splits - 1))
        train_end = start_idx + train_len
        test_end = min(train_end + test_len, total_len)
        
        if test_end - train_end < test_len * 0.5:  # 测试集太小，跳过
            break
        
        # 训练集回测
        train_df = df.iloc[start_idx:train_end].copy()
        train_data_path = data_path.parent / f"{symbol}_train_{i}.parquet"
        train_df.to_parquet(train_data_path)
        
        train_res = run_symbol_backtest(symbol, train_data_path, cfg, cfg.get("strategy_name"))
        train_stats = train_res["stats"]
        
        # 测试集回测（样本外）
        test_df = df.iloc[train_end:test_end].copy()
        test_data_path = data_path.parent / f"{symbol}_test_{i}.parquet"
        test_df.to_parquet(test_data_path)
        
        test_res = run_symbol_backtest(symbol, test_data_path, cfg, cfg.get("strategy_name"))
        test_stats = test_res["stats"]
        
        results.append({
            "split": i,
            "train_start": train_df.index[0],
            "train_end": train_df.index[-1],
            "test_start": test_df.index[0],
            "test_end": test_df.index[-1],
            "train_return": train_stats.get("Total Return [%]", 0),
            "test_return": test_stats.get("Total Return [%]", 0),
            "train_sharpe": train_stats.get("Sharpe Ratio", 0),
            "test_sharpe": test_stats.get("Sharpe Ratio", 0),
            "train_dd": train_stats.get("Max Drawdown [%]", 0),
            "test_dd": test_stats.get("Max Drawdown [%]", 0),
        })
        
        # 清理临时文件
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
    参数敏感性分析 - 检测过拟合
    
    测试不同参数组合，观察策略稳定性
    
    Args:
        symbol: 交易对符号
        data_path: 数据路径
        base_cfg: 基础回测配置
        param_grid: 参数网格，例如 {"fast": [10, 20, 30], "slow": [50, 60, 70]}
    
    Returns:
        包含所有参数组合结果的 DataFrame
    """
    import itertools
    
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    results = []
    
    for combo in itertools.product(*param_values):
        params = dict(zip(param_names, combo))
        cfg = base_cfg.copy()
        cfg["strategy_params"] = {**base_cfg["strategy_params"], **params}
        
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
    
    return pd.DataFrame(results)


def detect_overfitting(
    train_metrics: pd.Series,
    test_metrics: pd.Series,
    threshold: float = 0.3,
) -> Dict[str, bool]:
    """
    检测过拟合指标
    
    Args:
        train_metrics: 训练集指标
        test_metrics: 测试集指标
        threshold: 性能下降阈值（30%）
    
    Returns:
        过拟合检测结果
    """
    warnings = {}
    
    # 检查收益率下降
    train_return = train_metrics.get("Total Return [%]", 0)
    test_return = test_metrics.get("Total Return [%]", 0)
    if train_return > 0:
        return_drop = (train_return - test_return) / abs(train_return)
        warnings["return_drop"] = return_drop > threshold
    
    # 检查夏普比率下降
    train_sharpe = train_metrics.get("Sharpe Ratio", 0)
    test_sharpe = test_metrics.get("Sharpe Ratio", 0)
    if train_sharpe > 0:
        sharpe_drop = (train_sharpe - test_sharpe) / abs(train_sharpe)
        warnings["sharpe_drop"] = sharpe_drop > threshold
    
    # 检查回撤增加
    train_dd = abs(train_metrics.get("Max Drawdown [%]", 0))
    test_dd = abs(test_metrics.get("Max Drawdown [%]", 0))
    if train_dd > 0:
        dd_increase = (test_dd - train_dd) / train_dd
        warnings["drawdown_increase"] = dd_increase > threshold
    
    warnings["overfitting_risk"] = any(warnings.values())
    
    return warnings

