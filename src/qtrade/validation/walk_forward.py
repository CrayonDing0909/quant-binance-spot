"""
Walk-Forward Analysis & Parameter Sensitivity

提供：
- Expanding-window Walk-Forward 驗證（含正確 warmup）
- 參數敏感性分析
- 過擬合檢測
- Walk-Forward 結果摘要統計
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════
# Walk-Forward Analysis（Expanding Window）
# ══════════════════════════════════════════════════════════════

def walk_forward_analysis(
    symbol: str,
    data_path: Path,
    cfg: dict,
    n_splits: int = 5,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Expanding-window Walk-Forward 驗證（含正確 warmup）

    將數據等分成 (n_splits + 1) 個區間：
      Split 1: train = 區間[0]，           test = 區間[1]
      Split 2: train = 區間[0:1]（累加），  test = 區間[2]
      ...
      Split N: train = 區間[0:N-1]，       test = 區間[N]

    **關鍵修正**：測試集回測時，策略從 bar 0 開始跑（確保指標 warmup），
    但只用 test 區間的交易計算績效（利用 _apply_date_filter）。

    Args:
        symbol: 交易對符號
        data_path: 數據文件路徑
        cfg: 回測配置字典（to_backtest_dict() 的格式）
        n_splits: 分割數量（預設 5，產生 6 個區間）
        data_dir: 數據根目錄（用於載入 funding rate）

    Returns:
        包含每個 split 結果的 DataFrame
    """
    from ..data.storage import load_klines
    from ..backtest.run_backtest import run_symbol_backtest

    df = load_klines(data_path)
    total_len = len(df)

    # Walk-forward 自行管理數據切片，移除 cfg 中的 start/end 避免日期過濾衝突
    wf_cfg = {k: v for k, v in cfg.items() if k not in ("start", "end")}
    strategy_name = wf_cfg.get("strategy_name")

    # 等分成 n_splits+1 個區間
    n_segments = n_splits + 1
    seg_len = total_len // n_segments
    if seg_len < 500:  # 至少 500 根 bar（1h × 500 ≈ 21 天）
        print(f"  ⚠️  數據太短，每段只有 {seg_len} 根 bar，自動縮減 splits")
        n_segments = max(2, total_len // 500)
        n_splits = n_segments - 1
        seg_len = total_len // n_segments

    print(f"\n  📊 Walk-Forward 設定:")
    print(f"     數據量: {total_len:,} bars ({df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')})")
    print(f"     Splits: {n_splits} (每段 ~{seg_len:,} bars ≈ {seg_len/24:.0f} 天)")
    print()

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

        # ── Train 回測 ──────────────────────────────────
        # 數據：[0, train_end)，無日期過濾
        train_df = df.iloc[:train_end].copy()
        train_tmp = data_path.parent / f"_wf_{symbol}_train_{i}.parquet"
        train_df.to_parquet(train_tmp)

        try:
            train_res = run_symbol_backtest(
                symbol, train_tmp, wf_cfg, strategy_name,
                data_dir=data_dir,
            )
            train_stats = train_res.stats
        except Exception as e:
            print(f" ❌ train failed: {e}")
            train_tmp.unlink(missing_ok=True)
            continue

        # ── Test 回測（含 warmup）────────────────────────
        # 數據：[0, test_end)（包含 train 區間作為 warmup）
        # 日期過濾：只取 [test_start, test_end) 的績效
        #
        # 這樣做的好處：
        # 1. 策略從 bar 0 開始跑，指標有完整 warmup
        # 2. _apply_date_filter 只截取 OOS 區間的交易績效
        # 3. 不會有 look-ahead bias（策略只看到 test_end 之前的數據）
        test_full_df = df.iloc[:test_end].copy()
        test_tmp = data_path.parent / f"_wf_{symbol}_test_{i}.parquet"
        test_full_df.to_parquet(test_tmp)

        test_cfg = {**wf_cfg}
        test_cfg["start"] = str(df.index[test_start])
        test_cfg["end"] = str(df.index[test_end - 1])

        try:
            test_res = run_symbol_backtest(
                symbol, test_tmp, test_cfg, strategy_name,
                data_dir=data_dir,
            )
            test_stats = test_res.stats
        except Exception as e:
            print(f" ❌ test failed: {e}")
            train_tmp.unlink(missing_ok=True)
            test_tmp.unlink(missing_ok=True)
            continue

        # 如果有 adjusted_stats（含 funding 成本），優先使用
        # BacktestResult 是 dataclass，不能用 .get()
        train_adj = getattr(train_res, "adjusted_stats", None)
        test_adj = getattr(test_res, "adjusted_stats", None)

        # adjusted_stats 是 dict；stats 是 pd.Series — 兩者都支援 .get()
        train_ret = (train_adj or train_stats).get("Total Return [%]", 0)
        test_ret = (test_adj or test_stats).get("Total Return [%]", 0)
        train_sharpe = (train_adj or train_stats).get("Sharpe Ratio", 0)
        test_sharpe = (test_adj or test_stats).get("Sharpe Ratio", 0)
        train_mdd = (train_adj or train_stats).get("Max Drawdown [%]", 0)
        test_mdd = (test_adj or test_stats).get("Max Drawdown [%]", 0)
        train_trades = train_stats.get("Total Trades", 0)
        test_trades = test_stats.get("Total Trades", 0)

        cost_tag = " 💰" if train_adj or test_adj else ""
        print(f"  → train: {train_ret:+.1f}% (SR {train_sharpe:.2f})"
              f"  test: {test_ret:+.1f}% (SR {test_sharpe:.2f}){cost_tag}")

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
            "train_dd": train_mdd,
            "test_dd": test_mdd,
            "train_trades": train_trades,
            "test_trades": test_trades,
        })

        # 清理臨時文件
        train_tmp.unlink(missing_ok=True)
        test_tmp.unlink(missing_ok=True)

    return pd.DataFrame(results)


def walk_forward_summary(wf_results: pd.DataFrame) -> dict:
    """
    Walk-Forward 結果摘要統計

    計算 Sharpe 衰退率、OOS 一致性、年間穩定性等。

    Args:
        wf_results: walk_forward_analysis 返回的 DataFrame

    Returns:
        {
            "n_splits": int,
            "avg_train_sharpe": float,
            "avg_test_sharpe": float,
            "std_test_sharpe": float,
            "sharpe_degradation_pct": float,  # Sharpe 衰退率 (%)
            "oos_positive_pct": float,         # 測試集 Sharpe > 0 的比例
            "oos_profitable_pct": float,       # 測試集 Return > 0 的比例
            "worst_test_sharpe": float,
            "best_test_sharpe": float,
            "avg_test_return": float,
            "avg_test_dd": float,
            "is_robust": bool,                 # Sharpe 衰退 < 30% 且 OOS 全部 > 0
            "summary_text": str,               # 人類可讀摘要
        }
    """
    if wf_results.empty:
        return {"n_splits": 0, "summary_text": "❌ 沒有成功的 Walk-Forward split"}

    n = len(wf_results)
    avg_train_sr = wf_results["train_sharpe"].mean()
    avg_test_sr = wf_results["test_sharpe"].mean()
    std_test_sr = wf_results["test_sharpe"].std()

    # Sharpe 衰退率
    if abs(avg_train_sr) > 0.01:
        degradation_pct = (avg_train_sr - avg_test_sr) / abs(avg_train_sr) * 100
    else:
        degradation_pct = 0.0

    # OOS 一致性
    oos_positive = (wf_results["test_sharpe"] > 0).mean() * 100
    oos_profitable = (wf_results["test_return"] > 0).mean() * 100

    # 穩健性判斷
    is_robust = degradation_pct < 30 and oos_positive == 100

    # 構建摘要文字
    lines = [
        f"  === 過擬合風險評估 ===",
        f"  平均 Train Sharpe:  {avg_train_sr:.2f}",
        f"  平均 Test Sharpe:   {avg_test_sr:.2f} (±{std_test_sr:.2f})",
        f"  Sharpe 衰退率:      {degradation_pct:.1f}%",
        f"  OOS Sharpe > 0:     {oos_positive:.0f}% ({int(oos_positive/100*n)}/{n} splits)",
        f"  OOS 盈利:           {oos_profitable:.0f}% ({int(oos_profitable/100*n)}/{n} splits)",
        f"  最差 Test Sharpe:   {wf_results['test_sharpe'].min():.2f}",
        f"  最佳 Test Sharpe:   {wf_results['test_sharpe'].max():.2f}",
        f"  Test Sharpe 範圍:   {wf_results['test_sharpe'].max() - wf_results['test_sharpe'].min():.2f}x",
        "",
    ]

    if is_robust:
        lines.append(f"  ✅ 通過：衰退 < 30% 且 OOS 全正 → 低過擬合風險")
    elif degradation_pct < 50 and oos_positive >= 80:
        lines.append(f"  ⚠️  中度風險：衰退 {degradation_pct:.0f}%，建議進一步用 CPCV 驗證")
    else:
        lines.append(f"  ❌ 高風險：衰退 {degradation_pct:.0f}%，OOS 正向 {oos_positive:.0f}% → 可能過擬合")

    return {
        "n_splits": n,
        "avg_train_sharpe": avg_train_sr,
        "avg_test_sharpe": avg_test_sr,
        "std_test_sharpe": std_test_sr,
        "sharpe_degradation_pct": degradation_pct,
        "oos_positive_pct": oos_positive,
        "oos_profitable_pct": oos_profitable,
        "worst_test_sharpe": wf_results["test_sharpe"].min(),
        "best_test_sharpe": wf_results["test_sharpe"].max(),
        "avg_test_return": wf_results["test_return"].mean(),
        "avg_test_dd": wf_results["test_dd"].mean(),
        "is_robust": is_robust,
        "summary_text": "\n".join(lines),
    }


# ══════════════════════════════════════════════════════════════
# 參數敏感性分析
# ══════════════════════════════════════════════════════════════

def parameter_sensitivity_analysis(
    symbol: str,
    data_path: Path,
    base_cfg: dict,
    param_grid: Dict[str, List],
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """
    參數敏感性分析 - 檢測過擬合
    
    測試不同參數組合的表現，檢查策略對參數變化的敏感度。
    高敏感度可能表示過擬合。
    
    Args:
        symbol: 交易對符號
        data_path: 數據文件路徑
        base_cfg: 基礎配置
        param_grid: 參數網格 {參數名: [值列表]}
        data_dir: 數據根目錄（用於載入 funding rate）
        
    Returns:
        包含所有參數組合結果的 DataFrame
    """
    import itertools
    from ..backtest.run_backtest import run_symbol_backtest

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
            res = run_symbol_backtest(
                symbol, data_path, cfg, cfg.get("strategy_name"),
                data_dir=data_dir,
            )
            stats = res.stats
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


# ══════════════════════════════════════════════════════════════
# 過擬合檢測
# ══════════════════════════════════════════════════════════════

def detect_overfitting(
    train_metrics: pd.Series,
    test_metrics: pd.Series,
    threshold: float = 0.3,
) -> Dict[str, bool]:
    """
    檢測過擬合指標
    
    比較訓練集和測試集的績效指標，檢測是否存在過擬合。
    
    Args:
        train_metrics: 訓練集績效指標
        test_metrics: 測試集績效指標
        threshold: 衰退閾值（超過此比例視為過擬合）
        
    Returns:
        包含各項過擬合檢測結果的字典
    """
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
