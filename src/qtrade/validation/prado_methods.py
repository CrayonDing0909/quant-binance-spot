"""
López de Prado 進階過擬合驗證方法

實現頂級驗證方法：
1. Deflated Sharpe Ratio (DSR) - 校正多重測試偏差
2. Probability of Backtest Overfitting (PBO) - 過擬合機率估計
3. Combinatorial Purged Cross-Validation (CPCV) - 組合式交叉驗證

References:
- López de Prado, M. (2018). Advances in Financial Machine Learning
- Bailey, D. H., & López de Prado, M. (2014). The Deflated Sharpe Ratio
- Bailey et al. (2017). Probability of Backtest Overfitting
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import comb


# ══════════════════════════════════════════════════════════════════════════════
# 1. Deflated Sharpe Ratio (DSR)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DeflatedSharpeResult:
    """Deflated Sharpe Ratio 結果"""
    observed_sharpe: float          # 觀察到的 Sharpe
    deflated_sharpe: float          # 校正後的 Sharpe
    p_value: float                  # p-value
    is_significant: bool            # 是否統計顯著
    expected_max_sharpe: float      # 預期最大 Sharpe（在隨機試驗下）
    variance_of_sharpe: float       # Sharpe 的變異數
    n_trials: int                   # 測試次數
    confidence_level: float         # 信賴水準


def expected_max_sharpe(n_trials: int, variance: float = 1.0) -> float:
    """
    計算在 n 次獨立試驗下，預期的最大 Sharpe Ratio
    
    基於 order statistics 的期望值
    E[max(Z_1, ..., Z_n)] ≈ (1 - γ) * Φ^(-1)(1 - 1/n) + γ * Φ^(-1)(1 - 1/(n*e))
    
    簡化近似：E[max] ≈ sqrt(2 * log(n)) for large n
    """
    if n_trials <= 1:
        return 0.0
    
    # 更精確的近似（Bailey & López de Prado, 2014）
    euler_gamma = 0.5772156649  # Euler-Mascheroni 常數
    z = stats.norm.ppf(1 - 1 / n_trials)
    expected_max = z * np.sqrt(variance) - euler_gamma / (z * np.sqrt(variance)) if z > 0 else 0
    
    return max(expected_max, 0)


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    n_observations: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    confidence_level: float = 0.95,
) -> DeflatedSharpeResult:
    """
    計算 Deflated Sharpe Ratio
    
    當你測試多個策略/參數組合時，最好的那個可能只是運氣好。
    DSR 校正這種多重測試偏差。
    
    Args:
        observed_sharpe: 觀察到的 Sharpe Ratio（年化）
        n_trials: 測試的策略/參數組合數量
        n_observations: 收益觀察數量（交易次數或時間週期數）
        skewness: 收益的偏度
        kurtosis: 收益的峰度（正態分布 = 3）
        confidence_level: 信賴水準
    
    Returns:
        DeflatedSharpeResult
    
    Reference:
        Bailey & López de Prado (2014) "The Deflated Sharpe Ratio"
    """
    if n_observations < 2:
        raise ValueError("需要至少 2 個觀察值")
    
    # Sharpe Ratio 的標準誤（考慮非正態性）
    # Var(SR) ≈ (1 + 0.5*SR² - γ₃*SR + (γ₄-3)/4*SR²) / T
    # 其中 γ₃ = skewness, γ₄ = kurtosis
    sr = observed_sharpe
    var_sr = (
        1 
        + 0.5 * sr**2 
        - skewness * sr 
        + ((kurtosis - 3) / 4) * sr**2
    ) / n_observations
    
    std_sr = np.sqrt(max(var_sr, 1e-10))
    
    # 預期最大 Sharpe（在隨機試驗下）
    exp_max_sr = expected_max_sharpe(n_trials, variance=var_sr)
    
    # Deflated Sharpe = 觀察值 - 預期最大值
    deflated_sr = observed_sharpe - exp_max_sr
    
    # p-value: P(SR > observed | H0: true SR = 0)
    # 使用 t 分布近似
    t_stat = observed_sharpe / std_sr
    p_value = 1 - stats.t.cdf(t_stat, df=n_observations - 1)
    
    # 是否顯著
    is_significant = deflated_sr > 0 and p_value < (1 - confidence_level)
    
    return DeflatedSharpeResult(
        observed_sharpe=observed_sharpe,
        deflated_sharpe=deflated_sr,
        p_value=p_value,
        is_significant=is_significant,
        expected_max_sharpe=exp_max_sr,
        variance_of_sharpe=var_sr,
        n_trials=n_trials,
        confidence_level=confidence_level,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 2. Probability of Backtest Overfitting (PBO)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PBOResult:
    """Probability of Backtest Overfitting 結果"""
    pbo: float                      # 過擬合機率 [0, 1]
    logits: np.ndarray             # 每個組合的 logit 值
    performance_degradation: float  # 平均績效衰退
    rank_correlation: float         # In-sample vs OOS 排名相關性
    n_combinations: int             # 測試的組合數
    is_likely_overfitted: bool      # 是否可能過擬合


def probability_of_backtest_overfitting(
    in_sample_sharpes: np.ndarray,
    out_of_sample_sharpes: np.ndarray,
    threshold: float = 0.5,
) -> PBOResult:
    """
    計算 Probability of Backtest Overfitting (PBO)
    
    PBO 衡量的是：在 in-sample 選出的最佳策略，
    在 out-of-sample 表現低於中位數的機率。
    
    Args:
        in_sample_sharpes: 各策略的 in-sample Sharpe 陣列
        out_of_sample_sharpes: 對應的 out-of-sample Sharpe 陣列
        threshold: PBO 閾值（> threshold 視為過擬合）
    
    Returns:
        PBOResult
    
    Reference:
        Bailey et al. (2017) "Probability of Backtest Overfitting"
    """
    n = len(in_sample_sharpes)
    if n < 2:
        raise ValueError("需要至少 2 個策略比較")
    
    if len(out_of_sample_sharpes) != n:
        raise ValueError("in_sample 和 out_of_sample 長度必須相同")
    
    # 找出 in-sample 最佳策略
    best_is_idx = np.argmax(in_sample_sharpes)
    best_is_sharpe = in_sample_sharpes[best_is_idx]
    best_oos_sharpe = out_of_sample_sharpes[best_is_idx]
    
    # 計算 OOS 中位數
    oos_median = np.median(out_of_sample_sharpes)
    
    # Logit: log(rank_oos / (n - rank_oos))
    # 其中 rank_oos 是最佳 IS 策略在 OOS 的排名
    oos_rank = np.sum(out_of_sample_sharpes <= best_oos_sharpe)
    
    # 避免除以零
    if oos_rank == 0:
        logit = -np.inf
    elif oos_rank == n:
        logit = np.inf
    else:
        logit = np.log(oos_rank / (n - oos_rank))
    
    # PBO = 最佳 IS 策略在 OOS 低於中位數的機率
    # 簡化估計：使用排名比例
    pbo = oos_rank / n
    
    # 績效衰退
    perf_degradation = (best_is_sharpe - best_oos_sharpe) / max(abs(best_is_sharpe), 0.01)
    
    # 排名相關性（Spearman）
    from scipy.stats import spearmanr
    rank_corr, _ = spearmanr(in_sample_sharpes, out_of_sample_sharpes)
    
    return PBOResult(
        pbo=pbo,
        logits=np.array([logit]),
        performance_degradation=perf_degradation,
        rank_correlation=rank_corr,
        n_combinations=n,
        is_likely_overfitted=pbo > threshold,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3. Combinatorial Purged Cross-Validation (CPCV)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CPCVResult:
    """CPCV 結果"""
    mean_train_sharpe: float
    mean_test_sharpe: float
    std_train_sharpe: float
    std_test_sharpe: float
    sharpe_degradation: float       # 平均 Sharpe 衰退
    all_train_sharpes: List[float]
    all_test_sharpes: List[float]
    n_combinations: int
    pbo: float                      # 基於 CPCV 的 PBO
    is_robust: bool


def combinatorial_purged_cv(
    df: pd.DataFrame,
    cfg: dict,
    strategy_name: str,
    n_splits: int = 6,
    n_test_splits: int = 2,
    purge_pct: float = 0.01,
    embargo_pct: float = 0.01,
    symbol: str = "BTCUSDT",
    data_path: Optional[Path] = None,
) -> CPCVResult:
    """
    Combinatorial Purged Cross-Validation (CPCV)
    
    比傳統 K-Fold 更嚴格的交叉驗證：
    1. 組合式選擇 train/test（不只是連續區間）
    2. Purge: 移除 train 尾部的數據，避免 lookahead bias
    3. Embargo: 移除 test 開頭的數據，確保時間間隔
    
    Args:
        df: OHLCV DataFrame
        cfg: 回測配置
        strategy_name: 策略名稱
        n_splits: 將數據分成幾個區間
        n_test_splits: 每次使用幾個區間作為測試集
        purge_pct: Purge 比例
        embargo_pct: Embargo 比例
        symbol: 交易對
        data_path: 臨時數據保存路徑
    
    Returns:
        CPCVResult
    
    Reference:
        López de Prado (2018) Chapter 7
    """
    from ..backtest.run_backtest import run_symbol_backtest
    
    n = len(df)
    split_size = n // n_splits
    
    # 計算 purge 和 embargo 的 bar 數
    purge_bars = max(1, int(split_size * purge_pct))
    embargo_bars = max(1, int(split_size * embargo_pct))
    
    # 生成所有可能的測試集組合
    # C(n_splits, n_test_splits) 個組合
    test_combinations = list(combinations(range(n_splits), n_test_splits))
    n_combinations = len(test_combinations)
    
    print(f"  CPCV: {n_splits} splits, {n_test_splits} test splits")
    print(f"  總組合數: {n_combinations}")
    print(f"  Purge: {purge_bars} bars, Embargo: {embargo_bars} bars")
    
    train_sharpes = []
    test_sharpes = []
    
    # 臨時文件路徑
    if data_path is None:
        data_path = Path("./temp_cpcv")
    data_path.mkdir(parents=True, exist_ok=True)
    
    for i, test_indices in enumerate(test_combinations):
        # 確定 train 和 test 的區間
        train_indices = [j for j in range(n_splits) if j not in test_indices]
        
        # 收集數據
        train_dfs = []
        test_dfs = []
        
        for j in range(n_splits):
            start_idx = j * split_size
            end_idx = (j + 1) * split_size if j < n_splits - 1 else n
            segment = df.iloc[start_idx:end_idx].copy()
            
            if j in test_indices:
                # 測試集：應用 embargo（移除開頭）
                if embargo_bars < len(segment):
                    segment = segment.iloc[embargo_bars:]
                test_dfs.append(segment)
            else:
                # 訓練集：應用 purge（移除尾部）
                if purge_bars < len(segment):
                    segment = segment.iloc[:-purge_bars]
                train_dfs.append(segment)
        
        if not train_dfs or not test_dfs:
            continue
        
        train_df = pd.concat(train_dfs).sort_index()
        test_df = pd.concat(test_dfs).sort_index()
        
        if len(train_df) < 200 or len(test_df) < 100:
            continue
        
        # 保存臨時文件並回測
        train_path = data_path / f"train_{i}.parquet"
        test_path = data_path / f"test_{i}.parquet"
        
        try:
            train_df.to_parquet(train_path)
            test_df.to_parquet(test_path)
            
            # 回測
            train_res = run_symbol_backtest(symbol, train_path, cfg, strategy_name)
            test_res = run_symbol_backtest(symbol, test_path, cfg, strategy_name)
            
            train_sharpe = train_res["stats"].get("Sharpe Ratio", 0)
            test_sharpe = test_res["stats"].get("Sharpe Ratio", 0)
            
            train_sharpes.append(train_sharpe)
            test_sharpes.append(test_sharpe)
            
            if (i + 1) % 5 == 0 or i == n_combinations - 1:
                print(f"  進度: {i+1}/{n_combinations} | "
                      f"Train SR: {np.mean(train_sharpes):.2f} | "
                      f"Test SR: {np.mean(test_sharpes):.2f}")
        
        except Exception as e:
            print(f"  ⚠️  組合 {i} 失敗: {e}")
        
        finally:
            train_path.unlink(missing_ok=True)
            test_path.unlink(missing_ok=True)
    
    # 清理臨時目錄
    try:
        data_path.rmdir()
    except:
        pass
    
    if not train_sharpes:
        raise ValueError("沒有成功的 CPCV 組合")
    
    # 計算統計
    mean_train = np.mean(train_sharpes)
    mean_test = np.mean(test_sharpes)
    std_train = np.std(train_sharpes)
    std_test = np.std(test_sharpes)
    
    # Sharpe 衰退
    degradation = (mean_train - mean_test) / max(abs(mean_train), 0.01)
    
    # 基於 CPCV 的 PBO
    pbo_result = probability_of_backtest_overfitting(
        np.array(train_sharpes),
        np.array(test_sharpes),
    )
    
    return CPCVResult(
        mean_train_sharpe=mean_train,
        mean_test_sharpe=mean_test,
        std_train_sharpe=std_train,
        std_test_sharpe=std_test,
        sharpe_degradation=degradation,
        all_train_sharpes=train_sharpes,
        all_test_sharpes=test_sharpes,
        n_combinations=len(train_sharpes),
        pbo=pbo_result.pbo,
        is_robust=degradation < 0.5 and pbo_result.pbo < 0.5,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 便捷函數
# ══════════════════════════════════════════════════════════════════════════════

def run_all_advanced_validation(
    returns: pd.Series,
    sharpe_ratio: float,
    n_trials: int = 100,
    in_sample_sharpes: Optional[np.ndarray] = None,
    out_of_sample_sharpes: Optional[np.ndarray] = None,
) -> Dict:
    """
    執行所有進階驗證
    
    Args:
        returns: 策略收益率序列
        sharpe_ratio: 觀察到的 Sharpe Ratio
        n_trials: 測試的策略/參數組合數量
        in_sample_sharpes: IS Sharpe 陣列（用於 PBO）
        out_of_sample_sharpes: OOS Sharpe 陣列（用於 PBO）
    
    Returns:
        包含所有驗證結果的字典
    """
    results = {}
    
    # 1. Deflated Sharpe Ratio
    skewness = returns.skew() if len(returns) > 2 else 0
    kurtosis = returns.kurtosis() + 3 if len(returns) > 3 else 3  # scipy kurtosis is excess
    
    dsr_result = deflated_sharpe_ratio(
        observed_sharpe=sharpe_ratio,
        n_trials=n_trials,
        n_observations=len(returns),
        skewness=skewness,
        kurtosis=kurtosis,
    )
    results["deflated_sharpe"] = dsr_result
    
    # 2. PBO (if data provided)
    if in_sample_sharpes is not None and out_of_sample_sharpes is not None:
        pbo_result = probability_of_backtest_overfitting(
            in_sample_sharpes,
            out_of_sample_sharpes,
        )
        results["pbo"] = pbo_result
    
    return results
