"""
因子正交性分析模組

用於系統性偵測因子冗餘、同一因子穿不同衣服的問題。

核心功能：
1. compute_signal_correlation_matrix — NxN Spearman rank 相關矩陣
2. pca_decomposition — PCA 分解，回傳有效獨立因子數
3. marginal_information_ratio — 候選信號對既有因子的冗餘度
4. check_latent_factor_loading — 候選信號在潛在主成分上的 loading

Usage:
    from qtrade.validation.factor_orthogonality import (
        compute_signal_correlation_matrix,
        pca_decomposition,
        marginal_information_ratio,
        check_latent_factor_loading,
    )

    # 計算相關矩陣
    signals = {"tsmom": tsmom_signal, "htf": htf_signal, "lsr": lsr_signal}
    corr = compute_signal_correlation_matrix(signals)

    # PCA 分解
    pca = pca_decomposition(signals)
    print(f"有效獨立因子數: {pca.n_effective_factors}")

    # 檢查候選因子冗餘度
    result = marginal_information_ratio(candidate_signal, signals)
    if result.r_squared > 0.50:
        print("FAIL: 候選因子 > 50% 冗餘")

References:
    - López de Prado (2020). Machine Learning for Asset Managers (Factor Clustering)
    - Cont et al. (2014). Measuring factor redundancy in financial portfolios
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Dataclasses
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PCAResult:
    """PCA 分解結果"""
    n_effective_factors: int             # 解釋 threshold% 變異所需的主成分數
    explained_variance_ratio: np.ndarray  # 各 PC 的解釋變異比例
    cumulative_variance: np.ndarray      # 累積解釋變異
    loadings: pd.DataFrame               # PC loadings (signals × PCs)
    threshold: float                     # 使用的門檻 (default 0.95)
    n_signals: int                       # 輸入信號數


@dataclass
class MarginalInfoResult:
    """邊際資訊比率結果"""
    r_squared: float                     # 候選信號被既有因子解釋的比例
    residual_ic: float                   # 殘差信號的 IC (vs forward return)
    residual_signal: pd.Series           # 殘差信號序列
    is_redundant: bool                   # R-sq > threshold
    coefficients: Dict[str, float]       # 回歸係數 (各既有因子)
    threshold: float                     # 使用的冗餘門檻 (default 0.50)


@dataclass
class LatentLoadingResult:
    """潛在因子 Loading 結果"""
    pc_loadings: Dict[str, float]        # 候選在各 PC 上的 loading
    variance_explained_by_pcs: float     # 候選被既有 PCs 解釋的變異比例
    max_loading: float                   # 最大單 PC loading
    max_loading_pc: str                  # 最大 loading 對應的 PC
    is_redundant: bool                   # 最大 loading > threshold
    threshold: float                     # 使用的門檻 (default 0.70)


@dataclass
class RedundancyCluster:
    """冗餘叢集"""
    cluster_id: int
    signals: List[str]
    avg_intra_corr: float                # 叢集內平均相關性


@dataclass
class FactorGeometryReport:
    """完整因子幾何審計報告"""
    correlation_matrix: pd.DataFrame
    pca_result: PCAResult
    redundancy_clusters: List[RedundancyCluster]
    n_signals: int
    n_effective_factors: int
    summary: str                         # 人類可讀摘要


# ══════════════════════════════════════════════════════════════════════════════
# 1. Signal Correlation Matrix
# ══════════════════════════════════════════════════════════════════════════════

def compute_signal_correlation_matrix(
    signals: Dict[str, pd.Series],
    method: str = "spearman",
) -> pd.DataFrame:
    """
    計算信號間的 NxN 相關矩陣。

    Args:
        signals: {信號名: 信號序列}，所有序列應已對齊到相同 index
        method: "spearman"（default, rank correlation）或 "pearson"

    Returns:
        NxN 相關矩陣 DataFrame
    """
    if len(signals) < 2:
        logger.warning("需要至少 2 個信號才能計算相關矩陣")
        return pd.DataFrame()

    # 建立 DataFrame，對齊並丟棄缺失
    df = pd.DataFrame(signals).dropna()

    if len(df) < 50:
        logger.warning(f"對齊後僅 {len(df)} 筆資料，太少無法計算可靠的相關性")
        return pd.DataFrame()

    return df.corr(method=method)


# ══════════════════════════════════════════════════════════════════════════════
# 2. PCA Decomposition
# ══════════════════════════════════════════════════════════════════════════════

def pca_decomposition(
    signals: Dict[str, pd.Series],
    threshold: float = 0.95,
) -> PCAResult:
    """
    對信號矩陣做 PCA，回傳有效獨立因子數。

    Args:
        signals: {信號名: 信號序列}
        threshold: 解釋變異門檻（default 0.95 = 95%）

    Returns:
        PCAResult

    Raises:
        ValueError: 信號不足 2 個或數據太少
    """
    if len(signals) < 2:
        raise ValueError("需要至少 2 個信號做 PCA")

    # 建立對齊 DataFrame
    df = pd.DataFrame(signals).dropna()
    if len(df) < 50:
        raise ValueError(f"對齊後僅 {len(df)} 筆資料，需要至少 50 筆")

    signal_names = list(df.columns)
    n_signals = len(signal_names)

    # 標準化
    scaler = StandardScaler()
    X = scaler.fit_transform(df.values)

    # PCA
    pca = PCA()
    pca.fit(X)

    # 計算有效因子數
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    n_effective = int(np.searchsorted(cum_var, threshold) + 1)
    n_effective = min(n_effective, n_signals)

    # Loadings (signals × PCs)
    pc_names = [f"PC{i+1}" for i in range(n_signals)]
    loadings = pd.DataFrame(
        pca.components_.T,
        index=signal_names,
        columns=pc_names,
    )

    return PCAResult(
        n_effective_factors=n_effective,
        explained_variance_ratio=pca.explained_variance_ratio_,
        cumulative_variance=cum_var,
        loadings=loadings,
        threshold=threshold,
        n_signals=n_signals,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3. Marginal Information Ratio
# ══════════════════════════════════════════════════════════════════════════════

def marginal_information_ratio(
    candidate: pd.Series,
    existing_signals: Dict[str, pd.Series],
    forward_returns: Optional[pd.Series] = None,
    r_squared_threshold: float = 0.50,
) -> MarginalInfoResult:
    """
    計算候選信號相對於既有信號的邊際資訊。

    做法：把候選信號對所有既有信號做線性回歸，
    R² 表示候選信號被既有因子解釋的比例。
    殘差即為候選信號中「新」的信息。

    Args:
        candidate: 候選信號序列
        existing_signals: {既有信號名: 序列}
        forward_returns: 前瞻收益（用於計算殘差 IC），可選
        r_squared_threshold: R² 超過此值 → 判定為冗餘 (default 0.50)

    Returns:
        MarginalInfoResult
    """
    if not existing_signals:
        raise ValueError("需要至少 1 個既有信號")

    # 對齊所有序列
    all_data = {"_candidate": candidate}
    all_data.update(existing_signals)
    if forward_returns is not None:
        all_data["_fwd_ret"] = forward_returns

    df = pd.DataFrame(all_data).dropna()

    if len(df) < 50:
        raise ValueError(f"對齊後僅 {len(df)} 筆資料，需要至少 50 筆")

    y = df["_candidate"].values
    existing_names = [k for k in existing_signals.keys()]
    X = df[existing_names].values

    # 標準化
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_x.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    # 線性回歸
    reg = LinearRegression()
    reg.fit(X_scaled, y_scaled)
    r_squared = float(reg.score(X_scaled, y_scaled))

    # 殘差信號
    y_pred = reg.predict(X_scaled)
    residual_scaled = y_scaled - y_pred
    residual = pd.Series(residual_scaled, index=df.index, name="residual")

    # 殘差 IC（若提供前瞻收益）
    residual_ic = 0.0
    if forward_returns is not None and "_fwd_ret" in df.columns:
        fwd = df["_fwd_ret"].values
        valid = ~np.isnan(residual_scaled) & ~np.isnan(fwd)
        if valid.sum() >= 50:
            residual_ic, _ = stats.spearmanr(residual_scaled[valid], fwd[valid])
            residual_ic = float(residual_ic)

    # 回歸係數
    coefficients = {
        name: float(coef)
        for name, coef in zip(existing_names, reg.coef_)
    }

    return MarginalInfoResult(
        r_squared=r_squared,
        residual_ic=residual_ic,
        residual_signal=residual,
        is_redundant=r_squared > r_squared_threshold,
        coefficients=coefficients,
        threshold=r_squared_threshold,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 4. Latent Factor Loading Check
# ══════════════════════════════════════════════════════════════════════════════

def check_latent_factor_loading(
    candidate: pd.Series,
    existing_signals: Dict[str, pd.Series],
    loading_threshold: float = 0.70,
    pca_threshold: float = 0.95,
) -> LatentLoadingResult:
    """
    檢查候選信號在既有信號 PCA 主成分上的 loading。

    即使候選信號與每個既有信號的 pairwise 相關性都不高，
    如果候選在某個 PC 上的 loading 很高，
    說明它捕捉的是同一個潛在因子 — 「同一因子穿了不同衣服」。

    Args:
        candidate: 候選信號序列
        existing_signals: {既有信號名: 序列}
        loading_threshold: PC loading > 此值 → 冗餘 (default 0.70)
        pca_threshold: PCA 解釋變異門檻

    Returns:
        LatentLoadingResult
    """
    if not existing_signals:
        raise ValueError("需要至少 1 個既有信號")

    # 對齊所有序列
    all_data = {"_candidate": candidate}
    all_data.update(existing_signals)
    df = pd.DataFrame(all_data).dropna()

    if len(df) < 50:
        raise ValueError(f"對齊後僅 {len(df)} 筆資料，需要至少 50 筆")

    # 只用既有信號做 PCA
    existing_names = list(existing_signals.keys())
    X_existing = df[existing_names].values
    candidate_values = df["_candidate"].values

    # 標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_existing)
    candidate_scaled = (
        (candidate_values - candidate_values.mean()) / candidate_values.std()
        if candidate_values.std() > 0
        else candidate_values * 0.0
    )

    # 對既有信號做 PCA
    n_components = len(existing_names)
    pca = PCA(n_components=n_components)
    pca_scores = pca.fit_transform(X_scaled)

    # 計算候選在每個 PC 上的投影（loading = correlation）
    pc_loadings = {}
    for i in range(n_components):
        corr, _ = stats.spearmanr(candidate_scaled, pca_scores[:, i])
        pc_loadings[f"PC{i+1}"] = float(corr) if not np.isnan(corr) else 0.0

    # 被 PCs 解釋的變異比例：用回歸估算
    reg = LinearRegression()
    reg.fit(pca_scores, candidate_scaled)
    variance_explained = float(reg.score(pca_scores, candidate_scaled))

    # 最大 loading
    max_pc = max(pc_loadings, key=lambda k: abs(pc_loadings[k]))
    max_loading = abs(pc_loadings[max_pc])

    return LatentLoadingResult(
        pc_loadings=pc_loadings,
        variance_explained_by_pcs=variance_explained,
        max_loading=max_loading,
        max_loading_pc=max_pc,
        is_redundant=max_loading > loading_threshold,
        threshold=loading_threshold,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 5. Factor Geometry Audit（完整審計）
# ══════════════════════════════════════════════════════════════════════════════

def compute_redundancy_clusters(
    correlation_matrix: pd.DataFrame,
    corr_threshold: float = 0.30,
) -> List[RedundancyCluster]:
    """
    用階層式聚類找出冗餘信號叢集。

    Args:
        correlation_matrix: NxN 相關矩陣
        corr_threshold: 相關性門檻（> 此值歸為同一叢集）

    Returns:
        List of RedundancyCluster
    """
    if correlation_matrix.empty or len(correlation_matrix) < 2:
        return []

    signal_names = correlation_matrix.columns.tolist()

    # 將相關矩陣轉為距離矩陣
    dist_matrix = 1.0 - correlation_matrix.abs().values
    # 確保對角線為 0
    np.fill_diagonal(dist_matrix, 0.0)

    # 提取壓縮形式的距離矩陣（上三角）
    n = len(signal_names)
    condensed = []
    for i in range(n):
        for j in range(i + 1, n):
            condensed.append(dist_matrix[i, j])
    condensed = np.array(condensed)

    if len(condensed) == 0:
        return []

    # 階層式聚類
    Z = linkage(condensed, method="average")
    distance_threshold = 1.0 - corr_threshold
    labels = fcluster(Z, t=distance_threshold, criterion="distance")

    # 組織成叢集
    clusters_dict: Dict[int, List[str]] = {}
    for i, label in enumerate(labels):
        clusters_dict.setdefault(int(label), []).append(signal_names[i])

    # 計算每個叢集的內部平均相關性
    clusters = []
    for cid, members in clusters_dict.items():
        if len(members) < 2:
            continue  # 單一成員的叢集不算冗餘
        # 計算叢集內平均相關性
        intra_corrs = []
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                intra_corrs.append(abs(correlation_matrix.loc[members[i], members[j]]))
        avg_corr = float(np.mean(intra_corrs)) if intra_corrs else 0.0

        clusters.append(RedundancyCluster(
            cluster_id=cid,
            signals=members,
            avg_intra_corr=round(avg_corr, 4),
        ))

    # 按叢集大小排序（最大的先）
    clusters.sort(key=lambda c: len(c.signals), reverse=True)
    return clusters


def run_factor_geometry_audit(
    signals: Dict[str, pd.Series],
    pca_threshold: float = 0.95,
    cluster_corr_threshold: float = 0.30,
) -> FactorGeometryReport:
    """
    執行完整的因子幾何審計。

    Args:
        signals: {信號名: 信號序列}
        pca_threshold: PCA 解釋變異門檻
        cluster_corr_threshold: 冗餘叢集相關性門檻

    Returns:
        FactorGeometryReport
    """
    # 1. 相關矩陣
    corr_matrix = compute_signal_correlation_matrix(signals)

    # 2. PCA
    pca_result = pca_decomposition(signals, threshold=pca_threshold)

    # 3. 冗餘叢集
    clusters = compute_redundancy_clusters(corr_matrix, corr_threshold=cluster_corr_threshold)

    # 4. 摘要
    n = len(signals)
    n_eff = pca_result.n_effective_factors
    redundancy_info = ""
    if clusters:
        cluster_descs = []
        for c in clusters:
            cluster_descs.append(
                f"  - 叢集 {c.cluster_id}: {', '.join(c.signals)} "
                f"(avg corr={c.avg_intra_corr:.3f})"
            )
        redundancy_info = "\n冗餘叢集:\n" + "\n".join(cluster_descs)
    else:
        redundancy_info = "\n無冗餘叢集（所有信號相關性 < 門檻）"

    summary = (
        f"因子幾何審計結果\n"
        f"{'=' * 50}\n"
        f"信號數: {n}\n"
        f"有效獨立因子數: {n_eff} (解釋 {pca_threshold*100:.0f}% 變異)\n"
        f"冗餘比: {n - n_eff}/{n} 個信號可能是冗餘的\n"
        f"PC1 解釋變異: {pca_result.explained_variance_ratio[0]*100:.1f}%\n"
        f"{redundancy_info}"
    )

    return FactorGeometryReport(
        correlation_matrix=corr_matrix,
        pca_result=pca_result,
        redundancy_clusters=clusters,
        n_signals=n,
        n_effective_factors=n_eff,
        summary=summary,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Print Helpers
# ══════════════════════════════════════════════════════════════════════════════

def print_factor_geometry_report(report: FactorGeometryReport) -> None:
    """印出因子幾何審計報告"""
    print("\n" + "=" * 72)
    print("  因子幾何審計 (Factor Geometry Audit)")
    print("=" * 72)
    print(report.summary)

    # 相關矩陣
    if not report.correlation_matrix.empty:
        print("\n相關矩陣 (Spearman):")
        print(report.correlation_matrix.round(3).to_string())

    # PCA Loadings (top PCs only)
    pca = report.pca_result
    n_show = min(pca.n_effective_factors + 1, pca.n_signals)
    print(f"\nPCA Loadings (top {n_show} PCs):")
    loadings_show = pca.loadings.iloc[:, :n_show]
    print(loadings_show.round(3).to_string())

    print(f"\n累積解釋變異: {[f'{v:.1%}' for v in pca.cumulative_variance[:n_show]]}")
    print("=" * 72)


def print_marginal_info_result(
    name: str,
    result: MarginalInfoResult,
) -> None:
    """印出邊際資訊比率結果"""
    status = "FAIL (冗餘)" if result.is_redundant else "PASS (正交)"
    print(f"\n{'─' * 50}")
    print(f"  邊際資訊比率: {name}")
    print(f"  R²: {result.r_squared:.4f} (門檻: {result.threshold})")
    print(f"  殘差 IC: {result.residual_ic:.4f}")
    print(f"  判定: {status}")
    if result.coefficients:
        print(f"  回歸係數:")
        for sig_name, coef in result.coefficients.items():
            print(f"    {sig_name}: {coef:.4f}")
    print(f"{'─' * 50}")
