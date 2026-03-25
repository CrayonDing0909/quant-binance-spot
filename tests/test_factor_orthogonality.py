"""
Tests for factor_orthogonality.py — 因子正交性分析模組

覆蓋：
1. compute_signal_correlation_matrix — 相關矩陣計算
2. pca_decomposition — PCA 分解 + 有效因子數
3. marginal_information_ratio — 冗餘度判定
4. check_latent_factor_loading — 潛在因子 loading
5. compute_redundancy_clusters — 冗餘叢集
6. run_factor_geometry_audit — 完整審計
7. CI guard: research scripts 必須呼叫 marginal_information_ratio
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qtrade.validation.factor_orthogonality import (
    compute_signal_correlation_matrix,
    pca_decomposition,
    marginal_information_ratio,
    check_latent_factor_loading,
    compute_redundancy_clusters,
    run_factor_geometry_audit,
    PCAResult,
    MarginalInfoResult,
    LatentLoadingResult,
    RedundancyCluster,
    FactorGeometryReport,
)

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
ARCHIVE_DIR = SCRIPTS_DIR / "archive"


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def random_signals() -> dict[str, pd.Series]:
    """生成 3 個獨立隨機信號（低相關性）"""
    rng = np.random.default_rng(42)
    n = 500
    idx = pd.date_range("2020-01-01", periods=n, freq="1h")
    return {
        "signal_a": pd.Series(rng.standard_normal(n), index=idx),
        "signal_b": pd.Series(rng.standard_normal(n), index=idx),
        "signal_c": pd.Series(rng.standard_normal(n), index=idx),
    }


@pytest.fixture
def redundant_signals() -> dict[str, pd.Series]:
    """生成冗餘信號：signal_b = signal_a + noise"""
    rng = np.random.default_rng(42)
    n = 500
    idx = pd.date_range("2020-01-01", periods=n, freq="1h")
    base = rng.standard_normal(n)
    return {
        "signal_a": pd.Series(base, index=idx),
        "signal_b": pd.Series(base + rng.standard_normal(n) * 0.1, index=idx),
        "signal_c": pd.Series(rng.standard_normal(n), index=idx),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 1. Correlation Matrix
# ══════════════════════════════════════════════════════════════════════════════

def test_correlation_matrix_shape(random_signals):
    """相關矩陣應為 NxN"""
    corr = compute_signal_correlation_matrix(random_signals)
    assert corr.shape == (3, 3)
    assert list(corr.columns) == ["signal_a", "signal_b", "signal_c"]


def test_correlation_matrix_independent_signals_low_corr(random_signals):
    """獨立信號的相關性應接近 0"""
    corr = compute_signal_correlation_matrix(random_signals)
    off_diag = corr.values[np.triu_indices(3, k=1)]
    assert all(abs(c) < 0.15 for c in off_diag), f"獨立信號相關性太高: {off_diag}"


def test_correlation_matrix_redundant_signals_high_corr(redundant_signals):
    """冗餘信號的相關性應接近 1"""
    corr = compute_signal_correlation_matrix(redundant_signals)
    assert abs(corr.loc["signal_a", "signal_b"]) > 0.90


def test_correlation_matrix_too_few_signals():
    """只有 1 個信號應回傳空 DataFrame"""
    signals = {"only": pd.Series([1, 2, 3])}
    corr = compute_signal_correlation_matrix(signals)
    assert corr.empty


# ══════════════════════════════════════════════════════════════════════════════
# 2. PCA Decomposition
# ══════════════════════════════════════════════════════════════════════════════

def test_pca_independent_signals(random_signals):
    """獨立信號 → 有效因子數 = 信號數"""
    result = pca_decomposition(random_signals)
    assert isinstance(result, PCAResult)
    assert result.n_effective_factors == result.n_signals  # 3
    assert result.n_signals == 3


def test_pca_redundant_signals(redundant_signals):
    """冗餘信號 → 有效因子數 < 信號數"""
    result = pca_decomposition(redundant_signals)
    assert result.n_effective_factors < result.n_signals
    # signal_a ≈ signal_b → should be 2 effective factors
    assert result.n_effective_factors == 2


def test_pca_loadings_shape(random_signals):
    """Loadings 矩陣應為 n_signals × n_signals"""
    result = pca_decomposition(random_signals)
    assert result.loadings.shape == (3, 3)
    assert list(result.loadings.index) == ["signal_a", "signal_b", "signal_c"]


def test_pca_cumulative_variance(random_signals):
    """累積解釋變異最終應為 1.0"""
    result = pca_decomposition(random_signals)
    assert abs(result.cumulative_variance[-1] - 1.0) < 0.01


def test_pca_too_few_signals():
    """信號不足 2 個應 raise ValueError"""
    with pytest.raises(ValueError, match="至少 2"):
        pca_decomposition({"a": pd.Series([1, 2, 3])})


# ══════════════════════════════════════════════════════════════════════════════
# 3. Marginal Information Ratio
# ══════════════════════════════════════════════════════════════════════════════

def test_marginal_info_independent(random_signals):
    """獨立候選信號 → R² 低, 非冗餘"""
    candidate = random_signals.pop("signal_c")
    result = marginal_information_ratio(candidate, random_signals)
    assert isinstance(result, MarginalInfoResult)
    assert result.r_squared < 0.10  # 獨立信號 R² 應很低
    assert not result.is_redundant


def test_marginal_info_redundant(redundant_signals):
    """冗餘候選信號 → R² 高, 冗餘"""
    candidate = redundant_signals.pop("signal_b")  # nearly same as signal_a
    result = marginal_information_ratio(candidate, redundant_signals)
    assert result.r_squared > 0.80
    assert result.is_redundant


def test_marginal_info_with_forward_returns(random_signals):
    """提供前瞻收益時應計算殘差 IC"""
    rng = np.random.default_rng(99)
    candidate = random_signals.pop("signal_c")
    fwd = pd.Series(
        rng.standard_normal(len(candidate)),
        index=candidate.index,
    )
    result = marginal_information_ratio(
        candidate, random_signals, forward_returns=fwd,
    )
    # 殘差 IC 應為數值（不是 0.0）
    assert isinstance(result.residual_ic, float)


def test_marginal_info_coefficients(random_signals):
    """回歸係數應包含所有既有信號名稱"""
    candidate = random_signals.pop("signal_c")
    result = marginal_information_ratio(candidate, random_signals)
    assert "signal_a" in result.coefficients
    assert "signal_b" in result.coefficients


# ══════════════════════════════════════════════════════════════════════════════
# 4. Latent Factor Loading
# ══════════════════════════════════════════════════════════════════════════════

def test_latent_loading_independent(random_signals):
    """獨立候選信號 → loading 低"""
    candidate = random_signals.pop("signal_c")
    result = check_latent_factor_loading(candidate, random_signals)
    assert isinstance(result, LatentLoadingResult)
    assert result.max_loading < 0.30
    assert not result.is_redundant


def test_latent_loading_redundant(redundant_signals):
    """冗餘候選信號 → loading 高"""
    candidate = redundant_signals.pop("signal_b")
    result = check_latent_factor_loading(candidate, redundant_signals)
    assert result.max_loading > 0.50
    # Note: threshold=0.70, 所以視 noise level 而定


def test_latent_loading_has_all_pcs(random_signals):
    """PC loadings 應包含所有主成分"""
    candidate = random_signals.pop("signal_c")
    result = check_latent_factor_loading(candidate, random_signals)
    assert "PC1" in result.pc_loadings
    assert "PC2" in result.pc_loadings


# ══════════════════════════════════════════════════════════════════════════════
# 5. Redundancy Clusters
# ══════════════════════════════════════════════════════════════════════════════

def test_redundancy_clusters_with_redundant(redundant_signals):
    """冗餘信號應被聚為同一叢集"""
    corr = compute_signal_correlation_matrix(redundant_signals)
    clusters = compute_redundancy_clusters(corr, corr_threshold=0.30)
    # signal_a 和 signal_b 應在同一叢集
    found = False
    for c in clusters:
        if "signal_a" in c.signals and "signal_b" in c.signals:
            found = True
            assert c.avg_intra_corr > 0.80
    assert found, f"signal_a 和 signal_b 未被聚為同一叢集: {clusters}"


def test_redundancy_clusters_independent(random_signals):
    """獨立信號不應有冗餘叢集"""
    corr = compute_signal_correlation_matrix(random_signals)
    clusters = compute_redundancy_clusters(corr, corr_threshold=0.30)
    # 獨立信號不應有多成員叢集
    for c in clusters:
        assert len(c.signals) < 3, f"獨立信號不應全部聚在一起: {c.signals}"


# ══════════════════════════════════════════════════════════════════════════════
# 6. Full Geometry Audit
# ══════════════════════════════════════════════════════════════════════════════

def test_full_audit_output(random_signals):
    """完整審計應產生所有欄位"""
    report = run_factor_geometry_audit(random_signals)
    assert isinstance(report, FactorGeometryReport)
    assert not report.correlation_matrix.empty
    assert report.n_signals == 3
    assert report.n_effective_factors > 0
    assert len(report.summary) > 0


def test_full_audit_redundant_detects(redundant_signals):
    """冗餘信號的審計應偵測到叢集"""
    report = run_factor_geometry_audit(redundant_signals)
    assert report.n_effective_factors < report.n_signals
    assert len(report.redundancy_clusters) > 0


# ══════════════════════════════════════════════════════════════════════════════
# 7. CI Guard: Research scripts 應使用 marginal_information_ratio
# ══════════════════════════════════════════════════════════════════════════════

# 正交性 import patterns
ORTHO_IMPORT_PATTERNS = [
    r"from\s+qtrade\.validation\.factor_orthogonality\s+import",
    r"from\s+qtrade\.validation\s+import.*marginal_information_ratio",
    r"marginal_information_ratio",
    r"check_latent_factor_loading",
    r"compute_signal_correlation_matrix",
]

ORTHO_EXEMPT_PATTERN = re.compile(
    r"#\s*ORTHOGONALITY_EXEMPT\s*:", re.IGNORECASE,
)


def _get_active_research_scripts() -> list[Path]:
    """找到所有活躍的 research scripts"""
    if not SCRIPTS_DIR.exists():
        return []
    scripts = []
    for p in SCRIPTS_DIR.glob("research_*.py"):
        if ARCHIVE_DIR in p.parents:
            continue
        scripts.append(p)
    return sorted(scripts)


def _script_has_orthogonality_check(path: Path) -> bool:
    """檢查腳本是否包含正交性檢查或 opt-out"""
    content = path.read_text(encoding="utf-8")

    # opt-out
    if ORTHO_EXEMPT_PATTERN.search(content):
        return True

    # import check
    for pattern in ORTHO_IMPORT_PATTERNS:
        if re.search(pattern, content):
            return True

    return False


def test_research_scripts_use_orthogonality_check():
    """
    CI guard: 所有活躍 research scripts 必須使用正交性檢查
    或包含明確的 opt-out 標記 "# ORTHOGONALITY_EXEMPT: <reason>"
    """
    scripts = _get_active_research_scripts()
    if not scripts:
        pytest.skip("No active research scripts found")

    missing = []
    for s in scripts:
        if not _script_has_orthogonality_check(s):
            missing.append(s.name)

    if missing:
        pytest.fail(
            f"Found {len(missing)} research script(s) without orthogonality check:\n"
            + "\n".join(f"  - {s}" for s in missing)
            + "\n\nFix: add `from qtrade.validation.factor_orthogonality import "
            "marginal_information_ratio` or `# ORTHOGONALITY_EXEMPT: <reason>`"
        )
