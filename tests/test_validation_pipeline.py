"""
驗證管線單元測試

覆蓋 prado_methods（DSR, _simplified_pbo_estimate）、walk_forward_summary
以及確認已棄用函數不在公開 API 中。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from qtrade.validation.prado_methods import (
    deflated_sharpe_ratio,
    DeflatedSharpeResult,
    _simplified_pbo_estimate,
    PBOResult,
    expected_max_sharpe,
)


# ══════════════════════════════════════════════════════════════════════════════
# Deflated Sharpe Ratio
# ══════════════════════════════════════════════════════════════════════════════

class TestDeflatedSharpeRatio:
    """DSR 基本正確性"""

    def test_dsr_significant(self):
        """高 Sharpe + 少試驗 → 顯著"""
        result = deflated_sharpe_ratio(
            observed_sharpe=2.0,
            n_trials=10,
            n_observations=5000,
        )
        assert isinstance(result, DeflatedSharpeResult)
        assert result.is_significant == True  # noqa: E712 (may be numpy bool)
        assert result.deflated_sharpe > 0
        assert result.p_value < 0.05

    def test_dsr_not_significant(self):
        """低 Sharpe + 多試驗 → 不顯著"""
        result = deflated_sharpe_ratio(
            observed_sharpe=0.1,
            n_trials=1000,
            n_observations=100,
        )
        assert result.is_significant == False  # noqa: E712

    def test_dsr_requires_min_observations(self):
        """觀察數 < 2 應報錯"""
        with pytest.raises(ValueError, match="至少"):
            deflated_sharpe_ratio(
                observed_sharpe=1.0,
                n_trials=10,
                n_observations=1,
            )

    def test_dsr_with_skewness_kurtosis(self):
        """帶偏度/峰度的 DSR 應正常返回"""
        result = deflated_sharpe_ratio(
            observed_sharpe=1.5,
            n_trials=50,
            n_observations=2000,
            skewness=-0.5,
            kurtosis=5.0,
        )
        assert isinstance(result, DeflatedSharpeResult)
        assert result.variance_of_sharpe > 0


# ══════════════════════════════════════════════════════════════════════════════
# expected_max_sharpe
# ══════════════════════════════════════════════════════════════════════════════

class TestExpectedMaxSharpe:
    def test_single_trial(self):
        assert expected_max_sharpe(1) == 0.0

    def test_more_trials_higher_max(self):
        m10 = expected_max_sharpe(10)
        m100 = expected_max_sharpe(100)
        assert m100 > m10 > 0


# ══════════════════════════════════════════════════════════════════════════════
# Simplified PBO (internal)
# ══════════════════════════════════════════════════════════════════════════════

class TestSimplifiedPBO:
    """_simplified_pbo_estimate 基本正確性"""

    def test_basic_pbo(self):
        """基本 PBO 返回正確結構"""
        is_sharpes = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        oos_sharpes = np.array([0.8, 1.8, 2.8, 3.8, 4.8])
        result = _simplified_pbo_estimate(is_sharpes, oos_sharpes)
        assert isinstance(result, PBOResult)
        assert 0 <= result.pbo <= 1.0
        assert result.n_combinations == 5
        assert result.rank_correlation > 0.9  # 完美正相關

    def test_pbo_inverted_ranks_high(self):
        """IS 與 OOS 排名完全反轉 → 高 PBO"""
        is_sharpes = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        oos_sharpes = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _simplified_pbo_estimate(is_sharpes, oos_sharpes)
        # IS 最佳 (idx=0, SR=5.0) 在 OOS 最差 (1.0) → 高 PBO
        assert result.pbo <= 0.5  # rank = 1/5 = 0.2
        assert result.rank_correlation < 0  # 負相關

    def test_pbo_requires_min_strategies(self):
        with pytest.raises(ValueError, match="至少 2"):
            _simplified_pbo_estimate(np.array([1.0]), np.array([1.0]))

    def test_pbo_length_mismatch(self):
        with pytest.raises(ValueError, match="長度必須相同"):
            _simplified_pbo_estimate(np.array([1.0, 2.0]), np.array([1.0]))


# ══════════════════════════════════════════════════════════════════════════════
# Public API 不再包含 probability_of_backtest_overfitting
# ══════════════════════════════════════════════════════════════════════════════

class TestPBODeprecation:
    """確認 probability_of_backtest_overfitting 已從公開 API 移除"""

    def test_not_in_init(self):
        import qtrade.validation as val
        assert not hasattr(val, "probability_of_backtest_overfitting")

    def test_not_in_all(self):
        import qtrade.validation as val
        assert "probability_of_backtest_overfitting" not in val.__all__

    def test_internal_still_importable(self):
        """內部函數仍可直接 import"""
        from qtrade.validation.prado_methods import _simplified_pbo_estimate
        assert callable(_simplified_pbo_estimate)


# ══════════════════════════════════════════════════════════════════════════════
# Walk-Forward Summary
# ══════════════════════════════════════════════════════════════════════════════

class TestWalkForwardSummary:
    """walk_forward_summary 基本正確性"""

    def test_summary_with_valid_data(self):
        from qtrade.validation import walk_forward_summary

        wf_df = pd.DataFrame({
            "split": [1, 2, 3],
            "train_sharpe": [1.5, 1.6, 1.4],
            "test_sharpe": [1.2, 1.3, 1.1],
            "train_return": [50.0, 55.0, 45.0],
            "test_return": [30.0, 35.0, 25.0],
            "train_dd": [-10.0, -12.0, -8.0],
            "test_dd": [-15.0, -14.0, -13.0],
            "train_bars": [1000, 1000, 1000],
            "test_bars": [500, 500, 500],
        })
        summary = walk_forward_summary(wf_df)
        assert isinstance(summary, dict)
        assert "avg_train_sharpe" in summary
        assert "avg_test_sharpe" in summary
        assert summary["avg_train_sharpe"] > 0


def _load_validate_module():
    repo_root = Path(__file__).resolve().parent.parent
    validate_path = repo_root / "scripts" / "validate.py"
    spec = importlib.util.spec_from_file_location("validate_module", validate_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestValidateAlphaDecayConfig:
    def test_load_validation_config_reads_alpha_decay_section(self, tmp_path):
        validate_module = _load_validate_module()
        cfg_path = tmp_path / "validation.yaml"
        cfg_path.write_text(
            yaml.safe_dump(
                {
                    "alpha_decay": {
                        "enabled": True,
                        "forward_bars": 48,
                        "window_days": 120,
                        "recent_days": 90,
                        "recent_ic_min": 0.03,
                        "max_decay_pct": 0.4,
                        "max_critical_alerts": 1,
                        "min_ic_denominator": 0.02,
                    }
                }
            ),
            encoding="utf-8",
        )

        val_cfg = validate_module.load_validation_config(str(cfg_path))
        assert val_cfg.alpha_decay_enabled is True
        assert val_cfg.alpha_decay_forward_bars == 48
        assert val_cfg.alpha_decay_window_days == 120
        assert val_cfg.alpha_decay_recent_days == 90
        assert val_cfg.alpha_decay_recent_ic_min == 0.03
        assert val_cfg.alpha_decay_max_decay_pct == 0.4
        assert val_cfg.alpha_decay_max_critical_alerts == 1
        assert val_cfg.alpha_decay_min_ic_denominator == 0.02

    def test_load_validation_config_alpha_decay_defaults(self, tmp_path):
        """When alpha_decay section is empty, defaults should match governance spec."""
        validate_module = _load_validate_module()
        cfg_path = tmp_path / "validation.yaml"
        cfg_path.write_text(yaml.safe_dump({}), encoding="utf-8")

        val_cfg = validate_module.load_validation_config(str(cfg_path))
        assert val_cfg.alpha_decay_recent_ic_min == 0.005  # TSMOM-calibrated
        assert val_cfg.alpha_decay_max_decay_pct == 0.6
        assert val_cfg.alpha_decay_max_critical_alerts == 2
        assert val_cfg.alpha_decay_min_ic_denominator == 0.01

    def test_generate_summary_includes_alpha_decay_test(self, tmp_path):
        validate_module = _load_validate_module()
        summary = validate_module.generate_summary(
            walk_forward_results={},
            monte_carlo_results={},
            cross_asset_results={},
            prado_results={},
            kelly_results={},
            report_dir=tmp_path,
            alpha_decay_results={
                "passed": False,
                "avg_recent_ic": 0.01,
                "avg_decay_pct": 0.65,
                "n_critical_alerts": 2,
                "recent_ic_min": 0.02,
                "max_decay_pct": 0.5,
                "max_critical_alerts": 0,
            },
        )

        assert "alpha_decay" in summary["tests"]
        assert summary["tests"]["alpha_decay"]["passed"] is False
