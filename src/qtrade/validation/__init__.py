"""
統一驗證模組

整合所有策略驗證功能：
- Walk-Forward Analysis
- Parameter Sensitivity Analysis
- Monte Carlo Simulation
- Cross-Asset Validation (LOAO, Correlation, Market Regime)
- Advanced Methods (DSR, PBO, CPCV)
- Live/Backtest Consistency Validation

使用範例:
    from qtrade.validation import (
        # Walk-Forward
        walk_forward_analysis,
        
        # Sensitivity
        parameter_sensitivity_analysis,
        detect_overfitting,
        
        # Cross-Asset
        leave_one_asset_out,
        correlation_stratified_validation,
        market_regime_validation,
        
        # Advanced (Prado methods)
        deflated_sharpe_ratio,
        combinatorial_purged_cv,
        
        # Consistency
        ConsistencyValidator,
        run_consistency_check,
    )
"""
from __future__ import annotations

# Walk-Forward Analysis
from .walk_forward import (
    walk_forward_analysis,
    walk_forward_summary,
    parameter_sensitivity_analysis,
    detect_overfitting,
)

# Cross-Asset Validation
from .cross_asset import (
    # Config
    CrossAssetValidationConfig,
    CorrelationStratifiedConfig,
    MarketRegimeConfig,
    # Results
    AssetValidationResult,
    CrossAssetValidationResult,
    CorrelationGroupResult,
    MarketRegimeResult,
    # Enums
    ValidationMethod,
    MarketRegimeIndicator,
    RobustnessLevel,
    # Validators
    LeaveOneAssetOutValidator,
    CorrelationStratifiedValidator,
    MarketRegimeValidator,
    ValidationResultAnalyzer,
    # Convenience functions
    leave_one_asset_out,
    correlation_stratified_validation,
    market_regime_validation,
)

# Advanced Methods (López de Prado)
from .prado_methods import (
    deflated_sharpe_ratio,
    combinatorial_purged_cv,
    DeflatedSharpeResult,
    PBOResult,
    CPCVResult,
    run_all_advanced_validation,
)

# Live/Backtest Consistency
from .consistency import (
    ConsistencyValidator,
    ConsistencyReport,
    run_consistency_check,
)

# Backtest Self-Check (回測邏輯自檢)
from .consistency_checker import (
    ConsistencyChecker,
    ConsistencyReport as BacktestCheckReport,
    CheckResult,
    check_strategy_consistency,
)

# Regime Analysis
from .regime_analysis import (
    compute_regime_performance,
    auto_detect_regimes,
    print_regime_report,
    RegimePerformance,
    RegimeAnalysisResult,
)

# Red Flag Detection
from .red_flags import (
    check_red_flags,
    print_red_flags,
    RedFlag,
)

# Data Embargo (OOS Enforcement)
from .embargo import (
    EmbargoConfig,
    TemporalEmbargoConfig,
    SymbolEmbargoConfig,
    EmbargoHoldoutResult,
    EmbargoHoldoutSummary,
    load_embargo_config,
    enforce_temporal_embargo,
    get_research_symbols,
    is_symbol_embargoed,
    get_embargo_only_symbols,
    run_embargo_holdout_test,
    print_embargo_status,
)

__all__ = [
    # Walk-Forward
    "walk_forward_analysis",
    "walk_forward_summary",
    "parameter_sensitivity_analysis",
    "detect_overfitting",
    # Cross-Asset - Config
    "CrossAssetValidationConfig",
    "CorrelationStratifiedConfig",
    "MarketRegimeConfig",
    # Cross-Asset - Results
    "AssetValidationResult",
    "CrossAssetValidationResult",
    "CorrelationGroupResult",
    "MarketRegimeResult",
    # Cross-Asset - Enums
    "ValidationMethod",
    "MarketRegimeIndicator",
    "RobustnessLevel",
    # Cross-Asset - Validators
    "LeaveOneAssetOutValidator",
    "CorrelationStratifiedValidator",
    "MarketRegimeValidator",
    "ValidationResultAnalyzer",
    # Cross-Asset - Convenience
    "leave_one_asset_out",
    "correlation_stratified_validation",
    "market_regime_validation",
    # Advanced (Prado Methods)
    "deflated_sharpe_ratio",
    "combinatorial_purged_cv",
    "DeflatedSharpeResult",
    "PBOResult",
    "CPCVResult",
    "run_all_advanced_validation",
    # Consistency (Live vs Backtest)
    "ConsistencyValidator",
    "ConsistencyReport",
    "run_consistency_check",
    # Backtest Self-Check
    "ConsistencyChecker",
    "BacktestCheckReport",
    "CheckResult",
    "check_strategy_consistency",
    # Regime Analysis
    "compute_regime_performance",
    "auto_detect_regimes",
    "print_regime_report",
    "RegimePerformance",
    "RegimeAnalysisResult",
    # Red Flag Detection
    "check_red_flags",
    "print_red_flags",
    "RedFlag",
    # Data Embargo
    "EmbargoConfig",
    "TemporalEmbargoConfig",
    "SymbolEmbargoConfig",
    "EmbargoHoldoutResult",
    "EmbargoHoldoutSummary",
    "load_embargo_config",
    "enforce_temporal_embargo",
    "get_research_symbols",
    "is_symbol_embargoed",
    "get_embargo_only_symbols",
    "run_embargo_holdout_test",
    "print_embargo_status",
]
