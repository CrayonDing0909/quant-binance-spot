"""
回測模組

提供完整的回測功能：
- 單一資產回測
- 回測指標計算
- 策略驗證（Walk-Forward, 參數敏感性）
- Cross-Asset 驗證

注意: 驗證功能已移至 qtrade.validation 模組
此處的 import 保留向後相容性
"""
from __future__ import annotations

from .run_backtest import run_symbol_backtest
from .metrics import (
    pretty_stats,
    benchmark_buy_and_hold,
    full_report,
    trade_analysis,
    trade_summary,
)
from .plotting import plot_equity_curve
from .hyperopt_engine import (
    HyperoptEngine,
    ParamSpace,
    ParamDef,
    OptimizationResult,
    WalkForwardValidator,
    RSI_ADX_ATR_PARAM_SPACE,
    EMA_CROSS_PARAM_SPACE,
    OBJECTIVES,
)

# 向後相容: 從新的 validation 模組導入
# 建議直接使用 from qtrade.validation import ...
from ..validation import (
    # Walk-Forward
    walk_forward_analysis,
    parameter_sensitivity_analysis,
    detect_overfitting,
    # Cross-Asset Validation - Config
    CrossAssetValidationConfig,
    CorrelationStratifiedConfig,
    MarketRegimeConfig,
    # Cross-Asset Validation - Results
    AssetValidationResult,
    CrossAssetValidationResult,
    CorrelationGroupResult,
    MarketRegimeResult,
    # Cross-Asset Validation - Enums
    ValidationMethod,
    MarketRegimeIndicator,
    RobustnessLevel,
    # Cross-Asset Validation - Validators
    LeaveOneAssetOutValidator,
    CorrelationStratifiedValidator,
    MarketRegimeValidator,
    ValidationResultAnalyzer,
    # Cross-Asset Validation - Convenience
    leave_one_asset_out,
    correlation_stratified_validation,
    market_regime_validation,
)

__all__ = [
    # Core
    "run_symbol_backtest",
    # Metrics
    "pretty_stats",
    "benchmark_buy_and_hold",
    "full_report",
    "trade_analysis",
    "trade_summary",
    # Plotting
    "plot_equity_curve",
    # Hyperopt
    "HyperoptEngine",
    "ParamSpace",
    "ParamDef",
    "OptimizationResult",
    "WalkForwardValidator",
    "RSI_ADX_ATR_PARAM_SPACE",
    "EMA_CROSS_PARAM_SPACE",
    "OBJECTIVES",
    # Validation (backwards compatible, prefer qtrade.validation)
    "walk_forward_analysis",
    "parameter_sensitivity_analysis",
    "detect_overfitting",
    # Cross-Asset Validation - Config
    "CrossAssetValidationConfig",
    "CorrelationStratifiedConfig",
    "MarketRegimeConfig",
    # Cross-Asset Validation - Results
    "AssetValidationResult",
    "CrossAssetValidationResult",
    "CorrelationGroupResult",
    "MarketRegimeResult",
    # Cross-Asset Validation - Enums
    "ValidationMethod",
    "MarketRegimeIndicator",
    "RobustnessLevel",
    # Cross-Asset Validation - Validators
    "LeaveOneAssetOutValidator",
    "CorrelationStratifiedValidator",
    "MarketRegimeValidator",
    "ValidationResultAnalyzer",
    # Cross-Asset Validation - Convenience
    "leave_one_asset_out",
    "correlation_stratified_validation",
    "market_regime_validation",
]
