"""
回測模組

提供完整的回測功能：
- 單一資產回測
- 回測指標計算
- 策略驗證（Walk-Forward, 參數敏感性）
- Cross-Asset 驗證
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
from .validation import (
    walk_forward_analysis,
    parameter_sensitivity_analysis,
    detect_overfitting,
)
from .cross_asset_validation import (
    # 配置類
    CrossAssetValidationConfig,
    CorrelationStratifiedConfig,
    MarketRegimeConfig,
    # 結果類
    AssetValidationResult,
    CrossAssetValidationResult,
    CorrelationGroupResult,
    MarketRegimeResult,
    # 列舉
    ValidationMethod,
    MarketRegimeIndicator,
    RobustnessLevel,
    # 驗證器
    LeaveOneAssetOutValidator,
    CorrelationStratifiedValidator,
    MarketRegimeValidator,
    ValidationResultAnalyzer,
    # 便捷函數
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
    # Validation
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
