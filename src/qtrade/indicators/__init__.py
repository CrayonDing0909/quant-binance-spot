"""
統一的技術指標庫

提供常用的技術指標計算函數，避免在策略中重複實現。

指標列表:
- 趨勢類: EMA, SMA, WMA, MACD, ADX
- 動量類: RSI, Stochastic
- 波動類: Bollinger Bands, ATR
- Regime: Efficiency Ratio, Choppiness Index
- 成交量: OBV, VWAP
"""
from __future__ import annotations

from .rsi import calculate_rsi, calculate_rsi_divergence
from .macd import calculate_macd
from .bollinger import calculate_bollinger_bands
from .moving_average import calculate_ema, calculate_sma, calculate_wma
from .atr import calculate_atr, calculate_atr_percent, calculate_true_range
from .stochastic import calculate_stochastic
from .adx import calculate_adx
from .volume import calculate_obv, calculate_vwap
from .efficiency_ratio import calculate_efficiency_ratio, calculate_choppiness_index

__all__ = [
    # 趨勢
    "calculate_ema",
    "calculate_sma",
    "calculate_wma",
    "calculate_macd",
    "calculate_adx",
    # 動量
    "calculate_rsi",
    "calculate_rsi_divergence",
    "calculate_stochastic",
    # 波動
    "calculate_bollinger_bands",
    "calculate_atr",
    "calculate_atr_percent",
    "calculate_true_range",
    # Regime
    "calculate_efficiency_ratio",
    "calculate_choppiness_index",
    # 成交量
    "calculate_obv",
    "calculate_vwap",
]
