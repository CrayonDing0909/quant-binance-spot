"""
统一的技术指标库

提供常用的技术指标计算函数，避免在策略中重复实现。

指标列表:
- 趋势类: EMA, SMA, WMA, MACD, ADX
- 动量类: RSI, Stochastic
- 波动类: Bollinger Bands, ATR
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

__all__ = [
    # 趋势
    "calculate_ema",
    "calculate_sma",
    "calculate_wma",
    "calculate_macd",
    "calculate_adx",
    # 动量
    "calculate_rsi",
    "calculate_rsi_divergence",
    "calculate_stochastic",
    # 波动
    "calculate_bollinger_bands",
    "calculate_atr",
    "calculate_atr_percent",
    "calculate_true_range",
    # 成交量
    "calculate_obv",
    "calculate_vwap",
]
