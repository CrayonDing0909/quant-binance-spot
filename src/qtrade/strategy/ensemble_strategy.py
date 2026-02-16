"""
策略 Ensemble — 多策略信號加權組合

核心理念：
    1. 組合低相關策略，降低回撤、提高 Sharpe
    2. rsi_adx_atr（均值回歸 + 趨勢過濾）× macd_momentum（趨勢跟蹤）
    3. 信號相關性 ≈ 0.06 → 近乎獨立的 alpha 來源

組合方式（平均信號）：
    ensemble_signal = w1 × rsi_adx_atr_signal + w2 × macd_momentum_signal
    - 預設 50/50 等權（可配置）
    - 兩策略共識 → 全倉
    - 單策略觸發 → 半倉
    - 方向衝突 → 平倉

為什麼不是投票制？
    - 投票制（多數決）在只有 2 策略時退化為「全或無」
    - 加權平均保留連續倉位信息，對 position sizing 更友好
    - 與 volatility targeting 完美配合：連續信號 × 連續倉位大小
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from .base import StrategyContext
from . import register_strategy, get_strategy


@register_strategy("ensemble_rsi_macd")
def generate_positions(
    df: pd.DataFrame,
    ctx: StrategyContext,
    params: dict,
) -> pd.Series:
    """
    RSI-ADX-ATR + MACD Momentum 加權 Ensemble

    可配置參數:
        weight_rsi:  RSI 策略權重（預設 0.5）
        weight_macd: MACD 策略權重（預設 0.5）
        consensus_only: 只在兩策略方向一致時交易（預設 False）

    注意：
        - 子策略會繼承 params 中的共用參數（如 adx_period, atr_period 等）
        - 各子策略有各自的預設參數，除非在 YAML 中明確覆蓋
    """
    supports_short = ctx.market_type == "futures"

    # 權重
    w_rsi = float(params.get("weight_rsi", 0.5))
    w_macd = float(params.get("weight_macd", 0.5))
    consensus_only = bool(params.get("consensus_only", False))

    # 正規化權重
    total_w = w_rsi + w_macd
    w_rsi /= total_w
    w_macd /= total_w

    # 取得子策略函數
    rsi_func = get_strategy("rsi_adx_atr")
    macd_func = get_strategy("macd_momentum")

    # 計算子策略信號
    sig_rsi = rsi_func(df, ctx, params)
    sig_macd = macd_func(df, ctx, params)

    # 加權平均
    ensemble = w_rsi * sig_rsi + w_macd * sig_macd

    # Consensus 模式：方向不一致時歸零
    if consensus_only:
        # 同向才保留
        same_direction = (np.sign(sig_rsi) == np.sign(sig_macd)) & (sig_rsi != 0) & (sig_macd != 0)
        ensemble = ensemble.where(same_direction, 0.0)

    # Clip 信號範圍
    if supports_short:
        return ensemble.clip(-1.0, 1.0)
    else:
        return ensemble.clip(0.0, 1.0)
