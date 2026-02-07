from __future__ import annotations
import pandas as pd
from .base import StrategyContext
from . import register_strategy
from ..indicators import calculate_ema


@register_strategy("ema_cross")
def generate_positions(df: pd.DataFrame, ctx: StrategyContext, params: dict) -> pd.Series:
    fast = int(params.get("fast", 20))
    slow = int(params.get("slow", 60))
    close = df["close"]

    ema_fast = calculate_ema(close, fast, adjust=False)
    ema_slow = calculate_ema(close, slow, adjust=False)

    # Signal defined at bar close. Long when fast > slow.
    raw = (ema_fast > ema_slow).astype(float)

    # To avoid lookahead: execute next bar open => shift signal by 1 bar.
    pos = raw.shift(1).fillna(0.0)
    return pos.clip(0.0, 1.0)
