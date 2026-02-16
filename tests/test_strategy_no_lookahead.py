import pandas as pd
import numpy as np
from qtrade.strategy.rsi_adx_atr_strategy import generate_positions
from qtrade.strategy.base import StrategyContext


def test_positions_shifted_no_lookahead():
    """確保策略信號有 shift，不會 look-ahead"""
    np.random.seed(42)
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    close = pd.Series(np.cumsum(np.random.randn(n)) + 100, index=idx)
    df = pd.DataFrame({
        "open": close.shift(1).fillna(close.iloc[0]),
        "high": close + abs(np.random.randn(n)),
        "low": close - abs(np.random.randn(n)),
        "close": close,
        "volume": np.random.randint(100, 10000, n).astype(float),
    }, index=idx)

    ctx = StrategyContext("TEST", market_type="futures", direction="both")
    params = {
        "rsi_period": 10,
        "oversold": 30,
        "overbought": 70,
        "min_adx": 15,
        "adx_period": 14,
        "stop_loss_atr": 2.0,
        "atr_period": 14,
        "cooldown_bars": 3,
    }
    pos = generate_positions(df, ctx, params)

    # first bar must be 0 (indicators need warmup, no look-ahead)
    assert float(pos.iloc[0]) == 0.0
