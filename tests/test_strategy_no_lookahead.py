import pandas as pd
from qtrade.strategy.ema_cross import generate_positions
from qtrade.strategy.base import StrategyContext


def test_positions_shifted_no_lookahead():
    # dummy data
    idx = pd.date_range("2024-01-01", periods=5, freq="H", tz="UTC")
    df = pd.DataFrame({"close": [1, 2, 3, 2, 1], "open": [1, 2, 3, 2, 1]}, index=idx)

    pos = generate_positions(df, StrategyContext("TEST"), {"fast": 2, "slow": 3})

    # first bar must be 0 due to shift
    assert float(pos.iloc[0]) == 0.0
