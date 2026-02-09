"""
指標庫單元測試

確保指標計算正確，這樣回測結果才可信。
"""
import pandas as pd
import numpy as np
import pytest

# ── 測試數據 ──────────────────────────────────────────

def _make_ohlcv(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """生成模擬 OHLCV 數據"""
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.standard_normal(n) * 0.5)
    close = np.maximum(close, 10)  # 避免負數
    high = close + rng.uniform(0.5, 2.0, n)
    low = close - rng.uniform(0.5, 2.0, n)
    open_ = close + rng.uniform(-1.0, 1.0, n)
    volume = rng.uniform(100, 10000, n)

    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": close, "volume": volume,
    }, index=idx)


@pytest.fixture
def df():
    return _make_ohlcv()


@pytest.fixture
def close(df):
    return df["close"]


# ── RSI ───────────────────────────────────────────────

class TestRSI:
    def test_range(self, close):
        from qtrade.indicators import calculate_rsi
        rsi = calculate_rsi(close, period=14)
        assert rsi.min() >= 0, "RSI 不應小於 0"
        assert rsi.max() <= 100, "RSI 不應大於 100"

    def test_period_effect(self, close):
        from qtrade.indicators import calculate_rsi
        rsi_short = calculate_rsi(close, period=7)
        rsi_long = calculate_rsi(close, period=21)
        # 短週期 RSI 波動應該更大
        assert rsi_short.std() >= rsi_long.std() * 0.5

    def test_length(self, close):
        from qtrade.indicators import calculate_rsi
        rsi = calculate_rsi(close, period=14)
        assert len(rsi) == len(close)

    def test_all_up(self):
        """全部上漲 -> RSI 應接近 100"""
        from qtrade.indicators import calculate_rsi
        close = pd.Series(range(1, 31), dtype=float)
        rsi = calculate_rsi(close, period=14)
        assert rsi.iloc[-1] > 90

    def test_all_down(self):
        """全部下跌 -> RSI 應接近 0"""
        from qtrade.indicators import calculate_rsi
        close = pd.Series(range(30, 0, -1), dtype=float)
        rsi = calculate_rsi(close, period=14)
        assert rsi.iloc[-1] < 10


# ── EMA / SMA ────────────────────────────────────────

class TestMovingAverage:
    def test_sma_known_value(self):
        from qtrade.indicators import calculate_sma
        close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        sma = calculate_sma(close, period=3)
        assert abs(sma.iloc[-1] - 4.0) < 1e-10  # (3+4+5)/3 = 4

    def test_ema_follows_price(self, close):
        from qtrade.indicators import calculate_ema
        ema = calculate_ema(close, period=20)
        # EMA 最終值應接近收盤價
        assert abs(ema.iloc[-1] - close.iloc[-1]) < close.std() * 3

    def test_sma_length(self, close):
        from qtrade.indicators import calculate_sma
        sma = calculate_sma(close, period=20)
        assert len(sma) == len(close)


# ── MACD ──────────────────────────────────────────────

class TestMACD:
    def test_columns(self, close):
        from qtrade.indicators import calculate_macd
        macd = calculate_macd(close)
        assert "macd" in macd.columns
        assert "signal" in macd.columns
        assert "histogram" in macd.columns

    def test_histogram_equals_diff(self, close):
        from qtrade.indicators import calculate_macd
        macd = calculate_macd(close)
        diff = macd["macd"] - macd["signal"]
        np.testing.assert_allclose(macd["histogram"].dropna(), diff.dropna(), atol=1e-10)


# ── Bollinger Bands ───────────────────────────────────

class TestBollinger:
    def test_band_order(self, close):
        from qtrade.indicators import calculate_bollinger_bands
        bb = calculate_bollinger_bands(close, period=20)
        valid = bb.dropna()
        assert (valid["upper"] >= valid["middle"]).all()
        assert (valid["middle"] >= valid["lower"]).all()

    def test_columns(self, close):
        from qtrade.indicators import calculate_bollinger_bands
        bb = calculate_bollinger_bands(close, period=20)
        assert "%b" in bb.columns
        assert "bandwidth" in bb.columns


# ── ATR ───────────────────────────────────────────────

class TestATR:
    def test_positive(self, df):
        from qtrade.indicators import calculate_atr
        atr = calculate_atr(df, period=14)
        valid = atr.dropna()
        assert (valid >= 0).all(), "ATR 應全部 >= 0"

    def test_percent(self, df):
        from qtrade.indicators import calculate_atr_percent
        atr_pct = calculate_atr_percent(df, period=14)
        valid = atr_pct.dropna()
        assert (valid >= 0).all()
        # ATR% 一般不超過 50%（除非極端行情）
        assert valid.mean() < 50

    def test_length(self, df):
        from qtrade.indicators import calculate_atr
        atr = calculate_atr(df, period=14)
        assert len(atr) == len(df)


# ── Stochastic ────────────────────────────────────────

class TestStochastic:
    def test_range(self, df):
        from qtrade.indicators import calculate_stochastic
        stoch = calculate_stochastic(df, k_period=14)
        valid = stoch.dropna()
        assert valid["%K"].min() >= 0, "%K 不應小於 0"
        assert valid["%K"].max() <= 100, "%K 不應大於 100"

    def test_columns(self, df):
        from qtrade.indicators import calculate_stochastic
        stoch = calculate_stochastic(df)
        assert "%K" in stoch.columns
        assert "%D" in stoch.columns


# ── ADX ───────────────────────────────────────────────

class TestADX:
    def test_range(self, df):
        from qtrade.indicators import calculate_adx
        adx = calculate_adx(df, period=14)
        valid = adx.dropna()
        assert valid["ADX"].min() >= 0, "ADX 不應小於 0"

    def test_columns(self, df):
        from qtrade.indicators import calculate_adx
        adx = calculate_adx(df)
        assert "ADX" in adx.columns
        assert "+DI" in adx.columns
        assert "-DI" in adx.columns


# ── Volume ────────────────────────────────────────────

class TestVolume:
    def test_obv_length(self, df):
        from qtrade.indicators import calculate_obv
        obv = calculate_obv(df)
        assert len(obv) == len(df)

    def test_vwap_positive(self, df):
        from qtrade.indicators import calculate_vwap
        vwap = calculate_vwap(df)
        valid = vwap.dropna()
        assert (valid > 0).all(), "VWAP 應全部 > 0"
