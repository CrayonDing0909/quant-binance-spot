"""
Binance Real-Time Price Feed for Polymarket Strategy.

Fetches recent 1-minute klines and computes TA indicators
for the krajekis 5-layer strategy.

All indicators reuse existing qtrade.indicators functions.
No WebSocket — simple HTTP polling (called every 30s by runner).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx
import numpy as np
import pandas as pd

from qtrade.indicators import (
    calculate_atr,
    calculate_ema,
    calculate_macd,
    calculate_rsi,
    calculate_vwap,
)

logger = logging.getLogger(__name__)

# Binance symbol → Polymarket coin mapping
BINANCE_SYMBOLS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "XRP": "XRPUSDT",
}


@dataclass
class TASignals:
    """Technical analysis signals computed from recent klines."""

    # Current values
    close: float
    vwap: float
    ema_21: float
    ema_50: float
    rsi_14: float
    macd_hist: float       # MACD histogram (positive = bullish momentum)
    macd_hist_prev: float  # Previous histogram (for acceleration/deceleration)
    atr_14: float          # 15-minute ATR (computed on 15m resampled data)
    volume_ratio: float    # current volume / 20-period avg volume

    # Derived signals
    @property
    def trend_bullish(self) -> bool:
        """EMA21 > EMA50 and price > VWAP."""
        return self.ema_21 > self.ema_50 and self.close > self.vwap

    @property
    def trend_bearish(self) -> bool:
        """EMA21 < EMA50 and price < VWAP."""
        return self.ema_21 < self.ema_50 and self.close < self.vwap

    @property
    def rsi_overbought(self) -> bool:
        return self.rsi_14 > 70

    @property
    def rsi_oversold(self) -> bool:
        return self.rsi_14 < 30

    @property
    def macd_expanding_bull(self) -> bool:
        """MACD histogram positive and growing."""
        return self.macd_hist > 0 and self.macd_hist > self.macd_hist_prev

    @property
    def macd_expanding_bear(self) -> bool:
        """MACD histogram negative and growing (more negative)."""
        return self.macd_hist < 0 and self.macd_hist < self.macd_hist_prev

    @property
    def macd_exhaustion(self) -> bool:
        """MACD histogram shrinking (trend losing steam)."""
        return abs(self.macd_hist) < abs(self.macd_hist_prev)

    @property
    def price_above_vwap(self) -> bool:
        return self.close > self.vwap

    @property
    def vwap_distance_pct(self) -> float:
        """How far price is from VWAP as percentage."""
        if self.vwap == 0:
            return 0.0
        return (self.close - self.vwap) / self.vwap * 100


def fetch_recent_klines(
    coin: str,
    interval: str = "1m",
    limit: int = 100,
) -> pd.DataFrame | None:
    """
    Fetch recent klines from Binance Futures REST API.

    Args:
        coin: "BTC", "ETH", "SOL", "XRP"
        interval: candle interval ("1m", "5m", "15m")
        limit: number of candles (max 1500)

    Returns:
        DataFrame with OHLCV columns + datetime index, or None on error
    """
    symbol = BINANCE_SYMBOLS.get(coin.upper())
    if symbol is None:
        logger.error(f"Unknown coin: {coin}")
        return None

    try:
        resp = httpx.get(
            "https://fapi.binance.com/fapi/v1/klines",
            params={"symbol": symbol, "interval": interval, "limit": limit},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        records = []
        for k in data:
            records.append({
                "open_time": pd.Timestamp(k[0], unit="ms", tz="UTC"),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            })

        df = pd.DataFrame(records).set_index("open_time").sort_index()
        return df

    except Exception as e:
        logger.error(f"Failed to fetch {coin} {interval} klines: {e}")
        return None


def compute_ta_signals(coin: str) -> TASignals | None:
    """
    Fetch recent data and compute all TA signals for a coin.

    Fetches 1-minute klines for detailed TA + resamples to 15m for ATR.
    """
    # Fetch 1-minute data (100 bars = ~1.5 hours of history)
    df_1m = fetch_recent_klines(coin, interval="1m", limit=100)
    if df_1m is None or len(df_1m) < 60:
        return None

    close = df_1m["close"]
    volume = df_1m["volume"]

    # Core indicators on 1-minute data
    rsi = calculate_rsi(close, 14)
    macd_result = calculate_macd(close)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    vwap = calculate_vwap(df_1m)

    # MACD histogram
    if isinstance(macd_result, dict):
        macd_hist = macd_result.get("histogram", macd_result.get("MACD_hist", pd.Series()))
    elif isinstance(macd_result, pd.DataFrame):
        hist_col = [c for c in macd_result.columns if "hist" in c.lower()]
        macd_hist = macd_result[hist_col[0]] if hist_col else macd_result.iloc[:, -1]
    else:
        macd_hist = pd.Series(0.0, index=close.index)

    # Volume ratio (current vs 20-period average)
    vol_avg = volume.rolling(20).mean()
    vol_ratio = float(volume.iloc[-1] / vol_avg.iloc[-1]) if vol_avg.iloc[-1] > 0 else 1.0

    # ATR on 15-minute resampled data (for volatility regime)
    df_15m = df_1m.resample("15min").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna()

    if len(df_15m) >= 15:
        atr_15m = calculate_atr(df_15m, 14)
        atr_val = float(atr_15m.iloc[-1]) if not atr_15m.empty else 0.0
    else:
        # Fallback: estimate from 1m data
        atr_1m = calculate_atr(df_1m, 14)
        atr_val = float(atr_1m.iloc[-1] * np.sqrt(15)) if not atr_1m.empty else 0.0

    return TASignals(
        close=float(close.iloc[-1]),
        vwap=float(vwap.iloc[-1]) if not vwap.empty else float(close.iloc[-1]),
        ema_21=float(ema_21.iloc[-1]) if not ema_21.empty else float(close.iloc[-1]),
        ema_50=float(ema_50.iloc[-1]) if not ema_50.empty else float(close.iloc[-1]),
        rsi_14=float(rsi.iloc[-1]) if not rsi.empty else 50.0,
        macd_hist=float(macd_hist.iloc[-1]) if len(macd_hist) > 0 else 0.0,
        macd_hist_prev=float(macd_hist.iloc[-2]) if len(macd_hist) > 1 else 0.0,
        atr_14=atr_val,
        volume_ratio=vol_ratio,
    )
