"""
CB Premium 策略 — Coinbase-Binance 價差作為機構情緒 Standalone Signal

Alpha 來源：
    Coinbase 以美國機構/合規資金為主，Binance 以散戶/亞洲資金為主。
    當 Coinbase 出現溢價（Premium > 0），代表美國機構買入力量強勁 → bullish。
    當 Coinbase 出現折價（Premium < 0），代表亞洲主導拋售 → bearish。
    Premium 方向是機構資金流的 leading indicator。

信號定義：
    1. premium = (coinbase_close - binance_close) / binance_close
    2. Signal = rolling z-score of premium (24h or 168h window)
    3. Position = pctrank(signal) > long_threshold → long
                  pctrank(signal) < short_threshold → short
                  else → flat
    4. Rebalance cadence: every N hours (default 24h = daily)

Research Evidence:
    - IC: +0.022~+0.031 @ 1h, BTC+ETH both positive, pct+ 68-73%
    - Best config: LO 50 daily, BTC SR=1.462, ETH SR=1.368
    - TSMOM correlation: +0.009 (BTC), -0.068 (ETH) — near zero
    - Source: 华山论剑 (@huashanlunjians) methodology

Data dependency:
    Coinbase BTC/USD (or ETH/USD) 1h klines cached at data/coinbase/
    Auto-loaded via _data_dir param from backtest runner.

Changelog:
    v1 (2026-03-28): Initial implementation from EDA handoff
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from qtrade.strategy.base import StrategyContext
from qtrade.strategy import register_strategy
from qtrade.utils.log import get_logger

logger = get_logger(__name__)

# Coinbase symbol mapping: Binance symbol → Coinbase cache filename
_CB_SYMBOL_MAP = {
    "BTCUSDT": "BTC_USD_1h.parquet",
    "ETHUSDT": "ETH_USD_1h.parquet",
}


def _load_coinbase_close(symbol: str, data_dir: str | Path | None) -> pd.Series | None:
    """Load Coinbase close price from local parquet cache."""
    if data_dir is None:
        return None

    cb_filename = _CB_SYMBOL_MAP.get(symbol)
    if cb_filename is None:
        logger.warning(f"  CB Premium [{symbol}]: no Coinbase mapping, returning None")
        return None

    cb_path = Path(data_dir) / "coinbase" / cb_filename
    if not cb_path.exists():
        logger.warning(
            f"  CB Premium [{symbol}]: Coinbase cache not found: {cb_path}. "
            f"Run: PYTHONPATH=src python scripts/research/research_cb_premium_eda.py"
        )
        return None

    cb_df = pd.read_parquet(cb_path)
    logger.info(f"  CB Premium [{symbol}]: loaded Coinbase data ({len(cb_df)} bars)")
    return cb_df["close"]


def _compute_premium(binance_close: pd.Series, coinbase_close: pd.Series) -> pd.Series:
    """Compute (coinbase - binance) / binance premium, aligned on common index."""
    common = binance_close.index.intersection(coinbase_close.index)
    bn = binance_close.loc[common]
    cb = coinbase_close.loc[common]

    premium = (cb - bn) / bn
    premium = premium.replace([np.inf, -np.inf], np.nan)
    return premium


@register_strategy("cb_premium")
def generate_positions(
    df: pd.DataFrame,
    ctx: StrategyContext,
    params: dict,
) -> pd.Series:
    """
    CB Premium standalone strategy.

    Params:
        signal_variant: str = "zscore_24h"
            Signal processing variant:
            - "zscore_24h": 24h rolling z-score (default, best IC stability)
            - "zscore_168h": 168h rolling z-score (captures weekly cycle)
            - "ema_cross": EMA 12h - EMA 72h crossover
            - "ema4h": 4h EMA smoothed raw premium
            - "sma24h": 24h SMA of raw premium
        pctrank_lookback: int = 720
            Lookback window for percentile ranking (30 days)
        long_threshold: float = 0.5
            Percentile rank above which → long
        short_threshold: float = 0.3
            Percentile rank below which → short (set < 0 for long-only)
        rebalance_hours: int = 24
            Rebalance cadence (hours)
    """
    # ── Params ──
    signal_variant = str(params.get("signal_variant", "zscore_24h"))
    pctrank_lookback = int(params.get("pctrank_lookback", 720))
    long_threshold = float(params.get("long_threshold", 0.5))
    short_threshold = float(params.get("short_threshold", 0.3))
    rebalance_hours = int(params.get("rebalance_hours", 24))
    data_dir = params.get("_data_dir")

    # ── 1. Load Coinbase data ──
    cb_close = _load_coinbase_close(ctx.symbol, data_dir)
    if cb_close is None:
        logger.warning(f"  CB Premium [{ctx.symbol}]: no Coinbase data, returning flat")
        return pd.Series(0.0, index=df.index)

    # ── 2. Compute premium ──
    premium = _compute_premium(df["close"], cb_close)

    # Reindex to full Binance index (ffill for bars where Coinbase has gaps)
    premium = premium.reindex(df.index).ffill()

    coverage = (~premium.isna()).mean()
    if coverage < 0.3:
        logger.warning(
            f"  CB Premium [{ctx.symbol}]: premium coverage {coverage:.1%} < 30%, returning flat"
        )
        return pd.Series(0.0, index=df.index)

    logger.info(
        f"  CB Premium [{ctx.symbol}]: premium coverage={coverage:.1%}, "
        f"mean={premium.mean():.6f}, std={premium.std():.6f}"
    )

    # ── 3. Signal variant ──
    if signal_variant == "zscore_24h":
        roll = premium.rolling(24, min_periods=12)
        signal = (premium - roll.mean()) / roll.std().replace(0, np.nan)
    elif signal_variant == "zscore_168h":
        roll = premium.rolling(168, min_periods=84)
        signal = (premium - roll.mean()) / roll.std().replace(0, np.nan)
    elif signal_variant == "ema_cross":
        ema_fast = premium.ewm(span=12, adjust=False).mean()
        ema_slow = premium.ewm(span=72, adjust=False).mean()
        signal = ema_fast - ema_slow
    elif signal_variant == "ema4h":
        signal = premium.ewm(span=4, adjust=False).mean()
    elif signal_variant == "sma24h":
        signal = premium.rolling(24, min_periods=12).mean()
    else:
        logger.warning(f"  CB Premium [{ctx.symbol}]: unknown variant '{signal_variant}', using zscore_24h")
        roll = premium.rolling(24, min_periods=12)
        signal = (premium - roll.mean()) / roll.std().replace(0, np.nan)

    # ── 4. Percentile rank → position ──
    min_periods = max(pctrank_lookback // 4, 30)
    pctrank = signal.rolling(pctrank_lookback, min_periods=min_periods).rank(pct=True)

    n = len(df)
    pos_arr = np.zeros(n, dtype=np.float64)
    current = 0.0

    for i in range(n):
        pr = pctrank.iloc[i]
        if np.isnan(pr):
            pos_arr[i] = current
            continue

        if i % rebalance_hours == 0:
            if pr > long_threshold:
                current = 1.0
            elif pr < short_threshold:
                current = -1.0
            else:
                current = 0.0

        pos_arr[i] = current

    pos = pd.Series(pos_arr, index=df.index)

    logger.info(
        f"  CB Premium [{ctx.symbol}]: variant={signal_variant}, "
        f"TIM={float((pos.abs() > 0.01).mean()):.1%}, "
        f"long={float((pos > 0.01).mean()):.1%}, "
        f"short={float((pos < -0.01).mean()):.1%}"
    )

    return pos
