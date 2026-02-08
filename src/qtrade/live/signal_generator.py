"""
Signal Generator — 即时信号产生器

从 Binance 拉取最新 K 线数据，运行策略，输出交易信号。
设计为复用回测策略代码，无需改写策略。
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

from ..data.klines import fetch_klines
from ..data.storage import load_klines
from ..data.quality import clean_data
from ..strategy import get_strategy
from ..strategy.base import StrategyContext
from ..utils.log import get_logger

logger = get_logger("signal_gen")

# 策略至少需要多少根 K 线才能计算指标
MIN_BARS = 300


def fetch_recent_klines(
    symbol: str,
    interval: str,
    bars: int = MIN_BARS,
) -> pd.DataFrame:
    """
    从 Binance 拉取最近 N 根 **已收盘** K 线

    Binance API 总是返回当前未收盘的 K 线作为最后一根，
    在 Live Trading 中使用未收盘 K 线会导致指标不可靠（假信号）。
    因此这里会自动丢弃未收盘的 K 线。

    Args:
        symbol: 交易对, e.g. "BTCUSDT"
        interval: K 线周期, e.g. "1h"
        bars: 需要的 K 线数量

    Returns:
        DataFrame with OHLCV (只包含已收盘的 K 线)
    """
    from datetime import datetime, timezone, timedelta

    # 根据 interval 估算需要多少时间
    interval_minutes = {
        "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
        "1h": 60, "2h": 120, "4h": 240, "6h": 360, "8h": 480,
        "12h": 720, "1d": 1440,
    }

    minutes = interval_minutes.get(interval, 60)
    start_dt = datetime.now(timezone.utc) - timedelta(minutes=minutes * (bars + 10))
    start_str = start_dt.strftime("%Y-%m-%d")

    df = fetch_klines(symbol, interval, start_str)
    df = clean_data(df, fill_method="forward", remove_outliers=False, remove_duplicates=True)

    # ── 丢弃未收盘的 K 线 ──────────────────────────
    # Binance close_time 是该 K 线的结束时间 (e.g. 1h K 线 12:00 → close_time=12:59:59.999)
    # 如果 close_time > 当前时间 → 该 K 线尚未收盘，必须丢弃
    if "close_time" in df.columns:
        now = pd.Timestamp.now(tz="UTC")
        closed_mask = df["close_time"] <= now
        n_dropped = (~closed_mask).sum()
        if n_dropped > 0:
            logger.debug(f"  {symbol}: 丢弃 {n_dropped} 根未收盘 K 线")
        df = df[closed_mask]

    # 只保留最近 bars 根
    if len(df) > bars:
        df = df.iloc[-bars:]

    return df


def generate_signal(
    symbol: str,
    strategy_name: str,
    params: dict,
    interval: str = "1h",
    bars: int = MIN_BARS,
    df: pd.DataFrame | None = None,
) -> dict:
    """
    生成单个交易对的信号

    Args:
        symbol: 交易对
        strategy_name: 策略名称
        params: 策略参数
        interval: K 线周期
        bars: 需要的 K 线数量
        df: 可选，直接传入 K 线数据（测试用）

    Returns:
        {
            "symbol": str,
            "signal": float,          # 目标仓位 [0, 1]
            "price": float,           # 当前价格
            "timestamp": str,         # 最新 K 线时间
            "strategy": str,
            "indicators": dict,       # 关键指标值（调试用）
        }
    """
    # 获取数据
    if df is None:
        df = fetch_recent_klines(symbol, interval, bars)

    if len(df) < 50:
        logger.warning(f"⚠️  {symbol}: 数据不足 ({len(df)} bars)")
        return {
            "symbol": symbol,
            "signal": 0.0,
            "price": 0.0,
            "timestamp": "",
            "strategy": strategy_name,
            "indicators": {},
        }

    # 运行策略
    ctx = StrategyContext(symbol=symbol, interval=interval)
    strategy_func = get_strategy(strategy_name)
    positions = strategy_func(df, ctx, params)

    # 取最后一根 K 线的信号
    latest_signal = float(positions.iloc[-1])
    latest_price = float(df["close"].iloc[-1])
    latest_time = str(df.index[-1])

    # 收集关键指标（调试用）
    indicators = {
        "close": latest_price,
        "bars": len(df),
    }

    # 尝试计算常用指标
    try:
        from ..indicators import calculate_rsi, calculate_adx, calculate_atr
        rsi_period = int(params.get("rsi_period", 14))
        rsi = calculate_rsi(df["close"], rsi_period)
        indicators["rsi"] = round(float(rsi.iloc[-1]), 2)

        adx_period = int(params.get("adx_period", 14))
        adx_df = calculate_adx(df, adx_period)
        indicators["adx"] = round(float(adx_df["ADX"].iloc[-1]), 2)
        indicators["plus_di"] = round(float(adx_df["+DI"].iloc[-1]), 2)
        indicators["minus_di"] = round(float(adx_df["-DI"].iloc[-1]), 2)

        atr_period = int(params.get("atr_period", 14))
        atr = calculate_atr(df, atr_period)
        indicators["atr"] = round(float(atr.iloc[-1]), 2)
    except Exception:
        pass  # 指标计算失败不影响信号

    result = {
        "symbol": symbol,
        "signal": latest_signal,
        "price": latest_price,
        "timestamp": latest_time,
        "strategy": strategy_name,
        "indicators": indicators,
    }

    logger.info(
        f"📊 {symbol}: signal={latest_signal:.1f}, price={latest_price:.2f}, "
        f"RSI={indicators.get('rsi', '?')}, ADX={indicators.get('adx', '?')}"
    )

    return result

