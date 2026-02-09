"""
Signal Generator — 即時信號產生器

從 Binance 拉取最新 K 線數據，運行策略，輸出交易信號。
設計為復用回測策略程式碼，無需改寫策略。
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

# 策略至少需要多少根 K 線才能計算指標
MIN_BARS = 300


def fetch_recent_klines(
    symbol: str,
    interval: str,
    bars: int = MIN_BARS,
) -> pd.DataFrame:
    """
    從 Binance 拉取最近 N 根 **已收盤** K 線

    Binance API 總是返回當前未收盤的 K 線作為最後一根，
    在 Live Trading 中使用未收盤 K 線會導致指標不可靠（假信號）。
    因此這裡會自動丟棄未收盤的 K 線。

    Args:
        symbol: 交易對, e.g. "BTCUSDT"
        interval: K 線週期, e.g. "1h"
        bars: 需要的 K 線數量

    Returns:
        DataFrame with OHLCV (只包含已收盤的 K 線)
    """
    from datetime import datetime, timezone, timedelta

    # 根據 interval 估算需要多少時間
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

    # ── 丟棄未收盤的 K 線 ──────────────────────────
    # Binance close_time 是該 K 線的結束時間 (e.g. 1h K 線 12:00 → close_time=12:59:59.999)
    # 如果 close_time > 當前時間 → 該 K 線尚未收盤，必須丟棄
    if "close_time" in df.columns:
        now = pd.Timestamp.now(tz="UTC")
        closed_mask = df["close_time"] <= now
        n_dropped = (~closed_mask).sum()
        if n_dropped > 0:
            logger.debug(f"  {symbol}: 丟棄 {n_dropped} 根未收盤 K 線")
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
    生成單個交易對的信號

    Args:
        symbol: 交易對
        strategy_name: 策略名稱
        params: 策略參數
        interval: K 線週期
        bars: 需要的 K 線數量
        df: 可選，直接傳入 K 線數據（測試用）

    Returns:
        {
            "symbol": str,
            "signal": float,          # 目標倉位 [0, 1]
            "price": float,           # 當前價格
            "timestamp": str,         # 最新 K 線時間
            "strategy": str,
            "indicators": dict,       # 關鍵指標值（除錯用）
        }
    """
    # 獲取數據
    if df is None:
        df = fetch_recent_klines(symbol, interval, bars)

    if len(df) < 50:
        logger.warning(f"⚠️  {symbol}: 數據不足 ({len(df)} bars)")
        return {
            "symbol": symbol,
            "signal": 0.0,
            "price": 0.0,
            "timestamp": "",
            "strategy": strategy_name,
            "indicators": {},
        }

    # 運行策略
    ctx = StrategyContext(symbol=symbol, interval=interval)
    strategy_func = get_strategy(strategy_name)
    positions = strategy_func(df, ctx, params)

    # 取最後一根 K 線的信號
    latest_signal = float(positions.iloc[-1])
    latest_price = float(df["close"].iloc[-1])
    latest_time = str(df.index[-1])

    # 收集關鍵指標（除錯用）
    indicators = {
        "close": latest_price,
        "bars": len(df),
    }

    # 嘗試計算常用指標
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
        pass  # 指標計算失敗不影響信號

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
