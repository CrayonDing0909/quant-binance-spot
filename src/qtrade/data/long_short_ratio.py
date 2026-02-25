"""
Long/Short Ratio 數據模組

支援 provider：
    1. vision — data.binance.vision 每日 metrics CSV (2021-12 至今, 5m)
    2. api — Binance Futures API (~30 天歷史)

提供三種 LSR：
    - lsr: 全帳戶 Long/Short Ratio
    - top_lsr_account: 大戶帳戶數 Long/Short Ratio
    - top_lsr_position: 大戶持倉量 Long/Short Ratio

使用方式：
    from qtrade.data.long_short_ratio import (
        download_lsr,
        load_lsr,
        align_lsr_to_klines,
        compute_lsr_coverage,
    )
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

LSR_TYPES = ["lsr", "top_lsr_account", "top_lsr_position"]
_BASE_DIR = Path("data/binance/futures/derivatives")


def download_lsr(
    symbol: str,
    lsr_type: str = "lsr",
    start: str | None = None,
    end: str | None = None,
    interval: str = "1h",
    provider: str = "vision",
) -> pd.Series:
    """
    下載 Long/Short Ratio 數據

    Args:
        symbol: 交易對
        lsr_type: "lsr" / "top_lsr_account" / "top_lsr_position"
        provider: "vision" / "api"

    Returns:
        pd.Series indexed by UTC timestamp
    """
    if lsr_type not in LSR_TYPES:
        raise ValueError(f"Unknown lsr_type: {lsr_type}. Available: {LSR_TYPES}")

    from qtrade.data._derivatives_common import fetch_vision_single_metric, fetch_api_single_metric

    if provider == "vision":
        return fetch_vision_single_metric(symbol, lsr_type, start, end, interval)
    elif provider == "api":
        return fetch_api_single_metric(symbol, lsr_type, interval)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def save_lsr(series: pd.Series, symbol: str, lsr_type: str = "lsr", data_dir: Path = _BASE_DIR) -> Path:
    """儲存 LSR 到 parquet"""
    path = get_lsr_path(data_dir, symbol, lsr_type)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = series.to_frame(name=lsr_type)
    df.to_parquet(path, index=True)
    logger.info(f"💾 LSR saved: {path} ({len(df)} rows)")
    return path


def load_lsr(
    symbol: str,
    lsr_type: str = "lsr",
    data_dir: Path = _BASE_DIR,
) -> Optional[pd.Series]:
    """載入 LSR 數據"""
    path = get_lsr_path(data_dir, symbol, lsr_type)
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        col = lsr_type if lsr_type in df.columns else df.columns[0]
        return df[col]
    except Exception as e:
        logger.warning(f"⚠️  LSR load failed ({lsr_type}/{symbol}): {e}")
        return None


def get_lsr_path(data_dir: Path, symbol: str, lsr_type: str = "lsr") -> Path:
    """取得 LSR 標準儲存路徑"""
    return data_dir / lsr_type / f"{symbol}.parquet"


def align_lsr_to_klines(
    lsr_series: pd.Series | None,
    kline_index: pd.DatetimeIndex,
    max_ffill_bars: int = 2,
) -> pd.Series | None:
    """
    將 LSR 對齊到 K 線時間軸（forward-fill，嚴格因果）
    """
    if lsr_series is None or lsr_series.empty:
        return None

    series = lsr_series.copy()

    if kline_index.tz is None and series.index.tz is not None:
        series.index = series.index.tz_localize(None)
    elif kline_index.tz is not None and series.index.tz is None:
        series.index = series.index.tz_localize(kline_index.tz)

    aligned = series.reindex(kline_index, method="ffill", limit=max_ffill_bars)

    n_missing = aligned.isna().sum()
    n_total = len(aligned)
    if n_missing > 0:
        coverage = (n_total - n_missing) / n_total * 100
        logger.info(f"📊 LSR alignment: {n_total - n_missing}/{n_total} ({coverage:.1f}%)")

    return aligned


def compute_lsr_coverage(
    symbols: list[str],
    data_dir: Path = _BASE_DIR,
    lsr_type: str = "lsr",
) -> pd.DataFrame:
    """計算 LSR 覆蓋率"""
    rows = []
    for symbol in symbols:
        series = load_lsr(symbol, lsr_type, data_dir)
        if series is None or series.empty:
            rows.append({"symbol": symbol, "rows": 0, "start": None, "end": None, "days": 0})
        else:
            days = (series.index[-1] - series.index[0]).days
            rows.append({
                "symbol": symbol,
                "rows": len(series),
                "start": series.index[0].strftime("%Y-%m-%d"),
                "end": series.index[-1].strftime("%Y-%m-%d"),
                "days": days,
            })
    return pd.DataFrame(rows)
