"""
Taker Buy/Sell Volume 數據模組 — 包含 CVD 衍生計算

Taker Buy/Sell Volume Ratio:
    ratio > 1 → 主動買入量 > 主動賣出量（看多傾向）
    ratio < 1 → 主動賣出量 > 主動買入量（看空傾向）

CVD (Cumulative Volume Delta):
    從 Taker Buy/Sell Ratio 近似計算
    delta = (ratio - 1) / (ratio + 1) → 標準化 [-1, 1]
    CVD = cumsum(delta)

使用方式：
    from qtrade.data.taker_volume import (
        download_taker_volume,
        load_taker_volume,
        compute_cvd,
        align_taker_to_klines,
    )
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_BASE_DIR = Path("data/binance/futures/derivatives")


def download_taker_volume(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1h",
    provider: str = "vision",
) -> pd.Series:
    """
    下載 Taker Buy/Sell Volume Ratio

    Returns:
        pd.Series: taker_vol_ratio indexed by UTC timestamp
    """
    from qtrade.data._derivatives_common import fetch_vision_single_metric, fetch_api_single_metric

    if provider == "vision":
        return fetch_vision_single_metric(symbol, "taker_vol_ratio", start, end, interval)
    elif provider == "api":
        return fetch_api_single_metric(symbol, "taker_vol_ratio", interval)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def compute_cvd(taker_vol_ratio: pd.Series) -> pd.Series:
    """
    從 Taker Buy/Sell Volume Ratio 近似計算 CVD

    真正的 CVD 需要逐筆成交數據，但 taker ratio 的累積變化
    能捕捉同樣的趨勢方向訊號。

    Args:
        taker_vol_ratio: Taker Buy/Sell Volume Ratio Series

    Returns:
        CVD 累積序列
    """
    if taker_vol_ratio.empty:
        return pd.Series(dtype=float, name="cvd")

    ratio = taker_vol_ratio.copy()
    delta = (ratio - 1.0) / (ratio + 1.0)
    delta = delta.fillna(0.0).clip(-1.0, 1.0)

    cvd = delta.cumsum()
    cvd.name = "cvd"
    return cvd


def save_taker_volume(series: pd.Series, symbol: str, data_dir: Path = _BASE_DIR) -> Path:
    """儲存 Taker Volume Ratio 到 parquet"""
    path = data_dir / "taker_vol_ratio" / f"{symbol}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df = series.to_frame(name="taker_vol_ratio")
    df.to_parquet(path, index=True)
    logger.info(f"💾 Taker vol saved: {path} ({len(df)} rows)")
    return path


def save_cvd(series: pd.Series, symbol: str, data_dir: Path = _BASE_DIR) -> Path:
    """儲存 CVD 到 parquet"""
    path = data_dir / "cvd" / f"{symbol}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df = series.to_frame(name="cvd")
    df.to_parquet(path, index=True)
    logger.info(f"💾 CVD saved: {path} ({len(df)} rows)")
    return path


def load_taker_volume(symbol: str, data_dir: Path = _BASE_DIR) -> Optional[pd.Series]:
    """載入 Taker Volume Ratio"""
    path = data_dir / "taker_vol_ratio" / f"{symbol}.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        col = "taker_vol_ratio" if "taker_vol_ratio" in df.columns else df.columns[0]
        return df[col]
    except Exception as e:
        logger.warning(f"⚠️  Taker vol load failed ({symbol}): {e}")
        return None


def load_cvd(symbol: str, data_dir: Path = _BASE_DIR) -> Optional[pd.Series]:
    """載入 CVD"""
    path = data_dir / "cvd" / f"{symbol}.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        col = "cvd" if "cvd" in df.columns else df.columns[0]
        return df[col]
    except Exception as e:
        logger.warning(f"⚠️  CVD load failed ({symbol}): {e}")
        return None


def align_taker_to_klines(
    series: pd.Series | None,
    kline_index: pd.DatetimeIndex,
    max_ffill_bars: int = 2,
) -> pd.Series | None:
    """
    將 Taker Volume 或 CVD 對齊到 K 線時間軸
    """
    if series is None or series.empty:
        return None

    s = series.copy()
    if kline_index.tz is None and s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    elif kline_index.tz is not None and s.index.tz is None:
        s.index = s.index.tz_localize(kline_index.tz)

    aligned = s.reindex(kline_index, method="ffill", limit=max_ffill_bars)

    n_missing = aligned.isna().sum()
    if n_missing > 0:
        coverage = (len(aligned) - n_missing) / len(aligned) * 100
        logger.info(f"📊 {series.name} alignment: {coverage:.1f}% coverage")

    return aligned
