"""
清算/爆倉數據模組

支援：
    1. Binance /fapi/v1/allForceOrders — 最近 ~7 天清算訂單
    2. Coinglass — 歷史聚合清算數據（需 COINGLASS_API_KEY）

衍生指標：
    - liq_volume_long: 多頭清算量 (USDT)
    - liq_volume_short: 空頭清算量 (USDT)
    - liq_imbalance: 清算不平衡 [-1, 1]（正=空頭被清算多，看多）
    - liq_cascade_z: 清算瀑布 z-score

使用方式：
    from qtrade.data.liquidation import (
        load_liquidation,
        align_liquidation_to_klines,
    )
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_BASE_DIR = Path("data/binance/futures/liquidation")


def load_liquidation(
    symbol: str,
    data_dir: Path = _BASE_DIR,
) -> Optional[pd.DataFrame]:
    """載入清算數據（由 scripts/fetch_liquidation_data.py 下載）"""
    path = data_dir / f"{symbol}.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        if not df.empty:
            logger.debug(f"📂 Liquidation loaded: {symbol} ({len(df)} rows)")
        return df
    except Exception as e:
        logger.warning(f"⚠️  Liquidation load failed ({symbol}): {e}")
        return None


def get_liquidation_path(data_dir: Path, symbol: str) -> Path:
    """取得清算數據標準路徑"""
    return data_dir / f"{symbol}.parquet"


def align_liquidation_to_klines(
    liq_df: pd.DataFrame | None,
    kline_index: pd.DatetimeIndex,
    columns: list[str] | None = None,
    max_ffill_bars: int = 2,
) -> pd.DataFrame | None:
    """
    將清算數據對齊到 K 線時間軸

    Args:
        liq_df: 清算 DataFrame
        kline_index: K 線時間索引
        columns: 要對齊的欄位（預設全部）
        max_ffill_bars: 最大 forward-fill bars

    Returns:
        對齊後的 DataFrame
    """
    if liq_df is None or liq_df.empty:
        return None

    df = liq_df.copy()

    # Timezone alignment
    if kline_index.tz is None and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    elif kline_index.tz is not None and df.index.tz is None:
        df.index = df.index.tz_localize(kline_index.tz)

    if columns:
        df = df[[c for c in columns if c in df.columns]]

    aligned = df.reindex(kline_index, method="ffill", limit=max_ffill_bars)
    aligned = aligned.fillna(0)  # 無清算 = 0

    n_nonzero = (aligned.sum(axis=1) > 0).sum()
    logger.info(f"📊 Liquidation alignment: {n_nonzero}/{len(aligned)} bars with data")

    return aligned
