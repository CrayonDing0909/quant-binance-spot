"""
鏈上數據模組

支援：
    1. DeFi Llama — TVL、Stablecoin 流動性（免費，無需 API key）
    2. CryptoQuant (free tier) — Exchange Reserve（需 API key）
    3. Glassnode (free tier) — BTC/ETH 基礎鏈上指標（需 API key）

這些數據主要作為 Regime Indicator（風險偏好、宏觀環境），
不適合高頻信號（延遲 1-10 分鐘 ~ 數小時）。

使用方式：
    from qtrade.data.onchain import (
        load_onchain,
        save_onchain,
        align_onchain_to_klines,
        compute_onchain_coverage,
    )
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_BASE_DIR = Path("data/onchain")


def save_onchain(
    data: pd.Series | pd.DataFrame,
    provider: str,
    metric: str,
    data_dir: Path = _BASE_DIR,
) -> Path:
    """儲存鏈上數據到 parquet"""
    path = data_dir / provider / f"{metric}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, pd.Series):
        data = data.to_frame()
    data.to_parquet(path, index=True)
    logger.info(f"💾 Saved {provider}/{metric}: {len(data)} rows → {path}")
    return path


def load_onchain(
    provider: str,
    metric: str,
    data_dir: Path = _BASE_DIR,
) -> Optional[pd.DataFrame]:
    """載入鏈上數據"""
    path = data_dir / provider / f"{metric}.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        if not df.empty:
            logger.debug(f"📂 On-chain loaded: {provider}/{metric} ({len(df)} rows)")
        return df
    except Exception as e:
        logger.warning(f"⚠️  On-chain load failed ({provider}/{metric}): {e}")
        return None


def get_onchain_path(data_dir: Path, provider: str, metric: str) -> Path:
    """取得鏈上數據標準路徑"""
    return data_dir / provider / f"{metric}.parquet"


def align_onchain_to_klines(
    onchain_data: pd.Series | pd.DataFrame | None,
    kline_index: pd.DatetimeIndex,
    max_ffill_bars: int = 24,
) -> pd.Series | pd.DataFrame | None:
    """
    將鏈上數據對齊到 K 線時間軸

    鏈上數據通常是 daily 頻率，所以 max_ffill_bars 預設較高（24 bars = 1d for 1h klines）。
    嚴格因果：只使用 forward-fill，不使用未來資訊。

    Args:
        onchain_data: 鏈上數據（Series 或 DataFrame）
        kline_index: K 線時間索引
        max_ffill_bars: 最大 forward-fill bars（daily 數據對齊到 1h 時建議 24）

    Returns:
        對齊後的 Series / DataFrame
    """
    if onchain_data is None:
        return None

    if isinstance(onchain_data, pd.DataFrame) and onchain_data.empty:
        return None
    if isinstance(onchain_data, pd.Series) and onchain_data.empty:
        return None

    data = onchain_data.copy()

    # Timezone alignment
    if kline_index.tz is None and data.index.tz is not None:
        data.index = data.index.tz_localize(None)
    elif kline_index.tz is not None and data.index.tz is None:
        data.index = data.index.tz_localize(kline_index.tz)

    aligned = data.reindex(kline_index, method="ffill", limit=max_ffill_bars)

    if isinstance(aligned, pd.Series):
        n_missing = aligned.isna().sum()
        n_total = len(aligned)
    else:
        n_missing = aligned.isna().all(axis=1).sum()
        n_total = len(aligned)

    if n_missing > 0:
        coverage = (n_total - n_missing) / n_total * 100
        logger.info(f"📊 On-chain alignment: {n_total - n_missing}/{n_total} ({coverage:.1f}%)")

    return aligned


def compute_onchain_coverage(
    provider: str = "defillama",
    data_dir: Path = _BASE_DIR,
) -> pd.DataFrame:
    """計算鏈上數據覆蓋率"""
    provider_dir = data_dir / provider
    if not provider_dir.exists():
        return pd.DataFrame()

    rows = []
    for f in sorted(provider_dir.glob("*.parquet")):
        try:
            df = pd.read_parquet(f)
            metric = f.stem
            if df.empty:
                rows.append({"metric": metric, "rows": 0, "start": None, "end": None, "days": 0})
            else:
                days = (df.index[-1] - df.index[0]).days if hasattr(df.index, '__getitem__') else 0
                rows.append({
                    "metric": metric,
                    "rows": len(df),
                    "start": str(df.index[0])[:10] if len(df) > 0 else None,
                    "end": str(df.index[-1])[:10] if len(df) > 0 else None,
                    "days": days,
                })
        except Exception as e:
            rows.append({"metric": f.stem, "rows": 0, "start": None, "end": None, "days": 0})

    return pd.DataFrame(rows)
