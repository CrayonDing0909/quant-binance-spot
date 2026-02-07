from __future__ import annotations
from pathlib import Path
import pandas as pd


def save_klines(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=True)


def load_klines(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)
