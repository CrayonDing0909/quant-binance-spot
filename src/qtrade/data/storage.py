from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
import pandas as pd


def save_klines(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=True)


def load_klines(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def get_local_data_range(path: Path) -> tuple[Optional[datetime], Optional[datetime]]:
    """
    取得本地數據的時間範圍
    
    Returns:
        (最早時間, 最晚時間) 或 (None, None) 若檔案不存在
    """
    if not path.exists():
        return None, None
    
    try:
        df = pd.read_parquet(path)
        if df.empty:
            return None, None
        
        # index 是 open_time
        start = df.index.min()
        end = df.index.max()
        
        # 確保是 timezone-aware
        if start.tzinfo is None:
            start = start.tz_localize("UTC")
        if end.tzinfo is None:
            end = end.tz_localize("UTC")
            
        return start.to_pydatetime(), end.to_pydatetime()
    except Exception:
        return None, None


def merge_klines(existing_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """
    合併新舊數據，去重並排序
    
    Args:
        existing_df: 現有數據
        new_df: 新下載的數據
        
    Returns:
        合併後的 DataFrame
    """
    if existing_df.empty:
        return new_df
    if new_df.empty:
        return existing_df
    
    # 合併
    combined = pd.concat([existing_df, new_df])
    
    # 去重（保留最新的）
    combined = combined[~combined.index.duplicated(keep='last')]
    
    # 排序
    combined = combined.sort_index()
    
    return combined
