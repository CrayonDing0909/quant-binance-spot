"""
數據質量檢查模組

提供數據質量驗證和清洗功能：
- 數據完整性驗證
- 異常值檢測
- 數據清洗
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Tuple
import pandas as pd
import numpy as np
from enum import Enum


class DataQualityIssue(Enum):
    """數據質量問題類型"""
    MISSING_VALUES = "missing_values"
    DUPLICATE_TIMESTAMPS = "duplicate_timestamps"
    OUTLIERS = "outliers"
    GAPS = "gaps"
    INVALID_PRICES = "invalid_prices"
    VOLUME_ZERO = "volume_zero"
    PRICE_DECREASE = "price_decrease"  # 價格異常下跌（可能是數據錯誤）


@dataclass
class DataQualityReport:
    """數據質量報告"""
    total_rows: int
    issues: dict[DataQualityIssue, int]
    cleaned_rows: int
    missing_pct: float
    outlier_pct: float
    gaps: List[pd.Timestamp]
    is_valid: bool
    warnings: List[str]
    errors: List[str]


class DataQualityChecker:
    """數據質量檢查器"""
    
    def __init__(
        self,
        outlier_threshold: float = 3.0,  # Z-score 閾值
        max_price_change_pct: float = 0.5,  # 最大價格變化百分比（50%）
        min_volume: float = 0.0,  # 最小成交量
    ):
        """
        Args:
            outlier_threshold: 異常值檢測的 Z-score 閾值
            max_price_change_pct: 單根 K 線最大價格變化百分比
            min_volume: 最小成交量閾值
        """
        self.outlier_threshold = outlier_threshold
        self.max_price_change_pct = max_price_change_pct
        self.min_volume = min_volume
    
    def validate(
        self,
        df: pd.DataFrame,
        expected_columns: Optional[List[str]] = None
    ) -> DataQualityReport:
        """
        驗證數據質量
        
        Args:
            df: K線數據 DataFrame
            expected_columns: 期望的列名，預設 ["open", "high", "low", "close", "volume"]
        
        Returns:
            數據質量報告
        """
        if expected_columns is None:
            expected_columns = ["open", "high", "low", "close", "volume"]
        
        issues = {}
        warnings = []
        errors = []
        gaps = []
        
        # 1. 檢查必需列
        missing_columns = set(expected_columns) - set(df.columns)
        if missing_columns:
            errors.append(f"缺少必需列: {missing_columns}")
            return DataQualityReport(
                total_rows=len(df),
                issues={},
                cleaned_rows=0,
                missing_pct=100.0,
                outlier_pct=0.0,
                gaps=[],
                is_valid=False,
                warnings=warnings,
                errors=errors,
            )
        
        # 2. 檢查缺失值
        missing_counts = df[expected_columns].isnull().sum()
        total_missing = missing_counts.sum()
        issues[DataQualityIssue.MISSING_VALUES] = total_missing
        missing_pct = (total_missing / (len(df) * len(expected_columns))) * 100
        
        if total_missing > 0:
            warnings.append(f"發現 {total_missing} 個缺失值 ({missing_pct:.2f}%)")
        
        # 3. 檢查重複時間戳
        if df.index.duplicated().any():
            dup_count = df.index.duplicated().sum()
            issues[DataQualityIssue.DUPLICATE_TIMESTAMPS] = dup_count
            warnings.append(f"發現 {dup_count} 個重複時間戳")
        
        # 4. 檢查時間序列間隔
        gaps = self._detect_gaps(df)
        if gaps:
            issues[DataQualityIssue.GAPS] = len(gaps)
            warnings.append(f"發現 {len(gaps)} 個時間間隔")
        
        # 5. 檢查價格有效性
        invalid_prices = self._check_invalid_prices(df)
        if invalid_prices > 0:
            issues[DataQualityIssue.INVALID_PRICES] = invalid_prices
            errors.append(f"發現 {invalid_prices} 個無效價格")
        
        # 6. 檢查異常值
        outliers = self._detect_outliers(df)
        issues[DataQualityIssue.OUTLIERS] = outliers
        outlier_pct = (outliers / len(df)) * 100 if len(df) > 0 else 0.0
        
        if outliers > 0:
            warnings.append(f"發現 {outliers} 個異常值 ({outlier_pct:.2f}%)")
        
        # 7. 檢查成交量
        zero_volume = (df["volume"] <= self.min_volume).sum()
        if zero_volume > 0:
            issues[DataQualityIssue.VOLUME_ZERO] = zero_volume
            warnings.append(f"發現 {zero_volume} 根零成交量 K 線")
        
        # 8. 檢查價格異常變化
        price_decrease = self._check_price_decrease(df)
        if price_decrease > 0:
            issues[DataQualityIssue.PRICE_DECREASE] = price_decrease
            warnings.append(f"發現 {price_decrease} 根價格異常下跌的 K 線")
        
        # 判斷數據是否有效
        is_valid = (
            len(errors) == 0 and
            missing_pct < 10.0 and  # 缺失值少於 10%
            invalid_prices == 0
        )
        
        return DataQualityReport(
            total_rows=len(df),
            issues=issues,
            cleaned_rows=len(df) - total_missing,
            missing_pct=missing_pct,
            outlier_pct=outlier_pct,
            gaps=gaps,
            is_valid=is_valid,
            warnings=warnings,
            errors=errors,
        )
    
    def _detect_gaps(self, df: pd.DataFrame) -> List[pd.Timestamp]:
        """檢測時間序列中的間隔"""
        if len(df) < 2:
            return []
        
        gaps = []
        index = df.index
        
        # 計算預期的時間間隔（使用最常見的間隔）
        if len(index) > 1:
            intervals = index[1:] - index[:-1]
            
            # 使用 value_counts() 獲取最常見的間隔，如果沒有則使用中位數
            interval_counts = intervals.value_counts()
            if len(interval_counts) > 0:
                expected_interval = interval_counts.index[0]  # 最常見的間隔
            else:
                expected_interval = intervals.median()
            
            # 查找大於預期間隔 2 倍的間隔
            # intervals 是 TimedeltaIndex，直接使用索引訪問
            for i in range(len(intervals)):
                if intervals[i] > expected_interval * 2:
                    gaps.append(index[i + 1])
        
        return gaps
    
    def _check_invalid_prices(self, df: pd.DataFrame) -> int:
        """檢查無效價格（負數、NaN、Inf）"""
        price_columns = ["open", "high", "low", "close"]
        invalid_count = 0
        
        for col in price_columns:
            if col in df.columns:
                invalid = (
                    (df[col] <= 0) |
                    df[col].isnull() |
                    np.isinf(df[col])
                )
                invalid_count += invalid.sum()
        
        # 檢查 high >= low, high >= open, high >= close, low <= open, low <= close
        if all(col in df.columns for col in price_columns):
            invalid_ohlc = (
                (df["high"] < df["low"]) |
                (df["high"] < df["open"]) |
                (df["high"] < df["close"]) |
                (df["low"] > df["open"]) |
                (df["low"] > df["close"])
            )
            invalid_count += invalid_ohlc.sum()
        
        return invalid_count
    
    def _detect_outliers(self, df: pd.DataFrame) -> int:
        """使用 Z-score 檢測異常值"""
        if "close" not in df.columns or len(df) < 10:
            return 0
        
        close = df["close"]
        z_scores = np.abs((close - close.mean()) / close.std())
        outliers = (z_scores > self.outlier_threshold).sum()
        
        return outliers
    
    def _check_price_decrease(self, df: pd.DataFrame) -> int:
        """檢查價格異常下跌（可能是數據錯誤）"""
        if "close" not in df.columns or len(df) < 2:
            return 0
        
        price_change = df["close"].pct_change()
        abnormal_decrease = (price_change < -self.max_price_change_pct).sum()
        
        return abnormal_decrease
    
    def clean(
        self,
        df: pd.DataFrame,
        fill_method: str = "forward",
        remove_outliers: bool = False,
        remove_duplicates: bool = True
    ) -> pd.DataFrame:
        """
        清洗數據
        
        Args:
            df: 原始數據
            fill_method: 填充方法，"forward", "backward", "interpolate"
            remove_outliers: 是否移除異常值
            remove_duplicates: 是否移除重複時間戳
        
        Returns:
            清洗後的數據
        """
        cleaned_df = df.copy()
        
        # 1. 移除重複時間戳
        if remove_duplicates:
            cleaned_df = cleaned_df[~cleaned_df.index.duplicated(keep="last")]
        
        # 2. 填充缺失值
        if fill_method == "forward":
            cleaned_df = cleaned_df.ffill()
        elif fill_method == "backward":
            cleaned_df = cleaned_df.bfill()
        elif fill_method == "interpolate":
            cleaned_df = cleaned_df.interpolate(method="linear")
        
        # 如果還有缺失值，使用前向填充和後向填充
        cleaned_df = cleaned_df.ffill().bfill()
        
        # 3. 移除異常值
        if remove_outliers and "close" in cleaned_df.columns:
            close = cleaned_df["close"]
            z_scores = np.abs((close - close.mean()) / close.std())
            cleaned_df = cleaned_df[z_scores <= self.outlier_threshold]
        
        # 4. 移除無效價格
        price_columns = ["open", "high", "low", "close"]
        if all(col in cleaned_df.columns for col in price_columns):
            valid_mask = (
                (cleaned_df["high"] >= cleaned_df["low"]) &
                (cleaned_df["high"] >= cleaned_df["open"]) &
                (cleaned_df["high"] >= cleaned_df["close"]) &
                (cleaned_df["low"] <= cleaned_df["open"]) &
                (cleaned_df["low"] <= cleaned_df["close"]) &
                (cleaned_df[price_columns] > 0).all(axis=1)
            )
            cleaned_df = cleaned_df[valid_mask]
        
        return cleaned_df


def validate_data_quality(
    df: pd.DataFrame,
    expected_columns: Optional[List[str]] = None
) -> DataQualityReport:
    """
    驗證數據質量的便捷函數
    
    Args:
        df: K線數據 DataFrame
        expected_columns: 期望的列名
    
    Returns:
        數據質量報告
    """
    checker = DataQualityChecker()
    return checker.validate(df, expected_columns)


def clean_data(
    df: pd.DataFrame,
    fill_method: str = "forward",
    remove_outliers: bool = False,
    remove_duplicates: bool = True
) -> pd.DataFrame:
    """
    清洗數據的便捷函數
    
    Args:
        df: 原始數據
        fill_method: 填充方法
        remove_outliers: 是否移除異常值
        remove_duplicates: 是否移除重複時間戳
    
    Returns:
        清洗後的數據
    """
    checker = DataQualityChecker()
    return checker.clean(df, fill_method, remove_outliers, remove_duplicates)
