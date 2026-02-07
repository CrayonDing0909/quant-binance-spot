"""
数据质量检查模块

提供数据质量验证和清洗功能：
- 数据完整性验证
- 异常值检测
- 数据清洗
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Tuple
import pandas as pd
import numpy as np
from enum import Enum


class DataQualityIssue(Enum):
    """数据质量问题类型"""
    MISSING_VALUES = "missing_values"
    DUPLICATE_TIMESTAMPS = "duplicate_timestamps"
    OUTLIERS = "outliers"
    GAPS = "gaps"
    INVALID_PRICES = "invalid_prices"
    VOLUME_ZERO = "volume_zero"
    PRICE_DECREASE = "price_decrease"  # 价格异常下跌（可能是数据错误）


@dataclass
class DataQualityReport:
    """数据质量报告"""
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
    """数据质量检查器"""
    
    def __init__(
        self,
        outlier_threshold: float = 3.0,  # Z-score 阈值
        max_price_change_pct: float = 0.5,  # 最大价格变化百分比（50%）
        min_volume: float = 0.0,  # 最小成交量
    ):
        """
        Args:
            outlier_threshold: 异常值检测的 Z-score 阈值
            max_price_change_pct: 单根 K 线最大价格变化百分比
            min_volume: 最小成交量阈值
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
        验证数据质量
        
        Args:
            df: K线数据 DataFrame
            expected_columns: 期望的列名，默认 ["open", "high", "low", "close", "volume"]
        
        Returns:
            数据质量报告
        """
        if expected_columns is None:
            expected_columns = ["open", "high", "low", "close", "volume"]
        
        issues = {}
        warnings = []
        errors = []
        gaps = []
        
        # 1. 检查必需列
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
        
        # 2. 检查缺失值
        missing_counts = df[expected_columns].isnull().sum()
        total_missing = missing_counts.sum()
        issues[DataQualityIssue.MISSING_VALUES] = total_missing
        missing_pct = (total_missing / (len(df) * len(expected_columns))) * 100
        
        if total_missing > 0:
            warnings.append(f"发现 {total_missing} 个缺失值 ({missing_pct:.2f}%)")
        
        # 3. 检查重复时间戳
        if df.index.duplicated().any():
            dup_count = df.index.duplicated().sum()
            issues[DataQualityIssue.DUPLICATE_TIMESTAMPS] = dup_count
            warnings.append(f"发现 {dup_count} 个重复时间戳")
        
        # 4. 检查时间序列间隔
        gaps = self._detect_gaps(df)
        if gaps:
            issues[DataQualityIssue.GAPS] = len(gaps)
            warnings.append(f"发现 {len(gaps)} 个时间间隔")
        
        # 5. 检查价格有效性
        invalid_prices = self._check_invalid_prices(df)
        if invalid_prices > 0:
            issues[DataQualityIssue.INVALID_PRICES] = invalid_prices
            errors.append(f"发现 {invalid_prices} 个无效价格")
        
        # 6. 检查异常值
        outliers = self._detect_outliers(df)
        issues[DataQualityIssue.OUTLIERS] = outliers
        outlier_pct = (outliers / len(df)) * 100 if len(df) > 0 else 0.0
        
        if outliers > 0:
            warnings.append(f"发现 {outliers} 个异常值 ({outlier_pct:.2f}%)")
        
        # 7. 检查成交量
        zero_volume = (df["volume"] <= self.min_volume).sum()
        if zero_volume > 0:
            issues[DataQualityIssue.VOLUME_ZERO] = zero_volume
            warnings.append(f"发现 {zero_volume} 根零成交量 K 线")
        
        # 8. 检查价格异常变化
        price_decrease = self._check_price_decrease(df)
        if price_decrease > 0:
            issues[DataQualityIssue.PRICE_DECREASE] = price_decrease
            warnings.append(f"发现 {price_decrease} 根价格异常下跌的 K 线")
        
        # 判断数据是否有效
        is_valid = (
            len(errors) == 0 and
            missing_pct < 10.0 and  # 缺失值少于 10%
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
        """检测时间序列中的间隔"""
        if len(df) < 2:
            return []
        
        gaps = []
        index = df.index
        
        # 计算预期的时间间隔（使用最常见的间隔）
        if len(index) > 1:
            intervals = index[1:] - index[:-1]
            
            # 使用 value_counts() 获取最常见的间隔，如果没有则使用中位数
            interval_counts = intervals.value_counts()
            if len(interval_counts) > 0:
                expected_interval = interval_counts.index[0]  # 最常见的间隔
            else:
                expected_interval = intervals.median()
            
            # 查找大于预期间隔 2 倍的间隔
            # intervals 是 TimedeltaIndex，直接使用索引访问
            for i in range(len(intervals)):
                if intervals[i] > expected_interval * 2:
                    gaps.append(index[i + 1])
        
        return gaps
    
    def _check_invalid_prices(self, df: pd.DataFrame) -> int:
        """检查无效价格（负数、NaN、Inf）"""
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
        
        # 检查 high >= low, high >= open, high >= close, low <= open, low <= close
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
        """使用 Z-score 检测异常值"""
        if "close" not in df.columns or len(df) < 10:
            return 0
        
        close = df["close"]
        z_scores = np.abs((close - close.mean()) / close.std())
        outliers = (z_scores > self.outlier_threshold).sum()
        
        return outliers
    
    def _check_price_decrease(self, df: pd.DataFrame) -> int:
        """检查价格异常下跌（可能是数据错误）"""
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
        清洗数据
        
        Args:
            df: 原始数据
            fill_method: 填充方法，"forward", "backward", "interpolate"
            remove_outliers: 是否移除异常值
            remove_duplicates: 是否移除重复时间戳
        
        Returns:
            清洗后的数据
        """
        cleaned_df = df.copy()
        
        # 1. 移除重复时间戳
        if remove_duplicates:
            cleaned_df = cleaned_df[~cleaned_df.index.duplicated(keep="last")]
        
        # 2. 填充缺失值
        if fill_method == "forward":
            cleaned_df = cleaned_df.ffill()
        elif fill_method == "backward":
            cleaned_df = cleaned_df.bfill()
        elif fill_method == "interpolate":
            cleaned_df = cleaned_df.interpolate(method="linear")
        
        # 如果还有缺失值，使用前向填充和后向填充
        cleaned_df = cleaned_df.ffill().bfill()
        
        # 3. 移除异常值
        if remove_outliers and "close" in cleaned_df.columns:
            close = cleaned_df["close"]
            z_scores = np.abs((close - close.mean()) / close.std())
            cleaned_df = cleaned_df[z_scores <= self.outlier_threshold]
        
        # 4. 移除无效价格
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
    验证数据质量的便捷函数
    
    Args:
        df: K线数据 DataFrame
        expected_columns: 期望的列名
    
    Returns:
        数据质量报告
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
    清洗数据的便捷函数
    
    Args:
        df: 原始数据
        fill_method: 填充方法
        remove_outliers: 是否移除异常值
        remove_duplicates: 是否移除重复时间戳
    
    Returns:
        清洗后的数据
    """
    checker = DataQualityChecker()
    return checker.clean(df, fill_method, remove_outliers, remove_duplicates)

