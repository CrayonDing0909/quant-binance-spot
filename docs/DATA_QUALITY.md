# 数据质量检查使用指南

## 概述

数据质量检查模块提供了完整的数据验证和清洗功能，确保回测数据的准确性和可靠性。

## 数据验证

### 基本使用

```python
from qtrade.data import validate_data_quality

# 验证数据质量
report = validate_data_quality(df)

# 检查结果
if report.is_valid:
    print("数据质量良好")
else:
    print("数据质量问题:")
    for error in report.errors:
        print(f"  ❌ {error}")
    for warning in report.warnings:
        print(f"  ⚠️  {warning}")

# 查看详细报告
print(f"总行数: {report.total_rows}")
print(f"缺失值比例: {report.missing_pct:.2f}%")
print(f"异常值比例: {report.outlier_pct:.2f}%")
print(f"时间间隔数: {len(report.gaps)}")
```

### 自定义检查器

```python
from qtrade.data import DataQualityChecker

checker = DataQualityChecker(
    outlier_threshold=3.0,        # Z-score 阈值
    max_price_change_pct=0.5,     # 最大价格变化 50%
    min_volume=0.0                # 最小成交量
)

report = checker.validate(df, expected_columns=["open", "high", "low", "close", "volume"])
```

## 数据清洗

### 基本使用

```python
from qtrade.data import clean_data

# 清洗数据
cleaned_df = clean_data(
    df,
    fill_method="forward",      # 填充方法: forward, backward, interpolate
    remove_outliers=False,      # 是否移除异常值
    remove_duplicates=True      # 是否移除重复时间戳
)
```

### 高级清洗

```python
from qtrade.data import DataQualityChecker

checker = DataQualityChecker(
    outlier_threshold=3.0,
    max_price_change_pct=0.5
)

# 先验证
report = checker.validate(df)

# 根据问题选择清洗方法
if report.issues.get(DataQualityIssue.OUTLIERS, 0) > 0:
    # 如果有异常值，移除它们
    cleaned_df = checker.clean(df, remove_outliers=True)
else:
    # 否则只填充缺失值
    cleaned_df = checker.clean(df, remove_outliers=False)
```

## 检查的问题类型

数据质量检查器会检查以下问题：

1. **缺失值 (MISSING_VALUES)**: 检查必需列是否有缺失值
2. **重复时间戳 (DUPLICATE_TIMESTAMPS)**: 检查是否有重复的时间戳
3. **异常值 (OUTLIERS)**: 使用 Z-score 检测异常值
4. **时间间隔 (GAPS)**: 检测时间序列中的间隔
5. **无效价格 (INVALID_PRICES)**: 检查价格是否有效（负数、NaN、Inf，以及 OHLC 逻辑）
6. **零成交量 (VOLUME_ZERO)**: 检查是否有零成交量的 K 线
7. **价格异常下跌 (PRICE_DECREASE)**: 检测价格异常下跌（可能是数据错误）

## 在回测中使用

### 自动验证和清洗

在配置文件中启用：

```yaml
# config/base.yaml
backtest:
  validate_data: true   # 自动验证数据质量
  clean_data: true       # 自动清洗数据
```

回测框架会自动验证和清洗数据：

```python
from qtrade.backtest import run_symbol_backtest

# 数据会自动验证和清洗
result = run_symbol_backtest(
    symbol="BTCUSDT",
    data_path=data_path,
    cfg=cfg,
    validate_data=True,      # 验证数据（默认 True）
    clean_data_before=True   # 清洗数据（默认 True）
)
```

### 手动验证和清洗

```python
from qtrade.data import validate_data_quality, clean_data
from qtrade.data.storage import load_klines

# 加载数据
df = load_klines(data_path)

# 验证
report = validate_data_quality(df)
if not report.is_valid:
    print("数据质量问题，进行清洗...")
    df = clean_data(df)

# 继续使用清洗后的数据
```

## 数据质量报告

数据质量报告包含以下信息：

- `total_rows`: 总行数
- `issues`: 各种问题的数量
- `cleaned_rows`: 清洗后的行数
- `missing_pct`: 缺失值百分比
- `outlier_pct`: 异常值百分比
- `gaps`: 时间间隔列表
- `is_valid`: 数据是否有效
- `warnings`: 警告列表
- `errors`: 错误列表

## 最佳实践

1. **始终验证数据**: 在回测前验证数据质量
2. **谨慎处理异常值**: 不要盲目移除异常值，可能是真实的市场波动
3. **检查时间间隔**: 确保数据没有大的时间间隔
4. **验证价格逻辑**: 确保 OHLC 数据符合逻辑（high >= low, high >= open/close 等）
5. **记录清洗过程**: 记录清洗了哪些数据，以便后续分析

## 示例

```python
from qtrade.data import validate_data_quality, clean_data, DataQualityChecker

# 1. 基本验证
report = validate_data_quality(df)
print(f"数据有效性: {report.is_valid}")
print(f"缺失值: {report.missing_pct:.2f}%")

# 2. 自定义检查器
checker = DataQualityChecker(
    outlier_threshold=2.5,  # 更严格的异常值检测
    max_price_change_pct=0.3    # 更严格的价格变化限制
)
report = checker.validate(df)

# 3. 清洗数据
cleaned_df = clean_data(
    df,
    fill_method="interpolate",  # 使用插值填充
    remove_outliers=False,       # 保留异常值
    remove_duplicates=True       # 移除重复
)

# 4. 验证清洗后的数据
final_report = validate_data_quality(cleaned_df)
print(f"清洗后数据有效性: {final_report.is_valid}")
```

