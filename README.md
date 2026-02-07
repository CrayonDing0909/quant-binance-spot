# quant-binance-spot

Research/backtest with vectorbt, later extend to live trading on Binance Spot.

## 项目架构

```
quant-binance-spot/
├── config/              # 配置文件
│   ├── base.yaml        # 基础配置
│   └── dev.yaml         # 开发环境配置
├── data/                # 数据存储
│   └── binance/spot/    # Binance 现货数据
├── reports/             # 回测报告和图表
├── scripts/             # 脚本
│   ├── download_data.py      # 下载数据
│   ├── run_backtest.py        # 运行回测
│   ├── validate_strategy.py   # 策略验证（过拟合检测）
│   ├── create_strategy.py     # 策略模板生成器
│   └── optimize_params.py     # 参数优化工具
└── src/qtrade/
    ├── backtest/        # 回测模块
    │   ├── run_backtest.py    # 回测核心逻辑
    │   ├── metrics.py          # 指标计算
    │   ├── plotting.py         # 图表绘制
    │   └── validation.py       # 过拟合验证
    ├── indicators/      # 指标库
    │   ├── __init__.py         # 指标导出
    │   ├── rsi.py              # RSI 指标
    │   ├── macd.py             # MACD 指标
    │   ├── bollinger.py        # 布林带指标
    │   └── moving_average.py   # 移动平均线
    ├── strategy/        # 策略模块
    │   ├── base.py             # 策略基类（支持状态管理）
    │   ├── ema_cross.py        # EMA交叉策略示例
    │   ├── rsi_strategy.py     # RSI 策略
    │   ├── smc_strategy.py     # SMC 策略
    │   └── __init__.py         # 策略注册
    ├── data/            # 数据模块
    └── config.py        # 配置加载
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env

# Download data
python scripts/download_data.py

# Run backtest
python scripts/run_backtest.py

# Validate strategy (overfitting detection)
python scripts/validate_strategy.py

# Create new strategy (template generator)
python scripts/create_strategy.py --name my_strategy --type rsi

# Optimize strategy parameters
python scripts/optimize_params.py --strategy rsi
```

## 策略类型示例

项目已包含多种策略类型的示例：

### RSI 策略

**配置文件 (`config/rsi_example.yaml`):**
```yaml
strategy:
  name: "rsi"
  params:
    period: 14
    oversold: 30
    overbought: 70
```

**运行:**
```bash
python scripts/run_backtest.py --config config/rsi_example.yaml
```

### SMC (Smart Money Concept) 策略

**配置文件 (`config/smc_example.yaml`):**
```yaml
strategy:
  name: "smc_basic"
  params:
    order_block_lookback: 20
    liquidity_lookback: 50
```

**运行:**
```bash
python scripts/run_backtest.py --config config/smc_example.yaml
```

### EMA 交叉策略（默认）

**配置文件 (`config/base.yaml`):**
```yaml
strategy:
  name: "ema_cross"
  params:
    fast: 20
    slow: 60
```

## 快速开始

### 使用模板生成器创建策略（推荐）

```bash
# 创建 RSI 策略
python scripts/create_strategy.py --name my_rsi --type rsi

# 创建 EMA 策略
python scripts/create_strategy.py --name my_ema --type ema

# 创建自定义策略
python scripts/create_strategy.py --name my_strategy --type custom
```

### 使用指标库

所有策略都可以使用统一的指标库：

```python
from qtrade.indicators import calculate_rsi, calculate_macd, calculate_bollinger_bands

# 在策略中使用
rsi = calculate_rsi(df["close"], period=14)
macd_data = calculate_macd(df["close"], fast_period=12, slow_period=26)
bb = calculate_bollinger_bands(df["close"], period=20, std_mult=2.0)
```

### 优化策略参数

```bash
# 优化 RSI 策略参数
python scripts/optimize_params.py --strategy rsi

# 指定优化指标
python scripts/optimize_params.py --strategy ema_cross --metric "Sharpe Ratio"
```

## 开发新策略

### 方法 1: 使用模板生成器（推荐）

```bash
python scripts/create_strategy.py --name my_strategy --type rsi
```

### 方法 2: 手动创建策略文件

在 `src/qtrade/strategy/` 目录下创建新的策略文件，例如 `my_strategy.py`:

```python
from __future__ import annotations
import pandas as pd
from .base import StrategyContext
from . import register_strategy


@register_strategy("my_strategy")
def generate_positions(df: pd.DataFrame, ctx: StrategyContext, params: dict) -> pd.Series:
    """
    生成持仓信号
    
    Args:
        df: K线数据，包含 open, high, low, close, volume 等列
        ctx: 策略上下文（包含 symbol 等信息）
        params: 策略参数（从 config 中读取）
    
    Returns:
        pd.Series: 持仓比例序列，值在 [0, 1] 之间
        - 1.0 = 满仓
        - 0.0 = 空仓
    """
    # 使用指标库
    from qtrade.indicators import calculate_ema
    
    close = df["close"]
    
    # 使用指标库计算 EMA
    ema_fast = calculate_ema(close, params.get("fast", 20))
    ema_slow = calculate_ema(close, params.get("slow", 60))
    
    # 生成信号（在收盘时）
    signal = (ema_fast > ema_slow).astype(float)
    
    # 重要：避免未来信息泄露，将信号向后移动1个bar
    # 这样信号在 t 时刻生成，在 t+1 时刻执行
    pos = signal.shift(1).fillna(0.0)
    
    return pos.clip(0.0, 1.0)
```

### 2. 在策略模块中导入

编辑 `src/qtrade/strategy/__init__.py`，添加导入：

```python
from . import ema_cross  # noqa: E402
from . import my_strategy  # noqa: E402  # 添加这行
```

### 3. 配置策略

编辑 `config/base.yaml`:

```yaml
strategy:
  name: "my_strategy"  # 使用注册的策略名称
  params:
    fast: 20
    slow: 60
```

### 4. 运行回测

```bash
python scripts/run_backtest.py
```

回测结果会保存在 `reports/` 目录：
- `stats_{SYMBOL}.csv` - 统计指标
- `equity_curve_{SYMBOL}.png` - 资金曲线图

## 验证策略（过拟合检测）

### 1. 滚动窗口验证 (Walk-Forward Analysis)

将数据分成多个训练/测试窗口，观察策略在样本外的表现：

```bash
python scripts/validate_strategy.py
```

这会生成：
- `walk_forward_{SYMBOL}.csv` - 每个窗口的训练/测试结果

**如何判断过拟合：**
- 训练集收益率 >> 测试集收益率（下降 >30%）
- 训练集夏普比率 >> 测试集夏普比率
- 测试集回撤明显增加

### 2. 参数敏感性分析

测试不同参数组合，观察策略稳定性：

生成 `parameter_sensitivity_{SYMBOL}.csv`，包含所有参数组合的结果。

**如何判断过拟合：**
- 参数微小变化导致收益率大幅波动
- 收益率标准差 > 50%

### 3. 修正过拟合的方法

1. **简化策略**：减少参数数量，避免过度优化
2. **增加样本外测试**：使用更长的历史数据，更多的时间窗口
3. **正则化**：添加交易成本、滑点等约束
4. **避免数据挖掘偏差**：不要在同一个数据集上反复优化
5. **使用更保守的参数**：选择在多个时间窗口都表现稳定的参数

## 查看资金曲线

运行回测后，会在 `reports/` 目录生成 `equity_curve_{SYMBOL}.png`，包含：

1. **价格和信号图**：显示价格走势和买卖点
2. **持仓比例图**：显示持仓变化
3. **资金曲线图**：显示账户价值变化
4. **回撤图**：显示最大回撤

## 新功能

### 风险管理模块

提供完整的风险管理功能，包括仓位管理、风险限制和组合风险控制。

```python
from qtrade.risk import FixedPositionSizer, RiskLimits, apply_risk_limits

# 仓位管理
sizer = FixedPositionSizer(position_pct=0.8)
size = sizer.calculate_size(signal=1.0, equity=10000, price=50000)

# 风险限制
limits = RiskLimits(max_position_pct=1.0, max_drawdown_pct=0.5)
adjusted_pos, checks = apply_risk_limits(position_pct=1.0, equity_curve=equity_series, limits=limits)
```

详细文档: [风险管理指南](docs/RISK_MANAGEMENT.md)

### 数据质量检查

自动验证和清洗数据，确保回测数据的准确性。

```python
from qtrade.data import validate_data_quality, clean_data

# 验证数据质量
report = validate_data_quality(df)
if not report.is_valid:
    print(f"数据质量问题: {report.errors}")

# 清洗数据
cleaned_df = clean_data(df, fill_method="forward", remove_duplicates=True)
```

详细文档: [数据质量指南](docs/DATA_QUALITY.md)

### 策略组合功能

支持多策略组合，提供多种权重分配方法。

```python
from qtrade.strategy.portfolio import PortfolioStrategy, StrategyWeight, WeightMethod, StrategyPortfolio

config = PortfolioStrategy(
    strategies=[
        StrategyWeight("ema_cross", weight=0.6),
        StrategyWeight("rsi", weight=0.4),
    ],
    weight_method=WeightMethod.DYNAMIC
)
portfolio = StrategyPortfolio(config)
positions = portfolio.generate_positions(df, ctx, params)
```

详细文档: [策略组合指南](docs/STRATEGY_PORTFOLIO.md)

## 配置说明

### config/base.yaml

```yaml
market:
  symbols: ["BTCUSDT", "ETHUSDT"]  # 交易对
  interval: "1h"                    # K线周期
  start: "2022-01-01"              # 开始日期
  end: null                         # 结束日期（null = 现在）

backtest:
  initial_cash: 10000              # 初始资金
  validate_data: true              # 是否验证数据质量
  clean_data: true                  # 是否在回测前清洗数据

risk:
  max_position_pct: 1.0            # 最大仓位比例
  max_drawdown_pct: 0.5            # 最大回撤限制
  max_leverage: 1.0                # 最大杠杆

portfolio:
  enabled: false                   # 是否启用策略组合
  strategies:
    - name: "ema_cross"
      weight: 0.5
  weight_method: "equal"            # 权重分配方法
  fee_bps: 6                       # 手续费（基点，6 = 0.06%）
  slippage_bps: 5                  # 滑点（基点）
  trade_on: "next_open"            # 执行时机

strategy:
  name: "ema_cross"                # 策略名称
  params:                          # 策略参数
    fast: 20
    slow: 60

output:
  report_dir: "./reports"          # 报告输出目录
```

## 注意事项

1. **避免未来信息泄露**：信号必须使用 `shift(1)` 向后移动，确保在 t 时刻的信号在 t+1 时刻执行
2. **测试策略**：运行 `tests/test_strategy_no_lookahead.py` 确保没有未来信息泄露
3. **参数选择**：使用验证脚本测试参数稳定性，避免过度优化

## 📚 相关文档

### 🚀 新手必读（强烈推荐）
- **[快速开始指南](docs/QUICK_START_GUIDE.md)** ⭐⭐⭐ - 从策略发想到实现的完整教程，适合完全新手
- **[项目功能说明](docs/PROJECT_FEATURES.md)** ⭐⭐ - 项目提供的所有功能详细说明

### 📖 详细文档
- `ARCHITECTURE_ANALYSIS.md` - 架构分析文档
- `STRATEGY_DEVELOPMENT.md` - 策略开发指南
- `STRATEGY_TYPES_GUIDE.md` - 策略类型指南
- `IMPROVEMENTS_SUMMARY.md` - 改进总结

### 🔧 功能文档
- `docs/COMMAND_LINE_USAGE.md` - 命令行使用指南（命令行参数、输出目录管理）
- `docs/RISK_MANAGEMENT.md` - 风险管理指南
- `docs/DATA_QUALITY.md` - 数据质量检查指南
- `docs/STRATEGY_PORTFOLIO.md` - 策略组合指南
