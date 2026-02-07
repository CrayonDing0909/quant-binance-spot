# 命令行使用指南

本文档说明如何使用命令行参数运行各种脚本，以及输出目录的组织方式。

## 回测脚本 (run_backtest.py)

### 基本用法

```bash
# 使用配置文件（默认 config/base.yaml）
python scripts/run_backtest.py

# 指定配置文件
python scripts/run_backtest.py -c config/rsi.yaml

# 指定策略（覆盖配置文件中的策略）
python scripts/run_backtest.py -s rsi

# 指定策略和配置文件
python scripts/run_backtest.py -c config/rsi.yaml -s rsi

# 只回测指定交易对
python scripts/run_backtest.py -s rsi --symbol BTCUSDT

# 指定输出目录
python scripts/run_backtest.py -s rsi --output-dir reports/my_backtest
```

### 参数说明

- `-c, --config`: 配置文件路径（默认: `config/base.yaml`）
- `-s, --strategy`: 策略名称（覆盖配置文件中的策略）
- `--symbol`: 指定交易对（默认使用配置文件中的所有交易对）
- `--output-dir`: 输出目录（默认: `reports/{strategy_name}`）

### 输出目录组织

**默认行为**：输出会按策略名称分类到不同目录

```
reports/
├── rsi/                    # RSI 策略的结果
│   ├── stats_BTCUSDT.csv
│   ├── stats_ETHUSDT.csv
│   ├── equity_curve_BTCUSDT.png
│   └── equity_curve_ETHUSDT.png
├── ema_cross/              # EMA 交叉策略的结果
│   ├── stats_BTCUSDT.csv
│   └── equity_curve_BTCUSDT.png
└── my_strategy/            # 你的策略的结果
    ├── stats_BTCUSDT.csv
    └── equity_curve_BTCUSDT.png
```

**优势**：
- ✅ 不同策略的结果不会互相覆盖
- ✅ 可以轻松比较不同策略的表现
- ✅ 组织更清晰，易于管理

## 策略验证脚本 (validate_strategy.py)

### 基本用法

```bash
# 使用配置文件
python scripts/validate_strategy.py

# 指定配置文件
python scripts/validate_strategy.py -c config/rsi.yaml

# 指定策略
python scripts/validate_strategy.py -s rsi

# 指定输出目录
python scripts/validate_strategy.py -s rsi --output-dir reports/validation/rsi
```

### 参数说明

- `-c, --config`: 配置文件路径（默认: `config/base.yaml`）
- `-s, --strategy`: 策略名称（覆盖配置文件中的策略）
- `--output-dir`: 输出目录（默认: `reports/{strategy_name}`）

### 输出文件

验证脚本会在输出目录生成：
- `walk_forward_{SYMBOL}.csv` - 滚动窗口验证结果
- `parameter_sensitivity_{SYMBOL}.csv` - 参数敏感性分析结果

## 参数优化脚本 (optimize_params.py)

### 基本用法

```bash
# 优化策略参数
python scripts/optimize_params.py --strategy rsi

# 指定优化指标
python scripts/optimize_params.py --strategy rsi --metric "Sharpe Ratio"

# 指定配置文件
python scripts/optimize_params.py --strategy rsi -c config/rsi.yaml

# 只优化指定交易对
python scripts/optimize_params.py --strategy rsi --symbol BTCUSDT
```

### 参数说明

- `--strategy`: 策略名称（必需）
- `--metric`: 优化目标指标（默认: `Total Return [%]`）
- `-c, --config`: 配置文件路径（默认: `config/base.yaml`）
- `--symbol`: 指定交易对（默认使用配置文件中的所有交易对）

### 输出文件

优化脚本会在 `reports/` 目录生成：
- `optimization_{strategy}_{SYMBOL}.csv` - 优化结果

## 配置文件 vs 命令行参数

### 推荐使用方式

#### 新手：使用配置文件
- ✅ 简单直观
- ✅ 所有配置在一个文件中
- ✅ 易于版本控制

```yaml
# config/base.yaml
strategy:
  name: "rsi"
  params:
    period: 14
```

```bash
python scripts/run_backtest.py
```

#### 进阶：使用命令行参数
- ✅ 更灵活
- ✅ 可以快速切换策略
- ✅ 适合批量测试

```bash
# 快速测试不同策略
python scripts/run_backtest.py -s rsi
python scripts/run_backtest.py -s ema_cross
python scripts/run_backtest.py -s my_strategy
```

### 优先级

命令行参数 > 配置文件

如果同时指定，命令行参数会覆盖配置文件中的设置。

## 配置文件组织建议

### 方式一：每个策略一个配置文件（推荐）

```
config/
├── base.yaml          # 基础配置
├── rsi.yaml          # RSI 策略配置
├── ema_cross.yaml    # EMA 交叉策略配置
└── my_strategy.yaml  # 你的策略配置
```

**rsi.yaml**:
```yaml
market:
  symbols: ["BTCUSDT", "ETHUSDT"]
  interval: "1h"
  start: "2022-01-01"

backtest:
  initial_cash: 10000
  fee_bps: 6
  slippage_bps: 5

strategy:
  name: "rsi"
  params:
    period: 14
    oversold: 30
    overbought: 70

output:
  report_dir: "./reports"
```

**使用**:
```bash
python scripts/run_backtest.py -c config/rsi.yaml
```

### 方式二：使用基础配置 + 命令行参数

```yaml
# config/base.yaml（通用配置）
market:
  symbols: ["BTCUSDT", "ETHUSDT"]
  interval: "1h"
  start: "2022-01-01"

backtest:
  initial_cash: 10000
  fee_bps: 6
  slippage_bps: 5

strategy:
  name: "rsi"  # 默认策略
  params:
    period: 14
```

**使用**:
```bash
# 使用默认策略
python scripts/run_backtest.py

# 切换策略
python scripts/run_backtest.py -s ema_cross
```

## 输出目录管理

### 默认结构（按策略分类）

```
reports/
├── rsi/
│   ├── stats_BTCUSDT.csv
│   ├── equity_curve_BTCUSDT.png
│   ├── walk_forward_BTCUSDT.csv
│   └── parameter_sensitivity_BTCUSDT.csv
├── ema_cross/
│   └── ...
└── my_strategy/
    └── ...
```

### 自定义输出目录

```bash
# 指定自定义输出目录
python scripts/run_backtest.py -s rsi --output-dir reports/backtest_2024_01

# 结果会保存到 reports/backtest_2024_01/
```

### 清理旧结果

```bash
# 删除某个策略的所有结果
rm -rf reports/rsi/

# 删除所有结果
rm -rf reports/*
```

## 最佳实践

1. **使用配置文件管理策略参数**
   - 每个策略一个配置文件
   - 易于版本控制和分享

2. **使用命令行参数快速测试**
   - 快速切换策略
   - 批量测试不同参数

3. **利用输出目录分类**
   - 默认按策略分类，避免覆盖
   - 可以自定义输出目录用于特殊测试

4. **版本控制**
   - 将配置文件加入 Git
   - 忽略 reports/ 目录（添加到 .gitignore）

## 示例工作流

### 开发新策略

```bash
# 1. 创建策略
python scripts/create_strategy.py --name my_strategy --type custom

# 2. 创建配置文件
cp config/base.yaml config/my_strategy.yaml
# 编辑 config/my_strategy.yaml，设置策略参数

# 3. 运行回测
python scripts/run_backtest.py -c config/my_strategy.yaml

# 4. 优化参数
python scripts/optimize_params.py --strategy my_strategy -c config/my_strategy.yaml

# 5. 验证策略
python scripts/validate_strategy.py -c config/my_strategy.yaml
```

### 比较多个策略

```bash
# 快速测试多个策略
for strategy in rsi ema_cross my_strategy; do
    python scripts/run_backtest.py -s $strategy
done

# 结果会分别保存到 reports/rsi/, reports/ema_cross/, reports/my_strategy/
```

---

**提示**：使用 `--help` 查看所有可用参数

```bash
python scripts/run_backtest.py --help
python scripts/validate_strategy.py --help
python scripts/optimize_params.py --help
```

