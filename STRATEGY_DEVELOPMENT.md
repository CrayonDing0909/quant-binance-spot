# 策略开发指南

## 快速开始

### 1. 创建策略文件

在 `src/qtrade/strategy/` 目录下创建新文件，例如 `my_strategy.py`：

```python
from __future__ import annotations
import pandas as pd
from .base import StrategyContext
from . import register_strategy


@register_strategy("my_strategy")
def generate_positions(df: pd.DataFrame, ctx: StrategyContext, params: dict) -> pd.Series:
    # 你的策略逻辑
    close = df["close"]
    
    # 示例：简单策略
    signal = (close > close.shift(1)).astype(float)  # 价格上涨时买入
    
    # ⚠️ 重要：必须 shift(1) 避免未来信息泄露
    pos = signal.shift(1).fillna(0.0).clip(0.0, 1.0)
    
    return pos
```

### 2. 注册策略

编辑 `src/qtrade/strategy/__init__.py`，添加：

```python
from . import my_strategy  # noqa: E402
```

### 3. 配置策略

编辑 `config/base.yaml`：

```yaml
strategy:
  name: "my_strategy"
  params:
    param1: 10
    param2: 20
```

### 4. 运行回测

```bash
python scripts/run_backtest.py
```

## 策略函数签名

```python
def generate_positions(
    df: pd.DataFrame,           # K线数据
    ctx: StrategyContext,       # 策略上下文（包含 symbol）
    params: dict                # 策略参数（从 config 读取）
) -> pd.Series:                # 返回持仓比例序列 [0, 1]
```

### 输入数据 `df`

包含以下列：
- `open`: 开盘价
- `high`: 最高价
- `low`: 最低价
- `close`: 收盘价
- `volume`: 成交量
- `index`: 时间索引（DatetimeIndex）

### 返回值

- `pd.Series`: 持仓比例序列，与 `df` 相同的索引
  - `1.0`: 满仓（100% 资金）
  - `0.0`: 空仓（0% 资金）
  - `0.5`: 半仓（50% 资金）

## 重要注意事项

### ⚠️ 避免未来信息泄露（Look-ahead Bias）

**错误示例：**
```python
# ❌ 错误：在 t 时刻使用了 t 时刻的信号
signal = (ma_fast > ma_slow).astype(float)
pos = signal  # 这会导致未来信息泄露！
```

**正确示例：**
```python
# ✅ 正确：信号在 t 时刻生成，在 t+1 时刻执行
signal = (ma_fast > ma_slow).astype(float)
pos = signal.shift(1).fillna(0.0)  # 向后移动 1 个 bar
```

### 为什么需要 shift(1)？

1. 在 `t` 时刻的收盘价计算信号
2. 但实际交易在 `t+1` 时刻的开盘价执行
3. 所以信号必须向后移动 1 个 bar

## 策略开发流程

### 1. 设计策略逻辑

- 定义入场条件
- 定义出场条件
- 考虑风险控制

### 2. 实现策略代码

- 使用 `@register_strategy` 装饰器
- 实现 `generate_positions` 函数
- **确保使用 `shift(1)` 避免未来信息泄露**

### 3. 测试策略

```bash
# 运行测试确保没有未来信息泄露
pytest tests/test_strategy_no_lookahead.py
```

### 4. 运行回测

```bash
python scripts/run_backtest.py
```

查看结果：
- `reports/stats_{SYMBOL}.csv` - 统计指标
- `reports/equity_curve_{SYMBOL}.png` - 资金曲线图

### 5. 验证策略（过拟合检测）

```bash
python scripts/validate_strategy.py
```

检查：
- `reports/walk_forward_{SYMBOL}.csv` - 滚动窗口验证
- `reports/parameter_sensitivity_{SYMBOL}.csv` - 参数敏感性分析

### 6. 修正过拟合

如果发现过拟合：
1. 简化策略（减少参数）
2. 使用更保守的参数
3. 增加样本外测试
4. 避免在同一个数据集上反复优化

## 策略示例

### 示例 1: 双移动平均线

```python
@register_strategy("ma_cross")
def generate_positions(df: pd.DataFrame, ctx: StrategyContext, params: dict) -> pd.Series:
    fast = int(params.get("fast", 20))
    slow = int(params.get("slow", 60))
    close = df["close"]
    
    ma_fast = close.rolling(window=fast).mean()
    ma_slow = close.rolling(window=slow).mean()
    
    signal = (ma_fast > ma_slow).astype(float)
    pos = signal.shift(1).fillna(0.0).clip(0.0, 1.0)
    
    return pos
```

### 示例 2: RSI 策略

```python
@register_strategy("rsi_strategy")
def generate_positions(df: pd.DataFrame, ctx: StrategyContext, params: dict) -> pd.Series:
    period = int(params.get("period", 14))
    oversold = float(params.get("oversold", 30))
    overbought = float(params.get("overbought", 70))
    
    close = df["close"]
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # RSI < 30 买入，RSI > 70 卖出
    signal = (rsi < oversold).astype(float)
    pos = signal.shift(1).fillna(0.0).clip(0.0, 1.0)
    
    return pos
```

### 示例 3: 带止损的策略

```python
@register_strategy("ma_with_stop")
def generate_positions(df: pd.DataFrame, ctx: StrategyContext, params: dict) -> pd.Series:
    fast = int(params.get("fast", 20))
    slow = int(params.get("slow", 60))
    stop_loss_pct = float(params.get("stop_loss", 0.05))  # 5% 止损
    
    close = df["close"]
    ma_fast = close.rolling(window=fast).mean()
    ma_slow = close.rolling(window=slow).mean()
    
    # 基础信号
    base_signal = (ma_fast > ma_slow).astype(float)
    
    # 止损逻辑（简化版：如果价格从高点下跌超过止损比例，则平仓）
    high = df["high"].rolling(window=20).max()
    stop_loss_signal = (close < high * (1 - stop_loss_pct)).astype(float)
    
    # 组合信号
    signal = base_signal * (1 - stop_loss_signal)
    pos = signal.shift(1).fillna(0.0).clip(0.0, 1.0)
    
    return pos
```

## 常见问题

### Q: 如何实现部分仓位？

A: 返回 0.0 到 1.0 之间的值即可，例如 `0.5` 表示 50% 仓位。

### Q: 如何实现多空策略？

A: 当前框架只支持做多（long-only），返回 [0, 1] 之间的值。做空功能需要修改回测引擎。

### Q: 如何访问历史持仓？

A: 在策略函数中，你只能访问当前时刻及之前的数据。如果需要历史持仓信息，需要在策略逻辑中自己维护状态。

### Q: 策略参数如何优化？

A: 使用 `scripts/validate_strategy.py` 进行参数敏感性分析，避免过度优化。

## 参考

- 查看 `src/qtrade/strategy/example_strategy.py` 获取更多示例
- 查看 `src/qtrade/strategy/ema_cross.py` 查看实际策略实现

