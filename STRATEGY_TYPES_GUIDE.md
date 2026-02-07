# 不同策略类型的开发方式

## 概述

本项目支持多种策略类型，所有策略都遵循统一的开发模式：
1. **创建策略文件** → 2. **注册策略** → 3. **配置参数** → 4. **运行回测**

## 策略开发模式对比

### 模式 1: 技术指标策略（如 RSI, MACD, Bollinger Bands）

**特点：**
- 基于技术指标计算
- 参数相对固定（周期、阈值等）
- 逻辑相对简单

**示例：RSI 策略**

```python
# src/qtrade/strategy/rsi_strategy.py
@register_strategy("rsi")
def generate_positions(df, ctx, params):
    period = int(params.get("period", 14))
    oversold = float(params.get("oversold", 30))
    overbought = float(params.get("overbought", 70))
    
    rsi = calculate_rsi(df["close"], period)
    signal = (rsi < oversold).astype(float)
    pos = signal.shift(1).fillna(0.0)
    return pos.clip(0.0, 1.0)
```

**配置：**
```yaml
strategy:
  name: "rsi"
  params:
    period: 14
    oversold: 30
    overbought: 70
```

### 模式 2: 价格行为策略（如 SMC, Order Flow）

**特点：**
- 基于价格行为和市场结构
- 需要识别特定模式（订单块、流动性池等）
- 参数可能更复杂

**示例：SMC 策略**

```python
# src/qtrade/strategy/smc_strategy.py
@register_strategy("smc_basic")
def generate_positions(df, ctx, params):
    order_blocks = identify_order_blocks(df, params.get("order_block_lookback", 20))
    liquidity = identify_liquidity_pools(df, params.get("liquidity_lookback", 50))
    structure = identify_market_structure(df)
    
    signal = (order_blocks["bullish_ob"] > 0) & (liquidity > 0)
    pos = signal.shift(1).fillna(0.0)
    return pos.clip(0.0, 1.0)
```

**配置：**
```yaml
strategy:
  name: "smc_basic"
  params:
    order_block_lookback: 20
    liquidity_lookback: 50
    min_volume_multiplier: 1.5
```

### 模式 3: 趋势跟随策略（如 EMA Cross, MACD）

**特点：**
- 基于趋势识别
- 通常使用移动平均线
- 参数：快慢周期

**示例：EMA 交叉策略（已有）**

```python
# src/qtrade/strategy/ema_cross.py
@register_strategy("ema_cross")
def generate_positions(df, ctx, params):
    fast = int(params.get("fast", 20))
    slow = int(params.get("slow", 60))
    
    ema_fast = df["close"].ewm(span=fast).mean()
    ema_slow = df["close"].ewm(span=slow).mean()
    
    signal = (ema_fast > ema_slow).astype(float)
    pos = signal.shift(1).fillna(0.0)
    return pos.clip(0.0, 1.0)
```

**配置：**
```yaml
strategy:
  name: "ema_cross"
  params:
    fast: 20
    slow: 60
```

### 模式 4: 均值回归策略（如 Bollinger Bands, RSI 均值回归）

**特点：**
- 基于价格偏离均值的程度
- 在极端位置反向交易
- 参数：周期、标准差倍数

**示例：Bollinger Bands 策略（可扩展）**

```python
@register_strategy("bollinger_mean_reversion")
def generate_positions(df, ctx, params):
    period = int(params.get("period", 20))
    std_mult = float(params.get("std_mult", 2.0))
    
    close = df["close"]
    ma = close.rolling(period).mean()
    std = close.rolling(period).std()
    
    upper = ma + std * std_mult
    lower = ma - std * std_mult
    
    # 价格跌破下轨 -> 买入（均值回归）
    signal = (close < lower).astype(float)
    pos = signal.shift(1).fillna(0.0)
    return pos.clip(0.0, 1.0)
```

**配置：**
```yaml
strategy:
  name: "bollinger_mean_reversion"
  params:
    period: 20
    std_mult: 2.0
```

## 快速开发新策略的步骤

### 步骤 1: 确定策略类型

根据你的策略逻辑，选择对应的模式：
- **技术指标** → 参考 `rsi_strategy.py`
- **价格行为** → 参考 `smc_strategy.py`
- **趋势跟随** → 参考 `ema_cross.py`
- **均值回归** → 参考上面的 Bollinger 示例

### 步骤 2: 创建策略文件

在 `src/qtrade/strategy/` 创建新文件，例如 `my_rsi_strategy.py`：

```python
from __future__ import annotations
import pandas as pd
from .base import StrategyContext
from . import register_strategy

@register_strategy("my_rsi")
def generate_positions(df: pd.DataFrame, ctx: StrategyContext, params: dict) -> pd.Series:
    # 你的策略逻辑
    period = int(params.get("period", 14))
    # ... 计算指标和信号
    pos = signal.shift(1).fillna(0.0)
    return pos.clip(0.0, 1.0)
```

### 步骤 3: 注册策略

在 `src/qtrade/strategy/__init__.py` 添加：

```python
from . import my_rsi_strategy  # noqa: E402
```

### 步骤 4: 配置参数

创建或修改配置文件（如 `config/my_rsi.yaml`）：

```yaml
strategy:
  name: "my_rsi"
  params:
    period: 14
    oversold: 30
    overbought: 70
```

### 步骤 5: 运行回测

```bash
# 使用默认配置
python scripts/run_backtest.py

# 或指定配置文件（需要修改脚本支持）
# python scripts/run_backtest.py --config config/my_rsi.yaml
```

## 参数配置的最佳实践

### 1. 使用有意义的默认值

```python
period = int(params.get("period", 14))  # 提供默认值
```

### 2. 参数类型明确

```python
# ✅ 好：明确类型转换
period = int(params.get("period", 14))
threshold = float(params.get("threshold", 0.5))
enabled = bool(params.get("enabled", True))

# ❌ 不好：类型不明确
period = params.get("period", 14)  # 可能是字符串
```

### 3. 参数验证

```python
period = int(params.get("period", 14))
if period < 1 or period > 200:
    raise ValueError(f"Invalid period: {period}, must be between 1 and 200")
```

### 4. 参数文档

在配置文件中添加注释：

```yaml
strategy:
  name: "rsi"
  params:
    period: 14        # RSI 计算周期，建议 10-20
    oversold: 30      # 超卖阈值，建议 20-40
    overbought: 70    # 超买阈值，建议 60-80
```

## 常见策略类型的参数模式

| 策略类型 | 典型参数 | 示例值 |
|---------|---------|--------|
| RSI | period, oversold, overbought | 14, 30, 70 |
| MACD | fast, slow, signal | 12, 26, 9 |
| Bollinger | period, std_mult | 20, 2.0 |
| EMA Cross | fast, slow | 20, 60 |
| SMC | order_block_lookback, liquidity_lookback | 20, 50 |
| 成交量策略 | volume_period, volume_threshold | 20, 1.5 |

## 总结

**所有策略类型都遵循相同的开发模式：**

1. ✅ **统一的函数签名** - `generate_positions(df, ctx, params)`
2. ✅ **统一的注册方式** - `@register_strategy("name")`
3. ✅ **统一的配置格式** - YAML 配置文件
4. ✅ **统一的执行流程** - 运行回测脚本

**无论策略类型如何，开发步骤都是一样的：**
- 创建策略文件 → 注册 → 配置 → 运行

这使得**长期开发策略非常快速方便**，因为你只需要：
- 复制现有策略模板
- 修改策略逻辑
- 调整参数配置

**无需修改框架代码！**




