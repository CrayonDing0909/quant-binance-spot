# 策略组合使用指南

## 概述

策略组合功能允许你将多个策略组合在一起，通过不同的权重分配方法来优化整体表现。

## 基本使用

### 创建组合策略

```python
from qtrade.strategy.portfolio import (
    PortfolioStrategy,
    StrategyWeight,
    WeightMethod,
    StrategyPortfolio
)

# 配置组合策略
config = PortfolioStrategy(
    strategies=[
        StrategyWeight("ema_cross", weight=0.6, min_weight=0.2, max_weight=0.8),
        StrategyWeight("rsi", weight=0.4, min_weight=0.2, max_weight=0.8),
    ],
    weight_method=WeightMethod.EQUAL,  # 等权重
    rebalance_freq="D"                 # 每日再平衡
)

# 创建组合
portfolio = StrategyPortfolio(config)

# 生成持仓信号
positions = portfolio.generate_positions(df, ctx, params)
```

## 权重分配方法

### 1. 等权重 (EQUAL)

所有策略使用相同的权重。

```python
config = PortfolioStrategy(
    strategies=[
        StrategyWeight("ema_cross", weight=0.5),
        StrategyWeight("rsi", weight=0.5),
    ],
    weight_method=WeightMethod.EQUAL
)
```

### 2. 固定权重 (FIXED)

使用配置中指定的固定权重。

```python
config = PortfolioStrategy(
    strategies=[
        StrategyWeight("ema_cross", weight=0.7),  # 70%
        StrategyWeight("rsi", weight=0.3),         # 30%
    ],
    weight_method=WeightMethod.FIXED
)
```

### 3. 基于历史表现 (PERFORMANCE)

根据各策略的历史收益率分配权重，收益率越高，权重越高。

```python
config = PortfolioStrategy(
    strategies=[
        StrategyWeight("ema_cross", weight=0.5, min_weight=0.2, max_weight=0.8),
        StrategyWeight("rsi", weight=0.5, min_weight=0.2, max_weight=0.8),
    ],
    weight_method=WeightMethod.PERFORMANCE
)

# 需要提供历史收益率
historical_returns = {
    "ema_cross": ema_returns_series,
    "rsi": rsi_returns_series
}

positions = portfolio.generate_positions(
    df, ctx, params,
    historical_returns=historical_returns
)
```

### 4. 基于波动率 (VOLATILITY)

根据各策略的波动率分配权重，波动率越低，权重越高。

```python
config = PortfolioStrategy(
    strategies=[
        StrategyWeight("ema_cross", weight=0.5),
        StrategyWeight("rsi", weight=0.5),
    ],
    weight_method=WeightMethod.VOLATILITY
)

positions = portfolio.generate_positions(
    df, ctx, params,
    historical_returns=historical_returns
)
```

### 5. 基于夏普比率 (SHARPE)

根据各策略的夏普比率分配权重，夏普比率越高，权重越高。

```python
config = PortfolioStrategy(
    strategies=[
        StrategyWeight("ema_cross", weight=0.5),
        StrategyWeight("rsi", weight=0.5),
    ],
    weight_method=WeightMethod.SHARPE
)

historical_sharpe = {
    "ema_cross": 1.5,
    "rsi": 1.2
}

positions = portfolio.generate_positions(
    df, ctx, params,
    historical_sharpe=historical_sharpe
)
```

### 6. 动态调整 (DYNAMIC)

结合多个因素（表现和波动率）动态调整权重。

```python
config = PortfolioStrategy(
    strategies=[
        StrategyWeight("ema_cross", weight=0.5, min_weight=0.2, max_weight=0.8),
        StrategyWeight("rsi", weight=0.5, min_weight=0.2, max_weight=0.8),
    ],
    weight_method=WeightMethod.DYNAMIC,
    lookback_period=30  # 回看期 30 天
)

positions = portfolio.generate_positions(
    df, ctx, params,
    historical_returns=historical_returns
)
```

## 配置示例

### YAML 配置

```yaml
# config/base.yaml
portfolio:
  enabled: true
  strategies:
    - name: "ema_cross"
      weight: 0.6
      min_weight: 0.2
      max_weight: 0.8
    - name: "rsi"
      weight: 0.4
      min_weight: 0.2
      max_weight: 0.8
  weight_method: "dynamic"  # equal, fixed, performance, volatility, sharpe, dynamic
  rebalance_freq: "D"      # D, W, M
  lookback_period: 30
```

### 从配置创建组合

```python
from qtrade.config import load_config
from qtrade.strategy.portfolio import PortfolioStrategy, StrategyWeight, WeightMethod

cfg = load_config()

if cfg.portfolio.enabled:
    # 从配置创建组合策略
    strategies = [
        StrategyWeight(
            name=s["name"],
            weight=s["weight"],
            min_weight=s.get("min_weight", 0.0),
            max_weight=s.get("max_weight", 1.0)
        )
        for s in cfg.portfolio.strategies
    ]
    
    config = PortfolioStrategy(
        strategies=strategies,
        weight_method=WeightMethod(cfg.portfolio.weight_method),
        rebalance_freq=cfg.portfolio.rebalance_freq,
        lookback_period=cfg.portfolio.lookback_period
    )
    
    portfolio = StrategyPortfolio(config)
```

## 注册组合策略

你可以将组合策略注册为普通策略：

```python
from qtrade.strategy import register_strategy
from qtrade.strategy.portfolio import PortfolioStrategy, StrategyWeight, WeightMethod, StrategyPortfolio

# 创建组合配置
config = PortfolioStrategy(
    strategies=[
        StrategyWeight("ema_cross", weight=0.6),
        StrategyWeight("rsi", weight=0.4),
    ],
    weight_method=WeightMethod.DYNAMIC
)

portfolio = StrategyPortfolio(config)

# 注册为策略
@register_strategy("my_portfolio")
def generate_positions(df, ctx, params):
    return portfolio.generate_positions(df, ctx, params)
```

## 最佳实践

1. **设置权重限制**: 使用 `min_weight` 和 `max_weight` 防止某个策略权重过高或过低
2. **选择合适的权重方法**: 
   - 等权重：简单，适合策略表现相近的情况
   - 固定权重：适合有明确偏好的情况
   - 动态调整：适合策略表现差异较大的情况
3. **定期再平衡**: 根据 `rebalance_freq` 定期调整权重
4. **监控组合表现**: 定期检查组合的整体表现，必要时调整权重分配方法
5. **考虑相关性**: 如果策略相关性很高，组合的分散化效果可能有限

## 示例：三策略组合

```python
config = PortfolioStrategy(
    strategies=[
        StrategyWeight("ema_cross", weight=0.4, min_weight=0.1, max_weight=0.6),
        StrategyWeight("rsi", weight=0.3, min_weight=0.1, max_weight=0.5),
        StrategyWeight("smc_strategy", weight=0.3, min_weight=0.1, max_weight=0.5),
    ],
    weight_method=WeightMethod.DYNAMIC,
    rebalance_freq="W",  # 每周再平衡
    lookback_period=60   # 回看 60 天
)

portfolio = StrategyPortfolio(config)

# 计算历史收益率（用于动态权重）
historical_returns = {
    "ema_cross": calculate_returns(ema_positions, df),
    "rsi": calculate_returns(rsi_positions, df),
    "smc_strategy": calculate_returns(smc_positions, df),
}

# 生成组合持仓
portfolio_positions = portfolio.generate_positions(
    df, ctx, params,
    historical_returns=historical_returns
)
```

