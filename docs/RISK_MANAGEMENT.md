# 风险管理模块使用指南

## 概述

风险管理模块提供了完整的风险管理功能，包括仓位管理、风险限制和组合风险控制。

## 仓位管理

### 固定仓位 (FixedPositionSizer)

使用固定的仓位比例，不考虑信号强度。

```python
from qtrade.risk import FixedPositionSizer

sizer = FixedPositionSizer(position_pct=0.8)  # 80% 仓位
size = sizer.calculate_size(
    signal=1.0,      # 信号强度 [0, 1]
    equity=10000,    # 当前权益
    price=50000      # 当前价格
)
```

### Kelly 公式仓位 (KellyPositionSizer)

根据胜率和盈亏比计算最优仓位。

```python
from qtrade.risk import KellyPositionSizer

sizer = KellyPositionSizer(
    win_rate=0.6,        # 胜率 60%
    avg_win=0.05,        # 平均盈利 5%
    avg_loss=0.03,       # 平均亏损 3%
    kelly_fraction=0.5   # 使用 50% Kelly（更保守）
)
size = sizer.calculate_size(signal=1.0, equity=10000, price=50000)
```

### 波动率调整仓位 (VolatilityPositionSizer)

根据资产波动率调整仓位，波动率越高，仓位越小。

```python
from qtrade.risk import VolatilityPositionSizer

sizer = VolatilityPositionSizer(
    base_position_pct=1.0,      # 基础仓位 100%
    target_volatility=0.15,     # 目标波动率 15%
    lookback=20                 # 回看期 20
)
size = sizer.calculate_size(
    signal=1.0,
    equity=10000,
    price=50000,
    returns=returns_series      # 收益率序列
)
```

## 风险限制

### 基本使用

```python
from qtrade.risk import RiskLimits, apply_risk_limits

# 配置风险限制
limits = RiskLimits(
    max_position_pct=1.0,          # 最大仓位 100%
    max_drawdown_pct=0.5,          # 最大回撤 50%
    max_leverage=1.0,              # 最大杠杆 1.0（现货）
    max_single_position_pct=0.5,   # 单个资产最大仓位 50%
    min_cash_reserve_pct=0.1        # 最小现金储备 10%
)

# 应用风险限制
adjusted_pos, checks = apply_risk_limits(
    position_pct=1.0,               # 原始仓位
    equity_curve=equity_series,       # 权益曲线
    limits=limits,
    current_equity=9500,             # 当前权益
    cash=500                          # 当前现金
)

# 检查结果
if not checks["max_drawdown"]["passed"]:
    print(f"回撤超过限制: {checks['max_drawdown']['value']:.2%}")
```

### 单独检查

```python
from qtrade.risk import check_max_position_size, check_max_drawdown

# 检查仓位
passed, adjusted = check_max_position_size(1.2, limits)  # 超过限制

# 检查回撤
passed, current_dd = check_max_drawdown(equity_curve, limits)
```

## 组合风险管理

### 计算组合风险指标

```python
from qtrade.risk import PortfolioRiskManager

manager = PortfolioRiskManager(
    max_portfolio_var=0.05,      # 最大组合 VaR 5%
    max_correlation=0.8,         # 最大相关性 0.8
    diversification_threshold=0.3  # 分散化阈值 0.3
)

# 计算风险指标
risk_metrics = manager.calculate_portfolio_risk(
    returns={
        "BTCUSDT": btc_returns,
        "ETHUSDT": eth_returns
    },
    weights={
        "BTCUSDT": 0.6,
        "ETHUSDT": 0.4
    },
    confidence_level=0.95
)

print(f"组合 VaR: {risk_metrics['portfolio_var']:.4f}")
print(f"组合波动率: {risk_metrics['portfolio_volatility']:.2%}")
print(f"最大相关性: {risk_metrics['max_correlation']:.2f}")
print(f"分散化比率: {risk_metrics['diversification_ratio']:.2f}")
```

### 检查风险限制

```python
passed, metrics = manager.check_risk_limits(
    returns={"BTCUSDT": btc_returns, "ETHUSDT": eth_returns},
    weights={"BTCUSDT": 0.6, "ETHUSDT": 0.4}
)

if not passed:
    print("风险超过限制:")
    for issue in metrics["issues"]:
        print(f"  - {issue}")
```

## 在回测中使用

在回测配置中启用风险管理：

```yaml
# config/base.yaml
backtest:
  initial_cash: 10000
  validate_data: true
  clean_data: true

risk:
  max_position_pct: 1.0
  max_drawdown_pct: 0.5
  max_leverage: 1.0
```

然后在回测代码中：

```python
from qtrade.risk import RiskLimits
from qtrade.backtest import run_symbol_backtest

limits = RiskLimits(
    max_position_pct=1.0,
    max_drawdown_pct=0.5
)

result = run_symbol_backtest(
    symbol="BTCUSDT",
    data_path=data_path,
    cfg=cfg,
    risk_limits=limits
)
```

## 最佳实践

1. **保守的 Kelly 比例**: 使用 0.25-0.5 的 Kelly 比例，而不是全 Kelly
2. **设置回撤限制**: 根据风险承受能力设置合理的最大回撤限制
3. **组合分散化**: 确保组合中的资产相关性不要过高
4. **定期检查**: 定期检查风险指标，及时调整策略

