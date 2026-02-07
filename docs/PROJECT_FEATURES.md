# 项目功能完整说明

本文档详细说明项目提供的所有功能，帮助你了解可以做什么。

## 📚 目录

1. [策略开发功能](#策略开发功能)
2. [数据管理功能](#数据管理功能)
3. [回测功能](#回测功能)
4. [优化功能](#优化功能)
5. [验证功能](#验证功能)
6. [风险管理功能](#风险管理功能)
7. [策略组合功能](#策略组合功能)
8. [实用工具](#实用工具)

---

## 策略开发功能

### 1. 策略模板生成器

**功能**：快速创建策略文件，无需从零开始写代码。

**使用方法**：
```bash
# 创建 RSI 策略
python scripts/create_strategy.py --name my_rsi --type rsi

# 创建 EMA 交叉策略
python scripts/create_strategy.py --name my_ema --type ema

# 创建 MACD 策略
python scripts/create_strategy.py --name my_macd --type macd

# 创建自定义策略
python scripts/create_strategy.py --name my_custom --type custom
```

**优势**：
- ✅ 自动生成代码框架
- ✅ 自动注册策略
- ✅ 统一的代码风格
- ✅ 减少手动操作

### 2. 指标库

**功能**：提供常用的技术指标，无需重复实现。

**可用指标**：

#### RSI（相对强弱指标）
```python
from qtrade.indicators import calculate_rsi

rsi = calculate_rsi(close, period=14)
```

#### MACD
```python
from qtrade.indicators import calculate_macd

macd_data = calculate_macd(close, fast_period=12, slow_period=26, signal_period=9)
# 返回: {'macd': ..., 'signal': ..., 'histogram': ...}
```

#### 布林带
```python
from qtrade.indicators import calculate_bollinger_bands

bb = calculate_bollinger_bands(close, period=20, std_mult=2.0)
# 返回: {'upper': ..., 'middle': ..., 'lower': ...}
```

#### 移动平均线
```python
from qtrade.indicators import calculate_sma, calculate_ema, calculate_wma

sma = calculate_sma(close, period=20)
ema = calculate_ema(close, period=20)
wma = calculate_wma(close, period=20)
```

#### RSI 背离检测
```python
from qtrade.indicators import calculate_rsi_divergence

divergence = calculate_rsi_divergence(close, period=14, lookback=5)
```

**优势**：
- ✅ 统一接口
- ✅ 经过测试
- ✅ 性能优化
- ✅ 易于使用

### 3. 策略注册系统

**功能**：使用装饰器注册策略，自动管理。

**使用方法**：
```python
from qtrade.strategy import register_strategy

@register_strategy("my_strategy")
def generate_positions(df, ctx, params):
    # 策略逻辑
    return positions
```

**优势**：
- ✅ 自动注册
- ✅ 易于管理
- ✅ 支持热加载

### 4. 策略基类

**功能**：支持状态管理的策略基类。

**适用场景**：
- 需要维护状态（如止损、止盈）
- 复杂的策略逻辑

**使用方法**：
```python
from qtrade.strategy.base import BaseStrategy, StrategyState

class MyStrategy(BaseStrategy):
    def generate_positions(self, df, ctx, params, state):
        # 使用 state 维护状态
        if state.current_position > 0:
            if df["close"].iloc[-1] < state.stop_loss:
                return 0.0
        return 1.0
```

---

## 数据管理功能

### 1. 数据下载

**功能**：自动从币安下载历史 K 线数据。

**使用方法**：
```bash
# 下载单个交易对
python scripts/download_data.py --symbol BTCUSDT --interval 1h --start 2022-01-01

# 下载多个交易对（在配置文件中设置）
python scripts/download_data.py
```

**支持的时间周期**：
- `1m`, `3m`, `5m`, `15m`, `30m`
- `1h`, `2h`, `4h`, `6h`, `8h`, `12h`
- `1d`, `3d`, `1w`, `1M`

**数据格式**：
- 自动保存为 Parquet 格式（高效压缩）
- 包含：open, high, low, close, volume

### 2. 数据质量检查

**功能**：自动检查数据质量，发现潜在问题。

**检查内容**：
- ✅ 缺失值检查
- ✅ 重复时间戳检查
- ✅ 异常值检测（Z-score）
- ✅ 时间间隔检查
- ✅ 价格有效性检查（OHLC 逻辑）
- ✅ 零成交量检查
- ✅ 价格异常变化检查

**使用方法**：
```python
from qtrade.data import validate_data_quality

report = validate_data_quality(df)
if not report.is_valid:
    print("数据有问题")
    for error in report.errors:
        print(f"  - {error}")
```

**自动检查**：
在回测时自动检查（如果 `validate_data: true`）

### 3. 数据清洗

**功能**：自动清洗数据，修复常见问题。

**清洗功能**：
- 填充缺失值（前向填充、后向填充、插值）
- 移除重复时间戳
- 移除异常值（可选）
- 修复无效价格

**使用方法**：
```python
from qtrade.data import clean_data

cleaned_df = clean_data(
    df,
    fill_method="forward",      # forward, backward, interpolate
    remove_outliers=False,      # 是否移除异常值
    remove_duplicates=True       # 是否移除重复
)
```

**自动清洗**：
在回测时自动清洗（如果 `clean_data: true`）

---

## 回测功能

### 1. 向量化回测

**功能**：使用 vectorbt 进行快速向量化回测。

**特点**：
- ✅ 速度快（比循环快 100+ 倍）
- ✅ 支持手续费和滑点
- ✅ 避免未来信息泄露
- ✅ 自动计算各种指标

### 2. 回测配置

**配置文件** (`config/base.yaml`)：
```yaml
backtest:
  initial_cash: 10000        # 初始资金
  fee_bps: 6                 # 手续费 0.06%
  slippage_bps: 5            # 滑点 0.05%
  trade_on: "next_open"      # 执行时机
  validate_data: true        # 验证数据
  clean_data: true           # 清洗数据
```

### 3. 回测报告

**自动生成**：
- `stats_{SYMBOL}.csv` - 统计指标
- `equity_curve_{SYMBOL}.png` - 资金曲线图

**统计指标包括**：
- 总收益率
- 年化收益率
- 最大回撤
- 夏普比率
- 索提诺比率
- 胜率
- 总交易次数
- 平均持仓时间

### 4. 可视化

**资金曲线图包含**：
1. 价格和信号图：显示价格走势和买卖点
2. 持仓比例图：显示持仓变化
3. 资金曲线图：显示账户价值变化
4. 回撤图：显示最大回撤

---

## 优化功能

### 1. 参数网格搜索

**功能**：自动测试多个参数组合，找到最佳参数。

**使用方法**：
```bash
# 优化策略参数
python scripts/optimize_params.py --strategy rsi --metric "Sharpe Ratio"
```

**优化指标**：
- `Total Return [%]` - 总收益率
- `Sharpe Ratio` - 夏普比率（推荐）
- `Max Drawdown [%]` - 最大回撤
- `Win Rate [%]` - 胜率

### 2. 参数网格配置

**自动配置**：根据策略类型自动生成参数网格。

**手动配置**：在 `scripts/optimize_params.py` 中修改。

### 3. 优化报告

**生成文件**：`reports/optimization_{SYMBOL}.csv`

**包含内容**：
- 所有参数组合
- 对应的表现指标
- 最佳参数组合

---

## 验证功能

### 1. 滚动窗口验证 (Walk-Forward Analysis)

**功能**：将数据分成多个训练/测试窗口，检测过拟合。

**原理**：
- 在训练集上优化参数
- 在测试集上验证表现
- 如果测试集表现远差于训练集 → 可能过拟合

**使用方法**：
```bash
python scripts/validate_strategy.py
```

**输出**：
- `walk_forward_{SYMBOL}.csv` - 每个窗口的结果

### 2. 参数敏感性分析

**功能**：测试参数微小变化对结果的影响。

**判断标准**：
- ✅ 参数变化时表现稳定 → 策略稳健
- ❌ 参数微小变化导致结果大幅波动 → 策略不稳定

**输出**：
- `parameter_sensitivity_{SYMBOL}.csv` - 所有参数组合的结果

### 3. 过拟合检测

**自动检测**：
- 测试集收益率下降 > 30% → 警告
- 参数敏感性高 → 警告

---

## 风险管理功能

### 1. 仓位管理

**功能**：提供多种仓位计算方法。

#### 固定仓位
```python
from qtrade.risk import FixedPositionSizer

sizer = FixedPositionSizer(position_pct=0.8)  # 80% 仓位
```

#### Kelly 公式
```python
from qtrade.risk import KellyPositionSizer

sizer = KellyPositionSizer(
    win_rate=0.6,      # 胜率
    avg_win=0.05,      # 平均盈利
    avg_loss=0.03,     # 平均亏损
    kelly_fraction=0.5  # 保守因子
)
```

#### 波动率调整
```python
from qtrade.risk import VolatilityPositionSizer

sizer = VolatilityPositionSizer(
    base_position_pct=1.0,
    target_volatility=0.15,  # 目标波动率 15%
    lookback=20
)
```

### 2. 风险限制

**功能**：设置各种风险限制。

**限制类型**：
- 最大仓位限制
- 最大回撤限制
- 最大杠杆限制
- 现金储备要求

**使用方法**：
```python
from qtrade.risk import RiskLimits, apply_risk_limits

limits = RiskLimits(
    max_position_pct=1.0,
    max_drawdown_pct=0.5,
    max_leverage=1.0
)

adjusted_pos, checks = apply_risk_limits(
    position_pct=1.0,
    equity_curve=equity_series,
    limits=limits
)
```

### 3. 组合风险管理

**功能**：管理多个资产的组合风险。

**功能包括**：
- 组合 VaR 计算
- 相关性分析
- 分散化比率计算

**使用方法**：
```python
from qtrade.risk import PortfolioRiskManager

manager = PortfolioRiskManager()
risk_metrics = manager.calculate_portfolio_risk(
    returns={"BTCUSDT": btc_returns, "ETHUSDT": eth_returns},
    weights={"BTCUSDT": 0.6, "ETHUSDT": 0.4}
)
```

详细文档：[风险管理指南](RISK_MANAGEMENT.md)

---

## 策略组合功能

### 1. 多策略组合

**功能**：将多个策略组合在一起，分散风险。

**使用方法**：
```python
from qtrade.strategy.portfolio import PortfolioStrategy, StrategyWeight, StrategyPortfolio

config = PortfolioStrategy(
    strategies=[
        StrategyWeight("ema_cross", weight=0.6),
        StrategyWeight("rsi", weight=0.4),
    ],
    weight_method=WeightMethod.EQUAL
)

portfolio = StrategyPortfolio(config)
positions = portfolio.generate_positions(df, ctx, params)
```

### 2. 权重分配方法

**支持的方法**：
- `EQUAL` - 等权重
- `FIXED` - 固定权重
- `PERFORMANCE` - 基于历史表现
- `VOLATILITY` - 基于波动率
- `SHARPE` - 基于夏普比率
- `DYNAMIC` - 动态调整

### 3. 动态权重调整

**功能**：根据策略表现动态调整权重。

**适用场景**：
- 策略表现差异较大
- 需要自适应调整

详细文档：[策略组合指南](STRATEGY_PORTFOLIO.md)

---

## 实用工具

### 1. 配置管理

**功能**：使用 YAML 配置文件，易于管理。

**配置文件**：
- `config/base.yaml` - 基础配置
- `config/dev.yaml` - 开发环境配置（可选）

**配置内容**：
- 市场配置（交易对、时间周期）
- 回测配置（初始资金、手续费）
- 策略配置（策略名称、参数）
- 风险管理配置
- 输出配置

### 2. 日志系统

**功能**：统一的日志系统。

**使用方法**：
```python
from qtrade.utils.log import get_logger

logger = get_logger(__name__)
logger.info("策略运行中...")
```

### 3. 时间工具

**功能**：时间相关的工具函数。

**使用方法**：
```python
from qtrade.utils.time import parse_time, format_time
```

---

## 功能总结

### ✅ 已实现的功能

1. **策略开发**
   - ✅ 策略模板生成器
   - ✅ 指标库
   - ✅ 策略注册系统
   - ✅ 策略基类（状态管理）

2. **数据管理**
   - ✅ 数据下载
   - ✅ 数据质量检查
   - ✅ 数据清洗

3. **回测**
   - ✅ 向量化回测
   - ✅ 自动报告生成
   - ✅ 可视化

4. **优化**
   - ✅ 参数网格搜索
   - ✅ 多指标优化

5. **验证**
   - ✅ 滚动窗口验证
   - ✅ 参数敏感性分析
   - ✅ 过拟合检测

6. **风险管理**
   - ✅ 仓位管理
   - ✅ 风险限制
   - ✅ 组合风险管理

7. **策略组合**
   - ✅ 多策略组合
   - ✅ 动态权重调整

### 🚧 计划中的功能

1. **机器学习集成**
   - 特征工程
   - 模型训练
   - 预测信号生成

2. **实时监控**
   - 策略性能监控
   - 异常检测
   - 自动告警

---

## 使用建议

### 新手建议

1. **从简单开始**：先使用模板生成器创建简单策略
2. **理解指标**：学习常用技术指标的含义
3. **小步迭代**：每次只改一个参数，观察效果
4. **验证策略**：一定要运行验证脚本，避免过拟合

### 进阶建议

1. **策略组合**：组合多个策略，分散风险
2. **风险管理**：添加风险限制，保护资金
3. **数据质量**：确保数据质量，避免错误结果
4. **持续改进**：根据回测结果不断优化策略

---

## 相关文档

- [快速开始指南](QUICK_START_GUIDE.md) - 从零开始的完整教程
- [策略开发指南](../STRATEGY_DEVELOPMENT.md) - 策略开发详细说明
- [风险管理指南](RISK_MANAGEMENT.md) - 风险管理详细说明
- [数据质量指南](DATA_QUALITY.md) - 数据质量检查详细说明
- [策略组合指南](STRATEGY_PORTFOLIO.md) - 策略组合详细说明

---

**祝你交易顺利！** 🎉

