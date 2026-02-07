# 新手完整教程：从策略发想到实盘

本教程将带你从零开始，完整地开发一个交易策略，包括策略发想、实现、回测、优化和验证。

## 📋 目录

1. [项目功能概览](#项目功能概览)
2. [第一步：策略发想](#第一步策略发想)
3. [第二步：创建策略](#第二步创建策略)
4. [第三步：回测策略](#第三步回测策略)
5. [第四步：优化参数](#第四步优化参数)
6. [第五步：验证策略](#第五步验证策略)
7. [第六步：风险管理](#第六步风险管理)
8. [完整示例：RSI 策略](#完整示例rsi策略)

---

## ⚡ 快速理解：回测如何工作？

**重要概念**：回测脚本通过读取配置文件来确定使用哪个策略。

### 工作流程

```
1. 你创建策略代码 → 用 @register_strategy("策略名称") 注册
                    ↓
2. 你在 config/base.yaml 中配置 → strategy.name = "策略名称"
                    ↓
3. 你运行 python scripts/run_backtest.py
                    ↓
4. 脚本读取配置文件 → 找到策略名称
                    ↓
5. 脚本根据名称找到策略函数 → 使用配置的参数运行
                    ↓
6. 生成回测结果和报告
```

### 关键点

- ✅ **策略名称必须一致**：代码中的 `@register_strategy("my_strategy")` 和配置文件中的 `strategy.name: "my_strategy"` 必须完全一致
- ✅ **参数在配置文件中**：策略的参数在 `config/base.yaml` 的 `strategy.params` 中设置
- ✅ **切换策略很简单**：只需要修改配置文件，无需修改代码

### 示例

**代码中注册策略**：
```python
@register_strategy("my_rsi_strategy")  # ← 策略名称
def generate_positions(df, ctx, params):
    # 策略逻辑
    ...
```

**配置文件中指定策略**：
```yaml
strategy:
  name: "my_rsi_strategy"  # ← 必须与上面的名称一致
  params:
    period: 14
```

**运行回测**：
```bash
python scripts/run_backtest.py  # 自动读取配置，使用 my_rsi_strategy
```

---

## 项目功能概览

这个项目提供了完整的量化交易策略开发工具：

### 🎯 核心功能

1. **策略开发**
   - 快速创建策略模板
   - 统一的指标库（RSI、MACD、布林带等）
   - 策略注册系统

2. **数据管理**
   - 自动下载币安数据
   - 数据质量检查
   - 数据清洗

3. **回测系统**
   - 向量化回测（快速）
   - 支持手续费和滑点
   - 自动生成报告和图表

4. **策略优化**
   - 参数网格搜索
   - 多指标优化

5. **策略验证**
   - 过拟合检测
   - 滚动窗口验证
   - 参数敏感性分析

6. **风险管理**
   - 仓位管理
   - 风险限制
   - 组合风险管理

7. **策略组合**
   - 多策略组合
   - 动态权重调整

---

## 第一步：策略发想

### 什么是交易策略？

交易策略就是一套规则，告诉你在什么时候买入、什么时候卖出。

### 常见的策略思路

#### 1. 趋势跟踪策略
**思路**：跟随市场趋势，上涨时买入，下跌时卖出
- **例子**：移动平均线交叉策略
  - 当短期均线上穿长期均线 → 买入信号
  - 当短期均线下穿长期均线 → 卖出信号

#### 2. 均值回归策略
**思路**：价格偏离均值后会回归
- **例子**：RSI 超买超卖策略
  - RSI < 30（超卖）→ 买入
  - RSI > 70（超买）→ 卖出

#### 3. 突破策略
**思路**：价格突破关键位置时入场
- **例子**：布林带突破
  - 价格突破上轨 → 买入
  - 价格跌破下轨 → 卖出

### 策略发想步骤

1. **观察市场**：看看价格走势有什么规律
2. **形成假设**：比如"RSI 低于 30 时，价格可能会反弹"
3. **设计规则**：把假设转化为具体的买卖规则
4. **测试验证**：用历史数据回测，看是否有效

### 示例：我们要开发的策略

**策略名称**：RSI 超买超卖策略

**策略思路**：
- RSI（相对强弱指标）衡量价格动量
- RSI < 30：市场超卖，可能反弹 → 买入
- RSI > 70：市场超买，可能回调 → 卖出

**具体规则**：
1. 计算 RSI（周期 14）
2. 当 RSI < 30 时，买入（持仓 100%）
3. 当 RSI > 70 时，卖出（持仓 0%）
4. 其他情况保持当前持仓

---

## 第二步：创建策略

### 方法一：使用模板生成器（推荐新手）

项目提供了策略模板生成器，可以快速创建策略文件：

```bash
# 创建 RSI 策略
python scripts/create_strategy.py --name my_rsi_strategy --type rsi
```

这会自动创建策略文件，你只需要修改参数即可。

### 方法二：手动创建策略

#### 2.1 创建策略文件

在 `src/qtrade/strategy/` 目录下创建新文件，例如 `my_rsi_strategy.py`：

```python
"""
我的 RSI 策略

策略逻辑：
- RSI < 30：买入
- RSI > 70：卖出
"""
from __future__ import annotations
import pandas as pd
from .base import StrategyContext
from . import register_strategy
from ..indicators import calculate_rsi


@register_strategy("my_rsi_strategy")
def generate_positions(df: pd.DataFrame, ctx: StrategyContext, params: dict) -> pd.Series:
    """
    生成持仓信号
    
    Args:
        df: K线数据（包含 open, high, low, close, volume）
        ctx: 策略上下文（包含交易对信息）
        params: 策略参数
    
    Returns:
        持仓比例序列 [0, 1]，0 表示空仓，1 表示满仓
    """
    # 1. 获取参数（如果没有提供，使用默认值）
    period = params.get("period", 14)      # RSI 周期，默认 14
    oversold = params.get("oversold", 30)   # 超卖阈值，默认 30
    overbought = params.get("overbought", 70)  # 超买阈值，默认 70
    
    # 2. 获取收盘价
    close = df["close"]
    
    # 3. 计算 RSI
    rsi = calculate_rsi(close, period=period)
    
    # 4. 生成信号
    # RSI < 30：买入信号（持仓 = 1）
    # RSI > 70：卖出信号（持仓 = 0）
    # 其他情况：保持当前状态（这里简化处理，实际可以用更复杂的逻辑）
    
    pos = pd.Series(0.0, index=df.index)  # 初始化为空仓
    
    # 买入条件
    buy_signal = rsi < oversold
    pos[buy_signal] = 1.0
    
    # 卖出条件
    sell_signal = rsi > overbought
    pos[sell_signal] = 0.0
    
    # 5. 重要：避免未来信息泄露
    # 在 t 时刻的信号，应该在 t+1 时刻执行
    pos = pos.shift(1).fillna(0.0)
    
    return pos
```

#### 2.2 注册策略

在 `src/qtrade/strategy/__init__.py` 中导入你的策略：

```python
from . import my_rsi_strategy  # noqa: E402
```

#### 2.3 配置策略参数

在 `config/base.yaml` 中配置策略：

```yaml
strategy:
  name: "my_rsi_strategy"  # 策略名称（必须与 @register_strategy 中的名称一致）
  params:
    period: 14        # RSI 周期
    oversold: 30      # 超卖阈值
    overbought: 70    # 超买阈值
```

### 策略开发要点

1. **避免未来信息泄露**：必须使用 `shift(1)` 将信号向后移动
2. **返回值**：持仓比例必须在 [0, 1] 范围内
3. **使用指标库**：不要重复实现指标，使用 `qtrade.indicators` 中的函数

---

## 第三步：回测策略

### 3.1 准备数据

首先需要下载历史数据：

```bash
# 下载 BTCUSDT 的 1 小时 K 线数据
python scripts/download_data.py --symbol BTCUSDT --interval 1h --start 2022-01-01
```

### 3.2 配置策略（重要！）

**在运行回测之前，必须先配置要使用的策略。**

编辑 `config/base.yaml` 文件：

```yaml
strategy:
  name: "my_rsi_strategy"  # 这里指定要使用的策略名称
  params:
    period: 14
    oversold: 30
    overbought: 70
```

**重要说明**：
- `strategy.name` 必须与你在代码中 `@register_strategy("策略名称")` 的名称一致
- `strategy.params` 是策略的参数，不同策略需要的参数不同
- 修改配置后，直接运行回测即可，无需修改代码

### 3.3 运行回测

#### 方法一：使用配置文件（推荐新手）

```bash
# 运行回测（会自动读取 config/base.yaml 中的策略配置）
python scripts/run_backtest.py
```

#### 方法二：使用命令行参数（推荐进阶用户）

```bash
# 直接指定策略（覆盖配置文件中的策略）
python scripts/run_backtest.py -s my_rsi_strategy

# 指定配置文件
python scripts/run_backtest.py -c config/my_rsi.yaml

# 只回测指定交易对
python scripts/run_backtest.py -s my_rsi_strategy --symbol BTCUSDT
```

**回测脚本会**：
1. 读取配置文件（或使用命令行参数）
2. 找到对应的策略函数
3. 使用配置的参数运行回测
4. 生成报告和图表到 `reports/{strategy_name}/` 目录

**输出目录组织**：
- 默认情况下，输出会按策略名称自动分类
- 例如：`reports/my_rsi_strategy/stats_BTCUSDT.csv`
- 这样可以避免不同策略的结果互相覆盖

### 3.4 查看结果

回测完成后，会在 `reports/{strategy_name}/` 目录生成：

1. **统计报告** (`stats_BTCUSDT.csv`)
   - 总收益率
   - 最大回撤
   - 夏普比率
   - 胜率
   - 总交易次数

2. **资金曲线图** (`equity_curve_BTCUSDT.png`)
   - 价格走势
   - 买卖信号
   - 资金曲线
   - 回撤图

### 3.5 理解回测结果

#### 关键指标说明

- **总收益率 (Total Return)**：策略的总收益百分比
  - 正数 = 盈利
  - 负数 = 亏损

- **最大回撤 (Max Drawdown)**：从最高点到最低点的最大跌幅
  - 越小越好
  - 超过 50% 通常风险较大

- **夏普比率 (Sharpe Ratio)**：风险调整后的收益
  - > 1.0：不错
  - > 2.0：很好
  - < 0：策略表现差

- **胜率 (Win Rate)**：盈利交易占总交易的比例
  - 通常 50% 以上较好

- **总交易次数 (Total Trades)**：交易频率
  - 太少：可能错过机会
  - 太多：手续费成本高

### 3.5 分析资金曲线

查看 `equity_curve_BTCUSDT.png`：

1. **价格和信号图**：
   - 绿点：买入信号
   - 红点：卖出信号
   - 检查信号是否合理

2. **资金曲线图**：
   - 是否稳定上升？
   - 是否有大幅回撤？
   - 是否长期亏损？

3. **回撤图**：
   - 回撤是否在可接受范围内？

---

## 第四步：优化参数

如果回测结果不理想，可以尝试优化参数。

### 4.1 手动测试不同参数

修改 `config/base.yaml`：

```yaml
strategy:
  name: "my_rsi_strategy"
  params:
    period: 14        # 试试 10, 14, 20
    oversold: 30      # 试试 25, 30, 35
    overbought: 70    # 试试 65, 70, 75
```

然后重新运行回测，比较结果。

### 4.2 使用自动优化工具（推荐）

项目提供了参数优化工具，可以自动测试多个参数组合：

```bash
# 优化 RSI 策略参数
python scripts/optimize_params.py --strategy my_rsi_strategy --metric "Sharpe Ratio"
```

这会：
1. 测试所有参数组合
2. 找到表现最好的参数
3. 生成优化报告

### 4.3 优化指标选择

可以选择不同的优化指标：

- `Total Return [%]`：总收益率
- `Sharpe Ratio`：夏普比率（推荐）
- `Max Drawdown [%]`：最大回撤（越小越好）

```bash
# 优化夏普比率
python scripts/optimize_params.py --strategy my_rsi_strategy --metric "Sharpe Ratio"

# 优化总收益率
python scripts/optimize_params.py --strategy my_rsi_strategy --metric "Total Return [%]"
```

### 4.4 查看优化结果

优化完成后，会生成 `reports/optimization_BTCUSDT.csv`，包含：
- 所有参数组合
- 对应的表现指标
- 最佳参数组合

### ⚠️ 避免过度优化

**重要**：不要过度优化参数！

- 过度优化会导致过拟合
- 在历史数据上表现好，不代表未来也会好
- 应该选择在多个时间窗口都表现稳定的参数

---

## 第五步：验证策略

在实盘前，必须验证策略是否过拟合。

### 5.1 运行验证脚本

```bash
python scripts/validate_strategy.py
```

这会进行两种验证：

#### 1. 滚动窗口验证 (Walk-Forward Analysis)

将数据分成多个训练/测试窗口：
- 在训练集上优化参数
- 在测试集上验证表现

**如何判断过拟合**：
- ✅ 测试集表现接近训练集 → 策略稳定
- ❌ 测试集表现远差于训练集 → 可能过拟合

#### 2. 参数敏感性分析

测试参数微小变化对结果的影响：
- ✅ 参数变化时表现稳定 → 策略稳健
- ❌ 参数微小变化导致结果大幅波动 → 策略不稳定

### 5.2 理解验证结果

查看验证报告：

```
平均训练集收益率: 15.00%
平均测试集收益率: 12.00%
✓ 测试集表现稳定，收益率下降 20.0%
```

**好的结果**：
- 测试集收益率下降 < 30%
- 测试集夏普比率 > 0
- 参数敏感性低

**不好的结果**：
- 测试集收益率下降 > 50%
- 测试集夏普比率 < 0
- 参数敏感性高

如果验证结果不好，需要：
1. 简化策略
2. 减少参数数量
3. 使用更保守的参数
4. 增加样本外测试数据

---

## 第六步：风险管理

在实盘前，应该添加风险管理。

### 6.1 数据质量检查

项目会自动检查数据质量，但你可以手动验证：

```python
from qtrade.data import validate_data_quality

report = validate_data_quality(df)
if not report.is_valid:
    print("数据有问题，需要清洗")
```

### 6.2 设置风险限制

在 `config/base.yaml` 中配置：

```yaml
risk:
  max_position_pct: 1.0      # 最大仓位 100%
  max_drawdown_pct: 0.5     # 最大回撤限制 50%
  max_leverage: 1.0          # 最大杠杆（现货为 1.0）
```

### 6.3 仓位管理

可以使用不同的仓位管理方法：

```python
from qtrade.risk import FixedPositionSizer, KellyPositionSizer

# 固定仓位（80%）
sizer = FixedPositionSizer(position_pct=0.8)

# Kelly 公式（更科学，但需要历史数据）
sizer = KellyPositionSizer(
    win_rate=0.6,      # 胜率 60%
    avg_win=0.05,      # 平均盈利 5%
    avg_loss=0.03,     # 平均亏损 3%
    kelly_fraction=0.5  # 使用 50% Kelly（保守）
)
```

详细文档：[风险管理指南](RISK_MANAGEMENT.md)

---

## 完整示例：RSI 策略

让我们完整地开发一个 RSI 策略。

### 步骤 1：创建策略文件

创建 `src/qtrade/strategy/my_rsi_strategy.py`：

```python
"""
RSI 超买超卖策略

策略逻辑：
- RSI < 30：买入
- RSI > 70：卖出
"""
from __future__ import annotations
import pandas as pd
from .base import StrategyContext
from . import register_strategy
from ..indicators import calculate_rsi


@register_strategy("my_rsi_strategy")
def generate_positions(df: pd.DataFrame, ctx: StrategyContext, params: dict) -> pd.Series:
    period = params.get("period", 14)
    oversold = params.get("oversold", 30)
    overbought = params.get("overbought", 70)
    
    close = df["close"]
    rsi = calculate_rsi(close, period=period)
    
    pos = pd.Series(0.0, index=df.index)
    pos[rsi < oversold] = 1.0  # 超卖买入
    pos[rsi > overbought] = 0.0  # 超买卖出
    
    # 避免未来信息泄露
    pos = pos.shift(1).fillna(0.0)
    
    return pos
```

### 步骤 2：注册策略

在 `src/qtrade/strategy/__init__.py` 中添加：

```python
from . import my_rsi_strategy  # noqa: E402
```

### 步骤 3：配置策略（⭐ 重要！）

**这是关键步骤**：回测脚本会读取配置文件来确定使用哪个策略。

编辑 `config/base.yaml` 文件：

```yaml
strategy:
  name: "my_rsi_strategy"  # ⚠️ 必须与 @register_strategy("my_rsi_strategy") 中的名称完全一致
  params:
    period: 14
    oversold: 30
    overbought: 70
```

**工作原理**：
1. 当你运行 `python scripts/run_backtest.py` 时
2. 脚本会读取 `config/base.yaml` 文件
3. 根据 `strategy.name` 找到对应的策略函数（通过 `@register_strategy` 注册的）
4. 使用 `strategy.params` 中的参数运行策略
5. 对配置中的所有交易对进行回测

**如何切换策略？**
只需要修改配置文件，无需修改代码：
```yaml
strategy:
  name: "ema_cross"  # 切换到 EMA 交叉策略
  params:
    fast: 20
    slow: 60
```

**如果策略名称错误会怎样？**
会报错并显示所有可用的策略：
```
ValueError: Strategy 'wrong_name' not found. 
Available: ['ema_cross', 'rsi', 'my_rsi_strategy', ...]
```

### 步骤 4：下载数据

```bash
python scripts/download_data.py --symbol BTCUSDT --interval 1h --start 2022-01-01
```

### 步骤 5：运行回测

```bash
python scripts/run_backtest.py
```

### 步骤 6：查看结果

查看 `reports/stats_BTCUSDT.csv` 和 `reports/equity_curve_BTCUSDT.png`

### 步骤 7：优化参数

```bash
python scripts/optimize_params.py --strategy my_rsi_strategy --metric "Sharpe Ratio"
```

### 步骤 8：验证策略

```bash
python scripts/validate_strategy.py
```

### 步骤 9：如果结果好，可以实盘

如果验证通过，可以考虑实盘交易。

---

## 常见问题

### Q1: 策略总是亏损怎么办？

**可能原因**：
1. 策略逻辑有问题
2. 参数不合适
3. 市场环境不适合该策略

**解决方法**：
1. 检查信号是否合理（查看资金曲线图）
2. 尝试不同参数
3. 尝试不同交易对
4. 简化策略逻辑

### Q2: 回测结果很好，但验证失败？

**这是过拟合的典型表现**。

**解决方法**：
1. 简化策略
2. 减少参数
3. 使用更保守的参数
4. 增加更多历史数据

### Q3: 如何选择优化指标？

**推荐顺序**：
1. **夏普比率**：综合考虑收益和风险
2. **总收益率**：如果风险可控
3. **最大回撤**：如果风险承受能力低

### Q4: 策略在某个交易对上表现好，另一个不好？

**这是正常的**，不同资产有不同的特性。

**解决方法**：
1. 为不同资产使用不同参数
2. 使用策略组合
3. 只交易适合该策略的资产

---

## 下一步

1. **学习更多策略**：查看 `STRATEGY_DEVELOPMENT.md`
2. **了解风险管理**：查看 `RISK_MANAGEMENT.md`
3. **学习策略组合**：查看 `STRATEGY_PORTFOLIO.md`
4. **数据质量**：查看 `DATA_QUALITY.md`

---

## 总结

完整的策略开发流程：

1. ✅ **策略发想**：观察市场，形成假设
2. ✅ **创建策略**：编写代码，实现逻辑
3. ✅ **回测**：用历史数据测试
4. ✅ **优化**：找到最佳参数
5. ✅ **验证**：确保不过拟合
6. ✅ **风险管理**：添加风险控制
7. ✅ **实盘**：如果一切顺利

记住：**量化交易不是一夜暴富，而是持续改进的过程**。

祝你交易顺利！🎉

