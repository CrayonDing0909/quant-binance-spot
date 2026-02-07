"""
策略开发示例

这个文件展示了如何开发一个新策略。
你可以复制这个文件并修改策略逻辑。
"""
from __future__ import annotations
import pandas as pd
from .base import StrategyContext
from . import register_strategy


@register_strategy("example_strategy")
def generate_positions(df: pd.DataFrame, ctx: StrategyContext, params: dict) -> pd.Series:
    """
    示例策略：双移动平均线交叉
    
    策略逻辑：
    - 当快速均线上穿慢速均线时，买入（持仓 = 1.0）
    - 当快速均线下穿慢速均线时，卖出（持仓 = 0.0）
    
    Args:
        df: K线数据，包含以下列：
            - open: 开盘价
            - high: 最高价
            - low: 最低价
            - close: 收盘价
            - volume: 成交量
        ctx: 策略上下文，包含 symbol 等信息
        params: 策略参数字典，从 config/base.yaml 中读取
    
    Returns:
        pd.Series: 持仓比例序列，索引与 df 相同
        - 1.0 表示满仓（100% 资金投入）
        - 0.0 表示空仓（0% 资金投入）
        - 0.5 表示半仓（50% 资金投入）
    """
    # 从参数中获取均线周期，如果没有则使用默认值
    fast_period = int(params.get("fast", 20))
    slow_period = int(params.get("slow", 60))
    
    # 获取收盘价
    close = df["close"]
    
    # 计算快速和慢速移动平均线
    ma_fast = close.rolling(window=fast_period).mean()
    ma_slow = close.rolling(window=slow_period).mean()
    
    # 生成交易信号
    # 当快速均线 > 慢速均线时，信号为 1（买入）
    # 当快速均线 < 慢速均线时，信号为 0（卖出）
    signal = (ma_fast > ma_slow).astype(float)
    
    # ⚠️ 重要：避免未来信息泄露（Look-ahead Bias）
    # 在 t 时刻，我们只能看到 t 时刻及之前的数据
    # 但信号是在 t 时刻的收盘价计算出来的
    # 所以我们应该在 t+1 时刻执行交易
    # 使用 shift(1) 将信号向后移动 1 个 bar
    pos = signal.shift(1).fillna(0.0)
    
    # 确保持仓比例在 [0, 1] 之间
    pos = pos.clip(0.0, 1.0)
    
    return pos


# 如果你想开发更复杂的策略，可以参考以下示例：


@register_strategy("example_strategy_with_filter")
def generate_positions_with_filter(df: pd.DataFrame, ctx: StrategyContext, params: dict) -> pd.Series:
    """
    带过滤条件的策略示例
    
    在双均线策略基础上，添加成交量过滤：
    - 只有在成交量大于平均值时才交易
    """
    fast_period = int(params.get("fast", 20))
    slow_period = int(params.get("slow", 60))
    volume_threshold = float(params.get("volume_threshold", 1.2))  # 成交量阈值倍数
    
    close = df["close"]
    volume = df["volume"]
    
    # 计算均线
    ma_fast = close.rolling(window=fast_period).mean()
    ma_slow = close.rolling(window=slow_period).mean()
    
    # 计算成交量均线
    volume_ma = volume.rolling(window=20).mean()
    
    # 生成信号
    ma_signal = (ma_fast > ma_slow).astype(float)
    
    # 添加成交量过滤
    volume_filter = (volume > volume_ma * volume_threshold).astype(float)
    
    # 组合信号：均线信号 AND 成交量过滤
    signal = ma_signal * volume_filter
    
    # 避免未来信息泄露
    pos = signal.shift(1).fillna(0.0).clip(0.0, 1.0)
    
    return pos

