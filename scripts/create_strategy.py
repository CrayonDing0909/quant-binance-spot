"""
策略模板生成器

快速创建新策略文件的工具。

使用方法:
    python scripts/create_strategy.py --name my_strategy --type rsi
    python scripts/create_strategy.py --name my_strategy --type custom
"""
from __future__ import annotations
import argparse
from pathlib import Path


STRATEGY_TEMPLATES = {
    "rsi": '''"""
RSI 策略模板

基于 RSI 指标的交易策略。
"""
from __future__ import annotations
import pandas as pd
from ..strategy.base import StrategyContext
from ..strategy import register_strategy
from ..indicators import calculate_rsi


@register_strategy("{strategy_name}")
def generate_positions(df: pd.DataFrame, ctx: StrategyContext, params: dict) -> pd.Series:
    """
    RSI 策略
    
    Args:
        df: K线数据
        ctx: 策略上下文
        params: 策略参数
            - period: RSI 周期，默认 14
            - oversold: 超卖阈值，默认 30
            - overbought: 超买阈值，默认 70
    
    Returns:
        持仓比例序列 [0, 1]
    """
    period = int(params.get("period", 14))
    oversold = float(params.get("oversold", 30))
    overbought = float(params.get("overbought", 70))
    
    close = df["close"]
    rsi = calculate_rsi(close, period)
    
    # 生成信号
    signal = (rsi < oversold).astype(float)
    
    # 避免未来信息泄露
    pos = signal.shift(1).fillna(0.0).clip(0.0, 1.0)
    
    return pos
''',
    
    "ema": '''"""
EMA 交叉策略模板

基于双 EMA 交叉的交易策略。
"""
from __future__ import annotations
import pandas as pd
from ..strategy.base import StrategyContext
from ..strategy import register_strategy
from ..indicators import calculate_ema


@register_strategy("{strategy_name}")
def generate_positions(df: pd.DataFrame, ctx: StrategyContext, params: dict) -> pd.Series:
    """
    EMA 交叉策略
    
    Args:
        df: K线数据
        ctx: 策略上下文
        params: 策略参数
            - fast: 快线周期，默认 20
            - slow: 慢线周期，默认 60
    
    Returns:
        持仓比例序列 [0, 1]
    """
    fast = int(params.get("fast", 20))
    slow = int(params.get("slow", 60))
    
    close = df["close"]
    ema_fast = calculate_ema(close, fast)
    ema_slow = calculate_ema(close, slow)
    
    # 生成信号
    signal = (ema_fast > ema_slow).astype(float)
    
    # 避免未来信息泄露
    pos = signal.shift(1).fillna(0.0).clip(0.0, 1.0)
    
    return pos
''',
    
    "macd": '''"""
MACD 策略模板

基于 MACD 指标的交易策略。
"""
from __future__ import annotations
import pandas as pd
from ..strategy.base import StrategyContext
from ..strategy import register_strategy
from ..indicators import calculate_macd


@register_strategy("{strategy_name}")
def generate_positions(df: pd.DataFrame, ctx: StrategyContext, params: dict) -> pd.Series:
    """
    MACD 策略
    
    Args:
        df: K线数据
        ctx: 策略上下文
        params: 策略参数
            - fast_period: 快线周期，默认 12
            - slow_period: 慢线周期，默认 26
            - signal_period: 信号线周期，默认 9
    
    Returns:
        持仓比例序列 [0, 1]
    """
    fast_period = int(params.get("fast_period", 12))
    slow_period = int(params.get("slow_period", 26))
    signal_period = int(params.get("signal_period", 9))
    
    close = df["close"]
    macd_data = calculate_macd(close, fast_period, slow_period, signal_period)
    
    # MACD 线上穿信号线 -> 买入
    signal = (macd_data["macd"] > macd_data["signal"]).astype(float)
    
    # 避免未来信息泄露
    pos = signal.shift(1).fillna(0.0).clip(0.0, 1.0)
    
    return pos
''',
    
    "custom": '''"""
自定义策略模板

你可以在这里实现自己的策略逻辑。
"""
from __future__ import annotations
import pandas as pd
from ..strategy.base import StrategyContext
from ..strategy import register_strategy


@register_strategy("{strategy_name}")
def generate_positions(df: pd.DataFrame, ctx: StrategyContext, params: dict) -> pd.Series:
    """
    自定义策略
    
    Args:
        df: K线数据，包含以下列：
            - open: 开盘价
            - high: 最高价
            - low: 最低价
            - close: 收盘价
            - volume: 成交量
        ctx: 策略上下文，包含 symbol 等信息
        params: 策略参数，从 config 中读取
    
    Returns:
        持仓比例序列 [0, 1]
        - 1.0 = 满仓
        - 0.0 = 空仓
    """
    # TODO: 实现你的策略逻辑
    close = df["close"]
    
    # 示例：简单策略
    signal = (close > close.shift(1)).astype(float)
    
    # ⚠️ 重要：避免未来信息泄露，必须 shift(1)
    pos = signal.shift(1).fillna(0.0).clip(0.0, 1.0)
    
    return pos
''',
}


def create_strategy_file(strategy_name: str, strategy_type: str = "custom") -> None:
    """
    创建策略文件
    
    Args:
        strategy_name: 策略名称
        strategy_type: 策略类型（rsi, ema, macd, custom）
    """
    if strategy_type not in STRATEGY_TEMPLATES:
        raise ValueError(f"Unknown strategy type: {strategy_type}. Available: {list(STRATEGY_TEMPLATES.keys())}")
    
    # 确定文件路径
    project_root = Path(__file__).parent.parent
    strategy_dir = project_root / "src" / "qtrade" / "strategy"
    strategy_file = strategy_dir / f"{strategy_name}.py"
    
    # 检查文件是否已存在
    if strategy_file.exists():
        response = input(f"文件 {strategy_file} 已存在，是否覆盖？(y/N): ")
        if response.lower() != 'y':
            print("已取消")
            return
    
    # 生成策略代码
    template = STRATEGY_TEMPLATES[strategy_type]
    code = template.format(strategy_name=strategy_name)
    
    # 写入文件
    strategy_file.write_text(code, encoding="utf-8")
    print(f"✅ 已创建策略文件: {strategy_file}")
    
    # 更新 __init__.py
    init_file = strategy_dir / "__init__.py"
    init_content = init_file.read_text(encoding="utf-8")
    
    # 检查是否已导入
    import_line = f"from . import {strategy_name}  # noqa: E402"
    if import_line not in init_content:
        # 找到最后一个导入语句的位置
        lines = init_content.split("\n")
        last_import_idx = 0
        for i, line in enumerate(lines):
            if line.startswith("from . import") or line.startswith("import"):
                last_import_idx = i
        
        # 在最后一个导入后添加新导入
        lines.insert(last_import_idx + 1, f"from . import {strategy_name}  # noqa: E402")
        init_file.write_text("\n".join(lines), encoding="utf-8")
        print(f"✅ 已更新 {init_file}")
    else:
        print(f"ℹ️  {init_file} 中已存在导入语句")
    
    # 生成配置示例
    config_example = f'''# {strategy_name} 策略配置示例
strategy:
  name: "{strategy_name}"
  params:
    # TODO: 根据策略类型添加参数
    param1: value1
    param2: value2
'''
    config_file = project_root / "config" / f"{strategy_name}_example.yaml"
    if not config_file.exists():
        config_file.write_text(config_example, encoding="utf-8")
        print(f"✅ 已创建配置示例: {config_file}")
    
    print(f"\n📝 下一步:")
    print(f"1. 编辑策略文件: {strategy_file}")
    print(f"2. 配置参数: {config_file}")
    print(f"3. 运行回测: python scripts/run_backtest.py")


def main():
    parser = argparse.ArgumentParser(description="创建新策略文件")
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="策略名称（将用作文件名和注册名）"
    )
    parser.add_argument(
        "--type",
        type=str,
        default="custom",
        choices=list(STRATEGY_TEMPLATES.keys()),
        help="策略类型（rsi, ema, macd, custom）"
    )
    
    args = parser.parse_args()
    
    try:
        create_strategy_file(args.name, args.type)
    except Exception as e:
        print(f"❌ 错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

