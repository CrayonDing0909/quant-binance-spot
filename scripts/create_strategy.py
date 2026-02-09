"""
策略模板生成器

快速建立新策略檔案的工具。

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

基於 RSI 指標的交易策略。
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
        df: K線數據
        ctx: 策略上下文
        params: 策略參數
            - period: RSI 週期，預設 14
            - oversold: 超賣閾值，預設 30
            - overbought: 超買閾值，預設 70
    
    Returns:
        持倉比例序列 [0, 1]
    """
    period = int(params.get("period", 14))
    oversold = float(params.get("oversold", 30))
    overbought = float(params.get("overbought", 70))
    
    close = df["close"]
    rsi = calculate_rsi(close, period)
    
    # 生成信號
    signal = (rsi < oversold).astype(float)
    
    # 避免未來資訊洩露
    pos = signal.shift(1).fillna(0.0).clip(0.0, 1.0)
    
    return pos
''',
    
    "ema": '''"""
EMA 交叉策略模板

基於雙 EMA 交叉的交易策略。
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
        df: K線數據
        ctx: 策略上下文
        params: 策略參數
            - fast: 快線週期，預設 20
            - slow: 慢線週期，預設 60
    
    Returns:
        持倉比例序列 [0, 1]
    """
    fast = int(params.get("fast", 20))
    slow = int(params.get("slow", 60))
    
    close = df["close"]
    ema_fast = calculate_ema(close, fast)
    ema_slow = calculate_ema(close, slow)
    
    # 生成信號
    signal = (ema_fast > ema_slow).astype(float)
    
    # 避免未來資訊洩露
    pos = signal.shift(1).fillna(0.0).clip(0.0, 1.0)
    
    return pos
''',
    
    "macd": '''"""
MACD 策略模板

基於 MACD 指標的交易策略。
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
        df: K線數據
        ctx: 策略上下文
        params: 策略參數
            - fast_period: 快線週期，預設 12
            - slow_period: 慢線週期，預設 26
            - signal_period: 信號線週期，預設 9
    
    Returns:
        持倉比例序列 [0, 1]
    """
    fast_period = int(params.get("fast_period", 12))
    slow_period = int(params.get("slow_period", 26))
    signal_period = int(params.get("signal_period", 9))
    
    close = df["close"]
    macd_data = calculate_macd(close, fast_period, slow_period, signal_period)
    
    # MACD 線上穿信號線 -> 買入
    signal = (macd_data["macd"] > macd_data["signal"]).astype(float)
    
    # 避免未來資訊洩露
    pos = signal.shift(1).fillna(0.0).clip(0.0, 1.0)
    
    return pos
''',
    
    "custom": '''"""
自定義策略模板

你可以在這裡實現自己的策略邏輯。
"""
from __future__ import annotations
import pandas as pd
from ..strategy.base import StrategyContext
from ..strategy import register_strategy


@register_strategy("{strategy_name}")
def generate_positions(df: pd.DataFrame, ctx: StrategyContext, params: dict) -> pd.Series:
    """
    自定義策略
    
    Args:
        df: K線數據，包含以下列：
            - open: 開盤價
            - high: 最高價
            - low: 最低價
            - close: 收盤價
            - volume: 成交量
        ctx: 策略上下文，包含 symbol 等資訊
        params: 策略參數，從 config 中讀取
    
    Returns:
        持倉比例序列 [0, 1]
        - 1.0 = 滿倉
        - 0.0 = 空倉
    """
    # TODO: 實現你的策略邏輯
    close = df["close"]
    
    # 示例：簡單策略
    signal = (close > close.shift(1)).astype(float)
    
    # ⚠️ 重要：避免未來資訊洩露，必須 shift(1)
    pos = signal.shift(1).fillna(0.0).clip(0.0, 1.0)
    
    return pos
''',
}


def create_strategy_file(strategy_name: str, strategy_type: str = "custom") -> None:
    """
    建立策略檔案
    
    Args:
        strategy_name: 策略名稱
        strategy_type: 策略類型（rsi, ema, macd, custom）
    """
    if strategy_type not in STRATEGY_TEMPLATES:
        raise ValueError(f"Unknown strategy type: {strategy_type}. Available: {list(STRATEGY_TEMPLATES.keys())}")
    
    # 確定檔案路徑
    project_root = Path(__file__).parent.parent
    strategy_dir = project_root / "src" / "qtrade" / "strategy"
    strategy_file = strategy_dir / f"{strategy_name}.py"
    
    # 檢查檔案是否已存在
    if strategy_file.exists():
        response = input(f"檔案 {strategy_file} 已存在，是否覆蓋？(y/N): ")
        if response.lower() != 'y':
            print("已取消")
            return
    
    # 生成策略程式碼
    template = STRATEGY_TEMPLATES[strategy_type]
    code = template.format(strategy_name=strategy_name)
    
    # 寫入檔案
    strategy_file.write_text(code, encoding="utf-8")
    print(f"✅ 已建立策略檔案: {strategy_file}")
    
    # 更新 __init__.py
    init_file = strategy_dir / "__init__.py"
    init_content = init_file.read_text(encoding="utf-8")
    
    # 檢查是否已導入
    import_line = f"from . import {strategy_name}  # noqa: E402"
    if import_line not in init_content:
        # 找到最後一個導入語句的位置
        lines = init_content.split("\n")
        last_import_idx = 0
        for i, line in enumerate(lines):
            if line.startswith("from . import") or line.startswith("import"):
                last_import_idx = i
        
        # 在最後一個導入後添加新導入
        lines.insert(last_import_idx + 1, f"from . import {strategy_name}  # noqa: E402")
        init_file.write_text("\n".join(lines), encoding="utf-8")
        print(f"✅ 已更新 {init_file}")
    else:
        print(f"ℹ️  {init_file} 中已存在導入語句")
    
    # 生成配置示例
    config_example = f'''# {strategy_name} 策略配置示例
strategy:
  name: "{strategy_name}"
  params:
    # TODO: 根據策略類型添加參數
    param1: value1
    param2: value2
'''
    config_file = project_root / "config" / f"{strategy_name}_example.yaml"
    if not config_file.exists():
        config_file.write_text(config_example, encoding="utf-8")
        print(f"✅ 已建立配置示例: {config_file}")
    
    print(f"\n📝 下一步:")
    print(f"1. 編輯策略檔案: {strategy_file}")
    print(f"2. 配置參數: {config_file}")
    print(f"3. 運行回測: python scripts/run_backtest.py")


def main():
    parser = argparse.ArgumentParser(description="建立新策略檔案")
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="策略名稱（將用作檔案名和註冊名）"
    )
    parser.add_argument(
        "--type",
        type=str,
        default="custom",
        choices=list(STRATEGY_TEMPLATES.keys()),
        help="策略類型（rsi, ema, macd, custom）"
    )
    
    args = parser.parse_args()
    
    try:
        create_strategy_file(args.name, args.type)
    except Exception as e:
        print(f"❌ 錯誤: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
