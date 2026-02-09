"""
策略開發示例

這個檔案展示了如何開發一個新策略。
你可以複製這個檔案並修改策略邏輯。
"""
from __future__ import annotations
import pandas as pd
from .base import StrategyContext
from . import register_strategy


@register_strategy("example_strategy")
def generate_positions(df: pd.DataFrame, ctx: StrategyContext, params: dict) -> pd.Series:
    """
    示例策略：雙移動平均線交叉
    
    策略邏輯：
    - 當快速均線上穿慢速均線時，買入（持倉 = 1.0）
    - 當快速均線下穿慢速均線時，賣出（持倉 = 0.0）
    
    Args:
        df: K線數據，包含以下列：
            - open: 開盤價
            - high: 最高價
            - low: 最低價
            - close: 收盤價
            - volume: 成交量
        ctx: 策略上下文，包含 symbol 等資訊
        params: 策略參數字典，從 config/base.yaml 中讀取
    
    Returns:
        pd.Series: 持倉比例序列，索引與 df 相同
        - 1.0 表示滿倉（100% 資金投入）
        - 0.0 表示空倉（0% 資金投入）
        - 0.5 表示半倉（50% 資金投入）
    """
    # 從參數中獲取均線週期，如果沒有則使用預設值
    fast_period = int(params.get("fast", 20))
    slow_period = int(params.get("slow", 60))
    
    # 獲取收盤價
    close = df["close"]
    
    # 計算快速和慢速移動平均線
    ma_fast = close.rolling(window=fast_period).mean()
    ma_slow = close.rolling(window=slow_period).mean()
    
    # 生成交易信號
    # 當快速均線 > 慢速均線時，信號為 1（買入）
    # 當快速均線 < 慢速均線時，信號為 0（賣出）
    signal = (ma_fast > ma_slow).astype(float)
    
    # ⚠️ 重要：避免未來資訊洩漏（Look-ahead Bias）
    # 在 t 時刻，我們只能看到 t 時刻及之前的數據
    # 但信號是在 t 時刻的收盤價計算出來的
    # 所以我們應該在 t+1 時刻執行交易
    # 使用 shift(1) 將信號向後移動 1 個 bar
    pos = signal.shift(1).fillna(0.0)
    
    # 確保持倉比例在 [0, 1] 之間
    pos = pos.clip(0.0, 1.0)
    
    return pos


# 如果你想開發更複雜的策略，可以參考以下示例：


@register_strategy("example_strategy_with_filter")
def generate_positions_with_filter(df: pd.DataFrame, ctx: StrategyContext, params: dict) -> pd.Series:
    """
    帶過濾條件的策略示例
    
    在雙均線策略基礎上，添加成交量過濾：
    - 只有在成交量大於平均值時才交易
    """
    fast_period = int(params.get("fast", 20))
    slow_period = int(params.get("slow", 60))
    volume_threshold = float(params.get("volume_threshold", 1.2))  # 成交量閾值倍數
    
    close = df["close"]
    volume = df["volume"]
    
    # 計算均線
    ma_fast = close.rolling(window=fast_period).mean()
    ma_slow = close.rolling(window=slow_period).mean()
    
    # 計算成交量均線
    volume_ma = volume.rolling(window=20).mean()
    
    # 生成信號
    ma_signal = (ma_fast > ma_slow).astype(float)
    
    # 添加成交量過濾
    volume_filter = (volume > volume_ma * volume_threshold).astype(float)
    
    # 組合信號：均線信號 AND 成交量過濾
    signal = ma_signal * volume_filter
    
    # 避免未來資訊洩漏
    pos = signal.shift(1).fillna(0.0).clip(0.0, 1.0)
    
    return pos
