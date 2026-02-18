"""
策略基類和介面定義
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import pandas as pd


@dataclass(frozen=True)
class StrategyContext:
    """
    策略上下文資訊
    
    Attributes:
        symbol: 交易對
        interval: K 線週期
        market_type: 市場類型 "spot" 或 "futures"
        direction: 交易方向 "both", "long_only", "short_only"
        signal_delay: 信號延遲 bar 數（trade_on=next_open → 1）
    """
    symbol: str
    interval: str = "1h"
    market_type: str = "spot"
    direction: str = "both"  # "both", "long_only", "short_only"
    signal_delay: int = 0    # 0=同bar執行, 1=next_open（消除 look-ahead）

    @property
    def supports_short(self) -> bool:
        """是否支援做空（futures 且 direction 不是 long_only）"""
        return self.market_type == "futures" and self.direction != "long_only"

    @property
    def can_long(self) -> bool:
        """是否可以做多"""
        return self.direction != "short_only"

    @property
    def can_short(self) -> bool:
        """是否可以做空"""
        return self.market_type == "futures" and self.direction != "long_only"

    @property
    def is_futures(self) -> bool:
        """是否為合約模式"""
        return self.market_type == "futures"


@dataclass
class StrategyState:
    """
    策略狀態
    
    用於維護策略的內部狀態，如：
    - 當前持倉
    - 止損價格
    - 止盈價格
    - 其他自定義狀態
    """
    current_position: float = 0.0  # 當前持倉比例 [0, 1]
    entry_price: float | None = None  # 入場價格
    stop_loss: float | None = None  # 止損價格
    take_profit: float | None = None  # 止盈價格
    custom_state: dict[str, Any] = field(default_factory=dict)  # 自定義狀態


class BaseStrategy(ABC):
    """
    策略基類
    
    支援狀態管理的策略基類。如果你的策略需要維護狀態
    （如止損、止盈），可以繼承此類。
    
    示例:
        class MyStrategy(BaseStrategy):
            def generate_positions(self, df, ctx, params, state):
                # 使用 state 維護策略狀態
                if state.current_position > 0:
                    # 檢查止損
                    if df["close"].iloc[-1] < state.stop_loss:
                        return 0.0
                return 1.0
    """
    
    def __init__(self, name: str):
        self.name = name
        self._state: StrategyState | None = None
    
    @abstractmethod
    def generate_positions(
        self,
        df: pd.DataFrame,
        ctx: StrategyContext,
        params: dict,
        state: StrategyState | None = None
    ) -> pd.Series:
        """
        生成持倉信號
        
        Args:
            df: K線數據
            ctx: 策略上下文
            params: 策略參數
            state: 策略狀態（可選）
        
        Returns:
            持倉比例序列 [0, 1]
        """
        raise NotImplementedError
    
    def update_state(
        self,
        state: StrategyState,
        df: pd.DataFrame,
        position: pd.Series
    ) -> StrategyState:
        """
        更新策略狀態
        
        子類可以重寫此方法來實現自定義狀態更新邏輯。
        
        Args:
            state: 當前狀態
            df: K線數據
            position: 當前持倉序列
        
        Returns:
            更新後的狀態
        """
        # 預設實現：更新當前持倉
        if len(position) > 0:
            state.current_position = float(position.iloc[-1])
        
        return state


# 為了向後相容，保留函數式策略介面
def generate_positions(df: pd.DataFrame, ctx: StrategyContext, params: dict) -> pd.Series:
    """
    函數式策略介面（向後相容）
    
    這是函數式策略的介面定義。實際策略應該使用 @register_strategy 裝飾器。
    
    Returns:
        持倉比例序列:
        - Spot 模式: [0, 1]，0 = 空倉，1 = 滿倉做多
        - Futures 模式: [-1, 1]，-1 = 滿倉做空，0 = 空倉，1 = 滿倉做多
        
    Note:
        策略可以統一輸出 [-1, 1]，Spot 模式下負數會被自動 clip 到 0。
        使用 ctx.supports_short 或 ctx.is_futures 判斷是否支援做空。
    """
    raise NotImplementedError
