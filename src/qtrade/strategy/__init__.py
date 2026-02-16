from __future__ import annotations
from typing import Callable
from .base import StrategyContext

# 策略註冊表
_STRATEGY_REGISTRY: dict[str, Callable] = {}


def register_strategy(name: str):
    """裝飾器：註冊策略函數"""
    def decorator(func: Callable):
        _STRATEGY_REGISTRY[name] = func
        return func
    return decorator


def get_strategy(name: str) -> Callable:
    """根據名稱獲取策略函數"""
    if name not in _STRATEGY_REGISTRY:
        raise ValueError(f"Strategy '{name}' not found. Available: {list(_STRATEGY_REGISTRY.keys())}")
    return _STRATEGY_REGISTRY[name]


# 導入策略模組以觸發註冊
from . import ema_cross  # noqa: E402
from . import rsi_strategy  # noqa: E402
from . import smc_strategy  # noqa: E402
from . import my_strategy  # noqa: E402
from . import example_stateful_strategy  # noqa: E402
from . import rsi_adx_atr_strategy  # noqa: E402
from . import rsi_adx_atr_enhanced  # noqa: E402
from . import macd_momentum  # noqa: E402
from . import bb_mean_reversion  # noqa: E402
from . import multi_factor  # noqa: E402
from . import ensemble_strategy  # noqa: E402