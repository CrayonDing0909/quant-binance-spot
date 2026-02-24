from __future__ import annotations
import functools
from typing import Callable
import pandas as pd
from .base import StrategyContext

# 策略註冊表
_STRATEGY_REGISTRY: dict[str, Callable] = {}
_RAW_STRATEGY_REGISTRY: dict[str, Callable] = {}  # 原始函數（測試用）


def register_strategy(name: str, *, auto_delay: bool = True):
    """
    裝飾器：註冊策略函數，並自動套用安全防護

    ═══════════════════════════════════════════════════════
    ⚡ 防 Look-Ahead 機制（框架層強制，策略無法繞過）
    ═══════════════════════════════════════════════════════

    自動套用（策略函數不需要、也不應該自己做）：
      1. signal_delay shift（ctx.signal_delay > 0 時）
         → 回測中消除 look-ahead bias
         → 實盤中 signal_delay=0，不影響
      2. direction clip
         → spot 不做空（clip lower=0）
         → long_only/short_only 限制

    策略函數只需要：
      - 接收 (df, ctx, params)
      - 回傳 raw position Series [-1, 1]
      - 不需要 shift、不需要 clip

    Args:
        name: 策略名稱
        auto_delay: 是否自動套用 signal_delay（預設 True）
                    設為 False = 策略自行管理 delay 位置
                    （僅用於有 exit_rules 需要在 delay 後處理的策略）

    Example:
        @register_strategy("my_strategy")
        def my_strategy(df, ctx, params):
            # 只需回傳 raw signal，框架自動處理 delay + clip
            close = df["close"]
            signal = compute_something(close)
            return signal
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def safe_wrapper(df: pd.DataFrame, ctx: StrategyContext, params: dict) -> pd.Series:
            # ── 1. 呼叫原始策略函數 ──
            raw_pos = func(df, ctx, params)

            # ── 2. signal_delay shift（消除 look-ahead bias）──
            if auto_delay:
                signal_delay = getattr(ctx, "signal_delay", 0)
                if signal_delay > 0:
                    raw_pos = raw_pos.shift(signal_delay).fillna(0.0)

            # ── 3. direction clip ──
            if auto_delay:
                if not ctx.can_short:
                    raw_pos = raw_pos.clip(lower=0.0)
                if not ctx.can_long:
                    raw_pos = raw_pos.clip(upper=0.0)

            return raw_pos

        _STRATEGY_REGISTRY[name] = safe_wrapper
        _RAW_STRATEGY_REGISTRY[name] = func  # 保留原始函數供測試
        return func  # 回傳原始函數，讓模組內部還能直接呼叫
    return decorator


def get_strategy(name: str) -> Callable:
    """根據名稱獲取策略函數（已套用安全防護）"""
    if name not in _STRATEGY_REGISTRY:
        available = list(_STRATEGY_REGISTRY.keys())
        raise ValueError(f"Strategy '{name}' not found. Available: {available}")
    return _STRATEGY_REGISTRY[name]


def get_raw_strategy(name: str) -> Callable:
    """
    根據名稱獲取原始策略函數（無安全防護）

    ⚠️ 僅用於測試和調試。生產環境請用 get_strategy()。
    """
    if name not in _RAW_STRATEGY_REGISTRY:
        available = list(_RAW_STRATEGY_REGISTRY.keys())
        raise ValueError(f"Raw strategy '{name}' not found. Available: {available}")
    return _RAW_STRATEGY_REGISTRY[name]


def list_strategies() -> list[str]:
    """列出所有已註冊的策略名稱"""
    return list(_STRATEGY_REGISTRY.keys())


# 導入策略模組以觸發註冊
from . import rsi_adx_atr_strategy  # noqa: E402
from . import tsmom_strategy  # noqa: E402
from . import xsmom_strategy  # noqa: E402
from . import nwkl_strategy  # noqa: E402
from . import nw_envelope_regime_strategy  # noqa: E402
from . import breakout_vol_strategy  # noqa: E402
from . import funding_carry_strategy  # noqa: E402
from . import mr_microstructure_strategy  # noqa: E402
from . import oi_bb_rv_strategy  # noqa: E402
from . import mean_revert_liquidity  # noqa: E402
from . import x_model_weekend  # noqa: E402
from . import oi_liq_bounce_strategy  # noqa: E402
from . import tsmom_mr_composite_strategy  # noqa: E402
from . import tsmom_carry_v2_strategy  # noqa: E402
from . import meta_blend_strategy  # noqa: E402