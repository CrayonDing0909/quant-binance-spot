"""
通用出场规则（止损 / 止盈 / 移动止损 / 冷却期）

设计理念：
    策略只负责产生「进出场信号」（raw position），
    出场规则作为后处理器，叠加在任何策略上。

用法：
    from qtrade.strategy.exit_rules import apply_exit_rules

    raw_pos = my_indicator_logic(df, params)
    pos = apply_exit_rules(
        df, raw_pos,
        stop_loss_atr=2.0,      # 止损 = 入场价 - 2×ATR
        take_profit_atr=3.0,    # 止盈 = 入场价 + 3×ATR
        trailing_stop_atr=2.5,  # 移动止损 = 最高价 - 2.5×ATR
        cooldown_bars=6,        # 止损后 6 根 bar 不进场
    )
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from ..indicators.atr import calculate_atr


def apply_exit_rules(
    df: pd.DataFrame,
    raw_pos: pd.Series,
    stop_loss_atr: float | None = 2.0,
    take_profit_atr: float | None = 3.0,
    trailing_stop_atr: float | None = None,
    stop_loss_pct: float | None = None,
    take_profit_pct: float | None = None,
    atr_period: int = 14,
    cooldown_bars: int = 0,
) -> pd.Series:
    """
    对原始持仓信号叠加出场规则

    优先级：ATR-based > 固定百分比
    如果 stop_loss_atr 和 stop_loss_pct 都提供，使用 ATR 版本。

    Args:
        df:               K线数据（需要 high, low, close 列）
        raw_pos:          原始持仓信号 [0, 1]，已经 shift(1) 过
        stop_loss_atr:    止损距离（ATR 倍数），None = 不用
        take_profit_atr:  止盈距离（ATR 倍数），None = 不用
        trailing_stop_atr: 移动止损（ATR 倍数），None = 不用
        stop_loss_pct:    止损百分比（备选），None = 不用
        take_profit_pct:  止盈百分比（备选），None = 不用
        atr_period:       ATR 计算周期
        cooldown_bars:    出场后冷却期（bar 数）

    Returns:
        调整后的持仓序列 [0, 1]
    """
    close = df["close"].values
    high = df["high"].values
    n = len(df)

    # 预计算 ATR
    atr = calculate_atr(df, atr_period).values

    raw = raw_pos.values.copy()
    result = np.zeros(n, dtype=float)

    # 状态变量
    in_position = False
    entry_price = 0.0
    sl_price = 0.0
    tp_price = 0.0
    highest_since_entry = 0.0
    cooldown_remaining = 0

    for i in range(n):
        current_close = close[i]
        current_high = high[i]
        current_atr = atr[i] if not np.isnan(atr[i]) else 0.0

        # ── 冷却期检查 ──
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            result[i] = 0.0
            continue

        # ── 持仓中：检查 SL / TP / Trailing ──
        if in_position:
            triggered_exit = False

            # 更新移动止损的最高价
            if current_high > highest_since_entry:
                highest_since_entry = current_high
                # 更新移动止损价
                if trailing_stop_atr is not None and current_atr > 0:
                    sl_price = max(sl_price, highest_since_entry - trailing_stop_atr * current_atr)

            # 检查止损
            if sl_price > 0 and current_close <= sl_price:
                triggered_exit = True

            # 检查止盈
            if tp_price > 0 and current_close >= tp_price:
                triggered_exit = True

            # 策略本身发出平仓信号
            if raw[i] <= 0:
                triggered_exit = True

            if triggered_exit:
                result[i] = 0.0
                in_position = False
                cooldown_remaining = cooldown_bars
            else:
                result[i] = 1.0

        # ── 空仓中：检查是否该进场 ──
        else:
            if raw[i] > 0:
                in_position = True
                entry_price = current_close
                highest_since_entry = current_high

                # 设定止损价
                if stop_loss_atr is not None and current_atr > 0:
                    sl_price = entry_price - stop_loss_atr * current_atr
                elif stop_loss_pct is not None:
                    sl_price = entry_price * (1 - stop_loss_pct)
                else:
                    sl_price = 0.0

                # 设定止盈价
                if take_profit_atr is not None and current_atr > 0:
                    tp_price = entry_price + take_profit_atr * current_atr
                elif take_profit_pct is not None:
                    tp_price = entry_price * (1 + take_profit_pct)
                else:
                    tp_price = 0.0

                result[i] = 1.0
            else:
                result[i] = 0.0

    return pd.Series(result, index=raw_pos.index)

