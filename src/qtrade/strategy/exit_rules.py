"""
通用出場規則（止損 / 止盈 / 移動止損 / 冷卻期）

設計理念：
    策略只負責產生「進出場信號」（raw position），
    出場規則作為後處理器，疊加在任何策略上。

止損/止盈檢查邏輯（v2.0 更新）：
    - 止損：用 K 棒「最低價」檢查（模擬硬止損/預掛單行為）
    - 止盈：用 K 棒「最高價」檢查
    - 這比用「收盤價」更真實，與實盤預掛單行為一致

用法：
    from qtrade.strategy.exit_rules import apply_exit_rules

    raw_pos = my_indicator_logic(df, params)
    pos = apply_exit_rules(
        df, raw_pos,
        stop_loss_atr=2.0,      # 止損 = 入場價 - 2×ATR
        take_profit_atr=3.0,    # 止盈 = 入場價 + 3×ATR
        trailing_stop_atr=2.5,  # 移動止損 = 最高價 - 2.5×ATR
        cooldown_bars=6,        # 止損後 6 根 bar 不進場
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
    對原始持倉信號疊加出場規則（支援做空）

    優先級：ATR-based > 固定百分比
    如果 stop_loss_atr 和 stop_loss_pct 都提供，使用 ATR 版本。

    Args:
        df:               K線數據（需要 high, low, close 列）
        raw_pos:          原始持倉信號 [-1, 1]，已經 shift(1) 過
        stop_loss_atr:    止損距離（ATR 倍數），None = 不用
        take_profit_atr:  止盈距離（ATR 倍數），None = 不用
        trailing_stop_atr: 移動止損（ATR 倍數），None = 不用
        stop_loss_pct:    止損百分比（備選），None = 不用
        take_profit_pct:  止盈百分比（備選），None = 不用
        atr_period:       ATR 計算週期
        cooldown_bars:    出場後冷卻期（bar 數）

    Returns:
        調整後的持倉序列 [-1, 1]
    """
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(df)

    # 預計算 ATR
    atr = calculate_atr(df, atr_period).values

    raw = raw_pos.values.copy()
    result = np.zeros(n, dtype=float)

    # 狀態變數
    # position_state: 0 = flat, 1 = long, -1 = short
    position_state = 0
    entry_price = 0.0
    sl_price = 0.0
    tp_price = 0.0
    extreme_since_entry = 0.0  # 多倉追蹤最高價，空倉追蹤最低價
    cooldown_remaining = 0

    for i in range(n):
        current_close = close[i]
        current_high = high[i]
        current_low = low[i]
        current_atr = atr[i] if not np.isnan(atr[i]) else 0.0

        # ── 冷卻期檢查 ──
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            result[i] = 0.0
            continue

        # ── 持有多倉 ──
        if position_state == 1:
            triggered_exit = False

            # 更新最高價（用於移動止損）
            if current_high > extreme_since_entry:
                extreme_since_entry = current_high
                if trailing_stop_atr is not None and current_atr > 0:
                    sl_price = max(sl_price, extreme_since_entry - trailing_stop_atr * current_atr)

            # 檢查止損（用 low 價）
            if sl_price > 0 and current_low <= sl_price:
                triggered_exit = True

            # 檢查止盈（用 high 價）
            if tp_price > 0 and current_high >= tp_price:
                triggered_exit = True

            # 策略發出平倉或反向信號
            if raw[i] <= 0:
                triggered_exit = True

            if triggered_exit:
                # 平倉後是否反手做空
                if raw[i] < 0:
                    # 反手做空
                    position_state = -1
                    entry_price = current_close
                    extreme_since_entry = current_low
                    if stop_loss_atr is not None and current_atr > 0:
                        sl_price = entry_price + stop_loss_atr * current_atr
                    elif stop_loss_pct is not None:
                        sl_price = entry_price * (1 + stop_loss_pct)
                    else:
                        sl_price = 0.0
                    if take_profit_atr is not None and current_atr > 0:
                        tp_price = entry_price - take_profit_atr * current_atr
                    elif take_profit_pct is not None:
                        tp_price = entry_price * (1 - take_profit_pct)
                    else:
                        tp_price = 0.0
                    result[i] = -1.0
                else:
                    # 純平倉
                    position_state = 0
                    result[i] = 0.0
                    cooldown_remaining = cooldown_bars
            else:
                result[i] = 1.0

        # ── 持有空倉 ──
        elif position_state == -1:
            triggered_exit = False

            # 更新最低價（用於移動止損）
            if current_low < extreme_since_entry:
                extreme_since_entry = current_low
                if trailing_stop_atr is not None and current_atr > 0:
                    sl_price = min(sl_price, extreme_since_entry + trailing_stop_atr * current_atr)

            # 檢查止損（用 high 價，空倉止損在上方）
            if sl_price > 0 and current_high >= sl_price:
                triggered_exit = True

            # 檢查止盈（用 low 價，空倉止盈在下方）
            if tp_price > 0 and current_low <= tp_price:
                triggered_exit = True

            # 策略發出平倉或反向信號
            if raw[i] >= 0:
                triggered_exit = True

            if triggered_exit:
                # 平倉後是否反手做多
                if raw[i] > 0:
                    # 反手做多
                    position_state = 1
                    entry_price = current_close
                    extreme_since_entry = current_high
                    if stop_loss_atr is not None and current_atr > 0:
                        sl_price = entry_price - stop_loss_atr * current_atr
                    elif stop_loss_pct is not None:
                        sl_price = entry_price * (1 - stop_loss_pct)
                    else:
                        sl_price = 0.0
                    if take_profit_atr is not None and current_atr > 0:
                        tp_price = entry_price + take_profit_atr * current_atr
                    elif take_profit_pct is not None:
                        tp_price = entry_price * (1 + take_profit_pct)
                    else:
                        tp_price = 0.0
                    result[i] = 1.0
                else:
                    # 純平倉
                    position_state = 0
                    result[i] = 0.0
                    cooldown_remaining = cooldown_bars
            else:
                result[i] = -1.0

        # ── 空倉中：檢查是否該進場 ──
        else:
            if raw[i] > 0:
                # 開多
                position_state = 1
                entry_price = current_close
                extreme_since_entry = current_high

                if stop_loss_atr is not None and current_atr > 0:
                    sl_price = entry_price - stop_loss_atr * current_atr
                elif stop_loss_pct is not None:
                    sl_price = entry_price * (1 - stop_loss_pct)
                else:
                    sl_price = 0.0

                if take_profit_atr is not None and current_atr > 0:
                    tp_price = entry_price + take_profit_atr * current_atr
                elif take_profit_pct is not None:
                    tp_price = entry_price * (1 + take_profit_pct)
                else:
                    tp_price = 0.0

                result[i] = 1.0
            elif raw[i] < 0:
                # 開空
                position_state = -1
                entry_price = current_close
                extreme_since_entry = current_low

                if stop_loss_atr is not None and current_atr > 0:
                    sl_price = entry_price + stop_loss_atr * current_atr
                elif stop_loss_pct is not None:
                    sl_price = entry_price * (1 + stop_loss_pct)
                else:
                    sl_price = 0.0

                if take_profit_atr is not None and current_atr > 0:
                    tp_price = entry_price - take_profit_atr * current_atr
                elif take_profit_pct is not None:
                    tp_price = entry_price * (1 - take_profit_pct)
                else:
                    tp_price = 0.0

                result[i] = -1.0
            else:
                result[i] = 0.0

    return pd.Series(result, index=raw_pos.index)
