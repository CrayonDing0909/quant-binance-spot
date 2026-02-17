"""
通用出場規則（止損 / 止盈 / 移動止損 / 冷卻期）

設計理念：
    策略只負責產生「進出場信號」（raw position），
    出場規則作為後處理器，疊加在任何策略上。

止損/止盈檢查邏輯（v2.0 更新）：
    - 止損：用 K 棒「最低價」檢查（模擬硬止損/預掛單行為）
    - 止盈：用 K 棒「最高價」檢查
    - 這比用「收盤價」更真實，與實盤預掛單行為一致

入場價格（v2.1 更新）：
    - 入場價使用 open（開盤價），與 vectorbt from_orders(price=open_) 一致
    - 這確保 SL/TP 計算基於實際入場價，不存在收盤價 look-ahead

自適應止損（v3.0 更新）：
    - 可選傳入 adaptive_sl_er 參數（ER 序列）
    - ER 低（震盪）→ 放寬 SL → 扛過噪聲
    - ER 高（趨勢）→ 收緊 SL → 保護利潤
    - SL 乘數 = sl_atr_base × (er_sl_max - (er_sl_max - er_sl_min) × ER)
    - 例：ER=0.1 → SL=3.0x ATR，ER=0.8 → SL=1.5x ATR

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


def compute_adaptive_sl_multiplier(
    er_value: float,
    sl_atr_base: float,
    er_sl_min: float = 1.5,
    er_sl_max: float = 3.0,
) -> float:
    """
    根據 Efficiency Ratio 計算自適應 SL 乘數

    震盪市（ER 低）→ SL 放寬（更大的 ATR 倍數）→ 扛過噪聲
    趨勢市（ER 高）→ SL 收緊（更小的 ATR 倍數）→ 保護利潤

    映射：ER ∈ [0, 1] → SL ∈ [er_sl_max, er_sl_min]
        ER = 0.0 → sl_mult = er_sl_max (3.0)  最寬
        ER = 1.0 → sl_mult = er_sl_min (1.5)  最窄

    Args:
        er_value:     當前 ER 值 [0, 1]
        sl_atr_base:  原始 SL 基準（用於 fallback）
        er_sl_min:    ER=1 時的最窄 SL 乘數
        er_sl_max:    ER=0 時的最寬 SL 乘數

    Returns:
        SL ATR 乘數
    """
    if np.isnan(er_value):
        return sl_atr_base
    er_clamped = max(0.0, min(1.0, er_value))
    # 線性映射：ER=0 → er_sl_max，ER=1 → er_sl_min
    return er_sl_max - (er_sl_max - er_sl_min) * er_clamped


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
    adaptive_sl_er: pd.Series | np.ndarray | None = None,
    er_sl_min: float = 1.5,
    er_sl_max: float = 3.0,
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
        adaptive_sl_er:   Efficiency Ratio 序列 [0,1]（可選）
                          傳入時啟用自適應止損
        er_sl_min:        ER=1（趨勢）時的最窄 SL 乘數，預設 1.5
        er_sl_max:        ER=0（震盪）時的最寬 SL 乘數，預設 3.0

    Returns:
        調整後的持倉序列 [-1, 1]
    """
    open_ = df["open"].values
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(df)

    # 預計算 ATR
    atr = calculate_atr(df, atr_period).values

    # Adaptive SL 的 ER 序列
    use_adaptive_sl = adaptive_sl_er is not None and stop_loss_atr is not None
    if use_adaptive_sl:
        er_vals = np.asarray(adaptive_sl_er, dtype=float)
    else:
        er_vals = None

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
        current_open = open_[i]
        current_close = close[i]
        current_high = high[i]
        current_low = low[i]
        current_atr = atr[i] if not np.isnan(atr[i]) else 0.0

        # 計算本 bar 的 SL 乘數（自適應或固定）
        if use_adaptive_sl:
            _sl_mult = compute_adaptive_sl_multiplier(
                er_vals[i], stop_loss_atr, er_sl_min, er_sl_max
            )
        else:
            _sl_mult = stop_loss_atr

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
                    # 反手做空（入場價 = open，與 VBT 一致）
                    position_state = -1
                    entry_price = current_open
                    extreme_since_entry = current_low
                    if _sl_mult is not None and current_atr > 0:
                        sl_price = entry_price + _sl_mult * current_atr
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
                    # 反手做多（入場價 = open，與 VBT 一致）
                    position_state = 1
                    entry_price = current_open
                    extreme_since_entry = current_high
                    if _sl_mult is not None and current_atr > 0:
                        sl_price = entry_price - _sl_mult * current_atr
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
                # 開多（入場價 = open，與 VBT 一致）
                position_state = 1
                entry_price = current_open
                extreme_since_entry = current_high

                if _sl_mult is not None and current_atr > 0:
                    sl_price = entry_price - _sl_mult * current_atr
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
                # 開空（入場價 = open，與 VBT 一致）
                position_state = -1
                entry_price = current_open
                extreme_since_entry = current_low

                if _sl_mult is not None and current_atr > 0:
                    sl_price = entry_price + _sl_mult * current_atr
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
