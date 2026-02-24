"""
Mean Reversion — Liquidity Sweep Strategy
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

高勝率均值回歸策略，設計為 TSMOM (趨勢) 的互補策略。

核心邏輯 (Liquidity Sweep):
    機構經常在流動性集中區（前高/前低）觸發 stop-hunt，
    造成假突破後快速反轉。本策略利用此現象：
    1. 偵測價格掃過 24h 高低點（liquidity sweep）
    2. 確認反轉訊號（收盤無力 + RSI 極端）
    3. 以固定 TP/SL/Time-stop 短線交易

Timeframe: 15m (主要), 可用於其他週期
Key Level: 滾動 24h (96 bars @15m) 的 High/Low
Signal:
    Short: High[i] > 24h_Max[i-1] (創新高) + Close[i] < Close[i-1] (收盤反轉) + RSI > 70
    Long:  Low[i] < 24h_Min[i-1] (創新低) + Close[i] > Close[i-1] (收盤反轉) + RSI < 30
Exit:
    TP = 1.5% | SL = 1.0% | Time-stop = 12 bars (3h @15m)

Anti-Lookahead 保證:
    - 使用 auto_delay=False + 手動管理 signal_delay
    - entry 條件在 bar[i] 的 close 時判斷 (使用 close/high/low/RSI)
    - 實際入場在 bar[i + signal_delay] 的 open
    - SL/TP 使用 intrabar high/low 檢查 + 觸發價執行
    - 所有指標 (rolling_high/low, RSI) 使用 shift(1) 或只用過去資料

Note:
    direction clip 需手動處理 (auto_delay=False)。
    策略返回 positions + exec_prices (掛在 attrs)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import StrategyContext
from . import register_strategy
from ..indicators import calculate_rsi


# ──────────────────────────────────────────────
#  常數
# ──────────────────────────────────────────────

# 各 interval 對應 24h 的 bar 數
_BARS_24H = {
    "1m": 1440, "5m": 288, "15m": 96,
    "30m": 48, "1h": 24, "4h": 6, "1d": 1,
}


# ──────────────────────────────────────────────
#  主策略
# ──────────────────────────────────────────────

@register_strategy("mean_revert_liquidity", auto_delay=False)
def generate_mean_revert_liquidity(
    df: pd.DataFrame,
    ctx: StrategyContext,
    params: dict,
) -> pd.Series:
    """
    Liquidity Sweep 均值回歸策略

    此策略使用 auto_delay=False，自行管理：
        - signal_delay (entry 延遲 1 bar)
        - direction clip
        - SL/TP 以觸發價執行 (exec_prices)

    params:
        key_level_bars:   滾動高低點回看期 (bar 數), 預設 96 (=24h @15m)
        rsi_period:       RSI 計算週期, 預設 14
        rsi_overbought:   RSI 超買門檻 (做空條件), 預設 70
        rsi_oversold:     RSI 超賣門檻 (做多條件), 預設 30
        tp_pct:           止盈百分比, 預設 0.015 (1.5%)
        sl_pct:           止損百分比, 預設 0.010 (1.0%)
        max_hold_bars:    最大持倉 bar 數 (time-stop), 預設 12 (=3h @15m)
        cooldown_bars:    出場後冷卻 bar 數, 預設 4 (=1h @15m)

    Returns:
        pd.Series: 持倉信號 [-1, 0, 1]
                   attrs["exit_exec_prices"] 含 SL/TP 觸發價 (NaN=用 open)
    """
    # ── 參數解析 ──
    interval = getattr(ctx, "interval", "15m")
    bars_24h = _BARS_24H.get(interval, 96)
    key_level_bars = int(params.get("key_level_bars", bars_24h))
    rsi_period = int(params.get("rsi_period", 14))
    rsi_overbought = float(params.get("rsi_overbought", 70))
    rsi_oversold = float(params.get("rsi_oversold", 30))
    tp_pct = float(params.get("tp_pct", 0.015))
    sl_pct = float(params.get("sl_pct", 0.010))
    max_hold_bars = int(params.get("max_hold_bars", 12))
    cooldown_bars = int(params.get("cooldown_bars", 4))
    signal_delay = getattr(ctx, "signal_delay", 1)

    # ── 指標計算 ──
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    open_ = df["open"].values

    # 24h 滾動高低點 — shift(1) 確保只用過去資料 (不含當前 bar)
    rolling_high_24h = df["high"].rolling(key_level_bars).max().shift(1).values
    rolling_low_24h = df["low"].rolling(key_level_bars).min().shift(1).values

    # RSI — 不需要 shift，因為 RSI[i] 用的是 close[0..i]，
    # 且 entry signal 在 bar[i] close 時判斷，entry 執行在 bar[i+1] open
    rsi = calculate_rsi(df["close"], rsi_period).values

    # ── 逐 bar 狀態機 ──
    n = len(df)
    pos = np.zeros(n, dtype=np.float64)
    exec_prices = np.full(n, np.nan, dtype=np.float64)

    holding = 0         # 0: flat, 1: long, -1: short
    entry_price = 0.0   # 入場價 (next bar open)
    entry_bar = -999    # 入場 bar index
    cooldown_until = -1  # 冷卻期結束 bar

    warmup = max(key_level_bars + 2, rsi_period + 2)

    for i in range(warmup, n):
        # ─── 有持倉：檢查 SL/TP/Time-stop ───
        if holding != 0:
            bars_held = i - entry_bar

            if holding == 1:  # Long
                # TP: high 觸及 entry * (1 + tp_pct)
                tp_level = entry_price * (1.0 + tp_pct)
                if high[i] >= tp_level:
                    pos[i] = 0.0
                    exec_prices[i] = tp_level  # 以 TP 價執行
                    holding = 0
                    cooldown_until = i + cooldown_bars
                    continue

                # SL: low 觸及 entry * (1 - sl_pct)
                sl_level = entry_price * (1.0 - sl_pct)
                if low[i] <= sl_level:
                    pos[i] = 0.0
                    exec_prices[i] = sl_level  # 以 SL 價執行
                    holding = 0
                    cooldown_until = i + cooldown_bars
                    continue

                # Time-stop: 持倉超過 max_hold_bars
                if max_hold_bars > 0 and bars_held >= max_hold_bars:
                    pos[i] = 0.0
                    # time-stop 用 open 價出場 (NaN → 框架用 open)
                    holding = 0
                    cooldown_until = i + cooldown_bars
                    continue

                # 繼續持有
                pos[i] = 1.0

            elif holding == -1:  # Short
                # TP: low 觸及 entry * (1 - tp_pct)
                tp_level = entry_price * (1.0 - tp_pct)
                if low[i] <= tp_level:
                    pos[i] = 0.0
                    exec_prices[i] = tp_level
                    holding = 0
                    cooldown_until = i + cooldown_bars
                    continue

                # SL: high 觸及 entry * (1 + sl_pct)
                sl_level = entry_price * (1.0 + sl_pct)
                if high[i] >= sl_level:
                    pos[i] = 0.0
                    exec_prices[i] = sl_level
                    holding = 0
                    cooldown_until = i + cooldown_bars
                    continue

                # Time-stop
                if max_hold_bars > 0 and bars_held >= max_hold_bars:
                    pos[i] = 0.0
                    holding = 0
                    cooldown_until = i + cooldown_bars
                    continue

                pos[i] = -1.0

        # ─── 無持倉：檢查入場訊號 ───
        else:
            # 冷卻期中不入場
            if i < cooldown_until:
                pos[i] = 0.0
                continue

            # 需確保 i + signal_delay < n (有下一根 bar 可執行)
            if i + signal_delay >= n:
                pos[i] = 0.0
                continue

            # --- Short Signal ---
            # High[i] 突破前 24h 最高 (liquidity sweep upward)
            # Close[i] < Close[i-1] (收盤無力 → 假突破)
            # RSI > overbought (超買確認)
            if (not np.isnan(rolling_high_24h[i])
                    and high[i] > rolling_high_24h[i]
                    and close[i] < close[i - 1]
                    and rsi[i] > rsi_overbought):
                if ctx.can_short:
                    # 在 bar[i + signal_delay] 的 open 入場
                    exec_bar = i + signal_delay
                    pos[exec_bar] = -1.0
                    entry_price = open_[exec_bar]
                    entry_bar = exec_bar
                    holding = -1
                    continue

            # --- Long Signal ---
            # Low[i] 跌破前 24h 最低 (liquidity sweep downward)
            # Close[i] > Close[i-1] (收盤有力 → 假跌破)
            # RSI < oversold (超賣確認)
            if (not np.isnan(rolling_low_24h[i])
                    and low[i] < rolling_low_24h[i]
                    and close[i] > close[i - 1]
                    and rsi[i] < rsi_oversold):
                if ctx.can_long:
                    exec_bar = i + signal_delay
                    pos[exec_bar] = 1.0
                    entry_price = open_[exec_bar]
                    entry_bar = exec_bar
                    holding = 1
                    continue

            pos[i] = 0.0

    # ── 構建結果 ──
    result = pd.Series(pos, index=df.index, name="position")

    # 掛載 exec_prices 供 run_backtest 使用
    exec_prices_series = pd.Series(exec_prices, index=df.index)
    result.attrs["exit_exec_prices"] = exec_prices_series

    return result
