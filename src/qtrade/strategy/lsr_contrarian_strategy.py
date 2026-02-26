"""
LSR Contrarian 策略 — 散戶多空比極端值逆向交易

Alpha 來源：
    散戶 Long/Short Ratio（LSR）達到極端高/低時，
    群眾情緒過度一致，逆向交易有統計優勢。

學術 / 實踐背景：
    - LSR 反映散戶帳戶的多空比例，散戶普遍是反向指標
    - 初步研究顯示 LSR pctrank 168h vs 24h forward return 平均 IC = -0.0245（穩定為負）
    - 該逆向效果在盤整期和趨勢期**幾乎相同**（avg IC: -0.024 vs -0.024）
    - 收益分布為正偏態（WR=40%, R:R=1.97, Skew=+2.58）→ 類趨勢特徵
    - 散戶 LSR 的逆向 IC（0.026）遠高於大戶 LSR（0.002）→ 散戶情緒是關鍵

信號定義（v2 — Swing Mode）：
    入場：
        LSR percentile rank(window) > entry_pctile → 做空（多頭擁擠）
        LSR percentile rank(window) < (1 - entry_pctile) → 做多（空頭擁擠）
    出場（三擇一，先觸發者優先）：
        1. TP: 持空時，LSR pctrank < tp_pctile → 止盈（擺盪到對面）
              持多時，LSR pctrank > (1 - tp_pctile) → 止盈
        2. SL: 價格觸及 entry_price ± sl_atr * ATR → 硬止損
        3. 時間止損: 持倉超過 max_hold_bars → 強制出場
    方向：Both（多空皆可）

v1 vs v2 差異：
    - v1: exit at 50th percentile（只吃半段擺盪）
    - v2: exit at opposite extreme（吃完整擺盪）+ ATR SL 保護 + 時間止損
    - v2 移除 EMA 平滑（state machine 直接輸出）

參數：
    lsr_type: str = "lsr"           使用的 LSR 指標（散戶）
    lsr_window: int = 168           percentile rank 滾動窗口（bars）
    entry_pctile: float = 0.85      入場門檻（top/bottom 15%）
    tp_pctile: float = 0.25         止盈門檻（對面 25th percentile）
    sl_atr_mult: float = 2.5        ATR 止損乘數
    sl_atr_lookback: int = 14       ATR 計算窗口
    max_hold_bars: int = 120        最大持倉時間（120h = 5天）
    vol_target: float = 0.15        波動率目標（年化）
    vol_lookback: int = 168         波動率計算窗口

Anti-lookahead：
    - LSR 數據以 forward-fill 對齊到 K 線（嚴格因果）
    - SL 檢測使用 bar[i] 的 high/low（當前 bar 已結束）
    - 所有指標使用 bar[i] 的數據（不含策略內部 .shift）
    - signal_delay 由 @register_strategy 框架自動處理（trade_on=next_open → shift(1)）

Research Evidence：
    - Notebook: notebooks/research/20260226_lsr_extremes_contrarian.ipynb
    - 8/8 symbols gross Sharpe 為正（avg Sharpe = 9.33）
    - avg PnL/trade = +0.528%（通過 MR 鐵律）
    - IC stability: 51-67% 時間 IC 為負（逆向有效）
    - Cost estimate: ~130 trades/yr, net SR after costs = 3.1 ~ 32.2

Changelog:
    v1 (2026-02-26): Initial implementation from research notebook
    v2 (2026-02-27): Swing mode — TP at opposite extreme, ATR SL, time stop
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .base import StrategyContext
from . import register_strategy

logger = logging.getLogger(__name__)


def _compute_lsr_pctrank(
    lsr: pd.Series,
    window: int = 168,
) -> pd.Series:
    """
    計算 LSR 的滾動 percentile rank

    Args:
        lsr: Long/Short Ratio 序列
        window: 滾動窗口（bars）

    Returns:
        [0, 1] 之間的 percentile rank 序列
    """
    min_periods = max(window // 2, 24)
    pctrank = lsr.rolling(window, min_periods=min_periods).apply(
        lambda x: sp_stats.percentileofscore(x, x.iloc[-1]) / 100.0,
        raw=False,
    )
    return pctrank


def _vol_scale(
    close: pd.Series,
    vol_target: float = 0.15,
    vol_lookback: int = 168,
) -> pd.Series:
    """
    波動率目標縮放（倉位根據波動率反比調整）

    Args:
        close: 收盤價序列
        vol_target: 年化波動率目標
        vol_lookback: 波動率計算窗口

    Returns:
        縮放因子序列，clip 到 [0.2, 2.0]
    """
    returns = close.pct_change()
    vol = returns.rolling(vol_lookback, min_periods=max(vol_lookback // 4, 10)).std() * np.sqrt(8760)
    vol = vol.replace(0, np.nan).ffill().fillna(vol_target)
    scale = (vol_target / vol).clip(0.2, 2.0)
    return scale


def _compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    lookback: int = 14,
) -> pd.Series:
    """計算 ATR（Average True Range）"""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(lookback, min_periods=max(lookback // 2, 5)).mean()


@register_strategy("lsr_contrarian")
def generate_lsr_contrarian(
    df: pd.DataFrame,
    ctx: StrategyContext,
    params: dict,
) -> pd.Series:
    """
    LSR 逆向策略 v2 — Swing Mode

    信號邏輯：
        入場：
            LSR percentile rank > entry_pctile → 做空（多頭擁擠）
            LSR percentile rank < (1 - entry_pctile) → 做多（空頭擁擠）
        出場（三擇一，先觸發者優先）：
            1. TP: LSR 擺盪到對面極端 (tp_pctile)
            2. SL: 價格觸及 entry ± sl_atr * ATR（硬止損）
            3. 時間: 超過 max_hold_bars 強制出場

    params:
        lsr_type:         LSR 類型，"lsr"（散戶全帳戶，預設）
        lsr_window:       percentile rank 窗口（bars），預設 168
        entry_pctile:     入場門檻 [0,1]，預設 0.85
        tp_pctile:        止盈門檻 [0,1]，預設 0.25（對面 25th pctile）
        sl_atr_mult:      ATR 止損乘數，預設 2.5
        sl_atr_lookback:  ATR 計算窗口，預設 14
        max_hold_bars:    最大持倉時間（bars），預設 120（5天）
        vol_scale_enabled: 是否啟用波動率縮放，預設 True
        vol_target:       年化波動率目標，預設 0.15
        vol_lookback:     波動率回看期（bars），預設 168
        _data_dir:        數據根目錄（runner/script 注入）

    v1 compat params（向後相容，不再使用）:
        exit_pctile:      已棄用，改用 tp_pctile
        composite_ema:    已棄用，v2 移除 EMA 平滑
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    n = len(df)

    # ── 參數解析 ──
    lsr_type = str(params.get("lsr_type", "lsr"))
    lsr_window = int(params.get("lsr_window", 168))
    entry_pctile = float(params.get("entry_pctile", 0.85))
    tp_pctile = float(params.get("tp_pctile", 0.25))
    sl_atr_mult = float(params.get("sl_atr_mult", 2.5))
    sl_atr_lookback = int(params.get("sl_atr_lookback", 14))
    max_hold_bars = int(params.get("max_hold_bars", 120))
    vol_scale_enabled = bool(params.get("vol_scale_enabled", True))
    vol_target = float(params.get("vol_target", 0.15))
    vol_lookback = int(params.get("vol_lookback", 168))

    entry_hi = entry_pctile           # e.g., 0.85 → top 15%
    entry_lo = 1.0 - entry_pctile     # e.g., 0.15 → bottom 15%
    tp_for_short = tp_pctile          # 持空時 TP: pctrank < 0.25
    tp_for_long = 1.0 - tp_pctile    # 持多時 TP: pctrank > 0.75

    # ── 1. 取得 LSR 數據 ──
    # 優先從 ctx.derivatives_data 取（live / meta_blend 模式）
    lsr_series = ctx.get_derivative(lsr_type)

    if lsr_series is None:
        # 嘗試從 _data_dir 自動載入
        data_dir = params.get("_data_dir")
        if data_dir is not None:
            try:
                from ..data.long_short_ratio import load_lsr, align_lsr_to_klines
                from pathlib import Path

                data_dir_path = Path(data_dir)
                deriv_dir = data_dir_path / "binance" / "futures" / "derivatives"
                lsr_raw = load_lsr(ctx.symbol, lsr_type, data_dir=deriv_dir)
                if lsr_raw is not None and not lsr_raw.empty:
                    lsr_series = align_lsr_to_klines(lsr_raw, df.index, max_ffill_bars=2)
                    logger.info(f"  LSR Contrarian [{ctx.symbol}]: 自動載入 {lsr_type} 成功 ({len(lsr_raw)} rows)")
            except Exception as e:
                logger.warning(f"  LSR Contrarian [{ctx.symbol}]: LSR 自動載入失敗: {e}")

    if lsr_series is None:
        logger.warning(f"  ⚠️ LSR Contrarian [{ctx.symbol}]: no {lsr_type} data, returning flat")
        return pd.Series(0.0, index=df.index)

    # 確保 index 對齊
    lsr_aligned = lsr_series.reindex(df.index).ffill()

    # LSR 覆蓋率檢查
    lsr_coverage = (~lsr_aligned.isna()).mean()
    if lsr_coverage < 0.3:
        logger.warning(f"  ⚠️ LSR Contrarian [{ctx.symbol}]: LSR coverage {lsr_coverage:.1%} < 30%, returning flat")
        return pd.Series(0.0, index=df.index)

    # ── 2. 計算 LSR percentile rank ──
    lsr_pctrank = _compute_lsr_pctrank(lsr_aligned, lsr_window)

    # ── 3. 計算 ATR（用於止損）──
    atr = _compute_atr(high, low, close, sl_atr_lookback)

    # ── 4. Vol scaling ──
    if vol_scale_enabled:
        vol_scale = _vol_scale(close, vol_target, vol_lookback)
    else:
        vol_scale = pd.Series(1.0, index=df.index)

    # ── 5. State machine: 信號生成（v2 Swing Mode）──
    pctrank_vals = lsr_pctrank.fillna(0.5).values
    vol_vals = vol_scale.fillna(1.0).values
    atr_vals = atr.fillna(0.0).values
    close_vals = close.values
    high_vals = high.values
    low_vals = low.values
    open_vals = open_.values

    pos = np.zeros(n, dtype=float)
    state = 0  # 0 = flat, 1 = long, -1 = short
    entry_price = 0.0
    entry_bar = 0
    warmup = max(lsr_window + 10, sl_atr_lookback + 5)

    # 統計
    n_tp = 0
    n_sl = 0
    n_time = 0

    for i in range(warmup, n):
        pr = pctrank_vals[i]
        cur_atr = atr_vals[i]

        if state == 0:
            # ── 空倉 → 檢查入場 ──
            if pr > entry_hi:
                # 多頭擁擠 → 做空
                pos[i] = -1.0 * vol_vals[i]
                state = -1
                entry_price = open_vals[min(i + 1, n - 1)]  # next bar open (signal_delay 會再 shift)
                entry_bar = i
            elif pr < entry_lo:
                # 空頭擁擠 → 做多
                pos[i] = 1.0 * vol_vals[i]
                state = 1
                entry_price = open_vals[min(i + 1, n - 1)]
                entry_bar = i
            else:
                pos[i] = 0.0

        elif state == 1:
            # ── 持多中 → 檢查出場 ──
            hold_time = i - entry_bar

            # SL: 價格跌破 entry - sl_atr * ATR
            sl_price = entry_price - sl_atr_mult * cur_atr
            if low_vals[i] <= sl_price and cur_atr > 0:
                pos[i] = 0.0
                state = 0
                n_sl += 1
            # TP: LSR 擺盪到對面高位
            elif pr > tp_for_long:
                pos[i] = 0.0
                state = 0
                n_tp += 1
            # 時間止損
            elif hold_time >= max_hold_bars:
                pos[i] = 0.0
                state = 0
                n_time += 1
            else:
                # 繼續持有
                pos[i] = 1.0 * vol_vals[i]

        elif state == -1:
            # ── 持空中 → 檢查出場 ──
            hold_time = i - entry_bar

            # SL: 價格突破 entry + sl_atr * ATR
            sl_price = entry_price + sl_atr_mult * cur_atr
            if high_vals[i] >= sl_price and cur_atr > 0:
                pos[i] = 0.0
                state = 0
                n_sl += 1
            # TP: LSR 擺盪到對面低位
            elif pr < tp_for_short:
                pos[i] = 0.0
                state = 0
                n_tp += 1
            # 時間止損
            elif hold_time >= max_hold_bars:
                pos[i] = 0.0
                state = 0
                n_time += 1
            else:
                # 繼續持有
                pos[i] = -1.0 * vol_vals[i]

    pos_series = pd.Series(pos, index=df.index)
    pos_series = pos_series.clip(-1.0, 1.0).fillna(0.0)

    # ── 日誌 ──
    n_long = (pos_series > 0.1).sum()
    n_short = (pos_series < -0.1).sum()
    n_flat = n - n_long - n_short
    total_exits = n_tp + n_sl + n_time
    logger.info(
        f"  LSR Contrarian v2 [{ctx.symbol}]: "
        f"long={n_long}/{n}, short={n_short}/{n}, flat={n_flat}/{n}, "
        f"exits: TP={n_tp}, SL={n_sl}, Time={n_time} (total={total_exits}), "
        f"LSR coverage={lsr_coverage:.1%}"
    )

    return pos_series
