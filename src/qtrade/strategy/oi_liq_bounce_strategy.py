"""
OI Liquidation Bounce 策略 (v4.2)

Alpha 來源：
    OI 急降 + 價格急跌 → 清算瀑布結束 → 做多反彈
    Volume spike + 價格急跌 → 恐慌拋售結束 → 做多反彈（v4 新增 OR gate）

學術 / 實踐背景：
    - 大量清算（多頭爆倉）會造成 OI 急劇下降伴隨價格下跌
    - 清算瀑布結束後，賣壓消失 → 價格傾向反彈
    - 與趨勢動量（TSMOM）正交（相關性 ≈ 0.01），適合組合配置

信號定義：
    入場（OR gate，兩條路徑任一觸發即可）：
      Path A (OI Liquidation):
          OI_change z-score < oi_z_threshold AND Price_change z-score < price_z_threshold
      Path B (Volume Spike，v4 新增):
          Volume z-score > vol_spike_threshold AND Price_change z-score < price_z_threshold
      共同條件：AND HTF 趨勢過濾通過 AND (可選) FR spike filter
    出場：三種模式（exit_mode 參數控制）
          - fixed_hold: 固定持有 hold_bars 後出場（v1 行為）
          - atr_sl:     ATR 止損觸發出場，max_hold_bars 為時間止損上限
          - hybrid:     ATR 止損 + 固定持有時間，兩者先到先出（預設）
    方向：Long-only

風控元件：
    1. HTF Trend Filter — 三種模式（htf_regime_mode）
       - "ema":     Daily EMA50（價格在 EMA 之上 = 牛市）
       - "adx":     Daily ADX > 閾值 AND +DI > -DI（強上升趨勢）
       - "ema_adx": EMA50 AND ADX（最嚴格）
    2. FR Spike Filter — Funding Rate 急降確認清算事件（v3 新增）
    3. Volume Spike Path — 成交量急增作為替代入場條件（v4 新增）
    4. Vol Scaling — 根據波動率倒數縮放倉位
    5. Cooldown — 出場後 N bars 禁止再入場
    6. ATR Stop-Loss — 入場後設置 entry - ATR * stop_loss_atr 止損（v2 新增）

Anti-lookahead：
    - 所有指標使用 bar[i] 的數據（不含策略內部 .shift(1)，由框架統一延遲）
    - HTF trend 使用 resampled Daily 數據，ffill 到 1h（不含未來）
    - ATR SL 檢查使用 bar[i] 的 low/high，出場信號在 bar[i]
      → signal_delay 使得實際執行在 open[i+1]（保守：延遲出場）
    - entry_price 使用 close[i]（≈ open[i+1]），確保 SL/TP 基於接近實際入場價
    - signal_delay 由 @register_strategy 框架自動處理（trade_on=next_open → shift(1)）
    - v4.2 修正：移除策略內部所有 .shift(1)，消除 double-delay 問題

Note:
    使用 auto_delay=True，框架自動處理 signal_delay 和 direction clip。

Changelog:
    v1 (2026-02-24): Initial implementation — fixed_hold exit
    v2 (2026-02-24): Add ATR SL exit, symbol_overrides per exit mode
    v3 (2026-02-24): Add FR spike filter + ADX regime filter for BTC 2025 improvement
    v4 (2026-02-24): Add volume spike OR gate entry — BTC trades 31→55+, portfolio trade density ↑
    v4.2 (2026-02-25): Fix double-delay + SL/TP entry_price=close[i] + enable slippage_model
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from .base import StrategyContext
from . import register_strategy
from ..indicators import calculate_ema, calculate_atr, calculate_adx

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
#  核心指標計算
# ══════════════════════════════════════════════════════════════

def _compute_oi_change_zscore(
    oi_series: pd.Series,
    change_lookback: int = 24,
    z_window: int = 720,
) -> pd.Series:
    """
    計算 OI 變化率的滾動 z-score

    步驟：
        1. OI 24h 變化率 = (OI - OI_lag) / OI_lag
        2. z-score = (change_rate - rolling_mean) / rolling_std

    Args:
        oi_series: OI 數值序列（已對齊到 1h bar index）
        change_lookback: 計算變化率的回看期（bars），預設 24 = 24h
        z_window: 滾動 z-score 窗口（bars），預設 720 = 30 天

    Returns:
        OI 變化率 z-score 序列
    """
    if oi_series is None or oi_series.empty:
        return pd.Series(dtype=float)

    # 1. OI 變化率
    oi_lagged = oi_series.shift(change_lookback)
    change_rate = (oi_series - oi_lagged) / oi_lagged.replace(0, np.nan)

    # 2. 滾動 z-score
    min_periods = max(z_window // 4, 30)
    rolling_mean = change_rate.rolling(z_window, min_periods=min_periods).mean()
    rolling_std = change_rate.rolling(z_window, min_periods=min_periods).std()
    z = (change_rate - rolling_mean) / rolling_std.replace(0, np.nan)

    return z.fillna(0.0).clip(-5.0, 5.0)


def _compute_price_change_zscore(
    close: pd.Series,
    change_lookback: int = 8,
    z_window: int = 720,
) -> pd.Series:
    """
    計算價格變化率的滾動 z-score

    步驟：
        1. Price 8h 變化率 = (close - close_lag) / close_lag
        2. z-score = (change_rate - rolling_mean) / rolling_std

    Args:
        close: 收盤價序列
        change_lookback: 計算變化率的回看期（bars），預設 8 = 8h
        z_window: 滾動 z-score 窗口（bars），預設 720 = 30 天

    Returns:
        價格變化率 z-score 序列
    """
    # 1. 價格變化率
    close_lagged = close.shift(change_lookback)
    change_rate = (close - close_lagged) / close_lagged.replace(0, np.nan)

    # 2. 滾動 z-score
    min_periods = max(z_window // 4, 30)
    rolling_mean = change_rate.rolling(z_window, min_periods=min_periods).mean()
    rolling_std = change_rate.rolling(z_window, min_periods=min_periods).std()
    z = (change_rate - rolling_mean) / rolling_std.replace(0, np.nan)

    return z.fillna(0.0).clip(-5.0, 5.0)


def _compute_daily_ema_trend(
    df: pd.DataFrame,
    ema_period: int = 50,
) -> pd.Series:
    """
    計算 Daily EMA 趨勢過濾信號

    邏輯：
        1. 將 1h 數據重採樣為 Daily
        2. 計算 Daily EMA(ema_period)
        3. close > EMA → 上升趨勢（允許做多），反之不允許

    Args:
        df: 1h K 線 DataFrame（需含 close）
        ema_period: EMA 週期（在 Daily 上），預設 50

    Returns:
        1h 粒度的趨勢信號：1.0 = 上升趨勢，0.0 = 非上升趨勢
    """
    # 重採樣到 Daily
    daily_df = df.resample("1D").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()

    if len(daily_df) < ema_period + 5:
        # 數據不足 → 不過濾（全部允許）
        return pd.Series(1.0, index=df.index)

    # Daily EMA
    daily_ema = calculate_ema(daily_df["close"], ema_period)

    # close > EMA → 上升趨勢
    daily_trend = pd.Series(0.0, index=daily_df.index)
    daily_trend[daily_df["close"] > daily_ema] = 1.0

    # 映射回 1h（forward fill，不用未來數據）
    trend_1h = daily_trend.reindex(df.index, method="ffill").fillna(0.0)

    return trend_1h


def _compute_fr_change_zscore(
    fr_series: pd.Series,
    change_lookback: int = 3,
    z_window: int = 720,
) -> pd.Series:
    """
    計算 Funding Rate 變化的滾動 z-score

    邏輯：
        FR 急降 = 多頭被清算 → FR 從正值降至負值或低值
        FR 每 8h 更新一次，forward-fill 到 1h。
        change_lookback=3 → 以 8h 為單位 = 24h 變化。

    Steps:
        1. FR 差分（使用 8h 間隔的 diff）
        2. 滾動 z-score = (diff - rolling_mean) / rolling_std

    Args:
        fr_series: Funding rate 序列（已 forward-fill 至 1h bar index）
        change_lookback: FR 差分回看期（以 8h 為單位），預設 3 = 24h
        z_window: 滾動窗口（1h bars），預設 720 = 30 天

    Returns:
        FR 變化 z-score 序列（負值 = FR 急降）
    """
    if fr_series is None or fr_series.empty:
        return pd.Series(dtype=float)

    # FR 每 8h 更新，forward-fill 到 1h。diff(8*lookback) 取 lookback 個 8h 週期
    fr_diff = fr_series.diff(8 * change_lookback)

    # 滾動 z-score
    min_periods = max(z_window // 4, 30)
    rolling_mean = fr_diff.rolling(z_window, min_periods=min_periods).mean()
    rolling_std = fr_diff.rolling(z_window, min_periods=min_periods).std()
    z = (fr_diff - rolling_mean) / rolling_std.replace(0, np.nan)

    return z.fillna(0.0).clip(-5.0, 5.0)


def _compute_daily_adx_regime(
    df: pd.DataFrame,
    adx_period: int = 14,
    adx_threshold: float = 25.0,
) -> pd.Series:
    """
    計算 Daily ADX 趨勢 regime 過濾信號

    邏輯：
        1. 將 1h 數據重採樣為 Daily
        2. 計算 Daily ADX、+DI、-DI
        3. ADX > threshold AND +DI > -DI → 強上升趨勢（允許做多）

    比 EMA50 更精確：EMA50 只看價位，ADX 判斷趨勢強度和方向。
    適合避免 2025 年 BTC「價格在 EMA50 上方但無實質上升動能」的陷阱。

    Args:
        df: 1h K 線 DataFrame
        adx_period: ADX 計算週期（Daily），預設 14
        adx_threshold: ADX 閾值（> 此值為趨勢市場），預設 25

    Returns:
        1h 粒度的趨勢信號：1.0 = 強上升趨勢，0.0 = 否
    """
    # 重採樣到 Daily
    daily_df = df.resample("1D").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()

    if len(daily_df) < adx_period + 10:
        return pd.Series(1.0, index=df.index)

    # Daily ADX
    adx_data = calculate_adx(daily_df, adx_period)
    adx = adx_data["ADX"]
    plus_di = adx_data["+DI"]
    minus_di = adx_data["-DI"]

    # 強上升趨勢: ADX > threshold AND +DI > -DI
    daily_regime = pd.Series(0.0, index=daily_df.index)
    daily_regime[(adx > adx_threshold) & (plus_di > minus_di)] = 1.0

    # 映射回 1h（forward fill）
    regime_1h = daily_regime.reindex(df.index, method="ffill").fillna(0.0)

    return regime_1h


def _compute_volume_zscore(
    volume: pd.Series,
    vol_sum_window: int = 24,
    z_window: int = 720,
) -> pd.Series:
    """
    計算成交量的滾動 z-score（v4 新增）

    邏輯：
        Volume spike 常見於恐慌拋售 / 清算瀑布事件。
        與 OI 下降只有 30% 重疊 → 作為 OR gate 的替代入場路徑可顯著增加交易頻率。

    步驟：
        1. 計算 vol_sum_window 窗口的累計成交量
        2. z-score = (cum_vol - rolling_mean) / rolling_std

    Args:
        volume: 成交量序列
        vol_sum_window: 累計窗口（bars），預設 24 = 24h
        z_window: 滾動 z-score 窗口（bars），預設 720 = 30 天

    Returns:
        成交量 z-score 序列（正值 = 成交量高於平均）
    """
    cum_vol = volume.rolling(vol_sum_window, min_periods=1).sum()

    min_periods = max(z_window // 4, 30)
    rolling_mean = cum_vol.rolling(z_window, min_periods=min_periods).mean()
    rolling_std = cum_vol.rolling(z_window, min_periods=min_periods).std()
    z = (cum_vol - rolling_mean) / rolling_std.replace(0, np.nan)

    return z.fillna(0.0).clip(-5.0, 5.0)


def _vol_scale(
    close: pd.Series,
    vol_target: float = 0.15,
    vol_lookback: int = 168,
) -> pd.Series:
    """
    波動率目標縮放（倉位大小根據波動率反比調整）

    Args:
        close: 收盤價序列
        vol_target: 年化波動率目標（預設 0.15 = 15%）
        vol_lookback: 波動率計算回看期（bars）

    Returns:
        縮放因子序列，clip 到 [0.2, 1.0]
    """
    returns = close.pct_change()
    # 年化波動率（1h bar, 8760 = 365 * 24）
    vol = returns.rolling(vol_lookback, min_periods=max(vol_lookback // 4, 10)).std() * np.sqrt(8760)
    vol = vol.replace(0, np.nan).ffill().fillna(vol_target)

    scale = (vol_target / vol).clip(0.2, 1.0)
    return scale


# ══════════════════════════════════════════════════════════════
#  策略：OI Liquidation Bounce
# ══════════════════════════════════════════════════════════════

@register_strategy("oi_liq_bounce")
def generate_oi_liq_bounce(
    df: pd.DataFrame, ctx: StrategyContext, params: dict,
) -> pd.Series:
    """
    OI Liquidation Bounce 策略 (v4.2)

    信號邏輯（OR gate，兩條入場路徑任一觸發即可）：
        Path A (OI Liquidation):
            OI_change z-score < oi_z_threshold AND Price_change z-score < price_z_threshold
        Path B (Volume Spike，v4 新增):
            Volume z-score > vol_spike_threshold AND Price_change z-score < price_z_threshold
        共同條件：AND HTF 趨勢過濾通過 AND (可選) FR spike filter

        Exit:  三種模式（exit_mode 控制）：
               - fixed_hold: 持有 hold_bars 後出場
               - atr_sl: ATR 止損 + max_hold_bars 時間止損
               - hybrid: ATR 止損 + hold_bars 時間止損（先到先出）
        Direction: Long-only

    params:
        # Core signal — Path A (OI)
        oi_change_lookback:     OI 變化率回看期（bars），預設 24
        price_change_lookback:  價格變化率回看期（bars），預設 8
        z_window:               z-score 滾動窗口（bars），預設 720
        oi_z_threshold:         OI z-score 入場門檻（負值），預設 -1.5
        price_z_threshold:      Price z-score 入場門檻（負值），預設 -1.0

        # Core signal — Path B (Volume Spike, v4 新增)
        vol_spike_enabled:      是否啟用 volume spike 入場路徑，預設 False
        vol_spike_threshold:    Volume z-score 門檻（正值），預設 2.0
        vol_spike_lookback:     Volume 累計窗口（bars），預設 24

        # Exit — 模式選擇
        exit_mode:              出場模式，"fixed_hold" | "atr_sl" | "hybrid"，預設 "hybrid"
        hold_bars:              固定持有期（bars），用於 fixed_hold/hybrid，預設 24
        max_hold_bars:          最大持有期（bars），用於 atr_sl 模式的時間止損，預設 72
        stop_loss_atr:          止損 ATR 倍數（多頭: entry - ATR * mult），預設 2.5
        take_profit_atr:        止盈 ATR 倍數（多頭: entry + ATR * mult），None=不用

        # HTF trend filter — v3: 三種模式
        htf_filter_enabled:     是否啟用 HTF 過濾，預設 True
        htf_regime_mode:        過濾模式 "ema" | "adx" | "ema_adx"，預設 "ema"
        htf_ema_period:         Daily EMA 週期（ema/ema_adx 模式），預設 50
        htf_adx_period:         Daily ADX 週期（adx/ema_adx 模式），預設 14
        htf_adx_threshold:      Daily ADX 閾值（adx/ema_adx 模式），預設 25

        # Funding Rate spike filter — v3 新增
        fr_spike_enabled:       是否啟用 FR spike 過濾，預設 False
        fr_z_threshold:         FR z-score 入場門檻（負值），預設 -1.5
        fr_change_lookback:     FR 差分回看期（8h 為單位），預設 3 (=24h)

        # Vol scaling
        vol_scale_enabled:      是否啟用波動率縮放，預設 True
        vol_target:             年化波動率目標，預設 0.15
        vol_lookback:           波動率回看期（bars），預設 168

        # Cooldown
        cooldown_bars:          出場後冷卻期（bars），預設 12

        # Data injection (by runner/script)
        _oi_series:             預注入的 OI Series
        _fr_series:             預注入的 FR Series（v3 新增）
        _data_dir:              數據根目錄（用於自動載入 OI/FR）
    """
    close = df["close"]
    low = df["low"].values
    high = df["high"].values
    open_ = df["open"].values
    n = len(df)

    # ── 參數解析 ──
    oi_change_lookback = int(params.get("oi_change_lookback", 24))
    price_change_lookback = int(params.get("price_change_lookback", 8))
    z_window = int(params.get("z_window", 720))
    oi_z_threshold = float(params.get("oi_z_threshold", -1.5))
    price_z_threshold = float(params.get("price_z_threshold", -1.0))

    # Exit params
    exit_mode = str(params.get("exit_mode", "hybrid"))  # fixed_hold | atr_sl | hybrid
    hold_bars = int(params.get("hold_bars", 24))
    max_hold_bars = int(params.get("max_hold_bars", 72))
    stop_loss_atr_mult = params.get("stop_loss_atr", 2.5)
    stop_loss_atr_mult = float(stop_loss_atr_mult) if stop_loss_atr_mult is not None else None
    take_profit_atr_mult = params.get("take_profit_atr", None)
    take_profit_atr_mult = float(take_profit_atr_mult) if take_profit_atr_mult is not None else None

    # HTF filter params — v3: regime mode
    htf_filter_enabled = bool(params.get("htf_filter_enabled", True))
    htf_regime_mode = str(params.get("htf_regime_mode", "ema"))  # ema | adx | ema_adx
    htf_ema_period = int(params.get("htf_ema_period", 50))
    htf_adx_period = int(params.get("htf_adx_period", 14))
    htf_adx_threshold = float(params.get("htf_adx_threshold", 25.0))

    # Volume spike params — v4 新增
    vol_spike_enabled = bool(params.get("vol_spike_enabled", False))
    vol_spike_threshold = float(params.get("vol_spike_threshold", 2.0))
    vol_spike_lookback = int(params.get("vol_spike_lookback", 24))

    # FR spike filter params — v3 新增
    fr_spike_enabled = bool(params.get("fr_spike_enabled", False))
    fr_z_threshold = float(params.get("fr_z_threshold", -1.5))
    fr_change_lookback = int(params.get("fr_change_lookback", 3))

    vol_scale_enabled = bool(params.get("vol_scale_enabled", True))
    vol_target = float(params.get("vol_target", 0.15))
    vol_lookback = int(params.get("vol_lookback", 168))

    cooldown_bars = int(params.get("cooldown_bars", 12))

    # ── 1. OI 信號 ──
    oi_raw = params.get("_oi_series")
    if oi_raw is not None and isinstance(oi_raw, pd.Series) and not oi_raw.empty:
        # OI 可能需要對齊到 df index
        if not oi_raw.index.equals(df.index):
            # Timezone alignment
            if df.index.tz is None and oi_raw.index.tz is not None:
                oi_raw = oi_raw.copy()
                oi_raw.index = oi_raw.index.tz_localize(None)
            elif df.index.tz is not None and oi_raw.index.tz is None:
                oi_raw = oi_raw.copy()
                oi_raw.index = oi_raw.index.tz_localize(df.index.tz)
            oi_aligned = oi_raw.reindex(df.index, method="ffill", limit=2)
        else:
            oi_aligned = oi_raw
    else:
        # 嘗試自動載入 OI
        data_dir = params.get("_data_dir")
        if data_dir is not None:
            try:
                from ..data.open_interest import get_oi_path, load_open_interest, align_oi_to_klines
                for prov in ["merged", "binance_vision", "coinglass", "binance"]:
                    oi_path = get_oi_path(data_dir, ctx.symbol, prov)
                    oi_df = load_open_interest(oi_path)
                    if oi_df is not None and not oi_df.empty:
                        oi_aligned = align_oi_to_klines(oi_df, df.index, max_ffill_bars=2)
                        break
                else:
                    logger.warning(f"{ctx.symbol}: 無法自動載入 OI 數據")
                    return pd.Series(0.0, index=df.index)
            except Exception as e:
                logger.warning(f"{ctx.symbol}: OI 自動載入失敗: {e}")
                return pd.Series(0.0, index=df.index)
        else:
            # 無 OI 數據 → 無法產生信號，返回全 0
            return pd.Series(0.0, index=df.index)

    oi_coverage = (~oi_aligned.isna()).mean()
    if oi_coverage < 0.3:
        # OI 覆蓋率太低，返回全 0
        return pd.Series(0.0, index=df.index)

    # OI 變化率 z-score（框架 signal_delay 統一處理延遲，不再 .shift(1)）
    oi_z = _compute_oi_change_zscore(oi_aligned, oi_change_lookback, z_window)

    # ── 2. Price 信號 ──
    price_z = _compute_price_change_zscore(close, price_change_lookback, z_window)

    # ── 3. HTF 趨勢過濾（v3: 支援三種模式）──
    if htf_filter_enabled:
        if htf_regime_mode == "ema":
            htf_trend = _compute_daily_ema_trend(df, htf_ema_period)
        elif htf_regime_mode == "adx":
            htf_trend = _compute_daily_adx_regime(df, htf_adx_period, htf_adx_threshold)
        elif htf_regime_mode == "ema_adx":
            ema_trend = _compute_daily_ema_trend(df, htf_ema_period)
            adx_trend = _compute_daily_adx_regime(df, htf_adx_period, htf_adx_threshold)
            htf_trend = (ema_trend * adx_trend)  # 兩者都要 = 1.0
        else:
            logger.warning(f"Unknown htf_regime_mode: {htf_regime_mode}, fallback to ema")
            htf_trend = _compute_daily_ema_trend(df, htf_ema_period)
    else:
        htf_trend = pd.Series(1.0, index=df.index)

    # ── 4. FR Spike Filter（v3 新增）──
    if fr_spike_enabled:
        fr_raw = params.get("_fr_series")
        if fr_raw is None:
            # 嘗試自動載入 FR
            data_dir = params.get("_data_dir")
            if data_dir is not None:
                try:
                    from ..data.funding_rate import (
                        get_funding_rate_path, load_funding_rates, align_funding_to_klines,
                    )
                    fr_path = get_funding_rate_path(data_dir, ctx.symbol)
                    fr_df = load_funding_rates(fr_path)
                    if fr_df is not None:
                        fr_raw = align_funding_to_klines(fr_df, df.index)
                except Exception as e:
                    logger.warning(f"{ctx.symbol}: FR 自動載入失敗: {e}")

        if fr_raw is not None and isinstance(fr_raw, pd.Series) and not fr_raw.empty:
            # 對齊到 df index
            if not fr_raw.index.equals(df.index):
                if df.index.tz is None and fr_raw.index.tz is not None:
                    fr_raw = fr_raw.copy()
                    fr_raw.index = fr_raw.index.tz_localize(None)
                elif df.index.tz is not None and fr_raw.index.tz is None:
                    fr_raw = fr_raw.copy()
                    fr_raw.index = fr_raw.index.tz_localize(df.index.tz)
                fr_aligned = fr_raw.reindex(df.index, method="ffill")
            else:
                fr_aligned = fr_raw

            fr_z = _compute_fr_change_zscore(fr_aligned, fr_change_lookback, z_window)
        else:
            logger.warning(f"{ctx.symbol}: FR spike enabled but no FR data → disabled")
            fr_z = pd.Series(0.0, index=df.index)  # 不過濾
            fr_spike_enabled = False
    else:
        fr_z = pd.Series(0.0, index=df.index)

    # ── 5. Volume Spike 信號（v4 新增 OR gate Path B）──
    if vol_spike_enabled:
        vol_z = _compute_volume_zscore(df["volume"], vol_spike_lookback, z_window)
    else:
        vol_z = pd.Series(0.0, index=df.index)

    # ── 6. Vol Scaling ──
    if vol_scale_enabled:
        vol_scale = _vol_scale(close, vol_target, vol_lookback)
    else:
        vol_scale = pd.Series(1.0, index=df.index)

    # ── 7. Pre-compute ATR for SL/TP ──
    use_atr_exit = exit_mode in ("atr_sl", "hybrid")
    if use_atr_exit:
        atr_vals = calculate_atr(df, period=14).values
    else:
        atr_vals = None

    # ── 8. State Machine: 信號生成（v4.2: 不再使用 _lagged，由框架統一延遲）──
    oi_z_vals = oi_z.fillna(0.0).values
    price_z_vals = price_z.fillna(0.0).values
    htf_vals = htf_trend.fillna(0.0).values
    fr_z_vals = fr_z.fillna(0.0).values
    vol_z_vals = vol_z.fillna(0.0).values  # v4: volume spike z-score
    vol_vals = vol_scale.fillna(0.5).values
    close_vals = close.values  # v4.2: entry_price = close[i] ≈ open[i+1]

    pos = np.zeros(n, dtype=float)
    state = 0           # 0 = flat, 1 = long
    hold_count = 0      # bars since entry
    cooldown_remaining = 0  # cooldown counter
    entry_price = 0.0   # 入場價（用於 ATR SL/TP）
    sl_price = 0.0      # 止損價
    tp_price = 0.0      # 止盈價

    warmup = max(z_window, oi_change_lookback, price_change_lookback) + 50

    for i in range(warmup, n):
        # ── Cooldown 檢查 ──
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            pos[i] = 0.0
            continue

        if state == 1:
            # ── 持倉中 ──
            hold_count += 1
            should_exit = False

            if exit_mode == "fixed_hold":
                # v1 行為：固定持有 hold_bars 後出場
                if hold_count >= hold_bars:
                    should_exit = True

            elif exit_mode == "atr_sl":
                # 純 ATR 止損模式
                # SL 檢查：用 low 價（保守，因為 signal_delay 使執行在 open[i+1]）
                if sl_price > 0 and low[i] <= sl_price:
                    should_exit = True
                # TP 檢查：用 high 價
                if tp_price > 0 and high[i] >= tp_price:
                    should_exit = True
                # 時間止損：max_hold_bars
                if hold_count >= max_hold_bars:
                    should_exit = True

            elif exit_mode == "hybrid":
                # ATR 止損 + 固定持有時間，先到先出
                if sl_price > 0 and low[i] <= sl_price:
                    should_exit = True
                if tp_price > 0 and high[i] >= tp_price:
                    should_exit = True
                if hold_count >= hold_bars:
                    should_exit = True

            if should_exit:
                pos[i] = 0.0
                state = 0
                hold_count = 0
                cooldown_remaining = cooldown_bars
            else:
                # 繼續持有
                pos[i] = vol_vals[i]  # vol scaling 持續作用
        else:
            # ── 空倉 → 檢查入場條件（v4: OR gate）──
            oi_trigger = oi_z_vals[i] < oi_z_threshold
            price_trigger = price_z_vals[i] < price_z_threshold
            vol_spike_trigger = vol_spike_enabled and (vol_z_vals[i] > vol_spike_threshold)
            htf_ok = htf_vals[i] > 0.5  # 上升趨勢
            fr_ok = (not fr_spike_enabled) or (fr_z_vals[i] < fr_z_threshold)

            # OR gate: Path A (OI + price) OR Path B (vol spike + price)
            # 共同條件: htf_ok AND fr_ok
            path_a = oi_trigger and price_trigger    # OI liquidation path
            path_b = vol_spike_trigger and price_trigger  # Volume spike path
            entry_signal = (path_a or path_b) and htf_ok and fr_ok

            if entry_signal:
                # 入場做多
                pos[i] = vol_vals[i]  # vol scaling 決定倉位大小
                state = 1
                hold_count = 0
                entry_price = close_vals[i]  # v4.2: close[i] ≈ open[i+1]，配合 signal_delay

                # 計算 ATR SL/TP（僅 atr_sl 或 hybrid 模式）
                if use_atr_exit:
                    current_atr = atr_vals[i] if not np.isnan(atr_vals[i]) else 0.0
                    if stop_loss_atr_mult is not None and current_atr > 0:
                        sl_price = entry_price - stop_loss_atr_mult * current_atr
                    else:
                        sl_price = 0.0
                    if take_profit_atr_mult is not None and current_atr > 0:
                        tp_price = entry_price + take_profit_atr_mult * current_atr
                    else:
                        tp_price = 0.0
            else:
                pos[i] = 0.0

    return pd.Series(pos, index=df.index)
