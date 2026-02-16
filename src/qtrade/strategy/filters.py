"""
通用信號過濾器

設計理念：
    過濾器作為後處理器，疊加在策略的原始信號上。
    如果當前 bar 不滿足過濾條件，不允許新開倉（但不強制平倉）。

用法：
    from qtrade.strategy.filters import trend_filter, volume_filter, htf_trend_filter, volatility_filter

    raw_pos = my_indicator_logic(df, params)
    pos = trend_filter(df, raw_pos, min_adx=25)
    pos = volume_filter(df, pos, min_volume_ratio=1.2)
    pos = volatility_filter(df, pos, min_atr_ratio=0.005)  # 波動率過濾
    pos = htf_trend_filter(df, pos, htf_interval="4h")  # 高級時間框架趨勢
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from ..indicators.adx import calculate_adx
from ..indicators.atr import calculate_atr
from ..indicators.volume import calculate_obv
from ..indicators.moving_average import calculate_ema


def trend_filter(
    df: pd.DataFrame,
    raw_pos: pd.Series,
    min_adx: float = 25.0,
    adx_period: int = 14,
    require_uptrend: bool = True,
) -> pd.Series:
    """
    ADX 趨勢過濾器（支援做空）

    規則：
    - ADX < min_adx → 無趨勢，禁止新開倉
    - 做多：require_uptrend=True 時，要求 +DI > -DI（上升趨勢）
    - 做空：require_uptrend=True 時，要求 -DI > +DI（下降趨勢）
    - 已有持倉不受影響（由出場規則或策略信號決定平倉）

    Args:
        df:              K線數據
        raw_pos:         原始持倉信號 [-1, 1]
        min_adx:         最小 ADX 值，低於此值不開新倉
        adx_period:      ADX 週期
        require_uptrend: 是否要求趨勢方向配合（做多配上升趨勢，做空配下降趨勢）

    Returns:
        過濾後的持倉序列
    """
    adx_data = calculate_adx(df, adx_period)
    adx = adx_data["ADX"].values
    plus_di = adx_data["+DI"].values
    minus_di = adx_data["-DI"].values

    raw = raw_pos.values.copy()
    result = np.zeros(len(raw), dtype=float)
    position_state = 0  # 0 = flat, 1 = long, -1 = short

    for i in range(len(raw)):
        has_trend = not np.isnan(adx[i]) and adx[i] >= min_adx
        is_uptrend = plus_di[i] > minus_di[i]
        is_downtrend = minus_di[i] > plus_di[i]

        if raw[i] > 0:  # 做多信號
            if position_state == 1:
                # 已有多倉 → 保持
                result[i] = raw[i]
            elif has_trend and (is_uptrend or not require_uptrend):
                # 允許新開多倉
                result[i] = raw[i]
                position_state = 1
            else:
                # 不允許新開倉
                result[i] = 0.0
        elif raw[i] < 0:  # 做空信號
            if position_state == -1:
                # 已有空倉 → 保持
                result[i] = raw[i]
            elif has_trend and (is_downtrend or not require_uptrend):
                # 允許新開空倉（需要下降趨勢）
                result[i] = raw[i]
                position_state = -1
            else:
                # 不允許新開倉
                result[i] = 0.0
        else:
            result[i] = 0.0
            position_state = 0

    return pd.Series(result, index=raw_pos.index)


def volume_filter(
    df: pd.DataFrame,
    raw_pos: pd.Series,
    min_volume_ratio: float = 1.2,
    volume_period: int = 20,
) -> pd.Series:
    """
    成交量過濾器（支援做空）

    規則：當前成交量 > 均量 × min_volume_ratio 時才允許新開倉。
    避免在低流動性時段進場。

    Args:
        df:                K線數據
        raw_pos:           原始持倉信號 [-1, 1]
        min_volume_ratio:  最小成交量倍數
        volume_period:     均量計算週期

    Returns:
        過濾後的持倉序列
    """
    volume = df["volume"].values
    vol_ma = df["volume"].rolling(volume_period).mean().values

    raw = raw_pos.values.copy()
    result = np.zeros(len(raw), dtype=float)
    position_state = 0  # 0 = flat, 1 = long, -1 = short

    for i in range(len(raw)):
        vol_ok = not np.isnan(vol_ma[i]) and volume[i] >= vol_ma[i] * min_volume_ratio

        if raw[i] > 0:  # 做多信號
            if position_state == 1:
                result[i] = raw[i]
            elif vol_ok:
                result[i] = raw[i]
                position_state = 1
            else:
                result[i] = 0.0
        elif raw[i] < 0:  # 做空信號
            if position_state == -1:
                result[i] = raw[i]
            elif vol_ok:
                result[i] = raw[i]
                position_state = -1
            else:
                result[i] = 0.0
        else:
            result[i] = 0.0
            position_state = 0

    return pd.Series(result, index=raw_pos.index)


def volatility_filter(
    df: pd.DataFrame,
    raw_pos: pd.Series,
    min_atr_ratio: float = 0.005,
    atr_period: int = 14,
    lookback: int = 50,
    use_percentile: bool = False,
    min_percentile: float = 25.0,
) -> pd.Series:
    """
    波動率過濾器（ATR Ratio）（支援做空）

    專業量化交易的標準做法：當波動率不足時暫停交易，避免：
    - 假突破信號增多
    - 止損被噪音觸發
    - 獲利空間不足以覆蓋手續費

    兩種模式：
    1. 絕對閾值模式 (use_percentile=False)：
       - ATR / Close >= min_atr_ratio 時允許交易
       - min_atr_ratio=0.005 表示波動率至少 0.5%

    2. 百分位模式 (use_percentile=True)：
       - ATR 在歷史 lookback 期的百分位 >= min_percentile 時允許交易
       - min_percentile=25 表示波動率要高於歷史 25% 的水平

    Args:
        df:              K線數據
        raw_pos:         原始持倉信號 [-1, 1]
        min_atr_ratio:   最小 ATR/Price 比率（絕對閾值模式）
        atr_period:      ATR 計算週期
        lookback:        百分位計算的回看週期
        use_percentile:  是否使用百分位模式
        min_percentile:  最小百分位閾值（百分位模式）

    Returns:
        過濾後的持倉序列

    Example:
        # 絕對閾值：波動率至少 0.8%
        pos = volatility_filter(df, raw_pos, min_atr_ratio=0.008)

        # 百分位：波動率要高於歷史 30% 水平
        pos = volatility_filter(df, raw_pos, use_percentile=True, min_percentile=30)
    """
    atr = calculate_atr(df, atr_period)
    close = df["close"]

    # 計算 ATR Ratio
    atr_ratio = (atr / close).values

    if use_percentile:
        # 百分位模式：計算滾動百分位
        atr_series = pd.Series(atr_ratio, index=df.index)
        rolling_pct = atr_series.rolling(lookback).apply(
            lambda x: (x.iloc[-1] > x).sum() / len(x) * 100 if len(x) > 0 else 50,
            raw=False,
        ).values
        vol_ok_arr = rolling_pct >= min_percentile
    else:
        # 絕對閾值模式
        vol_ok_arr = atr_ratio >= min_atr_ratio

    raw = raw_pos.values.copy()
    result = np.zeros(len(raw), dtype=float)
    position_state = 0  # 0 = flat, 1 = long, -1 = short

    for i in range(len(raw)):
        vol_ok = not np.isnan(atr_ratio[i]) and vol_ok_arr[i]

        if raw[i] > 0:  # 做多信號
            if position_state == 1:
                # 已有多倉 → 保持（不因波動率下降而強制平倉）
                result[i] = raw[i]
            elif vol_ok:
                # 波動率足夠 → 允許新開多倉
                result[i] = raw[i]
                position_state = 1
            else:
                # 波動率不足 → 禁止新開倉
                result[i] = 0.0
        elif raw[i] < 0:  # 做空信號
            if position_state == -1:
                # 已有空倉 → 保持
                result[i] = raw[i]
            elif vol_ok:
                # 波動率足夠 → 允許新開空倉
                result[i] = raw[i]
                position_state = -1
            else:
                # 波動率不足 → 禁止新開倉
                result[i] = 0.0
        else:
            result[i] = 0.0
            position_state = 0

    return pd.Series(result, index=raw_pos.index)


# ── 高級時間框架重採樣映射 ──────────────────────────────────
_RESAMPLE_MAP = {
    # 當前週期 → 可用的高級週期及其 pandas resample rule
    "1m":  {"5m": "5min",  "15m": "15min", "1h": "1h",  "4h": "4h"},
    "5m":  {"15m": "15min", "30m": "30min", "1h": "1h",  "4h": "4h"},
    "15m": {"1h": "1h",     "4h": "4h",     "1d": "1D"},
    "30m": {"2h": "2h",     "4h": "4h",     "1d": "1D"},
    "1h":  {"4h": "4h",     "6h": "6h",     "1d": "1D"},
    "2h":  {"8h": "8h",     "1d": "1D"},
    "4h":  {"1d": "1D"},
}


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    將低級時間框架 K 線重採樣為高級時間框架

    Args:
        df:   低級 K 線數據（必須有 DatetimeIndex）
        rule: pandas resample rule, e.g. "4h", "1D"

    Returns:
        高級 K 線 DataFrame（OHLCV）
    """
    return df.resample(rule).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()


def htf_trend_filter(
    df: pd.DataFrame,
    raw_pos: pd.Series,
    htf_interval: str = "4h",
    ema_fast: int = 20,
    ema_slow: int = 50,
    current_interval: str = "1h",
) -> pd.Series:
    """
    多時間框架趨勢過濾器（支援做空）

    邏輯：
        1. 將 1h 數據重採樣為 4h
        2. 在 4h 上計算 EMA20 和 EMA50
        3. EMA20 > EMA50 = 上升趨勢 → 允許做多
        4. EMA20 < EMA50 = 下降趨勢 → 允許做空（Futures）
        5. 已有持倉不強制平倉

    為什麼用 EMA 交叉而不是 ADX？
        - ADX 只判斷趨勢強度，不判斷方向
        - EMA 交叉同時判斷趨勢方向和強度（間距越大趨勢越強）
        - 4h EMA20/50 對應日線 EMA5/12.5，是經典趨勢判斷

    Args:
        df:               低級 K 線數據（e.g. 1h）
        raw_pos:          原始持倉信號 [-1, 1]
        htf_interval:     高級時間框架，e.g. "4h"
        ema_fast:         快速 EMA 週期（在高級 TF 上），預設 20
        ema_slow:         慢速 EMA 週期（在高級 TF 上），預設 50
        current_interval: 當前 K 線週期，e.g. "1h"

    Returns:
        過濾後的持倉序列
    """
    # 獲取重採樣規則
    resample_options = _RESAMPLE_MAP.get(current_interval, {})
    resample_rule = resample_options.get(htf_interval)

    if resample_rule is None:
        # 無法重採樣，原樣返回
        return raw_pos

    # 重採樣到高級時間框架
    htf_df = _resample_ohlcv(df, resample_rule)
    if len(htf_df) < ema_slow + 5:
        # 數據不足以計算 EMA，原樣返回
        return raw_pos

    # 在高級 TF 上計算 EMA
    ema_f = calculate_ema(htf_df["close"], ema_fast)
    ema_s = calculate_ema(htf_df["close"], ema_slow)

    # 判斷趨勢方向
    # +1 = 上升趨勢，-1 = 下降趨勢
    htf_trend = pd.Series(0.0, index=htf_df.index)
    htf_trend[ema_f > ema_s] = 1.0
    htf_trend[ema_f < ema_s] = -1.0

    # 將高級 TF 的信號映射回低級 TF
    htf_trend_ltf = htf_trend.reindex(df.index, method="ffill").fillna(0.0)

    # 應用過濾邏輯
    raw = raw_pos.values.copy()
    trend = htf_trend_ltf.values
    result = np.zeros(len(raw), dtype=float)
    position_state = 0  # 0 = flat, 1 = long, -1 = short

    for i in range(len(raw)):
        is_uptrend = trend[i] > 0.5
        is_downtrend = trend[i] < -0.5

        if raw[i] > 0:  # 做多信號
            if position_state == 1:
                # 已有多倉 → 保持
                result[i] = raw[i]
            elif is_uptrend:
                # 高級 TF 上升趨勢 → 允許新開多倉
                result[i] = raw[i]
                position_state = 1
            else:
                # 高級 TF 非上升趨勢 → 禁止新開多倉
                result[i] = 0.0
        elif raw[i] < 0:  # 做空信號
            if position_state == -1:
                # 已有空倉 → 保持
                result[i] = raw[i]
            elif is_downtrend:
                # 高級 TF 下降趨勢 → 允許新開空倉
                result[i] = raw[i]
                position_state = -1
            else:
                # 高級 TF 非下降趨勢 → 禁止新開空倉
                result[i] = 0.0
        else:
            result[i] = 0.0
            position_state = 0

    return pd.Series(result, index=raw_pos.index)


def htf_soft_trend_filter(
    df: pd.DataFrame,
    raw_pos: pd.Series,
    htf_interval: str = "4h",
    ema_fast: int = 20,
    ema_slow: int = 50,
    current_interval: str = "1h",
    align_weight: float = 1.0,
    counter_weight: float = 0.5,
    neutral_weight: float = 0.75,
) -> pd.Series:
    """
    多時間框架「軟」趨勢過濾器

    與硬過濾器 (htf_trend_filter) 的區別：
      - 硬過濾器：逆趨勢交易完全禁止 → 交易次數大幅減少
      - 軟過濾器：逆趨勢交易以縮小倉位進行 → 保留交易機會，降低風險

    權重邏輯：
      - 順趨勢（e.g. 上升趨勢做多）→ align_weight (1.0)，全倉
      - 逆趨勢（e.g. 上升趨勢做空）→ counter_weight (0.5)，半倉
      - 無趨勢（EMA 糾纏）       → neutral_weight (0.75)，七五折

    為什麼「軟」比「硬」好？
      1. 避免在趨勢轉換期完全踏空
      2. 逆趨勢的均值回歸交易仍有盈利機會
      3. 降低倉位而非禁止，是更穩健的風控方式
      4. 交易頻率保持穩定，不會出現長期空倉

    Args:
        df:               低級 K 線數據（e.g. 1h）
        raw_pos:          原始持倉信號 [-1, 1]
        htf_interval:     高級時間框架，e.g. "4h"
        ema_fast:         快速 EMA 週期（在高級 TF 上），預設 20
        ema_slow:         慢速 EMA 週期（在高級 TF 上），預設 50
        current_interval: 當前 K 線週期，e.g. "1h"
        align_weight:     順趨勢權重（預設 1.0，全倉）
        counter_weight:   逆趨勢權重（預設 0.5，半倉）
        neutral_weight:   無趨勢權重（預設 0.75）

    Returns:
        加權後的持倉序列（連續值）
    """
    # 獲取重採樣規則
    resample_options = _RESAMPLE_MAP.get(current_interval, {})
    resample_rule = resample_options.get(htf_interval)

    if resample_rule is None:
        return raw_pos

    # 重採樣到高級時間框架
    htf_df = _resample_ohlcv(df, resample_rule)
    if len(htf_df) < ema_slow + 5:
        return raw_pos

    # 在高級 TF 上計算 EMA
    ema_f = calculate_ema(htf_df["close"], ema_fast)
    ema_s = calculate_ema(htf_df["close"], ema_slow)

    # 計算趨勢強度（連續值，基於 EMA 間距）
    # spread > 0 → 上升趨勢，spread < 0 → 下降趨勢
    ema_spread = (ema_f - ema_s) / ema_s  # 正規化 spread

    # 判斷離散趨勢方向（+1, 0, -1）
    htf_trend = pd.Series(0.0, index=htf_df.index)
    htf_trend[ema_f > ema_s] = 1.0
    htf_trend[ema_f < ema_s] = -1.0

    # 將趨勢映射回低級 TF
    htf_trend_ltf = htf_trend.reindex(df.index, method="ffill").fillna(0.0)

    # 計算權重
    raw = raw_pos.values.copy()
    trend = htf_trend_ltf.values
    result = np.zeros(len(raw), dtype=float)

    for i in range(len(raw)):
        if raw[i] == 0:
            result[i] = 0.0
            continue

        t = trend[i]
        if t > 0.5:  # 上升趨勢
            if raw[i] > 0:  # 做多 = 順趨勢
                weight = align_weight
            else:            # 做空 = 逆趨勢
                weight = counter_weight
        elif t < -0.5:  # 下降趨勢
            if raw[i] < 0:  # 做空 = 順趨勢
                weight = align_weight
            else:            # 做多 = 逆趨勢
                weight = counter_weight
        else:  # 無明確趨勢
            weight = neutral_weight

        result[i] = raw[i] * weight

    return pd.Series(result, index=raw_pos.index)


def funding_rate_filter(
    df: pd.DataFrame,
    raw_pos: pd.Series,
    funding_rates: pd.Series | None,
    max_positive_rate: float = 0.0002,  # 0.02%
    max_negative_rate: float = -0.0002, # -0.02%
) -> pd.Series:
    """
    Funding Rate 過濾器（擁擠交易過濾）

    邏輯：
    - Funding Rate > max_positive (0.02%) → 市場過熱 (Longs paying Shorts) → 禁止做多
    - Funding Rate < max_negative (-0.02%) → 市場過冷 (Shorts paying Longs) → 禁止做空
    - 如果 funding_rates 為 None，不過濾

    Args:
        df: K線數據
        raw_pos: 原始持倉信號
        funding_rates: Funding Rate 序列 (需與 df 對齊)
        max_positive_rate: 正費率閾值
        max_negative_rate: 負費率閾值
    """
    if funding_rates is None or funding_rates.empty:
        return raw_pos

    raw = raw_pos.values.copy()
    
    # 確保 funding_rates 與 raw_pos 對齊
    if len(funding_rates) != len(raw):
        # 嘗試重新對齊
        fr = funding_rates.reindex(raw_pos.index, method='ffill').fillna(0.0).values
    else:
        fr = funding_rates.values
        
    result = np.zeros(len(raw), dtype=float)
    position_state = 0

    for i in range(len(raw)):
        # Check current funding rate
        curr_fr = fr[i]
        
        # Is extreme?
        is_extreme_pos = not np.isnan(curr_fr) and curr_fr > max_positive_rate
        is_extreme_neg = not np.isnan(curr_fr) and curr_fr < max_negative_rate

        if raw[i] > 0:  # Long Signal
            if position_state == 1:
                # 保持持倉
                result[i] = raw[i]
            elif not is_extreme_pos: # Allow only if NOT extreme positive
                result[i] = raw[i]
                position_state = 1
            else:
                # 市場過熱，禁止做多
                result[i] = 0.0
        elif raw[i] < 0: # Short Signal
            if position_state == -1:
                # 保持持倉
                result[i] = raw[i]
            elif not is_extreme_neg: # Allow only if NOT extreme negative
                result[i] = raw[i]
                position_state = -1
            else:
                # 市場過冷，禁止做空
                result[i] = 0.0
        else:
            result[i] = 0.0
            position_state = 0
            
    return pd.Series(result, index=raw_pos.index)
