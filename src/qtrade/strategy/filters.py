"""
通用信號過濾器

設計理念：
    過濾器作為後處理器，疊加在策略的原始信號上。
    如果當前 bar 不滿足過濾條件，不允許新開倉（但不強制平倉）。

用法：
    from qtrade.strategy.filters import trend_filter, volume_filter, htf_trend_filter, volatility_filter, time_of_day_filter, oi_regime_filter, onchain_regime_filter, causal_resample_align

    raw_pos = my_indicator_logic(df, params)
    pos = trend_filter(df, raw_pos, min_adx=25)
    pos = volume_filter(df, pos, min_volume_ratio=1.2)
    pos = volatility_filter(df, pos, min_atr_ratio=0.005)  # 波動率過濾
    pos = htf_trend_filter(df, pos, htf_interval="4h")  # 高級時間框架趨勢
    pos = onchain_regime_filter(pos, tvl_sc_ratio_mom_30d_series)  # 鏈上 regime 過濾
"""
from __future__ import annotations
from typing import Callable
import logging
import traceback
import pandas as pd
import numpy as np
from ..indicators.adx import calculate_adx
from ..indicators.atr import calculate_atr
from ..indicators.volume import calculate_obv
from ..indicators.moving_average import calculate_ema

_logger = logging.getLogger(__name__)


def trend_filter(
    df: pd.DataFrame,
    raw_pos: pd.Series,
    min_adx: float = 25.0,
    short_min_adx: float | None = None,
    adx_period: int = 14,
    require_uptrend: bool = True,
) -> pd.Series:
    """
    ADX 趨勢過濾器（支援做空，多空可分別設門檻）

    規則：
    - ADX < min_adx → 無趨勢，禁止新開多倉
    - ADX < short_min_adx → 趨勢不夠強，禁止新開空倉
    - 做多：require_uptrend=True 時，要求 +DI > -DI（上升趨勢）
    - 做空：require_uptrend=True 時，要求 -DI > +DI（下降趨勢）
    - 已有持倉不受影響（由出場規則或策略信號決定平倉）

    為什麼空頭要更高的 ADX 門檻？
        加密市場有天然的多頭偏差（long bias）。做空本質上逆風，
        需要更強的下跌趨勢確認才能獲利。數據顯示空頭 WR 比多頭低 15-20pp，
        提高空頭 ADX 門檻可過濾大量低品質空頭交易。

    Args:
        df:              K線數據
        raw_pos:         原始持倉信號 [-1, 1]
        min_adx:         做多最小 ADX 值，低於此值不開新多倉
        short_min_adx:   做空最小 ADX 值，None = 與 min_adx 相同
        adx_period:      ADX 週期
        require_uptrend: 是否要求趨勢方向配合

    Returns:
        過濾後的持倉序列
    """
    _short_adx = short_min_adx if short_min_adx is not None else min_adx

    adx_data = calculate_adx(df, adx_period)
    adx = adx_data["ADX"].values
    plus_di = adx_data["+DI"].values
    minus_di = adx_data["-DI"].values

    raw = raw_pos.values.copy()
    result = np.zeros(len(raw), dtype=float)
    position_state = 0  # 0 = flat, 1 = long, -1 = short

    for i in range(len(raw)):
        has_long_trend = not np.isnan(adx[i]) and adx[i] >= min_adx
        has_short_trend = not np.isnan(adx[i]) and adx[i] >= _short_adx
        is_uptrend = plus_di[i] > minus_di[i]
        is_downtrend = minus_di[i] > plus_di[i]

        if raw[i] > 0:  # 做多信號
            if position_state == 1:
                # 已有多倉 → 保持
                result[i] = raw[i]
            elif has_long_trend and (is_uptrend or not require_uptrend):
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
            elif has_short_trend and (is_downtrend or not require_uptrend):
                # 允許新開空倉（需要更強的下降趨勢）
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


def _resample_ohlcv(df: pd.DataFrame, rule: str, *, _caller: str = "") -> pd.DataFrame:
    """
    將低級時間框架 K 線重採樣為高級時間框架

    ⚠️ 注意：此函數不保證因果性。
    resample 使用 pandas 預設 closed='left', label='left'，
    HTF bar 的 label 在 bar 起始（例如 4h bar [00:00, 04:00) 標記在 00:00），
    但 close 實際上 03:59 才可用。

    如果要將結果映射回低級 TF（reindex + ffill），
    必須先 shift(1) 延遲 1 個 HTF bar，否則會有 intra-bar look-ahead。
    建議直接使用 causal_resample_align() 代替手動 resample + reindex。

    Args:
        df:       低級 K 線數據（必須有 DatetimeIndex）
        rule:     pandas resample rule, e.g. "4h", "1D"
        _caller:  內部參數。causal_resample_align 傳 "causal" 以跳過警告。

    Returns:
        高級 K 線 DataFrame（OHLCV）
    """
    # Runtime 防護：如果不是從 causal_resample_align 呼叫的，發出警告
    if _caller != "causal":
        caller_info = "".join(traceback.format_stack(limit=3)[:-1]).strip()
        _logger.warning(
            "⚠️  _resample_ohlcv() 被直接呼叫！如需將結果映射回低級 TF，"
            "請使用 causal_resample_align() 避免 look-ahead。\n"
            f"  呼叫位置:\n{caller_info}"
        )
    return df.resample(rule).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()


def causal_resample_align(
    df: pd.DataFrame,
    freq: str,
    compute_fn: Callable[[pd.DataFrame], pd.Series | pd.DataFrame],
    target_index: pd.DatetimeIndex,
    fillna_value: float = 0.0,
) -> pd.Series | pd.DataFrame:
    """Resample df 到 freq，套用 compute_fn，shift(1)，再 reindex 到 target_index。

    這是在低級 TF 策略中使用 HTF 特徵的**唯一正確方式**。
    shift(1) 確保 HTF bar 的結果只在 bar **收盤後**才被使用，
    避免 intra-bar look-ahead（例如 4h bar close 在 03:59 才可用，
    但 label='left' 把它標記在 00:00）。

    用法範例::

        # 計算 4h EMA 趨勢並因果對齊到 1h
        def compute_trend(htf_df):
            ema_f = calculate_ema(htf_df["close"], 20)
            ema_s = calculate_ema(htf_df["close"], 50)
            trend = pd.Series(0.0, index=htf_df.index)
            trend[ema_f > ema_s] = 1.0
            trend[ema_f < ema_s] = -1.0
            return trend

        trend_1h = causal_resample_align(df, "4h", compute_trend, df.index)

    Args:
        df:           低級 K 線數據（必須有 OHLCV + DatetimeIndex）
        freq:         目標 HTF 頻率，如 "4h", "1D"
        compute_fn:   接收 HTF OHLCV DataFrame，回傳 Series 或 DataFrame
        target_index: 要對齊回的低級 TF index
        fillna_value: ffill 後的 NaN 填充值（預設 0.0）

    Returns:
        已因果對齊到 target_index 的 Series 或 DataFrame
    """
    htf_df = _resample_ohlcv(df, freq, _caller="causal")
    result = compute_fn(htf_df)
    result = result.shift(1)  # 等 bar 收盤後才用
    if isinstance(result, pd.DataFrame):
        return pd.DataFrame({
            c: result[c].reindex(target_index, method="ffill").fillna(fillna_value)
            for c in result.columns
        })
    return result.reindex(target_index, method="ffill").fillna(fillna_value)


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

    # 使用 causal_resample_align 避免 intra-bar look-ahead
    def _compute_trend(htf_df: pd.DataFrame) -> pd.Series:
        if len(htf_df) < ema_slow + 5:
            return pd.Series(0.0, index=htf_df.index)
        ema_f = calculate_ema(htf_df["close"], ema_fast)
        ema_s = calculate_ema(htf_df["close"], ema_slow)
        trend = pd.Series(0.0, index=htf_df.index)
        trend[ema_f > ema_s] = 1.0
        trend[ema_f < ema_s] = -1.0
        return trend

    htf_trend_ltf = causal_resample_align(df, resample_rule, _compute_trend, df.index)

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

    # 使用 causal_resample_align 避免 intra-bar look-ahead
    def _compute_trend(htf_df: pd.DataFrame) -> pd.Series:
        if len(htf_df) < ema_slow + 5:
            return pd.Series(0.0, index=htf_df.index)
        ema_f = calculate_ema(htf_df["close"], ema_fast)
        ema_s = calculate_ema(htf_df["close"], ema_slow)
        trend = pd.Series(0.0, index=htf_df.index)
        trend[ema_f > ema_s] = 1.0
        trend[ema_f < ema_s] = -1.0
        return trend

    htf_trend_ltf = causal_resample_align(df, resample_rule, _compute_trend, df.index)

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


def efficiency_ratio_filter(
    df: pd.DataFrame,
    raw_pos: pd.Series,
    er_period: int = 10,
    er_mode: str = "gate",
    er_min: float = 0.20,
    er_threshold_low: float = 0.20,
    er_threshold_high: float = 0.50,
    weight_at_low: float = 0.30,
    weight_at_high: float = 1.0,
) -> pd.Series:
    """
    Efficiency Ratio 震盪過濾器

    Kaufman Efficiency Ratio 衡量價格移動的「效率」：
        ER = |淨位移| / 總路徑
        ER → 1 = 趨勢（價格走直線），ER → 0 = 震盪（價格來回走）

    支援兩種模式：

    1. **gate（硬閘門）**：ER < er_min 時不允許新開倉，已有倉位不受影響。
       類似 ADX filter 的行為 — 只擋入場，不強制出場，不產生額外交易。
       **推薦使用**：避免倉位頻繁微調造成交易摩擦。

    2. **scale（軟縮放）**：根據 ER 值線性縮放倉位大小。
       ER 高 → 全倉，ER 低 → 降倉。
       注意：ER 每 bar 都在變，會導致倉位持續微調 → 增加交易次數。

    Args:
        df:                 K 線數據
        raw_pos:            原始持倉信號 [-1, 1]
        er_period:          ER 回看週期（預設 10 bars = Kaufman 建議值）
        er_mode:            "gate"（硬閘門，推薦）或 "scale"（軟縮放）
        er_min:             gate 模式的最低 ER（低於此值不開新倉）
        er_threshold_low:   scale 模式的低 ER 閾值
        er_threshold_high:  scale 模式的高 ER 閾值
        weight_at_low:      scale 模式低 ER 時的倉位權重
        weight_at_high:     scale 模式高 ER 時的倉位權重

    Returns:
        過濾後的持倉序列
    """
    from ..indicators.efficiency_ratio import calculate_efficiency_ratio

    er = calculate_efficiency_ratio(df["close"], period=er_period).values
    raw = raw_pos.values.copy()
    result = np.zeros(len(raw), dtype=float)

    if er_mode == "gate":
        # ── Gate 模式：只擋新開倉，已有倉位保持不變 ──
        position_state = 0  # 0=flat, 1=long, -1=short

        for i in range(len(raw)):
            er_ok = not np.isnan(er[i]) and er[i] >= er_min

            if raw[i] > 0:  # 做多信號
                if position_state == 1:
                    result[i] = raw[i]  # 已有多倉 → 保持
                elif er_ok:
                    result[i] = raw[i]  # ER 夠高 → 允許開多
                    position_state = 1
                else:
                    result[i] = 0.0     # ER 太低 → 擋掉
            elif raw[i] < 0:  # 做空信號
                if position_state == -1:
                    result[i] = raw[i]  # 已有空倉 → 保持
                elif er_ok:
                    result[i] = raw[i]  # ER 夠高 → 允許開空
                    position_state = -1
                else:
                    result[i] = 0.0     # ER 太低 → 擋掉
            else:
                result[i] = 0.0
                position_state = 0

    else:  # scale 模式
        # ── Scale 模式：根據 ER 值連續縮放倉位 ──
        for i in range(len(raw)):
            if raw[i] == 0 or np.isnan(er[i]):
                result[i] = 0.0 if raw[i] == 0 else raw[i]
                continue

            if er[i] <= er_threshold_low:
                w = weight_at_low
            elif er[i] >= er_threshold_high:
                w = weight_at_high
            else:
                ratio = (er[i] - er_threshold_low) / (er_threshold_high - er_threshold_low)
                w = weight_at_low + ratio * (weight_at_high - weight_at_low)

            result[i] = raw[i] * w

    return pd.Series(result, index=raw_pos.index)


def smooth_positions(
    pos: pd.Series,
    min_delta: float = 0.30,
) -> pd.Series:
    """
    倉位平滑器：忽略微小的倉位調整，減少不必要的交易。

    問題背景：
        HTF Soft Filter 每 bar 根據 4H EMA 趨勢狀態動態調整倉位權重
        (1.0 / 0.75 / 0.5)，當趨勢在邊界震盪時，會在相鄰狀態間頻繁切換
        (e.g. 0.75 ↔ 1.0)，每次切換 VBT 都算一筆交易 → 大量手續費損耗。

    規則：
        - 進場（0 → 非0）→ 永遠允許
        - 出場（非0 → 0）→ 永遠允許
        - 方向翻轉（多 ↔ 空）→ 永遠允許
        - 同方向微調 < min_delta → 保持原倉位，不交易

    為什麼 min_delta = 0.30？
        HTF soft filter 的權重值為 {0.5, 0.75, 1.0}，
        相鄰切換的 delta = 0.25（< 0.30 → 被擋），
        跨級切換的 delta = 0.50（> 0.30 → 通過）。
        這樣只阻擋噪音切換，不影響趨勢大翻轉。

    Args:
        pos:       倉位序列（經過所有 filter 後的值）
        min_delta: 同方向最小變化閾值（預設 0.30）

    Returns:
        平滑後的倉位序列
    """
    values = pos.values.copy()

    for i in range(1, len(values)):
        prev = values[i - 1]
        curr = values[i]

        # 出場或進場 → 永遠允許
        if curr == 0.0 or prev == 0.0:
            continue

        # 方向改變（多 ↔ 空）→ 永遠允許
        if (prev > 0) != (curr > 0):
            continue

        # 同方向：只在變化足夠大時才調倉
        if abs(curr - prev) < min_delta:
            values[i] = prev

    return pd.Series(values, index=pos.index)


def time_of_day_filter(
    df: pd.DataFrame,
    raw_pos: pd.Series,
    blocked_hours: list[int] | None = None,
) -> pd.Series:
    """
    Time-of-Day 過濾器：在指定 UTC 小時禁止新開倉

    為什麼有效？
        加密市場雖然 24/7 交易，但不同時段的流動性和波動品質差異顯著：
        - 亞洲收盤 → 歐洲開盤過渡期（~09 UTC）：方向不明，whipsaw 多
        - 歐洲午休（~12 UTC）：流動性斷層，假突破多
        這些時段進場的交易，跨三幣種、跨年度都顯示負 Sharpe。

    規則：
        - 當前 bar 的 UTC 小時在 blocked_hours 中 → 不允許新開倉
        - 已有持倉不受影響（不強制平倉）
        - blocked_hours 為空或 None → 不過濾（原樣返回）

    Args:
        df:             K 線數據（index 為 DatetimeIndex UTC）
        raw_pos:        原始持倉信號 [-1, 1]
        blocked_hours:  要封鎖的 UTC 小時列表，e.g. [9, 12]

    Returns:
        過濾後的持倉序列
    """
    if not blocked_hours:
        return raw_pos

    blocked_set = set(blocked_hours)
    hours = df.index.hour  # UTC hours

    raw = raw_pos.values.copy()
    result = np.zeros(len(raw), dtype=float)
    position_state = 0  # 0=flat, 1=long, -1=short

    for i in range(len(raw)):
        is_blocked = hours[i] in blocked_set

        if raw[i] > 0:  # 做多信號
            if position_state == 1:
                result[i] = raw[i]  # 已有多倉 → 保持
            elif not is_blocked:
                result[i] = raw[i]  # 允許新開多倉
                position_state = 1
            else:
                result[i] = 0.0     # 封鎖時段 → 擋掉
        elif raw[i] < 0:  # 做空信號
            if position_state == -1:
                result[i] = raw[i]  # 已有空倉 → 保持
            elif not is_blocked:
                result[i] = raw[i]  # 允許新開空倉
                position_state = -1
            else:
                result[i] = 0.0     # 封鎖時段 → 擋掉
        else:
            result[i] = 0.0
            position_state = 0

    return pd.Series(result, index=raw_pos.index)


def volatility_regime_scaler(
    df: pd.DataFrame,
    raw_pos: pd.Series,
    atr_period: int = 14,
    lookback: int = 168,
    low_vol_percentile: float = 30.0,
    low_vol_weight: float = 0.5,
) -> pd.Series:
    """
    波動率 Regime 倉位縮放器

    根據歷史波動率百分位動態調整倉位大小：
    - 高波動（ATR percentile > threshold）→ 全倉
    - 低波動（ATR percentile ≤ threshold）→ 降倉（low_vol_weight）

    原理：
        Regime 分析顯示策略在高波動期 Sharpe 是低波動期的 2-3 倍。
        低波動時 edge 小、手續費佔比大 → 減小倉位可降低磨損。
        與 volatility_filter（二元閘門）不同，本函數是連續縮放器：
        - volatility_filter: 低波動 → 完全不開倉
        - regime_scaler:     低波動 → 倉位打折，仍保留交易機會

    Args:
        df:                  K 線數據
        raw_pos:             原始持倉信號 [-1, 1]
        atr_period:          ATR 計算週期
        lookback:            滾動百分位回看 bar 數（預設 168 = 1h × 7 天）
        low_vol_percentile:  低波動閾值百分位（低於此值為低波動 regime）
        low_vol_weight:      低波動時倉位權重（預設 0.5 = 半倉）

    Returns:
        縮放後的持倉序列
    """
    atr = calculate_atr(df, atr_period)
    close = df["close"]
    atr_ratio = atr / close

    # 滾動百分位排名（0~1 → 0~100）
    atr_pct = atr_ratio.rolling(lookback, min_periods=max(lookback // 2, 1)).rank(pct=True) * 100

    raw = raw_pos.values.copy()
    pct = atr_pct.values
    result = np.zeros(len(raw), dtype=float)

    for i in range(len(raw)):
        if raw[i] == 0:
            result[i] = 0.0
            continue

        if np.isnan(pct[i]):
            # 資料不足時保持原倉位
            result[i] = raw[i]
            continue

        if pct[i] <= low_vol_percentile:
            result[i] = raw[i] * low_vol_weight
        else:
            result[i] = raw[i]

    return pd.Series(result, index=raw_pos.index)


def oi_regime_filter(
    raw_pos: pd.Series,
    oi_series: pd.Series,
    lookback: int = 720,
    min_pctrank: float = 0.3,
) -> pd.Series:
    """
    OI Regime 過濾器：在 OI 水位過低時禁止新開倉

    根據 Alpha Researcher 2026-02-28 EDA 結果：
      - OI pctrank_720 < 0.3 → OI 在近 720 bar 的低位，市場參與度不足
      - F5 (pctrank > 0.3) filter: Δ SR +0.317, 8/8 symbols improved, freq loss 29.8%
      - Raw IC = -0.006 (弱)，但 quintile spread Q5-Q1 = -1.31 Sharpe (強條件效應)
      - 方向交互: Long+FallingOI SR=1.50 vs Short+FallingOI SR=0.01

    機制假說：
      低 OI → 低市場參與度 → 假突破多，動量信號不可靠。
      OI 是外部數據 (獨立結算時間戳)，reindex+ffill 是安全的（無 intra-bar look-ahead）。

    規則：
      - OI 的 rolling pctrank (0~1) < min_pctrank → 禁止新開倉
      - 已有持倉不受影響（不強制平倉）
      - OI 數據缺失時保持原有信號（不阻擋）

    Args:
        raw_pos:      原始持倉信號 [-1, 1]
        oi_series:    OI 值序列（已對齊到 kline index，forward-fill）
        lookback:     pctrank 回看窗口（預設 720 bars = 30 天 @ 1h）
        min_pctrank:  最低百分位排名閾值（預設 0.3）

    Returns:
        過濾後的持倉序列
    """
    if oi_series is None or oi_series.empty:
        return raw_pos

    # 計算 rolling pctrank (0~1)
    oi_pctrank = oi_series.rolling(lookback, min_periods=max(lookback // 2, 1)).rank(pct=True)

    raw = raw_pos.values.copy()
    pctrank = oi_pctrank.reindex(raw_pos.index, method="ffill").fillna(1.0).values
    result = np.zeros(len(raw), dtype=float)
    position_state = 0  # 0=flat, 1=long, -1=short

    for i in range(len(raw)):
        oi_ok = pctrank[i] >= min_pctrank

        if raw[i] > 0:  # 做多信號
            if position_state == 1:
                result[i] = raw[i]  # 已有多倉 → 保持
            elif oi_ok:
                result[i] = raw[i]  # OI 水位夠 → 允許新開多
                position_state = 1
            else:
                result[i] = 0.0     # OI 水位低 → 擋掉
        elif raw[i] < 0:  # 做空信號
            if position_state == -1:
                result[i] = raw[i]  # 已有空倉 → 保持
            elif oi_ok:
                result[i] = raw[i]  # OI 水位夠 → 允許新開空
                position_state = -1
            else:
                result[i] = 0.0     # OI 水位低 → 擋掉
        else:
            result[i] = 0.0
            position_state = 0

    return pd.Series(result, index=raw_pos.index)


def onchain_regime_filter(
    raw_pos: pd.Series,
    indicator_series: pd.Series,
    lookback: int = 720,
    min_pctrank: float = 0.3,
) -> pd.Series:
    """
    On-Chain Regime 過濾器：在鏈上 regime 指標低位時禁止新開倉

    根據 Alpha Researcher 2026-02-28 On-Chain Regime Overlay EDA：
      - 最佳指標: tvl_sc_ratio_mom_30d（TVL/穩定幣比率的 30d 動量）
      - IC = 0.065（>10× OI 的 IC=0.006）
      - Quintile spread +4.69（monotonic）
      - Filter ≥P30: 8/8 symbols improved, avg Δ SR = +0.409, freq loss ~30%
      - Risk-On vs Risk-Off: avg Δ SR = +1.454
      - Quality Gates: 6/6 PASS → GO

    機制假說：
      TVL/穩定幣比率上升 → DeFi 效率提高 → risk-on 環境 → 趨勢信號更可靠
      TVL/穩定幣比率下降 → 資金外流 → risk-off → 假突破增多

    鏈上數據特性：
      - Daily 頻率，1-24h 延遲（取決於 DeFi Llama 更新週期）
      - 外部數據（獨立時間戳），reindex+ffill 不存在 intra-bar look-ahead
      - 但必須 shift(1) 延遲 1 天確保因果性（當天 TVL 收盤時才可用）

    規則：
      - indicator_series 的 rolling pctrank (0~1) < min_pctrank → 禁止新開倉
      - 已有持倉不受影響（不強制平倉）
      - 數據缺失時保持原有信號（不阻擋）

    Args:
        raw_pos:          原始持倉信號 [-1, 1]
        indicator_series: 鏈上 regime 指標（已對齊到 kline index，forward-fill）
                          預期為 tvl_sc_ratio_mom_30d
        lookback:         pctrank 回看窗口（預設 720 bars = 30 天 @ 1h）
                          注意：indicator_series 已 reindex 到 1h，所以單位是 1h bars
        min_pctrank:      最低百分位排名閾值（預設 0.3 = P30）

    Returns:
        過濾後的持倉序列
    """
    if indicator_series is None or indicator_series.empty:
        return raw_pos

    # 計算 rolling pctrank (0~1)
    pctrank = indicator_series.rolling(
        lookback, min_periods=max(lookback // 2, 1)
    ).rank(pct=True)

    # 對齊到 raw_pos index (ffill for daily → 1h)
    pctrank_aligned = pctrank.reindex(raw_pos.index, method="ffill").fillna(1.0).values

    raw = raw_pos.values.copy()
    result = np.zeros(len(raw), dtype=float)
    position_state = 0  # 0=flat, 1=long, -1=short

    for i in range(len(raw)):
        oc_ok = pctrank_aligned[i] >= min_pctrank

        if raw[i] > 0:  # 做多信號
            if position_state == 1:
                result[i] = raw[i]  # 已有多倉 → 保持
            elif oc_ok:
                result[i] = raw[i]  # regime 良好 → 允許新開多
                position_state = 1
            else:
                result[i] = 0.0     # regime 低迷 → 擋掉
        elif raw[i] < 0:  # 做空信號
            if position_state == -1:
                result[i] = raw[i]  # 已有空倉 → 保持
            elif oc_ok:
                result[i] = raw[i]  # regime 良好 → 允許新開空
                position_state = -1
            else:
                result[i] = 0.0     # regime 低迷 → 擋掉
        else:
            result[i] = 0.0
            position_state = 0

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
