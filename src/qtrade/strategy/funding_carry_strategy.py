"""
Funding Rate Carry 策略

學術背景：
    - Koijen et al. (2018) "Carry" — carry factor 跨資產類別皆有效
    - 加密市場：funding rate = 永續合約 vs 現貨的持有成本
    - 正 funding = 多頭擁擠 → 做空收 funding → carry alpha
    - 負 funding = 空頭擁擠 → 做多收 funding → carry alpha

加密幣特殊處理：
    1. Funding rate 每 8h 結算一次（Binance）
    2. 波動率遠大於 carry 收益 → 必須波動率縮放
    3. 單幣 carry 策略容易被趨勢反噬 → 需要多幣截面排序
    4. 作為低頻、低相關 sleeve → 不追求高換手

策略邏輯：
    1. 讀取歷史 funding rate（8h 數據）
    2. 計算 rolling mean funding rate（平滑噪音）
    3. 信號生成：
       - 模式 A（單幣）：funding > threshold → 做空收 carry；< -threshold → 做多
       - 模式 B（截面排序）：跨幣種 funding 排名，top → 做空，bottom → 做多
    4. 持倉縮放：波動率目標
    5. Rebalance 間隔：每 N 小時重新評估（低頻）
    6. Min holding period：避免頻繁切換

策略變體：
    1. funding_carry       — 單幣 funding carry（proxy 模式也適用）
    2. funding_carry_xs    — 截面排序 carry（多幣種）

Note:
    signal_delay 和 direction clip 由 @register_strategy 框架自動處理。
    策略本身是低頻信號，1-bar delay 影響極小。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from .base import StrategyContext
from . import register_strategy

import logging
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
#  Funding Rate 數據載入
# ══════════════════════════════════════════════════════════════

def _load_funding_rate(
    symbol: str,
    data_dir: str | Path = "data",
    market_type: str = "futures",
) -> Optional[pd.Series]:
    """
    載入 funding rate 數據

    Returns:
        funding_rate Series（8h 頻率）或 None
    """
    fpath = Path(data_dir) / "binance" / market_type / "funding_rate" / f"{symbol}.parquet"
    if not fpath.exists():
        logger.warning(f"⚠️  Funding Carry: 找不到 {symbol} 的 funding rate: {fpath}")
        return None

    try:
        df = pd.read_parquet(fpath)
        if "funding_rate" in df.columns:
            return df["funding_rate"]
        else:
            logger.warning(f"⚠️  Funding Carry: {symbol} 無 funding_rate 欄位")
            return None
    except Exception as e:
        logger.warning(f"⚠️  Funding Carry: 載入 {symbol} 失敗: {e}")
        return None


def _align_funding_to_klines(
    funding: pd.Series,
    kline_index: pd.DatetimeIndex,
) -> pd.Series:
    """
    將 8h funding rate 對齊到 1h K 線 index

    使用 forward-fill（causal）：每個 1h bar 看到的是最近一次已結算的 funding rate
    """
    # Reindex to kline timestamps, forward-fill
    aligned = funding.reindex(kline_index, method="ffill")
    return aligned


# ══════════════════════════════════════════════════════════════
#  核心信號計算
# ══════════════════════════════════════════════════════════════

def _compute_carry_signal(
    funding_aligned: pd.Series,
    rolling_window: int = 72,
    entry_threshold: float = 0.0001,
    exit_threshold: float = 0.00005,
    rebalance_interval: int = 8,
    vol_target: float = 0.10,
    close: Optional[pd.Series] = None,
    vol_lookback: int = 168,
) -> pd.Series:
    """
    單幣 Funding Carry 信號

    邏輯：
      1. rolling mean funding rate（平滑）
      2. mean_fr > entry_threshold  → 做空（收正 funding）
         mean_fr < -entry_threshold → 做多（收負 funding）
         |mean_fr| < exit_threshold → 平倉
      3. Rebalance：每 N 小時才重新評估，中間保持倉位
      4. Vol scaling：以波動率目標縮放倉位

    Args:
        funding_aligned:    對齊到 1h 的 funding rate
        rolling_window:     rolling mean 窗口（小時，預設 72 = 3 天）
        entry_threshold:    入場 funding 門檻（預設 0.01% = 0.0001）
        exit_threshold:     出場 funding 門檻（預設 0.005% = 0.00005）
        rebalance_interval: 重新評估間隔（小時，預設 8 = 每次 funding 結算）
        vol_target:         年化波動率目標（預設 10% — 保守）
        close:              收盤價序列（用於 vol scaling）
        vol_lookback:       vol 計算回看期

    Returns:
        position Series [-1, 1]
    """
    n = len(funding_aligned)
    mean_fr = funding_aligned.rolling(rolling_window, min_periods=1).mean()

    # 生成 raw 信號（只在 rebalance 點更新）
    raw_signal = np.zeros(n, dtype=np.float64)
    current_pos = 0.0

    for i in range(n):
        if np.isnan(mean_fr.iloc[i]):
            raw_signal[i] = current_pos
            continue

        # Rebalance check
        if i % rebalance_interval == 0:
            fr_val = mean_fr.iloc[i]
            if fr_val > entry_threshold:
                # 正 funding → 做空收 carry
                current_pos = -1.0
            elif fr_val < -entry_threshold:
                # 負 funding → 做多收 carry
                current_pos = 1.0
            elif abs(fr_val) < exit_threshold:
                # funding 回到中性 → 平倉
                current_pos = 0.0
            # else: 保持

        raw_signal[i] = current_pos

    pos = pd.Series(raw_signal, index=funding_aligned.index)

    # Vol scaling
    if close is not None and vol_target > 0:
        returns = close.pct_change()
        vol = returns.rolling(vol_lookback, min_periods=max(vol_lookback // 2, 1)).std() * np.sqrt(8760)
        vol = vol.replace(0, np.nan).ffill().fillna(vol_target)
        scale = (vol_target / vol).clip(0.1, 2.0)
        pos = (pos * scale).clip(-1.0, 1.0)

    return pos.fillna(0.0)


# ══════════════════════════════════════════════════════════════
#  策略 1: 單幣 Funding Carry
# ══════════════════════════════════════════════════════════════

@register_strategy("funding_carry")
def generate_funding_carry(
    df: pd.DataFrame, ctx: StrategyContext, params: dict
) -> pd.Series:
    """
    單幣 Funding Rate Carry 策略

    低頻、低相關 sleeve：
      - 正 funding → 做空收 carry
      - 負 funding → 做多收 carry
      - 波動率縮放控制曝險

    params:
        data_dir:             數據目錄（預設 "data"）
        rolling_window:       funding 平滑窗口（預設 72 = 3 天）
        entry_threshold:      入場 funding 門檻（預設 0.0001 = 0.01%）
        exit_threshold:       出場 funding 門檻（預設 0.00005 = 0.005%）
        rebalance_interval:   重新評估間隔小時（預設 8）
        vol_target:           年化波動率目標（預設 0.10 = 10%）
        vol_lookback:         vol 計算回看期（預設 168 = 7 天）
        use_proxy:            若無 funding 數據，是否用 proxy（預設 True）
        proxy_ema_fast:       proxy 用快速 EMA（預設 24）
        proxy_ema_slow:       proxy 用慢速 EMA（預設 168）

        # F2: Basis/Premium Confirmation（預設 disabled）
        basis_confirm_enabled:    是否啟用 basis 確認（預設 False）
        basis_ema_fast:           basis proxy 快速 EMA（預設 24）
        basis_ema_slow:           basis proxy 慢速 EMA（預設 168）
        basis_confirm_threshold:  basis spread 門檻（預設 0.01 = 1%）

        # F3: Regime Filter（預設 disabled）
        regime_filter_enabled:    是否啟用 regime 過濾（預設 False）
        regime_adx_max:           ADX 上限（預設 25 — 只在低 ADX 非趨勢環境交易）
        regime_adx_period:        ADX 計算週期（預設 14）
    """
    data_dir = params.get("data_dir", "data")
    rolling_window = int(params.get("rolling_window", 72))
    entry_threshold = float(params.get("entry_threshold", 0.0001))
    exit_threshold = float(params.get("exit_threshold", 0.00005))
    rebalance_interval = int(params.get("rebalance_interval", 8))
    vol_target = float(params.get("vol_target", 0.10))
    vol_lookback = int(params.get("vol_lookback", 168))
    use_proxy = bool(params.get("use_proxy", True))

    # F2: Basis confirmation
    basis_confirm_enabled = bool(params.get("basis_confirm_enabled", False))
    basis_ema_fast = int(params.get("basis_ema_fast", 24))
    basis_ema_slow = int(params.get("basis_ema_slow", 168))
    basis_confirm_threshold = float(params.get("basis_confirm_threshold", 0.01))

    # F3: Regime filter
    regime_filter_enabled = bool(params.get("regime_filter_enabled", False))
    regime_adx_max = float(params.get("regime_adx_max", 25.0))
    regime_adx_period = int(params.get("regime_adx_period", 14))

    # 嘗試載入真實 funding rate
    funding = _load_funding_rate(ctx.symbol, data_dir, ctx.market_type)

    if funding is not None:
        # 對齊到 1h K 線
        funding_aligned = _align_funding_to_klines(funding, df.index)

        pos = _compute_carry_signal(
            funding_aligned,
            rolling_window=rolling_window,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            rebalance_interval=rebalance_interval,
            vol_target=vol_target,
            close=df["close"],
            vol_lookback=vol_lookback,
        )

        # ── F2: Basis/Premium Confirmation ──
        if basis_confirm_enabled:
            pos = _apply_basis_confirmation(
                df, pos,
                ema_fast=basis_ema_fast,
                ema_slow=basis_ema_slow,
                threshold=basis_confirm_threshold,
            )

        # ── F3: Regime Filter (low ADX only) ──
        if regime_filter_enabled:
            pos = _apply_regime_filter(
                df, pos,
                adx_max=regime_adx_max,
                adx_period=regime_adx_period,
            )

        return pos

    # ── Proxy 模式（若無 funding 數據） ──
    if use_proxy:
        logger.info(f"ℹ️  Funding Carry: {ctx.symbol} 無 funding 數據，使用 proxy")
        return _funding_proxy_signal(df, params)

    # 完全無法生成信號
    logger.warning(f"⚠️  Funding Carry: {ctx.symbol} 無 funding 且 proxy 關閉")
    return pd.Series(0.0, index=df.index)


def _apply_basis_confirmation(
    df: pd.DataFrame,
    pos: pd.Series,
    ema_fast: int = 24,
    ema_slow: int = 168,
    threshold: float = 0.01,
) -> pd.Series:
    """
    F2: Basis/Premium Confirmation Filter

    用 EMA spread 作為 basis/premium 的 proxy：
      - 做空信號 (pos < 0) 需要 basis > +threshold（市場溢價，確認多頭擁擠）
      - 做多信號 (pos > 0) 需要 basis < -threshold（市場折價，確認空頭擁擠）
      - 不確認 → 置零（不開倉）
      - 已有倉位但 basis 條件消失 → 保持（不強制平倉，等 carry 信號平倉）

    Anti-lookahead: EMA 是 causal，只用歷史數據。
    """
    close = df["close"]
    ema_f = close.ewm(span=ema_fast, adjust=False).mean()
    ema_s = close.ewm(span=ema_slow, adjust=False).mean()
    basis_spread = (ema_f - ema_s) / ema_s  # > 0 = premium, < 0 = discount

    result = pos.copy()
    pos_vals = pos.values
    basis_vals = basis_spread.values

    for i in range(len(pos_vals)):
        if np.isnan(basis_vals[i]):
            continue
        p = pos_vals[i]
        b = basis_vals[i]
        if p < 0 and b < threshold:
            # Short carry signal but no premium confirmation → block
            result.iloc[i] = 0.0
        elif p > 0 and b > -threshold:
            # Long carry signal but no discount confirmation → block
            result.iloc[i] = 0.0

    return result


def _apply_regime_filter(
    df: pd.DataFrame,
    pos: pd.Series,
    adx_max: float = 25.0,
    adx_period: int = 14,
) -> pd.Series:
    """
    F3: Regime Filter — Only trade carry in non-trending environments

    邏輯：
      - ADX > adx_max → 強趨勢環境 → carry 易被趨勢反噬 → 禁止新開倉
      - ADX <= adx_max → 震盪/低波動環境 → carry 策略適用
      - 已有倉位且 ADX 升高 → 保持（不強制平倉）

    Anti-lookahead: ADX 是 causal（只用當下及歷史數據）。
    """
    from ..indicators.adx import calculate_adx

    adx_df = calculate_adx(df, period=adx_period)
    adx = adx_df["ADX"] if isinstance(adx_df, pd.DataFrame) else adx_df
    adx_vals = adx.values
    pos_vals = pos.values
    result = pos.values.copy()

    prev_pos = 0.0
    for i in range(len(pos_vals)):
        if np.isnan(adx_vals[i]):
            result[i] = pos_vals[i]
            prev_pos = pos_vals[i]
            continue

        current_signal = pos_vals[i]

        if adx_vals[i] > adx_max:
            # High ADX: block new entries, keep existing positions
            if prev_pos == 0.0 and current_signal != 0.0:
                result[i] = 0.0  # Block new entry
            elif prev_pos != 0.0 and current_signal != 0.0:
                # Had position, keep it (allow regime filter to not force-close)
                result[i] = current_signal
            else:
                result[i] = current_signal
        else:
            result[i] = current_signal

        prev_pos = result[i]

    return pd.Series(result, index=pos.index)


def _funding_proxy_signal(
    df: pd.DataFrame,
    params: dict,
) -> pd.Series:
    """
    Funding Rate Proxy 信號

    當沒有真實 funding 數據時的替代方案：
    用 EMA spread 作為 funding 的 proxy：
      - 價格大幅高於長期均線 → 市場過熱 → 類似正 funding → 做空
      - 價格大幅低於長期均線 → 市場恐慌 → 類似負 funding → 做多

    這是均值回歸的一種，但邏輯上對應 carry 而非趨勢。
    """
    proxy_fast = int(params.get("proxy_ema_fast", 24))
    proxy_slow = int(params.get("proxy_ema_slow", 168))
    vol_target = float(params.get("vol_target", 0.10))
    vol_lookback = int(params.get("vol_lookback", 168))
    rebalance_interval = int(params.get("rebalance_interval", 8))
    proxy_threshold = float(params.get("proxy_threshold", 0.02))

    close = df["close"]

    # EMA spread 作為 funding proxy
    ema_fast = close.ewm(span=proxy_fast, adjust=False).mean()
    ema_slow = close.ewm(span=proxy_slow, adjust=False).mean()
    spread = (ema_fast - ema_slow) / ema_slow  # 正規化 spread

    n = len(df)
    raw_signal = np.zeros(n, dtype=np.float64)
    current_pos = 0.0

    spread_vals = spread.values

    for i in range(n):
        if np.isnan(spread_vals[i]):
            raw_signal[i] = current_pos
            continue

        if i % rebalance_interval == 0:
            if spread_vals[i] > proxy_threshold:
                current_pos = -1.0  # 過熱 → 做空
            elif spread_vals[i] < -proxy_threshold:
                current_pos = 1.0   # 恐慌 → 做多
            elif abs(spread_vals[i]) < proxy_threshold * 0.5:
                current_pos = 0.0   # 中性 → 平倉

        raw_signal[i] = current_pos

    pos = pd.Series(raw_signal, index=df.index)

    # Vol scaling
    returns = close.pct_change()
    vol = returns.rolling(vol_lookback, min_periods=max(vol_lookback // 2, 1)).std() * np.sqrt(8760)
    vol = vol.replace(0, np.nan).ffill().fillna(vol_target)
    scale = (vol_target / vol).clip(0.1, 2.0)
    pos = (pos * scale).clip(-1.0, 1.0)

    return pos.fillna(0.0)


# ══════════════════════════════════════════════════════════════
#  策略 2: 截面排序 Carry（多幣種）
# ══════════════════════════════════════════════════════════════

@register_strategy("funding_carry_xs")
def generate_funding_carry_xs(
    df: pd.DataFrame, ctx: StrategyContext, params: dict
) -> pd.Series:
    """
    截面排序 Funding Carry 策略

    邏輯：
      1. 載入 universe 所有幣種的 funding rate
      2. 計算 rolling mean funding
      3. 排序：funding 最高 → 做空（top carry），funding 最低 → 做多
      4. 當前 symbol 根據排名決定倉位

    優勢（vs 單幣 carry）：
      - 相對排名更穩定（扣除市場整體 funding 偏移）
      - 捕捉的是 carry 差異，而非絕對水準

    params:
        universe:             幣種列表
        data_dir:             數據目錄
        rolling_window:       funding 平滑窗口（預設 72）
        rebalance_interval:   重新評估間隔（預設 8）
        long_rank_threshold:  做多門檻（排名最低 N%，預設 0.3）
        short_rank_threshold: 做空門檻（排名最高 N%，預設 0.7）
        vol_target:           波動率目標（預設 0.10）
    """
    universe = params.get("universe", [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "AVAXUSDT"
    ])
    data_dir = params.get("data_dir", "data")
    rolling_window = int(params.get("rolling_window", 72))
    rebalance_interval = int(params.get("rebalance_interval", 8))
    long_rank_threshold = float(params.get("long_rank_threshold", 0.3))
    short_rank_threshold = float(params.get("short_rank_threshold", 0.7))
    vol_target = float(params.get("vol_target", 0.10))
    vol_lookback = int(params.get("vol_lookback", 168))

    # 載入所有 universe 的 funding rate
    fr_dict = {}
    for sym in universe:
        fr = _load_funding_rate(sym, data_dir, ctx.market_type)
        if fr is not None:
            fr_aligned = _align_funding_to_klines(fr, df.index)
            fr_dict[sym] = fr_aligned.rolling(rolling_window, min_periods=1).mean()

    if len(fr_dict) < 3:
        logger.warning(f"⚠️  Funding Carry XS: funding 數據不足 ({len(fr_dict)}/{len(universe)})")
        return pd.Series(0.0, index=df.index)

    # 逐時間點排名
    fr_df = pd.DataFrame(fr_dict)
    rank_df = fr_df.rank(axis=1, pct=True, na_option="keep")

    # 取當前 symbol 的排名
    if ctx.symbol not in rank_df.columns:
        logger.warning(f"⚠️  Funding Carry XS: {ctx.symbol} 不在 universe 中")
        return pd.Series(0.0, index=df.index)

    rank = rank_df[ctx.symbol]

    # 信號生成（每 rebalance_interval 更新）
    n = len(df)
    raw_signal = np.zeros(n, dtype=np.float64)
    current_pos = 0.0
    rank_vals = rank.values

    for i in range(n):
        if np.isnan(rank_vals[i]):
            raw_signal[i] = current_pos
            continue

        if i % rebalance_interval == 0:
            if rank_vals[i] >= short_rank_threshold:
                # Funding 排名最高 → 做空收 carry
                current_pos = -1.0
            elif rank_vals[i] <= long_rank_threshold:
                # Funding 排名最低 → 做多收 carry
                current_pos = 1.0
            else:
                current_pos = 0.0

        raw_signal[i] = current_pos

    pos = pd.Series(raw_signal, index=df.index)

    # Vol scaling
    close = df["close"]
    returns = close.pct_change()
    vol = returns.rolling(vol_lookback, min_periods=max(vol_lookback // 2, 1)).std() * np.sqrt(8760)
    vol = vol.replace(0, np.nan).ffill().fillna(vol_target)
    scale = (vol_target / vol).clip(0.1, 2.0)
    pos = (pos * scale).clip(-1.0, 1.0)

    return pos.fillna(0.0)
