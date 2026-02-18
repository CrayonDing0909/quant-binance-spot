"""
XSMOM（Cross-Sectional Momentum）策略 — 改進版

學術背景：
    Jegadeesh & Titman (1993) "Returns to Buying Winners and Selling Losers"
    Liu, Tsyvinski & Wu (2022) "Common Risk Factors in Cryptocurrency"
    Ehsani & Linnainmaa (2022) "Factor Momentum and the Momentum Factor"

加密幣特殊處理（vs 傳統股票 XSMOM）：
    1. 殘差動量（Residual Momentum）：
       - 加密幣相關性 ~0.7+，BTC 漲全部漲
       - 先迴歸掉 BTC（市場因子），用殘差排名
       - 這才能捕捉真正的「相對強弱」
    2. 跳過近期（Skip Recent）：
       - 短期（1-3 天）有反轉效應，不是動量
       - Jegadeesh & Titman 跳 1 個月，加密跳 24-72h
    3. 長 lookback：
       - 股票用 3-12 個月，加密用 14-90 天
    4. 多 lookback 集成：
       - 單一 lookback 噪音大，多期平均更穩

策略變體：
    1. xsmom           — 殘差動量排名（改進版）
    2. xsmom_tsmom     — XSMOM + TSMOM 組合（低相關）

Note:
    signal_delay 和 direction clip 由 @register_strategy 框架自動處理。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from .base import StrategyContext
from . import register_strategy
from ..indicators import calculate_ema
from ..data.storage import load_klines

import logging
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════
#  模組級快取（避免每個 symbol 重複載入全 universe 數據）
# ══════════════════════════════════════════════════════════

_UNIVERSE_CACHE: dict[str, pd.DataFrame] = {}
_CACHE_TIMESTAMP: Optional[float] = None
_CACHE_TTL_SECONDS = 3600  # 快取 1 小時


def _load_universe_data(
    universe: list[str],
    data_dir: str | Path,
    interval: str = "1h",
    market_type: str = "futures",
) -> dict[str, pd.DataFrame]:
    """載入 universe 所有幣種的 K 線數據（帶快取）"""
    global _UNIVERSE_CACHE, _CACHE_TIMESTAMP
    import time

    now = time.time()
    cache_valid = (
        _CACHE_TIMESTAMP is not None
        and (now - _CACHE_TIMESTAMP) < _CACHE_TTL_SECONDS
        and set(universe).issubset(_UNIVERSE_CACHE.keys())
    )

    if cache_valid:
        return {s: _UNIVERSE_CACHE[s] for s in universe}

    data_path = Path(data_dir) / "binance" / market_type / interval
    result = {}
    for sym in universe:
        fpath = data_path / f"{sym}.parquet"
        if fpath.exists():
            try:
                df = load_klines(fpath)
                result[sym] = df
            except Exception as e:
                logger.warning(f"⚠️  XSMOM: 載入 {sym} 失敗: {e}")
        else:
            logger.warning(f"⚠️  XSMOM: 找不到 {sym} 的數據: {fpath}")

    _UNIVERSE_CACHE = result
    _CACHE_TIMESTAMP = now
    return result


# ══════════════════════════════════════════════════════════
#  核心：殘差動量排名計算
# ══════════════════════════════════════════════════════════

def _compute_residual_returns(
    universe_data: dict[str, pd.DataFrame],
    market_symbol: str = "BTCUSDT",
) -> dict[str, pd.Series]:
    """
    計算殘差報酬（去除 BTC 市場因子）

    對每個幣種 i：
        ret_i = alpha_i + beta_i * ret_BTC + epsilon_i
        residual_i = ret_i - beta_i * ret_BTC
                   ≈ ret_i - ret_BTC（簡化版，假設 beta≈1）

    為什麼需要：
        加密幣相關性 ~0.7+，BTC 漲 10% 時 ETH 漲 12%
        原始排名：ETH 排在前面，但只是因為 beta 高
        殘差排名：ETH 超額 +2%，這才是真正的相對強弱

    Args:
        universe_data: {symbol: DataFrame}
        market_symbol: 市場因子（預設 BTC）

    Returns:
        {symbol: residual_return_series}
    """
    # 收集所有幣種的 1-bar 報酬
    ret_dict = {}
    for sym, df in universe_data.items():
        ret_dict[sym] = df["close"].pct_change()

    ret_df = pd.DataFrame(ret_dict)

    # 市場報酬（BTC）
    if market_symbol in ret_df.columns:
        market_ret = ret_df[market_symbol]
    else:
        # 沒有 BTC 就用等權平均
        market_ret = ret_df.mean(axis=1)

    # 簡化版殘差：ret_i - ret_market
    # （完整版需要 rolling OLS 估計 beta，但噪音更大，不適合小 universe）
    residual_dict = {}
    for sym in ret_df.columns:
        residual_dict[sym] = ret_df[sym] - market_ret

    return residual_dict


def _compute_xsmom_rankings(
    universe_data: dict[str, pd.DataFrame],
    lookback: int = 720,
    skip_recent: int = 24,
    use_residual: bool = True,
    market_symbol: str = "BTCUSDT",
) -> dict[str, pd.Series]:
    """
    計算截面動量排名（改進版）

    改進點：
    1. 殘差動量：先扣除 BTC 市場因子
    2. 跳過近期：避免短期反轉
    3. 用 rolling sum 而非 rolling mean（更穩定）

    Args:
        universe_data: {symbol: DataFrame}
        lookback: 排名用的回看期（小時），預設 720 = 30 天
        skip_recent: 跳過最近 N 小時（避免反轉），預設 24 = 1 天
        use_residual: 是否使用殘差動量（預設 True）
        market_symbol: 市場因子幣種（預設 BTC）

    Returns:
        {symbol: rank_series}  rank ∈ [0, 1]，0 = 最弱，1 = 最強
    """
    if use_residual:
        ret_dict = _compute_residual_returns(universe_data, market_symbol)
    else:
        ret_dict = {}
        for sym, df in universe_data.items():
            ret_dict[sym] = df["close"].pct_change()

    ret_df = pd.DataFrame(ret_dict)

    # 動量分數 = rolling_sum(ret[t-lookback : t-skip_recent])
    # 跳過最近 skip_recent 根（短期反轉效應）
    if skip_recent > 0:
        # shift(skip_recent) 讓 rolling window 不包含最近 skip_recent 根
        momentum = ret_df.shift(skip_recent).rolling(lookback - skip_recent).sum()
    else:
        momentum = ret_df.rolling(lookback).sum()

    # 逐時間點 percentile rank
    rank_df = momentum.rank(axis=1, pct=True, na_option="keep")

    return {sym: rank_df[sym] for sym in rank_df.columns}


def _multi_lookback_rankings(
    universe_data: dict[str, pd.DataFrame],
    lookbacks: list[int],
    skip_recent: int = 24,
    use_residual: bool = True,
    market_symbol: str = "BTCUSDT",
) -> dict[str, pd.Series]:
    """
    多 lookback 集成排名

    對每個 lookback 計算排名，然後等權平均。
    這比單一 lookback 更穩健，避免了「剛好某個週期失效」的風險。
    """
    all_ranks = []
    for lb in lookbacks:
        ranks = _compute_xsmom_rankings(
            universe_data, lb, skip_recent, use_residual, market_symbol
        )
        ranks_df = pd.DataFrame(ranks)
        all_ranks.append(ranks_df)

    # 等權平均排名
    avg_rank_df = sum(all_ranks) / len(all_ranks)

    # 重新排名（平均排名可能不是 uniform 分佈）
    final_rank_df = avg_rank_df.rank(axis=1, pct=True, na_option="keep")

    return {sym: final_rank_df[sym] for sym in final_rank_df.columns}


# ══════════════════════════════════════════════════════════
#  策略 1: 改進版 XSMOM（殘差動量）
# ══════════════════════════════════════════════════════════

@register_strategy("xsmom")
def generate_xsmom(df: pd.DataFrame, ctx: StrategyContext, params: dict) -> pd.Series:
    """
    Cross-Sectional Momentum — 殘差動量排名

    改進版：
      1. 去除 BTC 市場因子 → 殘差動量（解決高相關性問題）
      2. 跳過最近 24h → 避免短期反轉效應
      3. 多 lookback 集成 → 更穩健
      4. 波動率目標縮放 → 控制曝險

    做多：排名 top（相對強勢，殘差為正）
    做空：排名 bottom（相對弱勢，殘差為負）

    params:
        universe:         幣種列表
        data_dir:         數據目錄
        lookbacks:        多 lookback 列表（預設 [336, 720, 1440] = 14d/30d/60d）
        skip_recent:      跳過最近 N 小時（預設 24 = 1 天）
        use_residual:     是否去除 BTC 因子（預設 True）
        market_symbol:    市場因子幣種（預設 BTCUSDT）
        long_threshold:   做多門檻（預設 0.7 = top 30%）
        short_threshold:  做空門檻（預設 0.3 = bottom 30%）
        vol_target:       年化波動率目標（預設 0.15）
        scale_mode:       "threshold" 二元 or "linear" 連續（預設 threshold）
    """
    universe = params.get("universe", [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
        "DOGEUSDT", "XRPUSDT", "AVAXUSDT", "LINKUSDT", "ADAUSDT",
    ])
    data_dir = params.get("data_dir", "data")

    lookbacks_raw = params.get("lookbacks", [336, 720, 1440])
    if isinstance(lookbacks_raw, str):
        lookbacks = [int(x) for x in lookbacks_raw.split(",")]
    else:
        lookbacks = [int(x) for x in lookbacks_raw]

    skip_recent = int(params.get("skip_recent", 24))
    use_residual = params.get("use_residual", True)
    market_symbol = params.get("market_symbol", "BTCUSDT")
    long_threshold = float(params.get("long_threshold", 0.7))
    short_threshold = float(params.get("short_threshold", 0.3))
    vol_target = float(params.get("vol_target", 0.15))
    scale_mode = params.get("scale_mode", "threshold")

    # 載入 universe 數據
    universe_data = _load_universe_data(
        universe, data_dir,
        interval=ctx.interval,
        market_type=ctx.market_type,
    )

    if len(universe_data) < 3:
        logger.warning(f"⚠️  XSMOM: universe 數據不足 ({len(universe_data)}/{len(universe)})")
        return pd.Series(0.0, index=df.index)

    # 多 lookback 集成排名
    rankings = _multi_lookback_rankings(
        universe_data, lookbacks, skip_recent, use_residual, market_symbol
    )

    # 取當前 symbol 的排名
    symbol = ctx.symbol
    if symbol not in rankings:
        logger.warning(f"⚠️  XSMOM: {symbol} 不在 universe 排名中")
        return pd.Series(0.0, index=df.index)

    rank = rankings[symbol].reindex(df.index)

    # 生成信號
    if scale_mode == "linear":
        # 連續信號：rank 0→-1, 0.5→0, 1→+1
        pos = (rank - 0.5) * 2.0
    else:
        # 二元信號：top → +1, bottom → -1, middle → 0
        pos = pd.Series(0.0, index=df.index)
        pos[rank >= long_threshold] = 1.0
        pos[rank <= short_threshold] = -1.0

    # 波動率目標縮放
    max_lb = max(lookbacks)
    close = df["close"]
    returns = close.pct_change()
    vol = returns.rolling(max_lb).std() * np.sqrt(8760)
    vol = vol.replace(0, np.nan).ffill().fillna(vol_target)
    scale = (vol_target / vol).clip(0.1, 2.0)
    pos = (pos * scale).clip(-1.0, 1.0).fillna(0.0)

    return pos


# ══════════════════════════════════════════════════════════
#  策略 2: XSMOM + TSMOM 組合
# ══════════════════════════════════════════════════════════

@register_strategy("xsmom_tsmom")
def generate_xsmom_tsmom(df: pd.DataFrame, ctx: StrategyContext, params: dict) -> pd.Series:
    """
    XSMOM + TSMOM 組合策略

    學術基礎：
        XSMOM 和 TSMOM 的相關性 ~0.3
        → 等權組合的 Sharpe ≈ (S1 + S2) / sqrt(2 + 2*corr)
        → 如果兩者 Sharpe 各 0.5, corr=0.3 → 組合 Sharpe ≈ 0.62

    params:
        xsmom_weight:     XSMOM 權重（預設 0.4）
        tsmom_weight:     TSMOM 權重（預設 0.6）
        tsmom_lookback:   TSMOM 專用回看期（預設 168 = 7 天）
        + XSMOM 的所有參數
        + EMA 對齊參數（可選）
    """
    xsmom_w = float(params.get("xsmom_weight", 0.4))
    tsmom_w = float(params.get("tsmom_weight", 0.6))
    vol_target = float(params.get("vol_target", 0.15))
    tsmom_lookback = int(params.get("tsmom_lookback", 168))

    # ── XSMOM 信號 ──
    xsmom_pos = generate_xsmom(df, ctx, params)

    # ── TSMOM 信號（用獨立 lookback，比 XSMOM 短）──
    from .tsmom_strategy import _tsmom_signal
    tsmom_pos = _tsmom_signal(df["close"], tsmom_lookback, vol_target)

    # ── EMA 趨勢對齊（可選）──
    ema_fast = params.get("ema_fast")
    if ema_fast is not None:
        ema_fast = int(ema_fast)
        ema_slow = int(params.get("ema_slow", 50))
        agree_weight = float(params.get("agree_weight", 1.0))
        disagree_weight = float(params.get("disagree_weight", 0.3))

        ema_f = calculate_ema(df["close"], ema_fast)
        ema_s = calculate_ema(df["close"], ema_slow)
        ema_trend = pd.Series(0.0, index=df.index)
        ema_trend[ema_f > ema_s] = 1.0
        ema_trend[ema_f < ema_s] = -1.0

        agree = np.sign(tsmom_pos) == np.sign(ema_trend)
        tsmom_pos = tsmom_pos.copy()
        tsmom_pos[agree] *= agree_weight
        tsmom_pos[~agree] *= disagree_weight

    # ── 加權組合 ──
    combined = xsmom_pos * xsmom_w + tsmom_pos * tsmom_w
    pos = combined.clip(-1.0, 1.0).fillna(0.0)

    return pos
