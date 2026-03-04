"""
Orderflow Composite 獨立策略

文獻基礎：
    - Cont, Kukanov & Stoikov (2014): OFI — 方向信號
    - Easley, López de Prado & O'Hara (2012): VPIN — 信心/regime 信號
    - Kyle (1985): informed trading → temporary impact → reversion
    - 標準 CVD: 趨勢確認

核心特點：
    1. 這是一個**獨立策略**，不是 TSMOM 的 filter 或 overlay
    2. 使用微結構數據（taker buy/sell flow）而非價格動量
    3. 設計上與 TSMOM 低相關（看的是「誰在交易」而非「價格走了多遠」）
    4. 透過 meta_blend 或獨立 Runner 在組合層級與 TSMOM 互補

信號邏輯（contrarian / mean-reversion）：
    在 1h 頻率，OFI 是**逆向指標**：
    - 高 taker buy → 散戶 FOMO 追漲 → 價格已反映 → 預期回落 → SHORT
    - 高 taker sell → 散戶恐慌拋售 → 價格已反映 → 預期反彈 → LONG
    這在微結構理論中有充分支撐（Kyle 1985 的 temporary price impact + reversion）

信號組件：
    1. OFI Signal (contrarian): -smoothed_OFI → fade the flow
    2. VPIN Confidence: 知情交易者活躍度 → 更大反轉 → 更大倉位
    3. CVD Confirmation: CVD 趨勢確認 → 同向加強

與 TSMOM 的結構性差異：
    - TSMOM 看 168h/720h 價格累積收益 → 歷史價格動量 → 追趨勢
    - Orderflow 看 6h-72h taker flow → fade 散戶流 → 逆向交易
    - 低相關性：TSMOM 追漲時 orderflow 可能在 fade 同一波

數據來源：
    taker_vol_ratio（Binance Vision 衍生品數據，2021-12 至今）
    → OFI = (ratio - 1) / (ratio + 1)  ≡ (buy_vol - sell_vol) / total_vol
    → CVD = cumsum(OFI)
    → VPIN_proxy = rolling_mean(|OFI|, N)

Note:
    signal_delay 和 direction clip 由 @register_strategy 框架自動處理，
    策略函數只需回傳 raw position [-1, 1]。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..indicators import calculate_ema
from . import register_strategy
from .base import StrategyContext

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
#  核心信號組件
# ══════════════════════════════════════════════════════════════


def compute_ofi_from_taker_ratio(taker_vol_ratio: pd.Series) -> pd.Series:
    """
    從 taker buy/sell volume ratio 計算 OFI

    數學上等價於 (buy_vol - sell_vol) / (buy_vol + sell_vol)
    值域 [-1, 1]：+1 = 全部 taker buy, -1 = 全部 taker sell
    """
    ratio = taker_vol_ratio.copy()
    ofi = (ratio - 1.0) / (ratio + 1.0)
    return ofi.fillna(0.0).clip(-1.0, 1.0)


def _ofi_direction_signal(
    ofi: pd.Series,
    ema_span: int = 12,
    lookback: int = 24,
    tanh_scale: float = 2.0,
) -> pd.Series:
    """
    OFI 方向信號

    使用 EMA-smoothed 累積 OFI 的 z-score → tanh 壓縮到 [-1, 1]。
    累積窗口捕捉「持續買入/賣出壓力」，而非單 bar 雜訊。

    Args:
        ofi: 原始 OFI 序列 [-1, 1]
        ema_span: OFI 先做 EMA 平滑的週期
        lookback: 累積 OFI 的窗口（小時）
        tanh_scale: z-score → tanh 的縮放因子
    """
    # 先 EMA 平滑去除 tick noise
    smoothed = ofi.ewm(span=ema_span, adjust=False).mean()

    # 累積窗口：捕捉持續性的 flow imbalance
    cumulative = smoothed.rolling(lookback, min_periods=lookback // 2).sum()

    # 標準化
    vol = smoothed.rolling(lookback, min_periods=lookback // 2).std().clip(lower=1e-8)
    z = cumulative / vol / (lookback ** 0.5)  # scale-free z-score

    # tanh 壓縮到 [-1, 1]
    signal = np.tanh(z * tanh_scale)
    return signal.fillna(0.0)


def _vpin_proxy_pctrank(
    ofi: pd.Series,
    window: int = 50,
    lookback: int = 720,
) -> pd.Series:
    """
    VPIN proxy: rolling mean of |OFI| 的百分位排名

    概念：|OFI| 大 = 買賣嚴重失衡 = 知情交易者活躍 = 信號更可靠
    用百分位排名而非絕對值，避免不同時期 vol regime 干擾。

    Args:
        ofi: 原始 OFI 序列
        window: VPIN rolling 平均窗口
        lookback: 百分位排名的回看期
    """
    abs_ofi = ofi.abs()
    vpin = abs_ofi.rolling(window, min_periods=window // 2).mean()
    pctrank = vpin.rolling(lookback, min_periods=lookback // 4).rank(pct=True)
    return pctrank.fillna(0.5)


def _cvd_trend_signal(
    ofi: pd.Series,
    lookback: int = 72,
    ema_span: int = 24,
) -> pd.Series:
    """
    CVD 趨勢信號

    CVD = cumsum(OFI)：累積買賣力量差。
    CVD 趨勢 = sign(CVD - EMA(CVD))：CVD 在均值之上/下。

    用途：當 CVD 趨勢與 OFI 方向一致時，信號更可靠。
    """
    cvd = ofi.cumsum()
    cvd_ema = calculate_ema(cvd, ema_span)
    cvd_trend = np.sign(cvd - cvd_ema)
    return cvd_trend.fillna(0.0)


# ══════════════════════════════════════════════════════════════
#  Turnover 控制
# ══════════════════════════════════════════════════════════════


def _apply_threshold_positions(
    composite: pd.Series,
    entry_threshold: float = 0.25,
    exit_threshold: float = 0.10,
    ema_span: int = 24,
) -> pd.Series:
    """
    Threshold-based positions with hysteresis.

    遠比連續持倉量化更適合低頻微結構策略：
    - 只在信號強度超過閾值時進場 → 更高確信度
    - 只在信號弱於退出閾值時出場 → 避免頻繁震盪
    - hysteresis 帶（exit < signal < entry）→ 持倉不變 → 大幅降低 turnover

    Args:
        composite: 原始組合信號 [-1, 1]
        entry_threshold: 進場信號強度（|composite| > threshold → 建倉）
        exit_threshold: 出場信號強度（|composite| < threshold → 平倉）
        ema_span: 信號先做 EMA 平滑
    """
    # EMA smoothing first
    if ema_span > 0:
        sig = composite.ewm(span=ema_span, adjust=False).mean()
    else:
        sig = composite.copy()

    # Threshold crossings with hysteresis
    event = pd.Series(np.nan, index=sig.index)
    event[sig > entry_threshold] = 1.0      # strong bullish → long
    event[sig < -entry_threshold] = -1.0    # strong bearish → short
    event[sig.abs() < exit_threshold] = 0.0  # weak signal → flat

    # Forward-fill: between entry and exit, hold position (hysteresis)
    pos = event.ffill().fillna(0.0)

    return pos


# ══════════════════════════════════════════════════════════════
#  數據載入
# ══════════════════════════════════════════════════════════════


def _load_and_align_taker_vol(
    symbol: str,
    kline_index: pd.DatetimeIndex,
    data_dir: str | Path = "data",
) -> Optional[pd.Series]:
    """載入並對齊 taker volume ratio 到 K 線時間軸"""
    from qtrade.data.taker_volume import align_taker_to_klines, load_taker_volume

    derivatives_dir = Path(data_dir) / "binance" / "futures" / "derivatives"
    taker = load_taker_volume(symbol, data_dir=derivatives_dir)
    if taker is None or taker.empty:
        logger.warning(f"⚠️ {symbol}: taker_vol_ratio 數據不存在 ({derivatives_dir})")
        return None

    aligned = align_taker_to_klines(taker, kline_index, max_ffill_bars=4)
    if aligned is None:
        return None

    coverage = 1.0 - aligned.isna().mean()
    if coverage < 0.10:
        logger.warning(
            f"⚠️ {symbol}: taker_vol_ratio coverage {coverage:.1%} < 10%, "
            "returning None"
        )
        return None

    logger.info(f"📊 {symbol}: taker_vol_ratio loaded, coverage={coverage:.1%}")
    return aligned


# ══════════════════════════════════════════════════════════════
#  策略主入口
# ══════════════════════════════════════════════════════════════


@register_strategy("orderflow_composite")
def generate_orderflow_composite(
    df: pd.DataFrame, ctx: StrategyContext, params: dict
) -> pd.Series:
    """
    Orderflow Composite 獨立策略

    依靠 taker buy/sell flow 的微結構信號生成倉位，
    結構上與 TSMOM 正交（看的維度不同）。

    Parameters:
        # ── Signal Mode ──
        contrarian: bool = True          True=逆向(fade the flow), False=動量(follow the flow)

        # ── OFI Direction ──
        ofi_ema_span: int = 12           OFI 平滑 EMA
        ofi_lookback: int = 24           OFI 累積窗口（h）
        ofi_tanh_scale: float = 2.0      z-score → tanh 縮放

        # ── VPIN Confidence ──
        vpin_window: int = 50            VPIN rolling 窗口
        vpin_lookback: int = 720         VPIN 百分位回看期
        vpin_min_scale: float = 0.3      最低 VPIN 倉位縮放

        # ── CVD Confirmation ──
        cvd_lookback: int = 72           CVD 趨勢回看期
        cvd_ema_span: int = 24           CVD EMA 平滑
        cvd_agree_scale: float = 1.0     CVD 與 OFI 同向時的縮放
        cvd_disagree_scale: float = 0.5  CVD 與 OFI 反向時的縮放
        cvd_neutral_scale: float = 0.7   CVD 或 OFI 為零時的縮放

        # ── Threshold Positions (low turnover) ──
        entry_threshold: float = 0.25    進場閾值（|signal| > entry → 建倉）
        exit_threshold: float = 0.10     出場閾值（|signal| < exit → 平倉）
        composite_ema: int = 24          信號 EMA 平滑週期

    Returns:
        pd.Series: 倉位信號 [-1, 1]
    """
    data_dir = params.get("_data_dir", "data")
    contrarian = params.get("contrarian", True)  # default: contrarian mode

    # ── 載入 taker volume 數據 ──
    taker = _load_and_align_taker_vol(ctx.symbol, df.index, data_dir)
    if taker is None:
        logger.warning(
            f"⚠️ {ctx.symbol}: No taker vol data available, "
            "returning 0 positions (no trades)"
        )
        return pd.Series(0.0, index=df.index)

    # ── 計算 OFI ──
    ofi = compute_ofi_from_taker_ratio(taker)

    # ── Signal 1: OFI Direction ──
    ofi_signal = _ofi_direction_signal(
        ofi,
        ema_span=params.get("ofi_ema_span", 12),
        lookback=params.get("ofi_lookback", 24),
        tanh_scale=params.get("ofi_tanh_scale", 2.0),
    )

    # Contrarian mode: fade the flow
    # 在 1h 頻率，taker flow 是反向指標（散戶 FOMO/恐慌）
    if contrarian:
        ofi_signal = -ofi_signal

    # ── Signal 2: VPIN Confidence (position sizing) ──
    vpin_min_scale = params.get("vpin_min_scale", 0.3)
    vpin_pctrank = _vpin_proxy_pctrank(
        ofi,
        window=params.get("vpin_window", 50),
        lookback=params.get("vpin_lookback", 720),
    )
    # High VPIN → informed trading active → larger reversal expected → bigger position
    # Low VPIN → noise → smaller position
    vpin_scale = vpin_min_scale + (1.0 - vpin_min_scale) * vpin_pctrank

    # ── Signal 3: CVD Confirmation ──
    cvd_agree = params.get("cvd_agree_scale", 1.0)
    cvd_disagree = params.get("cvd_disagree_scale", 0.5)
    cvd_neutral = params.get("cvd_neutral_scale", 0.7)

    cvd_trend = _cvd_trend_signal(
        ofi,
        lookback=params.get("cvd_lookback", 72),
        ema_span=params.get("cvd_ema_span", 24),
    )

    # In contrarian mode, CVD trend is also inverted for confirmation logic
    if contrarian:
        cvd_trend = -cvd_trend

    ofi_dir = np.sign(ofi_signal)
    agreement = ofi_dir * cvd_trend

    cvd_scale = pd.Series(cvd_neutral, index=df.index)
    cvd_scale[agreement > 0] = cvd_agree      # CVD confirms OFI direction
    cvd_scale[agreement < 0] = cvd_disagree    # CVD opposes OFI direction

    # ── Composite Position ──
    raw_pos = ofi_signal * vpin_scale * cvd_scale

    # ── Threshold-based positions (low turnover) ──
    result = _apply_threshold_positions(
        raw_pos,
        entry_threshold=params.get("entry_threshold", 0.25),
        exit_threshold=params.get("exit_threshold", 0.10),
        ema_span=params.get("composite_ema", 24),
    )

    return result
