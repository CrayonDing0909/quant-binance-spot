"""
aggTrades 聚合指標模組 — VPIN / Real CVD / OFI

從 Binance Vision aggTrades（逐筆成交）計算高品質微結構指標，
取代先前從 taker_vol_ratio 和 1h OHLCV 近似的劣質 proxy。

核心指標：
    VPIN (Volume-Synchronized Probability of Informed Trading)
        - Easley, López de Prado & O'Hara (2012)
        - 用 volume clock 偵測 informed trading 密度
        - 高 VPIN = 高毒性 = 高風險 → 適合作為 regime indicator

    Real CVD (Cumulative Volume Delta)
        - 逐筆計算 buy_vol - sell_vol 的累積值
        - 比 taker_vol_ratio proxy 精確得多

    OFI (Order Flow Imbalance) — 簡化版
        - 按小時聚合的 signed trade flow
        - Cont, Kukanov & Stoikov (2014) 的簡化近似

處理流程：
    1. 逐月下載 aggTrades（binance_vision.download_single_month_aggtrades）
    2. 每月立即聚合到 hourly metrics + volume-clock bars
    3. 丟棄原始 tick data（節省記憶體）
    4. 從 volume-clock bars 計算 VPIN
    5. 從 hourly metrics 計算 Real CVD 和 OFI
    6. 儲存聚合結果到 parquet

儲存路徑：
    data/binance/futures/aggtrades_agg/{SYMBOL}_hourly.parquet
    data/binance/futures/aggtrades_agg/{SYMBOL}_vpin.parquet

使用方式：
    from qtrade.data.agg_trades import (
        download_and_process_aggtrades,
        load_hourly_metrics,
        load_vpin,
        compute_vpin_from_bars,
    )
"""
from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_BASE_DIR = Path("data/binance/futures/aggtrades_agg")


# ══════════════════════════════════════════════════════════════
#  Track A: Hourly Aggregation（buy/sell volume, num_trades, VWAP）
# ══════════════════════════════════════════════════════════════


def aggregate_trades_to_hourly(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    將原始 aggTrades 聚合為小時級指標

    Args:
        trades_df: 原始 aggTrades DataFrame
            index = transact_time (DatetimeIndex, UTC)
            columns = agg_trade_id, price, qty, ..., is_buyer_maker

    Returns:
        DataFrame (index=hourly DatetimeIndex) with columns:
            buy_volume:   taker buy 成交量（USDT quote）
            sell_volume:  taker sell 成交量（USDT quote）
            total_volume: buy + sell
            num_trades:   成交筆數
            vwap:         成交量加權平均價格
    """
    if trades_df.empty:
        return pd.DataFrame(
            columns=["buy_volume", "sell_volume", "total_volume", "num_trades", "vwap"],
            dtype=float,
        )

    # is_buyer_maker=True → taker 是賣方（sell-initiated）
    # is_buyer_maker=False → taker 是買方（buy-initiated）
    trade_quote_vol = trades_df["price"] * trades_df["qty"]  # USDT volume
    is_buy = ~trades_df["is_buyer_maker"]

    buy_vol = trade_quote_vol.where(is_buy, 0.0)
    sell_vol = trade_quote_vol.where(~is_buy, 0.0)

    hourly = pd.DataFrame(
        {
            "buy_volume": buy_vol.resample("1h").sum(),
            "sell_volume": sell_vol.resample("1h").sum(),
            "num_trades": trades_df["qty"].resample("1h").count(),
            "_sum_quote_vol": trade_quote_vol.resample("1h").sum(),
            "_sum_base_vol": trades_df["qty"].resample("1h").sum(),
        }
    )

    hourly["total_volume"] = hourly["buy_volume"] + hourly["sell_volume"]
    hourly["vwap"] = hourly["_sum_quote_vol"] / hourly["_sum_base_vol"].replace(0, np.nan)
    hourly = hourly.drop(columns=["_sum_quote_vol", "_sum_base_vol"])

    # 移除全零行（無交易的小時）
    hourly = hourly[hourly["total_volume"] > 0]

    return hourly


# ══════════════════════════════════════════════════════════════
#  Track B: Volume-Clock Bars + VPIN
# ══════════════════════════════════════════════════════════════


def compute_volume_clock_bars(
    trades_df: pd.DataFrame,
    bucket_vol: float,
) -> pd.DataFrame:
    """
    將 aggTrades 分割為等成交量的 volume-clock bars

    Easley et al. (2012) VPIN 的核心前置步驟：
    用 volume clock（等成交量桶）替代 time clock，使每個 bar 包含相同的信息量。

    Binance 提供 is_buyer_maker flag，可直接分類 buy/sell volume，
    比 BVC (Bulk Volume Classification) 更精確。

    Args:
        trades_df:  原始 aggTrades DataFrame (index=transact_time)
        bucket_vol: 每個桶的目標成交量（USDT quote volume）
                    建議：median_daily_volume / 50

    Returns:
        DataFrame (index=bucket_end_time) with columns:
            open, high, low, close:  桶內 OHLC 價格
            total_volume:            桶內總成交量（USDT）
            buy_volume:              taker buy 成交量
            sell_volume:             taker sell 成交量
            num_trades:              成交筆數
            order_imbalance:         |buy - sell| / total
            duration_seconds:        桶持續時間（秒）
    """
    if trades_df.empty or bucket_vol <= 0:
        return pd.DataFrame()

    prices = trades_df["price"].values
    qtys = trades_df["qty"].values
    is_buyer_maker = trades_df["is_buyer_maker"].values
    timestamps = trades_df.index.values

    # 計算每筆交易的 quote volume (USDT)
    trade_vol = prices * qtys
    cum_vol = np.cumsum(trade_vol)

    # 用 cumulative volume 分配 bucket ID
    # bucket_id = floor(cumulative_vol / bucket_vol)
    bucket_ids = (cum_vol / bucket_vol).astype(np.int64)

    # 找出每個 bucket 的邊界（bucket_id 變化的位置）
    changes = np.where(np.diff(bucket_ids) > 0)[0]
    # boundaries[i] 是第 i 個 bucket 的起始 trade index
    boundaries = np.concatenate([[0], changes + 1])

    if len(boundaries) < 2:
        return pd.DataFrame()

    bars: list[dict] = []
    for i in range(len(boundaries) - 1):
        s = boundaries[i]
        e = boundaries[i + 1]

        p_slice = prices[s:e]
        v_slice = trade_vol[s:e]
        bm_slice = is_buyer_maker[s:e]

        buy_v = float(v_slice[~bm_slice].sum())
        sell_v = float(v_slice[bm_slice].sum())
        total_v = buy_v + sell_v

        ts_start = timestamps[s]
        ts_end = timestamps[e - 1]
        duration_ns = int(ts_end) - int(ts_start)
        duration_sec = max(duration_ns / 1e9, 0.001)

        bars.append(
            {
                "bucket_start": ts_start,
                "bucket_end": ts_end,
                "open": float(p_slice[0]),
                "high": float(p_slice.max()),
                "low": float(p_slice.min()),
                "close": float(p_slice[-1]),
                "total_volume": total_v,
                "buy_volume": buy_v,
                "sell_volume": sell_v,
                "num_trades": int(e - s),
                "order_imbalance": abs(buy_v - sell_v) / total_v if total_v > 0 else 0.0,
                "duration_seconds": duration_sec,
            }
        )

    # 處理最後一個不完整的 bucket（只加如果有足夠 volume）
    last_s = boundaries[-1]
    if last_s < len(prices):
        p_slice = prices[last_s:]
        v_slice = trade_vol[last_s:]
        bm_slice = is_buyer_maker[last_s:]
        remaining_vol = float(v_slice.sum())
        # 只有 >= 50% bucket_vol 才納入
        if remaining_vol >= bucket_vol * 0.5:
            buy_v = float(v_slice[~bm_slice].sum())
            sell_v = float(v_slice[bm_slice].sum())
            ts_end = timestamps[-1]
            ts_start = timestamps[last_s]
            duration_ns = int(ts_end) - int(ts_start)
            duration_sec = max(duration_ns / 1e9, 0.001)
            bars.append(
                {
                    "bucket_start": ts_start,
                    "bucket_end": ts_end,
                    "open": float(p_slice[0]),
                    "high": float(p_slice.max()),
                    "low": float(p_slice.min()),
                    "close": float(p_slice[-1]),
                    "total_volume": remaining_vol,
                    "buy_volume": buy_v,
                    "sell_volume": sell_v,
                    "num_trades": int(len(p_slice)),
                    "order_imbalance": abs(buy_v - sell_v) / remaining_vol
                    if remaining_vol > 0
                    else 0.0,
                    "duration_seconds": duration_sec,
                }
            )

    if not bars:
        return pd.DataFrame()

    result = pd.DataFrame(bars)
    result["bucket_end"] = pd.DatetimeIndex(result["bucket_end"])
    result = result.set_index("bucket_end")
    return result


def compute_vpin_from_bars(
    volume_bars: pd.DataFrame,
    n_buckets: int = 50,
) -> pd.Series:
    """
    從 volume-clock bars 計算 VPIN

    VPIN = rolling mean of order_imbalance over last n_buckets

    VPIN 值域 [0, 1]:
        ~0.0: 買賣平衡，低毒性
        ~0.5: 高度不平衡，知情交易者活躍
        >0.7: 極端毒性（如 2010 Flash Crash 前）

    Args:
        volume_bars: compute_volume_clock_bars() 的輸出
        n_buckets:   rolling window 大小（Easley 2012 建議 50）

    Returns:
        pd.Series: VPIN 序列，index = bucket_end timestamp
    """
    if volume_bars.empty or "order_imbalance" not in volume_bars.columns:
        return pd.Series(dtype=float, name="vpin")

    vpin = (
        volume_bars["order_imbalance"]
        .rolling(n_buckets, min_periods=n_buckets)
        .mean()
    )
    vpin.name = "vpin"
    return vpin.dropna()


def resample_vpin_to_time(
    vpin: pd.Series,
    freq: str = "1h",
) -> pd.Series:
    """
    將 volume-clock VPIN 重採樣到 time-clock

    因為 volume-clock bars 不等時距，需要 resample 到固定 freq。
    使用 last() 取每個時間窗口內最新的 VPIN 值。

    Args:
        vpin:  compute_vpin_from_bars() 的輸出
        freq:  目標頻率（"1h", "1D" 等）

    Returns:
        time-clock VPIN Series
    """
    if vpin.empty:
        return pd.Series(dtype=float, name="vpin")

    resampled = vpin.resample(freq).last()
    # ffill 填充無交易的時段（最多 forward-fill 24 bars）
    resampled = resampled.ffill(limit=24)
    resampled.name = "vpin"
    return resampled.dropna()


# ══════════════════════════════════════════════════════════════
#  Real CVD + OFI（從 hourly metrics 計算）
# ══════════════════════════════════════════════════════════════


def compute_real_cvd(hourly_df: pd.DataFrame) -> pd.Series:
    """
    從 hourly buy/sell volume 計算 Real CVD

    CVD = cumsum(buy_volume - sell_volume)

    與 taker_volume.py 的 proxy CVD 不同：
    - proxy 用 ratio 近似：delta = (ratio - 1) / (ratio + 1)
    - real CVD 用逐筆成交的買賣量差值

    Args:
        hourly_df: aggregate_trades_to_hourly() 的輸出

    Returns:
        pd.Series: CVD 累積序列
    """
    if hourly_df.empty:
        return pd.Series(dtype=float, name="real_cvd")

    delta = hourly_df["buy_volume"] - hourly_df["sell_volume"]
    cvd = delta.cumsum()
    cvd.name = "real_cvd"
    return cvd


def compute_ofi_hourly(hourly_df: pd.DataFrame) -> pd.Series:
    """
    從 hourly metrics 計算 Order Flow Imbalance (OFI)

    OFI = (buy_volume - sell_volume) / total_volume

    這是 Cont et al. (2014) OFI 的簡化近似版。
    真正的 OFI 需要 order book 變化數據，此處用 taker flow 代替。

    值域 [-1, 1]:
        +1: 全部是 taker buy
        -1: 全部是 taker sell
         0: 買賣均衡

    Args:
        hourly_df: aggregate_trades_to_hourly() 的輸出

    Returns:
        pd.Series: OFI 序列
    """
    if hourly_df.empty:
        return pd.Series(dtype=float, name="ofi")

    total = hourly_df["total_volume"].replace(0, np.nan)
    ofi = (hourly_df["buy_volume"] - hourly_df["sell_volume"]) / total
    ofi.name = "ofi"
    return ofi.fillna(0.0)


# ══════════════════════════════════════════════════════════════
#  高級 API：下載 + 處理 + 儲存
# ══════════════════════════════════════════════════════════════


def calibrate_bucket_vol(
    trades_df: pd.DataFrame,
    buckets_per_day: int = 50,
) -> float:
    """
    自動校準 volume-clock bucket 大小

    使用 median daily volume / buckets_per_day。
    Easley (2012) 建議 ~50 buckets/day。

    Args:
        trades_df: 至少一個月的 aggTrades 數據
        buckets_per_day: 每天目標桶數

    Returns:
        bucket_vol (USDT)
    """
    trade_vol = trades_df["price"] * trades_df["qty"]
    daily_vol = trade_vol.resample("1D").sum()
    median_daily = daily_vol.median()

    if median_daily <= 0 or np.isnan(median_daily):
        logger.warning("⚠️  Daily volume median is 0 or NaN, using fallback 1e8")
        return 1e8 / buckets_per_day

    bucket_vol = median_daily / buckets_per_day
    logger.info(
        f"📊 Bucket calibration: median daily vol = ${median_daily:,.0f}, "
        f"bucket_vol = ${bucket_vol:,.0f} ({buckets_per_day}/day)"
    )
    return bucket_vol


def download_and_process_aggtrades(
    symbol: str,
    start: str,
    end: str,
    data_dir: Path = _BASE_DIR,
    buckets_per_day: int = 50,
    n_vpin_buckets: int = 50,
) -> dict[str, pd.DataFrame | pd.Series]:
    """
    下載 aggTrades 並逐月處理，計算 VPIN + hourly metrics

    處理流程：
        1. 逐月下載 aggTrades
        2. 每月聚合到 hourly metrics（Track A）
        3. 每月計算 volume-clock bars（Track B）
        4. 釋放原始 tick data
        5. 合併所有月份
        6. 從 volume-clock bars 計算 VPIN
        7. 從 hourly metrics 計算 Real CVD + OFI
        8. 儲存到 parquet

    Args:
        symbol:           交易對，如 "BTCUSDT"
        start:            開始日期 "YYYY-MM-DD"
        end:              結束日期 "YYYY-MM-DD"
        data_dir:         儲存路徑
        buckets_per_day:  每天目標 volume-clock 桶數（預設 50）
        n_vpin_buckets:   VPIN rolling window 大小（預設 50）

    Returns:
        dict with keys:
            "hourly":     pd.DataFrame — 小時級 buy/sell volume metrics
            "vpin_1h":    pd.Series   — 1h VPIN（從 volume-clock resample）
            "vpin_daily": pd.Series   — daily VPIN
            "real_cvd":   pd.Series   — 真實 CVD
            "ofi":        pd.Series   — hourly OFI
    """
    import requests

    from qtrade.data.binance_vision import (
        download_single_month_aggtrades,
        generate_monthly_aggtrades_urls,
    )

    urls = generate_monthly_aggtrades_urls(symbol, start, end)
    if not urls:
        logger.warning(f"⚠️  {symbol}: 沒有需要下載的月份")
        return {}

    logger.info(
        f"🚀 aggTrades {symbol}: 下載 {len(urls)} 個月 ({start} → {end})"
    )

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    all_hourly: list[pd.DataFrame] = []
    all_vclock: list[pd.DataFrame] = []
    bucket_vol: float | None = None

    session = requests.Session()

    for i, (url, year_month) in enumerate(urls):
        logger.info(f"📥 [{i + 1}/{len(urls)}] {symbol} {year_month}...")

        trades = download_single_month_aggtrades(symbol, year_month, session)
        if trades is None or trades.empty:
            continue

        # ── Track A: hourly aggregation ──
        hourly = aggregate_trades_to_hourly(trades)
        all_hourly.append(hourly)

        # ── Track B: volume-clock bars ──
        if bucket_vol is None:
            # 用第一個月數據校準 bucket size
            bucket_vol = calibrate_bucket_vol(trades, buckets_per_day)

        vclock = compute_volume_clock_bars(trades, bucket_vol)
        if not vclock.empty:
            all_vclock.append(vclock)

        # 釋放原始 tick data
        del trades
        gc.collect()

    session.close()

    if not all_hourly:
        logger.warning(f"⚠️  {symbol}: 沒有下載到任何數據")
        return {}

    # ── 合併所有月份 ──
    hourly_df = pd.concat(all_hourly).sort_index()
    hourly_df = hourly_df[~hourly_df.index.duplicated(keep="last")]

    # ── VPIN ──
    vpin_1h = pd.Series(dtype=float, name="vpin")
    vpin_daily = pd.Series(dtype=float, name="vpin")

    if all_vclock:
        vclock_df = pd.concat(all_vclock).sort_index()
        vclock_df = vclock_df[~vclock_df.index.duplicated(keep="last")]

        vpin_raw = compute_vpin_from_bars(vclock_df, n_vpin_buckets)
        if not vpin_raw.empty:
            vpin_1h = resample_vpin_to_time(vpin_raw, "1h")
            vpin_daily = resample_vpin_to_time(vpin_raw, "1D")

        del vclock_df
        gc.collect()

    # ── Real CVD + OFI ──
    real_cvd = compute_real_cvd(hourly_df)
    ofi = compute_ofi_hourly(hourly_df)

    # ── 儲存 ──
    hourly_path = data_dir / f"{symbol}_hourly.parquet"
    hourly_df.to_parquet(hourly_path)
    logger.info(f"💾 Hourly metrics saved: {hourly_path} ({len(hourly_df)} rows)")

    if not vpin_1h.empty:
        vpin_path = data_dir / f"{symbol}_vpin.parquet"
        vpin_combined = pd.DataFrame(
            {"vpin_1h": vpin_1h, "vpin_daily": vpin_daily}
        )
        # 分別儲存 1h 和 daily
        vpin_1h.to_frame().to_parquet(data_dir / f"{symbol}_vpin_1h.parquet")
        vpin_daily.to_frame().to_parquet(data_dir / f"{symbol}_vpin_daily.parquet")
        logger.info(
            f"💾 VPIN saved: {data_dir / symbol}_vpin_*.parquet "
            f"(1h: {len(vpin_1h)}, daily: {len(vpin_daily)} rows)"
        )

    # CVD + OFI
    real_cvd.to_frame().to_parquet(data_dir / f"{symbol}_cvd.parquet")
    ofi.to_frame().to_parquet(data_dir / f"{symbol}_ofi.parquet")
    logger.info(
        f"💾 CVD+OFI saved: {data_dir / symbol}_cvd/ofi.parquet "
        f"({len(real_cvd)} rows)"
    )

    return {
        "hourly": hourly_df,
        "vpin_1h": vpin_1h,
        "vpin_daily": vpin_daily,
        "real_cvd": real_cvd,
        "ofi": ofi,
    }


# ══════════════════════════════════════════════════════════════
#  載入 / 對齊 helpers
# ══════════════════════════════════════════════════════════════


def load_hourly_metrics(
    symbol: str,
    data_dir: Path = _BASE_DIR,
) -> Optional[pd.DataFrame]:
    """載入已處理的 hourly aggTrades metrics"""
    path = data_dir / f"{symbol}_hourly.parquet"
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception as e:
        logger.warning(f"⚠️  Load hourly metrics failed ({symbol}): {e}")
        return None


def load_vpin(
    symbol: str,
    freq: str = "1h",
    data_dir: Path = _BASE_DIR,
) -> Optional[pd.Series]:
    """
    載入 VPIN 序列

    Args:
        symbol: 交易對
        freq:   "1h" 或 "daily"
        data_dir: 資料目錄

    Returns:
        VPIN Series，或 None（檔案不存在）
    """
    suffix = "1h" if freq in ("1h", "1H", "hourly") else "daily"
    path = data_dir / f"{symbol}_vpin_{suffix}.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        col = "vpin" if "vpin" in df.columns else df.columns[0]
        series = df[col]
        series.name = "vpin"
        return series
    except Exception as e:
        logger.warning(f"⚠️  Load VPIN failed ({symbol}): {e}")
        return None


def load_real_cvd(
    symbol: str,
    data_dir: Path = _BASE_DIR,
) -> Optional[pd.Series]:
    """載入 Real CVD"""
    path = data_dir / f"{symbol}_cvd.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        col = "real_cvd" if "real_cvd" in df.columns else df.columns[0]
        return df[col]
    except Exception as e:
        logger.warning(f"⚠️  Load CVD failed ({symbol}): {e}")
        return None


def load_ofi(
    symbol: str,
    data_dir: Path = _BASE_DIR,
) -> Optional[pd.Series]:
    """載入 OFI"""
    path = data_dir / f"{symbol}_ofi.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        col = "ofi" if "ofi" in df.columns else df.columns[0]
        return df[col]
    except Exception as e:
        logger.warning(f"⚠️  Load OFI failed ({symbol}): {e}")
        return None


def load_avg_trade_size(
    symbol: str,
    data_dir: Path = _BASE_DIR,
) -> Optional[pd.Series]:
    """
    載入 avg_trade_size (= total_volume / num_trades) 從 hourly aggTrades metrics

    avg_trade_size 反映鯨魚交易行為：
      - 高 avg_trade_size → 大戶在分銷 → 未來收益偏低 (IC=-0.030)
      - 低 avg_trade_size → 散戶主導 → 正常
      - corr(TSMOM)=0.04, corr(ATR)=0.25 — 幾乎完全正交

    Returns:
        avg_trade_size Series (hourly)，或 None
    """
    hourly = load_hourly_metrics(symbol, data_dir)
    if hourly is None:
        return None

    if "total_volume" not in hourly.columns or "num_trades" not in hourly.columns:
        logger.warning(f"⚠️  {symbol}: hourly metrics 缺少 total_volume 或 num_trades")
        return None

    ats = hourly["total_volume"] / hourly["num_trades"].replace(0, np.nan)
    ats.name = "avg_trade_size"
    return ats


def align_avg_trade_size_to_klines(
    ats: pd.Series | None,
    kline_index: pd.DatetimeIndex,
    max_ffill_bars: int = 4,
) -> pd.Series | None:
    """
    將 avg_trade_size 對齊到 K 線時間軸

    注意：avg_trade_size 是從 aggTrades 聚合的外部數據（有獨立時間戳），
    reindex+ffill 不涉及 intra-bar look-ahead 問題。
    """
    if ats is None or ats.empty:
        return None

    s = ats.copy()
    if kline_index.tz is None and s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    elif kline_index.tz is not None and s.index.tz is None:
        s.index = s.index.tz_localize(kline_index.tz)

    aligned = s.reindex(kline_index, method="ffill", limit=max_ffill_bars)

    n_missing = aligned.isna().sum()
    if n_missing > 0:
        coverage = (len(aligned) - n_missing) / len(aligned) * 100
        logger.info(f"📊 avg_trade_size alignment: {coverage:.1f}% coverage")

    return aligned


def align_vpin_to_klines(
    vpin: pd.Series | None,
    kline_index: pd.DatetimeIndex,
    max_ffill_bars: int = 4,
) -> pd.Series | None:
    """
    將 VPIN 對齊到 K 線時間軸

    注意：VPIN 是外部 aggregated 數據（有獨立的時間戳），
    不涉及 intra-bar look-ahead 問題（同 FR、OI、LSR 的 reindex 是安全的）。
    """
    if vpin is None or vpin.empty:
        return None

    s = vpin.copy()
    if kline_index.tz is None and s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    elif kline_index.tz is not None and s.index.tz is None:
        s.index = s.index.tz_localize(kline_index.tz)

    aligned = s.reindex(kline_index, method="ffill", limit=max_ffill_bars)

    n_missing = aligned.isna().sum()
    if n_missing > 0:
        coverage = (len(aligned) - n_missing) / len(aligned) * 100
        logger.info(f"📊 VPIN alignment: {coverage:.1f}% coverage")

    return aligned
