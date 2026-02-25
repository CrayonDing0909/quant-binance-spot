"""
清算/爆倉數據下載工具 (Phase 0B)

從 Binance Futures API 下載清算數據：
    1. Force Orders (近期強制平倉訂單)
    2. Aggregate Liquidation Volume (從 Coinglass，如有 API key)

數據來源：
    A. Binance API: GET /fapi/v1/allForceOrders — 最近清算訂單
       限制: 不需 API key, 但只有最近 ~7 天, 最多 1000 筆
    B. Coinglass API: /api/futures/liquidation/v2/history — 歷史清算數據
       需要 COINGLASS_API_KEY, 支援更長歷史

衍生指標：
    - liq_volume_long:  做多清算量 (USDT)
    - liq_volume_short: 做空清算量 (USDT)
    - liq_imbalance:    清算不平衡 = (long_liq - short_liq) / (long_liq + short_liq)
    - liq_cascade:      清算瀑布指標 = rolling z-score of total liq volume

儲存路徑：
    data/binance/futures/liquidation/{SYMBOL}.parquet

使用範例：
    # 從 Binance 下載最近清算數據
    PYTHONPATH=src python scripts/fetch_liquidation_data.py --symbols BTCUSDT ETHUSDT

    # 從 Coinglass 下載歷史清算（需要 API key）
    PYTHONPATH=src python scripts/fetch_liquidation_data.py --symbols BTCUSDT --source coinglass

    # 查看覆蓋率報告
    PYTHONPATH=src python scripts/fetch_liquidation_data.py --symbols BTCUSDT --coverage
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/binance/futures/liquidation")


# ══════════════════════════════════════════════════════════════
#  Binance Force Orders
# ══════════════════════════════════════════════════════════════

def fetch_binance_force_orders(
    symbol: str,
    limit: int = 1000,
) -> pd.DataFrame:
    """
    從 Binance /fapi/v1/allForceOrders 下載最近清算訂單

    回傳 DataFrame 欄位:
        timestamp, symbol, side, price, qty, quote_qty, time_in_force
    """
    from qtrade.data.binance_futures_client import BinanceFuturesHTTP

    client = BinanceFuturesHTTP()
    try:
        records = client.get("/fapi/v1/allForceOrders", {
            "symbol": symbol,
            "limit": min(limit, 1000),
        })
    except Exception as e:
        logger.error(f"❌ Binance force orders {symbol}: {e}")
        return pd.DataFrame()

    if not records:
        logger.warning(f"⚠️  No force orders for {symbol}")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["origQty"] = pd.to_numeric(df["origQty"], errors="coerce")
    df["executedQty"] = pd.to_numeric(df["executedQty"], errors="coerce")
    df["averagePrice"] = pd.to_numeric(df["averagePrice"], errors="coerce")

    # 計算清算金額 (USDT)
    df["quote_qty"] = df["executedQty"] * df["averagePrice"]

    result = df[["timestamp", "symbol", "side", "averagePrice", "executedQty", "quote_qty"]].copy()
    result.columns = ["timestamp", "symbol", "side", "price", "qty", "quote_qty"]
    result = result.set_index("timestamp").sort_index()

    logger.info(
        f"✅ Binance force orders {symbol}: {len(result)} orders "
        f"({result.index[0]:%Y-%m-%d %H:%M} → {result.index[-1]:%Y-%m-%d %H:%M})"
    )
    return result


# ══════════════════════════════════════════════════════════════
#  Coinglass Liquidation History
# ══════════════════════════════════════════════════════════════

def fetch_coinglass_liquidation(
    symbol: str,
    interval: str = "1h",
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """
    從 Coinglass API 下載歷史清算聚合數據

    Returns:
        DataFrame with columns: liq_volume_long, liq_volume_short
    """
    import requests

    api_key = os.getenv("COINGLASS_API_KEY", "")
    if not api_key:
        logger.warning("⚠️  COINGLASS_API_KEY not set. Cannot fetch liquidation data.")
        return pd.DataFrame()

    cg_symbol_map = {
        "BTCUSDT": "BTC", "ETHUSDT": "ETH", "SOLUSDT": "SOL",
        "BNBUSDT": "BNB", "DOGEUSDT": "DOGE", "ADAUSDT": "ADA",
        "AVAXUSDT": "AVAX", "LINKUSDT": "LINK", "XRPUSDT": "XRP",
    }
    cg_symbol = cg_symbol_map.get(symbol, symbol.replace("USDT", ""))

    interval_map = {
        "1h": "h1", "2h": "h2", "4h": "h4", "12h": "h12", "1d": "1d",
    }
    cg_interval = interval_map.get(interval, "h1")

    headers = {
        "accept": "application/json",
        "CoinGlass-API-Key": api_key,
    }

    # 分頁下載
    if start:
        start_ts = int(pd.Timestamp(start, tz="UTC").timestamp())
    else:
        start_ts = int(pd.Timestamp("2022-01-01", tz="UTC").timestamp())
    if end:
        end_ts = int(pd.Timestamp(end, tz="UTC").timestamp())
    else:
        end_ts = int(pd.Timestamp.now(tz="UTC").timestamp())

    all_records = []
    current_end = end_ts
    page = 0

    logger.info(f"📥 Coinglass liquidation: {symbol} ({cg_symbol}) {interval}")

    while page < 100:
        params = {
            "symbol": cg_symbol,
            "timeType": cg_interval,
            "endTime": current_end,
            "limit": 500,
        }

        try:
            resp = requests.get(
                "https://open-api-v3.coinglass.com/api/futures/liquidation/v2/history",
                params=params,
                headers=headers,
                timeout=30,
            )
            resp.raise_for_status()
            body = resp.json()
        except Exception as e:
            logger.warning(f"⚠️  Coinglass liquidation page {page}: {e}")
            break

        data = body.get("data", [])
        if not data:
            break

        all_records.extend(data)
        page += 1

        timestamps = [r.get("t", 0) for r in data if r.get("t")]
        if not timestamps:
            break
        earliest = min(timestamps)
        if earliest <= start_ts:
            break

        current_end = earliest - 1
        time.sleep(2.5)  # Rate limiting

    if not all_records:
        logger.warning(f"⚠️  Coinglass: no liquidation data for {symbol}")
        return pd.DataFrame()

    rows = []
    for r in all_records:
        ts = r.get("t")
        if ts is None:
            continue
        rows.append({
            "timestamp": pd.Timestamp(ts, unit="s", tz="UTC"),
            "liq_volume_long": float(r.get("longVolUsd", 0)),
            "liq_volume_short": float(r.get("shortVolUsd", 0)),
        })

    df = pd.DataFrame(rows)
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="last")]

    if not df.empty:
        logger.info(
            f"✅ Coinglass liq {symbol}: {len(df)} records "
            f"({df.index[0]:%Y-%m-%d} → {df.index[-1]:%Y-%m-%d})"
        )
    return df


# ══════════════════════════════════════════════════════════════
#  從 Force Orders 聚合成時間序列
# ══════════════════════════════════════════════════════════════

def aggregate_force_orders(
    orders: pd.DataFrame,
    interval: str = "1h",
) -> pd.DataFrame:
    """
    將逐筆清算訂單聚合成固定時間區間的清算量

    Returns:
        DataFrame with columns: liq_volume_long, liq_volume_short, liq_count_long, liq_count_short
    """
    if orders.empty:
        return pd.DataFrame(
            columns=["liq_volume_long", "liq_volume_short", "liq_count_long", "liq_count_short"]
        )

    resample_map = {
        "5m": "5min", "15m": "15min", "30m": "30min",
        "1h": "1h", "2h": "2h", "4h": "4h", "1d": "1D",
    }
    freq = resample_map.get(interval, "1h")

    # SELL side = 多頭被清算 (long liquidation)
    # BUY side = 空頭被清算 (short liquidation)
    long_liq = orders[orders["side"] == "SELL"]["quote_qty"]
    short_liq = orders[orders["side"] == "BUY"]["quote_qty"]

    result = pd.DataFrame(index=pd.DatetimeIndex([], name="timestamp"))

    if not long_liq.empty:
        result["liq_volume_long"] = long_liq.resample(freq).sum()
        result["liq_count_long"] = long_liq.resample(freq).count()
    if not short_liq.empty:
        result["liq_volume_short"] = short_liq.resample(freq).sum()
        result["liq_count_short"] = short_liq.resample(freq).count()

    result = result.fillna(0)
    return result


# ══════════════════════════════════════════════════════════════
#  衍生指標計算
# ══════════════════════════════════════════════════════════════

def compute_liquidation_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    計算清算衍生指標

    新增欄位：
        liq_total:      總清算量
        liq_imbalance:  清算不平衡 [-1, 1]（正=空頭被清算多，看多）
        liq_cascade_z:  清算瀑布 z-score（極端值=清算事件）
    """
    if df.empty:
        return df

    result = df.copy()

    long_vol = result.get("liq_volume_long", pd.Series(0, index=result.index))
    short_vol = result.get("liq_volume_short", pd.Series(0, index=result.index))

    result["liq_total"] = long_vol + short_vol

    # 不平衡: (short - long) / (short + long)
    # 正值 = 空頭被清算多 = 潛在看多信號
    total = long_vol + short_vol
    result["liq_imbalance"] = np.where(
        total > 0,
        (short_vol - long_vol) / total,
        0.0,
    )

    # 清算瀑布 z-score
    rolling_mean = result["liq_total"].rolling(168, min_periods=24).mean()  # 7d
    rolling_std = result["liq_total"].rolling(168, min_periods=24).std()
    result["liq_cascade_z"] = np.where(
        rolling_std > 0,
        (result["liq_total"] - rolling_mean) / rolling_std,
        0.0,
    )
    result["liq_cascade_z"] = result["liq_cascade_z"].clip(-5, 5).fillna(0)

    return result


# ══════════════════════════════════════════════════════════════
#  儲存 / 載入
# ══════════════════════════════════════════════════════════════

def save_liquidation(
    df: pd.DataFrame,
    symbol: str,
    data_dir: Path = DATA_DIR,
) -> Path:
    """儲存清算數據到 parquet"""
    path = data_dir / f"{symbol}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=True)
    logger.info(f"💾 Saved liquidation/{symbol}: {len(df)} rows → {path}")
    return path


def load_liquidation(
    symbol: str,
    data_dir: Path = DATA_DIR,
) -> pd.DataFrame | None:
    """載入清算數據"""
    path = data_dir / f"{symbol}.parquet"
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception as e:
        logger.warning(f"⚠️  Load liquidation/{symbol} failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════
#  主程式
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="清算/爆倉數據下載工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 從 Binance 下載最近清算數據
  PYTHONPATH=src python scripts/fetch_liquidation_data.py --symbols BTCUSDT ETHUSDT

  # 從 Coinglass 下載歷史清算（需 COINGLASS_API_KEY）
  PYTHONPATH=src python scripts/fetch_liquidation_data.py --symbols BTCUSDT --source coinglass

  # 查看已下載數據
  PYTHONPATH=src python scripts/fetch_liquidation_data.py --symbols BTCUSDT --coverage
        """,
    )
    parser.add_argument(
        "--symbols", nargs="+", required=True,
        help="交易對列表",
    )
    parser.add_argument(
        "--source", default="binance",
        choices=["binance", "coinglass", "both"],
        help="數據來源 (預設: binance)",
    )
    parser.add_argument(
        "--interval", default="1h",
        help="聚合區間 (預設: 1h)",
    )
    parser.add_argument(
        "--start", default=None,
        help="開始日期 (Coinglass 模式, YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end", default=None,
        help="結束日期 (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--coverage", action="store_true",
        help="只顯示覆蓋率報告",
    )
    parser.add_argument(
        "--data-dir", default=str(DATA_DIR),
        help=f"數據儲存目錄 (預設: {DATA_DIR})",
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    # 覆蓋率報告
    if args.coverage:
        print("\n📊 清算數據覆蓋率報告")
        print("=" * 60)
        for symbol in args.symbols:
            df = load_liquidation(symbol, data_dir)
            if df is None or df.empty:
                print(f"  {symbol}: ❌ 無數據")
            else:
                print(
                    f"  {symbol}: ✅ {len(df)} rows  "
                    f"{df.index[0]:%Y-%m-%d} → {df.index[-1]:%Y-%m-%d}  "
                    f"({(df.index[-1] - df.index[0]).days}d)"
                )
                if "liq_total" in df.columns:
                    print(f"           avg_total={df['liq_total'].mean():,.0f} USDT/bar")
        print()
        return

    # 下載模式
    for symbol in args.symbols:
        print(f"\n{'='*50}")
        print(f"  📥 {symbol} 清算數據")
        print(f"{'='*50}")

        frames = []

        # Binance force orders
        if args.source in ("binance", "both"):
            orders = fetch_binance_force_orders(symbol)
            if not orders.empty:
                agg = aggregate_force_orders(orders, args.interval)
                frames.append(agg)

        # Coinglass
        if args.source in ("coinglass", "both"):
            cg = fetch_coinglass_liquidation(
                symbol, args.interval, args.start, args.end,
            )
            if not cg.empty:
                frames.append(cg)

        if not frames:
            logger.warning(f"⚠️  {symbol}: 無清算數據")
            continue

        # 合併
        if len(frames) == 1:
            combined = frames[0]
        else:
            combined = pd.concat(frames)
            combined = combined[~combined.index.duplicated(keep="last")]
            combined = combined.sort_index()
            combined = combined.fillna(0)

        # 計算衍生指標
        combined = compute_liquidation_indicators(combined)

        # 儲存
        save_liquidation(combined, symbol, data_dir)

    print(f"\n✅ 下載完成！數據目錄: {data_dir}")


if __name__ == "__main__":
    main()
