from __future__ import annotations
from datetime import datetime, timezone
import pandas as pd
from .binance_client import BinanceHTTP


KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "num_trades",
    "taker_buy_base", "taker_buy_quote", "ignore"
]


def _to_millis(dt_str: str) -> int:
    # Expect "YYYY-MM-DD"
    dt = datetime.strptime(dt_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def fetch_klines(
    symbol: str,
    interval: str,
    start: str,
    end: str | None = None,
    limit: int = 1000,
) -> pd.DataFrame:
    http = BinanceHTTP()
    start_ms = _to_millis(start)
    end_ms = _to_millis(end) if end else None

    rows: list[list] = []
    cur = start_ms

    while True:
        params = {"symbol": symbol, "interval": interval, "startTime": cur, "limit": limit}
        if end_ms:
            params["endTime"] = end_ms

        chunk = http.get("/api/v3/klines", params=params)
        if not chunk:
            break

        rows.extend(chunk)
        last_open_time = chunk[-1][0]
        # advance by 1 ms to avoid duplicate last bar
        cur = last_open_time + 1

        if len(chunk) < limit:
            break
        if end_ms and cur >= end_ms:
            break

    df = pd.DataFrame(rows, columns=KLINE_COLS)
    # types
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df = df.set_index("open_time").sort_index()
    return df[["open", "high", "low", "close", "volume", "close_time"]]
