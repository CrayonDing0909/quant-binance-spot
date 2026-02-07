from __future__ import annotations
from pathlib import Path
from qtrade.config import load_config
from qtrade.data.klines import fetch_klines
from qtrade.data.storage import save_klines


def main() -> None:
    cfg = load_config()
    m = cfg.market
    for sym in m.symbols:
        df = fetch_klines(sym, m.interval, m.start, m.end)
        out = cfg.data_dir / "binance" / "spot" / m.interval / f"{sym}.parquet"
        save_klines(df, out)
        print(f"saved {sym}: {len(df)} rows -> {out}")


if __name__ == "__main__":
    main()
