from __future__ import annotations
import argparse
from pathlib import Path
from qtrade.config import load_config
from qtrade.data.klines import fetch_klines
from qtrade.data.storage import save_klines


def main() -> None:
    parser = argparse.ArgumentParser(description="下載 Binance K 線數據")
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="config/base.yaml",
        help="配置檔案路徑（默認: config/base.yaml）"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="只下載指定的交易對"
    )
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    m = cfg.market
    
    # 如果指定了 symbol，只下載該交易對
    symbols = [args.symbol] if args.symbol else m.symbols
    
    for sym in symbols:
        print(f"下載 {sym}...")
        df = fetch_klines(sym, m.interval, m.start, m.end)
        out = cfg.data_dir / "binance" / "spot" / m.interval / f"{sym}.parquet"
        save_klines(df, out)
        print(f"saved {sym}: {len(df)} rows -> {out}")


if __name__ == "__main__":
    main()
