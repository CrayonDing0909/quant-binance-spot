from __future__ import annotations
import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path
from qtrade.config import load_config
from qtrade.data.klines import fetch_klines
from qtrade.data.storage import save_klines, load_klines, get_local_data_range, merge_klines


def _interval_to_timedelta(interval: str) -> timedelta:
    """將 K 線週期轉換為 timedelta"""
    mapping = {
        "1m": timedelta(minutes=1),
        "3m": timedelta(minutes=3),
        "5m": timedelta(minutes=5),
        "15m": timedelta(minutes=15),
        "30m": timedelta(minutes=30),
        "1h": timedelta(hours=1),
        "2h": timedelta(hours=2),
        "4h": timedelta(hours=4),
        "6h": timedelta(hours=6),
        "8h": timedelta(hours=8),
        "12h": timedelta(hours=12),
        "1d": timedelta(days=1),
    }
    return mapping.get(interval, timedelta(hours=1))


def download_incremental(
    symbol: str,
    interval: str,
    start_date: str,
    end_date: str | None,
    data_path: Path,
    force_full: bool = False,
    market_type: str = "spot",
) -> tuple[int, int]:
    """
    增量下載 K 線數據
    
    Args:
        symbol: 交易對
        interval: K 線週期
        start_date: 開始日期
        end_date: 結束日期
        data_path: 儲存路徑
        force_full: 是否強制全量下載
        market_type: 市場類型 "spot" 或 "futures"
    
    Returns:
        (下載的新資料筆數, 總資料筆數)
    """
    # 取得本地數據範圍
    local_start, local_end = get_local_data_range(data_path)
    
    # 解析目標範圍
    target_start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    target_end = (
        datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if end_date
        else datetime.now(timezone.utc)
    )
    
    interval_delta = _interval_to_timedelta(interval)
    new_rows = 0
    
    # 判斷是否需要下載
    if force_full or local_start is None:
        # 全量下載
        print(f"  📥 全量下載 {start_date} → {end_date or '現在'}...")
        df = fetch_klines(symbol, interval, start_date, end_date, market_type=market_type)
        save_klines(df, data_path)
        return len(df), len(df)
    
    # 增量下載策略
    existing_df = load_klines(data_path)
    chunks_to_merge = [existing_df]
    
    # 1. 檢查是否需要補齊前面的數據
    if target_start < local_start:
        gap_end = (local_start - interval_delta).strftime("%Y-%m-%d")
        print(f"  📥 補齊前段: {start_date} → {gap_end}")
        front_df = fetch_klines(symbol, interval, start_date, gap_end, market_type=market_type)
        if not front_df.empty:
            chunks_to_merge.append(front_df)
            new_rows += len(front_df)
    
    # 2. 檢查是否需要下載後面的新數據
    # 加一個小緩衝，確保有重疊以處理可能的數據更新
    overlap_buffer = interval_delta * 2
    fetch_start = local_end - overlap_buffer
    
    if target_end > fetch_start:
        fetch_start_str = fetch_start.strftime("%Y-%m-%d")
        fetch_end_str = target_end.strftime("%Y-%m-%d") if end_date else None
        print(f"  📥 更新後段: {fetch_start_str} → {fetch_end_str or '現在'}")
        back_df = fetch_klines(symbol, interval, fetch_start_str, fetch_end_str, market_type=market_type)
        if not back_df.empty:
            # 計算真正的新數據（排除重疊部分）
            truly_new = back_df[back_df.index > local_end]
            new_rows += len(truly_new)
            chunks_to_merge.append(back_df)
    
    # 合併所有數據
    if len(chunks_to_merge) > 1:
        from functools import reduce
        merged = reduce(merge_klines, chunks_to_merge)
        save_klines(merged, data_path)
        return new_rows, len(merged)
    
    return 0, len(existing_df)


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
    parser.add_argument(
        "--full",
        action="store_true",
        help="強制全量下載（忽略本地緩存）"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="只顯示本地數據狀態，不下載"
    )
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    m = cfg.market
    market_type = m.market_type.value  # "spot" or "futures"
    
    # 如果指定了 symbol，只處理該交易對
    symbols = [args.symbol] if args.symbol else m.symbols
    
    # 市場類型標籤
    market_emoji = "🟢" if market_type == "spot" else "🔴"
    market_label = "SPOT" if market_type == "spot" else "FUTURES"
    
    # 顯示狀態模式
    if args.status:
        print(f"\n📊 本地數據狀態 {market_emoji} [{market_label}]:")
        print("-" * 60)
        for sym in symbols:
            data_path = cfg.data_dir / "binance" / market_type / m.interval / f"{sym}.parquet"
            local_start, local_end = get_local_data_range(data_path)
            if local_start:
                print(f"  {sym}: {local_start.strftime('%Y-%m-%d')} → {local_end.strftime('%Y-%m-%d %H:%M')}")
            else:
                print(f"  {sym}: ❌ 無本地數據")
        print("-" * 60)
        return
    
    # 下載模式
    print(f"\n🚀 開始下載 K 線數據 {market_emoji} [{market_label}]")
    print("-" * 60)
    
    total_new = 0
    for sym in symbols:
        # 根據 market_type 決定存儲路徑
        data_path = cfg.data_dir / "binance" / market_type / m.interval / f"{sym}.parquet"
        
        # 先顯示本地狀態
        local_start, local_end = get_local_data_range(data_path)
        if local_start and not args.full:
            print(f"\n📁 {sym} 本地: {local_start.strftime('%Y-%m-%d')} → {local_end.strftime('%Y-%m-%d %H:%M')}")
        else:
            print(f"\n📁 {sym} 本地: 無數據")
        
        # 下載
        new_rows, total_rows = download_incremental(
            symbol=sym,
            interval=m.interval,
            start_date=m.start,
            end_date=m.end,
            data_path=data_path,
            force_full=args.full,
            market_type=market_type,
        )
        
        total_new += new_rows
        
        if new_rows > 0:
            print(f"  ✅ 新增 {new_rows} 筆，共 {total_rows} 筆 → {data_path}")
        else:
            print(f"  ✅ 數據已是最新，共 {total_rows} 筆")
    
    print("-" * 60)
    print(f"🎉 完成！共新增 {total_new} 筆數據")


if __name__ == "__main__":
    main()
