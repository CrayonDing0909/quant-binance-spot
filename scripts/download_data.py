"""
多數據源 K 線數據下載工具

支援的數據源:
1. binance (默認) - Binance API，支援最近的數據
2. binance_vision - Binance 官方歷史數據，2017-08 開始
3. yfinance - Yahoo Finance，BTC 數據可追溯到 2014-09
4. ccxt - 多交易所 API (kraken, coinbasepro, bitstamp 等)

使用範例:
    # 默認從 Binance 下載
    python scripts/download_data.py -c config/base.yaml
    
    # 從 Yahoo Finance 下載長期歷史 (2015 年開始)
    python scripts/download_data.py --source yfinance --start 2015-01-01
    
    # 從 Kraken 下載 (2013 年開始)
    python scripts/download_data.py --source ccxt --exchange kraken --start 2013-10-01
    
    # 從 Binance Data Vision 批量下載 (更快)
    python scripts/download_data.py --source binance_vision --start 2017-08-17
    
    # 查看可用數據源資訊
    python scripts/download_data.py --list-sources
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pandas as pd
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


def fetch_from_source(
    source: str,
    symbol: str,
    interval: str,
    start: str,
    end: str | None,
    market_type: str = "spot",
    exchange: str | None = None,
) -> "pd.DataFrame":
    """
    從指定數據源獲取 K 線數據
    
    Args:
        source: 數據源名稱
        symbol: 交易對
        interval: K 線週期
        start: 開始日期
        end: 結束日期
        market_type: 市場類型
        exchange: CCXT 交易所名稱
    """
    if source == "binance":
        return fetch_klines(symbol, interval, start, end, market_type=market_type)
    
    elif source == "binance_vision":
        from qtrade.data.binance_vision import download_binance_vision_klines
        return download_binance_vision_klines(symbol, interval, start, end, market_type=market_type)
    
    elif source == "yfinance":
        from qtrade.data.yfinance_client import fetch_yfinance_klines
        return fetch_yfinance_klines(symbol, interval, start, end)
    
    elif source == "ccxt":
        from qtrade.data.ccxt_client import fetch_ccxt_klines
        exchange_id = exchange or "binance"
        return fetch_ccxt_klines(symbol, interval, start, end, exchange=exchange_id)
    
    else:
        raise ValueError(f"不支援的數據源: {source}")


def download_incremental(
    symbol: str,
    interval: str,
    start_date: str,
    end_date: str | None,
    data_path: Path,
    force_full: bool = False,
    market_type: str = "spot",
    source: str = "binance",
    exchange: str | None = None,
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
        source: 數據源名稱
        exchange: CCXT 交易所名稱
    
    Returns:
        (下載的新資料筆數, 總資料筆數)
    """
    import pandas as pd
    
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
        print(f"  📥 全量下載 {start_date} → {end_date or '現在'} (來源: {source})")
        df = fetch_from_source(source, symbol, interval, start_date, end_date, market_type, exchange)
        if not df.empty:
            save_klines(df, data_path)
            return len(df), len(df)
        return 0, 0
    
    # 增量下載策略
    existing_df = load_klines(data_path)
    chunks_to_merge = [existing_df]
    
    # 1. 檢查是否需要補齊前面的數據
    if target_start < local_start:
        gap_end = (local_start - interval_delta).strftime("%Y-%m-%d")
        print(f"  📥 補齊前段: {start_date} → {gap_end} (來源: {source})")
        front_df = fetch_from_source(source, symbol, interval, start_date, gap_end, market_type, exchange)
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
        print(f"  📥 更新後段: {fetch_start_str} → {fetch_end_str or '現在'} (來源: {source})")
        back_df = fetch_from_source(source, symbol, interval, fetch_start_str, fetch_end_str, market_type, exchange)
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


def list_data_sources() -> None:
    """列出所有可用的數據源及其資訊"""
    print("\n📊 可用的數據源:")
    print("=" * 70)
    
    # Binance API
    print("\n1️⃣  binance (默認)")
    print("   - 來源: Binance REST API")
    print("   - BTC 起始: 2017-08-17")
    print("   - 優點: 實時數據、支援 spot/futures")
    print("   - 用法: --source binance")
    
    # Binance Data Vision
    print("\n2️⃣  binance_vision")
    print("   - 來源: Binance 官方歷史數據庫")
    print("   - BTC 起始: 2017-08-17")
    print("   - 優點: 批量下載、速度快、完整歷史")
    print("   - 用法: --source binance_vision")
    
    # Yahoo Finance
    print("\n3️⃣  yfinance")
    print("   - 來源: Yahoo Finance")
    print("   - BTC 起始: ~2014-09")
    print("   - 優點: 最長免費歷史、無需 API key")
    print("   - 缺點: 只支援主流幣、數據可能有延遲")
    print("   - 用法: --source yfinance --start 2015-01-01")
    
    # CCXT
    print("\n4️⃣  ccxt")
    print("   - 來源: 多交易所統一 API")
    print("   - 支援交易所及 BTC 起始時間:")
    
    try:
        from qtrade.data.ccxt_client import EXCHANGE_HISTORY
        for ex_id, info in EXCHANGE_HISTORY.items():
            print(f"      • {ex_id}: {info['btc_start']} ({info['note']})")
    except ImportError:
        print("      • bitstamp: 2011-08 (最早)")
        print("      • kraken: 2013-10")
        print("      • bitfinex: 2013-04")
        print("      • coinbasepro: 2015-01")
        print("      • binance: 2017-08")
    
    print("   - 用法: --source ccxt --exchange kraken --start 2013-10-01")
    
    print("\n" + "=" * 70)
    print("\n💡 建議:")
    print("   • 如需 2017 年前的數據: 使用 yfinance 或 ccxt (kraken/bitstamp)")
    print("   • 如需 2017-現在完整數據: 使用 binance_vision + binance 組合")
    print("   • 如需實時更新: 使用 binance (默認)")


def check_data_availability(symbol: str, source: str, exchange: str | None = None) -> None:
    """檢查指定數據源的數據可用性"""
    print(f"\n🔍 檢查 {symbol} 在 {source} 的數據可用性...")
    
    if source == "yfinance":
        try:
            from qtrade.data.yfinance_client import get_yfinance_data_range
            earliest, latest = get_yfinance_data_range(symbol)
            if earliest:
                print(f"   ✅ {symbol}: {earliest} → {latest}")
            else:
                print(f"   ❌ {symbol}: 數據不可用")
        except ImportError:
            print("   ❌ yfinance 未安裝")
    
    elif source == "ccxt":
        try:
            from qtrade.data.ccxt_client import get_earliest_data_timestamp
            exchange_id = exchange or "binance"
            earliest = get_earliest_data_timestamp(exchange_id, symbol)
            if earliest:
                print(f"   ✅ {symbol} @ {exchange_id}: 從 {earliest} 開始")
            else:
                print(f"   ❌ {symbol} @ {exchange_id}: 數據不可用")
        except ImportError:
            print("   ❌ ccxt 未安裝")
    
    elif source == "binance_vision":
        try:
            from qtrade.data.binance_vision import check_data_availability
            result = check_data_availability(symbol, "1h")
            print(f"   {result['message']}")
        except ImportError:
            print("   ❌ binance_vision 模組錯誤")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="多數據源 K 線數據下載工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 默認從 Binance 下載
  python scripts/download_data.py -c config/base.yaml
  
  # 從 Yahoo Finance 下載長期歷史
  python scripts/download_data.py --source yfinance --symbol BTCUSDT --start 2015-01-01
  
  # 從 Kraken 下載更早的數據
  python scripts/download_data.py --source ccxt --exchange kraken --symbol BTCUSDT --start 2013-10-01
  
  # 查看可用數據源
  python scripts/download_data.py --list-sources
        """
    )
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
    
    # 多數據源選項
    parser.add_argument(
        "--source",
        type=str,
        default="binance",
        choices=["binance", "binance_vision", "yfinance", "ccxt"],
        help="數據源 (默認: binance)"
    )
    parser.add_argument(
        "--exchange",
        type=str,
        default=None,
        help="CCXT 交易所名稱 (用於 --source ccxt)"
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="覆蓋配置檔案中的開始日期 (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="覆蓋配置檔案中的結束日期 (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--interval",
        type=str,
        default=None,
        help="覆蓋配置檔案中的 K 線週期（支援逗號分隔批量下載: 5m,15m,1h,4h,1d）"
    )
    
    # Funding rate 下載
    parser.add_argument(
        "--funding-rate",
        action="store_true",
        help="同時下載 Futures 歷史 Funding Rate（合約模式自動啟用）"
    )
    
    # OI 下載
    parser.add_argument(
        "--oi",
        action="store_true",
        help="同時下載 Open Interest 數據（oi_liq_bounce 等策略自動啟用）"
    )
    parser.add_argument(
        "--clean-cache",
        action="store_true",
        help="OI 合併後自動刪除 vision_cache 原始 CSV（節省 ~400MB 磁碟）"
    )
    
    # 衍生品數據下載（LSR, Taker Vol, CVD, Liquidation）
    parser.add_argument(
        "--derivatives",
        action="store_true",
        help="同時下載衍生品數據（LSR、Taker Vol、CVD）到 data/binance/futures/derivatives/"
    )
    
    # 資訊查詢選項
    parser.add_argument(
        "--list-sources",
        action="store_true",
        help="列出所有可用的數據源"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="檢查指定數據源的數據可用性"
    )
    
    args = parser.parse_args()
    
    # 列出數據源
    if args.list_sources:
        list_data_sources()
        return
    
    # 載入配置
    cfg = load_config(args.config)
    m = cfg.market
    market_type = cfg.market_type_str  # "spot" or "futures"
    
    # 使用命令行參數覆蓋配置
    start_date = args.start or m.start
    end_date = args.end or m.end

    # 支援逗號分隔的多時間框架下載 (e.g. "5m,15m,1h,4h,1d")
    interval_arg = args.interval or m.interval
    intervals = [iv.strip() for iv in interval_arg.split(",")]
    
    # 如果指定了 symbol，只處理該交易對
    symbols = [args.symbol] if args.symbol else m.symbols
    
    # 檢查數據可用性
    if args.check:
        for sym in symbols:
            check_data_availability(sym, args.source, args.exchange)
        return
    
    # 市場類型標籤
    market_emoji = "🟢" if market_type == "spot" else "🔴"
    market_label = "SPOT" if market_type == "spot" else "FUTURES"
    
    # 數據源標籤
    source_label = args.source.upper()
    if args.source == "ccxt" and args.exchange:
        source_label = f"CCXT/{args.exchange.upper()}"
    
    # 顯示狀態模式
    if args.status:
        print(f"\n📊 本地數據狀態 {market_emoji} [{market_label}]:")
        print("-" * 60)
        for interval in intervals:
            for sym in symbols:
                data_path = cfg.data_dir / "binance" / market_type / interval / f"{sym}.parquet"
                local_start, local_end = get_local_data_range(data_path)
                if local_start:
                    print(f"  {sym} @ {interval}: {local_start.strftime('%Y-%m-%d')} → {local_end.strftime('%Y-%m-%d %H:%M')}")
                else:
                    print(f"  {sym} @ {interval}: ❌ 無本地數據")
        print("-" * 60)
        return
    
    # 下載模式 — 遍歷所有 interval
    total_new = 0
    for interval in intervals:
        print(f"\n🚀 開始下載 K 線數據 {market_emoji} [{market_label}] 📡 [{source_label}] ⏱ {interval}")
        print("-" * 60)
        print(f"   時間範圍: {start_date} → {end_date or '現在'}")
        print(f"   K 線週期: {interval}")
        print(f"   交易對: {', '.join(symbols)}")
        print("-" * 60)
        
        for sym in symbols:
            # 根據 market_type 決定存儲路徑
            data_path = cfg.data_dir / "binance" / market_type / interval / f"{sym}.parquet"
            
            # 先顯示本地狀態
            local_start, local_end = get_local_data_range(data_path)
            if local_start and not args.full:
                print(f"\n📁 {sym} @ {interval} 本地: {local_start.strftime('%Y-%m-%d')} → {local_end.strftime('%Y-%m-%d %H:%M')}")
            else:
                print(f"\n📁 {sym} @ {interval} 本地: 無數據")
            
            # 下載
            try:
                new_rows, total_rows = download_incremental(
                    symbol=sym,
                    interval=interval,
                    start_date=start_date,
                    end_date=end_date,
                    data_path=data_path,
                    force_full=args.full,
                    market_type=market_type,
                    source=args.source,
                    exchange=args.exchange,
                )
                
                total_new += new_rows
                
                if new_rows > 0:
                    print(f"  ✅ 新增 {new_rows} 筆，共 {total_rows} 筆 → {data_path}")
                else:
                    print(f"  ✅ 數據已是最新，共 {total_rows} 筆")
                    
            except Exception as e:
                print(f"  ❌ 下載失敗: {e}")
    
    print("-" * 60)
    print(f"🎉 完成！共新增 {total_new} 筆數據")

    # 使用主要 interval（第一個）作為 FR / OI 的 interval
    primary_interval = intervals[0]

    # ── Funding Rate 下載 ──────────────────────────
    # 合約模式下 --funding-rate 或 config 啟用時自動下載
    should_download_fr = (
        args.funding_rate
        or (market_type == "futures" and getattr(cfg.backtest.funding_rate, 'enabled', False))
    )
    if should_download_fr and market_type == "futures":
        from qtrade.data.funding_rate import (
            download_funding_rates,
            save_funding_rates,
            get_funding_rate_path,
            load_funding_rates,
        )
        print(f"\n📥 下載 Futures Funding Rate...")
        print("-" * 60)
        for sym in symbols:
            fr_path = get_funding_rate_path(cfg.data_dir, sym)
            try:
                existing = load_funding_rates(fr_path)
                if existing is not None and not args.full:
                    last_date = existing.index[-1].strftime("%Y-%m-%d")
                    print(f"  📥 {sym} Funding rate 增量更新: {last_date} → {end_date or '現在'}")
                    
                    # 增量下載
                    new_df = download_funding_rates(sym, last_date, end_date)
                    
                    if not new_df.empty:
                        # 過濾掉舊數據 (保留 index > existing.last)
                        new_data = new_df[new_df.index > existing.index[-1]]
                        if not new_data.empty:
                            merged = pd.concat([existing, new_data])
                            merged = merged[~merged.index.duplicated(keep='last')]
                            save_funding_rates(merged, fr_path)
                            print(f"  ✅ 新增 {len(new_data)} 筆，共 {len(merged)} 筆")
                        else:
                            print(f"  ✅ 數據已是最新")
                    else:
                        print(f"  ⚠️  無新數據")
                else:
                    fr_df = download_funding_rates(sym, start_date, end_date)
                    if not fr_df.empty:
                        save_funding_rates(fr_df, fr_path)
                        print(f"  ✅ {sym} Funding rate: {len(fr_df)} 筆 → {fr_path}")
                    else:
                        print(f"  ⚠️  {sym} 無 funding rate 資料")
            except Exception as e:
                print(f"  ❌ {sym} Funding rate 下載失敗: {e}")
        print("-" * 60)

    # ── OI 下載 ──────────────────────────────────────
    # oi_liq_bounce 等策略自動啟用，或 --oi 手動啟用
    oi_strategies = {"oi_liq_bounce", "oi_bb_rv"}
    strategy_name = getattr(cfg.strategy, "name", "")
    should_download_oi = (
        args.oi
        or (market_type == "futures" and strategy_name in oi_strategies)
    )
    if should_download_oi:
        try:
            from qtrade.data.open_interest import (
                download_open_interest,
                save_open_interest,
                load_open_interest,
                get_oi_path,
                merge_oi_sources,
            )
        except ImportError:
            print("⚠️  open_interest 模組不可用，跳過 OI 下載")
            should_download_oi = False

    if should_download_oi:
        print(f"\n📥 下載 Open Interest 數據...")
        print("-" * 60)

        # 1) binance_vision（完整歷史）
        for sym in symbols:
            try:
                print(f"  📥 {sym} OI via binance_vision...")
                df_vision = download_open_interest(
                    symbol=sym,
                    start=start_date,
                    end=end_date,
                    interval=primary_interval,
                    provider="binance_vision",
                )
                if not df_vision.empty:
                    path = get_oi_path(cfg.data_dir, sym, "binance_vision")
                    save_open_interest(df_vision, path)
                    print(f"  ✅ {sym} binance_vision: {len(df_vision)} 筆")
                else:
                    print(f"  ⚠️  {sym} binance_vision: 無數據")
            except Exception as e:
                print(f"  ❌ {sym} binance_vision OI 下載失敗: {e}")

        # 2) binance API（近期補齊）
        for sym in symbols:
            try:
                print(f"  📥 {sym} OI via binance API...")
                df_api = download_open_interest(
                    symbol=sym,
                    start=start_date,
                    end=end_date,
                    interval=primary_interval,
                    provider="binance",
                )
                if not df_api.empty:
                    path = get_oi_path(cfg.data_dir, sym, "binance")
                    save_open_interest(df_api, path)
                    print(f"  ✅ {sym} binance API: {len(df_api)} 筆")
                else:
                    print(f"  ⚠️  {sym} binance API: 無數據")
            except Exception as e:
                print(f"  ❌ {sym} binance API OI 下載失敗: {e}")

        # 3) 合併所有來源
        print(f"\n  🔀 合併 OI 來源...")
        for sym in symbols:
            try:
                sources = []
                for prov in ["binance_vision", "coinglass", "binance"]:
                    path = get_oi_path(cfg.data_dir, sym, prov)
                    loaded = load_open_interest(path)
                    if loaded is not None and not loaded.empty:
                        sources.append(loaded)
                if sources:
                    combined = merge_oi_sources(sources, max_ffill_bars=2)
                    save_path = get_oi_path(cfg.data_dir, sym, "merged")
                    save_open_interest(combined, save_path)
                    print(f"  ✅ {sym} merged: {len(combined)} 筆")
                else:
                    print(f"  ⚠️  {sym}: 無任何 OI 來源可合併")
            except Exception as e:
                print(f"  ❌ {sym} OI 合併失敗: {e}")

        # 4) 清理 vision_cache（合併後原始 CSV 不再需要）
        if getattr(args, "clean_cache", False):
            import shutil
            cache_base = cfg.data_dir / "binance" / "futures" / "open_interest" / "vision_cache"
            if cache_base.exists():
                n_files = sum(1 for f in cache_base.rglob("*") if f.is_file())
                size_mb = sum(f.stat().st_size for f in cache_base.rglob("*") if f.is_file()) / (1024 * 1024)
                shutil.rmtree(cache_base)
                print(f"  🗑️  vision_cache 已清理: {n_files} 檔案, {size_mb:.1f} MB")
            else:
                print(f"  ⏭️  vision_cache 不存在，無需清理")

        print("-" * 60)

    # ── 衍生品數據下載 (LSR, Taker Vol, CVD) ──────────
    should_download_derivatives = args.derivatives and market_type == "futures"
    if should_download_derivatives:
        from qtrade.data.long_short_ratio import download_lsr, save_lsr, LSR_TYPES
        from qtrade.data.taker_volume import (
            download_taker_volume, save_taker_volume,
            compute_cvd, save_cvd,
        )

        derivatives_dir = cfg.data_dir / "binance" / "futures" / "derivatives"

        print(f"\n📥 下載衍生品數據 (LSR + Taker Vol + CVD)...")
        print("-" * 60)

        for sym in symbols:
            # 1) Long/Short Ratio（全帳戶 + 大戶帳戶 + 大戶持倉）
            for lsr_type in LSR_TYPES:
                try:
                    series = download_lsr(
                        sym, lsr_type=lsr_type, start=start_date, end=end_date,
                        interval=primary_interval, provider="vision",
                    )
                    if not series.empty:
                        save_lsr(series, sym, lsr_type=lsr_type, data_dir=derivatives_dir)
                        print(f"  ✅ {sym} {lsr_type}: {len(series)} 筆")
                    else:
                        print(f"  ⚠️  {sym} {lsr_type}: 無數據")
                except Exception as e:
                    print(f"  ❌ {sym} {lsr_type}: {e}")

            # 2) Taker Buy/Sell Volume Ratio
            try:
                taker = download_taker_volume(
                    sym, start=start_date, end=end_date,
                    interval=primary_interval, provider="vision",
                )
                if not taker.empty:
                    save_taker_volume(taker, sym, data_dir=derivatives_dir)
                    print(f"  ✅ {sym} taker_vol_ratio: {len(taker)} 筆")

                    # 3) CVD 衍生計算
                    cvd = compute_cvd(taker)
                    save_cvd(cvd, sym, data_dir=derivatives_dir)
                    print(f"  ✅ {sym} cvd: {len(cvd)} 筆")
                else:
                    print(f"  ⚠️  {sym} taker_vol: 無數據")
            except Exception as e:
                print(f"  ❌ {sym} taker_vol/cvd: {e}")

        print("-" * 60)
        print(f"🎉 衍生品數據下載完成！存放位置: {derivatives_dir}")


if __name__ == "__main__":
    main()
