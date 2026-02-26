"""
數據清理工具 — 釋放磁碟空間

清理不再需要的暫存/中間數據，保留生產和回測所需檔案。

可清理項目：
1. vision_cache — OI 原始 CSV（合併到 merged parquet 後不再需要）
2. 5m klines — 僅研究用，生產策略未使用
3. kline_cache — live runner 的增量 K 線快取（重啟後會自動重建）

使用範例：
    # 列出可清理項目（dry-run，不刪除）
    python scripts/cleanup_data.py --dry-run

    # 清理 vision_cache（最大節省）
    python scripts/cleanup_data.py --vision-cache

    # 清理所有項目
    python scripts/cleanup_data.py --all

    # 僅清理 kline_cache
    python scripts/cleanup_data.py --kline-cache
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _dir_size_mb(path: Path) -> float:
    """計算目錄大小（MB）"""
    if not path.exists():
        return 0.0
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 * 1024)


def _file_count(path: Path) -> int:
    """計算目錄下檔案數量"""
    if not path.exists():
        return 0
    return sum(1 for f in path.rglob("*") if f.is_file())


def _remove_dir(path: Path, dry_run: bool) -> float:
    """刪除目錄，回傳釋放的 MB"""
    if not path.exists():
        print(f"  ⏭️  不存在: {path}")
        return 0.0
    size_mb = _dir_size_mb(path)
    n_files = _file_count(path)
    if dry_run:
        print(f"  🔍 [DRY-RUN] 將刪除: {path} ({size_mb:.1f} MB, {n_files} 檔案)")
    else:
        shutil.rmtree(path)
        print(f"  🗑️  已刪除: {path} ({size_mb:.1f} MB, {n_files} 檔案)")
    return size_mb


def clean_vision_cache(data_dir: Path, dry_run: bool) -> float:
    """
    清理 OI vision_cache — 合併後的原始 CSV

    Binance Vision OI 數據下載後會暫存為每日 CSV 到 vision_cache/，
    再合併成 merged parquet。合併完成後 CSV 不再需要。
    """
    print("\n📦 [1] OI Vision Cache (原始 CSV)")
    cache_dir = data_dir / "binance" / "futures" / "open_interest" / "vision_cache"
    return _remove_dir(cache_dir, dry_run)


def clean_5m_klines(data_dir: Path, dry_run: bool) -> float:
    """
    清理 5m K 線 — 僅研究/回測用

    生產策略使用 1h K 線，5m 數據僅在研究階段的 multi-TF 分析使用。
    需要時可透過 download_data.py 重新下載。
    """
    print("\n📦 [2] 5m K 線 (僅研究用)")
    kline_5m_dir = data_dir / "binance" / "futures" / "5m"
    return _remove_dir(kline_5m_dir, dry_run)


def clean_kline_cache(data_dir: Path, dry_run: bool) -> float:
    """
    清理 live runner 的 kline_cache

    增量 K 線快取會在 runner 重啟時自動重建（seed_bars）。
    清理後下次啟動會多花幾秒拉取種子數據。
    """
    print("\n📦 [3] Live K 線快取 (重啟自動重建)")
    total = 0.0
    reports_dir = data_dir.parent / "reports"
    if reports_dir.exists():
        for cache_dir in reports_dir.rglob("kline_cache"):
            if cache_dir.is_dir():
                total += _remove_dir(cache_dir, dry_run)
    if total == 0.0:
        print("  ⏭️  無 kline_cache 可清理")
    return total


def main():
    parser = argparse.ArgumentParser(
        description="數據清理工具 — 釋放磁碟空間",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  python scripts/cleanup_data.py --dry-run          # 列出可清理項目
  python scripts/cleanup_data.py --vision-cache     # 清理 OI 原始 CSV (~398MB)
  python scripts/cleanup_data.py --5m               # 清理 5m K 線 (~247MB)
  python scripts/cleanup_data.py --all              # 清理所有項目
        """,
    )
    parser.add_argument(
        "--data-dir", type=str, default="data",
        help="數據目錄 (預設: data)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="僅列出可清理項目，不實際刪除",
    )
    parser.add_argument(
        "--vision-cache", action="store_true",
        help="清理 OI vision_cache (合併後的原始 CSV)",
    )
    parser.add_argument(
        "--5m", dest="five_min", action="store_true",
        help="清理 5m K 線 (僅研究用，生產未使用)",
    )
    parser.add_argument(
        "--kline-cache", action="store_true",
        help="清理 live kline_cache (重啟後自動重建)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="清理所有項目",
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    # 預設 dry-run 如果沒有指定任何清理項目
    if not any([args.vision_cache, args.five_min, args.kline_cache, args.all]):
        args.dry_run = True
        args.all = True
        print("⚠️  未指定清理項目，預設 --dry-run --all（僅列出可清理項目）")

    do_all = args.all
    total_freed = 0.0

    print(f"📂 數據目錄: {data_dir.resolve()}")
    print(f"🔧 模式: {'DRY-RUN (不刪除)' if args.dry_run else '⚠️  實際刪除'}")

    if do_all or args.vision_cache:
        total_freed += clean_vision_cache(data_dir, args.dry_run)

    if do_all or args.five_min:
        total_freed += clean_5m_klines(data_dir, args.dry_run)

    if do_all or args.kline_cache:
        total_freed += clean_kline_cache(data_dir, args.dry_run)

    # 摘要
    print(f"\n{'='*50}")
    action = "可釋放" if args.dry_run else "已釋放"
    print(f"✅ 總計{action}: {total_freed:.1f} MB")
    if args.dry_run:
        print("💡 移除 --dry-run 以實際執行清理")


if __name__ == "__main__":
    main()
