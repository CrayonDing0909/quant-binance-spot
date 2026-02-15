#!/usr/bin/env python3
"""
Combinatorial Purged Cross-Validation (CPCV) 驗證腳本

基於 López de Prado (2018) 的 CPCV 方法，嚴格驗證策略是否過擬合。
策略在完整數據上只執行一次（正確 warmup），然後對 returns 做組合式交叉驗證。

使用方法:
    # 基本用法
    python scripts/run_cpcv.py -c config/futures_rsi_adx_atr.yaml

    # 自定義 splits
    python scripts/run_cpcv.py -c config/futures_rsi_adx_atr.yaml --splits 6 --test-splits 2

    # 只測 BTC
    python scripts/run_cpcv.py -c config/futures_rsi_adx_atr.yaml --symbol BTCUSDT

輸出:
    - Train/Test Sharpe 分布
    - PBO (Probability of Backtest Overfitting)
    - Sharpe 衰退率
    - 報告 CSV 保存至 reports/{market_type}/{strategy_name}/validation/
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# 確保 src/ 在 sys.path 中
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from qtrade.config import load_config
from qtrade.validation.prado_methods import combinatorial_purged_cv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CPCV 驗證 (López de Prado)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="config/futures_rsi_adx_atr.yaml",
        help="配置檔路徑",
    )
    parser.add_argument(
        "--splits",
        type=int,
        default=6,
        help="CPCV splits 數（預設: 6）",
    )
    parser.add_argument(
        "--test-splits",
        type=int,
        default=2,
        help="每次用幾個 split 當測試集（預設: 2）",
    )
    parser.add_argument(
        "--purge-bars",
        type=int,
        default=10,
        help="Purge bars（訓練段尾部移除，預設: 10）",
    )
    parser.add_argument(
        "--embargo-bars",
        type=int,
        default=10,
        help="Embargo bars（測試段開頭移除，預設: 10）",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="只測指定交易對",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="不保存報告",
    )

    args = parser.parse_args()

    # ── 載入配置 ─────────────────────────────────────
    cfg = load_config(args.config)
    market_type = cfg.market_type_str
    strategy_name = cfg.strategy.name
    symbols = [args.symbol] if args.symbol else cfg.market.symbols

    from scipy.special import comb as sp_comb
    n_combos = int(sp_comb(args.splits, args.test_splits))

    market_emoji = "🟢" if market_type == "spot" else "🔴"
    print(f"{'='*70}")
    print(f"  CPCV 驗證  {market_emoji} {market_type.upper()} | {strategy_name}")
    print(f"  Splits: {args.splits}, Test splits: {args.test_splits}")
    print(f"  組合數: C({args.splits},{args.test_splits}) = {n_combos}")
    print(f"  Purge: {args.purge_bars} bars, Embargo: {args.embargo_bars} bars")
    print(f"{'='*70}")

    # ── 輸出目錄 ─────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = cfg.get_report_dir("validation") / f"cpcv_{timestamp}"
    if not args.no_save:
        report_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for sym in symbols:
        print(f"\n{'─'*60}")
        print(f"  {sym}")
        print(f"{'─'*60}")

        # 數據路徑
        data_path = cfg.data_dir / "binance" / market_type / cfg.market.interval / f"{sym}.parquet"
        if not data_path.exists():
            print(f"  ⚠️  數據不存在: {data_path}")
            continue

        bt_cfg = cfg.to_backtest_dict(symbol=sym)

        try:
            result = combinatorial_purged_cv(
                symbol=sym,
                data_path=data_path,
                cfg=bt_cfg,
                strategy_name=strategy_name,
                n_splits=args.splits,
                n_test_splits=args.test_splits,
                purge_bars=args.purge_bars,
                embargo_bars=args.embargo_bars,
                data_dir=cfg.data_dir,
            )
        except Exception as e:
            print(f"  ❌ CPCV 失敗: {e}")
            continue

        all_results[sym] = result

        # ── 顯示結果 ─────────────────────────────────
        print(f"\n  === CPCV 結果 ({sym}) ===")
        print(f"  成功組合數:      {result.n_combinations}/{n_combos}")
        print(f"  平均 Train SR:   {result.mean_train_sharpe:.2f} (±{result.std_train_sharpe:.2f})")
        print(f"  平均 Test SR:    {result.mean_test_sharpe:.2f} (±{result.std_test_sharpe:.2f})")
        print(f"  Sharpe 衰退:     {result.sharpe_degradation*100:.1f}%")
        print(f"  PBO (過擬合機率): {result.pbo:.2f}")

        # 分布統計
        test_arr = np.array(result.all_test_sharpes)
        pct_positive = (test_arr > 0).mean() * 100
        print(f"  Test SR > 0:     {pct_positive:.0f}%")
        print(f"  Test SR 範圍:    [{test_arr.min():.2f}, {test_arr.max():.2f}]")
        print(f"  Test SR 中位數:  {np.median(test_arr):.2f}")

        # 判定
        if result.is_robust:
            print(f"\n  ✅ 通過：PBO < 0.5 且 Sharpe 衰退 < 50% → 低過擬合風險")
        elif result.pbo < 0.5:
            print(f"\n  ⚠️  中度風險：PBO={result.pbo:.2f} 但 Sharpe 衰退 {result.sharpe_degradation*100:.0f}%")
        else:
            print(f"\n  ❌ 高風險：PBO={result.pbo:.2f} → 策略可能過擬合")

        # ── 保存報告 ─────────────────────────────────
        if not args.no_save:
            import pandas as pd
            cpcv_df = pd.DataFrame({
                "train_sharpe": result.all_train_sharpes,
                "test_sharpe": result.all_test_sharpes,
            })
            cpcv_path = report_dir / f"cpcv_{sym}.csv"
            cpcv_df.to_csv(cpcv_path, index=False)

            summary = {
                "symbol": sym,
                "n_splits": args.splits,
                "n_test_splits": args.test_splits,
                "n_combinations": int(result.n_combinations),
                "mean_train_sharpe": float(result.mean_train_sharpe),
                "mean_test_sharpe": float(result.mean_test_sharpe),
                "sharpe_degradation": float(result.sharpe_degradation),
                "pbo": float(result.pbo),
                "is_robust": bool(result.is_robust),
                "pct_test_positive": float(pct_positive),
            }
            summary_path = report_dir / f"cpcv_summary_{sym}.json"
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)

            print(f"  ✅ CPCV 報告: {cpcv_path}")

    # ── 全域摘要 ─────────────────────────────────────
    if all_results:
        print(f"\n{'='*70}")
        print(f"  全域摘要")
        print(f"{'='*70}")
        print(f"  {'Symbol':<12} {'Train SR':>10} {'Test SR':>10} {'Degrad':>10} {'PBO':>8} {'Robust':>8}")
        print(f"  {'-'*58}")
        for sym, r in all_results.items():
            icon = "✅" if r.is_robust else "❌"
            print(
                f"  {sym:<12}"
                f" {r.mean_train_sharpe:>10.2f}"
                f" {r.mean_test_sharpe:>10.2f}"
                f" {r.sharpe_degradation*100:>9.1f}%"
                f" {r.pbo:>8.2f}"
                f" {icon:>8}"
            )

        all_robust = all(r.is_robust for r in all_results.values())
        print()
        if all_robust:
            print(f"  ✅ 所有交易對通過 CPCV 驗證")
        else:
            print(f"  ⚠️  部分交易對未通過 CPCV 驗證")

    if not args.no_save:
        print(f"\n📁 報告目錄: {report_dir}")


if __name__ == "__main__":
    main()
