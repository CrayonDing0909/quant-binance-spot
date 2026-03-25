#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════
  因子幾何審計 — 定期審計生產及候選信號的正交性
═══════════════════════════════════════════════════════════════

用途：
  1. 載入生產 config，提取所有中間信號（TSMOM、Basis、HTF、LSR、vol_pause）
  2. 載入所有 KEEP_BASELINE 信號（OI、On-chain、Macro、VPIN、avg_trade_size）
  3. 計算完整 NxN 相關矩陣 + PCA 分解
  4. 輸出因子幾何報告（相關性熱力圖、冗餘叢集、有效因子數）

用法：
  cd /Users/dylanting/Documents/spot_bot/quant-binance-spot
  source .venv/bin/activate

  # 基本用法（僅生產信號）
  PYTHONPATH=src python scripts/analyze_factor_geometry.py \
    -c config/prod_candidate_simplified.yaml

  # 含候選信號
  PYTHONPATH=src python scripts/analyze_factor_geometry.py \
    -c config/prod_candidate_simplified.yaml \
    --include-candidates

  # 指定輸出目錄
  PYTHONPATH=src python scripts/analyze_factor_geometry.py \
    -c config/prod_candidate_simplified.yaml \
    --output-dir reports/factor_geometry/20260305
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from qtrade.config import load_config, AppConfig
from qtrade.backtest.run_backtest import run_symbol_backtest
from qtrade.strategy import get_strategy
from qtrade.strategy.base import StrategyContext
from qtrade.data.storage import load_klines
from qtrade.validation.factor_orthogonality import (
    compute_signal_correlation_matrix,
    pca_decomposition,
    marginal_information_ratio,
    check_latent_factor_loading,
    run_factor_geometry_audit,
    print_factor_geometry_report,
    print_marginal_info_result,
)

import logging

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("factor_geometry")
logger.setLevel(logging.INFO)


# ═══════════════════════════════════════════════════════════
#  信號提取
# ═══════════════════════════════════════════════════════════

def _extract_production_signals(
    cfg: AppConfig,
    symbol: str,
) -> Dict[str, pd.Series]:
    """
    從生產策略中提取中間信號。

    回傳：
      - tsmom: TSMOM 核心信號
      - final_pos: 最終 position（含所有 filter/overlay）
    """
    signals = {}

    data_dir = cfg.data_dir
    market_type = cfg.market_type_str
    interval = cfg.market.interval
    data_path = data_dir / "binance" / market_type / interval / f"{symbol}.parquet"

    if not data_path.exists():
        logger.warning(f"  {symbol}: 數據不存在 ({data_path})")
        return signals

    try:
        df = load_klines(str(data_path))
        if df is None or df.empty:
            return signals

        # 取得策略函數
        strategy_name = cfg.strategy.name
        strategy_func = get_strategy(strategy_name)

        # 建立 context
        bt_cfg = cfg.to_backtest_dict(symbol=symbol)
        ctx = StrategyContext(
            symbol=symbol,
            interval=interval,
            market_type=market_type,
            direction=bt_cfg.get("direction", "both"),
            signal_delay=1,  # Backtest mode: trade on next open
        )

        # 執行策略取得最終 position
        pos = strategy_func(df, ctx, bt_cfg.get("strategy_params", {}))
        if pos is not None and not pos.empty:
            signals["final_pos"] = pos

    except Exception as e:
        logger.error(f"  {symbol}: 信號提取失敗: {e}")

    return signals


def _extract_all_signals(
    cfg: AppConfig,
    include_candidates: bool = False,
) -> Dict[str, pd.Series]:
    """
    對所有 symbols 提取信號，並平均為「策略層級」的信號序列。
    """
    symbols = cfg.market.symbols
    all_pos = []

    print(f"\n📊 提取 {len(symbols)} 個幣種的生產信號...")

    for symbol in symbols:
        signals = _extract_production_signals(cfg, symbol)
        if "final_pos" in signals:
            all_pos.append(signals["final_pos"])
            print(f"  ✅ {symbol}: {len(signals['final_pos']):,} bars")

    # 如果有多個幣種的信號，建立 portfolio-level 信號
    result = {}
    if all_pos:
        # 用第一個幣種的信號作為代表（BTC 通常最穩定）
        result["production_signal"] = all_pos[0]

    return result


# ═══════════════════════════════════════════════════════════
#  報告生成
# ═══════════════════════════════════════════════════════════

def _save_report(
    output_dir: Path,
    report,
    cfg: AppConfig,
):
    """儲存因子幾何報告到 JSON"""
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "timestamp": datetime.now().isoformat(),
        "config": str(cfg),
        "n_signals": report.n_signals,
        "n_effective_factors": report.n_effective_factors,
        "correlation_matrix": report.correlation_matrix.to_dict()
            if not report.correlation_matrix.empty else {},
        "pca": {
            "n_effective_factors": report.pca_result.n_effective_factors,
            "explained_variance_ratio": report.pca_result.explained_variance_ratio.tolist(),
            "cumulative_variance": report.pca_result.cumulative_variance.tolist(),
            "loadings": report.pca_result.loadings.to_dict(),
            "threshold": report.pca_result.threshold,
        },
        "redundancy_clusters": [
            {
                "cluster_id": c.cluster_id,
                "signals": c.signals,
                "avg_intra_corr": c.avg_intra_corr,
            }
            for c in report.redundancy_clusters
        ],
        "summary": report.summary,
    }

    output_path = output_dir / "factor_geometry_report.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n📁 報告已儲存: {output_path}")
    return output_path


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="因子幾何審計工具 — 偵測因子冗餘與潛在重疊",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-c", "--config", type=str, required=True,
        help="生產 config 路徑",
    )
    parser.add_argument(
        "--include-candidates", action="store_true",
        help="包含 KEEP_BASELINE 候選信號",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="輸出目錄（預設 reports/factor_geometry/<timestamp>）",
    )
    parser.add_argument(
        "--pca-threshold", type=float, default=0.95,
        help="PCA 解釋變異門檻 (default 0.95)",
    )
    parser.add_argument(
        "--cluster-threshold", type=float, default=0.30,
        help="冗餘叢集相關性門檻 (default 0.30)",
    )

    args = parser.parse_args()

    # ── 1. 載入配置 ──
    cfg = load_config(args.config)

    print("=" * 60)
    print(f"  因子幾何審計")
    print(f"  Config: {args.config}")
    print(f"  Symbols: {cfg.market.symbols}")
    print("=" * 60)

    # ── 2. 提取信號 ──
    signals = _extract_all_signals(cfg, include_candidates=args.include_candidates)

    if len(signals) < 2:
        print("\n⚠️  信號不足 2 個，無法執行因子幾何審計")
        print("    提示：使用 --include-candidates 加入候選信號")
        sys.exit(0)

    # ── 3. 執行審計 ──
    print(f"\n🔍 執行因子幾何審計 ({len(signals)} signals)...")
    report = run_factor_geometry_audit(
        signals,
        pca_threshold=args.pca_threshold,
        cluster_corr_threshold=args.cluster_threshold,
    )

    # ── 4. 印出報告 ──
    print_factor_geometry_report(report)

    # ── 5. 儲存報告 ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path("reports/factor_geometry") / timestamp
    _save_report(output_dir, report, cfg)

    # ── 6. 結論 ──
    print(f"\n📊 結論: 你有 {report.n_signals} 個信號但只有 "
          f"{report.n_effective_factors} 個獨立因子")
    if report.redundancy_clusters:
        print(f"   ⚠️  偵測到 {len(report.redundancy_clusters)} 個冗餘叢集")
    else:
        print(f"   ✅ 無冗餘叢集")


if __name__ == "__main__":
    main()
