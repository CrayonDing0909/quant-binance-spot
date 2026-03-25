#!/usr/bin/env python3
# EMBARGO_EXEMPT: ablation uses aggTrades data (not kline embargo scope), symbols are production set
# ORTHOGONALITY_EXEMPT: orthogonality already verified in EDA #22 (corr(TSMOM)=0.04, corr(ATR)=0.25)
"""
avg_trade_size Overlay/Filter Ablation — Quant Developer (#22 Handoff)

根據 Alpha Researcher Tick OFI EDA (#22 WEAK GO):
  - avg_trade_size IC=-0.030（aggTrades 最強信號）
  - corr(TSMOM)=0.04, corr(ATR)=0.25（幾乎完全正交）
  - 7/8 same sign (A3 PASS), A1 4/7 (弱 → 建議 overlay > filter)

Ablation 設計:
  A: Baseline = 現行生產（HTF filter + LSR overlay）
  B: HTF + avg_trade_size overlay + LSR overlay（stacking overlay）
  C: avg_trade_size overlay only（no HTF filter，replace attempt）

注意: AVAX 沒有 aggTrades 數據，ablation 以 5/6 production symbols 進行。
      最終評估時 AVAX 使用 baseline（無 avg_trade_size）。

Usage:
    PYTHONPATH=src python scripts/research_avg_trade_size_ablation.py
"""
from __future__ import annotations

import copy
import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qtrade.config import load_config
from qtrade.backtest.run_backtest import run_symbol_backtest, BacktestResult
from qtrade.data.storage import load_klines

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ats_ablation")


# ══════════════════════════════════════════════════════════════
#  Config
# ══════════════════════════════════════════════════════════════

PROD_CONFIG = Path("config/prod_candidate_simplified.yaml")
REPORT_DIR = Path("reports/research/avg_trade_size_ablation")

# 有 aggTrades 數據的生產幣種
ATS_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "LINKUSDT"]
# 無 aggTrades 數據的生產幣種（只跑 baseline）
NO_ATS_SYMBOLS = ["AVAXUSDT"]
ALL_SYMBOLS = ATS_SYMBOLS + NO_ATS_SYMBOLS

# Ablation configs (param overrides on top of production config)
ABLATION_CONFIGS = {
    "A_baseline": {
        # Current production: HTF filter + LSR overlay, no avg_trade_size
    },
    # ── Round 1: Overlay (continuous scaling) ──
    "B_htf_ats_overlay_mild": {
        # HTF (keep) + MILD avg_trade_size overlay
        "ats_overlay_enabled": True,
        "ats_lookback": 720,
        "ats_scale_threshold": 0.70,   # only reduce above P70
        "ats_min_scale": 0.70,         # at most reduce to 70% (mild)
    },
    "C_htf_ats_overlay_moderate": {
        # HTF (keep) + MODERATE avg_trade_size overlay
        "ats_overlay_enabled": True,
        "ats_lookback": 720,
        "ats_scale_threshold": 0.60,
        "ats_min_scale": 0.50,
    },
    # ── Round 2: Filter (binary gate) ──
    "D_htf_ats_filter_p80": {
        # HTF (keep) + avg_trade_size filter P80
        "ats_filter_enabled": True,
        "ats_lookback": 720,
        "ats_max_pctrank": 0.80,
    },
    "E_htf_ats_filter_p90": {
        # HTF (keep) + avg_trade_size filter P90 (extreme only)
        "ats_filter_enabled": True,
        "ats_lookback": 720,
        "ats_max_pctrank": 0.90,
    },
    # ── Round 3: Replace HTF ──
    "F_ats_filter_p80_no_htf": {
        # avg_trade_size filter (replace HTF)
        "ats_filter_enabled": True,
        "ats_lookback": 720,
        "ats_max_pctrank": 0.80,
        "htf_filter_enabled": False,
    },
}


# ══════════════════════════════════════════════════════════════
#  Backtest Runner
# ══════════════════════════════════════════════════════════════

def run_single_backtest(
    cfg,
    symbol: str,
    param_overrides: dict | None = None,
) -> BacktestResult | None:
    """Run a single symbol backtest with optional param overrides.

    For meta_blend strategy, param_overrides are injected into each
    sub-strategy's params dict (not the top-level meta_blend params).
    """
    bt_cfg = copy.deepcopy(cfg.to_backtest_dict(symbol=symbol))

    # Apply param overrides — inject into sub-strategy params for meta_blend
    if param_overrides:
        sp = bt_cfg["strategy_params"]
        if "sub_strategies" in sp:
            # meta_blend: inject into each sub-strategy's params
            for sub in sp["sub_strategies"]:
                sub_params = sub.get("params", {})
                for key, value in param_overrides.items():
                    sub_params[key] = value
                sub["params"] = sub_params
        else:
            # Non-meta_blend: inject at top level
            for key, value in param_overrides.items():
                sp[key] = value

    market_type = "futures" if cfg.market.market_type == "futures" else "spot"
    data_path = (
        cfg.data_dir / "binance" / market_type
        / cfg.market.interval / f"{symbol}.parquet"
    )

    if not data_path.exists():
        logger.warning(f"⚠️  {symbol}: 數據不存在 ({data_path})")
        return None

    try:
        res = run_symbol_backtest(
            symbol, data_path, bt_cfg,
            strategy_name=cfg.strategy.name,
            data_dir=cfg.data_dir,
        )
        return res
    except Exception as e:
        logger.error(f"❌ {symbol}: Backtest failed: {e}")
        return None


def compute_portfolio_metrics(
    results: dict[str, BacktestResult],
    weights: dict[str, float],
) -> dict:
    """Compute portfolio-level metrics from per-symbol BacktestResult objects."""
    # Collect equity curves
    equity_curves = {}
    for sym, res in results.items():
        if res is None or res.pf is None:
            continue
        eq = res.pf.value()
        if isinstance(eq, pd.DataFrame):
            eq = eq.iloc[:, 0]
        # Normalize to returns
        ret = eq.pct_change().fillna(0)
        equity_curves[sym] = ret

    if not equity_curves:
        return {"sharpe": np.nan, "total_return": np.nan, "max_dd": np.nan, "calmar": np.nan}

    # Weighted portfolio returns
    port_ret = pd.Series(0.0, index=list(equity_curves.values())[0].index)
    total_weight = 0.0
    for sym, ret in equity_curves.items():
        w = weights.get(sym, 0.0)
        aligned_ret = ret.reindex(port_ret.index, fill_value=0.0)
        port_ret = port_ret + aligned_ret * w
        total_weight += w

    if total_weight > 0 and total_weight != 1.0:
        port_ret = port_ret / total_weight * 1.0  # keep raw weighted

    # Metrics
    cumulative = (1 + port_ret).cumprod()
    total_return = cumulative.iloc[-1] - 1.0

    annual_factor = 8760 / 1  # 1h bars, 8760 hours/year
    n_bars = len(port_ret)
    years = n_bars / annual_factor
    if years <= 0:
        return {"sharpe": np.nan, "total_return": np.nan, "max_dd": np.nan, "calmar": np.nan}

    ann_return = (1 + total_return) ** (1 / years) - 1
    ann_vol = port_ret.std() * np.sqrt(annual_factor)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()

    calmar = ann_return / abs(max_dd) if abs(max_dd) > 0 else 0.0

    return {
        "sharpe": round(sharpe, 3),
        "total_return": round(total_return * 100, 1),
        "max_dd": round(max_dd * 100, 2),
        "calmar": round(calmar, 2),
        "ann_return": round(ann_return * 100, 1),
        "ann_vol": round(ann_vol * 100, 1),
        "years": round(years, 1),
    }


# ══════════════════════════════════════════════════════════════
#  Main Ablation
# ══════════════════════════════════════════════════════════════

def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    cfg = load_config(str(PROD_CONFIG))
    weights = {}
    if hasattr(cfg, "portfolio") and cfg.portfolio and hasattr(cfg.portfolio, "allocation"):
        weights = cfg.portfolio.allocation or {}

    print("=" * 70)
    print("  avg_trade_size Ablation — Quant Developer (#22 Handoff)")
    print("=" * 70)
    print(f"  Config: {PROD_CONFIG}")
    print(f"  ATS symbols: {ATS_SYMBOLS}")
    print(f"  No-ATS symbols: {NO_ATS_SYMBOLS} (baseline only)")
    print(f"  Weights: {weights}")
    print()

    all_results = {}  # config_name → {symbol → BacktestResult}
    all_portfolio = {}  # config_name → portfolio_metrics

    for config_name, overrides in ABLATION_CONFIGS.items():
        print(f"\n{'═' * 60}")
        print(f"  Config: {config_name}")
        if overrides:
            for k, v in overrides.items():
                print(f"    {k}: {v}")
        else:
            print("    (production baseline)")
        print(f"{'═' * 60}")

        sym_results = {}

        for sym in ALL_SYMBOLS:
            # 無 aggTrades 數據的幣種只跑 baseline 或不加 ats 參數
            if sym in NO_ATS_SYMBOLS and overrides:
                # 去掉 ats 相關參數，但保留 htf_filter_enabled 的覆蓋
                non_ats_overrides = {
                    k: v for k, v in overrides.items()
                    if not k.startswith("ats_")
                }
                res = run_single_backtest(cfg, sym, non_ats_overrides if non_ats_overrides else None)
            else:
                res = run_single_backtest(cfg, sym, overrides if overrides else None)

            sym_results[sym] = res

            if res is not None:
                sr = res.sharpe()
                ret = res.total_return_pct()
                mdd = res.max_drawdown_pct()
                print(f"    {sym}: SR={sr:.2f}, Return={ret:+.1f}%, MDD={mdd:.1f}%")
            else:
                print(f"    {sym}: FAILED")

        all_results[config_name] = sym_results

        # Portfolio metrics
        port = compute_portfolio_metrics(sym_results, weights)
        all_portfolio[config_name] = port
        print(f"\n  📊 Portfolio [{config_name}]:")
        print(f"    Sharpe={port['sharpe']}, Return={port['total_return']}%, "
              f"MDD={port['max_dd']}%, Calmar={port['calmar']}")

    # ══════════════════════════════════════════════════════════════
    #  Summary
    # ══════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 70)
    print("  ABLATION SUMMARY")
    print("=" * 70)

    # Portfolio comparison table
    baseline = all_portfolio["A_baseline"]
    print(f"\n  {'Config':<30s} {'SR':>8s} {'ΔSR':>8s} {'Δ%':>8s} {'Return':>8s} {'MDD':>8s} {'Calmar':>8s}")
    print(f"  {'─' * 30} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8}")

    for name, port in all_portfolio.items():
        sr = port["sharpe"]
        delta_sr = sr - baseline["sharpe"]
        delta_pct = (delta_sr / baseline["sharpe"] * 100) if baseline["sharpe"] != 0 else 0
        print(f"  {name:<30s} {sr:>8.3f} {delta_sr:>+8.3f} {delta_pct:>+7.1f}% "
              f"{port['total_return']:>+7.1f}% {port['max_dd']:>7.2f}% {port['calmar']:>8.2f}")

    # Per-symbol comparison table
    print(f"\n\n  Per-Symbol Sharpe Ratio:")
    print(f"  {'Symbol':<12s}", end="")
    for name in ABLATION_CONFIGS:
        print(f"  {name:<25s}", end="")
    print()
    print(f"  {'─' * 12}", end="")
    for _ in ABLATION_CONFIGS:
        print(f"  {'─' * 25}", end="")
    print()

    for sym in ALL_SYMBOLS:
        print(f"  {sym:<12s}", end="")
        base_sr = None
        for name in ABLATION_CONFIGS:
            res = all_results[name].get(sym)
            if res is not None:
                sr = res.sharpe()
                if base_sr is None:
                    base_sr = sr
                delta = sr - base_sr if base_sr is not None else 0
                print(f"  {sr:>6.2f} (Δ{delta:>+.2f}){'':<10s}", end="")
            else:
                print(f"  {'FAIL':<25s}", end="")
        print()

    # Improvement count
    print(f"\n\n  Symbols improved vs baseline:")
    for name, sym_results in all_results.items():
        if name == "A_baseline":
            continue
        improved = 0
        total = 0
        for sym in ATS_SYMBOLS:
            baseline_res = all_results["A_baseline"].get(sym)
            current_res = sym_results.get(sym)
            if baseline_res is not None and current_res is not None:
                total += 1
                if current_res.sharpe() > baseline_res.sharpe():
                    improved += 1
        print(f"    {name}: {improved}/{total} symbols improved")

    # Verdict
    print(f"\n\n  ═══ VERDICT ═══")
    best_name = None
    best_delta = -999
    for name, port in all_portfolio.items():
        if name == "A_baseline":
            continue
        delta_pct = ((port["sharpe"] - baseline["sharpe"]) / baseline["sharpe"] * 100
                     if baseline["sharpe"] != 0 else 0)
        if delta_pct > best_delta:
            best_delta = delta_pct
            best_name = name

    if best_delta >= 5.0:
        print(f"  ✅ GO — {best_name}: +{best_delta:.1f}% SR (>= 5% threshold)")
    elif best_delta > 0:
        print(f"  ⚠️  KEEP_BASELINE — Best: {best_name} +{best_delta:.1f}% SR (< 5% threshold)")
    else:
        print(f"  ❌ FAIL — All configs worse than baseline")

    # ══════════════════════════════════════════════════════════════
    #  Save Report
    # ══════════════════════════════════════════════════════════════
    report = {
        "timestamp": ts,
        "config": str(PROD_CONFIG),
        "ats_symbols": ATS_SYMBOLS,
        "no_ats_symbols": NO_ATS_SYMBOLS,
        "weights": weights,
        "ablation_configs": {
            name: overrides for name, overrides in ABLATION_CONFIGS.items()
        },
        "portfolio": all_portfolio,
        "per_symbol": {},
        "verdict": {
            "best_config": best_name,
            "best_delta_pct": round(best_delta, 2),
            "decision": "GO" if best_delta >= 5.0 else (
                "KEEP_BASELINE" if best_delta > 0 else "FAIL"
            ),
        },
    }

    for name, sym_results in all_results.items():
        report["per_symbol"][name] = {}
        for sym, res in sym_results.items():
            if res is not None:
                report["per_symbol"][name][sym] = {
                    "sharpe": round(res.sharpe(), 4),
                    "total_return_pct": round(res.total_return_pct(), 2),
                    "max_drawdown_pct": round(res.max_drawdown_pct(), 2),
                    "n_trades": int(res.stats.get("Total Trades", 0)) if res.stats is not None else None,
                }
            else:
                report["per_symbol"][name][sym] = None

    report_path = REPORT_DIR / "ablation_results.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  📄 Report saved: {report_path}")


if __name__ == "__main__":
    main()
