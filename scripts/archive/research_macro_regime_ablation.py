"""
Macro Cross-Market Regime Filter — 3-Way Ablation

根據 Alpha Researcher 2026-03-01 EDA 結果：
  - GLD_mom_60d IC=-0.049, VIX_mom_30d IC=+0.043
  - Combined IC=0.056, binary filter Δ SR +0.20~+0.61, 8/8 improved
  - 與現有信號獨立性極高: corr(BTC_mom)=0.125, corr(TVL)=0.034

Ablation 設計：
  Config A: Baseline（現有生產 = HTF filter only）
  Config B: Macro filter only（關閉 HTF，只用 macro regime）
  Config C: HTF + Macro（兩者疊加）

決策規則：
  - Incremental SR (C vs A) > 5% → GO_NEXT
  - B standalone > A → macro filter 獨立有效
  - C < A → over-filter, 疊加退化

Usage:
  PYTHONPATH=src python scripts/research_macro_regime_ablation.py
"""
from __future__ import annotations

import copy
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from qtrade.config import load_config
from qtrade.backtest.run_backtest import run_symbol_backtest, BacktestResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("macro_ablation")


# ══════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════
PROD_CONFIG = "config/prod_candidate_simplified.yaml"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT"]
REPORT_DIR = Path("reports/research/macro_regime_ablation")

# Macro filter 默認參數（從 EDA 結果推導）
MACRO_PARAMS = {
    "macro_regime_filter_enabled": True,
    "macro_gld_window": 60,      # GLD 60d momentum
    "macro_vix_window": 30,      # VIX 30d momentum
    "macro_zscore_lookback": 252, # ~1 trading year
    "macro_pctrank_lookback": 720,  # 30 天 @ 1h
    "macro_min_pctrank": 0.20,   # 阻擋底部 20%
}


def _build_configs() -> dict[str, dict]:
    """Build 3 ablation configs from production baseline."""
    base_cfg = load_config(PROD_CONFIG)

    configs = {}

    # ── Config A: Baseline (HTF filter only, production) ──
    configs["A_baseline"] = base_cfg

    # ── Config B: Macro filter only (disable HTF, enable macro) ──
    configs["B_macro_only"] = base_cfg

    # ── Config C: HTF + Macro (both enabled) ──
    configs["C_htf_macro"] = base_cfg

    return configs


def _inject_into_sub_strategies(cfg: dict, params_to_inject: dict) -> None:
    """Inject params into all sub_strategies' params dicts (meta_blend routing)."""
    sp = cfg.get("strategy_params", {})
    subs = sp.get("sub_strategies", [])
    for sub in subs:
        sub_params = sub.get("params", {})
        for k, v in params_to_inject.items():
            sub_params[k] = v


def _get_backtest_dict(base_cfg, symbol: str, config_name: str) -> dict:
    """Get backtest dict with config-specific modifications."""
    cfg = copy.deepcopy(base_cfg.to_backtest_dict(symbol))

    if config_name == "B_macro_only":
        # Disable HTF filter, enable macro filter
        _inject_into_sub_strategies(cfg, {"htf_filter_enabled": False})
        _inject_into_sub_strategies(cfg, MACRO_PARAMS)

    elif config_name == "C_htf_macro":
        # Keep HTF (already enabled in prod), add macro filter
        _inject_into_sub_strategies(cfg, MACRO_PARAMS)

    # A_baseline: no changes needed

    return cfg


def _run_single(symbol: str, cfg_name: str, cfg_dict: dict, data_dir: Path, kline_path: Path) -> dict:
    """Run single symbol backtest, return metrics dict."""
    if not kline_path.exists():
        logger.warning(f"  {symbol}: kline 不存在 ({kline_path})")
        return {}

    try:
        result: BacktestResult = run_symbol_backtest(
            symbol=symbol,
            data_path=kline_path,
            cfg=cfg_dict,
            data_dir=data_dir,
        )

        sr = result.sharpe()
        tr = result.total_return_pct()
        mdd = result.max_drawdown_pct()
        trades = result.stats.get("Total Trades", 0) if result.stats is not None else 0

        # Time-in-market
        pos_abs = result.pos.abs()
        tim = (pos_abs > 0.01).mean() * 100

        return {
            "sharpe": round(sr, 3),
            "total_return_pct": round(tr, 2),
            "max_dd_pct": round(mdd, 2),
            "trades": int(trades),
            "time_in_market_pct": round(tim, 1),
        }
    except Exception as e:
        logger.error(f"  {symbol} / {cfg_name}: 回測失敗: {e}")
        return {}


def _portfolio_stats(per_symbol: dict[str, dict], weights: dict[str, float]) -> dict:
    """Compute weighted portfolio stats from per-symbol results."""
    valid = {s: m for s, m in per_symbol.items() if m and "sharpe" in m}
    if not valid:
        return {}

    # Weighted average
    total_w = sum(weights.get(s, 1.0) for s in valid)
    w_sr = sum(m["sharpe"] * weights.get(s, 1.0) for s, m in valid.items()) / total_w
    w_tr = sum(m["total_return_pct"] * weights.get(s, 1.0) for s, m in valid.items()) / total_w
    w_mdd = min(m["max_dd_pct"] for m in valid.values())  # worst MDD
    w_tim = sum(m["time_in_market_pct"] * weights.get(s, 1.0) for s, m in valid.items()) / total_w

    return {
        "portfolio_sharpe": round(w_sr, 3),
        "portfolio_return_pct": round(w_tr, 2),
        "portfolio_worst_mdd_pct": round(w_mdd, 2),
        "portfolio_avg_tim_pct": round(w_tim, 1),
        "n_symbols": len(valid),
    }


def main():
    logger.info("=" * 70)
    logger.info("Macro Cross-Market Regime Filter — 3-Way Ablation")
    logger.info("=" * 70)

    base_cfg = load_config(PROD_CONFIG)
    data_dir = base_cfg.data_dir

    # Portfolio weights from config
    alloc = {}
    if base_cfg.portfolio and base_cfg.portfolio.allocation:
        alloc = base_cfg.portfolio.allocation
    for s in SYMBOLS:
        if s not in alloc:
            alloc[s] = 1.0 / len(SYMBOLS)

    # Resolve kline paths
    kline_paths = base_cfg.resolve_kline_paths()
    logger.info(f"  Available kline paths: {list(kline_paths.keys())}")

    results = {}
    for cfg_name in ["A_baseline", "B_macro_only", "C_htf_macro"]:
        logger.info(f"\n{'='*60}")
        logger.info(f"Config {cfg_name}")
        logger.info(f"{'='*60}")

        per_sym = {}
        for symbol in SYMBOLS:
            if symbol not in kline_paths:
                logger.warning(f"  {symbol}: kline not found, skipping")
                continue
            cfg_dict = _get_backtest_dict(base_cfg, symbol, cfg_name)
            logger.info(f"  Running {symbol}...")
            metrics = _run_single(symbol, cfg_name, cfg_dict, data_dir, kline_paths[symbol])
            per_sym[symbol] = metrics
            if metrics:
                logger.info(
                    f"  {symbol}: SR={metrics['sharpe']:.3f}, "
                    f"Return={metrics['total_return_pct']:.1f}%, "
                    f"MDD={metrics['max_dd_pct']:.2f}%, "
                    f"TIM={metrics['time_in_market_pct']:.1f}%"
                )

        portfolio = _portfolio_stats(per_sym, alloc)
        results[cfg_name] = {
            "per_symbol": per_sym,
            "portfolio": portfolio,
        }

        if portfolio:
            logger.info(
                f"\n  Portfolio: SR={portfolio['portfolio_sharpe']:.3f}, "
                f"Return={portfolio['portfolio_return_pct']:.1f}%, "
                f"Worst MDD={portfolio['portfolio_worst_mdd_pct']:.2f}%, "
                f"TIM={portfolio['portfolio_avg_tim_pct']:.1f}%"
            )

    # ══════════════════════════════════════════════════════════════
    # 比較與判決
    # ══════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("ABLATION COMPARISON")
    logger.info("=" * 70)

    sr_a = results["A_baseline"]["portfolio"].get("portfolio_sharpe", 0)
    sr_b = results["B_macro_only"]["portfolio"].get("portfolio_sharpe", 0)
    sr_c = results["C_htf_macro"]["portfolio"].get("portfolio_sharpe", 0)

    logger.info(f"  A (Baseline / HTF only):    SR = {sr_a:.3f}")
    logger.info(f"  B (Macro only, no HTF):     SR = {sr_b:.3f}")
    logger.info(f"  C (HTF + Macro):            SR = {sr_c:.3f}")

    # Incremental
    if sr_a > 0:
        delta_b = (sr_b - sr_a) / sr_a * 100
        delta_c = (sr_c - sr_a) / sr_a * 100
    else:
        delta_b = delta_c = 0.0

    logger.info(f"\n  Δ B vs A: {delta_b:+.1f}%")
    logger.info(f"  Δ C vs A: {delta_c:+.1f}%")

    # Per-symbol comparison
    logger.info("\n  Per-Symbol SR Comparison:")
    logger.info(f"  {'Symbol':<12} {'A (Baseline)':>12} {'B (Macro)':>12} {'C (HTF+Macro)':>12} {'B>A?':>6} {'C>A?':>6}")
    logger.info(f"  {'-'*66}")

    n_b_better = 0
    n_c_better = 0
    for symbol in SYMBOLS:
        sa = results["A_baseline"]["per_symbol"].get(symbol, {}).get("sharpe", 0)
        sb = results["B_macro_only"]["per_symbol"].get(symbol, {}).get("sharpe", 0)
        sc = results["C_htf_macro"]["per_symbol"].get(symbol, {}).get("sharpe", 0)
        b_win = "✅" if sb > sa else "❌"
        c_win = "✅" if sc > sa else "❌"
        if sb > sa:
            n_b_better += 1
        if sc > sa:
            n_c_better += 1
        logger.info(f"  {symbol:<12} {sa:>12.3f} {sb:>12.3f} {sc:>12.3f} {b_win:>6} {c_win:>6}")

    logger.info(f"\n  B beats A: {n_b_better}/{len(SYMBOLS)} symbols")
    logger.info(f"  C beats A: {n_c_better}/{len(SYMBOLS)} symbols")

    # Verdict
    logger.info("\n" + "=" * 70)
    if delta_c > 5.0 and n_c_better >= 5:
        verdict = "GO_NEXT"
        reason = f"C vs A: Δ SR = {delta_c:+.1f}% > 5%, {n_c_better}/8 symbols improved"
    elif delta_c > 0 and n_c_better >= 6:
        verdict = "WEAK_GO"
        reason = f"C vs A: Δ SR = {delta_c:+.1f}% (positive), {n_c_better}/8 symbols improved, but < 5% threshold"
    elif sr_b > sr_a and delta_b > 5.0:
        verdict = "MACRO_STANDALONE_BETTER"
        reason = f"B standalone better (Δ {delta_b:+.1f}%), but stacking degrades → over-filter"
    else:
        verdict = "FAIL"
        reason = f"Macro filter Δ SR = {delta_c:+.1f}%, insufficient incremental value"

    logger.info(f"VERDICT: {verdict}")
    logger.info(f"REASON:  {reason}")
    logger.info("=" * 70)

    # ══════════════════════════════════════════════════════════════
    # 保存結果
    # ══════════════════════════════════════════════════════════════
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": datetime.now().isoformat(),
        "config_file": PROD_CONFIG,
        "macro_params": MACRO_PARAMS,
        "results": results,
        "comparison": {
            "sr_a": sr_a,
            "sr_b": sr_b,
            "sr_c": sr_c,
            "delta_b_pct": round(delta_b, 2),
            "delta_c_pct": round(delta_c, 2),
            "n_b_better": n_b_better,
            "n_c_better": n_c_better,
        },
        "verdict": verdict,
        "reason": reason,
    }

    report_path = REPORT_DIR / "ablation_results.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
