#!/usr/bin/env python3
"""
VPIN Regime Filter — 3-Way Ablation Study

Configs:
    A: HTF only (baseline, same as prod_candidate_simplified)
    B: VPIN only (no HTF)
    C: HTF + VPIN (stacking)

Hypothesis:
    VPIN ⊥ vol (corr=0.025), so C(HTF+VPIN) may NOT over-filter
    (unlike OI/On-chain/Macro which all over-filtered when stacked with HTF)

Output:
    reports/research/vpin_ablation/ablation_results.json

# EMBARGO_EXEMPT: Completed research (2026-03-02, KEEP_BASELINE).
# Results already obtained before embargo infrastructure was built.
# Future re-runs should apply embargo via enforce_temporal_embargo().
"""
from __future__ import annotations

import copy
import json
import logging
import sys
from pathlib import Path

import pandas as pd

# ── Setup ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-25s  %(levelname)-7s  %(message)s",
)
logger = logging.getLogger("vpin_ablation")

from qtrade.config import load_config
from qtrade.backtest.run_backtest import run_symbol_backtest

# ═══════════════════════════════════════════
# Ablation configurations
# ═══════════════════════════════════════════
CONFIGS = {
    "A_htf_only": "config/research_vpin_ablation_A.yaml",
    "B_vpin_only": "config/research_vpin_ablation_B.yaml",
    "C_htf_vpin": "config/research_vpin_ablation_C.yaml",
}

OUTPUT_DIR = Path("reports/research/vpin_ablation")


def run_portfolio_for_config(config_path: str, label: str) -> dict:
    """Run portfolio backtest for a single config and return metrics."""
    logger.info(f"\n{'='*60}")
    logger.info(f"  ABLATION {label}: {config_path}")
    logger.info(f"{'='*60}")

    cfg = load_config(config_path)
    strategy_name = cfg.strategy.name
    market_type = cfg.market_type_str
    symbols = cfg.market.symbols
    results = {}
    all_equity = []

    for sym in symbols:
        bt_cfg = cfg.to_backtest_dict(symbol=sym)

        data_path = (
            cfg.data_dir / "binance" / market_type
            / cfg.market.interval / f"{sym}.parquet"
        )
        if not data_path.exists():
            logger.warning(f"  {sym}: 數據不存在 ({data_path})")
            results[sym] = {"sharpe": 0, "return_pct": 0, "mdd_pct": 0, "calmar": 0, "trades": 0}
            continue

        try:
            bt = run_symbol_backtest(
                sym, data_path, bt_cfg, strategy_name,
                data_dir=cfg.data_dir,
            )
            if bt and bt.pf is not None:
                pf = bt.pf
                sr = float(pf.sharpe_ratio())
                ret = float(pf.total_return()) * 100
                mdd = float(pf.max_drawdown()) * 100
                calmar = float(pf.annualized_return() / pf.max_drawdown()) if pf.max_drawdown() > 0 else 0
                trades = int(pf.trades.count())
                results[sym] = {
                    "sharpe": round(sr, 3),
                    "return_pct": round(ret, 2),
                    "mdd_pct": round(-abs(mdd), 2),
                    "calmar": round(calmar, 2),
                    "trades": trades,
                }
                logger.info(
                    f"  {sym}: SR={sr:.3f}, Return={ret:.1f}%, "
                    f"MDD={-abs(mdd):.2f}%, Trades={trades}"
                )

                # Equity curve for portfolio aggregation
                eq = pf.value()
                eq.name = sym
                all_equity.append(eq)
            else:
                logger.warning(f"  {sym}: NO RESULT")
                results[sym] = {"sharpe": 0, "return_pct": 0, "mdd_pct": 0, "calmar": 0, "trades": 0}
        except Exception as e:
            logger.error(f"  {sym}: FAILED — {e}")
            import traceback
            traceback.print_exc()
            results[sym] = {"sharpe": 0, "return_pct": 0, "mdd_pct": 0, "calmar": 0, "trades": 0}

    # ── Portfolio metrics ──
    portfolio_sr = 0
    portfolio_ret = 0
    portfolio_mdd = 0
    portfolio_calmar = 0

    if all_equity:
        alloc = cfg.portfolio.allocation or {}
        total_alloc = sum(alloc.get(s, 0.35) for s in symbols)

        # Weighted equity
        weighted_eq = None
        for eq in all_equity:
            sym = eq.name
            w = alloc.get(sym, 0.35) / total_alloc
            normed = eq / eq.iloc[0]
            contrib = normed * w
            if weighted_eq is None:
                weighted_eq = contrib
            else:
                # Align indices
                common = weighted_eq.index.intersection(contrib.index)
                weighted_eq = weighted_eq.reindex(common) + contrib.reindex(common)

        if weighted_eq is not None and len(weighted_eq) > 100:
            import numpy as np

            rets = weighted_eq.pct_change().dropna()
            ann_ret = rets.mean() * 8760
            ann_std = rets.std() * np.sqrt(8760)
            portfolio_sr = round(ann_ret / ann_std if ann_std > 0 else 0, 3)
            portfolio_ret = round((weighted_eq.iloc[-1] / weighted_eq.iloc[0] - 1) * 100, 2)

            running_max = weighted_eq.cummax()
            dd = (weighted_eq - running_max) / running_max
            portfolio_mdd = round(dd.min() * 100, 2)

            years = len(weighted_eq) / 8760
            cagr = ((weighted_eq.iloc[-1] / weighted_eq.iloc[0]) ** (1 / years) - 1) if years > 0 else 0
            portfolio_calmar = round(cagr * 100 / abs(portfolio_mdd) if portfolio_mdd != 0 else 0, 2)

    summary = {
        "label": label,
        "config": config_path,
        "portfolio": {
            "sharpe": portfolio_sr,
            "return_pct": portfolio_ret,
            "mdd_pct": portfolio_mdd,
            "calmar": portfolio_calmar,
        },
        "per_symbol": results,
    }

    logger.info(f"\n  PORTFOLIO [{label}]: SR={portfolio_sr}, "
                f"Return={portfolio_ret}%, MDD={portfolio_mdd}%")

    return summary


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for label, cfg_path in CONFIGS.items():
        all_results[label] = run_portfolio_for_config(cfg_path, label)

    # ── Comparative analysis ──
    logger.info("\n" + "=" * 80)
    logger.info("  ABLATION COMPARISON")
    logger.info("=" * 80)

    header = f"{'Config':<20} {'Portfolio SR':>12} {'Return%':>10} {'MDD%':>10} {'Calmar':>10}"
    logger.info(header)
    logger.info("-" * 62)

    baseline_sr = all_results["A_htf_only"]["portfolio"]["sharpe"]

    for label, res in all_results.items():
        p = res["portfolio"]
        delta = p["sharpe"] - baseline_sr
        delta_pct = delta / baseline_sr * 100 if baseline_sr != 0 else 0
        logger.info(
            f"{label:<20} {p['sharpe']:>12.3f} {p['return_pct']:>10.2f} "
            f"{p['mdd_pct']:>10.2f} {p['calmar']:>10.2f}  "
            f"(Δ SR: {delta:+.3f} = {delta_pct:+.1f}%)"
        )

    # ── Per-symbol comparison ──
    logger.info("\n  PER-SYMBOL SHARPE:")
    symbols = list(all_results["A_htf_only"]["per_symbol"].keys())
    header2 = f"{'Symbol':<12}" + "".join(f"  {label:>16}" for label in all_results.keys())
    logger.info(header2)
    logger.info("-" * (12 + 18 * len(all_results)))

    improved_B = 0
    improved_C = 0
    for sym in symbols:
        row = f"{sym:<12}"
        a_sr = all_results["A_htf_only"]["per_symbol"][sym]["sharpe"]
        for label, res in all_results.items():
            sr = res["per_symbol"][sym]["sharpe"]
            delta = sr - a_sr
            row += f"  {sr:>8.3f} ({delta:+.3f})"
        logger.info(row)

        b_sr = all_results["B_vpin_only"]["per_symbol"][sym]["sharpe"]
        c_sr = all_results["C_htf_vpin"]["per_symbol"][sym]["sharpe"]
        if b_sr > a_sr:
            improved_B += 1
        if c_sr > a_sr:
            improved_C += 1

    logger.info(f"\n  B (VPIN only) improved: {improved_B}/{len(symbols)} symbols")
    logger.info(f"  C (HTF+VPIN) improved: {improved_C}/{len(symbols)} symbols")

    # ── Verdict ──
    b_sr = all_results["B_vpin_only"]["portfolio"]["sharpe"]
    c_sr = all_results["C_htf_vpin"]["portfolio"]["sharpe"]
    b_delta_pct = (b_sr - baseline_sr) / baseline_sr * 100 if baseline_sr != 0 else 0
    c_delta_pct = (c_sr - baseline_sr) / baseline_sr * 100 if baseline_sr != 0 else 0

    logger.info(f"\n  B standalone Δ SR: {b_delta_pct:+.1f}%")
    logger.info(f"  C stacking  Δ SR: {c_delta_pct:+.1f}%")

    # Decision criteria:
    # - If C > A by >5% AND improved >= 6/8 → GO (stacking works due to orthogonality)
    # - If B > A by >5% AND improved >= 6/8 → Consider replacement
    # - If both < 5% → KEEP_BASELINE
    if c_delta_pct > 5 and improved_C >= 6:
        verdict = "GO_NEXT"
        reason = f"C(HTF+VPIN) > A by {c_delta_pct:.1f}%, {improved_C}/8 improved — orthogonality hypothesis confirmed"
    elif b_delta_pct > 5 and improved_B >= 6:
        verdict = "CONSIDER_REPLACEMENT"
        reason = f"B(VPIN only) > A by {b_delta_pct:.1f}%, {improved_B}/8 improved — VPIN standalone strong"
    elif c_delta_pct > 0 and c_sr > b_sr:
        verdict = "KEEP_BASELINE"
        reason = f"C marginally better ({c_delta_pct:+.1f}%) but < 5% threshold"
    else:
        verdict = "KEEP_BASELINE"
        reason = f"No significant improvement (B: {b_delta_pct:+.1f}%, C: {c_delta_pct:+.1f}%)"

    logger.info(f"\n  VERDICT: {verdict}")
    logger.info(f"  Reason: {reason}")

    # ── Save results ──
    output = {
        "ablation": "VPIN Regime Filter",
        "date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "hypothesis": "VPIN ⊥ vol (corr=0.025), HTF+VPIN may not over-filter",
        "verdict": verdict,
        "reason": reason,
        "configs": all_results,
        "comparison": {
            "baseline_sr": baseline_sr,
            "B_delta_pct": round(b_delta_pct, 2),
            "C_delta_pct": round(c_delta_pct, 2),
            "B_improved_symbols": improved_B,
            "C_improved_symbols": improved_C,
        },
    }

    out_path = OUTPUT_DIR / "ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"\n  Results saved: {out_path}")
    return verdict


if __name__ == "__main__":
    verdict = main()
    sys.exit(0 if verdict != "FAIL" else 1)
