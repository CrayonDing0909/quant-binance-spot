#!/usr/bin/env python
"""
NW Strategy Architecture Experiment Matrix
==========================================

8 組實驗矩陣，回答三件事：
  1. NW 更適合做回歸還是趨勢？
  2. 多級別（1D/12H/4H/1H）分工是否能顯著提升 OOS？
  3. 最終應該是單 NW，還是 NW + TSMOM 組合？

強制約束：
  - signal_delay=1, trade_on=next_open, exit_exec_prices
  - fee=5bps + slippage=3bps + funding_rate (全開)
  - 不可關閉 anti-lookahead

Usage:
    # 全部 8 組（Phase 1: screening + Phase 2: WF for survivors）
    python scripts/run_experiment_matrix.py -c config/futures_nwkl.yaml

    # 只跑 Phase 1 screening
    python scripts/run_experiment_matrix.py -c config/futures_nwkl.yaml --phase1-only

    # 只跑特定實驗
    python scripts/run_experiment_matrix.py -c config/futures_nwkl.yaml --experiments E0 E4 E5

    # 指定單幣
    python scripts/run_experiment_matrix.py -c config/futures_nwkl.yaml --symbol SOLUSDT
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qtrade.config import load_config
from qtrade.backtest.run_backtest import run_symbol_backtest, BacktestResult
from qtrade.backtest.metrics import long_short_split_analysis
from qtrade.validation.walk_forward import walk_forward_analysis, walk_forward_summary

warnings.filterwarnings("ignore", category=UserWarning, message=".*slippage_model.*")


# ══════════════════════════════════════════════════════════════
#  Experiment Definitions
# ══════════════════════════════════════════════════════════════

# Base NW V3 params (from config/futures_nwkl.yaml)
_BASE_PARAMS = {
    "use_mtf": True,
    "mtf_alignment_mode": "right_ffill",
    "module_a_enabled": True,
    "module_b_enabled": False,
    "entry_mode": "pullback_nw",
    "use_ltf_proxy": True,
    "ltf_ema_fast": 20,
    "ltf_ema_slow": 50,
    "use_momentum_confirm": True,
    "momentum_lookback": 3,
    "min_pullback_depth": 0.3,
    "reentry_lockout": True,
    "kernel_bandwidth": 8.0,
    "kernel_alpha": 1.0,
    "kernel_lookback": 200,
    "envelope_multiplier": 2.0,
    "envelope_window": 200,
    "regime_mode": "adx",
    "adx_period": 14,
    "adx_threshold": 20.0,
    "slope_window": 10,
    "slope_threshold": 0.001,
    "htf_bandwidth": 8.0,
    "htf_lookback": 50,
    "htf_adx_period": 14,
    "htf_adx_threshold": 20.0,
    "htf_slope_window": 5,
    "htf_slope_threshold": 0.002,
    "trend_scale": 1.0,
    "range_scale": 0.5,
    "mr_exit_target": "nw",
    "atr_period": 14,
    "stop_loss_atr": 3.0,
    "take_profit_atr": 4.0,
    "trailing_stop_atr": None,
    "cooldown_bars": 8,
    "max_holding_bars": 0,
    "min_hold_bars": 4,
    # New: multi-TF defaults OFF
    "use_1d_risk_gate": False,
    "use_12h_regime": False,
    "use_entry_volume_expansion": False,
}


def _exp_params(overrides: dict) -> dict:
    """Create experiment params from base + overrides."""
    p = dict(_BASE_PARAMS)
    p.update(overrides)
    return p


EXPERIMENTS: dict[str, dict] = {
    "E0_baseline_nw_v3": {
        "desc": "現有 NW V3 (right_ffill + Phase B filters)",
        "strategy": "nw_envelope_regime",
        "params": _exp_params({}),
    },
    "E1_risk_gate_1d": {
        "desc": "+1D risk-on/off gate，不改 entry",
        "strategy": "nw_envelope_regime",
        "params": _exp_params({
            "use_1d_risk_gate": True,
            "risk_gate_lookback": 20,
            "risk_gate_adx_period": 14,
            "risk_gate_adx_threshold": 25.0,
        }),
    },
    "E2_dual_regime_12h_4h": {
        "desc": "12H + 4H regime 同向才開倉",
        "strategy": "nw_envelope_regime",
        "params": _exp_params({
            "use_12h_regime": True,
            "htf_12h_lookback": 25,
            "htf_12h_adx_period": 14,
            "htf_12h_adx_threshold": 20.0,
            "dual_regime_require_agree": True,
        }),
    },
    "E3_entry_trigger_quality": {
        "desc": "1H 入場加 momentum turn + volume expansion",
        "strategy": "nw_envelope_regime",
        "params": _exp_params({
            "use_1d_risk_gate": True,
            "risk_gate_lookback": 20,
            "risk_gate_adx_threshold": 25.0,
            "use_12h_regime": True,
            "htf_12h_lookback": 25,
            "dual_regime_require_agree": True,
            "use_entry_volume_expansion": True,
            "volume_expansion_period": 20,
            "volume_expansion_ratio": 1.5,
        }),
    },
    "E4_trend_only_nw": {
        "desc": "僅 trend-pullback 模組（禁 MR）",
        "strategy": "nw_envelope_regime",
        "params": _exp_params({
            "module_a_enabled": True,
            "module_b_enabled": False,
            "entry_mode": "pullback_nw",
        }),
    },
    "E5_mr_only_nw": {
        "desc": "僅 mean-reversion 模組（禁 trend）",
        "strategy": "nw_envelope_regime",
        "params": _exp_params({
            "module_a_enabled": False,
            "module_b_enabled": True,
            "entry_mode": "pullback_nw",
            # MR 不需要 regime trending — 在 non-trending 時觸發
            "use_mtf": True,
            "range_scale": 1.0,
            "mr_exit_target": "nw",
            # 禁用 trend-only filters
            "use_ltf_proxy": False,
            "use_momentum_confirm": False,
            "min_pullback_depth": 0.0,
            "reentry_lockout": False,
        }),
    },
    "E6_dual_module_regime_switch": {
        "desc": "trend/mr 雙模組由 regime 分流",
        "strategy": "nw_envelope_regime",
        "params": _exp_params({
            "module_a_enabled": True,
            "module_b_enabled": True,
            "entry_mode": "dual",
            "range_scale": 0.5,
            "trend_scale": 1.0,
        }),
    },
    "E7_ensemble_nw_tsmom": {
        "desc": "NW（SOL）+ TSMOM（BTC/ETH）symbol-level routing",
        "strategy": "ensemble",  # special handling
        "params": {},
        "symbol_strategies": {
            "BTCUSDT": {
                "name": "tsmom_multi_ema",
                "params": {
                    "lookbacks": [72, 168, 336, 720],
                    "vol_target": 0.15,
                    "ema_fast": 20,
                    "ema_slow": 50,
                    "agree_weight": 1.0,
                    "disagree_weight": 0.3,
                },
            },
            "ETHUSDT": {
                "name": "tsmom_multi_ema",
                "params": {
                    "lookbacks": [72, 168, 336, 720],
                    "vol_target": 0.15,
                    "ema_fast": 20,
                    "ema_slow": 50,
                    "agree_weight": 1.0,
                    "disagree_weight": 0.3,
                },
            },
            "SOLUSDT": {
                "name": "nw_envelope_regime",
                "params": _exp_params({}),  # Use E0 baseline for SOL
            },
        },
    },
}


# ══════════════════════════════════════════════════════════════
#  Metrics Collection
# ══════════════════════════════════════════════════════════════

def _extract_metrics(res: BacktestResult, symbol: str) -> dict:
    """Extract standard metrics from BacktestResult."""
    adj = res.adjusted_stats
    stats = res.stats

    src = adj if adj else stats
    total_ret = src.get("Total Return [%]", 0.0)
    sharpe = src.get("Sharpe Ratio", 0.0)
    sortino = src.get("Sortino Ratio", 0.0)
    mdd = abs(src.get("Max Drawdown [%]", 0.0))
    trades = stats.get("Total Trades", 0)
    win_rate = stats.get("Win Rate [%]", 0.0)

    # CAGR from equity curve
    equity = res.equity()
    if len(equity) > 1:
        years = (equity.index[-1] - equity.index[0]).total_seconds() / (365.25 * 86400)
        if years > 0 and equity.iloc[0] > 0:
            cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0
        else:
            cagr = 0.0
    else:
        cagr = 0.0

    calmar = (cagr * 100) / mdd if mdd > 0 else 0.0

    # Turnover: trades per year
    if len(equity) > 1:
        years = max((equity.index[-1] - equity.index[0]).total_seconds() / (365.25 * 86400), 0.01)
        turnover = trades / years
    else:
        turnover = 0.0

    # Funding cost
    fr_cost_pct = 0.0
    if res.funding_cost:
        fr_cost_pct = res.funding_cost.total_cost_pct * 100

    return {
        "symbol": symbol,
        "total_return_pct": total_ret,
        "cagr_pct": cagr * 100,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd_pct": mdd,
        "calmar": calmar,
        "trades": trades,
        "win_rate_pct": win_rate,
        "turnover_pa": turnover,
        "fr_cost_pct": fr_cost_pct,
    }


# ══════════════════════════════════════════════════════════════
#  Experiment Runner
# ══════════════════════════════════════════════════════════════

def run_single_experiment(
    exp_name: str,
    exp_def: dict,
    cfg,
    symbols: list[str],
    out_dir: Path,
) -> dict[str, Any]:
    """Run single experiment across all symbols."""
    print(f"\n{'='*70}")
    print(f"  {exp_name}: {exp_def['desc']}")
    print(f"{'='*70}")

    results = {"name": exp_name, "desc": exp_def["desc"], "symbols": {}}
    strategy = exp_def["strategy"]
    is_ensemble = strategy == "ensemble"

    for sym in symbols:
        # Determine strategy + params for this symbol
        if is_ensemble:
            sym_strats = exp_def.get("symbol_strategies", {})
            if sym not in sym_strats:
                print(f"  ⚠️  {sym} not in ensemble map, skipping")
                continue
            sym_strategy = sym_strats[sym]["name"]
            sym_params = sym_strats[sym]["params"]
        else:
            sym_strategy = strategy
            sym_params = exp_def["params"]

        # Build backtest config
        bt_cfg = cfg.to_backtest_dict(symbol=sym)
        bt_cfg["strategy_name"] = sym_strategy
        bt_cfg["strategy_params"] = sym_params

        # Data path
        data_path = (
            Path(cfg.data_dir) / "binance" / cfg.market_type_str
            / cfg.market.interval / f"{sym}.parquet"
        )
        if not data_path.exists():
            print(f"  ❌ {sym}: data not found at {data_path}")
            continue

        t0 = time.time()
        try:
            res = run_symbol_backtest(
                sym, data_path, bt_cfg, sym_strategy,
                data_dir=Path(cfg.data_dir),
            )
        except Exception as e:
            print(f"  ❌ {sym}: backtest failed: {e}")
            continue
        elapsed = time.time() - t0

        metrics = _extract_metrics(res, sym)
        results["symbols"][sym] = metrics

        # Long/Short split
        ls_info = ""
        try:
            ls = long_short_split_analysis(res.pf, res.pos)
            if ls["df"] is not None and not ls["df"].empty:
                long_ret = ls["df"].loc["Long", "Return [%]"] if "Long" in ls["df"].index else 0
                short_ret = ls["df"].loc["Short", "Return [%]"] if "Short" in ls["df"].index else 0
                ls_info = f"  L:{long_ret:+.1f}% S:{short_ret:+.1f}%"
                metrics["long_return_pct"] = long_ret
                metrics["short_return_pct"] = short_ret
        except Exception:
            pass

        sr_tag = f"SR={metrics['sharpe']:.2f}"
        mdd_tag = f"MDD={metrics['max_dd_pct']:.1f}%"
        ret_tag = f"Ret={metrics['total_return_pct']:+.1f}%"
        trades_tag = f"T={metrics['trades']}"
        print(f"  ✅ {sym}: {ret_tag} {sr_tag} {mdd_tag} {trades_tag}{ls_info}  [{elapsed:.1f}s]")

    return results


def run_walk_forward_experiment(
    exp_name: str,
    exp_def: dict,
    cfg,
    symbols: list[str],
    n_splits: int = 5,
) -> dict[str, Any]:
    """Run walk-forward for one experiment."""
    print(f"\n  📊 Walk-Forward: {exp_name}")
    strategy = exp_def["strategy"]
    is_ensemble = strategy == "ensemble"
    wf_results = {}

    for sym in symbols:
        if is_ensemble:
            sym_strats = exp_def.get("symbol_strategies", {})
            if sym not in sym_strats:
                continue
            sym_strategy = sym_strats[sym]["name"]
            sym_params = sym_strats[sym]["params"]
        else:
            sym_strategy = strategy
            sym_params = exp_def["params"]

        bt_cfg = cfg.to_backtest_dict(symbol=sym)
        bt_cfg["strategy_name"] = sym_strategy
        bt_cfg["strategy_params"] = sym_params

        data_path = (
            Path(cfg.data_dir) / "binance" / cfg.market_type_str
            / cfg.market.interval / f"{sym}.parquet"
        )
        if not data_path.exists():
            continue

        try:
            wf_df = walk_forward_analysis(
                symbol=sym,
                data_path=data_path,
                cfg=bt_cfg,
                n_splits=n_splits,
                data_dir=Path(cfg.data_dir),
            )
            if not wf_df.empty:
                summary = walk_forward_summary(wf_df)
                wf_results[sym] = {
                    "avg_oos_sharpe": summary["avg_test_sharpe"],
                    "std_oos_sharpe": summary["std_test_sharpe"],
                    "oos_positive_pct": summary["oos_positive_pct"],
                    "degradation_pct": summary["sharpe_degradation_pct"],
                    "worst_oos_sharpe": summary["worst_test_sharpe"],
                    "best_oos_sharpe": summary["best_test_sharpe"],
                    "avg_oos_return": summary["avg_test_return"],
                    "splits": wf_df.to_dict("records"),
                }
                print(
                    f"    {sym}: avg_OOS_SR={summary['avg_test_sharpe']:.2f} "
                    f"±{summary['std_test_sharpe']:.2f}, "
                    f"OOS+={summary['oos_positive_pct']:.0f}%, "
                    f"degrad={summary['sharpe_degradation_pct']:.0f}%"
                )
        except Exception as e:
            print(f"    ❌ {sym} WF failed: {e}")

    return wf_results


# ══════════════════════════════════════════════════════════════
#  Elimination & Ranking
# ══════════════════════════════════════════════════════════════

def apply_elimination(
    all_results: dict[str, dict],
    wf_results: dict[str, dict],
    baseline_turnover: dict[str, float],
) -> dict[str, dict]:
    """Apply elimination rules. Returns dict of eliminated experiments with reasons."""
    eliminated = {}

    for exp_name, exp_data in all_results.items():
        reasons = []
        sym_data = exp_data.get("symbols", {})

        if not sym_data:
            reasons.append("no symbol results")
            eliminated[exp_name] = {"reasons": reasons}
            continue

        # Rule 1: Cost-adjusted total return <= 0 for ALL symbols
        all_negative = all(
            m.get("total_return_pct", 0) <= 0 for m in sym_data.values()
        )
        if all_negative:
            reasons.append("cost-adjusted return <= 0 for all symbols")

        # Rule 2: Avg OOS Sharpe < 0.5 (if WF was run)
        if exp_name in wf_results:
            wf = wf_results[exp_name]
            avg_oos_sharpes = [v["avg_oos_sharpe"] for v in wf.values() if "avg_oos_sharpe" in v]
            if avg_oos_sharpes:
                avg_oos_sr = np.mean(avg_oos_sharpes)
                if avg_oos_sr < 0.5:
                    reasons.append(f"avg OOS Sharpe {avg_oos_sr:.2f} < 0.5")

        # Rule 3: MaxDD > 35%
        max_mdd = max(m.get("max_dd_pct", 0) for m in sym_data.values())
        if max_mdd > 35:
            reasons.append(f"MaxDD {max_mdd:.1f}% > 35%")

        # Rule 4: Turnover > baseline * 1.5 without Sharpe improvement
        if baseline_turnover:
            for sym, m in sym_data.items():
                bl_to = baseline_turnover.get(sym, 0)
                if bl_to > 0 and m.get("turnover_pa", 0) > bl_to * 1.5:
                    bl_sharpe = all_results.get("E0_baseline_nw_v3", {}).get("symbols", {}).get(sym, {}).get("sharpe", 0)
                    if m.get("sharpe", 0) <= bl_sharpe:
                        reasons.append(f"{sym}: turnover {m['turnover_pa']:.0f} > 1.5× baseline ({bl_to:.0f}) without Sharpe gain")
                        break

        # Rule 5: WF 5 splits, OOS positive < 3/5
        if exp_name in wf_results:
            wf = wf_results[exp_name]
            for sym, wf_data in wf.items():
                if wf_data.get("oos_positive_pct", 100) < 60:
                    reasons.append(f"{sym}: OOS positive {wf_data['oos_positive_pct']:.0f}% < 60%")
                    break

        if reasons:
            eliminated[exp_name] = {"reasons": reasons}

    return eliminated


# ══════════════════════════════════════════════════════════════
#  Report Generation
# ══════════════════════════════════════════════════════════════

def generate_report(
    all_results: dict,
    wf_results: dict,
    eliminated: dict,
    out_dir: Path,
) -> str:
    """Generate comprehensive experiment report."""
    lines = []
    lines.append("=" * 80)
    lines.append("  NW STRATEGY ARCHITECTURE EXPERIMENT REPORT")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)

    # ── 1. Experiment Ledger ──
    lines.append("\n\n## EXPERIMENT LEDGER (E0~E7)")
    lines.append("-" * 80)
    for exp_name, exp_data in all_results.items():
        desc = exp_data.get("desc", "")
        elim_tag = " ❌ ELIMINATED" if exp_name in eliminated else " ✅"
        lines.append(f"\n### {exp_name}{elim_tag}")
        lines.append(f"    {desc}")

        sym_data = exp_data.get("symbols", {})
        if sym_data:
            lines.append(f"    {'Symbol':<10} {'Return%':>10} {'CAGR%':>8} {'Sharpe':>8} {'Sortino':>8} "
                         f"{'MaxDD%':>8} {'Calmar':>8} {'Trades':>8} {'WR%':>6} {'TO/yr':>8}")
            lines.append(f"    {'-'*96}")
            for sym, m in sym_data.items():
                lines.append(
                    f"    {sym:<10} {m['total_return_pct']:>+10.1f} {m['cagr_pct']:>8.1f} "
                    f"{m['sharpe']:>8.2f} {m['sortino']:>8.2f} {m['max_dd_pct']:>8.1f} "
                    f"{m['calmar']:>8.2f} {m['trades']:>8.0f} {m['win_rate_pct']:>6.1f} "
                    f"{m['turnover_pa']:>8.0f}"
                )

    # ── 2. Ablation Table ──
    lines.append("\n\n## ABLATION TABLE (橫向比較)")
    lines.append("-" * 80)

    # Per-symbol comparison
    symbols_seen = set()
    for exp_data in all_results.values():
        symbols_seen.update(exp_data.get("symbols", {}).keys())

    for sym in sorted(symbols_seen):
        lines.append(f"\n### {sym}")
        lines.append(f"    {'Experiment':<30} {'Return%':>10} {'Sharpe':>8} {'MaxDD%':>8} {'Trades':>8} {'Calmar':>8}")
        lines.append(f"    {'-'*82}")
        for exp_name, exp_data in all_results.items():
            m = exp_data.get("symbols", {}).get(sym, {})
            if m:
                tag = " ❌" if exp_name in eliminated else ""
                lines.append(
                    f"    {exp_name:<30} {m['total_return_pct']:>+10.1f} {m['sharpe']:>8.2f} "
                    f"{m['max_dd_pct']:>8.1f} {m['trades']:>8.0f} {m['calmar']:>8.2f}{tag}"
                )

    # ── 3. Walk-Forward Results ──
    if wf_results:
        lines.append("\n\n## WALK-FORWARD RESULTS")
        lines.append("-" * 80)
        for exp_name, wf in wf_results.items():
            lines.append(f"\n### {exp_name}")
            for sym, wf_data in wf.items():
                lines.append(
                    f"    {sym}: avg_OOS_SR={wf_data['avg_oos_sharpe']:.2f} "
                    f"±{wf_data['std_oos_sharpe']:.2f}, "
                    f"OOS+={wf_data['oos_positive_pct']:.0f}%, "
                    f"degrad={wf_data['degradation_pct']:.0f}%, "
                    f"worst={wf_data['worst_oos_sharpe']:.2f}, "
                    f"best={wf_data['best_oos_sharpe']:.2f}"
                )

    # ── 4. Elimination List ──
    lines.append("\n\n## ELIMINATION LIST")
    lines.append("-" * 80)
    if eliminated:
        for exp_name, info in eliminated.items():
            lines.append(f"  ❌ {exp_name}:")
            for r in info["reasons"]:
                lines.append(f"      - {r}")
    else:
        lines.append("  (no eliminations)")

    # ── 5. Top-2 Candidates ──
    lines.append("\n\n## TOP-2 CANDIDATES")
    lines.append("-" * 80)
    survivors = {k: v for k, v in all_results.items() if k not in eliminated}
    if survivors:
        # Rank by average Sharpe across symbols
        ranked = []
        for exp_name, exp_data in survivors.items():
            sym_data = exp_data.get("symbols", {})
            if sym_data:
                avg_sr = np.mean([m["sharpe"] for m in sym_data.values()])
                avg_ret = np.mean([m["total_return_pct"] for m in sym_data.values()])
                avg_mdd = np.mean([m["max_dd_pct"] for m in sym_data.values()])
                ranked.append((exp_name, avg_sr, avg_ret, avg_mdd))
        ranked.sort(key=lambda x: x[1], reverse=True)

        for i, (name, sr, ret, mdd) in enumerate(ranked[:2], 1):
            lines.append(f"  #{i}: {name}")
            lines.append(f"      Avg Sharpe={sr:.2f}, Avg Return={ret:+.1f}%, Avg MDD={mdd:.1f}%")
    else:
        lines.append("  ⚠️  No survivors — all experiments eliminated")

    # ── 6. Verdict ──
    lines.append("\n\n## VERDICT")
    lines.append("-" * 80)
    if not survivors:
        lines.append("  ❌ NO-GO: All experiments failed elimination criteria")
        lines.append("  Next: Consider structural changes beyond NW framework")
    elif ranked:
        best = ranked[0]
        # Check soft thresholds
        soft_pass = True
        notes = []

        best_data = survivors[best[0]]
        avg_sr = best[1]
        avg_mdd = best[3]

        if avg_sr < 0.8:
            notes.append(f"avg Sharpe {avg_sr:.2f} < 0.8 target")
            soft_pass = False
        if avg_mdd > 20:
            notes.append(f"avg MDD {avg_mdd:.1f}% > 20% target")

        if soft_pass and avg_mdd <= 20:
            lines.append(f"  ✅ GO: {best[0]}")
        elif avg_sr > 0.5:
            lines.append(f"  ⚠️  CONDITIONAL GO: {best[0]}")
            for n in notes:
                lines.append(f"      - {n}")
        else:
            lines.append(f"  ❌ NO-GO: Best candidate insufficient")
            for n in notes:
                lines.append(f"      - {n}")

    # ── 7. Key Findings ──
    lines.append("\n\n## KEY FINDINGS")
    lines.append("-" * 80)

    # Q1: MR vs Trend
    e4 = all_results.get("E4_trend_only_nw", {}).get("symbols", {})
    e5 = all_results.get("E5_mr_only_nw", {}).get("symbols", {})
    if e4 and e5:
        e4_avg = np.mean([m["sharpe"] for m in e4.values()])
        e5_avg = np.mean([m["sharpe"] for m in e5.values()])
        if e4_avg > e5_avg:
            lines.append(f"  Q1: NW 更適合做 TREND (E4 SR={e4_avg:.2f} > E5 SR={e5_avg:.2f})")
        else:
            lines.append(f"  Q1: NW 更適合做 MR (E5 SR={e5_avg:.2f} > E4 SR={e4_avg:.2f})")

    # Q2: Multi-TF
    e0 = all_results.get("E0_baseline_nw_v3", {}).get("symbols", {})
    e2 = all_results.get("E2_dual_regime_12h_4h", {}).get("symbols", {})
    if e0 and e2:
        e0_avg = np.mean([m["sharpe"] for m in e0.values()])
        e2_avg = np.mean([m["sharpe"] for m in e2.values()])
        delta = e2_avg - e0_avg
        if delta > 0.1:
            lines.append(f"  Q2: 多級別分工有顯著提升 (ΔSR={delta:+.2f})")
        else:
            lines.append(f"  Q2: 多級別分工無顯著提升 (ΔSR={delta:+.2f})")

    # Q3: NW vs Ensemble
    e7 = all_results.get("E7_ensemble_nw_tsmom", {}).get("symbols", {})
    if e0 and e7:
        e0_avg = np.mean([m["sharpe"] for m in e0.values()])
        e7_avg = np.mean([m["sharpe"] for m in e7.values()])
        if e7_avg > e0_avg + 0.1:
            lines.append(f"  Q3: 應該用 NW+TSMOM 組合 (E7 SR={e7_avg:.2f} vs E0 SR={e0_avg:.2f})")
        else:
            lines.append(f"  Q3: 單 NW 足夠，組合未顯著改善 (E7 SR={e7_avg:.2f} vs E0 SR={e0_avg:.2f})")

    lines.append("")
    lines.append("=" * 80)
    report_text = "\n".join(lines)

    # Save report
    report_path = out_dir / "experiment_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n📄 Report saved: {report_path}")

    return report_text


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="NW Strategy Experiment Matrix")
    parser.add_argument("-c", "--config", type=str, default="config/futures_nwkl.yaml")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--symbol", type=str, default=None)
    parser.add_argument("--experiments", nargs="+", default=None,
                        help="Run specific experiments (e.g., E0 E4 E5)")
    parser.add_argument("--phase1-only", action="store_true",
                        help="Only run Phase 1 (single-symbol screening)")
    parser.add_argument("--wf-splits", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    symbols = (
        [args.symbol] if args.symbol
        else args.symbols if args.symbols
        else cfg.market.symbols
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else Path("reports/experiments") / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    # Filter experiments
    if args.experiments:
        exp_keys = []
        for e in args.experiments:
            matched = [k for k in EXPERIMENTS if e.upper() in k.upper()]
            exp_keys.extend(matched)
        exp_dict = {k: EXPERIMENTS[k] for k in exp_keys if k in EXPERIMENTS}
    else:
        exp_dict = EXPERIMENTS

    print(f"🧪 NW Strategy Experiment Matrix")
    print(f"   Config: {args.config}")
    print(f"   Symbols: {symbols}")
    print(f"   Experiments: {list(exp_dict.keys())}")
    print(f"   Output: {out_dir}")
    print(f"   Phase 1 only: {args.phase1_only}")

    # ══════════════════════════════════════════════
    #  Phase 1: Single-Symbol Screening
    # ══════════════════════════════════════════════
    print(f"\n{'#'*70}")
    print(f"  PHASE 1: Single-Symbol Screening")
    print(f"{'#'*70}")

    all_results = {}
    t_start = time.time()

    for exp_name, exp_def in exp_dict.items():
        results = run_single_experiment(exp_name, exp_def, cfg, symbols, out_dir)
        all_results[exp_name] = results

    phase1_time = time.time() - t_start
    print(f"\n⏱️  Phase 1 完成: {phase1_time:.0f}s ({phase1_time/60:.1f} min)")

    # Save Phase 1 results
    phase1_path = out_dir / "phase1_results.json"
    with open(phase1_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    # ══════════════════════════════════════════════
    #  Phase 2: Walk-Forward for survivors
    # ══════════════════════════════════════════════
    wf_results = {}
    baseline_turnover = {}

    # Get baseline turnover
    e0_syms = all_results.get("E0_baseline_nw_v3", {}).get("symbols", {})
    for sym, m in e0_syms.items():
        baseline_turnover[sym] = m.get("turnover_pa", 0)

    if not args.phase1_only:
        print(f"\n{'#'*70}")
        print(f"  PHASE 2: Walk-Forward Validation")
        print(f"{'#'*70}")

        # Quick pre-elimination to skip obviously bad experiments
        for exp_name, exp_data in all_results.items():
            sym_data = exp_data.get("symbols", {})
            if not sym_data:
                continue
            # Skip if ALL symbols have negative return AND negative sharpe
            all_bad = all(
                m.get("total_return_pct", 0) <= 0 and m.get("sharpe", 0) < 0
                for m in sym_data.values()
            )
            if all_bad:
                print(f"  ⏭️  Skipping WF for {exp_name} (all symbols negative)")
                continue

            t0 = time.time()
            wf = run_walk_forward_experiment(
                exp_name, exp_dict.get(exp_name, exp_data),
                cfg, symbols, args.wf_splits,
            )
            wf_results[exp_name] = wf
            print(f"    ⏱️  {exp_name} WF: {time.time()-t0:.0f}s")

        # Save WF results
        wf_path = out_dir / "wf_results.json"
        with open(wf_path, "w", encoding="utf-8") as f:
            json.dump(wf_results, f, indent=2, ensure_ascii=False, default=str)

    # ══════════════════════════════════════════════
    #  Phase 3: Elimination & Report
    # ══════════════════════════════════════════════
    eliminated = apply_elimination(all_results, wf_results, baseline_turnover)

    report = generate_report(all_results, wf_results, eliminated, out_dir)
    print(report)

    total_time = time.time() - t_start
    print(f"\n⏱️  Total: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"📁 All results saved to: {out_dir}")


if __name__ == "__main__":
    main()
