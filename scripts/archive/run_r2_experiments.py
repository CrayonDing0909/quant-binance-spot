#!/usr/bin/env python3
"""
R2-100 Experiment Matrix Runner
Runs E1–E6 with strict cost backtests and produces evidence paths.

Usage:
    PYTHONPATH=src python -W ignore scripts/run_r2_experiments.py
"""
from __future__ import annotations
import sys, json, os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import yaml

from qtrade.config import load_config
from qtrade.backtest.run_backtest import run_symbol_backtest, BacktestResult
from qtrade.data.storage import load_klines

# ── helpers ──────────────────────────────────────────────

def _build_portfolio_equity(
    sleeve_equities: dict[str, pd.Series],
    weights: dict[str, float],
    initial_cash: float = 10_000,
) -> pd.Series:
    all_eq = list(sleeve_equities.values())
    min_start = max(eq.index[0] for eq in all_eq)
    max_end = min(eq.index[-1] for eq in all_eq)
    normed = {}
    for sym, eq in sleeve_equities.items():
        e = eq.loc[min_start:max_end]
        normed[sym] = e / e.iloc[0]
    port = sum(normed[s] * weights.get(s, 0) for s in normed)
    return port * initial_cash


def _compute_metrics(equity: pd.Series) -> dict:
    ret = equity.pct_change().dropna()
    total_ret = float(equity.iloc[-1] / equity.iloc[0] - 1)
    n_hours = len(equity)
    years = n_hours / 8760
    cagr = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0
    sr = float(ret.mean() / ret.std() * np.sqrt(8760)) if ret.std() > 0 else 0
    peak = equity.expanding().max()
    dd = (equity - peak) / peak
    mdd = abs(float(dd.min()))
    calmar = cagr / mdd if mdd > 0 else 0
    return {
        "total_return_pct": round(total_ret * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "sharpe": round(sr, 2),
        "max_dd_pct": round(mdd * 100, 2),
        "calmar": round(calmar, 2),
    }


def _yearly_breakdown(equity: pd.Series) -> list[dict]:
    equity = equity.copy()
    equity.index = pd.to_datetime(equity.index)
    rows = []
    for yr in sorted(equity.index.year.unique()):
        yr_eq = equity[equity.index.year == yr]
        if len(yr_eq) < 48:
            continue
        yr_ret = float(yr_eq.iloc[-1] / yr_eq.iloc[0] - 1)
        yr_rets = yr_eq.pct_change().dropna()
        yr_sr = float(yr_rets.mean() / yr_rets.std() * np.sqrt(8760)) if yr_rets.std() > 0 else 0
        yr_peak = yr_eq.expanding().max()
        yr_dd = (yr_eq - yr_peak) / yr_peak
        yr_mdd = abs(float(yr_dd.min()))
        label = f"{yr} YTD" if yr >= 2026 else str(yr)
        rows.append({"year": label, "return_pct": round(yr_ret * 100, 2),
                      "sharpe": round(yr_sr, 2), "max_dd_pct": round(yr_mdd * 100, 2)})
    return rows


def _sleeve_table(sleeve_equities: dict[str, pd.Series]) -> list[dict]:
    rows = []
    for sym, eq in sleeve_equities.items():
        m = _compute_metrics(eq)
        rows.append({"symbol": sym, **m})
    return rows


def _run_backtests_for_config(cfg_path, symbols, weights, cost_mult=1.0):
    cfg = load_config(cfg_path)
    market_type = cfg.market_type_str
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)
    ens = raw.get("ensemble", {})
    ens_strats = ens.get("strategies", {}) if ens.get("enabled") else {}
    sleeve_eq = {}
    for sym in symbols:
        data_path = cfg.data_dir / "binance" / market_type / cfg.market.interval / f"{sym}.parquet"
        if not data_path.exists():
            print(f"  ⚠️  {sym}: data not found at {data_path}")
            continue
        bt_cfg = cfg.to_backtest_dict(symbol=sym)
        strat_name = cfg.strategy.name
        if sym in ens_strats:
            strat_name = ens_strats[sym]["name"]
            bt_cfg["strategy_params"] = ens_strats[sym].get("params", {})
        if cost_mult != 1.0:
            bt_cfg["fee_bps"] = bt_cfg["fee_bps"] * cost_mult
            bt_cfg["slippage_bps"] = bt_cfg["slippage_bps"] * cost_mult
        res = run_symbol_backtest(sym, data_path, bt_cfg, strategy_name=strat_name, data_dir=cfg.data_dir)
        sleeve_eq[sym] = res.equity()
    initial_cash = cfg.backtest.initial_cash
    port_eq = _build_portfolio_equity(sleeve_eq, weights, initial_cash)
    port_metrics = _compute_metrics(port_eq)
    return port_eq, sleeve_eq, port_metrics


def _print_table(headers, rows, title=""):
    if title:
        print(f"\n{'─'*70}")
        print(f"  {title}")
        print(f"{'─'*70}")
    col_widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=4)) + 2
                  for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    print("  " + fmt.format(*headers))
    print("  " + fmt.format(*["─" * (w - 2) for w in col_widths]))
    for r in rows:
        print("  " + fmt.format(*r))


def run_experiment(name, cfg_path, symbols, weights, out_dir):
    exp_dir = out_dir / name
    exp_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for cm_label, cm in [("1.0x", 1.0), ("1.5x", 1.5)]:
        print(f"\n  ▶ {name} cost_mult={cm_label}")
        port_eq, sleeve_eq, metrics = _run_backtests_for_config(cfg_path, symbols, weights, cost_mult=cm)
        yearly = _yearly_breakdown(port_eq)
        sleeves = _sleeve_table(sleeve_eq)
        results[cm_label] = {"metrics": metrics, "yearly": yearly, "sleeves": sleeves}
        port_eq.to_csv(exp_dir / f"portfolio_equity_{cm_label}.csv")
        with open(exp_dir / f"portfolio_stats_{cm_label}.json", "w") as f:
            json.dump({"metrics": metrics, "yearly": yearly, "sleeves": sleeves}, f, indent=2)
    return results


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path("reports/R2_experiments") / ts
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"📂 Output: {out_root}\n")
    all_results = {}

    # ── R1 Baseline
    print("=" * 70); print("  R1 BASELINE"); print("=" * 70)
    r1_res = run_experiment("R1_baseline", "config/prod_candidate_R1.yaml",
        ["BTCUSDT", "ETHUSDT", "SOLUSDT"], {"BTCUSDT": 0.34, "ETHUSDT": 0.33, "SOLUSDT": 0.33}, out_root)
    all_results["R1_baseline"] = r1_res

    # ── E1
    print("\n" + "=" * 70); print("  E1 — UNIVERSE EXPANSION (9 symbols)"); print("=" * 70)
    e1_syms = ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT","ADAUSDT","DOGEUSDT","AVAXUSDT","LINKUSDT"]
    e1_wts = {"BTCUSDT":0.18,"ETHUSDT":0.14,"SOLUSDT":0.14,"BNBUSDT":0.10,"XRPUSDT":0.10,"ADAUSDT":0.08,"DOGEUSDT":0.08,"AVAXUSDT":0.10,"LINKUSDT":0.08}
    all_results["E1_universe"] = run_experiment("E1_universe","config/R2_E1_universe.yaml",e1_syms,e1_wts,out_root)

    # ── E2
    print("\n" + "=" * 70); print("  E2 — REGIME ROUTER"); print("=" * 70)
    all_results["E2_regime"] = run_experiment("E2_regime","config/R2_E2_regime.yaml",
        ["BTCUSDT","ETHUSDT","SOLUSDT"],{"BTCUSDT":0.34,"ETHUSDT":0.33,"SOLUSDT":0.33},out_root)

    # ── E3
    print("\n" + "=" * 70); print("  E3 — BTC BREAKOUT 2.0"); print("=" * 70)
    all_results["E3_btc_quality"] = run_experiment("E3_btc_quality","config/R2_E3_btc_quality.yaml",
        ["BTCUSDT","ETHUSDT","SOLUSDT"],{"BTCUSDT":0.34,"ETHUSDT":0.33,"SOLUSDT":0.33},out_root)

    # ── E4 (carry sleeve)
    print("\n" + "=" * 70); print("  E4 — CARRY SLEEVE"); print("=" * 70)
    cfg_e4 = load_config("config/prod_candidate_R1.yaml")
    mt = cfg_e4.market_type_str
    with open("config/prod_candidate_R1.yaml") as f:
        raw_e4 = yaml.safe_load(f)
    ens_e4 = raw_e4["ensemble"]["strategies"]
    for cm_label, cm in [("1.0x", 1.0), ("1.5x", 1.5)]:
        print(f"\n  ▶ E4_carry cost_mult={cm_label}")
        sleeve_eq_e4 = {}
        for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
            data_path = cfg_e4.data_dir / "binance" / mt / "1h" / f"{sym}.parquet"
            bt_cfg = cfg_e4.to_backtest_dict(symbol=sym)
            sn = ens_e4[sym]["name"]
            bt_cfg["strategy_params"] = ens_e4[sym].get("params", {})
            if cm != 1.0: bt_cfg["fee_bps"] *= cm; bt_cfg["slippage_bps"] *= cm
            res = run_symbol_backtest(sym, data_path, bt_cfg, strategy_name=sn, data_dir=cfg_e4.data_dir)
            sleeve_eq_e4[sym] = res.equity()
        bt_carry = cfg_e4.to_backtest_dict(symbol="BTCUSDT")
        bt_carry["strategy_params"] = {"mode":"single","funding_lookback_8h":90,"threshold":0.0001,"vol_target":0.10,"rebalance_interval":24,"min_holding_hours":72}
        if cm != 1.0: bt_carry["fee_bps"] *= cm; bt_carry["slippage_bps"] *= cm
        data_path_btc = cfg_e4.data_dir / "binance" / mt / "1h" / "BTCUSDT.parquet"
        carry_res = run_symbol_backtest("BTCUSDT", data_path_btc, bt_carry, strategy_name="funding_carry", data_dir=cfg_e4.data_dir)
        sleeve_eq_e4["BTC_carry"] = carry_res.equity()
        e4_wts = {"BTCUSDT":0.29,"ETHUSDT":0.28,"SOLUSDT":0.28,"BTC_carry":0.15}
        port_eq = _build_portfolio_equity(sleeve_eq_e4, e4_wts, cfg_e4.backtest.initial_cash)
        metrics = _compute_metrics(port_eq); yearly = _yearly_breakdown(port_eq); sleeves = _sleeve_table(sleeve_eq_e4)
        exp_dir = out_root / "E4_carry"; exp_dir.mkdir(parents=True, exist_ok=True)
        port_eq.to_csv(exp_dir / f"portfolio_equity_{cm_label}.csv")
        with open(exp_dir / f"portfolio_stats_{cm_label}.json", "w") as f:
            json.dump({"metrics":metrics,"yearly":yearly,"sleeves":sleeves},f,indent=2)
        if "E4_carry" not in all_results: all_results["E4_carry"] = {}
        all_results["E4_carry"][cm_label] = {"metrics":metrics,"yearly":yearly,"sleeves":sleeves}

    # ── E5 (dynamic risk budget)
    print("\n" + "=" * 70); print("  E5 — DYNAMIC RISK BUDGET"); print("=" * 70)
    cfg_e5 = load_config("config/prod_candidate_R1.yaml")
    with open("config/prod_candidate_R1.yaml") as f:
        raw_e5 = yaml.safe_load(f)
    ens_e5 = raw_e5["ensemble"]["strategies"]
    for cm_label, cm in [("1.0x", 1.0), ("1.5x", 1.5)]:
        print(f"\n  ▶ E5_dynrisk cost_mult={cm_label}")
        sleeve_eq_e5 = {}
        for sym in ["BTCUSDT","ETHUSDT","SOLUSDT"]:
            data_path = cfg_e5.data_dir / "binance" / mt / "1h" / f"{sym}.parquet"
            bt_cfg = cfg_e5.to_backtest_dict(symbol=sym)
            sn = ens_e5[sym]["name"]
            bt_cfg["strategy_params"] = ens_e5[sym].get("params", {})
            if cm != 1.0: bt_cfg["fee_bps"] *= cm; bt_cfg["slippage_bps"] *= cm
            res = run_symbol_backtest(sym, data_path, bt_cfg, strategy_name=sn, data_dir=cfg_e5.data_dir)
            sleeve_eq_e5[sym] = res.equity()
        all_eqs = list(sleeve_eq_e5.values())
        t0 = max(eq.index[0] for eq in all_eqs); t1 = min(eq.index[-1] for eq in all_eqs)
        aligned = {s: sleeve_eq_e5[s].loc[t0:t1] for s in sleeve_eq_e5}
        rets = {s: aligned[s].pct_change().fillna(0) for s in aligned}
        WINDOW = 720; MIN_W, MAX_W = 0.15, 0.55; syms = ["BTCUSDT","ETHUSDT","SOLUSDT"]
        idx = aligned["BTCUSDT"].index; n = len(idx)
        roll_sr = {}
        for s in syms:
            r = rets[s]; roll_mean = r.rolling(WINDOW).mean(); roll_std = r.rolling(WINDOW).std()
            roll_sr[s] = (roll_mean / roll_std * np.sqrt(8760)).clip(lower=0.01).fillna(0.01)
        w_arr = np.full((n, 3), 1.0/3)
        for i in range(WINDOW, n):
            scores = np.array([float(roll_sr[s].iloc[i]) for s in syms])
            raw_w = scores / scores.sum()
            raw_w = np.clip(raw_w, MIN_W, MAX_W); raw_w /= raw_w.sum()
            w_arr[i] = raw_w
        normed = {s: aligned[s]/aligned[s].iloc[0] for s in aligned}
        port_vals = np.ones(n)
        for i in range(1, n):
            daily_ret = sum(w_arr[i-1,j]*float(normed[syms[j]].iloc[i]/normed[syms[j]].iloc[i-1]-1) for j in range(3))
            port_vals[i] = port_vals[i-1]*(1+daily_ret)
        port_eq = pd.Series(port_vals*cfg_e5.backtest.initial_cash, index=idx)
        metrics = _compute_metrics(port_eq); yearly = _yearly_breakdown(port_eq); sleeves = _sleeve_table(sleeve_eq_e5)
        exp_dir = out_root / "E5_dynrisk"; exp_dir.mkdir(parents=True, exist_ok=True)
        port_eq.to_csv(exp_dir / f"portfolio_equity_{cm_label}.csv")
        with open(exp_dir / f"portfolio_stats_{cm_label}.json", "w") as f:
            json.dump({"metrics":metrics,"yearly":yearly,"sleeves":sleeves},f,indent=2)
        if "E5_dynrisk" not in all_results: all_results["E5_dynrisk"] = {}
        all_results["E5_dynrisk"][cm_label] = {"metrics":metrics,"yearly":yearly,"sleeves":sleeves}

    # ── E6 (execution robustness)
    print("\n" + "=" * 70); print("  E6 — EXECUTION ROBUSTNESS"); print("=" * 70)
    for cm_label, cm in [("1.5x", 1.5), ("2.0x", 2.0)]:
        print(f"\n  ▶ E6_exec cost_mult={cm_label}")
        port_eq, sleeve_eq, metrics = _run_backtests_for_config("config/prod_candidate_R1.yaml",
            ["BTCUSDT","ETHUSDT","SOLUSDT"],{"BTCUSDT":0.34,"ETHUSDT":0.33,"SOLUSDT":0.33},cost_mult=cm)
        yearly = _yearly_breakdown(port_eq); sleeves = _sleeve_table(sleeve_eq)
        exp_dir = out_root / "E6_exec"; exp_dir.mkdir(parents=True, exist_ok=True)
        port_eq.to_csv(exp_dir / f"portfolio_equity_{cm_label}.csv")
        with open(exp_dir / f"portfolio_stats_{cm_label}.json", "w") as f:
            json.dump({"metrics":metrics,"yearly":yearly,"sleeves":sleeves},f,indent=2)
        if "E6_exec" not in all_results: all_results["E6_exec"] = {}
        all_results["E6_exec"][cm_label] = {"metrics":metrics,"yearly":yearly,"sleeves":sleeves}

    # ── Final Report
    print("\n" + "═"*70); print("  R2-100 FINAL SUMMARY"); print("═"*70)
    headers = ["Experiment","Cost","Return%","CAGR%","Sharpe","MaxDD%","Calmar"]
    rows = []
    for exp_name, exp_data in all_results.items():
        for cm_label in sorted(exp_data.keys()):
            m = exp_data[cm_label]["metrics"]
            rows.append([exp_name,cm_label,f'{m["total_return_pct"]:+.1f}',f'{m["cagr_pct"]:+.1f}',
                         f'{m["sharpe"]:.2f}',f'{m["max_dd_pct"]:.1f}',f'{m["calmar"]:.2f}'])
    _print_table(headers, rows, "SUMMARY TABLE")
    with open(out_root / "r2_all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n📄 Master results: {out_root / 'r2_all_results.json'}")
    print(f"\n{'─'*70}\n  EVIDENCE PATHS\n{'─'*70}")
    for d in sorted(out_root.rglob("*.json")): print(f"  {d}")
    for d in sorted(out_root.rglob("*.csv")): print(f"  {d}")


if __name__ == "__main__":
    main()
