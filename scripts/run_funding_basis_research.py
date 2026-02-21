#!/usr/bin/env python3
"""
Funding / Basis Alpha Research — Phase 1

Runs three factor variants (F1/F2/F3) across BTC/ETH/SOL,
computes single-strategy backtest, walk-forward, cost stress,
and blend analysis vs the R2 production portfolio.

Usage:
    PYTHONPATH=src python scripts/run_funding_basis_research.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure src on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from qtrade.config import load_config
from qtrade.backtest.run_backtest import run_symbol_backtest, BacktestResult
from qtrade.data.storage import load_klines

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIGS = {
    "F1": BASE_DIR / "config" / "research_funding_F1.yaml",
    "F2": BASE_DIR / "config" / "research_funding_F2.yaml",
    "F3": BASE_DIR / "config" / "research_funding_F3.yaml",
}
R2_CONFIG = BASE_DIR / "config" / "prod_candidate_R2.yaml"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
COST_MULTS = [1.0, 1.5, 2.0]
BLEND_WEIGHTS = [0.10, 0.15, 0.20]  # carry weight in R2+carry blend
OUTPUT_DIR = BASE_DIR / "reports" / "funding_research"


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# Single-Symbol Backtest
# ─────────────────────────────────────────────────────────────
def _get_data_path(cfg, symbol: str) -> Path:
    """Construct data path for a symbol from config."""
    mt = cfg.market_type_str
    return cfg.data_dir / "binance" / mt / cfg.market.interval / f"{symbol}.parquet"


def run_single_backtest(
    cfg_path: Path,
    symbol: str,
    cost_mult: float = 1.0,
) -> dict | None:
    """Run a single-symbol backtest and return stats dict."""
    cfg = load_config(cfg_path)
    data_path = _get_data_path(cfg, symbol)
    if not data_path.exists():
        print(f"  ⚠️  Data not found: {data_path}")
        return None

    bt_cfg = cfg.to_backtest_dict()

    # Apply cost multiplier
    bt_cfg["fee_bps"] = bt_cfg.get("fee_bps", 5) * cost_mult
    bt_cfg["slippage_bps"] = bt_cfg.get("slippage_bps", 3) * cost_mult

    try:
        result = run_symbol_backtest(
            symbol=symbol,
            data_path=data_path,
            cfg=bt_cfg,
            strategy_name=cfg.strategy.name,
            market_type=cfg.market_type_str,
            direction=cfg.direction,
            data_dir=cfg.data_dir,
        )
    except Exception as e:
        print(f"  ❌ Backtest failed for {symbol}: {e}")
        return None

    equity = result.equity()
    returns = equity.pct_change().dropna()
    total_ret = (equity.iloc[-1] / equity.iloc[0]) - 1
    ann_factor = np.sqrt(8760)
    sharpe = (returns.mean() / returns.std() * ann_factor) if returns.std() > 0 else 0
    cum = (1 + returns).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()
    years = len(returns) / (365.25 * 24)
    cagr = (1 + total_ret) ** (1 / years) - 1 if years > 0 and total_ret > -1 else 0
    calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-9 else 0

    return {
        "symbol": symbol,
        "total_return": total_ret,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "calmar": calmar,
        "n_bars": len(returns),
        "equity": equity,
        "returns": returns,
    }


# ─────────────────────────────────────────────────────────────
# Portfolio Backtest (equal weight across symbols)
# ─────────────────────────────────────────────────────────────
def run_portfolio_factor(
    cfg_path: Path,
    cost_mult: float = 1.0,
    weights: dict | None = None,
) -> dict | None:
    """Run portfolio backtest for a factor config."""
    if weights is None:
        weights = {s: 1.0 / len(SYMBOLS) for s in SYMBOLS}

    all_equity = {}
    all_returns = {}

    for sym in SYMBOLS:
        res = run_single_backtest(cfg_path, sym, cost_mult)
        if res is None:
            return None
        all_equity[sym] = res["equity"]
        all_returns[sym] = res["returns"]

    # Align on common index
    eq_df = pd.DataFrame(all_equity)
    eq_df = eq_df.dropna()
    ret_df = eq_df.pct_change().dropna()

    # Weighted portfolio returns
    w = pd.Series(weights)
    port_ret = (ret_df * w).sum(axis=1)
    port_cum = (1 + port_ret).cumprod()
    port_cum = port_cum / port_cum.iloc[0]  # normalize

    total_ret = port_cum.iloc[-1] - 1
    years = len(port_ret) / (365.25 * 24)
    cagr = (1 + total_ret) ** (1 / years) - 1 if years > 0 and total_ret > -1 else 0
    sharpe = (port_ret.mean() / port_ret.std() * np.sqrt(8760)) if port_ret.std() > 0 else 0
    max_dd = (port_cum / port_cum.cummax() - 1).min()
    calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-9 else 0

    return {
        "total_return": total_ret,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "calmar": calmar,
        "port_returns": port_ret,
        "port_equity": port_cum,
        "per_symbol": all_returns,
    }


# ─────────────────────────────────────────────────────────────
# Walk-Forward (expanding window, 5 splits)
# ─────────────────────────────────────────────────────────────
def run_walk_forward_factor(cfg_path: Path, symbol: str, n_splits: int = 5) -> dict:
    """Simple expanding-window walk-forward for a single symbol."""
    cfg = load_config(cfg_path)
    data_path = _get_data_path(cfg, symbol)
    if not data_path.exists():
        return {"oos_sharpes": [], "avg_oos_sharpe": 0, "oos_positive": 0, "n_splits": 0}

    df = load_klines(data_path)
    if df is None or df.empty:
        return {"oos_sharpes": [], "avg_oos_sharpe": 0, "oos_positive": 0, "n_splits": 0}

    n = len(df)
    split_size = n // (n_splits + 1)  # expanding window

    bt_cfg = cfg.to_backtest_dict()
    oos_sharpes = []
    split_details = []

    for i in range(n_splits):
        train_end = split_size * (i + 1)
        test_start = train_end
        test_end = min(train_end + split_size, n)
        if test_end <= test_start:
            break

        # We run test period only
        test_df = df.iloc[test_start:test_end].copy()
        if len(test_df) < 100:
            break

        # Save test data to temp file for backtest
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            test_df.to_parquet(tmp_path)

        try:
            result = run_symbol_backtest(
                symbol=symbol,
                data_path=tmp_path,
                cfg=bt_cfg,
                strategy_name=cfg.strategy.name,
                market_type=cfg.market_type_str,
                direction=cfg.direction,
                data_dir=cfg.data_dir,
            )
            equity = result.equity()
            returns = equity.pct_change().dropna()
            if len(returns) > 0 and returns.std() > 0:
                sr = returns.mean() / returns.std() * np.sqrt(8760)
            else:
                sr = 0.0
            total_ret = (equity.iloc[-1] / equity.iloc[0]) - 1 if len(equity) > 1 else 0
            oos_sharpes.append(sr)
            split_details.append({
                "split": i + 1,
                "test_bars": len(test_df),
                "oos_return": total_ret,
                "oos_sharpe": sr,
            })
        except Exception as e:
            oos_sharpes.append(0.0)
            split_details.append({"split": i + 1, "error": str(e)})
        finally:
            tmp_path.unlink(missing_ok=True)

    avg_sr = np.mean(oos_sharpes) if oos_sharpes else 0
    oos_pos = sum(1 for s in oos_sharpes if s > 0)

    return {
        "oos_sharpes": oos_sharpes,
        "avg_oos_sharpe": avg_sr,
        "oos_positive": oos_pos,
        "n_splits": len(oos_sharpes),
        "details": split_details,
    }


# ─────────────────────────────────────────────────────────────
# Correlation & Blend Analysis vs R2
# ─────────────────────────────────────────────────────────────
def compute_blend_analysis(
    factor_port_ret: pd.Series,
    r2_port_ret: pd.Series,
    blend_weights: list[float],
) -> list[dict]:
    """Compute correlation and blend metrics."""
    # Align
    common = factor_port_ret.index.intersection(r2_port_ret.index)
    f_ret = factor_port_ret.loc[common]
    r_ret = r2_port_ret.loc[common]

    corr = f_ret.corr(r_ret)

    results = []
    for cw in blend_weights:
        blended = r_ret * (1 - cw) + f_ret * cw
        cum = (1 + blended).cumprod()
        total_ret = cum.iloc[-1] - 1
        years = len(blended) / (365.25 * 24)
        cagr = (1 + total_ret) ** (1 / years) - 1 if years > 0 and total_ret > -1 else 0
        sharpe = blended.mean() / blended.std() * np.sqrt(8760) if blended.std() > 0 else 0
        max_dd = (cum / cum.cummax() - 1).min()
        calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-9 else 0

        results.append({
            "carry_weight": cw,
            "corr_vs_r2": corr,
            "blend_cagr": cagr,
            "blend_sharpe": sharpe,
            "blend_max_dd": max_dd,
            "blend_calmar": calmar,
        })

    return results


def get_r2_portfolio_returns() -> pd.Series | None:
    """Load R2 portfolio equity and compute returns."""
    cfg = load_config(R2_CONFIG)
    weights = {"BTCUSDT": 0.34, "ETHUSDT": 0.33, "SOLUSDT": 0.33}

    all_equity = {}
    for sym in SYMBOLS:
        data_path = _get_data_path(cfg, sym)
        if not data_path.exists():
            return None
        bt_cfg = cfg.to_backtest_dict()

        # Get ensemble strategy for this symbol
        import yaml
        with open(R2_CONFIG, "r") as f:
            raw = yaml.safe_load(f)
        ens = raw.get("ensemble", {})
        strategies = ens.get("strategies", {})
        if sym in strategies:
            strat_name = strategies[sym]["name"]
            strat_params = strategies[sym].get("params", {})
            bt_cfg["params"] = strat_params
        else:
            strat_name = cfg.strategy.name

        try:
            result = run_symbol_backtest(
                symbol=sym,
                data_path=data_path,
                cfg=bt_cfg,
                strategy_name=strat_name,
                market_type=cfg.market_type_str,
                direction=cfg.direction,
                data_dir=cfg.data_dir,
            )
            all_equity[sym] = result.equity()
        except Exception as e:
            print(f"  ❌ R2 backtest failed for {sym}: {e}")
            return None

    eq_df = pd.DataFrame(all_equity).dropna()
    ret_df = eq_df.pct_change().dropna()
    w = pd.Series(weights)
    port_ret = (ret_df * w).sum(axis=1)
    return port_ret


# ─────────────────────────────────────────────────────────────
# Year-by-Year Analysis
# ─────────────────────────────────────────────────────────────
def year_by_year(port_returns: pd.Series) -> list[dict]:
    """Compute year-by-year stats from portfolio returns."""
    results = []
    for year in sorted(port_returns.index.year.unique()):
        yr = port_returns[port_returns.index.year == year]
        total_ret = (1 + yr).prod() - 1
        sharpe = yr.mean() / yr.std() * np.sqrt(8760) if yr.std() > 0 else 0
        cum = (1 + yr).cumprod()
        max_dd = (cum / cum.cummax() - 1).min()
        results.append({
            "year": year,
            "return": total_ret,
            "sharpe": sharpe,
            "max_dd": max_dd,
        })
    return results


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_DIR / ts
    _ensure_dir(out_dir)

    print("=" * 70)
    print("  Funding / Basis Alpha Research — Phase 1")
    print(f"  Output: {out_dir}")
    print("=" * 70)

    all_results = {}

    # ─── A. Full-period backtests with cost stress ───
    print("\n" + "=" * 70)
    print("  A. Full-Period Backtests (per-symbol + portfolio)")
    print("=" * 70)

    for factor, cfg_path in CONFIGS.items():
        print(f"\n{'─' * 50}")
        print(f"  Factor: {factor} ({cfg_path.name})")
        print(f"{'─' * 50}")

        factor_results = {"per_symbol": {}, "portfolio": {}}

        for cost_mult in COST_MULTS:
            label = f"{cost_mult}x"
            print(f"\n  📊 Cost {label}:")

            # Per-symbol
            sym_stats = {}
            for sym in SYMBOLS:
                res = run_single_backtest(cfg_path, sym, cost_mult)
                if res:
                    sym_stats[sym] = {
                        "total_return": res["total_return"],
                        "cagr": res["cagr"],
                        "sharpe": res["sharpe"],
                        "max_dd": res["max_dd"],
                        "calmar": res["calmar"],
                    }
                    print(
                        f"    {sym}: Return {res['total_return']*100:+.1f}%, "
                        f"SR {res['sharpe']:.2f}, MDD {res['max_dd']*100:.1f}%"
                    )
                else:
                    sym_stats[sym] = None
                    print(f"    {sym}: FAILED")

            factor_results["per_symbol"][label] = sym_stats

            # Portfolio
            port = run_portfolio_factor(cfg_path, cost_mult)
            if port:
                factor_results["portfolio"][label] = {
                    "total_return": port["total_return"],
                    "cagr": port["cagr"],
                    "sharpe": port["sharpe"],
                    "max_dd": port["max_dd"],
                    "calmar": port["calmar"],
                }
                print(
                    f"    PORTFOLIO: Return {port['total_return']*100:+.1f}%, "
                    f"SR {port['sharpe']:.2f}, MDD {port['max_dd']*100:.1f}%, "
                    f"Calmar {port['calmar']:.2f}"
                )

                # Save portfolio returns for blend analysis (1.0x only)
                if cost_mult == 1.0:
                    factor_results["_port_returns"] = port["port_returns"]
                    factor_results["_port_equity"] = port["port_equity"]

                    # Year-by-year
                    yby = year_by_year(port["port_returns"])
                    factor_results["year_by_year"] = yby
                    print(f"\n    Year-by-Year (1.0x):")
                    for y in yby:
                        print(
                            f"      {y['year']}: Ret {y['return']*100:+.1f}%, "
                            f"SR {y['sharpe']:.2f}, MDD {y['max_dd']*100:.1f}%"
                        )
            else:
                factor_results["portfolio"][label] = None
                print(f"    PORTFOLIO: FAILED")

        all_results[factor] = factor_results

    # ─── B. Walk-Forward (5 splits) ───
    print("\n" + "=" * 70)
    print("  B. Walk-Forward Analysis (5 splits)")
    print("=" * 70)

    for factor, cfg_path in CONFIGS.items():
        print(f"\n  Factor: {factor}")
        wf_results = {}
        for sym in SYMBOLS:
            wf = run_walk_forward_factor(cfg_path, sym)
            wf_results[sym] = wf
            print(
                f"    {sym}: OOS+={wf['oos_positive']}/{wf['n_splits']}, "
                f"Avg OOS SR={wf['avg_oos_sharpe']:.2f}"
            )
            if wf.get("details"):
                for d in wf["details"]:
                    if "error" not in d:
                        print(
                            f"      Split {d['split']}: "
                            f"Ret {d['oos_return']*100:+.1f}%, SR {d['oos_sharpe']:.2f}"
                        )
        all_results[factor]["walk_forward"] = wf_results

    # ─── C. Correlation & Blend vs R2 ───
    print("\n" + "=" * 70)
    print("  C. Correlation & Blend Analysis vs R2")
    print("=" * 70)

    r2_returns = get_r2_portfolio_returns()
    if r2_returns is not None:
        # Also compute R2 full-period stats for reference
        r2_cum = (1 + r2_returns).cumprod()
        r2_total = r2_cum.iloc[-1] - 1
        r2_years = len(r2_returns) / (365.25 * 24)
        r2_cagr = (1 + r2_total) ** (1 / r2_years) - 1 if r2_years > 0 else 0
        r2_sharpe = r2_returns.mean() / r2_returns.std() * np.sqrt(8760)
        r2_mdd = (r2_cum / r2_cum.cummax() - 1).min()
        r2_calmar = r2_cagr / abs(r2_mdd) if abs(r2_mdd) > 1e-9 else 0

        print(
            f"\n  R2 Baseline: CAGR {r2_cagr*100:.1f}%, "
            f"SR {r2_sharpe:.2f}, MDD {r2_mdd*100:.1f}%, Calmar {r2_calmar:.2f}"
        )

        for factor in CONFIGS:
            port_ret = all_results[factor].get("_port_returns")
            if port_ret is None:
                print(f"\n  {factor}: No portfolio returns → skip blend")
                continue

            blend_results = compute_blend_analysis(port_ret, r2_returns, BLEND_WEIGHTS)
            all_results[factor]["blend"] = blend_results

            print(f"\n  {factor}:")
            corr = blend_results[0]["corr_vs_r2"]
            print(f"    Corr vs R2: {corr:.3f}")
            for b in blend_results:
                delta_sr = b["blend_sharpe"] - r2_sharpe
                delta_mdd = b["blend_max_dd"] - r2_mdd
                print(
                    f"    Weight {b['carry_weight']:.0%}: "
                    f"CAGR {b['blend_cagr']*100:.1f}%, "
                    f"SR {b['blend_sharpe']:.2f} (Δ{delta_sr:+.2f}), "
                    f"MDD {b['blend_max_dd']*100:.1f}% (Δ{delta_mdd*100:+.1f}pp), "
                    f"Calmar {b['blend_calmar']:.2f}"
                )
    else:
        print("  ⚠️  Could not load R2 portfolio returns — skipping blend analysis")

    # ─── D. Save Results ───
    print("\n" + "=" * 70)
    print("  D. Saving Results")
    print("=" * 70)

    # Serialize — remove non-JSON-serializable items
    save_results = {}
    for factor in CONFIGS:
        fr = all_results[factor]
        save_results[factor] = {
            "per_symbol": fr.get("per_symbol", {}),
            "portfolio": fr.get("portfolio", {}),
            "year_by_year": fr.get("year_by_year", []),
            "walk_forward": {
                sym: {
                    "oos_sharpes": wf["oos_sharpes"],
                    "avg_oos_sharpe": wf["avg_oos_sharpe"],
                    "oos_positive": wf["oos_positive"],
                    "n_splits": wf["n_splits"],
                    "details": wf.get("details", []),
                }
                for sym, wf in fr.get("walk_forward", {}).items()
            },
            "blend": fr.get("blend", []),
        }

    results_path = out_dir / "research_results.json"
    with open(results_path, "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"  ✅ Results: {results_path}")

    # Save portfolio equity CSVs
    for factor in CONFIGS:
        peq = all_results[factor].get("_port_equity")
        if peq is not None:
            eq_path = out_dir / f"{factor}_portfolio_equity.csv"
            peq.to_csv(eq_path)
            print(f"  ✅ Equity: {eq_path}")

    # ─── E. Summary Table ───
    print("\n" + "=" * 70)
    print("  E. SUMMARY")
    print("=" * 70)

    print(f"\n{'Factor':<6} | {'Cost':>5} | {'Return':>9} | {'CAGR':>7} | {'Sharpe':>7} | {'MaxDD':>7} | {'Calmar':>7}")
    print("-" * 65)
    for factor in CONFIGS:
        for cm_label in ["1.0x", "1.5x", "2.0x"]:
            port = all_results[factor].get("portfolio", {}).get(cm_label)
            if port:
                print(
                    f"{factor:<6} | {cm_label:>5} | "
                    f"{port['total_return']*100:>+8.1f}% | "
                    f"{port['cagr']*100:>6.1f}% | "
                    f"{port['sharpe']:>7.2f} | "
                    f"{port['max_dd']*100:>6.1f}% | "
                    f"{port['calmar']:>7.2f}"
                )

    print(f"\n{'Factor':<6} | {'Sym':>8} | {'OOS+':>5} | {'Avg OOS SR':>11}")
    print("-" * 45)
    for factor in CONFIGS:
        wf = all_results[factor].get("walk_forward", {})
        for sym in SYMBOLS:
            w = wf.get(sym, {})
            print(
                f"{factor:<6} | {sym:>8} | "
                f"{w.get('oos_positive', 0)}/{w.get('n_splits', 0):>3} | "
                f"{w.get('avg_oos_sharpe', 0):>11.2f}"
            )

    if r2_returns is not None:
        print(f"\n{'Factor':<6} | {'Corr':>6} | {'Weight':>7} | {'Blend SR':>9} | {'ΔSR':>6} | {'Blend MDD':>10} | {'ΔMDD':>7}")
        print("-" * 70)
        for factor in CONFIGS:
            blend = all_results[factor].get("blend", [])
            for b in blend:
                delta_sr = b["blend_sharpe"] - r2_sharpe
                delta_mdd = (b["blend_max_dd"] - r2_mdd) * 100
                print(
                    f"{factor:<6} | {b['corr_vs_r2']:>6.3f} | "
                    f"{b['carry_weight']:>6.0%} | "
                    f"{b['blend_sharpe']:>9.2f} | "
                    f"{delta_sr:>+5.2f} | "
                    f"{b['blend_max_dd']*100:>9.1f}% | "
                    f"{delta_mdd:>+6.1f}pp"
                )

    # ─── F. Acceptance Gate Check ───
    print("\n" + "=" * 70)
    print("  F. ACCEPTANCE GATE")
    print("=" * 70)

    for factor in CONFIGS:
        print(f"\n  {factor}:")
        port_10 = all_results[factor].get("portfolio", {}).get("1.0x", {})
        port_15 = all_results[factor].get("portfolio", {}).get("1.5x", {})
        wf = all_results[factor].get("walk_forward", {})
        blend = all_results[factor].get("blend", [])

        # Gate 1: OOS Sharpe > 0.5
        avg_oos = np.mean([
            wf.get(sym, {}).get("avg_oos_sharpe", 0) for sym in SYMBOLS
        ])
        g1 = avg_oos > 0.5
        print(f"    G1: Avg OOS Sharpe > 0.5  → {avg_oos:.2f}  {'✅ PASS' if g1 else '❌ FAIL'}")

        # Gate 2: cost_mult=1.5 positive return
        ret_15 = port_15.get("total_return", -1) if port_15 else -1
        g2 = ret_15 > 0
        print(f"    G2: 1.5x cost positive   → {ret_15*100:+.1f}%  {'✅ PASS' if g2 else '❌ FAIL'}")

        # Gate 3: |corr| < 0.5
        corr_val = blend[0]["corr_vs_r2"] if blend else 1.0
        g3 = abs(corr_val) < 0.5
        print(f"    G3: |corr vs R2| < 0.5   → {corr_val:.3f}  {'✅ PASS' if g3 else '❌ FAIL'}")

        # Gate 4: Blend improves SR or Calmar, MDD not +2pp worse
        g4 = False
        if blend and r2_returns is not None:
            for b in blend:
                if b["carry_weight"] <= 0.20:
                    delta_sr = b["blend_sharpe"] - r2_sharpe
                    delta_calmar = b["blend_calmar"] - r2_calmar
                    delta_mdd = (b["blend_max_dd"] - r2_mdd) * 100
                    if (delta_sr > 0 or delta_calmar > 0) and delta_mdd < 2.0:
                        g4 = True
                        break
        print(f"    G4: Blend improves & MDD ok → {'✅ PASS' if g4 else '❌ FAIL'}")

        overall = g1 and g2 and g3 and g4
        print(f"    OVERALL: {'✅ ALL PASS' if overall else '❌ FAIL'}")

    print(f"\n{'=' * 70}")
    print(f"  Evidence path: {out_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
