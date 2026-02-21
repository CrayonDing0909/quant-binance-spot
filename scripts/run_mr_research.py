#!/usr/bin/env python3
"""
MR Research Matrix — Phase MR-1

自動化研究 pipeline：
  1. 對 3 個 MR 變體 × 2 個 TF (5m/15m) × 2 個 symbol → 跑 full backtest
  2. 成本壓力測試：cost_mult = 1.0, 1.5, 2.0
  3. Walk-Forward 5 splits
  4. Holdout (最後 12 個月)
  5. 年別統計
  6. 與 R2 的融合測試 (correlation + weight sweep)

Usage:
    cd /Users/dylanting/Documents/spot_bot/quant-binance-spot
    source .venv/bin/activate
    PYTHONPATH=src python scripts/run_mr_research.py
"""
from __future__ import annotations

import json
import sys
import warnings
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Suppress vectorbt warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qtrade.config import load_config
from qtrade.backtest.run_backtest import run_symbol_backtest, BacktestResult
from qtrade.validation.walk_forward import walk_forward_analysis, walk_forward_summary
from qtrade.data.storage import load_klines


# ══════════════════════════════════════════════════════════════
#  研究矩陣配置
# ══════════════════════════════════════════════════════════════

MR_CONFIGS = {
    "MR-A_5m":  "config/research_mr_5m_A.yaml",
    "MR-A_15m": "config/research_mr_15m_A.yaml",
    "MR-B_5m":  "config/research_mr_5m_B.yaml",
    "MR-B_15m": "config/research_mr_15m_B.yaml",
    "MR-C_5m":  "config/research_mr_5m_C.yaml",
    "MR-C_15m": "config/research_mr_15m_C.yaml",
}

R2_CONFIG = "config/prod_candidate_R2.yaml"

COST_MULTS = [1.0, 1.5, 2.0]
WF_SPLITS = 5
HOLDOUT_MONTHS = 12  # last 12 months

BLEND_WEIGHTS = [0.10, 0.15, 0.20]  # MR weight in blend

# Evaluation thresholds
THRESH_OOS_SHARPE = 0.7
THRESH_COST15_POSITIVE = True
THRESH_COST20_SHARPE = 0.2
THRESH_WORST_YEAR = -20.0  # %


# ══════════════════════════════════════════════════════════════
#  Utilities
# ══════════════════════════════════════════════════════════════

def _run_backtest_with_cost_mult(
    cfg_path: str, symbol: str, cost_mult: float = 1.0,
) -> BacktestResult | None:
    """Run single backtest with optional cost multiplier."""
    try:
        cfg = load_config(cfg_path)
        bt_cfg = cfg.to_backtest_dict(symbol=symbol)
        strategy_name = cfg.strategy.name

        # Apply cost multiplier
        if cost_mult != 1.0:
            bt_cfg["fee_bps"] = bt_cfg["fee_bps"] * cost_mult
            bt_cfg["slippage_bps"] = bt_cfg["slippage_bps"] * cost_mult

        data_path = (
            cfg.data_dir / "binance" / cfg.market_type_str
            / cfg.market.interval / f"{symbol}.parquet"
        )
        if not data_path.exists():
            print(f"  ⚠️  Data not found: {data_path}")
            return None

        res = run_symbol_backtest(
            symbol, data_path, bt_cfg, strategy_name,
            data_dir=cfg.data_dir,
        )
        return res
    except Exception as e:
        print(f"  ❌ Backtest failed ({symbol}, cost={cost_mult}x): {e}")
        return None


def _extract_stats(res: BacktestResult) -> dict:
    """Extract key stats from BacktestResult."""
    adj = res.adjusted_stats
    stats = res.stats

    src = adj if adj else stats
    total_ret = src.get("Total Return [%]", 0)
    sharpe = src.get("Sharpe Ratio", 0)
    mdd = abs(src.get("Max Drawdown [%]", 0))
    total_trades = stats.get("Total Trades", 0)

    # CAGR calculation
    equity = res.equity()
    if len(equity) > 1:
        n_years = (equity.index[-1] - equity.index[0]).total_seconds() / (365.25 * 86400)
        if n_years > 0 and equity.iloc[0] > 0:
            cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1
        else:
            cagr = 0.0
    else:
        cagr = 0.0
        n_years = 0.0

    # Calmar
    calmar = cagr / (mdd / 100) if mdd > 0 else 0.0

    return {
        "total_return_pct": total_ret,
        "cagr_pct": cagr * 100,
        "sharpe": sharpe,
        "max_dd_pct": mdd,
        "calmar": calmar,
        "total_trades": total_trades,
        "n_years": n_years,
    }


def _yearly_stats(res: BacktestResult) -> pd.DataFrame:
    """Compute per-year returns and stats."""
    equity = res.equity()
    if len(equity) < 2:
        return pd.DataFrame()

    # Daily returns (resample to daily first)
    daily_eq = equity.resample("1D").last().dropna()
    daily_ret = daily_eq.pct_change().dropna()

    if daily_ret.empty:
        return pd.DataFrame()

    # Group by year
    yearly = daily_ret.groupby(daily_ret.index.year)
    rows = []
    for year, rets in yearly:
        annual_ret = (1 + rets).prod() - 1
        sharpe = rets.mean() / rets.std() * np.sqrt(365) if rets.std() > 0 else 0
        cum = (1 + rets).cumprod()
        peak = cum.cummax()
        dd = ((cum - peak) / peak).min() * 100  # negative
        rows.append({
            "year": int(year),
            "return_pct": annual_ret * 100,
            "sharpe": sharpe,
            "max_dd_pct": abs(dd),
        })
    return pd.DataFrame(rows)


def _holdout_backtest(
    cfg_path: str, symbol: str, holdout_months: int = 12,
) -> dict | None:
    """Run holdout backtest on last N months."""
    try:
        cfg = load_config(cfg_path)
        bt_cfg = cfg.to_backtest_dict(symbol=symbol)
        strategy_name = cfg.strategy.name

        data_path = (
            cfg.data_dir / "binance" / cfg.market_type_str
            / cfg.market.interval / f"{symbol}.parquet"
        )
        if not data_path.exists():
            return None

        df = load_klines(data_path)
        holdout_start = df.index[-1] - pd.DateOffset(months=holdout_months)
        bt_cfg["start"] = str(holdout_start)
        bt_cfg["end"] = str(df.index[-1])

        res = run_symbol_backtest(
            symbol, data_path, bt_cfg, strategy_name,
            data_dir=cfg.data_dir,
        )
        return _extract_stats(res)
    except Exception as e:
        print(f"  ❌ Holdout failed ({symbol}): {e}")
        return None


def _walk_forward(cfg_path: str, symbol: str, n_splits: int = 5) -> dict | None:
    """Run walk-forward analysis."""
    try:
        cfg = load_config(cfg_path)
        bt_cfg = cfg.to_backtest_dict(symbol=symbol)
        strategy_name = cfg.strategy.name
        bt_cfg["strategy_name"] = strategy_name

        data_path = (
            cfg.data_dir / "binance" / cfg.market_type_str
            / cfg.market.interval / f"{symbol}.parquet"
        )
        if not data_path.exists():
            return None

        wf_df = walk_forward_analysis(
            symbol=symbol,
            data_path=data_path,
            cfg=bt_cfg,
            n_splits=n_splits,
            data_dir=cfg.data_dir,
        )
        if wf_df.empty:
            return None

        summary = walk_forward_summary(wf_df)
        return {
            "avg_oos_sharpe": summary.get("avg_test_sharpe", 0),
            "oos_positive_pct": summary.get("oos_positive_pct", 0),
            "sharpe_degradation_pct": summary.get("sharpe_degradation_pct", 0),
            "is_robust": summary.get("is_robust", False),
            "worst_test_sharpe": summary.get("worst_test_sharpe", 0),
            "wf_df": wf_df,
        }
    except Exception as e:
        print(f"  ❌ Walk-Forward failed ({symbol}): {e}")
        return None


# ══════════════════════════════════════════════════════════════
#  R2 Blend Analysis
# ══════════════════════════════════════════════════════════════

def _get_r2_equity(symbol: str) -> pd.Series | None:
    """Get R2 portfolio equity for a single symbol."""
    try:
        cfg = load_config(R2_CONFIG)
        bt_cfg = cfg.to_backtest_dict(symbol=symbol)

        # Check ensemble override
        import yaml
        with open(R2_CONFIG, "r") as f:
            raw = yaml.safe_load(f)
        ens = raw.get("ensemble", {})
        if ens.get("enabled", False):
            strategies = ens.get("strategies", {})
            if symbol in strategies:
                s = strategies[symbol]
                bt_cfg["strategy_params"] = s.get("params", {})
                strategy_name = s["name"]
            else:
                strategy_name = cfg.strategy.name
        else:
            strategy_name = cfg.strategy.name

        data_path = (
            cfg.data_dir / "binance" / cfg.market_type_str
            / cfg.market.interval / f"{symbol}.parquet"
        )
        if not data_path.exists():
            print(f"  ⚠️  R2 data not found: {data_path}")
            return None

        res = run_symbol_backtest(
            symbol, data_path, bt_cfg, strategy_name,
            data_dir=cfg.data_dir,
        )
        return res.equity()
    except Exception as e:
        print(f"  ❌ R2 equity failed ({symbol}): {e}")
        return None


def _blend_analysis(
    mr_equity: pd.Series,
    r2_equity: pd.Series,
    mr_weight: float,
) -> dict:
    """
    Compute blend stats for MR + R2.

    mr_equity & r2_equity are normalized to start at 1.0,
    then blended: blend = mr_weight * mr + (1-mr_weight) * r2
    """
    # Normalize to returns → blend returns → reconstruct equity
    # Align indices
    common = mr_equity.index.intersection(r2_equity.index)
    if len(common) < 100:
        return {"sharpe": 0, "cagr_pct": 0, "max_dd_pct": 0, "calmar": 0}

    mr_eq = mr_equity.reindex(common).ffill().dropna()
    r2_eq = r2_equity.reindex(common).ffill().dropna()

    # Normalize
    mr_norm = mr_eq / mr_eq.iloc[0]
    r2_norm = r2_eq / r2_eq.iloc[0]

    mr_ret = mr_norm.pct_change().fillna(0)
    r2_ret = r2_norm.pct_change().fillna(0)

    blend_ret = mr_weight * mr_ret + (1 - mr_weight) * r2_ret
    blend_eq = (1 + blend_ret).cumprod()

    n_years = (blend_eq.index[-1] - blend_eq.index[0]).total_seconds() / (365.25 * 86400)
    if n_years > 0:
        cagr = blend_eq.iloc[-1] ** (1 / n_years) - 1
    else:
        cagr = 0

    # Daily resampling for Sharpe
    daily_ret = blend_ret.resample("1D").sum().dropna()
    sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(365) if daily_ret.std() > 0 else 0

    # MaxDD
    peak = blend_eq.cummax()
    dd = ((blend_eq - peak) / peak).min() * 100
    max_dd = abs(dd)

    calmar = cagr / (max_dd / 100) if max_dd > 0 else 0

    return {
        "sharpe": sharpe,
        "cagr_pct": cagr * 100,
        "max_dd_pct": max_dd,
        "calmar": calmar,
    }


# ══════════════════════════════════════════════════════════════
#  Main Research Pipeline
# ══════════════════════════════════════════════════════════════

def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = Path("reports") / "mr_research" / ts
    report_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*70}")
    print(f"  MR Research Matrix — Phase MR-1")
    print(f"  {ts}")
    print(f"  Output: {report_dir}")
    print(f"{'='*70}\n")

    # ─── 1. Full Backtest Matrix ─────────────────────────────
    print("=" * 70)
    print("  STEP 1: Full Backtest Matrix (MR variants × TF × Symbol × Cost)")
    print("=" * 70)

    all_results = []
    mr_equities = {}  # {variant_label: {symbol: equity_series}}

    for variant, cfg_path in MR_CONFIGS.items():
        cfg = load_config(cfg_path)
        symbols = cfg.market.symbols
        tf = cfg.market.interval

        for sym in symbols:
            for cm in COST_MULTS:
                label = f"{variant}_{sym}_cost{cm:.1f}"
                print(f"\n  ▶ {label} ...", end="", flush=True)
                res = _run_backtest_with_cost_mult(cfg_path, sym, cm)
                if res is None:
                    print(" SKIP")
                    all_results.append({
                        "variant": variant, "tf": tf, "symbol": sym,
                        "cost_mult": cm, "status": "FAIL",
                    })
                    continue

                st = _extract_stats(res)
                st.update({
                    "variant": variant, "tf": tf, "symbol": sym,
                    "cost_mult": cm, "status": "OK",
                })
                all_results.append(st)
                print(f" SR={st['sharpe']:.2f} Ret={st['total_return_pct']:.1f}% MDD={st['max_dd_pct']:.1f}%")

                # Store 1.0x equity for blend analysis later
                if cm == 1.0:
                    mr_equities.setdefault(variant, {})[sym] = res.equity()

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(report_dir / "mr_backtest_matrix.csv", index=False)

    # ─── 2. Year-by-Year Table ───────────────────────────────
    print(f"\n{'='*70}")
    print("  STEP 2: Year-by-Year Statistics")
    print("=" * 70)

    yearly_rows = []
    for variant, cfg_path in MR_CONFIGS.items():
        cfg = load_config(cfg_path)
        for sym in cfg.market.symbols:
            res = _run_backtest_with_cost_mult(cfg_path, sym, 1.0)
            if res is None:
                continue
            ydf = _yearly_stats(res)
            if ydf.empty:
                continue
            ydf["variant"] = variant
            ydf["symbol"] = sym
            yearly_rows.append(ydf)
            for _, row in ydf.iterrows():
                print(f"  {variant} {sym} {int(row['year'])}: "
                      f"Ret={row['return_pct']:+.1f}% SR={row['sharpe']:.2f} MDD={row['max_dd_pct']:.1f}%")

    if yearly_rows:
        yearly_df = pd.concat(yearly_rows, ignore_index=True)
        yearly_df.to_csv(report_dir / "mr_yearly_stats.csv", index=False)
    else:
        yearly_df = pd.DataFrame()

    # ─── 3. Walk-Forward ─────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  STEP 3: Walk-Forward Analysis ({WF_SPLITS} splits)")
    print("=" * 70)

    wf_rows = []
    for variant, cfg_path in MR_CONFIGS.items():
        cfg = load_config(cfg_path)
        for sym in cfg.market.symbols:
            print(f"\n  ▶ WF {variant} {sym}...")
            wf_res = _walk_forward(cfg_path, sym, WF_SPLITS)
            if wf_res is None:
                wf_rows.append({
                    "variant": variant, "symbol": sym,
                    "avg_oos_sharpe": np.nan, "status": "FAIL",
                })
                continue

            wf_rows.append({
                "variant": variant, "symbol": sym,
                "avg_oos_sharpe": wf_res["avg_oos_sharpe"],
                "oos_positive_pct": wf_res["oos_positive_pct"],
                "sharpe_degradation_pct": wf_res["sharpe_degradation_pct"],
                "worst_test_sharpe": wf_res["worst_test_sharpe"],
                "is_robust": wf_res["is_robust"],
                "status": "OK",
            })

            # Save individual WF detail
            if wf_res.get("wf_df") is not None:
                wf_res["wf_df"].to_csv(
                    report_dir / f"wf_detail_{variant}_{sym}.csv", index=False,
                )

            print(f"    OOS SR={wf_res['avg_oos_sharpe']:.2f}, "
                  f"positive={wf_res['oos_positive_pct']:.0f}%, "
                  f"degradation={wf_res['sharpe_degradation_pct']:.1f}%")

    wf_df = pd.DataFrame(wf_rows)
    wf_df.to_csv(report_dir / "mr_walk_forward.csv", index=False)

    # ─── 4. Holdout (Last 12 months) ────────────────────────
    print(f"\n{'='*70}")
    print(f"  STEP 4: Holdout Test (Last {HOLDOUT_MONTHS} Months)")
    print("=" * 70)

    holdout_rows = []
    for variant, cfg_path in MR_CONFIGS.items():
        cfg = load_config(cfg_path)
        for sym in cfg.market.symbols:
            print(f"  ▶ Holdout {variant} {sym}...", end="", flush=True)
            ho = _holdout_backtest(cfg_path, sym, HOLDOUT_MONTHS)
            if ho is None:
                holdout_rows.append({
                    "variant": variant, "symbol": sym, "status": "FAIL",
                })
                print(" FAIL")
                continue

            ho.update({"variant": variant, "symbol": sym, "status": "OK"})
            holdout_rows.append(ho)
            print(f" SR={ho['sharpe']:.2f} Ret={ho['total_return_pct']:.1f}%")

    holdout_df = pd.DataFrame(holdout_rows)
    holdout_df.to_csv(report_dir / "mr_holdout.csv", index=False)

    # ─── 5. R2 Blend Analysis ────────────────────────────────
    print(f"\n{'='*70}")
    print("  STEP 5: R2 Blend Analysis")
    print("=" * 70)

    # Get R2 equities (1h data)
    r2_cfg = load_config(R2_CONFIG)
    r2_equities = {}
    r2_baseline_stats = {}
    for sym in r2_cfg.market.symbols:
        print(f"  Loading R2 equity for {sym}...")
        eq = _get_r2_equity(sym)
        if eq is not None:
            r2_equities[sym] = eq

    # Compute R2 baseline (equal-weight portfolio)
    if r2_equities:
        # Build R2 portfolio equity
        r2_rets = {}
        common_idx = None
        for sym, eq in r2_equities.items():
            norm = eq / eq.iloc[0]
            ret = norm.pct_change().fillna(0)
            # Resample to daily for correlation
            r2_rets[sym] = ret.resample("1D").sum()
            idx = r2_rets[sym].index
            common_idx = idx if common_idx is None else common_idx.intersection(idx)

        if common_idx is not None and len(common_idx) > 30:
            # Equal-weight R2 portfolio
            r2_weights = r2_cfg.portfolio.allocation or {s: 1/len(r2_equities) for s in r2_equities}
            r2_port_ret = sum(
                r2_rets[s].reindex(common_idx).fillna(0) * r2_weights.get(s, 0)
                for s in r2_equities
            )
            r2_port_eq = (1 + r2_port_ret).cumprod()
            n_yr = (common_idx[-1] - common_idx[0]).days / 365.25
            r2_cagr = (r2_port_eq.iloc[-1] ** (1/n_yr) - 1) * 100 if n_yr > 0 else 0
            r2_sr = r2_port_ret.mean() / r2_port_ret.std() * np.sqrt(365) if r2_port_ret.std() > 0 else 0
            r2_peak = r2_port_eq.cummax()
            r2_mdd = abs(((r2_port_eq - r2_peak)/r2_peak).min() * 100)
            r2_calmar = (r2_cagr/100) / (r2_mdd/100) if r2_mdd > 0 else 0

            r2_baseline_stats = {
                "cagr_pct": r2_cagr,
                "sharpe": r2_sr,
                "max_dd_pct": r2_mdd,
                "calmar": r2_calmar,
            }
            print(f"\n  R2 Baseline: CAGR={r2_cagr:.1f}% SR={r2_sr:.2f} MDD={r2_mdd:.1f}% Calmar={r2_calmar:.2f}")

    blend_rows = []
    for variant in MR_CONFIGS:
        mr_eqs = mr_equities.get(variant, {})
        if not mr_eqs:
            continue

        # Build MR daily returns for correlation
        mr_daily_rets = {}
        for sym, eq in mr_eqs.items():
            norm = eq / eq.iloc[0]
            mr_daily_rets[sym] = norm.pct_change().fillna(0).resample("1D").sum()

        # Correlation with R2 per-symbol
        corr_vals = []
        for sym in mr_daily_rets:
            if sym in r2_rets:
                ci = mr_daily_rets[sym].index.intersection(r2_rets[sym].index)
                if len(ci) > 30:
                    c = mr_daily_rets[sym].reindex(ci).corr(r2_rets[sym].reindex(ci))
                    corr_vals.append(c)

        avg_corr = np.mean(corr_vals) if corr_vals else np.nan

        # For blend: build a simple equal-weight MR portfolio
        mr_syms = list(mr_eqs.keys())
        mr_w = 1 / len(mr_syms) if mr_syms else 0
        mr_port_daily = sum(
            mr_daily_rets[s] * mr_w for s in mr_syms if s in mr_daily_rets
        )

        for bw in BLEND_WEIGHTS:
            # r2_port_ret is daily, mr_port_daily is daily
            # Align
            if 'r2_port_ret' not in dir():
                continue
            ci2 = mr_port_daily.index.intersection(r2_port_ret.index)
            if len(ci2) < 30:
                continue

            mr_r = mr_port_daily.reindex(ci2).fillna(0)
            r2_r = r2_port_ret.reindex(ci2).fillna(0)
            blend_r = bw * mr_r + (1 - bw) * r2_r
            blend_eq = (1 + blend_r).cumprod()

            n_yr2 = (ci2[-1] - ci2[0]).days / 365.25
            b_cagr = (blend_eq.iloc[-1] ** (1/n_yr2) - 1) * 100 if n_yr2 > 0 else 0
            b_sr = blend_r.mean() / blend_r.std() * np.sqrt(365) if blend_r.std() > 0 else 0
            b_peak = blend_eq.cummax()
            b_mdd = abs(((blend_eq - b_peak)/b_peak).min() * 100)
            b_calmar = (b_cagr/100)/(b_mdd/100) if b_mdd > 0 else 0

            # Verdict
            verdict = "PASS" if (
                (b_sr > r2_baseline_stats.get("sharpe", 0) or b_calmar > r2_baseline_stats.get("calmar", 0))
                and b_mdd <= r2_baseline_stats.get("max_dd_pct", 100) + 3.0
            ) else "FAIL"

            blend_rows.append({
                "variant": variant,
                "corr_vs_r2": avg_corr,
                "mr_weight": bw,
                "blend_cagr_pct": b_cagr,
                "blend_sharpe": b_sr,
                "blend_mdd_pct": b_mdd,
                "blend_calmar": b_calmar,
                "r2_cagr_pct": r2_baseline_stats.get("cagr_pct", 0),
                "r2_sharpe": r2_baseline_stats.get("sharpe", 0),
                "r2_mdd_pct": r2_baseline_stats.get("max_dd_pct", 0),
                "verdict": verdict,
            })

    blend_df = pd.DataFrame(blend_rows)
    blend_df.to_csv(report_dir / "mr_r2_blend.csv", index=False)

    # ══════════════════════════════════════════════════════════
    #  Print Summary Tables
    # ══════════════════════════════════════════════════════════

    print(f"\n\n{'='*70}")
    print("  ═══════════════════════════════════")
    print("    MR RESEARCH RESULTS SUMMARY")
    print("  ═══════════════════════════════════")
    print(f"{'='*70}")

    # ── 2) MR Single-Strategy Table ──
    print("\n\n  ─── MR Single-Strategy Table ───")
    header = f"{'Variant':<12} {'TF':<5} {'Symbol':<10} {'Cost':<5} {'TotRet%':>8} {'CAGR%':>7} {'Sharpe':>7} {'MDD%':>6} {'Trades':>7}"
    print(f"  {header}")
    print(f"  {'─'*len(header)}")
    for _, r in results_df.iterrows():
        if r.get("status") == "FAIL":
            print(f"  {r['variant']:<12} {r['tf']:<5} {r['symbol']:<10} {r['cost_mult']:<5.1f} {'FAIL':>8}")
            continue
        print(
            f"  {r['variant']:<12} {r.get('tf',''):<5} {r['symbol']:<10} {r['cost_mult']:<5.1f} "
            f"{r['total_return_pct']:>8.1f} {r.get('cagr_pct',0):>7.1f} {r['sharpe']:>7.2f} "
            f"{r['max_dd_pct']:>6.1f} {r.get('total_trades',0):>7.0f}"
        )

    # ── 3) Year-by-Year Table ──
    if not yearly_df.empty:
        print("\n\n  ─── Year-by-Year Table ───")
        header = f"{'Variant':<12} {'Symbol':<10} {'Year':<5} {'Return%':>8} {'Sharpe':>7} {'MDD%':>6}"
        print(f"  {header}")
        print(f"  {'─'*len(header)}")
        for _, r in yearly_df.iterrows():
            print(
                f"  {r['variant']:<12} {r['symbol']:<10} {int(r['year']):<5} "
                f"{r['return_pct']:>8.1f} {r['sharpe']:>7.2f} {r['max_dd_pct']:>6.1f}"
            )

    # ── WF Summary ──
    if not wf_df.empty:
        print("\n\n  ─── Walk-Forward Summary ───")
        header = f"{'Variant':<12} {'Symbol':<10} {'OOS SR':>7} {'OOS+%':>6} {'Degrad%':>8} {'Robust':>7}"
        print(f"  {header}")
        print(f"  {'─'*len(header)}")
        for _, r in wf_df.iterrows():
            if r.get("status") == "FAIL":
                print(f"  {r['variant']:<12} {r['symbol']:<10} {'FAIL':>7}")
                continue
            print(
                f"  {r['variant']:<12} {r['symbol']:<10} "
                f"{r['avg_oos_sharpe']:>7.2f} {r['oos_positive_pct']:>5.0f}% "
                f"{r['sharpe_degradation_pct']:>7.1f}% {'✓' if r['is_robust'] else '✗':>7}"
            )

    # ── Holdout ──
    if not holdout_df.empty:
        print(f"\n\n  ─── Holdout ({HOLDOUT_MONTHS}m) ───")
        header = f"{'Variant':<12} {'Symbol':<10} {'Return%':>8} {'Sharpe':>7} {'MDD%':>6}"
        print(f"  {header}")
        print(f"  {'─'*len(header)}")
        for _, r in holdout_df.iterrows():
            if r.get("status") == "FAIL":
                print(f"  {r['variant']:<12} {r['symbol']:<10} {'FAIL':>8}")
                continue
            print(
                f"  {r['variant']:<12} {r['symbol']:<10} "
                f"{r.get('total_return_pct', 0):>8.1f} {r.get('sharpe', 0):>7.2f} "
                f"{r.get('max_dd_pct', 0):>6.1f}"
            )

    # ── 4) R2 Blend Table ──
    if not blend_df.empty:
        print("\n\n  ─── R2 Blend Table ───")
        print(f"  R2 Baseline: CAGR={r2_baseline_stats.get('cagr_pct',0):.1f}% "
              f"SR={r2_baseline_stats.get('sharpe',0):.2f} "
              f"MDD={r2_baseline_stats.get('max_dd_pct',0):.1f}% "
              f"Calmar={r2_baseline_stats.get('calmar',0):.2f}")
        header = (f"{'Variant':<12} {'CorrR2':>7} {'MR Wt':>6} {'CAGR%':>7} "
                  f"{'Sharpe':>7} {'MDD%':>6} {'Calmar':>7} {'Verdict':>8}")
        print(f"  {header}")
        print(f"  {'─'*len(header)}")
        for _, r in blend_df.iterrows():
            print(
                f"  {r['variant']:<12} {r['corr_vs_r2']:>7.2f} {r['mr_weight']:>5.0%} "
                f"{r['blend_cagr_pct']:>7.1f} {r['blend_sharpe']:>7.2f} "
                f"{r['blend_mdd_pct']:>6.1f} {r['blend_calmar']:>7.2f} {r['verdict']:>8}"
            )

    # ─── 5) Failure Analysis ────────────────────────────────
    print("\n\n  ─── Failure Analysis ───")
    for variant in MR_CONFIGS:
        # Check cost stress
        v_results = results_df[results_df["variant"] == variant]
        if v_results.empty:
            print(f"  {variant}: No results (data missing?)")
            continue

        base = v_results[v_results["cost_mult"] == 1.0]
        c15 = v_results[v_results["cost_mult"] == 1.5]
        c20 = v_results[v_results["cost_mult"] == 2.0]

        issues = []

        # Check base Sharpe
        avg_base_sr = base["sharpe"].mean() if not base.empty else 0
        if avg_base_sr < THRESH_OOS_SHARPE:
            issues.append(f"base SR={avg_base_sr:.2f} < {THRESH_OOS_SHARPE}")

        # Check 1.5x cost
        if not c15.empty:
            avg_c15_ret = c15["total_return_pct"].mean()
            if avg_c15_ret <= 0:
                issues.append(f"cost 1.5x avg return={avg_c15_ret:.1f}% ≤ 0")

        # Check 2.0x cost
        if not c20.empty:
            avg_c20_sr = c20["sharpe"].mean()
            if avg_c20_sr < THRESH_COST20_SHARPE:
                issues.append(f"cost 2.0x SR={avg_c20_sr:.2f} < {THRESH_COST20_SHARPE}")

        # Check worst year
        if not yearly_df.empty:
            v_yearly = yearly_df[yearly_df["variant"] == variant]
            if not v_yearly.empty:
                worst = v_yearly["return_pct"].min()
                if worst < THRESH_WORST_YEAR:
                    issues.append(f"worst year={worst:.1f}% < {THRESH_WORST_YEAR}%")

        if issues:
            print(f"  {variant}: FAIL — {'; '.join(issues)}")
        else:
            print(f"  {variant}: PASS — all thresholds met")

    # ─── 6) Final Recommendation ─────────────────────────────
    print(f"\n\n  ─── Final Recommendation ───")

    # Score each variant
    candidates = []
    for variant in MR_CONFIGS:
        v_res = results_df[(results_df["variant"] == variant) & (results_df["cost_mult"] == 1.0)]
        if v_res.empty or (v_res.get("status") == "FAIL").any():
            continue

        avg_sr = v_res["sharpe"].mean()
        avg_ret = v_res["total_return_pct"].mean()

        # Cost 1.5x check
        v_c15 = results_df[(results_df["variant"] == variant) & (results_df["cost_mult"] == 1.5)]
        c15_ok = v_c15["total_return_pct"].mean() > 0 if not v_c15.empty else False

        # WF check
        v_wf = wf_df[wf_df["variant"] == variant] if not wf_df.empty else pd.DataFrame()
        wf_ok = v_wf["avg_oos_sharpe"].mean() > 0 if not v_wf.empty else False

        score = avg_sr * 2 + (1 if c15_ok else 0) + (1 if wf_ok else 0)
        candidates.append({
            "variant": variant,
            "avg_sharpe": avg_sr,
            "avg_return": avg_ret,
            "cost15_ok": c15_ok,
            "wf_ok": wf_ok,
            "score": score,
        })

    if candidates:
        candidates.sort(key=lambda x: x["score"], reverse=True)
        best = candidates[0]
        print(f"\n  🏆 Top-1 Candidate: {best['variant']}")
        print(f"     Avg Sharpe: {best['avg_sharpe']:.2f}")
        print(f"     Avg Return: {best['avg_return']:.1f}%")
        print(f"     Cost 1.5x OK: {'✓' if best['cost15_ok'] else '✗'}")
        print(f"     Walk-Forward OK: {'✓' if best['wf_ok'] else '✗'}")

        if best["avg_sharpe"] >= THRESH_OOS_SHARPE and best["cost15_ok"]:
            print(f"\n  ✅ RECOMMENDATION: {best['variant']} → paper trading Phase MR-2")
        else:
            print(f"\n  ⚠️  CONDITIONAL: {best['variant']} shows promise but needs parameter tuning")
            print(f"     Next step: deeper parameter search on {best['variant']}")
    else:
        print("\n  ❌ FAIL — No MR variant passes all thresholds.")
        print("     Possible causes:")
        print("       - MR strategies overtrading at 5m/15m → costs eat alpha")
        print("       - Trend-dominated market regimes since 2022")
        print("     Next steps:")
        print("       1. Try 1h/4h timeframes (lower turnover)")
        print("       2. Increase z_enter / bb_std thresholds (fewer trades)")
        print("       3. Add stronger vol filter (avoid trend periods)")
        print("       4. Test asymmetric MR (long-only in uptrend, skip shorts)")

    # ─── 7) Evidence Paths ───────────────────────────────────
    print(f"\n\n  ─── Evidence Paths ───")
    print(f"  {report_dir}/mr_backtest_matrix.csv")
    print(f"  {report_dir}/mr_yearly_stats.csv")
    print(f"  {report_dir}/mr_walk_forward.csv")
    print(f"  {report_dir}/mr_holdout.csv")
    print(f"  {report_dir}/mr_r2_blend.csv")
    for f in sorted(report_dir.glob("wf_detail_*.csv")):
        print(f"  {f}")

    # Save summary JSON
    summary = {
        "timestamp": ts,
        "configs": MR_CONFIGS,
        "cost_multipliers": COST_MULTS,
        "wf_splits": WF_SPLITS,
        "holdout_months": HOLDOUT_MONTHS,
        "r2_baseline": r2_baseline_stats if r2_baseline_stats else {},
        "candidates": candidates if candidates else [],
    }
    with open(report_dir / "research_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"  MR Research Complete — {report_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
