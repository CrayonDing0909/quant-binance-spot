#!/usr/bin/env python3
"""
Mini Rerun — Portfolio-Level Equity Aggregation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Runs 4 configs on 10-symbol and 19-symbol universes using
portfolio-level equity aggregation (NOT avg-of-metrics).

Configs:
  B0_baseline        — naked TSMOM-EMA
  P0_parity          — rb=3%, fg=80%, mt=2%
  B_ema4_rb3_fg80    — EMA(4) + rb=3%, fg=80%
  B_ema3_rb4_fg80    — EMA(3) + rb=4%, fg=80%

Portfolio equity = initial_cash × (1 + Σ wᵢ rᵢ(t)).cumprod()
  where rᵢ(t) = per-symbol funding-adjusted equity pct_change()
  and wᵢ = vol-parity weight normalised to sum=1.

Usage:
  cd /Users/dylanting/Documents/spot_bot/quant-binance-spot
  source .venv/bin/activate
  PYTHONPATH=src python scripts/mini_rerun_portfolio.py 2>&1
"""
from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from qtrade.data.storage import load_klines
from qtrade.data.quality import clean_data
from qtrade.data.funding_rate import (
    load_funding_rates,
    get_funding_rate_path,
    align_funding_to_klines,
)
from qtrade.strategy.base import StrategyContext
from qtrade.strategy import get_strategy
from qtrade.strategy.exit_rules import apply_exit_rules
from qtrade.backtest.run_backtest import clip_positions_by_direction
from qtrade.backtest.costs import (
    compute_funding_costs,
    adjust_equity_for_funding,
)

try:
    import vectorbt as vbt
except ImportError:
    sys.exit("vectorbt required")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ══════════════════════════════════════════════════════════════
#  Constants
# ══════════════════════════════════════════════════════════════

DATA_DIR = Path("data")
OUT_ROOT = Path("reports/research/churn_cost_optimization")

BASE_CFG = dict(
    strategy_name="tsmom_ema",
    strategy_params=dict(
        lookback=168, vol_target=0.15,
        ema_fast=20, ema_slow=50,
        agree_weight=1.0, disagree_weight=0.3,
    ),
    initial_cash=10_000,
    fee_bps=5.0,
    slippage_bps=3.0,
    interval="1h",
    market_type="futures",
    direction="both",
    trade_on="next_open",
    leverage=3,
)

PERIODS = {
    "IS":   ("2022-01-01", "2024-06-30"),
    "OOS":  ("2024-07-01", "2025-06-30"),
    "Live": ("2025-07-01", None),
    "Full": ("2022-01-01", None),
}

# 10-symbol universe (prod live R3C)
SYMS_10 = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT",
]

# 19-symbol universe (historical R3C)
SYMS_19 = SYMS_10 + [
    "DOTUSDT", "NEARUSDT", "APTUSDT", "ARBUSDT", "OPUSDT",
    "FILUSDT", "ATOMUSDT", "UNIUSDT", "AAVEUSDT",
]

# Vol-parity base weights from prod_candidate_R3C_universe.yaml
_W19 = {
    "BTCUSDT": 0.0722, "ETHUSDT": 0.0538, "SOLUSDT": 0.0509,
    "BNBUSDT": 0.0707, "XRPUSDT": 0.0491, "DOGEUSDT": 0.0512,
    "ADAUSDT": 0.0511, "AVAXUSDT": 0.0545, "LINKUSDT": 0.0538,
    "DOTUSDT": 0.0540, "LTCUSDT": 0.0605, "NEARUSDT": 0.0495,
    "APTUSDT": 0.0483, "ARBUSDT": 0.0495, "OPUSDT": 0.0426,
    "FILUSDT": 0.0489, "ATOMUSDT": 0.0523, "UNIUSDT": 0.0394,
    "AAVEUSDT": 0.0477,
}


def get_weights(symbols: list[str]) -> dict[str, float]:
    """Return normalized weights (sum=1) for the given symbol list."""
    raw = {s: _W19[s] for s in symbols}
    total = sum(raw.values())
    return {s: w / total for s, w in raw.items()}


# ══════════════════════════════════════════════════════════════
#  Config
# ══════════════════════════════════════════════════════════════

@dataclass
class ECfg:
    name: str
    rb: float = 0.0
    fg: float = 1.0
    mt: float = 0.0
    ema: int = 0
    sl: float | None = None
    cd: int = 0


CONFIGS = [
    ECfg("B0_baseline"),
    ECfg("P0_parity", rb=0.03, fg=0.80, mt=0.02),
    ECfg("B_ema4_rb3_fg80", rb=0.03, fg=0.80, ema=4),
    ECfg("B_ema3_rb4_fg80", rb=0.04, fg=0.80, ema=3),
]


# ══════════════════════════════════════════════════════════════
#  Signal processing (same as research script)
# ══════════════════════════════════════════════════════════════

def apply_gates(raw_pos: pd.Series, rb: float, fg: float, mt: float) -> pd.Series:
    vals = raw_pos.values.astype(np.float64)
    out = np.empty(len(vals), dtype=np.float64)
    prev = 0.0
    for i in range(len(vals)):
        tgt = vals[i]
        diff = abs(tgt - prev)
        same = (tgt > 0 and prev > 0) or (tgt < 0 and prev < 0)
        flip = (tgt > 0 and prev < 0) or (tgt < 0 and prev > 0)
        ok = True
        if same and fg < 1.0 and tgt != 0.0:
            if abs(prev / tgt) >= fg:
                ok = False
        if ok and rb > 0 and not flip and diff < rb:
            ok = False
        if ok and mt > 0 and not flip and diff < mt:
            ok = False
        out[i] = tgt if ok else prev
        prev = out[i]
    return pd.Series(out, index=raw_pos.index)


def smooth_signal(pos: pd.Series, span: int) -> pd.Series:
    return pos.ewm(span=span, adjust=False).mean().clip(-1.0, 1.0)


def disaster_sl(
    df: pd.DataFrame, pos: pd.Series, sl_atr: float, cd: int,
) -> tuple[pd.Series, pd.Series]:
    binary = pd.Series(np.sign(pos.values).astype(float), index=pos.index)
    sl_pos, ep = apply_exit_rules(
        df, binary,
        stop_loss_atr=sl_atr, take_profit_atr=None,
        trailing_stop_atr=None, cooldown_bars=cd,
    )
    result = pos.copy()
    forced_flat = sl_pos.abs() < 0.01
    result[forced_flat] = 0.0
    return result, ep


# ══════════════════════════════════════════════════════════════
#  Per-symbol equity curve
# ══════════════════════════════════════════════════════════════

def run_sym_equity(
    sym: str,
    df_full: pd.DataFrame,
    ec: ECfg,
    p_start: str | None,
    p_end: str | None,
) -> tuple[pd.Series, float, float, float] | None:
    """Run one backtest, return (equity_curve, total_fees, funding_pct, turnover).

    Returns None on failure or insufficient data.
    """
    try:
        df = df_full.copy()

        # Strategy signal (signal_delay=1)
        ctx = StrategyContext(
            symbol=sym, interval="1h",
            market_type="futures", direction="both", signal_delay=1,
        )
        pos = get_strategy(BASE_CFG["strategy_name"])(df, ctx, BASE_CFG["strategy_params"])

        # Signal smoothing
        if ec.ema > 0:
            pos = smooth_signal(pos, ec.ema)

        # Disaster SL
        ep_sl = pd.Series(np.nan, index=df.index)
        if ec.sl is not None:
            pos, ep_sl = disaster_sl(df, pos, ec.sl, ec.cd)

        # Direction clip
        pos = clip_positions_by_direction(pos, "futures", "both")

        # Gates
        if ec.rb > 0 or ec.fg < 1.0 or ec.mt > 0:
            pos = apply_gates(pos, ec.rb, ec.fg, ec.mt)

        # Date filter
        if p_start:
            ts = pd.Timestamp(p_start, tz="UTC") if df.index.tz else pd.Timestamp(p_start)
            df = df[df.index >= ts]
        if p_end:
            ts = pd.Timestamp(p_end, tz="UTC") if df.index.tz else pd.Timestamp(p_end)
            df = df[df.index <= ts]
        pos = pos.reindex(df.index).fillna(0.0)
        ep_sl = ep_sl.reindex(df.index)

        if len(df) < 100:
            return None

        # Execution price
        open_ = df["open"]
        close = df["close"]
        exec_price = open_.copy()
        sl_mask = ep_sl.notna()
        if sl_mask.any():
            exec_price[sl_mask] = ep_sl[sl_mask]

        # VBT Portfolio
        pf = vbt.Portfolio.from_orders(
            close=close,
            size=pos,
            size_type="targetpercent",
            price=exec_price,
            fees=BASE_CFG["fee_bps"] / 10_000,
            slippage=BASE_CFG["slippage_bps"] / 10_000,
            init_cash=BASE_CFG["initial_cash"],
            freq="1h",
            direction="both",
        )

        # Funding cost
        fr_path = get_funding_rate_path(DATA_DIR, sym)
        fdf = load_funding_rates(fr_path)
        fr = align_funding_to_klines(fdf, df.index, default_rate_8h=0.0001)
        eq_raw = pf.value()
        fc = compute_funding_costs(
            pos=pos, equity=eq_raw,
            funding_rates=fr, leverage=BASE_CFG["leverage"],
        )
        eq = adjust_equity_for_funding(eq_raw, fc)

        # Fees
        try:
            fees = float(pf.orders.records["fees"].sum())
        except Exception:
            fees = 0.0

        # Turnover
        n_yr = max(0.5, len(df) / 8760)
        turnover = float(pos.diff().abs().sum() / n_yr)

        return eq, fees, fc.total_cost_pct * 100, turnover

    except Exception as e:
        print(f"  ✗ {sym}/{ec.name}: {e}")
        return None


# ══════════════════════════════════════════════════════════════
#  Portfolio-level aggregation
# ══════════════════════════════════════════════════════════════

def portfolio_equity(
    eq_dict: dict[str, pd.Series],
    weights: dict[str, float],
    initial_cash: float = 10_000,
) -> pd.Series:
    """Combine per-symbol equity curves into portfolio equity using weighted returns.

    eq_dict: {symbol: equity_series} — each starts at initial_cash
    weights: {symbol: weight} — already normalised to sum=1
    """
    # Collect per-symbol returns
    rets = {}
    for sym, eq in eq_dict.items():
        r = eq.pct_change().fillna(0.0)
        rets[sym] = r

    # Align to common index (intersection)
    common_idx = None
    for sym, r in rets.items():
        if common_idx is None:
            common_idx = r.index
        else:
            common_idx = common_idx.intersection(r.index)
    common_idx = common_idx.sort_values()

    # Weighted portfolio return
    port_ret = pd.Series(0.0, index=common_idx)
    for sym, r in rets.items():
        w = weights.get(sym, 0.0)
        port_ret += w * r.reindex(common_idx).fillna(0.0)

    # Portfolio equity
    port_eq = initial_cash * (1 + port_ret).cumprod()
    return port_eq


def equity_metrics(eq: pd.Series, cash: float) -> dict:
    """Compute CAGR, Sharpe, Sortino, MaxDD, Calmar from equity curve."""
    n_yr = max(0.5, len(eq) / 8760)
    final = float(eq.iloc[-1])
    cagr = ((final / cash) ** (1 / n_yr) - 1) * 100

    r = eq.pct_change().dropna()
    sharpe = float(r.mean() / r.std() * np.sqrt(8760)) if len(r) > 1 and r.std() > 0 else 0.0
    dn = r[r < 0]
    sortino = float(r.mean() / dn.std() * np.sqrt(8760)) if len(dn) > 1 and dn.std() > 0 else 0.0

    dd = (eq - eq.cummax()) / eq.cummax()
    mdd = abs(float(dd.min())) * 100
    calmar = cagr / mdd if mdd > 0 else 0.0

    return dict(CAGR=round(cagr, 2), Sharpe=round(sharpe, 3),
                Sortino=round(sortino, 3), MaxDD=round(mdd, 2),
                Calmar=round(calmar, 3))


# ══════════════════════════════════════════════════════════════
#  Main orchestrator
# ══════════════════════════════════════════════════════════════

def run_universe(
    label: str,
    symbols: list[str],
    weights: dict[str, float],
    cache: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Run all configs × periods for a universe, return results DataFrame."""
    print(f"\n{'═' * 72}")
    print(f"  🌐 Universe: {label} ({len(symbols)} symbols)")
    print(f"{'═' * 72}")

    rows = []
    for ec in CONFIGS:
        for pname, (ps, pe) in PERIODS.items():
            eq_dict: dict[str, pd.Series] = {}
            total_fees = 0.0
            total_funding_pct = 0.0
            total_turnover = 0.0
            n_ok = 0

            for sym in symbols:
                if sym not in cache:
                    continue
                result = run_sym_equity(sym, cache[sym], ec, ps, pe)
                if result is None:
                    continue
                eq, fees, funding_pct, turnover = result
                w = weights.get(sym, 0.0)
                eq_dict[sym] = eq
                total_fees += w * fees
                total_funding_pct += w * funding_pct
                total_turnover += w * turnover
                n_ok += 1

            if n_ok < max(2, len(symbols) // 2):
                print(f"  ⚠ {ec.name}/{pname}: only {n_ok}/{len(symbols)} ok, skip")
                continue

            # Portfolio equity
            port_eq = portfolio_equity(eq_dict, weights, BASE_CFG["initial_cash"])
            em = equity_metrics(port_eq, BASE_CFG["initial_cash"])

            row = {
                "Config": ec.name,
                "Period": pname,
                "N_Syms": n_ok,
                **em,
                "Turnover": round(total_turnover, 1),
                "Fee% [cap]": round(total_fees / BASE_CFG["initial_cash"] * 100, 2),
                "Funding%": round(total_funding_pct, 2),
            }
            rows.append(row)
            print(f"   {ec.name:>20} {pname:>4}: Sh={em['Sharpe']:.3f} CAGR={em['CAGR']:.1f}% "
                  f"MDD={em['MaxDD']:.1f}% TO={total_turnover:.0f} "
                  f"Fee%={total_fees / BASE_CFG['initial_cash'] * 100:.1f}%")

    return pd.DataFrame(rows)


def write_decision_memo(df10: pd.DataFrame, df19: pd.DataFrame, out: Path):
    """Generate DECISION_MEMO.md."""
    lines = [
        "# DECISION_MEMO.md",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Context",
        "- 4 configs tested at portfolio-level (weighted equity aggregation)",
        "- 10-symbol: prod live universe (R3C 10S)",
        "- 19-symbol: historical research universe",
        "- Aggregation: portfolio equity = Σ(wᵢ × rᵢ) with vol-parity weights normalised to sum=1",
        "- Metrics computed from combined equity curve (NOT avg-of-metrics)",
        "",
    ]

    # Extract key numbers
    for lbl, df in [("10-symbol", df10), ("19-symbol", df19)]:
        lines.append(f"## {lbl} Universe — Full Period")
        full = df[df["Period"] == "Full"].copy()
        if full.empty:
            lines.append("(no data)\n")
            continue
        lines.append("")
        lines.append("| Config | Sharpe | CAGR | MaxDD | Calmar | Turnover | Fee% |")
        lines.append("|--------|--------|------|-------|--------|----------|------|")
        for _, r in full.sort_values("Sharpe", ascending=False).iterrows():
            lines.append(
                f"| {r['Config']} | {r['Sharpe']:.3f} | {r['CAGR']:.1f}% | "
                f"{r['MaxDD']:.1f}% | {r['Calmar']:.3f} | {r['Turnover']:.0f} | "
                f"{r['Fee% [cap]']:.1f}% |"
            )
        lines.append("")

        # Delta vs B0
        b0 = full[full["Config"] == "B0_baseline"]
        if not b0.empty:
            b0v = b0.iloc[0]
            lines.append(f"### Δ vs B0 ({lbl})")
            lines.append("")
            lines.append("| Config | ΔSharpe | ΔCAGR | ΔMaxDD | ΔFee% | ΔTurnover |")
            lines.append("|--------|---------|-------|--------|-------|-----------|")
            for _, r in full.iterrows():
                if r["Config"] == "B0_baseline":
                    continue
                lines.append(
                    f"| {r['Config']} | {r['Sharpe'] - b0v['Sharpe']:+.3f} | "
                    f"{r['CAGR'] - b0v['CAGR']:+.1f}% | "
                    f"{r['MaxDD'] - b0v['MaxDD']:+.1f}% | "
                    f"{r['Fee% [cap]'] - b0v['Fee% [cap]']:+.1f}% | "
                    f"{r['Turnover'] - b0v['Turnover']:+.0f} |"
                )
            lines.append("")

    # OOS check for best config
    lines.append("## OOS Robustness — B_ema4_rb3_fg80")
    lines.append("")
    for lbl, df in [("10-sym", df10), ("19-sym", df19)]:
        lines.append(f"### {lbl}")
        lines.append("")
        lines.append("| Period | Sharpe | CAGR | MaxDD |")
        lines.append("|--------|--------|------|-------|")
        sub = df[df["Config"] == "B_ema4_rb3_fg80"]
        for pn in ["IS", "OOS", "Live", "Full"]:
            pr = sub[sub["Period"] == pn]
            if not pr.empty:
                r = pr.iloc[0]
                lines.append(f"| {pn} | {r['Sharpe']:.3f} | {r['CAGR']:.1f}% | {r['MaxDD']:.1f}% |")
        lines.append("")

    # Decision
    lines.append("---")
    lines.append("")
    lines.append("## Decision Questions")
    lines.append("")

    # Extract values for decision
    best_10 = df10[(df10["Config"] == "B_ema4_rb3_fg80") & (df10["Period"] == "Full")]
    b0_10 = df10[(df10["Config"] == "B0_baseline") & (df10["Period"] == "Full")]
    best_19 = df19[(df19["Config"] == "B_ema4_rb3_fg80") & (df19["Period"] == "Full")]
    b0_19 = df19[(df19["Config"] == "B0_baseline") & (df19["Period"] == "Full")]
    oos_10 = df10[(df10["Config"] == "B_ema4_rb3_fg80") & (df10["Period"] == "OOS")]
    oos_19 = df19[(df19["Config"] == "B_ema4_rb3_fg80") & (df19["Period"] == "OOS")]

    # Q1: Should mainline switch?
    lines.append("### Q1: 主線是否應切換到 B_ema4_rb3_fg80？")
    lines.append("")
    if not best_10.empty and not b0_10.empty:
        d_sh = best_10.iloc[0]["Sharpe"] - b0_10.iloc[0]["Sharpe"]
        d_cagr = best_10.iloc[0]["CAGR"] - b0_10.iloc[0]["CAGR"]
        cagr_ratio = best_10.iloc[0]["CAGR"] / b0_10.iloc[0]["CAGR"] * 100 if b0_10.iloc[0]["CAGR"] != 0 else 0
        mdd_ok = best_10.iloc[0]["MaxDD"] <= 50
        oos_positive = not oos_10.empty and oos_10.iloc[0]["Sharpe"] > -0.5

        recommend = d_sh > 0.05 and mdd_ok
        if recommend:
            lines.append(f"**建議切換**。Portfolio-level Sharpe 提升 {d_sh:+.3f} (10-sym)，")
            lines.append(f"CAGR {d_cagr:+.1f}%，MaxDD {'✅' if mdd_ok else '❌'} {best_10.iloc[0]['MaxDD']:.1f}%。")
        else:
            lines.append(f"**暫不切換**。ΔSharpe={d_sh:+.3f} 不足 / MaxDD 超限。")
    lines.append("")

    # Q2: Risk & rollback
    lines.append("### Q2: 若切換，風險與回退條件")
    lines.append("")
    lines.append("**風險:**")
    lines.append("1. EMA(4) 平滑尚未在實盤運行過 — 需 shadow 至少 2 週")
    lines.append("2. OOS 期間（2024H2）所有配置均為負 Sharpe — 此策略有明確弱勢環境")
    lines.append("3. 回測不含 overlays (OI/Vol/micro_accel) — 實盤可能有額外信號調整")
    lines.append("")
    lines.append("**回退條件:**")
    lines.append("1. Rolling 30d Sharpe < -0.5 持續 5 天 → 回退到 B0/P0")
    lines.append("2. MDD > 30% → 立即回退")
    lines.append("3. Fee% 日均偏離回測 > 50% → 檢查 gate 參數")
    lines.append("4. Shadow vs Live PnL 偏差 > 20% 持續 1 週 → 暫停切換")
    lines.append("")

    # Q3: What more is needed?
    lines.append("### Q3: 若不切換，還缺什麼證據？")
    lines.append("")
    lines.append("1. **Shadow trading**: B_ema4_rb3_fg80 shadow run ≥ 2 週，比對 live B0")
    lines.append("2. **Overlay 交互**: 加入 OI/Vol/micro_accel overlay 後的 portfolio Sharpe")
    lines.append("3. **更長 OOS**: 2024 OOS 全負是否為系統性（市場環境）還是策略缺陷")
    lines.append("4. **tsmom_multi_ema vs tsmom_ema**: ETHUSDT 在 prod 用 multi_ema，")
    lines.append("   需確認 signal smoothing 對 multi_ema 效果是否一致")
    lines.append("")

    with open(out / "DECISION_MEMO.md", "w") as f:
        f.write("\n".join(lines))
    print(f"\n  📄 DECISION_MEMO.md written")


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUT_ROOT / ts
    out.mkdir(parents=True, exist_ok=True)

    print(f"{'═' * 72}")
    print(f"  Mini Rerun — Portfolio-Level Equity Aggregation")
    print(f"  Output: {out}")
    print(f"  Time:   {datetime.now().isoformat()}")
    print(f"{'═' * 72}")

    # Look-ahead check
    assert BASE_CFG["trade_on"] == "next_open", "trade_on must be next_open"
    print("  ✅ trade_on=next_open, signal_delay=1")

    # Load all data
    print("\n📥 Loading data...")
    cache: dict[str, pd.DataFrame] = {}
    for sym in SYMS_19:
        p = DATA_DIR / "binance" / "futures" / "1h" / f"{sym}.parquet"
        if not p.exists():
            print(f"  ⚠ {sym}: data not found, skip")
            continue
        df = load_klines(p)
        df = clean_data(df, fill_method="forward", remove_outliers=False, remove_duplicates=True)
        cache[sym] = df
        print(f"   {sym}: {len(df):,} bars  {df.index[0]} → {df.index[-1]}")

    # Run 10-symbol universe
    w10 = get_weights(SYMS_10)
    df10 = run_universe("10-symbol (prod live)", SYMS_10, w10, cache)
    df10.to_csv(out / "MINI_RERUN_10sym.csv", index=False)
    print(f"\n  💾 MINI_RERUN_10sym.csv: {len(df10)} rows")

    # Run 19-symbol universe
    w19 = get_weights(SYMS_19)
    df19 = run_universe("19-symbol (historical)", SYMS_19, w19, cache)
    df19.to_csv(out / "MINI_RERUN_19sym.csv", index=False)
    print(f"\n  💾 MINI_RERUN_19sym.csv: {len(df19)} rows")

    # Decision memo
    write_decision_memo(df10, df19, out)

    # Print summary
    print(f"\n{'═' * 72}")
    print(f"  📊 Quick Summary — Full Period, Portfolio-Level")
    print(f"{'═' * 72}")
    for lbl, df in [("10sym", df10), ("19sym", df19)]:
        full = df[df["Period"] == "Full"].sort_values("Sharpe", ascending=False)
        if full.empty:
            continue
        print(f"\n  {lbl}:")
        for _, r in full.iterrows():
            print(f"    {r['Config']:>20}: Sh={r['Sharpe']:.3f} CAGR={r['CAGR']:.1f}% "
                  f"MDD={r['MaxDD']:.1f}% Cal={r['Calmar']:.3f} "
                  f"TO={r['Turnover']:.0f} Fee%={r['Fee% [cap]']:.1f}%")

    print(f"\n{'═' * 72}")
    print(f"  ✅ Mini rerun complete → {out}")
    print(f"{'═' * 72}")

    return out


if __name__ == "__main__":
    main()
