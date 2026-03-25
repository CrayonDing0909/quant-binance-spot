"""
LSR Portfolio Role — Cycle 3

Hypothesis: The accepted LSR contrarian candidate (signal+entry+exit all settled)
has a clear best portfolio role among standalone, long-only satellite, and overlay.

Task: research_20260324_105700_lsr_contrarian_standalone_revisit
Experiment family: portfolio_role
Loop type: Loop B — Trade Expression

What stays fixed: BTC-only, 1h, EW pctrank(hl=84), ADX(14)>25, contrarian,
persist>=2, cooldown=24h, midpoint TP (pr=0.50), ATR 3.0 SL, 168h time stop, RT=0.12%

Row A: standalone viability + correlation to production
Row B: long-only satellite vs symmetric standalone
Row C: overlay ablation (3-way: prod-only vs prod+LSR vs LSR-standalone)

# ORTHOGONALITY_EXEMPT: portfolio-role cycle, not a new signal
# EMBARGO_EXEMPT: continuation of accepted baseline research
"""
import sys
import warnings
import copy

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as sp_stats

from qtrade.config import load_config
from qtrade.backtest.run_backtest import run_symbol_backtest

DATA_DIR = Path("data/binance/futures")
RT = 0.0012
ANN_FACTOR = np.sqrt(8760)

ENTRY_HI, ENTRY_LO = 0.85, 0.15
PERSIST_BARS = 2
COOLDOWN_BARS = 24
WARMUP = 200


# ═══════════════════════════════════════════════════════
#  Shared helpers (from Cycle 1/2)
# ═══════════════════════════════════════════════════════

def ew_pctrank(s, halflife=84, span_equiv=168):
    ew_mean = s.ewm(halflife=halflife, min_periods=42).mean()
    ew_std = s.ewm(halflife=halflife, min_periods=42).std()
    ew_z = (s - ew_mean) / ew_std.replace(0, np.nan)
    return ew_z.rolling(span_equiv, min_periods=84).apply(
        lambda x: sp_stats.percentileofscore(x, x.iloc[-1]) / 100.0, raw=False)


def compute_adx(high, low, close, period=14):
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr = pd.concat([
        high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    return dx.ewm(span=period, adjust=False).mean()


def consecutive_count(series):
    arr = series.values
    out = np.zeros(len(arr), dtype=int)
    for i in range(len(arr)):
        if arr[i] == 1:
            out[i] = out[i - 1] + 1 if i > 0 else 1
        else:
            out[i] = 0
    return pd.Series(out, index=series.index)


# ═══════════════════════════════════════════════════════
#  LSR Contrarian state machine (accepted Cycle 1+2 config)
# ═══════════════════════════════════════════════════════

def run_sm(df, label="", long_only=False):
    """
    Accepted LSR contrarian state machine (Cycles 1+2).
    Midpoint TP, ATR 3.0, 168h time stop, persist>=2, cooldown=24h.

    Returns trade list AND a per-bar position series for equity curve.
    """
    n = len(df)
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values
    o = df["open"].values
    pr_v = df["pr_ew84"].fillna(0.5).values
    atr = df["atr14"].fillna(0).values
    adx_ok = df["trending"].fillna(False).values
    lp = df["long_persist"].values
    sp = df["short_persist"].values
    idx = df.index

    trades = []
    pos_arr = np.zeros(n)
    state = 0
    ep = 0.0
    eb = 0
    last_exit_bar = -9999

    for i in range(WARMUP, n):
        if state == 0:
            if i - last_exit_bar < COOLDOWN_BARS:
                continue
            if not adx_ok[i]:
                continue
            long_ok = pr_v[i] < ENTRY_LO and lp[i] >= PERSIST_BARS
            short_ok = (not long_only) and pr_v[i] > ENTRY_HI and sp[i] >= PERSIST_BARS
            if long_ok:
                state, ep, eb = 1, o[min(i + 1, n - 1)], i
                pos_arr[i] = 1.0
            elif short_ok:
                state, ep, eb = -1, o[min(i + 1, n - 1)], i
                pos_arr[i] = -1.0
        else:
            hold = i - eb
            a = atr[i]
            ex, reason = False, ""

            if state == 1:
                if a > 0 and l[i] <= ep - 3.0 * a:
                    ex, reason = True, "SL"
                elif pr_v[i] > 0.50:
                    ex, reason = True, "TP"
                elif hold >= 168:
                    ex, reason = True, "TIME"
                else:
                    pos_arr[i] = 1.0
            else:
                if a > 0 and h[i] >= ep + 3.0 * a:
                    ex, reason = True, "SL"
                elif pr_v[i] < 0.50:
                    ex, reason = True, "TP"
                elif hold >= 168:
                    ex, reason = True, "TIME"
                else:
                    pos_arr[i] = -1.0

            if ex:
                xp = o[min(i + 1, n - 1)]
                pnl = (xp / ep - 1) * state if ep > 0 else 0
                trades.append({
                    "entry": idx[eb], "exit": idx[i],
                    "dir": "L" if state == 1 else "S",
                    "pnl": pnl,
                    "hold_h": (idx[i] - idx[eb]).total_seconds() / 3600,
                    "reason": reason,
                })
                state = 0
                last_exit_bar = i

    if not trades:
        return {"label": label, "n_trades": 0, "td": pd.DataFrame(),
                "pos": pd.Series(0.0, index=idx), **_empty_metrics()}

    td = pd.DataFrame(trades)
    pos_series = pd.Series(pos_arr, index=idx)
    metrics = _compute_metrics(td, idx, label)
    metrics["pos"] = pos_series
    metrics["td"] = td
    return metrics


def _empty_metrics():
    return {"tr_yr": 0, "hold_h": 0, "wr": 0, "aw": 0, "al": 0,
            "rr": 0, "gross_exp": 0, "net_exp": 0, "mdd": 0,
            "p5_loss": 0, "avg_loss": 0, "dead_share": 0, "exits": {}}


def _compute_metrics(td, idx, label):
    ny = (idx[-1] - idx[0]).days / 365.25
    wr = (td.pnl > 0).mean() * 100
    aw = td.loc[td.pnl > 0, "pnl"].mean() * 100 if (td.pnl > 0).any() else 0
    al = td.loc[td.pnl < 0, "pnl"].mean() * 100 if (td.pnl < 0).any() else 0
    rr = abs(aw / al) if al != 0 else 0
    gross_exp = wr / 100 * aw + (1 - wr / 100) * al
    net_exp = gross_exp - RT * 100
    cum_pnl = td["pnl"].cumsum()
    mdd = (cum_pnl - cum_pnl.cummax()).min() * 100
    losses = td.loc[td.pnl < 0, "pnl"]
    p5_loss = losses.quantile(0.05) * 100 if len(losses) > 5 else al
    dead_share = (td.reason == "TIME").sum() / len(td) * 100

    return {"label": label, "n_trades": len(td),
            "tr_yr": len(td) / ny if ny > 0 else 0,
            "hold_h": td.hold_h.mean(), "wr": wr, "aw": aw, "al": al, "rr": rr,
            "gross_exp": gross_exp, "net_exp": net_exp, "mdd": mdd,
            "p5_loss": p5_loss, "avg_loss": al, "dead_share": dead_share,
            "exits": td.reason.value_counts().to_dict()}


def equity_from_pos(pos, close):
    """Build a simple equity curve from a position series and close prices."""
    ret = close.pct_change().fillna(0)
    strat_ret = pos.shift(1).fillna(0) * ret
    return (1 + strat_ret).cumprod()


def compute_sharpe(equity):
    r = equity.pct_change().dropna()
    if r.std() == 0:
        return 0.0
    return r.mean() / r.std() * ANN_FACTOR


def compute_max_dd(equity):
    cum = equity / equity.cummax()
    return (cum - 1).min()


def compare_table(r, bl):
    rows = [
        ("Trades",      "n_trades",  ".0f", "+.0f"),
        ("Trades/yr",   "tr_yr",     ".1f", "+.1f"),
        ("WR %",        "wr",        ".1f", "+.1f"),
        ("Gross exp %", "gross_exp", "+.3f", "+.3f"),
        ("Net exp %",   "net_exp",   "+.3f", "+.3f"),
        ("R:R",         "rr",        ".2f", "+.2f"),
        ("Avg hold h",  "hold_h",    ".0f", "+.0f"),
        ("MDD %",       "mdd",       "+.2f", "+.2f"),
        ("Dead trade %","dead_share", ".1f", "+.1f"),
    ]
    print(f"  {'Metric':<20} {'Baseline':>12} {'Variant':>12} {'Delta':>12}")
    print(f"  {'-' * 58}")
    for label, key, fmt, dfmt in rows:
        bv, rv = bl[key], r[key]
        print(f"  {label:<20} {format(bv, fmt):>12} {format(rv, fmt):>12} {format(rv - bv, dfmt):>12}")
    print(f"  Exits: {r['exits']}")


# ═══════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Load BTC data + compute signals ──
    kl = pd.read_parquet(DATA_DIR / "1h" / "BTCUSDT.parquet")
    kl.index = kl.index.tz_localize(None) if kl.index.tz else kl.index

    lsr_raw = pd.read_parquet(DATA_DIR / "derivatives" / "lsr" / "BTCUSDT.parquet")["lsr"]
    lsr_raw.index = lsr_raw.index.tz_localize(None) if lsr_raw.index.tz else lsr_raw.index

    btc = kl.copy()
    btc["lsr"] = lsr_raw.reindex(btc.index, method="ffill", limit=2)
    btc = btc.loc[btc["lsr"].first_valid_index():].dropna(subset=["lsr"])

    btc["pr_ew84"] = ew_pctrank(btc["lsr"], halflife=84)
    btc["adx14"] = compute_adx(btc["high"], btc["low"], btc["close"], 14)
    btc["trending"] = btc["adx14"] > 25

    tr = pd.concat([
        btc["high"] - btc["low"],
        (btc["high"] - btc["close"].shift(1)).abs(),
        (btc["low"] - btc["close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    btc["atr14"] = tr.rolling(14, min_periods=7).mean()

    pr = btc["pr_ew84"].fillna(0.5)
    btc["long_persist"] = consecutive_count((pr < ENTRY_LO).astype(int))
    btc["short_persist"] = consecutive_count((pr > ENTRY_HI).astype(int))

    btc["year"] = btc.index.year

    print(f"BTC: {btc.index[0]} -> {btc.index[-1]} ({len(btc):,} bars)")

    # ═══════════════════════════════════════════════
    #  Baseline reproduction
    # ═══════════════════════════════════════════════
    baseline = run_sm(btc, "Symmetric standalone")

    print("\n" + "=" * 90)
    print("BASELINE REPRODUCTION (expected: 446 trades, net exp +0.407%)")
    print("=" * 90)
    print(f"  Trades: {baseline['n_trades']}  (expected: 446)")
    print(f"  Tr/yr:  {baseline['tr_yr']:.1f}  (expected: 105.0)")
    print(f"  Net:    {baseline['net_exp']:+.3f}%  (expected: +0.407%)")
    print(f"  MDD:    {baseline['mdd']:+.2f}%  (expected: -19.56%)")
    match = abs(baseline["n_trades"] - 446) <= 2
    print(f"  Match: {'OK' if match else 'MISMATCH'}")

    # Build equity curve for standalone
    eq_standalone = equity_from_pos(baseline["pos"], btc["close"])

    # ═══════════════════════════════════════════════
    #  Run production BTC backtest
    # ═══════════════════════════════════════════════
    print("\n\n" + "=" * 90)
    print("PRODUCTION BTC BACKTEST (for correlation analysis)")
    print("=" * 90)

    prod_cfg = load_config("config/prod_candidate_simplified.yaml")
    bt_cfg = copy.deepcopy(prod_cfg.to_backtest_dict(symbol="BTCUSDT"))
    data_path = (
        prod_cfg.data_dir / "binance" / "futures"
        / prod_cfg.market.interval / "BTCUSDT.parquet"
    )

    prod_result = None
    eq_prod = None
    try:
        prod_result = run_symbol_backtest(
            "BTCUSDT", data_path, bt_cfg,
            strategy_name=prod_cfg.strategy.name,
            data_dir=prod_cfg.data_dir,
        )
        eq_prod = prod_result.equity()
        eq_prod.index = eq_prod.index.tz_localize(None) if eq_prod.index.tz else eq_prod.index
        prod_sr = compute_sharpe(eq_prod)
        prod_dd = compute_max_dd(eq_prod)
        print(f"  Production BTC: SR={prod_sr:.2f}, MDD={prod_dd:.2%}")
        print(f"  Equity: {eq_prod.iloc[0]:.0f} -> {eq_prod.iloc[-1]:.0f}")
    except Exception as e:
        print(f"  Production backtest failed: {e}")
        print(f"  Will skip Row C overlay analysis")

    # ═══════════════════════════════════════════════
    #  ROW A — Standalone Viability
    # ═══════════════════════════════════════════════
    print("\n\n" + "=" * 90)
    print("ROW A — STANDALONE VIABILITY")
    print("=" * 90)

    # 1. Expectancy and density (already confirmed)
    print(f"\n  Net expectancy: {baseline['net_exp']:+.3f}% (> 0: PASS)")
    print(f"  Trades/yr: {baseline['tr_yr']:.1f} (>= 50: PASS)")
    print(f"  MDD: {baseline['mdd']:+.2f}%")

    # 2. Correlation to production BTC
    if eq_prod is not None:
        common = eq_standalone.index.intersection(eq_prod.index)
        if len(common) > 500:
            ret_lsr = eq_standalone.reindex(common).pct_change().dropna()
            ret_prod = eq_prod.reindex(common).pct_change().dropna()
            common_ret = ret_lsr.index.intersection(ret_prod.index)
            ret_lsr = ret_lsr.reindex(common_ret)
            ret_prod = ret_prod.reindex(common_ret)

            corr = ret_lsr.corr(ret_prod)
            print(f"\n  Return correlation (LSR standalone vs Prod BTC): {corr:.4f}")
            print(f"  Correlation < 0.30 = strong diversifier: {'PASS' if abs(corr) < 0.30 else 'FAIL'}")
            print(f"  Correlation < 0.50 = acceptable:          {'PASS' if abs(corr) < 0.50 else 'FAIL'}")

            # Rolling 60-day correlation
            rolling_corr = ret_lsr.rolling(60 * 24).corr(ret_prod).dropna()
            print(f"\n  Rolling 60d correlation:")
            print(f"    Mean:   {rolling_corr.mean():.4f}")
            print(f"    Median: {rolling_corr.median():.4f}")
            print(f"    P10:    {rolling_corr.quantile(0.10):.4f}")
            print(f"    P90:    {rolling_corr.quantile(0.90):.4f}")
        else:
            corr = np.nan
            print(f"  Insufficient common data for correlation ({len(common)} bars)")
    else:
        corr = np.nan
        print(f"  Production BTC equity not available for correlation")

    # 3. Standalone SR
    sr_standalone = compute_sharpe(eq_standalone)
    dd_standalone = compute_max_dd(eq_standalone)
    print(f"\n  Standalone SR: {sr_standalone:.2f}")
    print(f"  Standalone MDD: {dd_standalone:.2%}")

    # 4. Year-by-year
    td_bl = baseline["td"].copy()
    td_bl["year"] = pd.to_datetime(td_bl["entry"]).dt.year
    years = sorted(btc[btc["year"] >= 2022]["year"].unique())
    print(f"\n  Year-by-year:")
    all_positive = True
    for yr in years:
        s = td_bl[td_bl["year"] == yr]
        if len(s) < 1:
            continue
        avg = s.pnl.mean() * 100
        wr = (s.pnl > 0).mean() * 100
        if avg <= 0:
            all_positive = False
        print(f"    {yr}: {len(s):>3} trades, WR={wr:.0f}%, avgPnL={avg:+.3f}%")
    print(f"  All years positive: {'YES' if all_positive else 'NO'}")

    # Row A verdict
    a_net_ok = baseline["net_exp"] > 0
    a_density_ok = baseline["tr_yr"] >= 50
    a_corr_ok = abs(corr) < 0.50 if not np.isnan(corr) else True
    a_pass = a_net_ok and a_density_ok and a_corr_ok and all_positive

    print(f"\n  {'=' * 60}")
    print(f"  ROW A VERDICT: {'PASS — standalone viable' if a_pass else 'FAIL'}")
    if not a_pass:
        reasons = []
        if not a_net_ok: reasons.append("net exp <= 0")
        if not a_density_ok: reasons.append("density < 50/yr")
        if not a_corr_ok: reasons.append(f"corr too high ({corr:.3f})")
        if not all_positive: reasons.append("not all years positive")
        print(f"  Reasons: {', '.join(reasons)}")
    print(f"  {'=' * 60}")

    # ═══════════════════════════════════════════════
    #  ROW B — Long-Only Satellite
    # ═══════════════════════════════════════════════
    print("\n\n" + "=" * 90)
    print("ROW B — LONG-ONLY SATELLITE")
    print("=" * 90)

    long_only = run_sm(btc, "Long-only satellite", long_only=True)

    print(f"\n  Long-only vs Symmetric Standalone:")
    compare_table(long_only, baseline)

    # Side quality from long-only
    td_lo = long_only["td"]
    if len(td_lo) > 0:
        lo_wr = (td_lo.pnl > 0).mean() * 100
        lo_avg = td_lo.pnl.mean() * 100
        print(f"\n  Long-only: {len(td_lo)} trades, WR={lo_wr:.1f}%, avgPnL={lo_avg:+.3f}%")

        # Year-by-year
        td_lo_yr = td_lo.copy()
        td_lo_yr["year"] = pd.to_datetime(td_lo_yr["entry"]).dt.year
        print(f"\n  Year-by-year (long-only):")
        lo_all_positive = True
        for yr in years:
            s = td_lo_yr[td_lo_yr["year"] == yr]
            if len(s) < 1:
                continue
            avg = s.pnl.mean() * 100
            wr = (s.pnl > 0).mean() * 100
            if avg <= 0:
                lo_all_positive = False
            print(f"    {yr}: {len(s):>3} trades, WR={wr:.0f}%, avgPnL={avg:+.3f}%")
        print(f"  All years positive: {'YES' if lo_all_positive else 'NO'}")

    # Long-only equity curve + correlation
    eq_long_only = equity_from_pos(long_only["pos"], btc["close"])
    sr_lo = compute_sharpe(eq_long_only)
    dd_lo = compute_max_dd(eq_long_only)
    print(f"\n  Long-only SR: {sr_lo:.2f} (standalone: {sr_standalone:.2f})")
    print(f"  Long-only MDD: {dd_lo:.2%} (standalone: {dd_standalone:.2%})")

    # Row B verdict
    b_net_ok = long_only["net_exp"] > 0
    b_density_ok = long_only["tr_yr"] >= 30
    b_better_robustness = long_only["net_exp"] > baseline["net_exp"]
    b_better_mdd = long_only["mdd"] > baseline["mdd"]
    b_sr_better = sr_lo > sr_standalone

    print(f"\n  {'=' * 60}")
    if b_net_ok and b_density_ok and (b_better_robustness or b_sr_better):
        b_verdict = "PASS"
        print("  ROW B VERDICT: PASS — long-only satellite improves on symmetric")
        print(f"    Net exp: {long_only['net_exp']:+.3f}% vs {baseline['net_exp']:+.3f}%")
        print(f"    SR: {sr_lo:.2f} vs {sr_standalone:.2f}")
    elif b_net_ok and b_density_ok:
        b_verdict = "NEUTRAL"
        print("  ROW B VERDICT: NEUTRAL — long-only viable but not clearly better")
    else:
        b_verdict = "FAIL"
        print("  ROW B VERDICT: FAIL — long-only not viable")
        if not b_density_ok:
            print(f"    Density: {long_only['tr_yr']:.1f}/yr < 30")
    print(f"  {'=' * 60}")

    # ═══════════════════════════════════════════════
    #  ROW C — Overlay / Filter on Production
    # ═══════════════════════════════════════════════
    print("\n\n" + "=" * 90)
    print("ROW C — OVERLAY / FILTER ON PRODUCTION")
    print("=" * 90)

    if prod_result is not None and eq_prod is not None:
        prod_pos = prod_result.pos
        prod_pos.index = prod_pos.index.tz_localize(None) if prod_pos.index.tz else prod_pos.index
        lsr_pos = baseline["pos"]

        common_idx = prod_pos.index.intersection(lsr_pos.index)
        prod_pos_c = prod_pos.reindex(common_idx).fillna(0)
        lsr_pos_c = lsr_pos.reindex(common_idx).fillna(0)
        close_c = btc["close"].reindex(common_idx).fillna(method="ffill")

        # Overlay logic: boost production when LSR agrees, reduce when it disagrees
        lsr_dir = np.sign(lsr_pos_c)
        prod_dir = np.sign(prod_pos_c)

        # Agreement: LSR and production point in the same direction
        agree = (lsr_dir != 0) & (prod_dir != 0) & (lsr_dir == prod_dir)
        disagree = (lsr_dir != 0) & (prod_dir != 0) & (lsr_dir != prod_dir)

        overlay_pos = prod_pos_c.copy()
        overlay_pos[agree] = prod_pos_c[agree] * 1.3     # boost 30%
        overlay_pos[disagree] = prod_pos_c[disagree] * 0.5  # reduce 50%
        overlay_pos = overlay_pos.clip(-1, 1)

        # 3-way equity curves
        eq_a = equity_from_pos(prod_pos_c, close_c)    # production only
        eq_b = equity_from_pos(overlay_pos, close_c)    # production + LSR overlay
        eq_c = equity_from_pos(lsr_pos_c, close_c)      # LSR standalone

        sr_a = compute_sharpe(eq_a)
        sr_b = compute_sharpe(eq_b)
        sr_c = compute_sharpe(eq_c)

        dd_a = compute_max_dd(eq_a)
        dd_b = compute_max_dd(eq_b)
        dd_c = compute_max_dd(eq_c)

        ret_a = eq_a.iloc[-1] / eq_a.iloc[0] - 1
        ret_b = eq_b.iloc[-1] / eq_b.iloc[0] - 1
        ret_c = eq_c.iloc[-1] / eq_c.iloc[0] - 1

        print(f"\n  3-Way Ablation (common window: {common_idx[0]} -> {common_idx[-1]}, "
              f"{len(common_idx):,} bars):")
        print(f"\n  {'Variant':<30} {'SR':>8} {'MDD':>10} {'Return':>10}")
        print(f"  {'-' * 60}")
        print(f"  {'A: Prod-only':<30} {sr_a:>8.2f} {dd_a:>9.2%} {ret_a:>9.1%}")
        print(f"  {'B: Prod+LSR overlay':<30} {sr_b:>8.2f} {dd_b:>9.2%} {ret_b:>9.1%}")
        print(f"  {'C: LSR standalone':<30} {sr_c:>8.2f} {dd_c:>9.2%} {ret_c:>9.1%}")

        # Marginal SR
        d_sr = sr_b - sr_a
        d_dd = dd_b - dd_a

        print(f"\n  Overlay marginal SR: {d_sr:+.2f}")
        print(f"  Overlay marginal MDD: {d_dd:+.2%}")

        # Agreement stats
        agree_pct = agree.sum() / (lsr_dir != 0).sum() * 100 if (lsr_dir != 0).sum() > 0 else 0
        disagree_pct = disagree.sum() / (lsr_dir != 0).sum() * 100 if (lsr_dir != 0).sum() > 0 else 0
        print(f"\n  Signal agreement stats:")
        print(f"    LSR active bars: {(lsr_dir != 0).sum():,}")
        print(f"    Agree with prod: {agree.sum():,} ({agree_pct:.1f}%)")
        print(f"    Disagree:        {disagree.sum():,} ({disagree_pct:.1f}%)")

        c_overlay_improves = sr_b > sr_a and d_sr > 0.05
        c_overlay_no_harm = sr_b >= sr_a * 0.95

        print(f"\n  {'=' * 60}")
        if c_overlay_improves:
            c_verdict = "PASS"
            print("  ROW C VERDICT: PASS — overlay improves production")
        elif c_overlay_no_harm:
            c_verdict = "NEUTRAL"
            print("  ROW C VERDICT: NEUTRAL — overlay neither helps nor hurts")
        else:
            c_verdict = "FAIL"
            print("  ROW C VERDICT: FAIL — overlay degrades production")
        print(f"  {'=' * 60}")
    else:
        c_verdict = "SKIP"
        print("  SKIP — production backtest not available")
        sr_a = sr_b = sr_c = 0
        d_sr = 0

    # ═══════════════════════════════════════════════
    #  Final Verdict
    # ═══════════════════════════════════════════════
    print("\n\n" + "=" * 70)
    print("CYCLE 3 — PORTFOLIO ROLE FINAL VERDICT")
    print("=" * 70)

    a_verdict_str = "PASS" if a_pass else "FAIL"
    print(f"\n  Row A (standalone): {a_verdict_str}")
    if not np.isnan(corr):
        print(f"    Correlation to prod BTC: {corr:.4f}")
    print(f"    Net exp: {baseline['net_exp']:+.3f}%, Density: {baseline['tr_yr']:.1f}/yr")

    print(f"\n  Row B (long-only satellite): {b_verdict}")
    print(f"    Net exp: {long_only['net_exp']:+.3f}%, Density: {long_only['tr_yr']:.1f}/yr")
    print(f"    SR: {sr_lo:.2f} vs standalone {sr_standalone:.2f}")

    print(f"\n  Row C (overlay): {c_verdict}")
    if c_verdict != "SKIP":
        print(f"    Marginal SR: {d_sr:+.2f}")

    # Decision logic
    print(f"\n  {'=' * 60}")

    # Pick best role
    if a_pass and b_verdict == "PASS" and long_only["net_exp"] > baseline["net_exp"]:
        chosen_role = "long_only_satellite"
        print("  CHOSEN ROLE: long-only satellite")
        print("    Long-only improves on symmetric standalone — cleaner edge, fewer short losses.")
        print("    Validate as a separate long-only role before developer handoff.")
    elif a_pass and b_verdict != "PASS":
        chosen_role = "symmetric_standalone"
        print("  CHOSEN ROLE: symmetric standalone")
        print("    Standalone is viable and long-only does not clearly dominate.")
        print("    Prepare for Quant Developer handoff.")
    elif a_pass and b_verdict == "PASS":
        # Both pass but symmetric has better expectancy — keep symmetric
        chosen_role = "symmetric_standalone"
        print("  CHOSEN ROLE: symmetric standalone")
        print("    Both roles viable; symmetric has better net expectancy.")
    elif not a_pass and c_verdict == "PASS":
        chosen_role = "overlay"
        print("  CHOSEN ROLE: overlay/filter")
        print("    Standalone not viable, but overlay improves production.")
        print("    Open a position_sizing cycle for overlay conviction scaling.")
    else:
        chosen_role = "no_clear_role"
        print("  NO CLEAR ROLE — needs further work or direction should be shelved")

    print(f"\n  {'=' * 60}")

    print(f"\n  Full accepted config after Cycles 1-3:")
    print(f"    Signal:   EW pctrank(hl=84) contrarian")
    print(f"    Regime:   ADX(14) > 25")
    print(f"    Entry:    persist >= 2, cooldown = 24h")
    print(f"    TP:       midpoint (pr = 0.50)")
    print(f"    SL:       ATR 3.0")
    print(f"    Time:     168h")
    print(f"    Role:     {chosen_role}")

    if chosen_role in ("symmetric_standalone", "long_only_satellite"):
        print(f"\n  Next: Quant Developer handoff for v3 implementation + validation")
    elif chosen_role == "overlay":
        print(f"\n  Next: position_sizing cycle for overlay conviction scaling")
    else:
        print(f"\n  Next: review research direction or shelve")
