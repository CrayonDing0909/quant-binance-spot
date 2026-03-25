"""
LSR Exit Design — Cycle 2

Hypothesis: The accepted LSR contrarian entry (persist>=2, cooldown=24h) can be
better monetized by optimizing TP target, SL multiplier, and time stop independently.

Task: research_20260324_105700_lsr_contrarian_standalone_revisit
Experiment family: exit_design
Loop type: Loop B — Trade Expression

What stays fixed: BTC-only, 1h, EW pctrank(hl=84), ADX(14)>25 regime,
contrarian direction, persist>=2, cooldown=24h, RT=0.12%

Row A: Structural TP (opposite-extreme vs midpoint vs NW center)
Row B: ATR SL multiplier (2.0 - 4.0 sweep)
Row C: Time stop (72h - 216h sweep)

# ORTHOGONALITY_EXEMPT: exit-design cycle, not a new signal
# EMBARGO_EXEMPT: continuation of accepted baseline research
"""
import sys
import warnings
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as sp_stats

DATA_DIR = Path("data/binance/futures")
RT = 0.0012

ENTRY_HI, ENTRY_LO = 0.85, 0.15
PERSIST_BARS = 2
COOLDOWN_BARS = 24
WARMUP = 200


# ═══════════════════════════════════════════════════════
#  Shared helpers (copied from Cycle 1 for reproducibility)
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
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
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
#  State machine — exit variants only
# ═══════════════════════════════════════════════════════

def run_sm(df, label="",
           tp_mode="opposite_extreme",
           tp_hi=0.75, tp_lo=0.25,
           sl_atr_mult=3.0,
           max_hold=168):
    """
    State machine with FIXED entry (persist>=2, cooldown=24h).
    Only exit logic varies.

    tp_mode: opposite_extreme | midpoint | nw_center | nw_or_midpoint
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
    nw = df["nw_center"].values
    idx = df.index

    trades = []
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
            short_ok = pr_v[i] > ENTRY_HI and sp[i] >= PERSIST_BARS
            if long_ok:
                state, ep, eb = 1, o[min(i + 1, n - 1)], i
            elif short_ok:
                state, ep, eb = -1, o[min(i + 1, n - 1)], i
        else:
            hold = i - eb
            a = atr[i]
            ex, reason = False, ""

            if state == 1:
                if a > 0 and l[i] <= ep - sl_atr_mult * a:
                    ex, reason = True, "SL"
                elif tp_mode == "opposite_extreme" and pr_v[i] > tp_hi:
                    ex, reason = True, "TP"
                elif tp_mode == "midpoint" and pr_v[i] > 0.50:
                    ex, reason = True, "TP"
                elif tp_mode == "nw_center" and c[i] >= nw[i]:
                    ex, reason = True, "TP"
                elif tp_mode == "nw_or_midpoint" and (pr_v[i] > 0.50 or c[i] >= nw[i]):
                    ex, reason = True, "TP"
                elif hold >= max_hold:
                    ex, reason = True, "TIME"
            else:
                if a > 0 and h[i] >= ep + sl_atr_mult * a:
                    ex, reason = True, "SL"
                elif tp_mode == "opposite_extreme" and pr_v[i] < tp_lo:
                    ex, reason = True, "TP"
                elif tp_mode == "midpoint" and pr_v[i] < 0.50:
                    ex, reason = True, "TP"
                elif tp_mode == "nw_center" and c[i] <= nw[i]:
                    ex, reason = True, "TP"
                elif tp_mode == "nw_or_midpoint" and (pr_v[i] < 0.50 or c[i] <= nw[i]):
                    ex, reason = True, "TP"
                elif hold >= max_hold:
                    ex, reason = True, "TIME"

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
                "mdd": 0, "p5_loss": 0, "avg_loss": 0, "dead_share": 0,
                "tr_yr": 0, "hold_h": 0, "wr": 0, "aw": 0, "al": 0,
                "rr": 0, "gross_exp": 0, "net_exp": 0, "exits": {}}

    td = pd.DataFrame(trades)
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

    return {
        "label": label, "n_trades": len(td),
        "tr_yr": len(td) / ny if ny > 0 else 0,
        "hold_h": td.hold_h.mean(), "wr": wr, "aw": aw, "al": al, "rr": rr,
        "gross_exp": gross_exp, "net_exp": net_exp, "mdd": mdd,
        "p5_loss": p5_loss, "avg_loss": al, "dead_share": dead_share,
        "exits": td.reason.value_counts().to_dict(), "td": td,
    }


def compare_table(r, bl):
    rows = [
        ("Trades",     "n_trades",  ".0f", "+.0f"),
        ("Trades/yr",  "tr_yr",     ".1f", "+.1f"),
        ("WR %",       "wr",        ".1f", "+.1f"),
        ("Gross exp %","gross_exp", "+.3f","+.3f"),
        ("Net exp %",  "net_exp",  "+.3f", "+.3f"),
        ("R:R",        "rr",        ".2f", "+.2f"),
        ("Avg hold h", "hold_h",    ".0f", "+.0f"),
        ("MDD %",      "mdd",      "+.2f", "+.2f"),
        ("P5 loss %",  "p5_loss",  "+.2f", "+.2f"),
        ("Dead trade %","dead_share",".1f","+.1f"),
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
    from qtrade.strategy.nwkl_strategy import _rq_kernel_weights, _causal_nw_regression

    # ── Load data ──
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

    weights = _rq_kernel_weights(200, 8.0, 1.0)
    btc["nw_center"] = _causal_nw_regression(
        btc["close"].values.astype(np.float64), weights, 200)

    btc["year"] = btc.index.year

    print(f"BTC: {btc.index[0]} -> {btc.index[-1]} ({len(btc):,} bars)")

    # ═══════════════════════════════════════════════
    #  Baseline reproduction
    # ═══════════════════════════════════════════════
    baseline = run_sm(btc, "Baseline (opp-extreme TP, ATR3.0, 168h)")

    print("\n" + "=" * 90)
    print("BASELINE REPRODUCTION (expected: 381 trades, net exp +0.521%)")
    print("=" * 90)
    print(f"  Trades: {baseline['n_trades']}  (expected: 381)")
    print(f"  Tr/yr:  {baseline['tr_yr']:.1f}  (expected: 89.7)")
    print(f"  WR:     {baseline['wr']:.1f}%  (expected: 36.5%)")
    print(f"  Gross:  {baseline['gross_exp']:+.3f}%  (expected: +0.641%)")
    print(f"  Net:    {baseline['net_exp']:+.3f}%  (expected: +0.521%)")
    print(f"  R:R:    {baseline['rr']:.2f}  (expected: 2.75)")
    print(f"  Hold:   {baseline['hold_h']:.0f}h  (expected: 60h)")
    print(f"  Exits:  {baseline['exits']}")
    print(f"  MDD:    {baseline['mdd']:+.2f}%")
    match = abs(baseline["n_trades"] - 381) <= 2
    print(f"\n  Match: {'OK' if match else 'MISMATCH'}")

    # ═══════════════════════════════════════════════
    #  Row A — Structural TP
    # ═══════════════════════════════════════════════
    print("\n\n" + "=" * 90)
    print("ROW A — STRUCTURAL TP")
    print("=" * 90)

    tp_variants = {
        "Baseline: opp-extreme": run_sm(btc, "Baseline: opp-extreme",
                                        tp_mode="opposite_extreme"),
        "A1: midpoint (pr=0.50)": run_sm(btc, "A1: midpoint",
                                          tp_mode="midpoint"),
        "A2: NW center": run_sm(btc, "A2: NW center",
                                 tp_mode="nw_center"),
        "A3: NW or midpoint": run_sm(btc, "A3: NW or midpoint",
                                      tp_mode="nw_or_midpoint"),
    }

    hdr = (f"  {'Variant':<25} {'#Tr':>6} {'Tr/yr':>8} {'WR%':>7} {'GrossE':>10} "
           f"{'NetE':>10} {'R:R':>7} {'Hold':>7} {'MDD':>8} {'P5Loss':>8} {'Dead%':>7}")
    print(f"\n{hdr}")
    print(f"  {'-' * 110}")
    for name, r in tp_variants.items():
        tag = " <-- baseline" if "Baseline" in name else ""
        print(f"  {name:<25} {r['n_trades']:>6} {r['tr_yr']:>8.1f} {r['wr']:>6.1f}% "
              f"{r['gross_exp']:>+9.3f}% {r['net_exp']:>+9.3f}% "
              f"{r['rr']:>7.2f} {r['hold_h']:>6.0f}h {r['mdd']:>+7.2f}% "
              f"{r['p5_loss']:>+7.2f}% {r['dead_share']:>6.1f}%{tag}")

    bl = tp_variants["Baseline: opp-extreme"]
    for name, r in tp_variants.items():
        if "Baseline" in name:
            continue
        print(f"\n--- {name} vs Baseline ---")
        compare_table(r, bl)

    # Row A verdict
    best_tp_name = "Baseline: opp-extreme"
    best_tp = bl
    for name, r in tp_variants.items():
        if "Baseline" in name:
            continue
        mdd_improves = r["mdd"] > bl["mdd"] + 0.5
        exp_positive = r["net_exp"] > 0
        net_better = r["net_exp"] > bl["net_exp"]
        if exp_positive and (mdd_improves or net_better):
            if r["net_exp"] > best_tp["net_exp"] or r["mdd"] > best_tp["mdd"] + 1.0:
                best_tp_name = name
                best_tp = r

    tp_mode_map = {
        "Baseline: opp-extreme": "opposite_extreme",
        "A1: midpoint (pr=0.50)": "midpoint",
        "A2: NW center": "nw_center",
        "A3: NW or midpoint": "nw_or_midpoint",
    }
    accepted_tp_mode = tp_mode_map[best_tp_name]

    print(f"\n{'=' * 70}")
    print("ROW A VERDICT")
    print(f"{'=' * 70}")
    if best_tp_name == "Baseline: opp-extreme":
        a_verdict = "KEEP_BASELINE"
        print("  KEEP BASELINE — no TP variant offers a clear improvement")
    else:
        a_verdict = "PASS"
        print(f"  PASS — {best_tp_name}")
        print(f"  Net exp: {best_tp['net_exp']:+.3f}% (baseline: {bl['net_exp']:+.3f}%)")
        print(f"  MDD: {best_tp['mdd']:+.2f}% (baseline: {bl['mdd']:+.2f}%)")
    print(f"  Accepted TP: {accepted_tp_mode}")

    # ═══════════════════════════════════════════════
    #  Row B — ATR SL
    # ═══════════════════════════════════════════════
    print("\n\n" + "=" * 90)
    print(f"ROW B — ATR SL MULTIPLIER (TP: {accepted_tp_mode})")
    print("=" * 90)

    sl_results = {}
    for mult in [2.0, 2.5, 3.0, 3.5, 4.0]:
        sl_results[mult] = run_sm(btc, f"ATR {mult:.1f}",
                                   tp_mode=accepted_tp_mode,
                                   sl_atr_mult=mult)

    print(f"\n  {'ATR':>5} {'#Tr':>6} {'Tr/yr':>8} {'WR%':>7} {'GrossE':>10} "
          f"{'NetE':>10} {'R:R':>7} {'Hold':>7} {'MDD':>8} {'P5Loss':>8}")
    print(f"  {'-' * 95}")
    for mult, r in sl_results.items():
        tag = " <-- current" if mult == 3.0 else ""
        print(f"  {mult:>5.1f} {r['n_trades']:>6} {r['tr_yr']:>8.1f} {r['wr']:>6.1f}% "
              f"{r['gross_exp']:>+9.3f}% {r['net_exp']:>+9.3f}% "
              f"{r['rr']:>7.2f} {r['hold_h']:>6.0f}h {r['mdd']:>+7.2f}% "
              f"{r['p5_loss']:>+7.2f}%{tag}")

    bl_sl = sl_results[3.0]
    best_sl_mult = 3.0
    best_sl = bl_sl
    for mult, r in sl_results.items():
        if mult == 3.0:
            continue
        tail_better = r["p5_loss"] > bl_sl["p5_loss"] + 0.1
        mdd_better = r["mdd"] > bl_sl["mdd"] + 0.3
        exp_ok = r["net_exp"] > bl_sl["net_exp"] * 0.8
        exp_better = r["net_exp"] > bl_sl["net_exp"]
        hold_ok = r["hold_h"] < bl_sl["hold_h"] * 1.5
        if hold_ok and exp_ok and (tail_better or mdd_better or exp_better):
            score_new = r["net_exp"] + r["mdd"] * 0.1
            score_curr = best_sl["net_exp"] + best_sl["mdd"] * 0.1
            if score_new > score_curr:
                best_sl_mult = mult
                best_sl = r

    accepted_sl_mult = best_sl_mult

    print(f"\n{'=' * 70}")
    print("ROW B VERDICT")
    print(f"{'=' * 70}")
    if best_sl_mult == 3.0:
        b_verdict = "KEEP_BASELINE"
        print("  KEEP BASELINE — ATR 3.0 remains the best stop")
    else:
        b_verdict = "PASS"
        print(f"  PASS — ATR {best_sl_mult:.1f}")
        print(f"  Net exp: {best_sl['net_exp']:+.3f}% (baseline: {bl_sl['net_exp']:+.3f}%)")
        print(f"  MDD: {best_sl['mdd']:+.2f}% (baseline: {bl_sl['mdd']:+.2f}%)")
    print(f"  Accepted SL: ATR {accepted_sl_mult:.1f}")

    # ═══════════════════════════════════════════════
    #  Row C — Time Stop
    # ═══════════════════════════════════════════════
    print("\n\n" + "=" * 90)
    print(f"ROW C — TIME STOP (TP: {accepted_tp_mode}, SL: ATR {accepted_sl_mult:.1f})")
    print("=" * 90)

    ts_results = {}
    for hours in [72, 96, 120, 144, 168, 216]:
        ts_results[hours] = run_sm(btc, f"TIME={hours}h",
                                    tp_mode=accepted_tp_mode,
                                    sl_atr_mult=accepted_sl_mult,
                                    max_hold=hours)

    print(f"\n  {'Time':>6} {'#Tr':>6} {'Tr/yr':>8} {'WR%':>7} {'GrossE':>10} "
          f"{'NetE':>10} {'R:R':>7} {'Hold':>7} {'MDD':>8} {'Dead%':>7}")
    print(f"  {'-' * 95}")
    for hours, r in ts_results.items():
        tag = " <-- current" if hours == 168 else ""
        print(f"  {hours:>5}h {r['n_trades']:>6} {r['tr_yr']:>8.1f} {r['wr']:>6.1f}% "
              f"{r['gross_exp']:>+9.3f}% {r['net_exp']:>+9.3f}% "
              f"{r['rr']:>7.2f} {r['hold_h']:>6.0f}h {r['mdd']:>+7.2f}% "
              f"{r['dead_share']:>6.1f}%{tag}")

    bl_ts = ts_results[168]
    best_ts = 168
    best_ts_r = bl_ts
    for hours, r in ts_results.items():
        if hours == 168:
            continue
        hold_better = r["hold_h"] < bl_ts["hold_h"]
        exp_better = r["net_exp"] > bl_ts["net_exp"]
        dead_better = r["dead_share"] < bl_ts["dead_share"]
        mdd_ok = r["mdd"] > bl_ts["mdd"] - 1.0
        if mdd_ok and (exp_better or (hold_better and dead_better)):
            score_new = r["net_exp"] - r["dead_share"] * 0.01
            score_curr = best_ts_r["net_exp"] - best_ts_r["dead_share"] * 0.01
            if score_new > score_curr:
                best_ts = hours
                best_ts_r = r

    accepted_ts = best_ts

    print(f"\n{'=' * 70}")
    print("ROW C VERDICT")
    print(f"{'=' * 70}")
    if best_ts == 168:
        c_verdict = "KEEP_BASELINE"
        print("  KEEP BASELINE — 168h remains the best time stop")
    else:
        c_verdict = "PASS"
        print(f"  PASS — {best_ts}h")
        print(f"  Net exp: {best_ts_r['net_exp']:+.3f}% (baseline: {bl_ts['net_exp']:+.3f}%)")
        print(f"  Hold: {best_ts_r['hold_h']:.0f}h, Dead: {best_ts_r['dead_share']:.1f}%")
    print(f"  Accepted time stop: {accepted_ts}h")

    # ═══════════════════════════════════════════════
    #  Full combination
    # ═══════════════════════════════════════════════
    print("\n\n" + "=" * 90)
    print("FULL COMBINATION — ALL ACCEPTED EXITS")
    print("=" * 90)

    combo = run_sm(btc, "Cycle 2 Combined",
                   tp_mode=accepted_tp_mode,
                   sl_atr_mult=accepted_sl_mult,
                   max_hold=accepted_ts)

    print(f"  TP: {accepted_tp_mode}, SL: ATR {accepted_sl_mult:.1f}, Time: {accepted_ts}h\n")
    compare_table(combo, baseline)

    # ═══════════════════════════════════════════════
    #  Year-by-year
    # ═══════════════════════════════════════════════
    print("\n\n" + "=" * 90)
    print("YEAR-BY-YEAR COMPARISON")
    print("=" * 90)

    years = sorted(btc[btc["year"] >= 2022]["year"].unique())
    for vname, vr in [("Cycle1 Baseline", baseline), ("Cycle2 Combined", combo)]:
        td = vr["td"].copy()
        td["year"] = pd.to_datetime(td["entry"]).dt.year
        print(f"\n{vname}:")
        print(f"  {'Year':<6} {'#Tr':>5} {'WR%':>6} {'AvgPnL':>10} {'NetExp':>10}")
        print(f"  {'-' * 42}")
        for yr in years:
            s = td[td["year"] == yr]
            if len(s) < 1:
                continue
            wr = (s.pnl > 0).mean() * 100
            avg = s.pnl.mean() * 100
            w = s.loc[s.pnl > 0, "pnl"].mean() * 100 if (s.pnl > 0).any() else 0
            lo = s.loc[s.pnl < 0, "pnl"].mean() * 100 if (s.pnl < 0).any() else 0
            g = wr / 100 * w + (1 - wr / 100) * lo
            n = g - RT * 100
            print(f"  {yr:<6} {len(s):>5} {wr:>5.0f}% {avg:>+9.3f}% {n:>+9.3f}%")

    # ═══════════════════════════════════════════════
    #  Side split
    # ═══════════════════════════════════════════════
    print("\n\n" + "=" * 90)
    print("SIDE-SPLIT ANALYSIS")
    print("=" * 90)

    for vname, vr in [("Cycle1 Baseline", baseline), ("Cycle2 Combined", combo)]:
        td = vr["td"]
        print(f"\n{vname}:")
        print(f"  {'Side':<8} {'#Tr':>5} {'WR%':>6} {'AvgPnL':>10} {'R:R':>7}")
        print(f"  {'-' * 40}")
        for side in ["L", "S"]:
            s = td[td["dir"] == side]
            if len(s) == 0:
                continue
            wr = (s.pnl > 0).mean() * 100
            avg = s.pnl.mean() * 100
            w = s.loc[s.pnl > 0, "pnl"].mean() * 100 if (s.pnl > 0).any() else 0
            lo = s.loc[s.pnl < 0, "pnl"].mean() * 100 if (s.pnl < 0).any() else 0
            rr_side = abs(w / lo) if lo != 0 else 0
            print(f"  {side:<8} {len(s):>5} {wr:>5.0f}% {avg:>+9.3f}% {rr_side:>7.2f}")

    # ═══════════════════════════════════════════════
    #  Final verdict
    # ═══════════════════════════════════════════════
    any_change = (accepted_tp_mode != "opposite_extreme"
                  or accepted_sl_mult != 3.0
                  or accepted_ts != 168)

    print("\n\n" + "=" * 70)
    print("CYCLE 2 — EXIT DESIGN FINAL VERDICT")
    print("=" * 70)

    print(f"\n  Row A (TP):      {a_verdict} → {accepted_tp_mode}")
    print(f"  Row B (SL):      {b_verdict} → ATR {accepted_sl_mult:.1f}")
    print(f"  Row C (Time):    {c_verdict} → {accepted_ts}h")

    print(f"\n  {'=' * 60}")
    if any_change and combo["net_exp"] > 0:
        overall = "PASS"
        print("  OVERALL: PASS — exit design improved over Cycle 1 baseline")
    elif not any_change:
        overall = "KEEP_BASELINE"
        print("  OVERALL: KEEP BASELINE — no exit variant improves on accepted")
    else:
        overall = "KEEP_BASELINE"
        print("  OVERALL: MIXED — some changes but combined weaker")
    print(f"  {'=' * 60}")

    print(f"\n  Full accepted config after Cycle 1+2:")
    print(f"    Signal:   EW pctrank(hl=84) contrarian")
    print(f"    Regime:   ADX(14) > 25")
    print(f"    Entry:    persist >= 2, cooldown = 24h")
    print(f"    TP:       {accepted_tp_mode}")
    print(f"    SL:       ATR {accepted_sl_mult:.1f}")
    print(f"    Time:     {accepted_ts}h")
    print(f"    Net exp:  {combo['net_exp']:+.3f}%")
    print(f"    Tr/yr:    {combo['tr_yr']:.1f}")
    print(f"    MDD:      {combo['mdd']:+.2f}%")

    print(f"\n  Next: Cycle 3 — portfolio_role")
