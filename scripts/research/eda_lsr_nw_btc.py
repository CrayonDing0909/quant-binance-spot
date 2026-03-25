"""
BTC-only LSR Contrarian + NW Envelope Mean-Reversion EDA
=========================================================
Date: 2026-03-24
Task: research_20260324_105700_lsr_contrarian_standalone_revisit

Hypothesis: Retail LSR extreme + price deviation from NW Envelope
= high-conviction mean-reversion entry. BTC-only focus.

All IC computations use causal shift(1) per protocol.
"""

import sys, warnings
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as sp_stats

DATA_DIR = Path("data/binance/futures")
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
           "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT"]

NW_BANDWIDTH = 8.0
NW_ALPHA = 1.0
NW_LOOKBACK = 200
NW_ENVELOPE_MULT = 2.5
NW_ENVELOPE_WINDOW = 200

LSR_WINDOW = 168
ENTRY_HI = 0.85
ENTRY_LO = 0.15
TP_HI = 0.75
TP_LO = 0.25
SL_ATR_MULT = 3.0
MAX_HOLD = 168

# ── Pure-numpy NW functions (no numba dependency) ─────────────────
def rq_kernel_weights(lookback, bandwidth, alpha):
    d = np.arange(lookback + 1, dtype=np.float64)
    return (1.0 + d * d / (2.0 * alpha * bandwidth * bandwidth)) ** (-alpha)

def causal_nw_regression(close_arr, weights, lookback):
    n = len(close_arr)
    est = np.empty(n)
    est[0] = close_arr[0]
    cum_wc = weights[0] * close_arr[0]
    cum_w = weights[0]
    for t in range(1, n):
        ws = min(t + 1, lookback + 1)
        num = 0.0
        den = 0.0
        for j in range(ws):
            w = weights[j]
            num += w * close_arr[t - j]
            den += w
        est[t] = num / den
    return est

def compute_nw_envelope(close_series, bw=8.0, alpha=1.0, lb=200, mult=2.5, ew=200):
    arr = close_series.values.astype(np.float64)
    w = rq_kernel_weights(lb, bw, alpha)
    est = causal_nw_regression(arr, w, lb)
    nw = pd.Series(est, index=close_series.index)
    mae = (close_series - nw).abs().rolling(ew, min_periods=1).mean()
    return nw, nw + mult * mae, nw - mult * mae

def compute_pctrank(series, window=168):
    min_p = max(window // 2, 24)
    return series.rolling(window, min_periods=min_p).apply(
        lambda x: sp_stats.percentileofscore(x, x.iloc[-1]) / 100.0, raw=False)

def causal_rank_ic(signal, fwd_ret):
    """Causal IC: signal.shift(1).corr(fwd_ret, method='spearman')"""
    s = signal.shift(1)
    mask = s.notna() & fwd_ret.notna()
    n = mask.sum()
    if n < 200:
        return np.nan, n
    return s[mask].corr(fwd_ret[mask], method="spearman"), n

# ── Load Data ─────────────────────────────────────────────────────
print("=" * 80)
print("BTC-only LSR Contrarian + NW Envelope EDA — 2026-03-24")
print("=" * 80)

klines_all = {}
lsr_all = {}
for sym in SYMBOLS:
    kl = pd.read_parquet(DATA_DIR / "1h" / f"{sym}.parquet")
    kl.index = kl.index.tz_localize(None) if kl.index.tz else kl.index
    klines_all[sym] = kl
    lp = DATA_DIR / "derivatives" / "lsr" / f"{sym}.parquet"
    if lp.exists():
        s = pd.read_parquet(lp)["lsr"]
        s.index = s.index.tz_localize(None) if s.index.tz else s.index
        lsr_all[sym] = s

btc = klines_all["BTCUSDT"].copy()
btc["lsr"] = lsr_all["BTCUSDT"].reindex(btc.index, method="ffill", limit=2)

oi_df = pd.read_parquet(DATA_DIR / "open_interest" / "merged" / "BTCUSDT.parquet")
oi_df.index = oi_df.index.tz_localize(None) if oi_df.index.tz else oi_df.index
btc["oi"] = oi_df["sumOpenInterestValue"].reindex(btc.index, method="ffill", limit=4)

fr_df = pd.read_parquet(DATA_DIR / "funding_rate" / "BTCUSDT.parquet")
fr_df.index = fr_df.index.tz_localize(None) if fr_df.index.tz else fr_df.index
btc["fr"] = fr_df["funding_rate"].reindex(btc.index, method="ffill")

btc = btc.loc[btc["lsr"].first_valid_index():].dropna(subset=["lsr"])
print(f"BTC period: {btc.index[0]} → {btc.index[-1]}  ({len(btc):,} bars)")

# ── Features ──────────────────────────────────────────────────────
print("Computing features (pctrank ~60s, NW ~30s)...")
btc["lsr_pctrank"] = compute_pctrank(btc["lsr"], LSR_WINDOW)

for h in [24, 48, 72]:
    btc[f"fwd_{h}h"] = btc["close"].shift(-h) / btc["close"] - 1

nw, nw_up, nw_lo = compute_nw_envelope(btc["close"], NW_BANDWIDTH, NW_ALPHA,
                                         NW_LOOKBACK, NW_ENVELOPE_MULT, NW_ENVELOPE_WINDOW)
btc["nw"] = nw; btc["nw_up"] = nw_up; btc["nw_lo"] = nw_lo

btc["nw_regime"] = "inside"
btc.loc[btc["close"] > btc["nw_up"], "nw_regime"] = "above_upper"
btc.loc[btc["close"] < btc["nw_lo"], "nw_regime"] = "below_lower"

btc["oi_chg24"] = btc["oi"].pct_change(24)
btc["oi_rising"] = btc["oi_chg24"] > 0
btc["fr_pctrank"] = compute_pctrank(btc["fr"].dropna(), LSR_WINDOW).reindex(btc.index)
btc["atr14"] = (btc["high"] - btc["low"]).rolling(14).mean()
btc["year"] = btc.index.year

print(f"NW regime dist: {btc['nw_regime'].value_counts().to_dict()}")
print(f"Valid pctrank: {btc['lsr_pctrank'].notna().sum():,}")

# ══════════════════════════════════════════════════════════════════
# TASK 1 — BTC-only Causal IC
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("TASK 1: BTC-only Causal IC  (LSR pctrank_168)")
print("=" * 80)

print("\n┌─── 1a. Overall BTC IC ───┐")
for h in [24, 48, 72]:
    ic, n = causal_rank_ic(btc["lsr_pctrank"], btc[f"fwd_{h}h"])
    print(f"  vs fwd_{h}h: IC = {ic:+.4f}  (n={n:,})")

print("\n┌─── 1b. Year-by-Year IC (BTC) ───┐")
yearly = {}
for yr in sorted(btc["year"].unique()):
    sub = btc[btc["year"] == yr]
    row = {"year": yr}
    for h in [24, 48, 72]:
        ic, n = causal_rank_ic(sub["lsr_pctrank"], sub[f"fwd_{h}h"])
        row[f"ic_{h}h"] = ic
        row[f"n_{h}h"] = n
    yearly[yr] = row
    tag = "NEG (contrarian)" if row["ic_24h"] < 0 else "POS ⚠️ (FLIP)"
    print(f"  {yr}: 24h={row['ic_24h']:+.4f}  48h={row['ic_48h']:+.4f}  72h={row['ic_72h']:+.4f}  "
          f"(n≈{row['n_24h']:,})  {tag}")

print("\n┌─── 1c. BTC vs 8-Symbol Average ───┐")
sym_ics = {}
for sym in SYMBOLS:
    kl = klines_all[sym].copy()
    lsr = lsr_all.get(sym)
    if lsr is None:
        continue
    kl["lsr"] = lsr.reindex(kl.index, method="ffill", limit=2)
    kl = kl.loc[kl["lsr"].first_valid_index():].dropna(subset=["lsr"])
    kl["pr"] = compute_pctrank(kl["lsr"], LSR_WINDOW)
    kl["fwd"] = kl["close"].shift(-24) / kl["close"] - 1
    ic, n = causal_rank_ic(kl["pr"], kl["fwd"])
    sym_ics[sym] = ic
    flag = " ← BTC" if sym == "BTCUSDT" else ""
    print(f"  {sym:12s}: IC = {ic:+.4f}  (n={n:,}){flag}")

avg_ic = np.nanmean(list(sym_ics.values()))
btc_ic = sym_ics["BTCUSDT"]
print(f"\n  8-symbol avg |IC|: {abs(avg_ic):.4f}     BTC |IC|: {abs(btc_ic):.4f}")
f1_pass = abs(btc_ic) > abs(avg_ic)
print(f"  F1 BTC stronger? {'YES ✓' if f1_pass else 'NO ✗'}")

# Year-by-year for all symbols (BTC vs avg)
print("\n┌─── 1d. Year-by-Year BTC IC vs All-Symbol Avg IC ───┐")
for yr in sorted(btc["year"].unique()):
    yr_ics = []
    for sym in SYMBOLS:
        kl = klines_all[sym].copy()
        lsr = lsr_all.get(sym)
        if lsr is None: continue
        kl["lsr"] = lsr.reindex(kl.index, method="ffill", limit=2)
        kl = kl.loc[kl["lsr"].first_valid_index():].dropna(subset=["lsr"])
        kl["pr"] = compute_pctrank(kl["lsr"], LSR_WINDOW)
        kl["fwd"] = kl["close"].shift(-24) / kl["close"] - 1
        sub = kl[kl.index.year == yr]
        ic, n = causal_rank_ic(sub["pr"], sub["fwd"])
        if not np.isnan(ic):
            yr_ics.append({"sym": sym, "ic": ic})
    if yr_ics:
        avg_yr = np.mean([r["ic"] for r in yr_ics])
        btc_yr = next((r["ic"] for r in yr_ics if r["sym"] == "BTCUSDT"), np.nan)
        btc_stronger = abs(btc_yr) > abs(avg_yr) if not np.isnan(btc_yr) else False
        print(f"  {yr}: BTC={btc_yr:+.4f}  avg={avg_yr:+.4f}  BTC stronger? {'YES' if btc_stronger else 'NO'}")

# ══════════════════════════════════════════════════════════════════
# TASK 2 — NW Envelope Regime-Conditioned IC
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("TASK 2: NW Envelope Regime-Conditioned IC")
print("=" * 80)

uncond_ic, uncond_n = causal_rank_ic(btc["lsr_pctrank"], btc["fwd_24h"])
print(f"\n  Unconditional IC (24h): {uncond_ic:+.4f}  (n={uncond_n:,})")

print("\n┌─── 2a. IC by NW Band Position ───┐")
regime_ics = {}
for regime in ["below_lower", "inside", "above_upper"]:
    sub = btc[btc["nw_regime"] == regime]
    pct = len(sub) / len(btc) * 100
    for h in [24, 48, 72]:
        ic, n = causal_rank_ic(sub["lsr_pctrank"], sub[f"fwd_{h}h"])
        regime_ics[(regime, h)] = ic
        if h == 24:
            print(f"  {regime:15s}: IC_24h={ic:+.4f}  (n={n:,}, {pct:.1f}% of time)")

print(f"\n  |IC| outside bands avg: {(abs(regime_ics.get(('above_upper',24), 0)) + abs(regime_ics.get(('below_lower',24), 0)))/2:.4f}")
print(f"  |IC| inside band:       {abs(regime_ics.get(('inside',24), 0)):.4f}")

extreme_avg = (abs(regime_ics.get(("above_upper",24),0)) + abs(regime_ics.get(("below_lower",24),0))) / 2
inside_abs = abs(regime_ics.get(("inside",24),0))
f2_pass = extreme_avg > inside_abs * 1.2
print(f"  F2 NW conditioning helps (>20% improvement)? {'YES ✓' if f2_pass else 'NO ✗'}")

print("\n┌─── 2b. Conditional IC: LSR Extreme + NW Band ───┐")
lsr_ext_long = btc["lsr_pctrank"] < ENTRY_LO
lsr_ext_short = btc["lsr_pctrank"] > ENTRY_HI
lsr_ext = lsr_ext_long | lsr_ext_short

for regime in ["below_lower", "inside", "above_upper"]:
    sub = btc[(btc["nw_regime"] == regime) & lsr_ext]
    if len(sub) < 50:
        print(f"  {regime:15s} + LSR extreme: n={len(sub)} (too few)")
        continue
    ic, n = causal_rank_ic(sub["lsr_pctrank"], sub["fwd_24h"])
    print(f"  {regime:15s} + LSR extreme: IC={ic:+.4f}  (n={n})")

print("\n┌─── 2c. Directional Conditional Returns ───┐")
# Long setup: LSR < 0.15 (bears crowded) + price < lower NW band
long_setup = lsr_ext_long & (btc["nw_regime"] == "below_lower")
short_setup = lsr_ext_short & (btc["nw_regime"] == "above_upper")

for label, mask in [("LONG (LSR<0.15 + price<lower)", long_setup),
                    ("SHORT (LSR>0.85 + price>upper)", short_setup)]:
    sub = btc[mask]
    n = mask.sum()
    print(f"\n  {label}: n={n}")
    if n >= 10:
        for h in [24, 48, 72]:
            fwd = sub[f"fwd_{h}h"].shift(1)  # causal
            m = fwd.mean() * 100
            med = fwd.median() * 100
            wr = (fwd > 0).mean() * 100
            print(f"    fwd_{h}h: mean={m:+.3f}%  median={med:+.3f}%  win_rate={wr:.1f}%")

# Long w/o NW filter for comparison
long_base = lsr_ext_long
sub_base = btc[long_base]
n_base = long_base.sum()
print(f"\n  LONG baseline (LSR<0.15, no NW filter): n={n_base}")
if n_base >= 10:
    for h in [24, 48, 72]:
        fwd = sub_base[f"fwd_{h}h"].shift(1)
        m = fwd.mean() * 100
        print(f"    fwd_{h}h: mean={m:+.3f}%")

short_base = lsr_ext_short
sub_base = btc[short_base]
n_base = short_base.sum()
print(f"\n  SHORT baseline (LSR>0.85, no NW filter): n={n_base}")
if n_base >= 10:
    for h in [24, 48, 72]:
        fwd = sub_base[f"fwd_{h}h"].shift(1)
        m = fwd.mean() * 100
        print(f"    fwd_{h}h: mean={m:+.3f}%")

# ══════════════════════════════════════════════════════════════════
# TASK 3 & 4 — Combined Signal Candidates + Estimates
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("TASK 3 & 4: Combined Signal Candidates (BTC-only)")
print("=" * 80)

def run_state_machine(df, entry_long_mask, entry_short_mask, exit_mode="lsr_tp",
                      sl_mult=3.0, max_hold=168, label=""):
    """
    Run v2-style state machine.
    exit_mode:
      'lsr_tp': TP when LSR swings to opposite extreme (baseline)
      'nw_center': TP when price reaches NW center
    """
    n = len(df)
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    open_ = df["open"].values
    pr = df["lsr_pctrank"].fillna(0.5).values
    atr = df["atr14"].fillna(0).values
    nw_vals = df["nw"].values if "nw" in df.columns else np.full(n, np.nan)
    idx = df.index
    el = entry_long_mask.reindex(df.index, fill_value=False).values
    es = entry_short_mask.reindex(df.index, fill_value=False).values

    trades = []
    state = 0
    ep = 0.0
    eb = 0

    warmup = max(LSR_WINDOW + 10, NW_LOOKBACK + 10)

    for i in range(warmup, n):
        if state == 0:
            if el[i]:
                state = 1; ep = open_[min(i+1, n-1)]; eb = i
            elif es[i]:
                state = -1; ep = open_[min(i+1, n-1)]; eb = i
        else:
            hold = i - eb
            a = atr[i]
            should_exit = False
            exit_reason = ""

            if state == 1:
                if a > 0 and low[i] <= ep - sl_mult * a:
                    should_exit = True; exit_reason = "SL"
                elif exit_mode == "lsr_tp" and pr[i] > TP_HI:
                    should_exit = True; exit_reason = "TP"
                elif exit_mode == "nw_center" and close[i] >= nw_vals[i]:
                    should_exit = True; exit_reason = "TP_NW"
                elif hold >= max_hold:
                    should_exit = True; exit_reason = "TIME"
            else:
                if a > 0 and high[i] >= ep + sl_mult * a:
                    should_exit = True; exit_reason = "SL"
                elif exit_mode == "lsr_tp" and pr[i] < TP_LO:
                    should_exit = True; exit_reason = "TP"
                elif exit_mode == "nw_center" and close[i] <= nw_vals[i]:
                    should_exit = True; exit_reason = "TP_NW"
                elif hold >= max_hold:
                    should_exit = True; exit_reason = "TIME"

            if should_exit:
                xp = open_[min(i+1, n-1)]
                pnl = (xp / ep - 1) * state if ep > 0 else 0
                trades.append({
                    "entry": idx[eb], "exit": idx[i],
                    "dir": "L" if state == 1 else "S",
                    "pnl": pnl,
                    "hold_h": (idx[i] - idx[eb]).total_seconds() / 3600,
                    "reason": exit_reason,
                })
                state = 0

    if not trades:
        return {"label": label, "n_trades": 0}

    td = pd.DataFrame(trades)
    n_years = (idx[-1] - idx[0]).days / 365.25
    wr = (td["pnl"] > 0).mean() * 100
    avg_pnl = td["pnl"].mean() * 100
    avg_w = td.loc[td["pnl"] > 0, "pnl"].mean() * 100 if (td["pnl"] > 0).any() else 0
    avg_l = td.loc[td["pnl"] < 0, "pnl"].mean() * 100 if (td["pnl"] < 0).any() else 0
    rr = abs(avg_w / avg_l) if avg_l != 0 else 0
    exp = wr/100 * avg_w + (1 - wr/100) * avg_l

    cum_pnl = td["pnl"].cumsum()
    peak = cum_pnl.cummax()
    max_dd = (cum_pnl - peak).min() * 100

    return {
        "label": label,
        "n_trades": len(td),
        "trades_yr": len(td) / n_years if n_years > 0 else 0,
        "avg_hold_h": td["hold_h"].mean(),
        "wr": wr, "avg_pnl": avg_pnl,
        "avg_w": avg_w, "avg_l": avg_l,
        "rr": rr, "expectancy": exp,
        "max_dd": max_dd,
        "td": td,
        "exit_breakdown": td["reason"].value_counts().to_dict(),
    }

pr_vals = btc["lsr_pctrank"].fillna(0.5)

# Version A: Baseline
el_a = pr_vals < ENTRY_LO
es_a = pr_vals > ENTRY_HI
res_a = run_state_machine(btc, el_a, es_a, "lsr_tp", label="A: Baseline LSR")

# Version B: LSR + NW band filter
el_b = (pr_vals < ENTRY_LO) & (btc["close"] < btc["nw_lo"])
es_b = (pr_vals > ENTRY_HI) & (btc["close"] > btc["nw_up"])
res_b = run_state_machine(btc, el_b, es_b, "lsr_tp", label="B: LSR+NW band")

# Version C: B + OI/FR confirmation
fr_lo = btc["fr_pctrank"] < 0.20
fr_hi = btc["fr_pctrank"] > 0.80
oi_up = btc["oi_rising"].fillna(False)
el_c = el_b & (oi_up | fr_lo)
es_c = es_b & (oi_up | fr_hi)
res_c = run_state_machine(btc, el_c, es_c, "lsr_tp", label="C: B+OI/FR")

# Version D: B + NW center TP
res_d = run_state_machine(btc, el_b, es_b, "nw_center", label="D: B+NW TP")

# ── Print comparison table ──
print(f"\n{'Version':<22} {'#Tr':>5} {'Tr/yr':>6} {'Hold':>6} {'WR%':>6} {'AvgPnL':>8} "
      f"{'AvgW':>7} {'AvgL':>7} {'R:R':>5} {'Expect':>8} {'MaxDD':>7}")
print("─" * 100)
for r in [res_a, res_b, res_c, res_d]:
    if r["n_trades"] == 0:
        print(f"{r['label']:<22} {'NO TRADES':>5}")
        continue
    print(f"{r['label']:<22} {r['n_trades']:>5} {r['trades_yr']:>6.1f} {r['avg_hold_h']:>5.0f}h "
          f"{r['wr']:>6.1f} {r['avg_pnl']:>+7.3f}% {r['avg_w']:>+6.2f}% {r['avg_l']:>+6.2f}% "
          f"{r['rr']:>5.2f} {r['expectancy']:>+7.3f}% {r['max_dd']:>+6.1f}%")
    print(f"{'':22s} exits: {r['exit_breakdown']}")

# ── Cost-adjusted estimates ──
print("\n┌─── Cost-Adjusted Estimates (RT=0.12%) ───┐")
RT_COST = 0.0012
for r in [res_a, res_b, res_c, res_d]:
    if r["n_trades"] == 0:
        continue
    gross_exp = r["expectancy"]
    net_exp = gross_exp - RT_COST * 100
    cost_pct = RT_COST * 100 / abs(gross_exp) * 100 if gross_exp != 0 else float("inf")
    n_years = r["n_trades"] / r["trades_yr"] if r["trades_yr"] > 0 else 1
    ann_gross = r["trades_yr"] * gross_exp / 100
    ann_net = r["trades_yr"] * net_exp / 100
    print(f"  {r['label']:<22}: gross_exp={gross_exp:+.3f}%  cost={RT_COST*100:.2f}%  "
          f"net_exp={net_exp:+.3f}%  cost_drag={cost_pct:.0f}%  "
          f"ann_net≈{ann_net*100:+.1f}%")

# ── Year-by-year performance for Version B ──
print("\n┌─── Year-by-Year Trade Stats (Version B) ───┐")
if res_b["n_trades"] > 0:
    td_b = res_b["td"]
    td_b["year"] = pd.to_datetime(td_b["entry"]).dt.year
    for yr in sorted(td_b["year"].unique()):
        sub = td_b[td_b["year"] == yr]
        if len(sub) < 2:
            continue
        wr = (sub["pnl"] > 0).mean() * 100
        avg = sub["pnl"].mean() * 100
        print(f"  {yr}: trades={len(sub)}, WR={wr:.0f}%, avgPnL={avg:+.3f}%")

# ── Conditional IC at entry bars (Versions A vs B) ──
print("\n┌─── Conditional IC at Entry Bars ───┐")
for label, mask in [("A (any LSR extreme)", el_a | es_a),
                    ("B (LSR ext + NW band)", el_b | es_b)]:
    m = mask & btc["lsr_pctrank"].notna() & btc["fwd_24h"].notna()
    sub = btc[m]
    if len(sub) > 30:
        ic, n = causal_rank_ic(sub["lsr_pctrank"], sub["fwd_24h"])
        print(f"  {label}: IC={ic:+.4f}  (n={n})")

# ══════════════════════════════════════════════════════════════════
# TASK 5 — Falsification Summary
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("TASK 5: FALSIFICATION SUMMARY")
print("=" * 80)

print(f"\n  F1. BTC IC stronger than 8-symbol avg?")
print(f"      BTC |IC|={abs(btc_ic):.4f}  avg |IC|={abs(avg_ic):.4f}")
print(f"      {'PASS ✓' if f1_pass else 'FAIL ✗ — BTC-only hypothesis weakened'}")

print(f"\n  F2. NW band conditioning improves IC?")
print(f"      |IC| outside bands avg: {extreme_avg:.4f}")
print(f"      |IC| inside band:       {inside_abs:.4f}")
print(f"      {'PASS ✓ (>20% improvement)' if f2_pass else 'FAIL ✗ — NW conditioning adds no value'}")

print(f"\n  F3. 2025-2026 IC sign stability?")
f3_status = "PASS"
for yr in [2025, 2026]:
    if yr in yearly:
        ic = yearly[yr]["ic_24h"]
        if np.isnan(ic):
            print(f"      {yr}: insufficient data")
        elif ic > 0:
            print(f"      {yr}: IC={ic:+.4f} POSITIVE → contrarian FLIP ⚠️")
            f3_status = "KILL"
        else:
            print(f"      {yr}: IC={ic:+.4f} negative ✓")
    else:
        print(f"      {yr}: no data")
print(f"      {f3_status}")

# ── Final Verdict ──
print("\n" + "═" * 80)
print("                        FINAL VERDICT")
print("═" * 80)

kills = []
if not f1_pass:
    kills.append("F1: BTC-only hypothesis not supported")
if not f2_pass:
    kills.append("F2: NW conditioning adds no value")
if f3_status == "KILL":
    kills.append("F3: 2025/2026 IC flipped positive — contrarian edge dead")

if kills:
    print("\n  ISSUES FOUND:")
    for k in kills:
        print(f"    ✗ {k}")
    if f3_status == "KILL":
        print("\n  >>> VERDICT: KILL / SHELVE — contrarian edge structurally weakening.")
    elif not f1_pass and not f2_pass:
        print("\n  >>> VERDICT: SHELVE — neither BTC-only nor NW conditioning hypothesis holds.")
    elif not f1_pass:
        print("\n  >>> VERDICT: WEAKEN — BTC-only not justified; consider multi-symbol version.")
    elif not f2_pass:
        print("\n  >>> VERDICT: WEAKEN — NW envelope adds no value; stick to raw LSR contrarian.")
else:
    print("\n  All falsification checks passed.")
    if res_b["n_trades"] > 0 and res_b["expectancy"] > RT_COST * 100:
        print(f"  >>> VERDICT: TENTATIVE GO — Version B (LSR+NW) is promising.")
        print(f"      Recommend formal backtest with WFA/CPCV validation.")
    else:
        print(f"  >>> VERDICT: MARGINAL — signal exists but edge may not survive costs.")

print("\nDone.")
