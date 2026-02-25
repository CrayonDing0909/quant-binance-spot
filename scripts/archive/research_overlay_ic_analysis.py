"""
P0: Overlay-Adjusted IC Analysis — Quant Researcher

比較 3 層 position 的 IC：
  Layer 0: Base signal (strategy only)
  Layer 1: + Vol Pause overlay (production pipeline)
  Layer 2: + Micro Accel overlay (theoretical — not wired in live)

目的：判斷 overlay 是否為真正的 alpha 來源，
      或者 base signal IC ≈ 0 的問題無法被 overlay 解決。

使用方式:
    PYTHONPATH=src python scripts/research_overlay_ic_analysis.py \
        -c config/prod_live_R3C_E3.yaml \
        --output-dir reports/ic_deep_analysis
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qtrade.config import load_config
from qtrade.data.storage import load_klines
from qtrade.strategy.base import StrategyContext
from qtrade.strategy import get_strategy
from qtrade.strategy.overlays.oi_vol_exit_overlay import (
    apply_overlay_by_mode,
)
from qtrade.strategy.overlays.microstructure_accel_overlay import (
    apply_full_micro_accel_overlay,
)

# Suppress noisy overlay logging
logging.getLogger("qtrade.strategy.overlays").setLevel(logging.WARNING)
logging.getLogger("qtrade").setLevel(logging.WARNING)


# ═══════════════════════════════════════════════════════════
# 工具函數
# ═══════════════════════════════════════════════════════════

def _load_ensemble_strategies(config_path: str) -> dict:
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    ens = raw.get("ensemble", {})
    if ens.get("enabled", False):
        return ens.get("strategies", {})
    return {}


def _load_raw_config(config_path: str) -> dict:
    """Load raw YAML config"""
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _get_strategy_for_symbol(
    symbol: str,
    default_name: str,
    default_params: dict,
    ensemble_strategies: dict,
) -> tuple[str, dict]:
    if ensemble_strategies and symbol in ensemble_strategies:
        sym_cfg = ensemble_strategies[symbol]
        return sym_cfg["name"], sym_cfg.get("params", {})
    return default_name, default_params


def compute_ic(signals: pd.Series, prices: pd.Series, forward_bars: int = 24) -> dict:
    """計算 IC + 統計量"""
    fwd_ret = prices.pct_change(forward_bars).shift(-forward_bars)
    valid = signals.notna() & fwd_ret.notna() & (signals != 0)
    sig = signals[valid]
    fwd = fwd_ret[valid]

    if len(sig) < 30:
        return {"ic": None, "pval": None, "n": len(sig), "significant": False}

    ic, pval = stats.spearmanr(sig, fwd)
    return {
        "ic": round(ic, 5),
        "pval": round(pval, 5),
        "n": len(sig),
        "significant": pval < 0.05,
    }


def compute_yearly_ic(signals: pd.Series, prices: pd.Series, forward_bars: int = 24) -> dict:
    """按年計算 IC"""
    fwd_ret = prices.pct_change(forward_bars).shift(-forward_bars)
    combined = pd.DataFrame({"signal": signals, "fwd_ret": fwd_ret}).dropna()
    combined = combined[combined["signal"] != 0]

    yearly = {}
    for year, group in combined.groupby(combined.index.year):
        if len(group) >= 30:
            ic, _ = stats.spearmanr(group["signal"], group["fwd_ret"])
            yearly[int(year)] = round(ic, 4)
    return yearly


def compute_signal_diff_stats(base: pd.Series, adjusted: pd.Series) -> dict:
    """比較兩個 position series 的差異"""
    diff = adjusted - base
    changed = (diff.abs() > 0.001)
    n_changed = changed.sum()
    total = len(base)

    # 分析 overlay 的影響
    base_active = (base.abs() > 0.001)
    adj_active = (adjusted.abs() > 0.001)

    # 被 overlay 清零的 bars
    zeroed = base_active & ~adj_active
    # 被 overlay 增強的 bars
    boosted = (adjusted.abs() > base.abs() + 0.001)
    # 被 overlay 減弱的 bars (但未完全清零)
    reduced = (adjusted.abs() < base.abs() - 0.001) & adj_active

    return {
        "total_bars": total,
        "bars_changed": int(n_changed),
        "pct_changed": round(n_changed / total, 4) if total > 0 else 0,
        "bars_zeroed": int(zeroed.sum()),
        "bars_boosted": int(boosted.sum()),
        "bars_reduced": int(reduced.sum()),
        "base_active_pct": round(base_active.mean(), 4),
        "adj_active_pct": round(adj_active.mean(), 4),
        "avg_abs_base": round(base[base_active].abs().mean(), 4) if base_active.any() else 0,
        "avg_abs_adj": round(adjusted[adj_active].abs().mean(), 4) if adj_active.any() else 0,
    }


def load_oi_series(data_dir: Path, symbol: str, df_index: pd.DatetimeIndex) -> pd.Series | None:
    """嘗試載入 OI 數據"""
    try:
        from qtrade.data.open_interest import get_oi_path, load_open_interest, align_oi_to_klines
        for provider in ["merged", "coinglass", "binance"]:
            oi_path = get_oi_path(data_dir, symbol, provider)
            oi_df = load_open_interest(oi_path)
            if oi_df is not None and not oi_df.empty:
                return align_oi_to_klines(oi_df, df_index, max_ffill_bars=2)
    except Exception:
        pass
    return None


# ═══════════════════════════════════════════════════════════
# 主分析
# ═══════════════════════════════════════════════════════════

def analyze_symbol_layers(
    symbol: str,
    df: pd.DataFrame,
    strategy_name: str,
    params: dict,
    market_type: str,
    direction: str,
    overlay_cfg: dict | None,
    micro_accel_cfg: dict | None,
    oi_series: pd.Series | None,
    interval: str = "1h",
) -> dict:
    """分析 3 層 position 的 IC"""

    ctx = StrategyContext(
        symbol=symbol,
        interval=interval,
        market_type=market_type,
        direction=direction,
    )

    strategy_func = get_strategy(strategy_name)
    prices = df["close"]

    # ── Layer 0: Base signal ──
    base_pos = strategy_func(df, ctx, params)

    # ── Layer 1: + Vol Pause overlay ──
    vol_pos = base_pos.copy()
    vol_overlay_applied = False
    if overlay_cfg and overlay_cfg.get("enabled", False):
        overlay_mode = overlay_cfg.get("mode", "vol_pause")
        overlay_params = overlay_cfg.get("params", {})
        vol_pos = apply_overlay_by_mode(
            position=base_pos.copy(),
            price_df=df,
            oi_series=oi_series,
            params=overlay_params,
            mode=overlay_mode,
        )
        vol_overlay_applied = True

    # ── Layer 2: + Micro Accel overlay (on top of vol overlay) ──
    micro_pos = vol_pos.copy()
    micro_overlay_applied = False
    if micro_accel_cfg and micro_accel_cfg.get("enabled", False):
        micro_params = micro_accel_cfg.get("params", {})
        try:
            micro_pos = apply_full_micro_accel_overlay(
                base_position=vol_pos.copy(),
                df_1h=df,
                df_5m=None,
                df_15m=None,
                oi_series=oi_series,
                params=micro_params,
            )
            micro_overlay_applied = True
        except Exception as e:
            print(f"    ⚠️  Micro accel overlay failed: {e}")

    # ── Compute IC for each layer ──
    horizons = [24, 48, 168]
    results = {
        "symbol": symbol,
        "strategy": strategy_name,
        "data_range": f"{df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}",
        "vol_overlay_applied": vol_overlay_applied,
        "micro_overlay_applied": micro_overlay_applied,
        "layers": {},
    }

    for layer_name, pos in [
        ("L0_base", base_pos),
        ("L1_vol_overlay", vol_pos),
        ("L2_micro_accel", micro_pos),
    ]:
        layer_data = {}

        # Multi-horizon IC
        for h in horizons:
            ic_data = compute_ic(pos, prices, forward_bars=h)
            layer_data[f"ic_{h}h"] = ic_data

        # Yearly IC (24h horizon)
        layer_data["yearly_ic"] = compute_yearly_ic(pos, prices, 24)

        # Signal stats
        active = (pos.abs() > 0.001)
        layer_data["active_pct"] = round(active.mean(), 4)
        layer_data["avg_abs_pos"] = round(pos[active].abs().mean(), 4) if active.any() else 0
        layer_data["signal_std"] = round(pos[active].std(), 4) if active.any() else 0

        results["layers"][layer_name] = layer_data

    # ── Overlay diff stats ──
    results["vol_diff"] = compute_signal_diff_stats(base_pos, vol_pos)
    results["micro_diff"] = compute_signal_diff_stats(vol_pos, micro_pos)

    return results


def print_symbol_result(r: dict):
    """美觀輸出"""
    sym = r["symbol"]
    print(f"\n{'═' * 90}")
    print(f"  {sym}  ({r['strategy']})  —  {r['data_range']}")
    print(f"  Vol overlay: {'✅' if r['vol_overlay_applied'] else '❌'}  "
          f"| Micro accel: {'✅' if r['micro_overlay_applied'] else '❌'}")
    print(f"{'═' * 90}")

    # IC comparison table
    print(f"\n  ── IC Comparison (Spearman rank correlation) ──")
    print(f"  {'Layer':<20} {'IC_24h':>10} {'p-val':>8} {'sig':>5}  "
          f"{'IC_48h':>10} {'IC_168h':>10} {'Active%':>8} {'AvgAbs':>8}")
    print(f"  {'─' * 86}")

    for layer_name in ["L0_base", "L1_vol_overlay", "L2_micro_accel"]:
        ld = r["layers"][layer_name]
        ic24 = ld["ic_24h"]
        ic48 = ld["ic_48h"]
        ic168 = ld["ic_168h"]

        ic24_str = f"{ic24['ic']:+.5f}" if ic24["ic"] is not None else "N/A"
        p24_str = f"{ic24['pval']:.5f}" if ic24["pval"] is not None else "N/A"
        sig24 = "***" if ic24.get("pval") and ic24["pval"] < 0.001 else (
            "**" if ic24.get("pval") and ic24["pval"] < 0.01 else (
                "*" if ic24.get("pval") and ic24["pval"] < 0.05 else "ns"))
        ic48_str = f"{ic48['ic']:+.5f}" if ic48["ic"] is not None else "N/A"
        ic168_str = f"{ic168['ic']:+.5f}" if ic168["ic"] is not None else "N/A"

        label_map = {
            "L0_base": "L0: Base Signal",
            "L1_vol_overlay": "L1: +Vol Overlay",
            "L2_micro_accel": "L2: +Micro Accel",
        }

        print(f"  {label_map[layer_name]:<20} {ic24_str:>10} {p24_str:>8} {sig24:>5}  "
              f"{ic48_str:>10} {ic168_str:>10} {ld['active_pct']:>7.1%} {ld['avg_abs_pos']:>8.4f}")

    # IC delta (L1 vs L0)
    l0_ic = r["layers"]["L0_base"]["ic_24h"]["ic"]
    l1_ic = r["layers"]["L1_vol_overlay"]["ic_24h"]["ic"]
    l2_ic = r["layers"]["L2_micro_accel"]["ic_24h"]["ic"]

    if l0_ic is not None and l1_ic is not None:
        delta_vol = l1_ic - l0_ic
        print(f"\n  IC Δ (Vol overlay effect):    {delta_vol:+.5f}  "
              f"({'improved' if delta_vol > 0.001 else 'degraded' if delta_vol < -0.001 else 'neutral'})")
    if l1_ic is not None and l2_ic is not None:
        delta_micro = l2_ic - l1_ic
        print(f"  IC Δ (Micro accel effect):    {delta_micro:+.5f}  "
              f"({'improved' if delta_micro > 0.001 else 'degraded' if delta_micro < -0.001 else 'neutral'})")

    # Overlay diff stats
    vd = r["vol_diff"]
    print(f"\n  ── Vol Overlay Impact ──")
    print(f"    Bars changed: {vd['bars_changed']:,} ({vd['pct_changed']:.1%})  "
          f"| Zeroed: {vd['bars_zeroed']:,}  | Reduced: {vd['bars_reduced']:,}  "
          f"| Boosted: {vd['bars_boosted']:,}")

    md = r["micro_diff"]
    print(f"  ── Micro Accel Impact ──")
    print(f"    Bars changed: {md['bars_changed']:,} ({md['pct_changed']:.1%})  "
          f"| Zeroed: {md['bars_zeroed']:,}  | Reduced: {md['bars_reduced']:,}  "
          f"| Boosted: {md['bars_boosted']:,}")

    # Yearly IC comparison
    print(f"\n  ── Yearly IC: L0 vs L1 vs L2 ──")
    all_years = sorted(set(
        list(r["layers"]["L0_base"]["yearly_ic"].keys()) +
        list(r["layers"]["L1_vol_overlay"]["yearly_ic"].keys()) +
        list(r["layers"]["L2_micro_accel"]["yearly_ic"].keys())
    ))
    print(f"    {'Year':<6} {'L0_base':>10} {'L1_vol':>10} {'L2_micro':>10}  {'L1-L0':>8} {'L2-L1':>8}")
    for year in all_years:
        l0 = r["layers"]["L0_base"]["yearly_ic"].get(year)
        l1 = r["layers"]["L1_vol_overlay"]["yearly_ic"].get(year)
        l2 = r["layers"]["L2_micro_accel"]["yearly_ic"].get(year)

        l0_s = f"{l0:+.4f}" if l0 is not None else "N/A"
        l1_s = f"{l1:+.4f}" if l1 is not None else "N/A"
        l2_s = f"{l2:+.4f}" if l2 is not None else "N/A"

        d1 = f"{l1 - l0:+.4f}" if (l0 is not None and l1 is not None) else "--"
        d2 = f"{l2 - l1:+.4f}" if (l1 is not None and l2 is not None) else "--"

        print(f"    {year:<6} {l0_s:>10} {l1_s:>10} {l2_s:>10}  {d1:>8} {d2:>8}")


def print_portfolio_summary(results: list[dict]):
    """Portfolio 層面摘要"""
    print(f"\n\n{'═' * 110}")
    print(f"  PORTFOLIO OVERLAY IC SUMMARY — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'═' * 110}")

    print(f"\n  ── IC_24h Comparison (all symbols) ──")
    print(f"  {'Symbol':<10} {'L0_base':>10} {'L1_vol':>10} {'L2_micro':>10}  "
          f"{'L1-L0':>8} {'L2-L1':>8} {'L2-L0':>8}  {'Verdict'}")
    print(f"  {'─' * 100}")

    l0_ics = []
    l1_ics = []
    l2_ics = []

    for r in results:
        sym = r["symbol"]
        l0 = r["layers"]["L0_base"]["ic_24h"]["ic"]
        l1 = r["layers"]["L1_vol_overlay"]["ic_24h"]["ic"]
        l2 = r["layers"]["L2_micro_accel"]["ic_24h"]["ic"]

        l0_s = f"{l0:+.5f}" if l0 is not None else "N/A"
        l1_s = f"{l1:+.5f}" if l1 is not None else "N/A"
        l2_s = f"{l2:+.5f}" if l2 is not None else "N/A"

        d_vol = (l1 - l0) if (l0 is not None and l1 is not None) else None
        d_micro = (l2 - l1) if (l1 is not None and l2 is not None) else None
        d_total = (l2 - l0) if (l0 is not None and l2 is not None) else None

        d_vol_s = f"{d_vol:+.5f}" if d_vol is not None else "--"
        d_micro_s = f"{d_micro:+.5f}" if d_micro is not None else "--"
        d_total_s = f"{d_total:+.5f}" if d_total is not None else "--"

        # Verdict
        if l2 is not None and l2 > 0.03:
            verdict = "🟢 STRONG"
        elif l2 is not None and l2 > 0.01:
            verdict = "🟡 WEAK"
        elif l2 is not None and l2 > -0.01:
            verdict = "⚫ NOISE"
        else:
            verdict = "🔴 NEGATIVE"

        if l0 is not None: l0_ics.append(l0)
        if l1 is not None: l1_ics.append(l1)
        if l2 is not None: l2_ics.append(l2)

        print(f"  {sym:<10} {l0_s:>10} {l1_s:>10} {l2_s:>10}  "
              f"{d_vol_s:>8} {d_micro_s:>8} {d_total_s:>8}  {verdict}")

    # Averages
    print(f"  {'─' * 100}")
    avg_l0 = np.mean(l0_ics) if l0_ics else 0
    avg_l1 = np.mean(l1_ics) if l1_ics else 0
    avg_l2 = np.mean(l2_ics) if l2_ics else 0
    print(f"  {'AVERAGE':<10} {avg_l0:+10.5f} {avg_l1:+10.5f} {avg_l2:+10.5f}  "
          f"{avg_l1 - avg_l0:+8.5f} {avg_l2 - avg_l1:+8.5f} {avg_l2 - avg_l0:+8.5f}")

    # Key findings
    print(f"\n  ── Key Findings ──")

    # 1. Vol overlay effect
    vol_improved = sum(1 for r in results
                       if r["layers"]["L1_vol_overlay"]["ic_24h"]["ic"] is not None
                       and r["layers"]["L0_base"]["ic_24h"]["ic"] is not None
                       and r["layers"]["L1_vol_overlay"]["ic_24h"]["ic"] > r["layers"]["L0_base"]["ic_24h"]["ic"] + 0.001)
    vol_degraded = sum(1 for r in results
                       if r["layers"]["L1_vol_overlay"]["ic_24h"]["ic"] is not None
                       and r["layers"]["L0_base"]["ic_24h"]["ic"] is not None
                       and r["layers"]["L1_vol_overlay"]["ic_24h"]["ic"] < r["layers"]["L0_base"]["ic_24h"]["ic"] - 0.001)
    print(f"    Vol Overlay:   improved {vol_improved}/{len(results)}, degraded {vol_degraded}/{len(results)}")

    # 2. Micro accel effect
    micro_improved = sum(1 for r in results
                         if r["layers"]["L2_micro_accel"]["ic_24h"]["ic"] is not None
                         and r["layers"]["L1_vol_overlay"]["ic_24h"]["ic"] is not None
                         and r["layers"]["L2_micro_accel"]["ic_24h"]["ic"] > r["layers"]["L1_vol_overlay"]["ic_24h"]["ic"] + 0.001)
    micro_degraded = sum(1 for r in results
                         if r["layers"]["L2_micro_accel"]["ic_24h"]["ic"] is not None
                         and r["layers"]["L1_vol_overlay"]["ic_24h"]["ic"] is not None
                         and r["layers"]["L2_micro_accel"]["ic_24h"]["ic"] < r["layers"]["L1_vol_overlay"]["ic_24h"]["ic"] - 0.001)
    print(f"    Micro Accel:   improved {micro_improved}/{len(results)}, degraded {micro_degraded}/{len(results)}")

    # 3. Symbols with strong final IC
    strong = [r["symbol"] for r in results
              if r["layers"]["L2_micro_accel"]["ic_24h"]["ic"] is not None
              and r["layers"]["L2_micro_accel"]["ic_24h"]["ic"] > 0.03]
    weak_pos = [r["symbol"] for r in results
                if r["layers"]["L2_micro_accel"]["ic_24h"]["ic"] is not None
                and 0 < r["layers"]["L2_micro_accel"]["ic_24h"]["ic"] <= 0.03]
    noise = [r["symbol"] for r in results
             if r["layers"]["L2_micro_accel"]["ic_24h"]["ic"] is not None
             and abs(r["layers"]["L2_micro_accel"]["ic_24h"]["ic"]) <= 0.01]
    negative = [r["symbol"] for r in results
                if r["layers"]["L2_micro_accel"]["ic_24h"]["ic"] is not None
                and r["layers"]["L2_micro_accel"]["ic_24h"]["ic"] < -0.01]

    print(f"\n    🟢 Strong IC (>0.03):     {', '.join(strong) if strong else 'NONE'}")
    print(f"    🟡 Weak positive IC:      {', '.join(weak_pos) if weak_pos else 'NONE'}")
    print(f"    ⚫ Noise (|IC| ≤ 0.01):   {', '.join(noise) if noise else 'NONE'}")
    print(f"    🔴 Negative IC (<-0.01):  {', '.join(negative) if negative else 'NONE'}")

    # ── FINAL VERDICT ──
    print(f"\n  ╔{'═' * 80}╗")

    # Does overlay save the portfolio?
    l0_pos_count = sum(1 for ic in l0_ics if ic > 0.01)
    l2_pos_count = sum(1 for ic in l2_ics if ic > 0.01)
    avg_improvement = avg_l2 - avg_l0

    if avg_l2 > 0.02:
        print(f"  ║ 🟢 Overlays bring portfolio IC to acceptable level (avg={avg_l2:.4f})")
        print(f"  ║    Overlay contribution: {avg_improvement:+.5f} ({avg_improvement/max(abs(avg_l0), 0.001)*100:+.0f}%)")
        print(f"  ║    RECOMMENDATION: Overlays are important — ensure they're wired correctly in live")
    elif avg_l2 > 0 and l2_pos_count >= 3:
        print(f"  ║ 🟡 Overlays provide marginal improvement (avg={avg_l2:.4f})")
        print(f"  ║    {l2_pos_count} symbols with IC > 0.01 after overlay")
        print(f"  ║    RECOMMENDATION: Overlays help but base signal is fundamentally weak")
    elif avg_improvement > 0.005:
        print(f"  ║ 🟡 Overlays improve IC but portfolio remains weak (avg={avg_l2:.4f})")
        print(f"  ║    RECOMMENDATION: Overlays add marginal value; consider parameter refresh")
    else:
        print(f"  ║ 🔴 Overlays do NOT solve the IC problem (avg L0={avg_l0:.4f} → L2={avg_l2:.4f})")
        print(f"  ║    Overlay contribution: {avg_improvement:+.5f}")
        print(f"  ║    RECOMMENDATION: Base signal alpha is the root issue.")
        print(f"  ║    Overlays (vol-pause only reduces, micro-accel not wired) cannot")
        print(f"  ║    generate alpha where none exists. Need fundamental strategy review.")

    print(f"  ║")

    # Micro accel wiring finding
    micro_any_applied = any(r["micro_overlay_applied"] for r in results)
    if not micro_any_applied:
        print(f"  ║ ⚠️  FINDING: micro_accel_overlay is configured but NOT applied in backtest/live!")
        print(f"  ║    This is a dead config — no micro-accel effect in production.")

    print(f"  ╚{'═' * 80}╝")


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="P0: Overlay-Adjusted IC Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("--symbol", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_cfg = _load_raw_config(args.config)
    market_type = cfg.market_type_str
    direction = cfg.direction
    interval = cfg.market.interval
    symbols = [args.symbol] if args.symbol else cfg.market.symbols

    ensemble_strategies = _load_ensemble_strategies(args.config)
    default_strategy_name = cfg.strategy.name
    default_params = cfg.strategy.params

    # Extract overlay configs from raw YAML
    overlay_cfg = raw_cfg.get("strategy", {}).get("overlay")
    micro_accel_cfg = raw_cfg.get("strategy", {}).get("micro_accel_overlay")

    print(f"📊 P0: Overlay-Adjusted IC Analysis")
    print(f"   Config: {args.config}")
    print(f"   Market: {market_type} | Direction: {direction} | Interval: {interval}")
    print(f"   Default Strategy: {default_strategy_name}")
    if ensemble_strategies:
        routing = ", ".join(f"{s}→{v['name']}" for s, v in ensemble_strategies.items())
        print(f"   Ensemble Routing: {routing}")
    print(f"   OI/Vol Overlay: {'✅ ' + overlay_cfg.get('mode', '?') if overlay_cfg and overlay_cfg.get('enabled') else '❌ disabled'}")
    print(f"   Micro Accel:    {'✅' if micro_accel_cfg and micro_accel_cfg.get('enabled') else '❌ disabled'}")
    print(f"   Symbols: {', '.join(symbols)}")

    results = []

    for sym in symbols:
        sym_strategy, sym_params = _get_strategy_for_symbol(
            sym, default_strategy_name, default_params, ensemble_strategies
        )
        if sym_strategy == default_strategy_name:
            sym_params = cfg.strategy.get_params(sym)

        data_path = cfg.data_dir / "binance" / market_type / interval / f"{sym}.parquet"
        if not data_path.exists():
            print(f"\n⚠️  {sym}: 數據不存在，跳過")
            continue

        df = load_klines(data_path)
        if cfg.market.start:
            df = df[df.index >= cfg.market.start]

        # Load OI data
        oi_series = load_oi_series(cfg.data_dir, sym, df.index)
        oi_status = f"✅ {len(oi_series.dropna())} bars" if oi_series is not None else "❌"

        print(f"\n🔍 {sym} ({sym_strategy}) | {len(df)} bars | OI: {oi_status}")

        try:
            result = analyze_symbol_layers(
                symbol=sym,
                df=df,
                strategy_name=sym_strategy,
                params=sym_params,
                market_type=market_type,
                direction=direction,
                overlay_cfg=overlay_cfg,
                micro_accel_cfg=micro_accel_cfg,
                oi_series=oi_series,
                interval=interval,
            )
            results.append(result)
            print_symbol_result(result)
        except Exception as e:
            print(f"❌ {sym} 失敗: {e}")
            import traceback
            traceback.print_exc()

    if not results:
        print("\n❌ 無結果")
        return

    print_portfolio_summary(results)

    # Save JSON
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = out_dir / f"overlay_ic_analysis_{timestamp}.json"

        def _default(obj):
            import numpy as np
            if isinstance(obj, (np.bool_, bool)): return bool(obj)
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return str(obj)

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(
                {"timestamp": datetime.now(timezone.utc).isoformat(),
                 "config": args.config,
                 "results": {r["symbol"]: r for r in results}},
                f, indent=2, default=_default, ensure_ascii=False,
            )
        print(f"\n📁 JSON 報告: {report_path}")


if __name__ == "__main__":
    main()
