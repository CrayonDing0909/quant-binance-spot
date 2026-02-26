"""
BTC HTF Filter 變體研究：為什麼 BTC 的 HTF filter 改善最小 (+0.04 SR)?

假說：
1. BTC 使用 meta_blend (30% breakout_vol_atr + 70% tsmom_carry_v2)，
   HTF filter 只影響 70% 的 tsmom_carry_v2，效果被稀釋
2. BTC 的 TSMOM lookback=720h + EMA 30/100，但 4h HTF filter 用 EMA 20/50
   → 時間尺度不匹配，4h 過濾可能增加雜訊
3. Alpha Researcher P1 發現 daily_only alignment hit rate=70.9%，
   而 4h_only 是最差的 (25.5%) → BTC 可能更適合 daily-dominant filter

測試方案：
  A: No HTF filter (baseline)
  B: Default HTF (4h EMA 20/50) — 現有 research config
  C: Longer 4h EMA (30/100) — 匹配 BTC 的內建 EMA 週期
  D: Daily-dominant — 放鬆 4h veto (htf_no_confirm=0.7, htf_4h_only_confirm=0.3)
  E: Daily-dominant + Longer EMA — 結合 C+D
  F: Pure daily — 幾乎完全忽略 4h (htf_no_confirm=0.85, htf_4h_only_confirm=0.0)

每個方案跑兩次：
  1. meta_blend 模式（含 30% breakout_vol_atr，真實部署場景）
  2. 純 tsmom_carry_v2 btc_enhanced（隔離 HTF filter 效果）

Usage:
    PYTHONPATH=src python scripts/research_btc_htf_variants.py
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# ── Setup paths ────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))

from qtrade.config import load_config
from qtrade.backtest.run_backtest import run_symbol_backtest, BacktestResult

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────
SYMBOL = "BTCUSDT"
BASELINE_CFG = "config/prod_candidate_meta_blend.yaml"
RESEARCH_CFG = "config/research_htf_filter.yaml"
DATA_DIR = Path("data")
KLINE_DIR = DATA_DIR / "binance" / "futures" / "1h"

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = Path("reports/research/btc_htf_variants") / TIMESTAMP
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── BTC tsmom_carry_v2 base params (btc_enhanced tier) ────
BTC_V2_BASE_PARAMS = {
    "combination_mode": "confirmatory",
    "tier": "btc_enhanced",
    "tsmom_lookback": 720,
    "tsmom_ema_fast": 30,
    "tsmom_ema_slow": 100,
    "tsmom_vol_target": 0.15,
    "tsmom_agree_weight": 1.0,
    "tsmom_disagree_weight": 0.3,
    "basis_ema_fast": 24,
    "basis_ema_slow": 168,
    "basis_tanh_scale": 50.0,
    "carry_agree_scale": 1.0,
    "carry_disagree_scale": 0.6,
    "carry_neutral_scale": 0.85,
    "carry_confirm_smoothing": 24,
    "composite_ema": 12,
    "position_step": 0.1,
    "min_change_threshold": 0.05,
}

# ── HTF Variants ───────────────────────────────────────────
HTF_VARIANTS = {
    "A_no_htf": {
        "desc": "Baseline — no HTF filter",
        "htf_overrides": {"htf_filter_enabled": False},
    },
    "B_default_htf": {
        "desc": "Default HTF (4h EMA 20/50)",
        "htf_overrides": {"htf_filter_enabled": True},
    },
    "C_longer_ema": {
        "desc": "Longer 4h EMA (30/100) to match BTC TSMOM",
        "htf_overrides": {
            "htf_filter_enabled": True,
            "htf_4h_ema_fast": 30,
            "htf_4h_ema_slow": 100,
        },
    },
    "D_daily_dominant": {
        "desc": "Daily-dominant (relax 4h veto, trust daily)",
        "htf_overrides": {
            "htf_filter_enabled": True,
            "htf_no_confirm": 0.7,          # was 0.0 → allow 70% even if 4h disagrees
            "htf_4h_only_confirm": 0.3,      # was 0.7 → penalize 4h-only agreement
        },
    },
    "E_daily_dom_long_ema": {
        "desc": "Daily-dominant + Longer 4h EMA (30/100)",
        "htf_overrides": {
            "htf_filter_enabled": True,
            "htf_4h_ema_fast": 30,
            "htf_4h_ema_slow": 100,
            "htf_no_confirm": 0.7,
            "htf_4h_only_confirm": 0.3,
        },
    },
    "F_pure_daily": {
        "desc": "Pure daily (almost ignore 4h entirely)",
        "htf_overrides": {
            "htf_filter_enabled": True,
            "htf_no_confirm": 0.85,          # almost no penalty for 4h disagree
            "htf_4h_only_confirm": 0.0,       # zero when only 4h agrees (daily is master)
        },
    },
}


def _run_backtest(
    cfg_path: str,
    symbol: str,
    strategy_name: str | None = None,
    strategy_params: dict | None = None,
) -> BacktestResult | None:
    """Run a single-symbol backtest."""
    cfg = load_config(cfg_path)
    bt_cfg = cfg.to_backtest_dict(symbol=symbol)

    if strategy_name:
        name = strategy_name
    else:
        name = cfg.strategy.name

    if strategy_params:
        bt_cfg["strategy_params"] = {**bt_cfg["strategy_params"], **strategy_params}

    data_path = KLINE_DIR / f"{symbol}.parquet"
    if not data_path.exists():
        print(f"  ⚠️  Data not found: {data_path}")
        return None

    try:
        return run_symbol_backtest(
            symbol, data_path, bt_cfg,
            strategy_name=name,
            data_dir=DATA_DIR,
        )
    except Exception as e:
        print(f"  ❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def _extract_stats(res: BacktestResult) -> dict:
    """Extract key metrics from a BacktestResult."""
    pf = res.pf
    stats = pf.stats()

    return {
        "total_return_pct": round(float(stats.get("Total Return [%]", 0)), 2),
        "sharpe": round(float(stats.get("Sharpe Ratio", 0)), 3),
        "sortino": round(float(stats.get("Sortino Ratio", 0)), 3),
        "max_dd_pct": round(float(stats.get("Max Drawdown [%]", 0)), 2),
        "calmar": round(float(stats.get("Calmar Ratio", 0)), 3),
        "total_trades": int(stats.get("Total Trades", 0)),
        "win_rate": round(float(stats.get("Win Rate [%]", 0)), 1),
    }


def run_meta_blend_variants():
    """Test HTF variants via meta_blend (realistic BTC deployment scenario)."""
    print("\n" + "=" * 70)
    print("  Phase 1: meta_blend 模式（30% breakout + 70% tsmom_carry_v2）")
    print("  HTF filter 只影響 70% 的 tsmom_carry_v2 部分")
    print("=" * 70)

    results = {}

    for name, variant in HTF_VARIANTS.items():
        print(f"\n  [{name}] {variant['desc']}")
        htf = variant["htf_overrides"]

        if name == "A_no_htf":
            # Use production baseline (no HTF filter)
            res = _run_backtest(BASELINE_CFG, SYMBOL)
        else:
            # Use research config and override HTF params in the
            # tsmom_carry_v2 sub-strategy params for BTC
            cfg = load_config(RESEARCH_CFG)
            bt_cfg = cfg.to_backtest_dict(symbol=SYMBOL)

            # The config already has htf_filter_enabled: true for BTC's
            # tsmom_carry_v2 sub-strategy. We need to override specific HTF params.
            params = bt_cfg["strategy_params"]
            params.update(htf)

            res = _run_backtest(
                RESEARCH_CFG, SYMBOL,
                strategy_params=params,
            )

        if res:
            stats = _extract_stats(res)
            results[name] = {**stats, "desc": variant["desc"]}
            print(f"    SR={stats['sharpe']:.3f}, Return={stats['total_return_pct']:.1f}%, "
                  f"MDD={stats['max_dd_pct']:.1f}%, Trades={stats['total_trades']}")
        else:
            print(f"    ❌ FAILED")
            results[name] = {"desc": variant["desc"], "error": True}

    return results


def run_pure_v2_variants():
    """Test HTF variants on pure tsmom_carry_v2 btc_enhanced (isolate effect)."""
    print("\n" + "=" * 70)
    print("  Phase 2: 純 tsmom_carry_v2 btc_enhanced（隔離 HTF filter 效果）")
    print("  不含 breakout_vol_atr，純粹看 HTF filter 對 BTC TSMOM 的影響")
    print("=" * 70)

    results = {}

    for name, variant in HTF_VARIANTS.items():
        print(f"\n  [{name}] {variant['desc']}")
        htf = variant["htf_overrides"]

        # Build params: BTC v2 base + HTF overrides
        params = {**BTC_V2_BASE_PARAMS, **htf}

        res = _run_backtest(
            BASELINE_CFG, SYMBOL,
            strategy_name="tsmom_carry_v2",
            strategy_params=params,
        )

        if res:
            stats = _extract_stats(res)
            results[name] = {**stats, "desc": variant["desc"]}
            print(f"    SR={stats['sharpe']:.3f}, Return={stats['total_return_pct']:.1f}%, "
                  f"MDD={stats['max_dd_pct']:.1f}%, Trades={stats['total_trades']}")
        else:
            print(f"    ❌ FAILED")
            results[name] = {"desc": variant["desc"], "error": True}

    return results


def print_comparison(title: str, results: dict):
    """Pretty-print a comparison table."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")

    baseline = results.get("A_no_htf", {})
    baseline_sr = baseline.get("sharpe", 0)

    header = f"  {'Variant':<25} {'Sharpe':>8} {'Δ SR':>8} {'Return%':>9} {'MDD%':>7} {'Calmar':>8} {'Trades':>7}"
    print(header)
    print("  " + "-" * 78)

    for name, stats in results.items():
        if stats.get("error"):
            print(f"  {name:<25} {'FAILED':>8}")
            continue

        sr = stats["sharpe"]
        delta = sr - baseline_sr
        marker = "  ★" if sr == max(s.get("sharpe", 0) for s in results.values() if not s.get("error")) else ""

        print(
            f"  {name:<25} {sr:>8.3f} {delta:>+8.3f} "
            f"{stats['total_return_pct']:>8.1f}% {stats['max_dd_pct']:>6.1f}% "
            f"{stats['calmar']:>8.3f} {stats['total_trades']:>7d}{marker}"
        )


def main():
    print("🔬 BTC HTF Filter 變體研究")
    print(f"📁 輸出: {OUT_DIR}")
    print(f"📅 時間: {TIMESTAMP}")

    # Phase 1: meta_blend mode
    mb_results = run_meta_blend_variants()
    print_comparison("Phase 1: meta_blend (30% breakout + 70% V2)", mb_results)

    # Phase 2: pure tsmom_carry_v2
    v2_results = run_pure_v2_variants()
    print_comparison("Phase 2: pure tsmom_carry_v2 btc_enhanced", v2_results)

    # ── Find best variants ──
    best_mb = max(
        ((k, v) for k, v in mb_results.items() if not v.get("error")),
        key=lambda x: x[1]["sharpe"],
    )
    best_v2 = max(
        ((k, v) for k, v in v2_results.items() if not v.get("error")),
        key=lambda x: x[1]["sharpe"],
    )

    print(f"\n{'=' * 80}")
    print(f"  📊 SUMMARY")
    print(f"{'=' * 80}")
    print(f"  Best meta_blend variant:  {best_mb[0]} (SR={best_mb[1]['sharpe']:.3f})")
    print(f"  Best pure V2 variant:     {best_v2[0]} (SR={best_v2[1]['sharpe']:.3f})")

    # ── Diagnosis ──
    a_mb = mb_results.get("A_no_htf", {}).get("sharpe", 0)
    b_mb = mb_results.get("B_default_htf", {}).get("sharpe", 0)
    a_v2 = v2_results.get("A_no_htf", {}).get("sharpe", 0)
    b_v2 = v2_results.get("B_default_htf", {}).get("sharpe", 0)

    print(f"\n  Diagnosis: HTF filter effect dilution")
    print(f"    Pure V2 delta (default HTF):    {b_v2 - a_v2:+.3f} SR")
    print(f"    meta_blend delta (default HTF):  {b_mb - a_mb:+.3f} SR")
    if a_v2 > 0 and b_v2 > a_v2:
        dilution = 1 - (b_mb - a_mb) / (b_v2 - a_v2) if (b_v2 - a_v2) != 0 else 0
        print(f"    Dilution from breakout_vol_atr:  {dilution:.0%}")

    # ── Save results ──
    all_results = {
        "timestamp": TIMESTAMP,
        "symbol": SYMBOL,
        "meta_blend": mb_results,
        "pure_v2": v2_results,
        "best_meta_blend": best_mb[0],
        "best_pure_v2": best_v2[0],
    }

    out_file = OUT_DIR / "btc_htf_variants.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  ✅ 結果已保存: {out_file}")


if __name__ == "__main__":
    main()
