#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════
  Overlay 4-Way Comparison — R3C vs Meta-Blend × Overlay ON/OFF
═══════════════════════════════════════════════════════════════

目的：
    驗證 overlay（vol_pause）對 R3C 和 Meta-Blend 的績效影響，
    確認「carry confirmation 取代 overlay」的假設是否成立。

4 個組合：
    A. R3C naked          — overlay OFF
    B. R3C + overlay      — overlay ON（現行生產設定）
    C. Meta-Blend naked   — overlay OFF（目前候選設定）
    D. Meta-Blend + overlay — overlay ON（vol_pause）

輸出：
    - 8 幣種個別 + portfolio 彙總的 Sharpe / Return / MDD / Calmar / time-in-market
    - A vs B vs C vs D 完整對比表
    - Overlay 貢獻量化（delta Sharpe）
    - 結果寫入 reports/research/overlay_4way/

Usage:
    cd /Users/dylanting/Documents/spot_bot/quant-binance-spot
    source .venv/bin/activate
    PYTHONPATH=src python scripts/research_overlay_4way.py
"""
from __future__ import annotations

import json
import sys
import yaml
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from qtrade.config import load_config
from qtrade.backtest.run_backtest import run_symbol_backtest, BacktestResult

import logging

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("overlay_4way")
logger.setLevel(logging.INFO)


# ═══════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════

R3C_CONFIG = "config/prod_live_R3C_E3.yaml"
META_BLEND_CONFIG = "config/prod_candidate_meta_blend.yaml"

# 8 shared symbols (Meta-Blend universe, excludes LTC and XRP)
SHARED_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT",
]

KLINE_DIR = Path("data/binance")   # exchange-specific kline root
DATA_ROOT = Path("data")           # project data root (for FR, OI — functions prepend binance/)

# R3C vol_pause overlay config (from prod_live_R3C_E3.yaml)
VOL_PAUSE_OVERLAY = {
    "enabled": True,
    "mode": "oi_vol",
    "params": {
        "oi_extreme_z": 999.0,       # OI 部分實質關閉
        "oi_reversal_window": 6,
        "reduce_pct": 0.5,
        "oi_lookback": 24,
        "oi_z_window": 168,
        "vol_spike_z": 2.0,          # Vol 部分啟用
        "overlay_cooldown_bars": 12,
        "trend_lookback": 20,
        "atr_period": 14,
        "vol_z_window": 168,
    },
}

# Meta-Blend portfolio allocation weights
META_BLEND_WEIGHTS = {
    "BTCUSDT": 0.1500, "ETHUSDT": 0.1500,
    "SOLUSDT": 0.1167, "BNBUSDT": 0.1167,
    "DOGEUSDT": 0.1167, "ADAUSDT": 0.1167,
    "AVAXUSDT": 0.1166, "LINKUSDT": 0.1166,
}

# R3C allocation weights (only 8 shared symbols, renormalized)
R3C_RAW_WEIGHTS = {
    "BTCUSDT": 0.4450, "ETHUSDT": 0.3316,
    "SOLUSDT": 0.3137, "BNBUSDT": 0.4358,
    "DOGEUSDT": 0.3156, "ADAUSDT": 0.3150,
    "AVAXUSDT": 0.3359, "LINKUSDT": 0.3316,
}


# ═══════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════

def _get_r3c_strategy_for_symbol(symbol: str) -> tuple[str, dict]:
    """
    R3C ensemble routing: BTC→breakout_vol_atr, ETH→tsmom_multi_ema, others→tsmom_ema
    """
    with open(R3C_CONFIG, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    ens = raw.get("ensemble", {})
    strategies = ens.get("strategies", {})

    if symbol in strategies:
        sym_cfg = strategies[symbol]
        return sym_cfg["name"], sym_cfg.get("params", {})

    base_params = raw.get("strategy", {}).get("params", {})
    return "tsmom_ema", base_params


def _run_single_backtest(
    symbol: str,
    cfg_path: str,
    strategy_name: str | None = None,
    strategy_params: dict | None = None,
    overlay_override: dict | None = None,
    remove_overlay: bool = False,
) -> BacktestResult | None:
    """
    Run a single-symbol backtest with optional overlay override.

    Args:
        symbol: trading pair
        cfg_path: path to YAML config
        strategy_name: override strategy name (for R3C ensemble)
        strategy_params: override strategy params
        overlay_override: inject this overlay config (replaces config's overlay)
        remove_overlay: if True, remove overlay from config entirely
    """
    cfg_obj = load_config(cfg_path)
    bt_dict = cfg_obj.to_backtest_dict(symbol=symbol)

    if strategy_name:
        bt_dict["strategy_name"] = strategy_name
    if strategy_params:
        bt_dict["strategy_params"] = strategy_params

    # Overlay control
    if remove_overlay:
        bt_dict.pop("overlay", None)
    elif overlay_override is not None:
        bt_dict["overlay"] = overlay_override

    data_path = KLINE_DIR / "futures" / cfg_obj.market.interval / f"{symbol}.parquet"
    if not data_path.exists():
        logger.warning(f"Missing data: {data_path}")
        return None

    try:
        return run_symbol_backtest(
            symbol=symbol,
            data_path=data_path,
            cfg=bt_dict,
            market_type="futures",
            direction="both",
            data_dir=DATA_ROOT,
        )
    except Exception as e:
        logger.error(f"Backtest failed for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return None


def _extract_daily_returns(result: BacktestResult) -> pd.Series | None:
    """Extract daily return series from backtest result."""
    eq = result.equity()
    if eq is None or eq.empty:
        return None
    daily_eq = eq.resample("1D").last().dropna()
    daily_ret = daily_eq.pct_change().dropna()
    return daily_ret


def _extract_symbol_metrics(result: BacktestResult) -> dict:
    """Extract key metrics from a single-symbol BacktestResult."""
    sharpe = result.sharpe()
    total_ret = result.total_return_pct()
    mdd = result.max_drawdown_pct()

    # time-in-market: fraction of bars where |pos| > 0
    pos = result.pos
    tim = (pos.abs() > 0.01).mean() * 100  # as percentage

    return {
        "sharpe": float(sharpe),
        "total_ret_pct": float(total_ret),
        "mdd_pct": float(mdd),
        "time_in_market_pct": float(tim),
    }


def _portfolio_metrics(daily_returns: pd.Series) -> dict:
    """Compute portfolio-level metrics from daily return series."""
    if daily_returns.empty or daily_returns.std() == 0:
        return {"sharpe": 0.0, "total_ret_pct": 0.0, "mdd_pct": 0.0, "calmar": 0.0, "cagr_pct": 0.0}

    sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(365)
    cum = (1 + daily_returns).cumprod()
    total_ret = (cum.iloc[-1] - 1) * 100
    running_max = cum.cummax()
    dd = (cum - running_max) / running_max
    mdd = dd.min() * 100  # negative

    # CAGR
    n_years = len(daily_returns) / 365.0
    if n_years > 0 and cum.iloc[-1] > 0:
        cagr = (cum.iloc[-1] ** (1 / n_years) - 1) * 100
    else:
        cagr = 0.0

    calmar = cagr / abs(mdd) if mdd != 0 else 0.0

    return {
        "sharpe": float(sharpe),
        "total_ret_pct": float(total_ret),
        "mdd_pct": float(mdd),
        "cagr_pct": float(cagr),
        "calmar": float(calmar),
    }


def _build_portfolio_returns(
    daily_returns: dict[str, pd.Series],
    weights: dict[str, float],
) -> pd.Series:
    """Build weighted portfolio daily return series."""
    if not daily_returns:
        return pd.Series(dtype=float)

    df = pd.DataFrame(daily_returns)
    df = df.dropna(how="all")

    avail = [s for s in df.columns if s in weights]
    w_vals = np.array([weights[s] for s in avail])
    w_vals = w_vals / w_vals.sum()  # renormalize
    port_ret = (df[avail] * w_vals).sum(axis=1)
    return port_ret


# ═══════════════════════════════════════════════════════════
#  Run 4-Way Comparison
# ═══════════════════════════════════════════════════════════

def run_4way() -> dict:
    """
    Run all 4 combinations × 8 symbols.

    Returns:
        {
            "A_r3c_naked": {"per_symbol": {...}, "portfolio": {...}},
            "B_r3c_overlay": {"per_symbol": {...}, "portfolio": {...}},
            "C_meta_naked": {"per_symbol": {...}, "portfolio": {...}},
            "D_meta_overlay": {"per_symbol": {...}, "portfolio": {...}},
        }
    """
    combos = {
        "A_r3c_naked": {
            "label": "R3C (no overlay)",
            "config": R3C_CONFIG,
            "use_r3c_routing": True,
            "overlay_action": "remove",
            "weights": R3C_RAW_WEIGHTS,
        },
        "B_r3c_overlay": {
            "label": "R3C + vol_pause (production)",
            "config": R3C_CONFIG,
            "use_r3c_routing": True,
            "overlay_action": "keep",  # config already has overlay ON
            "weights": R3C_RAW_WEIGHTS,
        },
        "C_meta_naked": {
            "label": "Meta-Blend (no overlay)",
            "config": META_BLEND_CONFIG,
            "use_r3c_routing": False,
            "overlay_action": "remove",
            "weights": META_BLEND_WEIGHTS,
        },
        "D_meta_overlay": {
            "label": "Meta-Blend + vol_pause",
            "config": META_BLEND_CONFIG,
            "use_r3c_routing": False,
            "overlay_action": "inject",  # inject vol_pause
            "weights": META_BLEND_WEIGHTS,
        },
    }

    all_results = {}

    for combo_key, combo_def in combos.items():
        label = combo_def["label"]
        print(f"\n{'═' * 60}")
        print(f"  {combo_key}: {label}")
        print(f"{'═' * 60}")

        per_symbol_metrics = {}
        per_symbol_daily = {}

        for symbol in SHARED_SYMBOLS:
            # Determine strategy name and params
            if combo_def["use_r3c_routing"]:
                strat_name, strat_params = _get_r3c_strategy_for_symbol(symbol)
            else:
                strat_name = None
                strat_params = None

            # Determine overlay handling
            overlay_override = None
            remove_overlay = False
            if combo_def["overlay_action"] == "remove":
                remove_overlay = True
            elif combo_def["overlay_action"] == "inject":
                overlay_override = VOL_PAUSE_OVERLAY

            result = _run_single_backtest(
                symbol=symbol,
                cfg_path=combo_def["config"],
                strategy_name=strat_name,
                strategy_params=strat_params,
                overlay_override=overlay_override,
                remove_overlay=remove_overlay,
            )

            if result is None:
                logger.error(f"  {symbol}: FAILED")
                continue

            metrics = _extract_symbol_metrics(result)
            per_symbol_metrics[symbol] = metrics
            daily_ret = _extract_daily_returns(result)
            if daily_ret is not None:
                per_symbol_daily[symbol] = daily_ret

            strat_display = strat_name or "meta_blend"
            print(
                f"  {symbol:10s} [{strat_display:18s}] "
                f"SR={metrics['sharpe']:+6.3f}  "
                f"Ret={metrics['total_ret_pct']:+7.1f}%  "
                f"MDD={metrics['mdd_pct']:+6.1f}%  "
                f"TiM={metrics['time_in_market_pct']:.1f}%"
            )

        # Portfolio aggregate
        port_ret = _build_portfolio_returns(per_symbol_daily, combo_def["weights"])
        port_metrics = _portfolio_metrics(port_ret)

        print(f"\n  Portfolio: SR={port_metrics['sharpe']:+.3f}  "
              f"Ret={port_metrics['total_ret_pct']:+.1f}%  "
              f"MDD={port_metrics['mdd_pct']:+.1f}%  "
              f"CAGR={port_metrics['cagr_pct']:+.1f}%  "
              f"Calmar={port_metrics['calmar']:+.2f}")

        all_results[combo_key] = {
            "label": label,
            "per_symbol": per_symbol_metrics,
            "portfolio": port_metrics,
        }

    return all_results


# ═══════════════════════════════════════════════════════════
#  Report
# ═══════════════════════════════════════════════════════════

def print_report(results: dict) -> None:
    """Print comprehensive comparison report."""

    print("\n\n" + "═" * 80)
    print("  OVERLAY 4-WAY COMPARISON REPORT")
    print("═" * 80)

    # ── 1. Portfolio Summary ──
    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│  Portfolio-Level Summary                                            │")
    print("├─────────────────────────────┬────────┬─────────┬────────┬──────────┤")
    print("│ Combination                 │ Sharpe │ Return  │  MDD   │  Calmar  │")
    print("├─────────────────────────────┼────────┼─────────┼────────┼──────────┤")

    for key in ["A_r3c_naked", "B_r3c_overlay", "C_meta_naked", "D_meta_overlay"]:
        r = results[key]
        p = r["portfolio"]
        print(
            f"│ {r['label']:27s} │ {p['sharpe']:+6.3f} │ {p['total_ret_pct']:+6.1f}% │"
            f" {p['mdd_pct']:+5.1f}% │ {p['calmar']:+7.2f} │"
        )

    print("└─────────────────────────────┴────────┴─────────┴────────┴──────────┘")

    # ── 2. Overlay Impact Analysis ──
    print("\n┌─────────────────────────────────────────────────────────┐")
    print("│  Overlay Impact Analysis (delta = with - without)      │")
    print("├─────────────────────────────┬──────────────┬────────────┤")
    print("│ Strategy                    │ Sharpe Delta │ MDD Delta  │")
    print("├─────────────────────────────┼──────────────┼────────────┤")

    r3c_delta_sr = results["B_r3c_overlay"]["portfolio"]["sharpe"] - results["A_r3c_naked"]["portfolio"]["sharpe"]
    r3c_delta_mdd = results["B_r3c_overlay"]["portfolio"]["mdd_pct"] - results["A_r3c_naked"]["portfolio"]["mdd_pct"]
    meta_delta_sr = results["D_meta_overlay"]["portfolio"]["sharpe"] - results["C_meta_naked"]["portfolio"]["sharpe"]
    meta_delta_mdd = results["D_meta_overlay"]["portfolio"]["mdd_pct"] - results["C_meta_naked"]["portfolio"]["mdd_pct"]

    print(f"│ R3C (B - A)                 │ {r3c_delta_sr:+12.3f} │ {r3c_delta_mdd:+9.1f}% │")
    print(f"│ Meta-Blend (D - C)          │ {meta_delta_sr:+12.3f} │ {meta_delta_mdd:+9.1f}% │")
    print("└─────────────────────────────┴──────────────┴────────────┘")

    # ── 3. Production comparison: B vs D ──
    print("\n┌─────────────────────────────────────────────────────────┐")
    print("│  Production Comparison: R3C+overlay(B) vs MetaBlend(D) │")
    print("├───────────────────┬────────────────┬────────────────────┤")
    print("│ Metric            │  B (R3C prod)  │ D (Meta+overlay)   │")
    print("├───────────────────┼────────────────┼────────────────────┤")

    pb = results["B_r3c_overlay"]["portfolio"]
    pd_ = results["D_meta_overlay"]["portfolio"]

    for metric_name, key in [
        ("Sharpe", "sharpe"), ("Return %", "total_ret_pct"),
        ("MDD %", "mdd_pct"), ("CAGR %", "cagr_pct"), ("Calmar", "calmar"),
    ]:
        vb = pb[key]
        vd = pd_[key]
        winner = " <--" if vd > vb else ""
        if key == "mdd_pct":
            winner = " <--" if vd > vb else ""  # less negative = better
        print(f"│ {metric_name:17s} │ {vb:+14.3f} │ {vd:+14.3f}      │{winner}")

    print("└───────────────────┴────────────────┴────────────────────┘")

    # ── 4. Per-symbol detail ──
    print("\n┌───────────────────────────────────────────────────────────────────────────┐")
    print("│  Per-Symbol Sharpe Comparison                                            │")
    print("├──────────┬──────────┬──────────┬──────────┬──────────┬───────────────────┤")
    print("│ Symbol   │ A(R3C-N) │ B(R3C+O) │ C(MB-N)  │ D(MB+O)  │ Best             │")
    print("├──────────┼──────────┼──────────┼──────────┼──────────┼───────────────────┤")

    for sym in SHARED_SYMBOLS:
        vals = {}
        for key in ["A_r3c_naked", "B_r3c_overlay", "C_meta_naked", "D_meta_overlay"]:
            ps = results[key].get("per_symbol", {})
            vals[key] = ps.get(sym, {}).get("sharpe", float("nan"))

        best_key = max(vals, key=lambda k: vals[k] if not np.isnan(vals[k]) else -999)
        best_labels = {"A_r3c_naked": "A", "B_r3c_overlay": "B", "C_meta_naked": "C", "D_meta_overlay": "D"}
        print(
            f"│ {sym:8s} │ {vals['A_r3c_naked']:+8.3f} │ {vals['B_r3c_overlay']:+8.3f} │"
            f" {vals['C_meta_naked']:+8.3f} │ {vals['D_meta_overlay']:+8.3f} │"
            f" {best_labels[best_key]:17s} │"
        )

    print("└──────────┴──────────┴──────────┴──────────┴──────────┴───────────────────┘")

    # ── 5. Key conclusions ──
    print("\n" + "─" * 70)
    print("  KEY FINDINGS:")
    print("─" * 70)

    # R3C overlay impact
    if r3c_delta_sr > 0.05:
        print(f"  1. Vol_pause overlay HELPS R3C significantly (Sharpe +{r3c_delta_sr:.3f})")
    elif r3c_delta_sr > 0:
        print(f"  1. Vol_pause overlay helps R3C modestly (Sharpe +{r3c_delta_sr:.3f})")
    else:
        print(f"  1. Vol_pause overlay does NOT help R3C (Sharpe {r3c_delta_sr:+.3f})")

    # Meta-Blend overlay impact
    if meta_delta_sr > 0.05:
        print(f"  2. Vol_pause overlay ALSO HELPS Meta-Blend (Sharpe +{meta_delta_sr:.3f})")
        print(f"     → carry confirmation does NOT fully replace vol_pause")
        print(f"     → RECOMMENDATION: Enable overlay in prod_candidate_meta_blend.yaml")
    elif meta_delta_sr > 0:
        print(f"  2. Vol_pause overlay helps Meta-Blend marginally (Sharpe +{meta_delta_sr:.3f})")
        print(f"     → carry confirmation mostly replaces vol_pause but not entirely")
        print(f"     → RECOMMENDATION: Consider enabling overlay for extra protection")
    else:
        print(f"  2. Vol_pause overlay does NOT help Meta-Blend (Sharpe {meta_delta_sr:+.3f})")
        print(f"     → carry confirmation successfully replaces vol_pause")
        print(f"     → RECOMMENDATION: Keep overlay disabled (current setting is correct)")

    # Fair comparison
    b_sr = results["B_r3c_overlay"]["portfolio"]["sharpe"]
    d_sr = results["D_meta_overlay"]["portfolio"]["sharpe"]
    if d_sr > b_sr:
        print(f"\n  3. Meta-Blend+overlay (D) BEATS R3C+overlay (B): {d_sr:+.3f} vs {b_sr:+.3f}")
        print(f"     → Meta-Blend is a genuine upgrade over production R3C")
    else:
        print(f"\n  3. R3C+overlay (B) BEATS Meta-Blend+overlay (D): {b_sr:+.3f} vs {d_sr:+.3f}")
        print(f"     → Meta-Blend is NOT necessarily better in production conditions")

    print("─" * 70)


def save_results(results: dict, output_dir: Path) -> None:
    """Save results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Make JSON serializable
    serializable = {}
    for key, val in results.items():
        serializable[key] = {
            "label": val["label"],
            "per_symbol": val["per_symbol"],
            "portfolio": val["portfolio"],
        }

    out_path = output_dir / "overlay_4way_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to: {out_path}")


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  OVERLAY 4-WAY COMPARISON")
    print(f"  R3C Config:        {R3C_CONFIG}")
    print(f"  Meta-Blend Config: {META_BLEND_CONFIG}")
    print(f"  Symbols:           {SHARED_SYMBOLS}")
    print(f"  Overlay params:    vol_spike_z=2.0, oi_extreme_z=999 (OI off)")
    print("=" * 70)

    results = run_4way()

    if len(results) < 4:
        print("ERROR: Not all combinations completed!")
        sys.exit(1)

    print_report(results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("reports/research/overlay_4way") / timestamp
    save_results(results, output_dir)

    print(f"\nDone! ({timestamp})")


if __name__ == "__main__":
    main()
