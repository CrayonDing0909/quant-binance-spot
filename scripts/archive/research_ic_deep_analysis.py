"""
IC 深度分析腳本 — Quant Researcher 專用

目的：診斷 6 symbols IC decay >50% 的根因
      判斷是 structural decay（需 parameter refresh / deweight）
      還是 cyclical dip（可持有等待回升）

分析維度：
  1. Multi-horizon IC: 6h, 24h, 48h, 168h
  2. 年度 + 季度 IC 時間分解
  3. 多窗口 Rolling IC (90d, 180d, 360d)
  4. 最近 6/12 個月 vs 全歷史對比
  5. IC 結構性斷裂檢測 (mean shift test)
  6. 信號活躍度 + 方向偏態分析

使用方式:
    PYTHONPATH=src python scripts/research_ic_deep_analysis.py \
        -c config/prod_live_R3C_E3.yaml \
        --output-dir reports/ic_deep_analysis
"""
from __future__ import annotations

import argparse
import json
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
from qtrade.validation.ic_monitor import RollingICMonitor


# ═══════════════════════════════════════════════════════════
# 工具函數
# ═══════════════════════════════════════════════════════════

def _load_ensemble_strategies(config_path: str) -> dict:
    """從 YAML 讀取 ensemble per-symbol 策略路由"""
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    ens = raw.get("ensemble", {})
    if ens.get("enabled", False):
        return ens.get("strategies", {})
    return {}


def _get_strategy_for_symbol(
    symbol: str,
    default_name: str,
    default_params: dict,
    ensemble_strategies: dict,
) -> tuple[str, dict]:
    """取得指定 symbol 的策略名稱和參數"""
    if ensemble_strategies and symbol in ensemble_strategies:
        sym_cfg = ensemble_strategies[symbol]
        return sym_cfg["name"], sym_cfg.get("params", {})
    return default_name, default_params


def compute_ic_spearman(signals: pd.Series, forward_returns: pd.Series) -> tuple[float, float]:
    """計算 Spearman IC + p-value，排除 NaN 和零信號"""
    valid = signals.notna() & forward_returns.notna() & (signals != 0)
    sig = signals[valid]
    fwd = forward_returns[valid]
    if len(sig) < 30:
        return np.nan, np.nan
    ic, pval = stats.spearmanr(sig, fwd)
    return ic, pval


def compute_quarterly_ic(signals: pd.Series, prices: pd.Series, forward_bars: int) -> dict:
    """按季度計算 IC"""
    fwd_ret = prices.pct_change(forward_bars).shift(-forward_bars)
    combined = pd.DataFrame({"signal": signals, "fwd_ret": fwd_ret}).dropna()
    combined = combined[combined["signal"] != 0]

    quarterly = {}
    for (year, quarter), group in combined.groupby(
        [combined.index.year, combined.index.quarter]
    ):
        key = f"{year}Q{quarter}"
        if len(group) >= 30:
            ic, pval = stats.spearmanr(group["signal"], group["fwd_ret"])
            quarterly[key] = {"ic": round(ic, 5), "pval": round(pval, 5), "n": len(group)}
        else:
            quarterly[key] = {"ic": np.nan, "pval": np.nan, "n": len(group)}
    return quarterly


def compute_multi_horizon_ic(signals: pd.Series, prices: pd.Series, horizons: list[int]) -> dict:
    """多前瞻期 IC"""
    results = {}
    for h in horizons:
        fwd_ret = prices.pct_change(h).shift(-h)
        ic, pval = compute_ic_spearman(signals, fwd_ret)
        results[f"{h}h"] = {"ic": round(ic, 5) if not np.isnan(ic) else None,
                            "pval": round(pval, 5) if not np.isnan(pval) else None}
    return results


def compute_rolling_ic_series(
    signals: pd.Series,
    prices: pd.Series,
    forward_bars: int,
    window_bars: int,
    step_bars: int = 24,
) -> pd.Series:
    """計算滾動 IC 時間序列"""
    fwd_ret = prices.pct_change(forward_bars).shift(-forward_bars)
    combined = pd.DataFrame({"signal": signals, "fwd_ret": fwd_ret}).dropna()
    combined = combined[combined["signal"] != 0]

    if len(combined) < window_bars:
        ic_val, _ = stats.spearmanr(combined["signal"], combined["fwd_ret"])
        return pd.Series(ic_val, index=combined.index[-1:])

    ic_values = {}
    for end_idx in range(window_bars, len(combined), step_bars):
        start_idx = end_idx - window_bars
        window_data = combined.iloc[start_idx:end_idx]
        if len(window_data) >= 30:
            ic, _ = stats.spearmanr(window_data["signal"], window_data["fwd_ret"])
            ic_values[combined.index[end_idx - 1]] = ic

    return pd.Series(ic_values, name="rolling_ic")


def detect_structural_break(rolling_ic: pd.Series, min_segment: int = 30) -> dict:
    """
    簡易結構性斷裂檢測 (Mean Shift Test)

    將 rolling IC 序列分為前半和後半，
    用 Welch t-test 檢測均值是否有顯著變化。
    也做最近 1/3 vs 前 2/3 的比較。
    """
    valid = rolling_ic.dropna()
    if len(valid) < min_segment * 2:
        return {"test": "insufficient_data", "n": len(valid)}

    # 前半 vs 後半
    half = len(valid) // 2
    first_half = valid.iloc[:half]
    second_half = valid.iloc[half:]
    t_stat_half, p_half = stats.ttest_ind(first_half, second_half, equal_var=False)

    # 前 2/3 vs 後 1/3
    split = len(valid) * 2 // 3
    early = valid.iloc[:split]
    recent = valid.iloc[split:]
    t_stat_third, p_third = stats.ttest_ind(early, recent, equal_var=False)

    return {
        "half_split": {
            "first_mean": round(first_half.mean(), 5),
            "second_mean": round(second_half.mean(), 5),
            "t_stat": round(t_stat_half, 3),
            "p_value": round(p_half, 5),
            "significant": p_half < 0.05,
        },
        "third_split": {
            "early_mean": round(early.mean(), 5),
            "recent_mean": round(recent.mean(), 5),
            "t_stat": round(t_stat_third, 3),
            "p_value": round(p_third, 5),
            "significant": p_third < 0.05,
        },
    }


def compute_signal_stats(signals: pd.Series) -> dict:
    """信號活躍度和方向偏態分析"""
    total = len(signals)
    active = (signals != 0).sum()
    long_pct = (signals > 0).sum() / total if total > 0 else 0
    short_pct = (signals < 0).sum() / total if total > 0 else 0
    flat_pct = (signals == 0).sum() / total if total > 0 else 0

    return {
        "total_bars": total,
        "active_bars": int(active),
        "active_pct": round(active / total, 4) if total > 0 else 0,
        "long_pct": round(long_pct, 4),
        "short_pct": round(short_pct, 4),
        "flat_pct": round(flat_pct, 4),
        "long_short_ratio": round(long_pct / short_pct, 2) if short_pct > 0 else float("inf"),
        "signal_mean": round(signals[signals != 0].mean(), 4) if active > 0 else 0,
        "signal_std": round(signals[signals != 0].std(), 4) if active > 0 else 0,
    }


def classify_decay(report: dict) -> dict:
    """
    診斷 IC decay 的類型

    Returns:
        {
            "diagnosis": "structural" | "cyclical" | "healthy" | "noise",
            "confidence": float (0-1),
            "recommendation": str,
            "detail": str,
        }
    """
    ic_24h = report.get("multi_horizon", {}).get("24h", {}).get("ic")
    quarterly = report.get("quarterly_ic", {})
    structural = report.get("structural_break", {})
    overall_ic = report.get("overall_ic")
    recent_ic = report.get("recent_6m_ic")
    historical_ic = report.get("historical_ic")
    recent_12m_ic = report.get("recent_12m_ic")

    # 季度 IC 序列
    q_ics = [v["ic"] for v in quarterly.values() if v.get("ic") is not None and not np.isnan(v["ic"])]

    # 最近 4 個季度
    recent_quarters = q_ics[-4:] if len(q_ics) >= 4 else q_ics
    early_quarters = q_ics[:-4] if len(q_ics) > 4 else []

    # ── 判斷邏輯 ──

    # Case 1: Overall IC 不顯著或接近零
    if overall_ic is not None and abs(overall_ic) < 0.02:
        return {
            "diagnosis": "noise",
            "confidence": 0.8,
            "recommendation": "DEWEIGHT — 信號本身就弱，IC ≈ 0",
            "detail": f"Overall IC = {overall_ic:.4f}, 信號品質不足",
        }

    # Case 2: 結構性斷裂顯著
    third_split = structural.get("third_split", {})
    if third_split.get("significant", False):
        recent_mean = third_split.get("recent_mean", 0)
        early_mean = third_split.get("early_mean", 0)

        if recent_mean < 0.02 and early_mean > 0.04:
            return {
                "diagnosis": "structural",
                "confidence": 0.85,
                "recommendation": "PARAMETER_REFRESH — IC 結構性下降，需要重新優化參數",
                "detail": (
                    f"Welch t-test p={third_split['p_value']:.4f}, "
                    f"early IC={early_mean:.4f} → recent IC={recent_mean:.4f}"
                ),
            }

    # Case 3: 最近季度回升 → 可能是 cyclical
    if len(recent_quarters) >= 2:
        # 如果最近 2 季中有回升（IC > 0.03），可能只是 dip
        recovering = sum(1 for q in recent_quarters[-2:] if q > 0.03)
        if recovering >= 1 and recent_ic is not None and recent_ic > 0.02:
            return {
                "diagnosis": "cyclical",
                "confidence": 0.7,
                "recommendation": "HOLD — 最近季度有回升跡象，建議觀察 1-2 季",
                "detail": f"最近 2 季 IC: {[round(q, 4) for q in recent_quarters[-2:]]}, recent_6m IC={recent_ic:.4f}",
            }

    # Case 4: 持續正 IC 但衰減
    if recent_ic is not None and historical_ic is not None:
        if recent_ic > 0.02 and historical_ic > 0.04:
            decay_pct = 1 - recent_ic / historical_ic
            if decay_pct > 0.5:
                return {
                    "diagnosis": "cyclical",
                    "confidence": 0.6,
                    "recommendation": "HOLD — IC 衰減超 50% 但仍為正，建議密切監控",
                    "detail": f"IC decay {decay_pct:.0%}, recent={recent_ic:.4f}, hist={historical_ic:.4f}",
                }

    # Case 5: IC 仍然健康
    if recent_ic is not None and recent_ic > 0.03:
        return {
            "diagnosis": "healthy",
            "confidence": 0.8,
            "recommendation": "MAINTAIN — IC 穩健",
            "detail": f"recent_6m IC={recent_ic:.4f}",
        }

    # Default: 不確定
    return {
        "diagnosis": "uncertain",
        "confidence": 0.4,
        "recommendation": "MONITOR — 證據不足，建議持續追蹤",
        "detail": f"overall IC={overall_ic}, recent_6m IC={recent_ic}, hist IC={historical_ic}",
    }


# ═══════════════════════════════════════════════════════════
# 主分析函數
# ═══════════════════════════════════════════════════════════

def analyze_symbol(
    symbol: str,
    df: pd.DataFrame,
    strategy_name: str,
    params: dict,
    market_type: str,
    direction: str,
    interval: str = "1h",
) -> dict:
    """對單一 symbol 執行完整 IC 深度分析"""

    ctx = StrategyContext(
        symbol=symbol,
        interval=interval,
        market_type=market_type,
        direction=direction,
    )

    strategy_func = get_strategy(strategy_name)
    signals = strategy_func(df, ctx, params)
    prices = df["close"]

    # ── 1. 基礎 IC ──
    forward_bars_24h = 24
    fwd_ret_24h = prices.pct_change(forward_bars_24h).shift(-forward_bars_24h)
    overall_ic, overall_pval = compute_ic_spearman(signals, fwd_ret_24h)

    # ── 2. Multi-horizon IC ──
    multi_horizon = compute_multi_horizon_ic(signals, prices, horizons=[6, 24, 48, 168])

    # ── 3. 年度 IC ──
    monitor = RollingICMonitor(window=180 * 24, forward_bars=24, interval=interval)
    full_report = monitor.compute(signals, prices)
    yearly_ic = full_report.yearly_ic

    # ── 4. 季度 IC ──
    quarterly_ic = compute_quarterly_ic(signals, prices, forward_bars_24h)

    # ── 5. 多窗口 Rolling IC ──
    bars_per_day = 24  # 1h interval
    rolling_90d = compute_rolling_ic_series(signals, prices, 24, 90 * bars_per_day)
    rolling_180d = compute_rolling_ic_series(signals, prices, 24, 180 * bars_per_day)
    rolling_360d = compute_rolling_ic_series(signals, prices, 24, 360 * bars_per_day)

    # ── 6. Recent vs Historical ──
    recent_6m_bars = 180 * bars_per_day
    recent_12m_bars = 365 * bars_per_day

    valid_90 = rolling_90d.dropna()
    if len(valid_90) > recent_6m_bars // 24:
        recent_6m_ic = valid_90.iloc[-(recent_6m_bars // 24):].mean()
    elif len(valid_90) > 0:
        recent_6m_ic = valid_90.iloc[-(len(valid_90) // 2):].mean()
    else:
        recent_6m_ic = None

    if len(valid_90) > recent_12m_bars // 24:
        recent_12m_ic = valid_90.iloc[-(recent_12m_bars // 24):].mean()
    else:
        recent_12m_ic = valid_90.mean() if len(valid_90) > 0 else None

    historical_ic = full_report.historical_ic

    # ── 7. 結構性斷裂檢測 ──
    structural_break = detect_structural_break(rolling_180d)

    # ── 8. 信號統計 ──
    signal_stats = compute_signal_stats(signals)

    # ── 9. Rolling IC 統計摘要 ──
    rolling_summary = {}
    for label, series in [("90d", rolling_90d), ("180d", rolling_180d), ("360d", rolling_360d)]:
        v = series.dropna()
        if len(v) > 0:
            rolling_summary[label] = {
                "mean": round(v.mean(), 5),
                "std": round(v.std(), 5),
                "ir": round(v.mean() / v.std(), 3) if v.std() > 0 else 0,
                "min": round(v.min(), 5),
                "max": round(v.max(), 5),
                "pct_positive": round((v > 0).mean(), 3),
                "n_observations": len(v),
            }
        else:
            rolling_summary[label] = {"mean": None, "n_observations": 0}

    result = {
        "symbol": symbol,
        "strategy": strategy_name,
        "data_range": f"{df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}",
        "data_bars": len(df),
        "overall_ic": round(overall_ic, 5) if not np.isnan(overall_ic) else None,
        "overall_pval": round(overall_pval, 5) if not np.isnan(overall_pval) else None,
        "multi_horizon": multi_horizon,
        "yearly_ic": yearly_ic,
        "quarterly_ic": quarterly_ic,
        "rolling_summary": rolling_summary,
        "recent_6m_ic": round(recent_6m_ic, 5) if recent_6m_ic is not None else None,
        "recent_12m_ic": round(recent_12m_ic, 5) if recent_12m_ic is not None else None,
        "historical_ic": round(historical_ic, 5) if historical_ic is not None else None,
        "ic_decay_pct": full_report.ic_decay_pct,
        "is_decaying": full_report.is_decaying,
        "structural_break": structural_break,
        "signal_stats": signal_stats,
    }

    # ── 10. 診斷 ──
    result["diagnosis"] = classify_decay(result)

    return result


# ═══════════════════════════════════════════════════════════
# 報告輸出
# ═══════════════════════════════════════════════════════════

def print_symbol_report(r: dict):
    """美觀輸出單一 symbol 分析結果"""
    sym = r["symbol"]
    diag = r["diagnosis"]

    print(f"\n{'═' * 72}")
    print(f"  {sym}  ({r['strategy']})  —  {r['data_range']}")
    print(f"{'═' * 72}")

    # Overall IC
    overall_ic = r["overall_ic"]
    pval = r["overall_pval"]
    sig_mark = "***" if pval and pval < 0.001 else "**" if pval and pval < 0.01 else "*" if pval and pval < 0.05 else "ns"
    print(f"  Overall IC: {overall_ic:+.5f}  (p={pval:.5f}) {sig_mark}")

    # Multi-horizon
    print(f"\n  ── Multi-Horizon IC ──")
    for h, vals in r["multi_horizon"].items():
        ic = vals["ic"]
        p = vals["pval"]
        if ic is not None:
            print(f"    {h:>5s}: {ic:+.5f}  (p={p:.5f})")
        else:
            print(f"    {h:>5s}: N/A")

    # Yearly IC
    print(f"\n  ── Yearly IC ──")
    for year, ic in sorted(r["yearly_ic"].items()):
        bar_len = max(1, int(abs(ic) * 200))
        bar = "█" * min(bar_len, 40)
        emoji = "🟢" if ic > 0.03 else "🟡" if ic > 0 else "🔴"
        print(f"    {year}: {ic:+.4f}  {emoji} {bar}")

    # Quarterly IC
    print(f"\n  ── Quarterly IC ──")
    for qkey, qval in sorted(r["quarterly_ic"].items()):
        ic = qval["ic"]
        n = qval["n"]
        if ic is not None and not np.isnan(ic):
            emoji = "🟢" if ic > 0.03 else "🟡" if ic > 0 else "🔴"
            print(f"    {qkey}: {ic:+.5f}  (n={n})  {emoji}")
        else:
            print(f"    {qkey}: N/A  (n={n})")

    # Rolling IC summary
    print(f"\n  ── Rolling IC Summary ──")
    print(f"    {'Window':<8} {'Mean':>8} {'Std':>8} {'IR':>6} {'%Pos':>6} {'Min':>8} {'Max':>8}")
    for label, stats_d in r["rolling_summary"].items():
        if stats_d.get("mean") is not None:
            print(
                f"    {label:<8} {stats_d['mean']:+8.5f} {stats_d['std']:8.5f} "
                f"{stats_d['ir']:6.3f} {stats_d['pct_positive']:6.1%} "
                f"{stats_d['min']:+8.5f} {stats_d['max']:+8.5f}"
            )
        else:
            print(f"    {label:<8} N/A")

    # Decay detection
    print(f"\n  ── Alpha Decay Detection ──")
    print(f"    Historical IC:  {r['historical_ic']}")
    print(f"    Recent 6M IC:   {r['recent_6m_ic']}")
    print(f"    Recent 12M IC:  {r['recent_12m_ic']}")
    decay_pct = r["ic_decay_pct"]
    decay_emoji = "🔴" if r["is_decaying"] else "🟢"
    print(f"    IC Decay:       {decay_pct:+.0%}  {decay_emoji}")

    # Structural break
    sb = r["structural_break"]
    if "half_split" in sb:
        hs = sb["half_split"]
        ts = sb["third_split"]
        print(f"\n  ── Structural Break Test ──")
        sb_mark = "⚠️  SIGNIFICANT" if hs["significant"] else "✅ Not significant"
        print(f"    Half-split: first={hs['first_mean']:+.5f} vs second={hs['second_mean']:+.5f}  "
              f"(t={hs['t_stat']:.2f}, p={hs['p_value']:.4f}) {sb_mark}")
        sb_mark2 = "⚠️  SIGNIFICANT" if ts["significant"] else "✅ Not significant"
        print(f"    2/3 split:  early={ts['early_mean']:+.5f} vs recent={ts['recent_mean']:+.5f}  "
              f"(t={ts['t_stat']:.2f}, p={ts['p_value']:.4f}) {sb_mark2}")
    else:
        print(f"\n  ── Structural Break Test ──")
        print(f"    {sb.get('test', 'N/A')}")

    # Signal stats
    ss = r["signal_stats"]
    print(f"\n  ── Signal Stats ──")
    print(f"    Active: {ss['active_pct']:.1%}  |  Long: {ss['long_pct']:.1%}  |  Short: {ss['short_pct']:.1%}  |  L/S ratio: {ss['long_short_ratio']:.2f}")
    print(f"    Signal mean: {ss['signal_mean']:+.4f}  |  Signal std: {ss['signal_std']:.4f}")

    # Diagnosis
    print(f"\n  ╔{'═' * 66}╗")
    diag_emoji = {"structural": "🔴", "cyclical": "🟡", "healthy": "🟢", "noise": "⚫", "uncertain": "🔵"}
    emoji = diag_emoji.get(diag["diagnosis"], "❓")
    print(f"  ║ {emoji} DIAGNOSIS: {diag['diagnosis'].upper():<12}  (confidence: {diag['confidence']:.0%})")
    print(f"  ║ {diag['recommendation']}")
    print(f"  ║ {diag['detail']}")
    print(f"  ╚{'═' * 66}╝")


def print_summary_table(results: list[dict]):
    """印出全 symbol 摘要比較表"""
    print(f"\n\n{'═' * 110}")
    print(f"  PORTFOLIO IC SUMMARY — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'═' * 110}")
    print(
        f"  {'Symbol':<10} {'Strategy':<20} {'IC_all':>8} {'IC_6h':>8} {'IC_24h':>8} "
        f"{'IC_48h':>8} {'IC_168h':>8} {'Rec_6M':>8} {'Hist':>8} {'Decay':>7} {'Diag':<12} {'Action'}"
    )
    print(f"  {'─' * 108}")

    decay_count = 0
    structural_count = 0

    for r in results:
        sym = r["symbol"]
        strat = r["strategy"][:18]
        ic_all = f"{r['overall_ic']:+.4f}" if r["overall_ic"] is not None else "N/A"
        mh = r["multi_horizon"]
        ic_6 = f"{mh['6h']['ic']:+.4f}" if mh.get("6h", {}).get("ic") is not None else "N/A"
        ic_24 = f"{mh['24h']['ic']:+.4f}" if mh.get("24h", {}).get("ic") is not None else "N/A"
        ic_48 = f"{mh['48h']['ic']:+.4f}" if mh.get("48h", {}).get("ic") is not None else "N/A"
        ic_168 = f"{mh['168h']['ic']:+.4f}" if mh.get("168h", {}).get("ic") is not None else "N/A"
        rec_6m = f"{r['recent_6m_ic']:+.4f}" if r["recent_6m_ic"] is not None else "N/A"
        hist = f"{r['historical_ic']:+.4f}" if r["historical_ic"] is not None else "N/A"
        decay = f"{r['ic_decay_pct']:+.0%}" if r["ic_decay_pct"] is not None else "N/A"
        diag = r["diagnosis"]["diagnosis"]
        action = r["diagnosis"]["recommendation"].split("—")[0].strip()

        if r["is_decaying"]:
            decay_count += 1
        if diag == "structural":
            structural_count += 1

        diag_emoji = {"structural": "🔴", "cyclical": "🟡", "healthy": "🟢", "noise": "⚫", "uncertain": "🔵"}
        emoji = diag_emoji.get(diag, "❓")

        print(
            f"  {sym:<10} {strat:<20} {ic_all:>8} {ic_6:>8} {ic_24:>8} "
            f"{ic_48:>8} {ic_168:>8} {rec_6m:>8} {hist:>8} {decay:>7} {emoji} {diag:<10} {action}"
        )

    print(f"  {'─' * 108}")
    print(f"\n  📊 Summary: {len(results)} symbols analyzed")
    print(f"     IC decay >50%: {decay_count} symbols")
    print(f"     Structural decay: {structural_count} symbols")
    print(f"     Cyclical dip: {sum(1 for r in results if r['diagnosis']['diagnosis'] == 'cyclical')} symbols")
    print(f"     Healthy: {sum(1 for r in results if r['diagnosis']['diagnosis'] == 'healthy')} symbols")


def print_final_verdict(results: list[dict]):
    """印出最終研判報告"""
    decay_symbols = [r for r in results if r["is_decaying"]]
    structural = [r for r in results if r["diagnosis"]["diagnosis"] == "structural"]
    cyclical = [r for r in results if r["diagnosis"]["diagnosis"] == "cyclical"]
    healthy = [r for r in results if r["diagnosis"]["diagnosis"] == "healthy"]
    noise = [r for r in results if r["diagnosis"]["diagnosis"] == "noise"]

    print(f"\n\n{'═' * 72}")
    print(f"  QUANT RESEARCHER — FINAL VERDICT")
    print(f"{'═' * 72}")

    print(f"\n  ── Falsification Matrix ──")
    gates = [
        ("IC decay >50% count", f"{len(decay_symbols)}/10 symbols", len(decay_symbols) <= 3),
        ("Structural break detected", f"{len(structural)} symbols", len(structural) <= 2),
        ("Portfolio avg IC > 0.02", f"{np.mean([r['overall_ic'] for r in results if r['overall_ic'] is not None]):.4f}", 
         np.mean([r['overall_ic'] for r in results if r['overall_ic'] is not None]) > 0.02),
        ("Recent 6M avg IC > 0", f"{np.mean([r['recent_6m_ic'] for r in results if r['recent_6m_ic'] is not None]):.4f}",
         np.mean([r['recent_6m_ic'] for r in results if r['recent_6m_ic'] is not None]) > 0),
        ("Noise symbols ≤ 2", f"{len(noise)} symbols", len(noise) <= 2),
    ]

    for gate_name, value, passed in gates:
        emoji = "✅" if passed else "❌"
        print(f"    {emoji} {gate_name}: {value}")

    all_passed = all(p for _, _, p in gates)

    # Per-symbol recommendations
    print(f"\n  ── Per-Symbol Recommendations ──")
    for r in results:
        diag = r["diagnosis"]
        diag_emoji = {"structural": "🔴", "cyclical": "🟡", "healthy": "🟢", "noise": "⚫", "uncertain": "🔵"}
        emoji = diag_emoji.get(diag["diagnosis"], "❓")
        print(f"    {emoji} {r['symbol']:<10}  {diag['recommendation']}")

    # Final recommendation
    print(f"\n  ╔{'═' * 66}╗")
    if len(structural) >= 3:
        print(f"  ║ 🔴 VERDICT: PARAMETER_REFRESH_NEEDED")
        print(f"  ║ {len(structural)} symbols 有結構性 IC 衰退。")
        print(f"  ║ 建議對這些 symbols 進行參數重新優化。")
    elif len(structural) >= 1:
        print(f"  ║ 🟡 VERDICT: SELECTIVE_REFRESH")
        print(f"  ║ {len(structural)} symbols 結構性衰退 + {len(cyclical)} 週期性。")
        print(f"  ║ 建議僅對結構性衰退的 symbols 做 parameter refresh。")
        print(f"  ║ 週期性衰退的 symbols 暫時 HOLD，下季再評估。")
    elif len(decay_symbols) >= 5 and len(structural) == 0:
        print(f"  ║ 🟡 VERDICT: HOLD_AND_MONITOR")
        print(f"  ║ {len(decay_symbols)} symbols IC decay >50%，但無結構性斷裂。")
        print(f"  ║ 判斷為市場 regime 導致的週期性衰退。")
        print(f"  ║ 不建議立即 deweight。建議 1-2 季後 re-evaluate。")
    else:
        print(f"  ║ 🟢 VERDICT: MAINTAIN_CURRENT")
        print(f"  ║ 整體 IC 品質仍然穩健。")
    print(f"  ╚{'═' * 66}╝")


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="IC Deep Analysis — Quant Researcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-c", "--config", type=str, required=True, help="生產配置 YAML 路徑")
    parser.add_argument("--symbol", type=str, default=None, help="只分析指定 symbol")
    parser.add_argument("--output-dir", type=str, default=None, help="JSON 報告輸出目錄")
    args = parser.parse_args()

    cfg = load_config(args.config)
    market_type = cfg.market_type_str
    direction = cfg.direction
    interval = cfg.market.interval
    symbols = [args.symbol] if args.symbol else cfg.market.symbols

    # Load ensemble strategies
    ensemble_strategies = _load_ensemble_strategies(args.config)
    default_strategy_name = cfg.strategy.name
    default_params = cfg.strategy.params

    print(f"📊 IC Deep Analysis — Quant Researcher")
    print(f"   Config: {args.config}")
    print(f"   Market: {market_type} | Direction: {direction} | Interval: {interval}")
    print(f"   Default Strategy: {default_strategy_name}")
    if ensemble_strategies:
        routing = ", ".join(f"{s}→{v['name']}" for s, v in ensemble_strategies.items())
        print(f"   Ensemble Routing: {routing}")
    print(f"   Symbols: {', '.join(symbols)}")
    print(f"   Analysis Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    results = []

    for sym in symbols:
        # Resolve per-symbol strategy
        sym_strategy, sym_params = _get_strategy_for_symbol(
            sym, default_strategy_name, default_params, ensemble_strategies
        )

        # Also merge symbol_overrides from cfg.strategy
        if sym_strategy == default_strategy_name:
            sym_params = cfg.strategy.get_params(sym)

        # Load data
        data_path = (
            cfg.data_dir / "binance" / market_type / interval / f"{sym}.parquet"
        )
        if not data_path.exists():
            print(f"\n⚠️  {sym}: 數據不存在 ({data_path})，跳過")
            continue

        df = load_klines(data_path)

        # Filter by config date range
        if cfg.market.start:
            df = df[df.index >= cfg.market.start]

        print(f"\n🔍 Analyzing {sym} ({sym_strategy})... ({len(df)} bars)")

        try:
            result = analyze_symbol(
                symbol=sym,
                df=df,
                strategy_name=sym_strategy,
                params=sym_params,
                market_type=market_type,
                direction=direction,
                interval=interval,
            )
            results.append(result)
            print_symbol_report(result)
        except Exception as e:
            print(f"❌ {sym} 分析失敗: {e}")
            import traceback
            traceback.print_exc()

    if not results:
        print("\n❌ 無有效結果，退出")
        return

    # Summary table
    print_summary_table(results)

    # Final verdict
    print_final_verdict(results)

    # Save JSON
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = out_dir / f"ic_deep_analysis_{timestamp}.json"

        def _json_default(obj):
            if isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            return str(obj)

        save_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": args.config,
            "symbols": {r["symbol"]: r for r in results},
        }
        # Remove rolling_ic series from JSON (too large)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2, default=_json_default, ensure_ascii=False)
        print(f"\n📁 JSON 報告已儲存: {report_path}")


if __name__ == "__main__":
    main()
