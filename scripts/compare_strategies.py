#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════
  策略比較工具 — 多策略組合評估 + 邊際 Sharpe 分析
═══════════════════════════════════════════════════════════════

用途：
  評估新策略是否值得納入現有組合。提供：
  1. 各策略獨立績效（SR, MDD, Calmar, 交易數, 曝險率）
  2. 跨策略收益率相關性矩陣
  3. 邊際 Sharpe 測試（加入新策略後組合 SR 是否提升）
  4. 最佳權重配置（均值-方差最佳化 + 約束條件）
  5. 冗餘警告（新策略與現有策略相關性 > 0.5）
  6. 納入/跳過建議

用法：
  cd /Users/dylanting/Documents/spot_bot/quant-binance-spot
  source .venv/bin/activate

  # 比較新策略 vs 現有組合
  PYTHONPATH=src python scripts/compare_strategies.py \\
    --existing config/prod_live_R3C_E3.yaml \\
    --candidate config/research_oi_liq_bounce.yaml

  # 比較多個候選策略
  PYTHONPATH=src python scripts/compare_strategies.py \\
    --existing config/prod_live_R3C_E3.yaml \\
    --candidate config/research_oi_liq_bounce.yaml \\
    --candidate config/prod_candidate_meta_blend.yaml

  # 快速模式（關閉成本模型）
  PYTHONPATH=src python scripts/compare_strategies.py \\
    --existing config/prod_live_R3C_E3.yaml \\
    --candidate config/research_oi_liq_bounce.yaml \\
    --simple
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from qtrade.config import load_config, AppConfig
from qtrade.backtest.run_backtest import run_symbol_backtest, BacktestResult
from qtrade.data.storage import load_klines

import logging
import yaml

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("compare_strategies")
logger.setLevel(logging.INFO)


# ═══════════════════════════════════════════════════════════
#  策略回測 + 收益率提取
# ═══════════════════════════════════════════════════════════

def _get_ensemble_strategies(config_path: str) -> dict | None:
    """從 YAML 讀取 ensemble 路由配置"""
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    ens = raw.get("ensemble")
    if ens and ens.get("enabled", False):
        return ens.get("strategies", {})
    return None


def _run_strategy_backtests(
    config_path: str,
    simple_mode: bool = False,
    label: str = "",
) -> dict:
    """
    跑完一個 config 裡所有幣種的回測，回傳 per-symbol 結果。

    Returns:
        {
            "label": str,
            "config_path": str,
            "strategy_name": str,
            "symbols": [str],
            "results": {symbol: BacktestResult},
            "daily_returns": {symbol: pd.Series},
            "portfolio_daily_returns": pd.Series,
            "stats": {
                "sharpe": float,
                "total_return": float,
                "max_drawdown": float,
                "calmar": float,
                "trade_count": int,
                "exposure": float,
            },
        }
    """
    cfg = load_config(config_path)
    symbols = cfg.market.symbols
    market_type = cfg.market_type_str
    data_dir = cfg.data_dir

    # 檢查 ensemble 路由
    ensemble_strategies = _get_ensemble_strategies(config_path)

    strategy_name = cfg.strategy.name
    if label == "":
        label = strategy_name

    results: dict[str, BacktestResult] = {}
    daily_returns: dict[str, pd.Series] = {}

    for symbol in symbols:
        # Ensemble 路由
        if ensemble_strategies and symbol in ensemble_strategies:
            sym_strat = ensemble_strategies[symbol]
            strat_name = sym_strat["name"]
            bt_cfg = cfg.to_backtest_dict(symbol=symbol)
            bt_cfg["strategy_params"] = sym_strat.get("params", bt_cfg["strategy_params"])
        else:
            strat_name = strategy_name
            bt_cfg = cfg.to_backtest_dict(symbol=symbol)

        if simple_mode:
            bt_cfg["funding_rate"] = {"enabled": False}
            bt_cfg["slippage_model"] = {"enabled": False}

        data_path = (
            data_dir / "binance" / market_type
            / cfg.market.interval / f"{symbol}.parquet"
        )
        if not data_path.exists():
            logger.warning(f"  {symbol}: 數據不存在 ({data_path})")
            continue

        try:
            res = run_symbol_backtest(
                symbol, data_path, bt_cfg,
                strategy_name=strat_name,
                data_dir=data_dir,
            )
            results[symbol] = res

            # 提取日收益率
            eq = res.equity()
            if eq is not None and not eq.empty:
                daily_eq = eq.resample("1D").last().dropna()
                daily_returns[symbol] = daily_eq.pct_change().dropna()

        except Exception as e:
            logger.error(f"  {symbol} 回測失敗: {e}")

    if not results:
        return {
            "label": label,
            "config_path": config_path,
            "strategy_name": strategy_name,
            "symbols": [],
            "results": {},
            "daily_returns": {},
            "portfolio_daily_returns": pd.Series(dtype=float),
            "stats": {},
        }

    # ── 計算 portfolio 收益率 ──
    # 讀取配置中的權重，若無則等權
    active_symbols = list(results.keys())
    if cfg.portfolio.allocation:
        weights = {}
        for sym in active_symbols:
            weights[sym] = cfg.portfolio.get_weight(sym, len(active_symbols))
    else:
        weights = {sym: 1.0 / len(active_symbols) for sym in active_symbols}

    # 正規化
    total_w = sum(weights.values())
    weights = {k: v / total_w for k, v in weights.items()}

    # 對齊日收益率
    if daily_returns:
        dr_df = pd.DataFrame(daily_returns).dropna()
        port_ret = pd.Series(0.0, index=dr_df.index)
        for sym in active_symbols:
            if sym in dr_df.columns:
                port_ret += dr_df[sym] * weights.get(sym, 0)
    else:
        port_ret = pd.Series(dtype=float)

    # 計算組合統計
    stats = _compute_stats(port_ret, results, active_symbols)

    return {
        "label": label,
        "config_path": config_path,
        "strategy_name": strategy_name,
        "symbols": active_symbols,
        "results": results,
        "daily_returns": daily_returns,
        "portfolio_daily_returns": port_ret,
        "weights": weights,
        "stats": stats,
    }


def _compute_stats(
    port_ret: pd.Series,
    results: dict[str, BacktestResult],
    symbols: list[str],
) -> dict:
    """計算組合統計指標"""
    if port_ret.empty or len(port_ret) < 2:
        return {}

    years = len(port_ret) / 365.0
    cum_ret = (1 + port_ret).prod() - 1
    annual_ret = (1 + cum_ret) ** (1 / years) - 1 if years > 0 else 0

    sharpe = (
        np.sqrt(365) * port_ret.mean() / port_ret.std()
        if port_ret.std() > 0 else 0
    )

    # MDD
    cum_eq = (1 + port_ret).cumprod()
    rolling_max = cum_eq.expanding().max()
    dd = (cum_eq - rolling_max) / rolling_max
    max_dd = abs(dd.min())

    calmar = annual_ret / max_dd if max_dd > 0 else 0

    # 交易數
    total_trades = 0
    for sym in symbols:
        if sym in results:
            try:
                total_trades += results[sym].stats.get("Total Trades", 0)
            except Exception:
                pass

    # 曝險率（有持倉的比例）
    exposures = []
    for sym in symbols:
        if sym in results:
            pos = results[sym].pos
            if pos is not None:
                exposures.append((pos.abs() > 0.01).mean())
    avg_exposure = np.mean(exposures) if exposures else 0

    downside_ret = port_ret[port_ret < 0]
    downside_std = downside_ret.std() if len(downside_ret) > 0 else 0.001
    sortino = (
        np.sqrt(365) * port_ret.mean() / downside_std
        if downside_std > 0 else 0
    )

    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "total_return_pct": round(cum_ret * 100, 2),
        "annual_return_pct": round(annual_ret * 100, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "calmar": round(calmar, 3),
        "total_trades": int(total_trades),
        "avg_exposure": round(avg_exposure, 4),
        "years": round(years, 2),
    }


# ═══════════════════════════════════════════════════════════
#  分析函數
# ═══════════════════════════════════════════════════════════

def compute_strategy_correlation(
    strategy_data: list[dict],
) -> pd.DataFrame:
    """
    計算跨策略收益率相關性矩陣。
    使用 portfolio-level 日收益率。
    """
    rets = {}
    for sd in strategy_data:
        label = sd["label"]
        pr = sd["portfolio_daily_returns"]
        if pr is not None and not pr.empty:
            rets[label] = pr

    if len(rets) < 2:
        return pd.DataFrame()

    df = pd.DataFrame(rets).dropna()
    return df.corr()


def marginal_sharpe_test(
    existing_data: dict,
    candidate_data: dict,
    weight_for_candidate: float = 0.30,
) -> dict:
    """
    邊際 Sharpe 測試：把候選策略加入現有組合，看 SR 是否提升。

    使用簡單的固定權重混合：
        new_portfolio = (1 - w) * existing + w * candidate

    Returns:
        {
            "existing_sharpe": float,
            "combined_sharpe": float,
            "marginal_sharpe": float,  # combined - existing
            "improves": bool,
            "weight_used": float,
        }
    """
    ex_ret = existing_data["portfolio_daily_returns"]
    ca_ret = candidate_data["portfolio_daily_returns"]

    if ex_ret.empty or ca_ret.empty:
        return {"error": "收益率數據為空"}

    # 對齊時間
    common = ex_ret.index.intersection(ca_ret.index)
    if len(common) < 30:
        return {"error": f"共同時間範圍太短 ({len(common)} 天)"}

    ex_aligned = ex_ret.loc[common]
    ca_aligned = ca_ret.loc[common]

    # 現有組合 SR
    ex_sr = (
        np.sqrt(365) * ex_aligned.mean() / ex_aligned.std()
        if ex_aligned.std() > 0 else 0
    )

    # 混合後 SR
    w = weight_for_candidate
    combined = (1 - w) * ex_aligned + w * ca_aligned
    co_sr = (
        np.sqrt(365) * combined.mean() / combined.std()
        if combined.std() > 0 else 0
    )

    return {
        "existing_sharpe": round(ex_sr, 3),
        "combined_sharpe": round(co_sr, 3),
        "marginal_sharpe": round(co_sr - ex_sr, 3),
        "improves": co_sr > ex_sr,
        "weight_used": w,
        "common_days": len(common),
    }


def optimize_weights(
    strategy_data: list[dict],
    min_weight: float = 0.10,
    max_weight: float = 0.60,
    n_samples: int = 10000,
) -> dict:
    """
    蒙地卡羅權重最佳化：找出最大 Sharpe 的策略權重配置。

    約束：
        - 每個策略權重 in [min_weight, max_weight]
        - 權重總和 = 1.0

    Returns:
        {
            "optimal_weights": {label: float},
            "optimal_sharpe": float,
            "equal_weight_sharpe": float,
        }
    """
    labels = [sd["label"] for sd in strategy_data]
    rets_list = []
    for sd in strategy_data:
        pr = sd["portfolio_daily_returns"]
        if pr is None or pr.empty:
            return {"error": f"策略 {sd['label']} 無收益率數據"}
        rets_list.append(pr)

    # 對齊
    common_idx = rets_list[0].index
    for r in rets_list[1:]:
        common_idx = common_idx.intersection(r.index)

    if len(common_idx) < 30:
        return {"error": f"共同時間範圍太短 ({len(common_idx)} 天)"}

    rets_aligned = np.column_stack([r.loc[common_idx].values for r in rets_list])
    n_strats = len(labels)

    # 等權重基線
    eq_weights = np.ones(n_strats) / n_strats
    eq_port = rets_aligned @ eq_weights
    eq_sharpe = np.sqrt(365) * eq_port.mean() / eq_port.std() if eq_port.std() > 0 else 0

    # 蒙地卡羅搜索
    best_sharpe = -np.inf
    best_weights = eq_weights.copy()

    rng = np.random.default_rng(42)
    for _ in range(n_samples):
        # 生成滿足約束的隨機權重
        raw = rng.dirichlet(np.ones(n_strats))
        w = np.clip(raw, min_weight, max_weight)
        w = w / w.sum()

        port = rets_aligned @ w
        sr = np.sqrt(365) * port.mean() / port.std() if port.std() > 0 else 0

        if sr > best_sharpe:
            best_sharpe = sr
            best_weights = w.copy()

    return {
        "optimal_weights": {labels[i]: round(float(best_weights[i]), 4) for i in range(n_strats)},
        "optimal_sharpe": round(float(best_sharpe), 3),
        "equal_weight_sharpe": round(float(eq_sharpe), 3),
        "common_days": len(common_idx),
    }


def check_redundancy(
    strategy_data: list[dict],
    threshold: float = 0.50,
) -> list[dict]:
    """
    冗餘檢查：找出高相關性的策略對。

    Returns:
        [{"pair": (label_a, label_b), "correlation": float, "warning": str}]
    """
    corr = compute_strategy_correlation(strategy_data)
    if corr.empty:
        return []

    warnings = []
    labels = corr.columns.tolist()
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            c = corr.iloc[i, j]
            if abs(c) > threshold:
                warnings.append({
                    "pair": (labels[i], labels[j]),
                    "correlation": round(float(c), 3),
                    "warning": (
                        f"高相關性! {labels[i]} vs {labels[j]}: "
                        f"corr={c:.3f} > {threshold}"
                    ),
                })

    return warnings


def generate_recommendation(
    existing_data: dict,
    candidate_data: dict,
    corr_matrix: pd.DataFrame,
    marginal_result: dict,
) -> dict:
    """
    根據治理規則生成納入建議。

    治理門檻（from STRATEGY_PORTFOLIO_GOVERNANCE.md）：
      A2: 策略相關性 < 0.30
      A3: 邊際 SR > 0
      A4: 交易數 >= 30 per symbol
      A5: 無年份虧損 > -5%
    """
    c_label = candidate_data["label"]
    e_label = existing_data["label"]
    c_stats = candidate_data["stats"]

    checks = []

    # A2: 相關性
    if not corr_matrix.empty and e_label in corr_matrix.columns and c_label in corr_matrix.columns:
        corr_val = corr_matrix.loc[e_label, c_label]
        passed_a2 = abs(corr_val) < 0.30
        checks.append({
            "gate": "A2 (策略相關性 < 0.30)",
            "value": round(float(corr_val), 3),
            "passed": passed_a2,
        })
    else:
        checks.append({"gate": "A2 (策略相關性)", "value": "N/A", "passed": None})

    # A3: 邊際 Sharpe > 0
    if "error" not in marginal_result:
        passed_a3 = marginal_result.get("improves", False)
        checks.append({
            "gate": "A3 (邊際 SR > 0)",
            "value": marginal_result.get("marginal_sharpe", 0),
            "passed": passed_a3,
        })
    else:
        checks.append({"gate": "A3 (邊際 SR)", "value": "N/A", "passed": None})

    # A4: 交易數
    n_symbols = len(candidate_data["symbols"])
    total_trades = c_stats.get("total_trades", 0)
    trades_per_symbol = total_trades / n_symbols if n_symbols > 0 else 0
    passed_a4 = trades_per_symbol >= 30
    checks.append({
        "gate": "A4 (交易數 >= 30/幣種)",
        "value": round(trades_per_symbol, 1),
        "passed": passed_a4,
    })

    # 判定
    pass_count = sum(1 for c in checks if c["passed"] is True)
    total_count = sum(1 for c in checks if c["passed"] is not None)
    all_pass = pass_count == total_count and total_count > 0

    if all_pass:
        verdict = "ADD"
        reason = "所有治理門檻通過，建議納入組合"
    elif pass_count >= total_count - 1 and total_count > 0:
        verdict = "CONDITIONAL"
        failed = [c["gate"] for c in checks if c["passed"] is False]
        reason = f"大部分通過，但 {', '.join(failed)} 未通過，建議進一步評估"
    else:
        verdict = "SKIP"
        failed = [c["gate"] for c in checks if c["passed"] is False]
        reason = f"多項未通過：{', '.join(failed)}"

    return {
        "verdict": verdict,
        "reason": reason,
        "checks": checks,
    }


# ═══════════════════════════════════════════════════════════
#  報告輸出
# ═══════════════════════════════════════════════════════════

def print_report(
    all_strategy_data: list[dict],
    existing_idx: int,
    corr_matrix: pd.DataFrame,
    marginal_results: list[dict],
    optimal: dict,
    redundancy_warnings: list[dict],
    recommendations: list[dict],
):
    """印出完整分析報告"""
    print("\n")
    print("=" * 80)
    print("  策略組合比較報告")
    print(f"  生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # ── 1. 各策略績效 ──
    print("\n┌─────────────────────────────────────────────────────────┐")
    print("│  1. 各策略獨立績效                                       │")
    print("└─────────────────────────────────────────────────────────┘")
    print()

    header = f"{'策略':<25} {'SR':>8} {'Sortino':>8} {'Return%':>9} {'MDD%':>8} {'Calmar':>8} {'交易數':>7} {'曝險%':>7}"
    print(header)
    print("-" * 85)
    for sd in all_strategy_data:
        s = sd["stats"]
        if not s:
            print(f"{sd['label']:<25} {'(無數據)':>8}")
            continue
        role = " [基準]" if sd == all_strategy_data[existing_idx] else " [候選]"
        print(
            f"{sd['label'] + role:<25} "
            f"{s.get('sharpe', 0):>8.3f} "
            f"{s.get('sortino', 0):>8.3f} "
            f"{s.get('total_return_pct', 0):>8.1f}% "
            f"{s.get('max_drawdown_pct', 0):>7.1f}% "
            f"{s.get('calmar', 0):>8.3f} "
            f"{s.get('total_trades', 0):>7d} "
            f"{s.get('avg_exposure', 0)*100:>6.1f}%"
        )

    # ── 2. 相關性矩陣 ──
    print("\n┌─────────────────────────────────────────────────────────┐")
    print("│  2. 跨策略收益率相關性矩陣                                 │")
    print("└─────────────────────────────────────────────────────────┘")
    print()

    if not corr_matrix.empty:
        # 格式化輸出
        labels = corr_matrix.columns.tolist()
        max_label_len = max(len(l) for l in labels)
        header_str = " " * (max_label_len + 2) + "  ".join(f"{l:>10}" for l in labels)
        print(header_str)
        for i, label in enumerate(labels):
            row_str = f"{label:<{max_label_len + 2}}"
            for j in range(len(labels)):
                val = corr_matrix.iloc[i, j]
                if i == j:
                    row_str += f"{'1.000':>10}  "
                else:
                    marker = " *" if abs(val) > 0.50 else "  "
                    row_str += f"{val:>8.3f}{marker}"
            print(row_str)
        print("\n  (* = 相關性 > 0.50，有冗餘風險)")
    else:
        print("  (需要至少 2 個策略才能計算)")

    # ── 3. 邊際 Sharpe 測試 ──
    print("\n┌─────────────────────────────────────────────────────────┐")
    print("│  3. 邊際 Sharpe 測試                                     │")
    print("└─────────────────────────────────────────────────────────┘")
    print()

    for i, mr in enumerate(marginal_results):
        c_label = all_strategy_data[i + 1]["label"] if i + 1 < len(all_strategy_data) else "?"
        # Skip existing
        if "error" in mr:
            print(f"  {c_label}: {mr['error']}")
            continue
        marker = "✅" if mr["improves"] else "❌"
        print(
            f"  {c_label} (候選權重={mr['weight_used']*100:.0f}%): "
            f"現有 SR={mr['existing_sharpe']:.3f} → "
            f"混合 SR={mr['combined_sharpe']:.3f} "
            f"(Δ={mr['marginal_sharpe']:+.3f}) "
            f"{marker}"
        )

    # ── 4. 最佳權重配置 ──
    print("\n┌─────────────────────────────────────────────────────────┐")
    print("│  4. 最佳權重配置（蒙地卡羅最佳化）                          │")
    print("└─────────────────────────────────────────────────────────┘")
    print()

    if "error" in optimal:
        print(f"  {optimal['error']}")
    else:
        print(f"  等權重 Sharpe:  {optimal['equal_weight_sharpe']:.3f}")
        print(f"  最佳化 Sharpe:  {optimal['optimal_sharpe']:.3f}")
        print(f"  共同天數:       {optimal['common_days']}")
        print()
        print(f"  {'策略':<25} {'最佳權重':>10}")
        print(f"  {'-'*40}")
        for label, w in optimal["optimal_weights"].items():
            print(f"  {label:<25} {w*100:>9.1f}%")

    # ── 5. 冗餘警告 ──
    if redundancy_warnings:
        print("\n┌─────────────────────────────────────────────────────────┐")
        print("│  5. 冗餘警告                                             │")
        print("└─────────────────────────────────────────────────────────┘")
        print()
        for rw in redundancy_warnings:
            print(f"  ⚠️  {rw['warning']}")
    else:
        print("\n┌─────────────────────────────────────────────────────────┐")
        print("│  5. 冗餘警告：無                                         │")
        print("└─────────────────────────────────────────────────────────┘")

    # ── 6. 納入建議 ──
    print("\n┌─────────────────────────────────────────────────────────┐")
    print("│  6. 納入建議                                             │")
    print("└─────────────────────────────────────────────────────────┘")
    print()

    for rec in recommendations:
        label = rec["label"]
        r = rec["recommendation"]
        verdict_marker = {"ADD": "✅", "CONDITIONAL": "⚠️", "SKIP": "❌"}.get(r["verdict"], "?")
        print(f"  {verdict_marker} {label}: {r['verdict']} — {r['reason']}")
        for check in r["checks"]:
            c_marker = "✅" if check["passed"] else "❌" if check["passed"] is False else "⬜"
            print(f"    {c_marker} {check['gate']}: {check['value']}")

    print("\n" + "=" * 80)
    print("  報告結束")
    print("=" * 80)


def save_report(
    output_dir: Path,
    all_strategy_data: list[dict],
    corr_matrix: pd.DataFrame,
    marginal_results: list[dict],
    optimal: dict,
    redundancy_warnings: list[dict],
    recommendations: list[dict],
):
    """儲存報告到 JSON"""
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "timestamp": datetime.now().isoformat(),
        "strategies": [],
        "correlation_matrix": corr_matrix.to_dict() if not corr_matrix.empty else {},
        "marginal_sharpe_tests": marginal_results,
        "optimal_weights": optimal,
        "redundancy_warnings": [
            {"pair": list(rw["pair"]), "correlation": rw["correlation"]}
            for rw in redundancy_warnings
        ],
        "recommendations": [
            {"label": r["label"], **r["recommendation"]}
            for r in recommendations
        ],
    }

    for sd in all_strategy_data:
        report["strategies"].append({
            "label": sd["label"],
            "config_path": sd["config_path"],
            "strategy_name": sd["strategy_name"],
            "symbols": sd["symbols"],
            "stats": sd["stats"],
            "weights": sd.get("weights", {}),
        })

    output_path = output_dir / "strategy_comparison.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n📁 報告已儲存: {output_path}")
    return output_path


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="策略組合比較工具 — 邊際 Sharpe 分析 + 最佳權重配置",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--existing", type=str, required=True,
        help="現有組合的 config 路徑（基準策略）",
    )
    parser.add_argument(
        "--candidate", type=str, action="append", default=[],
        help="候選策略的 config 路徑（可多次使用）",
    )
    parser.add_argument(
        "--simple", action="store_true",
        help="快速模式：關閉 FR/Slippage 成本模型",
    )
    parser.add_argument(
        "--candidate-weight", type=float, default=0.30,
        help="邊際 Sharpe 測試中候選策略的假設權重（預設 0.30）",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="輸出目錄（預設 reports/strategy_comparison/<timestamp>）",
    )
    parser.add_argument(
        "--min-weight", type=float, default=0.10,
        help="最佳化約束：最低策略權重（預設 0.10）",
    )
    parser.add_argument(
        "--max-weight", type=float, default=0.60,
        help="最佳化約束：最高策略權重（預設 0.60）",
    )

    args = parser.parse_args()

    if not args.candidate:
        print("❌ 至少需要一個 --candidate 配置")
        sys.exit(1)

    # ── 1. 跑回測 ──
    print("=" * 60)
    print("  Step 1: 執行各策略回測")
    print("=" * 60)

    # 載入 existing 策略名稱
    existing_cfg = load_config(args.existing)
    existing_label = f"{existing_cfg.strategy.name}"

    print(f"\n📊 基準策略: {existing_label} ({args.existing})")
    existing_data = _run_strategy_backtests(
        args.existing, simple_mode=args.simple, label=existing_label,
    )

    candidate_data_list = []
    for cpath in args.candidate:
        c_cfg = load_config(cpath)
        c_label = f"{c_cfg.strategy.name}"
        # 避免重名
        existing_labels = [existing_label] + [cd["label"] for cd in candidate_data_list]
        if c_label in existing_labels:
            c_label = f"{c_label}_{Path(cpath).stem}"

        print(f"\n📊 候選策略: {c_label} ({cpath})")
        cd = _run_strategy_backtests(cpath, simple_mode=args.simple, label=c_label)
        candidate_data_list.append(cd)

    all_strategy_data = [existing_data] + candidate_data_list

    # ── 2. 分析 ──
    print("\n" + "=" * 60)
    print("  Step 2: 分析")
    print("=" * 60)

    # 相關性矩陣
    print("\n計算跨策略相關性...")
    corr_matrix = compute_strategy_correlation(all_strategy_data)

    # 邊際 Sharpe
    print("計算邊際 Sharpe...")
    marginal_results = []
    for cd in candidate_data_list:
        mr = marginal_sharpe_test(existing_data, cd, weight_for_candidate=args.candidate_weight)
        marginal_results.append(mr)

    # 最佳權重
    print("最佳化權重配置...")
    optimal = optimize_weights(
        all_strategy_data,
        min_weight=args.min_weight,
        max_weight=args.max_weight,
    )

    # 冗餘檢查
    print("冗餘檢查...")
    redundancy_warnings = check_redundancy(all_strategy_data)

    # 納入建議
    recommendations = []
    for i, cd in enumerate(candidate_data_list):
        mr = marginal_results[i]
        rec = generate_recommendation(existing_data, cd, corr_matrix, mr)
        recommendations.append({"label": cd["label"], "recommendation": rec})

    # ── 3. 報告 ──
    print_report(
        all_strategy_data,
        existing_idx=0,
        corr_matrix=corr_matrix,
        marginal_results=marginal_results,
        optimal=optimal,
        redundancy_warnings=redundancy_warnings,
        recommendations=recommendations,
    )

    # ── 4. 儲存 ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("reports/strategy_comparison") / timestamp

    save_report(
        output_dir,
        all_strategy_data,
        corr_matrix,
        marginal_results,
        optimal,
        redundancy_warnings,
        recommendations,
    )


if __name__ == "__main__":
    main()
