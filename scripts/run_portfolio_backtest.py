"""
組合回測腳本 - 同時回測多個幣種的組合表現

⚠️  重構重點（v2.0）：
    所有幣種的回測都透過 run_symbol_backtest() 執行，
    確保 Funding Rate / Volume Slippage 等成本模型一致。
    舊版直接建 VBT Portfolio 會繞過成本模型，產生「快樂表」。

支援：
- 等權重分配（預設）
- 自訂權重分配
- 從 config 讀取 portfolio.allocation 權重
- 組合績效統計（含成本調整）
- **Ensemble 模式**：每個 symbol 使用不同策略（v3.0 新增）
- **Vol-Parity 權重**：基於波動率反比分配權重（v3.0 新增）

使用範例：
    # 使用 config 中的 allocation 權重
    python scripts/run_portfolio_backtest.py -c config/futures_rsi_adx_atr.yaml

    # 等權重 BTC + ETH 組合
    python scripts/run_portfolio_backtest.py -c config/futures_rsi_adx_atr.yaml --symbols ETHUSDT SOLUSDT

    # 自訂權重 (ETH 60%, SOL 40%)
    python scripts/run_portfolio_backtest.py -c config/futures_rsi_adx_atr.yaml --symbols ETHUSDT SOLUSDT --weights 0.6 0.4

    # 快速模式（關閉成本模型，用於快速迭代）
    python scripts/run_portfolio_backtest.py -c config/futures_rsi_adx_atr.yaml --simple

    # Ensemble 模式（per-symbol 策略路由，從 config ensemble 段讀取）
    python scripts/run_portfolio_backtest.py -c config/futures_ensemble_nw_tsmom.yaml

    # Ensemble + 波動率平價權重
    python scripts/run_portfolio_backtest.py -c config/futures_ensemble_nw_tsmom.yaml --weight-mode vol_parity

    # 成本敏感度測試
    python scripts/run_portfolio_backtest.py -c config/futures_ensemble_nw_tsmom.yaml --cost-mult 0.5
    python scripts/run_portfolio_backtest.py -c config/futures_ensemble_nw_tsmom.yaml --cost-mult 1.5
"""
from __future__ import annotations
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import json
import yaml

from qtrade.config import load_config
from qtrade.backtest.run_backtest import (
    run_symbol_backtest,
    BacktestResult,
)
from qtrade.data.storage import load_klines


def apply_dd_throttle(
    portfolio_returns: pd.Series,
    dd_on: float = 0.10,
    dd_off: float = 0.07,
    scale: float = 0.70,
) -> pd.Series:
    """
    Portfolio-level drawdown throttle（風控覆蓋層）

    規則：
        - 當 running DD > dd_on → gross exposure *= scale
        - 當 running DD < dd_off → gross exposure back to 1.0

    不改變策略信號，只縮放 portfolio-level 收益率。

    Args:
        portfolio_returns:  原始 portfolio 收益率序列
        dd_on:              啟動 throttle 的 DD 門檻（預設 10%）
        dd_off:             關閉 throttle 的 DD 門檻（預設 7%）
        scale:              throttle 啟動時的曝險縮放倍數（預設 0.7）

    Returns:
        throttled portfolio 收益率序列
    """
    n = len(portfolio_returns)
    ret_arr = portfolio_returns.values.copy()
    throttled = np.zeros(n, dtype=float)

    equity = 1.0
    peak = 1.0
    throttle_active = False

    for i in range(n):
        # 決定本 bar 的 exposure（基於上一 bar 的 DD 狀態）
        current_scale = scale if throttle_active else 1.0
        throttled[i] = ret_arr[i] * current_scale

        # 更新 equity
        equity *= (1.0 + throttled[i])
        if equity > peak:
            peak = equity

        # 更新 DD 狀態（用於下一 bar）
        running_dd = (peak - equity) / peak if peak > 0 else 0.0
        if not throttle_active and running_dd > dd_on:
            throttle_active = True
        elif throttle_active and running_dd < dd_off:
            throttle_active = False

    return pd.Series(throttled, index=portfolio_returns.index)


def compute_vol_parity_weights(
    symbols: list[str],
    cfg,
    lookback: int = 720,
    min_weight: float = 0.20,
    max_weight: float = 0.50,
) -> dict[str, float]:
    """
    計算波動率反比（Risk Parity 近似）權重

    低波動 symbol → 高權重，高波動 → 低權重
    再用 min/max 限制避免極端偏斜

    Args:
        symbols: 交易對列表
        cfg: AppConfig
        lookback: 波動率計算回看期（bar 數）
        min_weight: 最低配置比例
        max_weight: 最高配置比例

    Returns:
        {symbol: weight} dict，已正規化
    """
    market_type = cfg.market_type_str
    vols = {}

    for sym in symbols:
        data_path = (
            cfg.data_dir / "binance" / market_type
            / cfg.market.interval / f"{sym}.parquet"
        )
        if not data_path.exists():
            print(f"⚠️  {sym}: 數據不存在，無法計算波動率")
            continue

        df = load_klines(data_path)
        returns = df["close"].pct_change()
        # 使用最近 lookback 根 bar 的波動率
        vol = returns.iloc[-lookback:].std() * np.sqrt(8760)  # 年化
        vols[sym] = vol
        print(f"  {sym}: 年化波動率 = {vol:.1%}")

    if not vols:
        return {s: 1.0 / len(symbols) for s in symbols}

    # 波動率反比
    inv_vols = {s: 1.0 / v for s, v in vols.items() if v > 0}
    total_inv = sum(inv_vols.values())
    raw_weights = {s: v / total_inv for s, v in inv_vols.items()}

    # 應用上下限
    clamped = {s: np.clip(w, min_weight, max_weight) for s, w in raw_weights.items()}

    # 迭代正規化（多輪 clip + renorm 確保收斂）
    for _ in range(5):
        total = sum(clamped.values())
        clamped = {s: w / total for s, w in clamped.items()}
        clamped = {s: np.clip(w, min_weight, max_weight) for s, w in clamped.items()}

    # 最終正規化
    total = sum(clamped.values())
    final = {s: w / total for s, w in clamped.items()}

    print(f"\n📊 Vol-Parity 權重:")
    for s, w in final.items():
        print(f"   {s}: {w*100:.1f}% (vol={vols.get(s, 0):.1%})")

    return final


def run_portfolio_backtest(
    symbols: list[str],
    weights: list[float],
    cfg,
    output_dir: Path,
    direction: str | None = None,
    simple_mode: bool = False,
    ensemble_strategies: dict | None = None,
    cost_mult: float = 1.0,
    dd_throttle_cfg: dict | None = None,
) -> dict:
    """
    執行組合回測（透過 run_symbol_backtest 確保成本一致性）

    Args:
        symbols: 交易對列表
        weights: 權重列表（與 symbols 對應）
        cfg: AppConfig 配置對象
        output_dir: 輸出目錄
        direction: 交易方向覆蓋（None 則自動從 config 判斷）
        simple_mode: True = 關閉 FR/Slippage 成本模型（快速迭代用）
        ensemble_strategies: per-symbol 策略配置（{symbol: {"name": ..., "params": ...}}）
        cost_mult: 成本乘數（1.0 = baseline, 0.5 = 低成本, 1.5 = 高成本）

    Returns:
        組合回測結果 dict
    """
    # 正規化權重
    weights = np.array(weights)
    weights = weights / weights.sum()

    market_type = cfg.market_type_str
    direction = direction or cfg.direction

    is_ensemble = ensemble_strategies is not None and len(ensemble_strategies) > 0

    print(f"\n📊 組合配置:")
    for sym, w in zip(symbols, weights):
        if is_ensemble and sym in ensemble_strategies:
            strat_name = ensemble_strategies[sym]["name"]
            print(f"   {sym}: {w*100:.1f}% → {strat_name}")
        else:
            print(f"   {sym}: {w*100:.1f}% → {cfg.strategy.name}")
    print(f"\n📈 交易方向: {direction}")
    print(f"🏷️  市場類型: {market_type}")
    if is_ensemble:
        print(f"🧩 模式: ENSEMBLE（per-symbol 策略路由）")
    if cost_mult != 1.0:
        print(f"💰 成本乘數: {cost_mult:.2f}x")
    if simple_mode:
        print(f"⚡ 模式: SIMPLE（成本模型關閉，僅供快速迭代）")
    else:
        print(f"🔒 模式: STRICT（含 Funding Rate + Volume Slippage）")
    print()

    # ── 使用 run_symbol_backtest 統一入口 ─────────────────
    # 這確保每個幣種都經過完整的成本模型處理
    per_symbol_results: dict[str, BacktestResult] = {}
    initial_cash = cfg.backtest.initial_cash

    for symbol in symbols:
        # ── Ensemble: 決定該 symbol 使用哪個策略 ──
        if is_ensemble and symbol in ensemble_strategies:
            sym_strat = ensemble_strategies[symbol]
            strategy_name = sym_strat["name"]
            # 用 ensemble 的 params 覆蓋預設
            bt_cfg = cfg.to_backtest_dict(symbol=symbol)
            bt_cfg["strategy_params"] = sym_strat.get("params", bt_cfg["strategy_params"])
        else:
            strategy_name = cfg.strategy.name
            bt_cfg = cfg.to_backtest_dict(symbol=symbol)

        # 如果命令列覆蓋 direction
        if direction:
            bt_cfg["direction"] = direction

        # Simple mode：關閉成本模型
        if simple_mode:
            bt_cfg["funding_rate"] = {"enabled": False}
            bt_cfg["slippage_model"] = {"enabled": False}

        # 成本乘數（用於敏感度分析）
        if cost_mult != 1.0:
            bt_cfg["fee_bps"] = bt_cfg["fee_bps"] * cost_mult
            bt_cfg["slippage_bps"] = bt_cfg["slippage_bps"] * cost_mult

        data_path = (
            cfg.data_dir / "binance" / market_type
            / cfg.market.interval / f"{symbol}.parquet"
        )

        if not data_path.exists():
            print(f"⚠️  {symbol}: 數據不存在 ({data_path})")
            continue

        res = run_symbol_backtest(
            symbol, data_path, bt_cfg,
            strategy_name=strategy_name,
            data_dir=cfg.data_dir,
        )
        per_symbol_results[symbol] = res

        # 顯示單幣結果
        pf = res.pf
        print(
            f"  {symbol} [{strategy_name}]: "
            f"Return {res.total_return_pct():+.1f}%, "
            f"Sharpe {res.sharpe():.2f}, "
            f"MDD {res.max_drawdown_pct():.1f}% "
            f"[{res.cost_summary()}]"
        )

    if not per_symbol_results:
        print("❌ 沒有成功的回測結果")
        return {}

    # ── 組合資金曲線 ─────────────────────────────────────
    # 使用 adjusted equity（含成本）如果有的話
    active_symbols = list(per_symbol_results.keys())
    active_weights = np.array([
        weights[symbols.index(s)] for s in active_symbols
    ])
    active_weights = active_weights / active_weights.sum()  # 重新正規化

    # 取得每個幣的資金曲線
    equity_curves = {}
    for sym, res in per_symbol_results.items():
        equity_curves[sym] = res.equity()

    # 對齊到共同時間範圍
    min_start = max(eq.index[0] for eq in equity_curves.values())
    max_end = min(eq.index[-1] for eq in equity_curves.values())
    print(f"\n📅 共同時間範圍: {min_start} → {max_end}")

    for sym in active_symbols:
        equity_curves[sym] = equity_curves[sym].loc[min_start:max_end]

    # 標準化淨值曲線（都從 1 開始）
    normalized = {}
    for sym in active_symbols:
        eq = equity_curves[sym]
        normalized[sym] = eq / eq.iloc[0]

    # 組合淨值 = 加權平均
    portfolio_normalized = sum(
        normalized[s] * w for s, w in zip(active_symbols, active_weights)
    )
    portfolio_equity = portfolio_normalized * initial_cash

    # Buy & Hold 組合
    bh_normalized = {}
    for sym in active_symbols:
        df = per_symbol_results[sym].df
        bh_eq = df["close"] / df["close"].iloc[0]
        bh_eq = bh_eq.loc[min_start:max_end]
        bh_normalized[sym] = bh_eq
    bh_portfolio_normalized = sum(
        bh_normalized[s] * w for s, w in zip(active_symbols, active_weights)
    )
    bh_equity = bh_portfolio_normalized * initial_cash

    # 計算組合收益率序列
    portfolio_returns = portfolio_equity.pct_change().fillna(0)
    bh_returns = bh_equity.pct_change().fillna(0)

    # ── DD Throttle（組合層風控覆蓋） ──
    if dd_throttle_cfg and dd_throttle_cfg.get("enabled", False):
        _dd_on = dd_throttle_cfg.get("dd_on", 0.10)
        _dd_off = dd_throttle_cfg.get("dd_off", 0.07)
        _dd_scale = dd_throttle_cfg.get("scale", 0.70)
        print(f"🛡️  DD Throttle: ON>{_dd_on*100:.0f}% → scale {_dd_scale:.0%}, OFF<{_dd_off*100:.0f}%")
        portfolio_returns = apply_dd_throttle(
            portfolio_returns,
            dd_on=_dd_on, dd_off=_dd_off, scale=_dd_scale,
        )
        # 用 throttled returns 重建 equity curve
        portfolio_equity = (1 + portfolio_returns).cumprod() * initial_cash

    # 計算統計指標
    stats = calculate_portfolio_stats(portfolio_returns, portfolio_equity, initial_cash)
    bh_stats = calculate_portfolio_stats(bh_returns, bh_equity, initial_cash)

    # ── 輸出結果 ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  組合回測結果: {' + '.join(active_symbols)}")
    mode_label = "SIMPLE（無成本）" if simple_mode else "STRICT（含 FR + Slippage）"
    print(f"  模式: {mode_label}")
    print("=" * 70)

    print(f"\n{'指標':<30} {'組合策略':>18} {'組合 Buy&Hold':>18}")
    print("-" * 70)
    print(f"{'Start':<30} {str(min_start)[:10]:>18} {str(min_start)[:10]:>18}")
    print(f"{'End':<30} {str(max_end)[:10]:>18} {str(max_end)[:10]:>18}")
    print(f"{'Total Return [%]':<30} {stats['total_return']*100:>18.2f} {bh_stats['total_return']*100:>18.2f}")
    print(f"{'Annualized Return [%]':<30} {stats['annual_return']*100:>18.2f} {bh_stats['annual_return']*100:>18.2f}")
    print(f"{'Max Drawdown [%]':<30} {stats['max_drawdown']*100:>18.2f} {bh_stats['max_drawdown']*100:>18.2f}")
    print(f"{'Sharpe Ratio':<30} {stats['sharpe']:>18.2f} {bh_stats['sharpe']:>18.2f}")
    print(f"{'Sortino Ratio':<30} {stats['sortino']:>18.2f} {bh_stats['sortino']:>18.2f}")
    print(f"{'Calmar Ratio':<30} {stats['calmar']:>18.2f} {bh_stats['calmar']:>18.2f}")

    # ── 成本模型影響摘要 ────────────────────────────────
    if not simple_mode:
        print(f"\n{'─'*70}")
        print(f"  💰 成本模型影響:")
        total_funding = 0.0
        for sym, res in per_symbol_results.items():
            if res.funding_cost:
                fc = res.funding_cost
                total_funding += fc.total_cost
                fr_sign = "支出" if fc.total_cost >= 0 else "收入"
                print(
                    f"    {sym}: Funding {fr_sign} "
                    f"${abs(fc.total_cost):,.0f} "
                    f"({fc.total_cost_pct*100:+.2f}%)"
                )
            if res.slippage_result:
                sr = res.slippage_result
                print(
                    f"    {sym}: Slippage avg={sr.avg_slippage_bps:.1f}bps, "
                    f"high_impact={sr.high_impact_bars} bars"
                )

    # ── 繪圖 ─────────────────────────────────────────────
    plot_portfolio_equity(
        portfolio_equity,
        bh_equity,
        active_symbols,
        active_weights,
        output_dir / "portfolio_equity_curve.png",
        mode_label=mode_label,
    )

    # ── 儲存結果 ─────────────────────────────────────────
    results = {
        "symbols": active_symbols,
        "weights": active_weights.tolist(),
        "start": str(min_start),
        "end": str(max_end),
        "mode": "simple" if simple_mode else "strict",
        "ensemble": is_ensemble,
        "cost_mult": cost_mult,
        "strategy_stats": stats,
        "buyhold_stats": bh_stats,
        "per_symbol": {
            sym: {
                "strategy": (ensemble_strategies.get(sym, {}).get("name", cfg.strategy.name)
                             if is_ensemble else cfg.strategy.name),
                "total_return_pct": res.total_return_pct(),
                "sharpe": res.sharpe(),
                "max_drawdown_pct": res.max_drawdown_pct(),
                "funding_rate_enabled": res.funding_rate_enabled,
                "slippage_model_enabled": res.slippage_model_enabled,
            }
            for sym, res in per_symbol_results.items()
        },
    }

    with open(output_dir / "portfolio_stats.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # 儲存資金曲線
    equity_df = pd.DataFrame({
        "strategy": portfolio_equity,
        "buyhold": bh_equity,
    })
    equity_df.to_csv(output_dir / "portfolio_equity.csv")

    print(f"\n✅ 組合資金曲線圖: {output_dir / 'portfolio_equity_curve.png'}")
    print(f"✅ 組合統計: {output_dir / 'portfolio_stats.json'}")

    return results


def calculate_portfolio_stats(
    returns: pd.Series, equity: pd.Series, initial_cash: float
) -> dict:
    """計算組合統計指標"""
    total_return = (equity.iloc[-1] - initial_cash) / initial_cash

    n_periods = len(returns)
    years = n_periods / (365 * 24)
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    rolling_max = equity.expanding().max()
    drawdown = (equity - rolling_max) / rolling_max
    max_drawdown = abs(drawdown.min())

    excess_returns = returns - 0
    sharpe = (
        np.sqrt(365 * 24) * excess_returns.mean() / excess_returns.std()
        if excess_returns.std() > 0 else 0
    )

    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else 0.001
    sortino = (
        np.sqrt(365 * 24) * returns.mean() / downside_std
        if downside_std > 0 else 0
    )

    calmar = annual_return / max_drawdown if max_drawdown > 0 else 0

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
    }


def plot_portfolio_equity(
    strategy_equity: pd.Series,
    bh_equity: pd.Series,
    symbols: list[str],
    weights: np.ndarray,
    save_path: Path,
    mode_label: str = "",
):
    """繪製組合資金曲線"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]}
    )

    weight_str = " + ".join([f"{s} {w*100:.0f}%" for s, w in zip(symbols, weights)])

    # 資金曲線
    ax1 = axes[0]
    ax1.plot(
        strategy_equity.index, strategy_equity.values,
        label="Portfolio Strategy", color="blue", linewidth=1.5,
    )
    ax1.plot(
        bh_equity.index, bh_equity.values,
        label="Portfolio Buy & Hold", color="gray", linestyle="--", alpha=0.7,
    )

    final_strat = (strategy_equity.iloc[-1] / strategy_equity.iloc[0] - 1) * 100
    final_bh = (bh_equity.iloc[-1] / bh_equity.iloc[0] - 1) * 100
    ax1.annotate(
        f"+{final_strat:.1f}%",
        xy=(strategy_equity.index[-1], strategy_equity.iloc[-1]),
        fontsize=10, color="blue", fontweight="bold",
    )
    ax1.annotate(
        f"+{final_bh:.1f}%",
        xy=(bh_equity.index[-1], bh_equity.iloc[-1]),
        fontsize=10, color="gray",
    )

    title = f"Portfolio Backtest: {weight_str}"
    if mode_label:
        title += f"  [{mode_label}]"
    ax1.set_title(title, fontsize=14, fontweight="bold")
    ax1.set_ylabel("Portfolio Value")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    # 回撤曲線
    ax2 = axes[1]
    rolling_max = strategy_equity.expanding().max()
    drawdown = (strategy_equity - rolling_max) / rolling_max * 100
    ax2.fill_between(
        drawdown.index, drawdown.values, 0,
        color="red", alpha=0.3, label="Strategy DD",
    )

    bh_rolling_max = bh_equity.expanding().max()
    bh_drawdown = (bh_equity - bh_rolling_max) / bh_rolling_max * 100
    ax2.plot(
        bh_drawdown.index, bh_drawdown.values,
        color="gray", linestyle="--", alpha=0.5, label="B&H DD",
    )

    ax2.set_ylabel("Drawdown [%]")
    ax2.set_xlabel("Date")
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def load_ensemble_config(config_path: str) -> dict | None:
    """
    從 YAML 配置檔讀取 ensemble 段落

    Returns:
        ensemble dict（含 strategies, weight_mode 等），若不存在則 None
    """
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    ens = raw.get("ensemble")
    if ens and ens.get("enabled", False):
        return ens
    return None


def main():
    parser = argparse.ArgumentParser(
        description="組合回測（v3.0 — 統一成本模型 + Ensemble 支援）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-c", "--config", type=str,
        default="config/rsi_adx_atr.yaml",
        help="配置檔案",
    )
    parser.add_argument(
        "--symbols", nargs="+", default=None,
        help="交易對列表（預設從 config 讀取）",
    )
    parser.add_argument(
        "--weights", nargs="+", type=float, default=None,
        help="權重列表（預設從 config portfolio.allocation 讀取）",
    )
    parser.add_argument(
        "--direction", "-d", type=str,
        choices=["both", "long_only", "short_only"],
        default=None,
        help="交易方向（預設從 config 讀取）",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="輸出目錄",
    )
    parser.add_argument(
        "--simple", action="store_true",
        help="⚡ 快速模式：關閉 FR/Slippage 成本模型（僅供快速迭代，結果不可信）",
    )
    parser.add_argument(
        "--weight-mode", type=str, default=None,
        choices=["fixed", "vol_parity"],
        help="權重模式（覆蓋 config ensemble.weight_mode）",
    )
    parser.add_argument(
        "--cost-mult", type=float, default=1.0,
        help="成本乘數（1.0=baseline, 0.5=低成本, 1.5=高成本）",
    )

    args = parser.parse_args()

    # 載入配置
    cfg = load_config(args.config)

    # ── 檢查 ensemble 配置 ──
    ensemble_raw = load_ensemble_config(args.config)
    ensemble_strategies = None
    if ensemble_raw:
        ensemble_strategies = ensemble_raw.get("strategies", {})
        print(f"🧩 偵測到 Ensemble 配置:")
        for sym, strat in ensemble_strategies.items():
            print(f"   {sym} → {strat['name']}")

    # ── 檢查 risk_overlay 配置（DD throttle 等） ──
    with open(args.config, "r", encoding="utf-8") as _f:
        _raw_cfg = yaml.safe_load(_f)
    dd_throttle_cfg = None
    risk_overlay = _raw_cfg.get("risk_overlay", {})
    if risk_overlay and risk_overlay.get("dd_throttle", {}).get("enabled", False):
        dd_throttle_cfg = risk_overlay["dd_throttle"]
        print(f"🛡️  偵測到 DD Throttle 配置: ON>{dd_throttle_cfg.get('dd_on', 0.10)*100:.0f}%, scale={dd_throttle_cfg.get('scale', 0.7):.0%}")

    # 確定交易對
    symbols = args.symbols or cfg.market.symbols
    if not symbols:
        print("❌ 未指定交易對，且 config 中也沒有設定")
        return

    # ── 設定權重 ──
    weight_mode = args.weight_mode
    if weight_mode is None and ensemble_raw:
        weight_mode = ensemble_raw.get("weight_mode", "fixed")

    if args.weights is not None:
        # 命令列明確指定 → 最高優先
        if len(args.weights) != len(symbols):
            raise ValueError(
                f"權重數量 ({len(args.weights)}) "
                f"與交易對數量 ({len(symbols)}) 不符"
            )
        weights = args.weights
        print(f"📋 使用命令列指定權重")
    elif weight_mode == "vol_parity":
        # Vol-Parity 權重
        vp_cfg = ensemble_raw.get("vol_parity", {}) if ensemble_raw else {}
        vp_weights = compute_vol_parity_weights(
            symbols, cfg,
            lookback=vp_cfg.get("lookback", 720),
            min_weight=vp_cfg.get("min_weight", 0.20),
            max_weight=vp_cfg.get("max_weight", 0.50),
        )
        weights = [vp_weights.get(s, 1.0 / len(symbols)) for s in symbols]
        print(f"📋 使用 vol_parity 權重")
    elif ensemble_raw and "fixed_weights" in ensemble_raw:
        # Ensemble 固定權重
        fw = ensemble_raw["fixed_weights"]
        weights = [fw.get(s, 1.0 / len(symbols)) for s in symbols]
        print(f"📋 使用 ensemble fixed_weights")
    elif cfg.portfolio.allocation:
        # 從 config 的 portfolio.allocation 讀取
        weights = []
        for sym in symbols:
            w = cfg.portfolio.get_weight(sym, len(symbols))
            weights.append(w)
        print(f"📋 使用 config portfolio.allocation 權重")
    else:
        weights = [1.0 / len(symbols)] * len(symbols)
        print(f"📋 使用等權重分配")

    # 設定輸出目錄
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Ensemble 模式：用 "ensemble_nw_tsmom" 作為策略名稱路徑
        if ensemble_strategies:
            report_base = Path(cfg.output.report_dir) / cfg.market_type_str / "ensemble_nw_tsmom" / "portfolio"
        else:
            report_base = cfg.get_report_dir("portfolio")
        output_dir = report_base / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📊 組合回測: {' + '.join(symbols)}")
    print(f"📁 輸出目錄: {output_dir}")

    # 執行回測
    run_portfolio_backtest(
        symbols, weights, cfg, output_dir,
        direction=args.direction,
        simple_mode=args.simple,
        ensemble_strategies=ensemble_strategies,
        cost_mult=args.cost_mult,
        dd_throttle_cfg=dd_throttle_cfg,
    )


if __name__ == "__main__":
    main()
