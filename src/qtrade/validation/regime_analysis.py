"""
Regime-Specific Performance Analysis

根據市場環境（牛市/熊市/盤整）分段計算策略績效，
識別策略在不同 regime 下的表現差異。

提供兩種模式：
1. 手動定義 regime（從 validation.yaml market_regimes）
2. 自動偵測 regime（基於 BTC drawdown + rally 規則）

References:
    - BTC drawdown > 20% = bear
    - BTC rally > 50% from recent trough = bull
    - else = sideways

Usage:
    from qtrade.validation.regime_analysis import (
        compute_regime_performance,
        auto_detect_regimes,
    )
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# Dataclasses
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RegimePerformance:
    """單一 regime 的績效摘要"""
    name: str
    start: str
    end: str
    sharpe: float
    total_return_pct: float
    max_drawdown_pct: float
    win_rate: float          # 正收益 bar 佔比
    n_bars: int
    annualized_return_pct: float


@dataclass
class RegimeAnalysisResult:
    """整體 regime 分析結果"""
    regimes: List[RegimePerformance]
    warnings: List[str]      # 紅旗（例如某 regime SR < 0）


# ══════════════════════════════════════════════════════════════════════════════
# Auto-Detect Regimes（基於 BTC drawdown / rally）
# ══════════════════════════════════════════════════════════════════════════════

def auto_detect_regimes(
    btc_prices: pd.Series,
    bear_dd_threshold: float = -0.20,
    bull_rally_threshold: float = 0.50,
) -> List[dict]:
    """
    自動偵測 BTC 市場環境。

    規則：
    - drawdown from ATH > 20% → bear
    - rally from recent trough > 50% → bull
    - 其他 → sideways

    Args:
        btc_prices: BTC 收盤價序列（DatetimeIndex）
        bear_dd_threshold: 熊市閾值（負值，例如 -0.20 = -20%）
        bull_rally_threshold: 牛市閾值（正值，例如 0.50 = +50%）

    Returns:
        List of dict with keys: name, start, end, description
    """
    if len(btc_prices) < 100:
        return []

    # 計算 drawdown from rolling ATH
    rolling_max = btc_prices.expanding().max()
    drawdown = (btc_prices - rolling_max) / rolling_max

    # 計算 rally from rolling trough (trailing 720h ~ 30 days)
    rolling_min = btc_prices.rolling(window=720, min_periods=1).min()
    rally = (btc_prices - rolling_min) / rolling_min

    # 分類每個 bar
    regime_labels = pd.Series("sideways", index=btc_prices.index)
    regime_labels[drawdown <= bear_dd_threshold] = "bear"
    regime_labels[rally >= bull_rally_threshold] = "bull"

    # 合併連續相同 regime 為區段
    regimes = []
    current_regime = regime_labels.iloc[0]
    current_start = regime_labels.index[0]

    for i in range(1, len(regime_labels)):
        if regime_labels.iloc[i] != current_regime:
            regimes.append({
                "name": f"{current_regime}_{current_start.strftime('%Y%m')}",
                "start": current_start.strftime("%Y-%m-%d"),
                "end": regime_labels.index[i - 1].strftime("%Y-%m-%d"),
                "description": f"Auto-detected {current_regime} market",
            })
            current_regime = regime_labels.iloc[i]
            current_start = regime_labels.index[i]

    # 最後一段
    regimes.append({
        "name": f"{current_regime}_{current_start.strftime('%Y%m')}",
        "start": current_start.strftime("%Y-%m-%d"),
        "end": regime_labels.index[-1].strftime("%Y-%m-%d"),
        "description": f"Auto-detected {current_regime} market",
    })

    # 過濾太短的 regime（< 168 bars = 1 week）
    filtered = [r for r in regimes if _regime_bar_count(r, btc_prices.index) >= 168]

    return filtered


def _regime_bar_count(regime: dict, index: pd.DatetimeIndex) -> int:
    """計算 regime 涵蓋的 bar 數"""
    start = pd.Timestamp(regime["start"])
    end = pd.Timestamp(regime["end"])
    if hasattr(index, 'tz') and index.tz is not None:
        if start.tz is None:
            start = start.tz_localize(index.tz)
        if end.tz is None:
            end = end.tz_localize(index.tz)
    mask = (index >= start) & (index <= end)
    return int(mask.sum())


# ══════════════════════════════════════════════════════════════════════════════
# Regime Performance Computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_regime_performance(
    equity: pd.Series,
    regimes: List[dict],
) -> RegimeAnalysisResult:
    """
    計算每個 regime 的績效。

    Args:
        equity: 策略資金曲線（DatetimeIndex，等間距如 1h）
        regimes: regime 定義列表，每個 dict 含 name, start, end, description

    Returns:
        RegimeAnalysisResult 含各 regime 績效及紅旗警告
    """
    results = []
    warnings = []

    for regime in regimes:
        name = regime["name"]
        start = pd.Timestamp(regime["start"])
        end = pd.Timestamp(regime["end"])

        # 若 equity index 有 timezone，確保比較時 tz-aware
        if hasattr(equity.index, 'tz') and equity.index.tz is not None:
            if start.tz is None:
                start = start.tz_localize(equity.index.tz)
            if end.tz is None:
                end = end.tz_localize(equity.index.tz)

        # 切片
        mask = (equity.index >= start) & (equity.index <= end)
        segment = equity[mask]

        if len(segment) < 50:
            continue

        returns = segment.pct_change().dropna()
        if len(returns) < 10:
            continue

        n_bars = len(returns)

        # Sharpe（年化，1h bar = 8760 bars/yr）
        hourly_std = returns.std()
        sharpe = (
            float(np.sqrt(8760) * returns.mean() / hourly_std)
            if hourly_std > 0 else 0.0
        )

        # Total Return
        total_return_pct = float(
            (segment.iloc[-1] / segment.iloc[0] - 1) * 100
        )

        # Max Drawdown
        rolling_max = segment.expanding().max()
        dd = (segment - rolling_max) / rolling_max
        max_dd_pct = float(abs(dd.min()) * 100)

        # Win Rate（正收益 bar 佔比）
        win_rate = float((returns > 0).mean())

        # Annualized Return
        hours = n_bars
        years = hours / (365 * 24) if hours > 0 else 1.0
        total_ret = segment.iloc[-1] / segment.iloc[0] - 1
        ann_ret_pct = float(
            ((1 + total_ret) ** (1 / years) - 1) * 100
        ) if years > 0 else 0.0

        rp = RegimePerformance(
            name=name,
            start=regime["start"],
            end=regime["end"],
            sharpe=round(sharpe, 2),
            total_return_pct=round(total_return_pct, 2),
            max_drawdown_pct=round(max_dd_pct, 2),
            win_rate=round(win_rate, 4),
            n_bars=n_bars,
            annualized_return_pct=round(ann_ret_pct, 2),
        )
        results.append(rp)

        # 紅旗：regime SR < 0
        if sharpe < 0:
            warnings.append(
                f"⚠️  Regime '{name}' ({regime['start']}→{regime['end']}): "
                f"SR={sharpe:.2f} < 0 — 策略在此環境下虧損"
            )

    return RegimeAnalysisResult(regimes=results, warnings=warnings)


def print_regime_report(result: RegimeAnalysisResult) -> None:
    """印出 regime 績效表"""
    print("\n" + "=" * 72)
    print("  📊 Regime-Specific Performance（市場環境分段績效）")
    print("     策略在不同市場環境下的表現，用於判斷策略是否 regime-dependent")
    print("=" * 72)

    if not result.regimes:
        print("  ⚠️  無有效 regime 數據")
        return

    # 表頭
    header = (
        f"  {'Regime':<30} {'SR':>6} {'Return%':>9} {'MDD%':>7} "
        f"{'WinRate':>8} {'AnnRet%':>9} {'Bars':>7}"
    )
    print(header)
    print("  " + "-" * 78)

    for rp in result.regimes:
        sr_icon = "✅" if rp.sharpe > 0 else "🚩"
        print(
            f"  {sr_icon} {rp.name:<28} {rp.sharpe:>6.2f} "
            f"{rp.total_return_pct:>+8.1f}% {rp.max_drawdown_pct:>6.1f}% "
            f"{rp.win_rate:>7.1%} {rp.annualized_return_pct:>+8.1f}% "
            f"{rp.n_bars:>7,}"
        )

    # 警告
    if result.warnings:
        print()
        for w in result.warnings:
            print(f"  {w}")
    else:
        print(f"\n  ✅ 所有 regime SR > 0")
