"""
回測成本模型

提供：
1. Funding Rate 模型 — 計算永續合約持倉的資金費率成本
2. Volume-based 滑點模型 — 根據成交量估算真實滑點
3. 策略容量分析 — 估算策略可承載的最大資金量

使用方式：
    from qtrade.backtest.costs import (
        compute_funding_costs,
        compute_volume_slippage,
        capacity_analysis,
    )
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# 1. Funding Rate 成本模型
# ══════════════════════════════════════════════════════════════


@dataclass
class FundingCostResult:
    """Funding Rate 成本計算結果"""

    cumulative_cost: pd.Series  # 累計 funding 成本（正=支出）
    per_settlement_cost: pd.Series  # 每次結算的 funding 成本
    total_cost: float  # 總 funding 成本
    total_cost_pct: float  # 總 funding 成本佔初始資金的比例
    avg_rate_8h: float  # 平均 8h funding rate
    n_settlements: int  # 結算次數
    annualized_cost_pct: float  # 年化 funding 成本率


def compute_funding_costs(
    pos: pd.Series,
    equity: pd.Series,
    funding_rates: pd.Series,
    leverage: int = 1,
) -> FundingCostResult:
    """
    計算持倉期間的 Funding Rate 成本

    Funding 機制：
    - 每 8 小時結算一次（00:00, 08:00, 16:00 UTC）
    - 多頭(pos>0) 且 rate>0 → 付費；rate<0 → 收費
    - 空頭(pos<0) 且 rate>0 → 收費；rate<0 → 付費
    - cost = position_value × funding_rate
    - position_value = equity × pos × leverage

    Args:
        pos: 持倉信號 Series（[-1, 1]）
        equity: 資金曲線 Series（VBT 的 portfolio value）
        funding_rates: 對齊到 kline 的 funding rate Series（非結算時刻=0）
        leverage: 槓桿倍數

    Returns:
        FundingCostResult
    """
    # 逐 bar 計算 funding 成本
    # cost = equity × pos × leverage × funding_rate
    # 正值 = 支出，負值 = 收入
    position_value = equity * pos * leverage
    per_bar_cost = position_value * funding_rates

    # 只保留有結算的 bar（funding_rate != 0）
    settlement_mask = funding_rates != 0
    per_settlement_cost = per_bar_cost[settlement_mask]

    cumulative_cost = per_bar_cost.cumsum()
    total_cost = per_bar_cost.sum()

    # 統計
    initial_equity = equity.iloc[0] if len(equity) > 0 else 1.0
    total_cost_pct = total_cost / initial_equity if initial_equity > 0 else 0.0

    non_zero_rates = funding_rates[settlement_mask]
    avg_rate_8h = non_zero_rates.mean() if len(non_zero_rates) > 0 else 0.0
    n_settlements = int(settlement_mask.sum())

    # 年化成本率：根據持倉比例加權
    # 每年 365 * 3 = 1095 次結算
    hours = len(pos)
    years = hours / (365 * 24) if hours > 0 else 1.0
    annualized_cost_pct = (total_cost_pct / years) if years > 0 else 0.0

    return FundingCostResult(
        cumulative_cost=cumulative_cost,
        per_settlement_cost=per_settlement_cost,
        total_cost=total_cost,
        total_cost_pct=total_cost_pct,
        avg_rate_8h=avg_rate_8h,
        n_settlements=n_settlements,
        annualized_cost_pct=annualized_cost_pct,
    )


def adjust_equity_for_funding(
    equity: pd.Series,
    funding_result: FundingCostResult,
) -> pd.Series:
    """
    用 funding 成本調整資金曲線

    adjusted_equity = original_equity - cumulative_funding_cost
    """
    return equity - funding_result.cumulative_cost


def compute_adjusted_stats(
    adjusted_equity: pd.Series,
    initial_cash: float,
) -> dict:
    """
    從調整後的資金曲線重新計算核心統計指標

    Returns:
        dict with keys: total_return_pct, max_drawdown_pct, sharpe, sortino, calmar
    """
    returns = adjusted_equity.pct_change().fillna(0.0)

    # Total Return
    total_return = (adjusted_equity.iloc[-1] - initial_cash) / initial_cash
    total_return_pct = total_return * 100

    # Annualized Return
    n_hours = len(returns)
    years = n_hours / (365 * 24) if n_hours > 0 else 1.0
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0

    # Max Drawdown
    rolling_max = adjusted_equity.expanding().max()
    drawdown = (adjusted_equity - rolling_max) / rolling_max
    max_drawdown_pct = abs(drawdown.min()) * 100

    # Sharpe Ratio (ann., 1h freq, risk-free=0)
    hourly_std = returns.std()
    sharpe = (
        np.sqrt(365 * 24) * returns.mean() / hourly_std if hourly_std > 0 else 0.0
    )

    # Sortino Ratio
    downside = returns[returns < 0]
    downside_std = downside.std() if len(downside) > 0 else 0.001
    sortino = (
        np.sqrt(365 * 24) * returns.mean() / downside_std if downside_std > 0 else 0.0
    )

    # Calmar Ratio
    calmar = annual_return / (max_drawdown_pct / 100) if max_drawdown_pct > 0 else 0.0

    return {
        "Total Return [%]": round(total_return_pct, 2),
        "Annualized Return [%]": round(annual_return * 100, 2),
        "Max Drawdown [%]": round(max_drawdown_pct, 2),
        "Sharpe Ratio": round(sharpe, 4),
        "Sortino Ratio": round(sortino, 4),
        "Calmar Ratio": round(calmar, 4),
    }


# ══════════════════════════════════════════════════════════════
# 2. Volume-based 滑點模型
# ══════════════════════════════════════════════════════════════


@dataclass
class SlippageResult:
    """滑點模型計算結果"""

    slippage_array: pd.Series  # 每 bar 的滑點比例 (0~1)
    avg_slippage_bps: float  # 平均滑點 (bps)
    max_slippage_bps: float  # 最大滑點 (bps)
    avg_participation_rate: float  # 平均市場佔比
    high_impact_bars: int  # 高衝擊 bar 數量（滑點 > 10bps）


def compute_volume_slippage(
    pos: pd.Series,
    df: pd.DataFrame,
    capital: float,
    base_bps: float = 2.0,
    impact_coefficient: float = 0.1,
    impact_power: float = 0.5,
    adv_lookback: int = 20,
    participation_rate: float = 0.10,
    leverage: int = 1,
) -> SlippageResult:
    """
    基於成交量的滑點模型（Square-Root Market Impact Model）

    模型：
        slippage = base_spread + k × (trade_value / ADV)^power

    其中：
    - base_spread: 最小買賣價差（固定成本）
    - k: 衝擊係數（經驗值，與市場深度有關）
    - trade_value: 該 bar 的交易金額 = |Δpos| × capital × leverage
    - ADV: 過去 N bar 的平均成交額
    - power: 衝擊指數（0.5 = 平方根模型，學術標準）

    Args:
        pos: 持倉信號 Series
        df: K 線 DataFrame（需要 'volume' 和 'close' 欄位）
        capital: 回測初始資金
        base_bps: 最低滑點 (bps)
        impact_coefficient: 衝擊係數 k
        impact_power: 衝擊指數（預設 0.5 = 平方根）
        adv_lookback: 平均成交量回看期（bar 數）
        participation_rate: 最大市場佔比（用於 clip）
        leverage: 槓桿倍數

    Returns:
        SlippageResult
    """
    # 計算每 bar 的交易金額
    delta_pos = pos.diff().fillna(pos.iloc[0])
    trade_value = delta_pos.abs() * capital * leverage

    # 計算平均每日成交額（ADV）
    # volume 是以基礎貨幣計價，乘以 close 轉為 USDT
    bar_volume_usd = df["volume"] * df["close"]
    adv = bar_volume_usd.rolling(window=adv_lookback, min_periods=1).mean()

    # 避免除以 0
    adv = adv.clip(lower=1.0)

    # 計算市場佔比 (participation rate)
    actual_participation = trade_value / adv
    # Clip 到合理範圍
    actual_participation = actual_participation.clip(upper=participation_rate * 10)

    # Square-root market impact model
    # slippage_pct = base_spread + k × (trade_value / ADV)^power
    base_spread = base_bps / 10_000.0
    impact = impact_coefficient * (trade_value / adv).pow(impact_power)
    slippage_pct = base_spread + impact

    # 無交易的 bar 滑點為 0（VBT 不會對無交易的 bar 收滑點）
    no_trade = delta_pos.abs() < 1e-10
    slippage_pct = slippage_pct.where(~no_trade, 0.0)

    # Clip 到合理範圍（最大 500bps = 5%）
    slippage_pct = slippage_pct.clip(upper=0.05)

    # 統計
    traded_slippage = slippage_pct[~no_trade]
    avg_slippage_bps = traded_slippage.mean() * 10_000 if len(traded_slippage) > 0 else 0
    max_slippage_bps = traded_slippage.max() * 10_000 if len(traded_slippage) > 0 else 0

    traded_participation = actual_participation[~no_trade]
    avg_participation = (
        traded_participation.mean() if len(traded_participation) > 0 else 0
    )
    high_impact = int((traded_slippage > 0.001).sum())  # > 10bps

    return SlippageResult(
        slippage_array=slippage_pct,
        avg_slippage_bps=round(avg_slippage_bps, 2),
        max_slippage_bps=round(max_slippage_bps, 2),
        avg_participation_rate=round(avg_participation, 6),
        high_impact_bars=high_impact,
    )


# ══════════════════════════════════════════════════════════════
# 3. 策略容量分析
# ══════════════════════════════════════════════════════════════


@dataclass
class CapacityResult:
    """策略容量分析結果"""

    results: pd.DataFrame  # 各資金等級的績效
    max_capacity_usd: float  # Sharpe > 1 的最大資金量
    capacity_at_half_sharpe: float  # Sharpe 衰減到一半的資金量
    summary: str  # 人類可讀的摘要


def capacity_analysis(
    pos: pd.Series,
    df: pd.DataFrame,
    cfg: dict,
    capital_levels: list[float] | None = None,
    leverage: int = 1,
    slippage_params: dict | None = None,
) -> CapacityResult:
    """
    策略容量分析

    以不同資金量重新計算滑點，觀察績效衰減。
    找出策略可承載的最大資金量。

    Args:
        pos: 持倉信號 Series
        df: K 線 DataFrame
        cfg: 回測配置 dict
        capital_levels: 測試的資金等級（預設從 1K 到 10M）
        leverage: 槓桿倍數
        slippage_params: 滑點模型參數 dict

    Returns:
        CapacityResult
    """
    import vectorbt as vbt

    from .run_backtest import to_vbt_direction

    if capital_levels is None:
        capital_levels = [
            1_000, 5_000, 10_000, 25_000, 50_000,
            100_000, 250_000, 500_000, 1_000_000,
            2_500_000, 5_000_000, 10_000_000,
        ]

    if slippage_params is None:
        slippage_params = {}

    fee = cfg.get("fee_bps", 4) / 10_000
    direction = cfg.get("direction", "both")
    vbt_direction = to_vbt_direction(direction)

    results = []
    base_sharpe = None

    for capital in capital_levels:
        # 計算該資金量下的滑點
        slip_result = compute_volume_slippage(
            pos=pos,
            df=df,
            capital=capital,
            leverage=leverage,
            **slippage_params,
        )

        # 回測
        try:
            pf = vbt.Portfolio.from_orders(
                close=df["close"],
                size=pos,
                size_type="targetpercent",
                price=df["open"],
                fees=fee,
                slippage=slip_result.slippage_array,
                init_cash=capital,
                freq="1h",
                direction=vbt_direction,
            )

            stats = pf.stats()
            total_return = stats.get("Total Return [%]", 0)
            sharpe = stats.get("Sharpe Ratio", 0)
            max_dd = abs(stats.get("Max Drawdown [%]", 0))

            if base_sharpe is None and sharpe > 0:
                base_sharpe = sharpe

            results.append({
                "Capital ($)": capital,
                "Total Return [%]": round(total_return, 2),
                "Sharpe": round(sharpe, 4),
                "Max DD [%]": round(max_dd, 2),
                "Avg Slippage (bps)": slip_result.avg_slippage_bps,
                "Max Slippage (bps)": slip_result.max_slippage_bps,
                "Avg Participation": f"{slip_result.avg_participation_rate:.4%}",
                "High Impact Bars": slip_result.high_impact_bars,
            })
        except Exception as e:
            logger.warning(f"  容量分析失敗 (${capital:,.0f}): {e}")
            results.append({
                "Capital ($)": capital,
                "Total Return [%]": 0,
                "Sharpe": 0,
                "Max DD [%]": 0,
                "Avg Slippage (bps)": slip_result.avg_slippage_bps,
                "Max Slippage (bps)": slip_result.max_slippage_bps,
                "Avg Participation": f"{slip_result.avg_participation_rate:.4%}",
                "High Impact Bars": slip_result.high_impact_bars,
            })

    results_df = pd.DataFrame(results)

    # 找出最大容量（Sharpe > 1.0）
    viable = results_df[results_df["Sharpe"] > 1.0]
    max_capacity = viable["Capital ($)"].max() if len(viable) > 0 else 0

    # 找出 Sharpe 衰減到一半的資金量
    half_sharpe = (base_sharpe or 0) / 2
    half_mask = results_df["Sharpe"] < half_sharpe
    cap_at_half = (
        results_df.loc[half_mask, "Capital ($)"].min()
        if half_mask.any()
        else capital_levels[-1]
    )

    # 生成摘要
    lines = [
        "📊 策略容量分析",
        f"   Base Sharpe (最小資金): {base_sharpe:.2f}" if base_sharpe else "   Base Sharpe: N/A",
        f"   最大可行資金 (Sharpe>1): ${max_capacity:,.0f}" if max_capacity > 0 else "   最大可行資金: 無（Sharpe 始終 < 1）",
        f"   Sharpe 半衰資金: ${cap_at_half:,.0f}",
        "",
    ]
    summary = "\n".join(lines)

    return CapacityResult(
        results=results_df,
        max_capacity_usd=max_capacity,
        capacity_at_half_sharpe=cap_at_half,
        summary=summary,
    )
