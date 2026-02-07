"""
回测指标计算

提供：
- 策略 vs Buy & Hold 基准对比
- 完整的风险/收益指标
- 逐笔交易分析
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import vectorbt as vbt


def pretty_stats(stats: pd.Series) -> pd.Series:
    """原始精简输出（向后兼容）"""
    keys = [
        "Start", "End", "Total Return [%]", "Max Drawdown [%]",
        "Sharpe Ratio", "Win Rate [%]", "Total Trades"
    ]
    out = stats.reindex([k for k in keys if k in stats.index])
    return out


def benchmark_buy_and_hold(df: pd.DataFrame, initial_cash: float,
                           fee_bps: float = 0, slippage_bps: float = 0) -> vbt.Portfolio:
    """
    计算 Buy & Hold 基准
    
    在第一根 bar 全仓买入，持有到最后。
    """
    close = df["close"]
    open_ = df["open"]
    fee = fee_bps / 10_000.0
    slippage = slippage_bps / 10_000.0

    # 全程持仓 100%
    bh_pos = pd.Series(1.0, index=df.index)
    # 第一根 bar 买入
    bh_pos.iloc[0] = 0.0  # shift 效果：第一根信号，第二根执行

    pf_bh = vbt.Portfolio.from_orders(
        close=close,
        size=bh_pos,
        size_type="targetpercent",
        price=open_,
        fees=fee,
        slippage=slippage,
        init_cash=initial_cash,
        freq="1h",
        direction="longonly",
    )
    return pf_bh


def full_report(pf: vbt.Portfolio, pf_bh: vbt.Portfolio,
                strategy_name: str = "Strategy") -> pd.DataFrame:
    """
    生成完整回测报告：策略 vs Buy & Hold 对比

    Returns:
        DataFrame，两列：Strategy / Buy & Hold
    """
    s = pf.stats()
    b = pf_bh.stats()

    def _get(series: pd.Series, key: str, default=0):
        return series.get(key, default)

    rows = {
        "Start":              [_get(s, "Start"), _get(b, "Start")],
        "End":                [_get(s, "End"), _get(b, "End")],
        "Total Return [%]":   [_get(s, "Total Return [%]"), _get(b, "Total Return [%]")],
        "Annualized Return [%]": [
            _annualized_return(pf), _annualized_return(pf_bh)
        ],
        "Max Drawdown [%]":   [_get(s, "Max Drawdown [%]"), _get(b, "Max Drawdown [%]")],
        "Sharpe Ratio":       [_get(s, "Sharpe Ratio"), _get(b, "Sharpe Ratio")],
        "Sortino Ratio":      [_get(s, "Sortino Ratio"), _get(b, "Sortino Ratio")],
        "Calmar Ratio":       [_get(s, "Calmar Ratio"), _get(b, "Calmar Ratio")],
        "Win Rate [%]":       [_get(s, "Win Rate [%]"), _get(b, "Win Rate [%]")],
        "Profit Factor":      [_get(s, "Profit Factor"), _get(b, "Profit Factor")],
        "Total Trades":       [_get(s, "Total Trades"), _get(b, "Total Trades")],
        "Avg Winning Trade [%]": [_get(s, "Avg Winning Trade [%]"), _get(b, "Avg Winning Trade [%]")],
        "Avg Losing Trade [%]":  [_get(s, "Avg Losing Trade [%]"), _get(b, "Avg Losing Trade [%]")],
        "Max Drawdown Duration": [_get(s, "Max Drawdown Duration"), _get(b, "Max Drawdown Duration")],
        "Expectancy":         [_get(s, "Expectancy"), _get(b, "Expectancy")],
    }

    report = pd.DataFrame(rows, index=[strategy_name, "Buy & Hold"]).T
    
    # 添加 alpha（策略超额收益）
    strat_ret = _get(s, "Total Return [%]", 0)
    bh_ret = _get(b, "Total Return [%]", 0)
    alpha_row = pd.DataFrame(
        {"Alpha [%]": [strat_ret - bh_ret, 0.0]},
        index=[strategy_name, "Buy & Hold"]
    ).T
    alpha_row.columns = report.columns
    report = pd.concat([report, alpha_row])

    return report


def _annualized_return(pf: vbt.Portfolio) -> float:
    """计算年化收益率"""
    total_ret = pf.stats().get("Total Return [%]", 0) / 100.0
    equity = pf.value()
    if len(equity) < 2:
        return 0.0
    days = (equity.index[-1] - equity.index[0]).total_seconds() / 86400.0
    if days <= 0:
        return 0.0
    years = days / 365.25
    if total_ret <= -1:
        return -100.0
    ann_ret = ((1 + total_ret) ** (1 / years) - 1) * 100
    return round(ann_ret, 2)


def trade_analysis(pf: vbt.Portfolio) -> pd.DataFrame:
    """
    逐笔交易分析
    
    Returns:
        DataFrame: 每笔交易的详情
            - Entry Time, Exit Time
            - Entry Price, Exit Price
            - PnL, Return [%]
            - Duration
    """
    try:
        trades = pf.trades.records_readable
    except Exception:
        return pd.DataFrame()

    if len(trades) == 0:
        return pd.DataFrame()

    result = pd.DataFrame()
    result["Entry Time"] = trades["Entry Timestamp"]
    result["Exit Time"] = trades["Exit Timestamp"]
    result["Entry Price"] = trades["Avg Entry Price"]
    result["Exit Price"] = trades["Avg Exit Price"]
    result["PnL"] = trades["PnL"]
    result["Return [%]"] = trades["Return"].apply(lambda x: round(x * 100, 2))
    result["Duration"] = trades["Exit Timestamp"] - trades["Entry Timestamp"]
    # vectorbt 可能返回 int (0=Open,1=Closed) 或字符串
    def _parse_status(x):
        if isinstance(x, str):
            return x
        return "Closed" if x == 1 else "Open"
    result["Status"] = trades["Status"].apply(_parse_status)

    return result.reset_index(drop=True)


def trade_summary(pf: vbt.Portfolio) -> pd.Series:
    """
    交易摘要统计
    
    Returns:
        Series: 交易层面的汇总指标
    """
    trades_df = trade_analysis(pf)
    if trades_df.empty:
        return pd.Series(dtype=float)

    closed = trades_df[trades_df["Status"] == "Closed"]
    if closed.empty:
        return pd.Series(dtype=float)

    winners = closed[closed["PnL"] > 0]
    losers = closed[closed["PnL"] < 0]

    summary = {
        "Total Trades": len(closed),
        "Winning Trades": len(winners),
        "Losing Trades": len(losers),
        "Win Rate [%]": round(len(winners) / len(closed) * 100, 1) if len(closed) > 0 else 0,
        "Avg Trade PnL": round(closed["PnL"].mean(), 2),
        "Avg Trade Return [%]": round(closed["Return [%]"].mean(), 2),
        "Best Trade [%]": round(closed["Return [%]"].max(), 2) if len(closed) > 0 else 0,
        "Worst Trade [%]": round(closed["Return [%]"].min(), 2) if len(closed) > 0 else 0,
        "Avg Win [%]": round(winners["Return [%]"].mean(), 2) if len(winners) > 0 else 0,
        "Avg Loss [%]": round(losers["Return [%]"].mean(), 2) if len(losers) > 0 else 0,
        "Largest Win": round(winners["PnL"].max(), 2) if len(winners) > 0 else 0,
        "Largest Loss": round(losers["PnL"].min(), 2) if len(losers) > 0 else 0,
        "Avg Duration": str(closed["Duration"].mean()).split(".")[0] if len(closed) > 0 else "N/A",
        "Max Consecutive Wins": _max_consecutive(closed["PnL"] > 0),
        "Max Consecutive Losses": _max_consecutive(closed["PnL"] <= 0),
    }
    return pd.Series(summary)


def _max_consecutive(mask: pd.Series) -> int:
    """计算最大连续 True 的次数"""
    if mask.empty:
        return 0
    groups = (~mask).cumsum()
    counts = mask.groupby(groups).sum()
    return int(counts.max()) if len(counts) > 0 else 0
