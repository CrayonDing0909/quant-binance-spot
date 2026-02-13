"""
回測指標計算

提供：
- 策略 vs Buy & Hold 基準對比
- 完整的風險/收益指標
- 逐筆交易分析
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import vectorbt as vbt


def pretty_stats(stats: pd.Series) -> pd.Series:
    """原始精簡輸出（向後相容）"""
    keys = [
        "Start", "End", "Total Return [%]", "Max Drawdown [%]",
        "Sharpe Ratio", "Win Rate [%]", "Total Trades"
    ]
    out = stats.reindex([k for k in keys if k in stats.index])
    return out


def benchmark_buy_and_hold(df: pd.DataFrame, initial_cash: float,
                           fee_bps: float = 0, slippage_bps: float = 0) -> vbt.Portfolio:
    """
    計算 Buy & Hold 基準
    
    在第一根 bar 全倉買入，持有到最後。
    """
    close = df["close"]
    open_ = df["open"]
    fee = fee_bps / 10_000.0
    slippage = slippage_bps / 10_000.0

    # 全程持倉 100%
    bh_pos = pd.Series(1.0, index=df.index)
    # 第一根 bar 買入
    bh_pos.iloc[0] = 0.0  # shift 效果：第一根信號，第二根執行

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
    生成完整回測報告：策略 vs Buy & Hold 對比

    Returns:
        DataFrame，兩列：Strategy / Buy & Hold
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
    
    # 添加 alpha（策略超額收益）
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
    """計算年化收益率"""
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
    逐筆交易分析
    
    Returns:
        DataFrame: 每筆交易的詳情
            - Entry Time, Exit Time
            - Entry Price, Exit Price
            - PnL, Return [%]
            - Duration
    """
    try:
        # 使用 positions 而不是 trades，獲得完整的 round-trip 交易
        # trades 會把部分平倉拆成多筆，positions 則是完整的進出場
        positions = pf.positions.records_readable
    except Exception:
        return pd.DataFrame()

    if len(positions) == 0:
        return pd.DataFrame()

    result = pd.DataFrame()
    result["Entry Time"] = positions["Entry Timestamp"]
    result["Exit Time"] = positions["Exit Timestamp"]
    result["Entry Price"] = positions["Avg Entry Price"]
    result["Exit Price"] = positions["Avg Exit Price"]
    result["PnL"] = positions["PnL"]
    result["Return [%]"] = positions["Return"].apply(lambda x: round(x * 100, 2))
    result["Duration"] = positions["Exit Timestamp"] - positions["Entry Timestamp"]
    # vectorbt 可能返回 int (0=Open,1=Closed) 或字串
    def _parse_status(x):
        if isinstance(x, str):
            return x
        return "Closed" if x == 1 else "Open"
    result["Status"] = positions["Status"].apply(_parse_status)

    return result.reset_index(drop=True)


def trade_summary(pf: vbt.Portfolio) -> pd.Series:
    """
    交易摘要統計
    
    Returns:
        Series: 交易層面的彙總指標
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
    """計算最大連續 True 的次數"""
    if mask.empty:
        return 0
    groups = (~mask).cumsum()
    counts = mask.groupby(groups).sum()
    return int(counts.max()) if len(counts) > 0 else 0


# ══════════════════════════════════════════════════════════════
# Long / Short 分開統計
# ══════════════════════════════════════════════════════════════


def long_short_split_analysis(
    pf: vbt.Portfolio,
    pos: pd.Series,
) -> dict:
    """
    Long / Short 分開統計分析

    從 portfolio 的交易記錄中，根據持倉方向分類，
    分別計算 Long 和 Short 的績效指標。

    Args:
        pf:  vectorbt Portfolio 物件
        pos: 持倉信號 Series（[-1, 1]）

    Returns:
        dict with keys:
            "long":    Long 交易統計 dict
            "short":   Short 交易統計 dict
            "summary": 人類可讀的摘要字串
            "df":      DataFrame (Long vs Short 對比表)
    """
    trades_df = trade_analysis(pf)
    if trades_df.empty:
        return {
            "long": {},
            "short": {},
            "summary": "無交易記錄",
            "df": pd.DataFrame(),
        }

    closed = trades_df[trades_df["Status"] == "Closed"].copy()
    if closed.empty:
        return {
            "long": {},
            "short": {},
            "summary": "無已平倉交易",
            "df": pd.DataFrame(),
        }

    # 判斷交易方向：
    # 如果入場價 < 出場價 且 PnL > 0 → 做多盈利
    # 使用持倉信號來判斷更精確
    # 在入場時間點查看 pos 的值
    trade_directions = []
    for _, trade in closed.iterrows():
        entry_time = trade["Entry Time"]
        # 查找入場時間附近的持倉信號
        if entry_time in pos.index:
            p = pos.loc[entry_time]
        else:
            # 找最近的時間點
            idx = pos.index.get_indexer([entry_time], method="nearest")[0]
            p = pos.iloc[idx] if idx >= 0 else 0

        if p > 0:
            trade_directions.append("Long")
        elif p < 0:
            trade_directions.append("Short")
        else:
            # 用價格推斷：如果入場 < 出場且 PnL > 0，是做多
            if trade["PnL"] > 0:
                is_long = trade["Exit Price"] > trade["Entry Price"]
            else:
                is_long = trade["Exit Price"] < trade["Entry Price"]
            trade_directions.append("Long" if is_long else "Short")

    closed["Direction"] = trade_directions

    long_trades = closed[closed["Direction"] == "Long"]
    short_trades = closed[closed["Direction"] == "Short"]

    def _calc_side_stats(side_trades: pd.DataFrame, side_name: str) -> dict:
        """計算單邊統計"""
        n = len(side_trades)
        if n == 0:
            return {
                "Total Trades": 0,
                "Winning Trades": 0,
                "Losing Trades": 0,
                "Win Rate [%]": 0.0,
                "Total PnL": 0.0,
                "Avg PnL": 0.0,
                "Avg Return [%]": 0.0,
                "Best Trade [%]": 0.0,
                "Worst Trade [%]": 0.0,
                "Profit Factor": 0.0,
                "Avg Duration": "N/A",
            }

        winners = side_trades[side_trades["PnL"] > 0]
        losers = side_trades[side_trades["PnL"] < 0]

        gross_profit = winners["PnL"].sum() if len(winners) > 0 else 0
        gross_loss = abs(losers["PnL"].sum()) if len(losers) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        return {
            "Total Trades": n,
            "Winning Trades": len(winners),
            "Losing Trades": len(losers),
            "Win Rate [%]": round(len(winners) / n * 100, 1),
            "Total PnL": round(side_trades["PnL"].sum(), 2),
            "Avg PnL": round(side_trades["PnL"].mean(), 2),
            "Avg Return [%]": round(side_trades["Return [%]"].mean(), 2),
            "Best Trade [%]": round(side_trades["Return [%]"].max(), 2),
            "Worst Trade [%]": round(side_trades["Return [%]"].min(), 2),
            "Profit Factor": round(profit_factor, 2) if profit_factor != float("inf") else "∞",
            "Avg Duration": str(side_trades["Duration"].mean()).split(".")[0] if n > 0 else "N/A",
        }

    long_stats = _calc_side_stats(long_trades, "Long")
    short_stats = _calc_side_stats(short_trades, "Short")

    # 建立對比 DataFrame
    comparison_df = pd.DataFrame({
        "Long": long_stats,
        "Short": short_stats,
    }).T

    # 生成摘要文字
    lines = [
        "📊 Long / Short 分開統計",
        f"   Long  交易: {long_stats['Total Trades']} 筆, "
        f"勝率 {long_stats['Win Rate [%]']}%, "
        f"總 PnL ${long_stats['Total PnL']:,.2f}, "
        f"平均報酬 {long_stats['Avg Return [%]']}%",
        f"   Short 交易: {short_stats['Total Trades']} 筆, "
        f"勝率 {short_stats['Win Rate [%]']}%, "
        f"總 PnL ${short_stats['Total PnL']:,.2f}, "
        f"平均報酬 {short_stats['Avg Return [%]']}%",
    ]

    # 判斷哪邊更好
    long_pnl = long_stats["Total PnL"]
    short_pnl = short_stats["Total PnL"]
    if long_pnl > short_pnl and long_pnl > 0:
        lines.append(f"   → Long 側貢獻更多利潤 (+${long_pnl:,.2f})")
    elif short_pnl > long_pnl and short_pnl > 0:
        lines.append(f"   → Short 側貢獻更多利潤 (+${short_pnl:,.2f})")
    elif long_pnl <= 0 and short_pnl <= 0:
        lines.append("   ⚠️ 兩側都在虧損")

    summary = "\n".join(lines)

    return {
        "long": long_stats,
        "short": short_stats,
        "summary": summary,
        "df": comparison_df,
    }
