"""
回測視覺化

提供：
- 策略 vs Buy & Hold 對比資金曲線
- 價格 + 買賣信號
- 持倉比例
- 回撤圖
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless rendering
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import vectorbt as vbt


def plot_backtest_summary(pf: vbt.Portfolio, df: pd.DataFrame, pos: pd.Series,
                          symbol: str, save_path: Path | None = None,
                          pf_benchmark: vbt.Portfolio | None = None,
                          strategy_name: str = "Strategy") -> None:
    """
    繪製完整的回測摘要圖

    4 個子圖：
      1. 價格 + 買賣信號
      2. 持倉比例
      3. 資金曲線（策略 vs Buy & Hold）
      4. 回撤
    """
    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1, 3, 2]})
    fig.suptitle(f"{symbol}  —  {strategy_name}  Backtest Report",
                 fontsize=16, fontweight="bold", y=0.98)

    # ── 1. 價格 + 買賣信號 ──────────────────────────────
    ax = axes[0]
    ax.plot(df.index, df["close"], linewidth=1.2, alpha=0.85, color="#333", label="Close")

    pos_changes = pos.diff()
    buy_idx = df.index[pos_changes > 0]
    sell_idx = df.index[pos_changes < 0]
    if len(buy_idx) > 0:
        ax.scatter(buy_idx, df.loc[buy_idx, "close"],
                   color="#26a69a", marker="^", s=70, zorder=5, label="Buy")
    if len(sell_idx) > 0:
        ax.scatter(sell_idx, df.loc[sell_idx, "close"],
                   color="#ef5350", marker="v", s=70, zorder=5, label="Sell")
    ax.set_ylabel("Price (USDT)")
    ax.set_title("Price & Signals", fontsize=12)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.25)

    # ── 2. 持倉比例 ────────────────────────────────────
    ax = axes[1]
    # 支援 [-1, 1] 的多空倉位顯示
    pos_long = pos.clip(lower=0)   # 多頭部分
    pos_short = pos.clip(upper=0)  # 空頭部分
    ax.fill_between(pos.index, 0, pos_long.values, alpha=0.4, color="#26a69a", label="Long")
    ax.fill_between(pos.index, 0, pos_short.values, alpha=0.4, color="#ef5350", label="Short")
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    
    # 動態 Y 軸範圍
    pos_min, pos_max = pos.min(), pos.max()
    if pos_min < 0:  # 有空頭
        ax.set_ylabel("Position [-1, 1]")
        ax.set_ylim(min(-1.15, pos_min - 0.1), max(1.15, pos_max + 0.1))
        ax.legend(loc="upper left", fontsize=8)
    else:  # 純多頭
        ax.set_ylabel("Position [0, 1]")
        ax.set_ylim(-0.05, 1.15)
    ax.grid(True, alpha=0.25)

    # ── 3. 資金曲線：策略 vs Buy & Hold ────────────────
    ax = axes[2]
    equity = pf.value()
    ax.plot(equity.index, equity.values, linewidth=2, color="#1565c0",
            label=f"{strategy_name}")

    if pf_benchmark is not None:
        eq_bh = pf_benchmark.value()
        ax.plot(eq_bh.index, eq_bh.values, linewidth=1.8, color="#9e9e9e",
                linestyle="--", label="Buy & Hold")

    ax.axhline(y=equity.iloc[0], color="gray", linestyle=":", alpha=0.4)
    ax.set_ylabel("Portfolio Value")
    ax.set_title("Equity Curve", fontsize=12)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.25)

    # 在圖上標註最終收益
    strat_ret = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
    label_x = equity.index[-1]
    ax.annotate(f"{strat_ret:+.1f}%", xy=(label_x, equity.iloc[-1]),
                fontsize=10, fontweight="bold", color="#1565c0",
                xytext=(5, 5), textcoords="offset points")
    if pf_benchmark is not None:
        bh_ret = (eq_bh.iloc[-1] / eq_bh.iloc[0] - 1) * 100
        ax.annotate(f"{bh_ret:+.1f}%", xy=(label_x, eq_bh.iloc[-1]),
                    fontsize=10, fontweight="bold", color="#757575",
                    xytext=(5, -12), textcoords="offset points")

    # ── 4. 回撤 ────────────────────────────────────────
    ax = axes[3]
    dd = pf.drawdown() * 100  # 轉為百分比
    ax.fill_between(dd.index, dd.values, 0, alpha=0.35, color="#ef5350", label="Strategy DD")

    if pf_benchmark is not None:
        dd_bh = pf_benchmark.drawdown() * 100
        ax.plot(dd_bh.index, dd_bh.values, linewidth=1, color="#9e9e9e",
                linestyle="--", alpha=0.7, label="Buy & Hold DD")

    ax.set_ylabel("Drawdown [%]")
    ax.set_xlabel("Date")
    ax.set_title("Drawdown", fontsize=12)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.25)

    # ── 格式 ────────────────────────────────────────────
    for a in axes:
        a.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def plot_equity_curve(pf: vbt.Portfolio, symbol: str,
                      save_path: Path | None = None) -> None:
    """簡版資金曲線圖（向後相容）"""
    plot_backtest_summary(pf, pd.DataFrame({"close": pf.close}), 
                          pd.Series(0.0, index=pf.close.index),
                          symbol, save_path)
