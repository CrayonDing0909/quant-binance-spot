#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════════
 Dual-Momentum Rotation Strategy — Institutional Crypto Portfolio Management
═══════════════════════════════════════════════════════════════════════════════════

Strategy Logic:
    1. ABSOLUTE MOMENTUM (Market Regime Filter):
       - BTC Close vs. 200-day SMA
       - Risk-On:  BTC > SMA(200) → Allocate to strongest asset
       - Risk-Off: BTC < SMA(200) → 100% Cash (0% interest)

    2. RELATIVE MOMENTUM (Asset Selection):
       - Universe: [BTC, ETH, SOL, BNB]
       - Metric:   90-day Return (simple momentum)
       - Rule:     Pick Top-1 asset with highest momentum
       - Rebalance: Weekly (every 7 calendar days)

    3. VOLATILITY TARGETING (Risk Management):
       - Annualized Vol = StdDev(30-day daily returns) × √365
       - Target Volatility = 40%
       - Position Size = min(Target_Vol / Current_Vol, 2.0)
       - Max Leverage: 2.0× (cap)
       - Excess cash earns 0%

    4. BENCHMARK:
       - BTC-USD Buy & Hold (fully invested from day 1)

Key Design Decisions:
    - Weekly rebalance avoids excessive turnover while capturing regime shifts
    - 200-day SMA filter historically avoids 60-80% of major drawdowns
    - Vol-targeting smooths the equity curve and enables consistent risk allocation
    - 2× leverage cap prevents over-leveraging in quiet markets

Usage:
    python scripts/research_dual_momentum.py
    python scripts/research_dual_momentum.py --start 2021-01-01 --end 2024-12-31
    python scripts/research_dual_momentum.py --target-vol 0.30 --max-leverage 1.5

Author: Quantitative Research Engineer
Date:   2026-02-19
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────
TICKERS = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "SOL": "SOL-USD",
    "BNB": "BNB-USD",
}
TRADING_DAYS_PER_YEAR = 365  # crypto = 365
RISK_FREE_RATE = 0.0  # cash earns 0%


# ═════════════════════════════════════════════════════════════════════════════
# Configuration
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class DualMomentumConfig:
    """Strategy configuration — all parameters in one place."""

    # Universe
    tickers: dict = field(default_factory=lambda: dict(TICKERS))

    # Absolute Momentum (Regime Filter)
    regime_sma_window: int = 200       # BTC 200-day SMA

    # Relative Momentum (Asset Selection)
    momentum_window: int = 90          # 90-day return lookback
    rebalance_days: int = 7            # rebalance every N calendar days
    top_n: int = 1                     # pick top N assets

    # Volatility Targeting (Risk Management)
    vol_window: int = 30               # 30-day rolling std dev
    target_vol: float = 0.40           # 40% annualized target
    max_leverage: float = 2.0          # cap at 2×
    min_leverage: float = 0.0          # floor (0 = can go to cash)

    # Costs
    rebalance_fee_bps: float = 10.0    # 0.10% round-trip fee per rebalance

    # Data
    start_date: str = "2020-01-01"
    end_date: Optional[str] = None     # None = present


# ═════════════════════════════════════════════════════════════════════════════
# Data Loader
# ═════════════════════════════════════════════════════════════════════════════
def load_data(config: DualMomentumConfig) -> pd.DataFrame:
    """
    Download daily close prices for all assets via yfinance.

    Returns:
        DataFrame with columns = asset names (BTC, ETH, SOL, BNB),
        index = DatetimeIndex (daily).
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance not installed. Run: pip install yfinance")

    end = config.end_date or datetime.now().strftime("%Y-%m-%d")
    logger.info(f"📥 Downloading data: {list(config.tickers.keys())} "
                f"({config.start_date} → {end})")

    # Download all tickers at once for efficiency
    ticker_list = list(config.tickers.values())
    raw = yf.download(
        ticker_list,
        start=config.start_date,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    # Extract Close prices
    if isinstance(raw.columns, pd.MultiIndex):
        closes = raw["Close"]
    else:
        # Single ticker edge case
        closes = raw[["Close"]].copy()
        closes.columns = [ticker_list[0]]

    # Rename columns to short names
    rename_map = {v: k for k, v in config.tickers.items()}
    closes = closes.rename(columns=rename_map)

    # Ensure all expected columns exist
    for name in config.tickers.keys():
        if name not in closes.columns:
            logger.warning(f"⚠️  {name} data not available — will be excluded")

    # Forward fill small gaps, then drop leading NaNs
    closes = closes.ffill().dropna(how="all")

    logger.info(f"✅ Data loaded: {len(closes)} days, "
                f"assets: {list(closes.columns)}")
    logger.info(f"   Date range: {closes.index[0].date()} → {closes.index[-1].date()}")

    for col in closes.columns:
        valid = closes[col].notna().sum()
        first_valid = closes[col].first_valid_index()
        logger.info(f"   {col}: {valid} valid days "
                    f"(from {first_valid.date() if first_valid else 'N/A'})")

    return closes


# ═════════════════════════════════════════════════════════════════════════════
# Core Strategy Engine
# ═════════════════════════════════════════════════════════════════════════════
class DualMomentumStrategy:
    """
    Dual-Momentum Rotation Strategy with Volatility Targeting.

    Combines:
        1. Absolute Momentum: BTC vs. 200-day SMA (regime filter)
        2. Relative Momentum: 90-day returns → Top-1 selection
        3. Volatility Targeting: position sizing based on inverse vol
    """

    def __init__(self, config: DualMomentumConfig):
        self.config = config

    def run(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Execute the full backtest.

        Args:
            prices: DataFrame of daily close prices (columns=assets)

        Returns:
            DataFrame with columns:
                strategy_return, strategy_equity, benchmark_equity,
                regime, selected_asset, position_size, ...
        """
        cfg = self.config
        assets = [c for c in prices.columns if c in cfg.tickers]

        logger.info("=" * 70)
        logger.info(" DUAL-MOMENTUM ROTATION STRATEGY")
        logger.info("=" * 70)
        logger.info(f"  Universe:        {assets}")
        logger.info(f"  Regime Filter:   BTC vs SMA({cfg.regime_sma_window})")
        logger.info(f"  Momentum:        {cfg.momentum_window}-day return, Top-{cfg.top_n}")
        logger.info(f"  Rebalance:       Every {cfg.rebalance_days} days")
        logger.info(f"  Vol Target:      {cfg.target_vol:.0%}")
        logger.info(f"  Max Leverage:    {cfg.max_leverage:.1f}×")
        logger.info(f"  Rebalance Fee:   {cfg.rebalance_fee_bps:.1f} bps")
        logger.info("=" * 70)

        # ─── Pre-compute signals ──────────────────────────────────────
        daily_returns = prices.pct_change()

        # 1. Absolute Momentum: BTC regime
        btc_close = prices["BTC"]
        btc_sma = btc_close.rolling(window=cfg.regime_sma_window, min_periods=cfg.regime_sma_window).mean()
        regime = (btc_close > btc_sma).astype(int)  # 1 = Risk-On, 0 = Risk-Off

        # 2. Relative Momentum: N-day returns for each asset
        momentum = prices.pct_change(periods=cfg.momentum_window)

        # 3. Volatility: 30-day annualized vol for each asset
        rolling_vol = daily_returns.rolling(window=cfg.vol_window, min_periods=cfg.vol_window).std() \
            * np.sqrt(TRADING_DAYS_PER_YEAR)

        # ─── Run backtest loop ────────────────────────────────────────
        dates = prices.index
        n = len(dates)

        # Output arrays
        strat_returns = np.zeros(n)
        selected_assets = ["CASH"] * n
        position_sizes = np.zeros(n)
        regime_arr = np.zeros(n, dtype=int)
        leverage_arr = np.zeros(n)

        # State
        current_asset = "CASH"
        current_leverage = 0.0
        last_rebalance_date = None

        # Warm-up period
        warmup = max(cfg.regime_sma_window, cfg.momentum_window, cfg.vol_window) + 1

        for i in range(n):
            date = dates[i]
            regime_arr[i] = regime.iloc[i] if not np.isnan(regime.iloc[i]) else 0

            # Skip warm-up
            if i < warmup:
                selected_assets[i] = "WARMUP"
                continue

            # ── Check if rebalance day ──
            needs_rebalance = False
            if last_rebalance_date is None:
                needs_rebalance = True
            else:
                days_since = (date - last_rebalance_date).days
                if days_since >= cfg.rebalance_days:
                    needs_rebalance = True

            if needs_rebalance:
                last_rebalance_date = date

                # ── 1. Regime Filter ──
                if regime_arr[i] == 0:
                    # Risk-Off: go to cash
                    new_asset = "CASH"
                    new_leverage = 0.0
                else:
                    # Risk-On: find best momentum asset
                    mom_values = {}
                    for asset in assets:
                        m = momentum[asset].iloc[i]
                        v = rolling_vol[asset].iloc[i]
                        if not np.isnan(m) and not np.isnan(v):
                            mom_values[asset] = m

                    if not mom_values:
                        new_asset = "CASH"
                        new_leverage = 0.0
                    else:
                        # Pick top-1 by momentum
                        sorted_assets = sorted(mom_values.items(),
                                               key=lambda x: x[1], reverse=True)
                        new_asset = sorted_assets[0][0]

                        # ── 3. Volatility Targeting ──
                        current_vol = rolling_vol[new_asset].iloc[i]
                        if current_vol > 0 and not np.isnan(current_vol):
                            raw_leverage = cfg.target_vol / current_vol
                            new_leverage = np.clip(raw_leverage,
                                                   cfg.min_leverage,
                                                   cfg.max_leverage)
                        else:
                            new_leverage = 1.0

                # Apply rebalance fee if changing position
                if new_asset != current_asset or abs(new_leverage - current_leverage) > 0.01:
                    fee = cfg.rebalance_fee_bps / 10_000
                    strat_returns[i] -= fee  # deduct fee

                current_asset = new_asset
                current_leverage = new_leverage

            # ── Calculate daily return ──
            if current_asset != "CASH" and current_asset != "WARMUP":
                asset_ret = daily_returns[current_asset].iloc[i]
                if not np.isnan(asset_ret):
                    strat_returns[i] += current_leverage * asset_ret

            selected_assets[i] = current_asset
            position_sizes[i] = current_leverage
            leverage_arr[i] = current_leverage

        # ─── Build results DataFrame ──────────────────────────────────
        results = pd.DataFrame(index=dates)
        results["strategy_return"] = strat_returns
        results["strategy_equity"] = (1 + results["strategy_return"]).cumprod()
        results["benchmark_return"] = daily_returns["BTC"].fillna(0)
        results["benchmark_equity"] = (1 + results["benchmark_return"]).cumprod()
        results["regime"] = regime_arr
        results["selected_asset"] = selected_assets
        results["position_size"] = position_sizes
        results["leverage"] = leverage_arr

        # Add component data for analysis
        results["btc_sma"] = btc_sma

        return results


# ═════════════════════════════════════════════════════════════════════════════
# Performance Analytics
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class PerformanceMetrics:
    """Container for strategy performance metrics."""
    total_return: float
    cagr: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration_days: int
    annual_vol: float
    calmar_ratio: float
    win_rate: float
    risk_on_pct: float
    avg_leverage: float
    num_rebalances: int
    exposure_pct: float


def compute_metrics(results: pd.DataFrame, label: str = "Strategy") -> PerformanceMetrics:
    """Compute comprehensive performance metrics."""
    returns = results[f"{'strategy' if label == 'Strategy' else 'benchmark'}_return"]
    equity = results[f"{'strategy' if label == 'Strategy' else 'benchmark'}_equity"]

    # Basic returns
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    n_years = (results.index[-1] - results.index[0]).days / 365.25
    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    # Risk metrics
    daily_vol = returns.std()
    annual_vol = daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = (returns.mean() / daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR)
              if daily_vol > 0 else 0)

    # Sortino (downside vol only)
    downside = returns[returns < 0]
    downside_vol = downside.std() * np.sqrt(TRADING_DAYS_PER_YEAR) if len(downside) > 0 else 1e-9
    sortino = cagr / downside_vol if downside_vol > 0 else 0

    # Max Drawdown
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_dd = drawdown.min()

    # Max Drawdown Duration
    underwater = drawdown < 0
    if underwater.any():
        groups = (~underwater).cumsum()
        underwater_groups = underwater.groupby(groups)
        dd_durations = []
        for _, group in underwater_groups:
            if group.any():
                duration = (group.index[-1] - group.index[0]).days
                dd_durations.append(duration)
        max_dd_duration = max(dd_durations) if dd_durations else 0
    else:
        max_dd_duration = 0

    # Calmar
    calmar = cagr / abs(max_dd) if abs(max_dd) > 0 else 0

    # Win rate (daily)
    non_zero = returns[returns != 0]
    win_rate = (non_zero > 0).mean() if len(non_zero) > 0 else 0

    # Strategy-specific metrics
    if label == "Strategy":
        regime = results["regime"]
        risk_on_pct = regime.mean()
        leverage = results["leverage"]
        avg_leverage = leverage[leverage > 0].mean() if (leverage > 0).any() else 0
        # Count rebalances (changes in selected asset or leverage)
        asset_changes = (results["selected_asset"] != results["selected_asset"].shift(1))
        num_rebalances = asset_changes.sum()
        exposure_pct = (leverage > 0).mean()
    else:
        risk_on_pct = 1.0
        avg_leverage = 1.0
        num_rebalances = 0
        exposure_pct = 1.0

    return PerformanceMetrics(
        total_return=total_return,
        cagr=cagr,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        max_drawdown_duration_days=max_dd_duration,
        annual_vol=annual_vol,
        calmar_ratio=calmar,
        win_rate=win_rate,
        risk_on_pct=risk_on_pct,
        avg_leverage=avg_leverage,
        num_rebalances=num_rebalances,
        exposure_pct=exposure_pct,
    )


def print_report(strat_metrics: PerformanceMetrics,
                 bench_metrics: PerformanceMetrics,
                 config: DualMomentumConfig) -> None:
    """Print a comprehensive side-by-side performance report."""
    print()
    print("═" * 78)
    print(" DUAL-MOMENTUM ROTATION STRATEGY — PERFORMANCE REPORT")
    print("═" * 78)
    print()

    # Configuration summary
    print("┌─ Configuration ──────────────────────────────────────────────────┐")
    print(f"│  Universe:          {str(list(config.tickers.keys())):<44s}│")
    print(f"│  Regime Filter:     BTC vs SMA({config.regime_sma_window})"
          f"{'':<31s}│")
    print(f"│  Momentum Lookback: {config.momentum_window} days"
          f"{'':<41s}│")
    print(f"│  Rebalance Freq:    Every {config.rebalance_days} days"
          f"{'':<36s}│")
    print(f"│  Vol Target:        {config.target_vol:.0%}"
          f"{'':<44s}│")
    print(f"│  Max Leverage:      {config.max_leverage:.1f}×"
          f"{'':<43s}│")
    print(f"│  Rebalance Fee:     {config.rebalance_fee_bps:.0f} bps"
          f"{'':<40s}│")
    print("└──────────────────────────────────────────────────────────────────┘")
    print()

    # Side-by-side metrics
    s = strat_metrics
    b = bench_metrics

    def row(label: str, sv: str, bv: str, highlight: bool = False):
        marker = " ◀" if highlight else ""
        print(f"  {label:<28s} {sv:>16s}  {bv:>16s}{marker}")

    print(f"  {'Metric':<28s} {'Strategy':>16s}  {'BTC B&H':>16s}")
    print(f"  {'─' * 28} {'─' * 16}  {'─' * 16}")

    row("Total Return",
        f"{s.total_return:+.2%}", f"{b.total_return:+.2%}",
        s.total_return > b.total_return)
    row("CAGR",
        f"{s.cagr:+.2%}", f"{b.cagr:+.2%}",
        s.cagr > b.cagr)
    row("Sharpe Ratio",
        f"{s.sharpe_ratio:.3f}", f"{b.sharpe_ratio:.3f}",
        s.sharpe_ratio > b.sharpe_ratio)
    row("Sortino Ratio",
        f"{s.sortino_ratio:.3f}", f"{b.sortino_ratio:.3f}",
        s.sortino_ratio > b.sortino_ratio)
    row("Max Drawdown",
        f"{s.max_drawdown:.2%}", f"{b.max_drawdown:.2%}",
        s.max_drawdown > b.max_drawdown)
    row("Max DD Duration",
        f"{s.max_drawdown_duration_days} days",
        f"{b.max_drawdown_duration_days} days",
        s.max_drawdown_duration_days < b.max_drawdown_duration_days)
    row("Annual Volatility",
        f"{s.annual_vol:.2%}", f"{b.annual_vol:.2%}")
    row("Calmar Ratio",
        f"{s.calmar_ratio:.3f}", f"{b.calmar_ratio:.3f}",
        s.calmar_ratio > b.calmar_ratio)
    row("Win Rate (daily)",
        f"{s.win_rate:.1%}", f"{b.win_rate:.1%}")

    print()
    print(f"  {'─' * 28} {'─' * 16}")
    print(f"  {'Strategy Internals':<28s}")
    print(f"  {'Risk-On Time':<28s} {s.risk_on_pct:>15.1%}")
    print(f"  {'Avg Leverage (when active)':<28s} {s.avg_leverage:>15.2f}×")
    print(f"  {'Exposure %':<28s} {s.exposure_pct:>15.1%}")
    print(f"  {'# Rebalances':<28s} {s.num_rebalances:>15d}")
    print()

    # Verdict
    print("┌─ Verdict ────────────────────────────────────────────────────────┐")
    if s.sharpe_ratio > b.sharpe_ratio:
        print("│  ✅ Strategy delivers SUPERIOR risk-adjusted returns vs B&H      │")
    else:
        print("│  ⚠️  Strategy UNDERPERFORMS B&H on risk-adjusted basis            │")

    if s.max_drawdown > b.max_drawdown:
        print("│  ✅ Strategy has LOWER max drawdown than B&H                      │")
    else:
        print("│  ⚠️  Strategy has HIGHER max drawdown than B&H                    │")

    if s.total_return > b.total_return:
        print("│  ✅ Strategy OUTPERFORMS B&H on absolute returns                  │")
    else:
        ratio = (s.total_return / b.total_return * 100) if b.total_return != 0 else 0
        print(f"│  📊 Strategy captures {ratio:.0f}% of B&H return "
              f"with {abs(s.max_drawdown)/abs(b.max_drawdown)*100:.0f}% of "
              f"the drawdown     │")
    print("└──────────────────────────────────────────────────────────────────┘")
    print()


# ═════════════════════════════════════════════════════════════════════════════
# Visualization
# ═════════════════════════════════════════════════════════════════════════════
def plot_results(results: pd.DataFrame, prices: pd.DataFrame,
                 config: DualMomentumConfig,
                 save_path: Optional[str] = None) -> None:
    """Generate institutional-grade performance visualization."""
    fig, axes = plt.subplots(5, 1, figsize=(16, 22),
                             gridspec_kw={"height_ratios": [3, 1.5, 1, 1, 1]})
    fig.suptitle("Dual-Momentum Rotation Strategy — Full Cycle Analysis",
                 fontsize=16, fontweight="bold", y=0.98)

    dates = results.index

    # ─── Panel 1: Equity Curves ───────────────────────────────────────
    ax1 = axes[0]
    ax1.semilogy(dates, results["strategy_equity"], color="#1a73e8",
                 linewidth=2.0, label="Dual-Momentum Strategy", zorder=5)
    ax1.semilogy(dates, results["benchmark_equity"], color="#ea4335",
                 linewidth=1.5, alpha=0.7, label="BTC Buy & Hold", zorder=4)

    # Shade Risk-Off periods
    regime = results["regime"]
    risk_off_starts = []
    in_risk_off = False
    start_idx = None
    for i in range(len(regime)):
        if regime.iloc[i] == 0 and not in_risk_off:
            in_risk_off = True
            start_idx = dates[i]
        elif regime.iloc[i] == 1 and in_risk_off:
            in_risk_off = False
            risk_off_starts.append((start_idx, dates[i]))
    if in_risk_off:
        risk_off_starts.append((start_idx, dates[-1]))

    for start, end in risk_off_starts:
        ax1.axvspan(start, end, color="#ffcdd2", alpha=0.3, zorder=1)

    ax1.set_ylabel("Portfolio Value (log scale)", fontsize=11)
    ax1.legend(loc="upper left", fontsize=11, framealpha=0.9)
    ax1.set_title("Equity Curve — Strategy vs. BTC Buy & Hold", fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.text(0.02, 0.05, "Red zones = Risk-Off (BTC < SMA200)",
             transform=ax1.transAxes, fontsize=9, color="#c62828", alpha=0.8)

    # ─── Panel 2: Drawdown ────────────────────────────────────────────
    ax2 = axes[1]
    strat_eq = results["strategy_equity"]
    bench_eq = results["benchmark_equity"]
    strat_dd = (strat_eq - strat_eq.cummax()) / strat_eq.cummax()
    bench_dd = (bench_eq - bench_eq.cummax()) / bench_eq.cummax()

    ax2.fill_between(dates, strat_dd, 0, color="#1a73e8", alpha=0.4,
                     label="Strategy DD")
    ax2.fill_between(dates, bench_dd, 0, color="#ea4335", alpha=0.3,
                     label="BTC B&H DD")
    ax2.set_ylabel("Drawdown", fontsize=11)
    ax2.set_ylim(min(bench_dd.min(), strat_dd.min()) * 1.1, 0.05)
    ax2.legend(loc="lower left", fontsize=10)
    ax2.set_title("Underwater Equity (Drawdown)", fontsize=13)
    ax2.grid(True, alpha=0.3)

    # ─── Panel 3: Selected Asset ──────────────────────────────────────
    ax3 = axes[2]
    asset_names = list(config.tickers.keys()) + ["CASH"]
    asset_to_num = {a: i for i, a in enumerate(asset_names)}
    asset_nums = [asset_to_num.get(a, -1) for a in results["selected_asset"]]

    colors = {"BTC": "#f7931a", "ETH": "#627eea", "SOL": "#9945ff",
              "BNB": "#f3ba2f", "CASH": "#90a4ae"}

    for asset_name in asset_names:
        mask = results["selected_asset"] == asset_name
        if mask.any():
            ax3.fill_between(dates, 0, 1, where=mask,
                             color=colors.get(asset_name, "#ccc"),
                             alpha=0.7, label=asset_name)

    ax3.set_ylabel("Allocation", fontsize=11)
    ax3.set_yticks([])
    ax3.legend(loc="upper right", ncol=5, fontsize=9, framealpha=0.9)
    ax3.set_title("Asset Allocation Over Time", fontsize=13)

    # ─── Panel 4: Leverage / Position Size ────────────────────────────
    ax4 = axes[3]
    ax4.fill_between(dates, results["leverage"], 0,
                     color="#1a73e8", alpha=0.4)
    ax4.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="1.0×")
    ax4.axhline(y=config.max_leverage, color="#c62828", linestyle="--",
                alpha=0.5, label=f"{config.max_leverage:.1f}× (max)")
    ax4.set_ylabel("Leverage", fontsize=11)
    ax4.set_ylim(-0.1, config.max_leverage + 0.3)
    ax4.legend(loc="upper right", fontsize=9)
    ax4.set_title("Position Leverage (Vol-Targeting)", fontsize=13)
    ax4.grid(True, alpha=0.3)

    # ─── Panel 5: BTC Price + SMA ─────────────────────────────────────
    ax5 = axes[4]
    ax5.plot(prices.index, prices["BTC"], color="#f7931a",
             linewidth=1.0, alpha=0.8, label="BTC Price")
    ax5.plot(results.index, results["btc_sma"], color="#1a73e8",
             linewidth=1.5, alpha=0.8, label=f"SMA({config.regime_sma_window})")
    ax5.set_ylabel("BTC Price (USD)", fontsize=11)
    ax5.set_yscale("log")
    ax5.legend(loc="upper left", fontsize=10)
    ax5.set_title("Bitcoin — Regime Filter (SMA 200)", fontsize=13)
    ax5.grid(True, alpha=0.3)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"📊 Chart saved: {save_path}")
    plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# Rolling Analysis
# ═════════════════════════════════════════════════════════════════════════════
def print_yearly_breakdown(results: pd.DataFrame) -> None:
    """Print year-by-year performance comparison."""
    print()
    print("┌─ Yearly Breakdown ───────────────────────────────────────────────┐")
    print(f"  {'Year':<8s} {'Strat Return':>14s} {'BTC Return':>14s} "
          f"{'Strat Sharpe':>14s} {'Risk-On %':>12s} {'Avg Lev':>10s}")
    print(f"  {'─' * 8} {'─' * 14} {'─' * 14} {'─' * 14} {'─' * 12} {'─' * 10}")

    for year in sorted(results.index.year.unique()):
        yr = results[results.index.year == year]
        if len(yr) < 30:
            continue

        s_ret = yr["strategy_equity"].iloc[-1] / yr["strategy_equity"].iloc[0] - 1
        b_ret = yr["benchmark_equity"].iloc[-1] / yr["benchmark_equity"].iloc[0] - 1
        s_sharpe = (yr["strategy_return"].mean() / yr["strategy_return"].std()
                    * np.sqrt(365)) if yr["strategy_return"].std() > 0 else 0
        risk_on = yr["regime"].mean()
        avg_lev = yr["leverage"][yr["leverage"] > 0].mean() if (yr["leverage"] > 0).any() else 0

        winner = "◀" if s_ret > b_ret else ""
        print(f"  {year:<8d} {s_ret:>+13.2%} {b_ret:>+13.2%} "
              f"{s_sharpe:>13.3f} {risk_on:>11.1%} {avg_lev:>9.2f}×  {winner}")

    print("└──────────────────────────────────────────────────────────────────┘")
    print()


def print_regime_analysis(results: pd.DataFrame) -> None:
    """Analyze performance by market regime."""
    print()
    print("┌─ Regime Analysis ────────────────────────────────────────────────┐")

    for regime_val, regime_name in [(1, "Risk-On (BTC > SMA200)"),
                                     (0, "Risk-Off (BTC < SMA200)")]:
        mask = results["regime"] == regime_val
        regime_data = results[mask]
        if len(regime_data) < 10:
            continue

        s_ret = regime_data["strategy_return"]
        b_ret = regime_data["benchmark_return"]

        ann_s = s_ret.mean() * 365
        ann_b = b_ret.mean() * 365
        vol_s = s_ret.std() * np.sqrt(365)

        print(f"  {regime_name}:")
        print(f"    Days:           {len(regime_data)}")
        print(f"    Strat Ann. Ret: {ann_s:+.2%}")
        print(f"    BTC Ann. Ret:   {ann_b:+.2%}")
        print(f"    Strat Vol:      {vol_s:.2%}")
        print()

    print("└──────────────────────────────────────────────────────────────────┘")


def print_asset_stats(results: pd.DataFrame) -> None:
    """Print statistics on which assets were selected and for how long."""
    print()
    print("┌─ Asset Selection Statistics ──────────────────────────────────────┐")

    total_days = len(results)
    asset_counts = results["selected_asset"].value_counts()

    for asset, count in asset_counts.items():
        pct = count / total_days
        print(f"  {asset:<8s}: {count:>5d} days ({pct:>6.1%})")

    print("└───────────────────────────────────────────────────────────────────┘")
    print()


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dual-Momentum Rotation Strategy Backtest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--start", default="2020-01-01",
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None,
                        help="End date (YYYY-MM-DD), default=present")
    parser.add_argument("--sma-window", type=int, default=200,
                        help="BTC SMA window for regime filter")
    parser.add_argument("--momentum-window", type=int, default=90,
                        help="Momentum lookback (days)")
    parser.add_argument("--rebalance-days", type=int, default=7,
                        help="Rebalance frequency (days)")
    parser.add_argument("--target-vol", type=float, default=0.40,
                        help="Target annualized volatility")
    parser.add_argument("--max-leverage", type=float, default=2.0,
                        help="Maximum leverage cap")
    parser.add_argument("--fee-bps", type=float, default=10.0,
                        help="Rebalance fee in basis points")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip chart generation")
    parser.add_argument("--save-plot", default=None,
                        help="Save chart to file path")
    return parser.parse_args()


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()

    # Build config
    config = DualMomentumConfig(
        start_date=args.start,
        end_date=args.end,
        regime_sma_window=args.sma_window,
        momentum_window=args.momentum_window,
        rebalance_days=args.rebalance_days,
        target_vol=args.target_vol,
        max_leverage=args.max_leverage,
        rebalance_fee_bps=args.fee_bps,
    )

    # ── 1. Load data ──
    logger.info("=" * 70)
    logger.info(" PHASE 1: DATA ACQUISITION")
    logger.info("=" * 70)
    prices = load_data(config)

    # ── 2. Run strategy ──
    logger.info("")
    logger.info("=" * 70)
    logger.info(" PHASE 2: STRATEGY EXECUTION")
    logger.info("=" * 70)
    strategy = DualMomentumStrategy(config)
    results = strategy.run(prices)

    # ── 3. Performance Analysis ──
    logger.info("")
    logger.info("=" * 70)
    logger.info(" PHASE 3: PERFORMANCE ANALYSIS")
    logger.info("=" * 70)

    strat_metrics = compute_metrics(results, "Strategy")
    bench_metrics = compute_metrics(results, "Benchmark")

    print_report(strat_metrics, bench_metrics, config)
    print_yearly_breakdown(results)
    print_regime_analysis(results)
    print_asset_stats(results)

    # ── 4. Plot ──
    if not args.no_plot:
        save_path = args.save_plot or "reports/dual_momentum_results.png"
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plot_results(results, prices, config, save_path=save_path)

    logger.info("✅ Backtest complete.")
    return results, strat_metrics, bench_metrics


if __name__ == "__main__":
    main()
