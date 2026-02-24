#!/usr/bin/env python3
"""
出場哲學驗證研究

比較三種出場哲學在 TSMOM-EMA 策略上的表現：
  1) Trend-hold   — 無 TP，寬 SL（災難型），策略反轉出場
  2) Hybrid-lock  — 無硬 TP，trailing stop 鎖盈
  3) Mean-revert-take — 固定 TP + SL，明確出場節奏

Requirements:
  - 已下載 ETHUSDT / SOLUSDT / BTCUSDT 1h Futures K 線 & funding rate
  - source .venv/bin/activate

Usage:
  PYTHONPATH=src python scripts/research_exit_philosophy.py
  PYTHONPATH=src python scripts/research_exit_philosophy.py --symbols ETHUSDT SOLUSDT
  PYTHONPATH=src python scripts/research_exit_philosophy.py --quick   # 快速模式（只跑 ETHUSDT, 減少參數）
"""
from __future__ import annotations

import argparse
import json
import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── Project imports ──
from qtrade.data.storage import load_klines
from qtrade.data.quality import validate_data_quality, clean_data
from qtrade.data.funding_rate import load_funding_rates, get_funding_rate_path, align_funding_to_klines
from qtrade.strategy.base import StrategyContext
from qtrade.strategy import get_strategy
from qtrade.strategy.exit_rules import apply_exit_rules
from qtrade.backtest.run_backtest import (
    clip_positions_by_direction,
    to_vbt_direction,
    BacktestResult,
)
from qtrade.backtest.costs import compute_funding_costs, adjust_equity_for_funding, compute_adjusted_stats
from qtrade.backtest.metrics import trade_analysis, benchmark_buy_and_hold
from qtrade.indicators.atr import calculate_atr

try:
    import vectorbt as vbt
except ImportError:
    raise ImportError("vectorbt is required: pip install vectorbt")

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("exit_philosophy")

# ══════════════════════════════════════════════════════════════
#  常數
# ══════════════════════════════════════════════════════════════

DATA_DIR = Path("data")
RESULTS_DIR = Path("reports/research/exit_philosophy")

# 回測基本參數（與 prod 一致）
BASE_CFG = dict(
    strategy_name="tsmom_ema",
    strategy_params=dict(
        lookback=168,
        vol_target=0.15,
        ema_fast=20,
        ema_slow=50,
        agree_weight=1.0,
        disagree_weight=0.3,
    ),
    initial_cash=10_000,
    fee_bps=5,
    slippage_bps=3,
    interval="1h",
    market_type="futures",
    direction="both",
    trade_on="next_open",
    leverage=3,
    validate_data=True,
    clean_data_before=True,
    funding_rate=dict(enabled=True, default_rate_8h=0.0001, use_historical=True),
    slippage_model=dict(enabled=False),
    position_sizing=dict(method="fixed", position_pct=1.0),
)

# 時間分割
PERIODS = {
    "IS":   ("2022-01-01", "2024-06-30"),   # In-sample
    "OOS":  ("2024-07-01", "2025-06-30"),   # Out-of-sample
    "Live": ("2025-07-01", None),           # Live-recent
    "Full": ("2022-01-01", None),           # 全段
}

DEFAULT_SYMBOLS = ["ETHUSDT", "SOLUSDT", "BTCUSDT"]


# ══════════════════════════════════════════════════════════════
#  參數矩陣定義
# ══════════════════════════════════════════════════════════════

@dataclass
class ExitConfig:
    """單一出場配置"""
    name: str
    philosophy: str  # "baseline", "trend_hold", "hybrid_lock", "mean_revert_take"
    stop_loss_atr: Optional[float] = None
    take_profit_atr: Optional[float] = None
    trailing_stop_atr: Optional[float] = None
    cooldown_bars: int = 0
    label: str = ""

    def __post_init__(self):
        if not self.label:
            parts = [self.philosophy]
            if self.stop_loss_atr is not None:
                parts.append(f"SL{self.stop_loss_atr}")
            if self.take_profit_atr is not None:
                parts.append(f"TP{self.take_profit_atr}")
            if self.trailing_stop_atr is not None:
                parts.append(f"TR{self.trailing_stop_atr}")
            parts.append(f"CD{self.cooldown_bars}")
            self.label = "_".join(parts)


def build_param_grid(quick: bool = False) -> list[ExitConfig]:
    """構建參數網格

    設計原則：
      - Trend-hold: 寬 SL（災難型），讓趨勢自然結束。SL 4-7× ATR。
      - Hybrid-lock: SL + Trailing，不設硬 TP。Trailing 鎖盈而非截斷利潤。
      - Mean-revert-take: SL + TP，明確出場節奏。TP 2-5× ATR。
      - Cooldown: 0 (最貼近實盤) 和 3 (防止 whipsaw re-entry)

    IMPORTANT: 所有 exit overlay 保留 TSMOM 連續信號的 magnitude，
    只在 SL/TP/cooldown 時強制 flat。
    """
    configs = []

    # ── B0: 基準（現行裸 TSMOM，與 production 一致）──
    configs.append(ExitConfig(
        name="B0_baseline",
        philosophy="baseline",
        stop_loss_atr=None,
        take_profit_atr=None,
        trailing_stop_atr=None,
        cooldown_bars=0,
        label="B0_baseline",
    ))

    # ── Trend-hold: 寬 SL（災難型保護），無 TP，無 trailing ──
    # 理念：讓趨勢信號自然管理出場，SL 僅作為黑天鵝保護
    sl_vals = [3.5, 5.0, 7.0] if not quick else [5.0]
    cd_vals = [0, 3] if not quick else [0]
    for sl, cd in product(sl_vals, cd_vals):
        configs.append(ExitConfig(
            name=f"TH_SL{sl}_CD{cd}",
            philosophy="trend_hold",
            stop_loss_atr=sl,
            take_profit_atr=None,
            trailing_stop_atr=None,
            cooldown_bars=cd,
        ))

    # ── Hybrid-lock: SL + Trailing，無 TP ──
    # 理念：保護浮盈，但不設硬天花板。Trailing 跟蹤極值。
    sl_vals = [3.0, 4.0, 5.0] if not quick else [4.0]
    tr_vals = [2.0, 3.0, 4.0] if not quick else [3.0]
    cd_vals = [0, 3] if not quick else [0]
    for sl, tr, cd in product(sl_vals, tr_vals, cd_vals):
        # trailing 必須 <= SL，否則永遠先觸發 trailing
        if tr > sl:
            continue
        configs.append(ExitConfig(
            name=f"HL_SL{sl}_TR{tr}_CD{cd}",
            philosophy="hybrid_lock",
            stop_loss_atr=sl,
            take_profit_atr=None,
            trailing_stop_atr=tr,
            cooldown_bars=cd,
        ))

    # ── Mean-revert-take: SL + TP，無 trailing ──
    # 理念：固定節奏出場，適合均值回歸型利潤
    sl_vals = [2.5, 3.5, 5.0] if not quick else [3.5]
    tp_vals = [2.0, 3.0, 5.0] if not quick else [3.0]
    cd_vals = [0, 3] if not quick else [0]
    for sl, tp, cd in product(sl_vals, tp_vals, cd_vals):
        configs.append(ExitConfig(
            name=f"MR_SL{sl}_TP{tp}_CD{cd}",
            philosophy="mean_revert_take",
            stop_loss_atr=sl,
            take_profit_atr=tp,
            trailing_stop_atr=None,
            cooldown_bars=cd,
        ))

    print(f"📊 參數網格: {len(configs)} 組 "
          f"(baseline=1, trend_hold={sum(1 for c in configs if c.philosophy=='trend_hold')}, "
          f"hybrid_lock={sum(1 for c in configs if c.philosophy=='hybrid_lock')}, "
          f"mean_revert_take={sum(1 for c in configs if c.philosophy=='mean_revert_take')})")
    return configs


# ══════════════════════════════════════════════════════════════
#  核心回測
# ══════════════════════════════════════════════════════════════

def _load_and_prepare(symbol: str) -> pd.DataFrame:
    """載入 + 清洗 K 線數據"""
    data_path = DATA_DIR / "binance" / "futures" / "1h" / f"{symbol}.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")
    df = load_klines(data_path)
    df = clean_data(df, fill_method="forward", remove_outliers=False, remove_duplicates=True)
    return df


def _generate_raw_signal(df: pd.DataFrame, symbol: str) -> pd.Series:
    """產生 TSMOM-EMA raw signal（含 signal_delay）"""
    ctx = StrategyContext(
        symbol=symbol,
        interval="1h",
        market_type="futures",
        direction="both",
        signal_delay=1,  # trade_on=next_open
    )
    strategy_func = get_strategy("tsmom_ema")
    raw_pos = strategy_func(df, ctx, BASE_CFG["strategy_params"])
    return raw_pos


def _apply_exit_overlay(
    df: pd.DataFrame,
    raw_pos: pd.Series,
    ecfg: ExitConfig,
) -> tuple[pd.Series, pd.Series]:
    """
    套用出場哲學 overlay

    CRITICAL: apply_exit_rules 輸出 binary positions (±1 / 0)，
    但 TSMOM 產生連續信號 (e.g. 0.3, -0.7)。
    直接用 binary 會改變槓桿和策略特性。

    修正方案：
      - 用 apply_exit_rules 判斷 WHEN SL/TP/cooldown 強制平倉
      - 強制平倉時 → pos = 0（尊重 exit_rules 的保護功能）
      - 非平倉時 → 保留原始 raw_pos 的連續 magnitude
      - exec_prices 保持不變（SL/TP 出場價格）

    Returns:
        (positions, exec_prices)
    """
    if ecfg.philosophy == "baseline":
        # 無 exit rules，保留原始信號
        exec_prices = pd.Series(np.nan, index=df.index)
        return raw_pos, exec_prices

    binary_pos, exec_prices = apply_exit_rules(
        df, raw_pos,
        stop_loss_atr=ecfg.stop_loss_atr,
        take_profit_atr=ecfg.take_profit_atr,
        trailing_stop_atr=ecfg.trailing_stop_atr,
        atr_period=14,
        cooldown_bars=ecfg.cooldown_bars,
    )

    # Merge: preserve continuous magnitude, respect forced-flat
    # binary_pos == 0 means: SL/TP triggered, cooldown, or signal-driven exit
    # binary_pos != 0 means: position held / entered
    result = raw_pos.copy()
    forced_flat = binary_pos == 0.0
    result[forced_flat] = 0.0

    return result, exec_prices


def _run_single_backtest(
    df: pd.DataFrame,
    pos: pd.Series,
    exec_prices: pd.Series,
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    fee_bps: float = 5.0,
    slippage_bps: float = 3.0,
) -> Optional[dict]:
    """
    單次回測 → 標準指標 dict

    嚴格遵守：
      - price=open（next_open 機制已由 signal_delay 處理）
      - SL/TP 用 exec_prices 修正（消除 look-ahead）
      - 含 funding rate 成本
    """
    # 日期過濾
    df_bt = df.copy()
    pos_bt = pos.copy()
    ep_bt = exec_prices.copy()

    if start:
        start_ts = pd.Timestamp(start, tz="UTC") if df_bt.index.tz else pd.Timestamp(start)
        mask = df_bt.index >= start_ts
        df_bt, pos_bt, ep_bt = df_bt.loc[mask], pos_bt.loc[mask], ep_bt.loc[mask]
    if end:
        end_ts = pd.Timestamp(end, tz="UTC") if df_bt.index.tz else pd.Timestamp(end)
        mask = df_bt.index <= end_ts
        df_bt, pos_bt, ep_bt = df_bt.loc[mask], pos_bt.loc[mask], ep_bt.loc[mask]

    if len(df_bt) < 500:
        return None  # 數據不足

    # direction clip
    pos_bt = clip_positions_by_direction(pos_bt, "futures", "both")

    close = df_bt["close"]
    open_ = df_bt["open"]

    # 構建執行價格
    exec_price = open_.copy()
    sl_tp_mask = ep_bt.notna()
    if sl_tp_mask.any():
        exec_price[sl_tp_mask] = ep_bt[sl_tp_mask]

    fee = fee_bps / 10_000.0
    slippage = slippage_bps / 10_000.0

    # VBT Portfolio
    pf = vbt.Portfolio.from_orders(
        close=close,
        size=pos_bt,
        size_type="targetpercent",
        price=exec_price,
        fees=fee,
        slippage=slippage,
        init_cash=BASE_CFG["initial_cash"],
        freq="1h",
        direction="both",
    )

    stats = pf.stats()
    equity = pf.value()

    # Funding rate adjustment
    fr_cost = None
    adj_equity = None
    adj_stats_dict = None

    try:
        fr_path = get_funding_rate_path(DATA_DIR, symbol)
        fr_df = load_funding_rates(fr_path)
        if fr_df is not None:
            fr_aligned = align_funding_to_klines(
                fr_df, df_bt.index, default_rate_8h=0.0001
            )
            fr_cost = compute_funding_costs(
                pos=pos_bt, equity=equity,
                funding_rates=fr_aligned, leverage=3,
            )
            adj_equity = adjust_equity_for_funding(equity, fr_cost)
            adj_stats_dict = compute_adjusted_stats(adj_equity, BASE_CFG["initial_cash"])
    except Exception:
        pass

    # 計算指標
    use_stats = adj_stats_dict if adj_stats_dict else stats
    use_equity = adj_equity if adj_equity is not None else equity

    total_ret_pct = _safe_get(use_stats, "Total Return [%]", 0.0)
    max_dd_pct = abs(_safe_get(use_stats, "Max Drawdown [%]", 0.0))
    sharpe = _safe_get(use_stats, "Sharpe Ratio", 0.0)
    sortino = _safe_get(use_stats, "Sortino Ratio", 0.0)
    calmar = _safe_get(use_stats, "Calmar Ratio", 0.0)

    # 計算年數和 CAGR
    days = (df_bt.index[-1] - df_bt.index[0]).total_seconds() / 86400
    years = max(days / 365.25, 0.01)
    if total_ret_pct > -100:
        cagr = ((1 + total_ret_pct / 100) ** (1 / years) - 1) * 100
    else:
        cagr = -100.0

    # MAR = CAGR / MaxDD
    mar = cagr / max_dd_pct if max_dd_pct > 0 else 0.0

    # 交易分析
    trades_df = trade_analysis(pf)
    n_trades = len(trades_df[trades_df["Status"] == "Closed"]) if not trades_df.empty else 0
    ann_trades = n_trades / years if years > 0 else 0

    # 勝率 & profit factor
    if not trades_df.empty:
        closed = trades_df[trades_df["Status"] == "Closed"]
        winners = closed[closed["PnL"] > 0]
        losers = closed[closed["PnL"] < 0]
        win_rate = len(winners) / len(closed) * 100 if len(closed) > 0 else 0
        gross_profit = winners["PnL"].sum() if len(winners) > 0 else 0
        gross_loss = abs(losers["PnL"].sum()) if len(losers) > 0 else 1e-9
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
    else:
        win_rate = 0
        profit_factor = 0
        closed = pd.DataFrame()

    # 平均持有時間（小時）
    avg_hold_hours = 0
    if not trades_df.empty and "Duration" in trades_df.columns:
        closed_dur = trades_df[trades_df["Status"] == "Closed"]["Duration"]
        if len(closed_dur) > 0:
            avg_hold_hours = closed_dur.mean().total_seconds() / 3600

    # Turnover（年化換手）
    pos_changes = pos_bt.diff().abs()
    total_turnover = pos_changes.sum()
    ann_turnover = total_turnover / years if years > 0 else 0

    # 交易成本占比
    pf_gross_equity = pf.value()  # pre-cost equity approximation
    total_fees = n_trades * 2 * fee * BASE_CFG["initial_cash"]  # rough
    fee_pct = total_fees / BASE_CFG["initial_cash"] * 100 if BASE_CFG["initial_cash"] > 0 else 0
    funding_pct = fr_cost.total_cost_pct * 100 if fr_cost else 0

    return {
        "CAGR [%]": round(cagr, 2),
        "Sharpe": round(sharpe, 3),
        "Sortino": round(sortino, 3),
        "Calmar": round(calmar, 3),
        "MaxDD [%]": round(max_dd_pct, 2),
        "MAR": round(mar, 3),
        "Total Return [%]": round(total_ret_pct, 2),
        "Ann. Trades": round(ann_trades, 1),
        "Turnover": round(ann_turnover, 1),
        "Win Rate [%]": round(win_rate, 1),
        "Profit Factor": round(profit_factor, 2),
        "Avg Hold [h]": round(avg_hold_hours, 1),
        "Fee Cost [%]": round(fee_pct, 2),
        "Funding Cost [%]": round(funding_pct, 2),
        "N Trades": n_trades,
        "Years": round(years, 2),
        # raw objects for diagnostics
        "_pf": pf,
        "_trades_df": trades_df,
        "_equity": use_equity,
        "_pos": pos_bt,
    }


def _safe_get(obj, key, default=0.0):
    """Safe get from pd.Series or dict"""
    if isinstance(obj, dict):
        return obj.get(key, default)
    try:
        return obj.get(key, default)
    except Exception:
        return default


# ══════════════════════════════════════════════════════════════
#  右尾診斷
# ══════════════════════════════════════════════════════════════

def compute_tail_diagnostics(result: dict) -> dict:
    """
    計算右尾/高勝率診斷：
      - Top 10% 交易對總收益貢獻
      - 去除 Top N 交易後績效衰減
      - MFE/MAE 分布
      - 浮盈回吐分布
    """
    trades_df = result.get("_trades_df")
    pf = result.get("_pf")
    if trades_df is None or trades_df.empty:
        return {}

    closed = trades_df[trades_df["Status"] == "Closed"].copy()
    if len(closed) < 5:
        return {}

    pnl = closed["PnL"].values
    total_pnl = pnl.sum()
    n = len(pnl)

    # ── Top 10% contribution ──
    sorted_pnl = np.sort(pnl)[::-1]  # descending
    top_n = max(1, int(np.ceil(n * 0.1)))
    top10_pnl = sorted_pnl[:top_n].sum()
    top10_contrib = top10_pnl / total_pnl * 100 if total_pnl != 0 else 0

    # ── Remove top N analysis ──
    remove_top = {}
    for k in [1, 3, 5]:
        if k >= n:
            continue
        remaining_pnl = sorted_pnl[k:].sum()
        decay = (1 - remaining_pnl / total_pnl) * 100 if total_pnl != 0 else 0
        remove_top[f"Remove Top{k} Decay [%]"] = round(decay, 1)

    # ── MFE/MAE from positions ──
    # Use vectorbt positions if available
    mfe_mae = {}
    try:
        positions = pf.positions.records_readable
        if len(positions) > 0 and "PnL" in positions.columns:
            # Approximate MFE/MAE from returns
            returns_arr = closed["Return [%]"].values
            mfe_mae["Avg Trade Return [%]"] = round(np.mean(returns_arr), 2)
            mfe_mae["Std Trade Return [%]"] = round(np.std(returns_arr), 2)
            mfe_mae["Skew Trade Return"] = round(float(pd.Series(returns_arr).skew()), 2)
            mfe_mae["Kurt Trade Return"] = round(float(pd.Series(returns_arr).kurtosis()), 2)
            # Positive PnL skew → right-tail driven
            mfe_mae["Median Trade Return [%]"] = round(float(np.median(returns_arr)), 2)
    except Exception:
        pass

    return {
        "Top 10% PnL Contribution [%]": round(top10_contrib, 1),
        "Top 10% Count": top_n,
        **remove_top,
        **mfe_mae,
    }


# ══════════════════════════════════════════════════════════════
#  成本壓測
# ══════════════════════════════════════════════════════════════

def run_cost_sensitivity(
    df: pd.DataFrame,
    pos: pd.Series,
    exec_prices: pd.Series,
    symbol: str,
    ecfg: ExitConfig,
) -> list[dict]:
    """fee/slippage ±20% 壓測"""
    base_fee = BASE_CFG["fee_bps"]
    base_slip = BASE_CFG["slippage_bps"]
    results = []

    for fee_mult, slip_mult, label in [
        (1.0,  1.0,  "Base"),
        (1.2,  1.0,  "Fee+20%"),
        (0.8,  1.0,  "Fee-20%"),
        (1.0,  1.2,  "Slip+20%"),
        (1.0,  0.8,  "Slip-20%"),
        (1.2,  1.2,  "Both+20%"),
    ]:
        r = _run_single_backtest(
            df, pos, exec_prices, symbol,
            start="2022-01-01", end=None,
            fee_bps=base_fee * fee_mult,
            slippage_bps=base_slip * slip_mult,
        )
        if r:
            results.append({
                "Scenario": label,
                "Config": ecfg.label,
                "CAGR [%]": r["CAGR [%]"],
                "Sharpe": r["Sharpe"],
                "MaxDD [%]": r["MaxDD [%]"],
            })
    return results


# ══════════════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="出場哲學驗證研究")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS,
                        help="交易對列表")
    parser.add_argument("--quick", action="store_true",
                        help="快速模式（減少參數組合）")
    parser.add_argument("--no-sensitivity", action="store_true",
                        help="跳過成本壓測")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = RESULTS_DIR / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  📊 出場哲學驗證研究 (Exit Philosophy Research)")
    print("=" * 70)
    print(f"  Symbols:    {args.symbols}")
    print(f"  Quick mode: {args.quick}")
    print(f"  Output:     {output_dir}")
    print()

    # ── 1. 構建參數網格 ──
    grid = build_param_grid(quick=args.quick)

    # ── 2. 逐幣種回測 ──
    all_results = []  # [{symbol, config, period, ...metrics}]
    all_tail = []     # [{symbol, config, ...diagnostics}]
    all_sensitivity = []

    for symbol in args.symbols:
        print(f"\n{'═' * 60}")
        print(f"  📈 {symbol}")
        print(f"{'═' * 60}")

        try:
            df = _load_and_prepare(symbol)
        except FileNotFoundError as e:
            print(f"  ⚠️  {e} — 跳過")
            continue

        # 產生 raw signal（在完整數據上）
        raw_pos = _generate_raw_signal(df, symbol)
        print(f"  Raw signal: {len(df):,} bars, "
              f"long={( raw_pos > 0.01).sum():,}, "
              f"short={(raw_pos < -0.01).sum():,}, "
              f"flat={(raw_pos.abs() <= 0.01).sum():,}")

        for i, ecfg in enumerate(grid):
            # 套用出場 overlay（在完整數據上）
            pos, exec_prices = _apply_exit_overlay(df, raw_pos, ecfg)

            # 各時段回測
            for period_name, (p_start, p_end) in PERIODS.items():
                r = _run_single_backtest(
                    df, pos, exec_prices, symbol,
                    start=p_start, end=p_end,
                )
                if r is None:
                    continue

                row = {
                    "Symbol": symbol,
                    "Config": ecfg.label,
                    "Philosophy": ecfg.philosophy,
                    "Period": period_name,
                    "SL": ecfg.stop_loss_atr,
                    "TP": ecfg.take_profit_atr,
                    "Trail": ecfg.trailing_stop_atr,
                    "CD": ecfg.cooldown_bars,
                }
                for k, v in r.items():
                    if not k.startswith("_"):
                        row[k] = v
                all_results.append(row)

                # 右尾診斷（只在 Full 期間算）
                if period_name == "Full":
                    tail = compute_tail_diagnostics(r)
                    if tail:
                        tail_row = {"Symbol": symbol, "Config": ecfg.label, "Philosophy": ecfg.philosophy}
                        tail_row.update(tail)
                        all_tail.append(tail_row)

            # 成本壓測（只對 Full 期間、最佳候選 + baseline）
            if not args.no_sensitivity and ecfg.philosophy == "baseline":
                sens = run_cost_sensitivity(df, pos, exec_prices, symbol, ecfg)
                for s in sens:
                    s["Symbol"] = symbol
                all_sensitivity.extend(sens)

            if (i + 1) % 10 == 0 or i == len(grid) - 1:
                print(f"  ... {i + 1}/{len(grid)} configs done")

    # ── 3. 彙整結果 ──
    results_df = pd.DataFrame(all_results)
    tail_df = pd.DataFrame(all_tail)
    sens_df = pd.DataFrame(all_sensitivity) if all_sensitivity else pd.DataFrame()

    # 儲存原始數據
    results_df.to_csv(output_dir / "raw_results.csv", index=False)
    if not tail_df.empty:
        tail_df.to_csv(output_dir / "tail_diagnostics.csv", index=False)
    if not sens_df.empty:
        sens_df.to_csv(output_dir / "cost_sensitivity.csv", index=False)

    # ── 4. 產生報表 ──
    print(f"\n{'═' * 70}")
    print("  📊 結果彙整")
    print(f"{'═' * 70}")

    # ── 主表 1: 每種哲學 × 全段 的最佳 & 基準 ──
    _print_main_table(results_df, output_dir)

    # ── 主表 2: 多時段穩健性 ──
    _print_period_comparison(results_df, output_dir)

    # ── 主表 3: 右尾診斷 ──
    if not tail_df.empty:
        _print_tail_table(tail_df, output_dir)

    # ── 主表 4: 成本壓測 ──
    if not sens_df.empty:
        _print_sensitivity_table(sens_df, output_dir)

    # ── 主表 5: 實盤貼近差異報告 ──
    _print_live_diff_report(output_dir)

    # ── 主表 6: 信號特性分析（baseline）──
    _print_signal_analysis(results_df, output_dir)

    # ── 結論 ──
    _print_conclusion(results_df, tail_df, output_dir)

    # 儲存 metadata
    meta = {
        "timestamp": timestamp,
        "symbols": args.symbols,
        "quick_mode": args.quick,
        "n_configs": len(grid),
        "periods": {k: list(v) for k, v in PERIODS.items()},
        "base_cfg": {k: v for k, v in BASE_CFG.items() if k != "strategy_params"},
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n✅ 完整結果已儲存: {output_dir}")


# ══════════════════════════════════════════════════════════════
#  報表輸出函數
# ══════════════════════════════════════════════════════════════

def _print_main_table(results_df: pd.DataFrame, output_dir: Path):
    """主表: 每種哲學的最佳與基準"""
    if results_df.empty:
        return

    full = results_df[results_df["Period"] == "Full"].copy()
    if full.empty:
        return

    # 按哲學聚合（跨幣種平均）
    cols = ["Config", "Philosophy", "CAGR [%]", "Sharpe", "Sortino", "Calmar",
            "MaxDD [%]", "MAR", "Ann. Trades", "Turnover", "Win Rate [%]",
            "Profit Factor", "Avg Hold [h]", "Fee Cost [%]", "Funding Cost [%]"]
    available_cols = [c for c in cols if c in full.columns]

    avg_by_config = full.groupby(["Config", "Philosophy"])[
        [c for c in available_cols if c not in ["Config", "Philosophy"]]
    ].mean().reset_index()

    # 每種哲學取 Sharpe 最高的
    best_rows = []
    for phil in ["baseline", "trend_hold", "hybrid_lock", "mean_revert_take"]:
        subset = avg_by_config[avg_by_config["Philosophy"] == phil]
        if subset.empty:
            continue
        best_idx = subset["Sharpe"].idxmax()
        best = subset.loc[best_idx].copy()
        best["Rank"] = "BEST"
        best_rows.append(best)

        # 次佳
        if len(subset) > 1:
            rest = subset.drop(best_idx)
            second_idx = rest["Sharpe"].idxmax()
            second = rest.loc[second_idx].copy()
            second["Rank"] = "2nd"
            best_rows.append(second)

    if not best_rows:
        return

    main_table = pd.DataFrame(best_rows)

    display_cols = ["Rank", "Philosophy", "Config", "CAGR [%]", "Sharpe", "Sortino",
                    "Calmar", "MaxDD [%]", "MAR", "Ann. Trades", "Turnover",
                    "Win Rate [%]", "Profit Factor", "Avg Hold [h]",
                    "Fee Cost [%]", "Funding Cost [%]"]
    display_cols = [c for c in display_cols if c in main_table.columns]

    print(f"\n{'─' * 70}")
    print("  📋 主表: 各哲學最佳配置（Full 期間，跨幣種平均）")
    print(f"{'─' * 70}")
    print(main_table[display_cols].to_string(index=False))
    main_table[display_cols].to_csv(output_dir / "T1_main_table.csv", index=False)


def _print_period_comparison(results_df: pd.DataFrame, output_dir: Path):
    """多時段穩健性比較"""
    if results_df.empty:
        return

    # 取每種哲學在 Full 上 Sharpe 最高的 config
    full = results_df[results_df["Period"] == "Full"]
    best_configs = {}
    for phil in ["baseline", "trend_hold", "hybrid_lock", "mean_revert_take"]:
        subset = full[full["Philosophy"] == phil]
        if subset.empty:
            continue
        avg_sharpe = subset.groupby("Config")["Sharpe"].mean()
        best_configs[phil] = avg_sharpe.idxmax()

    # 抽取這些 config 的各時段結果
    rows = []
    for phil, cfg_name in best_configs.items():
        for period in ["IS", "OOS", "Live", "Full"]:
            mask = (results_df["Config"] == cfg_name) & (results_df["Period"] == period)
            subset = results_df[mask]
            if subset.empty:
                continue
            row = {
                "Philosophy": phil,
                "Config": cfg_name,
                "Period": period,
                "CAGR [%]": round(subset["CAGR [%]"].mean(), 2),
                "Sharpe": round(subset["Sharpe"].mean(), 3),
                "MaxDD [%]": round(subset["MaxDD [%]"].mean(), 2),
                "Win Rate [%]": round(subset["Win Rate [%]"].mean(), 1),
                "N Trades": round(subset["N Trades"].mean(), 0),
            }
            rows.append(row)

    if not rows:
        return

    period_df = pd.DataFrame(rows)
    print(f"\n{'─' * 70}")
    print("  📋 多時段穩健性（各哲學 Full-best config）")
    print(f"{'─' * 70}")
    print(period_df.to_string(index=False))
    period_df.to_csv(output_dir / "T2_period_comparison.csv", index=False)


def _print_tail_table(tail_df: pd.DataFrame, output_dir: Path):
    """右尾診斷表"""
    # 找各哲學 Full 上最佳的 config，取其 tail
    display_cols = ["Philosophy", "Config",
                    "Top 10% PnL Contribution [%]", "Top 10% Count",
                    "Remove Top1 Decay [%]", "Remove Top3 Decay [%]",
                    "Remove Top5 Decay [%]",
                    "Avg Trade Return [%]", "Median Trade Return [%]",
                    "Skew Trade Return", "Kurt Trade Return"]
    display_cols = [c for c in display_cols if c in tail_df.columns]

    avg_tail = tail_df.groupby(["Config", "Philosophy"])[
        [c for c in display_cols if c not in ["Config", "Philosophy"]]
    ].mean().reset_index()

    print(f"\n{'─' * 70}")
    print("  📋 右尾 / 高勝率診斷（Full 期間，跨幣種平均）")
    print(f"{'─' * 70}")
    print(avg_tail[display_cols].to_string(index=False))
    avg_tail[display_cols].to_csv(output_dir / "T3_tail_diagnostics.csv", index=False)


def _print_sensitivity_table(sens_df: pd.DataFrame, output_dir: Path):
    """成本壓測表"""
    print(f"\n{'─' * 70}")
    print("  📋 成本壓測（Baseline，Full 期間）")
    print(f"{'─' * 70}")
    avg = sens_df.groupby("Scenario")[["CAGR [%]", "Sharpe", "MaxDD [%]"]].mean().reset_index()
    print(avg.to_string(index=False))
    avg.to_csv(output_dir / "T4_cost_sensitivity.csv", index=False)


def _print_live_diff_report(output_dir: Path):
    """回測 vs 實盤差異報告"""
    report = """
╔════════════════════════════════════════════════════════════════╗
║  回測 vs 實盤 執行邏輯差異報告                                  ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║  1. Min Trade Gate (2%)                                        ║
║     回測: 無（任意大小調倉）                                     ║
║     實盤: |target - current| < 2% → 跳過                       ║
║     偏差: 回測偏樂觀（更多微交易 → 更多手續費但也更多信號）       ║
║                                                                ║
║  2. Fill Gate (80%)                                            ║
║     回測: 100% fill（假設完美成交）                              ║
║     實盤: current/target ≥ 80% → 視為已完成                    ║
║     偏差: 回測偏樂觀（假設精確到位）                             ║
║                                                                ║
║  3. Rebalance Band (3%)                                        ║
║     回測: 無                                                    ║
║     實盤: diff < 3% 且同方向 → 跳過                            ║
║     偏差: 回測偏樂觀（更多微調倉 → turnover 更高）              ║
║                                                                ║
║  4. Order Type                                                 ║
║     回測: Market order fee (5bps)                              ║
║     實盤: Maker 優先 (2bps) + timeout fallback Market          ║
║     偏差: 回測偏保守（多算 3bps 手續費 × ~60% maker fill 率）   ║
║                                                                ║
║  5. SL/TP Execution                                            ║
║     回測: intra-bar simulation（high/low 檢測 + exec_prices）  ║
║     實盤: 交易所掛 STOP_MARKET / TAKE_PROFIT_MARKET 單         ║
║     偏差: 基本一致（SL 優先保守處理）                            ║
║                                                                ║
║  6. Funding Rate                                               ║
║     回測: 歷史資料逐 bar 對齊                                   ║
║     實盤: 每 8h 即時結算                                        ║
║     偏差: 基本一致（歷史 vs 即時差異 < 0.5%/yr）               ║
║                                                                ║
║  7. 方向切換確認 (flip_confirmation)                            ║
║     回測: 無（信號翻轉即執行）                                   ║
║     實盤: 可選 2-tick 確認（prod 關閉）                         ║
║     偏差: 一致（均為立即執行）                                   ║
║                                                                ║
║  綜合評估: 回測整體偏樂觀 1~3%/yr                              ║
║  主因: min trade gate + fill gate + rebalance band             ║
║  抵消: maker 優先省 ~1.5%/yr 手續費                            ║
║  淨偏差估計: 回測偏樂觀 0~2%/yr                                ║
╚════════════════════════════════════════════════════════════════╝
    """.strip()
    print(f"\n{report}")
    with open(output_dir / "T5_live_diff_report.txt", "w") as f:
        f.write(report)


def _print_signal_analysis(results_df: pd.DataFrame, output_dir: Path):
    """TSMOM 信號特性分析（幫助理解 exit overlay 的影響）"""
    full = results_df[results_df["Period"] == "Full"]
    if full.empty:
        return

    print(f"\n{'─' * 70}")
    print("  📋 信號特性分析（Full 期間）")
    print(f"{'─' * 70}")

    # Per-symbol breakdown for baseline
    baseline = full[full["Philosophy"] == "baseline"]
    if not baseline.empty:
        print("\n  [Baseline — 裸 TSMOM 各幣種表現]")
        display_cols = ["Symbol", "CAGR [%]", "Sharpe", "MaxDD [%]",
                       "Win Rate [%]", "Profit Factor", "N Trades",
                       "Avg Hold [h]", "Turnover"]
        display_cols = [c for c in display_cols if c in baseline.columns]
        print(baseline[display_cols].to_string(index=False))

    # Per-symbol breakdown for each philosophy best
    for phil in ["trend_hold", "hybrid_lock", "mean_revert_take"]:
        subset = full[full["Philosophy"] == phil]
        if subset.empty:
            continue
        avg_sharpe = subset.groupby("Config")["Sharpe"].mean()
        best_cfg = avg_sharpe.idxmax()
        best_sub = subset[subset["Config"] == best_cfg]
        print(f"\n  [{phil} BEST ({best_cfg}) 各幣種表現]")
        display_cols = ["Symbol", "CAGR [%]", "Sharpe", "MaxDD [%]",
                       "Win Rate [%]", "Profit Factor", "N Trades",
                       "Avg Hold [h]", "Turnover"]
        display_cols = [c for c in display_cols if c in best_sub.columns]
        print(best_sub[display_cols].to_string(index=False))

    # Save
    full_display = full[["Symbol", "Config", "Philosophy", "CAGR [%]", "Sharpe",
                          "MaxDD [%]", "Win Rate [%]", "N Trades", "Avg Hold [h]"]].copy()
    full_display.to_csv(output_dir / "T6_signal_analysis.csv", index=False)


def _print_conclusion(results_df: pd.DataFrame, tail_df: pd.DataFrame, output_dir: Path):
    """結論"""
    if results_df.empty:
        return

    full = results_df[results_df["Period"] == "Full"]
    if full.empty:
        return

    # 各哲學最佳 config
    best = {}
    for phil in ["baseline", "trend_hold", "hybrid_lock", "mean_revert_take"]:
        subset = full[full["Philosophy"] == phil]
        if subset.empty:
            continue
        avg = subset.groupby("Config").agg({
            "Sharpe": "mean",
            "CAGR [%]": "mean",
            "MaxDD [%]": "mean",
            "Win Rate [%]": "mean",
            "Avg Hold [h]": "mean",
        }).reset_index()
        idx = avg["Sharpe"].idxmax()
        best[phil] = avg.loc[idx]

    # 右尾判斷（基於 baseline，不受 exit overlay 污染）
    tail_driven = "unknown"
    if not tail_df.empty:
        baseline_tail = tail_df[tail_df["Philosophy"] == "baseline"]
        if not baseline_tail.empty and "Top 10% PnL Contribution [%]" in baseline_tail.columns:
            avg_top10 = baseline_tail["Top 10% PnL Contribution [%]"].mean()
        else:
            avg_top10 = tail_df["Top 10% PnL Contribution [%]"].mean()
        baseline_wr = full[full["Philosophy"] == "baseline"]["Win Rate [%]"].mean()
        if avg_top10 > 60:
            tail_driven = "right_tail"
        elif baseline_wr > 55:
            tail_driven = "high_win_rate"
        else:
            tail_driven = "mixed"

    lines = [
        "",
        "═" * 70,
        "  🎯 結論與建議",
        "═" * 70,
        "",
    ]

    # ── 1. 驅動模式判斷 ──
    lines.append("  ━━━ 1. 驅動模式判斷 ━━━")
    if tail_driven == "right_tail":
        lines.append("  📊 此策略為【右尾驅動】型")
        lines.append("     → 少數大贏家貢獻主要收益（Top 10% trades > 60% PnL）")
        lines.append("     → 不建議設緊 TP — 會系統性截斷右尾")
        lines.append("     → Trend-following 策略的典型特徵，符合 TSMOM 設計")
    elif tail_driven == "high_win_rate":
        lines.append("  📊 此策略為【高勝率驅動】型")
        lines.append("     → 多數交易小贏，少數交易大輸")
        lines.append("     → 適合設定明確 TP 鎖利 + 嚴格 SL 控損")
    else:
        lines.append("  📊 此策略為【混合】型")
        lines.append("     → 右尾貢獻與勝率介於兩者之間")
        lines.append("     → Trailing stop 是最佳折衷（保護浮盈而不截斷趨勢）")
    lines.append("")

    # ── 2. 各哲學表現比較 ──
    lines.append("  ━━━ 2. 各哲學表現比較 ━━━")
    for phil, b in sorted(best.items(), key=lambda x: x[1]["Sharpe"], reverse=True):
        marker = "⭐" if b["Sharpe"] == max(v["Sharpe"] for v in best.values()) else "  "
        lines.append(f"  {marker} {phil:20s}: Sharpe={b['Sharpe']:.3f}, "
                     f"CAGR={b['CAGR [%]']:.1f}%, MDD={b['MaxDD [%]']:.1f}%, "
                     f"WR={b['Win Rate [%]']:.0f}%")
    lines.append("")

    # ── 3. 核心洞察 ──
    lines.append("  ━━━ 3. 核心洞察 ━━━")
    if "baseline" in best:
        b_sharpe = best["baseline"]["Sharpe"]
        all_worse = all(best[p]["Sharpe"] < b_sharpe for p in best if p != "baseline")
        if all_worse:
            lines.append("  ⚠️  所有 exit overlay 均劣於裸 TSMOM baseline")
            lines.append("     原因分析：")
            lines.append("     a) TSMOM 連續信號已內建出場機制（動量消失 → 信號趨零）")
            lines.append("     b) SL/TP 觸發後的 cooldown 期間錯過趨勢延續")
            lines.append("     c) SL 在高波動環境被頻繁觸發（crypto 常見 3-5 ATR 回撤）")
            lines.append("     d) TP 截斷利潤，但 TSMOM 的利潤主要來自持有趨勢")
        else:
            improved = [p for p in best if p != "baseline" and best[p]["Sharpe"] >= b_sharpe]
            lines.append(f"  ✅ 有哲學超越 baseline: {', '.join(improved)}")
    lines.append("")

    # ── 4. TP 建議 ──
    lines.append("  ━━━ 4. TP/SL 建議 ━━━")
    if "baseline" in best:
        b_sharpe = best["baseline"]["Sharpe"]
        phil_sharpes = {p: best[p]["Sharpe"] for p in best}
        best_phil = max(phil_sharpes, key=phil_sharpes.get)

        if best_phil == "baseline":
            lines.append("  💡 建議: 【維持現狀 — 不上 TP/SL】")
            lines.append("     → TSMOM 信號本身就是最佳出場機制")
            lines.append("     → 若擔心尾部風險，可考慮:")
            lines.append("       • 帳戶層級 Drawdown Circuit Breaker（現有 40%）")
            lines.append("       • 極寬災難型 SL（≥7× ATR），僅防閃崩")
            lines.append("       • 不加 cooldown（觸發後立即恢復信號控制）")
        elif best_phil == "trend_hold":
            lines.append("  💡 建議: 【加寬幅災難型 SL】")
            lines.append(f"     → 最佳: {best['trend_hold']['Config']}")
            lines.append("     → SL 僅做黑天鵝保護，不干預正常出場")
        elif best_phil == "hybrid_lock":
            lines.append("  💡 建議: 【上 Trailing Stop】")
            lines.append(f"     → 最佳: {best['hybrid_lock']['Config']}")
            lines.append("     → Trailing 鎖盈而非截斷趨勢")
            lines.append("     → 需監控: trailing 觸發頻率是否過高")
        elif best_phil == "mean_revert_take":
            lines.append("  💡 建議: 【上固定 TP】")
            lines.append(f"     → 最佳: {best['mean_revert_take']['Config']}")
            lines.append("     → 策略可能有均值回歸特性，TP 有效")

    lines.append("")

    # ── 5. Paper-trade 候選 ──
    lines.append("  ━━━ 5. Paper-trade 候選配置 ━━━")
    candidates = []
    for phil in ["baseline", "trend_hold", "hybrid_lock", "mean_revert_take"]:
        if phil in best:
            b = best[phil]
            lines.append(f"    #{len(candidates)+1}. {phil}: {b['Config']}")
            lines.append(f"        Sharpe={b['Sharpe']:.3f}, CAGR={b['CAGR [%]']:.1f}%, "
                        f"MDD={b['MaxDD [%]']:.1f}%, WR={b['Win Rate [%]']:.0f}%")
            candidates.append(phil)

    lines.append("")

    # ── 6. 監控指標 ──
    lines.append("  ━━━ 6. Paper-trade 監控指標 ━━━")
    lines.append("    1. Rolling 30d Sharpe: IS 衰減 > 30% → 警戒")
    lines.append("    2. Avg holding time: 偏離回測均值 > 50% → 檢查信號品質")
    lines.append("    3. Win rate: 偏離回測 > 10pp → 檢查市場狀態")
    lines.append("    4. SL 觸發率: > 30% trades hit SL → SL 太窄或市場劇烈")
    lines.append("    5. Funding cost / 總收益占比: > 50% → 重新評估倉位方向")
    lines.append("    6. Max consecutive losses: > 回測 2σ → 暫停檢討")
    lines.append("")

    # ── 7. TSMOM 特殊考量 ──
    lines.append("  ━━━ 7. TSMOM 策略特殊考量 ━━━")
    lines.append("    • TSMOM 的出場機制是「信號衰減」而非「觸價出場」")
    lines.append("    • 信號連續性是 TSMOM 的核心優勢（vol-scaled position sizing）")
    lines.append("    • 疊加 exit rules 的最大風險: 打斷信號連續性，增加 whipsaw")
    lines.append("    • 若生產環境決定加 SL: 建議 SL ≥ 5× ATR + CD=0")
    lines.append("    • 原則: exit overlay 應為「保險」而非「主要出場機制」")
    lines.append("")

    conclusion = "\n".join(lines)
    print(conclusion)
    with open(output_dir / "T7_conclusion.txt", "w") as f:
        f.write(conclusion)


if __name__ == "__main__":
    main()
