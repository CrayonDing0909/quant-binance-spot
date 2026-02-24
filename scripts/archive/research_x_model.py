#!/usr/bin/env python3
"""
X-Model Weekend Liquidity Sweep — Research Backtest v8
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ERL→IRL X-Model 策略回測研究腳本。

v8 核心修正:
    - Weekly Profile Management: 'Monday Scalper' → 'Weekly Swing Trader'
    - Breakeven Logic: price +0.5% → SL to Entry (risk-free)
    - Trailing Stop: trail SL at 1.0%~2.0% from peak (lock profits)
    - Hold 4-5 days: capture Wed/Thu expansion (not exit Mon night)
    - Retain v7 HTF Bias + v6 Rejection + v5 Weekend Range

Usage:
    cd /path/to/quant-binance-spot
    source .venv/bin/activate
    PYTHONPATH=src python scripts/research_x_model.py
"""
from __future__ import annotations

import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import vectorbt as vbt

warnings.filterwarnings("ignore")

# ── Ensure src on path ──
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from qtrade.backtest.run_backtest import run_symbol_backtest, validate_backtest_config
from qtrade.strategy.x_model_weekend import (
    compute_daily_levels_utc8,
    is_x_model_window_utc8,
)


# ══════════════════════════════════════════════════════════════
#  Config
# ══════════════════════════════════════════════════════════════

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
INTERVAL = "1h"
DATA_DIR = ROOT / "data" / "binance" / "futures" / "1h"
DATA_DIR_15M = ROOT / "data" / "binance" / "futures" / "15m"
REPORT_DIR = ROOT / "reports" / "research" / "x_model"

BASE = dict(
    confirm_window=4,
    require_fvg=True,
    require_mss=True,
    require_15m=True,
    sl_buffer_pct=0.001,
    tp_mode="stdv",
    stdv_mult=2.0,
    swing_lookback=12,
    mss_lookback=10,
    max_hold_bars=48,
    force_close_window_end=True,
    cooldown_bars=2,
    min_range_pct=0.002,
)

CONFIGS: dict[str, dict] = {
    # === STDV sweep (full ICT: 1H MSS+FVG + 15m MSS+FVG) ===
    "A_full_15":      BASE | {"stdv_mult": 1.5},
    "B_full_175":     BASE | {"stdv_mult": 1.75},
    "C_full_20":      BASE | {"stdv_mult": 2.0},

    # === No 15m requirement (H1 MSS+FVG only) ===
    "D_h1_15":        BASE | {"require_15m": False, "stdv_mult": 1.5},
    "E_h1_20":        BASE | {"require_15m": False, "stdv_mult": 2.0},

    # === No MSS (FVG only, no 15m) ===
    "F_fvg_only":     BASE | {"require_mss": False, "require_15m": False},

    # === Pure SFP (no confirmation) ===
    "G_pure_sfp":     BASE | {"require_fvg": False, "require_mss": False, "require_15m": False},

    # === TP modes ===
    "H_opposing":     BASE | {"tp_mode": "opposing"},
    "I_min_tp":       BASE | {"tp_mode": "min_stdv_opp"},

    # === No force-close ===
    "J_no_close":     BASE | {"force_close_window_end": False},

    # === Wider confirm window ===
    "K_wide_cw":      BASE | {"confirm_window": 6},

    # ════ v4 Experimental Configs ════
    # Trend Filter (200 SMA)
    "L_trend_200":    BASE | {"trend_filter_period": 200, "stdv_mult": 2.0},
    
    # Kill Zone (UTC+8 15:00-19:00 = London Open)
    "M_london_kz":    BASE | {"kill_zone_hours": (15, 19), "stdv_mult": 2.0},

    # Relaxed Confirmation (MSS OR FVG)
    "N_relaxed":      BASE | {"relax_confirmation": True, "stdv_mult": 2.0},

    # v4 Combined (Smart Money Logic)
    "O_v4_smart":     BASE | {
        "trend_filter_period": 200,      # Filter against HTF Trend
        "kill_zone_hours": (15, 19),     # Only trade London Open manipulation
        "relax_confirmation": True,      # Entry on first sign (MSS or FVG)
        "stdv_mult": 2.0,
        "require_15m": False,            # Rely on 1H + Trend
    },

    # ════ v5 True Weekend Range (X-Model Logic) ════
    # P_true_x: Use Weekend Range (Sat+Sun) + Trade on Mon/Tue + London KZ
    "P_true_x":       BASE | {
        "use_weekend_levels": True,      # Key Fix: Weekend High/Low
        "kill_zone_hours": (15, 19),     # London Open
        "relax_confirmation": True,      # Speed
        "stdv_mult": 2.0,
        "require_15m": False,
        "trend_filter_period": 0,        # Let's test raw X-Model first
    },
    
    # Q_true_x_trend: + Trend Filter (EMA? No, simple MA for now)
    "Q_true_x_trend": BASE | {
        "use_weekend_levels": True,
        "kill_zone_hours": (15, 19),
        "relax_confirmation": True,
        "stdv_mult": 2.0,
        "require_15m": False,
        "trend_filter_period": 200,
    },

    # ════ v6 Rejection Quality Filter ════
    # R_reject_kz: Rolling PDH/PDL + London KZ + Rejection Filter
    "R_reject_kz":    BASE | {
        "kill_zone_hours": (15, 19),
        "relax_confirmation": True,
        "stdv_mult": 2.0,
        "require_15m": False,
        "rejection_filter": True,
        "min_wick_ratio": 0.5,
        "max_body_position": 0.4,
    },

    # S_reject_vol: + Volume Filter (1.2x avg)
    "S_reject_vol":   BASE | {
        "kill_zone_hours": (15, 19),
        "relax_confirmation": True,
        "stdv_mult": 2.0,
        "require_15m": False,
        "rejection_filter": True,
        "min_wick_ratio": 0.5,
        "max_body_position": 0.4,
        "min_volume_factor": 1.2,
    },

    # T_wx_reject: Weekend Range + Rejection Filter (True X-Model v6)
    "T_wx_reject":    BASE | {
        "use_weekend_levels": True,
        "kill_zone_hours": (15, 19),
        "relax_confirmation": True,
        "stdv_mult": 2.0,
        "require_15m": False,
        "rejection_filter": True,
        "min_wick_ratio": 0.5,
        "max_body_position": 0.4,
    },

    # U_wx_reject_vol: Weekend Range + Rejection + Volume (Full v6)
    "U_wx_reject_vol": BASE | {
        "use_weekend_levels": True,
        "kill_zone_hours": (15, 19),
        "relax_confirmation": True,
        "stdv_mult": 2.0,
        "require_15m": False,
        "rejection_filter": True,
        "min_wick_ratio": 0.5,
        "max_body_position": 0.4,
        "min_volume_factor": 1.2,
    },

    # V_wx_reject_trend: Weekend Range + Rejection + Trend (v6 + v4)
    "V_wx_reject_trend": BASE | {
        "use_weekend_levels": True,
        "kill_zone_hours": (15, 19),
        "relax_confirmation": True,
        "stdv_mult": 2.0,
        "require_15m": False,
        "trend_filter_period": 200,
        "rejection_filter": True,
        "min_wick_ratio": 0.5,
        "max_body_position": 0.4,
    },

    # ════ v7 HTF Directional Bias (Trend Following Liquidity Raids) ════

    # W_htf_ema20: Minimal — HTF Bias only (Daily EMA 20)
    #   Bull→ only long (sweep Low), Bear→ only short (sweep High)
    "W_htf_ema20":   BASE | {
        "htf_trend_period": 20,
        "relax_confirmation": True,
        "stdv_mult": 2.0,
        "require_15m": False,
    },

    # X_htf_kz: HTF Bias + Kill Zone (London Open)
    "X_htf_kz":      BASE | {
        "htf_trend_period": 20,
        "kill_zone_hours": (15, 19),
        "relax_confirmation": True,
        "stdv_mult": 2.0,
        "require_15m": False,
    },

    # Y_htf_wx: HTF Bias + Weekend Range + Kill Zone
    "Y_htf_wx":      BASE | {
        "htf_trend_period": 20,
        "use_weekend_levels": True,
        "kill_zone_hours": (15, 19),
        "relax_confirmation": True,
        "stdv_mult": 2.0,
        "require_15m": False,
    },

    # Z1_htf_reject: HTF Bias + Weekend Range + KZ + Rejection + Volume
    #   The "Full Stack v7" — Trend + Structure + Quality
    "Z1_htf_reject": BASE | {
        "htf_trend_period": 20,
        "use_weekend_levels": True,
        "kill_zone_hours": (15, 19),
        "relax_confirmation": True,
        "stdv_mult": 2.0,
        "require_15m": False,
        "rejection_filter": True,
        "min_wick_ratio": 0.5,
        "max_body_position": 0.4,
        "min_volume_factor": 1.2,
    },

    # Z2_htf_noclose: HTF Bias + WX + KZ + No Force Close
    #   Test if letting trades run past window improves with trend alignment
    "Z2_htf_noclose": BASE | {
        "htf_trend_period": 20,
        "use_weekend_levels": True,
        "kill_zone_hours": (15, 19),
        "relax_confirmation": True,
        "stdv_mult": 2.0,
        "require_15m": False,
        "force_close_window_end": False,
    },

    # Z3_htf_ema50: Test longer EMA (50 days) for less whipsaw
    "Z3_htf_ema50":  BASE | {
        "htf_trend_period": 50,
        "use_weekend_levels": True,
        "kill_zone_hours": (15, 19),
        "relax_confirmation": True,
        "stdv_mult": 2.0,
        "require_15m": False,
    },

    # ════ v8 Weekly Profile Management (Swing Trader) ════

    # SW1_be_only: Breakeven only (0.5%) — eliminate risk, hold for expansion
    "SW1_be_only":    BASE | {
        "htf_trend_period": 20,
        "kill_zone_hours": (15, 19),
        "relax_confirmation": True,
        "stdv_mult": 2.0,
        "require_15m": False,
        "force_close_window_end": False,
        "max_hold_bars": 96,
        "breakeven_trigger_pct": 0.005,      # 0.5% move → SL to entry
    },

    # SW2_be_trail: Breakeven + Trailing (lock profits during expansion)
    "SW2_be_trail":   BASE | {
        "htf_trend_period": 20,
        "kill_zone_hours": (15, 19),
        "relax_confirmation": True,
        "stdv_mult": 2.0,
        "require_15m": False,
        "force_close_window_end": False,
        "max_hold_bars": 96,
        "breakeven_trigger_pct": 0.005,      # 0.5% → BE
        "trailing_stop_pct": 0.015,          # 1.5% trailing from peak
    },

    # SW3_be_trail_wx: BE + Trail + Weekend Range
    "SW3_be_trail_wx": BASE | {
        "htf_trend_period": 20,
        "use_weekend_levels": True,
        "kill_zone_hours": (15, 19),
        "relax_confirmation": True,
        "stdv_mult": 2.0,
        "require_15m": False,
        "force_close_window_end": False,
        "max_hold_bars": 96,
        "breakeven_trigger_pct": 0.005,
        "trailing_stop_pct": 0.015,
    },

    # SW4_tight_trail: Tighter trailing (1.0%) for quicker profit lock
    "SW4_tight_trail": BASE | {
        "htf_trend_period": 20,
        "kill_zone_hours": (15, 19),
        "relax_confirmation": True,
        "stdv_mult": 2.5,                    # wider TP target
        "require_15m": False,
        "force_close_window_end": False,
        "max_hold_bars": 96,
        "breakeven_trigger_pct": 0.003,      # 0.3% → BE (faster)
        "trailing_stop_pct": 0.010,          # 1.0% trailing
    },

    # SW5_wide_trail: Wider trailing (2.0%) — max room for weekly expansion
    "SW5_wide_trail":  BASE | {
        "htf_trend_period": 20,
        "kill_zone_hours": (15, 19),
        "relax_confirmation": True,
        "stdv_mult": 3.0,                    # even wider TP
        "require_15m": False,
        "force_close_window_end": False,
        "max_hold_bars": 120,                # 5 days
        "breakeven_trigger_pct": 0.005,
        "trailing_stop_pct": 0.020,          # 2.0% trailing
    },

    # SW6_full_stack: HTF + WX + KZ + Reject + Vol + BE + Trail
    #   The "Complete v8 Swing"
    "SW6_full_stack":  BASE | {
        "htf_trend_period": 20,
        "use_weekend_levels": True,
        "kill_zone_hours": (15, 19),
        "relax_confirmation": True,
        "stdv_mult": 2.0,
        "require_15m": False,
        "rejection_filter": True,
        "min_wick_ratio": 0.5,
        "max_body_position": 0.4,
        "min_volume_factor": 1.2,
        "force_close_window_end": False,
        "max_hold_bars": 96,
        "breakeven_trigger_pct": 0.005,
        "trailing_stop_pct": 0.015,
    },

    # SW7_no_tp: Trail only, NO TP target — let trailing stop do all exits
    "SW7_no_tp":       BASE | {
        "htf_trend_period": 20,
        "kill_zone_hours": (15, 19),
        "relax_confirmation": True,
        "stdv_mult": 99.0,                   # effectively disable TP
        "require_15m": False,
        "force_close_window_end": False,
        "max_hold_bars": 120,
        "breakeven_trigger_pct": 0.005,
        "trailing_stop_pct": 0.015,
    },
}

SEGMENTS = {
    "full": ("2024-01-01", "2026-02-28"),
    "IS":   ("2024-01-01", "2025-03-31"),
    "OOS":  ("2025-04-01", "2026-02-28"),
}


# ══════════════════════════════════════════════════════════════
#  Data Loading
# ══════════════════════════════════════════════════════════════

_DATA_CACHE: dict[str, pd.DataFrame] = {}
_15M_CACHE: dict[str, pd.DataFrame | None] = {}


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize DataFrame: ensure DatetimeIndex in UTC."""
    if not isinstance(df.index, pd.DatetimeIndex):
        if "open_time" in df.columns:
            df = df.set_index("open_time")
        elif "timestamp" in df.columns:
            df = df.set_index("timestamp")
        else:
            df.index = pd.to_datetime(df.index)

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    return df.sort_index()


def load_data(symbol: str) -> pd.DataFrame:
    """Load 1H parquet data and cache it."""
    if symbol in _DATA_CACHE:
        return _DATA_CACHE[symbol]

    path = DATA_DIR / f"{symbol}.parquet"
    if not path.exists():
        print(f"  ⚠ {symbol} 1H data not found at {path}")
        return pd.DataFrame()

    df = _normalize_df(pd.read_parquet(path))
    _DATA_CACHE[symbol] = df
    return df


def load_15m_data(symbol: str) -> pd.DataFrame | None:
    """Load 15m parquet data and cache it (returns None if not available)."""
    if symbol in _15M_CACHE:
        return _15M_CACHE[symbol]

    path = DATA_DIR_15M / f"{symbol}.parquet"
    if not path.exists():
        print(f"  ⓘ {symbol} 15m data not found — 15m confirmation disabled")
        _15M_CACHE[symbol] = None
        return None

    df = _normalize_df(pd.read_parquet(path))
    _15M_CACHE[symbol] = df
    return df


# ══════════════════════════════════════════════════════════════
#  Weekend / Sweep Statistics
# ══════════════════════════════════════════════════════════════

def weekend_statistics(df: pd.DataFrame, start: str, end: str) -> dict:
    """
    Compute PDH/PDL sweep statistics during X-Model active windows.
    """
    mask = (df.index >= pd.Timestamp(start, tz="UTC")) & (
        df.index <= pd.Timestamp(end, tz="UTC")
    )
    sub = df.loc[mask].copy()
    if sub.empty:
        return {"weekends": 0, "pdh_sweeps": 0, "pdl_sweeps": 0, "total_sweeps": 0}

    is_active = is_x_model_window_utc8(sub.index)
    high = sub["high"].values
    low = sub["low"].values
    close = sub["close"].values

    pdh, pdl = compute_daily_levels_utc8(sub.index, high, low)

    pdh_sweeps = 0
    pdl_sweeps = 0
    for i in range(len(sub)):
        if not is_active[i] or np.isnan(pdh[i]) or np.isnan(pdl[i]):
            continue
        if high[i] > pdh[i] and close[i] < pdh[i]:
            pdh_sweeps += 1
        if low[i] < pdl[i] and close[i] > pdl[i]:
            pdl_sweeps += 1

    # Count unique weekends (Sat/Sun/Mon UTC+8)
    utc8_times = sub.index + pd.Timedelta(hours=8)
    active_weeks = set()
    for t, a in zip(utc8_times, is_active):
        if a:
            active_weeks.add((t.isocalendar()[0], t.isocalendar()[1]))

    return {
        "weekends": len(active_weeks),
        "pdh_sweeps": pdh_sweeps,
        "pdl_sweeps": pdl_sweeps,
        "total_sweeps": pdh_sweeps + pdl_sweeps,
    }


# ══════════════════════════════════════════════════════════════
#  Single Backtest
# ══════════════════════════════════════════════════════════════

def run_single(
    symbol: str,
    config_name: str,
    params: dict,
    segment: str,
) -> dict | None:
    """Run a single backtest for one symbol/config/segment."""
    data_path = DATA_DIR / f"{symbol}.parquet"
    if not data_path.exists():
        print(f"  ⚠ {symbol} data not found at {data_path}")
        return None

    start, end = SEGMENTS[segment]

    # Inject 15m data if available
    strat_params = params.copy()
    df_15m = load_15m_data(symbol)
    if df_15m is not None:
        strat_params["_df_15m"] = df_15m

    cfg = {
        "strategy_name": "x_model_weekend",
        "strategy_params": strat_params,
        "interval": INTERVAL,
        "market_type": "futures",
        "direction": "both",
        "trade_on": "next_open",
        "position_sizing": {"method": "fixed", "position_pct": 1.0},
        "initial_cash": 10_000,
        "fee_bps": 5.0,
        "slippage_bps": 3.0,
        "start": start,
        "end": end,
    }

    try:
        result = run_symbol_backtest(
            symbol=symbol,
            data_path=data_path,
            cfg=cfg,
            strategy_name="x_model_weekend",
            market_type="futures",
            direction="both",
        )
    except Exception as e:
        print(f"  ✗ {symbol}/{config_name}/{segment}: {e}")
        return None

    if result is None or result.pf is None:
        return None

    pf = result.pf
    df = result.df
    stats = pf.stats()

    # Extract trade info from stats
    n_trades = int(stats.get("Total Trades", 0))
    win_rate = float(stats.get("Win Rate [%]", 0.0))

    # Average holding from positions
    avg_hold_hours = 0.0
    try:
        recs = pf.positions.records_readable
        if len(recs) > 0:
            dur = recs["Duration"]
            avg_td = dur.mean()
            if hasattr(avg_td, "total_seconds"):
                avg_hold_hours = avg_td.total_seconds() / 3600
            elif isinstance(avg_td, (int, float)):
                avg_hold_hours = float(avg_td)
    except Exception:
        pass

    total_return = stats.get("Total Return [%]", 0.0)
    days = (df.index[-1] - df.index[0]).days
    years = max(days / 365.25, 0.01)
    cagr = ((1 + total_return / 100) ** (1 / years) - 1) * 100 if total_return > -100 else -100.0

    # Turnover
    pos_series = result.pos
    turnover = pos_series.diff().abs().sum()

    return {
        "symbol": symbol,
        "config": config_name,
        "segment": segment,
        "CAGR%": round(cagr, 2),
        "Sharpe": round(stats.get("Sharpe Ratio", 0.0), 3),
        "Sortino": round(stats.get("Sortino Ratio", 0.0), 3),
        "MaxDD%": round(stats.get("Max Drawdown [%]", 0.0), 2),
        "Calmar": round(cagr / max(stats.get("Max Drawdown [%]", 100.0), 0.01), 3),
        "WinRate%": round(win_rate, 1),
        "Trades": n_trades,
        "AvgHold_h": round(avg_hold_hours, 1),
        "Turnover": round(turnover, 1),
        "TotalRet%": round(total_return, 2),
        "PF": round(stats.get("Profit Factor", 0.0), 3) if stats.get("Profit Factor") else 0.0,
    }


# ══════════════════════════════════════════════════════════════
#  Trade Analysis
# ══════════════════════════════════════════════════════════════

def trade_analysis(symbol: str, config_name: str, params: dict) -> pd.DataFrame | None:
    """Extract detailed per-trade records for best config."""
    data_path = DATA_DIR / f"{symbol}.parquet"
    if not data_path.exists():
        return None

    start, end = SEGMENTS["full"]

    strat_params = params.copy()
    df_15m = load_15m_data(symbol)
    if df_15m is not None:
        strat_params["_df_15m"] = df_15m

    cfg = {
        "strategy_name": "x_model_weekend",
        "strategy_params": strat_params,
        "interval": INTERVAL,
        "market_type": "futures",
        "direction": "both",
        "trade_on": "next_open",
        "position_sizing": {"method": "fixed", "position_pct": 1.0},
        "initial_cash": 10_000,
        "fee_bps": 5.0,
        "slippage_bps": 3.0,
        "start": start,
        "end": end,
    }

    try:
        result = run_symbol_backtest(
            symbol=symbol,
            data_path=data_path,
            cfg=cfg,
            strategy_name="x_model_weekend",
            market_type="futures",
            direction="both",
        )
    except Exception:
        return None

    if result is None or result.pf is None:
        return None

    pf = result.pf
    try:
        recs = pf.positions.records_readable
        if len(recs) > 0:
            cols = [c for c in recs.columns if c in (
                "Entry Timestamp", "Exit Timestamp", "PnL", "P&L",
                "Return", "Direction", "Duration", "Side",
            )]
            out = recs[cols].copy() if cols else recs.copy()
            return out
    except Exception:
        pass
    return None


# ══════════════════════════════════════════════════════════════
#  Right-tail Analysis
# ══════════════════════════════════════════════════════════════

def right_tail_analysis(trades_df: pd.DataFrame) -> dict:
    """Analyze right-tail concentration of PnL."""
    if trades_df is None or len(trades_df) == 0:
        return {}

    pnl = trades_df["PnL"].values
    total_pnl = np.sum(pnl)
    sorted_pnl = np.sort(pnl)[::-1]  # descending

    n = len(pnl)
    top10_n = max(1, int(n * 0.1))

    if total_pnl <= 0:
        top10_pct = np.nan
    else:
        top10_pct = np.sum(sorted_pnl[:top10_n]) / total_pnl * 100

    # Remove top 1, 3, 5 trades
    rm1_pnl = total_pnl - np.sum(sorted_pnl[:1])
    rm3_pnl = total_pnl - np.sum(sorted_pnl[:min(3, n)])
    rm5_pnl = total_pnl - np.sum(sorted_pnl[:min(5, n)])

    return {
        "Total PnL": round(total_pnl, 2),
        "Top10% Contribution%": round(top10_pct, 1) if not np.isnan(top10_pct) else "N/A",
        "Rm Top1 PnL": round(rm1_pnl, 2),
        "Rm Top3 PnL": round(rm3_pnl, 2),
        "Rm Top5 PnL": round(rm5_pnl, 2),
        "Skew": round(float(pd.Series(pnl).skew()), 3),
        "Kurtosis": round(float(pd.Series(pnl).kurtosis()), 3),
    }


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPORT_DIR / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"═══ X-Model v8 Research: Weekly Profile Management (Swing Trader) ═══")
    print(f"Output: {out_dir}")
    print(f"Symbols: {SYMBOLS}")
    print(f"Configs: {len(CONFIGS)}")
    print(f"Segments: {list(SEGMENTS.keys())}")
    print(f"v8 Core: Breakeven + Trailing Stop → 'Monday Scalper' → 'Weekly Swing'")
    print(f"         BE trigger: 0.3%~0.5% → SL to Entry (risk-free)")
    print(f"         Trailing: 1.0%~2.0% from peak (lock profits)")
    print(f"         Holding: 96-120 bars (4-5 days, capture Wed/Thu expansion)")
    print(f"v7: HTF Bias (EMA 20), v6: Reject+Vol, v5: WX Range, v4: KZ")
    print(f"Kill Zone: London Open (15:00-19:00 UTC+8)")
    print()

    # ── Check 15m data availability ──
    print("─── 15m Data Availability ───")
    for sym in SYMBOLS:
        df_15m = load_15m_data(sym)
        if df_15m is not None:
            print(f"  ✓ {sym}: {len(df_15m)} bars ({df_15m.index[0]} ~ {df_15m.index[-1]})")
        else:
            print(f"  ✗ {sym}: not available")
    print()

    # ── 1. Weekend Statistics ──
    print("─── Weekend / Sweep Statistics ───")
    for sym in SYMBOLS:
        df = load_data(sym)
        if df.empty:
            continue
        for seg_name, (s, e) in SEGMENTS.items():
            stats = weekend_statistics(df, s, e)
            print(
                f"  {sym} [{seg_name}]: {stats['weekends']} weekends, "
                f"{stats['pdh_sweeps']} PDH sweeps, {stats['pdl_sweeps']} PDL sweeps, "
                f"{stats['total_sweeps']} total sweeps"
            )
    print()

    # ── 2. Run all backtests ──
    print("─── Running Backtests ───")
    all_results = []
    total = len(SYMBOLS) * len(CONFIGS) * len(SEGMENTS)
    done = 0

    for sym in SYMBOLS:
        for cfg_name, params in CONFIGS.items():
            for seg_name in SEGMENTS:
                done += 1
                print(f"\r  [{done}/{total}] {sym}/{cfg_name}/{seg_name}", end="", flush=True)
                res = run_single(sym, cfg_name, params, seg_name)
                if res:
                    all_results.append(res)

    print(f"\n  → {len(all_results)} results collected\n")

    if not all_results:
        print("⚠ No results! Check data availability.")
        return

    results_df = pd.DataFrame(all_results)

    # ── 3. Performance Matrix ──
    print("─── Performance Matrix (full segment, avg across symbols) ───")
    full_df = results_df[results_df["segment"] == "full"].copy()
    perf = None
    if len(full_df) > 0:
        perf = (
            full_df.groupby("config")
            .agg({
                "CAGR%": "mean",
                "Sharpe": "mean",
                "Sortino": "mean",
                "MaxDD%": "mean",
                "Calmar": "mean",
                "WinRate%": "mean",
                "Trades": "sum",
                "AvgHold_h": "mean",
                "Turnover": "sum",
                "PF": "mean",
            })
            .sort_values("Sharpe", ascending=False)
        )
        print(perf.to_string())
        perf.to_csv(out_dir / "performance_matrix.csv")
    print()

    # ── 4. Multi-period Consistency ──
    print("─── Multi-period Consistency ───")
    period_df = (
        results_df.groupby(["config", "segment"])
        .agg({
            "CAGR%": "mean",
            "Sharpe": "mean",
            "MaxDD%": "mean",
            "Trades": "sum",
            "WinRate%": "mean",
        })
    )
    period_pivot = period_df.reset_index().pivot_table(
        index="config", columns="segment", values="Sharpe", aggfunc="first"
    )
    print(period_pivot.to_string())
    period_pivot.to_csv(out_dir / "multi_period_sharpe.csv")
    print()

    # ── 5. Confirmation Layer Comparison ──
    print("─── Confirmation Layer Comparison (full, avg across symbols) ───")
    # Full ICT (A_full_15) vs H1 only (D_h1_15) vs FVG only (F_fvg_only) vs Pure SFP (G_pure_sfp)
    layer_configs = ["A_full_15", "D_h1_15", "F_fvg_only", "G_pure_sfp"]
    layer_df = full_df[full_df["config"].isin(layer_configs)]
    if len(layer_df) > 0:
        layer_cmp = layer_df.groupby("config").agg({
            "CAGR%": "mean", "Sharpe": "mean", "WinRate%": "mean",
            "Trades": "sum", "AvgHold_h": "mean", "PF": "mean",
        })
        print(layer_cmp.to_string())
    print()

    # ── 6. STDV Sensitivity ──
    print("─── STDV Sensitivity (full, avg across symbols) ───")
    stdv_configs = ["A_full_15", "B_full_175", "C_full_20"]
    stdv_df = full_df[full_df["config"].isin(stdv_configs)]
    if len(stdv_df) > 0:
        stdv_cmp = stdv_df.groupby("config").agg({
            "CAGR%": "mean", "Sharpe": "mean", "WinRate%": "mean",
            "Trades": "sum", "AvgHold_h": "mean",
        })
        print(stdv_cmp.to_string())
    print()

    # ── 7. TP Mode Comparison ──
    print("─── TP Mode Comparison (full, avg across symbols) ───")
    tp_configs = ["C_full_20", "H_opposing", "I_min_tp"]
    tp_df = full_df[full_df["config"].isin(tp_configs)]
    if len(tp_df) > 0:
        tp_cmp = tp_df.groupby("config").agg({
            "CAGR%": "mean", "Sharpe": "mean", "WinRate%": "mean",
            "Trades": "sum", "AvgHold_h": "mean",
        })
        print(tp_cmp.to_string())
    print()

    # ── 8. v4 Filter Impact Analysis ──
    print("─── v4 Filter Impact (full, avg across symbols) ───")
    v4_configs = ["C_full_20", "L_trend_200", "M_london_kz", "N_relaxed", "O_v4_smart"]
    v4_df = full_df[full_df["config"].isin(v4_configs)]
    if len(v4_df) > 0:
        v4_cmp = v4_df.groupby("config").agg({
            "CAGR%": "mean", "Sharpe": "mean", "Sortino": "mean",
            "WinRate%": "mean", "Trades": "sum", "MaxDD%": "mean",
            "PF": "mean", "AvgHold_h": "mean",
        })
        # Reindex to show in logical order
        v4_order = [c for c in ["C_full_20", "L_trend_200", "M_london_kz", "N_relaxed", "O_v4_smart"]
                     if c in v4_cmp.index]
        v4_cmp = v4_cmp.loc[v4_order]
        print(v4_cmp.to_string())
        v4_cmp.to_csv(out_dir / "v4_filter_comparison.csv")

        print()
        print("  Legend:")
        print("    C_full_20   = v3 Baseline (MSS AND FVG, no trend, no KZ)")
        print("    L_trend_200 = + Trend Filter only (200 SMA)")
        print("    M_london_kz = + Kill Zone only (15:00-19:00 UTC+8)")
        print("    N_relaxed   = + Relaxed Confirm only (MSS OR FVG)")
        print("    O_v4_smart  = Trend + KZ + Relaxed combined")
    print()

    # ── 8b. v4 Per-symbol Breakdown (O_v4_smart) ──
    print("─── O_v4_smart Per-Symbol (full segment) ───")
    v4s_df = full_df[full_df["config"] == "O_v4_smart"]
    if len(v4s_df) > 0:
        for _, row in v4s_df.iterrows():
            print(
                f"  {row['symbol']}: Sharpe={row['Sharpe']:.3f}, "
                f"CAGR={row['CAGR%']:.1f}%, WR={row['WinRate%']:.1f}%, "
                f"Trades={row['Trades']}, MaxDD={row['MaxDD%']:.1f}%, "
                f"PF={row['PF']:.3f}"
            )
    print()

    # ── 9. v5 Weekend Range Comparison ──
    print("─── v5 Weekend Range vs Rolling PDH/PDL (full, avg across symbols) ───")
    v5_configs = ["C_full_20", "M_london_kz", "O_v4_smart", "P_true_x", "Q_true_x_trend"]
    v5_df = full_df[full_df["config"].isin(v5_configs)]
    if len(v5_df) > 0:
        v5_cmp = v5_df.groupby("config").agg({
            "CAGR%": "mean", "Sharpe": "mean", "Sortino": "mean",
            "WinRate%": "mean", "Trades": "sum", "MaxDD%": "mean",
            "PF": "mean", "AvgHold_h": "mean",
        })
        v5_order = [c for c in v5_configs if c in v5_cmp.index]
        v5_cmp = v5_cmp.loc[v5_order]
        print(v5_cmp.to_string())
        v5_cmp.to_csv(out_dir / "v5_weekend_range_comparison.csv")
        print()
        print("  Legend:")
        print("    C_full_20     = v3 Baseline (rolling PDH/PDL, Sat~Mon, MSS AND FVG)")
        print("    M_london_kz   = v4 + Kill Zone only (rolling PDH/PDL, London Open)")
        print("    O_v4_smart    = v4 Combined (rolling PDH/PDL + Trend + KZ + Relaxed)")
        print("    P_true_x      = v5 TRUE X-Model (Static Weekend Range + Mon/Tue + London KZ)")
        print("    Q_true_x_trend= v5 + 200 SMA Trend Filter")
    print()

    # ── 9b. v5 Per-symbol Breakdown ──
    print("─── v5 P_true_x Per-Symbol (full segment) ───")
    px_df = full_df[full_df["config"] == "P_true_x"]
    if len(px_df) > 0:
        for _, row in px_df.iterrows():
            print(
                f"  {row['symbol']}: Sharpe={row['Sharpe']:.3f}, "
                f"CAGR={row['CAGR%']:.1f}%, WR={row['WinRate%']:.1f}%, "
                f"Trades={row['Trades']}, MaxDD={row['MaxDD%']:.1f}%, "
                f"PF={row['PF']:.3f}"
            )
    print()

    print("─── v5 Q_true_x_trend Per-Symbol (full segment) ───")
    qx_df = full_df[full_df["config"] == "Q_true_x_trend"]
    if len(qx_df) > 0:
        for _, row in qx_df.iterrows():
            print(
                f"  {row['symbol']}: Sharpe={row['Sharpe']:.3f}, "
                f"CAGR={row['CAGR%']:.1f}%, WR={row['WinRate%']:.1f}%, "
                f"Trades={row['Trades']}, MaxDD={row['MaxDD%']:.1f}%, "
                f"PF={row['PF']:.3f}"
            )
    print()

    # ── 9c. v5 Multi-period Breakdown ──
    print("─── v5 Multi-period Consistency (P_true_x, Q_true_x_trend) ───")
    v5_period_df = results_df[results_df["config"].isin(["P_true_x", "Q_true_x_trend"])]
    if len(v5_period_df) > 0:
        v5_pivot = v5_period_df.pivot_table(
            index=["config", "symbol"], columns="segment",
            values=["Sharpe", "WinRate%", "Trades", "CAGR%"], aggfunc="first"
        )
        print(v5_pivot.to_string())
    print()

    # ── 10. v6 Rejection Quality Filter Analysis ──
    print("─── v6 Rejection Quality Filter (full, avg across symbols) ───")
    v6_configs = ["M_london_kz", "P_true_x", "R_reject_kz", "S_reject_vol",
                  "T_wx_reject", "U_wx_reject_vol", "V_wx_reject_trend"]
    v6_df = full_df[full_df["config"].isin(v6_configs)]
    if len(v6_df) > 0:
        v6_cmp = v6_df.groupby("config").agg({
            "CAGR%": "mean", "Sharpe": "mean", "Sortino": "mean",
            "WinRate%": "mean", "Trades": "sum", "MaxDD%": "mean",
            "PF": "mean",
        })
        v6_order = [c for c in v6_configs if c in v6_cmp.index]
        v6_cmp = v6_cmp.loc[v6_order]
        print(v6_cmp.to_string())
        v6_cmp.to_csv(out_dir / "v6_rejection_comparison.csv")
        print()
        print("  Legend:")
        print("    M_london_kz   = v4 Baseline (rolling PDH, KZ, no rejection filter)")
        print("    P_true_x      = v5 Weekend Range (no rejection filter)")
        print("    R_reject_kz   = v6 Rolling PDH + KZ + Rejection Filter")
        print("    S_reject_vol  = v6 + Volume Filter (1.2x avg)")
        print("    T_wx_reject   = v6 Weekend Range + Rejection")
        print("    U_wx_reject_vol = v6 Weekend Range + Rejection + Volume")
        print("    V_wx_reject_trend = v6 Weekend Range + Rejection + Trend")
    print()

    # ── 10b. v6 Per-Symbol (key configs) ──
    for cfg_label in ["R_reject_kz", "T_wx_reject", "V_wx_reject_trend"]:
        print(f"─── v6 {cfg_label} Per-Symbol (full segment) ───")
        cfg_df = full_df[full_df["config"] == cfg_label]
        if len(cfg_df) > 0:
            for _, row in cfg_df.iterrows():
                print(
                    f"  {row['symbol']}: Sharpe={row['Sharpe']:.3f}, "
                    f"CAGR={row['CAGR%']:.1f}%, WR={row['WinRate%']:.1f}%, "
                    f"Trades={row['Trades']}, MaxDD={row['MaxDD%']:.1f}%, "
                    f"PF={row['PF']:.3f}"
                )
        print()

    # ── 10c. v6 Multi-period Breakdown ──
    print("─── v6 Multi-period Consistency (rejection configs) ───")
    v6_period_configs = ["R_reject_kz", "T_wx_reject", "V_wx_reject_trend"]
    v6_period_df = results_df[results_df["config"].isin(v6_period_configs)]
    if len(v6_period_df) > 0:
        v6_pivot = v6_period_df.pivot_table(
            index=["config", "symbol"], columns="segment",
            values=["Sharpe", "WinRate%", "Trades", "CAGR%"], aggfunc="first"
        )
        print(v6_pivot.to_string())
    print()

    # ── 11. v7 HTF Directional Bias Analysis ──
    print("═══ v7 HTF Directional Bias: Trend Following Liquidity Raids ═══")
    print()
    v7_configs = ["M_london_kz", "P_true_x", "U_wx_reject_vol",
                  "W_htf_ema20", "X_htf_kz", "Y_htf_wx",
                  "Z1_htf_reject", "Z2_htf_noclose", "Z3_htf_ema50"]
    v7_df = full_df[full_df["config"].isin(v7_configs)]
    if len(v7_df) > 0:
        v7_cmp = v7_df.groupby("config").agg({
            "CAGR%": "mean", "Sharpe": "mean", "Sortino": "mean",
            "WinRate%": "mean", "Trades": "sum", "MaxDD%": "mean",
            "PF": "mean",
        })
        v7_order = [c for c in v7_configs if c in v7_cmp.index]
        v7_cmp = v7_cmp.loc[v7_order]
        print("─── v7 HTF Bias Comparison (full, avg across symbols) ───")
        print(v7_cmp.to_string())
        v7_cmp.to_csv(out_dir / "v7_htf_bias_comparison.csv")
        print()
        print("  Legend:")
        print("    M_london_kz     = v4 Baseline (market neutral, KZ)")
        print("    P_true_x        = v5 WX Range (market neutral)")
        print("    U_wx_reject_vol = v6 WX + Reject + Volume (market neutral)")
        print("    W_htf_ema20     = v7 HTF Bias only (Daily EMA 20)")
        print("    X_htf_kz        = v7 HTF + Kill Zone")
        print("    Y_htf_wx        = v7 HTF + Weekend Range + KZ")
        print("    Z1_htf_reject   = v7 Full Stack (HTF + WX + KZ + Reject + Vol)")
        print("    Z2_htf_noclose  = v7 HTF + WX + KZ + No Force Close")
        print("    Z3_htf_ema50    = v7 HTF (EMA 50) + WX + KZ")
    print()

    # ── 11b. v7 Per-Symbol Breakdown (key configs) ──
    for cfg_label in ["W_htf_ema20", "Y_htf_wx", "Z1_htf_reject", "Z2_htf_noclose"]:
        print(f"─── v7 {cfg_label} Per-Symbol (full segment) ───")
        cfg_df = full_df[full_df["config"] == cfg_label]
        if len(cfg_df) > 0:
            for _, row in cfg_df.iterrows():
                print(
                    f"  {row['symbol']}: Sharpe={row['Sharpe']:.3f}, "
                    f"CAGR={row['CAGR%']:.1f}%, WR={row['WinRate%']:.1f}%, "
                    f"Trades={row['Trades']}, MaxDD={row['MaxDD%']:.1f}%, "
                    f"PF={row['PF']:.3f}"
                )
        print()

    # ── 11c. v7 Multi-period Consistency ──
    print("─── v7 Multi-period Consistency (HTF bias configs) ───")
    v7_period_configs = ["W_htf_ema20", "X_htf_kz", "Y_htf_wx",
                         "Z1_htf_reject", "Z2_htf_noclose"]
    v7_period_df = results_df[results_df["config"].isin(v7_period_configs)]
    if len(v7_period_df) > 0:
        v7_pivot = v7_period_df.pivot_table(
            index=["config", "symbol"], columns="segment",
            values=["Sharpe", "WinRate%", "Trades", "CAGR%"], aggfunc="first"
        )
        print(v7_pivot.to_string())
    print()

    # ── 12. v8 Weekly Profile Management (Swing Trader) ──
    print("═══ v8 Weekly Profile Management: Breakeven + Trailing Stop ═══")
    print()
    v8_configs = ["X_htf_kz", "Z2_htf_noclose",
                  "SW1_be_only", "SW2_be_trail", "SW3_be_trail_wx",
                  "SW4_tight_trail", "SW5_wide_trail",
                  "SW6_full_stack", "SW7_no_tp"]
    v8_df = full_df[full_df["config"].isin(v8_configs)]
    if len(v8_df) > 0:
        v8_cmp = v8_df.groupby("config").agg({
            "CAGR%": "mean", "Sharpe": "mean", "Sortino": "mean",
            "WinRate%": "mean", "Trades": "sum", "MaxDD%": "mean",
            "PF": "mean", "AvgHold_h": "mean",
        })
        v8_order = [c for c in v8_configs if c in v8_cmp.index]
        v8_cmp = v8_cmp.loc[v8_order]
        print("─── v8 Swing Comparison (full, avg across symbols) ───")
        print(v8_cmp.to_string())
        v8_cmp.to_csv(out_dir / "v8_swing_comparison.csv")
        print()
        print("  Legend:")
        print("    X_htf_kz        = v7 HTF + KZ (force close ON, no BE/trail)")
        print("    Z2_htf_noclose  = v7 HTF + KZ (force close OFF, no BE/trail)")
        print("    SW1_be_only     = v8 HTF + KZ + BE 0.5% (no trail)")
        print("    SW2_be_trail    = v8 HTF + KZ + BE 0.5% + Trail 1.5%")
        print("    SW3_be_trail_wx = v8 HTF + WX + KZ + BE + Trail")
        print("    SW4_tight_trail = v8 BE 0.3% + Trail 1.0% (tight)")
        print("    SW5_wide_trail  = v8 BE 0.5% + Trail 2.0% (wide, 5d hold)")
        print("    SW6_full_stack  = v8 Complete (HTF+WX+KZ+Reject+Vol+BE+Trail)")
        print("    SW7_no_tp       = v8 Trail-only exit (no TP target)")
    print()

    # ── 12b. v8 Per-Symbol Breakdown ──
    for cfg_label in ["SW1_be_only", "SW2_be_trail", "SW5_wide_trail", "SW7_no_tp"]:
        print(f"─── v8 {cfg_label} Per-Symbol (full segment) ───")
        cfg_df = full_df[full_df["config"] == cfg_label]
        if len(cfg_df) > 0:
            for _, row in cfg_df.iterrows():
                print(
                    f"  {row['symbol']}: Sharpe={row['Sharpe']:.3f}, "
                    f"CAGR={row['CAGR%']:.1f}%, WR={row['WinRate%']:.1f}%, "
                    f"Trades={row['Trades']}, MaxDD={row['MaxDD%']:.1f}%, "
                    f"PF={row['PF']:.3f}, AvgHold={row['AvgHold_h']:.0f}h"
                )
        print()

    # ── 12c. v8 Multi-period Consistency ──
    print("─── v8 Multi-period Consistency (swing configs) ───")
    v8_period_configs = ["SW1_be_only", "SW2_be_trail", "SW3_be_trail_wx",
                         "SW5_wide_trail", "SW7_no_tp"]
    v8_period_df = results_df[results_df["config"].isin(v8_period_configs)]
    if len(v8_period_df) > 0:
        v8_pivot = v8_period_df.pivot_table(
            index=["config", "symbol"], columns="segment",
            values=["Sharpe", "WinRate%", "Trades", "CAGR%"], aggfunc="first"
        )
        print(v8_pivot.to_string())
    print()

    # ── 13. Per-symbol detail ──
    results_df.to_csv(out_dir / "per_symbol_detail.csv", index=False)

    # ── 14. Trade Analysis (best overall + best v8 config) ──
    if perf is not None and len(perf) > 0:
        best_cfg = perf.index[0]
        best_params = CONFIGS[best_cfg]
        print(f"─── Trade Analysis (best config: {best_cfg}) ───")

        all_trade_dfs = []
        for sym in SYMBOLS:
            tdf = trade_analysis(sym, best_cfg, best_params)
            if tdf is not None and len(tdf) > 0:
                tdf["symbol"] = sym
                all_trade_dfs.append(tdf)

        if all_trade_dfs:
            trades_all = pd.concat(all_trade_dfs, ignore_index=True)
            trades_all.to_csv(out_dir / "trade_analysis.csv", index=False)
            print(f"  Total trades: {len(trades_all)}")

            pnl_col = "PnL" if "PnL" in trades_all.columns else "P&L" if "P&L" in trades_all.columns else None
            dir_col = "Direction" if "Direction" in trades_all.columns else "Side" if "Side" in trades_all.columns else None

            if dir_col and pnl_col:
                for d in trades_all[dir_col].unique():
                    sub = trades_all[trades_all[dir_col] == d]
                    wr = (sub[pnl_col] > 0).mean() * 100
                    avg_pnl = sub[pnl_col].mean()
                    print(f"  {d}: {len(sub)} trades, WR={wr:.1f}%, AvgPnL={avg_pnl:.4f}")

            if pnl_col and pnl_col != "PnL":
                trades_all = trades_all.rename(columns={pnl_col: "PnL"})
            rt = right_tail_analysis(trades_all)
            if rt:
                print(f"  Right-tail: {rt}")
        else:
            print("  No trades to analyze")
        print()

    # ── 15. Summary ──
    summary_lines = [
        f"X-Model v8 Research Summary — Weekly Profile Management",
        f"{'='*60}",
        f"Date: {ts}",
        f"Strategy: x_model_weekend v8 (ERL→IRL + HTF Bias + BE/Trail)",
        f"Symbols: {SYMBOLS}",
        f"Timeframe: {INTERVAL} (+ Daily EMA for HTF Bias)",
        f"v8 Core: Breakeven + Trailing → Weekly Swing Trader",
        f"v7: HTF Bias, v6: Reject+Vol, v5: WX Range, v4: KZ",
        f"Configs tested: {len(CONFIGS)}",
        "",
    ]

    if perf is not None and len(perf) > 0:
        summary_lines.append("Performance Ranking (full period, avg across symbols):")
        summary_lines.append(perf.to_string())
        summary_lines.append("")

        summary_lines.append("Multi-period Sharpe:")
        summary_lines.append(period_pivot.to_string())
        summary_lines.append("")

        # Verdict
        best_sharpe = perf["Sharpe"].iloc[0]
        best_cfg_name = perf.index[0]
        best_wr = perf["WinRate%"].iloc[0]
        best_cagr = perf["CAGR%"].iloc[0]

        summary_lines.append("─── Verdict ───")
        if best_sharpe > 0:
            summary_lines.append(
                f"✅ Best config: {best_cfg_name} "
                f"(Sharpe={best_sharpe:.3f}, CAGR={best_cagr:.1f}%, WR={best_wr:.1f}%)"
            )
            if best_wr > 55:
                summary_lines.append(
                    "  Strategy shows HIGH WIN RATE — suitable for weekly swing complement to TSMOM."
                )
            elif best_sharpe > 0.5:
                summary_lines.append("  Strategy shows positive alpha — worth iterating further.")
            else:
                summary_lines.append("  Strategy shows marginal alpha — needs further refinement.")
        else:
            summary_lines.append(
                f"❌ All configs negative Sharpe. Best: {best_cfg_name} ({best_sharpe:.3f})"
            )

        # v8 Weekly Profile Impact
        summary_lines.append("")
        summary_lines.append("─── v8 Weekly Profile Impact (Swing vs Scalp) ───")
        v8_compare = [
            ("X_htf_kz",        "v7 Scalp (force close, no BE)"),
            ("Z2_htf_noclose",  "v7 Hold (no force close, no BE)"),
            ("SW1_be_only",     "v8 BE only (risk-free hold)"),
            ("SW2_be_trail",    "v8 BE + Trail 1.5%"),
            ("SW4_tight_trail", "v8 BE + Trail 1.0% (tight)"),
            ("SW5_wide_trail",  "v8 BE + Trail 2.0% (wide)"),
            ("SW7_no_tp",       "v8 Trail-only (no TP)"),
        ]
        for lbl, desc in v8_compare:
            if lbl in perf.index:
                s = perf.loc[lbl, "Sharpe"]
                wr = perf.loc[lbl, "WinRate%"]
                tr = int(perf.loc[lbl, "Trades"])
                cagr = perf.loc[lbl, "CAGR%"]
                mdd = perf.loc[lbl, "MaxDD%"]
                summary_lines.append(
                    f"  {lbl:20s} ({desc:30s}): Sharpe={s:+.3f}, "
                    f"CAGR={cagr:+.1f}%, WR={wr:.1f}%, Trades={tr}, MaxDD={mdd:.1f}%"
                )

        # Key comparison: best v8 swing vs v7 scalp
        best_v8_cfg = None
        best_v8_sharpe = -999.0
        for cfg_label in ["SW1_be_only", "SW2_be_trail", "SW3_be_trail_wx",
                          "SW4_tight_trail", "SW5_wide_trail",
                          "SW6_full_stack", "SW7_no_tp"]:
            if cfg_label in perf.index:
                s = perf.loc[cfg_label, "Sharpe"]
                if s > best_v8_sharpe:
                    best_v8_sharpe = s
                    best_v8_cfg = cfg_label

        v7_scalp = perf.loc["X_htf_kz", "Sharpe"] if "X_htf_kz" in perf.index else -999
        summary_lines.append("")
        if best_v8_cfg and best_v8_sharpe > v7_scalp:
            summary_lines.append(
                f"  ✅ Swing ({best_v8_cfg}: {best_v8_sharpe:.3f}) beats "
                f"Scalp (X_htf_kz: {v7_scalp:.3f})"
            )
            summary_lines.append(
                f"  → Weekly Profile Management is a meaningful improvement"
            )
            summary_lines.append(
                f"  → 'Monday accumulation → Wed/Thu distribution' captured"
            )
        elif best_v8_cfg:
            summary_lines.append(
                f"  ❌ Swing ({best_v8_cfg}: {best_v8_sharpe:.3f}) did NOT beat "
                f"Scalp (X_htf_kz: {v7_scalp:.3f})"
            )
        else:
            summary_lines.append("  ⚠ No v8 configs produced results")

        # IS vs OOS stability
        if "IS" in period_pivot.columns and "OOS" in period_pivot.columns:
            is_sharpe = period_pivot["IS"].mean()
            oos_sharpe = period_pivot["OOS"].mean()
            summary_lines.append("")
            summary_lines.append(
                f"IS/OOS stability: IS avg Sharpe={is_sharpe:.3f}, OOS avg Sharpe={oos_sharpe:.3f}"
            )

            # v8 IS/OOS for key swing configs
            for cfg_key in ["SW2_be_trail", "SW5_wide_trail", "SW7_no_tp"]:
                v8_rows = results_df[results_df["config"] == cfg_key]
                if len(v8_rows) > 0:
                    for seg in ["IS", "OOS"]:
                        seg_rows = v8_rows[v8_rows["segment"] == seg]
                        if len(seg_rows) > 0:
                            avg_s = seg_rows["Sharpe"].mean()
                            summary_lines.append(f"  {cfg_key} {seg}: Sharpe={avg_s:.3f}")

            if is_sharpe > 0 and oos_sharpe > 0:
                summary_lines.append("  ✅ Both positive — signal appears robust")
            elif is_sharpe > 0 > oos_sharpe:
                summary_lines.append("  ⚠ IS positive but OOS negative — possible overfitting")
            else:
                summary_lines.append("  ❌ Both negative — strategy needs fundamental redesign")

    summary_text = "\n".join(summary_lines)
    print(summary_text)
    (out_dir / "summary.txt").write_text(summary_text, encoding="utf-8")

    print(f"\n✅ All output saved to: {out_dir}")


if __name__ == "__main__":
    main()
