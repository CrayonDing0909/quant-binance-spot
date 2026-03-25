#!/usr/bin/env python3
"""
Funding Carry Strategy Evaluation

1. Run funding_carry backtest on BTC/ETH/SOL
2. Run TSMOM (prod) backtest on same symbols
3. Compare: SR, MDD, return correlation, daily PnL correlation
4. Simulate a blended portfolio (TSMOM 70% + Carry 30%)
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from qtrade.config import load_config
from qtrade.backtest.run_backtest import run_symbol_backtest
from qtrade.data.storage import load_klines


BASE_DIR = Path(__file__).resolve().parent.parent.parent
CARRY_CONFIG = BASE_DIR / "config" / "futures_funding_carry.yaml"
TSMOM_CONFIG = BASE_DIR / "config" / "prod_candidate_simplified.yaml"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def run_strategy(config_path: Path, symbols: list[str], label: str) -> dict:
    """Run backtest for all symbols, return {symbol: BacktestResult}."""
    cfg = load_config(config_path)

    # Load reference df for regime gate (TSMOM only)
    ref_df = None
    rg = getattr(cfg, '_regime_gate_cfg', None)
    if rg and rg.get("enabled", False):
        ref_sym = rg.get("reference_symbol", "BTCUSDT")
        mt = cfg.market_type_str
        ref_path = cfg.data_dir / "binance" / mt / cfg.market.interval / f"{ref_sym}.parquet"
        if ref_path.exists():
            ref_df = load_klines(ref_path)

    results = {}
    for symbol in symbols:
        mt = cfg.market_type_str
        data_path = cfg.data_dir / "binance" / mt / cfg.market.interval / f"{symbol}.parquet"
        if not data_path.exists():
            print(f"  ⚠️ {label} {symbol}: data not found")
            continue

        bt_cfg = cfg.to_backtest_dict(symbol)
        if ref_df is not None:
            bt_cfg["_regime_gate_ref_df"] = ref_df

        try:
            result = run_symbol_backtest(symbol=symbol, data_path=data_path, cfg=bt_cfg)
            results[symbol] = result
            print(f"  ✅ {label} {symbol}: done")
        except Exception as e:
            print(f"  ❌ {label} {symbol}: {e}")

    return results


def extract_returns(results: dict) -> pd.DataFrame:
    """Extract hourly returns from results."""
    rets = {}
    for symbol, res in results.items():
        if res.pf is not None:
            eq = res.pf.value()
            rets[symbol] = eq.pct_change().fillna(0)
    return pd.DataFrame(rets)


def portfolio_returns(rets_df: pd.DataFrame, weights: dict | None = None) -> pd.Series:
    """Compute equal-weight or custom-weight portfolio returns."""
    if weights is None:
        weights = {c: 1.0 / len(rets_df.columns) for c in rets_df.columns}
    port = sum(rets_df[c] * weights.get(c, 0) for c in rets_df.columns)
    return port


def stats_from_returns(rets: pd.Series, label: str = "") -> dict:
    """Compute key stats from return series."""
    cumret = (1 + rets).cumprod()
    total_ret = float(cumret.iloc[-1] - 1)
    n_years = len(rets) / (365.25 * 24)
    cagr = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
    sr = float(rets.mean() / rets.std() * np.sqrt(8760)) if rets.std() > 0 else 0
    peak = cumret.cummax()
    dd = (cumret - peak) / peak
    mdd = float(dd.min())
    return {
        "Strategy": label,
        "SR": round(sr, 2),
        "CAGR": f"{cagr:.1%}",
        "MDD": f"{mdd:.1%}",
        "Total Return": f"{total_ret:.1%}",
    }


def main():
    print("=" * 60)
    print("FUNDING CARRY EVALUATION")
    print("=" * 60)

    # Run both strategies
    print("\n📊 Running Funding Carry...")
    carry_results = run_strategy(CARRY_CONFIG, SYMBOLS, "CARRY")

    print("\n📊 Running TSMOM (prod)...")
    tsmom_results = run_strategy(TSMOM_CONFIG, SYMBOLS, "TSMOM")

    # Extract returns
    carry_rets = extract_returns(carry_results)
    tsmom_rets = extract_returns(tsmom_results)

    if carry_rets.empty or tsmom_rets.empty:
        print("❌ Not enough data to compare")
        return

    # Align on common index
    common_idx = carry_rets.index.intersection(tsmom_rets.index)
    carry_rets = carry_rets.loc[common_idx]
    tsmom_rets = tsmom_rets.loc[common_idx]

    # Portfolio returns
    carry_port = portfolio_returns(carry_rets)
    tsmom_port = portfolio_returns(tsmom_rets)

    # Blend: 70% TSMOM + 30% Carry
    blend_port = 0.7 * tsmom_port + 0.3 * carry_port

    # Stats
    print("\n" + "=" * 60)
    print("STRATEGY COMPARISON")
    print("=" * 60)
    stats = [
        stats_from_returns(carry_port, "Funding Carry"),
        stats_from_returns(tsmom_port, "TSMOM (prod)"),
        stats_from_returns(blend_port, "Blend 70/30"),
    ]
    df = pd.DataFrame(stats).set_index("Strategy")
    print(df.to_string())

    # Correlation analysis
    print("\n\n📊 CORRELATION ANALYSIS")
    print("-" * 40)

    # Daily returns correlation
    carry_daily = carry_port.resample("D").sum()
    tsmom_daily = tsmom_port.resample("D").sum()
    common_daily = carry_daily.index.intersection(tsmom_daily.index)
    if len(common_daily) > 30:
        corr = carry_daily.loc[common_daily].corr(tsmom_daily.loc[common_daily])
        print(f"Daily return correlation: {corr:.3f}")
    else:
        print("Not enough daily data for correlation")

    # Per-symbol correlation
    print("\nPer-symbol daily PnL correlation (TSMOM vs Carry):")
    for sym in SYMBOLS:
        if sym in carry_rets.columns and sym in tsmom_rets.columns:
            c_daily = carry_rets[sym].resample("D").sum()
            t_daily = tsmom_rets[sym].resample("D").sum()
            common = c_daily.index.intersection(t_daily.index)
            if len(common) > 30:
                corr = c_daily.loc[common].corr(t_daily.loc[common])
                print(f"  {sym}: {corr:.3f}")

    # Position correlation
    print("\nPer-symbol position correlation:")
    for sym in SYMBOLS:
        if sym in carry_results and sym in tsmom_results:
            c_pos = carry_results[sym].pos
            t_pos = tsmom_results[sym].pos
            if c_pos is not None and t_pos is not None:
                common = c_pos.index.intersection(t_pos.index)
                if len(common) > 100:
                    corr = c_pos.loc[common].corr(t_pos.loc[common])
                    print(f"  {sym}: {corr:.3f}")

    # Time in market
    print("\n\n📊 TIME IN MARKET")
    print("-" * 40)
    for label, results in [("Carry", carry_results), ("TSMOM", tsmom_results)]:
        for sym, res in results.items():
            if res.pos is not None:
                tim = (res.pos.abs() > 0.01).mean()
                avg_pos = res.pos.abs().mean()
                print(f"  {label} {sym}: TIM={tim:.1%}, avg |pos|={avg_pos:.3f}")


if __name__ == "__main__":
    main()
