#!/usr/bin/env python3
"""
Evaluate KEEP_BASELINE regime indicators as STANDALONE SIGNAL GENERATORS.

Previous research tested these as filters on TSMOM (stacking → over-filter).
New approach: convert regime indicators into directional position signals.

Strategies tested:
  A: On-chain (TVL/SC momentum) → positive momentum = long, negative = flat
  B: Macro (GLD/VIX composite) → risk-on = long, risk-off = flat
  C: Combined A+B ensemble

For each: compute SR, MDD, correlation with TSMOM, blend potential.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from qtrade.data.storage import load_klines

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"


# ═══════════════════════════════════════════════════════════
#  Data Loaders (causal, with shift(1) for daily→hourly)
# ═══════════════════════════════════════════════════════════

def load_onchain_signal(kline_index: pd.DatetimeIndex) -> pd.Series:
    """Load TVL/SC ratio momentum as directional signal."""
    tvl_path = DATA_DIR / "onchain" / "defillama" / "tvl_total.parquet"
    sc_path = DATA_DIR / "onchain" / "defillama" / "stablecoin_mcap.parquet"

    if not tvl_path.exists() or not sc_path.exists():
        print(f"  ⚠️ On-chain data not found: {tvl_path}, {sc_path}")
        return None

    tvl = pd.read_parquet(tvl_path)
    sc = pd.read_parquet(sc_path)

    # Get total TVL and stablecoin mcap
    if "totalLiquidityUSD" in tvl.columns:
        tvl_series = tvl["totalLiquidityUSD"]
    elif "tvl" in tvl.columns:
        tvl_series = tvl["tvl"]
    else:
        tvl_series = tvl.iloc[:, 0]

    if "totalCirculatingPeggedUSD" in sc.columns:
        sc_series = sc["totalCirculatingPeggedUSD"]
    elif "mcap" in sc.columns:
        sc_series = sc["mcap"]
    else:
        sc_series = sc.iloc[:, 0]

    # Align dates
    common = tvl_series.index.intersection(sc_series.index)
    if len(common) < 60:
        print(f"  ⚠️ On-chain: insufficient common dates ({len(common)})")
        return None

    ratio = tvl_series.loc[common] / sc_series.loc[common].replace(0, np.nan)
    mom_30d = ratio.pct_change(30)

    # Causal: shift 1 day, then ffill to 1h
    mom_shifted = mom_30d.shift(1)
    # Ensure timezone compatibility
    if mom_shifted.index.tz is None and kline_index.tz is not None:
        mom_shifted.index = mom_shifted.index.tz_localize(kline_index.tz)
    elif mom_shifted.index.tz is not None and kline_index.tz is None:
        mom_shifted.index = mom_shifted.index.tz_localize(None)
    signal = mom_shifted.reindex(kline_index, method="ffill")

    return signal


def load_macro_signal(kline_index: pd.DatetimeIndex) -> pd.Series:
    """Load GLD/VIX composite as directional signal."""
    gld_path = DATA_DIR / "macro" / "gld_daily.parquet"
    vix_path = DATA_DIR / "macro" / "vix_daily.parquet"

    if not gld_path.exists() or not vix_path.exists():
        print(f"  ⚠️ Macro data not found: {gld_path}, {vix_path}")
        return None

    gld = pd.read_parquet(gld_path)
    vix = pd.read_parquet(vix_path)

    gld_close = gld["Close"] if "Close" in gld.columns else gld.iloc[:, 0]
    vix_close = vix["Close"] if "Close" in vix.columns else vix.iloc[:, 0]

    # GLD 60d momentum, VIX 30d momentum
    gld_mom = gld_close.pct_change(60)
    vix_mom = vix_close.pct_change(30)

    # Z-score over 252 days
    common = gld_mom.index.intersection(vix_mom.index)
    gld_z = (gld_mom.loc[common] - gld_mom.loc[common].rolling(252).mean()) / gld_mom.loc[common].rolling(252).std()
    vix_z = (vix_mom.loc[common] - vix_mom.loc[common].rolling(252).mean()) / vix_mom.loc[common].rolling(252).std()

    # Combined: -GLD + VIX (GLD up = risk-off, VIX up from bottom = recovery)
    combined = (-gld_z + vix_z) / 2

    # Causal: shift 1 day, then ffill to 1h
    combined_shifted = combined.shift(1)
    # Ensure timezone compatibility
    if combined_shifted.index.tz is None and kline_index.tz is not None:
        combined_shifted.index = combined_shifted.index.tz_localize(kline_index.tz)
    elif combined_shifted.index.tz is not None and kline_index.tz is None:
        combined_shifted.index = combined_shifted.index.tz_localize(None)
    signal = combined_shifted.reindex(kline_index, method="ffill")

    return signal


# ═══════════════════════════════════════════════════════════
#  Signal → Position Converter
# ═══════════════════════════════════════════════════════════

def signal_to_position(
    signal: pd.Series,
    lookback: int = 720,
    long_threshold_pctile: float = 0.5,
    short_threshold_pctile: float = 0.3,
    rebalance_hours: int = 24,
) -> pd.Series:
    """
    Convert a raw indicator signal to a position using percentile ranking.

    - pctrank > long_threshold → long (+1)
    - pctrank < short_threshold → short (-1) or flat (0)
    - between → flat (0)
    - Rebalance every N hours to keep it low frequency
    """
    pctrank = signal.rolling(lookback, min_periods=max(lookback // 4, 30)).rank(pct=True)

    n = len(signal)
    pos = np.zeros(n, dtype=np.float64)
    current = 0.0

    for i in range(n):
        if np.isnan(pctrank.iloc[i]):
            pos[i] = current
            continue

        if i % rebalance_hours == 0:
            if pctrank.iloc[i] > long_threshold_pctile:
                current = 1.0
            elif pctrank.iloc[i] < short_threshold_pctile:
                current = -1.0
            else:
                current = 0.0

        pos[i] = current

    return pd.Series(pos, index=signal.index)


# ═══════════════════════════════════════════════════════════
#  Backtest Helper
# ═══════════════════════════════════════════════════════════

def backtest_position(
    pos: pd.Series,
    close: pd.Series,
    fee_bps: float = 5,
    label: str = "",
) -> dict:
    """Simple backtest: pos * returns - fees on trades."""
    returns = close.pct_change().fillna(0)

    # Shift position by 1 for signal delay (trade on next open ≈ next bar)
    pos_delayed = pos.shift(1).fillna(0)

    # Strategy returns
    strat_ret = pos_delayed * returns

    # Fee calculation (on position changes)
    trades = pos_delayed.diff().abs().fillna(0)
    fee_cost = trades * (fee_bps / 10000)
    strat_ret -= fee_cost

    cumret = (1 + strat_ret).cumprod()
    total_ret = float(cumret.iloc[-1] - 1)
    n_years = len(strat_ret) / (365.25 * 24)
    cagr = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
    sr = float(strat_ret.mean() / strat_ret.std() * np.sqrt(8760)) if strat_ret.std() > 0 else 0
    peak = cumret.cummax()
    mdd = float(((cumret - peak) / peak).min())
    tim = float((pos_delayed.abs() > 0.01).mean())
    turnover = float(trades.sum() / len(trades))

    return {
        "label": label,
        "SR": round(sr, 2),
        "CAGR": f"{cagr:.1%}",
        "MDD": f"{mdd:.1%}",
        "Return": f"{total_ret:.1%}",
        "TIM": f"{tim:.1%}",
        "Turnover/bar": f"{turnover:.5f}",
        "returns": strat_ret,  # for correlation
    }


def main():
    print("=" * 70)
    print("KEEP_BASELINE → STANDALONE SIGNAL GENERATORS")
    print("=" * 70)

    # Load BTC 1h as reference
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    results_by_strategy = {}

    for sym in symbols:
        kline_path = DATA_DIR / "binance" / "futures" / "1h" / f"{sym}.parquet"
        if not kline_path.exists():
            print(f"⚠️ {sym} kline not found")
            continue

        df = load_klines(kline_path)
        close = df["close"]
        kline_idx = df.index

        print(f"\n{'─' * 50}")
        print(f"  {sym} ({len(df)} bars, {df.index[0].date()} → {df.index[-1].date()})")
        print(f"{'─' * 50}")

        # ── A: On-chain Signal (long/short) ──
        onchain_raw = load_onchain_signal(kline_idx)
        if onchain_raw is not None:
            onchain_pos = signal_to_position(
                onchain_raw, lookback=720,
                long_threshold_pctile=0.5, short_threshold_pctile=0.3,
                rebalance_hours=24,
            )
            res_a = backtest_position(onchain_pos, close, label=f"A: On-chain ({sym})")
            print(f"  A On-chain L/S:  SR={res_a['SR']}, CAGR={res_a['CAGR']}, MDD={res_a['MDD']}, TIM={res_a['TIM']}")
            results_by_strategy.setdefault("A_onchain", []).append(res_a)

            # Long-only variant: positive momentum = long, else flat
            onchain_lo = signal_to_position(
                onchain_raw, lookback=720,
                long_threshold_pctile=0.4, short_threshold_pctile=-1.0,  # never short
                rebalance_hours=24,
            )
            onchain_lo = onchain_lo.clip(0, 1)
            res_a2 = backtest_position(onchain_lo, close, label=f"A2: On-chain LO ({sym})")
            print(f"  A2 On-chain LO:  SR={res_a2['SR']}, CAGR={res_a2['CAGR']}, MDD={res_a2['MDD']}, TIM={res_a2['TIM']}")
            results_by_strategy.setdefault("A2_onchain_LO", []).append(res_a2)

        # ── B: Macro Signal (long/short) ──
        macro_raw = load_macro_signal(kline_idx)
        if macro_raw is not None:
            macro_pos = signal_to_position(
                macro_raw, lookback=720,
                long_threshold_pctile=0.5, short_threshold_pctile=0.3,
                rebalance_hours=24,
            )
            res_b = backtest_position(macro_pos, close, label=f"B: Macro ({sym})")
            print(f"  B Macro L/S:     SR={res_b['SR']}, CAGR={res_b['CAGR']}, MDD={res_b['MDD']}, TIM={res_b['TIM']}")
            results_by_strategy.setdefault("B_macro", []).append(res_b)

            # Long-only variant
            macro_lo = signal_to_position(
                macro_raw, lookback=720,
                long_threshold_pctile=0.4, short_threshold_pctile=-1.0,
                rebalance_hours=24,
            )
            macro_lo = macro_lo.clip(0, 1)
            res_b2 = backtest_position(macro_lo, close, label=f"B2: Macro LO ({sym})")
            print(f"  B2 Macro LO:     SR={res_b2['SR']}, CAGR={res_b2['CAGR']}, MDD={res_b2['MDD']}, TIM={res_b2['TIM']}")
            results_by_strategy.setdefault("B2_macro_LO", []).append(res_b2)

        # ── C: Combined On-chain + Macro (long-only) ──
        if onchain_raw is not None and macro_raw is not None:
            combined_pos = (onchain_lo + macro_lo) / 2
            combined_pos = combined_pos.clip(0, 1.0)
            res_c = backtest_position(combined_pos, close, label=f"C: Combined LO ({sym})")
            print(f"  C Combined LO:   SR={res_c['SR']}, CAGR={res_c['CAGR']}, MDD={res_c['MDD']}, TIM={res_c['TIM']}")
            results_by_strategy.setdefault("C_combined_LO", []).append(res_c)

    # ── TSMOM comparison (load from recent backtest) ──
    # Run quick TSMOM backtest for correlation
    print("\n" + "=" * 70)
    print("TSMOM COMPARISON (for correlation analysis)")
    print("=" * 70)

    from qtrade.config import load_config
    from qtrade.backtest.run_backtest import run_symbol_backtest

    tsmom_cfg_path = BASE_DIR / "config" / "prod_candidate_simplified.yaml"
    tsmom_cfg = load_config(tsmom_cfg_path)
    ref_path = DATA_DIR / "binance" / "futures" / "1h" / "BTCUSDT.parquet"
    ref_df = load_klines(ref_path)

    tsmom_returns = {}
    for sym in symbols:
        mt = tsmom_cfg.market_type_str
        data_path = tsmom_cfg.data_dir / "binance" / mt / tsmom_cfg.market.interval / f"{sym}.parquet"
        bt_cfg = tsmom_cfg.to_backtest_dict(sym)
        if ref_df is not None:
            bt_cfg["_regime_gate_ref_df"] = ref_df
        try:
            res = run_symbol_backtest(symbol=sym, data_path=data_path, cfg=bt_cfg)
            eq = res.pf.value()
            tsmom_returns[sym] = eq.pct_change().fillna(0)
        except Exception as e:
            print(f"  ⚠️ TSMOM {sym}: {e}")

    # ── Correlation Analysis ──
    print("\n" + "=" * 70)
    print("CORRELATION: New Signals vs TSMOM (daily returns)")
    print("=" * 70)

    for strat_name, strat_results in results_by_strategy.items():
        print(f"\n{strat_name}:")
        for res in strat_results:
            sym = res["label"].split("(")[1].rstrip(")")
            if sym in tsmom_returns:
                # Daily correlation
                strat_daily = res["returns"].resample("D").sum()
                tsmom_daily = tsmom_returns[sym].resample("D").sum()
                common = strat_daily.index.intersection(tsmom_daily.index)
                if len(common) > 30:
                    corr = strat_daily.loc[common].corr(tsmom_daily.loc[common])
                    print(f"  {sym}: corr={corr:.3f}")

    # ── Portfolio Blend Test ──
    print("\n" + "=" * 70)
    print("BLEND TEST: TSMOM + New Signal (equal-weight across symbols)")
    print("=" * 70)

    for strat_name, strat_results in results_by_strategy.items():
        # Build portfolio returns for this strategy
        strat_port = None
        tsmom_port = None
        n_syms = 0

        for res in strat_results:
            sym = res["label"].split("(")[1].rstrip(")")
            if sym in tsmom_returns:
                w = 1.0 / len(symbols)
                if strat_port is None:
                    common = res["returns"].index.intersection(tsmom_returns[sym].index)
                    strat_port = res["returns"].loc[common] * w
                    tsmom_port = tsmom_returns[sym].loc[common] * w
                else:
                    common = res["returns"].index.intersection(tsmom_returns[sym].index)
                    common = common.intersection(strat_port.index)
                    strat_port = strat_port.loc[common] + res["returns"].loc[common] * w
                    tsmom_port = tsmom_port.loc[common] + tsmom_returns[sym].loc[common] * w
                n_syms += 1

        if strat_port is not None and tsmom_port is not None:
            for blend_w in [0.1, 0.2, 0.3]:
                blend = (1 - blend_w) * tsmom_port + blend_w * strat_port
                cumret = (1 + blend).cumprod()
                total_ret = float(cumret.iloc[-1] - 1)
                n_years = len(blend) / (365.25 * 24)
                cagr = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
                sr = float(blend.mean() / blend.std() * np.sqrt(8760)) if blend.std() > 0 else 0
                mdd = float(((cumret - cumret.cummax()) / cumret.cummax()).min())

                tsmom_only = (1 + tsmom_port).cumprod()
                tsmom_sr = float(tsmom_port.mean() / tsmom_port.std() * np.sqrt(8760))

                print(f"  {strat_name} {blend_w:.0%} blend: SR={sr:.2f} (TSMOM-only={tsmom_sr:.2f}), "
                      f"MDD={mdd:.1%}, CAGR={cagr:.1%}")


if __name__ == "__main__":
    main()
