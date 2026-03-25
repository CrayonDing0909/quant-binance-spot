#!/usr/bin/env python3
"""
# EMBARGO_EXEMPT: Production-layer audit on a frozen baseline; this comparison intentionally uses the full deployment sample.
# ORTHOGONALITY_EXEMPT: This study evaluates portfolio-level exposure scaling, not a new candidate alpha factor.

Formal evaluation of `low_freq_portfolio` as a portfolio layer / exposure scalar.

The script compares:
1. Baseline (`config/prod_candidate_simplified.yaml`)
2. Baseline + low-frequency portfolio layer

It always reports both:
- production-matched: existing overlay ON
- naked: existing overlay OFF

Output:
- `summary.json`
- `report.md`
- per-comparison `portfolio_equity.csv`
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from qtrade.backtest.costs import (
    adjust_equity_for_funding,
    compute_adjusted_stats,
    compute_funding_costs,
    compute_volume_slippage,
)
from qtrade.backtest.metrics import benchmark_buy_and_hold
from qtrade.backtest.run_backtest import (
    BacktestResult,
    run_symbol_backtest,
    safe_portfolio_from_orders,
    to_vbt_direction,
)
from qtrade.config import AppConfig, load_config
from qtrade.data.funding_rate import (
    align_funding_to_klines,
    get_funding_rate_path,
    load_funding_rates,
)
from qtrade.data.quality import clean_data, validate_data_quality
from qtrade.data.storage import load_klines
from qtrade.strategy import get_strategy
from qtrade.strategy.base import StrategyContext


@dataclass
class PortfolioSummary:
    label: str
    overlay_enabled: bool
    experiment_name: str
    portfolio_equity: pd.Series
    portfolio_returns: pd.Series
    gross_exposure: pd.Series
    stats: dict
    per_symbol: dict[str, BacktestResult]
    common_index: pd.DatetimeIndex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate low_freq_portfolio as a portfolio exposure layer over the frozen production baseline."
    )
    parser.add_argument(
        "-c",
        "--config",
        default="config/research_low_freq_portfolio_layer.yaml",
        help="Research config path",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional explicit output directory",
    )
    return parser.parse_args()


def load_research_cfg(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_symbol_weights(cfg: AppConfig) -> dict[str, float]:
    symbols = list(cfg.market.symbols)
    raw = {sym: cfg.portfolio.get_weight(sym, len(symbols)) for sym in symbols}
    total = sum(raw.values())
    return {sym: w / total for sym, w in raw.items()}


def build_bt_cfg(cfg: AppConfig, symbol: str, overlay_enabled: bool) -> dict:
    bt_cfg = copy.deepcopy(cfg.to_backtest_dict(symbol=symbol))
    if not overlay_enabled:
        bt_cfg.pop("overlay", None)
    return bt_cfg


def load_full_symbol_df(cfg: AppConfig, symbol: str) -> pd.DataFrame:
    data_path = (
        cfg.data_dir
        / "binance"
        / cfg.market_type_str
        / cfg.market.interval
        / f"{symbol}.parquet"
    )
    df = load_klines(data_path)
    if cfg.backtest.validate_data:
        report = validate_data_quality(df)
        if not report.is_valid:
            print(f"⚠️  {symbol}: data quality warnings during layer audit")
            for issue in report.errors + report.warnings:
                print(f"   - {issue}")
    if cfg.backtest.clean_data:
        df = clean_data(
            df,
            fill_method="forward",
            remove_outliers=False,
            remove_duplicates=True,
        )
    return df


def compute_regime_components(
    df: pd.DataFrame,
    *,
    adx_period: int,
    vol_lookback: int,
    vol_percentile_window: int,
) -> tuple[pd.Series, pd.Series]:
    close = df["close"]
    log_returns = np.log(close / close.shift(1))
    realized_vol = log_returns.rolling(vol_lookback).std() * np.sqrt(365 * 24)
    vol_pctile = realized_vol.rolling(vol_percentile_window, min_periods=60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1],
        raw=False,
    )

    high = df["high"]
    low = df["low"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(adx_period).mean()

    plus_dm = (high - high.shift(1)).clip(lower=0)
    minus_dm = (low.shift(1) - low).clip(lower=0)
    plus_mask = plus_dm < minus_dm
    plus_dm[plus_mask] = 0
    minus_mask = minus_dm < plus_dm
    minus_dm[minus_mask] = 0

    plus_di = 100 * (plus_dm.rolling(adx_period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(adx_period).mean() / atr)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10))
    adx = dx.rolling(adx_period).mean()
    return adx, vol_pctile


def compute_layer_scalar_and_regime_inputs(
    cfg: AppConfig,
    symbol: str,
    target_index: pd.DatetimeIndex,
    layer_strategy_name: str,
    layer_params: dict,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    df_full = load_full_symbol_df(cfg, symbol)
    signal_delay = 1 if cfg.backtest.trade_on == "next_open" else 0
    ctx = StrategyContext(
        symbol=symbol,
        interval=cfg.market.interval,
        market_type=cfg.market_type_str,
        direction=cfg.direction,
        signal_delay=signal_delay,  # Backtest mode: execute next-open, must delay 1 bar
    )
    params = copy.deepcopy(layer_params)
    params.setdefault("_data_dir", cfg.data_dir)

    layer_func = get_strategy(layer_strategy_name)
    scalar_full = layer_func(df_full, ctx, params).clip(lower=0.0, upper=1.0)
    scalar = scalar_full.reindex(target_index).fillna(0.0)

    adx, vol_pctile = compute_regime_components(
        df_full,
        adx_period=params.get("adx_period", 14),
        vol_lookback=params.get("vol_lookback", 20),
        vol_percentile_window=params.get("vol_percentile_window", 252),
    )
    adx = adx.reindex(target_index).ffill()
    vol_pctile = vol_pctile.reindex(target_index).ffill()
    return scalar, adx, vol_pctile


def rebuild_backtest_from_positions(
    symbol: str,
    df: pd.DataFrame,
    pos: pd.Series,
    bt_cfg: dict,
    data_dir: Path,
) -> BacktestResult:
    fee = bt_cfg["fee_bps"] / 10_000.0
    sm_cfg = bt_cfg.get("slippage_model", {})
    slippage_result = None
    if sm_cfg.get("enabled", False):
        slippage_result = compute_volume_slippage(
            pos=pos,
            df=df,
            capital=bt_cfg["initial_cash"],
            base_bps=sm_cfg.get("base_bps", 2.0),
            impact_coefficient=sm_cfg.get("impact_coefficient", 0.1),
            impact_power=sm_cfg.get("impact_power", 0.5),
            adv_lookback=sm_cfg.get("adv_lookback", 20),
            participation_rate=sm_cfg.get("participation_rate", 0.10),
            leverage=bt_cfg.get("leverage", 1),
        )
        slippage = slippage_result.slippage_array
    else:
        slippage = bt_cfg["slippage_bps"] / 10_000.0

    pf = safe_portfolio_from_orders(
        df=df,
        pos=pos,
        fee=fee,
        slippage=slippage,
        init_cash=bt_cfg["initial_cash"],
        freq=bt_cfg.get("interval", "1h"),
        direction=to_vbt_direction(bt_cfg.get("direction", "both")),
        exit_exec_prices=pos.attrs.get("exit_exec_prices"),
    )
    pf_bh = benchmark_buy_and_hold(
        df,
        initial_cash=bt_cfg["initial_cash"],
        fee_bps=bt_cfg["fee_bps"],
        slippage_bps=bt_cfg["slippage_bps"],
        interval=bt_cfg.get("interval", "1h"),
    )

    stats = pf.stats()
    fr_cfg = bt_cfg.get("funding_rate", {})
    funding_cost = None
    adjusted_equity = None
    adjusted_stats = None

    if fr_cfg.get("enabled", False) and bt_cfg.get("market_type") == "futures":
        fr_path = get_funding_rate_path(data_dir, symbol)
        funding_df = None
        if fr_cfg.get("use_historical", True):
            funding_df = load_funding_rates(fr_path)
        funding_rates = align_funding_to_klines(
            funding_df,
            df.index,
            default_rate_8h=fr_cfg.get("default_rate_8h", 0.0001),
        )
        funding_cost = compute_funding_costs(
            pos=pos,
            equity=pf.value(),
            funding_rates=funding_rates,
            leverage=bt_cfg.get("leverage", 1),
        )
        adjusted_equity = adjust_equity_for_funding(pf.value(), funding_cost)
        adjusted_stats = compute_adjusted_stats(
            adjusted_equity,
            bt_cfg["initial_cash"],
        )

    return BacktestResult(
        pf=pf,
        pf_bh=pf_bh,
        stats=stats,
        df=df,
        pos=pos,
        funding_cost=funding_cost,
        slippage_result=slippage_result,
        adjusted_stats=adjusted_stats,
        adjusted_equity=adjusted_equity,
        funding_rate_enabled=fr_cfg.get("enabled", False),
        slippage_model_enabled=sm_cfg.get("enabled", False),
    )


def calculate_portfolio_stats(
    returns: pd.Series,
    equity: pd.Series,
    initial_cash: float,
) -> dict:
    total_return = float((equity.iloc[-1] - initial_cash) / initial_cash)
    years = len(returns) / (365 * 24)
    annual_return = float((1 + total_return) ** (1 / years) - 1) if years > 0 else 0.0
    rolling_max = equity.expanding().max()
    drawdown = (equity - rolling_max) / rolling_max
    max_drawdown = float(abs(drawdown.min()))
    std = returns.std()
    sharpe = float(np.sqrt(365 * 24) * returns.mean() / std) if std > 0 else 0.0
    downside = returns[returns < 0]
    downside_std = downside.std() if len(downside) > 0 else 0.0
    sortino = (
        float(np.sqrt(365 * 24) * returns.mean() / downside_std)
        if downside_std > 0
        else 0.0
    )
    calmar = float(annual_return / max_drawdown) if max_drawdown > 0 else 0.0
    return {
        "total_return_pct": round(total_return * 100, 2),
        "annual_return_pct": round(annual_return * 100, 2),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "calmar": round(calmar, 3),
    }


def aggregate_portfolio(
    label: str,
    experiment_name: str,
    overlay_enabled: bool,
    per_symbol_results: dict[str, BacktestResult],
    weights: dict[str, float],
    initial_cash: float,
) -> PortfolioSummary:
    active_symbols = list(per_symbol_results.keys())
    common_index = per_symbol_results[active_symbols[0]].equity().index
    for sym in active_symbols[1:]:
        common_index = common_index.intersection(per_symbol_results[sym].equity().index)

    eq_curves = {
        sym: per_symbol_results[sym].equity().loc[common_index] for sym in active_symbols
    }
    normalized = {sym: eq / eq.iloc[0] for sym, eq in eq_curves.items()}
    portfolio_normalized = sum(normalized[sym] * weights[sym] for sym in active_symbols)
    portfolio_equity = portfolio_normalized * initial_cash
    portfolio_returns = portfolio_equity.pct_change().fillna(0.0)

    gross_exposure = pd.Series(0.0, index=common_index)
    for sym in active_symbols:
        sym_pos = per_symbol_results[sym].pos.reindex(common_index).fillna(0.0)
        gross_exposure = gross_exposure.add(sym_pos.abs() * weights[sym], fill_value=0.0)

    stats = calculate_portfolio_stats(portfolio_returns, portfolio_equity, initial_cash)
    return PortfolioSummary(
        label=label,
        overlay_enabled=overlay_enabled,
        experiment_name=experiment_name,
        portfolio_equity=portfolio_equity,
        portfolio_returns=portfolio_returns,
        gross_exposure=gross_exposure,
        stats=stats,
        per_symbol=per_symbol_results,
        common_index=common_index,
    )


def compute_contiguous_segment_drawdowns(
    equity: pd.Series,
    mask: pd.Series,
    *,
    min_bars: int = 24,
) -> list[float]:
    mask = mask.reindex(equity.index).fillna(False).astype(bool)
    dd_values: list[float] = []
    start_idx = None
    for i, flag in enumerate(mask):
        if flag and start_idx is None:
            start_idx = i
        elif not flag and start_idx is not None:
            if i - start_idx >= min_bars:
                seg = equity.iloc[start_idx:i]
                dd = (seg / seg.cummax() - 1.0).min()
                dd_values.append(abs(float(dd)) * 100)
            start_idx = None
    if start_idx is not None and len(mask) - start_idx >= min_bars:
        seg = equity.iloc[start_idx:]
        dd = (seg / seg.cummax() - 1.0).min()
        dd_values.append(abs(float(dd)) * 100)
    return dd_values


def build_portfolio_regime_masks(
    regime_inputs: dict[str, dict[str, pd.Series]],
    weights: dict[str, float],
    index: pd.DatetimeIndex,
) -> dict[str, pd.Series]:
    agg_adx = pd.Series(0.0, index=index)
    agg_vol = pd.Series(0.0, index=index)
    for sym, comp in regime_inputs.items():
        agg_adx = agg_adx.add(comp["adx"].reindex(index).ffill() * weights[sym], fill_value=0.0)
        agg_vol = agg_vol.add(
            comp["vol_pctile"].reindex(index).ffill() * weights[sym],
            fill_value=0.0,
        )

    high_vol_no_trend = (agg_adx < 20) & (agg_vol > 0.90)
    chop = (agg_adx < 20) & (agg_vol >= 0.70) & (agg_vol <= 0.90)
    sideways = (agg_adx < 20) & (agg_vol < 0.70)
    return {
        "sideways": sideways,
        "chop": chop,
        "high_vol_no_trend": high_vol_no_trend,
    }


def compare_regime_drawdowns(
    baseline: PortfolioSummary,
    layered: PortfolioSummary,
    regime_masks: dict[str, pd.Series],
) -> dict[str, dict]:
    out = {}
    for regime_name, mask in regime_masks.items():
        base_dds = compute_contiguous_segment_drawdowns(baseline.portfolio_equity, mask)
        layer_dds = compute_contiguous_segment_drawdowns(layered.portfolio_equity, mask)
        out[regime_name] = {
            "bars": int(mask.reindex(baseline.common_index).fillna(False).sum()),
            "share_pct": round(float(mask.reindex(baseline.common_index).fillna(False).mean()) * 100, 2),
            "baseline_worst_segment_mdd_pct": round(max(base_dds), 2) if base_dds else None,
            "layer_worst_segment_mdd_pct": round(max(layer_dds), 2) if layer_dds else None,
            "delta_worst_segment_mdd_pct": (
                round(max(layer_dds) - max(base_dds), 2)
                if base_dds and layer_dds
                else None
            ),
            "baseline_avg_segment_mdd_pct": round(float(np.mean(base_dds)), 2) if base_dds else None,
            "layer_avg_segment_mdd_pct": round(float(np.mean(layer_dds)), 2) if layer_dds else None,
        }
    return out


def run_baseline_set(
    cfg: AppConfig,
    overlay_enabled: bool,
) -> tuple[dict[str, BacktestResult], dict[str, pd.DataFrame]]:
    results: dict[str, BacktestResult] = {}
    cleaned_dfs: dict[str, pd.DataFrame] = {}
    for symbol in cfg.market.symbols:
        bt_cfg = build_bt_cfg(cfg, symbol, overlay_enabled=overlay_enabled)
        data_path = (
            cfg.data_dir
            / "binance"
            / cfg.market_type_str
            / cfg.market.interval
            / f"{symbol}.parquet"
        )
        res = run_symbol_backtest(
            symbol,
            data_path,
            bt_cfg,
            strategy_name=cfg.strategy.name,
            data_dir=cfg.data_dir,
        )
        results[symbol] = res
        cleaned_dfs[symbol] = res.df
    return results, cleaned_dfs


def run_layered_set(
    cfg: AppConfig,
    baseline_results: dict[str, BacktestResult],
    overlay_enabled: bool,
    layer_strategy_name: str,
    layer_params: dict,
) -> tuple[dict[str, BacktestResult], dict[str, dict[str, pd.Series]]]:
    results: dict[str, BacktestResult] = {}
    regime_inputs: dict[str, dict[str, pd.Series]] = {}
    for symbol, base_res in baseline_results.items():
        scalar, adx, vol_pctile = compute_layer_scalar_and_regime_inputs(
            cfg,
            symbol,
            base_res.df.index,
            layer_strategy_name,
            layer_params,
        )
        scaled_pos = (base_res.pos * scalar).clip(lower=-1.0, upper=1.0)
        if base_res.pos.attrs:
            scaled_pos.attrs.update(dict(base_res.pos.attrs))
        bt_cfg = build_bt_cfg(cfg, symbol, overlay_enabled=overlay_enabled)
        results[symbol] = rebuild_backtest_from_positions(
            symbol=symbol,
            df=base_res.df,
            pos=scaled_pos,
            bt_cfg=bt_cfg,
            data_dir=cfg.data_dir,
        )
        regime_inputs[symbol] = {"adx": adx, "vol_pctile": vol_pctile}
    return results, regime_inputs


def summarise_comparison(
    baseline: PortfolioSummary,
    layered: PortfolioSummary,
    regime_drawdowns: dict[str, dict],
) -> dict:
    base_avg_exp = float(baseline.gross_exposure.mean())
    layer_avg_exp = float(layered.gross_exposure.mean())
    base_tim = float((baseline.gross_exposure > 0.05).mean())
    layer_tim = float((layered.gross_exposure > 0.05).mean())

    return {
        "baseline": baseline.stats,
        "layered": layered.stats,
        "delta": {
            "sharpe": round(layered.stats["sharpe"] - baseline.stats["sharpe"], 3),
            "total_return_pct": round(
                layered.stats["total_return_pct"] - baseline.stats["total_return_pct"], 2
            ),
            "max_drawdown_pct": round(
                layered.stats["max_drawdown_pct"] - baseline.stats["max_drawdown_pct"], 2
            ),
            "calmar": round(layered.stats["calmar"] - baseline.stats["calmar"], 3),
            "annual_return_pct": round(
                layered.stats["annual_return_pct"] - baseline.stats["annual_return_pct"], 2
            ),
        },
        "exposure": {
            "baseline_avg_gross_exposure": round(base_avg_exp, 4),
            "layer_avg_gross_exposure": round(layer_avg_exp, 4),
            "exposure_retention_pct": round(100 * layer_avg_exp / base_avg_exp, 2)
            if base_avg_exp > 0
            else None,
            "baseline_time_in_market_pct": round(base_tim * 100, 2),
            "layer_time_in_market_pct": round(layer_tim * 100, 2),
            "tim_retention_pct": round(100 * layer_tim / base_tim, 2) if base_tim > 0 else None,
        },
        "return_efficiency": {
            "annual_return_retention_pct": round(
                100 * layered.stats["annual_return_pct"] / baseline.stats["annual_return_pct"],
                2,
            )
            if baseline.stats["annual_return_pct"] != 0
            else None,
            "calmar_retention_pct": round(
                100 * layered.stats["calmar"] / baseline.stats["calmar"], 2
            )
            if baseline.stats["calmar"] != 0
            else None,
        },
        "regime_drawdowns": regime_drawdowns,
    }


def decide_verdict(prod_matched: dict, naked: dict) -> tuple[str, str]:
    prod_delta = prod_matched["delta"]
    prod_exp = prod_matched["exposure"]
    prod_regimes = prod_matched["regime_drawdowns"]

    improved_high_vol = (
        prod_regimes["high_vol_no_trend"]["delta_worst_segment_mdd_pct"] is not None
        and prod_regimes["high_vol_no_trend"]["delta_worst_segment_mdd_pct"] < 0
    )
    improved_chop = (
        prod_regimes["chop"]["delta_worst_segment_mdd_pct"] is not None
        and prod_regimes["chop"]["delta_worst_segment_mdd_pct"] < 0
    )

    return_retention = prod_matched["return_efficiency"]["annual_return_retention_pct"] or 0.0
    exposure_retention = prod_exp["exposure_retention_pct"] or 0.0

    if (
        prod_delta["max_drawdown_pct"] <= -0.30
        and prod_delta["calmar"] > 0
        and return_retention >= 85.0
        and exposure_retention >= 85.0
        and (improved_high_vol or improved_chop)
    ):
        return "GO_NEXT", "Production-matched MDD improved with acceptable return retention and without a large exposure collapse."

    if (
        prod_delta["max_drawdown_pct"] < 0
        and prod_delta["annual_return_pct"] <= -5.0
        and exposure_retention < 85.0
    ):
        return "KEEP_BASELINE", "Drawdown improved mainly via lower gross exposure / time-in-market; this behaves more like an over-filter than a robust portfolio layer."

    if prod_delta["max_drawdown_pct"] >= 0 and naked["delta"]["max_drawdown_pct"] >= 0:
        return "FAIL", "Neither production-matched nor naked comparison reduced drawdown."

    return "NEED_MORE_WORK", "Results are mixed: some defensive benefit appears, but the production-matched trade-off is not yet strong enough for promotion."


def write_equity_csv(path: Path, baseline: PortfolioSummary, layered: PortfolioSummary) -> None:
    df = pd.DataFrame(
        {
            "baseline_equity": baseline.portfolio_equity,
            "layered_equity": layered.portfolio_equity,
            "baseline_gross_exposure": baseline.gross_exposure,
            "layered_gross_exposure": layered.gross_exposure,
        }
    )
    df.to_csv(path)


def write_markdown_report(
    output_path: Path,
    baseline_config: str,
    experiments: list[dict],
    verdict: str,
    verdict_reason: str,
) -> None:
    lines = [
        "# Low-Frequency Portfolio Layer Evaluation",
        "",
        f"- Baseline: `{baseline_config}`",
        f"- Verdict: `{verdict}`",
        f"- Summary: {verdict_reason}",
        "",
    ]

    for exp in experiments:
        lines.append(f"## {exp['name']}")
        lines.append("")
        lines.append(exp["description"])
        lines.append("")
        for mode_name, comp in exp["comparisons"].items():
            lines.append(f"### {mode_name}")
            lines.append("")
            lines.append("| Metric | Baseline | Layered | Delta |")
            lines.append("|---|---:|---:|---:|")
            for key in ("sharpe", "total_return_pct", "max_drawdown_pct", "calmar", "annual_return_pct"):
                lines.append(
                    f"| {key} | {comp['baseline'][key]} | {comp['layered'][key]} | {comp['delta'][key]} |"
                )
            lines.append("")
            lines.append(
                f"- Avg gross exposure: {comp['exposure']['baseline_avg_gross_exposure']:.4f} -> {comp['exposure']['layer_avg_gross_exposure']:.4f}"
            )
            lines.append(
                f"- Time in market: {comp['exposure']['baseline_time_in_market_pct']:.2f}% -> {comp['exposure']['layer_time_in_market_pct']:.2f}%"
            )
            for regime_name, regime_stats in comp["regime_drawdowns"].items():
                lines.append(
                    f"- {regime_name}: worst-segment MDD {regime_stats['baseline_worst_segment_mdd_pct']}% -> "
                    f"{regime_stats['layer_worst_segment_mdd_pct']}% (Δ {regime_stats['delta_worst_segment_mdd_pct']}%)"
                )
            lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    research_cfg_path = Path(args.config)
    research_cfg = load_research_cfg(research_cfg_path)

    baseline_config_path = Path(research_cfg["research"]["baseline_config"])
    baseline_cfg = load_config(str(baseline_config_path))
    weights = normalize_symbol_weights(baseline_cfg)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = (
            Path("reports")
            / "research"
            / research_cfg["research"].get("output_subdir", "low_freq_portfolio_layer")
            / timestamp
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    mode_flags = []
    comparison_cfg = research_cfg.get("comparison", {})
    if comparison_cfg.get("include_overlay_matched", True):
        mode_flags.append(("production_matched", True))
    if comparison_cfg.get("include_naked", True):
        mode_flags.append(("naked", False))

    baseline_cache: dict[bool, PortfolioSummary] = {}
    baseline_results_cache: dict[bool, dict[str, BacktestResult]] = {}

    print(f"Baseline config: {baseline_config_path}")
    print(f"Output dir: {output_dir}")

    for mode_name, overlay_enabled in mode_flags:
        print(f"\n=== Building baseline: {mode_name} (overlay={'ON' if overlay_enabled else 'OFF'}) ===")
        per_symbol_base, _ = run_baseline_set(baseline_cfg, overlay_enabled=overlay_enabled)
        baseline_results_cache[overlay_enabled] = per_symbol_base
        baseline_cache[overlay_enabled] = aggregate_portfolio(
            label="baseline",
            experiment_name="baseline",
            overlay_enabled=overlay_enabled,
            per_symbol_results=per_symbol_base,
            weights=weights,
            initial_cash=baseline_cfg.backtest.initial_cash,
        )

    experiment_summaries: list[dict] = []
    best_prod_comparison = None

    for exp_cfg in research_cfg["experiments"]:
        exp_name = exp_cfg["name"]
        description = exp_cfg.get("description", "")
        layer_cfg = exp_cfg["portfolio_layer"]
        layer_strategy_name = layer_cfg["strategy_name"]
        layer_params = layer_cfg.get("params", {})
        print(f"\n{'=' * 72}\nExperiment: {exp_name}\n{'=' * 72}")

        comparisons = {}
        for mode_name, overlay_enabled in mode_flags:
            print(f"  -> {mode_name} (overlay={'ON' if overlay_enabled else 'OFF'})")
            layered_results, regime_inputs = run_layered_set(
                baseline_cfg,
                baseline_results_cache[overlay_enabled],
                overlay_enabled=overlay_enabled,
                layer_strategy_name=layer_strategy_name,
                layer_params=layer_params,
            )
            layered_summary = aggregate_portfolio(
                label="layered",
                experiment_name=exp_name,
                overlay_enabled=overlay_enabled,
                per_symbol_results=layered_results,
                weights=weights,
                initial_cash=baseline_cfg.backtest.initial_cash,
            )
            base_summary = baseline_cache[overlay_enabled]
            regime_masks = build_portfolio_regime_masks(
                regime_inputs,
                weights,
                layered_summary.common_index,
            )
            regime_drawdowns = compare_regime_drawdowns(
                base_summary,
                layered_summary,
                regime_masks,
            )
            comparison = summarise_comparison(
                base_summary,
                layered_summary,
                regime_drawdowns,
            )
            comparisons[mode_name] = comparison

            subdir = output_dir / exp_name / mode_name
            subdir.mkdir(parents=True, exist_ok=True)
            write_equity_csv(subdir / "portfolio_equity.csv", base_summary, layered_summary)
            with (subdir / "comparison.json").open("w", encoding="utf-8") as f:
                json.dump(comparison, f, indent=2, default=str)

        experiment_summaries.append(
            {
                "name": exp_name,
                "description": description,
                "layer_strategy": layer_strategy_name,
                "layer_params": layer_params,
                "comparisons": comparisons,
            }
        )

        prod_comparison = comparisons.get("production_matched")
        if prod_comparison is not None:
            if best_prod_comparison is None:
                best_prod_comparison = (exp_name, prod_comparison)
            else:
                _, current_best = best_prod_comparison
                current_score = current_best["delta"]["calmar"]
                new_score = prod_comparison["delta"]["calmar"]
                if new_score > current_score:
                    best_prod_comparison = (exp_name, prod_comparison)

    if best_prod_comparison is None:
        raise RuntimeError("No production-matched comparison was generated")

    best_name, best_prod = best_prod_comparison
    best_naked = next(
        (
            exp["comparisons"]["naked"]
            for exp in experiment_summaries
            if exp["name"] == best_name and "naked" in exp["comparisons"]
        ),
        best_prod,
    )
    verdict, verdict_reason = decide_verdict(best_prod, best_naked)

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_config": str(baseline_config_path),
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "best_production_matched_experiment": best_name,
        "experiments": experiment_summaries,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    write_markdown_report(
        output_dir / "report.md",
        str(baseline_config_path),
        experiment_summaries,
        verdict,
        verdict_reason,
    )

    print(f"\nVerdict: {verdict}")
    print(verdict_reason)
    print(f"\nSummary: {output_dir / 'summary.json'}")
    print(f"Report : {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
