#!/usr/bin/env python3
"""
成本敏感性分析

測試不同成本假設下策略的 Sharpe 和 Total Return，
了解策略對手續費、滑點、funding rate 的容忍度。

使用方法:
    python scripts/run_cost_sensitivity.py -c config/futures_rsi_adx_atr.yaml
    python scripts/run_cost_sensitivity.py -c config/futures_rsi_adx_atr.yaml --symbol BTCUSDT

輸出:
    - 不同成本假設下的 Sharpe/Return 表格
    - 識別策略對哪個成本因素最敏感
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# 確保 src/ 在 sys.path 中
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from qtrade.config import load_config
from qtrade.backtest.run_backtest import run_symbol_backtest


def run_sensitivity(
    symbol: str,
    data_path: Path,
    base_cfg: dict,
    strategy_name: str,
    data_dir: Path,
) -> pd.DataFrame:
    """
    對單個幣種執行成本敏感性分析
    
    測試以下維度：
    1. Funding rate: 0.005%, 0.01% (基準), 0.02%, 0.03% per 8h
    2. Fee: 2bps, 4bps (基準), 8bps, 12bps
    3. Slippage: 1bps, 3bps (基準), 5bps, 10bps
    """
    results = []

    # ── 基準線 ──────────────────────────────────
    base_res = run_symbol_backtest(
        symbol, data_path, base_cfg, strategy_name, data_dir=data_dir,
    )
    base_stats = base_res["stats"]
    base_adj = base_res.get("adjusted_stats") or base_stats
    base_sharpe = base_adj.get("Sharpe Ratio", base_stats.get("Sharpe Ratio", 0))
    base_return = base_adj.get("Total Return [%]", base_stats.get("Total Return [%]", 0))
    base_mdd = base_adj.get("Max Drawdown [%]", base_stats.get("Max Drawdown [%]", 0))

    results.append({
        "scenario": "基準",
        "funding_rate_8h": base_cfg.get("funding_rate", {}).get("default_rate_8h", 0.0001),
        "fee_bps": base_cfg["fee_bps"],
        "slippage_bps": base_cfg["slippage_bps"],
        "sharpe": base_sharpe,
        "total_return_pct": base_return,
        "max_dd_pct": base_mdd,
    })

    # ── Funding Rate 敏感性 ──────────────────────
    for rate_8h in [0.00005, 0.0002, 0.0003]:
        cfg = {**base_cfg}
        cfg["funding_rate"] = {
            **base_cfg.get("funding_rate", {}),
            "enabled": True,
            "default_rate_8h": rate_8h,
            "use_historical": False,  # 用固定值測試敏感性
        }
        try:
            res = run_symbol_backtest(symbol, data_path, cfg, strategy_name, data_dir=data_dir)
            adj = res.get("adjusted_stats") or res["stats"]
            results.append({
                "scenario": f"FR={rate_8h*100:.3f}%",
                "funding_rate_8h": rate_8h,
                "fee_bps": cfg["fee_bps"],
                "slippage_bps": cfg["slippage_bps"],
                "sharpe": adj.get("Sharpe Ratio", res["stats"].get("Sharpe Ratio", 0)),
                "total_return_pct": adj.get("Total Return [%]", res["stats"].get("Total Return [%]", 0)),
                "max_dd_pct": adj.get("Max Drawdown [%]", res["stats"].get("Max Drawdown [%]", 0)),
            })
        except Exception as e:
            print(f"  ⚠️  FR={rate_8h}: {e}")

    # ── Fee 敏感性 ───────────────────────────────
    for fee in [2, 8, 12]:
        if fee == base_cfg["fee_bps"]:
            continue
        cfg = {**base_cfg, "fee_bps": fee}
        try:
            res = run_symbol_backtest(symbol, data_path, cfg, strategy_name, data_dir=data_dir)
            adj = res.get("adjusted_stats") or res["stats"]
            results.append({
                "scenario": f"Fee={fee}bps",
                "funding_rate_8h": base_cfg.get("funding_rate", {}).get("default_rate_8h", 0.0001),
                "fee_bps": fee,
                "slippage_bps": base_cfg["slippage_bps"],
                "sharpe": adj.get("Sharpe Ratio", res["stats"].get("Sharpe Ratio", 0)),
                "total_return_pct": adj.get("Total Return [%]", res["stats"].get("Total Return [%]", 0)),
                "max_dd_pct": adj.get("Max Drawdown [%]", res["stats"].get("Max Drawdown [%]", 0)),
            })
        except Exception as e:
            print(f"  ⚠️  Fee={fee}: {e}")

    # ── Slippage 敏感性 ──────────────────────────
    for slip in [1, 5, 10]:
        if slip == base_cfg["slippage_bps"]:
            continue
        cfg = {**base_cfg, "slippage_bps": slip}
        # 關閉 volume slippage model 以測試固定滑點影響
        cfg["slippage_model"] = {**base_cfg.get("slippage_model", {}), "enabled": False}
        try:
            res = run_symbol_backtest(symbol, data_path, cfg, strategy_name, data_dir=data_dir)
            adj = res.get("adjusted_stats") or res["stats"]
            results.append({
                "scenario": f"Slip={slip}bps",
                "funding_rate_8h": base_cfg.get("funding_rate", {}).get("default_rate_8h", 0.0001),
                "fee_bps": base_cfg["fee_bps"],
                "slippage_bps": slip,
                "sharpe": adj.get("Sharpe Ratio", res["stats"].get("Sharpe Ratio", 0)),
                "total_return_pct": adj.get("Total Return [%]", res["stats"].get("Total Return [%]", 0)),
                "max_dd_pct": adj.get("Max Drawdown [%]", res["stats"].get("Max Drawdown [%]", 0)),
            })
        except Exception as e:
            print(f"  ⚠️  Slip={slip}: {e}")

    return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="成本敏感性分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-c", "--config", type=str, default="config/futures_rsi_adx_atr.yaml")
    parser.add_argument("--symbol", type=str, default=None)
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    market_type = cfg.market_type_str
    strategy_name = cfg.strategy.name
    symbols = [args.symbol] if args.symbol else cfg.market.symbols

    print(f"{'='*70}")
    print(f"  成本敏感性分析  {'🟢' if market_type == 'spot' else '🔴'} {market_type.upper()} | {strategy_name}")
    print(f"{'='*70}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = cfg.get_report_dir("validation") / f"cost_sensitivity_{timestamp}"
    if not args.no_save:
        report_dir.mkdir(parents=True, exist_ok=True)

    for sym in symbols:
        print(f"\n{'─'*60}")
        print(f"  {sym}")
        print(f"{'─'*60}")

        data_path = cfg.data_dir / "binance" / market_type / cfg.market.interval / f"{sym}.parquet"
        if not data_path.exists():
            print(f"  ⚠️  數據不存在: {data_path}")
            continue

        bt_cfg = cfg.to_backtest_dict(symbol=sym)

        print(f"  🔄 運行成本敏感性分析...")
        df = run_sensitivity(sym, data_path, bt_cfg, strategy_name, cfg.data_dir)

        # 顯示結果
        print(f"\n  === 成本敏感性 ({sym}) ===")
        print(f"  {'Scenario':<20} {'FR (8h)':>10} {'Fee':>8} {'Slip':>8} {'Sharpe':>10} {'Return':>12} {'MDD':>10}")
        print(f"  {'-'*78}")

        base_sharpe = df.iloc[0]["sharpe"]
        for _, row in df.iterrows():
            marker = " ←基準" if row["scenario"] == "基準" else ""
            delta = row["sharpe"] - base_sharpe
            delta_str = f"({delta:+.2f})" if row["scenario"] != "基準" else ""
            print(
                f"  {row['scenario']:<20}"
                f" {row['funding_rate_8h']*100:>9.3f}%"
                f" {row['fee_bps']:>7.0f}"
                f" {row['slippage_bps']:>7.0f}"
                f" {row['sharpe']:>10.2f}"
                f" {row['total_return_pct']:>11.1f}%"
                f" {row['max_dd_pct']:>9.1f}%"
                f" {delta_str}{marker}"
            )

        # 識別最敏感的因素
        if len(df) > 1:
            # 計算每個維度的 Sharpe 變化
            fr_rows = df[df["scenario"].str.startswith("FR=")]
            fee_rows = df[df["scenario"].str.startswith("Fee=")]
            slip_rows = df[df["scenario"].str.startswith("Slip=")]

            sensitivities = {}
            if not fr_rows.empty:
                sensitivities["Funding Rate"] = abs(fr_rows["sharpe"] - base_sharpe).max()
            if not fee_rows.empty:
                sensitivities["手續費"] = abs(fee_rows["sharpe"] - base_sharpe).max()
            if not slip_rows.empty:
                sensitivities["滑點"] = abs(slip_rows["sharpe"] - base_sharpe).max()

            if sensitivities:
                most_sensitive = max(sensitivities, key=sensitivities.get)
                print(f"\n  → 策略對 {most_sensitive} 最敏感 (最大 Sharpe 變化: {sensitivities[most_sensitive]:.2f})")

        # 保存
        if not args.no_save:
            csv_path = report_dir / f"cost_sensitivity_{sym}.csv"
            df.to_csv(csv_path, index=False)
            print(f"  ✅ 報告: {csv_path}")

    if not args.no_save:
        print(f"\n📁 報告目錄: {report_dir}")


if __name__ == "__main__":
    main()
