"""
overbought 門檻掃描：測試不同做空入場門檻對 Long/Short 績效的影響

用法:
    python scripts/scan_overbought.py -c config/futures_rsi_adx_atr.yaml --symbol BTCUSDT
    python scripts/scan_overbought.py -c config/futures_rsi_adx_atr.yaml  # 測所有幣
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from qtrade.config import load_config
from qtrade.backtest.run_backtest import run_symbol_backtest
from qtrade.backtest.metrics import long_short_split_analysis


def scan_overbought(
    cfg_path: str,
    symbol: str | None = None,
    ob_values: list[int] | None = None,
) -> None:
    cfg = load_config(cfg_path)
    market_type = cfg.market_type_str
    direction = cfg.direction

    if market_type != "futures" or direction != "both":
        print("❌ 此掃描僅適用於 futures + both 模式")
        sys.exit(1)

    if ob_values is None:
        ob_values = [68, 70, 72, 73, 75, 78, 80, 82]

    symbols = [symbol] if symbol else cfg.market.symbols

    for sym in symbols:
        data_path = cfg.data_dir / "binance" / market_type / cfg.market.interval / f"{sym}.parquet"
        if not data_path.exists():
            print(f"⚠️  數據不存在: {data_path}")
            continue

        print(f"\n{'═'*80}")
        print(f"  📊 overbought 掃描: {sym}")
        print(f"  基準參數: oversold={cfg.strategy.params.get('oversold', 30)}, "
              f"stop_loss_atr={cfg.strategy.params.get('stop_loss_atr')}, "
              f"take_profit_atr={cfg.strategy.params.get('take_profit_atr')}")
        print(f"{'═'*80}")

        rows = []
        base_cfg = cfg.to_backtest_dict(symbol=sym)

        for ob in ob_values:
            t0 = time.time()
            # 修改 overbought 參數
            test_cfg = base_cfg.copy()
            test_cfg["strategy_params"] = {
                **test_cfg["strategy_params"],
                "overbought": ob,
            }

            try:
                res = run_symbol_backtest(
                    sym, data_path, test_cfg,
                    strategy_name=cfg.strategy.name,
                    data_dir=cfg.data_dir,
                )
            except Exception as e:
                print(f"  ❌ overbought={ob} 失敗: {e}")
                continue

            elapsed = time.time() - t0
            pf = res["pf"]
            stats = pf.stats()
            pos = res["pos"]

            # Long/Short 分開統計
            ls = long_short_split_analysis(pf, pos)
            long_s = ls["long"]
            short_s = ls["short"]

            row = {
                "overbought": ob,
                # 整體
                "Total Return [%]": round(stats.get("Total Return [%]", 0), 1),
                "Ann. Return [%]": round(
                    res.get("adjusted_stats", {}).get("Total Return [%]", stats.get("Total Return [%]", 0))
                    if res.get("adjusted_stats") else stats.get("Total Return [%]", 0), 1
                ),
                "Sharpe": round(stats.get("Sharpe Ratio", 0), 2),
                "MDD [%]": round(stats.get("Max Drawdown [%]", 0), 1),
                "Trades": int(stats.get("Total Trades", 0)),
                "PF": round(stats.get("Profit Factor", 0), 2),
                # Long 側
                "L_Trades": long_s.get("Total Trades", 0),
                "L_WR [%]": long_s.get("Win Rate [%]", 0),
                "L_PnL": round(long_s.get("Total PnL", 0), 0),
                "L_PF": long_s.get("Profit Factor", 0),
                "L_Avg [%]": long_s.get("Avg Return [%]", 0),
                # Short 側
                "S_Trades": short_s.get("Total Trades", 0),
                "S_WR [%]": short_s.get("Win Rate [%]", 0),
                "S_PnL": round(short_s.get("Total PnL", 0), 0),
                "S_PF": short_s.get("Profit Factor", 0),
                "S_Avg [%]": short_s.get("Avg Return [%]", 0),
                "time": f"{elapsed:.1f}s",
            }
            rows.append(row)
            print(f"  ✅ overbought={ob:>2}  Sharpe={row['Sharpe']:.2f}  "
                  f"L_PF={row['L_PF']}  S_PF={row['S_PF']}  ({elapsed:.1f}s)")

        if not rows:
            continue

        # 組成 DataFrame
        df_result = pd.DataFrame(rows)
        df_result = df_result.set_index("overbought")

        # ── 列印完整對比表 ──
        print(f"\n{'─'*80}")
        print(f"  {sym}  overbought 門檻掃描結果")
        print(f"{'─'*80}")

        # 整體指標
        print("\n📊 整體績效:")
        overall_cols = ["Total Return [%]", "Sharpe", "MDD [%]", "Trades", "PF"]
        print(df_result[overall_cols].to_string())

        # Long vs Short 對比
        print("\n📈 Long 側:")
        long_cols = ["L_Trades", "L_WR [%]", "L_PnL", "L_PF", "L_Avg [%]"]
        print(df_result[long_cols].to_string())

        print("\n📉 Short 側:")
        short_cols = ["S_Trades", "S_WR [%]", "S_PnL", "S_PF", "S_Avg [%]"]
        print(df_result[short_cols].to_string())

        # ── 找最佳值 ──
        print(f"\n{'─'*80}")
        print(f"  🏆 最佳 overbought 門檻分析")
        print(f"{'─'*80}")

        # 用 Sharpe 做主指標
        best_sharpe_idx = df_result["Sharpe"].idxmax()
        print(f"  最高 Sharpe:     overbought={best_sharpe_idx}  (Sharpe={df_result.loc[best_sharpe_idx, 'Sharpe']:.2f})")

        # 最高 Short PF
        s_pf_series = df_result["S_PF"].replace("∞", 0).astype(float)
        best_spf_idx = s_pf_series.idxmax()
        print(f"  最高 Short PF:   overbought={best_spf_idx}  (S_PF={df_result.loc[best_spf_idx, 'S_PF']})")

        # 最高總報酬
        best_ret_idx = df_result["Total Return [%]"].idxmax()
        print(f"  最高總報酬:      overbought={best_ret_idx}  (Return={df_result.loc[best_ret_idx, 'Total Return [%]']}%)")

        # Short PnL 正值且 Sharpe 最高的
        positive_short = df_result[s_pf_series > 1.0]
        if not positive_short.empty:
            best_balanced = positive_short["Sharpe"].idxmax()
            print(f"  最佳平衡點:      overbought={best_balanced}  "
                  f"(Sharpe={positive_short.loc[best_balanced, 'Sharpe']:.2f}, "
                  f"S_PF={positive_short.loc[best_balanced, 'S_PF']})")
        else:
            print(f"  ⚠️ 所有門檻下 Short PF < 1.0（做空端虧損）")

        # 保存 CSV
        csv_path = Path(f"reports/futures/{cfg.strategy.name}/scan_overbought_{sym}.csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df_result.to_csv(csv_path)
        print(f"\n✅ 結果已儲存: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="overbought 門檻掃描")
    parser.add_argument("-c", "--config", default="config/futures_rsi_adx_atr.yaml")
    parser.add_argument("--symbol", type=str, default=None)
    parser.add_argument(
        "--values", type=str, default=None,
        help="自訂掃描值，逗號分隔 (e.g. 68,70,72,75,78,80)"
    )
    args = parser.parse_args()

    ob_values = None
    if args.values:
        ob_values = [int(v.strip()) for v in args.values.split(",")]

    scan_overbought(args.config, args.symbol, ob_values)


if __name__ == "__main__":
    main()
