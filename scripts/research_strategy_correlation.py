"""
策略相關性矩陣分析

比較多策略的信號相關性 + 各策略單獨回測績效。
用於選擇低相關策略組合（策略 Ensemble）。

使用方式:
    python scripts/research_strategy_correlation.py -c config/futures_rsi_adx_atr.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qtrade.config import load_config
from qtrade.data.storage import load_klines
from qtrade.strategy.base import StrategyContext
from qtrade.strategy import get_strategy
from qtrade.backtest.run_backtest import run_symbol_backtest

# 測試的策略列表（排除 example / toy 策略）
STRATEGIES = [
    "rsi_adx_atr",
    "bb_mean_reversion",
    "macd_momentum",
    "multi_factor",
    "smc_basic",
]


def compute_signals(strategy_name: str, df: pd.DataFrame, ctx: StrategyContext, params: dict) -> pd.Series:
    """計算策略信號"""
    try:
        func = get_strategy(strategy_name)
        signals = func(df, ctx, params)
        return signals
    except Exception as e:
        print(f"  ⚠️  {strategy_name}: {e}")
        return pd.Series(0.0, index=df.index)


def run_quick_backtest(strategy_name: str, cfg, symbol: str, data_path: Path) -> dict:
    """跑單策略快速回測，回傳 key metrics"""
    try:
        bt_cfg = cfg.to_backtest_dict(symbol)
        bt_cfg["strategy_name"] = strategy_name

        result = run_symbol_backtest(
            symbol=symbol,
            data_path=data_path,
            cfg=bt_cfg,
            strategy_name=strategy_name,
        )

        if result is None:
            return {"sharpe": np.nan, "return": np.nan, "mdd": np.nan, "trades": 0}

        pf = result["pf"]
        stats = pf.stats()

        return {
            "sharpe": stats.get("Sharpe Ratio", np.nan),
            "return": stats.get("Total Return [%]", np.nan),
            "mdd": stats.get("Max Drawdown [%]", np.nan),
            "trades": int(stats.get("Total Trades", 0)),
        }
    except Exception as e:
        print(f"  ⚠️  回測 {strategy_name} 失敗: {e}")
        return {"sharpe": np.nan, "return": np.nan, "mdd": np.nan, "trades": 0}


def main():
    parser = argparse.ArgumentParser(description="策略相關性矩陣分析")
    parser.add_argument("-c", "--config", type=str, default="config/futures_rsi_adx_atr.yaml")
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    args = parser.parse_args()

    cfg = load_config(args.config)
    market_type = cfg.market_type_str
    symbol = args.symbol

    data_path = cfg.data_dir / "binance" / market_type / cfg.market.interval / f"{symbol}.parquet"
    if not data_path.exists():
        print(f"❌ 數據不存在: {data_path}")
        return

    df = load_klines(data_path)
    print(f"📊 策略相關性矩陣分析")
    print(f"   幣對: {symbol}")
    print(f"   數據: {df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')} ({len(df):,} bars)")
    print(f"   策略: {', '.join(STRATEGIES)}")
    print()

    # 1) 計算所有策略的信號
    ctx = StrategyContext(
        symbol=symbol,
        interval=cfg.market.interval,
        market_type=market_type,
        direction=cfg.direction,
    )
    params = cfg.strategy.get_params(symbol)

    signals_dict = {}
    for name in STRATEGIES:
        print(f"  計算 {name} 信號...", end="")
        sig = compute_signals(name, df, ctx, params)
        # 只保留非零信號的「方向」(-1, 0, +1)
        direction = np.sign(sig)
        signals_dict[name] = direction
        active = (direction != 0).sum()
        print(f" ✅ 活躍率 {active/len(df)*100:.1f}%")

    signals_df = pd.DataFrame(signals_dict, index=df.index)

    # 2) 信號相關性矩陣（只看非零 bar）
    print()
    print("═" * 70)
    print("  信號方向相關性矩陣 (Pearson)")
    print("═" * 70)

    corr_matrix = signals_df.corr()
    print(corr_matrix.round(3).to_string())
    print()

    # 找出低相關配對
    print("  低相關配對 (|corr| < 0.3):")
    for i, s1 in enumerate(STRATEGIES):
        for s2 in STRATEGIES[i+1:]:
            c = corr_matrix.loc[s1, s2]
            if abs(c) < 0.3:
                print(f"    {s1} × {s2}: {c:.3f} ✅")
    print()

    # 3) 各策略單獨回測
    print("═" * 70)
    print("  各策略單獨回測（使用相同的費用/滑點/風控）")
    print("═" * 70)

    results = {}
    for name in STRATEGIES:
        print(f"  回測 {name}...", end="")
        r = run_quick_backtest(name, cfg, symbol, data_path)
        results[name] = r
        sr = r["sharpe"]
        ret = r["return"]
        mdd = r["mdd"]
        trades = r["trades"]
        if not np.isnan(sr):
            print(f" SR={sr:.2f}, Ret={ret:.1f}%, MDD={mdd:.1f}%, Trades={trades}")
        else:
            print(f" ❌ 回測失敗")

    print()
    print("═" * 70)
    print("  績效排名")
    print("═" * 70)
    perf_df = pd.DataFrame(results).T
    perf_df = perf_df.sort_values("sharpe", ascending=False)
    print(perf_df.to_string())
    print()

    # 4) 推薦 Ensemble
    print("═" * 70)
    print("  Ensemble 推薦")
    print("═" * 70)

    # 選擇 Sharpe > 0 且相互低相關的策略
    viable = [name for name, r in results.items() if not np.isnan(r["sharpe"]) and r["sharpe"] > 0]

    if len(viable) >= 2:
        # 找出所有低相關配對
        low_corr_pairs = []
        for i, s1 in enumerate(viable):
            for s2 in viable[i+1:]:
                c = corr_matrix.loc[s1, s2]
                low_corr_pairs.append((s1, s2, c))

        low_corr_pairs.sort(key=lambda x: abs(x[2]))

        print(f"  可用策略 (SR > 0): {viable}")
        print(f"\n  配對相關性（由低到高）:")
        for s1, s2, c in low_corr_pairs:
            emoji = "✅" if abs(c) < 0.3 else "🟡" if abs(c) < 0.5 else "⚠️"
            sr1 = results[s1]["sharpe"]
            sr2 = results[s2]["sharpe"]
            # 理論組合 Sharpe (equal weight, ignoring vol scaling)
            combined_sr = (sr1 + sr2) / np.sqrt(2 * (1 + c))
            print(f"    {s1} × {s2}: corr={c:.3f} {emoji} | Combo SR≈{combined_sr:.2f}")
    else:
        print(f"  ⚠️  可用策略不足 ({len(viable)})，無法組合")

    print()
    print("✅ 分析完成")


if __name__ == "__main__":
    main()
