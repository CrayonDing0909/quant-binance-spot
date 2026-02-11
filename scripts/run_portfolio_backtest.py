"""
組合回測腳本 - 同時回測多個幣種的組合表現

支援：
- 等權重分配（預設）
- 自訂權重分配
- 組合績效統計

使用範例：
    # 等權重 BTC + ETH 組合
    python scripts/run_portfolio_backtest.py -c config/rsi_adx_atr.yaml --symbols BTCUSDT ETHUSDT
    
    # 自訂權重 (BTC 60%, ETH 40%)
    python scripts/run_portfolio_backtest.py -c config/rsi_adx_atr.yaml --symbols BTCUSDT ETHUSDT --weights 0.6 0.4
"""
from __future__ import annotations
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import json

from qtrade.config import load_config
from qtrade.data.storage import load_klines
from qtrade.data.quality import validate_data_quality, clean_data
from qtrade.strategy.base import StrategyContext
from qtrade.strategy import get_strategy


def run_portfolio_backtest(
    symbols: list[str],
    weights: list[float],
    cfg,
    output_dir: Path,
) -> dict:
    """
    執行組合回測
    
    Args:
        symbols: 交易對列表
        weights: 權重列表（與 symbols 對應）
        cfg: 配置對象
        output_dir: 輸出目錄
    
    Returns:
        組合回測結果
    """
    import vectorbt as vbt
    
    # 正規化權重
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    print(f"\n📊 組合配置:")
    for sym, w in zip(symbols, weights):
        print(f"   {sym}: {w*100:.1f}%")
    print()
    
    # 載入所有數據
    market_type = cfg.market.market_type.value
    interval = cfg.market.interval
    
    all_data = {}
    min_start = None
    max_end = None
    
    for symbol in symbols:
        data_path = cfg.data_dir / "binance" / market_type / interval / f"{symbol}.parquet"
        df = load_klines(data_path)
        df = clean_data(df, fill_method="forward", remove_outliers=False, remove_duplicates=True)
        all_data[symbol] = df
        
        if min_start is None or df.index[0] > min_start:
            min_start = df.index[0]
        if max_end is None or df.index[-1] < max_end:
            max_end = df.index[-1]
    
    print(f"📅 共同時間範圍: {min_start} → {max_end}")
    
    # 對齊所有數據到共同時間範圍
    for symbol in symbols:
        all_data[symbol] = all_data[symbol].loc[min_start:max_end]
    
    # 獲取策略和參數
    strategy_name = cfg.strategy.name
    strategy_func = get_strategy(strategy_name)
    base_params = cfg.strategy.params
    symbol_overrides = cfg.strategy.symbol_overrides or {}
    
    # 回測參數
    initial_cash = cfg.backtest.initial_cash
    fee = cfg.backtest.fee_bps / 10000
    slippage = cfg.backtest.slippage_bps / 10000
    
    # 使用 vectorbt 計算各幣種的淨值曲線
    equity_curves = {}
    all_positions = {}
    
    for symbol in symbols:
        df = all_data[symbol]
        
        # 合併參數
        params = base_params.copy()
        if symbol in symbol_overrides:
            params.update(symbol_overrides[symbol])
        
        ctx = StrategyContext(
            symbol=symbol,
            interval=interval,
            market_type=market_type,
            direction="long_only",
        )
        
        # 生成持倉信號
        pos = strategy_func(df, ctx, params)
        pos = pos.clip(lower=0.0)  # Spot 只做多
        all_positions[symbol] = pos
        
        # 用 vectorbt 計算（使用 open 價格執行，與 run_backtest.py 一致）
        pf = vbt.Portfolio.from_orders(
            close=df["close"],
            size=pos,
            size_type="targetpercent",
            price=df["open"],  # 關鍵：使用開盤價執行
            fees=fee,
            slippage=slippage,
            init_cash=initial_cash,
            freq="1h",
            direction="longonly",
        )
        
        equity_curves[symbol] = pf.value()
        print(f"  {symbol}: 回報 {pf.total_return()*100:.2f}%, MDD {pf.max_drawdown()*100:.2f}%")
    
    # 標準化淨值曲線（都從 1 開始）
    normalized = {}
    for symbol in symbols:
        eq = equity_curves[symbol]
        normalized[symbol] = eq / eq.iloc[0]
    
    # 組合淨值 = 加權平均
    portfolio_normalized = sum(normalized[s] * w for s, w in zip(symbols, weights))
    portfolio_equity = portfolio_normalized * initial_cash
    
    # Buy & Hold 組合
    bh_normalized = {}
    for symbol in symbols:
        df = all_data[symbol]
        bh_eq = df["close"] / df["close"].iloc[0]
        bh_normalized[symbol] = bh_eq
    bh_portfolio_normalized = sum(bh_normalized[s] * w for s, w in zip(symbols, weights))
    bh_equity = bh_portfolio_normalized * initial_cash
    
    # 計算組合收益率序列（用於統計）
    portfolio_returns = portfolio_equity.pct_change().fillna(0)
    bh_returns = bh_equity.pct_change().fillna(0)
    
    # 計算統計指標
    stats = calculate_portfolio_stats(portfolio_returns, portfolio_equity, initial_cash)
    bh_stats = calculate_portfolio_stats(bh_returns, bh_equity, initial_cash)
    
    # 輸出結果
    print("\n" + "=" * 70)
    print(f"  組合回測結果: {' + '.join(symbols)}")
    print("=" * 70)
    
    print(f"\n{'指標':<30} {'組合策略':>18} {'組合 Buy&Hold':>18}")
    print("-" * 70)
    print(f"{'Start':<30} {str(min_start)[:10]:>18} {str(min_start)[:10]:>18}")
    print(f"{'End':<30} {str(max_end)[:10]:>18} {str(max_end)[:10]:>18}")
    print(f"{'Total Return [%]':<30} {stats['total_return']*100:>18.2f} {bh_stats['total_return']*100:>18.2f}")
    print(f"{'Annualized Return [%]':<30} {stats['annual_return']*100:>18.2f} {bh_stats['annual_return']*100:>18.2f}")
    print(f"{'Max Drawdown [%]':<30} {stats['max_drawdown']*100:>18.2f} {bh_stats['max_drawdown']*100:>18.2f}")
    print(f"{'Sharpe Ratio':<30} {stats['sharpe']:>18.2f} {bh_stats['sharpe']:>18.2f}")
    print(f"{'Sortino Ratio':<30} {stats['sortino']:>18.2f} {bh_stats['sortino']:>18.2f}")
    print(f"{'Calmar Ratio':<30} {stats['calmar']:>18.2f} {bh_stats['calmar']:>18.2f}")
    
    # 繪製組合曲線
    plot_portfolio_equity(
        portfolio_equity, 
        bh_equity, 
        symbols, 
        weights,
        output_dir / "portfolio_equity_curve.png"
    )
    
    # 儲存結果
    results = {
        "symbols": symbols,
        "weights": weights.tolist(),
        "start": str(min_start),
        "end": str(max_end),
        "strategy_stats": stats,
        "buyhold_stats": bh_stats,
    }
    
    with open(output_dir / "portfolio_stats.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # 儲存資金曲線
    equity_df = pd.DataFrame({
        "strategy": portfolio_equity,
        "buyhold": bh_equity,
    })
    equity_df.to_csv(output_dir / "portfolio_equity.csv")
    
    print(f"\n✅ 組合資金曲線圖: {output_dir / 'portfolio_equity_curve.png'}")
    print(f"✅ 組合統計: {output_dir / 'portfolio_stats.json'}")
    
    return results


def calculate_portfolio_stats(returns: pd.Series, equity: pd.Series, initial_cash: float) -> dict:
    """計算組合統計指標"""
    # 總收益
    total_return = (equity.iloc[-1] - initial_cash) / initial_cash
    
    # 年化收益（假設每年 252 * 24 小時，1h 數據）
    n_periods = len(returns)
    years = n_periods / (365 * 24)
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    
    # 最大回撤
    rolling_max = equity.expanding().max()
    drawdown = (equity - rolling_max) / rolling_max
    max_drawdown = abs(drawdown.min())
    
    # Sharpe Ratio（年化）
    excess_returns = returns - 0  # 假設無風險利率為 0
    sharpe = np.sqrt(365 * 24) * excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
    
    # Sortino Ratio
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else 0.001
    sortino = np.sqrt(365 * 24) * returns.mean() / downside_std if downside_std > 0 else 0
    
    # Calmar Ratio
    calmar = annual_return / max_drawdown if max_drawdown > 0 else 0
    
    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
    }


def plot_portfolio_equity(
    strategy_equity: pd.Series,
    bh_equity: pd.Series,
    symbols: list[str],
    weights: np.ndarray,
    save_path: Path,
):
    """繪製組合資金曲線"""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # 組合名稱
    weight_str = " + ".join([f"{s} {w*100:.0f}%" for s, w in zip(symbols, weights)])
    
    # 資金曲線
    ax1 = axes[0]
    ax1.plot(strategy_equity.index, strategy_equity.values, label="Portfolio Strategy", color="blue", linewidth=1.5)
    ax1.plot(bh_equity.index, bh_equity.values, label="Portfolio Buy & Hold", color="gray", linestyle="--", alpha=0.7)
    
    # 標註最終收益
    final_strat = (strategy_equity.iloc[-1] / strategy_equity.iloc[0] - 1) * 100
    final_bh = (bh_equity.iloc[-1] / bh_equity.iloc[0] - 1) * 100
    ax1.annotate(f"+{final_strat:.1f}%", xy=(strategy_equity.index[-1], strategy_equity.iloc[-1]),
                 fontsize=10, color="blue", fontweight="bold")
    ax1.annotate(f"+{final_bh:.1f}%", xy=(bh_equity.index[-1], bh_equity.iloc[-1]),
                 fontsize=10, color="gray")
    
    ax1.set_title(f"Portfolio Backtest: {weight_str}", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Portfolio Value")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")
    
    # 回撤曲線
    ax2 = axes[1]
    rolling_max = strategy_equity.expanding().max()
    drawdown = (strategy_equity - rolling_max) / rolling_max * 100
    ax2.fill_between(drawdown.index, drawdown.values, 0, color="red", alpha=0.3, label="Strategy DD")
    
    bh_rolling_max = bh_equity.expanding().max()
    bh_drawdown = (bh_equity - bh_rolling_max) / bh_rolling_max * 100
    ax2.plot(bh_drawdown.index, bh_drawdown.values, color="gray", linestyle="--", alpha=0.5, label="B&H DD")
    
    ax2.set_ylabel("Drawdown [%]")
    ax2.set_xlabel("Date")
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="組合回測")
    parser.add_argument("-c", "--config", type=str, default="config/rsi_adx_atr.yaml", help="配置檔案")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"], help="交易對列表")
    parser.add_argument("--weights", nargs="+", type=float, default=None, help="權重列表（與 symbols 對應）")
    parser.add_argument("--output-dir", type=str, default=None, help="輸出目錄")
    
    args = parser.parse_args()
    
    # 載入配置
    cfg = load_config(args.config)
    
    # 設定權重
    if args.weights is None:
        weights = [1.0 / len(args.symbols)] * len(args.symbols)  # 等權重
    else:
        if len(args.weights) != len(args.symbols):
            raise ValueError(f"權重數量 ({len(args.weights)}) 與交易對數量 ({len(args.symbols)}) 不符")
        weights = args.weights
    
    # 設定輸出目錄
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("reports/portfolio") / f"{'+'.join(args.symbols)}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📊 組合回測: {' + '.join(args.symbols)}")
    print(f"📁 輸出目錄: {output_dir}")
    
    # 執行回測
    run_portfolio_backtest(args.symbols, weights, cfg, output_dir)


if __name__ == "__main__":
    main()
