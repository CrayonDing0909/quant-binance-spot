"""
Risk Parameter Scan
Find the optimal combination of Stop Loss and Cooldown Bars.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from qtrade.config import load_config
from qtrade.backtest.run_backtest import run_symbol_backtest

def main():
    config_path = "config/futures_rsi_adx_atr.yaml"
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        print(f"Config not found: {config_path}")
        return

    cfg = load_config(config_path)
    
    # Symbols to test
    symbols = ["BTCUSDT", "ETHUSDT"]
    
    # Parameter Grid
    stop_loss_grid = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    cooldown_grid = [1, 3, 5, 8, 12, 24]
    
    print(f"=== Risk Parameter Scan ===")
    print(f"Config: {config_path}")
    print(f"Stop Loss Grid: {stop_loss_grid}")
    print(f"Cooldown Grid: {cooldown_grid}")
    
    for symbol in symbols:
        print(f"\nScanning {symbol}...")
        
        # Prepare base config
        base_params = cfg.strategy.params.copy()
        
        # Resolve data path
        # Try standard path
        data_path = cfg.data_dir / "binance" / cfg.market_type_str / cfg.market.interval / f"{symbol}.parquet"
        if not data_path.exists():
            # Fallback
            data_path = cfg.data_dir / cfg.market_type_str / "klines" / f"{symbol}.parquet"
            
        if not data_path.exists():
            print(f"Data not found: {data_path}")
            continue
            
        # Results matrix
        sharpe_matrix = pd.DataFrame(index=stop_loss_grid, columns=cooldown_grid)
        
        # Use simple loops (not efficient but straightforward for small grid)
        total_runs = len(stop_loss_grid) * len(cooldown_grid)
        count = 0
        
        for sl in stop_loss_grid:
            for cd in cooldown_grid:
                count += 1
                print(f"\r  Progress: {count}/{total_runs} (SL={sl}, CD={cd})", end="", flush=True)
                
                # Update params
                current_params = base_params.copy()
                current_params["stop_loss_atr"] = sl
                current_params["cooldown_bars"] = cd
                
                # Build backtest config
                bt_cfg = cfg.to_backtest_dict(symbol=symbol)
                bt_cfg["strategy_params"] = current_params
                # Clear overrides to ensure we test exactly these params
                bt_cfg["strategy_params"].pop("symbol_overrides", None)
                
                # Run backtest (suppress output)
                try:
                    res = run_symbol_backtest(
                        symbol, data_path, bt_cfg, cfg.strategy.name,
                        data_dir=cfg.data_dir,
                    )
                    
                    # Prioritize adjusted stats if available
                    stats = res.get("adjusted_stats", res["pf"].stats())
                    sharpe = stats.get("Sharpe Ratio", 0)
                    
                    # Valid check
                    if np.isnan(sharpe): sharpe = 0
                    
                    sharpe_matrix.loc[sl, cd] = sharpe
                    
                except Exception as e:
                    # print(f"  Error scan SL={sl}, CD={cd}: {e}")
                    sharpe_matrix.loc[sl, cd] = np.nan

        print(f"\n\n{symbol} Sharpe Ratio Heatmap (Rows=SL, Cols=Cooldown):")
        print(sharpe_matrix.astype(float).round(2).to_string())
        
        # Find best
        # Stack to find max
        stacked = sharpe_matrix.astype(float).stack()
        best_idx = stacked.idxmax()
        best_sharpe = stacked.max()
        
        print(f"Best Config for {symbol}: SL={best_idx[0]}, CD={best_idx[1]} (Sharpe: {best_sharpe:.2f})")

if __name__ == "__main__":
    main()
