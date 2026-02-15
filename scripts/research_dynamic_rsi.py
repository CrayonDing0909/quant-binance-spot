"""
Research Script: Static vs Dynamic RSI Thresholds
Goal: Test if dynamic thresholds (rolling percentiles) can mitigate alpha decay in 2025-2026.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import vectorbt as vbt
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from qtrade.data.storage import load_klines

def calculate_ic(signal: pd.Series, returns: pd.Series) -> float:
    """Calculate Information Coefficient (Spearman Rank Correlation)"""
    return signal.corr(returns, method='spearman')

def run_analysis(symbol: str, data_path: Path):
    print(f"\n=== Analysis for {symbol} ===")
    
    # Load data
    df = load_klines(data_path)
    close = df['close']
    
    # Calculate returns (1h forward return)
    # Use shift(-1) because we want to predict the NEXT return based on CURRENT signal
    returns_1h = close.pct_change().shift(-1)
    
    # Calculate RSI using vectorbt
    rsi = vbt.RSI.run(close, window=14).rsi
    
    # --- Strategy 1: Static Thresholds ---
    # Long < 30, Short > 80
    static_long = (rsi < 30).astype(int)
    static_short = (rsi > 80).astype(int)
    static_signal = static_long - static_short
    
    # --- Strategy 2: Dynamic Thresholds (Rolling Percentile) ---
    # Window: 14 days = 14 * 24 = 336 hours
    window = 336
    rsi_rolling = rsi.rolling(window=window)
    rsi_low = rsi_rolling.quantile(0.10)  # 10th percentile
    rsi_high = rsi_rolling.quantile(0.90) # 90th percentile
    
    dynamic_long = (rsi < rsi_low).astype(int)
    dynamic_short = (rsi > rsi_high).astype(int)
    dynamic_signal = dynamic_long - dynamic_short
    
    # Align data
    valid_mask = returns_1h.notna() & rsi.notna() & rsi_low.notna()
    
    # --- Analyze by Year ---
    years = sorted(df.index.year.unique())
    
    print(f"{'Year':<6} | {'Static IC':<10} | {'Dynamic IC':<10} | {'Static Ret':<10} | {'Dynamic Ret':<10} | {'St. Trades':<10} | {'Dy. Trades':<10}")
    print("-" * 85)
    
    cumulative_static = 0.0
    cumulative_dynamic = 0.0
    
    for year in years:
        mask = (df.index.year == year) & valid_mask
        if not mask.any():
            continue
            
        # IC Analysis
        ic_static = calculate_ic(static_signal[mask], returns_1h[mask])
        ic_dynamic = calculate_ic(dynamic_signal[mask], returns_1h[mask])
        
        # Simple Returns (Signal * Return) - assuming 1h holding period
        # Note: This is a very rough approximation, ignoring costs/slippage/compounding
        ret_static = (static_signal[mask] * returns_1h[mask]).sum()
        ret_dynamic = (dynamic_signal[mask] * returns_1h[mask]).sum()
        
        cumulative_static += ret_static
        cumulative_dynamic += ret_dynamic
        
        # Count trades (signals != 0)
        trades_static = (static_signal[mask] != 0).sum()
        trades_dynamic = (dynamic_signal[mask] != 0).sum()
        
        print(f"{year:<6} | {ic_static:>10.4f} | {ic_dynamic:>10.4f} | {ret_static:>10.2%} | {ret_dynamic:>10.2%} | {trades_static:>10} | {trades_dynamic:>10}")
        
    print("-" * 85)
    print(f"TOTAL  | {'-':>10} | {'-':>10} | {cumulative_static:>10.2%} | {cumulative_dynamic:>10.2%} | {'-':>10} | {'-':>10}")
    
    # Compare 2025-2026 performance specifically
    mask_recent = (df.index.year >= 2025) & valid_mask
    ret_static_recent = (static_signal[mask_recent] * returns_1h[mask_recent]).sum()
    ret_dynamic_recent = (dynamic_signal[mask_recent] * returns_1h[mask_recent]).sum()
    
    print("\n=== Recent Performance (2025-2026) ===")
    print(f"Static Return:  {ret_static_recent:.2%}")
    print(f"Dynamic Return: {ret_dynamic_recent:.2%}")
    if ret_static_recent != 0:
        print(f"Improvement:    {(ret_dynamic_recent - ret_static_recent) / abs(ret_static_recent):.1%}")
    else:
        print(f"Improvement:    N/A")

if __name__ == "__main__":
    # Hardcoded path for now, based on previous findings
    # Assuming running from quant-binance-spot root
    data_dir = Path("data/binance/futures/1h")
    
    symbols = ["BTCUSDT", "ETHUSDT"]
    
    for sym in symbols:
        path = data_dir / f"{sym}.parquet"
        if path.exists():
            run_analysis(sym, path)
        else:
            print(f"Data not found for {sym} at {path}")
