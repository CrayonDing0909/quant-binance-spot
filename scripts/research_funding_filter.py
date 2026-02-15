"""
Research Script: Funding Rate Filter Analysis
Goal: Check if extreme funding rates correlate with poor strategy performance.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import vectorbt as vbt

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from qtrade.data.storage import load_klines
from qtrade.data.funding_rate import load_funding_rates, align_funding_to_klines

def run_analysis(symbol: str, data_dir: Path):
    print(f"\n=== Funding Rate Analysis for {symbol} ===")
    
    # Load Price Data
    # Try different potential paths
    price_path = data_dir / "binance" / "futures" / "1h" / f"{symbol}.parquet"
    if not price_path.exists():
         price_path = data_dir / "futures" / "klines" / f"{symbol}.parquet"
         
    if not price_path.exists():
        print(f"Price data not found for {symbol} at {price_path}")
        return

    df = load_klines(price_path)
    close = df['close']
    
    # Load Funding Rate Data
    fr_path = data_dir / "binance" / "futures" / "funding_rate" / f"{symbol}.parquet"
    if not fr_path.exists():
        print(f"Funding rate data not found for {symbol}")
        return
        
    fr_df = load_funding_rates(fr_path)
    
    # Align Funding Rate to Price (1h)
    funding_rates = align_funding_to_klines(fr_df, df.index)
    
    # Generate Strategy Signals (Dynamic RSI)
    # Using parameters from our recent config
    rsi = vbt.RSI.run(close, window=10).rsi
    
    # Rolling percentile (simplified)
    window = 14 * 24
    q_low = 0.10
    q_high = 0.90
    
    rsi_rolling = rsi.rolling(window=window)
    oversold = rsi_rolling.quantile(q_low)
    overbought = rsi_rolling.quantile(q_high)
    
    # Entry Signals
    long_entries = (rsi.shift(1) < oversold.shift(1)) & (rsi >= oversold)
    short_entries = (rsi.shift(1) > overbought.shift(1)) & (rsi <= overbought)
    
    # Calculate Forward Returns (24h return to capture trend/reversion impact)
    # Shift(-24) means return from t to t+24
    fwd_ret = close.pct_change(24).shift(-24)
    
    # Create Analysis DataFrame
    analysis = pd.DataFrame({
        'funding_rate': funding_rates,
        'fwd_ret': fwd_ret,
        'is_long': long_entries,
        'is_short': short_entries
    }).dropna()
    
    # Analyze Longs
    longs = analysis[analysis['is_long']].copy()
    shorts = analysis[analysis['is_short']].copy()
    
    print(f"Total Long Signals: {len(longs)}")
    print(f"Total Short Signals: {len(shorts)}")
    
    # Bins for Funding Rate
    # Funding Rate is usually around 0.01% per 8h (0.0001)
    # High is > 0.05% (0.0005)
    # Low is < -0.05% (-0.0005)
    bins = [-np.inf, -0.0005, -0.0001, 0.0001, 0.0005, np.inf]
    labels = ['Very Neg', 'Neg', 'Neutral', 'Pos', 'Very Pos']
    
    longs['fr_bin'] = pd.cut(longs['funding_rate'], bins=bins, labels=labels)
    shorts['fr_bin'] = pd.cut(shorts['funding_rate'], bins=bins, labels=labels)
    
    print("\n--- Long Performance (24h Fwd Return) by Funding Rate ---")
    # observed=False to handle empty bins gracefully in newer pandas
    long_stats = longs.groupby('fr_bin', observed=False)['fwd_ret'].agg(['count', 'mean', 'median', lambda x: (x>0).mean()])
    long_stats.columns = ['Count', 'Mean Ret', 'Median Ret', 'Win Rate']
    print(long_stats)
    
    print("\n--- Short Performance (24h Fwd Return) by Funding Rate ---")
    short_stats = shorts.groupby('fr_bin', observed=False)['fwd_ret'].agg(['count', 'mean', 'median', lambda x: (x<0).mean()]) 
    short_stats.columns = ['Count', 'Mean Ret', 'Median Ret', 'Win Rate']
    # Invert returns for shorts to show "profit"
    short_stats['Mean Ret'] = -short_stats['Mean Ret']
    short_stats['Median Ret'] = -short_stats['Median Ret']
    print(short_stats)

if __name__ == "__main__":
    # Assuming running from quant-binance-spot
    data_dir = Path("data")
    if not data_dir.exists():
        # Maybe running from one level up?
        if Path("quant-binance-spot/data").exists():
            data_dir = Path("quant-binance-spot/data")
        
    run_analysis("BTCUSDT", data_dir)
    run_analysis("ETHUSDT", data_dir)
