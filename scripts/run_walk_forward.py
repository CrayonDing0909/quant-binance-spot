"""
Walk-Forward Analysis Script

Usage:
    python scripts/run_walk_forward.py -c config/futures_rsi_adx_atr.yaml --splits 5
    python scripts/run_walk_forward.py -c config/futures_ensemble_nw_tsmom.yaml --splits 5
"""
import argparse
import sys
from pathlib import Path
import pandas as pd
import yaml
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from qtrade.config import load_config
from qtrade.validation.walk_forward import walk_forward_analysis, walk_forward_summary


def _load_ensemble_strategy(config_path: str, symbol: str) -> tuple | None:
    """從 ensemble 配置取得 symbol 的策略名與參數"""
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    ens = raw.get("ensemble")
    if ens and ens.get("enabled", False):
        strategies = ens.get("strategies", {})
        if symbol in strategies:
            s = strategies[symbol]
            return s["name"], s.get("params", {})
    return None

def main():
    parser = argparse.ArgumentParser(description="Run Walk-Forward Analysis")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--splits", type=int, default=5, help="Number of splits (default: 5)")
    parser.add_argument("--symbol", type=str, help="Specific symbol to test (overrides config)")
    
    args = parser.parse_args()
    
    # Load config
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"Error: Config file not found at {cfg_path}")
        sys.exit(1)
        
    cfg = load_config(cfg_path)
    
    # Override symbol if provided
    symbols = [args.symbol] if args.symbol else cfg.market.symbols
    
    print(f"=== Walk-Forward Analysis: {cfg.strategy.name} ===")
    print(f"Config: {args.config}")
    print(f"Splits: {args.splits}")
    print(f"Market Type: {cfg.market_type_str}")
    
    for symbol in symbols:
        # ── Ensemble: 檢查 per-symbol 策略路由 ──
        sym_strategy_name = cfg.strategy.name
        sym_params = None

        ensemble_override = _load_ensemble_strategy(str(cfg_path), symbol)
        if ensemble_override:
            sym_strategy_name, sym_params = ensemble_override
            print(f"\n🧩 Ensemble: {symbol} → {sym_strategy_name}")
        else:
            print(f"\nAnalyzing {symbol} [{sym_strategy_name}]...")
        
        # Resolve data path
        # Try exact path first (standard structure: data_dir/binance/market_type/interval/symbol.parquet)
        data_path = Path(cfg.data_dir) / "binance" / cfg.market_type_str / "klines" / f"{symbol}.parquet"
        if not data_path.exists():
             # Try without 'klines' (some setups might use interval directly)
             data_path = Path(cfg.data_dir) / "binance" / cfg.market_type_str / cfg.market.interval / f"{symbol}.parquet"
        
        if not data_path.exists():
             # Fallback: try without binance prefix (legacy?)
             data_path = Path(cfg.data_dir) / cfg.market_type_str / "klines" / f"{symbol}.parquet"

        
        if not data_path.exists():
            print(f"❌ Data file not found: {data_path}")
            continue
            
        # Prepare config dict for backtest
        backtest_cfg = cfg.to_backtest_dict(symbol=symbol)

        # Ensemble: 覆蓋策略名與參數
        if sym_params is not None:
            backtest_cfg["strategy_params"] = sym_params
        backtest_cfg["strategy_name"] = sym_strategy_name
        
        # Run WFA
        wf_df = walk_forward_analysis(
            symbol=symbol,
            data_path=data_path,
            cfg=backtest_cfg,
            n_splits=args.splits,
            data_dir=Path(cfg.data_dir)
        )
        
        if wf_df.empty:
            print(f"❌ No results for {symbol}")
            continue
            
        summary = walk_forward_summary(wf_df)
        print("\n" + summary["summary_text"])
        
        # Save results
        out_dir = Path("reports") / "walk_forward" / cfg.strategy.name / datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir.mkdir(parents=True, exist_ok=True)
        
        wf_csv_path = out_dir / f"wf_results_{symbol}.csv"
        wf_df.to_csv(wf_csv_path, index=False)
        print(f"  Saved details to: {wf_csv_path}")
        
        summary_path = out_dir / f"summary_{symbol}.txt"
        with open(summary_path, "w") as f:
            f.write(summary["summary_text"])
            
    print("\nDone.")

if __name__ == "__main__":
    main()
