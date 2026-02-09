#!/usr/bin/env python3
"""
Kelly 公式驗證腳本

驗證你的策略是否適合使用 Kelly 倉位管理：
1. 檢查統計穩定性（勝率、盈虧比是否穩定）
2. 比較不同 Kelly fraction 的回測表現
3. 給出是否使用 Kelly 的建議

使用方法：
    # 快速檢查
    python scripts/validate_kelly.py
    
    # 詳細分析（比較不同 fraction）
    python scripts/validate_kelly.py --detailed
    
    # 指定交易對
    python scripts/validate_kelly.py --symbols BTCUSDT ETHUSDT
    
    # 輸出 JSON
    python scripts/validate_kelly.py --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 將專案加入 path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qtrade.config import load_config
from qtrade.backtest.kelly_validation import (
    kelly_backtest_comparison,
    quick_kelly_check,
    is_strategy_suitable_for_kelly,
    calculate_kelly_stats,
    calculate_kelly_stats_from_portfolio,
    extract_trades_from_portfolio,
)
from qtrade.backtest.run_backtest import run_symbol_backtest
from qtrade.utils.log import get_logger


def get_data_path(data_dir: Path, symbol: str, interval: str) -> Path:
    """構建數據路徑"""
    return data_dir / "binance" / "spot" / interval / f"{symbol}.parquet"

logger = get_logger("kelly_validation")


def main():
    parser = argparse.ArgumentParser(
        description="Kelly 公式驗證",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", "-c",
        default="config/rsi_adx_atr.yaml",
        help="配置檔路徑",
    )
    parser.add_argument(
        "--symbols", "-s",
        nargs="+",
        help="指定交易對",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="詳細分析（比較不同 Kelly fraction）",
    )
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
        help="要比較的 Kelly fractions (default: 0.0 0.25 0.5 0.75 1.0)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="輸出 JSON 格式",
    )
    
    args = parser.parse_args()
    
    # 載入配置
    cfg = load_config(args.config)
    symbols = args.symbols or cfg.market.symbols
    
    print("=" * 60)
    print("  📊 Kelly 公式驗證")
    print("=" * 60)
    print(f"  配置: {args.config}")
    print(f"  策略: {cfg.strategy.name}")
    print(f"  交易對: {', '.join(symbols)}")
    print("=" * 60)
    
    results = {}
    
    for symbol in symbols:
        print(f"\n📈 分析 {symbol}...")
        
        # 獲取數據路徑
        data_path = get_data_path(cfg.data_dir, symbol, cfg.market.interval)
        
        if not data_path.exists():
            print(f"   ❌ 數據不存在: {data_path}")
            continue
        
        # 構建回測配置
        backtest_cfg = {
            "strategy_name": cfg.strategy.name,
            "strategy_params": cfg.strategy.get_params(symbol),
            "initial_cash": cfg.backtest.initial_cash,
            "fee_bps": cfg.backtest.fee_bps,
            "slippage_bps": cfg.backtest.slippage_bps,
        }
        
        if args.detailed:
            # 詳細分析
            try:
                report = kelly_backtest_comparison(
                    symbol=symbol,
                    data_path=data_path,
                    cfg=backtest_cfg,
                    kelly_fractions=args.fractions,
                    strategy_name=cfg.strategy.name,
                )
                
                results[symbol] = {
                    "suitable": report.kelly_stats.is_profitable(),
                    "full_kelly": report.kelly_stats.kelly_pct,
                    "recommended_fraction": report.recommended_fraction,
                    "reason": report.recommendation_reason,
                    "stats": {
                        "win_rate": report.kelly_stats.win_rate,
                        "win_loss_ratio": report.kelly_stats.win_loss_ratio,
                        "edge": report.kelly_stats.edge,
                        "total_trades": report.kelly_stats.total_trades,
                    },
                    "stability": {
                        "kelly": report.kelly_stability,
                        "win_rate": report.win_rate_stability,
                        "edge": report.edge_stability,
                    },
                }
                
                if not args.json:
                    print(report.summary())
                    
            except Exception as e:
                logger.error(f"❌ {symbol} 分析失敗: {e}")
                import traceback
                traceback.print_exc()
        else:
            # 快速檢查
            try:
                result = run_symbol_backtest(
                    symbol, data_path, backtest_cfg, cfg.strategy.name
                )
                # 從 vectorbt Portfolio 提取交易紀錄
                pf = result.get("pf")
                trades = extract_trades_from_portfolio(pf) if pf else []
                
                suitable, reason = is_strategy_suitable_for_kelly(trades)
                stats = calculate_kelly_stats(trades)
                
                results[symbol] = {
                    "suitable": suitable,
                    "reason": reason,
                    "full_kelly": stats.kelly_pct,
                    "recommended": stats.kelly_pct * 0.25 if suitable else 0,
                    "stats": {
                        "win_rate": stats.win_rate,
                        "win_loss_ratio": stats.win_loss_ratio,
                        "edge": stats.edge,
                        "total_trades": stats.total_trades,
                    },
                }
                
                if not args.json:
                    print(f"\n{stats.summary()}")
                    
                    if suitable:
                        print(f"\n   ✅ 適合使用 Kelly")
                        print(f"   建議: Quarter Kelly = {stats.kelly_pct * 0.25:.1%}")
                    else:
                        print(f"\n   ❌ 不建議使用 Kelly")
                        print(f"   原因: {reason}")
                        
            except Exception as e:
                logger.error(f"❌ {symbol} 分析失敗: {e}")
    
    # 總結
    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        print("\n" + "=" * 60)
        print("  📋 總結")
        print("=" * 60)
        
        for symbol, r in results.items():
            suitable = r.get("suitable", False)
            emoji = "✅" if suitable else "❌"
            
            if suitable:
                kelly = r.get("recommended", r.get("full_kelly", 0) * 0.25)
                print(f"  {emoji} {symbol}: 適合 Kelly (建議 {kelly:.1%})")
            else:
                print(f"  {emoji} {symbol}: 不建議 Kelly - {r.get('reason', '')}")
        
        print("=" * 60)
        
        # 配置建議
        all_suitable = all(r.get("suitable", False) for r in results.values())
        if all_suitable and results:
            avg_kelly = sum(r.get("full_kelly", 0) for r in results.values()) / len(results)
            print(f"\n💡 配置建議:")
            print(f"   在 config/{cfg.strategy.name}.yaml 中設置:")
            print(f"   ```yaml")
            print(f"   position_sizing:")
            print(f"     method: \"kelly\"")
            print(f"     kelly_fraction: 0.25  # Quarter Kelly")
            print(f"     min_trades_for_kelly: 30")
            print(f"   ```")
        else:
            print(f"\n💡 建議暫時使用固定倉位，等累積更多交易數據")


if __name__ == "__main__":
    main()
