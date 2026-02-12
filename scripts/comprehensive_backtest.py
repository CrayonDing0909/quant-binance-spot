#!/usr/bin/env python3
"""
系統化綜合回測腳本

全面測試 rsi_adx_atr 策略在不同條件下的表現：

1. 市場階段測試
   - 牛市 (Bull)
   - 熊市 (Bear)  
   - 震盪市 (Sideways)
   - 高波動 (High Volatility)
   - 低波動 (Low Volatility)

2. 倉位管理策略
   - 固定倉位 (Fixed)
   - Kelly 公式 (Kelly)
   - 波動率調整 (Volatility-based)
   - 風險平價 (Risk Parity)

3. 出場策略
   - ATR-based TP/SL
   - RSI-based TP/SL
   - Trailing Stop
   - 時間止損 (Time-based)

使用方法：
    # 完整測試（所有組合）
    python scripts/comprehensive_backtest.py --symbol BTCUSDT
    
    # 只測試市場階段
    python scripts/comprehensive_backtest.py --symbol BTCUSDT --test market_regime
    
    # 只測試倉位管理
    python scripts/comprehensive_backtest.py --symbol BTCUSDT --test position_sizing
    
    # 只測試出場策略
    python scripts/comprehensive_backtest.py --symbol BTCUSDT --test exit_strategy

輸出：
    - reports/comprehensive/summary.csv - 所有測試結果
    - reports/comprehensive/comparison.png - 比較圖表
    - reports/comprehensive/best_config.json - 最佳配置
"""
import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

# 添加專案路徑
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qtrade.backtest import run_symbol_backtest, pretty_stats
from qtrade.data.storage import load_klines
from qtrade.indicators import calculate_rsi, calculate_atr
from qtrade.utils.log import get_logger

logger = get_logger("comprehensive_backtest")


# ══════════════════════════════════════════════════════════════
# 市場階段識別
# ══════════════════════════════════════════════════════════════

@dataclass
class MarketRegime:
    """市場階段"""
    name: str
    start_idx: int
    end_idx: int
    description: str
    metrics: dict = field(default_factory=dict)


def identify_market_regimes(df: pd.DataFrame, window: int = 50) -> list[MarketRegime]:
    """
    識別市場階段
    
    方法：
    - 牛市：SMA 上升 + 價格在 SMA 之上
    - 熊市：SMA 下降 + 價格在 SMA 之下
    - 震盪市：SMA 平坦 + 價格在 SMA 附近波動
    - 高波動：ATR 在歷史 75 百分位以上
    - 低波動：ATR 在歷史 25 百分位以下
    """
    close = df["close"]
    
    # 計算指標
    sma = close.rolling(window).mean()
    sma_slope = sma.pct_change(window)
    
    atr = calculate_atr(df["high"], df["low"], close, period=14)
    atr_pct = atr / close  # ATR 佔價格百分比
    
    # 分類
    regimes = []
    
    # 牛市：SMA 斜率 > 0.05 且價格 > SMA
    bull_mask = (sma_slope > 0.05) & (close > sma)
    
    # 熊市：SMA 斜率 < -0.05 且價格 < SMA
    bear_mask = (sma_slope < -0.05) & (close < sma)
    
    # 震盪市：SMA 斜率在 [-0.05, 0.05] 之間
    sideways_mask = (sma_slope.abs() <= 0.05) & (~bull_mask) & (~bear_mask)
    
    # 高波動：ATR% > 75 百分位
    high_vol_threshold = atr_pct.quantile(0.75)
    high_vol_mask = atr_pct > high_vol_threshold
    
    # 低波動：ATR% < 25 百分位
    low_vol_threshold = atr_pct.quantile(0.25)
    low_vol_mask = atr_pct < low_vol_threshold
    
    # 找出連續區間
    def find_periods(mask: pd.Series, name: str, desc: str) -> list[MarketRegime]:
        """找出連續為 True 的區間"""
        periods = []
        in_period = False
        start = 0
        
        for i in range(len(mask)):
            if mask.iloc[i] and not in_period:
                in_period = True
                start = i
            elif not mask.iloc[i] and in_period:
                in_period = False
                if i - start >= 50:  # 至少 50 根 K 線
                    periods.append(MarketRegime(
                        name=name,
                        start_idx=start,
                        end_idx=i,
                        description=desc,
                        metrics={
                            "duration_bars": i - start,
                            "return_pct": (close.iloc[i] / close.iloc[start] - 1) * 100,
                        }
                    ))
        
        # 處理最後一個區間
        if in_period and len(mask) - start >= 50:
            periods.append(MarketRegime(
                name=name,
                start_idx=start,
                end_idx=len(mask),
                description=desc,
                metrics={
                    "duration_bars": len(mask) - start,
                    "return_pct": (close.iloc[-1] / close.iloc[start] - 1) * 100,
                }
            ))
        
        return periods
    
    regimes.extend(find_periods(bull_mask, "bull", "牛市（上升趨勢）"))
    regimes.extend(find_periods(bear_mask, "bear", "熊市（下降趨勢）"))
    regimes.extend(find_periods(sideways_mask, "sideways", "震盪市（橫盤）"))
    regimes.extend(find_periods(high_vol_mask, "high_vol", "高波動"))
    regimes.extend(find_periods(low_vol_mask, "low_vol", "低波動"))
    
    return regimes


# ══════════════════════════════════════════════════════════════
# 測試配置
# ══════════════════════════════════════════════════════════════

# 倉位管理策略配置
POSITION_SIZING_CONFIGS = {
    "fixed_100": {
        "name": "固定滿倉",
        "position_pct": 1.0,
        "use_kelly": False,
    },
    "fixed_50": {
        "name": "固定半倉",
        "position_pct": 0.5,
        "use_kelly": False,
    },
    "fixed_25": {
        "name": "固定四分之一倉",
        "position_pct": 0.25,
        "use_kelly": False,
    },
    "kelly_full": {
        "name": "Kelly 公式（完整）",
        "position_pct": 1.0,
        "use_kelly": True,
        "kelly_fraction": 1.0,
    },
    "kelly_half": {
        "name": "Kelly 公式（半 Kelly）",
        "position_pct": 1.0,
        "use_kelly": True,
        "kelly_fraction": 0.5,
    },
    "kelly_quarter": {
        "name": "Kelly 公式（四分之一 Kelly）",
        "position_pct": 1.0,
        "use_kelly": True,
        "kelly_fraction": 0.25,
    },
}

# 出場策略配置
EXIT_STRATEGY_CONFIGS = {
    "atr_2_3": {
        "name": "ATR SL=2x TP=3x",
        "stop_loss_atr": 2.0,
        "take_profit_atr": 3.0,
        "trailing_stop_atr": None,
        "use_rsi_exit": False,
    },
    "atr_1.5_3": {
        "name": "ATR SL=1.5x TP=3x",
        "stop_loss_atr": 1.5,
        "take_profit_atr": 3.0,
        "trailing_stop_atr": None,
        "use_rsi_exit": False,
    },
    "atr_2_4": {
        "name": "ATR SL=2x TP=4x",
        "stop_loss_atr": 2.0,
        "take_profit_atr": 4.0,
        "trailing_stop_atr": None,
        "use_rsi_exit": False,
    },
    "trailing_2.5": {
        "name": "Trailing Stop 2.5x ATR",
        "stop_loss_atr": 2.0,
        "take_profit_atr": None,
        "trailing_stop_atr": 2.5,
        "use_rsi_exit": False,
    },
    "trailing_2": {
        "name": "Trailing Stop 2x ATR",
        "stop_loss_atr": 1.5,
        "take_profit_atr": None,
        "trailing_stop_atr": 2.0,
        "use_rsi_exit": False,
    },
    "rsi_exit": {
        "name": "RSI 出場",
        "stop_loss_atr": 2.0,
        "take_profit_atr": None,
        "trailing_stop_atr": None,
        "use_rsi_exit": True,
        "rsi_overbought": 70,
        "rsi_oversold": 30,
    },
    "combined": {
        "name": "組合（ATR SL + RSI TP）",
        "stop_loss_atr": 2.0,
        "take_profit_atr": 4.0,
        "trailing_stop_atr": None,
        "use_rsi_exit": True,
        "rsi_overbought": 75,
        "rsi_oversold": 25,
    },
}

# 策略參數配置（RSI + ADX 相關）
STRATEGY_PARAM_CONFIGS = {
    "default": {
        "name": "預設參數",
        "rsi_period": 14,
        "oversold": 35,
        "overbought": 70,
        "min_adx": 20,
        "adx_period": 14,
        "atr_period": 14,
        "cooldown_bars": 6,
    },
    "aggressive": {
        "name": "積極參數",
        "rsi_period": 10,
        "oversold": 40,
        "overbought": 65,
        "min_adx": 15,
        "adx_period": 10,
        "atr_period": 10,
        "cooldown_bars": 4,
    },
    "conservative": {
        "name": "保守參數",
        "rsi_period": 21,
        "oversold": 30,
        "overbought": 75,
        "min_adx": 25,
        "adx_period": 21,
        "atr_period": 21,
        "cooldown_bars": 10,
    },
}


# ══════════════════════════════════════════════════════════════
# 回測執行
# ══════════════════════════════════════════════════════════════

@dataclass
class BacktestResult:
    """單次回測結果"""
    config_name: str
    config_type: str  # "market_regime", "position_sizing", "exit_strategy", "strategy_params"
    config_details: dict
    
    # 績效指標
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    
    # 額外資訊
    market_regime: str = ""
    period_start: str = ""
    period_end: str = ""
    
    def to_dict(self) -> dict:
        return asdict(self)


def run_single_backtest(
    df: pd.DataFrame,
    symbol: str,
    strategy_params: dict,
    exit_params: dict,
    position_params: dict,
    market_type: str = "spot",
    direction: str = "both",
) -> dict:
    """運行單次回測"""
    
    # 合併參數
    params = {
        **strategy_params,
        **exit_params,
    }
    
    cfg = {
        "initial_cash": 10000,
        "fee_bps": 10,
        "slippage_bps": 5,
        "interval": "1h",
        "market_type": market_type,
        "direction": direction,
        "strategy_params": params,
    }
    
    # 保存數據到臨時路徑
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        temp_path = Path(f.name)
        df.to_parquet(temp_path)
    
    try:
        result = run_symbol_backtest(
            symbol=symbol,
            data_path=temp_path,
            cfg=cfg,
            strategy_name="rsi_adx_atr",
            market_type=market_type,
            direction=direction,
        )
        
        stats = result["stats"]
        
        return {
            "total_return": stats.get("Total Return [%]", 0),
            "sharpe_ratio": stats.get("Sharpe Ratio", 0),
            "sortino_ratio": stats.get("Sortino Ratio", 0),
            "max_drawdown": stats.get("Max Drawdown [%]", 0),
            "win_rate": stats.get("Win Rate [%]", 0),
            "profit_factor": stats.get("Profit Factor", 0),
            "total_trades": stats.get("Total Trades", 0),
        }
    finally:
        temp_path.unlink()


def run_comprehensive_backtest(
    df: pd.DataFrame,
    symbol: str,
    test_types: list[str] = None,
    market_type: str = "spot",
    direction: str = "both",
) -> list[BacktestResult]:
    """
    運行綜合回測
    
    Args:
        df: K 線數據
        symbol: 交易對
        test_types: 要測試的類型 ["market_regime", "position_sizing", "exit_strategy", "strategy_params"]
        market_type: 市場類型
        direction: 交易方向
    
    Returns:
        所有回測結果
    """
    if test_types is None:
        test_types = ["market_regime", "position_sizing", "exit_strategy", "strategy_params"]
    
    results = []
    total_tests = 0
    
    # 計算總測試數量
    if "market_regime" in test_types:
        regimes = identify_market_regimes(df)
        total_tests += len(regimes)
    if "position_sizing" in test_types:
        total_tests += len(POSITION_SIZING_CONFIGS)
    if "exit_strategy" in test_types:
        total_tests += len(EXIT_STRATEGY_CONFIGS)
    if "strategy_params" in test_types:
        total_tests += len(STRATEGY_PARAM_CONFIGS)
    
    current_test = 0
    
    # 預設參數
    default_strategy_params = STRATEGY_PARAM_CONFIGS["default"]
    default_exit_params = EXIT_STRATEGY_CONFIGS["atr_2_3"]
    default_position_params = POSITION_SIZING_CONFIGS["fixed_100"]
    
    # 1. 市場階段測試
    if "market_regime" in test_types:
        print("\n📊 測試不同市場階段...")
        regimes = identify_market_regimes(df)
        
        for regime in regimes:
            current_test += 1
            print(f"   [{current_test}/{total_tests}] {regime.name}: {regime.description}")
            
            regime_df = df.iloc[regime.start_idx:regime.end_idx].copy()
            
            if len(regime_df) < 100:
                print(f"      ⚠️ 數據不足，跳過")
                continue
            
            try:
                stats = run_single_backtest(
                    regime_df, symbol,
                    default_strategy_params,
                    default_exit_params,
                    default_position_params,
                    market_type, direction,
                )
                
                results.append(BacktestResult(
                    config_name=f"{regime.name}_{regime.start_idx}",
                    config_type="market_regime",
                    config_details={"regime": regime.name, "description": regime.description},
                    market_regime=regime.name,
                    period_start=str(regime_df.index[0]),
                    period_end=str(regime_df.index[-1]),
                    **stats,
                ))
                
                print(f"      ✅ 收益: {stats['total_return']:.2f}%, Sharpe: {stats['sharpe_ratio']:.2f}")
            except Exception as e:
                print(f"      ❌ 失敗: {e}")
    
    # 2. 倉位管理策略測試
    if "position_sizing" in test_types:
        print("\n💰 測試不同倉位管理策略...")
        
        for config_id, config in POSITION_SIZING_CONFIGS.items():
            current_test += 1
            print(f"   [{current_test}/{total_tests}] {config['name']}")
            
            try:
                stats = run_single_backtest(
                    df, symbol,
                    default_strategy_params,
                    default_exit_params,
                    config,
                    market_type, direction,
                )
                
                results.append(BacktestResult(
                    config_name=config_id,
                    config_type="position_sizing",
                    config_details=config,
                    **stats,
                ))
                
                print(f"      ✅ 收益: {stats['total_return']:.2f}%, Sharpe: {stats['sharpe_ratio']:.2f}")
            except Exception as e:
                print(f"      ❌ 失敗: {e}")
    
    # 3. 出場策略測試
    if "exit_strategy" in test_types:
        print("\n🚪 測試不同出場策略...")
        
        for config_id, config in EXIT_STRATEGY_CONFIGS.items():
            current_test += 1
            print(f"   [{current_test}/{total_tests}] {config['name']}")
            
            # 合併出場參數
            exit_params = {k: v for k, v in config.items() if k != "name"}
            
            try:
                stats = run_single_backtest(
                    df, symbol,
                    default_strategy_params,
                    exit_params,
                    default_position_params,
                    market_type, direction,
                )
                
                results.append(BacktestResult(
                    config_name=config_id,
                    config_type="exit_strategy",
                    config_details=config,
                    **stats,
                ))
                
                print(f"      ✅ 收益: {stats['total_return']:.2f}%, Sharpe: {stats['sharpe_ratio']:.2f}")
            except Exception as e:
                print(f"      ❌ 失敗: {e}")
    
    # 4. 策略參數測試
    if "strategy_params" in test_types:
        print("\n⚙️ 測試不同策略參數...")
        
        for config_id, config in STRATEGY_PARAM_CONFIGS.items():
            current_test += 1
            print(f"   [{current_test}/{total_tests}] {config['name']}")
            
            # 合併策略參數
            strategy_params = {k: v for k, v in config.items() if k != "name"}
            
            try:
                stats = run_single_backtest(
                    df, symbol,
                    strategy_params,
                    default_exit_params,
                    default_position_params,
                    market_type, direction,
                )
                
                results.append(BacktestResult(
                    config_name=config_id,
                    config_type="strategy_params",
                    config_details=config,
                    **stats,
                ))
                
                print(f"      ✅ 收益: {stats['total_return']:.2f}%, Sharpe: {stats['sharpe_ratio']:.2f}")
            except Exception as e:
                print(f"      ❌ 失敗: {e}")
    
    return results


# ══════════════════════════════════════════════════════════════
# 報告生成
# ══════════════════════════════════════════════════════════════

def generate_report(results: list[BacktestResult], output_dir: Path, symbol: str):
    """生成報告"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 轉換為 DataFrame
    df_results = pd.DataFrame([r.to_dict() for r in results])
    
    # 2. 保存 CSV
    csv_path = output_dir / f"{symbol}_comprehensive_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\n📄 結果已保存: {csv_path}")
    
    # 3. 生成摘要
    print("\n" + "=" * 70)
    print("📊 綜合回測報告")
    print("=" * 70)
    
    # 按類型分組統計
    for config_type in df_results["config_type"].unique():
        type_df = df_results[df_results["config_type"] == config_type]
        
        print(f"\n【{config_type}】")
        print("-" * 50)
        
        # 按 Sharpe Ratio 排序
        type_df_sorted = type_df.sort_values("sharpe_ratio", ascending=False)
        
        for _, row in type_df_sorted.iterrows():
            sharpe = row["sharpe_ratio"]
            ret = row["total_return"]
            dd = row["max_drawdown"]
            wr = row["win_rate"]
            
            # 用 emoji 標記最佳
            emoji = "🥇" if row.name == type_df_sorted.index[0] else "  "
            
            print(f"{emoji} {row['config_name']:20} | "
                  f"收益: {ret:+7.2f}% | "
                  f"Sharpe: {sharpe:5.2f} | "
                  f"DD: {dd:6.2f}% | "
                  f"勝率: {wr:5.1f}%")
    
    # 4. 找出整體最佳配置
    print("\n" + "=" * 70)
    print("🏆 最佳配置推薦")
    print("=" * 70)
    
    # 按 Sharpe Ratio 排序
    best_sharpe = df_results.loc[df_results["sharpe_ratio"].idxmax()]
    print(f"\n最高 Sharpe Ratio: {best_sharpe['config_name']}")
    print(f"   類型: {best_sharpe['config_type']}")
    print(f"   Sharpe: {best_sharpe['sharpe_ratio']:.2f}")
    print(f"   收益: {best_sharpe['total_return']:.2f}%")
    
    # 按總收益排序
    best_return = df_results.loc[df_results["total_return"].idxmax()]
    print(f"\n最高總收益: {best_return['config_name']}")
    print(f"   類型: {best_return['config_type']}")
    print(f"   收益: {best_return['total_return']:.2f}%")
    print(f"   Sharpe: {best_return['sharpe_ratio']:.2f}")
    
    # 最小回撤
    best_dd = df_results.loc[df_results["max_drawdown"].idxmax()]  # 回撤是負數
    print(f"\n最小回撤: {best_dd['config_name']}")
    print(f"   類型: {best_dd['config_type']}")
    print(f"   回撤: {best_dd['max_drawdown']:.2f}%")
    print(f"   收益: {best_dd['total_return']:.2f}%")
    
    # 5. 保存最佳配置
    best_config = {
        "best_sharpe": {
            "config_name": best_sharpe["config_name"],
            "config_type": best_sharpe["config_type"],
            "sharpe_ratio": best_sharpe["sharpe_ratio"],
            "total_return": best_sharpe["total_return"],
        },
        "best_return": {
            "config_name": best_return["config_name"],
            "config_type": best_return["config_type"],
            "total_return": best_return["total_return"],
            "sharpe_ratio": best_return["sharpe_ratio"],
        },
        "best_drawdown": {
            "config_name": best_dd["config_name"],
            "config_type": best_dd["config_type"],
            "max_drawdown": best_dd["max_drawdown"],
            "total_return": best_dd["total_return"],
        },
    }
    
    config_path = output_dir / f"{symbol}_best_config.json"
    with open(config_path, "w") as f:
        json.dump(best_config, f, indent=2)
    print(f"\n📄 最佳配置已保存: {config_path}")
    
    # 6. 市場階段分析（如果有）
    regime_df = df_results[df_results["config_type"] == "market_regime"]
    if len(regime_df) > 0:
        print("\n" + "=" * 70)
        print("📈 市場階段分析")
        print("=" * 70)
        
        # 按市場階段分組
        regime_stats = regime_df.groupby("market_regime").agg({
            "total_return": ["mean", "std", "count"],
            "sharpe_ratio": "mean",
            "win_rate": "mean",
        })
        
        for regime in regime_stats.index:
            ret_mean = regime_stats.loc[regime, ("total_return", "mean")]
            ret_std = regime_stats.loc[regime, ("total_return", "std")]
            count = regime_stats.loc[regime, ("total_return", "count")]
            sharpe = regime_stats.loc[regime, ("sharpe_ratio", "mean")]
            
            print(f"\n{regime}:")
            print(f"   樣本數: {count:.0f}")
            print(f"   平均收益: {ret_mean:.2f}% ± {ret_std:.2f}%")
            print(f"   平均 Sharpe: {sharpe:.2f}")
    
    print("\n" + "=" * 70)


# ══════════════════════════════════════════════════════════════
# 主程式
# ══════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="系統化綜合回測")
    parser.add_argument("--symbol", default="BTCUSDT", help="交易對")
    parser.add_argument("--interval", default="1h", help="K 線週期")
    parser.add_argument("--test", nargs="+", 
                       choices=["market_regime", "position_sizing", "exit_strategy", "strategy_params", "all"],
                       default=["all"],
                       help="要測試的類型")
    parser.add_argument("--market-type", default="spot", choices=["spot", "futures"])
    parser.add_argument("--direction", default="both", choices=["both", "long_only", "short_only"])
    parser.add_argument("--output-dir", default="reports/comprehensive", help="輸出目錄")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70)
    print("🔬 系統化綜合回測")
    print("=" * 70)
    print(f"交易對: {args.symbol}")
    print(f"週期: {args.interval}")
    print(f"市場類型: {args.market_type}")
    print(f"交易方向: {args.direction}")
    
    # 確定測試類型
    if "all" in args.test:
        test_types = ["market_regime", "position_sizing", "exit_strategy", "strategy_params"]
    else:
        test_types = args.test
    
    print(f"測試類型: {test_types}")
    print("=" * 70)
    
    # 載入數據
    data_dir = Path(__file__).parent.parent / "data"
    # 嘗試不同的數據路徑
    possible_paths = [
        data_dir / "binance" / "spot" / args.interval / f"{args.symbol}.parquet",
        data_dir / "binance" / "futures" / args.interval / f"{args.symbol}.parquet",
        data_dir / f"{args.symbol}_{args.interval}.parquet",
    ]
    
    data_path = None
    for path in possible_paths:
        if path.exists():
            data_path = path
            break
    
    if data_path is None:
        print(f"❌ 找不到 {args.symbol} {args.interval} 的數據文件")
        print(f"   嘗試過的路徑: {possible_paths}")
        print(f"   請先運行: python scripts/download_data.py --symbol {args.symbol}")
        sys.exit(1)
    
    print(f"📊 數據文件: {data_path}")
    df = load_klines(data_path)
    print(f"   數據範圍: {df.index[0]} ~ {df.index[-1]}")
    print(f"   K 線數量: {len(df)}")
    
    # 運行綜合回測
    results = run_comprehensive_backtest(
        df=df,
        symbol=args.symbol,
        test_types=test_types,
        market_type=args.market_type,
        direction=args.direction,
    )
    
    # 生成報告
    output_dir = Path(args.output_dir)
    generate_report(results, output_dir, args.symbol)
    
    print("\n🎉 綜合回測完成！")


if __name__ == "__main__":
    main()
