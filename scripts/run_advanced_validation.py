#!/usr/bin/env python3
"""
進階驗證腳本

整合 Cross-Asset 驗證 + Monte Carlo 模擬，提供完整的策略穩健性分析。

使用方式:
    python scripts/run_advanced_validation.py --config config/rsi_adx_atr.yaml

功能:
    1. Cross-Asset 驗證
       - Leave-One-Asset-Out (LOAO)
       - 相關性分層驗證
       - 市場狀態驗證

    2. Monte Carlo 模擬
       - VaR / CVaR 計算
       - Bootstrap 績效信賴區間
       - Drawdown 分布分析
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


# 確保可以 import qtrade
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qtrade.config import load_config

from qtrade.backtest import (
    run_symbol_backtest,
    trade_analysis,
    leave_one_asset_out,
    correlation_stratified_validation,
    market_regime_validation,
    ValidationResultAnalyzer,
    CrossAssetValidationConfig,
)
from qtrade.risk import (
    MonteCarloSimulator,
    MonteCarloConfig,
    BootstrapConfig,
    bootstrap_strategy_ci,
    monte_carlo_var,
    simulate_strategy_outcomes,
)
from qtrade.data.storage import load_klines


# ══════════════════════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ValidationPipelineConfig:
    """驗證流程配置"""
    # 資產設定
    symbols: List[str]
    data_dir: Path
    
    # 驗證選項
    run_loao: bool = True
    run_correlation: bool = True
    run_regime: bool = True
    run_monte_carlo: bool = True
    
    # 數據分割設定（專業量化必備）
    train_ratio: float = 0.6  # 訓練集比例
    val_ratio: float = 0.2    # 驗證集比例（用於選模型）
    # test_ratio = 1 - train_ratio - val_ratio = 0.2（最終驗證）
    use_test_set_only: bool = True  # Monte Carlo 只用 Test Set
    
    # Monte Carlo 設定
    mc_n_simulations: int = 10000
    mc_confidence_levels: tuple = (0.95, 0.99)
    
    # 輸出設定
    output_dir: Optional[Path] = None
    verbose: bool = True


# ══════════════════════════════════════════════════════════════════════════════
# 數據分割工具
# ══════════════════════════════════════════════════════════════════════════════

def train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> Dict[str, pd.DataFrame]:
    """
    時間序列的 Train/Val/Test 分割
    
    專業量化交易的標準做法：
    - Train: 用於參數優化
    - Val: 用於選擇最佳模型/參數
    - Test: 最終驗證（只碰一次！）
    
    Args:
        df: K 線數據
        train_ratio: 訓練集比例
        val_ratio: 驗證集比例
    
    Returns:
        {"train": df, "val": df, "test": df, "periods": {...}}
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    return {
        "train": train_df,
        "val": val_df,
        "test": test_df,
        "periods": {
            "train": f"{train_df.index[0].strftime('%Y-%m-%d')} → {train_df.index[-1].strftime('%Y-%m-%d')}" if len(train_df) > 0 else "N/A",
            "val": f"{val_df.index[0].strftime('%Y-%m-%d')} → {val_df.index[-1].strftime('%Y-%m-%d')}" if len(val_df) > 0 else "N/A",
            "test": f"{test_df.index[0].strftime('%Y-%m-%d')} → {test_df.index[-1].strftime('%Y-%m-%d')}" if len(test_df) > 0 else "N/A",
        },
        "sizes": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        }
    }


def save_split_data(
    split_data: Dict[str, pd.DataFrame],
    symbol: str,
    data_dir: Path,
) -> Dict[str, Path]:
    """
    保存分割後的數據到臨時文件
    
    Returns:
        {"train": Path, "val": Path, "test": Path}
    """
    paths = {}
    for split_name in ["train", "val", "test"]:
        df = split_data[split_name]
        if len(df) > 0:
            path = data_dir / f"_temp_{symbol}_{split_name}.parquet"
            df.to_parquet(path)
            paths[split_name] = path
    return paths


def cleanup_split_data(paths: Dict[str, Path]) -> None:
    """清理臨時文件"""
    for path in paths.values():
        if path.exists():
            path.unlink()


# ══════════════════════════════════════════════════════════════════════════════
# 報告生成器
# ══════════════════════════════════════════════════════════════════════════════

class ValidationReporter:
    """驗證結果報告生成器"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def print_header(self, title: str, char: str = "═", width: int = 60):
        """打印區塊標題"""
        if self.verbose:
            print(f"\n{char * width}")
            print(f"  {title}")
            print(f"{char * width}")
    
    def print_subheader(self, title: str):
        """打印子標題"""
        if self.verbose:
            print(f"\n▶ {title}")
            print("-" * 50)
    
    def print_metric(self, name: str, value, format_spec: str = ""):
        """打印指標"""
        if self.verbose:
            if format_spec:
                print(f"  {name}: {value:{format_spec}}")
            else:
                print(f"  {name}: {value}")
    
    def print_table(self, df: pd.DataFrame, max_rows: int = 20):
        """打印表格"""
        if self.verbose:
            print(df.head(max_rows).to_string())
    
    def print_warning(self, message: str):
        """打印警告"""
        if self.verbose:
            print(f"  ⚠️  {message}")
    
    def print_success(self, message: str):
        """打印成功訊息"""
        if self.verbose:
            print(f"  ✅ {message}")
    
    def print_error(self, message: str):
        """打印錯誤"""
        if self.verbose:
            print(f"  ❌ {message}")


# ══════════════════════════════════════════════════════════════════════════════
# 驗證流程
# ══════════════════════════════════════════════════════════════════════════════

class ValidationPipeline:
    """
    驗證流程管理器
    
    整合所有驗證步驟，提供統一的執行介面。
    """
    
    def __init__(
        self,
        pipeline_config: ValidationPipelineConfig,
        strategy_config: dict,
        backtest_func=None,
        data_loader=None,
    ):
        self.pipeline_config = pipeline_config
        self.strategy_config = strategy_config
        self.reporter = ValidationReporter(pipeline_config.verbose)
        
        # 自定義函數（用於注入專案特定的回測邏輯）
        self._backtest_func = backtest_func
        self._data_loader = data_loader
        
        # 建立數據路徑映射
        # 支援兩種路徑格式：
        # 1. data/binance/spot/1h/BTCUSDT.parquet (專案標準格式)
        # 2. data/BTCUSDT_1h.parquet (簡化格式)
        self.data_paths = {}
        for s in pipeline_config.symbols:
            # 優先使用專案標準格式
            standard_path = pipeline_config.data_dir / f"{s}.parquet"
            simple_path = pipeline_config.data_dir / f"{s}_1h.parquet"
            
            if standard_path.exists():
                self.data_paths[s] = standard_path
            elif simple_path.exists():
                self.data_paths[s] = simple_path
            else:
                # 記錄預期路徑，後續會報告找不到
                self.data_paths[s] = standard_path
        
        # 過濾存在的資產
        self.available_symbols = [
            s for s in pipeline_config.symbols
            if self.data_paths[s].exists()
        ]
        
        if len(self.available_symbols) < len(pipeline_config.symbols):
            missing = set(pipeline_config.symbols) - set(self.available_symbols)
            self.reporter.print_warning(f"找不到數據: {missing}")
    
    def run(self) -> dict:
        """
        執行完整驗證流程
        
        Returns:
            包含所有驗證結果的字典
        """
        results = {}
        
        self.reporter.print_header("🔬 進階策略驗證", "═")
        self.reporter.print_metric("策略", self.strategy_config.get("strategy_name", "unknown"))
        self.reporter.print_metric("資產數量", len(self.available_symbols))
        self.reporter.print_metric("可用資產", self.available_symbols)
        
        # 1. Cross-Asset 驗證
        if self.pipeline_config.run_loao:
            results["loao"] = self._run_loao_validation()
        
        if self.pipeline_config.run_correlation:
            results["correlation"] = self._run_correlation_validation()
        
        if self.pipeline_config.run_regime:
            results["regime"] = self._run_regime_validation()
        
        # 2. Monte Carlo 模擬
        if self.pipeline_config.run_monte_carlo:
            results["monte_carlo"] = self._run_monte_carlo()
        
        # 3. 總結報告
        self._print_summary(results)
        
        return results
    
    def _run_loao_validation(self) -> dict:
        """執行 Leave-One-Asset-Out 驗證"""
        self.reporter.print_header("📊 Leave-One-Asset-Out 驗證", "─")
        
        if len(self.available_symbols) < 3:
            self.reporter.print_warning("資產數量不足（需至少 3 個）")
            return {}
        
        try:
            result = leave_one_asset_out(
                symbols=self.available_symbols,
                data_paths=self.data_paths,
                cfg=self.strategy_config,
                backtest_func=self._backtest_func,
                data_loader=self._data_loader,
                parallel=True,
            )
            
            # 打印結果
            self.reporter.print_subheader("驗證結果")
            self.reporter.print_table(result.to_dataframe())
            
            self.reporter.print_subheader("摘要")
            summary = ValidationResultAnalyzer.summarize(result)
            self.reporter.print_metric("穩健性等級", result.robustness_level.value)
            self.reporter.print_metric("平均績效衰退", f"{result.avg_sharpe_degradation:.1%}")
            self.reporter.print_metric("衰退標準差", f"{result.std_sharpe_degradation:.2f}")
            
            if result.overfitted_assets:
                self.reporter.print_warning(f"可能過擬合: {list(result.overfitted_assets)}")
            
            # 建議
            recommendations = ValidationResultAnalyzer.get_recommendations(result)
            self.reporter.print_subheader("建議")
            for rec in recommendations:
                print(f"  • {rec}")
            
            return {"result": result, "summary": summary}
        
        except Exception as e:
            self.reporter.print_error(f"LOAO 驗證失敗: {e}")
            return {"error": str(e)}
    
    def _run_correlation_validation(self) -> dict:
        """執行相關性分層驗證"""
        self.reporter.print_header("🔗 相關性分層驗證", "─")
        
        if len(self.available_symbols) < 4:
            self.reporter.print_warning("資產數量不足（需至少 4 個）")
            return {}
        
        try:
            result = correlation_stratified_validation(
                symbols=self.available_symbols,
                data_paths=self.data_paths,
                cfg=self.strategy_config,
                n_groups=min(3, len(self.available_symbols) // 2),
                backtest_func=self._backtest_func,
                data_loader=self._data_loader,
            )
            
            self.reporter.print_subheader("驗證結果")
            self.reporter.print_table(result.to_dataframe())
            
            self.reporter.print_metric("穩健性等級", result.robustness_level.value)
            
            return {"result": result}
        
        except Exception as e:
            self.reporter.print_error(f"相關性驗證失敗: {e}")
            return {"error": str(e)}
    
    def _run_regime_validation(self) -> dict:
        """執行市場狀態驗證（使用 Train/Test 分割）"""
        self.reporter.print_header("📈 市場狀態驗證", "─")
        
        try:
            # 準備數據路徑（使用 Test Set）
            temp_paths = {}
            if self.pipeline_config.use_test_set_only:
                self.reporter.print_subheader("數據分割")
                for symbol in self.available_symbols:
                    data_path = self.data_paths[symbol]
                    if self._data_loader:
                        full_df = self._data_loader(data_path)
                    else:
                        full_df = load_klines(data_path)
                    
                    split_data = train_val_test_split(
                        full_df,
                        train_ratio=self.pipeline_config.train_ratio,
                        val_ratio=self.pipeline_config.val_ratio,
                    )
                    
                    # 使用 Test Set
                    test_df = split_data["test"]
                    temp_path = data_path.parent / f"_temp_{symbol}_test.parquet"
                    test_df.to_parquet(temp_path)
                    temp_paths[symbol] = temp_path
                    self.reporter.print_metric(f"{symbol} Test Set", f"{split_data['periods']['test']}")
                
                data_paths_to_use = temp_paths
            else:
                data_paths_to_use = self.data_paths
            
            try:
                results_list, summary_df = market_regime_validation(
                    symbols=self.available_symbols,
                    data_paths=data_paths_to_use,
                    cfg=self.strategy_config,
                    indicator="volatility",
                    backtest_func=self._backtest_func,
                    data_loader=self._data_loader,
                )
            finally:
                # 清理臨時文件
                for path in temp_paths.values():
                    path.unlink(missing_ok=True)
            
            if not summary_df.empty:
                self.reporter.print_subheader("不同市場狀態下的表現")
                self.reporter.print_table(summary_df)
                
                # 分析高/低波動性表現差異
                high_vol = summary_df[summary_df["regime"].str.contains("high")]
                low_vol = summary_df[summary_df["regime"].str.contains("low")]
                
                if not high_vol.empty and not low_vol.empty:
                    self.reporter.print_subheader("波動性狀態比較")
                    self.reporter.print_metric(
                        "高波動 Sharpe",
                        f"{high_vol['sharpe'].mean():.2f}"
                    )
                    self.reporter.print_metric(
                        "低波動 Sharpe",
                        f"{low_vol['sharpe'].mean():.2f}"
                    )
            
            return {"results": results_list, "summary": summary_df}
        
        except Exception as e:
            self.reporter.print_error(f"市場狀態驗證失敗: {e}")
            return {"error": str(e)}
    
    def _run_monte_carlo(self) -> dict:
        """執行 Monte Carlo 模擬（使用 Train/Test 分割）"""
        self.reporter.print_header("🎲 Monte Carlo 模擬", "─")
        
        # 選擇第一個可用資產進行模擬
        if not self.available_symbols:
            self.reporter.print_warning("無可用資產")
            return {}
        
        symbol = self.available_symbols[0]
        data_path = self.data_paths[symbol]
        
        try:
            # 載入原始數據並分割
            if self._data_loader:
                full_df = self._data_loader(data_path)
            else:
                full_df = load_klines(data_path)
            
            # 數據分割
            split_data = train_val_test_split(
                full_df,
                train_ratio=self.pipeline_config.train_ratio,
                val_ratio=self.pipeline_config.val_ratio,
            )
            
            self.reporter.print_subheader(f"數據分割 ({symbol})")
            self.reporter.print_metric("Train", f"{split_data['periods']['train']} ({split_data['sizes']['train']} bars)")
            self.reporter.print_metric("Val", f"{split_data['periods']['val']} ({split_data['sizes']['val']} bars)")
            self.reporter.print_metric("Test", f"{split_data['periods']['test']} ({split_data['sizes']['test']} bars)")
            
            # 決定使用哪個數據集
            if self.pipeline_config.use_test_set_only:
                eval_df = split_data["test"]
                eval_name = "Test Set (Out-of-Sample)"
            else:
                eval_df = full_df
                eval_name = "Full Data (In-Sample)"
            
            self.reporter.print_subheader(f"分析資產: {symbol} - {eval_name}")
            
            if len(eval_df) < 100:
                self.reporter.print_warning(f"數據量不足（{len(eval_df)} bars），跳過")
                return {}
            
            # 保存臨時數據
            temp_path = data_path.parent / f"_temp_{symbol}_eval.parquet"
            eval_df.to_parquet(temp_path)
            
            try:
                # 執行回測
                if self._backtest_func:
                    res = self._backtest_func(
                        symbol,
                        temp_path,
                        self.strategy_config,
                        self.strategy_config.get("strategy_name"),
                    )
                else:
                    res = run_symbol_backtest(
                        symbol,
                        temp_path,
                        self.strategy_config,
                        self.strategy_config.get("strategy_name"),
                    )
            finally:
                temp_path.unlink(missing_ok=True)
            
            pf = res["pf"]
            
            # 計算日收益率（相容較新版本的 pandas）
            try:
                daily_returns = pf.daily_returns()
            except TypeError:
                # pandas 2.0+ 移除了 resample 的 axis 參數
                # 使用替代方式計算
                equity = pf.value()
                daily_returns = equity.resample('D').last().pct_change().dropna()
            
            # 取得交易數據
            trades_df = trade_analysis(pf)
            
            results = {}
            
            # 1. Monte Carlo VaR
            self.reporter.print_subheader("VaR / CVaR 分析")
            
            mc_config = MonteCarloConfig(
                n_simulations=self.pipeline_config.mc_n_simulations,
                confidence_levels=self.pipeline_config.mc_confidence_levels,
            )
            simulator = MonteCarloSimulator(mc_config=mc_config)
            
            var_result = simulator.calculate_var(
                daily_returns,
                portfolio_value=self.strategy_config["initial_cash"],
            )
            
            for conf in self.pipeline_config.mc_confidence_levels:
                self.reporter.print_metric(
                    f"{conf*100:.0f}% VaR",
                    f"${var_result.get_var(conf):,.0f}"
                )
                self.reporter.print_metric(
                    f"{conf*100:.0f}% CVaR",
                    f"${var_result.get_cvar(conf):,.0f}"
                )
            
            results["var"] = var_result
            
            # 2. Bootstrap 績效信賴區間
            if not trades_df.empty and "Return [%]" in trades_df.columns:
                trade_returns = trades_df["Return [%]"] / 100
                
                self.reporter.print_subheader("Bootstrap 績效信賴區間 (95%)")
                
                ci = bootstrap_strategy_ci(
                    trade_returns,
                    confidence=0.95,
                    n_simulations=self.pipeline_config.mc_n_simulations,
                )
                
                for metric, (lower, median, upper) in ci.items():
                    self.reporter.print_metric(
                        metric,
                        f"[{lower:.2%}, {median:.2%}, {upper:.2%}]"
                    )
                
                results["bootstrap_ci"] = ci
            
            # 3. 策略結果分布
            self.reporter.print_subheader("策略結果分布模擬")
            
            outcomes = simulate_strategy_outcomes(
                daily_returns,
                n_simulations=self.pipeline_config.mc_n_simulations,
            )
            
            self.reporter.print_metric(
                "期望最終收益",
                f"{outcomes['percentiles']['final_return'][50]:.1%}"
            )
            self.reporter.print_metric(
                "95% CI",
                f"[{outcomes['percentiles']['final_return'][5]:.1%}, "
                f"{outcomes['percentiles']['final_return'][95]:.1%}]"
            )
            self.reporter.print_metric(
                "虧損機率",
                f"{outcomes['probability_of_loss']:.1%}"
            )
            self.reporter.print_metric(
                ">20% Drawdown 機率",
                f"{outcomes['probability_of_drawdown_gt_20']:.1%}"
            )
            
            results["outcomes"] = outcomes
            
            return results
        
        except Exception as e:
            self.reporter.print_error(f"Monte Carlo 模擬失敗: {e}")
            return {"error": str(e)}
    
    def _print_summary(self, results: dict):
        """打印總結報告"""
        self.reporter.print_header("📋 驗證總結", "═")
        
        # 穩健性評估
        robust_checks = []
        
        if "loao" in results and "result" in results["loao"]:
            loao_result = results["loao"]["result"]
            is_robust = ValidationResultAnalyzer.is_strategy_robust(loao_result)
            robust_checks.append(("Cross-Asset", is_robust))
        
        if "monte_carlo" in results and "outcomes" in results["monte_carlo"]:
            outcomes = results["monte_carlo"]["outcomes"]
            is_safe = outcomes["probability_of_loss"] < 0.5
            robust_checks.append(("Monte Carlo", is_safe))
        
        self.reporter.print_subheader("穩健性檢查")
        for check_name, passed in robust_checks:
            status = "✅ 通過" if passed else "❌ 未通過"
            print(f"  {check_name}: {status}")
        
        # 總體評估
        all_passed = all(passed for _, passed in robust_checks)
        
        self.reporter.print_subheader("總體評估")
        if all_passed:
            self.reporter.print_success("策略通過所有穩健性檢查，可進行下一步測試")
        else:
            self.reporter.print_warning("策略未通過部分檢查，建議進一步優化")


# ══════════════════════════════════════════════════════════════════════════════
# CLI 介面
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    """解析命令列參數"""
    parser = argparse.ArgumentParser(
        description="進階策略驗證：Cross-Asset 驗證 + Monte Carlo 模擬",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
    # 使用預設配置
    python scripts/run_advanced_validation.py --config config/rsi_adx_atr.yaml

    # 指定資產
    python scripts/run_advanced_validation.py --config config/rsi_adx_atr.yaml \\
        --symbols BTCUSDT ETHUSDT BNBUSDT SOLUSDT ADAUSDT

    # 只執行 Monte Carlo 模擬
    python scripts/run_advanced_validation.py --config config/rsi_adx_atr.yaml \\
        --no-loao --no-correlation --no-regime
        """,
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="策略配置檔案路徑",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT"],
        help="要驗證的資產列表",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/binance/spot/1h"),
        help="數據目錄路徑（默認: data/binance/spot/1h）",
    )
    parser.add_argument(
        "--no-loao",
        action="store_true",
        help="跳過 Leave-One-Asset-Out 驗證",
    )
    parser.add_argument(
        "--no-correlation",
        action="store_true",
        help="跳過相關性分層驗證",
    )
    parser.add_argument(
        "--no-regime",
        action="store_true",
        help="跳過市場狀態驗證",
    )
    parser.add_argument(
        "--no-monte-carlo",
        action="store_true",
        help="跳過 Monte Carlo 模擬",
    )
    parser.add_argument(
        "--n-simulations",
        type=int,
        default=10000,
        help="Monte Carlo 模擬次數",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.6,
        help="訓練集比例（默認: 0.6）",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="驗證集比例（默認: 0.2，Test = 1 - train - val）",
    )
    parser.add_argument(
        "--use-full-data",
        action="store_true",
        help="使用全部數據（不分割，僅用於調試）",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="安靜模式（減少輸出）",
    )
    
    return parser.parse_args()


def build_backtest_config(cfg, symbol: str) -> dict:
    """
    將專案配置轉換為回測函數期望的格式
    
    Args:
        cfg: load_config() 返回的配置物件
        symbol: 交易對符號
    
    Returns:
        回測配置字典
    """
    return {
        "initial_cash": cfg.backtest.initial_cash,
        "fee_bps": cfg.backtest.fee_bps,
        "slippage_bps": cfg.backtest.slippage_bps,
        "strategy_params": cfg.strategy.get_params(symbol),
        "strategy_name": cfg.strategy.name,
        "validate_data": cfg.backtest.validate_data,
        "clean_data_before": cfg.backtest.clean_data,
        "interval": cfg.market.interval,
    }


def main():
    """主程式"""
    args = parse_args()
    
    # 載入策略配置（使用專案的 load_config）
    if not args.config.exists():
        print(f"❌ 找不到配置檔案: {args.config}")
        sys.exit(1)
    
    cfg = load_config(str(args.config))
    
    # 使用配置檔案中的交易對（如果沒有指定）
    symbols = args.symbols
    if symbols == ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT"]:
        # 使用預設值，嘗試用配置檔案中的交易對
        symbols = cfg.market.symbols
    
    # 確定數據目錄
    data_dir = args.data_dir
    if data_dir == Path("data/binance/spot/1h"):
        # 使用預設值，根據配置動態設定
        data_dir = cfg.data_dir / "binance" / "spot" / cfg.market.interval
    
    # 建立第一個交易對的回測配置（用於顯示策略名稱等）
    first_symbol = symbols[0] if symbols else "BTCUSDT"
    strategy_config = build_backtest_config(cfg, first_symbol)
    
    # 建立流程配置
    pipeline_config = ValidationPipelineConfig(
        symbols=symbols,
        data_dir=data_dir,
        run_loao=not args.no_loao,
        run_correlation=not args.no_correlation,
        run_regime=not args.no_regime,
        run_monte_carlo=not args.no_monte_carlo,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        use_test_set_only=not args.use_full_data,  # 預設使用 Test Set
        mc_n_simulations=args.n_simulations,
        verbose=not args.quiet,
    )
    
    # 建立自定義回測函數，為每個交易對使用正確的參數
    def custom_backtest_func(symbol: str, data_path: Path, bt_cfg: dict, strategy_name=None):
        """為每個交易對使用正確的參數覆寫"""
        # 使用該交易對的專屬參數
        symbol_cfg = build_backtest_config(cfg, symbol)
        return run_symbol_backtest(symbol, data_path, symbol_cfg, symbol_cfg["strategy_name"])
    
    # 執行驗證（使用自定義回測函數）
    from qtrade.data.storage import load_klines
    
    pipeline = ValidationPipeline(
        pipeline_config,
        strategy_config,
        backtest_func=custom_backtest_func,
        data_loader=load_klines,
    )
    
    results = pipeline.run()
    
    print("\n✅ 驗證完成！")


if __name__ == "__main__":
    main()
