#!/usr/bin/env python3
"""
統一驗證入口

整合所有策略驗證功能於一個腳本：
- Walk-Forward Analysis
- Monte Carlo Simulation
- Cross-Asset Validation (LOAO, Correlation, Market Regime)
- Advanced Methods (DSR, PBO)
- Kelly Formula Validation
- Live/Backtest Consistency

使用方式:
    # 執行標準驗證套件
    python scripts/validate.py -c config/strategies/rsi_adx_atr.yaml

    # 快速驗證（跳過耗時的測試）
    python scripts/validate.py -c config/rsi_adx_atr.yaml --quick

    # 只執行特定驗證
    python scripts/validate.py -c config/rsi_adx_atr.yaml --only walk_forward,monte_carlo

    # Kelly 公式驗證
    python scripts/validate.py -c config/rsi_adx_atr.yaml --only kelly

    # 完整驗證（包括所有測試）
    python scripts/validate.py -c config/rsi_adx_atr.yaml --full

    # 使用驗證配置覆蓋
    python scripts/validate.py -c config/rsi_adx_atr.yaml -v config/validation.yaml
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

# 添加 src 到路徑
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qtrade.config import load_config
from qtrade.data.storage import load_klines


@dataclass
class ValidationConfig:
    """驗證配置"""
    # Walk-Forward
    walk_forward_enabled: bool = True
    walk_forward_splits: int = 5
    
    # Monte Carlo
    monte_carlo_enabled: bool = True
    monte_carlo_simulations: int = 10000
    monte_carlo_confidence: List[float] = None
    
    # Cross-Asset
    loao_enabled: bool = True
    correlation_enabled: bool = True
    regime_enabled: bool = True
    
    # Advanced (Prado methods)
    dsr_enabled: bool = True
    dsr_n_trials: int = 729
    pbo_enabled: bool = True
    pbo_threshold: float = 0.5
    
    # Kelly
    kelly_enabled: bool = True
    kelly_fractions: List[float] = None
    
    # Consistency
    consistency_enabled: bool = False
    consistency_days: int = 7
    
    def __post_init__(self):
        if self.monte_carlo_confidence is None:
            self.monte_carlo_confidence = [0.95, 0.99]
        if self.kelly_fractions is None:
            self.kelly_fractions = [0.0, 0.25, 0.5, 0.75, 1.0]


def load_validation_config(config_path: Optional[str]) -> ValidationConfig:
    """載入驗證配置"""
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            data = yaml.safe_load(f)
        
        # 解析配置
        wf = data.get("walk_forward", {})
        mc = data.get("monte_carlo", {})
        ca = data.get("cross_asset", {})
        pm = data.get("prado_methods", {})
        ky = data.get("kelly", {})
        cs = data.get("consistency", {})
        
        return ValidationConfig(
            walk_forward_enabled=wf.get("enabled", True),
            walk_forward_splits=wf.get("n_splits", 5),
            monte_carlo_enabled=mc.get("enabled", True),
            monte_carlo_simulations=mc.get("n_simulations", 10000),
            monte_carlo_confidence=mc.get("confidence_levels", [0.95, 0.99]),
            loao_enabled=ca.get("run_loao", True),
            correlation_enabled=ca.get("run_correlation_stratified", True),
            regime_enabled=ca.get("run_regime_validation", True),
            dsr_enabled=pm.get("deflated_sharpe", {}).get("enabled", True),
            dsr_n_trials=pm.get("deflated_sharpe", {}).get("n_trials", 729),
            pbo_enabled=pm.get("pbo", {}).get("enabled", True),
            pbo_threshold=pm.get("pbo", {}).get("threshold", 0.5),
            kelly_enabled=ky.get("enabled", True),
            kelly_fractions=ky.get("fractions", [0.0, 0.25, 0.5, 0.75, 1.0]),
            consistency_enabled=cs.get("enabled", False),
            consistency_days=cs.get("days", 7),
        )
    
    return ValidationConfig()


def run_walk_forward(
    symbols: List[str],
    data_paths: Dict[str, Path],
    cfg: dict,
    n_splits: int,
    report_dir: Path,
) -> Dict[str, pd.DataFrame]:
    """執行 Walk-Forward 分析"""
    from qtrade.validation import walk_forward_analysis
    
    print("\n" + "=" * 72)
    print("  📊 Walk-Forward Analysis（前瞻驗證）")
    print("     用歷史訓練 → 在未來數據上測試，模擬真實使用場景")
    print("=" * 72)
    
    results = {}
    for symbol in symbols:
        if symbol not in data_paths:
            print(f"  ⚠️  {symbol}: 無數據")
            continue
        
        print(f"\n  {symbol}:")
        try:
            wf_result = walk_forward_analysis(
                symbol=symbol,
                data_path=data_paths[symbol],
                cfg=cfg,
                n_splits=n_splits,
            )
            results[symbol] = wf_result
            
            # 保存結果
            wf_path = report_dir / f"walk_forward_{symbol}.csv"
            wf_result.to_csv(wf_path, index=False)
            
            # 顯示摘要
            if len(wf_result) > 0:
                avg_train = wf_result["train_sharpe"].mean()
                avg_test = wf_result["test_sharpe"].mean()
                degradation = (avg_train - avg_test) / max(abs(avg_train), 0.01)
                deg_icon = "✅" if degradation < 0.5 else "⚠️"
                print(f"    平均 Train Sharpe: {avg_train:.2f}")
                print(f"    平均 Test Sharpe:  {avg_test:.2f}")
                print(f"    {deg_icon} 績效衰退: {degradation:.1%}（< 50% 為佳）")
            else:
                print(f"    ⚠️  無有效結果（可能數據太短）")
        except Exception as e:
            print(f"    ❌ 失敗: {e}")
    
    return results


def run_monte_carlo(
    symbols: List[str],
    data_paths: Dict[str, Path],
    cfg: dict,
    n_simulations: int,
    confidence_levels: List[float],
    report_dir: Path,
) -> Dict:
    """執行 Monte Carlo 模擬"""
    from qtrade.risk.monte_carlo import MonteCarloSimulator, MonteCarloConfig
    from qtrade.backtest.run_backtest import run_symbol_backtest
    
    print("\n" + "=" * 72)
    print("  🎲 Monte Carlo Simulation（壓力測試）")
    print("     隨機打亂收益順序模擬 10000 次，估計最壞情況")
    print("=" * 72)
    
    results = {}
    for symbol in symbols:
        if symbol not in data_paths:
            continue
        
        print(f"\n  {symbol}:")
        try:
            # 執行回測獲取收益率
            bt_result = run_symbol_backtest(
                symbol, data_paths[symbol], cfg, cfg.get("strategy_name")
            )
            
            # 從 Portfolio 物件提取收益率
            pf = bt_result.pf
            if pf is not None:
                returns = pf.returns()
            else:
                returns = None
            
            if returns is None or len(returns) == 0:
                print(f"    ⚠️  無收益數據")
                continue
            
            # Monte Carlo 模擬 - 使用正確的配置方式
            mc_config = MonteCarloConfig(
                n_simulations=n_simulations,
                confidence_levels=tuple(confidence_levels),
            )
            simulator = MonteCarloSimulator(mc_config=mc_config)
            
            # 計算 VaR
            var_result = simulator.calculate_var(returns)
            
            results[symbol] = {
                "var": var_result,
            }
            
            # 顯示結果
            var_95 = var_result.get_var(0.95)
            var_99 = var_result.get_var(0.99)
            cvar_95 = var_result.get_cvar(0.95)
            
            print(f"    VaR (95%): {var_95:.2%}")
            print(f"    VaR (99%): {var_99:.2%}")
            print(f"    CVaR (95%): {cvar_95:.2%}")
            
        except Exception as e:
            print(f"    ❌ 失敗: {e}")
    
    return results


def run_cross_asset(
    symbols: List[str],
    data_paths: Dict[str, Path],
    cfg: dict,
    run_loao: bool,
    run_correlation: bool,
    run_regime: bool,
    report_dir: Path,
) -> Dict:
    """執行 Cross-Asset 驗證"""
    from qtrade.validation import (
        leave_one_asset_out,
        market_regime_validation,
        ValidationResultAnalyzer,
    )
    
    print("\n" + "=" * 72)
    print("  🔄 Cross-Asset Validation（跨資產驗證）")
    print("     測試策略在不同幣種、不同市場環境下是否一致")
    print("=" * 72)
    
    results = {}
    
    # Leave-One-Asset-Out
    if run_loao and len(symbols) >= 3:
        print("\n  📌 Leave-One-Asset-Out (LOAO):")
        try:
            loao_result = leave_one_asset_out(
                symbols=symbols,
                data_paths=data_paths,
                cfg=cfg,
            )
            results["loao"] = loao_result
            
            print(f"    穩健性等級: {loao_result.robustness_level.value}")
            print(f"    平均 Sharpe 衰退: {loao_result.avg_sharpe_degradation:.1%}")
            
            if loao_result.overfitted_assets:
                print(f"    ⚠️  可能過擬合: {list(loao_result.overfitted_assets)}")
                
            # 保存結果
            loao_df = loao_result.to_dataframe()
            loao_df.to_csv(report_dir / "loao_results.csv", index=False)
            
        except Exception as e:
            print(f"    ❌ 失敗: {e}")
    
    # Market Regime
    if run_regime:
        print("\n  📌 Market Regime Validation:")
        try:
            regime_results, regime_df = market_regime_validation(
                symbols=symbols,
                data_paths=data_paths,
                cfg=cfg,
                indicator="volatility",
            )
            results["regime"] = regime_results
            
            if not regime_df.empty:
                regime_df.to_csv(report_dir / "regime_results.csv", index=False)
                
                # 顯示摘要
                for symbol in symbols:
                    sym_df = regime_df[regime_df["symbol"] == symbol]
                    if len(sym_df) >= 2:
                        high_sharpe = sym_df[sym_df["regime"].str.contains("high")]["sharpe"].iloc[0]
                        low_sharpe = sym_df[sym_df["regime"].str.contains("low")]["sharpe"].iloc[0]
                        print(f"    {symbol}: High Vol SR={high_sharpe:.2f}, Low Vol SR={low_sharpe:.2f}")
                        
        except Exception as e:
            print(f"    ❌ 失敗: {e}")
    
    return results


def run_prado_methods(
    symbols: List[str],
    walk_forward_results: Dict[str, pd.DataFrame],
    cfg: dict,
    dsr_enabled: bool,
    dsr_n_trials: int,
    pbo_enabled: bool,
    pbo_threshold: float,
    report_dir: Path,
) -> Dict:
    """執行 Prado 方法（DSR, PBO）"""
    from qtrade.validation import (
        deflated_sharpe_ratio,
        probability_of_backtest_overfitting,
    )
    
    print("\n" + "=" * 72)
    print("  🔬 Advanced Validation（Marcos López de Prado 方法）")
    print("     用學術方法檢測過擬合和 Sharpe Ratio 的真實性")
    print("=" * 72)
    
    results = {}
    
    # 收集所有 walk-forward 結果
    all_train_sharpes = []
    all_test_sharpes = []
    
    for symbol, wf_df in walk_forward_results.items():
        if "train_sharpe" in wf_df.columns and "test_sharpe" in wf_df.columns:
            all_train_sharpes.extend(wf_df["train_sharpe"].tolist())
            all_test_sharpes.extend(wf_df["test_sharpe"].tolist())
    
    # Deflated Sharpe Ratio
    if dsr_enabled and all_test_sharpes:
        print("\n  📌 Deflated Sharpe Ratio (DSR):")
        try:
            observed_sharpe = np.mean(all_test_sharpes)
            n_obs = len(all_test_sharpes) * 100  # 估計觀察數
            
            dsr_result = deflated_sharpe_ratio(
                observed_sharpe=observed_sharpe,
                n_trials=dsr_n_trials,
                n_observations=n_obs,
            )
            results["dsr"] = dsr_result
            
            print(f"    觀察 Sharpe: {dsr_result.observed_sharpe:.4f}")
            print(f"    校正 Sharpe: {dsr_result.deflated_sharpe:.4f}")
            print(f"    p-value: {dsr_result.p_value:.4f}")
            print(f"    顯著性: {'✅ 顯著' if dsr_result.is_significant else '⚠️  不顯著'}")
            
        except Exception as e:
            print(f"    ❌ 失敗: {e}")
    
    # Probability of Backtest Overfitting
    if pbo_enabled and len(all_train_sharpes) >= 2:
        print("\n  📌 Probability of Backtest Overfitting (PBO):")
        try:
            pbo_result = probability_of_backtest_overfitting(
                in_sample_sharpes=np.array(all_train_sharpes),
                out_of_sample_sharpes=np.array(all_test_sharpes),
                threshold=pbo_threshold,
            )
            results["pbo"] = pbo_result
            
            print(f"    PBO: {pbo_result.pbo:.2%}")
            print(f"    排名相關性: {pbo_result.rank_correlation:.4f}")
            print(f"    績效衰退: {pbo_result.performance_degradation:.1%}")
            
            if pbo_result.is_likely_overfitted:
                print(f"    ⚠️  可能過擬合 (PBO > {pbo_threshold:.0%})")
            else:
                print(f"    ✅ 過擬合風險可接受")
                
        except Exception as e:
            print(f"    ❌ 失敗: {e}")
    
    return results


def run_kelly_validation(
    symbols: List[str],
    data_paths: Dict[str, Path],
    cfg: dict,
    kelly_fractions: List[float],
    report_dir: Path,
) -> Dict:
    """執行 Kelly 公式驗證"""
    from qtrade.backtest.kelly_validation import (
        kelly_backtest_comparison,
        is_strategy_suitable_for_kelly,
    )
    
    print("\n" + "=" * 72)
    print("  💰 Kelly Formula Validation（最佳倉位驗證）")
    print("     根據勝率和盈虧比計算最佳資金比例，檢驗穩定性")
    print("=" * 72)
    
    results = {}
    all_suitable = True
    
    for symbol in symbols:
        if symbol not in data_paths:
            print(f"  ⚠️  {symbol}: 無數據")
            continue
        
        print(f"\n  {symbol}:")
        try:
            # 執行 Kelly 驗證
            report = kelly_backtest_comparison(
                symbol=symbol,
                data_path=data_paths[symbol],
                cfg=cfg,
                kelly_fractions=kelly_fractions,
                strategy_name=cfg.get("strategy_name"),
            )
            
            results[symbol] = report
            
            # 顯示結果摘要
            stats = report.kelly_stats
            print(f"    勝率: {stats.win_rate:.1%} ({stats.winning_trades}/{stats.total_trades})")
            print(f"    盈虧比: {stats.win_loss_ratio:.2f}")
            print(f"    Full Kelly: {stats.kelly_pct:.1%}")
            print(f"    穩定性 (CV): {report.kelly_stability:.2f}")
            print(f"    推薦倉位: {report.recommended_fraction:.0%} Kelly")
            print(f"    原因: {report.recommendation_reason}")
            
            if report.recommended_fraction == 0:
                all_suitable = False
                print(f"    ⚠️  不適合使用 Kelly")
            else:
                print(f"    ✅ 推薦使用 {report.recommended_fraction:.0%} Kelly = {stats.kelly_pct * report.recommended_fraction:.1%} 倉位")
            
            # 保存詳細報告
            report_path = report_dir / f"kelly_{symbol}.txt"
            with open(report_path, "w") as f:
                f.write(report.summary())
                
        except Exception as e:
            print(f"    ❌ 失敗: {e}")
            import traceback
            traceback.print_exc()
    
    # 保存摘要
    if results:
        summary_data = []
        for symbol, report in results.items():
            summary_data.append({
                "symbol": symbol,
                "win_rate": report.kelly_stats.win_rate,
                "win_loss_ratio": report.kelly_stats.win_loss_ratio,
                "full_kelly_pct": report.kelly_stats.kelly_pct,
                "recommended_fraction": report.recommended_fraction,
                "effective_kelly_pct": report.kelly_stats.kelly_pct * report.recommended_fraction,
                "stability_cv": report.kelly_stability,
                "total_trades": report.kelly_stats.total_trades,
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(report_dir / "kelly_summary.csv", index=False)
        
        print("\n  " + "-" * 60)
        if all_suitable:
            print("  ✅ 所有交易對都適合使用 Kelly 倉位管理")
        else:
            print("  ⚠️  部分交易對不適合使用 Kelly")
    
    return results


def run_consistency_check(
    symbols: List[str],
    cfg,
    days: int,
    report_dir: Path,
    use_binance_api: bool = True,
) -> Dict:
    """
    執行一致性檢查
    
    Args:
        symbols: 交易對列表
        cfg: 策略配置
        days: 回看天數
        report_dir: 報告目錄
        use_binance_api: 是否從 Binance API 獲取真實交易（推薦）
    """
    from qtrade.validation import ConsistencyValidator
    
    print("\n" + "=" * 72)
    print("  🔍 Live/Backtest Consistency Check（實盤一致性檢查）")
    print("     比對實盤交易與回測信號，確認兩者邏輯一致")
    print("=" * 72)
    print(f"  期間: 最近 {days} 天")
    print(f"  數據來源: {'Binance API' if use_binance_api else 'State 文件'}")
    
    results = {}
    
    validator = ConsistencyValidator(
        strategy_name=cfg.strategy.name,
        params=cfg.strategy.params,
        interval=cfg.market.interval,
        market_type=cfg.market_type_str,
        direction=cfg.direction,
    )
    
    for symbol in symbols:
        # 獲取該 symbol 的特定參數（含覆寫）
        symbol_params = cfg.strategy.get_params(symbol)
        validator.params = symbol_params
        
        print(f"\n  {symbol}:")
        try:
            # 找到對應的 state 文件
            live_dir = cfg.get_report_dir("live")
            live_state_path = live_dir / "real_state.json"
            if not live_state_path.exists():
                live_state_path = live_dir / "paper_state.json"
            
            report = validator.validate_recent(
                symbol=symbol,
                days=days,
                live_state_path=live_state_path if live_state_path.exists() else None,
                use_binance_api=use_binance_api,
            )
            results[symbol] = report
            
            # 顯示結果
            print(f"    信號一致性: {report.consistency_rate:.1%}")
            
            if report.trade_consistency_rate is not None:
                print(f"    交易一致性: {report.trade_consistency_rate:.1%}")
            
            if report.live_return_pct is not None:
                print(f"    Live 收益: {report.live_return_pct:+.2f}%")
                print(f"    Backtest 收益: {report.backtest_return_pct:+.2f}%")
            
            if not report.is_consistent:
                print(f"    ⚠️  未通過一致性檢查")
                for inc in report.inconsistencies:
                    print(f"       • {inc.description}")
            else:
                print(f"    ✅ 通過一致性檢查")
                
            # 保存報告
            report_path = report_dir / f"consistency_{symbol}.json"
            report.save(report_path)
            
        except Exception as e:
            print(f"    ❌ 失敗: {e}")
            import traceback
            traceback.print_exc()
    
    return results


def _to_native(obj):
    """將 numpy 類型轉換為 Python 原生類型，避免 YAML 序列化問題"""
    if isinstance(obj, (np.bool_, np.generic)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(i) for i in obj]
    return obj


def generate_summary(
    walk_forward_results: Dict,
    monte_carlo_results: Dict,
    cross_asset_results: Dict,
    prado_results: Dict,
    kelly_results: Dict,
    report_dir: Path,
):
    """生成驗證摘要報告（新手友善版）"""
    
    # ── 終端輸出 ──────────────────────────────────────
    print("\n" + "=" * 72)
    print("  📋 Validation Summary — 策略驗證總結")
    print("=" * 72)
    print()
    print("  每項測試檢查策略的不同面向，幫助你判斷策略是否可以上線。")
    print("  ✅ PASS = 通過   ⚠️ CHECK = 需注意   ❌ FAIL = 不建議上線")
    print()
    print("  ─────────────────────────────────────────────────────────────────")
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "tests": {},
    }
    
    # Walk-Forward 摘要
    if walk_forward_results:
        all_degradations = []
        per_symbol = {}
        for symbol, wf_df in walk_forward_results.items():
            if len(wf_df) > 0:
                avg_train = float(wf_df["train_sharpe"].mean())
                avg_test = float(wf_df["test_sharpe"].mean())
                deg = (avg_train - avg_test) / max(abs(avg_train), 0.01)
                all_degradations.append(deg)
                per_symbol[symbol] = {
                    "train_sharpe": round(avg_train, 2),
                    "test_sharpe": round(avg_test, 2),
                    "degradation": f"{deg:.1%}",
                    "splits_completed": int(len(wf_df)),
                }
        
        avg_deg = float(np.mean(all_degradations)) if all_degradations else 0
        passed = avg_deg < 0.5
        summary["tests"]["walk_forward"] = {
            "passed": bool(passed),
            "avg_degradation": f"{avg_deg:.1%}",
            "threshold": "< 50%",
            "meaning": "訓練期→測試期的績效衰退幅度，越低代表策略越穩健",
            "per_symbol": per_symbol,
        }
        icon = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {icon}  Walk-Forward（前瞻驗證）")
        print(f"         測試方法: 用歷史訓練 → 在新數據上驗證，模擬真實使用場景")
        print(f"         績效衰退: {avg_deg:.1%}（標準: < 50% 為佳）")
        for sym, info in per_symbol.items():
            print(f"           {sym}: 訓練 SR={info['train_sharpe']:.2f} → 測試 SR={info['test_sharpe']:.2f} (衰退 {info['degradation']})")
        print()
    
    # Monte Carlo 摘要
    if monte_carlo_results:
        var_95_list = []
        per_symbol_mc = {}
        for sym, r in monte_carlo_results.items():
            if "var" in r:
                var_result = r["var"]
                if hasattr(var_result, 'get_var'):
                    v95 = float(var_result.get_var(0.95))
                    v99 = float(var_result.get_var(0.99))
                elif hasattr(var_result, 'var_95'):
                    v95 = float(var_result.var_95)
                    v99 = float(getattr(var_result, 'var_99', 0))
                else:
                    continue
                var_95_list.append(v95)
                per_symbol_mc[sym] = {"var_95": f"{v95:.2%}", "var_99": f"{v99:.2%}"}
        
        avg_var = float(np.mean(var_95_list)) if var_95_list else 0
        passed = avg_var > -0.3  # VaR 95% < 30%
        summary["tests"]["monte_carlo"] = {
            "passed": bool(passed),
            "avg_var_95": f"{avg_var:.2%}",
            "threshold": "日 VaR 95% < 30%",
            "meaning": "模擬 10000 次隨機情境，估計最差情況的單日虧損",
            "per_symbol": per_symbol_mc,
        }
        icon = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {icon}  Monte Carlo（壓力測試）")
        print(f"         測試方法: 隨機模擬 10000 種市場情境，看最差情況虧多少")
        print(f"         平均 VaR 95%: {avg_var:.2%}（意思：95% 的情況下單日虧損 < 此值）")
        for sym, info in per_symbol_mc.items():
            print(f"           {sym}: VaR 95%={info['var_95']}, VaR 99%={info['var_99']}")
        print()
    
    # Cross-Asset 摘要
    if cross_asset_results:
        loao = cross_asset_results.get("loao")
        if loao:
            passed = loao.robustness_level.value in ["robust", "moderate"]
            summary["tests"]["cross_asset"] = {
                "passed": bool(passed),
                "robustness": loao.robustness_level.value,
                "threshold": "穩健度 robust 或 moderate",
                "meaning": "策略在不同幣種上表現一致嗎？防止只對特定幣過擬合",
            }
            icon = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {icon}  Cross-Asset（跨資產驗證）")
            print(f"         測試方法: 去掉一個幣種訓練，看剩下的表現")
            print(f"         穩健度: {loao.robustness_level.value}（標準: robust 或 moderate）")
            print()
    
    # Prado 摘要
    if prado_results:
        dsr = prado_results.get("dsr")
        pbo = prado_results.get("pbo")
        
        if dsr:
            is_sig = bool(dsr.is_significant)
            summary["tests"]["dsr"] = {
                "passed": is_sig,
                "deflated_sharpe": round(float(dsr.deflated_sharpe), 4),
                "p_value": round(float(dsr.p_value), 6),
                "threshold": "p-value < 0.05 (統計顯著)",
                "meaning": "考慮了『試了很多參數才找到這個結果』的情況後，Sharpe 是否仍然顯著？",
            }
            icon = "✅ PASS" if is_sig else "⚠️ CHECK"
            print(f"  {icon}  DSR（校正 Sharpe Ratio）")
            print(f"         測試方法: 把回測裡「調了很多參數」的因素扣除，看 Sharpe 是否仍顯著")
            print(f"         校正 SR: {dsr.deflated_sharpe:.4f}, p-value: {dsr.p_value:.4f}（標準: p < 0.05）")
            print()
        
        if pbo:
            not_overfitted = bool(not pbo.is_likely_overfitted)
            summary["tests"]["pbo"] = {
                "passed": not_overfitted,
                "pbo_pct": f"{pbo.pbo:.1%}",
                "rank_correlation": round(float(pbo.rank_correlation), 4),
                "threshold": "PBO < 50%",
                "meaning": "用交叉驗證估計策略是『真的好』還是『碰巧好』的機率",
            }
            icon = "✅ PASS" if not_overfitted else "⚠️ CHECK"
            print(f"  {icon}  PBO（過擬合機率）")
            print(f"         測試方法: 用排列組合計算「回測好但實盤差」的機率")
            print(f"         過擬合機率: {pbo.pbo:.1%}（標準: < 50%）")
            print()
    
    # Kelly 摘要
    if kelly_results:
        suitable_count = sum(
            1 for r in kelly_results.values() 
            if r.recommended_fraction > 0
        )
        total_count = len(kelly_results)
        passed = suitable_count == total_count
        
        per_symbol_kelly = {}
        for sym, r in kelly_results.items():
            per_symbol_kelly[sym] = {
                "win_rate": f"{r.kelly_stats.win_rate:.1%}",
                "win_loss_ratio": round(float(r.kelly_stats.win_loss_ratio), 2),
                "full_kelly": f"{r.kelly_stats.kelly_pct:.1%}",
                "recommended": f"{r.recommended_fraction:.0%} Kelly = {r.kelly_stats.kelly_pct * r.recommended_fraction:.1%}",
                "stability_cv": round(float(r.kelly_stability), 2),
            }
        
        summary["tests"]["kelly"] = {
            "passed": bool(passed),
            "suitable_assets": f"{suitable_count}/{total_count}",
            "threshold": "所有幣種都適合使用 Kelly",
            "meaning": "根據歷史勝率和盈虧比，計算最佳倉位大小並檢驗其穩定性",
            "per_symbol": per_symbol_kelly,
        }
        icon = "✅ PASS" if passed else "⚠️ CHECK"
        print(f"  {icon}  Kelly（最佳倉位驗證）")
        print(f"         測試方法: 用歷史勝率+盈虧比算最佳倉位，並檢查穩定性")
        for sym, info in per_symbol_kelly.items():
            print(f"           {sym}: 勝率={info['win_rate']}, 盈虧比={info['win_loss_ratio']}, 建議={info['recommended']}")
        print()
    
    # 總體判斷
    all_passed = all(
        t.get("passed", True) 
        for t in summary["tests"].values()
    )
    summary["overall_passed"] = bool(all_passed)
    
    print("  ─────────────────────────────────────────────────────────────────")
    if all_passed:
        print("  🎉 Overall: ✅ 策略驗證通過 — 可以考慮上線！")
    else:
        print("  ⚠️  Overall: 有項目未通過，建議了解後再決定是否上線")
    print()
    print("  💡 提示：驗證通過≠保證賺錢，只是表示策略在統計上有合理性。")
    print("           實盤會受滑價、funding rate、流動性等因素影響。")
    print("=" * 72)
    
    # ── 保存 YAML（轉換為原生類型，避免 numpy 序列化問題）──
    summary = _to_native(summary)
    summary_path = report_dir / "validation_summary.yaml"
    with open(summary_path, "w", encoding="utf-8") as f:
        yaml.dump(
            summary,
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="統一策略驗證工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  python scripts/validate.py -c config/rsi_adx_atr.yaml
  python scripts/validate.py -c config/rsi_adx_atr.yaml --quick
  python scripts/validate.py -c config/rsi_adx_atr.yaml --only walk_forward,monte_carlo
  python scripts/validate.py -c config/rsi_adx_atr.yaml --full
        """
    )
    
    parser.add_argument(
        "-c", "--config",
        required=True,
        help="策略配置文件路徑"
    )
    
    parser.add_argument(
        "-v", "--validation-config",
        default=None,
        help="驗證配置文件路徑（可選）"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="快速模式：只執行基本驗證"
    )
    
    parser.add_argument(
        "--full",
        action="store_true",
        help="完整模式：執行所有驗證（包括耗時測試）"
    )
    
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="只執行指定的驗證（逗號分隔）: walk_forward,monte_carlo,loao,regime,dsr,pbo,kelly,consistency,predeploy"
    )
    
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="報告輸出目錄"
    )
    
    args = parser.parse_args()
    
    # 載入配置
    cfg = load_config(args.config)
    val_cfg = load_validation_config(args.validation_config)
    
    # 解析 --only 參數
    only_tests = None
    if args.only:
        only_tests = set(args.only.lower().split(","))
    
    # 調整模式
    if args.quick:
        val_cfg.monte_carlo_simulations = 1000
        val_cfg.correlation_enabled = False
        val_cfg.consistency_enabled = False
    
    if args.full:
        val_cfg.consistency_enabled = True
    
    # 設置輸出目錄
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output:
        report_dir = Path(args.output)
    else:
        report_dir = cfg.get_report_dir("validation") / timestamp
    report_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 72)
    print(f"  🔬 Strategy Validation: {cfg.strategy.name}")
    print("=" * 72)
    print(f"  配置: {args.config}")
    print(f"  交易對: {cfg.market.symbols}")
    print(f"  報告目錄: {report_dir}")
    print()
    print("  本工具執行 6 項驗證，從不同角度檢查策略是否真的有效：")
    print("  ① Walk-Forward  — 策略在新數據上還行不行？")
    print("  ② Monte Carlo   — 最壞情況會虧多少？")
    print("  ③ Cross-Asset   — 換一個幣種還有效嗎？")
    print("  ④ DSR / PBO     — 是不是碰巧調出來的好結果？")
    print("  ⑤ Kelly         — 每次該下多大的注？")
    print("  ⑥ Pre-Deploy    — 回測和實盤的程式碼一致嗎？")
    
    # 準備數據路徑
    symbols = cfg.market.symbols
    # 根據 market_type 決定數據路徑
    market_type = cfg.market_type_str  # "spot" or "futures"
    data_dir = cfg.data_dir / "binance" / market_type / cfg.market.interval
    data_paths = {}
    
    for symbol in symbols:
        path = data_dir / f"{symbol}.parquet"
        if path.exists():
            data_paths[symbol] = path
        else:
            print(f"  ⚠️  {symbol}: 數據文件不存在，將跳過")
    
    if not data_paths:
        print("❌ 沒有可用的數據文件，請先下載數據")
        return 1
    
    # 使用 AppConfig 集中方法產生回測配置（包含 market_type / direction）
    backtest_cfg = cfg.to_backtest_dict()
    
    # 執行驗證
    walk_forward_results = {}
    monte_carlo_results = {}
    cross_asset_results = {}
    prado_results = {}
    kelly_results = {}
    
    def should_run(test_name: str, enabled: bool) -> bool:
        if only_tests is not None:
            return test_name in only_tests
        return enabled
    
    # 1. Walk-Forward
    if should_run("walk_forward", val_cfg.walk_forward_enabled):
        walk_forward_results = run_walk_forward(
            symbols=symbols,
            data_paths=data_paths,
            cfg=backtest_cfg,
            n_splits=val_cfg.walk_forward_splits,
            report_dir=report_dir,
        )
    
    # 2. Monte Carlo
    if should_run("monte_carlo", val_cfg.monte_carlo_enabled):
        monte_carlo_results = run_monte_carlo(
            symbols=symbols,
            data_paths=data_paths,
            cfg=backtest_cfg,
            n_simulations=val_cfg.monte_carlo_simulations,
            confidence_levels=val_cfg.monte_carlo_confidence,
            report_dir=report_dir,
        )
    
    # 3. Cross-Asset
    run_loao = should_run("loao", val_cfg.loao_enabled)
    run_correlation = should_run("correlation", val_cfg.correlation_enabled)
    run_regime = should_run("regime", val_cfg.regime_enabled)
    
    if run_loao or run_correlation or run_regime:
        cross_asset_results = run_cross_asset(
            symbols=symbols,
            data_paths=data_paths,
            cfg=backtest_cfg,
            run_loao=run_loao,
            run_correlation=run_correlation,
            run_regime=run_regime,
            report_dir=report_dir,
        )
    
    # 4. Prado Methods (需要 walk-forward 結果)
    run_dsr = should_run("dsr", val_cfg.dsr_enabled)
    run_pbo = should_run("pbo", val_cfg.pbo_enabled)
    
    if (run_dsr or run_pbo) and walk_forward_results:
        prado_results = run_prado_methods(
            symbols=symbols,
            walk_forward_results=walk_forward_results,
            cfg=backtest_cfg,
            dsr_enabled=run_dsr,
            dsr_n_trials=val_cfg.dsr_n_trials,
            pbo_enabled=run_pbo,
            pbo_threshold=val_cfg.pbo_threshold,
            report_dir=report_dir,
        )
    
    # 5. Kelly Validation
    if should_run("kelly", val_cfg.kelly_enabled):
        kelly_results = run_kelly_validation(
            symbols=symbols,
            data_paths=data_paths,
            cfg=backtest_cfg,
            kelly_fractions=val_cfg.kelly_fractions,
            report_dir=report_dir,
        )
    
    # 6. Consistency Check (需要 Paper Trading 運行中)
    if should_run("consistency", val_cfg.consistency_enabled):
        run_consistency_check(
            symbols=symbols,
            cfg=cfg,
            days=val_cfg.consistency_days,
            report_dir=report_dir,
        )
    
    # 7. Pre-Deploy 一致性檢查（回測↔實盤路徑比對）
    if should_run("predeploy", True):
        try:
            from validate_live_consistency import ConsistencyChecker, print_report
            checker = ConsistencyChecker(cfg, verbose=True)
            results = checker.run_all()
            print_report(results, verbose=True)
        except ImportError:
            print("  ⚠️  validate_live_consistency.py 未找到，跳過 pre-deploy 檢查")
        except Exception as e:
            print(f"  ⚠️  Pre-deploy 檢查異常: {e}")
    
    # 8. 生成摘要
    generate_summary(
        walk_forward_results=walk_forward_results,
        monte_carlo_results=monte_carlo_results,
        cross_asset_results=cross_asset_results,
        prado_results=prado_results,
        kelly_results=kelly_results,
        report_dir=report_dir,
    )
    
    print(f"\n📁 報告已保存至: {report_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
