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
import copy
import sys
from dataclasses import dataclass, field
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

    # Cost Stress Test
    cost_stress_enabled: bool = True
    cost_stress_multipliers: List[float] = None

    # Delay Stress Test
    delay_stress_enabled: bool = True
    delay_stress_extra_bars: int = 1
    delay_stress_max_drop_pct: float = 50.0

    # Holdout OOS
    holdout_oos_enabled: bool = True
    holdout_oos_max_degradation: float = 0.5
    holdout_oos_periods: List[dict] = None

    # Alpha Decay
    # Governance spec: .cursor/skills/validation/alpha-decay-governance.md
    alpha_decay_enabled: bool = True
    alpha_decay_forward_bars: int = 24
    alpha_decay_window_days: int = 180
    alpha_decay_recent_days: int = 180
    alpha_decay_recent_ic_min: float = 0.005
    alpha_decay_max_decay_pct: float = 0.6
    alpha_decay_max_critical_alerts: int = 2
    alpha_decay_min_ic_denominator: float = 0.01

    # Red Flags thresholds (from validation.yaml -> red_flags)
    red_flags_thresholds: Optional[Dict[str, float]] = None

    # Market Regimes (for regime analysis)
    market_regimes: List[dict] = None

    # Embargo Holdout OOS
    embargo_holdout_enabled: bool = True
    embargo_holdout_max_degradation: float = 0.5
    validation_config_path: Optional[str] = None  # 用於載入 embargo config
    
    def __post_init__(self):
        if self.monte_carlo_confidence is None:
            self.monte_carlo_confidence = [0.95, 0.99]
        if self.kelly_fractions is None:
            self.kelly_fractions = [0.0, 0.25, 0.5, 0.75, 1.0]
        if self.cost_stress_multipliers is None:
            self.cost_stress_multipliers = [1.5, 2.0]
        if self.holdout_oos_periods is None:
            self.holdout_oos_periods = []
        if self.market_regimes is None:
            self.market_regimes = []


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
        cost_s = data.get("cost_stress", {})
        delay_s = data.get("delay_stress", {})
        holdout = data.get("holdout_oos", {})
        alpha_decay = data.get("alpha_decay", {})
        embargo_raw = data.get("data_embargo", {})
        red_flags_raw = data.get("red_flags", {})
        
        # DSR n_trials: 優先使用 trial_registry.cumulative_n_trials（真實多重測試數）
        trial_reg = data.get("trial_registry", {})
        dsr_n_trials = trial_reg.get(
            "cumulative_n_trials",
            pm.get("deflated_sharpe", {}).get("n_trials", 729),
        )

        # Embargo holdout: 從 data_embargo section 讀取 enabled 狀態
        embargo_temporal = embargo_raw.get("temporal", {})
        embargo_holdout_enabled = embargo_temporal.get("enabled", True)
        
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
            dsr_n_trials=dsr_n_trials,
            pbo_enabled=pm.get("pbo", {}).get("enabled", True),
            pbo_threshold=pm.get("pbo", {}).get("threshold", 0.5),
            kelly_enabled=ky.get("enabled", True),
            kelly_fractions=ky.get("fractions", [0.0, 0.25, 0.5, 0.75, 1.0]),
            consistency_enabled=cs.get("enabled", False),
            consistency_days=cs.get("days", 7),
            cost_stress_enabled=cost_s.get("enabled", True),
            cost_stress_multipliers=cost_s.get("multipliers", [1.5, 2.0]),
            delay_stress_enabled=delay_s.get("enabled", True),
            delay_stress_extra_bars=delay_s.get("extra_delay_bars", 1),
            delay_stress_max_drop_pct=delay_s.get("max_sharpe_drop_pct", 50.0),
            holdout_oos_enabled=holdout.get("enabled", True),
            holdout_oos_max_degradation=holdout.get("max_degradation", 0.5),
            holdout_oos_periods=holdout.get("periods", []),
            alpha_decay_enabled=alpha_decay.get("enabled", True),
            alpha_decay_forward_bars=alpha_decay.get("forward_bars", 24),
            alpha_decay_window_days=alpha_decay.get("window_days", 180),
            alpha_decay_recent_days=alpha_decay.get("recent_days", 180),
            alpha_decay_recent_ic_min=alpha_decay.get("recent_ic_min", 0.005),
            alpha_decay_max_decay_pct=alpha_decay.get("max_decay_pct", 0.6),
            alpha_decay_max_critical_alerts=alpha_decay.get("max_critical_alerts", 2),
            alpha_decay_min_ic_denominator=alpha_decay.get("min_ic_denominator", 0.01),
            red_flags_thresholds=red_flags_raw if red_flags_raw else None,
            market_regimes=data.get("market_regimes", []),
            embargo_holdout_enabled=embargo_holdout_enabled,
            embargo_holdout_max_degradation=embargo_temporal.get("max_degradation", 0.5),
            validation_config_path=config_path,
        )
    
    return ValidationConfig()


def run_walk_forward(
    symbols: List[str],
    data_paths: Dict[str, Path],
    cfg: dict,
    n_splits: int,
    report_dir: Path,
    data_dir: Optional[Path] = None,
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
                data_dir=data_dir,
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
    data_dir: Optional[Path] = None,
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
                symbol, data_paths[symbol], cfg, cfg.get("strategy_name"),
                data_dir=data_dir,
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
    data_dir: Optional[Path] = None,
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
                data_dir=data_dir,
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
                data_dir=data_dir,
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
    data_paths: Optional[Dict[str, Path]] = None,
    data_dir: Optional[Path] = None,
) -> Dict:
    """執行 Prado 方法（DSR, PBO via CPCV）"""
    from qtrade.validation import (
        deflated_sharpe_ratio,
    )
    from qtrade.validation.prado_methods import combinatorial_purged_cv
    
    print("\n" + "=" * 72)
    print("  🔬 Advanced Validation（Marcos López de Prado 方法）")
    print("     用學術方法檢測過擬合和 Sharpe Ratio 的真實性")
    print("=" * 72)
    
    results = {}
    
    # 收集所有 walk-forward 結果（供 DSR 使用）
    all_train_sharpes = []
    all_test_sharpes = []
    
    for symbol, wf_df in walk_forward_results.items():
        if "train_sharpe" in wf_df.columns and "test_sharpe" in wf_df.columns:
            all_train_sharpes.extend(wf_df["train_sharpe"].tolist())
            all_test_sharpes.extend(wf_df["test_sharpe"].tolist())
    
    # Deflated Sharpe Ratio
    if dsr_enabled and all_test_sharpes:
        print("\n  📌 Deflated Sharpe Ratio (DSR):")
        print(f"     n_trials={dsr_n_trials}（累積歷史測試組合數，來自 trial_registry）")
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
            print(f"    n_trials (cumulative): {dsr_n_trials}")
            print(f"    顯著性: {'✅ 顯著' if dsr_result.is_significant else '⚠️  不顯著'}")
            
        except Exception as e:
            print(f"    ❌ 失敗: {e}")
    
    # Probability of Backtest Overfitting — 使用真正的 CPCV
    if pbo_enabled and data_paths:
        print("\n  📌 Probability of Backtest Overfitting (PBO via CPCV):")
        print("     使用 Combinatorial Purged Cross-Validation (10 splits, 2 test)")
        
        strategy_name = cfg.get("strategy_name", "")
        cpcv_pbos = []
        cpcv_degradations = []
        
        for symbol in symbols:
            if symbol not in data_paths:
                continue
            try:
                print(f"\n  {symbol}:")
                cpcv_result = combinatorial_purged_cv(
                    symbol=symbol,
                    data_path=data_paths[symbol],
                    cfg=cfg,
                    strategy_name=strategy_name,
                    n_splits=10,
                    n_test_splits=2,
                    purge_bars=10,
                    embargo_bars=10,
                    data_dir=data_dir,
                )
                cpcv_pbos.append(cpcv_result.pbo)
                cpcv_degradations.append(cpcv_result.sharpe_degradation)
                
                print(f"    Train SR: {cpcv_result.mean_train_sharpe:.2f} ± {cpcv_result.std_train_sharpe:.2f}")
                print(f"    Test SR:  {cpcv_result.mean_test_sharpe:.2f} ± {cpcv_result.std_test_sharpe:.2f}")
                print(f"    PBO: {cpcv_result.pbo:.1%}, 衰退: {cpcv_result.sharpe_degradation:.1%}")
                print(f"    組合數: {cpcv_result.n_combinations}")
                
            except Exception as e:
                print(f"    ❌ 失敗: {e}")
        
        if cpcv_pbos:
            from qtrade.validation.prado_methods import PBOResult
            avg_pbo = np.mean(cpcv_pbos)
            max_pbo = np.max(cpcv_pbos)
            avg_deg = np.mean(cpcv_degradations)
            
            print(f"\n  ── CPCV PBO 彙總 ──")
            print(f"    平均 PBO: {avg_pbo:.1%}")
            print(f"    最大 PBO: {max_pbo:.1%}")
            print(f"    平均衰退: {avg_deg:.1%}")
            
            if max_pbo < pbo_threshold:
                print(f"    ✅ 所有幣種 PBO < {pbo_threshold:.0%}，過擬合風險可接受")
            else:
                print(f"    ⚠️  部分幣種 PBO > {pbo_threshold:.0%}")
            
            # 建立與舊介面相容的 PBOResult
            results["pbo"] = PBOResult(
                pbo=avg_pbo,
                logits=np.array(cpcv_pbos),
                performance_degradation=avg_deg,
                rank_correlation=0.0,  # CPCV 不使用此欄位
                n_combinations=len(cpcv_pbos),
                is_likely_overfitted=max_pbo > pbo_threshold,
            )
    
    return results


def run_kelly_validation(
    symbols: List[str],
    data_paths: Dict[str, Path],
    cfg: dict,
    kelly_fractions: List[float],
    report_dir: Path,
    data_dir: Optional[Path] = None,
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
                data_dir=data_dir,
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


# ══════════════════════════════════════════════════════════════════════════════
# Cost Stress Test
# ══════════════════════════════════════════════════════════════════════════════

def run_cost_stress_test(
    symbols: List[str],
    data_paths: Dict[str, Path],
    cfg: dict,
    strategy_name: str,
    multipliers: List[float],
    report_dir: Path,
    data_dir: Optional[Path] = None,
) -> Dict:
    """
    成本壓力測試：在 1×, 1.5×, 2× 成本下檢查策略存活性。

    PASS 條件：2× 成本下所有幣種平均 SR > 0
    """
    from qtrade.backtest.run_backtest import run_symbol_backtest

    print("\n" + "=" * 72)
    print("  💰 Cost Stress Test（成本壓力測試）")
    print("     在倍增成本下驗證策略是否仍有正收益")
    print("=" * 72)

    all_multipliers = [1.0] + [m for m in multipliers if m != 1.0]
    # per-symbol, per-multiplier Sharpe
    results: Dict[str, Dict[float, float]] = {}

    for symbol in symbols:
        if symbol not in data_paths:
            continue
        results[symbol] = {}
        for mult in all_multipliers:
            try:
                stressed_cfg = copy.deepcopy(cfg)
                stressed_cfg["fee_bps"] = stressed_cfg["fee_bps"] * mult
                stressed_cfg["slippage_bps"] = stressed_cfg["slippage_bps"] * mult
                bt_res = run_symbol_backtest(
                    symbol, data_paths[symbol], stressed_cfg,
                    strategy_name, data_dir=data_dir,
                )
                sr = bt_res.sharpe()
                results[symbol][mult] = sr
            except Exception as e:
                print(f"    ❌ {symbol} @ {mult:.1f}x: {e}")
                results[symbol][mult] = float("nan")

    # ── 顯示表格 ──
    header_parts = [f"{'Symbol':<12}"]
    for mult in all_multipliers:
        header_parts.append(f"{'SR@' + f'{mult:.1f}x':>10}")
    print(f"\n  {' '.join(header_parts)}")
    print("  " + "-" * (12 + 11 * len(all_multipliers)))

    max_mult = max(multipliers)
    all_max_srs = []
    for symbol in results:
        parts = [f"  {symbol:<12}"]
        for mult in all_multipliers:
            sr_val = results[symbol].get(mult, float("nan"))
            parts.append(f"{sr_val:>10.2f}")
        print(" ".join(parts))
        max_sr = results[symbol].get(max_mult, float("nan"))
        if not np.isnan(max_sr):
            all_max_srs.append(max_sr)

    # PASS/FAIL
    avg_sr_at_max = np.mean(all_max_srs) if all_max_srs else 0.0
    passed = avg_sr_at_max > 0
    n_positive = sum(1 for s in all_max_srs if s > 0)
    icon = "✅ PASS" if passed else "❌ FAIL"
    print(f"\n  {icon}  @ {max_mult:.1f}x cost: avg SR={avg_sr_at_max:.2f}, "
          f"{n_positive}/{len(all_max_srs)} symbols SR > 0")

    # 保存結果
    rows = []
    for sym, mults in results.items():
        row = {"symbol": sym}
        for m, sr_val in mults.items():
            row[f"sr_{m:.1f}x"] = round(sr_val, 4) if not np.isnan(sr_val) else None
        rows.append(row)
    pd.DataFrame(rows).to_csv(report_dir / "cost_stress.csv", index=False)

    return {
        "passed": bool(passed),
        "avg_sr_at_max_cost": round(float(avg_sr_at_max), 4),
        "n_positive": n_positive,
        "n_total": len(all_max_srs),
        "max_multiplier": max_mult,
        "per_symbol": results,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Delay Stress Test
# ══════════════════════════════════════════════════════════════════════════════

def run_delay_stress_test(
    symbols: List[str],
    data_paths: Dict[str, Path],
    cfg: dict,
    strategy_name: str,
    extra_delay_bars: int,
    max_sharpe_drop_pct: float,
    report_dir: Path,
    data_dir: Optional[Path] = None,
) -> Dict:
    """
    延遲壓力測試：在 signal_delay + extra_delay_bars 下檢查 SR 衰退。

    方法：先跑正常回測 (delay=1)，再把 pos shift(extra_delay_bars)
    重建 portfolio，比較 SR drop。

    PASS 條件：SR drop < max_sharpe_drop_pct%
    """
    from qtrade.backtest.run_backtest import (
        run_symbol_backtest,
        safe_portfolio_from_orders,
        to_vbt_direction,
        _bps_to_pct,
    )

    print("\n" + "=" * 72)
    print(f"  ⏱️  Delay Stress Test（延遲壓力測試，+{extra_delay_bars} bar）")
    print("     檢測策略對執行延遲的敏感度")
    print("=" * 72)

    results: Dict[str, Dict] = {}

    for symbol in symbols:
        if symbol not in data_paths:
            continue
        try:
            # 正常回測 (delay=1)
            bt_res = run_symbol_backtest(
                symbol, data_paths[symbol], cfg,
                strategy_name, data_dir=data_dir,
            )
            baseline_sr = bt_res.sharpe()

            # 額外延遲：shift pos by extra_delay_bars
            delayed_pos = bt_res.pos.shift(extra_delay_bars).fillna(0)

            # 重建 portfolio
            dr = cfg.get("direction", "both")
            vbt_dir = to_vbt_direction(dr)
            fee = _bps_to_pct(cfg["fee_bps"])
            slippage = _bps_to_pct(cfg["slippage_bps"])

            pf_delayed = safe_portfolio_from_orders(
                df=bt_res.df,
                pos=delayed_pos,
                fee=fee,
                slippage=slippage,
                init_cash=cfg["initial_cash"],
                freq=cfg.get("interval", "1h"),
                direction=vbt_dir,
            )
            delayed_stats = pf_delayed.stats()
            delayed_sr = float(delayed_stats.get("Sharpe Ratio", 0))

            # SR drop %
            if abs(baseline_sr) > 0.001:
                drop_pct = (baseline_sr - delayed_sr) / abs(baseline_sr) * 100
            else:
                drop_pct = 0.0

            results[symbol] = {
                "baseline_sr": round(baseline_sr, 4),
                "delayed_sr": round(delayed_sr, 4),
                "drop_pct": round(drop_pct, 1),
            }

            icon = "✅" if drop_pct < max_sharpe_drop_pct else "⚠️"
            print(f"  {icon} {symbol}: SR {baseline_sr:.2f} → {delayed_sr:.2f} "
                  f"(drop {drop_pct:+.1f}%)")

        except Exception as e:
            print(f"    ❌ {symbol}: {e}")

    # PASS/FAIL
    drops = [r["drop_pct"] for r in results.values()]
    avg_drop = np.mean(drops) if drops else 0.0
    passed = avg_drop < max_sharpe_drop_pct
    icon = "✅ PASS" if passed else "❌ FAIL"
    print(f"\n  {icon}  avg SR drop: {avg_drop:.1f}%（閾值: < {max_sharpe_drop_pct:.0f}%）")

    # 保存結果
    rows = [{"symbol": sym, **vals} for sym, vals in results.items()]
    pd.DataFrame(rows).to_csv(report_dir / "delay_stress.csv", index=False)

    return {
        "passed": bool(passed),
        "avg_drop_pct": round(float(avg_drop), 1),
        "max_drop_pct": round(float(max(drops)) if drops else 0.0, 1),
        "threshold_pct": max_sharpe_drop_pct,
        "extra_delay_bars": extra_delay_bars,
        "per_symbol": results,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Holdout OOS Test
# ══════════════════════════════════════════════════════════════════════════════

def run_holdout_oos(
    symbols: List[str],
    data_paths: Dict[str, Path],
    cfg: dict,
    strategy_name: str,
    periods: List[dict],
    max_degradation: float,
    report_dir: Path,
    data_dir: Optional[Path] = None,
) -> Dict:
    """
    Holdout Out-of-Sample 測試。

    對每個 period：
      - 用 train_start~train_end 跑回測 → IS SR
      - 用 test_start~now 跑回測 → OOS SR
      - 比較退化程度

    PASS 條件：OOS SR > IS SR × (1 - max_degradation)
    """
    from qtrade.backtest.run_backtest import run_symbol_backtest

    print("\n" + "=" * 72)
    print("  📦 Holdout OOS Test（保留樣本外測試）")
    print("     用 train 期間回測，再在 test 期間評估，測試策略泛化能力")
    print("=" * 72)

    all_results = {}

    for period in periods:
        pname = period["name"]
        train_start = period.get("train_start")
        train_end = period.get("train_end")
        test_start = period.get("test_start")

        print(f"\n  📌 Period: {pname}")
        print(f"     Train: {train_start} → {train_end}")
        print(f"     Test:  {test_start} → now")

        period_results = {}

        for symbol in symbols:
            if symbol not in data_paths:
                continue
            try:
                # IS backtest (train period)
                is_cfg = copy.deepcopy(cfg)
                is_cfg["start"] = train_start
                is_cfg["end"] = train_end
                is_res = run_symbol_backtest(
                    symbol, data_paths[symbol], is_cfg,
                    strategy_name, data_dir=data_dir,
                )
                is_sr = is_res.sharpe()

                # OOS backtest (test period → end of data)
                oos_cfg = copy.deepcopy(cfg)
                oos_cfg["start"] = test_start
                oos_cfg.pop("end", None)  # 不限制結束日期
                oos_res = run_symbol_backtest(
                    symbol, data_paths[symbol], oos_cfg,
                    strategy_name, data_dir=data_dir,
                )
                oos_sr = oos_res.sharpe()

                # 退化度
                if abs(is_sr) > 0.001:
                    degradation = (is_sr - oos_sr) / abs(is_sr)
                else:
                    degradation = 0.0

                period_results[symbol] = {
                    "is_sr": round(is_sr, 4),
                    "oos_sr": round(oos_sr, 4),
                    "degradation": round(degradation, 4),
                }

                passed_sym = oos_sr >= is_sr * (1 - max_degradation) if is_sr > 0 else oos_sr > 0
                icon = "✅" if passed_sym else "⚠️"
                print(f"    {icon} {symbol}: IS SR={is_sr:.2f}, "
                      f"OOS SR={oos_sr:.2f}, 退化={degradation:.1%}")

            except Exception as e:
                print(f"    ❌ {symbol}: {e}")

        all_results[pname] = period_results

    # 彙總 PASS/FAIL
    all_oos_srs = []
    all_degs = []
    for pname, presults in all_results.items():
        for sym, vals in presults.items():
            all_oos_srs.append(vals["oos_sr"])
            all_degs.append(vals["degradation"])

    avg_deg = np.mean(all_degs) if all_degs else 0.0
    avg_oos_sr = np.mean(all_oos_srs) if all_oos_srs else 0.0
    passed = avg_deg < max_degradation and avg_oos_sr > 0
    icon = "✅ PASS" if passed else "❌ FAIL"
    print(f"\n  {icon}  平均退化: {avg_deg:.1%}（閾值: < {max_degradation:.0%}），"
          f"平均 OOS SR: {avg_oos_sr:.2f}")

    # 保存結果
    rows = []
    for pname, presults in all_results.items():
        for sym, vals in presults.items():
            rows.append({"period": pname, "symbol": sym, **vals})
    if rows:
        pd.DataFrame(rows).to_csv(report_dir / "holdout_oos.csv", index=False)

    return {
        "passed": bool(passed),
        "avg_degradation": round(float(avg_deg), 4),
        "avg_oos_sr": round(float(avg_oos_sr), 4),
        "threshold": max_degradation,
        "per_period": all_results,
    }


def run_alpha_decay_check(
    symbols: List[str],
    data_paths: Dict[str, Path],
    cfg_obj,
    strategy_name: str,
    forward_bars: int,
    window_days: int,
    recent_days: int,
    recent_ic_min: float,
    max_decay_pct: float,
    max_critical_alerts: int,
    report_dir: Path,
    min_ic_denominator: float = 0.01,
) -> Dict:
    """
    Alpha Decay 檢查：監控 recent IC、IC 衰退和 critical alerts。

    PASS 條件（全部滿足）：
      1) 平均 recent IC >= recent_ic_min
      2) 平均 IC decay <= max_decay_pct
      3) critical alerts 數量 <= max_critical_alerts
    """
    from qtrade.strategy import get_strategy
    from qtrade.strategy.base import StrategyContext
    from qtrade.validation.ic_monitor import RollingICMonitor

    print("\n" + "=" * 72)
    print("  📉 Alpha Decay Check（因子衰減檢查）")
    print("     用 IC 追蹤信號有效性，檢測最近期是否衰退")
    print("=" * 72)

    bars_per_day = {
        "1m": 1440,
        "5m": 288,
        "15m": 96,
        "1h": 24,
        "4h": 6,
        "1d": 1,
    }
    interval = cfg_obj.market.interval
    bpd = bars_per_day.get(interval, 24)
    window = window_days * bpd

    strategy_func = get_strategy(strategy_name)
    monitor = RollingICMonitor(
        window=window,
        forward_bars=forward_bars,
        recent_days=recent_days,
        decay_threshold=max_decay_pct,
        interval=interval,
        min_ic_denominator=min_ic_denominator,
    )

    symbol_rows = []
    recent_ics = []
    decay_pcts = []
    total_critical_alerts = 0

    for symbol in symbols:
        if symbol not in data_paths:
            continue
        try:
            df = load_klines(data_paths[symbol])
            params = cfg_obj.strategy.get_params(symbol)
            ctx = StrategyContext(
                symbol=symbol,
                interval=interval,
                market_type=cfg_obj.market_type_str,
                direction=cfg_obj.direction,
                signal_delay=1,  # Backtest validation: execute on next open.
            )

            signals = strategy_func(df, ctx, params)
            report = monitor.compute(signals, df["close"])
            alerts = monitor.check_alerts(report)
            n_critical = sum(1 for a in alerts if a.severity == "critical")
            total_critical_alerts += n_critical

            recent_ics.append(float(report.recent_ic))
            decay_pcts.append(float(report.ic_decay_pct))
            symbol_rows.append({
                "symbol": symbol,
                "overall_ic": round(float(report.overall_ic), 6),
                "recent_ic": round(float(report.recent_ic), 6),
                "historical_ic": round(float(report.historical_ic), 6),
                "ic_decay_pct": round(float(report.ic_decay_pct), 6),
                "ic_ir": round(float(report.ic_ir), 6),
                "overall_ic_pvalue": round(float(report.overall_ic_pvalue), 6),
                "signal_count": int(report.signal_count),
                "n_alerts": int(len(alerts)),
                "n_critical_alerts": int(n_critical),
                "is_decaying": bool(report.is_decaying),
            })

            icon = "✅" if n_critical == 0 and report.recent_ic >= recent_ic_min else "⚠️"
            print(
                f"  {icon} {symbol}: recent IC={report.recent_ic:+.4f}, "
                f"decay={report.ic_decay_pct:+.1%}, critical={n_critical}"
            )
        except Exception as e:
            print(f"    ❌ {symbol}: {e}")

    avg_recent_ic = float(np.mean(recent_ics)) if recent_ics else 0.0
    avg_decay_pct = float(np.mean(decay_pcts)) if decay_pcts else 0.0
    passed = (
        avg_recent_ic >= recent_ic_min
        and avg_decay_pct <= max_decay_pct
        and total_critical_alerts <= max_critical_alerts
    )
    icon = "✅ PASS" if passed else "❌ FAIL"
    print(
        f"\n  {icon}  avg recent IC={avg_recent_ic:+.4f} (>= {recent_ic_min:.4f}), "
        f"avg decay={avg_decay_pct:+.1%} (<= {max_decay_pct:.0%}), "
        f"critical alerts={total_critical_alerts} (<= {max_critical_alerts})"
    )

    if symbol_rows:
        pd.DataFrame(symbol_rows).to_csv(report_dir / "alpha_decay.csv", index=False)

    return {
        "passed": bool(passed),
        "avg_recent_ic": round(avg_recent_ic, 6),
        "avg_decay_pct": round(avg_decay_pct, 6),
        "n_critical_alerts": int(total_critical_alerts),
        "recent_ic_min": float(recent_ic_min),
        "max_decay_pct": float(max_decay_pct),
        "max_critical_alerts": int(max_critical_alerts),
        "forward_bars": int(forward_bars),
        "window_days": int(window_days),
        "recent_days": int(recent_days),
        "per_symbol": symbol_rows,
    }


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
    cost_stress_results: Optional[Dict] = None,
    delay_stress_results: Optional[Dict] = None,
    holdout_oos_results: Optional[Dict] = None,
    alpha_decay_results: Optional[Dict] = None,
    embargo_holdout_results=None,
    regime_analysis_result=None,
    red_flag_results: Optional[Dict] = None,
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
    
    # Cost Stress 摘要
    if cost_stress_results:
        passed = cost_stress_results.get("passed", True)
        summary["tests"]["cost_stress"] = {
            "passed": bool(passed),
            "avg_sr_at_max_cost": cost_stress_results.get("avg_sr_at_max_cost", 0),
            "max_multiplier": cost_stress_results.get("max_multiplier", 2.0),
            "threshold": f"SR > 0 at {cost_stress_results.get('max_multiplier', 2.0):.1f}x cost",
            "meaning": "策略在雙倍成本下是否仍能盈利，測試對手續費和滑點的敏感度",
        }
        icon = "✅ PASS" if passed else "❌ FAIL"
        max_m = cost_stress_results.get("max_multiplier", 2.0)
        avg_sr = cost_stress_results.get("avg_sr_at_max_cost", 0)
        n_pos = cost_stress_results.get("n_positive", 0)
        n_tot = cost_stress_results.get("n_total", 0)
        print(f"  {icon}  Cost Stress（成本壓力測試）")
        print(f"         測試方法: 將手續費和滑點加倍，看策略是否仍有正收益")
        print(f"         @ {max_m:.1f}x cost: avg SR={avg_sr:.2f}, "
              f"{n_pos}/{n_tot} symbols SR > 0")
        print()

    # Delay Stress 摘要
    if delay_stress_results:
        passed = delay_stress_results.get("passed", True)
        summary["tests"]["delay_stress"] = {
            "passed": bool(passed),
            "avg_drop_pct": delay_stress_results.get("avg_drop_pct", 0),
            "threshold": f"SR drop < {delay_stress_results.get('threshold_pct', 50):.0f}%",
            "meaning": "策略對執行延遲的敏感度，延遲增加 1 bar 後 SR 衰退多少",
        }
        icon = "✅ PASS" if passed else "❌ FAIL"
        avg_drop = delay_stress_results.get("avg_drop_pct", 0)
        extra = delay_stress_results.get("extra_delay_bars", 1)
        thresh = delay_stress_results.get("threshold_pct", 50)
        print(f"  {icon}  Delay Stress（延遲壓力測試）")
        print(f"         測試方法: 在正常延遲上再加 +{extra} bar，看 SR 衰退多少")
        print(f"         avg SR drop: {avg_drop:.1f}%（標準: < {thresh:.0f}%）")
        print()

    # Holdout OOS 摘要
    if holdout_oos_results:
        passed = holdout_oos_results.get("passed", True)
        summary["tests"]["holdout_oos"] = {
            "passed": bool(passed),
            "avg_degradation": holdout_oos_results.get("avg_degradation", 0),
            "avg_oos_sr": holdout_oos_results.get("avg_oos_sr", 0),
            "threshold": f"OOS 退化 < {holdout_oos_results.get('threshold', 0.5):.0%}",
            "meaning": "用歷史數據訓練、用新數據測試，檢查策略是否只在過去有效",
        }
        icon = "✅ PASS" if passed else "❌ FAIL"
        avg_deg = holdout_oos_results.get("avg_degradation", 0)
        avg_oos_sr = holdout_oos_results.get("avg_oos_sr", 0)
        thresh = holdout_oos_results.get("threshold", 0.5)
        print(f"  {icon}  Holdout OOS（保留樣本外測試）")
        print(f"         測試方法: 劃分訓練/測試期間，比較 IS 和 OOS 績效退化")
        print(f"         平均退化: {avg_deg:.1%}（標準: < {thresh:.0%}），"
              f"avg OOS SR: {avg_oos_sr:.2f}")
        print()

    # Alpha Decay 摘要
    if alpha_decay_results:
        passed = alpha_decay_results.get("passed", True)
        avg_recent_ic = alpha_decay_results.get("avg_recent_ic", 0.0)
        avg_decay_pct = alpha_decay_results.get("avg_decay_pct", 0.0)
        n_critical = alpha_decay_results.get("n_critical_alerts", 0)
        recent_ic_min = alpha_decay_results.get("recent_ic_min", 0.02)
        max_decay_pct = alpha_decay_results.get("max_decay_pct", 0.5)
        max_critical = alpha_decay_results.get("max_critical_alerts", 0)
        summary["tests"]["alpha_decay"] = {
            "passed": bool(passed),
            "avg_recent_ic": avg_recent_ic,
            "avg_decay_pct": avg_decay_pct,
            "n_critical_alerts": n_critical,
            "threshold": (
                f"recent IC >= {recent_ic_min:.4f}, "
                f"decay <= {max_decay_pct:.0%}, "
                f"critical alerts <= {max_critical}"
            ),
            "meaning": "追蹤信號 IC 是否衰退；防止策略 edge 在近期失效",
        }
        icon = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {icon}  Alpha Decay（因子衰減檢查）")
        print(f"         測試方法: 比較歷史與近期 IC，並檢查 critical 警報")
        print(
            f"         avg recent IC: {avg_recent_ic:+.4f}（標準: >= {recent_ic_min:.4f}）, "
            f"avg decay: {avg_decay_pct:+.1%}（標準: <= {max_decay_pct:.0%}）, "
            f"critical: {n_critical}（標準: <= {max_critical}）"
        )
        print()

    # Regime Split 摘要
    if regime_analysis_result is not None:
        from qtrade.validation.regime_analysis import RegimeAnalysisResult
        if isinstance(regime_analysis_result, RegimeAnalysisResult):
            has_warnings = bool(regime_analysis_result.warnings)
            passed = not has_warnings
            summary["tests"]["regime_split"] = {
                "passed": bool(passed),
                "n_regimes": len(regime_analysis_result.regimes),
                "warnings": regime_analysis_result.warnings,
                "threshold": "所有 regime SR > 0",
                "meaning": "策略在不同市場環境（牛市/熊市/盤整）下是否都有正收益",
            }
            icon = "✅ PASS" if passed else "⚠️ CHECK"
            print(f"  {icon}  Regime Split（市場環境分段）")
            print(f"         測試方法: 根據 BTC drawdown 自動劃分牛市/熊市/盤整")
            if has_warnings:
                for w in regime_analysis_result.warnings:
                    print(f"         {w}")
            else:
                print(f"         所有 {len(regime_analysis_result.regimes)} 個 regime SR > 0")
            print()

    # Embargo Holdout OOS 摘要
    if embargo_holdout_results is not None:
        passed = embargo_holdout_results.passed
        summary["tests"]["embargo_holdout"] = {
            "passed": bool(passed),
            "avg_is_sr": embargo_holdout_results.avg_is_sr,
            "avg_oos_sr": embargo_holdout_results.avg_oos_sr,
            "avg_degradation_pct": embargo_holdout_results.avg_degradation_pct,
            "n_oos_positive": embargo_holdout_results.n_oos_positive,
            "n_total": embargo_holdout_results.n_total,
            "embargo_cutoff": embargo_holdout_results.embargo_cutoff,
            "embargo_months": embargo_holdout_results.embargo_months,
            "threshold": "OOS SR > 0 且衰退 < 50%",
            "meaning": "用研究從未接觸的最近數據做測試 — 最嚴格的過擬合檢查",
        }
        icon = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {icon}  Embargo Holdout（🔒 數據隔離樣本外測試）")
        print(f"         測試方法: 用最近 {embargo_holdout_results.embargo_months} 個月"
              f"（研究從未使用）的數據做 OOS 驗證")
        print(f"         IS SR: {embargo_holdout_results.avg_is_sr:.2f} → "
              f"OOS SR: {embargo_holdout_results.avg_oos_sr:.2f} "
              f"(Δ={embargo_holdout_results.avg_degradation_pct:+.1f}%)")
        print(f"         OOS 正收益: {embargo_holdout_results.n_oos_positive}"
              f"/{embargo_holdout_results.n_total}")
        print(f"         Cutoff: {embargo_holdout_results.embargo_cutoff}")
        print()

    # Red Flags 摘要
    if red_flag_results is not None:
        passed = red_flag_results.get("passed", True)
        n_flags = red_flag_results.get("n_flags", 0)
        summary["tests"]["red_flags"] = {
            "passed": bool(passed),
            "n_flags": n_flags,
            "threshold": "無紅旗",
            "meaning": "自動偵測 look-ahead bias、過擬合或信號泄漏的異常指標",
        }
        icon = "✅ PASS" if passed else "⚠️ CHECK"
        print(f"  {icon}  Red Flags（紅旗檢查）")
        print(f"         測試方法: 檢查 SR>4、MDD<3%、WinRate>70% 等異常指標")
        if n_flags > 0:
            print(f"         發現 {n_flags} 個紅旗，建議審查")
        else:
            print(f"         無異常指標")
        print()

    # 總體判斷
    all_passed = all(
        t.get("passed", True) 
        for t in summary["tests"].values()
    )
    summary["overall_passed"] = bool(all_passed)
    
    # 統計
    n_pass = sum(1 for t in summary["tests"].values() if t.get("passed", True))
    n_warn = sum(1 for t in summary["tests"].values() if not t.get("passed", True))
    n_total = len(summary["tests"])

    print("  ─────────────────────────────────────────────────────────────────")
    print(f"  📊 {n_pass} PASS / {n_warn} WARN-or-FAIL / {n_total} total tests")
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
        help="只執行指定的驗證（逗號分隔）: walk_forward,monte_carlo,regime_analysis,loao,regime,dsr,pbo,kelly,cost_stress,delay_stress,holdout_oos,alpha_decay,embargo_holdout,consistency,predeploy"
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
        val_cfg.cost_stress_enabled = False
        val_cfg.delay_stress_enabled = False
        val_cfg.holdout_oos_enabled = False
        val_cfg.alpha_decay_enabled = False
        val_cfg.embargo_holdout_enabled = False
    
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
    print("  本工具執行 11 項驗證，從不同角度檢查策略是否真的有效：")
    print("  ① Walk-Forward   — 策略在新數據上還行不行？")
    print("  ② Monte Carlo    — 最壞情況會虧多少？")
    print("  ③ Regime Split   — 牛市/熊市/盤整都行嗎？")
    print("  ④ Cross-Asset    — 換一個幣種還有效嗎？")
    print("  ⑤ DSR / PBO      — 是不是碰巧調出來的好結果？")
    print("  ⑥ Kelly          — 每次該下多大的注？")
    print("  ⑦ Cost Stress    — 手續費翻倍還賺嗎？")
    print("  ⑧ Delay Stress   — 信號晚一根 bar 還行嗎？")
    print("  ⑨ Holdout OOS    — 用新數據驗證有效嗎？")
    print("  ⑩ Alpha Decay    — 因子在近期有沒有失效？")
    print("  ⑪ Embargo OOS    — 用研究從未接觸的數據做最終驗證")
    print("  ⑫ Red Flags      — 有沒有過擬合的跡象？")
    print("  ⑬ Pre-Deploy     — 回測和實盤的程式碼一致嗎？")
    
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
    cost_stress_results = {}
    delay_stress_results = {}
    holdout_oos_results = {}
    alpha_decay_results = {}
    embargo_holdout_results = None
    regime_analysis_result = None
    red_flag_results = None
    
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
            data_dir=cfg.data_dir,
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
            data_dir=cfg.data_dir,
        )
    
    # 2.5. Regime-Specific Performance
    if should_run("regime_analysis", True):
        try:
            from qtrade.validation.regime_analysis import (
                compute_regime_performance,
                auto_detect_regimes,
                print_regime_report,
            )
            from qtrade.backtest.run_backtest import run_symbol_backtest

            # 對每個 symbol 跑一次回測取得 equity curve
            all_equities = {}
            for symbol in symbols:
                if symbol not in data_paths:
                    continue
                try:
                    bt_res = run_symbol_backtest(
                        symbol, data_paths[symbol], backtest_cfg,
                        cfg.strategy.name, data_dir=cfg.data_dir,
                    )
                    all_equities[symbol] = bt_res.equity()
                except Exception as e:
                    print(f"  ⚠️  {symbol}: regime analysis backtest 失敗: {e}")

            if all_equities:
                # 使用手動定義 + 自動偵測 regime
                regimes = list(val_cfg.market_regimes)  # 手動定義

                # 自動偵測（如果有 BTC 數據）
                btc_sym = next(
                    (s for s in ["BTCUSDT", "BTCBUSD"] if s in data_paths),
                    None,
                )
                if btc_sym:
                    btc_df = pd.read_parquet(data_paths[btc_sym])
                    auto_regimes = auto_detect_regimes(btc_df["close"])
                    regimes.extend(auto_regimes)

                # 用第一個 symbol（或 portfolio 平均）作為績效衡量
                # 對所有 symbol 的 equity 取等權平均
                eq_df = pd.DataFrame(all_equities)
                # 正規化為初始值 1，再取平均
                eq_norm = eq_df.div(eq_df.iloc[0])
                avg_equity = eq_norm.mean(axis=1)

                regime_analysis_result = compute_regime_performance(
                    avg_equity, regimes,
                )
                print_regime_report(regime_analysis_result)

                # 保存結果
                if regime_analysis_result.regimes:
                    regime_rows = []
                    for rp in regime_analysis_result.regimes:
                        regime_rows.append({
                            "name": rp.name,
                            "start": rp.start,
                            "end": rp.end,
                            "sharpe": rp.sharpe,
                            "total_return_pct": rp.total_return_pct,
                            "max_drawdown_pct": rp.max_drawdown_pct,
                            "win_rate": rp.win_rate,
                            "n_bars": rp.n_bars,
                            "annualized_return_pct": rp.annualized_return_pct,
                        })
                    regime_df = pd.DataFrame(regime_rows)
                    regime_df.to_csv(
                        report_dir / "regime_analysis.csv", index=False,
                    )
        except Exception as e:
            print(f"  ⚠️  Regime analysis 失敗: {e}")

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
            data_dir=cfg.data_dir,
        )
    
    # 4. Prado Methods (需要 walk-forward 結果)
    run_dsr = should_run("dsr", val_cfg.dsr_enabled)
    run_pbo = should_run("pbo", val_cfg.pbo_enabled)
    
    if run_dsr or run_pbo:
        prado_results = run_prado_methods(
            symbols=symbols,
            walk_forward_results=walk_forward_results,
            cfg=backtest_cfg,
            dsr_enabled=run_dsr,
            dsr_n_trials=val_cfg.dsr_n_trials,
            pbo_enabled=run_pbo,
            pbo_threshold=val_cfg.pbo_threshold,
            report_dir=report_dir,
            data_paths=data_paths,
            data_dir=cfg.data_dir,
        )
    
    # 5. Kelly Validation
    if should_run("kelly", val_cfg.kelly_enabled):
        kelly_results = run_kelly_validation(
            symbols=symbols,
            data_paths=data_paths,
            cfg=backtest_cfg,
            kelly_fractions=val_cfg.kelly_fractions,
            report_dir=report_dir,
            data_dir=cfg.data_dir,
        )
    
    # 6. Cost Stress Test
    if should_run("cost_stress", val_cfg.cost_stress_enabled):
        cost_stress_results = run_cost_stress_test(
            symbols=symbols,
            data_paths=data_paths,
            cfg=backtest_cfg,
            strategy_name=cfg.strategy.name,
            multipliers=val_cfg.cost_stress_multipliers,
            report_dir=report_dir,
            data_dir=cfg.data_dir,
        )

    # 7. Delay Stress Test
    if should_run("delay_stress", val_cfg.delay_stress_enabled):
        delay_stress_results = run_delay_stress_test(
            symbols=symbols,
            data_paths=data_paths,
            cfg=backtest_cfg,
            strategy_name=cfg.strategy.name,
            extra_delay_bars=val_cfg.delay_stress_extra_bars,
            max_sharpe_drop_pct=val_cfg.delay_stress_max_drop_pct,
            report_dir=report_dir,
            data_dir=cfg.data_dir,
        )

    # 8. Holdout OOS Test
    if should_run("holdout_oos", val_cfg.holdout_oos_enabled) and val_cfg.holdout_oos_periods:
        holdout_oos_results = run_holdout_oos(
            symbols=symbols,
            data_paths=data_paths,
            cfg=backtest_cfg,
            strategy_name=cfg.strategy.name,
            periods=val_cfg.holdout_oos_periods,
            max_degradation=val_cfg.holdout_oos_max_degradation,
            report_dir=report_dir,
            data_dir=cfg.data_dir,
        )

    # 8.2. Alpha Decay Check
    if should_run("alpha_decay", val_cfg.alpha_decay_enabled):
        alpha_decay_results = run_alpha_decay_check(
            symbols=symbols,
            data_paths=data_paths,
            cfg_obj=cfg,
            strategy_name=cfg.strategy.name,
            forward_bars=val_cfg.alpha_decay_forward_bars,
            window_days=val_cfg.alpha_decay_window_days,
            recent_days=val_cfg.alpha_decay_recent_days,
            recent_ic_min=val_cfg.alpha_decay_recent_ic_min,
            max_decay_pct=val_cfg.alpha_decay_max_decay_pct,
            max_critical_alerts=val_cfg.alpha_decay_max_critical_alerts,
            report_dir=report_dir,
            min_ic_denominator=val_cfg.alpha_decay_min_ic_denominator,
        )

    # 8.5. Embargo Holdout OOS Test（數據隔離樣本外測試）
    if should_run("embargo_holdout", val_cfg.embargo_holdout_enabled):
        try:
            from qtrade.validation.embargo import (
                load_embargo_config,
                run_embargo_holdout_test,
            )
            embargo_cfg = load_embargo_config(val_cfg.validation_config_path)
            if embargo_cfg.temporal.enabled:
                embargo_holdout_results = run_embargo_holdout_test(
                    symbols=symbols,
                    data_paths=data_paths,
                    cfg=backtest_cfg,
                    strategy_name=cfg.strategy.name,
                    embargo=embargo_cfg,
                    max_degradation=val_cfg.embargo_holdout_max_degradation,
                    data_dir=cfg.data_dir,
                )
        except Exception as e:
            print(f"  ⚠️  Embargo holdout test 失敗: {e}")

    # 9. Consistency Check (需要 Paper Trading 運行中)
    if should_run("consistency", val_cfg.consistency_enabled):
        run_consistency_check(
            symbols=symbols,
            cfg=cfg,
            days=val_cfg.consistency_days,
            report_dir=report_dir,
        )
    
    # 10. Pre-Deploy 一致性檢查（回測↔實盤路徑比對）
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
    
    # 11. Red Flag Check（對每個 symbol 的回測結果）
    try:
        from qtrade.validation.red_flags import check_red_flags, print_red_flags
        from qtrade.backtest.run_backtest import run_symbol_backtest as _bt

        print("\n" + "=" * 72)
        print("  🚩 Red Flag Check（回測紅旗檢查）")
        print("     自動偵測可能的 look-ahead bias、過擬合或信號泄漏")
        print("=" * 72)

        all_flags = []
        for symbol in symbols:
            if symbol not in data_paths:
                continue
            try:
                bt_res = _bt(
                    symbol, data_paths[symbol], backtest_cfg,
                    cfg.strategy.name, data_dir=cfg.data_dir,
                )
                stats = bt_res.pf.stats()
                flags = check_red_flags(stats, thresholds=val_cfg.red_flags_thresholds)
                if flags:
                    print(f"\n  {symbol}:")
                    for flag in flags:
                        print(f"    {flag.emoji} {flag.metric} = {flag.value:.2f} ({flag.threshold})")
                        print(f"       → {flag.explanation}")
                    all_flags.extend(flags)
            except Exception:
                pass

        if not all_flags:
            print("\n  ✅ 所有幣種均無紅旗")
        else:
            print(f"\n  ⚠️  共 {len(all_flags)} 個紅旗，建議審查")

        red_flag_results = {
            "passed": len(all_flags) == 0,
            "n_flags": len(all_flags),
            "flags": [f"{f.emoji} {f.metric}={f.value:.2f}" for f in all_flags],
        }
    except Exception as e:
        print(f"  ⚠️  Red flag check 失敗: {e}")

    # 12. 生成摘要
    generate_summary(
        walk_forward_results=walk_forward_results,
        monte_carlo_results=monte_carlo_results,
        cross_asset_results=cross_asset_results,
        prado_results=prado_results,
        kelly_results=kelly_results,
        report_dir=report_dir,
        cost_stress_results=cost_stress_results or None,
        delay_stress_results=delay_stress_results or None,
        holdout_oos_results=holdout_oos_results or None,
        alpha_decay_results=alpha_decay_results or None,
        embargo_holdout_results=embargo_holdout_results,
        regime_analysis_result=regime_analysis_result,
        red_flag_results=red_flag_results,
    )
    
    print(f"\n📁 報告已保存至: {report_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
