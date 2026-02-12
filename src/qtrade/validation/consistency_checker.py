"""
回測與實盤一致性檢查器

系統化檢測回測邏輯與實盤邏輯之間的差異，避免常見陷阱。

常見問題來源：
    1. 未來資訊洩漏 (Look-ahead Bias)
    2. 信號時序問題（回測用 close，實盤用 open）
    3. 訂單執行差異（滑點、部分成交）
    4. 手續費計算差異
    5. 最小交易單位未考慮
    6. 數據質量問題
    7. 狀態管理不同步

使用方法：
    from qtrade.validation import ConsistencyChecker
    
    checker = ConsistencyChecker(
        strategy_name="rsi_adx_atr",
        backtest_cfg=cfg,
    )
    
    # 運行所有檢查
    report = checker.run_all_checks(df)
    print(report.summary())
    
    # 或單獨運行某項檢查
    checker.check_lookahead_bias(df)
    checker.check_signal_timing(df)
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Any
import numpy as np
import pandas as pd

from ..utils.log import get_logger
from ..strategy.base import StrategyContext
from ..strategy import get_strategy

logger = get_logger("consistency_checker")


# ══════════════════════════════════════════════════════════════
# 檢查結果
# ══════════════════════════════════════════════════════════════

@dataclass
class CheckResult:
    """單項檢查結果"""
    name: str
    passed: bool
    severity: str  # "error", "warning", "info"
    message: str
    details: dict = field(default_factory=dict)


@dataclass
class ConsistencyReport:
    """一致性檢查報告"""
    checks: list[CheckResult] = field(default_factory=list)
    
    @property
    def passed(self) -> bool:
        """是否所有檢查都通過（忽略 info）"""
        return all(c.passed for c in self.checks if c.severity != "info")
    
    @property
    def errors(self) -> list[CheckResult]:
        return [c for c in self.checks if c.severity == "error" and not c.passed]
    
    @property
    def warnings(self) -> list[CheckResult]:
        return [c for c in self.checks if c.severity == "warning" and not c.passed]
    
    def summary(self) -> str:
        """生成摘要報告"""
        lines = [
            "=" * 60,
            "🔍 回測與實盤一致性檢查報告",
            "=" * 60,
            "",
        ]
        
        # 統計
        total = len(self.checks)
        passed = sum(1 for c in self.checks if c.passed)
        errors = len(self.errors)
        warnings_count = len(self.warnings)
        
        status = "✅ 通過" if self.passed else "❌ 有問題"
        lines.append(f"狀態: {status}")
        lines.append(f"檢查項目: {passed}/{total} 通過")
        if errors:
            lines.append(f"🔴 錯誤: {errors}")
        if warnings_count:
            lines.append(f"🟡 警告: {warnings_count}")
        lines.append("")
        
        # 詳細結果
        for check in self.checks:
            emoji = "✅" if check.passed else ("🔴" if check.severity == "error" else "🟡")
            lines.append(f"{emoji} [{check.severity.upper()}] {check.name}")
            lines.append(f"   {check.message}")
            if check.details:
                for k, v in check.details.items():
                    lines.append(f"   • {k}: {v}")
            lines.append("")
        
        lines.append("=" * 60)
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# 一致性檢查器
# ══════════════════════════════════════════════════════════════

class ConsistencyChecker:
    """
    回測與實盤一致性檢查器
    
    系統化檢測常見的回測陷阱和實盤差異。
    """
    
    def __init__(
        self,
        strategy_name: str,
        backtest_cfg: dict,
        market_type: str = "spot",
        direction: str = "both",
    ):
        self.strategy_name = strategy_name
        self.backtest_cfg = backtest_cfg
        self.market_type = market_type
        self.direction = direction
        self.strategy_func = get_strategy(strategy_name)
        
        self._report = ConsistencyReport()
    
    def run_all_checks(self, df: pd.DataFrame) -> ConsistencyReport:
        """運行所有檢查"""
        self._report = ConsistencyReport()
        
        logger.info("🔍 開始一致性檢查...")
        
        # 1. 數據質量檢查
        self.check_data_quality(df)
        
        # 2. 未來資訊洩漏檢查
        self.check_lookahead_bias(df)
        
        # 3. 信號時序檢查
        self.check_signal_timing(df)
        
        # 4. 交易成本檢查
        self.check_trading_costs(df)
        
        # 5. 最小交易單位檢查
        self.check_min_trade_size(df)
        
        # 6. 信號一致性檢查（回測 vs 逐 bar）
        self.check_signal_consistency(df)
        
        # 7. 邊界條件檢查
        self.check_edge_cases(df)
        
        # 8. 狀態管理檢查
        self.check_state_management(df)
        
        logger.info(f"檢查完成: {len(self._report.checks)} 項")
        return self._report
    
    # ══════════════════════════════════════════════════════════════
    # 個別檢查方法
    # ══════════════════════════════════════════════════════════════
    
    def check_data_quality(self, df: pd.DataFrame) -> CheckResult:
        """檢查數據質量"""
        issues = []
        
        # 檢查缺失值
        missing = df[["open", "high", "low", "close", "volume"]].isnull().sum()
        if missing.any():
            issues.append(f"缺失值: {missing[missing > 0].to_dict()}")
        
        # 檢查時間連續性
        if isinstance(df.index, pd.DatetimeIndex):
            time_diff = df.index.to_series().diff()
            expected_freq = time_diff.mode().iloc[0] if len(time_diff.mode()) > 0 else None
            gaps = time_diff[time_diff > expected_freq * 2] if expected_freq else pd.Series()
            if len(gaps) > 0:
                issues.append(f"時間間隙: {len(gaps)} 處")
        
        # 檢查異常值
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            if col in df.columns:
                pct_change = df[col].pct_change().abs()
                extreme_moves = (pct_change > 0.5).sum()  # 單根 K 線 >50% 變動
                if extreme_moves > 0:
                    issues.append(f"{col} 異常波動: {extreme_moves} 根")
        
        # 檢查 OHLC 邏輯
        if all(c in df.columns for c in ["open", "high", "low", "close"]):
            invalid_ohlc = (
                (df["high"] < df["low"]) |
                (df["high"] < df["open"]) |
                (df["high"] < df["close"]) |
                (df["low"] > df["open"]) |
                (df["low"] > df["close"])
            ).sum()
            if invalid_ohlc > 0:
                issues.append(f"OHLC 邏輯錯誤: {invalid_ohlc} 根")
        
        passed = len(issues) == 0
        result = CheckResult(
            name="數據質量檢查",
            passed=passed,
            severity="error" if not passed else "info",
            message="數據質量良好" if passed else f"發現 {len(issues)} 個問題",
            details={"issues": issues} if issues else {},
        )
        self._report.checks.append(result)
        return result
    
    def check_lookahead_bias(self, df: pd.DataFrame) -> CheckResult:
        """
        檢查未來資訊洩漏
        
        方法：比較使用完整數據 vs 逐步增量數據的信號是否一致
        """
        ctx = StrategyContext(
            symbol="TEST",
            interval="1h",
            market_type=self.market_type,
            direction=self.direction,
        )
        params = self.backtest_cfg.get("strategy_params", {})
        
        # 使用完整數據計算信號
        full_signals = self.strategy_func(df, ctx, params)
        
        # 逐步計算信號（模擬實盤）
        incremental_signals = pd.Series(index=df.index, dtype=float)
        
        # 從一個安全的起點開始（避免指標計算期）
        warmup = max(50, len(df) // 10)
        
        for i in range(warmup, len(df)):
            # 只使用到目前為止的數據
            partial_df = df.iloc[:i+1].copy()
            try:
                partial_signals = self.strategy_func(partial_df, ctx, params)
                incremental_signals.iloc[i] = partial_signals.iloc[-1]
            except Exception:
                incremental_signals.iloc[i] = np.nan
        
        # 比較差異
        valid_idx = incremental_signals.notna()
        if valid_idx.sum() > 0:
            diff = (full_signals[valid_idx] - incremental_signals[valid_idx]).abs()
            mismatch_count = (diff > 0.01).sum()
            mismatch_pct = mismatch_count / valid_idx.sum() * 100
        else:
            mismatch_count = 0
            mismatch_pct = 0
        
        passed = mismatch_pct < 1.0  # 允許 1% 的微小差異（浮點誤差）
        
        result = CheckResult(
            name="未來資訊洩漏檢查 (Look-ahead Bias)",
            passed=passed,
            severity="error" if not passed else "info",
            message=f"信號不一致率: {mismatch_pct:.2f}%" if not passed else "未發現未來資訊洩漏",
            details={
                "mismatch_count": int(mismatch_count),
                "mismatch_pct": f"{mismatch_pct:.2f}%",
                "total_checked": int(valid_idx.sum()),
            },
        )
        self._report.checks.append(result)
        return result
    
    def check_signal_timing(self, df: pd.DataFrame) -> CheckResult:
        """
        檢查信號時序
        
        確保策略不會在當前 bar 結束前就使用當前 bar 的數據做決策
        """
        ctx = StrategyContext(
            symbol="TEST",
            interval="1h",
            market_type=self.market_type,
            direction=self.direction,
        )
        params = self.backtest_cfg.get("strategy_params", {})
        
        # 計算信號
        signals = self.strategy_func(df, ctx, params)
        
        # 檢查信號是否有 shift（應該 shift(1) 避免使用當前 bar 數據）
        # 方法：如果信號和 close 的相關性異常高，可能有問題
        if len(signals) > 100:
            # 計算信號變化和價格變化的相關性
            signal_change = signals.diff()
            price_change = df["close"].pct_change()
            
            # 同期相關性（應該低，因為信號基於歷史）
            same_bar_corr = signal_change.corr(price_change)
            
            # 滯後相關性（如果策略有效，應該有一定相關性）
            lagged_corr = signal_change.shift(1).corr(price_change)
            
            # 如果同期相關性異常高，可能有問題
            suspicious = abs(same_bar_corr) > 0.5
        else:
            same_bar_corr = 0
            lagged_corr = 0
            suspicious = False
        
        passed = not suspicious
        
        result = CheckResult(
            name="信號時序檢查",
            passed=passed,
            severity="warning" if not passed else "info",
            message="信號與當前價格相關性異常高，可能有時序問題" if not passed else "信號時序正常",
            details={
                "same_bar_correlation": f"{same_bar_corr:.3f}",
                "lagged_correlation": f"{lagged_corr:.3f}",
            },
        )
        self._report.checks.append(result)
        return result
    
    def check_trading_costs(self, df: pd.DataFrame) -> CheckResult:
        """
        檢查交易成本設定
        
        確保手續費和滑點設定合理
        """
        fee_bps = self.backtest_cfg.get("fee_bps", 0)
        slippage_bps = self.backtest_cfg.get("slippage_bps", 0)
        
        issues = []
        
        # Binance 現貨手續費通常是 0.1%（10 bps）
        if fee_bps < 5:
            issues.append(f"手續費可能設太低: {fee_bps} bps（Binance 通常 10 bps）")
        
        # 滑點應該有一些
        if slippage_bps == 0:
            issues.append("滑點設為 0，實盤可能會有 1-5 bps 滑點")
        
        # 總成本檢查
        total_cost = fee_bps + slippage_bps
        if total_cost < 10:
            issues.append(f"總交易成本可能低估: {total_cost} bps")
        
        passed = len(issues) == 0
        
        result = CheckResult(
            name="交易成本檢查",
            passed=passed,
            severity="warning" if not passed else "info",
            message="交易成本設定合理" if passed else "交易成本設定可能不準確",
            details={
                "fee_bps": fee_bps,
                "slippage_bps": slippage_bps,
                "total_cost_bps": total_cost,
                "issues": issues,
            },
        )
        self._report.checks.append(result)
        return result
    
    def check_min_trade_size(self, df: pd.DataFrame) -> CheckResult:
        """
        檢查最小交易單位
        
        確保回測中的交易數量符合交易所要求
        """
        # Binance 的最小交易金額通常是 10 USDT
        min_notional = 10.0
        initial_cash = self.backtest_cfg.get("initial_cash", 10000)
        
        # 如果初始資金太小，可能無法執行所有交易
        issues = []
        
        avg_price = df["close"].mean()
        min_qty = min_notional / avg_price
        
        # 估算每次交易的金額
        # 假設滿倉交易
        trade_value = initial_cash
        
        if trade_value < min_notional:
            issues.append(f"初始資金 ${initial_cash} 可能低於最小交易金額 ${min_notional}")
        
        # 檢查是否考慮了數量精度
        # Binance 通常要求特定的精度（如 BTC 是 0.00001）
        
        passed = len(issues) == 0
        
        result = CheckResult(
            name="最小交易單位檢查",
            passed=passed,
            severity="warning" if not passed else "info",
            message="交易單位設定合理" if passed else "可能有最小交易單位問題",
            details={
                "initial_cash": initial_cash,
                "min_notional": min_notional,
                "avg_price": f"{avg_price:.2f}",
                "issues": issues,
            },
        )
        self._report.checks.append(result)
        return result
    
    def check_signal_consistency(self, df: pd.DataFrame) -> CheckResult:
        """
        檢查信號一致性
        
        多次運行策略，確保結果完全一致（無隨機性）
        """
        ctx = StrategyContext(
            symbol="TEST",
            interval="1h",
            market_type=self.market_type,
            direction=self.direction,
        )
        params = self.backtest_cfg.get("strategy_params", {})
        
        # 運行多次
        results = []
        for _ in range(3):
            signals = self.strategy_func(df, ctx, params)
            results.append(signals.values)
        
        # 檢查是否完全一致
        all_same = all(np.allclose(results[0], r, equal_nan=True) for r in results[1:])
        
        result = CheckResult(
            name="信號一致性檢查（無隨機性）",
            passed=all_same,
            severity="error" if not all_same else "info",
            message="策略包含隨機性，每次運行結果不同" if not all_same else "信號完全一致",
            details={},
        )
        self._report.checks.append(result)
        return result
    
    def check_edge_cases(self, df: pd.DataFrame) -> CheckResult:
        """
        檢查邊界條件
        
        測試策略在極端情況下的行為
        """
        ctx = StrategyContext(
            symbol="TEST",
            interval="1h",
            market_type=self.market_type,
            direction=self.direction,
        )
        params = self.backtest_cfg.get("strategy_params", {})
        
        issues = []
        
        # 1. 測試空數據
        try:
            empty_df = pd.DataFrame(columns=df.columns)
            self.strategy_func(empty_df, ctx, params)
            issues.append("策略沒有處理空數據的情況")
        except (ValueError, IndexError, KeyError):
            pass  # 預期會拋出異常
        except Exception as e:
            issues.append(f"空數據時拋出意外異常: {type(e).__name__}")
        
        # 2. 測試只有一根 K 線
        try:
            single_bar = df.iloc[:1].copy()
            result = self.strategy_func(single_bar, ctx, params)
            if len(result) != 1:
                issues.append("單根 K 線時返回長度不正確")
        except Exception as e:
            # 可以接受拋出異常（指標計算期不足）
            pass
        
        # 3. 測試極端價格變動
        extreme_df = df.copy()
        extreme_df.loc[extreme_df.index[len(extreme_df)//2], "close"] *= 2  # 模擬閃崩
        try:
            result = self.strategy_func(extreme_df, ctx, params)
            if result.isna().all():
                issues.append("極端價格變動導致所有信號為 NaN")
        except Exception as e:
            issues.append(f"極端價格變動時崩潰: {type(e).__name__}")
        
        # 4. 測試全零成交量
        zero_vol_df = df.copy()
        zero_vol_df["volume"] = 0
        try:
            result = self.strategy_func(zero_vol_df, ctx, params)
            # 應該能正常運行
        except Exception as e:
            issues.append(f"零成交量時崩潰: {type(e).__name__}")
        
        # 5. 測試信號範圍
        signals = self.strategy_func(df, ctx, params)
        out_of_range = ((signals < -1.01) | (signals > 1.01)).sum()
        if out_of_range > 0:
            issues.append(f"信號超出 [-1, 1] 範圍: {out_of_range} 根")
        
        passed = len(issues) == 0
        
        result = CheckResult(
            name="邊界條件檢查",
            passed=passed,
            severity="warning" if not passed else "info",
            message=f"發現 {len(issues)} 個邊界問題" if not passed else "邊界條件處理正常",
            details={"issues": issues},
        )
        self._report.checks.append(result)
        return result
    
    def check_state_management(self, df: pd.DataFrame) -> CheckResult:
        """
        檢查狀態管理
        
        確保策略在分段運行時狀態正確延續
        """
        ctx = StrategyContext(
            symbol="TEST",
            interval="1h",
            market_type=self.market_type,
            direction=self.direction,
        )
        params = self.backtest_cfg.get("strategy_params", {})
        
        # 完整運行
        full_signals = self.strategy_func(df, ctx, params)
        
        # 分段運行（模擬實盤的分批數據）
        split_point = len(df) // 2
        
        # 第一段
        first_half = df.iloc[:split_point].copy()
        first_signals = self.strategy_func(first_half, ctx, params)
        
        # 第二段（包含重疊以計算指標）
        overlap = 100  # 重疊 100 根用於指標計算
        second_half = df.iloc[split_point-overlap:].copy()
        second_signals = self.strategy_func(second_half, ctx, params)
        
        # 比較重疊部分是否一致
        overlap_full = full_signals.iloc[split_point:split_point+10]
        overlap_second = second_signals.iloc[overlap:overlap+10]
        
        if len(overlap_full) > 0 and len(overlap_second) > 0:
            diff = (overlap_full.values - overlap_second.values)
            max_diff = np.abs(diff).max() if len(diff) > 0 else 0
            consistent = max_diff < 0.01
        else:
            consistent = True
            max_diff = 0
        
        result = CheckResult(
            name="狀態管理檢查（分段運行一致性）",
            passed=consistent,
            severity="warning" if not consistent else "info",
            message="分段運行結果不一致，可能有狀態管理問題" if not consistent else "分段運行結果一致",
            details={
                "max_difference": f"{max_diff:.4f}",
            },
        )
        self._report.checks.append(result)
        return result


# ══════════════════════════════════════════════════════════════
# 快捷函數
# ══════════════════════════════════════════════════════════════

def check_strategy_consistency(
    strategy_name: str,
    df: pd.DataFrame,
    cfg: dict,
    market_type: str = "spot",
    direction: str = "both",
) -> ConsistencyReport:
    """
    快捷函數：運行所有一致性檢查
    
    Args:
        strategy_name: 策略名稱
        df: K 線數據
        cfg: 回測配置
        market_type: 市場類型
        direction: 交易方向
    
    Returns:
        ConsistencyReport
    """
    checker = ConsistencyChecker(
        strategy_name=strategy_name,
        backtest_cfg=cfg,
        market_type=market_type,
        direction=direction,
    )
    return checker.run_all_checks(df)
