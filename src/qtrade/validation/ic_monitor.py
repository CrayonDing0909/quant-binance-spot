"""
Information Coefficient (IC) 監控模組

用於追蹤策略信號品質、偵測 Alpha Decay。

IC = Spearman Rank Correlation(signal, forward_return)

- IC > 0.05: 有效信號
- IC ≈ 0: 信號失效 / Alpha 衰退
- IC < -0.05: 信號反轉

使用方式:
    from qtrade.validation.ic_monitor import RollingICMonitor
    monitor = RollingICMonitor(window=180*24, forward_bars=24)
    report = monitor.compute(signals, forward_returns)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# 數據結構
# ══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ICReport:
    """IC 分析報告"""
    # 全局 IC
    overall_ic: float
    overall_ic_pvalue: float
    
    # Rolling IC 統計
    rolling_ic: pd.Series  # 滾動 IC 時間序列
    avg_ic: float
    ic_std: float
    ic_ir: float  # IC Information Ratio = mean(IC) / std(IC)
    
    # Alpha Decay 偵測
    recent_ic: float  # 最近 N 天的 IC
    historical_ic: float  # 較早期的 IC
    ic_decay_pct: float  # IC 衰退百分比
    is_decaying: bool  # 是否正在衰退
    
    # 時間分段分析
    yearly_ic: dict  # {year: IC}
    
    # 信號分佈
    signal_count: int
    active_signal_pct: float  # 非零信號的比例


@dataclass(frozen=True)
class AlphaDecayAlert:
    """Alpha Decay 警報"""
    severity: str  # "info", "warning", "critical"
    message: str
    metric: str
    current_value: float
    threshold: float


# ══════════════════════════════════════════════════════════════
# IC 監控器
# ══════════════════════════════════════════════════════════════

class RollingICMonitor:
    """
    Rolling Information Coefficient 監控器
    
    追蹤策略信號與未來收益的 rank correlation，
    用於偵測 Alpha Decay。
    
    Args:
        window: 滾動窗口大小（bar 數，如 180*24 = 180 天 for 1h data）
        forward_bars: 前瞻收益計算期（bar 數，如 24 = 24 小時）
        min_observations: 計算 IC 所需的最小觀測數
        decay_threshold: IC 衰退閾值（超過此比例視為衰退）
        recent_days: "最近"期間的天數（用於衰退偵測）
    """
    
    def __init__(
        self,
        window: int = 180 * 24,  # 180 天 (for 1h data)
        forward_bars: int = 24,  # 24 小時前瞻
        min_observations: int = 100,
        decay_threshold: float = 0.50,  # IC 下降超過 50% 視為衰退
        recent_days: int = 180,
        interval: str = "1h",
    ):
        self.window = window
        self.forward_bars = forward_bars
        self.min_observations = min_observations
        self.decay_threshold = decay_threshold
        self.recent_days = recent_days
        self.interval = interval
        
        # 根據 interval 計算 bars per day
        bars_per_day_map = {
            "1m": 1440, "5m": 288, "15m": 96,
            "1h": 24, "4h": 6, "1d": 1,
        }
        self.bars_per_day = bars_per_day_map.get(interval, 24)
    
    def compute(
        self,
        signals: pd.Series,
        prices: pd.Series,
    ) -> ICReport:
        """
        計算完整的 IC 報告
        
        Args:
            signals: 策略信號序列（-1 ~ 1）
            prices: 收盤價序列
            
        Returns:
            ICReport
        """
        # 計算前瞻收益
        forward_returns = prices.pct_change(self.forward_bars).shift(-self.forward_bars)
        
        # 對齊
        valid = signals.notna() & forward_returns.notna() & (signals != 0)
        sig = signals[valid]
        fwd = forward_returns[valid]
        
        if len(sig) < self.min_observations:
            logger.warning(f"觀測數不足: {len(sig)} < {self.min_observations}")
            return self._empty_report()
        
        # 全局 IC
        overall_ic, overall_pvalue = stats.spearmanr(sig, fwd)
        
        # Rolling IC
        rolling_ic = self._compute_rolling_ic(signals, forward_returns)
        
        # IC 統計
        valid_ic = rolling_ic.dropna()
        avg_ic = valid_ic.mean()
        ic_std = valid_ic.std()
        ic_ir = avg_ic / ic_std if ic_std > 0 else 0.0
        
        # Alpha Decay 偵測
        recent_bars = self.recent_days * self.bars_per_day
        if len(valid_ic) > recent_bars * 2:
            recent_ic = valid_ic.iloc[-recent_bars:].mean()
            historical_ic = valid_ic.iloc[:-recent_bars].mean()
        else:
            half = len(valid_ic) // 2
            recent_ic = valid_ic.iloc[half:].mean()
            historical_ic = valid_ic.iloc[:half].mean()
        
        ic_decay_pct = 1.0 - (recent_ic / historical_ic) if historical_ic != 0 else 0.0
        is_decaying = ic_decay_pct > self.decay_threshold
        
        # 年度分析
        yearly_ic = self._compute_yearly_ic(signals, forward_returns)
        
        # 信號分佈
        signal_count = len(sig)
        active_signal_pct = (signals != 0).sum() / len(signals) if len(signals) > 0 else 0.0
        
        return ICReport(
            overall_ic=overall_ic,
            overall_ic_pvalue=overall_pvalue,
            rolling_ic=rolling_ic,
            avg_ic=avg_ic,
            ic_std=ic_std,
            ic_ir=ic_ir,
            recent_ic=recent_ic,
            historical_ic=historical_ic,
            ic_decay_pct=ic_decay_pct,
            is_decaying=is_decaying,
            yearly_ic=yearly_ic,
            signal_count=signal_count,
            active_signal_pct=active_signal_pct,
        )
    
    def check_alerts(self, report: ICReport) -> list[AlphaDecayAlert]:
        """
        根據 IC 報告產生警報
        
        Returns:
            警報清單（按嚴重程度排序）
        """
        alerts = []
        
        # Alert 1: IC 衰退超過閾值
        if report.is_decaying:
            alerts.append(AlphaDecayAlert(
                severity="critical",
                message=(
                    f"Alpha Decay 偵測：IC 從 {report.historical_ic:.4f} "
                    f"下降到 {report.recent_ic:.4f} "
                    f"（衰退 {report.ic_decay_pct:.0%}）"
                ),
                metric="ic_decay_pct",
                current_value=report.ic_decay_pct,
                threshold=self.decay_threshold,
            ))
        
        # Alert 2: 最近 IC 接近零
        if abs(report.recent_ic) < 0.02:
            alerts.append(AlphaDecayAlert(
                severity="critical",
                message=f"最近 {self.recent_days} 天 IC ≈ 0 ({report.recent_ic:.4f})，信號可能已失效",
                metric="recent_ic",
                current_value=report.recent_ic,
                threshold=0.02,
            ))
        
        # Alert 3: IC IR 太低
        if report.ic_ir < 0.3 and report.ic_ir > 0:
            alerts.append(AlphaDecayAlert(
                severity="warning",
                message=f"IC IR = {report.ic_ir:.2f} < 0.3，信號穩定性不足",
                metric="ic_ir",
                current_value=report.ic_ir,
                threshold=0.3,
            ))
        
        # Alert 4: 全局 IC 不顯著
        if report.overall_ic_pvalue > 0.05:
            alerts.append(AlphaDecayAlert(
                severity="warning",
                message=f"全局 IC ({report.overall_ic:.4f}) 不顯著 (p={report.overall_ic_pvalue:.4f})",
                metric="overall_ic_pvalue",
                current_value=report.overall_ic_pvalue,
                threshold=0.05,
            ))
        
        # 排序：critical → warning → info
        severity_order = {"critical": 0, "warning": 1, "info": 2}
        alerts.sort(key=lambda a: severity_order.get(a.severity, 3))
        
        return alerts
    
    def _compute_rolling_ic(
        self,
        signals: pd.Series,
        forward_returns: pd.Series,
    ) -> pd.Series:
        """計算滾動 IC（Spearman rank correlation）"""
        # 合併數據
        combined = pd.DataFrame({
            "signal": signals,
            "fwd_ret": forward_returns,
        }).dropna()
        
        # 只取有信號的 bar
        combined = combined[combined["signal"] != 0]
        
        if len(combined) < self.window:
            # 數據不足一個窗口，用全部數據
            ic_val, _ = stats.spearmanr(combined["signal"], combined["fwd_ret"])
            return pd.Series(ic_val, index=combined.index)
        
        # 滾動計算（每天計算一次）
        step = self.bars_per_day
        ic_values = {}
        
        for end_idx in range(self.window, len(combined), step):
            start_idx = end_idx - self.window
            window_data = combined.iloc[start_idx:end_idx]
            
            if len(window_data) >= self.min_observations:
                ic, _ = stats.spearmanr(window_data["signal"], window_data["fwd_ret"])
                ic_values[combined.index[end_idx - 1]] = ic
        
        return pd.Series(ic_values, name="rolling_ic")
    
    def _compute_yearly_ic(
        self,
        signals: pd.Series,
        forward_returns: pd.Series,
    ) -> dict:
        """按年計算 IC"""
        combined = pd.DataFrame({
            "signal": signals,
            "fwd_ret": forward_returns,
        }).dropna()
        combined = combined[combined["signal"] != 0]
        
        yearly = {}
        for year, group in combined.groupby(combined.index.year):
            if len(group) >= self.min_observations:
                ic, _ = stats.spearmanr(group["signal"], group["fwd_ret"])
                yearly[int(year)] = round(ic, 4)
        
        return yearly
    
    def _empty_report(self) -> ICReport:
        """返回空報告"""
        return ICReport(
            overall_ic=0.0,
            overall_ic_pvalue=1.0,
            rolling_ic=pd.Series(dtype=float),
            avg_ic=0.0,
            ic_std=0.0,
            ic_ir=0.0,
            recent_ic=0.0,
            historical_ic=0.0,
            ic_decay_pct=0.0,
            is_decaying=False,
            yearly_ic={},
            signal_count=0,
            active_signal_pct=0.0,
        )


# ══════════════════════════════════════════════════════════════
# 便利函數
# ══════════════════════════════════════════════════════════════

def compute_strategy_ic(
    df: pd.DataFrame,
    strategy_func,
    ctx,
    params: dict,
    forward_bars: int = 24,
    window: int = 180 * 24,
    interval: str = "1h",
) -> ICReport:
    """
    一鍵計算策略的 IC 報告
    
    Args:
        df: K 線 DataFrame
        strategy_func: 策略函數
        ctx: StrategyContext
        params: 策略參數
        forward_bars: 前瞻期
        window: 滾動窗口
        interval: 時間間隔
        
    Returns:
        ICReport
    """
    signals = strategy_func(df, ctx, params)
    
    monitor = RollingICMonitor(
        window=window,
        forward_bars=forward_bars,
        interval=interval,
    )
    
    return monitor.compute(signals, df["close"])
