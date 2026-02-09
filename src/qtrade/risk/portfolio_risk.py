"""
組合風險管理模組

提供組合級別的風險管理功能：
- 組合 VaR 計算
- 相關性分析
- 組合風險控制
"""
from __future__ import annotations
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class PortfolioRiskManager:
    """組合風險管理器"""
    
    max_portfolio_var: float = 0.05  # 最大組合 VaR（5%）
    max_correlation: float = 0.8  # 最大允許相關性
    diversification_threshold: float = 0.3  # 分散化閾值
    
    def calculate_portfolio_risk(
        self,
        returns: Dict[str, pd.Series],
        weights: Dict[str, float],
        confidence_level: float = 0.95
    ) -> dict:
        """
        計算組合風險指標
        
        Args:
            returns: 各資產的收益率序列
            weights: 各資產的權重
            confidence_level: 置信水平，預設 95%
        
        Returns:
            包含各種風險指標的字典
        """
        # 對齊時間序列
        aligned_returns = self._align_returns(returns)
        
        if len(aligned_returns) == 0:
            return {
                "portfolio_var": 0.0,
                "portfolio_volatility": 0.0,
                "max_correlation": 0.0,
                "diversification_ratio": 0.0,
            }
        
        # 計算組合收益率
        portfolio_returns = self._calculate_portfolio_returns(aligned_returns, weights)
        
        # 計算組合 VaR
        portfolio_var = self._calculate_var(portfolio_returns, confidence_level)
        
        # 計算組合波動率
        portfolio_vol = portfolio_returns.std() * np.sqrt(252)  # 年化
        
        # 計算相關性矩陣
        corr_matrix = aligned_returns.corr()
        max_corr = self._get_max_correlation(corr_matrix)
        
        # 計算分散化比率
        div_ratio = self._calculate_diversification_ratio(aligned_returns, weights)
        
        return {
            "portfolio_var": portfolio_var,
            "portfolio_volatility": portfolio_vol,
            "max_correlation": max_corr,
            "diversification_ratio": div_ratio,
            "correlation_matrix": corr_matrix,
        }
    
    def _align_returns(self, returns: Dict[str, pd.Series]) -> pd.DataFrame:
        """對齊收益率時間序列"""
        df = pd.DataFrame(returns)
        return df.dropna()
    
    def _calculate_portfolio_returns(
        self,
        returns: pd.DataFrame,
        weights: Dict[str, float]
    ) -> pd.Series:
        """計算組合收益率"""
        # 確保權重歸一化
        total_weight = sum(weights.values())
        if total_weight == 0:
            return pd.Series(0.0, index=returns.index)
        
        normalized_weights = {k: v / total_weight for k, v in weights.items()}
        
        # 計算加權組合收益率
        portfolio_returns = pd.Series(0.0, index=returns.index)
        for symbol, weight in normalized_weights.items():
            if symbol in returns.columns:
                portfolio_returns += returns[symbol] * weight
        
        return portfolio_returns
    
    def _calculate_var(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95
    ) -> float:
        """計算 VaR（風險價值）"""
        if len(returns) == 0:
            return 0.0
        
        # 使用歷史模擬法
        var_percentile = (1 - confidence_level) * 100
        var = abs(np.percentile(returns, var_percentile))
        
        return var
    
    def _get_max_correlation(self, corr_matrix: pd.DataFrame) -> float:
        """獲取最大相關性（排除對角線）"""
        if len(corr_matrix) == 0:
            return 0.0
        
        # 排除對角線元素
        mask = ~np.eye(len(corr_matrix), dtype=bool)
        max_corr = corr_matrix.values[mask].max()
        
        return max_corr
    
    def _calculate_diversification_ratio(
        self,
        returns: pd.DataFrame,
        weights: Dict[str, float]
    ) -> float:
        """
        計算分散化比率
        
        分散化比率 = 加權平均波動率 / 組合波動率
        比率 > 1 表示有分散化效果
        """
        if len(returns) == 0:
            return 0.0
        
        # 計算各資產波動率
        asset_vols = returns.std() * np.sqrt(252)  # 年化
        
        # 計算加權平均波動率
        total_weight = sum(weights.values())
        if total_weight == 0:
            return 0.0
        
        weighted_avg_vol = 0.0
        for symbol, weight in weights.items():
            if symbol in asset_vols.index:
                weighted_avg_vol += asset_vols[symbol] * (weight / total_weight)
        
        # 計算組合波動率
        portfolio_returns = self._calculate_portfolio_returns(returns, weights)
        portfolio_vol = portfolio_returns.std() * np.sqrt(252)
        
        if portfolio_vol == 0:
            return 0.0
        
        return weighted_avg_vol / portfolio_vol
    
    def check_risk_limits(
        self,
        returns: Dict[str, pd.Series],
        weights: Dict[str, float],
        confidence_level: float = 0.95
    ) -> tuple[bool, dict]:
        """
        檢查組合風險是否在限制內
        
        Returns:
            (是否通過, 風險指標字典)
        """
        risk_metrics = self.calculate_portfolio_risk(returns, weights, confidence_level)
        
        passed = True
        issues = []
        
        # 檢查 VaR
        if risk_metrics["portfolio_var"] > self.max_portfolio_var:
            passed = False
            issues.append(f"組合 VaR ({risk_metrics['portfolio_var']:.4f}) 超過限制 ({self.max_portfolio_var:.4f})")
        
        # 檢查相關性
        if risk_metrics["max_correlation"] > self.max_correlation:
            passed = False
            issues.append(f"最大相關性 ({risk_metrics['max_correlation']:.2f}) 超過限制 ({self.max_correlation:.2f})")
        
        # 檢查分散化
        if risk_metrics["diversification_ratio"] < self.diversification_threshold:
            passed = False
            issues.append(f"分散化比率 ({risk_metrics['diversification_ratio']:.2f}) 低於閾值 ({self.diversification_threshold:.2f})")
        
        risk_metrics["passed"] = passed
        risk_metrics["issues"] = issues
        
        return passed, risk_metrics


def calculate_portfolio_var(
    returns: Dict[str, pd.Series],
    weights: Dict[str, float],
    confidence_level: float = 0.95
) -> float:
    """
    計算組合 VaR
    
    Args:
        returns: 各資產的收益率序列
        weights: 各資產的權重
        confidence_level: 置信水平
    
    Returns:
        組合 VaR
    """
    manager = PortfolioRiskManager()
    risk_metrics = manager.calculate_portfolio_risk(returns, weights, confidence_level)
    return risk_metrics["portfolio_var"]


def calculate_correlation_matrix(returns: Dict[str, pd.Series]) -> pd.DataFrame:
    """
    計算資產間的相關性矩陣
    
    Args:
        returns: 各資產的收益率序列
    
    Returns:
        相關性矩陣 DataFrame
    """
    df = pd.DataFrame(returns)
    df = df.dropna()
    return df.corr()
