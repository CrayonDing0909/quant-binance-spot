"""
组合风险管理模块

提供组合级别的风险管理功能：
- 组合 VaR 计算
- 相关性分析
- 组合风险控制
"""
from __future__ import annotations
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class PortfolioRiskManager:
    """组合风险管理器"""
    
    max_portfolio_var: float = 0.05  # 最大组合 VaR（5%）
    max_correlation: float = 0.8  # 最大允许相关性
    diversification_threshold: float = 0.3  # 分散化阈值
    
    def calculate_portfolio_risk(
        self,
        returns: Dict[str, pd.Series],
        weights: Dict[str, float],
        confidence_level: float = 0.95
    ) -> dict:
        """
        计算组合风险指标
        
        Args:
            returns: 各资产的收益率序列
            weights: 各资产的权重
            confidence_level: 置信水平，默认 95%
        
        Returns:
            包含各种风险指标的字典
        """
        # 对齐时间序列
        aligned_returns = self._align_returns(returns)
        
        if len(aligned_returns) == 0:
            return {
                "portfolio_var": 0.0,
                "portfolio_volatility": 0.0,
                "max_correlation": 0.0,
                "diversification_ratio": 0.0,
            }
        
        # 计算组合收益率
        portfolio_returns = self._calculate_portfolio_returns(aligned_returns, weights)
        
        # 计算组合 VaR
        portfolio_var = self._calculate_var(portfolio_returns, confidence_level)
        
        # 计算组合波动率
        portfolio_vol = portfolio_returns.std() * np.sqrt(252)  # 年化
        
        # 计算相关性矩阵
        corr_matrix = aligned_returns.corr()
        max_corr = self._get_max_correlation(corr_matrix)
        
        # 计算分散化比率
        div_ratio = self._calculate_diversification_ratio(aligned_returns, weights)
        
        return {
            "portfolio_var": portfolio_var,
            "portfolio_volatility": portfolio_vol,
            "max_correlation": max_corr,
            "diversification_ratio": div_ratio,
            "correlation_matrix": corr_matrix,
        }
    
    def _align_returns(self, returns: Dict[str, pd.Series]) -> pd.DataFrame:
        """对齐收益率时间序列"""
        df = pd.DataFrame(returns)
        return df.dropna()
    
    def _calculate_portfolio_returns(
        self,
        returns: pd.DataFrame,
        weights: Dict[str, float]
    ) -> pd.Series:
        """计算组合收益率"""
        # 确保权重归一化
        total_weight = sum(weights.values())
        if total_weight == 0:
            return pd.Series(0.0, index=returns.index)
        
        normalized_weights = {k: v / total_weight for k, v in weights.items()}
        
        # 计算加权组合收益率
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
        """计算 VaR（风险价值）"""
        if len(returns) == 0:
            return 0.0
        
        # 使用历史模拟法
        var_percentile = (1 - confidence_level) * 100
        var = abs(np.percentile(returns, var_percentile))
        
        return var
    
    def _get_max_correlation(self, corr_matrix: pd.DataFrame) -> float:
        """获取最大相关性（排除对角线）"""
        if len(corr_matrix) == 0:
            return 0.0
        
        # 排除对角线元素
        mask = ~np.eye(len(corr_matrix), dtype=bool)
        max_corr = corr_matrix.values[mask].max()
        
        return max_corr
    
    def _calculate_diversification_ratio(
        self,
        returns: pd.DataFrame,
        weights: Dict[str, float]
    ) -> float:
        """
        计算分散化比率
        
        分散化比率 = 加权平均波动率 / 组合波动率
        比率 > 1 表示有分散化效果
        """
        if len(returns) == 0:
            return 0.0
        
        # 计算各资产波动率
        asset_vols = returns.std() * np.sqrt(252)  # 年化
        
        # 计算加权平均波动率
        total_weight = sum(weights.values())
        if total_weight == 0:
            return 0.0
        
        weighted_avg_vol = 0.0
        for symbol, weight in weights.items():
            if symbol in asset_vols.index:
                weighted_avg_vol += asset_vols[symbol] * (weight / total_weight)
        
        # 计算组合波动率
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
        检查组合风险是否在限制内
        
        Returns:
            (是否通过, 风险指标字典)
        """
        risk_metrics = self.calculate_portfolio_risk(returns, weights, confidence_level)
        
        passed = True
        issues = []
        
        # 检查 VaR
        if risk_metrics["portfolio_var"] > self.max_portfolio_var:
            passed = False
            issues.append(f"组合 VaR ({risk_metrics['portfolio_var']:.4f}) 超过限制 ({self.max_portfolio_var:.4f})")
        
        # 检查相关性
        if risk_metrics["max_correlation"] > self.max_correlation:
            passed = False
            issues.append(f"最大相关性 ({risk_metrics['max_correlation']:.2f}) 超过限制 ({self.max_correlation:.2f})")
        
        # 检查分散化
        if risk_metrics["diversification_ratio"] < self.diversification_threshold:
            passed = False
            issues.append(f"分散化比率 ({risk_metrics['diversification_ratio']:.2f}) 低于阈值 ({self.diversification_threshold:.2f})")
        
        risk_metrics["passed"] = passed
        risk_metrics["issues"] = issues
        
        return passed, risk_metrics


def calculate_portfolio_var(
    returns: Dict[str, pd.Series],
    weights: Dict[str, float],
    confidence_level: float = 0.95
) -> float:
    """
    计算组合 VaR
    
    Args:
        returns: 各资产的收益率序列
        weights: 各资产的权重
        confidence_level: 置信水平
    
    Returns:
        组合 VaR
    """
    manager = PortfolioRiskManager()
    risk_metrics = manager.calculate_portfolio_risk(returns, weights, confidence_level)
    return risk_metrics["portfolio_var"]


def calculate_correlation_matrix(returns: Dict[str, pd.Series]) -> pd.DataFrame:
    """
    计算资产间的相关性矩阵
    
    Args:
        returns: 各资产的收益率序列
    
    Returns:
        相关性矩阵 DataFrame
    """
    df = pd.DataFrame(returns)
    df = df.dropna()
    return df.corr()

