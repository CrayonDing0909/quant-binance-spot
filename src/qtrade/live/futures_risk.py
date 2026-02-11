"""
Futures Risk Management — 合約風險管理模組

功能：
    1. 強平價格計算與預警
    2. 資金費率追蹤與統計
    3. 保證金率監控
    4. 風險指標計算

使用方式：
    risk_manager = FuturesRiskManager(broker)
    
    # 計算強平價格
    liq_price = risk_manager.calculate_liquidation_price("BTCUSDT")
    
    # 檢查風險等級
    risk_level = risk_manager.check_position_risk("BTCUSDT")
    
    # 獲取資金費率
    funding = risk_manager.get_funding_rate_info("BTCUSDT")
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from ..utils.log import get_logger
from ..data.binance_futures_client import BinanceFuturesHTTP

if TYPE_CHECKING:
    from .binance_futures_broker import BinanceFuturesBroker, FuturesPosition

logger = get_logger("futures_risk")


@dataclass
class LiquidationInfo:
    """強平資訊"""
    symbol: str
    position_side: str      # LONG / SHORT
    entry_price: float      # 開倉價格
    mark_price: float       # 標記價格
    liquidation_price: float  # 強平價格
    distance_pct: float     # 距離強平的百分比
    margin_ratio: float     # 保證金率
    leverage: int
    is_safe: bool           # 是否安全（距離 > 10%）


@dataclass
class FundingRateInfo:
    """資金費率資訊"""
    symbol: str
    current_rate: float     # 當前費率
    predicted_rate: float   # 預測費率
    next_funding_time: datetime  # 下次結算時間
    rate_8h_avg: float      # 8小時平均
    rate_24h_avg: float     # 24小時平均
    annualized_rate: float  # 年化費率
    position_impact: float  # 對當前持倉的影響（預估）


@dataclass
class RiskLevel:
    """風險等級"""
    level: str              # LOW / MEDIUM / HIGH / CRITICAL
    margin_ratio: float     # 保證金率
    liquidation_distance: float  # 距離強平百分比
    funding_exposure: float  # 資金費率曝險
    warnings: list[str]     # 警告訊息


class FuturesRiskManager:
    """
    合約風險管理器
    
    提供持倉風險監控、強平預警、資金費率追蹤等功能。
    """
    
    # 風險閾值
    MARGIN_RATIO_WARNING = 0.5      # 50% 保證金率警告
    MARGIN_RATIO_DANGER = 0.7       # 70% 保證金率危險
    MARGIN_RATIO_CRITICAL = 0.9     # 90% 保證金率緊急
    
    LIQUIDATION_WARNING_PCT = 0.15  # 距離強平 15% 警告
    LIQUIDATION_DANGER_PCT = 0.08   # 距離強平 8% 危險
    LIQUIDATION_CRITICAL_PCT = 0.03 # 距離強平 3% 緊急
    
    FUNDING_RATE_HIGH = 0.001       # 0.1% 高費率警告
    
    def __init__(self, broker: BinanceFuturesBroker | None = None):
        """
        Args:
            broker: BinanceFuturesBroker 實例（可選，用於獲取持倉資訊）
        """
        self.broker = broker
        self.http = broker.http if broker else BinanceFuturesHTTP()
        self._funding_cache: dict[str, dict] = {}
    
    # ── 強平價格計算 ────────────────────────────────────────
    
    def calculate_liquidation_price(
        self,
        symbol: str,
        position: FuturesPosition | None = None,
    ) -> LiquidationInfo | None:
        """
        計算強平價格
        
        Binance USDT-M 永續合約強平公式（逐倉模式）：
        
        多倉強平價格 = 開倉價格 × (1 - 1/槓桿 + 維持保證金率)
        空倉強平價格 = 開倉價格 × (1 + 1/槓桿 - 維持保證金率)
        
        實際上 Binance 的強平計算更複雜，會考慮：
        - 維持保證金率（根據持倉量分層）
        - 未實現盈虧
        - 其他費用
        
        這裡提供簡化版本的估算。
        
        Args:
            symbol: 交易對
            position: 持倉資訊（None 時從 broker 獲取）
            
        Returns:
            LiquidationInfo 或 None
        """
        if position is None and self.broker:
            position = self.broker.get_position(symbol)
        
        if position is None or not position.is_open:
            return None
        
        try:
            # 獲取標記價格
            data = self.http.get_mark_price(symbol)
            mark_price = float(data['markPrice'])
            
            # 獲取維持保證金率（簡化：使用固定值 0.4%）
            # 實際應該根據持倉量查詢分層維持保證金率
            maintenance_margin_rate = 0.004
            
            leverage = position.leverage
            entry_price = position.entry_price
            
            # 計算強平價格
            if position.qty > 0:  # 多倉
                # 多倉強平 = 開倉價 × (1 - 初始保證金率 + 維持保證金率)
                # 初始保證金率 = 1 / 槓桿
                liq_price = entry_price * (1 - 1/leverage + maintenance_margin_rate)
            else:  # 空倉
                # 空倉強平 = 開倉價 × (1 + 初始保證金率 - 維持保證金率)
                liq_price = entry_price * (1 + 1/leverage - maintenance_margin_rate)
            
            # 使用交易所返回的強平價格（如果有且合理）
            if position.liquidation_price > 0:
                liq_price = position.liquidation_price
            
            # 計算距離強平的百分比
            if position.qty > 0:  # 多倉
                distance_pct = (mark_price - liq_price) / mark_price
            else:  # 空倉
                distance_pct = (liq_price - mark_price) / mark_price
            
            # 計算保證金率
            # 保證金率 = (維持保證金 + 未實現盈虧) / (持倉價值 × 初始保證金率)
            position_value = abs(position.qty) * mark_price
            initial_margin = position_value / leverage
            margin_ratio = abs(position.unrealized_pnl) / initial_margin if initial_margin > 0 else 0
            
            is_safe = distance_pct > self.LIQUIDATION_WARNING_PCT
            
            return LiquidationInfo(
                symbol=symbol,
                position_side="LONG" if position.qty > 0 else "SHORT",
                entry_price=entry_price,
                mark_price=mark_price,
                liquidation_price=liq_price,
                distance_pct=distance_pct,
                margin_ratio=margin_ratio,
                leverage=leverage,
                is_safe=is_safe,
            )
            
        except Exception as e:
            logger.error(f"計算強平價格失敗 {symbol}: {e}")
            return None
    
    # ── 資金費率 ────────────────────────────────────────────
    
    def get_funding_rate_info(self, symbol: str) -> FundingRateInfo | None:
        """
        獲取資金費率資訊
        
        Args:
            symbol: 交易對
            
        Returns:
            FundingRateInfo 或 None
        """
        try:
            # 獲取當前資金費率
            data = self.http.get_mark_price(symbol)
            current_rate = float(data['lastFundingRate'])
            next_funding_time = datetime.fromtimestamp(
                int(data['nextFundingTime']) / 1000, 
                tz=timezone.utc
            )
            
            # 獲取歷史資金費率
            history = self.http.get_funding_rate(symbol, limit=100)
            
            # 計算平均值
            rates = [float(h['fundingRate']) for h in history]
            rate_8h_avg = sum(rates[:3]) / 3 if len(rates) >= 3 else current_rate
            rate_24h_avg = sum(rates[:8]) / 8 if len(rates) >= 8 else current_rate
            
            # 年化費率（每 8 小時收一次，一年 1095 次）
            annualized_rate = rate_24h_avg * 1095
            
            # 計算對當前持倉的影響
            position_impact = 0.0
            if self.broker:
                pos = self.broker.get_position(symbol)
                if pos and pos.is_open:
                    position_value = abs(pos.qty) * float(data['markPrice'])
                    # 多倉支付，空倉收取（當費率為正時）
                    sign = 1 if pos.qty > 0 else -1
                    position_impact = -sign * position_value * current_rate
            
            return FundingRateInfo(
                symbol=symbol,
                current_rate=current_rate,
                predicted_rate=current_rate,  # Binance 不提供預測費率
                next_funding_time=next_funding_time,
                rate_8h_avg=rate_8h_avg,
                rate_24h_avg=rate_24h_avg,
                annualized_rate=annualized_rate,
                position_impact=position_impact,
            )
            
        except Exception as e:
            logger.error(f"獲取資金費率失敗 {symbol}: {e}")
            return None
    
    def get_funding_rate_history(
        self,
        symbol: str,
        limit: int = 100,
    ) -> list[dict]:
        """
        獲取歷史資金費率
        
        Returns:
            列表，每個元素包含 fundingTime, fundingRate
        """
        try:
            return self.http.get_funding_rate(symbol, limit)
        except Exception as e:
            logger.error(f"獲取歷史資金費率失敗 {symbol}: {e}")
            return []
    
    # ── 風險等級評估 ────────────────────────────────────────
    
    def check_position_risk(self, symbol: str) -> RiskLevel | None:
        """
        檢查持倉風險等級
        
        Returns:
            RiskLevel 或 None（無持倉）
        """
        if not self.broker:
            logger.warning("未設置 broker，無法檢查持倉風險")
            return None
        
        pos = self.broker.get_position(symbol)
        if not pos or not pos.is_open:
            return None
        
        warnings = []
        
        # 1. 強平距離
        liq_info = self.calculate_liquidation_price(symbol, pos)
        liquidation_distance = liq_info.distance_pct if liq_info else 1.0
        
        if liquidation_distance < self.LIQUIDATION_CRITICAL_PCT:
            warnings.append(f"⚠️ 極度危險！距離強平僅 {liquidation_distance:.1%}")
        elif liquidation_distance < self.LIQUIDATION_DANGER_PCT:
            warnings.append(f"🔴 危險！距離強平 {liquidation_distance:.1%}")
        elif liquidation_distance < self.LIQUIDATION_WARNING_PCT:
            warnings.append(f"🟡 警告：距離強平 {liquidation_distance:.1%}")
        
        # 2. 保證金率
        margin_ratio = liq_info.margin_ratio if liq_info else 0
        
        if margin_ratio > self.MARGIN_RATIO_CRITICAL:
            warnings.append(f"⚠️ 保證金率過高 {margin_ratio:.1%}")
        elif margin_ratio > self.MARGIN_RATIO_DANGER:
            warnings.append(f"🔴 保證金率偏高 {margin_ratio:.1%}")
        elif margin_ratio > self.MARGIN_RATIO_WARNING:
            warnings.append(f"🟡 保證金率注意 {margin_ratio:.1%}")
        
        # 3. 資金費率
        funding_info = self.get_funding_rate_info(symbol)
        funding_exposure = funding_info.position_impact if funding_info else 0
        
        if funding_info and abs(funding_info.current_rate) > self.FUNDING_RATE_HIGH:
            if funding_info.current_rate > 0 and pos.qty > 0:
                warnings.append(f"🟡 高資金費率 {funding_info.current_rate*100:.3f}%，多倉需支付")
            elif funding_info.current_rate < 0 and pos.qty < 0:
                warnings.append(f"🟡 負資金費率 {funding_info.current_rate*100:.3f}%，空倉需支付")
        
        # 綜合評估風險等級
        if liquidation_distance < self.LIQUIDATION_CRITICAL_PCT or margin_ratio > self.MARGIN_RATIO_CRITICAL:
            level = "CRITICAL"
        elif liquidation_distance < self.LIQUIDATION_DANGER_PCT or margin_ratio > self.MARGIN_RATIO_DANGER:
            level = "HIGH"
        elif liquidation_distance < self.LIQUIDATION_WARNING_PCT or margin_ratio > self.MARGIN_RATIO_WARNING:
            level = "MEDIUM"
        else:
            level = "LOW"
        
        return RiskLevel(
            level=level,
            margin_ratio=margin_ratio,
            liquidation_distance=liquidation_distance,
            funding_exposure=funding_exposure,
            warnings=warnings,
        )
    
    def check_all_positions_risk(self) -> dict[str, RiskLevel]:
        """檢查所有持倉的風險"""
        if not self.broker:
            return {}
        
        results = {}
        for pos in self.broker.get_positions():
            risk = self.check_position_risk(pos.symbol)
            if risk:
                results[pos.symbol] = risk
        return results
    
    # ── 風險報告 ────────────────────────────────────────────
    
    def generate_risk_report(self) -> str:
        """
        生成風險報告
        
        Returns:
            格式化的風險報告字串
        """
        if not self.broker:
            return "⚠️ 未設置 broker，無法生成風險報告"
        
        lines = [
            "=" * 60,
            "  合約風險報告",
            "=" * 60,
        ]
        
        # 帳戶概覽
        try:
            account = self.broker.get_account_info()
            total_balance = float(account.get('totalWalletBalance', 0))
            available = float(account.get('availableBalance', 0))
            unrealized_pnl = float(account.get('totalUnrealizedProfit', 0))
            margin_balance = float(account.get('totalMarginBalance', 0))
            
            lines.extend([
                "",
                "📊 帳戶概覽",
                "-" * 40,
                f"  總餘額:       ${total_balance:,.2f}",
                f"  可用餘額:     ${available:,.2f}",
                f"  保證金餘額:   ${margin_balance:,.2f}",
                f"  未實現盈虧:   ${unrealized_pnl:+,.2f}",
            ])
        except Exception as e:
            lines.append(f"  ❌ 獲取帳戶資訊失敗: {e}")
        
        # 持倉風險
        positions = self.broker.get_positions()
        if positions:
            lines.extend([
                "",
                "📈 持倉風險",
                "-" * 40,
            ])
            
            for pos in positions:
                risk = self.check_position_risk(pos.symbol)
                liq = self.calculate_liquidation_price(pos.symbol, pos)
                funding = self.get_funding_rate_info(pos.symbol)
                
                side = "LONG" if pos.qty > 0 else "SHORT"
                emoji = "🟢" if risk and risk.level == "LOW" else (
                    "🟡" if risk and risk.level == "MEDIUM" else (
                        "🔴" if risk and risk.level == "HIGH" else "⚠️"
                    )
                )
                
                lines.append(f"\n  {emoji} {pos.symbol} [{side}]")
                lines.append(f"     數量: {abs(pos.qty):.6f}")
                lines.append(f"     開倉價: ${pos.entry_price:,.2f}")
                lines.append(f"     槓桿: {pos.leverage}x")
                
                if liq:
                    lines.append(f"     標記價: ${liq.mark_price:,.2f}")
                    lines.append(f"     強平價: ${liq.liquidation_price:,.2f}")
                    lines.append(f"     距強平: {liq.distance_pct:.1%}")
                
                if funding:
                    lines.append(f"     資金費率: {funding.current_rate*100:.4f}%")
                    if funding.position_impact != 0:
                        lines.append(f"     預估費用: ${funding.position_impact:+.2f}")
                
                lines.append(f"     未實現: ${pos.unrealized_pnl:+,.2f}")
                
                if risk and risk.warnings:
                    for w in risk.warnings:
                        lines.append(f"     {w}")
        else:
            lines.extend([
                "",
                "📈 持倉風險",
                "-" * 40,
                "  無持倉",
            ])
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def print_risk_report(self) -> None:
        """打印風險報告"""
        print(self.generate_risk_report())
