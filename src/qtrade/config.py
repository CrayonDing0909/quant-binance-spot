from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os
import yaml
from dotenv import load_dotenv


@dataclass(frozen=True)
class MarketConfig:
    symbols: list[str]
    interval: str
    start: str
    end: str | None


@dataclass(frozen=True)
class BacktestConfig:
    initial_cash: float
    fee_bps: float
    slippage_bps: float
    trade_on: str  # "next_open"
    validate_data: bool = True
    clean_data: bool = True


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    params: dict
    symbol_overrides: dict | None = None

    def get_params(self, symbol: str | None = None) -> dict:
        """返回合併後的參數：base params + symbol overrides"""
        merged = dict(self.params)
        if symbol and self.symbol_overrides and symbol in self.symbol_overrides:
            merged.update(self.symbol_overrides[symbol])
        return merged


@dataclass(frozen=True)
class PortfolioConfig:
    """
    多幣種倉位分配

    allocation: { "BTCUSDT": 0.6, "ETHUSDT": 0.4 }
        - 值為該幣種可用的最大權益比例
        - 總和應 <= 1.0（剩餘為現金儲備）
        - 未指定時，按 symbols 數量等權分配
    """
    allocation: dict[str, float] | None = None

    def get_weight(self, symbol: str, n_symbols: int = 1) -> float:
        """
        取得某幣種的權重 [0, 1]

        如果有明確配置就用配置值，否則等權分配。
        """
        if self.allocation and symbol in self.allocation:
            return float(self.allocation[symbol])
        # 等權分配
        return 1.0 / max(n_symbols, 1)


@dataclass(frozen=True)
class RiskConfig:
    """
    風險管理配置

    max_drawdown_pct: 最大回撤比例 [0, 1]，超過則觸發熔斷
        - 0.20 = 虧 20% 後停止交易（建議 Paper Trading 用 0.20）
        - 0.10 = 虧 10% 後停止交易（建議 Real Trading 用 0.10~0.15）
        - None / 0 = 不啟用熔斷
    """
    max_drawdown_pct: float | None = 0.20


@dataclass(frozen=True)
class PositionSizingConfig:
    """
    倉位計算配置
    
    method: 倉位計算方法
        - "fixed": 固定倉位比例（預設）
        - "kelly": 根據 Kelly 公式動態調整
        - "volatility": 根據波動率調整
    
    position_pct: 固定倉位比例 [0, 1]（method="fixed" 時使用）
    
    kelly_fraction: Kelly 比例因子 [0, 1]
        - 1.0 = Full Kelly（風險高）
        - 0.5 = Half Kelly（推薦）
        - 0.25 = Quarter Kelly（保守）
    
    win_rate, avg_win, avg_loss: Kelly 參數
        - None = 從歷史交易自動計算
        
    target_volatility: 目標年化波動率（method="volatility" 時使用）
    vol_lookback: 波動率計算回看期
    
    min_trades_for_kelly: 使用 Kelly 前需要的最小交易數量
        - 交易數不足時自動回退到固定倉位
    """
    method: str = "fixed"  # "fixed", "kelly", "volatility"
    
    # Fixed 參數
    position_pct: float = 1.0
    
    # Kelly 參數
    kelly_fraction: float = 0.25  # 預設 Quarter Kelly（保守）
    win_rate: float | None = None
    avg_win: float | None = None
    avg_loss: float | None = None
    min_trades_for_kelly: int = 20  # 至少 20 筆交易才啟用 Kelly
    
    # Volatility 參數
    target_volatility: float = 0.15
    vol_lookback: int = 20


@dataclass(frozen=True)
class OutputConfig:
    report_dir: str


@dataclass(frozen=True)
class AppConfig:
    market: MarketConfig
    backtest: BacktestConfig
    strategy: StrategyConfig
    output: OutputConfig
    data_dir: Path
    portfolio: PortfolioConfig = PortfolioConfig()
    risk: RiskConfig = RiskConfig()
    position_sizing: PositionSizingConfig = PositionSizingConfig()


def load_config(path: str = "config/base.yaml") -> AppConfig:
    load_dotenv()
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    data_dir = Path(os.getenv("DATA_DIR", "./data")).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    # portfolio 可選
    portfolio_raw = raw.get("portfolio", {})
    portfolio = PortfolioConfig(
        allocation=portfolio_raw.get("allocation"),
    )

    # risk 可選
    risk_raw = raw.get("risk", {})
    risk = RiskConfig(
        max_drawdown_pct=risk_raw.get("max_drawdown_pct", 0.20),
    )

    # position_sizing 可選
    ps_raw = raw.get("position_sizing", {})
    position_sizing = PositionSizingConfig(
        method=ps_raw.get("method", "fixed"),
        position_pct=ps_raw.get("position_pct", 1.0),
        kelly_fraction=ps_raw.get("kelly_fraction", 0.25),
        win_rate=ps_raw.get("win_rate"),
        avg_win=ps_raw.get("avg_win"),
        avg_loss=ps_raw.get("avg_loss"),
        min_trades_for_kelly=ps_raw.get("min_trades_for_kelly", 20),
        target_volatility=ps_raw.get("target_volatility", 0.15),
        vol_lookback=ps_raw.get("vol_lookback", 20),
    )

    return AppConfig(
        market=MarketConfig(**raw["market"]),
        backtest=BacktestConfig(**raw["backtest"]),
        strategy=StrategyConfig(
            name=raw["strategy"]["name"],
            params=raw["strategy"].get("params", {}),
            symbol_overrides=raw["strategy"].get("symbol_overrides"),
        ),
        output=OutputConfig(**raw["output"]),
        data_dir=data_dir,
        portfolio=portfolio,
        risk=risk,
        position_sizing=position_sizing,
    )
