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
        """返回合并后的参数：base params + symbol overrides"""
        merged = dict(self.params)
        if symbol and self.symbol_overrides and symbol in self.symbol_overrides:
            merged.update(self.symbol_overrides[symbol])
        return merged


@dataclass(frozen=True)
class PortfolioConfig:
    """
    多币种仓位分配

    allocation: { "BTCUSDT": 0.6, "ETHUSDT": 0.4 }
        - 值为该币种可用的最大权益比例
        - 总和应 <= 1.0（剩余为现金储备）
        - 未指定时，按 symbols 数量等权分配
    """
    allocation: dict[str, float] | None = None

    def get_weight(self, symbol: str, n_symbols: int = 1) -> float:
        """
        取得某币种的权重 [0, 1]

        如果有明确配置就用配置值，否则等权分配。
        """
        if self.allocation and symbol in self.allocation:
            return float(self.allocation[symbol])
        # 等权分配
        return 1.0 / max(n_symbols, 1)


@dataclass(frozen=True)
class RiskConfig:
    """
    风险管理配置

    max_drawdown_pct: 最大回撤比例 [0, 1]，超过则触发熔断
        - 0.20 = 亏 20% 后停止交易（建议 Paper Trading 用 0.20）
        - 0.10 = 亏 10% 后停止交易（建议 Real Trading 用 0.10~0.15）
        - None / 0 = 不启用熔断
    """
    max_drawdown_pct: float | None = 0.20


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


def load_config(path: str = "config/base.yaml") -> AppConfig:
    load_dotenv()
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    data_dir = Path(os.getenv("DATA_DIR", "./data")).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    # portfolio 可选
    portfolio_raw = raw.get("portfolio", {})
    portfolio = PortfolioConfig(
        allocation=portfolio_raw.get("allocation"),
    )

    # risk 可选
    risk_raw = raw.get("risk", {})
    risk = RiskConfig(
        max_drawdown_pct=risk_raw.get("max_drawdown_pct", 0.20),
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
    )
