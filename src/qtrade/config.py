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


def load_config(path: str = "config/base.yaml") -> AppConfig:
    load_dotenv()
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    data_dir = Path(os.getenv("DATA_DIR", "./data")).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    return AppConfig(
        market=MarketConfig(**raw["market"]),
        backtest=BacktestConfig(**raw["backtest"]),
        strategy=StrategyConfig(**raw["strategy"]),
        output=OutputConfig(**raw["output"]),
        data_dir=data_dir,
    )
