from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import os
import yaml
from dotenv import load_dotenv


class MarketType(str, Enum):
    """市場類型"""
    SPOT = "spot"
    FUTURES = "futures"


@dataclass(frozen=True)
class MarketConfig:
    symbols: list[str]
    interval: str
    start: str
    end: str | None
    market_type: MarketType = MarketType.SPOT  # 新增：預設現貨


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

    allocation: { "BTCUSDT": 0.3, "ETHUSDT": 0.3 } 或 null
        - 值為該幣種可用的最大權益比例
        - 總和應 <= 1.0（剩餘為現金儲備）
        - 設為 null 或不設定 → 自動等權分配（考慮 cash_reserve）
    
    cash_reserve: 現金保留比例 [0, 1]，預設 0.2 (20%)
        - 自動分配時：每幣權重 = (1 - cash_reserve) / n_symbols
        - 例如 4 幣 + 20% 現金 → 每幣 20%
    """
    allocation: dict[str, float] | None = None
    cash_reserve: float = 0.2  # 預設保留 20% 現金

    def get_weight(self, symbol: str, n_symbols: int = 1) -> float:
        """
        取得某幣種的權重 [0, 1]

        優先級：
        1. 有明確 allocation 且包含該幣種 → 用設定值
        2. 否則 → 自動等權分配，考慮 cash_reserve
        """
        if self.allocation and symbol in self.allocation:
            return float(self.allocation[symbol])
        # 自動等權分配（扣除現金保留）
        available = 1.0 - self.cash_reserve
        return available / max(n_symbols, 1)


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
class FuturesConfig:
    """
    合約專屬配置
    
    leverage: 槓桿倍數 [1, 125]
        - 建議新手用 1-3 倍
        - 高波動幣種建議低槓桿
    
    margin_type: 保證金模式
        - "ISOLATED": 逐倉（推薦，風險隔離）
        - "CROSSED": 全倉（共用保證金）
    
    position_mode: 持倉模式
        - "ONE_WAY": 單向持倉（預設，同時只能多或空）
        - "HEDGE": 雙向持倉（可同時持有多空倉位）
    
    direction: 交易方向
        - "both": 多空都做（預設）
        - "long_only": 只做多（合約但不做空）
        - "short_only": 只做空
    """
    leverage: int = 1
    margin_type: str = "ISOLATED"
    position_mode: str = "ONE_WAY"
    direction: str = "both"  # "both", "long_only", "short_only"


@dataclass(frozen=True)
class NotificationConfig:
    """
    通知配置（支援 Spot/Futures 分開通知）
    
    telegram_bot_token: Telegram Bot Token
        - 設定後覆蓋環境變數 TELEGRAM_BOT_TOKEN
        - 可用 ${ENV_VAR} 語法引用環境變數
    
    telegram_chat_id: Telegram Chat ID
        - 設定後覆蓋環境變數 TELEGRAM_CHAT_ID
    
    prefix: 訊息前綴
        - 例如 "🟢 [SPOT]" 或 "🔴 [FUTURES]"
        - 方便在同一個 Chat 區分不同策略
    
    enabled: 是否啟用通知
    """
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None
    prefix: str = ""
    enabled: bool = True


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
    futures: FuturesConfig | None = None  # 合約配置（僅 market_type=futures 時使用）
    notification: NotificationConfig | None = None  # 通知配置

    @property
    def is_futures(self) -> bool:
        """是否為合約模式"""
        return self.market.market_type == MarketType.FUTURES

    @property
    def supports_short(self) -> bool:
        """是否支援做空（合約模式才支援）"""
        return self.is_futures


def _resolve_env_var(value: str | None) -> str | None:
    """
    解析環境變數語法 ${VAR_NAME}
    
    例如：${SPOT_TELEGRAM_BOT_TOKEN} → 實際值
    """
    if not value or not isinstance(value, str):
        return value
    if value.startswith("${") and value.endswith("}"):
        env_name = value[2:-1]
        return os.getenv(env_name)
    return value


def load_config(path: str = "config/base.yaml") -> AppConfig:
    load_dotenv()
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    data_dir = Path(os.getenv("DATA_DIR", "./data")).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    # market 配置
    market_raw = dict(raw["market"])
    market_type_str = market_raw.pop("market_type", "spot")
    market_type = MarketType(market_type_str)
    market = MarketConfig(
        **market_raw,
        market_type=market_type,
    )

    # portfolio 可選
    portfolio_raw = raw.get("portfolio", {})
    portfolio = PortfolioConfig(
        allocation=portfolio_raw.get("allocation"),
        cash_reserve=portfolio_raw.get("cash_reserve", 0.2),
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

    # futures 可選（僅合約模式使用）
    futures: FuturesConfig | None = None
    if market_type == MarketType.FUTURES:
        futures_raw = raw.get("futures", {})
        futures = FuturesConfig(
            leverage=futures_raw.get("leverage", 1),
            margin_type=futures_raw.get("margin_type", "ISOLATED"),
            position_mode=futures_raw.get("position_mode", "ONE_WAY"),
            direction=futures_raw.get("direction", "both"),
        )

    # notification 可選
    notification: NotificationConfig | None = None
    notif_raw = raw.get("notification")
    if notif_raw:
        notification = NotificationConfig(
            telegram_bot_token=_resolve_env_var(notif_raw.get("telegram_bot_token")),
            telegram_chat_id=_resolve_env_var(notif_raw.get("telegram_chat_id")),
            prefix=notif_raw.get("prefix", ""),
            enabled=notif_raw.get("enabled", True),
        )

    return AppConfig(
        market=market,
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
        futures=futures,
        notification=notification,
    )
