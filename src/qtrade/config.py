from __future__ import annotations
import copy
from dataclasses import dataclass, field
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
    interval: str                               # 主要執行 interval
    start: str
    end: str | None
    market_type: MarketType = MarketType.SPOT   # 預設現貨
    auxiliary_intervals: list[str] = field(default_factory=list)  # 輔助 TF（信號生成用）


@dataclass(frozen=True)
class FundingRateModelConfig:
    """
    回測用 Funding Rate 成本模型

    enabled: 是否啟用 funding rate 扣除
    default_rate_8h: 無歷史資料時的預設 8h 費率（0.0001 = 0.01%）
    use_historical: 是否嘗試載入歷史資料（需先用 download_data.py --funding-rate）
    """
    enabled: bool = False
    default_rate_8h: float = 0.0001  # 0.01% per 8h (Binance 預設)
    use_historical: bool = True


@dataclass(frozen=True)
class SlippageModelConfig:
    """
    Volume-based 滑點模型（Square-Root Market Impact）

    slippage = base_spread + k × (trade_value / ADV)^power

    enabled: 啟用後取代固定 slippage_bps
    base_bps: 最低買賣價差 (bps)
    impact_coefficient: 衝擊係數 k（經驗值 0.05~0.20）
    impact_power: 衝擊指數（0.5 = 平方根模型）
    adv_lookback: 計算 ADV 的回看 bar 數
    participation_rate: 最大市場佔比上限（clip 用）
    """
    enabled: bool = False
    base_bps: float = 2.0
    impact_coefficient: float = 0.1
    impact_power: float = 0.5
    adv_lookback: int = 20
    participation_rate: float = 0.10


@dataclass(frozen=True)
class BacktestConfig:
    initial_cash: float
    fee_bps: float
    slippage_bps: float
    trade_on: str  # "next_open"
    validate_data: bool = True
    clean_data: bool = True
    funding_rate: FundingRateModelConfig = FundingRateModelConfig()
    slippage_model: SlippageModelConfig = SlippageModelConfig()


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    params: dict
    symbol_overrides: dict | None = None
    display_name: str | None = None

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
    report_dir: str = "./reports"


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
class SymbolGovernanceThresholds:
    """Symbol Governance 狀態轉移閾值"""
    edge_sharpe_deweight: float = 0.3
    edge_sharpe_quarantine: float = 0.0
    edge_sharpe_recover_active: float = 0.8
    edge_sharpe_recover_from_quarantine: float = 0.5
    edge_per_turnover_min: float = 0.0
    slippage_ratio_deweight: float = 1.5
    slippage_ratio_quarantine: float = 1.8
    slippage_ratio_recover_active: float = 1.2
    slippage_ratio_recover_from_quarantine: float = 1.3
    consistency_min: float = 99.0
    consistency_recover: float = 99.5
    missed_signals_quarantine: float = 5.0
    dd_quarantine_pct: float = 25.0


@dataclass(frozen=True)
class SymbolGovernanceWeights:
    """Symbol Governance 權重乘數與約束"""
    active_multiplier: float = 1.0
    deweight_multiplier: float = 0.5
    quarantine_multiplier: float = 0.0
    min_weight: float = 0.03
    max_weight: float = 0.20


@dataclass(frozen=True)
class SymbolGovernanceConfig:
    """
    Symbol Governance 配置

    Negative Edge Filter with Kill-List/Quarantine
    
    enabled: 是否啟用治理機制（False = 行為與無治理完全一致）
    review_frequency: 審查頻率 ("weekly")
    warmup_days: 首次動作前需要的最少天數
    quarantine_min_days: 隔離最短天數
    consecutive_reviews_for_quarantine: 連續幾次 review 才觸發 quarantine
    consecutive_reviews_for_recovery: 連續幾次 review 才觸發 recovery
    thresholds: 狀態轉移閾值
    weights: 權重乘數與約束
    artifacts_dir: Artifact 輸出目錄
    """
    enabled: bool = False
    review_frequency: str = "weekly"
    warmup_days: int = 14
    quarantine_min_days: int = 14
    consecutive_reviews_for_quarantine: int = 2
    consecutive_reviews_for_recovery: int = 2
    thresholds: SymbolGovernanceThresholds = SymbolGovernanceThresholds()
    weights: SymbolGovernanceWeights = SymbolGovernanceWeights()
    artifacts_dir: str = "reports/symbol_governance"


@dataclass(frozen=True)
class RebalanceBandConfig:
    """
    Rebalance Band — 抑制微幅調倉（減少實盤摩擦）

    enabled: 是否啟用 rebalance band gate
        - False（預設）：行為與無 band 完全一致（backward compatible）
    
    threshold_pct: band 閾值 [0, 1]
        - 0.03 = 3%（推薦值）
        - 語義：若 |target_pct - current_pct| < threshold_pct，跳過本次下單
    
    apply_on_same_direction_only: 僅同方向時才套用 band
        - True（推薦）：方向翻轉（多空切換）不受 band 限制，避免卡住該翻的倉
        - False：所有情境都套用 band（含翻倉）
    """
    enabled: bool = False
    threshold_pct: float = 0.03
    apply_on_same_direction_only: bool = True


@dataclass(frozen=True)
class LiveConfig:
    """
    實盤專屬配置

    kline_cache: 啟用增量 K 線快取（推薦）
        - True（預設）：每次 cron 只拉增量 K 線，累積歷史數據
          → 策略從 bar 0 跑到最新 bar，與回測行為一致
        - False：每次拉最近 300 bar（滑動窗口），可能與回測有差異

    flip_confirmation: 方向切換需 2-tick 確認
        - True：方向翻轉需連續 2 次 cron 產生同方向信號
          → 防止滑動窗口造成的頻繁翻轉（kline_cache=False 時建議開啟）
        - False（預設）：立即執行方向切換
          → kline_cache=True 時，數據穩定，不需額外確認
    """
    kline_cache: bool = True
    flip_confirmation: bool = False
    prefer_limit_order: bool = False   # 優先使用限價單（Maker fee 更低）
    limit_order_timeout_s: int = 10    # 限價單等待成交秒數（超時改市價單）
    watchdog: dict = field(default_factory=dict)  # Live Watchdog 可選配置
    symbol_governance: SymbolGovernanceConfig = SymbolGovernanceConfig()
    rebalance_band: RebalanceBandConfig = RebalanceBandConfig()


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
    live: LiveConfig = LiveConfig()  # 實盤配置

    @property
    def is_futures(self) -> bool:
        """是否為合約模式"""
        return self.market.market_type == MarketType.FUTURES

    @property
    def supports_short(self) -> bool:
        """是否支援做空（合約模式才支援）"""
        return self.is_futures

    @property
    def market_type_str(self) -> str:
        """市場類型字串 ('spot' / 'futures')"""
        return self.market.market_type.value

    @property
    def direction(self) -> str:
        """
        交易方向 ('both' / 'long_only' / 'short_only')
        
        - Spot → 強制 'long_only'
        - Futures → 從 FuturesConfig 讀取，預設 'both'
        """
        if not self.is_futures:
            return "long_only"
        if self.futures and self.futures.direction:
            return self.futures.direction
        return "both"

    def get_report_dir(self, run_type: str = "backtest") -> Path:
        """
        自動計算標準報告路徑。
        
        結構: reports/{market_type}/{strategy_name}/{run_type}/
        
        Args:
            run_type: "backtest" | "portfolio" | "validation" | "live"
        
        Returns:
            Path 物件，不含時間戳（由呼叫者加）
        """
        base = Path(self.output.report_dir)
        return base / self.market_type_str / self.strategy.name / run_type

    def to_backtest_dict(self, symbol: str | None = None) -> dict:
        """
        產生標準回測配置 dict（供 run_symbol_backtest / validation / optimize 使用）
        
        集中管理，避免各 script 重複建構。
        包含 start / end 日期，讓回測引擎可以過濾數據範圍。
        """
        d = {
            "strategy_name": self.strategy.name,
            "strategy_params": self.strategy.get_params(symbol),
            "initial_cash": self.backtest.initial_cash,
            "fee_bps": self.backtest.fee_bps,
            "slippage_bps": self.backtest.slippage_bps,
            "interval": self.market.interval,
            "market_type": self.market_type_str,
            "direction": self.direction,
            "validate_data": self.backtest.validate_data,
            "clean_data_before": self.backtest.clean_data,
            "start": self.market.start,
            "end": self.market.end,
            "leverage": self.futures.leverage if self.futures else 1,
        }
        # position sizing（回測用）
        ps = self.position_sizing
        d["position_sizing"] = {
            "method": ps.method,
            "position_pct": ps.position_pct,
            "kelly_fraction": ps.kelly_fraction,
            "target_volatility": ps.target_volatility,
            "vol_lookback": ps.vol_lookback,
        }
        # 成本模型（新增）
        fr = self.backtest.funding_rate
        d["funding_rate"] = {
            "enabled": fr.enabled,
            "default_rate_8h": fr.default_rate_8h,
            "use_historical": fr.use_historical,
        }
        sm = self.backtest.slippage_model
        d["slippage_model"] = {
            "enabled": sm.enabled,
            "base_bps": sm.base_bps,
            "impact_coefficient": sm.impact_coefficient,
            "impact_power": sm.impact_power,
            "adv_lookback": sm.adv_lookback,
            "participation_rate": sm.participation_rate,
        }
        # overlay（可選 — 從原始 YAML 讀取）
        # ⚠️ 必須 deepcopy：overlay_params 會在 run_symbol_backtest 中被注入
        # per-symbol 的 _lsr_series / _oi_series / _fr_series。如果用 reference，
        # portfolio backtest 跑多個 symbol 時，symbol 1 的數據會汙染 symbol 2-N。
        if hasattr(self, '_overlay_cfg') and self._overlay_cfg:
            d["overlay"] = copy.deepcopy(self._overlay_cfg)
        return d


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
    aux_intervals = market_raw.pop("auxiliary_intervals", [])
    market = MarketConfig(
        **market_raw,
        market_type=market_type,
        auxiliary_intervals=aux_intervals if aux_intervals else [],
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

    # live 可選
    live_raw = raw.get("live", {})

    # symbol_governance 子配置
    sg_raw = live_raw.get("symbol_governance", {})
    sg_thresh_raw = sg_raw.get("thresholds", {})
    sg_weights_raw = sg_raw.get("weights", {})
    sg_thresholds = SymbolGovernanceThresholds(
        edge_sharpe_deweight=sg_thresh_raw.get("edge_sharpe_deweight", 0.3),
        edge_sharpe_quarantine=sg_thresh_raw.get("edge_sharpe_quarantine", 0.0),
        edge_sharpe_recover_active=sg_thresh_raw.get("edge_sharpe_recover_active", 0.8),
        edge_sharpe_recover_from_quarantine=sg_thresh_raw.get("edge_sharpe_recover_from_quarantine", 0.5),
        edge_per_turnover_min=sg_thresh_raw.get("edge_per_turnover_min", 0.0),
        slippage_ratio_deweight=sg_thresh_raw.get("slippage_ratio_deweight", 1.5),
        slippage_ratio_quarantine=sg_thresh_raw.get("slippage_ratio_quarantine", 1.8),
        slippage_ratio_recover_active=sg_thresh_raw.get("slippage_ratio_recover_active", 1.2),
        slippage_ratio_recover_from_quarantine=sg_thresh_raw.get("slippage_ratio_recover_from_quarantine", 1.3),
        consistency_min=sg_thresh_raw.get("consistency_min", 99.0),
        consistency_recover=sg_thresh_raw.get("consistency_recover", 99.5),
        missed_signals_quarantine=sg_thresh_raw.get("missed_signals_quarantine", 5.0),
        dd_quarantine_pct=sg_thresh_raw.get("dd_quarantine_pct", 25.0),
    )
    sg_weights = SymbolGovernanceWeights(
        active_multiplier=sg_weights_raw.get("active_multiplier", 1.0),
        deweight_multiplier=sg_weights_raw.get("deweight_multiplier", 0.5),
        quarantine_multiplier=sg_weights_raw.get("quarantine_multiplier", 0.0),
        min_weight=sg_weights_raw.get("min_weight", 0.03),
        max_weight=sg_weights_raw.get("max_weight", 0.20),
    )
    sg_cfg = SymbolGovernanceConfig(
        enabled=sg_raw.get("enabled", False),
        review_frequency=sg_raw.get("review_frequency", "weekly"),
        warmup_days=sg_raw.get("warmup_days", 14),
        quarantine_min_days=sg_raw.get("quarantine_min_days", 14),
        consecutive_reviews_for_quarantine=sg_raw.get("consecutive_reviews_for_quarantine", 2),
        consecutive_reviews_for_recovery=sg_raw.get("consecutive_reviews_for_recovery", 2),
        thresholds=sg_thresholds,
        weights=sg_weights,
        artifacts_dir=sg_raw.get("artifacts_dir", "reports/symbol_governance"),
    )

    # rebalance_band 子配置
    rb_raw = live_raw.get("rebalance_band", {})
    rb_cfg = RebalanceBandConfig(
        enabled=rb_raw.get("enabled", False),
        threshold_pct=rb_raw.get("threshold_pct", 0.03),
        apply_on_same_direction_only=rb_raw.get("apply_on_same_direction_only", True),
    )

    live_cfg = LiveConfig(
        kline_cache=live_raw.get("kline_cache", True),
        flip_confirmation=live_raw.get("flip_confirmation", False),
        prefer_limit_order=live_raw.get("prefer_limit_order", False),
        limit_order_timeout_s=live_raw.get("limit_order_timeout_s", 10),
        watchdog=live_raw.get("watchdog", {}),
        symbol_governance=sg_cfg,
        rebalance_band=rb_cfg,
    )

    # output 可選（預設 ./reports）
    output_raw = raw.get("output", {})
    output = OutputConfig(**output_raw) if output_raw else OutputConfig()

    # backtest 配置（含成本模型子項）
    bt_raw = dict(raw["backtest"])
    
    # 解析 funding_rate 子配置
    fr_raw = bt_raw.pop("funding_rate", {})
    funding_rate_cfg = FundingRateModelConfig(
        enabled=fr_raw.get("enabled", False),
        default_rate_8h=fr_raw.get("default_rate_8h", 0.0001),
        use_historical=fr_raw.get("use_historical", True),
    )
    
    # 解析 slippage_model 子配置
    sm_raw = bt_raw.pop("slippage_model", {})
    slippage_model_cfg = SlippageModelConfig(
        enabled=sm_raw.get("enabled", False),
        base_bps=sm_raw.get("base_bps", 2.0),
        impact_coefficient=sm_raw.get("impact_coefficient", 0.1),
        impact_power=sm_raw.get("impact_power", 0.5),
        adv_lookback=sm_raw.get("adv_lookback", 20),
        participation_rate=sm_raw.get("participation_rate", 0.10),
    )
    
    backtest_cfg = BacktestConfig(
        **bt_raw,
        funding_rate=funding_rate_cfg,
        slippage_model=slippage_model_cfg,
    )

    app_cfg = AppConfig(
        market=market,
        backtest=backtest_cfg,
        strategy=StrategyConfig(
            name=raw["strategy"]["name"],
            params=raw["strategy"].get("params", {}),
            symbol_overrides=raw["strategy"].get("symbol_overrides"),
            display_name=raw["strategy"].get("display_name"),
        ),
        output=output,
        data_dir=data_dir,
        portfolio=portfolio,
        risk=risk,
        position_sizing=position_sizing,
        futures=futures,
        notification=notification,
        live=live_cfg,
    )
    # 附加原始配置路徑（供 ensemble 路由等需要重讀 YAML 的功能使用）
    object.__setattr__(app_cfg, '_config_path', str(path))
    # 附加 overlay 配置（供 run_symbol_backtest 使用，不改 frozen dataclass 結構）
    overlay_raw = raw.get("strategy", {}).get("overlay")
    object.__setattr__(app_cfg, '_overlay_cfg', overlay_raw)
    return app_cfg