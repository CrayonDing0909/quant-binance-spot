"""
Live/Backtest 一致性驗證

核心問題：Live 產生的信號是否與同期回測一致？
這是檢測 look-ahead bias、data snooping、實作 bug 的關鍵工具。

使用場景：
1. 定期驗證（例如每週）：用過去 7 天的 live 交易紀錄 vs 同期回測
2. 上線前驗證：用歷史數據模擬 live 環境，確保信號一致
3. 問題診斷：當發現 live 績效遠差於回測時

常見不一致原因：
- Look-ahead bias：回測不小心用了未來數據
- 數據處理差異：live 和 backtest 的數據清洗方式不同
- 時間對齊問題：live 用了未收盤 K 線
- 狀態問題：策略有內部狀態但初始化方式不同
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Callable

import numpy as np
import pandas as pd

from ..utils.log import get_logger

logger = get_logger("consistency_validator")


# ══════════════════════════════════════════════════════════════════════════════
# 資料結構
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SignalComparison:
    """單一時間點的信號比較"""
    timestamp: str
    symbol: str
    live_signal: float
    backtest_signal: float
    price: float
    diff: float  # live - backtest
    is_consistent: bool  # |diff| < threshold
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class InconsistencyAnalysis:
    """不一致原因分析"""
    type: str  # "systematic_bias", "time_cluster", "volatility_related"
    description: str
    severity: str  # "high", "medium", "low"
    details: dict = field(default_factory=dict)


@dataclass
class TradeComparison:
    """交易比較"""
    timestamp: str
    symbol: str
    live_side: Optional[str]  # BUY/SELL/None
    backtest_side: Optional[str]
    live_qty: float
    backtest_qty: float
    is_consistent: bool


@dataclass 
class ConsistencyReport:
    """一致性驗證報告"""
    # 基本資訊
    report_id: str
    created_at: str
    period_start: str
    period_end: str
    symbols: List[str]
    strategy_name: str
    interval: str
    
    # 信號一致性統計
    total_comparisons: int
    consistent_count: int
    consistency_rate: float  # [0, 1]
    
    # 信號差異統計
    avg_signal_diff: float
    max_signal_diff: float
    std_signal_diff: float
    
    # 詳細比較（可選，可能很大）
    comparisons: List[SignalComparison] = field(default_factory=list)
    
    # 不一致原因分析
    inconsistencies: List[InconsistencyAnalysis] = field(default_factory=list)
    
    # 交易比較
    trade_comparisons: List[TradeComparison] = field(default_factory=list)
    trade_consistency_rate: Optional[float] = None
    
    # 績效比較（如果有交易紀錄）
    live_return_pct: Optional[float] = None
    backtest_return_pct: Optional[float] = None
    return_diff_pct: Optional[float] = None
    
    @property
    def is_consistent(self) -> bool:
        """整體是否一致（閾值 95%）"""
        return self.consistency_rate >= 0.95
    
    @property
    def has_systematic_bias(self) -> bool:
        """是否有系統性偏差"""
        return any(i.type == "systematic_bias" for i in self.inconsistencies)
    
    def to_dict(self) -> dict:
        """轉換為字典（用於 JSON 序列化）"""
        result = asdict(self)
        # 轉換 dataclass 列表
        result["comparisons"] = [c.to_dict() if hasattr(c, "to_dict") else asdict(c) 
                                  for c in self.comparisons]
        result["inconsistencies"] = [asdict(i) for i in self.inconsistencies]
        result["trade_comparisons"] = [asdict(t) for t in self.trade_comparisons]
        return result
    
    def save(self, path: Path) -> None:
        """儲存報告"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"📄 報告已儲存: {path}")
    
    def summary(self) -> str:
        """產生摘要文字"""
        status = "✅ 通過" if self.is_consistent else "❌ 未通過"
        lines = [
            "=" * 60,
            f"  Live/Backtest 一致性驗證報告",
            "=" * 60,
            f"  狀態: {status}",
            f"  期間: {self.period_start} → {self.period_end}",
            f"  策略: {self.strategy_name}",
            f"  交易對: {', '.join(self.symbols)}",
            "-" * 60,
            f"  信號一致性: {self.consistency_rate:.1%} ({self.consistent_count}/{self.total_comparisons})",
            f"  平均信號差異: {self.avg_signal_diff:.4f}",
            f"  最大信號差異: {self.max_signal_diff:.4f}",
        ]
        
        if self.trade_consistency_rate is not None:
            lines.append(f"  交易一致性: {self.trade_consistency_rate:.1%}")
        
        if self.live_return_pct is not None and self.backtest_return_pct is not None:
            lines.extend([
                "-" * 60,
                f"  Live 收益: {self.live_return_pct:+.2f}%",
                f"  Backtest 收益: {self.backtest_return_pct:+.2f}%",
                f"  差異: {self.return_diff_pct:+.2f}%",
            ])
        
        if self.inconsistencies:
            lines.append("-" * 60)
            lines.append("  ⚠️  不一致原因分析:")
            for inc in self.inconsistencies:
                lines.append(f"    • [{inc.severity.upper()}] {inc.description}")
        
        lines.append("=" * 60)
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# 主要驗證器
# ══════════════════════════════════════════════════════════════════════════════

class ConsistencyValidator:
    """
    Live/Backtest 一致性驗證器
    
    驗證方法：
    1. 信號一致性：對歷史每個 K 線時間點，比較 live 產生的信號與回測
    2. 交易一致性：比較實際交易與回測交易（時間點、方向）
    3. 績效一致性：比較同期收益率
    
    使用範例：
        validator = ConsistencyValidator(
            strategy_name="rsi_adx_atr",
            params={"rsi_period": 10, "oversold": 30, ...},
            interval="1h",
        )
        
        # 驗證過去 7 天
        report = validator.validate_recent("BTCUSDT", days=7)
        print(report.summary())
        
        # 驗證指定期間
        report = validator.validate_period("BTCUSDT", "2026-01-01", "2026-02-01")
    """
    
    # 策略預熱所需的最小 K 線數量
    MIN_WARMUP_BARS = 300
    
    def __init__(
        self,
        strategy_name: str,
        params: dict,
        interval: str = "1h",
        signal_threshold: float = 0.05,  # 信號差異容忍度
        include_details: bool = True,  # 是否包含詳細比較（可能很大）
        market_type: str = "spot",
        direction: str = "both",
        overlay_cfg: dict | None = None,
        data_dir: str | None = None,
    ):
        """
        Args:
            strategy_name: 策略名稱
            params: 策略參數
            interval: K 線週期
            signal_threshold: 信號差異容忍度，|diff| <= threshold 視為一致
            include_details: 是否在報告中包含每個時間點的詳細比較
            market_type: 市場類型 "spot" 或 "futures"
            direction: 交易方向 "both", "long_only", "short_only"
            overlay_cfg: Overlay 配置（如有），格式同 YAML strategy.overlay 區塊
            data_dir: 數據目錄（供 overlay 載入 OI 等輔助數據）
        """
        self.strategy_name = strategy_name
        self.params = params
        self.interval = interval
        self.signal_threshold = signal_threshold
        self.include_details = include_details
        self.market_type = market_type
        self.direction = direction
        self.overlay_cfg = overlay_cfg
        self.data_dir = data_dir
        
        # 載入策略函數
        self._strategy_func = None
    
    def _get_strategy_func(self) -> Callable:
        """延遲載入策略函數"""
        if self._strategy_func is None:
            from ..strategy import get_strategy
            self._strategy_func = get_strategy(self.strategy_name)
        return self._strategy_func
    
    def validate_period(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        live_trades: Optional[List[dict]] = None,
    ) -> ConsistencyReport:
        """
        驗證指定期間的一致性
        
        Args:
            symbol: 交易對
            start_date: 開始日期 "YYYY-MM-DD"
            end_date: 結束日期
            live_trades: 實際 live 交易紀錄（用於交易比較）
            
        Returns:
            ConsistencyReport 驗證報告
        """
        logger.info(f"🔬 驗證 {symbol} {start_date} → {end_date}")
        
        # 1. 獲取歷史 K 線數據
        from ..data.klines import fetch_klines
        from ..data.quality import clean_data
        
        df = fetch_klines(symbol, self.interval, start_date, end_date)
        df = clean_data(df, fill_method="forward", remove_outliers=False, remove_duplicates=True)
        
        if len(df) < self.MIN_WARMUP_BARS + 50:
            raise ValueError(
                f"數據不足: {len(df)} bars，需要至少 {self.MIN_WARMUP_BARS + 50} bars"
            )
        
        # 2. 執行完整回測以獲取 backtest 信號序列
        backtest_signals = self._run_backtest_signals(df, symbol)
        
        # 3. 模擬 live 信號生成並比較
        comparisons = self._compare_signals(df, backtest_signals, symbol)
        
        # 4. 統計分析
        consistent_count = sum(1 for c in comparisons if c.is_consistent)
        consistency_rate = consistent_count / len(comparisons) if comparisons else 0
        
        diffs = [c.diff for c in comparisons]
        avg_diff = np.mean(diffs) if diffs else 0
        max_diff = max(abs(d) for d in diffs) if diffs else 0
        std_diff = np.std(diffs) if diffs else 0
        
        # 5. 分析不一致原因
        inconsistencies = self._analyze_inconsistencies(comparisons)
        
        # 6. 交易比較（如果提供了 live 交易紀錄）
        trade_comparisons = []
        trade_consistency_rate = None
        if live_trades:
            trade_comparisons = self._compare_trades(df, backtest_signals, live_trades, symbol)
            if trade_comparisons:
                trade_consistent = sum(1 for t in trade_comparisons if t.is_consistent)
                trade_consistency_rate = trade_consistent / len(trade_comparisons)
        
        # 7. 產生報告
        report = ConsistencyReport(
            report_id=f"{symbol}_{start_date}_{end_date}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            created_at=datetime.now(timezone.utc).isoformat(),
            period_start=start_date,
            period_end=end_date,
            symbols=[symbol],
            strategy_name=self.strategy_name,
            interval=self.interval,
            total_comparisons=len(comparisons),
            consistent_count=consistent_count,
            consistency_rate=consistency_rate,
            avg_signal_diff=avg_diff,
            max_signal_diff=max_diff,
            std_signal_diff=std_diff,
            comparisons=comparisons if self.include_details else [],
            inconsistencies=inconsistencies,
            trade_comparisons=trade_comparisons,
            trade_consistency_rate=trade_consistency_rate,
        )
        
        logger.info(f"   信號一致性: {consistency_rate:.1%} ({consistent_count}/{len(comparisons)})")
        if inconsistencies:
            for inc in inconsistencies:
                logger.warning(f"   ⚠️  {inc.description}")
        
        return report
    
    def validate_recent(
        self,
        symbol: str,
        days: int = 7,
        live_state_path: Optional[Path] = None,
        use_binance_api: bool = True,
    ) -> ConsistencyReport:
        """
        驗證最近 N 天的一致性
        
        這是最常用的方法：用過去 7 天的交易紀錄與同期回測比對
        
        Args:
            symbol: 交易對
            days: 回看天數
            live_state_path: live 狀態檔路徑（包含交易紀錄）
            use_binance_api: 是否從 Binance API 獲取真實交易（優先）
        """
        end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        start_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
        start_ts = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)
        
        live_trades = None
        
        # 1. 優先從 Binance API 獲取
        if use_binance_api:
            live_trades = self._fetch_trades_from_binance(symbol, start_ts)
            if live_trades:
                logger.info(f"   📡 從 Binance API 獲取 {len(live_trades)} 筆交易")
        
        # 2. Fallback 到 state 檔案
        if not live_trades and live_state_path and live_state_path.exists():
            try:
                with open(live_state_path) as f:
                    state = json.load(f)
                live_trades = state.get("trades", [])
                logger.info(f"   📄 從 state 檔案載入 {len(live_trades)} 筆交易")
            except Exception as e:
                logger.warning(f"   ⚠️  無法載入 live 狀態檔: {e}")
        
        return self.validate_period(symbol, start_date, end_date, live_trades)
    
    def _fetch_trades_from_binance(
        self, 
        symbol: str, 
        start_time: int,
    ) -> Optional[List[dict]]:
        """從 Binance API 獲取交易歷史"""
        try:
            from ..live.binance_futures_broker import BinanceFuturesBroker
            
            broker = BinanceFuturesBroker(dry_run=True)
            raw_trades = broker.get_trade_history(
                symbol=symbol, 
                limit=500,
                start_time=start_time,
            )
            
            if not raw_trades:
                return None
            
            # 轉換為標準格式
            trades = []
            for t in raw_trades:
                trades.append({
                    "timestamp": datetime.fromtimestamp(
                        t["time"] / 1000, tz=timezone.utc
                    ).isoformat(),
                    "symbol": t["symbol"],
                    "side": t["side"],
                    "qty": t["qty"],
                    "price": t["price"],
                    "pnl": t.get("realized_pnl", 0),
                })
            
            return trades
            
        except Exception as e:
            logger.warning(f"   ⚠️  無法從 Binance API 獲取交易: {e}")
            return None
    
    def validate_multiple(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
    ) -> Dict[str, ConsistencyReport]:
        """
        驗證多個交易對
        
        Returns:
            {symbol: ConsistencyReport}
        """
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.validate_period(symbol, start_date, end_date)
            except Exception as e:
                logger.error(f"❌ {symbol} 驗證失敗: {e}")
        return results
    
    # ══════════════════════════════════════════════════════════════════════════
    # 內部方法
    # ══════════════════════════════════════════════════════════════════════════
    
    def _run_backtest_signals(self, df: pd.DataFrame, symbol: str) -> pd.Series:
        """
        執行回測，獲取完整的信號序列
        
        這是使用完整數據一次性計算所有信號（回測模式）。
        如果提供了 overlay_cfg，會在策略信號後套用 overlay，
        與 run_symbol_backtest 和 live signal_generator 的行為一致。

        **重要**: 使用 signal_delay=1（模擬 trade_on="next_open"），
        讓回測信號與 run_symbol_backtest 行為一致。
        """
        from ..strategy.base import StrategyContext
        
        ctx = StrategyContext(
            symbol=symbol,
            interval=self.interval,
            market_type=self.market_type,
            direction=self.direction,
            signal_delay=1,  # 回測模式：signal_delay=1（trade_on="next_open"）
        )
        strategy_func = self._get_strategy_func()
        positions = strategy_func(df, ctx, self.params)
        
        # 套用 overlay（使用共用 pipeline，確保與 backtest / live 行為一致）
        if self.overlay_cfg and self.overlay_cfg.get("enabled", False):
            try:
                from ..strategy.overlays.overlay_pipeline import prepare_and_apply_overlay
                positions = prepare_and_apply_overlay(
                    positions, df, self.overlay_cfg, symbol,
                    data_dir=self.data_dir,
                )
            except ImportError:
                logger.warning(
                    "Consistency validator: overlay 模組無法載入，跳過 overlay"
                )
        
        return positions
    
    def _compare_signals(
        self,
        df: pd.DataFrame,
        backtest_signals: pd.Series,
        symbol: str,
    ) -> List[SignalComparison]:
        """
        比較 live 模式與 backtest 模式的信號
        
        關鍵：live 模式只用該時間點之前的數據（模擬真實 live 環境）
        """
        from ..strategy.base import StrategyContext
        
        comparisons = []
        strategy_func = self._get_strategy_func()
        
        # 從 warmup 之後開始比較
        start_idx = self.MIN_WARMUP_BARS
        
        for i in range(start_idx, len(df)):
            timestamp = df.index[i]
            current_price = float(df["close"].iloc[i])
            
            # Live 模式：只看到當前時間點及之前的數據
            # 這是關鍵！live 不應該看到未來數據
            live_df = df.iloc[:i+1].copy()
            
            ctx = StrategyContext(
                symbol=symbol,
                interval=self.interval,
                market_type=self.market_type,
                direction=self.direction,
                signal_delay=0,  # Live 模式：信號即時執行，不延遲
            )
            live_positions = strategy_func(live_df, ctx, self.params)
            live_signal = float(live_positions.iloc[-1])
            
            # Backtest 模式：取該時間點的信號
            backtest_signal = float(backtest_signals.iloc[i])
            
            diff = live_signal - backtest_signal
            is_consistent = abs(diff) <= self.signal_threshold
            
            comparisons.append(SignalComparison(
                timestamp=str(timestamp),
                symbol=symbol,
                live_signal=round(live_signal, 4),
                backtest_signal=round(backtest_signal, 4),
                price=current_price,
                diff=round(diff, 4),
                is_consistent=is_consistent,
            ))
        
        return comparisons
    
    def _compare_trades(
        self,
        df: pd.DataFrame,
        backtest_signals: pd.Series,
        live_trades: List[dict],
        symbol: str,
    ) -> List[TradeComparison]:
        """比較 live 交易與回測交易"""
        comparisons = []
        
        # 從 backtest signals 提取交易點
        backtest_trades = self._extract_trades_from_signals(df, backtest_signals, symbol)
        
        # 按時間戳建立索引
        live_by_time = {t.get("timestamp", ""): t for t in live_trades if t.get("symbol") == symbol}
        
        for bt_trade in backtest_trades:
            # 尋找同一時間的 live 交易
            ts = bt_trade["timestamp"]
            live_trade = live_by_time.get(ts)
            
            if live_trade:
                is_consistent = (
                    live_trade.get("side") == bt_trade["side"] and
                    abs(live_trade.get("qty", 0) - bt_trade["qty"]) / max(bt_trade["qty"], 1e-10) < 0.1
                )
            else:
                is_consistent = False
            
            comparisons.append(TradeComparison(
                timestamp=ts,
                symbol=symbol,
                live_side=live_trade.get("side") if live_trade else None,
                backtest_side=bt_trade["side"],
                live_qty=live_trade.get("qty", 0) if live_trade else 0,
                backtest_qty=bt_trade["qty"],
                is_consistent=is_consistent,
            ))
        
        return comparisons
    
    def _extract_trades_from_signals(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        symbol: str,
    ) -> List[dict]:
        """從信號序列提取交易點"""
        trades = []
        prev_signal = 0.0
        
        for i in range(1, len(signals)):
            curr_signal = float(signals.iloc[i])
            diff = curr_signal - prev_signal
            
            if abs(diff) >= 0.02:  # 與 runner 相同的閾值
                side = "BUY" if diff > 0 else "SELL"
                trades.append({
                    "timestamp": str(df.index[i]),
                    "symbol": symbol,
                    "side": side,
                    "qty": abs(diff),
                })
            
            prev_signal = curr_signal
        
        return trades
    
    def _analyze_inconsistencies(
        self,
        comparisons: List[SignalComparison],
    ) -> List[InconsistencyAnalysis]:
        """分析不一致的可能原因"""
        inconsistent = [c for c in comparisons if not c.is_consistent]
        if not inconsistent:
            return []
        
        analyses = []
        diffs = [c.diff for c in inconsistent]
        
        # 1. 檢查系統性偏差（live 總是高/低）
        avg_diff = np.mean(diffs)
        if abs(avg_diff) > self.signal_threshold:
            direction = "高" if avg_diff > 0 else "低"
            analyses.append(InconsistencyAnalysis(
                type="systematic_bias",
                description=f"系統性偏差: live 信號平均比 backtest {direction} {abs(avg_diff):.4f}",
                severity="high",
                details={"avg_diff": avg_diff, "direction": direction},
            ))
        
        # 2. 檢查是否集中在特定時間段
        if len(inconsistent) >= 5:
            timestamps = [c.timestamp for c in inconsistent]
            # 簡單檢查：是否有連續的不一致
            consecutive_count = 0
            max_consecutive = 0
            for i in range(1, len(timestamps)):
                # 這裡簡化處理，實際應該比較時間差
                consecutive_count += 1
                max_consecutive = max(max_consecutive, consecutive_count)
            
            if max_consecutive >= 5:
                analyses.append(InconsistencyAnalysis(
                    type="time_cluster",
                    description=f"不一致集中出現，最長連續 {max_consecutive} 次",
                    severity="medium",
                    details={"max_consecutive": max_consecutive},
                ))
        
        # 3. 檢查是否與價格波動相關
        prices = [c.price for c in inconsistent]
        all_prices = [c.price for c in comparisons]
        
        if prices and all_prices:
            inconsistent_avg_price = np.mean(prices)
            all_avg_price = np.mean(all_prices)
            
            price_diff_pct = abs(inconsistent_avg_price - all_avg_price) / all_avg_price
            if price_diff_pct > 0.05:  # 5% 差異
                analyses.append(InconsistencyAnalysis(
                    type="price_related",
                    description=f"不一致與特定價格區間相關 (差異 {price_diff_pct:.1%})",
                    severity="low",
                    details={"inconsistent_avg_price": inconsistent_avg_price, "all_avg_price": all_avg_price},
                ))
        
        # 4. 如果不一致比例很高，標記為嚴重問題
        inconsistency_rate = len(inconsistent) / len(comparisons) if comparisons else 0
        if inconsistency_rate > 0.1:  # 超過 10% 不一致
            analyses.append(InconsistencyAnalysis(
                type="high_inconsistency",
                description=f"不一致比例過高 ({inconsistency_rate:.1%})，可能有實作 bug 或 look-ahead bias",
                severity="high",
                details={"inconsistency_rate": inconsistency_rate},
            ))
        
        return analyses


# ══════════════════════════════════════════════════════════════════════════════
# 便利函數
# ══════════════════════════════════════════════════════════════════════════════

def run_consistency_check(
    config_path: str = "config/rsi_adx_atr.yaml",
    days: int = 7,
    symbols: Optional[List[str]] = None,
    output_dir: str = "reports/validation",
) -> Dict[str, ConsistencyReport]:
    """
    執行一致性驗證（便利函數）
    
    Args:
        config_path: 配置檔路徑
        days: 回看天數
        symbols: 交易對列表，None 則使用配置中的 symbols
        output_dir: 報告輸出目錄
        
    Returns:
        {symbol: ConsistencyReport}
    """
    from ..config import load_config
    
    cfg = load_config(config_path)
    
    symbols = symbols or cfg.market.symbols
    params = cfg.strategy.params
    
    # 讀取 overlay 配置（如有）
    overlay_cfg = getattr(cfg, '_overlay_cfg', None)
    
    validator = ConsistencyValidator(
        strategy_name=cfg.strategy.name,
        params=params,
        interval=cfg.market.interval,
        market_type=cfg.market_type_str,
        direction=cfg.direction,
        overlay_cfg=overlay_cfg,
        data_dir=str(cfg.data_dir) if cfg.data_dir else None,
    )
    
    results = {}
    output_path = Path(output_dir)
    
    for symbol in symbols:
        # 獲取該 symbol 的參數（含覆寫）
        symbol_params = cfg.strategy.get_params(symbol)
        validator.params = symbol_params
        
        try:
            # 檢查是否有 live state 檔案
            live_state_path = Path(f"reports/live/{cfg.strategy.name}/paper_state.json")
            
            report = validator.validate_recent(
                symbol=symbol,
                days=days,
                live_state_path=live_state_path,
            )
            
            results[symbol] = report
            
            # 儲存報告
            report_path = output_path / f"consistency_{symbol}_{datetime.now().strftime('%Y%m%d')}.json"
            report.save(report_path)
            
            # 印出摘要
            print(report.summary())
            
        except Exception as e:
            logger.error(f"❌ {symbol} 驗證失敗: {e}")
            import traceback
            traceback.print_exc()
    
    # 總結
    print("\n" + "=" * 60)
    print("  總結")
    print("=" * 60)
    for symbol, report in results.items():
        status = "✅" if report.is_consistent else "❌"
        print(f"  {status} {symbol}: {report.consistency_rate:.1%}")
    
    return results
