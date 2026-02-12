"""
Kelly 公式回測驗證

用途：
1. 驗證策略是否適合使用 Kelly
2. 比較不同 Kelly fraction 的表現
3. 檢測策略的 edge 是否穩定

核心問題：
- Kelly 計算依賴歷史統計（勝率、盈虧比）
- 如果這些統計不穩定，Kelly 倉位會劇烈波動
- 本模組透過回測驗證這些統計的穩定性

使用方法：
    results = kelly_backtest_comparison(
        symbol="BTCUSDT",
        data_path=Path("data/binance/spot/BTCUSDT_1h.parquet"),
        cfg=backtest_config,
        kelly_fractions=[0.0, 0.25, 0.5, 1.0],  # 比較不同 fraction
    )
    
    # 分析結果
    print(results.summary())
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd

from ..utils.log import get_logger

logger = get_logger("kelly_validation")


# ══════════════════════════════════════════════════════════════════════════════
# 資料結構
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class KellyStats:
    """Kelly 計算所需的統計數據"""
    win_rate: float  # 勝率 [0, 1]
    avg_win: float   # 平均盈利（正數）
    avg_loss: float  # 平均虧損（正數）
    total_trades: int
    winning_trades: int
    losing_trades: int
    
    @property
    def win_loss_ratio(self) -> float:
        """盈虧比 W = avg_win / avg_loss"""
        if self.avg_loss <= 0:
            return float('inf')
        return self.avg_win / self.avg_loss
    
    @property
    def kelly_pct(self) -> float:
        """
        計算 Full Kelly 比例
        
        Kelly% = (P × W - L) / W
        """
        W = self.win_loss_ratio
        if W <= 0 or W == float('inf'):
            return 0.0
        
        L = 1 - self.win_rate
        kelly = (self.win_rate * W - L) / W
        return max(0.0, kelly)  # Kelly 不能為負
    
    @property
    def edge(self) -> float:
        """
        策略優勢（期望值）
        
        Edge = P × avg_win - (1-P) × avg_loss
        """
        return self.win_rate * self.avg_win - (1 - self.win_rate) * self.avg_loss
    
    @property
    def edge_pct(self) -> float:
        """相對於平均交易金額的 edge 百分比"""
        avg_trade = (self.avg_win + self.avg_loss) / 2
        if avg_trade <= 0:
            return 0.0
        return self.edge / avg_trade
    
    def is_profitable(self) -> bool:
        """策略是否有正期望值"""
        return self.edge > 0
    
    def summary(self) -> str:
        lines = [
            f"  勝率: {self.win_rate:.1%} ({self.winning_trades}/{self.total_trades})",
            f"  盈虧比: {self.win_loss_ratio:.2f} (avg_win={self.avg_win:.2f}, avg_loss={self.avg_loss:.2f})",
            f"  Edge: {self.edge:.2f} ({self.edge_pct:.1%})",
            f"  Full Kelly: {self.kelly_pct:.1%}",
            f"  建議倉位:",
            f"    - Half Kelly (0.5): {self.kelly_pct * 0.5:.1%}",
            f"    - Quarter Kelly (0.25): {self.kelly_pct * 0.25:.1%}",
        ]
        return "\n".join(lines)


@dataclass
class KellyBacktestResult:
    """單一 Kelly fraction 的回測結果"""
    kelly_fraction: float
    effective_kelly_pct: float  # 實際使用的 Kelly 比例
    
    # 績效指標
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    calmar_ratio: float  # return / max_dd
    
    # 風險指標
    volatility: float
    var_95: float  # 95% VaR
    
    # 交易統計
    total_trades: int
    avg_position_size: float


@dataclass
class KellyValidationReport:
    """Kelly 驗證完整報告"""
    symbol: str
    period_start: str
    period_end: str
    
    # Kelly 統計
    kelly_stats: KellyStats
    
    # 不同 fraction 的回測結果
    backtest_results: List[KellyBacktestResult]
    
    # 穩定性分析
    kelly_stability: float  # Kelly 在滾動窗口中的標準差
    win_rate_stability: float
    edge_stability: float
    
    # 建議
    recommended_fraction: float
    recommendation_reason: str
    
    def summary(self) -> str:
        lines = [
            "=" * 60,
            f"  Kelly 公式驗證報告",
            "=" * 60,
            f"  交易對: {self.symbol}",
            f"  期間: {self.period_start} → {self.period_end}",
            "-" * 60,
            "  【策略統計】",
            self.kelly_stats.summary(),
            "-" * 60,
            "  【穩定性分析】",
            f"  Kelly 穩定性: {self.kelly_stability:.1%} (越低越好)",
            f"  勝率穩定性: {self.win_rate_stability:.1%}",
            f"  Edge 穩定性: {self.edge_stability:.1%}",
            "-" * 60,
            "  【不同 Kelly Fraction 比較】",
        ]
        
        # 表頭
        lines.append(f"  {'Fraction':<10} {'Return':<10} {'Sharpe':<10} {'MaxDD':<10} {'Calmar':<10}")
        lines.append("  " + "-" * 50)
        
        for r in self.backtest_results:
            lines.append(
                f"  {r.kelly_fraction:<10.2f} "
                f"{r.total_return_pct:>+8.1f}% "
                f"{r.sharpe_ratio:>9.2f} "
                f"{r.max_drawdown_pct:>9.1f}% "
                f"{r.calmar_ratio:>9.2f}"
            )
        
        lines.extend([
            "-" * 60,
            "  【建議】",
            f"  推薦 Kelly Fraction: {self.recommended_fraction}",
            f"  原因: {self.recommendation_reason}",
            "=" * 60,
        ])
        
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# 核心計算
# ══════════════════════════════════════════════════════════════════════════════

def calculate_kelly_stats(trades: List[Dict]) -> KellyStats:
    """
    從交易紀錄計算 Kelly 統計
    
    Args:
        trades: 交易紀錄列表，每個交易需有 'pnl' 欄位
    """
    if not trades:
        return KellyStats(
            win_rate=0.5, avg_win=1.0, avg_loss=1.0,
            total_trades=0, winning_trades=0, losing_trades=0
        )
    
    wins = [t['pnl'] for t in trades if t.get('pnl', 0) > 0]
    losses = [abs(t['pnl']) for t in trades if t.get('pnl', 0) < 0]
    
    total = len(wins) + len(losses)
    if total == 0:
        return KellyStats(
            win_rate=0.5, avg_win=1.0, avg_loss=1.0,
            total_trades=len(trades), winning_trades=0, losing_trades=0
        )
    
    return KellyStats(
        win_rate=len(wins) / total if total > 0 else 0.5,
        avg_win=sum(wins) / len(wins) if wins else 1.0,
        avg_loss=sum(losses) / len(losses) if losses else 1.0,
        total_trades=len(trades),
        winning_trades=len(wins),
        losing_trades=len(losses),
    )


def extract_trades_from_portfolio(pf) -> List[Dict]:
    """
    從 vectorbt Portfolio 提取交易紀錄
    
    使用 positions 而不是 trades，獲得完整的 round-trip 交易。
    trades 會把部分平倉拆成多筆，positions 則是完整的進出場。
    
    Args:
        pf: vectorbt Portfolio 對象
        
    Returns:
        交易紀錄列表，每個包含 'pnl'
    """
    try:
        # 使用 positions 獲得完整的 round-trip 交易
        positions_records = pf.positions.records_readable
        
        if positions_records.empty:
            return []
        
        trades = []
        for _, row in positions_records.iterrows():
            pnl = row.get('PnL', row.get('Return', 0))
            if pd.notna(pnl):
                trades.append({
                    'pnl': float(pnl),
                    'entry_price': row.get('Avg Entry Price', 0),
                    'exit_price': row.get('Avg Exit Price', 0),
                    'size': row.get('Size', 0),
                })
        
        return trades
    except Exception as e:
        logger.warning(f"無法從 Portfolio 提取交易: {e}")
        return []


def calculate_kelly_stats_from_portfolio(pf) -> KellyStats:
    """
    從 vectorbt Portfolio 計算 Kelly 統計
    
    Args:
        pf: vectorbt Portfolio 對象
    """
    trades = extract_trades_from_portfolio(pf)
    return calculate_kelly_stats(trades)


def calculate_rolling_kelly(
    trades: List[Dict],
    window_size: int = 30,
) -> pd.DataFrame:
    """
    計算滾動窗口的 Kelly 統計
    
    用於分析 Kelly 的穩定性
    """
    if len(trades) < window_size:
        return pd.DataFrame()
    
    results = []
    
    for i in range(window_size, len(trades) + 1):
        window_trades = trades[i - window_size:i]
        stats = calculate_kelly_stats(window_trades)
        
        results.append({
            'index': i,
            'win_rate': stats.win_rate,
            'win_loss_ratio': stats.win_loss_ratio,
            'kelly_pct': stats.kelly_pct,
            'edge': stats.edge,
        })
    
    return pd.DataFrame(results)


def calculate_kelly_stability(rolling_df: pd.DataFrame) -> Dict[str, float]:
    """
    計算 Kelly 相關指標的穩定性（變異係數）
    
    變異係數 = 標準差 / 平均值
    越低表示越穩定
    """
    if rolling_df.empty:
        return {'kelly': 1.0, 'win_rate': 1.0, 'edge': 1.0}
    
    def cv(series: pd.Series) -> float:
        """計算變異係數"""
        mean = series.mean()
        if abs(mean) < 1e-10:
            return 1.0
        return series.std() / abs(mean)
    
    return {
        'kelly': cv(rolling_df['kelly_pct']),
        'win_rate': cv(rolling_df['win_rate']),
        'edge': cv(rolling_df['edge']),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 回測比較
# ══════════════════════════════════════════════════════════════════════════════

def kelly_backtest_comparison(
    symbol: str,
    data_path: Path,
    cfg: dict,
    kelly_fractions: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0],
    strategy_name: Optional[str] = None,
) -> KellyValidationReport:
    """
    比較不同 Kelly fraction 的回測表現
    
    Args:
        symbol: 交易對
        data_path: K 線數據路徑
        cfg: 回測配置
        kelly_fractions: 要比較的 Kelly fractions
        strategy_name: 策略名稱
        
    Returns:
        KellyValidationReport
    """
    import vectorbt as vbt
    from .run_backtest import run_symbol_backtest
    from ..data.storage import load_klines
    
    logger.info(f"📊 Kelly 驗證: {symbol}")
    
    # 載入數據
    df = load_klines(data_path)
    period_start = df.index[0].strftime("%Y-%m-%d")
    period_end = df.index[-1].strftime("%Y-%m-%d")
    
    # 先跑一次基礎回測，獲取交易紀錄和 position 訊號
    base_result = run_symbol_backtest(symbol, data_path, cfg, strategy_name)
    
    # 從 Portfolio 物件提取交易紀錄
    pf = base_result.get("pf")
    if pf is not None:
        base_trades = extract_trades_from_portfolio(pf)
    else:
        base_trades = base_result.get("trades", [])
    
    # 獲取基礎 position 訊號（用於後續縮放）
    base_pos = base_result.get("pos")
    
    # 計算 Kelly 統計
    kelly_stats = calculate_kelly_stats(base_trades)
    logger.info(f"   Full Kelly: {kelly_stats.kelly_pct:.1%}, Edge: {kelly_stats.edge:.2f}")
    
    # 計算滾動 Kelly 穩定性
    rolling_df = calculate_rolling_kelly(base_trades, window_size=min(30, len(base_trades) // 3))
    stability = calculate_kelly_stability(rolling_df)
    
    # 對每個 fraction 執行回測
    backtest_results = []
    
    # 準備回測參數
    close = df["close"]
    open_ = df["open"]
    fee = cfg.get("fee_bps", 10) / 10_000.0
    slippage = cfg.get("slippage_bps", 5) / 10_000.0
    initial_cash = cfg.get("initial_cash", 10000)
    
    # 從 cfg 取得 direction，使用共用映射函數
    from .run_backtest import to_vbt_direction
    kelly_vbt_direction = to_vbt_direction(cfg.get("direction", "long_only"))
    
    for fraction in kelly_fractions:
        effective_kelly = kelly_stats.kelly_pct * fraction
        
        # 計算實際倉位比例（Kelly pct * fraction，但至少要有一個最小值來產生訊號）
        # 當 fraction=0 時，倉位為 0（不交易）
        # 當 fraction=1 時，倉位為策略原始訊號 * Kelly pct
        if base_pos is not None and fraction > 0:
            # 縮放 position: 原始訊號 * effective_kelly
            # 例如原始訊號是 1（全倉），effective_kelly 是 0.25，則實際倉位是 0.25
            position_scale = min(effective_kelly, 1.0) if effective_kelly > 0 else 0.01
            scaled_pos = base_pos * position_scale
        else:
            # fraction = 0 或沒有 position 訊號，使用極小值
            scaled_pos = base_pos * 0.001 if base_pos is not None else None
        
        try:
            if scaled_pos is not None:
                # 使用縮放後的倉位執行回測
                test_pf = vbt.Portfolio.from_orders(
                    close=close,
                    size=scaled_pos,
                    size_type="targetpercent",
                    price=open_,
                    fees=fee,
                    slippage=slippage,
                    init_cash=initial_cash,
                    freq="1h",
                    direction=kelly_vbt_direction,
                )
                stats = test_pf.stats()
            else:
                # 沒有 position 訊號，使用基礎回測結果
                stats = base_result.get("stats", {})
            
            # 計算 Calmar ratio
            total_return = stats.get("Total Return [%]", 0)
            max_dd = abs(stats.get("Max Drawdown [%]", 1))
            calmar = total_return / max_dd if max_dd > 0 else 0
            
            backtest_results.append(KellyBacktestResult(
                kelly_fraction=fraction,
                effective_kelly_pct=effective_kelly,
                total_return_pct=total_return,
                sharpe_ratio=stats.get("Sharpe Ratio", 0),
                max_drawdown_pct=max_dd,
                calmar_ratio=calmar,
                volatility=stats.get("Volatility (Ann.) [%]", 0),
                var_95=stats.get("Value at Risk", 0),
                total_trades=stats.get("Total Trades", 0),
                avg_position_size=effective_kelly,
            ))
            
            logger.info(
                f"   Fraction {fraction:.2f}: "
                f"Return={total_return:+.1f}%, Sharpe={stats.get('Sharpe Ratio', 0):.2f}, "
                f"MaxDD={max_dd:.1f}%"
            )
            
        except Exception as e:
            logger.warning(f"   Fraction {fraction} 回測失敗: {e}")
    
    # 決定推薦的 fraction
    recommended, reason = _recommend_kelly_fraction(
        kelly_stats, stability, backtest_results
    )
    
    return KellyValidationReport(
        symbol=symbol,
        period_start=period_start,
        period_end=period_end,
        kelly_stats=kelly_stats,
        backtest_results=backtest_results,
        kelly_stability=stability.get('kelly', 1.0),
        win_rate_stability=stability.get('win_rate', 1.0),
        edge_stability=stability.get('edge', 1.0),
        recommended_fraction=recommended,
        recommendation_reason=reason,
    )


def _recommend_kelly_fraction(
    stats: KellyStats,
    stability: Dict[str, float],
    results: List[KellyBacktestResult],
) -> Tuple[float, str]:
    """
    根據分析結果推薦 Kelly fraction
    """
    # 1. 檢查策略是否有正期望值
    if not stats.is_profitable():
        return 0.0, "策略期望值為負，不建議使用 Kelly"
    
    # 2. 檢查交易數量
    if stats.total_trades < 30:
        return 0.0, f"交易數量不足 ({stats.total_trades} < 30)，統計不可靠"
    
    # 3. 檢查穩定性
    kelly_cv = stability.get('kelly', 1.0)
    if kelly_cv > 0.5:
        return 0.25, f"Kelly 不穩定 (CV={kelly_cv:.2f})，建議保守使用 Quarter Kelly"
    
    # 4. 找最佳風險調整收益
    if results:
        # 用 Calmar ratio 找最佳平衡點
        best = max(results, key=lambda r: r.calmar_ratio if r.calmar_ratio > 0 else -float('inf'))
        
        if best.kelly_fraction <= 0.25:
            return best.kelly_fraction, f"Calmar ratio 最優 ({best.calmar_ratio:.2f})"
        elif best.kelly_fraction <= 0.5:
            return 0.25, "Half Kelly 風險較高，建議 Quarter Kelly"
        else:
            return 0.25, "Full Kelly 波動太大，建議 Quarter Kelly"
    
    # 默認保守
    return 0.25, "預設推薦 Quarter Kelly（風險較低）"


# ══════════════════════════════════════════════════════════════════════════════
# 便利函數
# ══════════════════════════════════════════════════════════════════════════════

def is_strategy_suitable_for_kelly(
    trades: List[Dict],
    min_trades: int = 30,
    min_edge_pct: float = 0.01,
    max_kelly_cv: float = 0.5,
) -> Tuple[bool, str]:
    """
    判斷策略是否適合使用 Kelly
    
    Args:
        trades: 交易紀錄
        min_trades: 最小交易數量
        min_edge_pct: 最小 edge 百分比
        max_kelly_cv: Kelly 最大變異係數
        
    Returns:
        (是否適合, 原因)
    """
    # 檢查交易數量
    if len(trades) < min_trades:
        return False, f"交易數量不足: {len(trades)} < {min_trades}"
    
    # 計算統計
    stats = calculate_kelly_stats(trades)
    
    # 檢查期望值
    if not stats.is_profitable():
        return False, f"期望值為負: edge = {stats.edge:.2f}"
    
    # 檢查 edge 強度
    if stats.edge_pct < min_edge_pct:
        return False, f"Edge 太弱: {stats.edge_pct:.2%} < {min_edge_pct:.2%}"
    
    # 檢查穩定性
    rolling_df = calculate_rolling_kelly(trades, window_size=min(30, len(trades) // 3))
    stability = calculate_kelly_stability(rolling_df)
    
    if stability.get('kelly', 1.0) > max_kelly_cv:
        return False, f"Kelly 不穩定: CV = {stability['kelly']:.2f} > {max_kelly_cv}"
    
    return True, f"適合使用 Kelly (Full Kelly = {stats.kelly_pct:.1%})"


def quick_kelly_check(symbol: str, data_path: Path, cfg: dict) -> str:
    """
    快速 Kelly 檢查（一行總結）
    """
    from .run_backtest import run_symbol_backtest
    
    result = run_symbol_backtest(symbol, data_path, cfg)
    
    # 從 Portfolio 物件提取交易紀錄
    pf = result.get("pf")
    if pf is not None:
        trades = extract_trades_from_portfolio(pf)
    else:
        trades = result.get("trades", [])
    
    suitable, reason = is_strategy_suitable_for_kelly(trades)
    
    stats = calculate_kelly_stats(trades)
    
    if suitable:
        return f"✅ {symbol}: 適合 Kelly (推薦 Quarter={stats.kelly_pct*0.25:.1%}) - {reason}"
    else:
        return f"❌ {symbol}: 不適合 Kelly - {reason}"
