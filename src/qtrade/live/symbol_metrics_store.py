"""
Symbol Metrics Store — 治理所需的 per-symbol 4 週指標

提供統一的資料模型 (SymbolMetrics) 與讀取介面。
指標可來自：
  - 實盤交易 DB (trading.db) 的已實現 PnL / 交易記錄
  - 外部注入 (dry-run / backtest 測試)
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional

import math


@dataclass
class SymbolMetrics:
    """
    某 symbol 在 4 週滾動窗口的治理指標

    所有欄位定義見 R3C_SYMBOL_GOVERNANCE_SPEC.md §4-5
    """
    symbol: str

    # 原始輸入
    net_pnl: float = 0.0               # 扣費後淨 PnL (USDT)
    turnover: float = 0.0              # 總成交額
    returns_series: List[float] = field(default_factory=list)  # 逐 bar 淨報酬率
    max_drawdown_pct: float = 0.0       # 滾動最大回撤 %
    realized_slippage_bps: float = 0.0  # 實際滑點 (bps)
    model_slippage_bps: float = 3.0     # 模型預設滑點 (bps)
    signal_execution_consistency_pct: float = 100.0  # 信號-執行一致率 %
    missed_signals_pct: float = 0.0     # 漏信號率 %
    trade_count: int = 0                # 交易筆數

    # 計算指標（呼叫 compute() 後填入）
    edge_sharpe_4w: float = 0.0
    edge_per_turnover_4w: float = 0.0
    slippage_ratio_4w: float = 0.0
    consistency_4w: float = 100.0
    missed_4w: float = 0.0
    dd_4w: float = 0.0

    # 元資料
    window_start: Optional[str] = None
    window_end: Optional[str] = None

    def compute(self) -> "SymbolMetrics":
        """根據原始輸入計算衍生指標，回傳 self 方便 chaining。"""
        eps = 1e-12

        # Sharpe on 4-week net returns
        if len(self.returns_series) >= 2:
            mean_ret = sum(self.returns_series) / len(self.returns_series)
            var = sum((r - mean_ret) ** 2 for r in self.returns_series) / (
                len(self.returns_series) - 1
            )
            std = math.sqrt(var) if var > 0 else eps
            # 年化 Sharpe (假設 1h bar → 8760 bars/year)
            self.edge_sharpe_4w = (mean_ret / std) * math.sqrt(8760)
        else:
            self.edge_sharpe_4w = 0.0

        # Edge per turnover
        self.edge_per_turnover_4w = self.net_pnl / max(self.turnover, eps)

        # Slippage ratio
        self.slippage_ratio_4w = self.realized_slippage_bps / max(
            self.model_slippage_bps, eps
        )

        # Pass-through
        self.consistency_4w = self.signal_execution_consistency_pct
        self.missed_4w = self.missed_signals_pct
        self.dd_4w = self.max_drawdown_pct

        return self

    def to_dict(self) -> dict:
        """序列化為 JSON-friendly dict。"""
        d = asdict(self)
        # returns_series 可能很長，不放進 artifact 概覽
        d.pop("returns_series", None)
        return d


def build_metrics_from_dict(raw: dict) -> SymbolMetrics:
    """從 JSON/dict 重建 SymbolMetrics（反序列化）。"""
    m = SymbolMetrics(
        symbol=raw["symbol"],
        net_pnl=raw.get("net_pnl", 0.0),
        turnover=raw.get("turnover", 0.0),
        returns_series=raw.get("returns_series", []),
        max_drawdown_pct=raw.get("max_drawdown_pct", 0.0),
        realized_slippage_bps=raw.get("realized_slippage_bps", 0.0),
        model_slippage_bps=raw.get("model_slippage_bps", 3.0),
        signal_execution_consistency_pct=raw.get("signal_execution_consistency_pct", 100.0),
        missed_signals_pct=raw.get("missed_signals_pct", 0.0),
        trade_count=raw.get("trade_count", 0),
        edge_sharpe_4w=raw.get("edge_sharpe_4w", 0.0),
        edge_per_turnover_4w=raw.get("edge_per_turnover_4w", 0.0),
        slippage_ratio_4w=raw.get("slippage_ratio_4w", 0.0),
        consistency_4w=raw.get("consistency_4w", 100.0),
        missed_4w=raw.get("missed_4w", 0.0),
        dd_4w=raw.get("dd_4w", 0.0),
        window_start=raw.get("window_start"),
        window_end=raw.get("window_end"),
    )
    return m
