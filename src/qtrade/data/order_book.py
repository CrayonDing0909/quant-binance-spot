"""
Order Book Depth 數據模組（Phase 4C）

提供 WebSocket depth stream 的訂閱與指標計算：
  - bid/ask imbalance: (bid_vol - ask_vol) / (bid_vol + ask_vol)
  - spread: (best_ask - best_bid) / mid_price
  - depth at levels: 各層級的掛單量

狀態：
  - 此模組在 Alpha Researcher 驗證 order book depth 有 alpha 後才啟用
  - 目前為基礎設施佔位（infrastructure placeholder）

使用方式（未來）：
    from qtrade.data.order_book import OrderBookSnapshot, compute_imbalance

Note:
    Order book depth stream 在 Oracle Cloud (1GB RAM) 環境下需謹慎使用：
    - 每個 symbol 的 depth stream 約產生 100+ msg/s
    - 建議僅訂閱 @depth5 或 @depth10（而非 @depth@100ms）
    - 記憶體估算：8 symbols × depth10 ≈ 額外 ~5MB
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class OrderBookSnapshot:
    """
    Order Book 快照

    Attributes:
        symbol: 交易對
        timestamp_ms: 時間戳（毫秒）
        bids: list of [price, qty] pairs, sorted by price desc
        asks: list of [price, qty] pairs, sorted by price asc
    """
    symbol: str
    timestamp_ms: int
    bids: list[list[float]] = field(default_factory=list)
    asks: list[list[float]] = field(default_factory=list)

    @property
    def best_bid(self) -> float:
        return self.bids[0][0] if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[0][0] if self.asks else 0.0

    @property
    def mid_price(self) -> float:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return 0.0

    @property
    def spread(self) -> float:
        """Spread as fraction of mid price."""
        mid = self.mid_price
        if mid > 0:
            return (self.best_ask - self.best_bid) / mid
        return 0.0


def compute_imbalance(
    snapshot: OrderBookSnapshot,
    levels: int = 5,
) -> float:
    """
    計算 bid/ask imbalance

    imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol)
    range: [-1, 1]
      +1 = 完全買方掛單
      -1 = 完全賣方掛單
       0 = 均衡

    Args:
        snapshot: Order book 快照
        levels: 使用前 N 層計算

    Returns:
        float: imbalance [-1, 1]
    """
    bid_vol = sum(b[1] for b in snapshot.bids[:levels])
    ask_vol = sum(a[1] for a in snapshot.asks[:levels])
    total = bid_vol + ask_vol

    if total <= 0:
        return 0.0

    return (bid_vol - ask_vol) / total


def compute_depth_profile(
    snapshot: OrderBookSnapshot,
    levels: int = 10,
) -> dict[str, float]:
    """
    計算 depth profile 指標

    Returns:
        dict with:
          - imbalance: bid/ask volume imbalance
          - spread_bps: spread in basis points
          - bid_depth: total bid volume (in base asset)
          - ask_depth: total ask volume (in base asset)
          - depth_ratio: bid_depth / ask_depth
          - bid_wall: 最大單層 bid 量
          - ask_wall: 最大單層 ask 量
    """
    bids = snapshot.bids[:levels]
    asks = snapshot.asks[:levels]

    bid_vol = sum(b[1] for b in bids) if bids else 0.0
    ask_vol = sum(a[1] for a in asks) if asks else 0.0
    total = bid_vol + ask_vol

    mid = snapshot.mid_price

    return {
        "imbalance": (bid_vol - ask_vol) / total if total > 0 else 0.0,
        "spread_bps": snapshot.spread * 10000,
        "bid_depth": bid_vol,
        "ask_depth": ask_vol,
        "depth_ratio": bid_vol / ask_vol if ask_vol > 0 else 0.0,
        "bid_wall": max((b[1] for b in bids), default=0.0),
        "ask_wall": max((a[1] for a in asks), default=0.0),
    }


class OrderBookCache:
    """
    Order Book 記憶體快取（per-symbol）

    保留最近 N 個快照用於短期分析（如 imbalance 趨勢）。

    ⚠️ 記憶體敏感：Oracle Cloud 1GB RAM 下建議 max_snapshots ≤ 60。
    """

    def __init__(self, max_snapshots: int = 60):
        self._cache: dict[str, list[OrderBookSnapshot]] = {}
        self._max = max_snapshots

    def update(self, snapshot: OrderBookSnapshot) -> None:
        """追加新快照"""
        sym = snapshot.symbol
        if sym not in self._cache:
            self._cache[sym] = []
        self._cache[sym].append(snapshot)
        # 超過上限時丟棄最舊的
        if len(self._cache[sym]) > self._max:
            self._cache[sym] = self._cache[sym][-self._max:]

    def get_latest(self, symbol: str) -> Optional[OrderBookSnapshot]:
        """取得最新快照"""
        snaps = self._cache.get(symbol)
        if snaps:
            return snaps[-1]
        return None

    def get_imbalance_series(self, symbol: str, levels: int = 5) -> pd.Series:
        """
        取得某 symbol 的 imbalance 時間序列

        Returns:
            pd.Series indexed by timestamp
        """
        snaps = self._cache.get(symbol, [])
        if not snaps:
            return pd.Series(dtype=float, name="imbalance")

        data = [
            (pd.Timestamp(s.timestamp_ms, unit="ms", tz="UTC"), compute_imbalance(s, levels))
            for s in snaps
        ]
        ts, vals = zip(*data)
        return pd.Series(vals, index=pd.DatetimeIndex(ts), name="imbalance")

    def clear(self, symbol: str | None = None) -> None:
        """清空快取"""
        if symbol:
            self._cache.pop(symbol, None)
        else:
            self._cache.clear()


def parse_depth_message(msg: dict) -> Optional[OrderBookSnapshot]:
    """
    從 Binance WebSocket depth 消息解析 OrderBookSnapshot

    Expected format (from @depth5 or @depth10 stream):
        {
            "e": "depthUpdate",
            "s": "BTCUSDT",
            "T": 1234567890123,
            "b": [["60000.00", "1.5"], ...],
            "a": [["60001.00", "0.8"], ...],
        }
    Or partial depth (@depth5/@depth10/@depth20):
        {
            "lastUpdateId": 123,
            "bids": [["60000.00", "1.5"], ...],
            "asks": [["60001.00", "0.8"], ...],
        }
    """
    try:
        # Full depth update
        if "e" in msg and msg.get("e") == "depthUpdate":
            symbol = msg.get("s", "")
            ts = msg.get("T", 0)
            bids = [[float(p), float(q)] for p, q in msg.get("b", [])]
            asks = [[float(p), float(q)] for p, q in msg.get("a", [])]
            return OrderBookSnapshot(symbol=symbol, timestamp_ms=ts, bids=bids, asks=asks)

        # Partial depth (from @depth5/@depth10 subscription)
        if "bids" in msg and "asks" in msg:
            bids = [[float(p), float(q)] for p, q in msg.get("bids", [])]
            asks = [[float(p), float(q)] for p, q in msg.get("asks", [])]
            return OrderBookSnapshot(
                symbol=msg.get("s", "UNKNOWN"),
                timestamp_ms=int(msg.get("E", 0)),
                bids=bids,
                asks=asks,
            )

    except (KeyError, ValueError, TypeError) as e:
        logger.debug(f"Order book parse error: {e}")

    return None
