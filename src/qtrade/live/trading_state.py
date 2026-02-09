"""
Real Trading 狀態持久化

安全設計原則：
1. 不存 API Keys
2. Order ID 用 hash 脫敏
3. 用比例而非絕對金額（可選）
4. 可選加密
5. 啟動時與交易所比對驗證

使用場景：
- 斷線恢復：從本地狀態檔恢復持仓資訊，與交易所比對
- 交易紀錄：記錄交易歷史用於分析（脫敏）
- 績效追蹤：追蹤累積 PnL、最大回撤等
- 一致性驗證：提供交易紀錄給 consistency_validator 使用

使用方法：
    # 初始化
    state_manager = TradingStateManager(
        state_path=Path("reports/live/my_strategy/real_state.json"),
        encrypt=False,  # 生產環境建議 True
    )
    
    # 啟動時驗證
    discrepancies = state_manager.verify_against_exchange(broker)
    if discrepancies:
        logger.warning(f"狀態不一致: {discrepancies}")
    
    # 記錄交易
    state_manager.log_trade(order_result, pnl=100.0)
    
    # 更新持仓
    state_manager.update_position("BTCUSDT", qty=0.1, avg_entry=50000)
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any
import base64

from ..utils.log import get_logger

logger = get_logger("trading_state")


# ══════════════════════════════════════════════════════════════════════════════
# 資料結構
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TradeLog:
    """
    交易紀錄（脫敏設計）
    
    不存敏感資訊：
    - order_id 用 hash
    - 不存 API 相關資訊
    """
    timestamp: str  # ISO format
    symbol: str
    side: str  # "BUY" / "SELL"
    qty: float
    price: float
    value: float  # qty * price
    fee: float
    pnl: Optional[float]  # 賣出時的 PnL
    reason: str
    order_hash: str = ""  # Order ID 的 hash（用於追蹤但不可逆）
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "TradeLog":
        return cls(**data)


@dataclass
class PositionState:
    """持仓狀態"""
    symbol: str
    qty: float
    avg_entry: float
    last_updated: str  # ISO format
    
    @property
    def is_open(self) -> bool:
        return self.qty > 1e-10
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "PositionState":
        return cls(**data)


@dataclass
class TradingState:
    """
    交易狀態（完整）
    
    安全設計：
    - 不存 API Keys
    - Order ID 用 hash
    - 可選：用比例而非絕對金額
    """
    # 版本資訊
    version: str = "1.0"
    
    # 基本資訊
    strategy_name: str = ""
    mode: str = "real"  # "paper" / "real"
    symbols: List[str] = field(default_factory=list)
    interval: str = "1h"
    
    # 持仓狀態
    positions: Dict[str, Dict] = field(default_factory=dict)
    
    # 交易紀錄
    trades: List[Dict] = field(default_factory=list)
    
    # 績效追蹤
    initial_equity: float = 0.0  # 初始權益（用於計算比例）
    cumulative_pnl: float = 0.0
    cumulative_pnl_pct: float = 0.0  # 相對初始權益的百分比
    peak_equity: float = 0.0
    current_drawdown_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    
    # 運行統計
    started_at: str = ""
    last_updated: str = ""
    total_ticks: int = 0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    
    # 元數據（用於診斷）
    last_error: Optional[str] = None
    restart_count: int = 0
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "TradingState":
        # 處理舊版本的遷移
        data.pop("__version__", None)
        
        # 確保所有必要欄位存在
        defaults = cls()
        for key in asdict(defaults).keys():
            if key not in data:
                data[key] = getattr(defaults, key)
        
        return cls(**data)


# ══════════════════════════════════════════════════════════════════════════════
# 狀態管理器
# ══════════════════════════════════════════════════════════════════════════════

class TradingStateManager:
    """
    交易狀態管理器
    
    安全特性：
    1. 不存敏感資訊
    2. 支援 state 檔案加密（可選）
    3. 斷線後可從交易所 API 恢復真實狀態並比對
    4. 自動備份
    
    使用範例：
        manager = TradingStateManager(
            state_path=Path("reports/live/rsi_adx_atr/real_state.json"),
            strategy_name="rsi_adx_atr",
            symbols=["BTCUSDT", "ETHUSDT"],
        )
        
        # 啟動時驗證
        discrepancies = manager.verify_against_exchange(broker)
        
        # 記錄交易
        manager.log_trade(order_result)
        
        # 更新持仓
        manager.update_position("BTCUSDT", 0.1, 50000)
    """
    
    # 加密用的環境變數名稱
    ENCRYPTION_KEY_ENV = "TRADING_STATE_ENCRYPTION_KEY"
    
    def __init__(
        self,
        state_path: Path,
        strategy_name: str = "",
        symbols: Optional[List[str]] = None,
        interval: str = "1h",
        mode: str = "real",
        encrypt: bool = False,
        auto_backup: bool = True,
        max_trade_history: int = 10000,  # 最多保留的交易紀錄數
    ):
        """
        Args:
            state_path: 狀態檔路徑
            strategy_name: 策略名稱
            symbols: 交易對列表
            interval: K 線週期
            mode: "paper" 或 "real"
            encrypt: 是否加密儲存
            auto_backup: 是否自動備份
            max_trade_history: 最多保留的交易紀錄數量
        """
        self.state_path = Path(state_path)
        self.encrypt = encrypt
        self.auto_backup = auto_backup
        self.max_trade_history = max_trade_history
        
        # 初始化或載入狀態
        if self.state_path.exists():
            self._load()
            self.state.restart_count += 1
            logger.info(
                f"📂 載入交易狀態: {len(self.state.trades)} 筆交易, "
                f"{len([p for p in self.state.positions.values() if p.get('qty', 0) > 1e-10])} 個持仓"
            )
        else:
            self.state = TradingState(
                strategy_name=strategy_name,
                symbols=symbols or [],
                interval=interval,
                mode=mode,
                started_at=datetime.now(timezone.utc).isoformat(),
            )
            logger.info("📂 建立新的交易狀態檔")
    
    # ══════════════════════════════════════════════════════════════════════════
    # 公開介面
    # ══════════════════════════════════════════════════════════════════════════
    
    def update_position(
        self,
        symbol: str,
        qty: float,
        avg_entry: float,
    ) -> None:
        """更新持仓狀態"""
        if qty > 1e-10:
            self.state.positions[symbol] = {
                "symbol": symbol,
                "qty": qty,
                "avg_entry": avg_entry,
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
        else:
            self.state.positions.pop(symbol, None)
        
        self._save()
    
    def log_trade(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        fee: float = 0.0,
        pnl: Optional[float] = None,
        reason: str = "",
        order_id: str = "",
    ) -> TradeLog:
        """
        記錄交易（脫敏）
        
        Args:
            symbol: 交易對
            side: "BUY" / "SELL"
            qty: 數量
            price: 成交價
            fee: 手續費
            pnl: 盈虧（賣出時）
            reason: 交易原因
            order_id: 訂單 ID（會被 hash）
            
        Returns:
            TradeLog 交易紀錄
        """
        trade = TradeLog(
            timestamp=datetime.now(timezone.utc).isoformat(),
            symbol=symbol,
            side=side,
            qty=qty,
            price=price,
            value=qty * price,
            fee=fee,
            pnl=pnl,
            reason=reason,
            order_hash=self._hash_order_id(order_id) if order_id else "",
        )
        
        self.state.trades.append(trade.to_dict())
        self.state.total_trades += 1
        
        # 更新勝率統計
        if pnl is not None:
            self.state.cumulative_pnl += pnl
            if self.state.initial_equity > 0:
                self.state.cumulative_pnl_pct = self.state.cumulative_pnl / self.state.initial_equity * 100
            
            if pnl > 0:
                self.state.winning_trades += 1
            elif pnl < 0:
                self.state.losing_trades += 1
        
        # 限制交易紀錄數量
        if len(self.state.trades) > self.max_trade_history:
            # 保留最近的紀錄
            self.state.trades = self.state.trades[-self.max_trade_history:]
        
        self._save()
        
        return trade
    
    def log_trade_from_order(
        self,
        order_result: Any,
        pnl: Optional[float] = None,
    ) -> TradeLog:
        """
        從 OrderResult 記錄交易
        
        Args:
            order_result: BinanceSpotBroker.OrderResult 或類似物件
            pnl: 盈虧
        """
        return self.log_trade(
            symbol=order_result.symbol,
            side=order_result.side,
            qty=order_result.qty,
            price=order_result.price,
            fee=getattr(order_result, "fee", 0.0),
            pnl=pnl,
            reason=getattr(order_result, "reason", ""),
            order_id=getattr(order_result, "order_id", ""),
        )
    
    def update_equity(self, current_equity: float) -> None:
        """
        更新權益和回撤
        
        Args:
            current_equity: 當前總權益
        """
        if self.state.initial_equity <= 0:
            self.state.initial_equity = current_equity
        
        # 更新峰值
        if current_equity > self.state.peak_equity:
            self.state.peak_equity = current_equity
        
        # 計算當前回撤
        if self.state.peak_equity > 0:
            self.state.current_drawdown_pct = (
                (self.state.peak_equity - current_equity) / self.state.peak_equity * 100
            )
            self.state.max_drawdown_pct = max(
                self.state.max_drawdown_pct,
                self.state.current_drawdown_pct,
            )
        
        self._save()
    
    def increment_tick(self) -> None:
        """增加 tick 計數"""
        self.state.total_ticks += 1
        self._save()
    
    def log_error(self, error_msg: str) -> None:
        """記錄錯誤"""
        self.state.last_error = f"{datetime.now(timezone.utc).isoformat()}: {error_msg}"
        self._save()
    
    def verify_against_exchange(
        self,
        broker: Any,
    ) -> Dict[str, Dict]:
        """
        與交易所狀態比對，檢測不一致
        
        斷線恢復時呼叫，確保本地狀態與交易所同步
        
        Args:
            broker: BinanceSpotBroker 實例
            
        Returns:
            不一致的持仓 {symbol: {"local": qty, "exchange": qty, "diff": qty}}
        """
        discrepancies = {}
        
        for symbol, local_pos in self.state.positions.items():
            local_qty = local_pos.get("qty", 0)
            
            try:
                exchange_qty = broker.get_position(symbol)
            except Exception as e:
                logger.warning(f"⚠️  無法查詢 {symbol} 交易所持仓: {e}")
                continue
            
            diff = exchange_qty - local_qty
            
            if abs(diff) > 1e-6:
                discrepancies[symbol] = {
                    "local": local_qty,
                    "exchange": exchange_qty,
                    "diff": diff,
                }
                logger.warning(
                    f"⚠️  {symbol} 持仓不一致: "
                    f"本地={local_qty:.6f}, 交易所={exchange_qty:.6f}, 差異={diff:+.6f}"
                )
        
        # 檢查交易所有但本地沒有的持仓
        for symbol in self.state.symbols:
            if symbol not in self.state.positions:
                try:
                    exchange_qty = broker.get_position(symbol)
                    if exchange_qty > 1e-6:
                        discrepancies[symbol] = {
                            "local": 0,
                            "exchange": exchange_qty,
                            "diff": exchange_qty,
                        }
                        logger.warning(
                            f"⚠️  {symbol} 本地無紀錄但交易所有持仓: {exchange_qty:.6f}"
                        )
                except Exception:
                    pass
        
        if not discrepancies:
            logger.info("✅ 本地狀態與交易所一致")
        
        return discrepancies
    
    def sync_from_exchange(self, broker: Any) -> None:
        """
        從交易所同步狀態
        
        用於修復不一致或首次啟動
        """
        for symbol in self.state.symbols:
            try:
                qty = broker.get_position(symbol)
                if qty > 1e-6:
                    price = broker.get_price(symbol)
                    self.update_position(symbol, qty, price)
                    logger.info(f"📥 同步 {symbol}: {qty:.6f} @ {price:.2f}")
                else:
                    self.state.positions.pop(symbol, None)
            except Exception as e:
                logger.warning(f"⚠️  無法同步 {symbol}: {e}")
        
        self._save()
    
    def get_trade_stats(self) -> Dict[str, float]:
        """
        計算交易統計（用於 Kelly 計算）
        
        Returns:
            {"win_rate": float, "avg_win": float, "avg_loss": float}
        """
        if not self.state.trades:
            return {"win_rate": 0.5, "avg_win": 1.0, "avg_loss": 1.0}
        
        wins = []
        losses = []
        
        for t in self.state.trades:
            pnl = t.get("pnl")
            if pnl is not None:
                if pnl > 0:
                    wins.append(pnl)
                elif pnl < 0:
                    losses.append(abs(pnl))
        
        total = len(wins) + len(losses)
        if total == 0:
            return {"win_rate": 0.5, "avg_win": 1.0, "avg_loss": 1.0}
        
        return {
            "win_rate": len(wins) / total,
            "avg_win": sum(wins) / len(wins) if wins else 1.0,
            "avg_loss": sum(losses) / len(losses) if losses else 1.0,
        }
    
    def summary(self) -> str:
        """產生狀態摘要"""
        stats = self.get_trade_stats()
        
        lines = [
            "=" * 50,
            f"  交易狀態摘要 [{self.state.mode.upper()}]",
            "=" * 50,
            f"  策略: {self.state.strategy_name}",
            f"  啟動: {self.state.started_at}",
            f"  更新: {self.state.last_updated}",
            f"  重啟: {self.state.restart_count} 次",
            "-" * 50,
            f"  總交易: {self.state.total_trades} 筆",
            f"  勝率: {stats['win_rate']:.1%}",
            f"  累積 PnL: ${self.state.cumulative_pnl:,.2f} ({self.state.cumulative_pnl_pct:+.2f}%)",
            f"  最大回撤: {self.state.max_drawdown_pct:.2f}%",
        ]
        
        # 持仓
        open_positions = [
            (s, p) for s, p in self.state.positions.items()
            if p.get("qty", 0) > 1e-10
        ]
        if open_positions:
            lines.append("-" * 50)
            lines.append("  持仓:")
            for symbol, pos in open_positions:
                lines.append(f"    {symbol}: {pos['qty']:.6f} @ ${pos['avg_entry']:,.2f}")
        
        lines.append("=" * 50)
        return "\n".join(lines)
    
    # ══════════════════════════════════════════════════════════════════════════
    # 內部方法
    # ══════════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def _hash_order_id(order_id: str) -> str:
        """
        Order ID hash（不可逆）
        
        用於追蹤但不暴露真實 order_id
        """
        if not order_id:
            return ""
        return hashlib.sha256(order_id.encode()).hexdigest()[:16]
    
    def _save(self) -> None:
        """儲存狀態到檔案"""
        self.state.last_updated = datetime.now(timezone.utc).isoformat()
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 自動備份
        if self.auto_backup and self.state_path.exists():
            backup_path = self.state_path.with_suffix(".json.bak")
            try:
                import shutil
                shutil.copy2(self.state_path, backup_path)
            except Exception as e:
                logger.warning(f"⚠️  備份失敗: {e}")
        
        content = json.dumps(self.state.to_dict(), indent=2, ensure_ascii=False)
        
        if self.encrypt:
            content = self._encrypt(content)
        
        with open(self.state_path, "w", encoding="utf-8") as f:
            f.write(content)
    
    def _load(self) -> None:
        """從檔案載入狀態"""
        with open(self.state_path, encoding="utf-8") as f:
            content = f.read()
        
        if self.encrypt:
            content = self._decrypt(content)
        
        data = json.loads(content)
        self.state = TradingState.from_dict(data)
    
    def _encrypt(self, content: str) -> str:
        """
        簡單加密（XOR + base64）
        
        注意：這不是強加密，只是防止明文曝露。
        生產環境建議使用更強的加密（如 Fernet）。
        """
        key = os.getenv(self.ENCRYPTION_KEY_ENV, "default_key_change_me")
        
        # 簡單 XOR 加密
        key_bytes = key.encode()
        content_bytes = content.encode()
        encrypted = bytes(
            c ^ key_bytes[i % len(key_bytes)]
            for i, c in enumerate(content_bytes)
        )
        
        return base64.b64encode(encrypted).decode()
    
    def _decrypt(self, content: str) -> str:
        """解密"""
        key = os.getenv(self.ENCRYPTION_KEY_ENV, "default_key_change_me")
        
        encrypted = base64.b64decode(content)
        key_bytes = key.encode()
        decrypted = bytes(
            c ^ key_bytes[i % len(key_bytes)]
            for i, c in enumerate(encrypted)
        )
        
        return decrypted.decode()


# ══════════════════════════════════════════════════════════════════════════════
# 便利函數
# ══════════════════════════════════════════════════════════════════════════════

def get_state_manager(
    strategy_name: str,
    mode: str = "real",
    symbols: Optional[List[str]] = None,
    base_dir: str = "reports/live",
) -> TradingStateManager:
    """
    取得或建立狀態管理器（便利函數）
    
    Args:
        strategy_name: 策略名稱
        mode: "paper" 或 "real"
        symbols: 交易對列表
        base_dir: 基礎目錄
        
    Returns:
        TradingStateManager
    """
    state_path = Path(base_dir) / strategy_name / f"{mode}_state.json"
    
    return TradingStateManager(
        state_path=state_path,
        strategy_name=strategy_name,
        symbols=symbols,
        mode=mode,
        encrypt=(mode == "real"),  # real 模式建議加密
    )
