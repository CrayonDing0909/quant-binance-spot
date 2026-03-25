"""
結構化交易資料庫 — SQLite

取代 JSON 散落存儲，提供可查詢、可分析的交易記錄。
SQLite 是 Serverless 資料庫，不需要額外伺服器進程，
對系統資源幾乎零負擔，非常適合 Oracle Cloud Free Tier。

Schema:
    trades      — 每筆交易（開倉/平倉/加減倉）
    signals     — 每次策略信號快照（含指標值）
    daily_equity — 每日權益快照（用於績效圖表）

使用方式:
    db = TradingDatabase("reports/futures/rsi_adx_atr/live/trading.db")
    db.log_trade(symbol="BTCUSDT", side="BUY", ...)
    db.log_signal(symbol="BTCUSDT", signal=1.0, ...)

    # 查詢
    trades = db.get_trades(symbol="BTCUSDT", limit=50)
    equity = db.get_daily_equity(days=30)
    stats = db.get_performance_summary()
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from ..utils.log import get_logger

logger = get_logger("trading_db")


class TradingDatabase:
    """
    SQLite 交易資料庫

    特性:
    - 自動建表（首次使用自動初始化 schema）
    - 線程安全（每次操作獨立 connection，或使用 WAL mode）
    - 輕量（單一 .db 檔案，零外部依賴）
    - 防腐蝕（WAL mode + 原子寫入）
    """

    SCHEMA_VERSION = 2

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """取得資料庫連線"""
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.row_factory = sqlite3.Row  # 讓查詢結果可用欄位名存取
        conn.execute("PRAGMA journal_mode=WAL")  # 寫前日誌，防止資料損壞
        conn.execute("PRAGMA busy_timeout=5000")  # 忙碌時等待 5 秒
        return conn

    def _init_db(self) -> None:
        """初始化資料庫 schema"""
        conn = self._get_conn()
        try:
            conn.executescript("""
                -- 交易記錄表
                CREATE TABLE IF NOT EXISTS trades (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp       TEXT    NOT NULL,           -- ISO 8601 UTC
                    symbol          TEXT    NOT NULL,
                    side            TEXT    NOT NULL,           -- BUY / SELL
                    position_side   TEXT    DEFAULT '',         -- LONG / SHORT / BOTH
                    qty             REAL    NOT NULL,
                    price           REAL    NOT NULL,
                    value           REAL    NOT NULL,           -- qty * price
                    fee             REAL    DEFAULT 0,
                    fee_rate        REAL    DEFAULT 0,          -- Maker/Taker 費率
                    pnl             REAL,                       -- 平倉盈虧 (NULL = 開倉)
                    reason          TEXT    DEFAULT '',
                    order_type      TEXT    DEFAULT 'MARKET',   -- LIMIT / MARKET / LIMIT+MARKET
                    order_id_hash   TEXT    DEFAULT ''          -- 脫敏的 orderId
                );

                -- 信號快照表
                CREATE TABLE IF NOT EXISTS signals (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp       TEXT    NOT NULL,           -- ISO 8601 UTC
                    symbol          TEXT    NOT NULL,
                    signal_value    REAL    NOT NULL,           -- [-1, 1]
                    price           REAL    NOT NULL,
                    rsi             REAL,
                    adx             REAL,
                    atr             REAL,
                    plus_di         REAL,
                    minus_di        REAL,
                    funding_rate    REAL,
                    target_pct      REAL,                       -- 目標倉位 %
                    current_pct     REAL,                       -- 當前倉位 %
                    action          TEXT    DEFAULT 'HOLD'      -- OPEN_LONG / OPEN_SHORT / CLOSE / HOLD
                );

                -- 每日權益快照表
                CREATE TABLE IF NOT EXISTS daily_equity (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    date            TEXT    NOT NULL UNIQUE,    -- YYYY-MM-DD
                    equity          REAL    NOT NULL,
                    cash            REAL    DEFAULT 0,
                    unrealized_pnl  REAL    DEFAULT 0,
                    pnl_day         REAL    DEFAULT 0,          -- 當日盈虧
                    cumulative_pnl  REAL    DEFAULT 0,
                    drawdown_pct    REAL    DEFAULT 0,
                    trade_count     INTEGER DEFAULT 0,          -- 當日交易筆數
                    position_count  INTEGER DEFAULT 0           -- 持倉數量
                );

                -- 索引（加速查詢）
                CREATE INDEX IF NOT EXISTS idx_trades_symbol_ts ON trades(symbol, timestamp);
                CREATE INDEX IF NOT EXISTS idx_trades_ts ON trades(timestamp);
                CREATE INDEX IF NOT EXISTS idx_signals_symbol_ts ON signals(symbol, timestamp);
                CREATE INDEX IF NOT EXISTS idx_signals_ts ON signals(timestamp);
                CREATE INDEX IF NOT EXISTS idx_equity_date ON daily_equity(date);

                -- Schema 版本追蹤
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER NOT NULL
                );
            """)

            # 設定 schema 版本
            cur = conn.execute("SELECT COUNT(*) FROM schema_version")
            if cur.fetchone()[0] == 0:
                conn.execute(
                    "INSERT INTO schema_version (version) VALUES (?)",
                    (self.SCHEMA_VERSION,),
                )

            # ── Schema migrations ──
            cur = conn.execute("SELECT version FROM schema_version LIMIT 1")
            current_version = cur.fetchone()[0]

            if current_version < 2:
                # v2: Add sleeve tracking columns
                try:
                    conn.execute("ALTER TABLE signals ADD COLUMN sleeve_signals TEXT DEFAULT ''")
                except sqlite3.OperationalError:
                    pass  # Column already exists
                try:
                    conn.execute("ALTER TABLE trades ADD COLUMN sleeve TEXT DEFAULT ''")
                except sqlite3.OperationalError:
                    pass  # Column already exists
                conn.execute(
                    "UPDATE schema_version SET version = ?", (2,)
                )
                logger.info("📦 DB migrated to schema v2 (sleeve tracking)")

            conn.commit()
            logger.info(f"📦 交易資料庫已就緒: {self.db_path}")
        except Exception as e:
            logger.error(f"❌ 初始化資料庫失敗: {e}")
            raise
        finally:
            conn.close()

    # ══════════════════════════════════════════════════════════════
    # 寫入操作
    # ══════════════════════════════════════════════════════════════

    def log_trade(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        fee: float = 0.0,
        fee_rate: float = 0.0,
        pnl: Optional[float] = None,
        reason: str = "",
        order_type: str = "MARKET",
        order_id_hash: str = "",
        position_side: str = "",
        timestamp: Optional[str] = None,
        sleeve: str = "",
    ) -> int:
        """
        記錄一筆交易

        Args:
            sleeve: dominant sub-strategy name (e.g. "tsmom_carry_v2")

        Returns:
            新增記錄的 id
        """
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        value = qty * price

        conn = self._get_conn()
        try:
            cur = conn.execute(
                """
                INSERT INTO trades
                    (timestamp, symbol, side, position_side, qty, price, value,
                     fee, fee_rate, pnl, reason, order_type, order_id_hash, sleeve)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (ts, symbol, side, position_side, qty, price, value,
                 fee, fee_rate, pnl, reason, order_type, order_id_hash, sleeve),
            )
            conn.commit()
            trade_id = cur.lastrowid
            logger.debug(
                f"📝 DB: trade #{trade_id} {side} {symbol} "
                f"{qty:.6f} @ ${price:,.2f} [{order_type}] sleeve={sleeve}"
            )
            return trade_id
        finally:
            conn.close()

    def log_signal(
        self,
        symbol: str,
        signal_value: float,
        price: float,
        rsi: Optional[float] = None,
        adx: Optional[float] = None,
        atr: Optional[float] = None,
        plus_di: Optional[float] = None,
        minus_di: Optional[float] = None,
        funding_rate: Optional[float] = None,
        target_pct: Optional[float] = None,
        current_pct: Optional[float] = None,
        action: str = "HOLD",
        timestamp: Optional[str] = None,
        sleeve_signals: str = "",
    ) -> int:
        """
        記錄一次信號快照

        Args:
            sleeve_signals: JSON string of per-sub-strategy signals
                e.g. '{"tsmom_carry_v2": 0.45, "lsr_contrarian": 0.0}'

        Returns:
            新增記錄的 id
        """
        ts = timestamp or datetime.now(timezone.utc).isoformat()

        conn = self._get_conn()
        try:
            cur = conn.execute(
                """
                INSERT INTO signals
                    (timestamp, symbol, signal_value, price, rsi, adx, atr,
                     plus_di, minus_di, funding_rate, target_pct, current_pct,
                     action, sleeve_signals)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (ts, symbol, signal_value, price, rsi, adx, atr,
                 plus_di, minus_di, funding_rate, target_pct, current_pct,
                 action, sleeve_signals),
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()

    def log_daily_equity(
        self,
        equity: float,
        cash: float = 0.0,
        unrealized_pnl: float = 0.0,
        pnl_day: float = 0.0,
        cumulative_pnl: float = 0.0,
        drawdown_pct: float = 0.0,
        trade_count: int = 0,
        position_count: int = 0,
        date: Optional[str] = None,
    ) -> None:
        """
        記錄每日權益快照（UPSERT: 同一天重複寫入則更新）
        """
        d = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")

        conn = self._get_conn()
        try:
            conn.execute(
                """
                INSERT INTO daily_equity
                    (date, equity, cash, unrealized_pnl, pnl_day,
                     cumulative_pnl, drawdown_pct, trade_count, position_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(date) DO UPDATE SET
                    equity=excluded.equity,
                    cash=excluded.cash,
                    unrealized_pnl=excluded.unrealized_pnl,
                    pnl_day=excluded.pnl_day,
                    cumulative_pnl=excluded.cumulative_pnl,
                    drawdown_pct=excluded.drawdown_pct,
                    trade_count=excluded.trade_count,
                    position_count=excluded.position_count
                """,
                (d, equity, cash, unrealized_pnl, pnl_day,
                 cumulative_pnl, drawdown_pct, trade_count, position_count),
            )
            conn.commit()
        finally:
            conn.close()

    # ══════════════════════════════════════════════════════════════
    # 查詢操作
    # ══════════════════════════════════════════════════════════════

    def get_trades(
        self,
        symbol: Optional[str] = None,
        days: Optional[int] = None,
        limit: int = 100,
    ) -> list[dict]:
        """
        查詢交易記錄

        Args:
            symbol: 過濾特定幣種（None = 全部）
            days: 過濾最近 N 天
            limit: 最大筆數

        Returns:
            交易記錄列表（最新在前）
        """
        conn = self._get_conn()
        try:
            conditions = []
            params = []

            if symbol:
                conditions.append("symbol = ?")
                params.append(symbol)
            if days:
                since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
                conditions.append("timestamp >= ?")
                params.append(since)

            where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
            params.append(limit)

            rows = conn.execute(
                f"SELECT * FROM trades {where} ORDER BY timestamp DESC LIMIT ?",
                params,
            ).fetchall()

            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_signals(
        self,
        symbol: Optional[str] = None,
        days: Optional[int] = None,
        limit: int = 100,
    ) -> list[dict]:
        """查詢信號記錄"""
        conn = self._get_conn()
        try:
            conditions = []
            params = []

            if symbol:
                conditions.append("symbol = ?")
                params.append(symbol)
            if days:
                since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
                conditions.append("timestamp >= ?")
                params.append(since)

            where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
            params.append(limit)

            rows = conn.execute(
                f"SELECT * FROM signals {where} ORDER BY timestamp DESC LIMIT ?",
                params,
            ).fetchall()

            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_daily_equity(self, days: int = 30) -> list[dict]:
        """查詢每日權益"""
        conn = self._get_conn()
        try:
            since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
            rows = conn.execute(
                "SELECT * FROM daily_equity WHERE date >= ? ORDER BY date",
                (since,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_performance_summary(self, days: Optional[int] = None) -> dict:
        """
        績效總覽

        Returns:
            {
                "total_trades": int,
                "winning_trades": int,
                "losing_trades": int,
                "win_rate": float,
                "total_pnl": float,
                "avg_pnl": float,
                "total_fees": float,
                "total_fee_savings": float,  # Maker 省下的手續費
                "maker_pct": float,          # Maker 成交比例
                "best_trade": float,
                "worst_trade": float,
                "avg_holding_trades_per_day": float,
            }
        """
        conn = self._get_conn()
        try:
            conditions = []
            params = []

            if days:
                since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
                conditions.append("timestamp >= ?")
                params.append(since)

            where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

            # 基本統計
            row = conn.execute(
                f"""
                SELECT
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
                    COALESCE(SUM(pnl), 0) as total_pnl,
                    COALESCE(AVG(pnl), 0) as avg_pnl,
                    COALESCE(SUM(fee), 0) as total_fees,
                    COALESCE(MAX(pnl), 0) as best_trade,
                    COALESCE(MIN(pnl), 0) as worst_trade
                FROM trades
                {where}
                """,
                params,
            ).fetchone()

            total = row["total_trades"] or 0
            winning = row["winning_trades"] or 0
            losing = row["losing_trades"] or 0
            decided = winning + losing

            # Maker / Taker 統計
            maker_row = conn.execute(
                f"""
                SELECT
                    COUNT(*) as maker_count,
                    COALESCE(SUM(value), 0) as maker_volume
                FROM trades
                {where}
                {"AND" if where else "WHERE"} order_type LIKE '%LIMIT%'
                """,
                params,
            ).fetchone()

            total_volume_row = conn.execute(
                f"SELECT COALESCE(SUM(value), 0) as total_volume FROM trades {where}",
                params,
            ).fetchone()

            maker_volume = maker_row["maker_volume"] or 0
            total_volume = total_volume_row["total_volume"] or 0
            maker_pct = maker_volume / total_volume if total_volume > 0 else 0

            # 估算 Maker 省下的手續費
            # savings = maker_volume × (taker_rate - maker_rate)
            fee_savings = maker_volume * (0.0004 - 0.0002)

            return {
                "total_trades": total,
                "winning_trades": winning,
                "losing_trades": losing,
                "win_rate": winning / decided if decided > 0 else 0.0,
                "total_pnl": row["total_pnl"],
                "avg_pnl": row["avg_pnl"],
                "total_fees": row["total_fees"],
                "total_fee_savings": fee_savings,
                "maker_pct": maker_pct,
                "best_trade": row["best_trade"],
                "worst_trade": row["worst_trade"],
            }
        finally:
            conn.close()

    def get_trade_count(self, symbol: Optional[str] = None) -> int:
        """快速取得交易筆數"""
        conn = self._get_conn()
        try:
            if symbol:
                row = conn.execute(
                    "SELECT COUNT(*) as cnt FROM trades WHERE symbol = ?",
                    (symbol,),
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) as cnt FROM trades").fetchone()
            return row["cnt"]
        finally:
            conn.close()

    # ══════════════════════════════════════════════════════════════
    # 工具
    # ══════════════════════════════════════════════════════════════

    def export_trades_csv(self, output_path: Path | str) -> int:
        """
        匯出交易記錄為 CSV

        Returns:
            匯出的筆數
        """
        import csv

        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM trades ORDER BY timestamp"
            ).fetchall()

            if not rows:
                return 0

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(rows[0].keys())  # header
                for row in rows:
                    writer.writerow(tuple(row))

            logger.info(f"📤 匯出 {len(rows)} 筆交易到 {output_path}")
            return len(rows)
        finally:
            conn.close()

    def compact(self, keep_days: int = 365) -> int:
        """
        清理舊資料（保留最近 N 天）

        Returns:
            刪除的筆數
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=keep_days)).isoformat()

        conn = self._get_conn()
        try:
            # 刪除舊信號（信號量最多，最需要清理）
            cur = conn.execute(
                "DELETE FROM signals WHERE timestamp < ?", (cutoff,)
            )
            deleted_signals = cur.rowcount

            # trades 和 daily_equity 保留更久（通常不需要清理）
            conn.commit()
            conn.execute("VACUUM")  # 回收空間

            if deleted_signals > 0:
                logger.info(f"🧹 清理了 {deleted_signals} 筆舊信號記錄")
            return deleted_signals
        finally:
            conn.close()
