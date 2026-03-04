"""
Base Runner — 交易執行器共享邏輯基類

LiveRunner（Polling）和 WebSocketRunner（Event-driven）共用：
  - 倉位計算器 (fixed / kelly / volatility)
  - Drawdown 熔斷
  - SL/TP 冷卻 + 孤兒掛單清理
  - SL/TP 補掛（含 Adaptive SL）
  - 方向錯誤 TP 偵測
  - 防不必要重平衡
  - 方向切換確認
  - 信號狀態持久化
  - SQLite 結構化記錄
  - 定期帳戶摘要
"""
from __future__ import annotations

import json
import time
import traceback
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from ..config import AppConfig
from ..monitor.notifier import TelegramNotifier
from ..risk.position_sizing import (
    PositionSizer,
    FixedPositionSizer,
    KellyPositionSizer,
    VolatilityPositionSizer,
)
from .paper_broker import PaperBroker
from .kline_cache import IncrementalKlineCache
from .signal_generator import SignalResult, PositionInfo
from ..utils.log import get_logger

logger = get_logger("base_runner")

# Live 模式輔助資料最大保留行數（節省記憶體）
# OI 策略 lookback ~500h，加上安全餘量，2000 行綽綽有餘
_LIVE_MAX_AUX_BARS = 2000


class BaseRunner(ABC):
    """
    交易執行器基類

    子類只需實現:
      - run()  → 主迴圈（Polling / WebSocket）
    """

    def __init__(
        self,
        cfg: AppConfig,
        broker,
        mode: str = "paper",
        notifier: TelegramNotifier | None = None,
    ):
        self.cfg = cfg
        self.broker = broker
        self.mode = mode
        self.strategy_name = cfg.strategy.name
        self.symbols = cfg.market.symbols
        self.interval = cfg.market.interval
        self.market_type = cfg.market_type_str
        self.is_running = False
        self.trade_count = 0
        self.start_time: float | None = None

        # Ensemble 路由：per-symbol 策略名與參數（從 YAML ensemble.strategies 載入）
        self._ensemble_strategies: dict[str, dict] = {}
        self._load_ensemble_strategies()

        # Telegram
        self.notifier = notifier or TelegramNotifier.from_config(cfg.notification)

        # 多幣種倉位分配權重
        self._weights: dict[str, float] = {}
        n = len(self.symbols)
        for sym in self.symbols:
            self._weights[sym] = cfg.portfolio.get_weight(sym, n)

        # Symbol Governance — 套用治理權重（僅在 enabled 時生效）
        self._apply_governance_to_weights()

        # Drawdown 熔斷
        self.max_drawdown_pct = cfg.risk.max_drawdown_pct if cfg.risk else None
        self._circuit_breaker_triggered = False
        self._initial_equity: float | None = None

        # 倉位計算器
        self.position_sizer: Optional[PositionSizer] = None
        self._init_position_sizer()

        # SQLite 結構化資料庫
        self.trading_db = None
        try:
            from .trading_db import TradingDatabase
            db_path = cfg.get_report_dir("live") / "trading.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self.trading_db = TradingDatabase(db_path)
            self._log.info(f"📦 SQLite 資料庫已就緒: {db_path}")
        except Exception as e:
            self._log.warning(f"⚠️  SQLite 資料庫初始化失敗（不影響交易）: {e}")

        # 信號狀態持久化
        self._signal_state_path = cfg.get_report_dir("live") / "signal_state.json"
        self._signal_state: dict[str, float] = self._load_signal_state()

        # K 線快取（子類在自己的 __init__ 中設定）
        self._kline_cache: IncrementalKlineCache | None = None

        # OI 記憶體快取（避免每次信號都重讀 parquet）
        self._oi_cache: dict[str, pd.Series] = {}
        self._oi_cache_ts: float = 0.0
        self._oi_refresh_interval_s: float = 1800.0  # 每 30 分鐘刷新一次
        self._init_oi_cache()

        # 衍生品記憶體快取（LSR, Taker Vol, CVD — Phase 4A）
        self._derivatives_cache: dict[str, dict[str, pd.Series]] = {}  # {symbol: {metric: Series}}
        self._derivatives_cache_ts: float = 0.0
        self._derivatives_refresh_interval_s: float = 3600.0  # 每 60 分鐘刷新一次
        self._init_derivatives_cache()

        # Rebalance Band 計數器（監控用）
        self._band_skip_count: int = 0
        self._band_skip_notional_est: float = 0.0

    @property
    def _log(self):
        """子類可覆寫以使用專用 logger"""
        return logger

    # ══════════════════════════════════════════════════════════
    #  Symbol Governance — 權重覆寫
    # ══════════════════════════════════════════════════════════

    def _apply_governance_to_weights(self) -> None:
        """
        套用治理層的 effective weights（若啟用且有 artifact）。

        - disabled → 不做任何事（backward compatible）
        - enabled + 有 artifact → 覆寫 self._weights
        - enabled + 無 artifact / 讀取失敗 → 保留 base weights + warning
        """
        gov_cfg = self.cfg.live.symbol_governance

        if not gov_cfg.enabled:
            self._log.info("📋 Symbol governance: disabled — 使用原始 base weights")
            return

        self._log.info("📋 Symbol governance: enabled — 嘗試讀取 artifact")

        try:
            from .symbol_governance import apply_governance_weights

            effective = apply_governance_weights(
                self._weights, gov_cfg, gov_cfg.artifacts_dir,
            )

            # apply_governance_weights returns base_weights copy when no artifact
            # Detect that case by reference equality with the dict content
            if effective == self._weights:
                self._log.warning(
                    "⚠️  Symbol governance: 未找到 artifact 或內容無變化 — 使用 base weights"
                )
                return

            self._weights = effective

            # Log summary
            summary_parts = []
            for sym in sorted(effective):
                summary_parts.append(f"{sym}={effective[sym]:.4f}")
            self._log.info(
                f"✅ Symbol governance: effective weights 已套用 — "
                f"{', '.join(summary_parts)}"
            )

        except Exception as e:
            self._log.warning(
                f"⚠️  Symbol governance: 讀取 artifact 失敗 ({e}) — 回退 base weights"
            )

    # ══════════════════════════════════════════════════════════
    #  Ensemble 路由
    # ══════════════════════════════════════════════════════════

    def _load_ensemble_strategies(self) -> None:
        """
        從 config YAML 的 ensemble.strategies 載入 per-symbol 策略路由。

        若 ensemble.enabled=true 且有 strategies map，
        則 _get_strategy_for_symbol() 會回傳 symbol 專屬策略名與參數，
        否則 fallback 到全域 strategy.name + strategy.params。
        """
        try:
            import yaml
            cfg_path = getattr(self.cfg, '_config_path', None)
            if cfg_path is None:
                return
            with open(cfg_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)
            ens = raw.get("ensemble")
            if ens and ens.get("enabled", False):
                strategies = ens.get("strategies", {})
                if strategies:
                    self._ensemble_strategies = strategies
                    routing = ", ".join(
                        f"{s}→{v['name']}" for s, v in strategies.items()
                    )
                    self._log.info(f"🧩 Ensemble 模式啟用: {routing}")
        except Exception as e:
            self._log.debug(f"Ensemble 配置載入失敗（使用全域策略）: {e}")

    def _get_strategy_for_symbol(self, symbol: str) -> tuple[str, dict]:
        """
        取得指定 symbol 的策略名稱和參數。

        優先使用 ensemble.strategies 路由，否則 fallback 到全域策略。
        自動注入 _data_dir，讓策略可在 live 模式自動載入輔助數據（OI, FR 等）。

        Returns:
            (strategy_name, params)
        """
        if self._ensemble_strategies and symbol in self._ensemble_strategies:
            sym_cfg = self._ensemble_strategies[symbol]
            name = sym_cfg["name"]
            params = dict(sym_cfg.get("params", {}))
        else:
            name = self.strategy_name
            params = self.cfg.strategy.get_params(symbol)

        # 自動注入 _data_dir（讓策略可以在 live 模式自動載入 OI/FR 等輔助數據）
        if "_data_dir" not in params:
            params["_data_dir"] = str(self.cfg.data_dir)

        # 注入 OI 記憶體快取（避免策略每次從磁碟讀取）
        if symbol in self._oi_cache and "_oi_series" not in params:
            self._maybe_refresh_oi_cache()
            if symbol in self._oi_cache:
                params["_oi_series"] = self._oi_cache[symbol]

        # 注入衍生品數據快取（LSR, CVD, taker vol, liquidation）
        if symbol in self._derivatives_cache and "_derivatives_data" not in params:
            self._maybe_refresh_derivatives_cache()
            if symbol in self._derivatives_cache:
                params["_derivatives_data"] = self._derivatives_cache[symbol]

        return name, params

    # ══════════════════════════════════════════════════════════
    #  OI 記憶體快取
    # ══════════════════════════════════════════════════════════

    def _needs_oi(self) -> bool:
        """判斷當前策略是否需要 OI 數據。"""
        oi_strategies = {"oi_liq_bounce", "oi_bb_rv"}
        if self.strategy_name in oi_strategies:
            return True
        for sym_cfg in self._ensemble_strategies.values():
            if sym_cfg.get("name") in oi_strategies:
                return True
        return False

    def _init_oi_cache(self) -> None:
        """啟動時載入 OI 到記憶體（僅需要 OI 的策略）。"""
        if not self._needs_oi():
            return
        self._load_oi_from_disk()

    def _load_oi_from_disk(self) -> None:
        """從 parquet 載入所有 symbol 的 OI 到記憶體快取（僅保留最近 N 行）。"""
        try:
            from ..data.open_interest import get_oi_path, load_open_interest, OI_PROVIDER_SEARCH_ORDER
        except ImportError:
            self._log.warning("⚠️  open_interest 模組不可用，OI 快取停用")
            return

        data_dir = self.cfg.data_dir
        loaded = 0
        for sym in self.symbols:
            for prov in OI_PROVIDER_SEARCH_ORDER:
                oi_path = get_oi_path(data_dir, sym, prov)
                oi_df = load_open_interest(oi_path)
                if oi_df is not None and not oi_df.empty:
                    # 只保留 sumOpenInterest 欄位作為 Series
                    if "sumOpenInterest" in oi_df.columns:
                        series = oi_df["sumOpenInterest"]
                    else:
                        # fallback: 取第一個數值欄
                        num_cols = oi_df.select_dtypes(include="number").columns
                        if len(num_cols) > 0:
                            series = oi_df[num_cols[0]]
                        else:
                            continue
                    # 裁剪：live 只需最近 N 行，節省記憶體
                    if _LIVE_MAX_AUX_BARS > 0 and len(series) > _LIVE_MAX_AUX_BARS:
                        series = series.iloc[-_LIVE_MAX_AUX_BARS:]
                    self._oi_cache[sym] = series
                    loaded += 1
                    break

        self._oi_cache_ts = time.time()
        if loaded > 0:
            self._log.info(f"📊 OI 記憶體快取已載入: {loaded}/{len(self.symbols)} symbols")
        else:
            self._log.warning("⚠️  OI 記憶體快取: 無任何 symbol 載入成功")

    def _maybe_refresh_oi_cache(self) -> None:
        """若快取過期，從磁碟重新載入 OI 數據。"""
        if not self._needs_oi():
            return
        now = time.time()
        if now - self._oi_cache_ts >= self._oi_refresh_interval_s:
            self._log.debug("🔄 OI 記憶體快取刷新中...")
            self._load_oi_from_disk()

    # ══════════════════════════════════════════════════════════
    #  衍生品記憶體快取 (LSR, Taker Vol, CVD — Phase 4A)
    # ══════════════════════════════════════════════════════════

    def _needs_derivatives(self) -> bool:
        """判斷當前策略是否需要衍生品數據。"""
        deriv_strategies = {
            "crowding_contrarian", "cvd_divergence", "liq_cascade_v2",
        }
        if self.strategy_name in deriv_strategies:
            return True
        for sym_cfg in self._ensemble_strategies.values():
            if sym_cfg.get("name") in deriv_strategies:
                return True
        # 檢查是否有 derivatives_enhanced overlay 配置
        overlay_cfg = getattr(self.cfg, '_overlay_cfg', None)
        if overlay_cfg and overlay_cfg.get("mode") == "derivatives_micro":
            return True
        return False

    def _init_derivatives_cache(self) -> None:
        """啟動時載入衍生品數據到記憶體，並啟動後台 API 輪詢線程。"""
        if not self._needs_derivatives():
            return
        self._load_derivatives_from_disk()
        # Phase 4B: 後台線程定期從 API 更新磁碟數據
        # （注意：is_running 需在 run() 開始後才為 True，
        #   所以 bg thread 的啟動延遲到子類呼叫 _start_derivatives_bg_refresh）

    @staticmethod
    def _trim_series(series: pd.Series, max_bars: int = _LIVE_MAX_AUX_BARS) -> pd.Series:
        """裁剪 Series 到最近 max_bars 行（節省記憶體）。"""
        if max_bars > 0 and len(series) > max_bars:
            return series.iloc[-max_bars:]
        return series

    def _load_derivatives_from_disk(self) -> None:
        """從 parquet 載入所有 symbol 的衍生品數據到記憶體快取（僅保留最近 N 行）。"""
        data_dir = self.cfg.data_dir
        loaded = 0

        for sym in self.symbols:
            sym_deriv: dict[str, pd.Series] = {}

            # LSR
            try:
                from ..data.long_short_ratio import load_lsr
                for lsr_type in ["lsr", "top_lsr_account"]:
                    series = load_lsr(sym, lsr_type, data_dir=data_dir / "binance" / "futures" / "derivatives" / lsr_type)
                    if series is not None and not series.empty:
                        sym_deriv[lsr_type] = self._trim_series(series)
            except Exception as e:
                self._log.debug(f"  {sym}: LSR 載入失敗: {e}")

            # Taker Vol
            try:
                from ..data.taker_volume import load_taker_volume
                tv = load_taker_volume(sym, data_dir=data_dir / "binance" / "futures" / "derivatives" / "taker_vol_ratio")
                if tv is not None and not tv.empty:
                    sym_deriv["taker_vol_ratio"] = self._trim_series(tv)
            except Exception as e:
                self._log.debug(f"  {sym}: Taker Vol 載入失敗: {e}")

            # CVD
            try:
                from ..data.taker_volume import load_cvd
                cvd = load_cvd(sym, data_dir=data_dir / "binance" / "futures" / "derivatives" / "cvd")
                if cvd is not None and not cvd.empty:
                    sym_deriv["cvd"] = self._trim_series(cvd)
            except Exception as e:
                self._log.debug(f"  {sym}: CVD 載入失敗: {e}")

            # Liquidation
            try:
                from ..data.liquidation import load_liquidation
                liq_df = load_liquidation(sym, data_dir=data_dir / "binance" / "futures" / "liquidation")
                if liq_df is not None and not liq_df.empty:
                    for col in ["liq_cascade_z", "liq_imbalance", "liq_total"]:
                        if col in liq_df.columns:
                            sym_deriv[col] = self._trim_series(liq_df[col])
            except Exception as e:
                self._log.debug(f"  {sym}: Liquidation 載入失敗: {e}")

            if sym_deriv:
                self._derivatives_cache[sym] = sym_deriv
                loaded += 1

        self._derivatives_cache_ts = time.time()
        if loaded > 0:
            total_metrics = sum(len(v) for v in self._derivatives_cache.values())
            self._log.info(
                f"📊 衍生品記憶體快取已載入: {loaded}/{len(self.symbols)} symbols, "
                f"{total_metrics} metrics total"
            )
        else:
            self._log.info("ℹ️  衍生品記憶體快取: 無數據可載入（可能尚未下載）")

    def _maybe_refresh_derivatives_cache(self) -> None:
        """若快取過期，從磁碟重新載入衍生品數據。"""
        if not self._needs_derivatives():
            return
        now = time.time()
        if now - self._derivatives_cache_ts >= self._derivatives_refresh_interval_s:
            self._log.debug("🔄 衍生品記憶體快取刷新中...")
            self._load_derivatives_from_disk()

    # ── Phase 4B: 衍生品 API 後台輪詢線程 ──

    def _start_derivatives_bg_refresh(self) -> None:
        """
        啟動後台線程，定期從 Binance API 拉取衍生品數據並寫入 parquet。

        磁碟快取（_load_derivatives_from_disk）會在下次信號生成時刷新到記憶體。
        此線程只負責「磁碟 ← API」的部分。
        """
        import threading

        if not self._needs_derivatives():
            return

        interval_sec = self._derivatives_refresh_interval_s  # 預設 1800s = 30min

        def _poll_loop():
            while self.is_running:
                try:
                    self._poll_derivatives_api()
                except Exception as e:
                    self._log.warning(f"⚠️ 衍生品 API 輪詢失敗: {e}")
                # 等待下一輪（每 10 秒檢查一次 is_running）
                waited = 0.0
                while waited < interval_sec and self.is_running:
                    import time as _time
                    _time.sleep(10)
                    waited += 10

        t = threading.Thread(target=_poll_loop, daemon=True, name="derivatives_bg")
        t.start()
        self._log.info(f"🔄 衍生品 API 後台輪詢已啟動（每 {interval_sec:.0f}s）")

    def _poll_derivatives_api(self) -> None:
        """
        從 Binance API 輪詢最新 LSR + Taker Vol 並寫入 parquet。

        使用 data modules 的 download + save 函數。
        liquidation 和 CVD 依賴非 API 數據源，這裡不輪詢。
        """
        data_dir = self.cfg.data_dir
        deriv_dir = data_dir / "binance" / "futures" / "derivatives"

        for sym in self.symbols:
            # LSR — api provider (最近 ~30 天)
            try:
                from ..data.long_short_ratio import download_lsr, save_lsr
                for lsr_type in ["lsr", "top_lsr_account"]:
                    series = download_lsr(sym, lsr_type=lsr_type, provider="api")
                    if series is not None and not series.empty:
                        save_lsr(series, sym, lsr_type=lsr_type, data_dir=deriv_dir)
            except Exception:
                pass  # debug-level, don't spam logs

            # Taker Vol — api provider
            try:
                from ..data.taker_volume import download_taker_volume, save_taker_volume, compute_cvd, save_cvd
                tv = download_taker_volume(sym, provider="api")
                if tv is not None and not tv.empty:
                    save_taker_volume(tv, sym, data_dir=deriv_dir)
                    cvd = compute_cvd(tv)
                    save_cvd(cvd, sym, data_dir=deriv_dir)
            except Exception:
                pass

        self._log.debug(f"📡 衍生品 API 輪詢完成 ({len(self.symbols)} symbols)")

    # ══════════════════════════════════════════════════════════
    #  倉位計算器
    # ══════════════════════════════════════════════════════════

    def _init_position_sizer(self) -> None:
        """根據配置初始化倉位計算器"""
        ps_cfg = self.cfg.position_sizing

        if ps_cfg.method == "kelly":
            stats = self._get_trade_stats()
            total_trades = stats.get("total_trades", 0)
            min_trades = getattr(ps_cfg, "min_trades_for_kelly", 30)

            if total_trades < min_trades:
                self._log.info(
                    f"📊 倉位計算: 交易數 ({total_trades}) < 最小要求 ({min_trades})，暫用固定倉位"
                )
                self.position_sizer = FixedPositionSizer(ps_cfg.position_pct)
            else:
                try:
                    win_rate = getattr(ps_cfg, "win_rate", None) or stats.get("win_rate", 0.5)
                    avg_win = getattr(ps_cfg, "avg_win", None) or stats.get("avg_win", 1.0)
                    avg_loss = getattr(ps_cfg, "avg_loss", None) or stats.get("avg_loss", 1.0)
                    self.position_sizer = KellyPositionSizer(
                        win_rate=win_rate,
                        avg_win=avg_win,
                        avg_loss=avg_loss,
                        kelly_fraction=ps_cfg.kelly_fraction,
                    )
                    self._log.info(
                        f"📊 倉位計算: Kelly (fraction={ps_cfg.kelly_fraction}, "
                        f"kelly_pct={self.position_sizer.kelly_pct:.1%})"
                    )
                except ValueError as e:
                    self._log.warning(f"⚠️  Kelly 參數無效: {e}，改用固定倉位")
                    self.position_sizer = FixedPositionSizer(ps_cfg.position_pct)

        elif ps_cfg.method == "volatility":
            self.position_sizer = VolatilityPositionSizer(
                base_position_pct=ps_cfg.position_pct,
                target_volatility=ps_cfg.target_volatility,
                lookback=ps_cfg.vol_lookback,
                interval=self.interval,
            )
            self._log.info(f"📊 倉位計算: 波動率目標 ({ps_cfg.target_volatility:.1%})")

        else:
            self.position_sizer = FixedPositionSizer(ps_cfg.position_pct)
            self._log.info(f"📊 倉位計算: 固定 ({ps_cfg.position_pct:.0%})")

    def _get_trade_stats(self) -> dict:
        """從 TradingDB 或 PaperBroker 取得交易統計（Kelly 用）"""
        if self.trading_db:
            try:
                summary = self.trading_db.get_performance_summary()
                return {
                    "win_rate": summary.get("win_rate", 0.5),
                    "avg_win": summary.get("avg_win_pnl", 1.0),
                    "avg_loss": abs(summary.get("avg_loss_pnl", 1.0)),
                    "total_trades": summary.get("total_trades", 0),
                }
            except Exception:
                pass

        if isinstance(self.broker, PaperBroker):
            trades = self.broker.account.trades
            if not trades:
                return {"win_rate": 0.5, "avg_win": 1.0, "avg_loss": 1.0, "total_trades": 0}
            wins = [t for t in trades if t.pnl and t.pnl > 0]
            losses = [t for t in trades if t.pnl and t.pnl < 0]
            total = len(wins) + len(losses)
            return {
                "win_rate": len(wins) / total if total > 0 else 0.5,
                "avg_win": sum(t.pnl for t in wins) / len(wins) if wins else 1.0,
                "avg_loss": abs(sum(t.pnl for t in losses) / len(losses)) if losses else 1.0,
                "total_trades": len(trades),
            }

        return {"win_rate": 0.5, "avg_win": 1.0, "avg_loss": 1.0, "total_trades": 0}

    def _get_equity(self) -> float | None:
        """取得當前權益（Paper / Real 通用）"""
        try:
            if isinstance(self.broker, PaperBroker):
                prices = {}
                for sym in self.symbols:
                    p = self._get_price(sym)
                    if p and p > 0:
                        prices[sym] = p
                return self.broker.get_equity(prices) if prices else None
            elif hasattr(self.broker, "get_equity"):
                return self.broker.get_equity()
        except Exception as e:
            self._log.warning(f"⚠️ 取得權益失敗（影響 position sizing）: {e}")
        return None

    def _get_price(self, symbol: str) -> float | None:
        """從 K 線快取或 Broker 取得當前價格"""
        if self._kline_cache is not None:
            df = self._kline_cache.get_cached(symbol)
            if df is not None and len(df) > 0:
                return float(df["close"].iloc[-1])
        if hasattr(self.broker, "get_price"):
            try:
                return self.broker.get_price(symbol)
            except Exception as e:
                self._log.debug(f"  {symbol}: 從 broker 取得價格失敗: {e}")
        return None

    def _apply_position_sizing(self, raw_signal: float, price: float, symbol: str) -> float:
        """應用倉位計算器調整信號"""
        if self.position_sizer is None:
            return raw_signal

        try:
            if isinstance(self.broker, PaperBroker):
                prices = {}
                for sym in self.symbols:
                    p = self._get_price(sym)
                    if p and p > 0:
                        prices[sym] = p
                equity = self.broker.get_equity(prices)
            elif hasattr(self.broker, "get_equity"):
                try:
                    equity = self.broker.get_equity()
                except TypeError:
                    equity = self.broker.get_equity([symbol])
            else:
                equity = 10000

            returns = None
            if isinstance(self.position_sizer, VolatilityPositionSizer):
                if self._kline_cache is not None:
                    df = self._kline_cache.get_cached(symbol)
                    if df is not None and len(df) > self.position_sizer.lookback:
                        returns = df["close"].pct_change()

            position_size = self.position_sizer.calculate_size(
                signal=raw_signal, equity=equity, price=price, returns=returns,
            )
            position_value = position_size * price
            adjusted_signal = position_value / equity if equity > 0 else raw_signal
            return max(-1.0, min(1.0, adjusted_signal))
        except Exception as e:
            # ⚠️ Position sizing 失敗 → 回退到 raw_signal（滿倉）
            # 這可能導致過度曝險，必須有可見性
            self._log.error(
                f"❌ {symbol}: Position sizing 計算失敗，回退到 raw_signal={raw_signal:.2f}: {e}"
            )
            # 一次性 TG 告警（避免每個 tick 都發）
            if not getattr(self, "_ps_error_notified", False):
                self._ps_error_notified = True
                if self.notifier:
                    self.notifier.send_error(
                        f"⚠️ Position sizing 異常！{symbol}: {e}\n"
                        f"已回退到 raw_signal={raw_signal:.2f}，請檢查"
                    )
            return raw_signal

    # ══════════════════════════════════════════════════════════
    #  信號狀態持久化
    # ══════════════════════════════════════════════════════════

    def _load_signal_state(self) -> dict[str, float]:
        try:
            if self._signal_state_path.exists():
                with open(self._signal_state_path) as f:
                    data = json.load(f)
                return data.get("signals", {})
        except Exception:
            pass
        return {}

    def _save_signal_state(self, signal_map: dict[str, float]) -> None:
        try:
            self._signal_state_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "signals": signal_map,
            }
            with open(self._signal_state_path, "w") as f:
                json.dump(payload, f, indent=2)
        except Exception:
            pass

    # ══════════════════════════════════════════════════════════
    #  安全機制
    # ══════════════════════════════════════════════════════════

    def _check_circuit_breaker(self) -> bool:
        """Drawdown 熔斷檢查"""
        if self._circuit_breaker_triggered:
            return True
        if not self.max_drawdown_pct:
            return False

        try:
            equity = self._get_equity()
            if equity is None or equity <= 0:
                return False

            if self._initial_equity is None:
                if isinstance(self.broker, PaperBroker):
                    self._initial_equity = self.broker.account.initial_cash
                else:
                    self._initial_equity = equity
                self._log.info(f"📊 熔斷基準權益: ${self._initial_equity:,.2f}")
                return False

            drawdown = 1.0 - (equity / self._initial_equity)

            if drawdown >= self.max_drawdown_pct:
                self._circuit_breaker_triggered = True
                self._log.warning(
                    f"🚨🚨🚨 CIRCUIT BREAKER 觸發！"
                    f"Drawdown={drawdown:.1%} >= {self.max_drawdown_pct:.0%} "
                    f"(權益 ${equity:,.2f} / 基準 ${self._initial_equity:,.2f})"
                )
                for sym in self.symbols:
                    try:
                        p = self._get_price(sym) or 0.0
                        if p <= 0:
                            continue
                        pct = self.broker.get_position_pct(sym, p)
                        if abs(pct) > 0.01:
                            self.broker.execute_target_position(
                                symbol=sym, target_pct=0.0,
                                current_price=p, reason="CIRCUIT_BREAKER",
                            )
                            self._log.warning(f"  🔴 強制平倉 {sym}")
                    except Exception as e:
                        self._log.error(f"  ❌ 強制平倉 {sym} 失敗: {e}")

                self.notifier.send_error(
                    f"🚨 <b>CIRCUIT BREAKER 熔斷觸發!</b>\n\n"
                    f"  Drawdown: <b>{drawdown:.1%}</b> (閾值 {self.max_drawdown_pct:.0%})\n"
                    f"  ⚠️ 已強制平倉所有持倉"
                )
                return True

            if drawdown >= self.max_drawdown_pct * 0.8:
                self._log.warning(f"⚠️  Drawdown 預警: {drawdown:.1%}")

        except Exception as e:
            self._log.debug(f"熔斷檢查失敗: {e}")
        return False

    def _check_sl_tp_cooldown(
        self, symbol: str, current_pct: float, target_pct: float,
    ) -> bool:
        """
        SL/TP 冷卻檢查 + 孤兒掛單清理

        Returns: True = 應跳過本次開倉（冷卻中）
        """
        if not (
            abs(current_pct) < 0.01
            and abs(target_pct) > 0.02
            and not isinstance(self.broker, PaperBroker)
            and hasattr(self.broker, "get_open_orders")
            and hasattr(self.broker, "get_trade_history")
        ):
            return False

        try:
            if hasattr(self.broker, "get_all_conditional_orders"):
                cond_orders = self.broker.get_all_conditional_orders(symbol)
            else:
                cond_orders = self.broker.get_open_orders(symbol)
            sl_tp_types = {"STOP_MARKET", "TAKE_PROFIT_MARKET", "STOP", "TAKE_PROFIT"}
            has_sl_tp = any(o.get("type") in sl_tp_types for o in cond_orders)

            if has_sl_tp:
                orphan_detail = [
                    f"{o.get('type')}[{o.get('positionSide', '?')}] "
                    f"@ ${float(o.get('stopPrice', 0) or o.get('triggerPrice', 0) or 0):,.2f}"
                    for o in cond_orders if o.get("type") in sl_tp_types
                ]
                self._log.warning(
                    f"🧹 {symbol}: 無持倉但有殘留掛單 {orphan_detail} → 取消孤兒 SL/TP"
                )
                if hasattr(self.broker, "cancel_all_open_orders"):
                    self.broker.cancel_all_open_orders(symbol)
                else:
                    self.broker.cancel_stop_loss(symbol)
                    self.broker.cancel_take_profit(symbol)
                if hasattr(self.broker, "_remove_algo_cache"):
                    self.broker._remove_algo_cache(symbol)
                has_sl_tp = False

            if not has_sl_tp:
                recent_trades = self.broker.get_trade_history(symbol=symbol, limit=5)
                now_ms = int(time.time() * 1000)
                cooldown_ms = 10 * 60 * 1000

                recently_closed = any(
                    now_ms - t.get("time", 0) < cooldown_ms
                    for t in (recent_trades or [])
                )
                if recently_closed:
                    self._log.warning(
                        f"⚠️  {symbol}: 無持倉且無 SL/TP，但最近 10min 有成交 → "
                        f"疑似 SL/TP 觸發，跳過本次開倉（冷卻等下根 bar）"
                    )
                    return True
        except Exception as e:
            self._log.debug(f"  {symbol}: SL/TP 冷卻檢查失敗: {e}（繼續正常流程）")
        return False

    def _calculate_sl_tp_prices(
        self,
        symbol: str,
        price: float,
        target_pct: float,
        params: dict,
        indicators: dict,
    ) -> tuple[float | None, float | None]:
        """計算 SL/TP 價格（含 Adaptive SL）"""
        stop_loss_atr = params.get("stop_loss_atr")
        take_profit_atr = params.get("take_profit_atr")
        atr_value = indicators.get("atr")

        if not (atr_value and target_pct != 0):
            return None, None

        sl_mult = float(stop_loss_atr) if stop_loss_atr else None
        if sl_mult and params.get("adaptive_sl", False):
            er_value = indicators.get("er")
            if er_value is not None:
                from ..strategy.exit_rules import compute_adaptive_sl_multiplier
                sl_mult = compute_adaptive_sl_multiplier(
                    er_value, sl_mult,
                    er_sl_min=float(params.get("er_sl_min", 1.5)),
                    er_sl_max=float(params.get("er_sl_max", 3.0)),
                )
                self._log.info(
                    f"🔧 {symbol}: Adaptive SL: ER={er_value:.3f} → SL={sl_mult:.2f}x ATR"
                )

        stop_loss_price = None
        take_profit_price = None
        if target_pct > 0:
            if sl_mult:
                stop_loss_price = price - sl_mult * float(atr_value)
            if take_profit_atr:
                take_profit_price = price + float(take_profit_atr) * float(atr_value)
        elif target_pct < 0:
            if sl_mult:
                stop_loss_price = price + sl_mult * float(atr_value)
            if take_profit_atr:
                take_profit_price = price - float(take_profit_atr) * float(atr_value)

        if stop_loss_price or take_profit_price:
            pos_side = "LONG" if target_pct > 0 else "SHORT"
            sl_str = f"${stop_loss_price:,.2f}" if stop_loss_price else "N/A"
            tp_str = f"${take_profit_price:,.2f}" if take_profit_price else "N/A"
            self._log.info(f"🛡️  {symbol} [{pos_side}] SL={sl_str}, TP={tp_str}")

        return stop_loss_price, take_profit_price

    def _ensure_sl_tp(self, symbol: str, sig: SignalResult, params: dict, actual_pct: float):
        """SL/TP 補掛機制（含 Adaptive SL + 方向錯誤 TP 偵測）"""
        if isinstance(self.broker, PaperBroker):
            return
        if abs(actual_pct) <= 0.01:
            return
        if not hasattr(self.broker, "place_stop_loss"):
            return
        if not hasattr(self.broker, "get_open_orders"):
            return

        stop_loss_atr = params.get("stop_loss_atr")
        take_profit_atr = params.get("take_profit_atr")
        atr_value = sig.indicators.get("atr")
        price = sig.price

        if not ((stop_loss_atr or take_profit_atr) and atr_value):
            return

        try:
            if hasattr(self.broker, "get_all_conditional_orders"):
                cond_orders = self.broker.get_all_conditional_orders(symbol)
            else:
                cond_orders = self.broker.get_open_orders(symbol)

            position_side = "LONG" if actual_pct > 0 else "SHORT"

            def _match_side(o: dict) -> bool:
                o_ps = o.get("positionSide", "")
                return not o_ps or o_ps == position_side or o_ps == "BOTH"

            has_sl = any(
                o.get("type") in {"STOP_MARKET", "STOP"} and _match_side(o)
                for o in cond_orders
            )
            has_tp = any(
                o.get("type") in {"TAKE_PROFIT_MARKET", "TAKE_PROFIT"} and _match_side(o)
                for o in cond_orders
            )

            # 方向錯誤 TP 偵測
            if has_tp and hasattr(self.broker, "get_position"):
                pos_check = self.broker.get_position(symbol)
                if pos_check and pos_check.entry_price > 0:
                    is_long = pos_check.qty > 0
                    for o in cond_orders:
                        otype = o.get("type", "")
                        if otype not in {"TAKE_PROFIT_MARKET", "TAKE_PROFIT"}:
                            continue
                        trigger = float(
                            o.get("stopPrice", 0) or o.get("triggerPrice", 0) or 0
                        )
                        if trigger <= 0:
                            continue
                        wrong_dir = (
                            (is_long and trigger < pos_check.entry_price * 0.99) or
                            (not is_long and trigger > pos_check.entry_price * 1.01)
                        )
                        if wrong_dir:
                            self._log.warning(
                                f"🚨 {symbol}: 方向錯誤 TP "
                                f"${trigger:,.2f} "
                                f"({'LONG' if is_long else 'SHORT'} 倉 "
                                f"entry=${pos_check.entry_price:,.2f}) → 取消"
                            )
                            self.broker.cancel_take_profit(symbol)
                            has_tp = False
                            break

            # 補掛 SL（支援 Adaptive SL）
            if not has_sl and stop_loss_atr:
                _sl_mult = float(stop_loss_atr)
                if params.get("adaptive_sl", False):
                    er_value = sig.indicators.get("er")
                    if er_value is not None:
                        from ..strategy.exit_rules import compute_adaptive_sl_multiplier
                        _sl_mult = compute_adaptive_sl_multiplier(
                            er_value, _sl_mult,
                            er_sl_min=float(params.get("er_sl_min", 1.5)),
                            er_sl_max=float(params.get("er_sl_max", 3.0)),
                        )

                if actual_pct > 0:
                    sl_price = price - _sl_mult * float(atr_value)
                else:
                    sl_price = price + _sl_mult * float(atr_value)
                self._log.info(
                    f"🔄 {symbol}: 補掛止損單 SL=${sl_price:,.2f} [{position_side}]"
                    + (f" (adaptive: {_sl_mult:.2f}x ATR)" if params.get("adaptive_sl") else "")
                )
                self.broker.place_stop_loss(
                    symbol=symbol, stop_price=sl_price,
                    position_side=position_side, reason="ensure_stop_loss",
                )

            # 補掛 TP
            if not has_tp and take_profit_atr:
                if actual_pct > 0:
                    tp_price = price + float(take_profit_atr) * float(atr_value)
                else:
                    tp_price = price - float(take_profit_atr) * float(atr_value)
                self._log.info(
                    f"🔄 {symbol}: 補掛止盈單 TP=${tp_price:,.2f} [{position_side}]"
                )
                self.broker.place_take_profit(
                    symbol=symbol, take_profit_price=tp_price,
                    position_side=position_side, reason="ensure_take_profit",
                )

            if has_sl and (has_tp or not take_profit_atr):
                self._log.debug(f"  {symbol}: SL/TP 掛單正常 ✓")

        except Exception as e:
            self._log.warning(f"⚠️  {symbol}: SL/TP 補掛檢查失敗: {e}")

    # ══════════════════════════════════════════════════════════
    #  DB 記錄
    # ══════════════════════════════════════════════════════════

    def _log_signal_to_db(self, symbol: str, sig: SignalResult) -> None:
        if not self.trading_db:
            return
        try:
            indicators = sig.indicators
            raw_signal = sig.signal
            price = sig.price
            current_pct = 0
            try:
                current_pct = self.broker.get_position_pct(symbol, price)
            except Exception:
                pass

            action = "HOLD"
            if raw_signal > 0.01 and current_pct <= 0.01:
                action = "OPEN_LONG"
            elif raw_signal < -0.01 and current_pct >= -0.01:
                action = "OPEN_SHORT"
            elif abs(raw_signal) < 0.01 and abs(current_pct) > 0.01:
                action = "CLOSE"

            self.trading_db.log_signal(
                symbol=symbol,
                signal_value=raw_signal,
                price=price,
                rsi=indicators.get("rsi"),
                adx=indicators.get("adx"),
                atr=indicators.get("atr"),
                plus_di=indicators.get("plus_di"),
                minus_di=indicators.get("minus_di"),
                target_pct=raw_signal * self._weights.get(symbol, 1.0),
                current_pct=current_pct,
                action=action,
                timestamp=sig.timestamp,
            )
        except Exception as e:
            self._log.debug(f"信號記錄失敗: {e}")

    def _log_trade_to_db(self, symbol: str, trade, reason: str) -> None:
        if not self.trading_db:
            return
        try:
            order_type = "MARKET"
            fee_rate = 0.0004
            if hasattr(trade, "raw") and trade.raw:
                order_type = trade.raw.get("_order_type", "MARKET")
                fee_rate = trade.raw.get("_fee_rate", 0.0004)
            self.trading_db.log_trade(
                symbol=symbol,
                side=trade.side,
                qty=trade.qty,
                price=trade.price,
                fee=getattr(trade, "fee", 0.0),
                fee_rate=fee_rate,
                pnl=trade.pnl,
                reason=reason,
                order_type=order_type,
                order_id_hash=getattr(trade, "order_id", "")[:8],
                position_side=getattr(trade, "position_side", ""),
            )
        except Exception as e:
            self._log.debug(f"  {symbol}: 交易寫入 DB 失敗: {e}")

    # ══════════════════════════════════════════════════════════
    #  信號處理（核心共享邏輯）
    # ══════════════════════════════════════════════════════════

    def _process_signal(self, symbol: str, sig: SignalResult) -> object | None:
        """
        處理單一幣種信號並下單（包含全部安全機制）

        Returns:
            Trade object if executed, None otherwise
        """
        raw_signal = sig.signal
        price = sig.price
        indicators = sig.indicators
        _, params = self._get_strategy_for_symbol(symbol)

        # 1. 記錄信號到 DB
        self._log_signal_to_db(symbol, sig)

        # 2. Spot clip
        if self.market_type == "spot" and raw_signal < 0:
            self._log.debug(f"  {symbol}: Spot 模式不支援做空，信號 {raw_signal:.0%} clip 到 0")
            raw_signal = 0.0

        # 3. 倉位計算
        weight = self._weights.get(symbol, 1.0 / max(len(self.symbols), 1))
        if price <= 0:
            return None

        adjusted_signal = self._apply_position_sizing(raw_signal, price, symbol)
        target_pct = adjusted_signal * weight

        current_pct = self.broker.get_position_pct(symbol, price)
        diff = abs(target_pct - current_pct)

        # 4. SL/TP 冷卻 + 孤兒掛單清理
        if self._check_sl_tp_cooldown(symbol, current_pct, target_pct):
            actual_pct = current_pct
            if not isinstance(self.broker, PaperBroker) and hasattr(self.broker, "get_position_pct"):
                try:
                    actual_pct = self.broker.get_position_pct(symbol, price)
                except Exception:
                    pass
            self._ensure_sl_tp(symbol, sig, params, actual_pct)
            return None

        # 5. 防不必要重平衡
        if target_pct != 0 and current_pct != 0:
            same_direction = (
                (target_pct > 0 and current_pct > 0) or
                (target_pct < 0 and current_pct < 0)
            )
            if same_direction:
                fill_ratio = abs(current_pct) / abs(target_pct)
                if fill_ratio >= 0.80:
                    diff = 0
                    self._log.debug(
                        f"  {symbol}: 方向一致且倉位充足 "
                        f"({current_pct:+.1%} / {target_pct:+.1%} = {fill_ratio:.0%})，跳過"
                    )
                else:
                    self._log.info(
                        f"  {symbol}: 方向一致但倉位不足 "
                        f"({current_pct:+.1%} / {target_pct:+.1%} = {fill_ratio:.0%})，需加倉"
                    )

        # 5b. Rebalance Band gate — 抑制微幅調倉
        rb_cfg = self.cfg.live.rebalance_band
        if rb_cfg.enabled and diff > 0:
            is_direction_flip_for_band = (
                (target_pct > 0 and current_pct < 0) or
                (target_pct < 0 and current_pct > 0)
            )
            # 方向翻轉 + apply_on_same_direction_only → 不受 band 限制
            apply_band = True
            if is_direction_flip_for_band and rb_cfg.apply_on_same_direction_only:
                apply_band = False

            if apply_band and diff < rb_cfg.threshold_pct:
                self._band_skip_count += 1
                equity = self._get_equity() or 10000.0
                self._band_skip_notional_est += diff * equity
                self._log.info(
                    f"  🔇 {symbol}: rebalance band SKIP — "
                    f"diff={diff:.4f} < band={rb_cfg.threshold_pct:.2%} "
                    f"(current={current_pct:+.4f}, target={target_pct:+.4f}) "
                    f"[total skips: {self._band_skip_count}]"
                )
                # 跳到 SL/TP 補掛（不執行交易）
                actual_pct = current_pct
                if not isinstance(self.broker, PaperBroker) and hasattr(self.broker, "get_position_pct"):
                    try:
                        actual_pct = self.broker.get_position_pct(symbol, price)
                    except Exception:
                        pass
                self._ensure_sl_tp(symbol, sig, params, actual_pct)
                return None

        # 6. 方向切換確認
        prev_signal = self._signal_state.get(symbol)

        is_direction_flip = (
            (target_pct > 0.01 and current_pct < -0.01) or
            (target_pct < -0.01 and current_pct > 0.01)
        )

        if is_direction_flip and self.cfg.live.flip_confirmation:
            if prev_signal is None:
                self._log.info(f"  {symbol}: 方向切換 (首次啟動) → 直接執行")
            else:
                new_dir = 1 if target_pct > 0 else -1
                prev_dir = 1 if prev_signal > 0 else (-1 if prev_signal < 0 else 0)
                if prev_dir == new_dir:
                    self._log.info(
                        f"✅ {symbol}: 方向切換已確認 "
                        f"(前次={prev_signal:+.0%}, 本次={raw_signal:+.0%})"
                    )
                else:
                    self._log.warning(
                        f"⚠️  {symbol}: 方向切換待確認 "
                        f"(持倉={current_pct:+.0%} → 信號={raw_signal:+.0%}) "
                        f"— 維持原方向"
                    )
                    if current_pct < 0:
                        target_pct = -1.0 * weight
                    else:
                        target_pct = 1.0 * weight
                    diff = abs(target_pct - current_pct)
        elif is_direction_flip:
            self._log.info(
                f"🔄 {symbol}: 方向切換 ({current_pct:+.0%} → {raw_signal:+.0%}) — 直接執行"
            )

        # 更新信號狀態
        self._signal_state[symbol] = sig.signal
        self._save_signal_state(self._signal_state)

        # Log 信號
        self._log.info(
            f"  📊 {symbol}: signal={raw_signal:.2f}, target={target_pct:.2f}, "
            f"current={current_pct:.2f}, diff={diff:.2f}, "
            f"RSI={indicators.get('rsi', '?')}, ADX={indicators.get('adx', '?')}"
        )

        # 7. 執行交易
        trade = None
        if diff >= 0.02:
            ps_method = self.cfg.position_sizing.method
            reason = f"signal={raw_signal:.0%}×{weight:.0%}"
            if ps_method != "fixed":
                reason += f" [{ps_method}→{adjusted_signal:.0%}]"

            stop_loss_price, take_profit_price = self._calculate_sl_tp_prices(
                symbol, price, target_pct, params, indicators,
            )

            try:
                trade = self.broker.execute_target_position(
                    symbol=symbol,
                    target_pct=target_pct,
                    current_price=price,
                    reason=reason,
                    stop_loss_price=stop_loss_price,
                    take_profit_price=take_profit_price,
                )
            except Exception as e:
                self._log.error(f"❌ {symbol} 交易執行失敗: {e}")
                self._log.error(traceback.format_exc())
                return None

            if trade:
                self.trade_count += 1
                self._log_trade_to_db(symbol, trade, reason)

                try:
                    leverage = self.cfg.futures.leverage if self.cfg.futures else None
                    self.notifier.send_trade(
                        symbol=symbol,
                        side=trade.side,
                        qty=trade.qty,
                        price=trade.price,
                        reason=reason,
                        pnl=trade.pnl,
                        weight=weight,
                        leverage=leverage if self.market_type == "futures" else None,
                        stop_loss_price=stop_loss_price,
                        take_profit_price=take_profit_price,
                    )
                except Exception as e:
                    self._log.debug(f"通知發送失敗: {e}")
        else:
            self._log.debug(
                f"  {symbol}: 倉位不變 (target={target_pct:.0%}, current={current_pct:.0%})"
            )

        # 8. SL/TP 補掛
        if trade:
            time.sleep(1)

        actual_pct = current_pct
        if not isinstance(self.broker, PaperBroker) and hasattr(self.broker, "get_position_pct"):
            try:
                actual_pct = self.broker.get_position_pct(symbol, price)
            except Exception:
                pass

        self._ensure_sl_tp(symbol, sig, params, actual_pct)

        # 9. Algo cache 清理
        if (
            abs(actual_pct) <= 0.01
            and not isinstance(self.broker, PaperBroker)
            and hasattr(self.broker, "_remove_algo_cache")
        ):
            self.broker._remove_algo_cache(symbol)

        return trade

    # ══════════════════════════════════════════════════════════
    #  定期任務
    # ══════════════════════════════════════════════════════════

    def _send_periodic_summary(self):
        """定期推送帳戶摘要"""
        # Log rebalance band stats if enabled
        rb_cfg = self.cfg.live.rebalance_band
        if rb_cfg.enabled and self._band_skip_count > 0:
            self._log.info(
                f"📊 Rebalance Band 統計: "
                f"skipped={self._band_skip_count}, "
                f"est_notional_saved=${self._band_skip_notional_est:,.2f}"
            )

        try:
            if isinstance(self.broker, PaperBroker):
                prices = {}
                for sym in self.symbols:
                    p = self._get_price(sym)
                    if p and p > 0:
                        prices[sym] = p
                if prices:
                    equity = self.broker.get_equity(prices)
                    positions_info = {
                        sym: {"qty": pos.qty, "avg_entry": pos.avg_entry}
                        for sym, pos in self.broker.account.positions.items()
                        if pos.is_open
                    }
                    self.notifier.send_account_summary(
                        initial_cash=self.broker.account.initial_cash,
                        equity=equity,
                        cash=self.broker.account.cash,
                        positions=positions_info,
                        trade_count=len(self.broker.account.trades),
                        mode=self.mode.upper(),
                    )
                    if self.trading_db:
                        try:
                            self.trading_db.log_daily_equity(
                                equity=equity,
                                cash=self.broker.account.cash,
                                pnl_day=equity - self.broker.account.initial_cash,
                                trade_count=len(self.broker.account.trades),
                                position_count=len(positions_info),
                            )
                        except Exception:
                            pass
            else:
                usdt = self.broker.get_balance("USDT")
                positions_info = {}
                total_value = usdt
                for sym in self.symbols:
                    pos = self.broker.get_position(sym)
                    if pos and pos.is_open:
                        p = self.broker.get_price(sym)
                        val = abs(pos.qty) * p
                        total_value += val
                        positions_info[sym] = {
                            "qty": pos.qty,
                            "avg_entry": pos.entry_price,
                            "side": "LONG" if pos.qty > 0 else "SHORT",
                        }

                self._log.info(
                    f"\n{'='*50}\n"
                    f"  帳戶摘要 [{self.mode.upper()}]\n"
                    f"{'='*50}\n"
                    f"  USDT: ${usdt:,.2f}\n"
                    f"  總權益: ${total_value:,.2f}\n"
                    f"{'='*50}"
                )

                self.notifier.send_account_summary(
                    initial_cash=0,
                    equity=total_value,
                    cash=usdt,
                    positions=positions_info,
                    trade_count=self.trade_count,
                    mode=self.mode.upper(),
                )

                if self.trading_db:
                    try:
                        self.trading_db.log_daily_equity(
                            equity=total_value,
                            cash=usdt,
                            trade_count=self.trade_count,
                            position_count=len(positions_info),
                        )
                    except Exception:
                        pass
        except Exception as e:
            self._log.warning(f"⚠️  週期報告失敗: {e}")

    # ══════════════════════════════════════════════════════════
    #  主迴圈（子類實現）
    # ══════════════════════════════════════════════════════════

    @abstractmethod
    def run(self):
        """啟動交易主迴圈"""
        ...
