"""
Signal Generator — 即時信號產生器

從 Binance 拉取最新 K 線數據，運行策略，輸出交易信號。
設計為復用回測策略程式碼，無需改寫策略。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd

from ..data.klines import fetch_klines
from ..data.storage import load_klines
from ..data.quality import clean_data
from ..strategy import get_strategy
from ..strategy.base import StrategyContext
from ..utils.log import get_logger

logger = get_logger("signal_gen")

# 策略至少需要多少根 K 線才能計算指標
MIN_BARS = 300


@dataclass
class PositionInfo:
    """即時持倉資訊（由 Runner 注入，供 Telegram 顯示）"""
    pct: float = 0.0
    entry: float = 0.0
    qty: float = 0.0
    side: str = ""           # "LONG" / "SHORT" / ""
    sl: float | None = None  # 止損價
    tp: float | None = None  # 止盈價


@dataclass
class SignalResult:
    """
    標準化信號結果

    取代原有的 raw dict，提供型別安全和 IDE 自動補全。
    """
    symbol: str
    signal: float               # 目標倉位 [-1, 1]（futures）或 [0, 1]（spot）
    price: float                # 當前價格
    timestamp: str              # 最新 K 線時間
    strategy: str               # 策略名稱
    indicators: dict = field(default_factory=dict)   # RSI, ADX, ATR, ER 等
    position_info: PositionInfo = field(default_factory=PositionInfo)  # Runner 注入

    def to_dict(self) -> dict:
        """序列化為 dict（JSON 輸出用）"""
        d = {
            "symbol": self.symbol,
            "signal": self.signal,
            "price": self.price,
            "timestamp": self.timestamp,
            "strategy": self.strategy,
            "indicators": self.indicators,
        }
        if self.position_info and self.position_info.pct != 0:
            d["_position"] = {
                "pct": self.position_info.pct,
                "entry": self.position_info.entry,
                "qty": self.position_info.qty,
                "side": self.position_info.side,
                "sl": self.position_info.sl,
                "tp": self.position_info.tp,
            }
        return d


def fetch_recent_klines(
    symbol: str,
    interval: str,
    bars: int = MIN_BARS,
) -> pd.DataFrame:
    """
    從 Binance 拉取最近 N 根 **已收盤** K 線

    Binance API 總是返回當前未收盤的 K 線作為最後一根，
    在 Live Trading 中使用未收盤 K 線會導致指標不可靠（假信號）。
    因此這裡會自動丟棄未收盤的 K 線。

    Args:
        symbol: 交易對, e.g. "BTCUSDT"
        interval: K 線週期, e.g. "1h"
        bars: 需要的 K 線數量

    Returns:
        DataFrame with OHLCV (只包含已收盤的 K 線)
    """
    from datetime import datetime, timezone, timedelta

    # 根據 interval 估算需要多少時間
    interval_minutes = {
        "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
        "1h": 60, "2h": 120, "4h": 240, "6h": 360, "8h": 480,
        "12h": 720, "1d": 1440,
    }

    minutes = interval_minutes.get(interval, 60)
    start_dt = datetime.now(timezone.utc) - timedelta(minutes=minutes * (bars + 10))
    start_str = start_dt.strftime("%Y-%m-%d")

    df = fetch_klines(symbol, interval, start_str)
    df = clean_data(df, fill_method="forward", remove_outliers=False, remove_duplicates=True)

    # ── 丟棄未收盤的 K 線 ──────────────────────────
    # Binance close_time 是該 K 線的結束時間 (e.g. 1h K 線 12:00 → close_time=12:59:59.999)
    # 如果 close_time > 當前時間 → 該 K 線尚未收盤，必須丟棄
    if "close_time" in df.columns:
        now = pd.Timestamp.now(tz="UTC")
        closed_mask = df["close_time"] <= now
        n_dropped = (~closed_mask).sum()
        if n_dropped > 0:
            logger.debug(f"  {symbol}: 丟棄 {n_dropped} 根未收盤 K 線")
        df = df[closed_mask]

    # 只保留最近 bars 根
    if len(df) > bars:
        df = df.iloc[-bars:]

    return df


def generate_signal(
    symbol: str,
    strategy_name: str,
    params: dict,
    interval: str = "1h",
    bars: int = MIN_BARS,
    df: pd.DataFrame | None = None,
    market_type: str = "spot",
    direction: str = "both",
    overlay_cfg: dict | None = None,
) -> SignalResult:
    """
    生成單個交易對的信號

    Args:
        symbol: 交易對
        strategy_name: 策略名稱
        params: 策略參數
        interval: K 線週期
        bars: 需要的 K 線數量
        df: 可選，直接傳入 K 線數據（測試用）
        market_type: 市場類型 "spot" 或 "futures"
        direction: 交易方向 "both", "long_only", "short_only"
        overlay_cfg: overlay 配置 dict（與 backtest pipeline 一致）
            例: {"enabled": True, "mode": "vol_pause", "params": {...}}

    Returns:
        SignalResult 標準化信號結果
    """
    # 獲取數據
    if df is None:
        df = fetch_recent_klines(symbol, interval, bars)

    if len(df) < 50:
        logger.warning(f"⚠️  {symbol}: 數據不足 ({len(df)} bars)")
        return SignalResult(
            symbol=symbol, signal=0.0, price=0.0,
            timestamp="", strategy=strategy_name,
        )

    # 運行策略（傳入正確的 market_type 和 direction）
    # 使用 .get() 而非 .pop() — 不修改呼叫者的 params dict
    derivatives_data = params.get("_derivatives_data")
    auxiliary_data = params.get("_auxiliary_data")

    ctx = StrategyContext(
        symbol=symbol,
        interval=interval,
        market_type=market_type,
        direction=direction,
        signal_delay=0,  # 明確設定：Live 模式不延遲（信號即時執行）
        auxiliary_data=auxiliary_data,
        derivatives_data=derivatives_data,
    )
    strategy_func = get_strategy(strategy_name)
    # 傳入不含已消費的 key 的 params（_derivatives_data / _auxiliary_data
    # 已被 StrategyContext 消費，不應傳給策略函數；
    # 其他 _data_dir / _oi_series 等仍需傳入，部分策略會讀取）
    _consumed_keys = {"_derivatives_data", "_auxiliary_data"}
    clean_params = {k: v for k, v in params.items() if k not in _consumed_keys}
    positions = strategy_func(df, ctx, clean_params)

    # ── Overlay 後處理 ─────────────────────────────────
    # 使用共用 overlay pipeline，確保 live 和 backtest 行為完全一致
    if overlay_cfg and overlay_cfg.get("enabled", False):
        from ..strategy.overlays.overlay_pipeline import prepare_and_apply_overlay

        positions = prepare_and_apply_overlay(
            positions, df, overlay_cfg, symbol,
            data_dir=params.get("_data_dir"),
            injected_oi_series=params.get("_oi_series"),
        )

    # ── Regime Gate 後處理 ──────────────────────────────
    # Portfolio-level regime gate scales all signals based on BTC trend regime.
    # Applied after overlay so it acts as the outermost defense layer.
    regime_gate_cfg = params.get("_regime_gate")
    if regime_gate_cfg and regime_gate_cfg.get("enabled", False):
        try:
            from ..strategy.filters import compute_portfolio_regime_gate
            ref_df = params.get("_regime_gate_ref_df")
            if ref_df is not None and len(ref_df) > 50:
                gate = compute_portfolio_regime_gate(
                    ref_df,
                    adx_period=regime_gate_cfg.get("adx_period", 14),
                    adx_trend_threshold=regime_gate_cfg.get("adx_trend_threshold", 25.0),
                    adx_weak_threshold=regime_gate_cfg.get("adx_weak_threshold", 15.0),
                    er_lookback=regime_gate_cfg.get("efficiency_ratio_lookback", 20),
                    er_trend_threshold=regime_gate_cfg.get("er_trend_threshold", 0.40),
                    er_weak_threshold=regime_gate_cfg.get("er_weak_threshold", 0.25),
                    scale_trending=regime_gate_cfg.get("scale_trending", 1.0),
                    scale_weak=regime_gate_cfg.get("scale_weak", 0.5),
                    scale_no_trend=regime_gate_cfg.get("scale_no_trend", 0.0),
                )
                latest_gate = float(gate.iloc[-1])
                if latest_gate < 1.0:
                    positions = positions * latest_gate
                    logger.info(
                        f"  🚦 {symbol}: regime_gate={latest_gate:.1f} "
                        f"(signal scaled to {float(positions.iloc[-1]):.2f})"
                    )
        except Exception as e:
            logger.warning(f"  ⚠️ {symbol}: regime gate failed: {e}")

    # 取最後一根 K 線的信號
    latest_signal = float(positions.iloc[-1])
    latest_price = float(df["close"].iloc[-1])
    latest_time = str(df.index[-1])

    # 收集關鍵指標（除錯用）
    indicators = {
        "close": latest_price,
        "bars": len(df),
    }

    # 優先使用策略自帶指標（strategy.attrs["indicators"]）
    strategy_indicators = getattr(positions, "attrs", {}).get("indicators")
    if strategy_indicators:
        for k, v in strategy_indicators.items():
            if not k.startswith("_"):  # 跳過內部欄位
                indicators[k] = v

    # 回退: 若策略未提供指標，計算通用 RSI/ADX/ATR
    if not strategy_indicators:
        try:
            from ..indicators import calculate_rsi, calculate_adx, calculate_atr
            rsi_period = int(params.get("rsi_period", 14))
            rsi = calculate_rsi(df["close"], rsi_period)
            indicators["rsi"] = round(float(rsi.iloc[-1]), 2)

            adx_period = int(params.get("adx_period", 14))
            adx_df = calculate_adx(df, adx_period)
            indicators["adx"] = round(float(adx_df["ADX"].iloc[-1]), 2)
            indicators["plus_di"] = round(float(adx_df["+DI"].iloc[-1]), 2)
            indicators["minus_di"] = round(float(adx_df["-DI"].iloc[-1]), 2)

            atr_period = int(params.get("atr_period", 14))
            atr = calculate_atr(df, atr_period)
            indicators["atr"] = round(float(atr.iloc[-1]), 2)

            # Efficiency Ratio（ER filter 或 adaptive SL 啟用時計算）
            er_period = params.get("er_period") or (
                params.get("adaptive_sl_er_period", 10) if params.get("adaptive_sl") else None
            )
            if er_period is not None:
                from ..indicators import calculate_efficiency_ratio
                er = calculate_efficiency_ratio(df["close"], period=int(er_period))
                indicators["er"] = round(float(er.iloc[-1]), 3)
        except Exception:
            pass  # 指標計算失敗不影響信號

    result = SignalResult(
        symbol=symbol,
        signal=latest_signal,
        price=latest_price,
        timestamp=latest_time,
        strategy=strategy_name,
        indicators=indicators,
    )

    # 動態 log 顯示策略指標
    _log_parts = [f"📊 {symbol}: signal={latest_signal:.1f}, price={latest_price:.2f}"]
    if strategy_indicators:
        for _k in ("tsmom", "carry", "ema_trend", "htf", "tier"):
            _v = indicators.get(_k)
            if _v is not None:
                _log_parts.append(f"{_k}={_v}")
    else:
        _log_parts.append(f"RSI={indicators.get('rsi', '?')}")
        _log_parts.append(f"ADX={indicators.get('adx', '?')}")
    logger.info(", ".join(_log_parts))

    return result
