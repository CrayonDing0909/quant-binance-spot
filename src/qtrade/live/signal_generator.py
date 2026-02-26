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
    # 如果 params 中有注入的衍生品/輔助數據，傳入 StrategyContext
    derivatives_data = params.pop("_derivatives_data", None)
    auxiliary_data = params.pop("_auxiliary_data", None)

    ctx = StrategyContext(
        symbol=symbol,
        interval=interval,
        market_type=market_type,
        direction=direction,
        auxiliary_data=auxiliary_data,
        derivatives_data=derivatives_data,
    )
    strategy_func = get_strategy(strategy_name)
    positions = strategy_func(df, ctx, params)

    # ── Overlay 後處理（與 run_symbol_backtest 一致）──────────
    # 確保 live pipeline 和 backtest pipeline 套用相同的 overlay
    if overlay_cfg and overlay_cfg.get("enabled", False):
        from ..strategy.overlays.oi_vol_exit_overlay import apply_overlay_by_mode

        overlay_mode = overlay_cfg.get("mode", "vol_pause")
        overlay_params = overlay_cfg.get("params", {})

        # OI 資料：與 run_symbol_backtest 一致的載入邏輯
        # 優先使用 params 中已注入的 _oi_series（來自 BaseRunner OI cache），
        # 否則從 _data_dir 自動載入（支援複合模式如 "oi_vol+lsr_confirmatory"）
        oi_series = params.get("_oi_series")
        _needs_oi = any(m in overlay_mode for m in ("oi_only", "oi_vol"))
        if oi_series is None and _needs_oi:
            data_dir = params.get("_data_dir")
            if data_dir:
                try:
                    from ..data.open_interest import (
                        get_oi_path, load_open_interest, align_oi_to_klines,
                    )
                    from pathlib import Path
                    data_dir_path = Path(data_dir)
                    for _prov in ["merged", "binance_vision", "coinglass", "binance"]:
                        _oi_path = get_oi_path(data_dir_path, symbol, _prov)
                        _oi_df = load_open_interest(_oi_path)
                        if _oi_df is not None and not _oi_df.empty:
                            oi_series = align_oi_to_klines(
                                _oi_df, df.index, max_ffill_bars=2,
                            )
                            logger.debug(f"  {symbol}: overlay OI 載入成功 (provider={_prov})")
                            break
                    else:
                        logger.warning(
                            f"  {symbol}: overlay 模式 {overlay_mode} 需要 OI 但無法載入"
                        )
                except Exception as e:
                    logger.warning(f"  {symbol}: overlay OI 載入失敗: {e}")

        # LSR 資料：overlay mode 含 lsr_confirmatory 時自動載入
        if "lsr_confirmatory" in overlay_mode and "_lsr_series" not in overlay_params:
            data_dir = params.get("_data_dir")
            if data_dir:
                try:
                    from ..data.long_short_ratio import load_lsr, align_lsr_to_klines
                    from pathlib import Path
                    data_dir_path = Path(data_dir)
                    deriv_dir = data_dir_path / "binance" / "futures" / "derivatives"
                    lsr_type = overlay_params.get("lsr_type", "lsr")
                    lsr_raw = load_lsr(symbol, lsr_type, data_dir=deriv_dir)
                    if lsr_raw is not None and not lsr_raw.empty:
                        lsr_aligned = align_lsr_to_klines(lsr_raw, df.index, max_ffill_bars=2)
                        overlay_params["_lsr_series"] = lsr_aligned
                        logger.debug(f"  {symbol}: overlay LSR 載入成功 ({len(lsr_raw)} rows)")
                    else:
                        logger.warning(f"  {symbol}: overlay LSR 數據不存在 ({lsr_type})")
                except Exception as e:
                    logger.warning(f"  {symbol}: overlay LSR 載入失敗: {e}")

        # OI 確認層數據：lsr_confirmatory + oi_confirm_enabled 時載入
        if ("lsr_confirmatory" in overlay_mode
                and overlay_params.get("oi_confirm_enabled", False)
                and "_oi_series" not in overlay_params):
            data_dir = params.get("_data_dir")
            if data_dir:
                try:
                    from ..data.open_interest import (
                        get_oi_path, load_open_interest, align_oi_to_klines,
                    )
                    from pathlib import Path
                    data_dir_path = Path(data_dir)
                    for _prov in ["merged", "binance_vision", "coinglass", "binance"]:
                        _oi_path = get_oi_path(data_dir_path, symbol, _prov)
                        _oi_df = load_open_interest(_oi_path)
                        if _oi_df is not None and not _oi_df.empty:
                            overlay_params["_oi_series"] = align_oi_to_klines(
                                _oi_df, df.index, max_ffill_bars=2,
                            )
                            logger.debug(f"  {symbol}: overlay OI (for LSR confirm) 載入成功")
                            break
                except Exception as e:
                    logger.warning(f"  {symbol}: overlay OI (for LSR confirm) 載入失敗: {e}")

        # FR 確認層數據：lsr_confirmatory + fr_confirm_enabled 時載入
        if ("lsr_confirmatory" in overlay_mode
                and overlay_params.get("fr_confirm_enabled", False)
                and "_fr_series" not in overlay_params):
            data_dir = params.get("_data_dir")
            if data_dir:
                try:
                    from ..data.funding_rate import load_funding_rates, get_funding_rate_path
                    from pathlib import Path
                    data_dir_path = Path(data_dir)
                    fr_path = get_funding_rate_path(data_dir_path, symbol)
                    funding_df = load_funding_rates(fr_path)
                    if funding_df is not None and not funding_df.empty:
                        fr_col = "fundingRate" if "fundingRate" in funding_df.columns else funding_df.columns[0]
                        fr_series = funding_df[fr_col]
                        fr_aligned = fr_series.reindex(df.index, method="ffill")
                        overlay_params["_fr_series"] = fr_aligned
                        logger.debug(f"  {symbol}: overlay FR (for LSR confirm) 載入成功")
                    else:
                        logger.warning(f"  {symbol}: overlay FR 數據不存在")
                except Exception as e:
                    logger.warning(f"  {symbol}: overlay FR (for LSR confirm) 載入失敗: {e}")

        positions = apply_overlay_by_mode(
            position=positions,
            price_df=df,
            oi_series=oi_series,
            params=overlay_params,
            mode=overlay_mode,
        )
        logger.info(f"📊 Live overlay applied: mode={overlay_mode}")

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
