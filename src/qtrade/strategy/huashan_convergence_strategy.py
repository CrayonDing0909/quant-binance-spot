"""
华山论剑 Multi-Signal Convergence 策略

Alpha 來源（5 大支柱）：
    1. OI 4-Quadrant：Price dir × OI dir 判斷市場狀態 (Gate.com: 60-70% accuracy)
    2. Funding Rate：持續負值 = 逆向做多；>0.10%/8h = extreme (arXiv:2212.06888)
    3. LSR 多空比：multi-σ extremes 才有效 (AInvest: >70% long = fade)
    4. CB Premium：60-day z-score (CryptoQuant: SMA-50 consensus)
    5. 穩定幣流入：場外資金 momentum (BDC: 229% ROI, 但 BTC 上衰退中)

v2 優化（基於外部研究 + 內部 LSR v3 驗證）：
    - OI 改為 4-Quadrant (price×OI direction) 替代純 z-score
    - CB Premium z-score window 24h → 60 days (外部研究共識)
    - 穩定幣降權 0.5 → 0.3 (CryptoQuant CEO: BTC 流動性走 ETF 非穩定幣)
    - ADX regime gate (>25) — LSR v3 已驗證能過濾震盪假信號
    - ATR 3.0x SL + 168h time stop — LSR v3 Cycle 2 驗證 MDD 改善 11pp
    - Entry persist ≥ 2 bars + cooldown 24h — LSR v3 Cycle 1 驗證

Research Evidence:
    - docs/research/20260328_huashan_external_research.md — 13 篇文獻整理
    - ACM 2025: 4-6 factor composite SR=2.5
    - CF Benchmarks: sentiment-gated SR=1.52
    - v1 backtest: BTC SR=0.55/MDD-30%, ETH SR=0.86/MDD-23%

Changelog:
    v1 (2026-03-28): Initial — 5 pillars convergence + profit target exit
    v2 (2026-03-28): OI 4-quadrant, CB 60d z-score, ADX gate, ATR SL, persist/cooldown
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from qtrade.strategy.base import StrategyContext
from qtrade.strategy import register_strategy
from qtrade.utils.log import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════
#  Sub-signal builders: each returns Series in [-1, +1]
# ═══════════════════════════════════════════════════════════

def _oi_signal(df: pd.DataFrame, symbol: str, data_dir: Path | None, params: dict) -> pd.Series | None:
    """
    OI 4-Quadrant 信號 (Gate.com framework, 60-70% accuracy):
        Price ↑ + OI ↑ = 強勢上漲確認           → +1.0 (bullish)
        Price ↑ + OI ↓ = 空頭回補 (非新需求)     → +0.3 (cautious bullish)
        Price ↓ + OI ↑ = 強勢下跌 (新空頭進場)   → -1.0 (bearish)
        Price ↓ + OI ↓ = 多頭清算完成            → -0.3 → contrarian +0.5 (bullish)
    """
    oi_lookback = int(params.get("oi_lookback", 24))

    if data_dir is None:
        return None

    try:
        from qtrade.data.open_interest import OI_PROVIDER_SEARCH_ORDER
        oi_base = Path(data_dir) / "binance" / "futures" / "open_interest"

        oi_df = None
        for provider in OI_PROVIDER_SEARCH_ORDER:
            path = oi_base / provider / f"{symbol}.parquet"
            if path.exists():
                oi_df = pd.read_parquet(path)
                break

        if oi_df is None or oi_df.empty:
            return None

        col = "sumOpenInterestValue" if "sumOpenInterestValue" in oi_df.columns else "sumOpenInterest"
        oi_series = oi_df[col].astype(float)

        # Align to kline index
        if oi_series.index.tz is None and df.index.tz is not None:
            oi_series.index = oi_series.index.tz_localize(df.index.tz)
        oi_aligned = oi_series.reindex(df.index, method="ffill")

        # 4-Quadrant: price direction × OI direction
        price_change = df["close"].pct_change(oi_lookback)
        oi_change = oi_aligned.pct_change(oi_lookback)

        signal = pd.Series(0.0, index=df.index)
        price_up = price_change > 0
        oi_up = oi_change > 0

        signal[price_up & oi_up] = 1.0      # Strong uptrend confirmed
        signal[price_up & ~oi_up] = 0.3     # Short squeeze (cautious)
        signal[~price_up & oi_up] = -1.0    # Strong downtrend
        signal[~price_up & ~oi_up] = 0.5    # Liquidation flush → contrarian bullish

        # Smooth with EMA to avoid bar-to-bar noise
        signal = signal.ewm(span=12, adjust=False).mean()

        logger.info(f"  华山 OI [{symbol}]: mean={signal.mean():.4f}, coverage={signal.notna().mean():.1%}")
        return signal

    except Exception as e:
        logger.warning(f"  华山 OI [{symbol}]: failed: {e}")
        return None


def _funding_rate_signal(df: pd.DataFrame, symbol: str, data_dir: Path | None, params: dict) -> pd.Series | None:
    """
    Funding Rate 信號：FR 的 z-score 逆向。
    FR 持續負值 = 空頭付費 = 逆向做多 (+1)
    FR 持續高位正值 = 多頭付費 = 逆向做空 (-1)
    """
    fr_zscore_window = int(params.get("fr_zscore_window", 168))

    if data_dir is None:
        return None

    try:
        fr_path = Path(data_dir) / "binance" / "futures" / "funding_rate" / f"{symbol}.parquet"
        if not fr_path.exists():
            return None

        fr_df = pd.read_parquet(fr_path)
        fr_series = fr_df["funding_rate"].astype(float)

        # Align to kline index (funding is 8h, ffill to 1h)
        if fr_series.index.tz is None and df.index.tz is not None:
            fr_series.index = fr_series.index.tz_localize(df.index.tz)
        fr_aligned = fr_series.reindex(df.index, method="ffill")

        # Z-score
        roll = fr_aligned.rolling(fr_zscore_window, min_periods=fr_zscore_window // 4)
        z = (fr_aligned - roll.mean()) / roll.std().replace(0, np.nan)

        # Invert: high FR → bearish, low FR → bullish
        signal = -z.clip(-3, 3) / 3.0

        logger.info(f"  华山 FR [{symbol}]: mean={signal.mean():.4f}, coverage={signal.notna().mean():.1%}")
        return signal

    except Exception as e:
        logger.warning(f"  华山 FR [{symbol}]: failed: {e}")
        return None


def _lsr_signal(df: pd.DataFrame, symbol: str, data_dir: Path | None, params: dict) -> pd.Series | None:
    """
    LSR 信號：散戶多空比的 z-score 逆向。
    LSR 極高（散戶做多擁擠）→ bearish (-1)
    LSR 極低（散戶做空擁擠）→ bullish (+1)
    """
    lsr_zscore_window = int(params.get("lsr_zscore_window", 168))

    if data_dir is None:
        return None

    try:
        from qtrade.data.long_short_ratio import load_lsr, align_lsr_to_klines

        deriv_dir = Path(data_dir) / "binance" / "futures" / "derivatives"
        lsr_raw = load_lsr(symbol, lsr_type="lsr", data_dir=deriv_dir)
        if lsr_raw is None:
            return None

        lsr_aligned = align_lsr_to_klines(lsr_raw, df.index, max_ffill_bars=2)
        if lsr_aligned is None:
            return None

        # Z-score
        roll = lsr_aligned.rolling(lsr_zscore_window, min_periods=lsr_zscore_window // 4)
        z = (lsr_aligned - roll.mean()) / roll.std().replace(0, np.nan)

        # Invert: high LSR (retail crowded long) → bearish
        signal = -z.clip(-3, 3) / 3.0

        logger.info(f"  华山 LSR [{symbol}]: mean={signal.mean():.4f}, coverage={signal.notna().mean():.1%}")
        return signal

    except Exception as e:
        logger.warning(f"  华山 LSR [{symbol}]: failed: {e}")
        return None


def _cb_premium_signal(df: pd.DataFrame, symbol: str, data_dir: Path | None, params: dict) -> pd.Series | None:
    """
    CB Premium 信號：Coinbase-Binance 價差 z-score。
    Premium > 0（美國機構買入）→ bullish (+1)
    Premium < 0（亞洲拋售）→ bearish (-1)
    """
    cb_zscore_window = int(params.get("cb_zscore_window", 60))

    if data_dir is None:
        return None

    # Map Binance symbol to Coinbase cache file
    cb_map = {"BTCUSDT": "BTC_USD_1h.parquet", "ETHUSDT": "ETH_USD_1h.parquet"}
    cb_file = cb_map.get(symbol)
    if cb_file is None:
        return None

    try:
        cb_path = Path(data_dir) / "coinbase" / cb_file
        if not cb_path.exists():
            return None

        cb_df = pd.read_parquet(cb_path)
        cb_close = cb_df["close"]

        # Compute premium
        common = df.index.intersection(cb_close.index)
        if len(common) < 500:
            return None

        bn = df["close"].loc[common]
        cb = cb_close.loc[common]
        premium = (cb - bn) / bn
        premium = premium.replace([np.inf, -np.inf], np.nan)
        premium = premium.reindex(df.index).ffill()

        # Z-score (positive z = premium above normal = bullish)
        roll = premium.rolling(cb_zscore_window, min_periods=cb_zscore_window // 2)
        z = (premium - roll.mean()) / roll.std().replace(0, np.nan)
        signal = z.clip(-3, 3) / 3.0

        logger.info(f"  华山 CB [{symbol}]: mean={signal.mean():.4f}, coverage={signal.notna().mean():.1%}")
        return signal

    except Exception as e:
        logger.warning(f"  华山 CB [{symbol}]: failed: {e}")
        return None


def _stablecoin_signal(df: pd.DataFrame, params: dict) -> pd.Series | None:
    """
    穩定幣信號：交易所穩定幣市值 30d momentum。
    穩定幣增加（場外資金進場）→ bullish (+1)
    穩定幣減少 → bearish (-1)
    """
    sc_mom_window = int(params.get("sc_mom_window", 30))
    sc_zscore_window = int(params.get("sc_zscore_window", 90))

    try:
        from qtrade.data.onchain import load_onchain

        sc_df = load_onchain(provider="defillama", metric="stablecoin_mcap")
        if sc_df is None or sc_df.empty:
            return None

        col = "totalCirculatingPeggedUSD" if "totalCirculatingPeggedUSD" in sc_df.columns else sc_df.columns[0]
        sc_series = sc_df[col].astype(float)

        # 30d momentum
        mom = sc_series.pct_change(sc_mom_window)

        # Causal: shift 1 day, then ffill to 1h
        mom_shifted = mom.shift(1)
        if mom_shifted.index.tz is None and df.index.tz is not None:
            mom_shifted.index = mom_shifted.index.tz_localize(df.index.tz)
        mom_aligned = mom_shifted.reindex(df.index, method="ffill")

        # Z-score
        roll = mom_aligned.rolling(sc_zscore_window * 24, min_periods=sc_zscore_window * 6)
        z = (mom_aligned - roll.mean()) / roll.std().replace(0, np.nan)
        signal = z.clip(-3, 3) / 3.0

        logger.info(f"  华山 SC: mean={signal.mean():.4f}, coverage={signal.notna().mean():.1%}")
        return signal

    except Exception as e:
        logger.warning(f"  华山 SC: failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════
#  Main Strategy
# ═══════════════════════════════════════════════════════════

@register_strategy("huashan_convergence")
def generate_positions(
    df: pd.DataFrame,
    ctx: StrategyContext,
    params: dict,
) -> pd.Series:
    """
    华山论剑 Multi-Signal Convergence Strategy v2.

    v2 optimizations (from external research + LSR v3 validated patterns):
        - OI: 4-Quadrant (price dir × OI dir) replacing pure z-score
        - CB Premium: 60-day z-score (CryptoQuant consensus)
        - Stablecoin: weight 0.3 (degraded for BTC per CryptoQuant CEO)
        - ADX regime gate (>25): filters choppy markets (LSR v3 validated)
        - ATR 3.0x SL: tail protection (LSR v3 Cycle 2: optimal)
        - 168h time stop: prevents zombie holdings (LSR v3 validated)
        - Entry persist ≥ 2 rebalance bars: reduces false signals
        - Cooldown 24h after exit: prevents immediate re-entry losses
    """
    # ── Params: weights ──
    w_oi = float(params.get("w_oi", 1.0))
    w_fr = float(params.get("w_fr", 1.0))
    w_lsr = float(params.get("w_lsr", 1.0))
    w_cb = float(params.get("w_cb", 1.0))
    w_sc = float(params.get("w_sc", 0.3))  # v2: lowered from 0.5

    # ── Params: entry/exit ──
    entry_threshold = float(params.get("entry_threshold", 0.3))
    exit_threshold = float(params.get("exit_threshold", 0.0))
    min_signals = int(params.get("min_signals", 3))
    rebalance_hours = int(params.get("rebalance_hours", 24))

    # ── Params: profit target (华山论剑 rule) ──
    use_profit_target = bool(params.get("use_profit_target", True))
    profit_target_pct = float(params.get("profit_target_pct", 0.08))

    # ── Params: v2 exit design (from LSR v3 research) ──
    sl_atr_mult = float(params.get("sl_atr_mult", 3.0))
    sl_atr_lookback = int(params.get("sl_atr_lookback", 14))
    max_hold_bars = int(params.get("max_hold_bars", 168))

    # ── Params: v2 entry filters (from LSR v3 research) ──
    adx_gate_enabled = bool(params.get("adx_gate_enabled", True))
    adx_period = int(params.get("adx_period", 14))
    adx_threshold = float(params.get("adx_threshold", 25.0))
    persist_bars = int(params.get("persist_bars", 2))
    cooldown_bars = int(params.get("cooldown_bars", 24))

    data_dir = params.get("_data_dir")

    # ── 1. Build sub-signals ──
    sub_signals: list[tuple[str, float, pd.Series]] = []

    oi_sig = _oi_signal(df, ctx.symbol, data_dir, params)
    if oi_sig is not None:
        sub_signals.append(("OI", w_oi, oi_sig))

    fr_sig = _funding_rate_signal(df, ctx.symbol, data_dir, params)
    if fr_sig is not None:
        sub_signals.append(("FR", w_fr, fr_sig))

    lsr_sig = _lsr_signal(df, ctx.symbol, data_dir, params)
    if lsr_sig is not None:
        sub_signals.append(("LSR", w_lsr, lsr_sig))

    cb_sig = _cb_premium_signal(df, ctx.symbol, data_dir, params)
    if cb_sig is not None:
        sub_signals.append(("CB", w_cb, cb_sig))

    sc_sig = _stablecoin_signal(df, params)
    if sc_sig is not None:
        sub_signals.append(("SC", w_sc, sc_sig))

    n_active = len(sub_signals)
    logger.info(f"  华山 [{ctx.symbol}]: {n_active}/5 sub-signals active: "
                f"{[name for name, _, _ in sub_signals]}")

    if n_active < min_signals:
        logger.warning(
            f"  华山 [{ctx.symbol}]: only {n_active} signals < min_signals={min_signals}, returning flat"
        )
        return pd.Series(0.0, index=df.index)

    # ── 2. Compute weighted convergence score ──
    total_weight = sum(w for _, w, _ in sub_signals)
    score = pd.Series(0.0, index=df.index)

    for name, weight, sig in sub_signals:
        aligned = sig.reindex(df.index).fillna(0.0)
        score += (weight / total_weight) * aligned

    # ── 3. ADX regime gate (v2) ──
    adx_ok = pd.Series(True, index=df.index)
    if adx_gate_enabled:
        from qtrade.indicators import calculate_adx
        adx_result = calculate_adx(df, adx_period)
        if isinstance(adx_result, dict):
            adx_series = adx_result["ADX"]
        elif isinstance(adx_result, pd.DataFrame):
            adx_series = adx_result["ADX"] if "ADX" in adx_result.columns else adx_result.iloc[:, 0]
        else:
            adx_series = adx_result
        adx_ok = adx_series > adx_threshold
        logger.info(f"  华山 ADX [{ctx.symbol}]: gate pass rate={float(adx_ok.mean()):.1%}")

    # ── 4. ATR for stop-loss (v2) ──
    from qtrade.indicators import calculate_atr
    atr = calculate_atr(df, sl_atr_lookback)

    # ── 5. State machine with full exit design ──
    n = len(df)
    pos_arr = np.zeros(n, dtype=np.float64)
    close_vals = df["close"].values
    high_vals = df["high"].values
    low_vals = df["low"].values
    score_vals = score.values
    adx_ok_vals = adx_ok.values
    atr_vals = atr.values

    current_pos = 0.0
    entry_price = 0.0
    entry_bar = -999
    bars_above_threshold = 0  # persist counter
    last_exit_bar = -999      # cooldown tracker

    for i in range(n):
        s = score_vals[i]

        if np.isnan(s):
            pos_arr[i] = current_pos
            continue

        # ── Exit checks (every bar, not just rebalance) ──
        if current_pos != 0.0 and entry_price > 0:
            bars_held = i - entry_bar
            current_atr = atr_vals[i] if not np.isnan(atr_vals[i]) else 0.0

            # (a) Profit target exit (华山论剑: BTC ≥8%)
            if use_profit_target:
                pnl_pct = (close_vals[i] - entry_price) / entry_price * current_pos
                if pnl_pct >= profit_target_pct:
                    current_pos = 0.0
                    entry_price = 0.0
                    last_exit_bar = i

            # (b) ATR stop-loss
            if current_pos != 0.0 and current_atr > 0:
                if current_pos > 0:
                    sl_price = entry_price - sl_atr_mult * current_atr
                    if low_vals[i] <= sl_price:
                        current_pos = 0.0
                        entry_price = 0.0
                        last_exit_bar = i
                elif current_pos < 0:
                    sl_price = entry_price + sl_atr_mult * current_atr
                    if high_vals[i] >= sl_price:
                        current_pos = 0.0
                        entry_price = 0.0
                        last_exit_bar = i

            # (c) Time stop
            if current_pos != 0.0 and bars_held >= max_hold_bars:
                current_pos = 0.0
                entry_price = 0.0
                last_exit_bar = i

            # (d) Score reversal exit (check every bar)
            if current_pos > 0 and s < exit_threshold:
                current_pos = 0.0
                entry_price = 0.0
                last_exit_bar = i
            elif current_pos < 0 and s > -exit_threshold:
                current_pos = 0.0
                entry_price = 0.0
                last_exit_bar = i

        # ── Entry checks (only on rebalance bars) ──
        if current_pos == 0.0 and i % rebalance_hours == 0:
            # Cooldown check
            if i - last_exit_bar < cooldown_bars:
                pos_arr[i] = 0.0
                continue

            # ADX gate check
            if not adx_ok_vals[i]:
                bars_above_threshold = 0
                pos_arr[i] = 0.0
                continue

            # Persist check: score must be above threshold for persist_bars consecutive rebalance bars
            if s > entry_threshold or (s < -entry_threshold and ctx.can_short):
                bars_above_threshold += 1
            else:
                bars_above_threshold = 0

            if bars_above_threshold >= persist_bars:
                if s > entry_threshold:
                    current_pos = 1.0
                    entry_price = close_vals[i]
                    entry_bar = i
                    bars_above_threshold = 0
                elif s < -entry_threshold and ctx.can_short:
                    current_pos = -1.0
                    entry_price = close_vals[i]
                    entry_bar = i
                    bars_above_threshold = 0

        pos_arr[i] = current_pos

    pos = pd.Series(pos_arr, index=df.index)

    tim = float((pos.abs() > 0.01).mean())
    long_pct = float((pos > 0.01).mean())
    short_pct = float((pos < -0.01).mean())

    logger.info(
        f"  华山 [{ctx.symbol}]: score mean={score.mean():.4f}, std={score.std():.4f}, "
        f"TIM={tim:.1%}, long={long_pct:.1%}, short={short_pct:.1%}"
    )

    return pos
