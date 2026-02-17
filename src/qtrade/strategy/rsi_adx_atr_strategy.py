"""
RSI + ADX + ATR 組合策略

核心理念：
    1. ADX 過濾 → 只在有趨勢的市場做單
    2. RSI 擇時  → 趨勢中找回調入場點
    3. ATR 止損  → 動態止損距離，適應波動率變化
    4. 冷卻期    → 止損後不追單

與純 RSI 策略的區別：
    - 純 RSI：隨時交易 → 震盪市頻繁虧損
    - 本策略：趨勢確認 + 回調入場 + 動態止損 → 減少無效交易

參數預設值經過初步調優，但建議使用 optimize_params.py 做網格搜索。
"""
from __future__ import annotations
import pandas as pd
import os
from pathlib import Path
from .base import StrategyContext
from . import register_strategy
from ..indicators import calculate_rsi, calculate_adx, calculate_atr
from .exit_rules import apply_exit_rules
from .filters import (
    trend_filter, htf_trend_filter, htf_soft_trend_filter,
    volatility_filter, funding_rate_filter, efficiency_ratio_filter,
    smooth_positions,
)
from ..data.funding_rate import load_funding_rates, get_funding_rate_path, align_funding_to_klines


@register_strategy("rsi_adx_atr")
def generate_positions(df: pd.DataFrame, ctx: StrategyContext, params: dict) -> pd.Series:
    """
    RSI 回調入場 + ADX 趨勢過濾 + ATR 動態止損

    入場條件（同時滿足）：
        1. ADX > min_adx（有趨勢）
        2. 做多：+DI > -DI（上升趨勢）+ RSI 從超賣區回升
        3. 做空：-DI > +DI（下降趨勢）+ RSI 從超買區回落（僅 Futures）

    出場條件（任一觸發）：
        1. RSI > overbought（動力衰竭）→ 平多（或開空）
        2. RSI < oversold（超賣）→ 平空（或開多）
        3. ATR 止損被觸發
        4. ATR 止盈被觸發
        5. 移動止損被觸發（如果啟用）

    params:
        rsi_period:        RSI 週期，預設 14
        oversold:          超賣線，預設 35
        overbought:        超買線，預設 70
        min_adx:           最小 ADX 值，預設 20
        adx_period:        ADX 週期，預設 14
        stop_loss_atr:     止損 ATR 倍數，預設 2.0
        take_profit_atr:   止盈 ATR 倍數，預設 3.0
        trailing_stop_atr: 移動止損 ATR 倍數，None = 不用，建議 2.5
        atr_period:        ATR 週期，預設 14
        cooldown_bars:     冷卻期，預設 6
        htf_interval:      高級時間框架，e.g. "4h"，None = 不使用
        htf_ema_fast:      高級 TF 快速 EMA，預設 20
        htf_ema_slow:      高級 TF 慢速 EMA，預設 50
        
        # ── 波動率過濾器（可選，防止低波動磨耗）──
        min_atr_ratio:     最小 ATR/Price 比率，None = 不使用，建議 0.005~0.01
        vol_filter_mode:   "absolute" 或 "percentile"，預設 "absolute"
        vol_min_percentile: 百分位模式下的閾值，預設 25
        
    信號輸出：
        - Spot 模式：[0, 1]，負數會被 clip 到 0
        - Futures 模式：[-1, 1]，支援做空
    """
    # ── 參數 ──
    rsi_period = int(params.get("rsi_period", 14))
    min_adx = float(params.get("min_adx", 20))
    adx_period = int(params.get("adx_period", 14))
    
    # 是否支援做空（從 context 取得）
    supports_short = ctx.supports_short if hasattr(ctx, 'supports_short') else False

    close = df["close"]

    # ── 指標計算 ──
    rsi = calculate_rsi(close, rsi_period)
    rsi_prev = rsi.shift(1)

    # ── 閾值計算 (Static vs Dynamic) ──
    rsi_mode = params.get("rsi_mode", "static")
    
    if rsi_mode == "dynamic":
        # Dynamic Thresholds (Rolling Percentile)
        # 預設 14 天 (336 小時)
        lookback_days = int(params.get("rsi_lookback_days", 14))
        bars_per_day = 24  # 假設 1h
        
        # 嘗試從 context 獲取 interval
        interval = ctx.interval if hasattr(ctx, "interval") else "1h"
        
        if interval == '15m': bars_per_day = 96
        elif interval == '30m': bars_per_day = 48
        elif interval == '4h': bars_per_day = 6
        elif interval == '1d': bars_per_day = 1
            
        window = int(params.get("rsi_window_bars", lookback_days * bars_per_day))
        q_low = float(params.get("rsi_quantile_low", 0.10))
        q_high = float(params.get("rsi_quantile_high", 0.90))
        
        rsi_rolling = rsi.rolling(window=window)
        oversold_threshold = rsi_rolling.quantile(q_low)
        overbought_threshold = rsi_rolling.quantile(q_high)
    else:
        # Static Thresholds
        oversold = float(params.get("oversold", 35))
        overbought = float(params.get("overbought", 70))
        # 轉為 Series 以便統一處理
        oversold_threshold = pd.Series(oversold, index=df.index)
        overbought_threshold = pd.Series(overbought, index=df.index)

    # ── 原始信號 ──
    # 做多信號：RSI 上一根 < 閾值 且 當前 >= 閾值（從超賣區回升）
    # 注意：Dynamic 模式下閾值是變動的，比較時也應使用當下的閾值
    long_entry = (rsi_prev < oversold_threshold.shift(1)) & (rsi >= oversold_threshold)
    long_exit = rsi > overbought_threshold
    
    # 做空信號（僅 Futures）：RSI 上一根 > 閾值 且 當前 <= 閾值（從超買區回落）
    short_entry = (rsi_prev > overbought_threshold.shift(1)) & (rsi <= overbought_threshold) if supports_short else pd.Series(False, index=df.index)
    short_exit = rsi < oversold_threshold

    # 狀態機：生成持倉序列
    # 狀態：0 = 空倉，1 = 多倉，-1 = 空倉（做空）
    raw_pos = pd.Series(0.0, index=df.index)
    state = 0  # 0 = flat, 1 = long, -1 = short

    for i in range(len(df)):
        if state == 0:  # 空倉
            if long_entry.iloc[i]:
                state = 1
                raw_pos.iloc[i] = 1.0
            elif supports_short and short_entry.iloc[i]:
                state = -1
                raw_pos.iloc[i] = -1.0
            else:
                raw_pos.iloc[i] = 0.0
        elif state == 1:  # 持有多倉
            if long_exit.iloc[i]:
                # 平倉 → 回到 Flat，不直接反手
                # 靠 cooldown + 新的入場信號再決定方向
                state = 0
                raw_pos.iloc[i] = 0.0
            else:
                raw_pos.iloc[i] = 1.0
        else:  # state == -1，持有空倉
            if short_exit.iloc[i]:
                # 平倉 → 回到 Flat，不直接反手
                state = 0
                raw_pos.iloc[i] = 0.0
            else:
                raw_pos.iloc[i] = -1.0

    # shift(1) 避免未來資訊洩漏
    raw_pos = raw_pos.shift(1).fillna(0.0)

    # ── ADX 趨勢過濾 ──
    filtered_pos = trend_filter(
        df, raw_pos,
        min_adx=min_adx,
        short_min_adx=params.get("short_min_adx"),  # None = 與 min_adx 相同
        adx_period=adx_period,
        require_uptrend=True,
    )

    # ── Funding Rate 過濾 (如果啟用) ──
    if params.get("use_funding_filter", False):
        _fr_applied = False
        try:
            # 嘗試多個可能的 data 目錄
            data_dir = None
            for candidate in [
                Path("data"),
                Path("quant-binance-spot/data"),
                Path(os.getenv("DATA_DIR", "")),
                Path.home() / "quant-binance-spot" / "data",
            ]:
                if candidate.exists() and str(candidate):
                    data_dir = candidate
                    break

            if data_dir is None or not data_dir.exists():
                import logging
                logging.getLogger("strategy").warning(
                    f"⚠️  {ctx.symbol}: Funding Rate 過濾啟用但找不到 data 目錄 "
                    f"(嘗試: data/, $DATA_DIR, ~/quant-binance-spot/data/)"
                )
            else:
                fr_path = get_funding_rate_path(data_dir, ctx.symbol)
                if fr_path.exists():
                    fr_df = load_funding_rates(fr_path)
                    if fr_df is not None and not fr_df.empty:
                        fr_series = align_funding_to_klines(fr_df, df.index)
                        filtered_pos = funding_rate_filter(
                            df, filtered_pos, fr_series,
                            max_positive_rate=float(params.get("fr_max_pos", 0.0002)),
                            max_negative_rate=float(params.get("fr_max_neg", -0.0002)),
                        )
                        _fr_applied = True
                else:
                    import logging
                    logging.getLogger("strategy").warning(
                        f"⚠️  {ctx.symbol}: Funding Rate 資料不存在: {fr_path}"
                    )
        except Exception as e:
            import logging
            logging.getLogger("strategy").warning(
                f"⚠️  {ctx.symbol}: Funding Rate 過濾失敗: {e}"
            )

    # ── 波動率過濾（可選，防止低波動磨耗）──
    min_atr_ratio = params.get("min_atr_ratio")
    if min_atr_ratio is not None:
        vol_mode = params.get("vol_filter_mode", "absolute")
        filtered_pos = volatility_filter(
            df, filtered_pos,
            min_atr_ratio=float(min_atr_ratio),
            atr_period=int(params.get("atr_period", 14)),
            use_percentile=(vol_mode == "percentile"),
            min_percentile=float(params.get("vol_min_percentile", 25)),
        )

    # ── 多時間框架「硬」過濾（可選，在 exit rules 之前阻擋逆趨勢入場）──
    htf_interval = params.get("htf_interval")
    htf_mode = params.get("htf_mode", "soft")
    current_interval = ctx.interval if hasattr(ctx, "interval") else "1h"

    # hard 模式在 exit rules 之前過濾（二元閘門）
    if htf_interval and htf_mode == "hard":
        filtered_pos = htf_trend_filter(
            df, filtered_pos,
            htf_interval=htf_interval,
            ema_fast=int(params.get("htf_ema_fast", 20)),
            ema_slow=int(params.get("htf_ema_slow", 50)),
            current_interval=current_interval,
        )

    # ── ATR 止損 / 止盈 / 移動止損 ──
    # 自適應止損：如果啟用，ER 低（震盪）放寬 SL，ER 高（趨勢）收緊 SL
    adaptive_sl_er = None
    if params.get("adaptive_sl", False):
        from ..indicators.efficiency_ratio import calculate_efficiency_ratio
        _er_period = int(params.get("adaptive_sl_er_period", 10))
        adaptive_sl_er = calculate_efficiency_ratio(df["close"], period=_er_period)

    pos, _exit_exec_prices = apply_exit_rules(
        df, filtered_pos,
        stop_loss_atr=params.get("stop_loss_atr", 2.0),
        take_profit_atr=params.get("take_profit_atr", 3.0),
        trailing_stop_atr=params.get("trailing_stop_atr", None),
        atr_period=int(params.get("atr_period", 14)),
        cooldown_bars=int(params.get("cooldown_bars", 6)),
        min_hold_bars=int(params.get("min_hold_bars", 0)),
        adaptive_sl_er=adaptive_sl_er,
        er_sl_min=float(params.get("er_sl_min", 1.5)),
        er_sl_max=float(params.get("er_sl_max", 3.0)),
    )

    # ── 多時間框架「軟」過濾（在 exit rules 之後縮放倉位大小）──
    # 必須在 exit rules 之後：exit rules 輸出 binary [-1, 0, 1]，
    # 軟過濾器再根據 HTF 趨勢將倉位縮放為連續值 [0.5, 0.75, 1.0]
    if htf_interval and htf_mode == "soft":
        pos = htf_soft_trend_filter(
            df, pos,
            htf_interval=htf_interval,
            ema_fast=int(params.get("htf_ema_fast", 20)),
            ema_slow=int(params.get("htf_ema_slow", 50)),
            current_interval=current_interval,
            align_weight=float(params.get("htf_align_weight", 1.0)),
            counter_weight=float(params.get("htf_counter_weight", 0.5)),
            neutral_weight=float(params.get("htf_neutral_weight", 0.75)),
        )

    # ── Efficiency Ratio 震盪過濾（可選，降低震盪市進場）──
    # 支援 gate（只擋新開倉）和 scale（連續縮放）兩種模式
    er_period = params.get("er_period")
    if er_period is not None:
        er_mode = str(params.get("er_mode", "gate"))
        pos = efficiency_ratio_filter(
            df, pos,
            er_period=int(er_period),
            er_mode=er_mode,
            er_min=float(params.get("er_min", 0.20)),
            er_threshold_low=float(params.get("er_threshold_low", 0.20)),
            er_threshold_high=float(params.get("er_threshold_high", 0.50)),
            weight_at_low=float(params.get("er_weight_at_low", 0.30)),
            weight_at_high=float(params.get("er_weight_at_high", 1.0)),
        )

    # ── 倉位平滑（減少 HTF 權重微調造成的假交易）──
    min_rebalance = params.get("min_rebalance_pct")
    if min_rebalance is not None:
        pos = smooth_positions(pos, min_delta=float(min_rebalance))

    # 根據市場類型 clip 信號範圍
    # Spot: [0, 1]，Futures: [-1, 1]
    if supports_short:
        result = pos.clip(-1.0, 1.0)
    else:
        result = pos.clip(0.0, 1.0)

    # 附加 SL/TP 出場價格，供 run_backtest 使用（消除 look-ahead bias）
    result.attrs['exit_exec_prices'] = _exit_exec_prices
    return result


@register_strategy("rsi_adx_atr_trailing")
def generate_positions_trailing(df: pd.DataFrame, ctx: StrategyContext, params: dict) -> pd.Series:
    """
    同 rsi_adx_atr，但預設啟用移動止損

    移動止損讓利潤奔跑，在趨勢延續時不會過早止盈。
    適合趨勢明顯的行情（如 BTC 單邊牛市）。
    """
    # 預設啟用 trailing stop，取消固定 TP
    params_with_trailing = {
        **params,
        "trailing_stop_atr": params.get("trailing_stop_atr", 2.5),
        "take_profit_atr": params.get("take_profit_atr", None),  # 用 trailing 替代 TP
    }
    return generate_positions(df, ctx, params_with_trailing)
