from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import logging
import warnings
import numpy as np
import pandas as pd
import vectorbt as vbt

from ..strategy.base import StrategyContext
from ..strategy import get_strategy
from ..data.storage import load_klines
from ..data.quality import validate_data_quality, clean_data
from ..risk.risk_limits import RiskLimits, apply_risk_limits
from .metrics import benchmark_buy_and_hold
from .costs import (
    compute_funding_costs,
    adjust_equity_for_funding,
    compute_adjusted_stats,
    compute_volume_slippage,
    FundingCostResult,
    SlippageResult,
)
from ..data.funding_rate import (
    load_funding_rates,
    get_funding_rate_path,
    align_funding_to_klines,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# BacktestResult — 標準化回測輸出（取代 raw dict）
# ══════════════════════════════════════════════════════════════

@dataclass
class BacktestResult:
    """
    標準化的回測結果。

    所有回測入口（single / portfolio / validation / kelly）都返回這個物件，
    確保成本模型一致、欄位不遺漏。
    """
    # ── 核心 ──
    pf: object                          # vbt.Portfolio（策略）
    pf_bh: object                       # vbt.Portfolio（Buy & Hold 基準）
    stats: object                       # pf.stats() 原始統計
    df: pd.DataFrame                    # K 線 DataFrame（已過濾）
    pos: pd.Series                      # 持倉序列

    # ── 成本模型 ──
    funding_cost: FundingCostResult | None = None
    slippage_result: SlippageResult | None = None

    # ── 調整後績效（含 funding 扣除）──
    adjusted_stats: dict | None = None
    adjusted_equity: pd.Series | None = None

    # ── 成本模型配置旗標（用於審計）──
    funding_rate_enabled: bool = False
    slippage_model_enabled: bool = False

    def equity(self) -> pd.Series:
        """取得策略資金曲線（優先用 adjusted，沒有則用原始）"""
        if self.adjusted_equity is not None:
            return self.adjusted_equity
        return self.pf.value()

    def total_return_pct(self) -> float:
        """總回報率 %（含成本調整）"""
        if self.adjusted_stats:
            return self.adjusted_stats.get("Total Return [%]", 0.0)
        return self.stats.get("Total Return [%]", 0.0)

    def sharpe(self) -> float:
        """Sharpe Ratio（含成本調整）"""
        if self.adjusted_stats:
            return self.adjusted_stats.get("Sharpe Ratio", 0.0)
        return self.stats.get("Sharpe Ratio", 0.0)

    def max_drawdown_pct(self) -> float:
        """Max Drawdown %（含成本調整）"""
        if self.adjusted_stats:
            return self.adjusted_stats.get("Max Drawdown [%]", 0.0)
        return abs(self.stats.get("Max Drawdown [%]", 0.0))

    def cost_summary(self) -> str:
        """成本模型摘要字串（用於日誌/報告）"""
        parts = []
        if self.funding_rate_enabled:
            if self.funding_cost:
                parts.append(
                    f"FR: ${self.funding_cost.total_cost:,.0f} "
                    f"({self.funding_cost.total_cost_pct*100:.2f}%)"
                )
            else:
                parts.append("FR: enabled (no data)")
        else:
            parts.append("FR: OFF")

        if self.slippage_model_enabled:
            if self.slippage_result:
                parts.append(
                    f"Slip: avg {self.slippage_result.avg_slippage_bps:.1f}bps"
                )
            else:
                parts.append("Slip: enabled (no data)")
        else:
            parts.append("Slip: fixed")

        return " | ".join(parts)


# ══════════════════════════════════════════════════════════════
# 配置安全驗證
# ══════════════════════════════════════════════════════════════

def validate_backtest_config(cfg: dict) -> None:
    """
    檢查回測配置的安全性，對可疑配置發出警告。

    這個函數在每次 run_symbol_backtest 開頭呼叫，確保使用者不會
    不小心跑出「快樂表」。

    規則：
    1. futures + funding_rate 未啟用 → WARNING
    2. futures + slippage_model 未啟用 → WARNING
    3. 缺少 funding_rate / slippage_model 鍵 → WARNING（可能是手動建構的 dict）
    """
    market_type = cfg.get("market_type", "spot")

    if market_type != "futures":
        return  # Spot 不需要 funding rate / volume slippage 檢查

    # ── Funding Rate ──
    fr_cfg = cfg.get("funding_rate")
    if fr_cfg is None:
        warnings.warn(
            "⚠️  Futures 回測缺少 'funding_rate' 配置！"
            "結果將不包含 funding 成本，可能嚴重高估收益。"
            "請用 cfg.to_backtest_dict() 產生配置，或手動加入 "
            "funding_rate: {enabled: true}。",
            UserWarning,
            stacklevel=3,
        )
    elif not fr_cfg.get("enabled", False):
        warnings.warn(
            "⚠️  Futures 回測 funding_rate.enabled=false！"
            "Funding rate 年化成本約 5-15%，關閉會嚴重高估收益。"
            "如需快速迭代，請使用 --simple flag 明確標記。",
            UserWarning,
            stacklevel=3,
        )

    # ── Slippage Model ──
    sm_cfg = cfg.get("slippage_model")
    if sm_cfg is None:
        warnings.warn(
            "⚠️  Futures 回測缺少 'slippage_model' 配置！"
            "將使用固定滑點，小幣種可能嚴重低估實際滑點。"
            "請用 cfg.to_backtest_dict() 產生配置。",
            UserWarning,
            stacklevel=3,
        )
    elif not sm_cfg.get("enabled", False):
        logger.info(
            "ℹ️  Volume slippage model 未啟用，使用固定 slippage_bps。"
            "如需更精確的成本估算，設定 slippage_model.enabled=true。"
        )


# ══════════════════════════════════════════════════════════════
# Shared constants — 所有回測相關模組共用
# ══════════════════════════════════════════════════════════════

# 我們的 direction → vectorbt direction 映射
VBT_DIRECTION_MAP: dict[str, str] = {
    "both": "both",
    "long_only": "longonly",
    "short_only": "shortonly",
}


def to_vbt_direction(direction: str) -> str:
    """將我們的 direction 字串轉為 vectorbt 接受的格式"""
    return VBT_DIRECTION_MAP.get(direction, "longonly")


def clip_positions_by_direction(
    pos: pd.Series,
    market_type: str,
    direction: str,
) -> pd.Series:
    """
    根據 market_type / direction 過濾持倉信號
    
    - spot / long_only  → clip 掉做空信號
    - short_only        → 轉換符號讓 vectorbt shortonly 正確運作
    - both              → 不做處理
    """
    if market_type == "spot" or direction == "long_only":
        return pos.clip(lower=0.0)
    elif direction == "short_only":
        # vectorbt shortonly: size>0 = 開空, size<0 = 平空
        # 策略的 pos=-1 表示做空 → 轉換為 +1
        return (-pos).clip(lower=0.0)
    return pos  # "both": 保留 [-1, 1]


def _bps_to_pct(bps: float) -> float:
    return bps / 10_000.0


def _resolve_backtest_params(cfg: dict, **kwargs) -> dict:
    """
    從 cfg dict + explicit kwargs 解析回測參數
    
    explicit kwargs 優先（如果傳入非 None 值），否則 fallback 到 cfg dict。
    這樣無論呼叫者是用 explicit args 還是 cfg dict 都能正確運作。
    """
    return {
        "market_type": kwargs.get("market_type") or cfg.get("market_type", "spot"),
        "direction": kwargs.get("direction") or cfg.get("direction", "both"),
        "validate_data": kwargs.get("validate_data") if kwargs.get("validate_data") is not None else cfg.get("validate_data", True),
        "clean_data_before": kwargs.get("clean_data_before") if kwargs.get("clean_data_before") is not None else cfg.get("clean_data_before", True),
        "start": kwargs.get("start") or cfg.get("start"),
        "end": kwargs.get("end") or cfg.get("end"),
    }


def _apply_date_filter(
    df: pd.DataFrame,
    pos: pd.Series,
    start: str | None,
    end: str | None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    根據 start / end 日期過濾數據和持倉信號
    
    策略在完整數據上計算（確保指標 warmup 正確），
    之後只截取 [start, end] 區間送入 VBT 回測。
    
    這樣做的好處：
    1. 指標不會有 NaN warmup 問題
    2. 回測結果只反映指定時間範圍
    3. Total Return / Sharpe / MDD 等指標更精確
    """
    if start is None and end is None:
        return df, pos
    
    original_len = len(df)
    
    if start is not None:
        start_ts = pd.Timestamp(start, tz="UTC") if df.index.tz is not None else pd.Timestamp(start)
        mask = df.index >= start_ts
        df = df.loc[mask]
        pos = pos.loc[mask]
    
    if end is not None:
        end_ts = pd.Timestamp(end, tz="UTC") if df.index.tz is not None else pd.Timestamp(end)
        mask = df.index <= end_ts
        df = df.loc[mask]
        pos = pos.loc[mask]
    
    if len(df) < original_len:
        logger.info(
            f"📅 日期過濾: {original_len} → {len(df)} bars "
            f"({df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')})"
        )
    
    return df, pos


# ══════════════════════════════════════════════════════════════
# 波動率目標倉位縮放
# ══════════════════════════════════════════════════════════════

def _apply_vol_scaling(
    pos: pd.Series,
    df: pd.DataFrame,
    target_vol: float = 0.15,
    vol_lookback: int = 168,
    max_scale: float = 2.0,
    min_scale: float = 0.1,
    interval: str = "1h",
) -> pd.Series:
    """
    根據實現波動率反向縮放倉位（Volatility Targeting）

    高波動期 → 降低倉位，低波動期 → 提高倉位（但不超過 max_scale）
    
    公式: scale = target_vol / realized_vol
    
    Args:
        pos: 原始信號 [-1, 1]
        df: K 線 DataFrame（需要 close 欄位）
        target_vol: 目標年化波動率（預設 15%）
        vol_lookback: 波動率計算回看期（bar 數）
        max_scale: 最大縮放倍數
        min_scale: 最小縮放倍數
        interval: 時間間隔（用於年化）
    
    Returns:
        縮放後的倉位信號（連續值）
    """
    # 根據 interval 決定年化因子
    annualize_factors = {
        "1m": np.sqrt(525_600),
        "5m": np.sqrt(105_120),
        "15m": np.sqrt(35_040),
        "1h": np.sqrt(8_760),
        "4h": np.sqrt(2_190),
        "1d": np.sqrt(365),
    }
    annualize = annualize_factors.get(interval, np.sqrt(8_760))
    
    returns = df["close"].pct_change()
    realized_vol = returns.rolling(window=vol_lookback).std() * annualize
    
    # 避免除以零 & warmup 期用 target_vol
    realized_vol = realized_vol.replace(0, np.nan).ffill().fillna(target_vol)
    
    scale = (target_vol / realized_vol).clip(lower=min_scale, upper=max_scale)
    
    scaled_pos = pos * scale
    # 最終仍然限制在 [-1, 1]
    scaled_pos = scaled_pos.clip(lower=-1.0, upper=1.0)
    
    logger.info(
        f"📊 Vol Targeting: target={target_vol:.0%}, "
        f"avg_realized={realized_vol.mean():.1%}, "
        f"avg_scale={scale.mean():.2f}, "
        f"avg_|pos|={scaled_pos.abs().mean():.3f}"
    )
    
    return scaled_pos


# ══════════════════════════════════════════════════════════════
# 核心回測函數
# ══════════════════════════════════════════════════════════════

def run_symbol_backtest(
    symbol: str,
    data_path: Path,
    cfg: dict,
    strategy_name: str = None,
    validate_data: Optional[bool] = None,
    clean_data_before: Optional[bool] = None,
    risk_limits: Optional[RiskLimits] = None,
    market_type: str | None = None,
    direction: str | None = None,
    data_dir: Path | None = None,
) -> BacktestResult:
    """
    運行單個交易對的回測（唯一的 VBT Portfolio 構建入口）

    **重要**：所有回測路徑（單幣 / 組合 / 驗證 / Kelly）都必須透過
    這個函數來產生 VBT Portfolio，確保成本模型一致。

    Args:
        symbol: 交易對
        data_path: K 線數據路徑
        cfg: 配置字典（建議用 AppConfig.to_backtest_dict() 產生）
        strategy_name: 策略名稱
        validate_data: 是否驗證數據
        clean_data_before: 是否清洗數據
        risk_limits: 風險限制
        market_type: "spot" 或 "futures"（None → 從 cfg 讀取，預設 "spot"）
        direction: "both" / "long_only" / "short_only"（None → 從 cfg 讀取）
        data_dir: 數據根目錄（用於載入 funding rate 等輔助數據）

    Returns:
        BacktestResult（標準化回測結果）
    """
    # ── 配置安全驗證（防止「快樂表」）──
    validate_backtest_config(cfg)

    df = load_klines(data_path)

    # 解析參數（explicit args 優先，fallback 到 cfg dict）
    resolved = _resolve_backtest_params(
        cfg,
        market_type=market_type,
        direction=direction,
        validate_data=validate_data,
        clean_data_before=clean_data_before,
    )
    mt = resolved["market_type"]
    dr = resolved["direction"]

    # 數據質量檢查
    if resolved["validate_data"]:
        quality_report = validate_data_quality(df)
        if not quality_report.is_valid:
            print(f"⚠️  警告: {symbol} 數據質量問題")
            for error in quality_report.errors:
                print(f"  - {error}")
            for warning in quality_report.warnings:
                print(f"  - {warning}")

    # 數據清洗
    if resolved["clean_data_before"]:
        df = clean_data(df, fill_method="forward", remove_outliers=False, remove_duplicates=True)

    # ── trade_on=next_open → signal_delay=1（消除 look-ahead bias）──
    trade_on = cfg.get("trade_on", "next_open")
    signal_delay = 1 if trade_on == "next_open" else 0

    ctx = StrategyContext(
        symbol=symbol,
        interval=cfg.get("interval", "1h"),
        market_type=mt,
        direction=dr,
        signal_delay=signal_delay,
    )

    # 獲取策略函數
    strategy_name = strategy_name or cfg.get("strategy_name", "rsi_adx_atr")
    strategy_func = get_strategy(strategy_name)

    # 自動注入 _data_dir（讓策略可以自動載入 OI/FR 等輔助數據）
    if data_dir is not None and "_data_dir" not in cfg["strategy_params"]:
        cfg["strategy_params"]["_data_dir"] = data_dir

    # positions: [-1, 1] (Futures) 或 [0, 1] (Spot)
    pos = strategy_func(df, ctx, cfg["strategy_params"])

    # ── Overlay 後處理（可選）──────────────────────────
    # 如果 cfg 中有 overlay 配置，在 direction clip 前套用
    # 這確保 overlay 也自動整合到 walk-forward pipeline
    overlay_cfg = cfg.get("overlay")
    if overlay_cfg and overlay_cfg.get("enabled", False):
        from ..strategy.overlays.oi_vol_exit_overlay import apply_overlay_by_mode
        overlay_mode = overlay_cfg.get("mode", "vol_pause")
        overlay_params = overlay_cfg.get("params", {})

        # OI 資料：優先使用呼叫者注入的 _oi_series，否則自動從 data_dir 載入
        # 支援複合模式：如 "oi_vol+lsr_confirmatory" 也需載入 OI
        oi_series = cfg.get("_oi_series")
        _needs_oi = any(m in overlay_mode for m in ("oi_only", "oi_vol"))
        if oi_series is None and _needs_oi and data_dir:
            from ..data.open_interest import get_oi_path, load_open_interest, align_oi_to_klines
            for _prov in ["merged", "coinglass", "binance"]:
                _oi_path = get_oi_path(data_dir, symbol, _prov)
                _oi_df = load_open_interest(_oi_path)
                if _oi_df is not None and not _oi_df.empty:
                    oi_series = align_oi_to_klines(_oi_df, df.index, max_ffill_bars=2)
                    break

        # LSR 資料：overlay mode 含 lsr_confirmatory 時自動載入
        if "lsr_confirmatory" in overlay_mode and "_lsr_series" not in overlay_params:
            if data_dir:
                try:
                    from ..data.long_short_ratio import load_lsr, align_lsr_to_klines
                    lsr_type = overlay_params.get("lsr_type", "lsr")
                    deriv_dir = data_dir / "binance" / "futures" / "derivatives"
                    lsr_raw = load_lsr(symbol, lsr_type, data_dir=deriv_dir)
                    if lsr_raw is not None and not lsr_raw.empty:
                        lsr_aligned = align_lsr_to_klines(lsr_raw, df.index, max_ffill_bars=2)
                        overlay_params["_lsr_series"] = lsr_aligned
                        logger.debug(f"  {symbol}: overlay LSR 載入成功 ({len(lsr_raw)} rows)")
                    else:
                        logger.warning(f"  {symbol}: overlay LSR 數據不存在 ({lsr_type})")
                except Exception as e:
                    logger.warning(f"  {symbol}: overlay LSR 載入失敗: {e}")

        # OI 確認層數據：lsr_confirmatory + oi_confirm_enabled 時載入 OI 到 overlay_params
        if ("lsr_confirmatory" in overlay_mode
                and overlay_params.get("oi_confirm_enabled", False)
                and "_oi_series" not in overlay_params):
            if data_dir:
                try:
                    from ..data.open_interest import get_oi_path, load_open_interest, align_oi_to_klines
                    for _prov in ["merged", "coinglass", "binance"]:
                        _oi_path = get_oi_path(data_dir, symbol, _prov)
                        _oi_df = load_open_interest(_oi_path)
                        if _oi_df is not None and not _oi_df.empty:
                            overlay_params["_oi_series"] = align_oi_to_klines(
                                _oi_df, df.index, max_ffill_bars=2
                            )
                            logger.debug(f"  {symbol}: overlay OI (for LSR confirm) 載入成功")
                            break
                except Exception as e:
                    logger.warning(f"  {symbol}: overlay OI (for LSR confirm) 載入失敗: {e}")

        # FR 確認層數據：lsr_confirmatory + fr_confirm_enabled 時載入 FR 到 overlay_params
        if ("lsr_confirmatory" in overlay_mode
                and overlay_params.get("fr_confirm_enabled", False)
                and "_fr_series" not in overlay_params):
            if data_dir:
                try:
                    fr_path = get_funding_rate_path(data_dir, symbol)
                    funding_df = load_funding_rates(fr_path)
                    if funding_df is not None and not funding_df.empty:
                        # 使用原始 funding rate 值（非對齊到結算時刻），用於 pctrank 計算
                        fr_col = "fundingRate" if "fundingRate" in funding_df.columns else funding_df.columns[0]
                        fr_series = funding_df[fr_col]
                        fr_aligned = fr_series.reindex(df.index, method="ffill")
                        overlay_params["_fr_series"] = fr_aligned
                        logger.debug(f"  {symbol}: overlay FR (for LSR confirm) 載入成功 ({len(funding_df)} rows)")
                    else:
                        logger.warning(f"  {symbol}: overlay FR 數據不存在")
                except Exception as e:
                    logger.warning(f"  {symbol}: overlay FR (for LSR confirm) 載入失敗: {e}")

        pos = apply_overlay_by_mode(
            position=pos,
            price_df=df,
            oi_series=oi_series,
            params=overlay_params,
            mode=overlay_mode,
        )
        logger.info(f"📊 Overlay applied: mode={overlay_mode}")

    # 根據 direction 過濾信號（使用共用函數）
    pos = clip_positions_by_direction(pos, mt, dr)

    # ── 倉位縮放（position sizing）──────────────────
    # 與實盤 runner 的 _apply_position_sizing 一致：
    #   - fixed:      pos *= position_pct（預設 1.0，不縮放）
    #   - kelly:      pos *= kelly_pct * kelly_fraction（使用配置的 win_rate/avg_win/avg_loss）
    #   - volatility: pos *= target_vol / realized_vol
    ps_cfg = cfg.get("position_sizing", {})
    ps_method = ps_cfg.get("method", "fixed")
    ps_pct = ps_cfg.get("position_pct", 1.0)

    if ps_method == "kelly":
        kelly_fraction = ps_cfg.get("kelly_fraction", 0.25)
        win_rate = ps_cfg.get("win_rate")
        avg_win = ps_cfg.get("avg_win")
        avg_loss = ps_cfg.get("avg_loss")

        if win_rate is not None and avg_win is not None and avg_loss is not None:
            # 使用配置提供的統計（與 live runner 一致）
            try:
                from ..risk.position_sizing import KellyPositionSizer
                ks = KellyPositionSizer(
                    win_rate=win_rate, avg_win=avg_win, avg_loss=avg_loss,
                    kelly_fraction=kelly_fraction,
                )
                scale = ks.kelly_pct  # 已含 fraction
                if scale > 0:
                    logger.info(
                        f"📊 Position Sizing [kelly]: "
                        f"win_rate={win_rate:.1%}, W/L={avg_win/avg_loss:.2f}, "
                        f"fraction={kelly_fraction}, scale={scale:.2f}"
                    )
                    pos = pos * scale
                else:
                    logger.info("📊 Position Sizing [kelly]: edge ≤ 0，不縮放")
            except ValueError as e:
                logger.warning(f"⚠️  Kelly 參數無效: {e}，不縮放")
        else:
            logger.info(
                "📊 Position Sizing [kelly]: 未設定 win_rate/avg_win/avg_loss，"
                "不縮放（請先跑 kelly_validation 取得統計後填入配置）"
            )
    elif ps_method == "volatility":
        target_vol = ps_cfg.get("target_volatility", 0.15)
        vol_lookback = ps_cfg.get("vol_lookback", 168)
        pos = _apply_vol_scaling(
            pos, df,
            target_vol=target_vol,
            vol_lookback=vol_lookback,
            interval=cfg.get("interval", "1h"),
        )
    elif ps_pct < 1.0:
        # fixed 但 position_pct < 1.0 → 線性縮放
        logger.info(f"📊 Position Sizing [fixed]: scale={ps_pct:.2f}")
        pos = pos * ps_pct

    # 應用風險限制（如果提供）
    if risk_limits is not None:
        equity_curve = (1 + df["close"].pct_change()).cumprod() * cfg["initial_cash"]
        adjusted_pos = pd.Series(0.0, index=pos.index)
        for i in range(len(pos)):
            current_equity = equity_curve.iloc[i] if i < len(equity_curve) else cfg["initial_cash"]
            adjusted_pos.iloc[i], _ = apply_risk_limits(
                pos.iloc[i],
                equity_curve.iloc[:i+1],
                risk_limits,
                current_equity=current_equity
            )
        pos = adjusted_pos

    # ── 日期過濾 ──────────────────────────────────────
    # 策略已在完整數據上計算完畢（確保指標 warmup），
    # 現在截取 [start, end] 區間送入 VBT 回測
    df, pos = _apply_date_filter(df, pos, resolved.get("start"), resolved.get("end"))

    close = df["close"]
    open_ = df["open"]
    fee = _bps_to_pct(cfg["fee_bps"])

    # ── 構建執行價格（消除 SL/TP look-ahead bias）──────
    # exit_exec_prices: SL/TP 觸發時為實際出場價，其餘為 NaN
    exit_exec_prices = pos.attrs.get("exit_exec_prices")
    if exit_exec_prices is not None:
        # 對齊到日期過濾後的索引
        exit_exec_prices = exit_exec_prices.reindex(pos.index)
        # 自定義執行價格: SL/TP bar 使用出場價，其餘用 open
        exec_price = open_.copy()
        sl_tp_mask = exit_exec_prices.notna()
        exec_price[sl_tp_mask] = exit_exec_prices[sl_tp_mask]
        logger.info(
            f"🔧 SL/TP 出場價修正: {sl_tp_mask.sum()} bars 使用實際 SL/TP 價格"
        )
    else:
        exec_price = open_

    # ── 滑點模型 ──────────────────────────────────────
    sm_cfg = cfg.get("slippage_model", {})
    slippage_result: SlippageResult | None = None

    if sm_cfg.get("enabled", False):
        leverage = cfg.get("leverage", 1)
        slippage_result = compute_volume_slippage(
            pos=pos,
            df=df,
            capital=cfg["initial_cash"],
            base_bps=sm_cfg.get("base_bps", 2.0),
            impact_coefficient=sm_cfg.get("impact_coefficient", 0.1),
            impact_power=sm_cfg.get("impact_power", 0.5),
            adv_lookback=sm_cfg.get("adv_lookback", 20),
            participation_rate=sm_cfg.get("participation_rate", 0.10),
            leverage=leverage,
        )
        slippage = slippage_result.slippage_array
        logger.info(
            f"📊 {symbol} Volume slippage: "
            f"avg={slippage_result.avg_slippage_bps:.1f}bps, "
            f"max={slippage_result.max_slippage_bps:.1f}bps, "
            f"high_impact={slippage_result.high_impact_bars} bars"
        )
    else:
        slippage = _bps_to_pct(cfg["slippage_bps"])

    # ── 策略 Portfolio ─────────────────────────────
    vbt_direction = to_vbt_direction(dr)
    
    pf = vbt.Portfolio.from_orders(
        close=close,
        size=pos,
        size_type="targetpercent",
        price=exec_price,
        fees=fee,
        slippage=slippage,
        init_cash=cfg["initial_cash"],
        freq=cfg.get("interval", "1h"),
        direction=vbt_direction,
    )

    # ── Buy & Hold 基準 ────────────────────────────
    pf_bh = benchmark_buy_and_hold(
        df,
        initial_cash=cfg["initial_cash"],
        fee_bps=cfg["fee_bps"],
        slippage_bps=cfg["slippage_bps"],
        interval=cfg.get("interval", "1h"),
    )

    stats = pf.stats()

    # ── Funding Rate 成本模型 ──────────────────────
    fr_cfg = cfg.get("funding_rate", {})
    funding_cost: FundingCostResult | None = None
    adjusted_stats: dict | None = None
    adjusted_equity: pd.Series | None = None

    if fr_cfg.get("enabled", False) and mt == "futures":
        # 嘗試載入歷史 funding rate
        funding_df = None
        if fr_cfg.get("use_historical", True) and data_dir is not None:
            fr_path = get_funding_rate_path(data_dir, symbol)
            funding_df = load_funding_rates(fr_path)
            if funding_df is not None:
                logger.info(f"📥 {symbol} 載入歷史 funding rate: {len(funding_df)} records")
            else:
                logger.info(f"ℹ️  {symbol} 無歷史 funding rate，使用預設費率")

        # 對齊到 kline 時間軸
        funding_rates = align_funding_to_klines(
            funding_df,
            df.index,
            default_rate_8h=fr_cfg.get("default_rate_8h", 0.0001),
        )

        # 計算 funding 成本
        leverage = cfg.get("leverage", 1)
        equity = pf.value()
        funding_cost = compute_funding_costs(
            pos=pos,
            equity=equity,
            funding_rates=funding_rates,
            leverage=leverage,
        )

        # 調整後的資金曲線和統計
        adjusted_equity = adjust_equity_for_funding(equity, funding_cost)
        adjusted_stats = compute_adjusted_stats(adjusted_equity, cfg["initial_cash"])

        logger.info(
            f"💰 {symbol} Funding cost: "
            f"total=${funding_cost.total_cost:,.2f} "
            f"({funding_cost.total_cost_pct*100:.2f}%), "
            f"annualized={funding_cost.annualized_cost_pct*100:.2f}%/yr, "
            f"settlements={funding_cost.n_settlements}"
        )

    result = BacktestResult(
        pf=pf,
        pf_bh=pf_bh,
        stats=stats,
        df=df,
        pos=pos,
        funding_cost=funding_cost,
        slippage_result=slippage_result,
        adjusted_stats=adjusted_stats,
        adjusted_equity=adjusted_equity,
        funding_rate_enabled=fr_cfg.get("enabled", False),
        slippage_model_enabled=sm_cfg.get("enabled", False),
    )

    logger.info(f"📊 {symbol} 回測完成 [{result.cost_summary()}]")
    return result
