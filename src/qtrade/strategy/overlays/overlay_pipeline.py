"""
Overlay Pipeline — 共用的 overlay 資料載入 + 套用流程

**唯一入口**：`prepare_and_apply_overlay()`

backtest（run_backtest.py）和 live（signal_generator.py）都呼叫此函數，
確保：
  1. OI/LSR/FR 資料載入邏輯只寫一次（消除 Shotgun Surgery）
  2. OI provider 搜尋順序一致（使用 OI_PROVIDER_SEARCH_ORDER）
  3. overlay_params 必定 deepcopy（防止 per-symbol 交叉汙染）
"""
from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from ...data.open_interest import (
    OI_PROVIDER_SEARCH_ORDER,
    get_oi_path,
    load_open_interest,
    align_oi_to_klines,
)
from .oi_vol_exit_overlay import apply_overlay_by_mode

logger = logging.getLogger(__name__)


def prepare_and_apply_overlay(
    pos: pd.Series,
    df: pd.DataFrame,
    overlay_cfg: dict,
    symbol: str,
    *,
    data_dir: Path | str | None = None,
    injected_oi_series: pd.Series | None = None,
    extra_params: dict[str, Any] | None = None,
) -> pd.Series:
    """
    載入所需的輔助資料（OI / LSR / FR）並套用 overlay。

    此函數是 backtest 和 live pipeline 的**唯一共用入口**，
    確保兩邊行為完全一致。

    Args:
        pos:               策略原始持倉信號 [-1, 1]
        df:                K 線 DataFrame（含 DatetimeIndex）
        overlay_cfg:       overlay 配置 dict（含 enabled, mode, params）
        symbol:            交易對名稱
        data_dir:          數據根目錄（用於從磁碟載入 OI/LSR/FR）
        injected_oi_series: 呼叫者已注入的 OI（來自記憶體快取或 cfg）
        extra_params:      呼叫者已注入的額外參數（_oi_series, _lsr_series 等）

    Returns:
        overlay 處理後的持倉信號
    """
    if not overlay_cfg or not overlay_cfg.get("enabled", False):
        return pos

    overlay_mode = overlay_cfg.get("mode", "vol_pause")
    # ⚠️ 必須 deepcopy：防止 per-symbol 的 _lsr_series / _oi_series / _fr_series
    # 注入後汙染共用 dict（portfolio backtest 會跑多個 symbol）
    overlay_params = copy.deepcopy(overlay_cfg.get("params", {}))

    data_dir_path = Path(data_dir) if data_dir is not None else None

    # 合併呼叫者已注入的額外參數（如 _oi_series, _lsr_series）
    if extra_params:
        for k, v in extra_params.items():
            if k not in overlay_params:
                overlay_params[k] = v

    # ── OI 資料載入 ──────────────────────────────────────
    oi_series = injected_oi_series
    _needs_oi = any(m in overlay_mode for m in ("oi_only", "oi_vol"))
    if oi_series is None and _needs_oi and data_dir_path:
        oi_series = _load_oi(data_dir_path, symbol, df.index)

    # ── LSR 資料載入 ──────────────────────────────────────
    if "lsr_confirmatory" in overlay_mode and "_lsr_series" not in overlay_params:
        if data_dir_path:
            _load_lsr_into_params(data_dir_path, symbol, df.index, overlay_params)

    # ── OI 確認層（LSR confirmatory + oi_confirm_enabled）──
    if ("lsr_confirmatory" in overlay_mode
            and overlay_params.get("oi_confirm_enabled", False)
            and "_oi_series" not in overlay_params):
        if data_dir_path:
            _oi = _load_oi(data_dir_path, symbol, df.index)
            if _oi is not None:
                overlay_params["_oi_series"] = _oi

    # ── FR 確認層（LSR confirmatory + fr_confirm_enabled）──
    if ("lsr_confirmatory" in overlay_mode
            and overlay_params.get("fr_confirm_enabled", False)
            and "_fr_series" not in overlay_params):
        if data_dir_path:
            _load_fr_into_params(data_dir_path, symbol, df.index, overlay_params)

    # ── 保存策略指標（overlay 操作會產生新 Series 丟失 attrs）──
    saved_indicators = getattr(pos, "attrs", {}).get("indicators")

    pos = apply_overlay_by_mode(
        position=pos,
        price_df=df,
        oi_series=oi_series,
        params=overlay_params,
        mode=overlay_mode,
    )

    # 恢復策略指標
    if saved_indicators:
        pos.attrs["indicators"] = saved_indicators

    logger.info(f"📊 Overlay applied: mode={overlay_mode}")
    return pos


# ══════════════════════════════════════════════════════════════
# 內部：資料載入 helper（不要在外部直接呼叫）
# ══════════════════════════════════════════════════════════════


def _load_oi(
    data_dir: Path,
    symbol: str,
    target_index: pd.DatetimeIndex,
) -> pd.Series | None:
    """
    依 OI_PROVIDER_SEARCH_ORDER 搜尋並載入 OI 資料。

    Returns:
        對齊到 target_index 的 OI Series，或 None（找不到資料）。
    """
    for prov in OI_PROVIDER_SEARCH_ORDER:
        oi_path = get_oi_path(data_dir, symbol, prov)
        oi_df = load_open_interest(oi_path)
        if oi_df is not None and not oi_df.empty:
            logger.debug(f"  {symbol}: overlay OI 載入成功 (provider={prov})")
            return align_oi_to_klines(oi_df, target_index, max_ffill_bars=2)

    logger.warning(f"  {symbol}: overlay OI 無法載入（所有 provider 均無資料）")
    return None


def _load_lsr_into_params(
    data_dir: Path,
    symbol: str,
    target_index: pd.DatetimeIndex,
    overlay_params: dict,
) -> None:
    """載入 LSR 資料並注入到 overlay_params["_lsr_series"]。"""
    try:
        from ...data.long_short_ratio import load_lsr, align_lsr_to_klines

        deriv_dir = data_dir / "binance" / "futures" / "derivatives"
        lsr_type = overlay_params.get("lsr_type", "lsr")
        lsr_raw = load_lsr(symbol, lsr_type, data_dir=deriv_dir)
        if lsr_raw is not None and not lsr_raw.empty:
            lsr_aligned = align_lsr_to_klines(lsr_raw, target_index, max_ffill_bars=2)
            overlay_params["_lsr_series"] = lsr_aligned
            logger.debug(f"  {symbol}: overlay LSR 載入成功 ({len(lsr_raw)} rows)")
        else:
            logger.warning(f"  {symbol}: overlay LSR 數據不存在 ({lsr_type})")
    except Exception as e:
        logger.warning(f"  {symbol}: overlay LSR 載入失敗: {e}")


def _load_fr_into_params(
    data_dir: Path,
    symbol: str,
    target_index: pd.DatetimeIndex,
    overlay_params: dict,
) -> None:
    """載入 Funding Rate 資料並注入到 overlay_params["_fr_series"]。"""
    try:
        from ...data.funding_rate import load_funding_rates, get_funding_rate_path

        fr_path = get_funding_rate_path(data_dir, symbol)
        funding_df = load_funding_rates(fr_path)
        if funding_df is not None and not funding_df.empty:
            fr_col = (
                "fundingRate"
                if "fundingRate" in funding_df.columns
                else funding_df.columns[0]
            )
            fr_series = funding_df[fr_col]
            fr_aligned = fr_series.reindex(target_index, method="ffill")
            overlay_params["_fr_series"] = fr_aligned
            logger.debug(
                f"  {symbol}: overlay FR 載入成功 ({len(funding_df)} rows)"
            )
        else:
            logger.warning(f"  {symbol}: overlay FR 數據不存在")
    except Exception as e:
        logger.warning(f"  {symbol}: overlay FR 載入失敗: {e}")
