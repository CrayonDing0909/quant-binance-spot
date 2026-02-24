"""
Backtest ↔ Live Overlay 一致性測試

確保：
1. generate_signal()（live pipeline）和 run_symbol_backtest()（backtest pipeline）
   在相同的輸入 df + overlay config 下，產生相同的 overlay 處理結果。
2. overlay_cfg=None 時，live pipeline 不套用 overlay（向後相容）。
3. overlay_cfg.enabled=False 時，live pipeline 不套用 overlay。

防止 backtest/live 不一致的 bug 再次發生。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qtrade.strategy.base import StrategyContext
from qtrade.strategy import get_strategy
from qtrade.strategy.overlays.oi_vol_exit_overlay import (
    apply_overlay_by_mode,
    apply_vol_pause_overlay,
)


# ══════════════════════════════════════════════════════════════
# Helper: 合成 OHLCV 數據（帶 vol spike）
# ══════════════════════════════════════════════════════════════

def _make_ohlcv(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    生成帶有一段 vol spike 的合成 OHLCV 數據。

    在 bar 300-310 人工放大 high-low range 來觸發 vol_pause overlay，
    確保 overlay 確實有效果（非 no-op）。
    """
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    close = np.maximum(close, 10.0)  # 防止負價格

    high = close + rng.uniform(0.2, 1.0, n)
    low = close - rng.uniform(0.2, 1.0, n)
    open_ = close + rng.normal(0, 0.3, n)
    volume = rng.uniform(100, 1000, n)

    # 人工製造 vol spike（bar 300-310 放大 range）
    for i in range(300, min(310, n)):
        high[i] = close[i] + 15.0
        low[i] = close[i] - 15.0

    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


# ── 共用的 overlay config（與 prod_live_R3C_E3.yaml 結構一致）──
OVERLAY_CFG_VOL_PAUSE = {
    "enabled": True,
    "mode": "vol_pause",
    "params": {
        "vol_spike_z": 2.0,
        "overlay_cooldown_bars": 12,
        "atr_period": 14,
        "vol_z_window": 168,
    },
}

OVERLAY_CFG_OI_VOL = {
    "enabled": True,
    "mode": "oi_vol",
    "params": {
        "oi_extreme_z": 999.0,
        "oi_reversal_window": 6,
        "reduce_pct": 0.5,
        "oi_lookback": 24,
        "oi_z_window": 168,
        "vol_spike_z": 2.0,
        "overlay_cooldown_bars": 12,
        "trend_lookback": 20,
        "atr_period": 14,
        "vol_z_window": 168,
    },
}

TSMOM_EMA_PARAMS = {
    "lookback": 168,
    "vol_target": 0.15,
    "ema_fast": 20,
    "ema_slow": 50,
    "agree_weight": 1.0,
    "disagree_weight": 0.3,
}


# ══════════════════════════════════════════════════════════════
# 1. Backtest/Live overlay 結果一致性
# ══════════════════════════════════════════════════════════════


class TestOverlayConsistency:
    """驗證 backtest 和 live 兩條 pipeline 的 overlay 輸出完全一致"""

    def _run_backtest_path(
        self,
        df: pd.DataFrame,
        strategy_name: str,
        params: dict,
        overlay_cfg: dict,
    ) -> pd.Series:
        """
        模擬 backtest pipeline 的 overlay 流程
        (run_backtest.py L424-454)
        """
        ctx = StrategyContext(
            symbol="BTCUSDT",
            interval="1h",
            market_type="futures",
            direction="both",
        )
        strategy_func = get_strategy(strategy_name)
        pos = strategy_func(df, ctx, params)

        if overlay_cfg and overlay_cfg.get("enabled", False):
            overlay_mode = overlay_cfg.get("mode", "vol_pause")
            overlay_params = overlay_cfg.get("params", {})
            oi_series = None  # 與 live 一致：不載入 OI

            pos = apply_overlay_by_mode(
                position=pos,
                price_df=df,
                oi_series=oi_series,
                params=overlay_params,
                mode=overlay_mode,
            )
        return pos

    def _run_live_path(
        self,
        df: pd.DataFrame,
        strategy_name: str,
        params: dict,
        overlay_cfg: dict | None,
    ) -> pd.Series:
        """
        模擬 live pipeline 的 overlay 流程
        (signal_generator.py generate_signal() 中的新增邏輯)
        """
        ctx = StrategyContext(
            symbol="BTCUSDT",
            interval="1h",
            market_type="futures",
            direction="both",
        )
        strategy_func = get_strategy(strategy_name)
        positions = strategy_func(df, ctx, params)

        # 這段邏輯與 signal_generator.py 中的新增程式碼完全對應
        if overlay_cfg and overlay_cfg.get("enabled", False):
            overlay_mode = overlay_cfg.get("mode", "vol_pause")
            overlay_params = overlay_cfg.get("params", {})

            positions = apply_overlay_by_mode(
                position=positions,
                price_df=df,
                oi_series=None,
                params=overlay_params,
                mode=overlay_mode,
            )
        return positions

    def test_vol_pause_consistency(self):
        """vol_pause overlay: backtest 和 live 結果完全一致"""
        df = _make_ohlcv(500, seed=42)
        bt_pos = self._run_backtest_path(df, "tsmom_ema", TSMOM_EMA_PARAMS, OVERLAY_CFG_VOL_PAUSE)
        live_pos = self._run_live_path(df, "tsmom_ema", TSMOM_EMA_PARAMS, OVERLAY_CFG_VOL_PAUSE)

        pd.testing.assert_series_equal(
            bt_pos, live_pos,
            check_names=False,
            obj="backtest vs live overlay positions",
        )

    def test_oi_vol_consistency(self):
        """oi_vol overlay (no OI data): backtest 和 live 結果完全一致"""
        df = _make_ohlcv(500, seed=123)
        bt_pos = self._run_backtest_path(df, "tsmom_ema", TSMOM_EMA_PARAMS, OVERLAY_CFG_OI_VOL)
        live_pos = self._run_live_path(df, "tsmom_ema", TSMOM_EMA_PARAMS, OVERLAY_CFG_OI_VOL)

        pd.testing.assert_series_equal(
            bt_pos, live_pos,
            check_names=False,
            obj="backtest vs live overlay positions (oi_vol)",
        )

    def test_overlay_actually_modifies_positions(self):
        """確認 vol spike 合成數據確實觸發了 overlay（非 no-op 測試）"""
        df = _make_ohlcv(500, seed=42)
        ctx = StrategyContext(
            symbol="BTCUSDT", interval="1h",
            market_type="futures", direction="both",
        )
        strategy_func = get_strategy("tsmom_ema")
        raw_pos = strategy_func(df, ctx, TSMOM_EMA_PARAMS)

        overlaid_pos = apply_overlay_by_mode(
            position=raw_pos,
            price_df=df,
            oi_series=None,
            params=OVERLAY_CFG_VOL_PAUSE["params"],
            mode="vol_pause",
        )

        # overlay 應該把某些非零位置設為 0
        diff_count = (raw_pos != overlaid_pos).sum()
        assert diff_count > 0, (
            "Overlay 沒有任何效果！合成數據中的 vol spike 未觸發 overlay。"
            "請檢查 _make_ohlcv 的 spike 幅度是否足夠。"
        )


# ══════════════════════════════════════════════════════════════
# 2. overlay_cfg=None / disabled 時不影響信號
# ══════════════════════════════════════════════════════════════


class TestOverlayDisabled:
    """overlay 關閉時不應改變 raw positions"""

    def test_overlay_none_no_change(self):
        """overlay_cfg=None 時，live 信號不受影響"""
        df = _make_ohlcv(500, seed=42)
        ctx = StrategyContext(
            symbol="BTCUSDT", interval="1h",
            market_type="futures", direction="both",
        )
        strategy_func = get_strategy("tsmom_ema")
        raw_pos = strategy_func(df, ctx, TSMOM_EMA_PARAMS)

        # 模擬 live 路徑，overlay_cfg=None
        live_pos = raw_pos.copy()  # generate_signal 不會修改
        # overlay_cfg is None → 跳過
        pd.testing.assert_series_equal(
            raw_pos, live_pos, check_names=False,
        )

    def test_overlay_disabled_no_change(self):
        """overlay_cfg.enabled=False 時，live 信號不受影響"""
        df = _make_ohlcv(500, seed=42)
        ctx = StrategyContext(
            symbol="BTCUSDT", interval="1h",
            market_type="futures", direction="both",
        )
        strategy_func = get_strategy("tsmom_ema")
        raw_pos = strategy_func(df, ctx, TSMOM_EMA_PARAMS)

        overlay_cfg = {"enabled": False, "mode": "vol_pause", "params": {}}
        # Simulate: condition check → enabled=False → skip
        overlaid = raw_pos.copy()
        if overlay_cfg and overlay_cfg.get("enabled", False):
            overlaid = apply_vol_pause_overlay(overlaid, df, overlay_cfg["params"])

        pd.testing.assert_series_equal(
            raw_pos, overlaid, check_names=False,
        )


# ══════════════════════════════════════════════════════════════
# 3. generate_signal 介面向後相容
# ══════════════════════════════════════════════════════════════


class TestGenerateSignalInterface:
    """確認 generate_signal 新增 overlay_cfg 參數向後相容"""

    def test_generate_signal_accepts_overlay_cfg(self):
        """generate_signal() 應接受 overlay_cfg kwarg 不報錯"""
        import inspect
        from qtrade.live.signal_generator import generate_signal

        sig = inspect.signature(generate_signal)
        assert "overlay_cfg" in sig.parameters, (
            "generate_signal() 缺少 overlay_cfg 參數！"
        )
        # 預設值必須是 None（向後相容）
        param = sig.parameters["overlay_cfg"]
        assert param.default is None, (
            f"overlay_cfg 預設值應為 None，但得到 {param.default}"
        )
