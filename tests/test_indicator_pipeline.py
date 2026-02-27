"""
Indicator Pipeline 整合測試

驗證策略自帶指標（via pos.attrs["indicators"]）能完整存活
從策略生成 → register_strategy wrapper (shift/clip) → meta_blend 轉發
→ signal_generator → SignalResult.indicators 的整條管線。

背景：
    2026-02-27 Telegram Bot 只顯示 RSI/ADX（fallback 指標），
    而非策略自帶的 tsmom/carry/ema_trend/htf。
    原因：策略未附帶 indicators、meta_blend 未轉發、
    signal_generator 未優先使用策略指標。

修復：
    1. tsmom_carry_v2: 附帶 indicators 到 pos.attrs
    2. meta_blend: 轉發子策略 indicators
    3. signal_generator: 優先使用策略指標，fallback 到 RSI/ADX
    4. register_strategy wrapper + signal_generator overlay 後保存/恢復 attrs（防禦性）
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ══════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════

@pytest.fixture
def sample_df():
    """生成最小可用的 OHLCV DataFrame (200 bars)"""
    n = 200
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=n, freq="1h")
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame({
        "open": close + np.random.randn(n) * 0.1,
        "high": close + abs(np.random.randn(n) * 0.5),
        "low": close - abs(np.random.randn(n) * 0.5),
        "close": close,
        "volume": np.random.randint(100, 10000, n).astype(float),
    }, index=dates)
    return df


@pytest.fixture
def strategy_context():
    """建立標準 StrategyContext"""
    from qtrade.strategy.base import StrategyContext
    return StrategyContext(
        symbol="ETHUSDT",
        interval="1h",
        market_type="futures",
        direction="both",
        signal_delay=0,  # 先不 shift，單獨測
    )


@pytest.fixture
def strategy_context_with_delay():
    """建立含 signal_delay=1 的 StrategyContext"""
    from qtrade.strategy.base import StrategyContext
    return StrategyContext(
        symbol="ETHUSDT",
        interval="1h",
        market_type="futures",
        direction="both",
        signal_delay=1,
    )


# ══════════════════════════════════════════════════════════════
# 1. pandas .attrs 在 shift/clip 後的行為驗證
# ══════════════════════════════════════════════════════════════

class TestPandasAttrsBehavior:
    """驗證 pandas .attrs 在各種操作後被保留（pandas 2.x+ 行為）。
    我們的防禦性 save/restore 機制確保即使未來 pandas 行為改變也不會破壞管線。"""

    def test_shift_preserves_attrs(self):
        """pandas 2.x+ .shift() 保留 .attrs"""
        s = pd.Series([1.0, 2.0, 3.0])
        s.attrs["indicators"] = {"foo": "bar"}
        shifted = s.shift(1)
        assert shifted.attrs.get("indicators") == {"foo": "bar"}

    def test_clip_preserves_attrs(self):
        """pandas 2.x+ .clip() 保留 .attrs"""
        s = pd.Series([1.0, 2.0, 3.0])
        s.attrs["indicators"] = {"foo": "bar"}
        clipped = s.clip(-1.0, 1.0)
        assert clipped.attrs.get("indicators") == {"foo": "bar"}

    def test_fillna_preserves_attrs(self):
        """pandas 2.x+ .fillna() 保留 .attrs"""
        s = pd.Series([1.0, None, 3.0])
        s.attrs["indicators"] = {"foo": "bar"}
        filled = s.fillna(0.0)
        assert filled.attrs.get("indicators") == {"foo": "bar"}


# ══════════════════════════════════════════════════════════════
# 2. register_strategy wrapper 保護 attrs
# ══════════════════════════════════════════════════════════════

class TestRegisterStrategyAttrsPreservation:
    """驗證 register_strategy wrapper 正確保存/恢復 attrs"""

    def test_attrs_survive_shift_in_wrapper(self, sample_df, strategy_context_with_delay):
        """attrs 在 register_strategy 的 shift(signal_delay=1) 後仍存在"""
        from qtrade.strategy import register_strategy, _STRATEGY_REGISTRY

        # 註冊一個測試策略
        test_name = "_test_attrs_shift"

        @register_strategy(test_name)
        def _test_strategy(df, ctx, params):
            pos = pd.Series(0.5, index=df.index)
            pos.attrs["indicators"] = {"test_key": "test_value", "score": 0.42}
            return pos

        try:
            fn = _STRATEGY_REGISTRY[test_name]
            result = fn(sample_df, strategy_context_with_delay, {})

            assert "indicators" in result.attrs, \
                "attrs['indicators'] 在 shift 後消失了！"
            assert result.attrs["indicators"]["test_key"] == "test_value"
            assert result.attrs["indicators"]["score"] == 0.42
        finally:
            _STRATEGY_REGISTRY.pop(test_name, None)

    def test_attrs_survive_clip_in_wrapper(self, sample_df):
        """attrs 在 register_strategy 的 clip（long_only 方向裁剪）後仍存在"""
        from qtrade.strategy import register_strategy, _STRATEGY_REGISTRY
        from qtrade.strategy.base import StrategyContext

        test_name = "_test_attrs_clip"

        # 使用 long_only context，觸發 clip(lower=0.0) 裁剪負值
        ctx_long_only = StrategyContext(
            symbol="ETHUSDT",
            interval="1h",
            market_type="futures",
            direction="long_only",
            signal_delay=0,
        )

        @register_strategy(test_name)
        def _test_strategy(df, ctx, params):
            pos = pd.Series(-0.5, index=df.index)  # 負值，long_only 會 clip 到 0
            pos.attrs["indicators"] = {"clipped": True}
            return pos

        try:
            fn = _STRATEGY_REGISTRY[test_name]
            result = fn(sample_df, ctx_long_only, {})

            # 值被 clip 了（負值變 0）
            assert result.min() >= 0.0, "long_only 應 clip 負值到 0"
            # attrs 仍在
            assert "indicators" in result.attrs, \
                "attrs['indicators'] 在 clip 後消失了！"
            assert result.attrs["indicators"]["clipped"] is True
        finally:
            _STRATEGY_REGISTRY.pop(test_name, None)


# ══════════════════════════════════════════════════════════════
# 3. tsmom_carry_v2 指標附帶
# ══════════════════════════════════════════════════════════════

class TestTsmomCarryV2Indicators:
    """驗證 tsmom_carry_v2 策略正確附帶指標"""

    @pytest.fixture
    def tsmom_df(self):
        """生成足夠長的 OHLCV (500 bars) + 基本結構"""
        n = 500
        np.random.seed(123)
        dates = pd.date_range("2025-01-01", periods=n, freq="1h")
        close = 2000 + np.cumsum(np.random.randn(n) * 5)
        df = pd.DataFrame({
            "open": close + np.random.randn(n) * 0.5,
            "high": close + abs(np.random.randn(n) * 2),
            "low": close - abs(np.random.randn(n) * 2),
            "close": close,
            "volume": np.random.randint(1000, 100000, n).astype(float),
        }, index=dates)
        return df

    def test_tsmom_carry_v2_attaches_indicators(self, tsmom_df, strategy_context):
        """tsmom_carry_v2 策略應附帶 indicators 到 attrs"""
        from qtrade.strategy.tsmom_carry_v2_strategy import generate_tsmom_carry_v2

        params = {
            "tsmom_lookback": 168,
            "tsmom_ema_fast": 12,
            "tsmom_ema_slow": 26,
            "composite_ema_span": 6,
            "position_step": 0.1,
            "min_change_threshold": 0.05,
        }

        result = generate_tsmom_carry_v2(tsmom_df, strategy_context, params)

        assert "indicators" in result.attrs, \
            "tsmom_carry_v2 未附帶 indicators！"

        ind = result.attrs["indicators"]
        assert "tier" in ind, "缺少 tier 指標"
        assert "tsmom" in ind, "缺少 tsmom 指標"
        assert "ema_trend" in ind, "缺少 ema_trend 指標"
        assert ind["ema_trend"] in ("UP", "DOWN"), \
            f"ema_trend 應為 UP/DOWN，得到 {ind['ema_trend']}"

    def test_tsmom_carry_v2_indicators_survive_register_wrapper(
        self, tsmom_df, strategy_context_with_delay
    ):
        """tsmom_carry_v2 的 indicators 在 register_strategy wrapper 後仍存在"""
        from qtrade.strategy import get_strategy

        strategy_fn = get_strategy("tsmom_carry_v2")
        params = {
            "tsmom_lookback": 168,
            "tsmom_ema_fast": 12,
            "tsmom_ema_slow": 26,
            "composite_ema_span": 6,
            "position_step": 0.1,
            "min_change_threshold": 0.05,
        }

        result = strategy_fn(tsmom_df, strategy_context_with_delay, params)

        assert "indicators" in result.attrs, \
            "tsmom_carry_v2 indicators 在 register_strategy wrapper (shift+clip) 後消失！"

        ind = result.attrs["indicators"]
        assert "tsmom" in ind
        assert "ema_trend" in ind


# ══════════════════════════════════════════════════════════════
# 4. meta_blend 子策略指標轉發
# ══════════════════════════════════════════════════════════════

class TestMetaBlendIndicatorForwarding:
    """驗證 meta_blend 正確轉發子策略的 indicators"""

    @pytest.fixture
    def blend_df(self):
        """生成足夠長的 OHLCV (500 bars)"""
        n = 500
        np.random.seed(456)
        dates = pd.date_range("2025-01-01", periods=n, freq="1h")
        close = 3000 + np.cumsum(np.random.randn(n) * 10)
        return pd.DataFrame({
            "open": close + np.random.randn(n),
            "high": close + abs(np.random.randn(n) * 3),
            "low": close - abs(np.random.randn(n) * 3),
            "close": close,
            "volume": np.random.randint(1000, 100000, n).astype(float),
        }, index=dates)

    def test_meta_blend_forwards_sub_strategy_indicators(self, blend_df):
        """meta_blend 應轉發最高權重子策略的 indicators"""
        from qtrade.strategy import get_strategy
        from qtrade.strategy.base import StrategyContext

        ctx = StrategyContext(
            symbol="ETHUSDT",
            interval="1h",
            market_type="futures",
            direction="both",
            signal_delay=1,
        )

        params = {
            "sub_strategies": [
                {
                    "name": "tsmom_carry_v2",
                    "weight": 1.0,
                    "params": {
                        "tsmom_lookback": 168,
                        "tsmom_ema_fast": 12,
                        "tsmom_ema_slow": 26,
                        "composite_ema_span": 6,
                        "position_step": 0.1,
                        "min_change_threshold": 0.05,
                    },
                }
            ],
        }

        strategy_fn = get_strategy("meta_blend")
        result = strategy_fn(blend_df, ctx, params)

        assert "indicators" in result.attrs, \
            "meta_blend 未轉發子策略的 indicators！"

        ind = result.attrs["indicators"]
        # 應包含 tsmom_carry_v2 的指標
        assert "tsmom" in ind or "tier" in ind, \
            f"meta_blend 未包含 tsmom_carry_v2 的指標，只有: {list(ind.keys())}"
        # _sub 記錄來源子策略
        assert ind.get("_sub") == "tsmom_carry_v2", \
            f"_sub 應為 tsmom_carry_v2，得到 {ind.get('_sub')}"


# ══════════════════════════════════════════════════════════════
# 5. signal_generator 完整管線
# ══════════════════════════════════════════════════════════════

class TestSignalGeneratorIndicatorPipeline:
    """驗證 generate_signal 最終輸出包含策略指標而非 fallback"""

    @pytest.fixture
    def long_df(self):
        """生成 500 bar OHLCV"""
        n = 500
        np.random.seed(789)
        dates = pd.date_range("2025-01-01", periods=n, freq="1h")
        close = 50000 + np.cumsum(np.random.randn(n) * 100)
        return pd.DataFrame({
            "open": close + np.random.randn(n) * 10,
            "high": close + abs(np.random.randn(n) * 50),
            "low": close - abs(np.random.randn(n) * 50),
            "close": close,
            "volume": np.random.randint(1000, 100000, n).astype(float),
        }, index=dates)

    def test_generate_signal_uses_strategy_indicators(self, long_df):
        """generate_signal 應優先使用策略自帶指標，非 RSI/ADX fallback"""
        from qtrade.live.signal_generator import generate_signal

        params = {
            "sub_strategies": [
                {
                    "name": "tsmom_carry_v2",
                    "weight": 1.0,
                    "params": {
                        "tsmom_lookback": 168,
                        "tsmom_ema_fast": 12,
                        "tsmom_ema_slow": 26,
                        "composite_ema_span": 6,
                        "position_step": 0.1,
                        "min_change_threshold": 0.05,
                    },
                }
            ],
        }

        result = generate_signal(
            symbol="BTCUSDT",
            strategy_name="meta_blend",
            params=params,
            interval="1h",
            df=long_df,
            market_type="futures",
            direction="both",
        )

        # 應包含策略指標
        assert "tsmom" in result.indicators, \
            f"SignalResult 缺少 tsmom 指標，只有: {list(result.indicators.keys())}"
        assert "ema_trend" in result.indicators, \
            f"SignalResult 缺少 ema_trend 指標"

        # 不應包含 RSI/ADX fallback（因為策略已提供指標）
        assert "rsi" not in result.indicators, \
            "策略已提供指標但仍 fallback 到 RSI"
        assert "adx" not in result.indicators, \
            "策略已提供指標但仍 fallback 到 ADX"

    def test_generate_signal_fallback_without_strategy_indicators(self, long_df):
        """無策略指標時應 fallback 到 RSI/ADX"""
        from qtrade.strategy import register_strategy, _STRATEGY_REGISTRY
        from qtrade.live.signal_generator import generate_signal

        test_name = "_test_no_indicators"

        @register_strategy(test_name)
        def _test_strategy(df, ctx, params):
            # 故意不設 attrs["indicators"]
            return pd.Series(0.5, index=df.index)

        try:
            result = generate_signal(
                symbol="BTCUSDT",
                strategy_name=test_name,
                params={},
                interval="1h",
                df=long_df,
                market_type="futures",
                direction="both",
            )

            # 應 fallback 到 RSI/ADX
            assert "rsi" in result.indicators, \
                "無策略指標時應 fallback 到 RSI"
        finally:
            _STRATEGY_REGISTRY.pop(test_name, None)
