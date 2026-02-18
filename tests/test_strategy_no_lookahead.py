"""
策略 Look-Ahead Bias 自動偵測測試

═══════════════════════════════════════════════════════
三道防線，任何一道失敗都代表策略有 look-ahead：
═══════════════════════════════════════════════════════

1. 截斷測試（Truncation Test）
   在完整數據 [0:N] 上跑策略 → 信號 A
   在截斷數據 [0:N-100] 上跑策略 → 信號 B
   A[0:N-100] 必須等於 B
   如果不等 → 策略使用了 bar N-100 之後的「未來數據」

2. 框架延遲測試（Framework Delay Test）
   get_strategy() 回傳的 wrapped 版本（有 delay）
   get_raw_strategy() 回傳的原始版本（無 delay）
   wrapped.shift(-1) 應該約等於 raw（驗證框架確實有 shift）

3. 因果測試（Causality Test）
   修改最後 50 根 K 線的數據
   前 N-50 根的信號不應改變
   如果改變 → 策略的前段信號依賴了後段數據

所有已註冊策略（@register_strategy）都會自動被測試。
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# 確保 src 在路徑中
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qtrade.strategy import list_strategies, get_strategy, get_raw_strategy
from qtrade.strategy.base import StrategyContext


# ══════════════════════════════════════════════════════════
#  測試數據生成
# ══════════════════════════════════════════════════════════

def _make_fake_ohlcv(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    生成有趨勢特徵的假 OHLCV 數據
    （純隨機數據對動量策略不公平，加入趨勢讓信號有變化）
    """
    np.random.seed(seed)
    # 帶趨勢的價格序列
    trend = np.cumsum(np.random.randn(n) * 0.5 + 0.02)
    close = 1000 + trend
    noise = np.abs(np.random.randn(n)) * 2

    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    df = pd.DataFrame({
        "open": np.roll(close, 1),
        "high": close + noise,
        "low": close - noise,
        "close": close,
        "volume": np.random.randint(1000, 100000, n).astype(float),
    }, index=idx)
    df.iloc[0, df.columns.get_loc("open")] = close[0]  # 修正第一根

    return df


# 策略的預設參數
_DEFAULT_PARAMS = {
    "rsi_adx_atr": {
        "rsi_period": 10, "oversold": 30, "overbought": 70,
        "min_adx": 15, "adx_period": 14, "stop_loss_atr": 2.0,
        "atr_period": 14, "cooldown_bars": 3,
    },
    "rsi_adx_atr_trailing": {
        "rsi_period": 10, "oversold": 30, "overbought": 70,
        "min_adx": 15, "adx_period": 14, "stop_loss_atr": 2.0,
        "atr_period": 14, "cooldown_bars": 3,
    },
    "tsmom": {"lookback": 72, "vol_target": 0.15},
    "tsmom_multi": {"lookbacks": [72, 168], "vol_target": 0.15},
    "tsmom_ema": {"lookback": 72, "vol_target": 0.15, "ema_fast": 20, "ema_slow": 50},
    "tsmom_multi_ema": {"lookbacks": [72, 168], "vol_target": 0.15, "ema_fast": 20, "ema_slow": 50},
}

# 需要 universe 數據的策略（在假數據測試中跳過，因為無法載入 parquet）
_NEEDS_UNIVERSE = {"xsmom", "xsmom_tsmom"}

# 所有需要測試的策略（自動發現，排除需要 universe 的策略）
_ALL_STRATEGIES = [s for s in list_strategies() if s not in _NEEDS_UNIVERSE]

# 需要跳過框架延遲測試的策略（auto_delay=False，自行管理 delay）
_MANUAL_DELAY_STRATEGIES = {"rsi_adx_atr", "rsi_adx_atr_trailing"}


def _get_params(name: str) -> dict:
    """取得策略的預設參數"""
    return _DEFAULT_PARAMS.get(name, {})


# ══════════════════════════════════════════════════════════
#  Test 1: 截斷測試（最核心的 look-ahead 偵測）
# ══════════════════════════════════════════════════════════

@pytest.mark.parametrize("strategy_name", _ALL_STRATEGIES)
def test_truncation_no_lookahead(strategy_name: str):
    """
    截斷測試：未來數據不應影響過去的信號

    如果策略在 [0:400] 和 [0:500] 上計算的 [0:400] 信號不同，
    就代表 bar 400 之後的數據「洩漏」到了前面的計算中。
    """
    df_full = _make_fake_ohlcv(500, seed=42)
    df_trunc = df_full.iloc[:400].copy()

    ctx = StrategyContext(
        symbol="TEST", market_type="futures", direction="both",
        signal_delay=0,  # 用 0 測試原始信號，避免 shift 引入的邊界差異
    )
    params = _get_params(strategy_name)

    strategy_func = get_strategy(strategy_name)
    pos_full = strategy_func(df_full, ctx, params)
    pos_trunc = strategy_func(df_trunc, ctx, params)

    # 比較重疊區域（跳過前 100 根，給 warmup 空間）
    overlap_start = 100
    overlap_end = len(df_trunc)
    full_overlap = pos_full.iloc[overlap_start:overlap_end].values
    trunc_overlap = pos_trunc.iloc[overlap_start:].values

    diff = np.abs(full_overlap - trunc_overlap)
    max_diff = np.nanmax(diff)

    assert max_diff < 1e-10, (
        f"❌ {strategy_name} LOOK-AHEAD DETECTED!\n"
        f"   信號在截斷前後不一致（max diff = {max_diff:.6f}）\n"
        f"   這代表策略使用了 bar {overlap_end}+ 的未來數據\n"
        f"   第一個差異位置: bar {overlap_start + np.argmax(diff > 1e-10)}"
    )


# ══════════════════════════════════════════════════════════
#  Test 2: 框架延遲測試（驗證 register_strategy 有 shift）
# ══════════════════════════════════════════════════════════

@pytest.mark.parametrize("strategy_name", [
    s for s in _ALL_STRATEGIES if s not in _MANUAL_DELAY_STRATEGIES
])
def test_framework_applies_signal_delay(strategy_name: str):
    """
    驗證 @register_strategy(auto_delay=True) 確實會 shift 信號

    wrapped(delay=1) 應該等於 raw(delay=0).shift(1)
    """
    df = _make_fake_ohlcv(500, seed=42)
    params = _get_params(strategy_name)

    # 無延遲的 raw 信號
    ctx_raw = StrategyContext(
        symbol="TEST", market_type="futures", direction="both",
        signal_delay=0,
    )
    raw_func = get_raw_strategy(strategy_name)
    pos_raw = raw_func(df, ctx_raw, params)

    # 有延遲的 wrapped 信號
    ctx_delayed = StrategyContext(
        symbol="TEST", market_type="futures", direction="both",
        signal_delay=1,
    )
    wrapped_func = get_strategy(strategy_name)
    pos_delayed = wrapped_func(df, ctx_delayed, params)

    # wrapped(delay=1) 應該 = raw.shift(1)
    expected = pos_raw.shift(1).fillna(0.0)

    # 跳過第一根（shift 後是 NaN → 0）
    diff = np.abs(pos_delayed.values[1:] - expected.values[1:])
    max_diff = np.nanmax(diff)

    assert max_diff < 1e-10, (
        f"❌ {strategy_name} 框架延遲未生效！\n"
        f"   wrapped(delay=1) != raw.shift(1)\n"
        f"   max diff = {max_diff:.6f}\n"
        f"   這代表 @register_strategy 沒有正確 shift 信號"
    )


# ══════════════════════════════════════════════════════════
#  Test 3: 因果測試（修改未來，過去不應變）
# ══════════════════════════════════════════════════════════

@pytest.mark.parametrize("strategy_name", _ALL_STRATEGIES)
def test_causality_future_data_doesnt_affect_past(strategy_name: str):
    """
    因果測試：修改最後 50 根 K 線，前面的信號不應改變

    這比截斷測試更嚴格：即使策略沒有直接使用未來數據，
    如果用了某種全局 normalize（如 rank 全序列），也會被抓到。
    """
    df_original = _make_fake_ohlcv(500, seed=42)
    df_modified = df_original.copy()

    # 修改最後 50 根的 close（大幅改變，確保差異明顯）
    df_modified.iloc[-50:, df_modified.columns.get_loc("close")] *= 2.0
    df_modified.iloc[-50:, df_modified.columns.get_loc("high")] *= 2.0
    df_modified.iloc[-50:, df_modified.columns.get_loc("low")] *= 2.0
    df_modified.iloc[-50:, df_modified.columns.get_loc("open")] *= 2.0

    ctx = StrategyContext(
        symbol="TEST", market_type="futures", direction="both",
        signal_delay=0,
    )
    params = _get_params(strategy_name)

    strategy_func = get_strategy(strategy_name)
    pos_orig = strategy_func(df_original, ctx, params)
    pos_mod = strategy_func(df_modified, ctx, params)

    # 前 N-50 根的信號（扣除 warmup）應該相同
    check_start = 100
    check_end = len(df_original) - 50

    orig_vals = pos_orig.iloc[check_start:check_end].values
    mod_vals = pos_mod.iloc[check_start:check_end].values

    diff = np.abs(orig_vals - mod_vals)
    max_diff = np.nanmax(diff)

    assert max_diff < 1e-10, (
        f"❌ {strategy_name} CAUSALITY VIOLATION!\n"
        f"   修改最後 50 根 K 線導致前面的信號改變（max diff = {max_diff:.6f}）\n"
        f"   這代表策略的前段計算依賴了後段數據\n"
        f"   第一個差異位置: bar {check_start + np.argmax(diff > 1e-10)}"
    )


# ══════════════════════════════════════════════════════════
#  Test 4: Direction Clip 測試
# ══════════════════════════════════════════════════════════

@pytest.mark.parametrize("strategy_name", [
    s for s in _ALL_STRATEGIES if s not in _MANUAL_DELAY_STRATEGIES
])
def test_direction_clip_spot_no_short(strategy_name: str):
    """
    Spot 模式下不應有負信號（做空信號應被 clip 到 0）
    """
    df = _make_fake_ohlcv(500, seed=42)
    params = _get_params(strategy_name)

    ctx = StrategyContext(
        symbol="TEST", market_type="spot", direction="both",
        signal_delay=0,
    )
    strategy_func = get_strategy(strategy_name)
    pos = strategy_func(df, ctx, params)

    min_val = pos.min()
    assert min_val >= -1e-10, (
        f"❌ {strategy_name} Spot 模式出現負信號 ({min_val:.4f})\n"
        f"   Spot 不能做空，框架應自動 clip lower=0"
    )


@pytest.mark.parametrize("strategy_name", [
    s for s in _ALL_STRATEGIES if s not in _MANUAL_DELAY_STRATEGIES
])
def test_direction_clip_long_only(strategy_name: str):
    """
    long_only 模式下不應有負信號
    """
    df = _make_fake_ohlcv(500, seed=42)
    params = _get_params(strategy_name)

    ctx = StrategyContext(
        symbol="TEST", market_type="futures", direction="long_only",
        signal_delay=0,
    )
    strategy_func = get_strategy(strategy_name)
    pos = strategy_func(df, ctx, params)

    min_val = pos.min()
    assert min_val >= -1e-10, (
        f"❌ {strategy_name} long_only 模式出現負信號 ({min_val:.4f})"
    )


@pytest.mark.parametrize("strategy_name", [
    s for s in _ALL_STRATEGIES if s not in _MANUAL_DELAY_STRATEGIES
])
def test_direction_clip_short_only(strategy_name: str):
    """
    short_only 模式下不應有正信號
    """
    df = _make_fake_ohlcv(500, seed=42)
    params = _get_params(strategy_name)

    ctx = StrategyContext(
        symbol="TEST", market_type="futures", direction="short_only",
        signal_delay=0,
    )
    strategy_func = get_strategy(strategy_name)
    pos = strategy_func(df, ctx, params)

    max_val = pos.max()
    assert max_val <= 1e-10, (
        f"❌ {strategy_name} short_only 模式出現正信號 ({max_val:.4f})"
    )


# ══════════════════════════════════════════════════════════
#  Test 5: 信號值範圍測試
# ══════════════════════════════════════════════════════════

@pytest.mark.parametrize("strategy_name", _ALL_STRATEGIES)
def test_signal_range(strategy_name: str):
    """信號必須在 [-1, 1] 範圍內"""
    df = _make_fake_ohlcv(500, seed=42)
    params = _get_params(strategy_name)

    ctx = StrategyContext(
        symbol="TEST", market_type="futures", direction="both",
        signal_delay=0,
    )
    strategy_func = get_strategy(strategy_name)
    pos = strategy_func(df, ctx, params)

    assert pos.min() >= -1.0 - 1e-10, f"{strategy_name}: min={pos.min():.4f} < -1"
    assert pos.max() <= 1.0 + 1e-10, f"{strategy_name}: max={pos.max():.4f} > 1"
    assert not pos.isna().any(), f"{strategy_name}: 信號包含 NaN"


# ══════════════════════════════════════════════════════════
#  XSMOM 專用測試（需要 universe 數據 → 用 real parquet）
# ══════════════════════════════════════════════════════════

_DATA_DIR = Path(__file__).parent.parent / "data"
_HAS_REAL_DATA = (_DATA_DIR / "binance" / "futures" / "1h" / "ETHUSDT.parquet").exists()


@pytest.mark.skipif(not _HAS_REAL_DATA, reason="需要 parquet 數據")
@pytest.mark.parametrize("strategy_name", ["xsmom", "xsmom_tsmom"])
def test_xsmom_truncation_no_lookahead(strategy_name: str):
    """XSMOM 截斷測試：用真實 parquet 數據"""
    from qtrade.data.storage import load_klines

    # 載入一個 symbol 的完整數據
    fpath = _DATA_DIR / "binance" / "futures" / "1h" / "ETHUSDT.parquet"
    df_all = load_klines(fpath)

    # 取最近 2000 根
    df_full = df_all.iloc[-2000:].copy()
    df_trunc = df_full.iloc[:-200].copy()

    ctx = StrategyContext(
        symbol="ETHUSDT", interval="1h",
        market_type="futures", direction="both",
        signal_delay=0,
    )
    params = {
        "data_dir": str(_DATA_DIR.parent / "data"),
        "lookback": 168,
        "long_threshold": 0.7,
        "short_threshold": 0.3,
        "vol_target": 0.15,
    }

    strategy_func = get_strategy(strategy_name)

    # 清快取，確保重新載入
    from qtrade.strategy.xsmom_strategy import _UNIVERSE_CACHE
    _UNIVERSE_CACHE.clear()
    pos_full = strategy_func(df_full, ctx, params)

    _UNIVERSE_CACHE.clear()
    pos_trunc = strategy_func(df_trunc, ctx, params)

    # 比較重疊區域
    overlap_start = 300  # 跳過 warmup
    overlap_end = len(df_trunc)
    full_overlap = pos_full.iloc[overlap_start:overlap_end].values
    trunc_overlap = pos_trunc.iloc[overlap_start:].values

    diff = np.abs(full_overlap - trunc_overlap)
    max_diff = np.nanmax(diff)

    assert max_diff < 1e-10, (
        f"❌ {strategy_name} XSMOM LOOK-AHEAD DETECTED!\n"
        f"   max diff = {max_diff:.6f}"
    )


@pytest.mark.skipif(not _HAS_REAL_DATA, reason="需要 parquet 數據")
@pytest.mark.parametrize("strategy_name", ["xsmom", "xsmom_tsmom"])
def test_xsmom_framework_delay(strategy_name: str):
    """XSMOM 框架延遲測試"""
    from qtrade.data.storage import load_klines

    fpath = _DATA_DIR / "binance" / "futures" / "1h" / "ETHUSDT.parquet"
    df = load_klines(fpath).iloc[-1000:].copy()

    params = {
        "data_dir": str(_DATA_DIR.parent / "data"),
        "lookback": 168,
        "vol_target": 0.15,
    }

    # 無延遲的 raw
    ctx_raw = StrategyContext(
        symbol="ETHUSDT", interval="1h",
        market_type="futures", direction="both",
        signal_delay=0,
    )
    from qtrade.strategy.xsmom_strategy import _UNIVERSE_CACHE
    _UNIVERSE_CACHE.clear()
    raw_func = get_raw_strategy(strategy_name)
    pos_raw = raw_func(df, ctx_raw, params)

    # 有延遲的 wrapped
    ctx_delayed = StrategyContext(
        symbol="ETHUSDT", interval="1h",
        market_type="futures", direction="both",
        signal_delay=1,
    )
    _UNIVERSE_CACHE.clear()
    wrapped_func = get_strategy(strategy_name)
    pos_delayed = wrapped_func(df, ctx_delayed, params)

    # wrapped(delay=1) 應該 = raw.shift(1)
    expected = pos_raw.shift(1).fillna(0.0)
    diff = np.abs(pos_delayed.values[1:] - expected.values[1:])
    max_diff = np.nanmax(diff)

    assert max_diff < 1e-10, (
        f"❌ {strategy_name} 框架延遲未生效！max diff = {max_diff:.6f}"
    )
