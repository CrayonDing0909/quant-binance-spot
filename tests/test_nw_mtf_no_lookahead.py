"""
MTF 4h→1h Anti-Lookahead 專用測試 + 對齊模式仲裁

═══════════════════════════════════════════════════════
覆蓋 3 種 MTF alignment modes：
  - legacy_left_ffill:  ⚠️ LOOK-AHEAD（僅供對照）
  - right_ffill:        ✅ CAUSAL, 最小延遲
  - left_shift1_ffill:  ✅ CAUSAL, 保守穩定
═══════════════════════════════════════════════════════

測試項目：
  A. Online vs Offline 一致性（causal modes）
  B. Legacy look-ahead 直接偵測
  C. 截斷不變性（all modes）
  D. 4h 邊界因果測試（causal modes vs legacy）
  E. 因果性 — 修改未來數據（all modes）
  F. 完整策略截斷不變性（causal modes）
  G. 仲裁摘要
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qtrade.strategy.nw_envelope_regime_strategy import (
    _compute_htf_regime,
    _compute_nw_envelope,
    _detect_regime,
    _resample_ohlcv,
    _resample_ohlcv_right,
)
from qtrade.strategy import get_strategy
from qtrade.strategy.base import StrategyContext


# ══════════════════════════════════════════════════════════
#  測試數據生成
# ══════════════════════════════════════════════════════════

def _make_1h_ohlcv(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """生成有趨勢特徵的 1h OHLCV 假數據"""
    np.random.seed(seed)
    trend = np.cumsum(np.random.randn(n) * 0.5 + 0.02)
    close = 1000 + trend
    noise = np.abs(np.random.randn(n)) * 2

    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "open": np.roll(close, 1),
            "high": close + noise,
            "low": close - noise,
            "close": close,
            "volume": np.random.randint(1000, 100000, n).astype(float),
        },
        index=idx,
    )
    df.iloc[0, df.columns.get_loc("open")] = close[0]
    return df


# ── 標準參數 ──
_HTF_PARAMS = dict(
    bandwidth=8.0, alpha=1.0, lookback=50,
    adx_period=14, adx_threshold=20.0, regime_mode="adx",
)

# Causal modes（應通過所有測試）
_CAUSAL_MODES = ["right_ffill", "left_shift1_ffill"]
# All modes（包含 legacy 供仲裁）
_ALL_MODES = ["legacy_left_ffill", "right_ffill", "left_shift1_ffill"]


# ══════════════════════════════════════════════════════════
#  Test A: Online vs Offline 一致性（causal modes）
# ══════════════════════════════════════════════════════════

@pytest.mark.parametrize("mode", _CAUSAL_MODES)
def test_online_vs_offline_consistency(mode):
    """
    Offline vs Online：每個 4h 邊界截斷重算，比對離線全量結果。
    Causal modes 應 mismatch = 0。
    """
    df = _make_1h_ohlcv(1200, seed=42)

    offline_trending, offline_dir = _compute_htf_regime(
        df, **_HTF_PARAMS, mtf_alignment_mode=mode,
    )

    if mode == "right_ffill":
        df_4h = _resample_ohlcv_right(df, "4h")
    else:
        df_4h = _resample_ohlcv(df, "4h")

    warmup_4h = 50 + 20
    mismatches = 0
    total = 0

    for t in range(warmup_4h, len(df_4h)):
        if mode == "right_ffill":
            end_ts = df_4h.index[t]
        else:
            end_ts = df_4h.index[t] + pd.Timedelta(hours=3)

        df_trunc = df[df.index <= end_ts]
        if len(df_trunc) == 0:
            continue

        t_trending, t_dir = _compute_htf_regime(
            df_trunc, **_HTF_PARAMS, mtf_alignment_mode=mode,
        )

        last_ts = df_trunc.index[-1]
        if last_ts not in offline_trending.index:
            continue

        if bool(t_trending.loc[last_ts]) != bool(offline_trending.loc[last_ts]):
            mismatches += 1
        if abs(float(t_dir.loc[last_ts]) - float(offline_dir.loc[last_ts])) > 1e-10:
            mismatches += 1
        total += 1

    assert total > 0, "未比較到任何點"
    assert mismatches == 0, (
        f"[{mode}] Online vs Offline mismatch: {mismatches}/{total}"
    )


# ══════════════════════════════════════════════════════════
#  Test B: Legacy look-ahead 直接偵測
# ══════════════════════════════════════════════════════════

def test_legacy_alignment_uses_future_close():
    """
    直接證明 legacy_left_ffill 使 4h bar 的 close（未來數據）
    在 4h 窗口開始時即可見。

    方法：比較 1h bar 在 4h 窗口起始時看到的 4h close
    vs 該 1h bar 自己的 close。legacy 模式下，它們應不同
    （因為 4h bar 的 close = 3h 後的 1h close）。
    """
    df = _make_1h_ohlcv(800, seed=42)
    df_4h = _resample_ohlcv(df, "4h")

    # 用 4h close 作為 proxy signal（繞過 NW/ADX 閾值化）
    signal_4h = df_4h["close"]

    # Legacy alignment: direct ffill
    signal_1h_legacy = signal_4h.reindex(df.index, method="ffill").fillna(0.0)

    # left_shift1_ffill alignment
    signal_4h_shifted = signal_4h.shift(1).fillna(0.0)
    signal_1h_causal = signal_4h_shifted.reindex(df.index, method="ffill").fillna(0.0)

    # 在 4h 窗口的起始時刻（label），legacy 讓 3h 後的 close 可見
    future_leak_count = 0
    causal_leak_count = 0
    total = 0

    for idx in range(10, len(df_4h)):
        label_ts = df_4h.index[idx]
        close_ts = label_ts + pd.Timedelta(hours=3)

        if label_ts not in df.index or close_ts not in df.index:
            continue

        bar_close = df_4h.loc[label_ts, "close"]  # = close at T+3h
        current_close = df.loc[label_ts, "close"]  # = close at T

        # legacy: 在 T 就能看到 bar_close (= T+3h 的 close)
        legacy_val = signal_1h_legacy.loc[label_ts]
        if abs(legacy_val - bar_close) < 1e-10:
            if abs(bar_close - current_close) > 1e-6:
                future_leak_count += 1

        # causal: 在 T 只能看到上一根 4h bar 的 close
        causal_val = signal_1h_causal.loc[label_ts]
        if idx > 0:
            prev_close = df_4h["close"].iloc[idx - 1]
            if abs(causal_val - bar_close) < 1e-10:
                if abs(bar_close - current_close) > 1e-6:
                    causal_leak_count += 1

        total += 1

    assert total > 0
    assert future_leak_count > 0, (
        f"Legacy 應有未來數據洩漏但偵測到 0/{total}。"
        f"4h bar 的 close（3h 後）應在窗口起始即可見。"
    )
    # Causal mode 不應有 leak
    assert causal_leak_count == 0, (
        f"left_shift1_ffill 不應有未來洩漏但偵測到 {causal_leak_count}/{total}"
    )


# ══════════════════════════════════════════════════════════
#  Test C: 截斷不變性（所有 mode）
# ══════════════════════════════════════════════════════════

@pytest.mark.parametrize("mode", _ALL_MODES)
def test_truncation_invariance_mtf(mode):
    """截斷尾部數據，重疊區的 regime 必須一致（所有 mode 都應 PASS）"""
    df_full = _make_1h_ohlcv(1200, seed=42)
    df_trunc = df_full.iloc[:-200].copy()

    trending_full, dir_full = _compute_htf_regime(
        df_full, **_HTF_PARAMS, mtf_alignment_mode=mode,
    )
    trending_trunc, dir_trunc = _compute_htf_regime(
        df_trunc, **_HTF_PARAMS, mtf_alignment_mode=mode,
    )

    warmup_1h = 300
    overlap_end = len(df_trunc)

    full_t = trending_full.iloc[warmup_1h:overlap_end].values.astype(float)
    trunc_t = trending_trunc.iloc[warmup_1h:].values.astype(float)

    full_d = dir_full.iloc[warmup_1h:overlap_end].values
    trunc_d = dir_trunc.iloc[warmup_1h:].values

    max_diff_t = np.nanmax(np.abs(full_t - trunc_t))
    max_diff_d = np.nanmax(np.abs(full_d - trunc_d))

    assert max_diff_t < 1e-10, (
        f"[{mode}] MTF trending 截斷不一致! max diff={max_diff_t:.6f}"
    )
    assert max_diff_d < 1e-10, (
        f"[{mode}] MTF direction 截斷不一致! max diff={max_diff_d:.6f}"
    )


# ══════════════════════════════════════════════════════════
#  Test D: 4h 邊界因果測試
# ══════════════════════════════════════════════════════════

def _get_intra_window_bars(mode, df_4h, target_idx, df_1h_index):
    """
    取得 4h 窗口內「更早於修改點」的 1h bars。
    這些 bar 不應受到修改影響。
    """
    if mode == "right_ffill":
        # right/right: bar labeled T contains (T-4h, T]
        # last bar = T, earlier bars = T-3h, T-2h, T-1h
        label = df_4h.index[target_idx]
        modified_ts = label  # last bar in window
        earlier_bars = []
        for delta_h in [3, 2, 1]:
            ts = label - pd.Timedelta(hours=delta_h)
            if ts in df_1h_index:
                earlier_bars.append(ts)
        return modified_ts, earlier_bars
    else:
        # left/left: bar labeled T contains [T, T+4h)
        # last bar = T+3h, earlier bars = T, T+1h, T+2h
        label = df_4h.index[target_idx]
        modified_ts = label + pd.Timedelta(hours=3)
        earlier_bars = []
        for delta_h in [0, 1, 2]:
            ts = label + pd.Timedelta(hours=delta_h)
            if ts in df_1h_index:
                earlier_bars.append(ts)
        return modified_ts, earlier_bars


@pytest.mark.parametrize("mode", _CAUSAL_MODES)
def test_4h_boundary_alignment(mode):
    """
    修改 4h 窗口的最後 1h bar → 同窗口內更早的 1h bar 不應受影響。
    Causal modes 應 PASS。
    """
    df = _make_1h_ohlcv(1200, seed=42)

    if mode == "right_ffill":
        df_4h = _resample_ohlcv_right(df, "4h")
    else:
        df_4h = _resample_ohlcv(df, "4h")

    # 測試多個 4h 窗口以增加信心
    any_fail = False
    for target_idx in range(75, 85):
        if target_idx >= len(df_4h):
            continue

        modified_ts, earlier_bars = _get_intra_window_bars(
            mode, df_4h, target_idx, df.index,
        )
        if not earlier_bars or modified_ts not in df.index:
            continue

        df_mod = df.copy()
        df_mod.loc[modified_ts, "close"] *= 5.0
        df_mod.loc[modified_ts, "high"] *= 5.0

        t_orig, d_orig = _compute_htf_regime(
            df, **_HTF_PARAMS, mtf_alignment_mode=mode,
        )
        t_mod, d_mod = _compute_htf_regime(
            df_mod, **_HTF_PARAMS, mtf_alignment_mode=mode,
        )

        for ts in earlier_bars:
            diff_t = abs(float(t_orig.loc[ts]) - float(t_mod.loc[ts]))
            diff_d = abs(float(d_orig.loc[ts]) - float(d_mod.loc[ts]))
            if diff_t > 1e-10 or diff_d > 1e-10:
                any_fail = True
                break
        if any_fail:
            break

    assert not any_fail, (
        f"[{mode}] 4h boundary look-ahead detected! "
        f"同窗口早期 bars 受到晚期修改影響。"
    )


def test_4h_boundary_legacy_has_lookahead():
    """
    legacy_left_ffill：修改 4h 窗口最後 1h bar →
    同窗口內更早 1h bar 的 4h close 也受影響（直接證明 look-ahead）。

    用 4h close 直接測試對齊（繞過 ADX 閾值化）。
    """
    df = _make_1h_ohlcv(1200, seed=42)
    df_4h = _resample_ohlcv(df, "4h")

    any_leak = False
    for target_idx in range(70, 90):
        if target_idx >= len(df_4h):
            continue

        label = df_4h.index[target_idx]
        last_ts = label + pd.Timedelta(hours=3)
        check_ts = label + pd.Timedelta(hours=1)

        if last_ts not in df.index or check_ts not in df.index:
            continue

        df_mod = df.copy()
        df_mod.loc[last_ts, "close"] *= 5.0
        df_mod.loc[last_ts, "high"] *= 5.0

        # 直接比較 4h close 的對齊值（非 regime）
        df_4h_orig = _resample_ohlcv(df, "4h")
        df_4h_mod = _resample_ohlcv(df_mod, "4h")

        close_1h_orig = df_4h_orig["close"].reindex(df.index, method="ffill")
        close_1h_mod = df_4h_mod["close"].reindex(df_mod.index, method="ffill")

        # 在 check_ts（窗口中間），legacy 會反映 last_ts 的修改
        if check_ts in close_1h_orig.index and check_ts in close_1h_mod.index:
            diff = abs(float(close_1h_orig.loc[check_ts]) - float(close_1h_mod.loc[check_ts]))
            if diff > 1e-6:
                any_leak = True
                break

    assert any_leak, (
        "legacy_left_ffill 應有 4h 邊界 look-ahead（"
        "修改窗口末 close 影響窗口中 1h bar），但未偵測到。"
    )


# ══════════════════════════════════════════════════════════
#  Test E: 因果性 — 修改尾部數據（all modes）
# ══════════════════════════════════════════════════════════

@pytest.mark.parametrize("mode", _ALL_MODES)
def test_causality_future_data(mode):
    """修改尾部 200 根 1h，前段 regime 不變（所有 mode 都應 PASS）"""
    df_orig = _make_1h_ohlcv(1200, seed=42)
    df_mod = df_orig.copy()

    df_mod.iloc[-200:, df_mod.columns.get_loc("close")] *= 2.0
    df_mod.iloc[-200:, df_mod.columns.get_loc("high")] *= 2.0
    df_mod.iloc[-200:, df_mod.columns.get_loc("low")] *= 2.0
    df_mod.iloc[-200:, df_mod.columns.get_loc("open")] *= 2.0

    trending_orig, dir_orig = _compute_htf_regime(
        df_orig, **_HTF_PARAMS, mtf_alignment_mode=mode,
    )
    trending_mod, dir_mod = _compute_htf_regime(
        df_mod, **_HTF_PARAMS, mtf_alignment_mode=mode,
    )

    warmup = 300
    check_end = len(df_orig) - 200

    diff_t = np.abs(
        trending_orig.iloc[warmup:check_end].values.astype(float)
        - trending_mod.iloc[warmup:check_end].values.astype(float)
    )
    diff_d = np.abs(
        dir_orig.iloc[warmup:check_end].values
        - dir_mod.iloc[warmup:check_end].values
    )

    max_diff_t = np.nanmax(diff_t) if len(diff_t) > 0 else 0.0
    max_diff_d = np.nanmax(diff_d) if len(diff_d) > 0 else 0.0

    assert max_diff_t < 1e-10, (
        f"[{mode}] causality violation (trending)! max diff={max_diff_t:.6f}"
    )
    assert max_diff_d < 1e-10, (
        f"[{mode}] causality violation (direction)! max diff={max_diff_d:.6f}"
    )


# ══════════════════════════════════════════════════════════
#  Test F: 完整策略截斷不變性（causal modes + entry filters）
# ══════════════════════════════════════════════════════════

@pytest.mark.parametrize("mode", _CAUSAL_MODES)
def test_full_strategy_truncation_with_mtf(mode):
    """完整策略（含 MTF + exit_rules + Phase B filters）的截斷不變性"""
    df_full = _make_1h_ohlcv(1200, seed=42)
    df_trunc = df_full.iloc[:-200].copy()

    ctx = StrategyContext(
        symbol="TEST", market_type="futures", direction="both",
        signal_delay=0,
    )

    params = {
        "kernel_bandwidth": 8.0, "kernel_lookback": 100,
        "envelope_multiplier": 2.0, "envelope_window": 100,
        "adx_period": 14, "adx_threshold": 20.0,
        "slope_window": 10, "slope_threshold": 0.001,
        "atr_period": 14, "stop_loss_atr": 2.0, "take_profit_atr": 3.0,
        "cooldown_bars": 3, "use_mtf": True,
        "mtf_alignment_mode": mode,
        "module_a_enabled": True, "module_b_enabled": False,
        "entry_mode": "pullback_nw",
        "htf_bandwidth": 8.0, "htf_lookback": 50,
        "htf_adx_period": 14, "htf_adx_threshold": 20.0,
        # Enable Phase B filters
        "use_ltf_proxy": True, "ltf_ema_fast": 20, "ltf_ema_slow": 50,
        "use_momentum_confirm": True, "momentum_lookback": 3,
    }

    strategy_func = get_strategy("nw_envelope_regime")
    pos_full = strategy_func(df_full, ctx, params)
    pos_trunc = strategy_func(df_trunc, ctx, params)

    warmup = 400
    overlap_end = len(df_trunc)
    diff = np.abs(
        pos_full.iloc[warmup:overlap_end].values - pos_trunc.iloc[warmup:].values
    )
    max_diff = np.nanmax(diff)

    assert max_diff < 1e-10, (
        f"❌ [{mode}] 完整策略截斷不一致! max diff={max_diff:.6f}"
    )


# ══════════════════════════════════════════════════════════
#  Test G: 仲裁摘要
# ══════════════════════════════════════════════════════════

def test_arbitration_summary(capsys):
    """
    仲裁摘要：對所有 mode 執行快速因果檢測並列表。
    此測試永遠 PASS，僅輸出摘要表格。
    """
    df = _make_1h_ohlcv(800, seed=42)
    warmup = 250

    print("\n")
    print("═" * 72)
    print("  MTF Alignment Mode Arbitration Summary")
    print("═" * 72)
    print(
        f"{'Mode':<25} {'Truncation':<12} {'Boundary':<12} "
        f"{'Causality':<12} {'Verdict'}"
    )
    print("-" * 72)

    for mode in _ALL_MODES:
        # ── Truncation test ──
        df_trunc = df.iloc[:-100].copy()
        t_full, d_full = _compute_htf_regime(
            df, **_HTF_PARAMS, mtf_alignment_mode=mode,
        )
        t_trunc, d_trunc = _compute_htf_regime(
            df_trunc, **_HTF_PARAMS, mtf_alignment_mode=mode,
        )
        overlap = len(df_trunc)
        trunc_ok = np.nanmax(np.abs(
            t_full.iloc[warmup:overlap].values.astype(float)
            - t_trunc.iloc[warmup:].values.astype(float)
        )) < 1e-10

        # ── Boundary test（直接測試 4h close 對齊） ──
        if mode == "right_ffill":
            df_4h = _resample_ohlcv_right(df, "4h")
        else:
            df_4h = _resample_ohlcv(df, "4h")

        boundary_ok = True
        for idx in range(60, 70):
            if idx >= len(df_4h):
                break

            modified_ts, earlier_bars = _get_intra_window_bars(
                mode, df_4h, idx, df.index,
            )
            if not earlier_bars or modified_ts not in df.index:
                continue

            df_m = df.copy()
            df_m.loc[modified_ts, "close"] *= 5.0
            df_m.loc[modified_ts, "high"] *= 5.0

            # 用 4h close 直接測試（避免 ADX 閾值問題）
            if mode == "right_ffill":
                df_4h_o = _resample_ohlcv_right(df, "4h")
                df_4h_m = _resample_ohlcv_right(df_m, "4h")
            else:
                df_4h_o = _resample_ohlcv(df, "4h")
                df_4h_m = _resample_ohlcv(df_m, "4h")

            if mode == "left_shift1_ffill":
                close_1h_o = df_4h_o["close"].shift(1).reindex(
                    df.index, method="ffill"
                ).fillna(0)
                close_1h_m = df_4h_m["close"].shift(1).reindex(
                    df_m.index, method="ffill"
                ).fillna(0)
            else:
                close_1h_o = df_4h_o["close"].reindex(
                    df.index, method="ffill"
                ).fillna(0)
                close_1h_m = df_4h_m["close"].reindex(
                    df_m.index, method="ffill"
                ).fillna(0)

            for ts in earlier_bars:
                if ts in close_1h_o.index and ts in close_1h_m.index:
                    d = abs(float(close_1h_o.loc[ts]) - float(close_1h_m.loc[ts]))
                    if d > 1e-6:
                        boundary_ok = False
                        break
            if not boundary_ok:
                break

        # ── Causality test ──
        df_mod = df.copy()
        df_mod.iloc[-100:, df_mod.columns.get_loc("close")] *= 2.0
        t_o2, _ = _compute_htf_regime(
            df, **_HTF_PARAMS, mtf_alignment_mode=mode,
        )
        t_m2, _ = _compute_htf_regime(
            df_mod, **_HTF_PARAMS, mtf_alignment_mode=mode,
        )
        check_end = len(df) - 100
        causal_ok = np.nanmax(np.abs(
            t_o2.iloc[warmup:check_end].values.astype(float)
            - t_m2.iloc[warmup:check_end].values.astype(float)
        )) < 1e-10

        all_pass = trunc_ok and boundary_ok and causal_ok
        verdict = "✅ PASS" if all_pass else "❌ FAIL"

        print(
            f"{mode:<25} "
            f"{'PASS' if trunc_ok else 'FAIL':<12} "
            f"{'PASS' if boundary_ok else 'FAIL':<12} "
            f"{'PASS' if causal_ok else 'FAIL':<12} "
            f"{verdict}"
        )

    print("═" * 72)
    print("  Recommended: right_ffill (causal + minimal delay)")
    print("  Fallback:    left_shift1_ffill (causal + exchange-standard bars)")
    print("═" * 72)
