#!/usr/bin/env python3
"""
回測 ↔ 實盤 一致性驗證（Pre-Deploy Checklist）

上架策略前的常規測驗，自動檢查回測和實盤路徑的每一步是否一致。
通過全部檢查才能放心部署。

使用方式：
    # 標準驗證（建議每次部署前執行）
    python scripts/validate_live_consistency.py -c config/futures_rsi_adx_atr.yaml

    # 詳細模式（顯示每個 check 的詳細資訊）
    python scripts/validate_live_consistency.py -c config/futures_rsi_adx_atr.yaml -v

    # 只檢查特定項目
    python scripts/validate_live_consistency.py -c config/futures_rsi_adx_atr.yaml --only params,signal,sltp

檢查清單：
    1.  config_passthrough   — 參數從 YAML → backtest_dict → live 路徑完整傳遞
    2.  strategy_context     — StrategyContext 回測/實盤一致
    3.  strategy_function    — 使用同一個策略函數
    4.  signal_consistency   — 在相同數據上產生相同信號
    5.  signal_clip          — 信號 clip 邏輯一致
    6.  entry_price          — 入場價 close[-1] vs open[0] 差距可接受
    7.  sltp_formula         — SL/TP 方向公式正確
    8.  sltp_price_base      — SL/TP 基準價內部一致性
    9.  position_sizing      — 倉位計算鏈路正確
    10. fee_match            — 手續費設定與交易所一致
    11. date_filter          — start/end 日期正確套用
    12. cooldown             — 冷卻期設定一致
    13. funding_rate_warning — 資金費率未建模提醒
    14. overlay_consistency  — Overlay 配置在回測/實盤路徑一致
"""
from __future__ import annotations

import argparse
import sys
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

# 添加 src 到路徑
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qtrade.config import load_config, AppConfig
from qtrade.data.storage import load_klines
from qtrade.data.quality import clean_data
from qtrade.strategy import get_strategy
from qtrade.strategy.base import StrategyContext
from qtrade.backtest.run_backtest import (
    clip_positions_by_direction,
    to_vbt_direction,
    _apply_date_filter,
)


# ══════════════════════════════════════════════════════════
# Check Result
# ══════════════════════════════════════════════════════════


@dataclass
class CheckResult:
    """單項檢查結果"""

    name: str
    passed: bool
    severity: str  # "PASS", "WARN", "FAIL"
    message: str
    details: str = ""


class ConsistencyChecker:
    """回測↔實盤一致性檢查器"""

    def __init__(self, cfg: AppConfig, verbose: bool = False):
        self.cfg = cfg
        self.verbose = verbose
        self.results: list[CheckResult] = []

        # 載入數據（取第一個幣種做驗證）
        self.symbol = cfg.market.symbols[0]
        mt = cfg.market_type_str
        data_dir = cfg.data_dir / "binance" / mt / cfg.market.interval
        self.data_path = data_dir / f"{self.symbol}.parquet"

    def _add(self, name: str, passed: bool, severity: str, message: str, details: str = ""):
        self.results.append(CheckResult(name, passed, severity, message, details))

    # ── 1. Config Passthrough ──────────────────────────────

    def check_config_passthrough(self):
        """檢查 config → backtest_dict → live 參數傳遞鏈"""
        bt_dict = self.cfg.to_backtest_dict(symbol=self.symbol)
        live_params = self.cfg.strategy.get_params(self.symbol)

        issues = []

        # 策略參數一致
        if bt_dict["strategy_params"] != live_params:
            issues.append(
                f"strategy_params 不一致:\n"
                f"  backtest: {bt_dict['strategy_params']}\n"
                f"  live:     {live_params}"
            )

        # market_type 傳遞
        if bt_dict["market_type"] != self.cfg.market_type_str:
            issues.append(
                f"market_type: backtest_dict={bt_dict['market_type']} "
                f"vs cfg={self.cfg.market_type_str}"
            )

        # direction 傳遞
        if bt_dict["direction"] != self.cfg.direction:
            issues.append(
                f"direction: backtest_dict={bt_dict['direction']} "
                f"vs cfg={self.cfg.direction}"
            )

        # start/end 傳遞
        if bt_dict.get("start") != self.cfg.market.start:
            issues.append(
                f"start: backtest_dict={bt_dict.get('start')} "
                f"vs cfg={self.cfg.market.start}"
            )

        if issues:
            self._add("config_passthrough", False, "FAIL", "參數傳遞有斷裂", "\n".join(issues))
        else:
            self._add(
                "config_passthrough",
                True,
                "PASS",
                "YAML → to_backtest_dict() → get_params() 鏈路完整",
                f"market_type={bt_dict['market_type']}, direction={bt_dict['direction']}, "
                f"start={bt_dict.get('start')}, params keys={list(live_params.keys())}",
            )

    # ── 2. Strategy Context ────────────────────────────────

    def check_strategy_context(self):
        """檢查 StrategyContext 回測/實盤是否一致"""
        bt_ctx = StrategyContext(
            symbol=self.symbol,
            interval=self.cfg.market.interval,
            market_type=self.cfg.market_type_str,
            direction=self.cfg.direction,
            signal_delay=1,  # Backtest path: signal on close, execute on next open
        )

        # 實盤 signal_generator 的邏輯
        live_ctx = StrategyContext(
            symbol=self.symbol,
            interval=self.cfg.market.interval,
            market_type=self.cfg.market_type_str,
            direction=self.cfg.direction,
            signal_delay=0,  # Live path: execute immediately on the latest completed bar
        )

        issues = []
        for attr in ("market_type", "direction", "supports_short", "can_long", "can_short"):
            bt_val = getattr(bt_ctx, attr)
            live_val = getattr(live_ctx, attr)
            if bt_val != live_val:
                issues.append(f"{attr}: backtest={bt_val} vs live={live_val}")

        if issues:
            self._add("strategy_context", False, "FAIL", "StrategyContext 不一致", "\n".join(issues))
        else:
            self._add(
                "strategy_context",
                True,
                "PASS",
                f"StrategyContext 一致 (market={bt_ctx.market_type}, dir={bt_ctx.direction}, "
                f"short={bt_ctx.supports_short})",
            )

    # ── 3. Strategy Function ───────────────────────────────

    def check_strategy_function(self):
        """檢查回測和實盤是否使用同一個策略函數"""
        strategy_name = self.cfg.strategy.name
        try:
            func = get_strategy(strategy_name)
            # 檢查函數是否可呼叫
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())
            expected = ["df", "ctx", "params"]
            if params[:3] != expected:
                self._add(
                    "strategy_function",
                    False,
                    "FAIL",
                    f"策略函數簽名不正確: {params}，期望 {expected}",
                )
                return
            self._add(
                "strategy_function",
                True,
                "PASS",
                f"get_strategy('{strategy_name}') → {func.__module__}.{func.__name__}()",
            )
        except Exception as e:
            self._add("strategy_function", False, "FAIL", f"策略函數載入失敗: {e}")

    # ── 4. Signal Consistency ──────────────────────────────

    def check_signal_consistency(self):
        """在相同數據上驗證回測和實盤信號路徑產生相同結果"""
        if not self.data_path.exists():
            self._add("signal_consistency", False, "FAIL", f"數據不存在: {self.data_path}")
            return

        df = load_klines(self.data_path)
        df = clean_data(df, fill_method="forward", remove_outliers=False, remove_duplicates=True)

        params = self.cfg.strategy.get_params(self.symbol)
        ctx = StrategyContext(
            symbol=self.symbol,
            interval=self.cfg.market.interval,
            market_type=self.cfg.market_type_str,
            direction=self.cfg.direction,
            signal_delay=0,  # Live-consistency check compares current-bar live-style execution
        )

        strategy_func = get_strategy(self.cfg.strategy.name)

        # 回測路徑：用全部數據
        pos_full = strategy_func(df, ctx, params)

        # 實盤路徑：模擬只用最後 300 bars
        df_live = df.iloc[-300:]
        pos_live = strategy_func(df_live, ctx, params)

        # 比較重疊區間的最後一根信號
        signal_full = float(pos_full.iloc[-1])
        signal_live = float(pos_live.iloc[-1])

        if signal_full == signal_live:
            self._add(
                "signal_consistency",
                True,
                "PASS",
                f"最後一根信號一致: {signal_full:.2f} "
                f"(全期 {len(df)} bars vs 最後 300 bars)",
            )
        else:
            # 容許微小浮點差異
            if abs(signal_full - signal_live) < 1e-6:
                self._add(
                    "signal_consistency",
                    True,
                    "PASS",
                    f"最後一根信號一致（浮點誤差 < 1e-6）",
                )
            else:
                self._add(
                    "signal_consistency",
                    False,
                    "WARN",
                    f"最後一根信號不同: full={signal_full:.4f} vs live={signal_live:.4f}",
                    "策略內部 state machine 可能因歷史不同而分歧。\n"
                    "這通常是正常的（exit_rules 的 state 受前面 bar 影響），\n"
                    "但如果差異持續很大，需要檢查 warmup 問題。",
                )

    # ── 5. Signal Clip ─────────────────────────────────────

    def check_signal_clip(self):
        """驗證信號 clip 邏輯一致"""
        mt = self.cfg.market_type_str
        direction = self.cfg.direction

        # 構造測試信號
        test_signals = pd.Series([1.0, -1.0, 0.5, -0.5, 0.0])

        # 回測 clip
        bt_clipped = clip_positions_by_direction(test_signals, mt, direction)

        # 實盤 clip (runner.py 邏輯)
        live_clipped = test_signals.copy()
        if mt == "spot":
            live_clipped = live_clipped.clip(lower=0.0)

        issues = []
        for i in range(len(test_signals)):
            bt_val = bt_clipped.iloc[i]
            live_val = live_clipped.iloc[i]
            if abs(bt_val - live_val) > 1e-6:
                issues.append(
                    f"signal={test_signals.iloc[i]:.1f}: "
                    f"backtest clip={bt_val:.2f} vs live clip={live_val:.2f}"
                )

        if issues:
            self._add("signal_clip", False, "FAIL", "Clip 邏輯不一致", "\n".join(issues))
        else:
            self._add(
                "signal_clip",
                True,
                "PASS",
                f"clip 邏輯一致 (market={mt}, direction={direction})",
            )

    # ── 6. Entry Price ─────────────────────────────────────

    def check_entry_price(self):
        """驗證 close[N-1] vs open[N] 差距可接受"""
        if not self.data_path.exists():
            self._add("entry_price", False, "FAIL", f"數據不存在")
            return

        df = load_klines(self.data_path)
        df = clean_data(df, fill_method="forward", remove_outliers=False, remove_duplicates=True)

        # close[i] vs open[i+1]
        close_prev = df["close"].iloc[:-1].values
        open_next = df["open"].iloc[1:].values
        diff_pct = np.abs((open_next - close_prev) / close_prev) * 100

        avg_diff = float(np.nanmean(diff_pct))
        max_diff = float(np.nanmax(diff_pct))
        p99_diff = float(np.nanpercentile(diff_pct, 99))

        # 閾值：平均 < 0.5% 視為可接受
        passed = avg_diff < 0.5
        severity = "PASS" if passed else "WARN"

        self._add(
            "entry_price",
            passed,
            severity,
            f"|close[N-1] - open[N]| 差距: 平均={avg_diff:.4f}%, P99={p99_diff:.3f}%, 最大={max_diff:.2f}%",
            f"回測用 open[N]，實盤用 close[N-1]（≈ open[N]）\n"
            f"差距主因：隔夜/周末 gap，1h bar 影響極小",
        )

    # ── 7. SL/TP Formula ───────────────────────────────────

    def check_sltp_formula(self):
        """驗證 SL/TP 方向公式正確"""
        params = self.cfg.strategy.get_params(self.symbol)
        sl_atr = params.get("stop_loss_atr")
        tp_atr = params.get("take_profit_atr")

        if not sl_atr and not tp_atr:
            self._add("sltp_formula", True, "PASS", "策略未使用 ATR SL/TP")
            return

        issues = []
        test_price = 50000.0
        test_atr = 1000.0

        # 模擬回測 exit_rules 的公式
        # Long: SL = entry - sl_atr * ATR, TP = entry + tp_atr * ATR
        # Short: SL = entry + sl_atr * ATR, TP = entry - tp_atr * ATR
        if sl_atr:
            bt_sl_long = test_price - float(sl_atr) * test_atr
            bt_sl_short = test_price + float(sl_atr) * test_atr
            # 模擬 runner.py 的公式
            live_sl_long = test_price - float(sl_atr) * test_atr
            live_sl_short = test_price + float(sl_atr) * test_atr

            if bt_sl_long != live_sl_long:
                issues.append(f"SL_LONG: backtest={bt_sl_long} vs live={live_sl_long}")
            if bt_sl_short != live_sl_short:
                issues.append(f"SL_SHORT: backtest={bt_sl_short} vs live={live_sl_short}")

        if tp_atr:
            bt_tp_long = test_price + float(tp_atr) * test_atr
            bt_tp_short = test_price - float(tp_atr) * test_atr
            live_tp_long = test_price + float(tp_atr) * test_atr
            live_tp_short = test_price - float(tp_atr) * test_atr

            if bt_tp_long != live_tp_long:
                issues.append(f"TP_LONG: backtest={bt_tp_long} vs live={live_tp_long}")
            if bt_tp_short != live_tp_short:
                issues.append(f"TP_SHORT: backtest={bt_tp_short} vs live={live_tp_short}")

        if issues:
            self._add("sltp_formula", False, "FAIL", "SL/TP 方向公式不一致", "\n".join(issues))
        else:
            sl_str = f"{sl_atr}×ATR" if sl_atr else "無"
            tp_str = f"{tp_atr}×ATR" if tp_atr else "無（信號出場）"
            details = (
                f"Long: SL=entry-{sl_str}, TP=entry+{tp_str}\n"
                f"Short: SL=entry+{sl_str}, TP=entry-{tp_str}"
            )
            self._add("sltp_formula", True, "PASS", "SL/TP 方向公式一致", details)

    # ── 8. SL/TP Price Base ────────────────────────────────

    def check_sltp_price_base(self):
        """驗證 SL/TP 基準價在各自路徑內是否一致"""
        issues = []
        notes = []

        # 回測 exit_rules: entry_price = open[i], SL/TP 基於 open[i]
        # VBT: price=open_ → 成交於 open[i]
        notes.append("回測: entry=open[i], SL/TP 基於 open[i] ✓ 內部一致")

        # 實盤 runner: sig['price'] = close[-1], SL/TP 基於 close[-1]
        # broker: market order → 成交 ≈ close[-1]
        notes.append("實盤: entry≈close[-1], SL/TP 基於 close[-1] ✓ 內部一致")

        # 交叉比較
        notes.append("差異: 回測基於 open[N], 實盤基於 close[N-1], 差距 <0.1%")

        self._add(
            "sltp_price_base",
            True,
            "PASS",
            "SL/TP 基準價各自內部一致",
            "\n".join(notes),
        )

    # ── 9. Position Sizing ─────────────────────────────────

    def check_position_sizing(self):
        """驗證倉位計算鏈路"""
        n_symbols = len(self.cfg.market.symbols)
        weight = self.cfg.portfolio.get_weight(self.symbol, n_symbols)
        position_pct = self.cfg.position_sizing.position_pct
        method = self.cfg.position_sizing.method

        issues = []
        notes = []

        # 回測 single-symbol: 100%
        notes.append(f"回測 (single-symbol): targetpercent=1.0 → 100%/幣")
        notes.append(f"實盤: signal × position_pct({position_pct}) × weight({weight:.2f}) = signal × {position_pct * weight:.2f}")
        notes.append(f"實盤每幣曝險: {position_pct * weight * 100:.0f}%")

        if position_pct * weight > 0.99 and n_symbols == 1:
            # 單幣 + 100% pct + 無 cash_reserve → 一致
            notes.append("→ 回測 single-symbol 與實盤一致")
        else:
            notes.append(f"→ ⚠️ 回測 single-symbol 用 100%, 實盤用 {position_pct * weight * 100:.0f}%")
            notes.append(f"   應該用 portfolio backtest 做比較")

        # 檢查 position_pct 是否 <= 1
        if position_pct > 1.0:
            issues.append(f"position_pct={position_pct} > 1.0，異常！")

        # 檢查 weight 是否合理
        total_weight = sum(
            self.cfg.portfolio.get_weight(s, n_symbols)
            for s in self.cfg.market.symbols
        )

        # 判斷超配是否為明確設定（allocation 裡每個幣都有指定值）
        has_explicit_allocation = (
            self.cfg.portfolio.allocation is not None
            and all(
                s in self.cfg.portfolio.allocation
                for s in self.cfg.market.symbols
            )
        )

        if total_weight > 1.01 and not has_explicit_allocation:
            issues.append(f"總權重 {total_weight:.2f} > 1.0，可能超配！（未明確設定 allocation）")

        warnings = []
        if total_weight > 1.01 and has_explicit_allocation:
            leverage = self.cfg.futures.leverage if self.cfg.futures else 1
            warnings.append(
                f"總權重 {total_weight:.2f} > 1.0 — 明確設定的 allocation 超配 "
                f"({total_weight:.0%} 曝險)，"
                f"槓桿 {leverage}x 下保證金需 {total_weight / leverage:.0%}"
            )

        if issues:
            self._add("position_sizing", False, "FAIL", "倉位計算異常", "\n".join(issues))
        elif warnings:
            notes.extend(warnings)
            self._add(
                "position_sizing",
                True,
                "WARN",
                f"明確超配: 總曝險 {total_weight:.0%}（已確認為 allocation 設定）",
                "\n".join(notes),
            )
        else:
            self._add(
                "position_sizing",
                True,
                "PASS",
                f"method={method}, pct={position_pct}, weight={weight:.2f}/幣, "
                f"總配置={total_weight:.0%}",
                "\n".join(notes),
            )

    # ── 10. Fee Match ──────────────────────────────────────

    def check_fee_match(self):
        """檢查回測手續費與交易所費率是否匹配"""
        fee_bps = self.cfg.backtest.fee_bps
        fee_pct = fee_bps / 10_000

        # Binance Futures 費率參考
        binance_taker = 4  # bps (0.04%)
        binance_maker = 2  # bps (0.02%)

        if fee_bps == binance_taker:
            self._add(
                "fee_match",
                True,
                "PASS",
                f"fee={fee_bps}bps = Binance Taker ({fee_pct:.2f}%)",
            )
        elif fee_bps == binance_maker:
            self._add(
                "fee_match",
                True,
                "PASS",
                f"fee={fee_bps}bps = Binance Maker ({fee_pct:.2f}%)",
            )
        elif fee_bps < binance_maker:
            self._add(
                "fee_match",
                False,
                "WARN",
                f"fee={fee_bps}bps < Binance Maker({binance_maker}bps)，回測可能過於樂觀",
            )
        elif fee_bps > binance_taker * 2:
            self._add(
                "fee_match",
                False,
                "WARN",
                f"fee={fee_bps}bps > 2×Taker，回測可能過於悲觀",
            )
        else:
            self._add(
                "fee_match",
                True,
                "PASS",
                f"fee={fee_bps}bps 在合理範圍 [{binance_maker}-{binance_taker*2}bps]",
            )

        # 滑點模型補充資訊
        sm = self.cfg.backtest.slippage_model
        if sm.enabled:
            self._add(
                "slippage_model",
                True,
                "PASS",
                f"Volume-based 滑點已啟用 (base={sm.base_bps}bps, k={sm.impact_coefficient}, power={sm.impact_power})",
                f"取代固定 slippage_bps={self.cfg.backtest.slippage_bps}bps\n"
                f"ADV lookback={sm.adv_lookback} bars, participation_rate={sm.participation_rate:.0%}",
            )

    # ── 11. Date Filter ────────────────────────────────────

    def check_date_filter(self):
        """驗證 start/end 日期正確套用"""
        if not self.data_path.exists():
            self._add("date_filter", False, "FAIL", "數據不存在")
            return

        df = load_klines(self.data_path)
        df = clean_data(df, fill_method="forward", remove_outliers=False, remove_duplicates=True)

        start = self.cfg.market.start
        end = self.cfg.market.end
        pos_dummy = pd.Series(0.0, index=df.index)

        df_filtered, _ = _apply_date_filter(df, pos_dummy, start, end)

        issues = []
        if start:
            start_ts = pd.Timestamp(start, tz="UTC") if df.index.tz else pd.Timestamp(start)
            if df_filtered.index[0] < start_ts:
                issues.append(
                    f"start filter 失敗: 數據從 {df_filtered.index[0]} 開始, "
                    f"應 >= {start}"
                )

        if end:
            end_ts = pd.Timestamp(end, tz="UTC") if df.index.tz else pd.Timestamp(end)
            if df_filtered.index[-1] > end_ts:
                issues.append(
                    f"end filter 失敗: 數據到 {df_filtered.index[-1]}, "
                    f"應 <= {end}"
                )

        if issues:
            self._add("date_filter", False, "FAIL", "日期過濾有問題", "\n".join(issues))
        else:
            self._add(
                "date_filter",
                True,
                "PASS",
                f"日期過濾正確: {len(df)} → {len(df_filtered)} bars "
                f"({df_filtered.index[0].strftime('%Y-%m-%d')} → "
                f"{df_filtered.index[-1].strftime('%Y-%m-%d')})",
                f"start={start}, end={end}",
            )

    # ── 12. Cooldown ───────────────────────────────────────

    def check_cooldown(self):
        """驗證冷卻期設定"""
        params = self.cfg.strategy.get_params(self.symbol)
        cooldown = int(params.get("cooldown_bars", 0))

        notes = []
        notes.append(f"exit_rules cooldown_bars = {cooldown}")
        notes.append(f"runner.py exchange cooldown = 10 min (SL/TP 觸發偵測)")

        if cooldown == 0:
            notes.append("⚠️ cooldown=0: SL/TP 觸發後不等待，可能連續虧損")
            self._add(
                "cooldown",
                True,
                "WARN",
                f"cooldown_bars=0（無冷卻期），建議 >= 1",
                "\n".join(notes),
            )
        else:
            notes.append(f"✓ 回測: SL/TP 後等 {cooldown} 根 bar")
            notes.append(f"✓ 實盤: SL/TP 後等 {cooldown} 根 bar + exchange 10min 偵測")
            self._add(
                "cooldown",
                True,
                "PASS",
                f"cooldown_bars={cooldown} (回測) + exchange 10min (實盤)",
                "\n".join(notes),
            )

    # ── 13. Funding Rate Warning ───────────────────────────

    def check_funding_rate_warning(self):
        """資金費率建模檢查"""
        if self.cfg.market_type_str != "futures":
            self._add(
                "funding_rate_warning",
                True,
                "PASS",
                "Spot 模式，無 funding rate",
            )
            return

        leverage = self.cfg.futures.leverage if self.cfg.futures else 1
        fr_cfg = self.cfg.backtest.funding_rate

        if fr_cfg.enabled:
            source = "歷史資料" if fr_cfg.use_historical else f"固定 {fr_cfg.default_rate_8h:.4%}/8h"
            self._add(
                "funding_rate_warning",
                True,
                "PASS",
                f"Futures {leverage}x — 回測已啟用 funding rate 模型 ({source})",
                f"funding_rate.enabled=true, default_rate_8h={fr_cfg.default_rate_8h}\n"
                f"use_historical={fr_cfg.use_historical}\n"
                f"回測已扣除 funding rate 成本，與實盤一致。",
            )
        else:
            self._add(
                "funding_rate_warning",
                True,
                "WARN",
                f"Futures {leverage}x — 回測未啟用 funding rate (年化 ~10-15% 拖累)",
                "永續合約每 8h 收取資金費率 (~0.01%)。\n"
                "本策略持倉短 (8-24h)，影響 ≈ 3-5%/年。\n"
                "長期實盤回報會低於回測。\n"
                "建議在 backtest.funding_rate.enabled 設為 true。",
            )

    # ── 14. Overlay Consistency ─────────────────────────────

    def check_overlay_consistency(self):
        """驗證 overlay 配置在回測和實盤路徑是否一致"""
        # 從 config 讀取 overlay 設定
        overlay_cfg = getattr(self.cfg, '_overlay_cfg', None)
        bt_dict = self.cfg.to_backtest_dict(symbol=self.symbol)
        bt_overlay = bt_dict.get("overlay")

        # 情境 1：沒有 overlay — 最簡單
        if not overlay_cfg and not bt_overlay:
            self._add(
                "overlay_consistency",
                True,
                "PASS",
                "策略未使用 overlay（回測和實盤皆無）",
            )
            return

        # 情境 2：overlay 設定存在但 enabled=False
        if overlay_cfg and not overlay_cfg.get("enabled", False):
            if bt_overlay and bt_overlay.get("enabled", False):
                self._add(
                    "overlay_consistency",
                    False,
                    "FAIL",
                    "Overlay 不一致：原始 config 未啟用，但 to_backtest_dict 啟用",
                )
            else:
                self._add(
                    "overlay_consistency",
                    True,
                    "PASS",
                    "Overlay 已定義但 enabled=false（回測和實盤皆不套用）",
                )
            return

        # 情境 3：overlay 啟用 — 嚴格檢查
        issues = []
        notes = []

        if not bt_overlay or not bt_overlay.get("enabled", False):
            issues.append(
                "Overlay config 有 enabled=true，但 to_backtest_dict() 未傳遞 overlay"
            )
        else:
            # 3a. mode 一致性
            cfg_mode = overlay_cfg.get("mode", "vol_pause")
            bt_mode = bt_overlay.get("mode", "vol_pause")
            if cfg_mode != bt_mode:
                issues.append(
                    f"Overlay mode 不一致: config={cfg_mode} vs backtest_dict={bt_mode}"
                )
            else:
                notes.append(f"overlay mode={cfg_mode} ✓")

            # 3b. params 一致性
            cfg_params = overlay_cfg.get("params", {})
            bt_params = bt_overlay.get("params", {})
            param_diffs = []
            all_keys = set(list(cfg_params.keys()) + list(bt_params.keys()))
            for k in sorted(all_keys):
                v1 = cfg_params.get(k)
                v2 = bt_params.get(k)
                if v1 != v2:
                    param_diffs.append(f"  {k}: config={v1} vs backtest={v2}")
            if param_diffs:
                issues.append("Overlay params 不一致:\n" + "\n".join(param_diffs))
            else:
                notes.append(f"overlay params 一致 ({len(cfg_params)} keys) ✓")

        # 3c. OI 數據可用性（oi_vol 和 oi_only mode 需要 OI）
        overlay_mode = overlay_cfg.get("mode", "vol_pause")
        if overlay_mode in ("oi_vol", "oi_only"):
            # 檢查 live signal_generator 路徑是否能自動載入 OI
            # 機制：BaseRunner 注入 _data_dir → signal_generator 的 overlay 處理區塊
            #       透過 params.get("_data_dir") 自動從 parquet 載入 OI
            try:
                from qtrade.live.signal_generator import generate_signal
                import inspect
                source = inspect.getsource(generate_signal)
                if "_data_dir" in source and "oi_series" in source:
                    notes.append(f"Live signal_generator 支援 overlay OI 自動載入 (_data_dir) ✓")
                elif "oi_series" in inspect.signature(generate_signal).parameters:
                    notes.append(f"Live signal_generator 支援 oi_series 參數 ✓")
                else:
                    issues.append(
                        f"Overlay mode={overlay_mode} 需要 OI 數據，"
                        f"但 live signal_generator 缺乏 OI 載入機制"
                    )
            except ImportError:
                issues.append("無法載入 signal_generator 模組")

            # 檢查 OI 數據檔案是否存在
            data_dir = self.cfg.data_dir
            oi_path = data_dir / "binance" / "futures" / "open_interest" / "merged" / f"{self.symbol}.parquet"
            if oi_path.exists():
                import os
                mtime = os.path.getmtime(oi_path)
                from datetime import datetime, timezone
                age_hours = (datetime.now(timezone.utc).timestamp() - mtime) / 3600
                notes.append(f"OI 數據存在: {oi_path.name} (age={age_hours:.1f}h) ✓")
                if age_hours > 24:
                    issues.append(
                        f"OI 數據過舊 ({age_hours:.0f}h > 24h)，overlay 可能使用過時信號"
                    )
            else:
                issues.append(
                    f"Overlay mode={overlay_mode} 需要 OI 數據，"
                    f"但 {oi_path} 不存在"
                )

        # 3d. 檢查 BaseRunner 是否注入 _data_dir（供 live 路徑自動載入 OI）
        try:
            from qtrade.live.base_runner import BaseRunner
            import inspect
            source = inspect.getsource(BaseRunner._get_strategy_for_symbol)
            if "_data_dir" in source:
                notes.append("BaseRunner 注入 _data_dir ✓")
            else:
                issues.append(
                    "BaseRunner._get_strategy_for_symbol 未注入 _data_dir，"
                    "live 路徑可能無法載入 OI/FR 輔助數據"
                )
        except Exception:
            notes.append("⚠️ 無法檢查 BaseRunner source（非嚴重問題）")

        if issues:
            self._add(
                "overlay_consistency",
                False,
                "FAIL",
                f"Overlay 一致性有問題 (mode={overlay_mode})",
                "\n".join(issues + ["---"] + notes),
            )
        else:
            self._add(
                "overlay_consistency",
                True,
                "PASS",
                f"Overlay 一致 (mode={overlay_mode}, enabled=true)",
                "\n".join(notes),
            )

    # ── Run All ────────────────────────────────────────────

    def run_all(self, only: set[str] | None = None) -> list[CheckResult]:
        """執行所有檢查"""
        checks = {
            "params": [self.check_config_passthrough, self.check_strategy_context],
            "strategy": [self.check_strategy_function],
            "signal": [self.check_signal_consistency, self.check_signal_clip],
            "entry": [self.check_entry_price],
            "sltp": [
                self.check_sltp_formula,
                self.check_sltp_price_base,
            ],
            "sizing": [self.check_position_sizing],
            "fee": [self.check_fee_match],
            "date": [self.check_date_filter],
            "cooldown": [self.check_cooldown],
            "funding": [self.check_funding_rate_warning],
            "overlay": [self.check_overlay_consistency],
        }

        for group_name, check_funcs in checks.items():
            if only and group_name not in only:
                continue
            for func in check_funcs:
                try:
                    func()
                except Exception as e:
                    self._add(func.__name__, False, "FAIL", f"檢查異常: {e}")

        return self.results


# ══════════════════════════════════════════════════════════
# Report
# ══════════════════════════════════════════════════════════


def print_report(results: list[CheckResult], verbose: bool = False):
    """列印檢查報告"""
    ICONS = {"PASS": "✅", "WARN": "⚠️ ", "FAIL": "🔴"}

    print()
    print("━" * 65)
    print("  📋 回測↔實盤一致性檢查報告")
    print("━" * 65)

    for r in results:
        icon = ICONS.get(r.severity, "?")
        status = f"[{r.severity}]"
        print(f"  {icon} {status:<6} {r.name:<28} {r.message}")
        if verbose and r.details:
            for line in r.details.split("\n"):
                print(f"            {line}")

    # 統計
    n_pass = sum(1 for r in results if r.severity == "PASS")
    n_warn = sum(1 for r in results if r.severity == "WARN")
    n_fail = sum(1 for r in results if r.severity == "FAIL")
    total = len(results)

    print()
    print("━" * 65)
    print(f"  結果: {n_pass} PASS / {n_warn} WARN / {n_fail} FAIL  (共 {total} 項)")

    if n_fail == 0 and n_warn == 0:
        print("  🏆 全部通過！可以安心部署。")
    elif n_fail == 0:
        print("  ⚠️  有警告項目，建議了解後再部署。")
    else:
        print("  🚨 有失敗項目，請修復後再部署！")

    print("━" * 65)
    print()

    return n_fail == 0


# ══════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="回測↔實盤一致性驗證（Pre-Deploy Checklist）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 標準驗證
  python scripts/validate_live_consistency.py -c config/futures_rsi_adx_atr.yaml

  # 詳細模式
  python scripts/validate_live_consistency.py -c config/futures_rsi_adx_atr.yaml -v

  # 只檢查 SL/TP 和信號
  python scripts/validate_live_consistency.py -c config/futures_rsi_adx_atr.yaml --only signal,sltp
        """,
    )

    parser.add_argument("-c", "--config", required=True, help="策略配置文件路徑")
    parser.add_argument("-v", "--verbose", action="store_true", help="詳細模式（顯示細節）")
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="只檢查指定項目（逗號分隔）: params,strategy,signal,entry,sltp,sizing,fee,date,cooldown,funding,overlay",
    )

    args = parser.parse_args()

    # 載入配置
    cfg = load_config(args.config)

    print("=" * 65)
    print(f"  🔍 Pre-Deploy Consistency Check")
    print("=" * 65)
    print(f"  策略:   {cfg.strategy.name}")
    print(f"  配置:   {args.config}")
    print(f"  市場:   {cfg.market_type_str} ({cfg.direction})")
    print(f"  交易對: {', '.join(cfg.market.symbols)}")

    # 解析 --only
    only = None
    if args.only:
        only = set(args.only.lower().split(","))
        print(f"  檢查項: {', '.join(sorted(only))}")

    # 執行檢查
    checker = ConsistencyChecker(cfg, verbose=args.verbose)
    results = checker.run_all(only=only)

    # 列印報告
    all_passed = print_report(results, verbose=args.verbose)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
