"""
Overlay Data Isolation 回歸測試

防止 overlay_params cross-contamination 再次發生。

背景：
    2026-02-27 發現的 bug — 在 portfolio backtest 中，run_symbol_backtest()
    將 per-symbol 的 _lsr_series / _oi_series / _fr_series 注入到 overlay_params dict。
    如果 cfg 是共用 reference（非 deepcopy），symbol 1 的數據會汙染 symbol 2-N。

    根因：config.py to_backtest_dict() 對 _overlay_cfg 用 reference 而非 deepcopy。

修復：
    1. config.py to_backtest_dict() 改用 copy.deepcopy(self._overlay_cfg)
    2. run_backtest.py overlay_params 額外 deepcopy（防禦性）
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest


# ══════════════════════════════════════════════════════════════
# 1. to_backtest_dict() 回傳的 overlay 是獨立副本
# ══════════════════════════════════════════════════════════════


class TestToBacktestDictOverlayIsolation:
    """驗證 AppConfig.to_backtest_dict() 產生的 overlay 不會互相汙染"""

    @pytest.fixture
    def overlay_app_config(self, tmp_path):
        """建立含 lsr_confirmatory overlay 的 AppConfig"""
        import yaml

        cfg_content = {
            "market": {
                "symbols": ["BTCUSDT", "ETHUSDT"],
                "interval": "1h",
                "start": "2024-01-01",
                "end": None,
                "market_type": "futures",
            },
            "futures": {
                "leverage": 1,
                "margin_type": "ISOLATED",
                "position_mode": "ONE_WAY",
                "direction": "both",
            },
            "strategy": {
                "name": "tsmom_ema",
                "params": {"lookback": 168},
                "overlay": {
                    "enabled": True,
                    "mode": "oi_vol+lsr_confirmatory",
                    "params": {
                        "vol_spike_z": 2.0,
                        "lsr_type": "lsr",
                        "lsr_window": 168,
                        "lsr_entry_pctile": 0.85,
                        "lsr_scale_boost": 1.3,
                        "lsr_scale_reduce": 0.3,
                        "oi_confirm_enabled": True,
                        "fr_confirm_enabled": True,
                    },
                },
            },
            "backtest": {
                "initial_cash": 10000,
                "fee_bps": 5,
                "slippage_bps": 3,
                "trade_on": "next_open",
            },
            "position_sizing": {"method": "fixed", "position_pct": 1.0},
            "output": {"report_dir": "./reports"},
        }

        cfg_path = tmp_path / "test_overlay_iso.yaml"
        with open(cfg_path, "w") as f:
            yaml.dump(cfg_content, f)

        from qtrade.config import load_config

        return load_config(str(cfg_path))

    def test_two_calls_produce_independent_overlay_dicts(self, overlay_app_config):
        """
        連續呼叫 to_backtest_dict() 兩次，修改第一份的 overlay params，
        第二份不應受影響。
        """
        cfg = overlay_app_config
        bt_cfg_1 = cfg.to_backtest_dict(symbol="BTCUSDT")
        bt_cfg_2 = cfg.to_backtest_dict(symbol="ETHUSDT")

        # 模擬 run_symbol_backtest 注入 per-symbol 數據
        fake_lsr = pd.Series(np.random.randn(100), name="lsr_btc")
        bt_cfg_1["overlay"]["params"]["_lsr_series"] = fake_lsr
        bt_cfg_1["overlay"]["params"]["_oi_series"] = fake_lsr
        bt_cfg_1["overlay"]["params"]["_fr_series"] = fake_lsr

        # bt_cfg_2 的 overlay 不應包含 BTC 的注入數據
        assert "_lsr_series" not in bt_cfg_2["overlay"]["params"], (
            "overlay_params cross-contamination! "
            "bt_cfg_2 contains _lsr_series injected into bt_cfg_1"
        )
        assert "_oi_series" not in bt_cfg_2["overlay"]["params"], (
            "overlay_params cross-contamination! "
            "bt_cfg_2 contains _oi_series injected into bt_cfg_1"
        )
        assert "_fr_series" not in bt_cfg_2["overlay"]["params"], (
            "overlay_params cross-contamination! "
            "bt_cfg_2 contains _fr_series injected into bt_cfg_1"
        )

    def test_mutation_does_not_affect_source(self, overlay_app_config):
        """
        修改 to_backtest_dict() 的回傳值，不應影響 AppConfig 內部的 _overlay_cfg。
        """
        cfg = overlay_app_config
        bt_cfg = cfg.to_backtest_dict(symbol="BTCUSDT")

        # 注入並修改
        bt_cfg["overlay"]["params"]["_lsr_series"] = "INJECTED"
        bt_cfg["overlay"]["params"]["lsr_scale_boost"] = 999.0

        # 再次呼叫，應取得乾淨的 overlay config
        bt_cfg_clean = cfg.to_backtest_dict(symbol="BTCUSDT")
        assert "_lsr_series" not in bt_cfg_clean["overlay"]["params"], (
            "Mutation leaked back to AppConfig._overlay_cfg!"
        )
        assert bt_cfg_clean["overlay"]["params"]["lsr_scale_boost"] == 1.3, (
            f"Expected lsr_scale_boost=1.3, got "
            f"{bt_cfg_clean['overlay']['params']['lsr_scale_boost']}"
        )


# ══════════════════════════════════════════════════════════════
# 2. run_symbol_backtest 的 overlay_params 不汙染 caller 的 dict
# ══════════════════════════════════════════════════════════════


class TestRunSymbolBacktestOverlayIsolation:
    """驗證 run_symbol_backtest 不會汙染傳入的 cfg dict"""

    def test_overlay_params_not_mutated_after_backtest(self, tmp_path):
        """
        直接構建 cfg dict（模擬 portfolio backtest 場景），
        呼叫 run_symbol_backtest 後確認 cfg 的 overlay.params 未被汙染。

        注意：此測試需要 kline 數據存在才能跑。如果沒有數據就 skip。
        """
        from pathlib import Path

        data_dir = Path("data")
        kline_path = data_dir / "binance" / "futures" / "1h" / "BTCUSDT.parquet"
        if not kline_path.exists():
            pytest.skip("BTCUSDT kline data not available for integration test")

        # 建立 cfg dict（模擬 to_backtest_dict 產出）
        overlay_params = {
            "vol_spike_z": 2.0,
            "overlay_cooldown_bars": 12,
            "atr_period": 14,
            "vol_z_window": 168,
            "lsr_type": "lsr",
            "lsr_window": 168,
            "lsr_entry_pctile": 0.85,
            "lsr_scale_boost": 1.3,
            "lsr_scale_reduce": 0.3,
            "lsr_min_coverage": 0.3,
            "lsr_pos_threshold": 0.05,
            "oi_confirm_enabled": True,
            "fr_confirm_enabled": True,
        }
        original_keys = set(overlay_params.keys())

        cfg = {
            "strategy_name": "tsmom_ema",
            "strategy_params": {
                "lookback": 168,
                "vol_target": 0.15,
                "ema_fast": 20,
                "ema_slow": 50,
                "agree_weight": 1.0,
                "disagree_weight": 0.3,
            },
            "initial_cash": 10000,
            "fee_bps": 5,
            "slippage_bps": 3,
            "interval": "1h",
            "market_type": "futures",
            "direction": "both",
            "validate_data": True,
            "clean_data_before": True,
            "start": "2024-01-01",
            "end": "2024-06-01",
            "leverage": 1,
            "position_sizing": {"method": "fixed", "position_pct": 1.0},
            "funding_rate": {"enabled": False},
            "slippage_model": {"enabled": False},
            "overlay": {
                "enabled": True,
                "mode": "oi_vol+lsr_confirmatory",
                "params": overlay_params,
            },
        }

        from qtrade.backtest.run_backtest import run_symbol_backtest

        # 第一次呼叫
        run_symbol_backtest(
            "BTCUSDT",
            kline_path,
            cfg,
            strategy_name="tsmom_ema",
            data_dir=data_dir,
        )

        # overlay_params 不應被注入 _lsr_series / _oi_series / _fr_series
        current_keys = set(overlay_params.keys())
        injected = current_keys - original_keys
        assert not injected, (
            f"run_symbol_backtest mutated the caller's overlay_params! "
            f"Injected keys: {injected}. "
            f"This causes cross-contamination in portfolio backtests."
        )


# ══════════════════════════════════════════════════════════════
# 3. deepcopy 正確性驗證
# ══════════════════════════════════════════════════════════════


class TestDeepCopyBehavior:
    """直接驗證 deepcopy 機制正確工作"""

    def test_nested_dict_deepcopy_isolates_mutations(self):
        """
        模擬 overlay_cfg 被 deepcopy 後，修改副本不影響原件。
        這是 bug fix 的核心機制。
        """
        original = {
            "enabled": True,
            "mode": "oi_vol+lsr_confirmatory",
            "params": {
                "lsr_type": "lsr",
                "lsr_window": 168,
                "oi_confirm_enabled": True,
            },
        }

        # 模擬 to_backtest_dict 的 deepcopy 行為
        copy1 = copy.deepcopy(original)
        copy2 = copy.deepcopy(original)

        # 注入 per-symbol 數據到 copy1
        copy1["params"]["_lsr_series"] = pd.Series([1, 2, 3])
        copy1["params"]["_oi_series"] = pd.Series([4, 5, 6])

        # copy2 和 original 都不應受影響
        assert "_lsr_series" not in copy2["params"]
        assert "_oi_series" not in copy2["params"]
        assert "_lsr_series" not in original["params"]
        assert "_oi_series" not in original["params"]

    def test_shallow_copy_would_fail(self):
        """
        證明淺拷貝無法防止汙染（反向測試，展示 bug 機制）。
        """
        original = {
            "enabled": True,
            "params": {"lsr_type": "lsr"},
        }

        # 淺拷貝：外層 dict 是新的，但 nested dict 仍是同一個引用
        shallow = dict(original)
        shallow["params"]["_injected"] = True

        # 淺拷貝下，original 也被汙染了
        assert "_injected" in original["params"], (
            "Expected shallow copy to share nested dict references"
        )

        # 清理
        del original["params"]["_injected"]
