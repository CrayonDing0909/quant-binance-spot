"""
Red Flag Detection — Unit & Integration Tests

覆蓋：
  1. 正常策略統計 → 無紅旗
  2. 各個 flag 獨立觸發
  3. 多個 flag 同時觸發
  4. 邊界值測試
  5. vectorbt stats 格式（鍵名相容）
  6. 空 / 缺失鍵 → 不報錯
"""
from __future__ import annotations

import pytest

from qtrade.validation.red_flags import check_red_flags, print_red_flags, RedFlag


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def healthy_stats() -> dict:
    """正常策略的統計（不應觸發任何紅旗）"""
    return {
        "Sharpe Ratio": 2.5,
        "Max Drawdown [%]": -8.5,
        "Win Rate [%]": 48.0,
        "Profit Factor": 1.8,
        "Total Trades": 350,
        "Calmar Ratio": 5.2,
    }


@pytest.fixture
def suspicious_stats() -> dict:
    """可疑策略的統計（應觸發多個紅旗）"""
    return {
        "Sharpe Ratio": 6.0,
        "Max Drawdown [%]": -1.5,
        "Win Rate [%]": 82.0,
        "Profit Factor": 8.3,
        "Total Trades": 15,
        "Calmar Ratio": 25.0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Basic Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestRedFlagBasic:
    """基本正確性測試"""

    def test_healthy_no_flags(self, healthy_stats):
        """正常統計 → 0 紅旗"""
        flags = check_red_flags(healthy_stats)
        assert len(flags) == 0

    def test_suspicious_multiple_flags(self, suspicious_stats):
        """可疑統計 → 多個紅旗"""
        flags = check_red_flags(suspicious_stats)
        assert len(flags) >= 4  # SR>4, MDD<3, WR>70, PF>5, trades<30, calmar>20
        metrics = {f.metric for f in flags}
        assert "Sharpe Ratio" in metrics
        assert "Max Drawdown" in metrics
        assert "Win Rate" in metrics
        assert "Profit Factor" in metrics

    def test_empty_stats(self):
        """空 dict → 0 紅旗（不報錯）"""
        flags = check_red_flags({})
        assert len(flags) == 0

    def test_return_type(self, healthy_stats):
        """返回值類型正確"""
        flags = check_red_flags(healthy_stats)
        assert isinstance(flags, list)
        # 空列表也是 list
        for flag in flags:
            assert isinstance(flag, RedFlag)


# ══════════════════════════════════════════════════════════════════════════════
# Individual Flag Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestIndividualFlags:
    """各個 flag 獨立觸發測試"""

    def test_high_sharpe(self):
        """Sharpe > 4.0 → 🚩"""
        flags = check_red_flags({"Sharpe Ratio": 4.5})
        assert len(flags) == 1
        assert flags[0].metric == "Sharpe Ratio"

    def test_low_mdd(self):
        """MDD < 3% → 🚩"""
        flags = check_red_flags({"Max Drawdown [%]": -2.0})
        assert len(flags) == 1
        assert flags[0].metric == "Max Drawdown"

    def test_high_win_rate(self):
        """Win Rate > 70% → 🚩"""
        flags = check_red_flags({"Win Rate [%]": 75.0})
        assert len(flags) == 1
        assert flags[0].metric == "Win Rate"

    def test_high_profit_factor(self):
        """Profit Factor > 5 → 🚩"""
        flags = check_red_flags({"Profit Factor": 6.0})
        assert len(flags) == 1
        assert flags[0].metric == "Profit Factor"

    def test_low_trades(self):
        """Total Trades < 30 → ⚠️"""
        flags = check_red_flags({"Total Trades": 10})
        assert len(flags) == 1
        assert flags[0].metric == "Total Trades"
        assert flags[0].emoji == "⚠️"

    def test_high_calmar(self):
        """Calmar > 20 → 🚩"""
        flags = check_red_flags({"Calmar Ratio": 22.0})
        assert len(flags) == 1
        assert flags[0].metric == "Calmar Ratio"


# ══════════════════════════════════════════════════════════════════════════════
# Boundary Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestBoundaryValues:
    """邊界值測試"""

    def test_sharpe_exactly_4(self):
        """Sharpe = 4.0 不觸發（> 才觸發）"""
        flags = check_red_flags({"Sharpe Ratio": 4.0})
        assert len(flags) == 0

    def test_mdd_exactly_3(self):
        """MDD = 3.0 不觸發（< 才觸發）"""
        flags = check_red_flags({"Max Drawdown [%]": -3.0})
        assert len(flags) == 0

    def test_win_rate_exactly_70(self):
        """Win Rate = 70 不觸發（> 才觸發）"""
        flags = check_red_flags({"Win Rate [%]": 70.0})
        assert len(flags) == 0

    def test_profit_factor_exactly_5(self):
        """PF = 5.0 不觸發（> 才觸發）"""
        flags = check_red_flags({"Profit Factor": 5.0})
        assert len(flags) == 0

    def test_trades_exactly_30(self):
        """Trades = 30 不觸發（< 才觸發）"""
        flags = check_red_flags({"Total Trades": 30})
        assert len(flags) == 0


# ══════════════════════════════════════════════════════════════════════════════
# Key Name Compatibility
# ══════════════════════════════════════════════════════════════════════════════

class TestKeyNameCompat:
    """替代鍵名相容測試"""

    def test_alternate_sharpe_key(self):
        """支援 'sharpe' 鍵名"""
        flags = check_red_flags({"sharpe": 5.0})
        assert len(flags) == 1
        assert flags[0].metric == "Sharpe Ratio"

    def test_alternate_mdd_key(self):
        """支援 'max_dd_pct' 鍵名"""
        flags = check_red_flags({"max_dd_pct": 1.0})
        assert len(flags) == 1
        assert flags[0].metric == "Max Drawdown"

    def test_alternate_win_rate_key(self):
        """支援 'win_rate' 鍵名"""
        flags = check_red_flags({"win_rate": 80.0})
        assert len(flags) == 1
        assert flags[0].metric == "Win Rate"

    def test_alternate_pf_key(self):
        """支援 'profit_factor' 鍵名"""
        flags = check_red_flags({"profit_factor": 6.0})
        assert len(flags) == 1
        assert flags[0].metric == "Profit Factor"

    def test_alternate_trades_key(self):
        """支援 'total_trades' 鍵名"""
        flags = check_red_flags({"total_trades": 5})
        assert len(flags) == 1
        assert flags[0].metric == "Total Trades"


# ══════════════════════════════════════════════════════════════════════════════
# print_red_flags Smoke Test
# ══════════════════════════════════════════════════════════════════════════════

class TestPrintRedFlags:
    """print_red_flags 不報錯（smoke test）"""

    def test_print_no_flags(self, capsys):
        """空列表 → 印出「無異常」"""
        print_red_flags([])
        captured = capsys.readouterr()
        assert "無異常" in captured.out

    def test_print_with_flags(self, capsys, suspicious_stats):
        """有紅旗 → 印出詳細報告"""
        flags = check_red_flags(suspicious_stats)
        print_red_flags(flags)
        captured = capsys.readouterr()
        assert "Red Flag Check" in captured.out
        assert "🚩" in captured.out
