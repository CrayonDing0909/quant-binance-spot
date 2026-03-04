"""
CI Guard: Data Embargo 強制執行

自動掃描所有 research scripts，確保它們使用 embargo 機制。
這防止研究者無意中使用 embargo 區間的數據。

規則：
1. 所有 scripts/research_*.py 必須 import enforce_temporal_embargo 或 load_embargo_config
2. 或者包含明確的 opt-out 註解 "# EMBARGO_EXEMPT: <reason>"
3. embargo.py 模組本身的公共 API 必須完整

白名單（不需要 embargo）：
- 非研究腳本（不以 research_ 開頭）
- archive/ 目錄下的腳本
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
ARCHIVE_DIR = SCRIPTS_DIR / "archive"

# embargo 相關的 import pattern
EMBARGO_IMPORT_PATTERNS = [
    r"from\s+qtrade\.validation\.embargo\s+import",
    r"from\s+qtrade\.validation\s+import.*embargo",
    r"import\s+qtrade\.validation\.embargo",
    r"enforce_temporal_embargo",
    r"load_embargo_config",
    r"get_research_symbols",
]

# 顯式 opt-out 標記
EMBARGO_EXEMPT_PATTERN = re.compile(r"#\s*EMBARGO_EXEMPT\s*:", re.IGNORECASE)


def _get_active_research_scripts() -> list[Path]:
    """找到所有活躍的 research scripts"""
    if not SCRIPTS_DIR.exists():
        return []
    scripts = []
    for p in SCRIPTS_DIR.glob("research_*.py"):
        # 排除 archive
        if ARCHIVE_DIR in p.parents:
            continue
        scripts.append(p)
    return sorted(scripts)


def _has_embargo_import(content: str) -> bool:
    """檢查檔案是否包含 embargo 相關 import"""
    for pattern in EMBARGO_IMPORT_PATTERNS:
        if re.search(pattern, content):
            return True
    return False


def _has_embargo_exempt(content: str) -> bool:
    """檢查檔案是否有顯式 opt-out 標記"""
    return bool(EMBARGO_EXEMPT_PATTERN.search(content))


class TestEmbargoGuard:
    """CI 掃描：research scripts 必須使用 embargo 機制"""

    def test_research_scripts_use_embargo(self):
        """所有 research_*.py 必須 import embargo 或標記 EMBARGO_EXEMPT"""
        scripts = _get_active_research_scripts()
        if not scripts:
            pytest.skip("No active research scripts found")

        violations = []
        for script_path in scripts:
            content = script_path.read_text(encoding="utf-8")
            has_import = _has_embargo_import(content)
            has_exempt = _has_embargo_exempt(content)

            if not has_import and not has_exempt:
                violations.append(script_path.name)

        if violations:
            msg = (
                f"Found {len(violations)} research script(s) without embargo enforcement:\n"
                + "\n".join(f"  - {v}" for v in violations)
                + "\n\nFix: Add one of:\n"
                "  1. from qtrade.validation.embargo import enforce_temporal_embargo\n"
                "  2. # EMBARGO_EXEMPT: <reason why embargo doesn't apply>"
            )
            pytest.fail(msg)

    def test_embargo_module_exists(self):
        """embargo.py 模組必須存在"""
        embargo_path = (
            Path(__file__).parent.parent
            / "src" / "qtrade" / "validation" / "embargo.py"
        )
        assert embargo_path.exists(), f"Missing: {embargo_path}"

    def test_embargo_module_public_api(self):
        """embargo.py 必須提供完整的公共 API"""
        from qtrade.validation.embargo import (
            EmbargoConfig,
            EmbargoHoldoutResult,
            EmbargoHoldoutSummary,
            SymbolEmbargoConfig,
            TemporalEmbargoConfig,
            enforce_temporal_embargo,
            get_embargo_only_symbols,
            get_research_symbols,
            is_symbol_embargoed,
            load_embargo_config,
            print_embargo_status,
            run_embargo_holdout_test,
        )
        # 只要 import 成功就是 PASS
        assert callable(enforce_temporal_embargo)
        assert callable(load_embargo_config)
        assert callable(get_research_symbols)
        assert callable(run_embargo_holdout_test)

    def test_embargo_config_loads_from_validation_yaml(self):
        """embargo config 能從 validation.yaml 正確載入"""
        from qtrade.validation.embargo import load_embargo_config

        config_path = Path(__file__).parent.parent / "config" / "validation.yaml"
        if not config_path.exists():
            pytest.skip("validation.yaml not found")

        embargo = load_embargo_config(config_path)

        # 基本結構驗證
        assert hasattr(embargo, "temporal")
        assert hasattr(embargo, "symbol")
        assert hasattr(embargo.temporal, "enabled")
        assert hasattr(embargo.temporal, "embargo_months")
        assert hasattr(embargo.symbol, "enabled")
        assert hasattr(embargo.symbol, "embargo_symbols")
        assert isinstance(embargo.symbol.embargo_symbols, list)

    def test_embargo_cutoff_is_in_past(self):
        """cutoff date 必須在過去"""
        import pandas as pd

        from qtrade.validation.embargo import load_embargo_config

        config_path = Path(__file__).parent.parent / "config" / "validation.yaml"
        if not config_path.exists():
            pytest.skip("validation.yaml not found")

        embargo = load_embargo_config(config_path)
        if not embargo.temporal.enabled:
            pytest.skip("Temporal embargo disabled")

        now = pd.Timestamp.now()
        cutoff = embargo.cutoff_date
        assert cutoff < now, f"Cutoff {cutoff} should be in the past (now={now})"

    def test_enforce_temporal_embargo_truncates(self):
        """enforce_temporal_embargo 必須正確截斷數據"""
        import pandas as pd

        from qtrade.validation.embargo import (
            EmbargoConfig,
            TemporalEmbargoConfig,
            enforce_temporal_embargo,
        )

        # 建立測試 DataFrame：過去 6 個月的小時數據
        dates = pd.date_range("2025-09-01", "2026-03-01", freq="1h")
        df = pd.DataFrame({"close": range(len(dates))}, index=dates)

        # embargo 3 個月 → cutoff ~2025-12-02
        embargo = EmbargoConfig(
            temporal=TemporalEmbargoConfig(enabled=True, embargo_months=3),
        )

        df_truncated = enforce_temporal_embargo(df, embargo)
        cutoff = embargo.cutoff_date

        # 截斷後的數據不應包含 cutoff 之後的
        assert df_truncated.index.max() <= cutoff
        # 截斷後的數據量應小於原始
        assert len(df_truncated) < len(df)

    def test_get_research_symbols_filters(self):
        """get_research_symbols 必須過濾掉 embargo symbols"""
        from qtrade.validation.embargo import (
            EmbargoConfig,
            SymbolEmbargoConfig,
            get_research_symbols,
        )

        embargo = EmbargoConfig(
            symbol=SymbolEmbargoConfig(
                enabled=True,
                embargo_symbols=["AVAXUSDT", "DOTUSDT"],
            ),
        )

        all_symbols = ["BTCUSDT", "ETHUSDT", "AVAXUSDT", "DOTUSDT", "SOLUSDT"]
        result = get_research_symbols(all_symbols, embargo)

        assert "BTCUSDT" in result
        assert "ETHUSDT" in result
        assert "SOLUSDT" in result
        assert "AVAXUSDT" not in result
        assert "DOTUSDT" not in result

    def test_is_symbol_embargoed(self):
        """is_symbol_embargoed 正確判斷"""
        from qtrade.validation.embargo import (
            EmbargoConfig,
            SymbolEmbargoConfig,
            is_symbol_embargoed,
        )

        embargo = EmbargoConfig(
            symbol=SymbolEmbargoConfig(
                enabled=True,
                embargo_symbols=["AVAXUSDT"],
            ),
        )

        assert is_symbol_embargoed("AVAXUSDT", embargo) is True
        assert is_symbol_embargoed("BTCUSDT", embargo) is False

    def test_disabled_embargo_returns_all(self):
        """禁用 embargo 時應返回所有數據/symbols"""
        import pandas as pd

        from qtrade.validation.embargo import (
            EmbargoConfig,
            SymbolEmbargoConfig,
            TemporalEmbargoConfig,
            enforce_temporal_embargo,
            get_research_symbols,
        )

        embargo = EmbargoConfig(
            temporal=TemporalEmbargoConfig(enabled=False),
            symbol=SymbolEmbargoConfig(enabled=False),
        )

        # Temporal disabled → 不截斷
        dates = pd.date_range("2025-01-01", "2026-03-01", freq="1h")
        df = pd.DataFrame({"close": range(len(dates))}, index=dates)
        assert len(enforce_temporal_embargo(df, embargo)) == len(df)

        # Symbol disabled → 不過濾
        all_symbols = ["BTCUSDT", "AVAXUSDT"]
        assert get_research_symbols(all_symbols, embargo) == all_symbols
