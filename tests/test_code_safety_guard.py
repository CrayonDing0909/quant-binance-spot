"""
CI Guard: Code Safety Anti-Pattern Scanner

靜態掃描 src/qtrade/ 中所有 .py 文件，
自動偵測 5 類已知的架構反模式（對應 .cursor/rules/code-safety.mdc）。

模式參考：test_resample_shift_guard.py

5 項掃描：
  1. Config dict 淺複製（應用 deepcopy）
  2. params.pop() 修改 caller dict
  3. StrategyContext() 缺少 signal_delay
  4. except Exception + pass 在 critical path
  5. 直接呼叫 vbt.Portfolio.from_orders 而非 safe_portfolio_from_orders
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).parent.parent / "src" / "qtrade"
SCRIPTS_ROOT = Path(__file__).parent.parent / "scripts"


# ══════════════════════════════════════════════════════════════
#  Test 1: Config dict 不應使用淺複製
# ══════════════════════════════════════════════════════════════

# 模式：變數名含 cfg/config/params 的 .copy() 呼叫（應改用 deepcopy）
_CFG_COPY_PATTERN = re.compile(
    r"""
    (?:                     # 變數名含 cfg / config / base_ / overlay_params
        \bcfg\b
      | \bconfig\b
      | \bbase_cfg\b
      | \boverlay_params\b
      | \bself\.base_cfg\b
    )
    \s*(?:=\s*.*)?          # 可能有 assignment
    \.copy\(\)              # .copy() 呼叫
    """,
    re.VERBOSE,
)

# 白名單檔案（不掃描）
_COPY_WHITELIST_FILES = {
    # tests 不在 src/qtrade/ 下，不會被掃到
}


def _find_shallow_copy_on_config() -> list[tuple[str, int, str]]:
    violations = []
    for py_file in SRC_ROOT.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        rel = str(py_file.relative_to(SRC_ROOT))
        if rel in _COPY_WHITELIST_FILES:
            continue

        try:
            lines = py_file.read_text(encoding="utf-8").split("\n")
        except Exception:
            continue

        for i, line in enumerate(lines, start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if _CFG_COPY_PATTERN.search(stripped):
                # 確認同行沒有 deepcopy
                if "deepcopy" not in stripped:
                    violations.append((rel, i, stripped))
    return violations


class TestNoShallowCopyOnConfig:
    """Config dict（含嵌套結構）不應使用 .copy()，應改用 copy.deepcopy()"""

    def test_no_shallow_copy_on_config(self):
        violations = _find_shallow_copy_on_config()
        if violations:
            msg = [
                "",
                "=" * 70,
                "Config dict shallow copy detected (should use copy.deepcopy):",
                "=" * 70,
            ]
            for f, n, c in violations:
                msg.append(f"  {f}:{n}  {c}")
            msg.extend([
                "",
                "See .cursor/rules/code-safety.mdc Section 1",
                "=" * 70,
            ])
            pytest.fail("\n".join(msg))


# ══════════════════════════════════════════════════════════════
#  Test 2: params.pop() 不應出現在 live/ 和 strategy/ 中
# ══════════════════════════════════════════════════════════════

_PARAMS_POP_DIRS = ["live", "strategy"]

# 白名單：某些 .pop() 是對自己建立的 dict 操作（安全）
_PARAMS_POP_WHITELIST = {
    # config.py 中 bt_raw.pop() 是對 dict(raw["backtest"]) 操作（已複製）
    "config.py": "bt_raw is a local copy",
    # overlays 的 apply_overlay_by_mode 中 params.pop 是對 deepcopy 後的 dict 操作
}

# 白名單行片段
_PARAMS_POP_WHITELIST_LINES = [
    "bt_raw.pop(",       # config.py — 操作的是 dict(raw["backtest"])
    "sm_raw.pop(",       # config.py — 操作的是 bt_raw.pop() 結果
    "fr_raw.pop(",       # config.py — 同上
    "raw.pop(",          # config.py — 操作原始 yaml dict
    "bt_raw.pop(",       # config.py
    "ps_raw.pop(",       # config.py
    "kwargs.pop(",       # 函數自己的 kwargs
    "options.pop(",      # 自建 dict
    "d.pop(",            # 自建 dict
]


def _find_params_pop() -> list[tuple[str, int, str]]:
    violations = []
    for subdir in _PARAMS_POP_DIRS:
        target = SRC_ROOT / subdir
        if not target.exists():
            continue
        for py_file in target.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            rel = str(py_file.relative_to(SRC_ROOT))

            try:
                lines = py_file.read_text(encoding="utf-8").split("\n")
            except Exception:
                continue

            for i, line in enumerate(lines, start=1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if "params.pop(" not in stripped:
                    continue
                # 檢查白名單行
                if any(wl in stripped for wl in _PARAMS_POP_WHITELIST_LINES):
                    continue
                violations.append((rel, i, stripped))
    return violations


class TestNoParamsPop:
    """params.pop() 會修改 caller 的 dict，應改用 params.get()"""

    def test_no_params_pop_in_live_strategy(self):
        violations = _find_params_pop()
        if violations:
            msg = [
                "",
                "=" * 70,
                "params.pop() detected in live/ or strategy/ (should use params.get()):",
                "=" * 70,
            ]
            for f, n, c in violations:
                msg.append(f"  {f}:{n}  {c}")
            msg.extend([
                "",
                "See .cursor/rules/code-safety.mdc Section 2",
                "=" * 70,
            ])
            pytest.fail("\n".join(msg))


# ══════════════════════════════════════════════════════════════
#  Test 3: StrategyContext() 必須包含 signal_delay
# ══════════════════════════════════════════════════════════════

# 白名單：允許不含 signal_delay 的檔案
_CTX_WHITELIST_FILES = {
    # StrategyContext 定義本身
    "strategy/base.py": "dataclass definition",
}


def _find_missing_signal_delay() -> list[tuple[str, int, str]]:
    """
    掃描 src/qtrade/ 中 StrategyContext( 呼叫，
    檢查是否包含 signal_delay= 參數。

    使用多行匹配：StrategyContext( 可能跨越多行直到 )。
    """
    violations = []

    for py_file in SRC_ROOT.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        rel = str(py_file.relative_to(SRC_ROOT))
        if rel in _CTX_WHITELIST_FILES:
            continue

        try:
            content = py_file.read_text(encoding="utf-8")
        except Exception:
            continue

        # 找所有 StrategyContext( 的位置
        for m in re.finditer(r"StrategyContext\s*\(", content):
            start = m.start()
            # 找到匹配的 ) — 簡化實作：向後找第一個配對的 )
            depth = 0
            end = start
            for j in range(m.end(), min(m.end() + 500, len(content))):
                ch = content[j]
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    if depth == 0:
                        end = j + 1
                        break
                    depth -= 1

            block = content[start:end]

            # 跳過 docstring / comment 中的提及
            line_start = content.rfind("\n", 0, start) + 1
            line_prefix = content[line_start:start].strip()
            if line_prefix.startswith("#") or line_prefix.startswith('"""') or line_prefix.startswith("'"):
                continue

            # 檢查是否包含 signal_delay
            if "signal_delay" not in block:
                # 計算行號
                lineno = content[:start].count("\n") + 1
                first_line = block.split("\n")[0].strip()
                violations.append((rel, lineno, first_line))

    if SCRIPTS_ROOT.exists():
        for py_file in SCRIPTS_ROOT.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            try:
                content = py_file.read_text(encoding="utf-8")
            except Exception:
                continue

            rel = f"scripts/{py_file.relative_to(SCRIPTS_ROOT)}"

            for m in re.finditer(r"StrategyContext\s*\(", content):
                start = m.start()
                depth = 0
                end = start
                for j in range(m.end(), min(m.end() + 500, len(content))):
                    ch = content[j]
                    if ch == "(":
                        depth += 1
                    elif ch == ")":
                        if depth == 0:
                            end = j + 1
                            break
                        depth -= 1

                block = content[start:end]
                line_start = content.rfind("\n", 0, start) + 1
                line_prefix = content[line_start:start].strip()
                if line_prefix.startswith("#") or line_prefix.startswith('"""') or line_prefix.startswith("'"):
                    continue
                if "signal_delay" not in block:
                    lineno = content[:start].count("\n") + 1
                    first_line = block.split("\n")[0].strip()
                    violations.append((rel, lineno, first_line))

    return violations


class TestStrategyContextHasSignalDelay:
    """StrategyContext() 必須顯式設定 signal_delay，避免默認值造成 look-ahead"""

    def test_all_strategy_context_have_signal_delay(self):
        violations = _find_missing_signal_delay()
        if violations:
            msg = [
                "",
                "=" * 70,
                "StrategyContext() without explicit signal_delay:",
                "=" * 70,
            ]
            for f, n, c in violations:
                msg.append(f"  {f}:{n}  {c}")
            msg.extend([
                "",
                "Every StrategyContext() MUST include signal_delay=0 or signal_delay=1",
                "See .cursor/rules/code-safety.mdc Section 3",
                "=" * 70,
            ])
            pytest.fail("\n".join(msg))


# ══════════════════════════════════════════════════════════════
#  Test 4: Critical path 不應有 silent except
# ══════════════════════════════════════════════════════════════

# 需要掃描的 critical 函數名
_CRITICAL_FUNCTIONS = {
    "_apply_position_sizing",
    "execute_target_position",
    "_process_signal",
    "_check_circuit_breaker",
}


def _find_silent_except_in_critical() -> list[tuple[str, int, str, str]]:
    """
    在 live/ 目錄中找 critical 函數內的 `except Exception:` + `pass`。
    回傳 [(file, lineno, line_content, function_name), ...]
    """
    violations = []
    live_dir = SRC_ROOT / "live"
    if not live_dir.exists():
        return violations

    for py_file in live_dir.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        rel = str(py_file.relative_to(SRC_ROOT))

        try:
            lines = py_file.read_text(encoding="utf-8").split("\n")
        except Exception:
            continue

        current_fn = None
        for i, line in enumerate(lines, start=1):
            stripped = line.strip()

            # 追蹤目前所在的函數
            fn_match = re.match(r"def\s+(\w+)\s*\(", stripped)
            if fn_match:
                current_fn = fn_match.group(1)

            # 只在 critical 函數中掃描
            if current_fn not in _CRITICAL_FUNCTIONS:
                continue

            # 偵測 except Exception 後接 pass（可能間隔 0-2 行）
            if re.match(r"except\s+Exception", stripped):
                # 向前看 3 行：如果 try 區塊只是 get_position_pct 之類的
                # 只讀查詢，silent pass 是可接受的 fallback
                context_lines = [lines[j].strip() for j in range(max(0, i - 4), i - 1)]
                context_block = " ".join(context_lines)
                is_readonly_query = any(
                    q in context_block
                    for q in ["get_position_pct", "get_balance", "get_ticker"]
                )
                if is_readonly_query:
                    continue

                # 向後看 3 行內是否有 pass（且沒有 logger / notify）
                has_logging = False
                has_pass = False
                for j in range(i, min(i + 4, len(lines))):
                    next_line = lines[j].strip()
                    if "logger" in next_line or "log" in next_line or "notify" in next_line:
                        has_logging = True
                        break
                    if next_line == "pass":
                        has_pass = True

                if has_pass and not has_logging:
                    violations.append((rel, i, stripped, current_fn))

    return violations


class TestNoSilentExceptOnCriticalPath:
    """Critical trading path 中 except Exception 必須有 logging，不能 pass"""

    def test_no_silent_except_in_critical_functions(self):
        violations = _find_silent_except_in_critical()
        if violations:
            msg = [
                "",
                "=" * 70,
                "Silent except on critical trading path:",
                "=" * 70,
            ]
            for f, n, c, fn in violations:
                msg.append(f"  {f}:{n} in {fn}()  {c}")
            msg.extend([
                "",
                "Critical functions MUST have logger.error() + notify on exception",
                "See .cursor/rules/code-safety.mdc Section 5",
                "=" * 70,
            ])
            pytest.fail("\n".join(msg))


# ══════════════════════════════════════════════════════════════
#  Test 5: 不應直接呼叫 vbt.Portfolio.from_orders
# ══════════════════════════════════════════════════════════════

# 白名單：允許直接呼叫的位置
_VBT_WHITELIST_FILES = {
    # safe_portfolio_from_orders 的定義本身
    "backtest/run_backtest.py": "safe_portfolio_from_orders definition",
    # buy_and_hold benchmark（不需要策略信號保護）
    "backtest/metrics.py": "buy-and-hold benchmark, always uses price=open_",
}


def _find_direct_vbt_portfolio() -> list[tuple[str, int, str]]:
    violations = []
    for py_file in SRC_ROOT.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        rel = str(py_file.relative_to(SRC_ROOT))
        if rel in _VBT_WHITELIST_FILES:
            continue

        try:
            lines = py_file.read_text(encoding="utf-8").split("\n")
        except Exception:
            continue

        for i, line in enumerate(lines, start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "vbt.Portfolio.from_orders(" in stripped or "Portfolio.from_orders(" in stripped:
                # 排除 import 行
                if "import" in stripped:
                    continue
                violations.append((rel, i, stripped))

    if SCRIPTS_ROOT.exists():
        for py_file in SCRIPTS_ROOT.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            rel = f"scripts/{py_file.relative_to(SCRIPTS_ROOT)}"

            try:
                lines = py_file.read_text(encoding="utf-8").split("\n")
            except Exception:
                continue

            for i, line in enumerate(lines, start=1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if "vbt.Portfolio.from_orders(" in stripped or "Portfolio.from_orders(" in stripped:
                    if "import" in stripped:
                        continue
                    violations.append((rel, i, stripped))
    return violations


class TestNoDirectPortfolioFromOrders:
    """應使用 safe_portfolio_from_orders()，不直接呼叫 vbt.Portfolio.from_orders()"""

    def test_no_direct_vbt_portfolio_from_orders(self):
        violations = _find_direct_vbt_portfolio()
        if violations:
            msg = [
                "",
                "=" * 70,
                "Direct vbt.Portfolio.from_orders() call (use safe_portfolio_from_orders):",
                "=" * 70,
            ]
            for f, n, c in violations:
                msg.append(f"  {f}:{n}  {c}")
            msg.extend([
                "",
                "Use safe_portfolio_from_orders() to enforce price=df['open']",
                "See anti-bias.mdc Section 1",
                "=" * 70,
            ])
            pytest.fail("\n".join(msg))


# ══════════════════════════════════════════════════════════════
#  Meta: 確認 code-safety.mdc 存在
# ══════════════════════════════════════════════════════════════

class TestCodeSafetyRuleExists:
    """確認 code-safety.mdc 規則檔存在"""

    def test_code_safety_mdc_exists(self):
        rule_path = Path(__file__).parent.parent / ".cursor" / "rules" / "code-safety.mdc"
        assert rule_path.exists(), (
            "code-safety.mdc not found!\n"
            "Expected at: .cursor/rules/code-safety.mdc"
        )
