"""
CI Guard: 偵測 resample 後接 reindex(method="ffill") 但缺少 shift(1) 的模式

═══════════════════════════════════════════════════════════════════
根因：
    _resample_ohlcv(df, "4h") 使用 label='left'，HTF bar 標記在 bar 起始。
    直接 .reindex(target_index, method="ffill") 會導致 intra-bar look-ahead：
    例如 4h bar [00:00, 04:00) 的 close 在 03:59 才可用，
    但 label=00:00 → ffill 從 00:00 起就用到 close → 提前 3 小時。

    正確做法：
    result = compute_fn(htf_df)
    result = result.shift(1)   # ← 必須有這一步
    result.reindex(target_index, method="ffill")

    或直接使用 causal_resample_align()。

白名單：
    外部資料（FR、OI、LSR 等）的 reindex(ffill) 是安全的，
    因為資料點在記錄時已經定稿，不涉及 intra-bar 問題。
═══════════════════════════════════════════════════════════════════
"""
import re
import ast
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).parent.parent / "src" / "qtrade"

# ── 白名單：這些檔案中的 reindex(method="ffill") 是安全的（外部資料對齊） ──
# key = 相對於 SRC_ROOT 的路徑, value = 白名單原因
WHITELIST_FILES = {
    # 外部資料模組：FR / OI / LSR / Taker Vol / Liquidation 對齊到 K 線 index
    "data/long_short_ratio.py": "外部 API 資料，時間戳已定稿",
    "data/taker_volume.py": "外部 API 資料，時間戳已定稿",
    "data/liquidation.py": "外部 API 資料，時間戳已定稿",
    "data/open_interest.py": "外部 API 資料，時間戳已定稿",
    "data/onchain.py": "外部 API 資料，時間戳已定稿",
    "data/multi_tf_loader.py": "MultiTFLoader 的 _align_to_primary 使用 max_ffill_bars 限制",
    # overlay 使用外部資料
    "strategy/overlays/lsr_confirmatory_overlay.py": "OI/FR 外部資料 ffill",
    "strategy/overlays/oi_cascade_overlay.py": "OI 外部資料 ffill",
    # funding carry 使用外部 FR 資料
    "strategy/funding_carry_strategy.py": "外部 FR 結算時間戳已定稿",
}

# ── 白名單：特定行的模式（用於混合檔案：同時有安全和不安全的 reindex） ──
# 格式：(file_relative_path, 行內容片段)
WHITELIST_LINES = [
    # tsmom_carry_v2 中已正確使用 shift(1) 的位置
    ("strategy/tsmom_carry_v2_strategy.py", "shift(1)"),
    # nw_envelope_regime 中 left_shift1_ffill 模式已正確實作
    ("strategy/nw_envelope_regime_strategy.py", "shift(1)"),
]

# ── 白名單：完整函數名稱（函數內部已有 shift(1) 保護） ──
WHITELIST_FUNCTIONS = {
    # causal_resample_align 本身的實作：內部已有 result.shift(1)
    ("strategy/filters.py", "causal_resample_align"),
}


def _find_dangerous_reindex_patterns() -> list[tuple[str, int, str]]:
    """
    掃描 src/qtrade/ 中所有 .py 文件，
    找出「resample 後 reindex(method='ffill')」但缺少 shift(1) 的模式。

    回傳 [(file_path, line_number, line_content), ...]
    """
    violations = []

    for py_file in SRC_ROOT.rglob("*.py"):
        rel_path = str(py_file.relative_to(SRC_ROOT))

        # 跳過白名單整檔
        if rel_path in WHITELIST_FILES:
            continue

        # 跳過測試 / __pycache__
        if "__pycache__" in str(py_file):
            continue

        try:
            content = py_file.read_text(encoding="utf-8")
        except Exception:
            continue

        lines = content.split("\n")

        for i, line in enumerate(lines, start=1):
            # 偵測 .reindex(..., method="ffill") 或 .reindex(..., method='ffill')
            if not re.search(r'\.reindex\(.*method\s*=\s*["\']ffill["\']', line):
                continue

            # 檢查是否在白名單行內
            is_whitelisted = False
            for wl_file, wl_fragment in WHITELIST_LINES:
                if rel_path == wl_file and wl_fragment in line:
                    is_whitelisted = True
                    break

            if is_whitelisted:
                continue

            # 檢查是否在白名單函數內
            # 回溯找到最近的 def 行
            for j in range(i - 2, max(0, i - 50) - 1, -1):
                candidate = lines[j].strip()
                if candidate.startswith("def "):
                    fn_name_match = re.match(r"def\s+(\w+)\s*\(", candidate)
                    if fn_name_match:
                        fn_name = fn_name_match.group(1)
                        if (rel_path, fn_name) in WHITELIST_FUNCTIONS:
                            is_whitelisted = True
                    break

            if is_whitelisted:
                continue

            # 檢查上下文：前 10 行內是否有 shift(1) 或 causal_resample_align
            context_start = max(0, i - 11)  # i is 1-based, lines[] is 0-based
            context = "\n".join(lines[context_start:i - 1])

            if "shift(1)" in context or "causal_resample_align" in context:
                continue

            # 同時檢查同函數範圍內是否有 _resample_ohlcv 或 resample(
            # 如果沒有，可能只是普通的 ffill（如 Series.reindex），不是 HTF 對齊
            fn_context_start = max(0, i - 30)
            fn_context = "\n".join(lines[fn_context_start:i])
            if "resample" not in fn_context and "_resample_ohlcv" not in fn_context:
                continue

            violations.append((rel_path, i, line.strip()))

    return violations


class TestResampleShiftGuard:
    """CI 級自動掃描：偵測未加 shift(1) 的 resample+reindex 模式"""

    def test_no_unshifted_resample_reindex(self):
        """
        掃描 src/qtrade/ 中所有 .py 文件，
        確保 resample 後接 reindex(ffill) 的位置都有 shift(1) 保護。

        如果此測試失敗，表示有新的 HTF 對齊沒有加 shift(1)，
        會導致 intra-bar look-ahead bias。

        修復方式：
            1. 使用 causal_resample_align()（推薦）
            2. 或手動在 reindex 前加 .shift(1)
            3. 如果確認是安全場景（外部資料對齊），加入 WHITELIST
        """
        violations = _find_dangerous_reindex_patterns()

        if violations:
            msg_lines = [
                "",
                "=" * 70,
                "⚠️  偵測到 resample + reindex(ffill) 但缺少 shift(1) 的位置：",
                "=" * 70,
            ]
            for fpath, lineno, content in violations:
                msg_lines.append(f"  {fpath}:{lineno}  →  {content}")
            msg_lines.extend([
                "",
                "修復方式：",
                "  1. 使用 causal_resample_align() 取代手動 resample + reindex",
                "  2. 或在 reindex 前加 .shift(1)",
                "  3. 若為安全場景（外部資料 ffill），加入 WHITELIST_FILES 或 WHITELIST_LINES",
                "=" * 70,
            ])
            pytest.fail("\n".join(msg_lines))

    def test_whitelist_files_exist(self):
        """確保白名單中的檔案都存在（避免路徑過時）"""
        for rel_path in WHITELIST_FILES:
            full_path = SRC_ROOT / rel_path
            assert full_path.exists(), (
                f"白名單檔案不存在: {rel_path}\n"
                f"請更新 test_resample_shift_guard.py 的 WHITELIST_FILES"
            )

    def test_causal_resample_align_exists(self):
        """確保 causal_resample_align 函數存在且可 import"""
        from qtrade.strategy.filters import causal_resample_align
        assert callable(causal_resample_align)
