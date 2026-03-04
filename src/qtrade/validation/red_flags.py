"""
Backtest Red Flag Detection

自動檢查回測結果中可能暗示 look-ahead bias、過擬合或信號泄漏的異常指標。
每個 flag 有明確的閾值和解釋，供人工審查。

Usage:
    from qtrade.validation.red_flags import check_red_flags, print_red_flags

    flags = check_red_flags(stats)
    print_red_flags(flags)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class RedFlag:
    """單一紅旗"""
    emoji: str       # 🚩 或 ⚠️
    metric: str      # 指標名稱
    value: float     # 實際值
    threshold: str   # 閾值描述
    explanation: str  # 可能原因


def check_red_flags(stats: Dict) -> List[RedFlag]:
    """
    檢查回測統計中的紅旗。

    接受 vbt pf.stats() 的原始 dict/Series，或 adjusted_stats dict。
    鍵名使用 vectorbt 標準格式（例如 "Sharpe Ratio", "Max Drawdown [%]"）。

    Args:
        stats: 回測統計字典，支援以下鍵：
            - "Sharpe Ratio" or "sharpe"
            - "Max Drawdown [%]" or "max_dd_pct"
            - "Win Rate [%]" or "win_rate"
            - "Profit Factor" or "profit_factor"
            - "Total Trades" or "total_trades"
            - "Total Return [%]" or "total_return_pct"

    Returns:
        紅旗列表（可能為空 = 無異常）
    """
    flags: List[RedFlag] = []

    # ── Helper: 取值（支援多種 key 名稱）──
    def _get(primary: str, *alternates: str, default: Optional[float] = None) -> Optional[float]:
        v = stats.get(primary)
        if v is not None:
            return float(v)
        for alt in alternates:
            v = stats.get(alt)
            if v is not None:
                return float(v)
        return default

    # ── 1. Sharpe > 4.0 ──
    sharpe = _get("Sharpe Ratio", "sharpe")
    if sharpe is not None and sharpe > 4.0:
        flags.append(RedFlag(
            emoji="🚩",
            metric="Sharpe Ratio",
            value=sharpe,
            threshold="> 4.0",
            explanation="可能存在 look-ahead bias 或過擬合。"
                        "真實多空策略長期 SR > 4 極為罕見。",
        ))

    # ── 2. MDD < 3% ──
    max_dd = _get("Max Drawdown [%]", "max_dd_pct")
    if max_dd is not None and abs(max_dd) < 3.0:
        flags.append(RedFlag(
            emoji="🚩",
            metric="Max Drawdown",
            value=abs(max_dd),
            threshold="< 3%",
            explanation="可能存在 look-ahead bias。"
                        "加密市場波動大，MDD < 3% 極度異常。",
        ))

    # ── 3. Win Rate > 70% ──
    win_rate = _get("Win Rate [%]", "win_rate")
    if win_rate is not None and win_rate > 70.0:
        flags.append(RedFlag(
            emoji="🚩",
            metric="Win Rate",
            value=win_rate,
            threshold="> 70%",
            explanation="可能存在信號泄漏或 look-ahead bias。"
                        "趨勢跟蹤策略典型勝率 35-55%。",
        ))

    # ── 4. Profit Factor > 5.0 ──
    pf = _get("Profit Factor", "profit_factor")
    if pf is not None and pf > 5.0:
        flags.append(RedFlag(
            emoji="🚩",
            metric="Profit Factor",
            value=pf,
            threshold="> 5.0",
            explanation="過於完美，可能存在數據問題或過擬合。"
                        "健康策略 PF 通常在 1.2-3.0。",
        ))

    # ── 5. Total Trades < 30（補充：樣本太小） ──
    trades = _get("Total Trades", "total_trades")
    if trades is not None and trades < 30:
        flags.append(RedFlag(
            emoji="⚠️",
            metric="Total Trades",
            value=trades,
            threshold="< 30",
            explanation="交易次數過少，統計推斷不可靠。"
                        "至少需要 30+ trades 才有意義。",
        ))

    # ── 6. Calmar Ratio > 20（補充：異常高 risk-adj return） ──
    calmar = _get("Calmar Ratio", "calmar")
    if calmar is not None and calmar > 20.0:
        flags.append(RedFlag(
            emoji="🚩",
            metric="Calmar Ratio",
            value=calmar,
            threshold="> 20",
            explanation="Calmar Ratio 異常高，可能存在 look-ahead bias。"
                        "生產級策略 Calmar 通常 < 10。",
        ))

    return flags


def print_red_flags(flags: List[RedFlag]) -> None:
    """印出紅旗報告"""
    if not flags:
        print("\n  ✅ Red Flag Check: 無異常指標")
        return

    print(f"\n  {'='*60}")
    print(f"  🚩 Red Flag Check — 發現 {len(flags)} 個警告")
    print(f"  {'='*60}")

    for flag in flags:
        print(f"  {flag.emoji} {flag.metric} = {flag.value:.2f} ({flag.threshold})")
        print(f"     → {flag.explanation}")

    print()
    print("  💡 紅旗不代表策略一定有問題，但建議仔細審查上述指標。")
    print(f"  {'='*60}")
