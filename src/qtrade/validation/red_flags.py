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


# ── Default thresholds (overridable via validation.yaml -> red_flags) ──
DEFAULT_RED_FLAG_THRESHOLDS: Dict[str, float] = {
    "max_sharpe": 4.0,
    "min_mdd_pct": 3.0,
    "max_win_rate": 70.0,
    "max_profit_factor": 5.0,
    "min_trades": 30,
    "max_calmar": 20.0,
}


def check_red_flags(
    stats: Dict,
    thresholds: Optional[Dict[str, float]] = None,
) -> List[RedFlag]:
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
        thresholds: 可選紅旗閾值 dict（來自 validation.yaml -> red_flags）。
            缺少的 key 自動回退到 DEFAULT_RED_FLAG_THRESHOLDS。

    Returns:
        紅旗列表（可能為空 = 無異常）
    """
    flags: List[RedFlag] = []
    t = {**DEFAULT_RED_FLAG_THRESHOLDS, **(thresholds or {})}

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

    # ── 1. Sharpe too high ──
    sharpe = _get("Sharpe Ratio", "sharpe")
    if sharpe is not None and sharpe > t["max_sharpe"]:
        flags.append(RedFlag(
            emoji="🚩",
            metric="Sharpe Ratio",
            value=sharpe,
            threshold=f"> {t['max_sharpe']}",
            explanation="可能存在 look-ahead bias 或過擬合。"
                        "真實多空策略長期 SR > 4 極為罕見。",
        ))

    # ── 2. MDD too small ──
    max_dd = _get("Max Drawdown [%]", "max_dd_pct")
    if max_dd is not None and abs(max_dd) < t["min_mdd_pct"]:
        flags.append(RedFlag(
            emoji="🚩",
            metric="Max Drawdown",
            value=abs(max_dd),
            threshold=f"< {t['min_mdd_pct']}%",
            explanation="可能存在 look-ahead bias。"
                        "加密市場波動大，MDD < 3% 極度異常。",
        ))

    # ── 3. Win Rate too high ──
    win_rate = _get("Win Rate [%]", "win_rate")
    if win_rate is not None and win_rate > t["max_win_rate"]:
        flags.append(RedFlag(
            emoji="🚩",
            metric="Win Rate",
            value=win_rate,
            threshold=f"> {t['max_win_rate']}%",
            explanation="可能存在信號泄漏或 look-ahead bias。"
                        "趨勢跟蹤策略典型勝率 35-55%。",
        ))

    # ── 4. Profit Factor too high ──
    pf = _get("Profit Factor", "profit_factor")
    if pf is not None and pf > t["max_profit_factor"]:
        flags.append(RedFlag(
            emoji="🚩",
            metric="Profit Factor",
            value=pf,
            threshold=f"> {t['max_profit_factor']}",
            explanation="過於完美，可能存在數據問題或過擬合。"
                        "健康策略 PF 通常在 1.2-3.0。",
        ))

    # ── 5. Too few trades ──
    trades = _get("Total Trades", "total_trades")
    if trades is not None and trades < t["min_trades"]:
        flags.append(RedFlag(
            emoji="⚠️",
            metric="Total Trades",
            value=trades,
            threshold=f"< {int(t['min_trades'])}",
            explanation="交易次數過少，統計推斷不可靠。"
                        "至少需要 30+ trades 才有意義。",
        ))

    # ── 6. Calmar too high ──
    calmar = _get("Calmar Ratio", "calmar")
    if calmar is not None and calmar > t["max_calmar"]:
        flags.append(RedFlag(
            emoji="🚩",
            metric="Calmar Ratio",
            value=calmar,
            threshold=f"> {t['max_calmar']}",
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
