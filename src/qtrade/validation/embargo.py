"""
Data Embargo — 確保 Alpha Research 有真正的 Out-of-Sample

核心概念：
1. 時間 Embargo：鎖住最近 N 個月的數據，研究只能用 cutoff 之前的資料
2. 幣種 Embargo：保留 1-2 個 symbol 完全不參與研究，用於 cross-symbol OOS

使用方式：
    # 在研究腳本開頭
    from qtrade.validation.embargo import (
        load_embargo_config,
        enforce_temporal_embargo,
        get_research_symbols,
    )

    embargo = load_embargo_config()
    df = enforce_temporal_embargo(df, embargo)
    research_symbols = get_research_symbols(all_symbols, embargo)

    # 在 validation 階段（holdout test）
    from qtrade.validation.embargo import run_embargo_holdout_test
    result = run_embargo_holdout_test(...)

References:
    - Bailey & López de Prado (2014): The importance of true OOS testing
    - Harvey et al. (2016): "…and the Cross-Section of Expected Returns"
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# Default validation config path
_DEFAULT_VALIDATION_CONFIG = Path("config/validation.yaml")


# ══════════════════════════════════════════════════════════════
# Data Classes
# ══════════════════════════════════════════════════════════════

@dataclass
class TemporalEmbargoConfig:
    """時間 embargo 設定"""
    enabled: bool = True
    embargo_months: int = 3  # ~2160 bars at 1h


@dataclass
class SymbolEmbargoConfig:
    """幣種 embargo 設定"""
    enabled: bool = True
    embargo_symbols: List[str] = field(default_factory=lambda: ["AVAXUSDT", "DOTUSDT"])


@dataclass
class EmbargoConfig:
    """完整 embargo 設定"""
    temporal: TemporalEmbargoConfig = field(default_factory=TemporalEmbargoConfig)
    symbol: SymbolEmbargoConfig = field(default_factory=SymbolEmbargoConfig)

    @property
    def cutoff_date(self) -> pd.Timestamp:
        """計算時間 embargo 的截止日期（研究數據的最後日期）"""
        if not self.temporal.enabled:
            return pd.Timestamp.max
        now = pd.Timestamp.now(tz="UTC").tz_localize(None)
        return now - pd.DateOffset(months=self.temporal.embargo_months)

    @property
    def embargo_start(self) -> pd.Timestamp:
        """embargo 區間的起始日期（= cutoff_date + 1 bar）"""
        return self.cutoff_date

    def describe(self) -> str:
        """人類可讀的描述"""
        lines = ["Data Embargo Configuration:"]
        if self.temporal.enabled:
            lines.append(
                f"  Temporal: last {self.temporal.embargo_months} months locked "
                f"(cutoff: {self.cutoff_date.strftime('%Y-%m-%d')})"
            )
        else:
            lines.append("  Temporal: DISABLED")

        if self.symbol.enabled and self.symbol.embargo_symbols:
            lines.append(
                f"  Symbol: {', '.join(self.symbol.embargo_symbols)} reserved for OOS"
            )
        else:
            lines.append("  Symbol: DISABLED")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# Loading
# ══════════════════════════════════════════════════════════════

def load_embargo_config(
    config_path: str | Path | None = None,
) -> EmbargoConfig:
    """
    從 validation.yaml 載入 embargo 設定。

    Args:
        config_path: validation.yaml 的路徑。
                     None 時使用 config/validation.yaml。

    Returns:
        EmbargoConfig
    """
    path = Path(config_path) if config_path else _DEFAULT_VALIDATION_CONFIG

    if not path.exists():
        logger.warning(f"Embargo config not found at {path}, using defaults")
        return EmbargoConfig()

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    embargo_raw = data.get("data_embargo", {})
    if not embargo_raw:
        logger.info("No data_embargo section in config, using defaults")
        return EmbargoConfig()

    # Parse temporal
    temp_raw = embargo_raw.get("temporal", {})
    temporal = TemporalEmbargoConfig(
        enabled=temp_raw.get("enabled", True),
        embargo_months=temp_raw.get("embargo_months", 3),
    )

    # Parse symbol
    sym_raw = embargo_raw.get("symbol", {})
    symbol = SymbolEmbargoConfig(
        enabled=sym_raw.get("enabled", True),
        embargo_symbols=sym_raw.get("embargo_symbols", ["AVAXUSDT", "DOTUSDT"]),
    )

    return EmbargoConfig(temporal=temporal, symbol=symbol)


# ══════════════════════════════════════════════════════════════
# Enforcement Functions（研究腳本使用）
# ══════════════════════════════════════════════════════════════

def enforce_temporal_embargo(
    df: pd.DataFrame,
    embargo: EmbargoConfig | None = None,
    *,
    config_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    對 DataFrame 施加時間 embargo — 截斷 cutoff 之後的數據。

    研究腳本必須在載入數據後立即調用此函數。

    Args:
        df: 帶 DatetimeIndex 的 DataFrame
        embargo: EmbargoConfig（None 時自動載入）
        config_path: 僅在 embargo=None 時使用

    Returns:
        截斷後的 DataFrame（只包含 cutoff 之前的數據）
    """
    if embargo is None:
        embargo = load_embargo_config(config_path)

    if not embargo.temporal.enabled:
        return df

    cutoff = embargo.cutoff_date

    if not isinstance(df.index, pd.DatetimeIndex):
        logger.warning("DataFrame index is not DatetimeIndex, skipping embargo")
        return df

    # 處理 timezone：如果 df.index 有 tz 但 cutoff 沒有（或反之）
    if df.index.tz is not None and cutoff.tz is None:
        cutoff = cutoff.tz_localize(df.index.tz)
    elif df.index.tz is None and cutoff.tz is not None:
        cutoff = cutoff.tz_localize(None)

    before = len(df)
    df_truncated = df[df.index <= cutoff]
    after = len(df_truncated)
    removed = before - after

    if removed > 0:
        logger.info(
            f"  Embargo: removed {removed} bars after {cutoff.strftime('%Y-%m-%d')} "
            f"({before} → {after} bars)"
        )

    return df_truncated


def get_research_symbols(
    all_symbols: List[str],
    embargo: EmbargoConfig | None = None,
    *,
    config_path: str | Path | None = None,
) -> List[str]:
    """
    過濾掉被 embargo 的 symbol，返回可用於研究的幣種列表。

    Args:
        all_symbols: 完整幣種列表
        embargo: EmbargoConfig（None 時自動載入）
        config_path: 僅在 embargo=None 時使用

    Returns:
        過濾後的幣種列表
    """
    if embargo is None:
        embargo = load_embargo_config(config_path)

    if not embargo.symbol.enabled or not embargo.symbol.embargo_symbols:
        return list(all_symbols)

    embargoed = set(embargo.symbol.embargo_symbols)
    research_symbols = [s for s in all_symbols if s not in embargoed]

    removed = [s for s in all_symbols if s in embargoed]
    if removed:
        logger.info(
            f"  Embargo: removed {len(removed)} symbols from research: "
            f"{', '.join(removed)}"
        )

    return research_symbols


def is_symbol_embargoed(
    symbol: str,
    embargo: EmbargoConfig | None = None,
    *,
    config_path: str | Path | None = None,
) -> bool:
    """檢查某個 symbol 是否在 embargo 列表中。"""
    if embargo is None:
        embargo = load_embargo_config(config_path)

    if not embargo.symbol.enabled:
        return False

    return symbol in embargo.symbol.embargo_symbols


def get_embargo_only_symbols(
    all_symbols: List[str],
    embargo: EmbargoConfig | None = None,
    *,
    config_path: str | Path | None = None,
) -> List[str]:
    """返回只在 embargo 中的 symbol（用於 cross-symbol OOS validation）。"""
    if embargo is None:
        embargo = load_embargo_config(config_path)

    if not embargo.symbol.enabled or not embargo.symbol.embargo_symbols:
        return []

    embargoed = set(embargo.symbol.embargo_symbols)
    return [s for s in all_symbols if s in embargoed]


# ══════════════════════════════════════════════════════════════
# Holdout Validation（validate.py 使用）
# ══════════════════════════════════════════════════════════════

@dataclass
class EmbargoHoldoutResult:
    """Embargo holdout test 單一 symbol 的結果"""
    symbol: str
    is_sr: float              # In-sample Sharpe (embargo 之前)
    oos_sr: float             # OOS Sharpe (embargo 區間)
    degradation_pct: float    # (IS_SR - OOS_SR) / |IS_SR| * 100
    is_bars: int
    oos_bars: int
    passed: bool


@dataclass
class EmbargoHoldoutSummary:
    """Embargo holdout test 總結"""
    results: List[EmbargoHoldoutResult]
    avg_is_sr: float
    avg_oos_sr: float
    avg_degradation_pct: float
    passed: bool
    n_oos_positive: int
    n_total: int
    embargo_cutoff: str
    embargo_months: int


def run_embargo_holdout_test(
    symbols: List[str],
    data_paths: Dict[str, Path],
    cfg: dict,
    strategy_name: str,
    embargo: EmbargoConfig | None = None,
    max_degradation: float = 0.5,
    data_dir: Path | None = None,
    config_path: str | Path | None = None,
) -> EmbargoHoldoutSummary | None:
    """
    用 embargo 區間做 holdout OOS 測試。

    邏輯：
    1. 對每個 symbol，用 cutoff 之前的數據跑回測 → IS SR
    2. 用 cutoff 之後（embargo 區間）的數據跑回測 → OOS SR
    3. 比較 IS vs OOS：OOS SR > IS SR × (1 - max_degradation) 且 OOS SR > 0 → PASS

    Args:
        symbols: 要測試的幣種列表
        data_paths: {symbol: Path} 數據路徑
        cfg: backtest config dict
        strategy_name: 策略名稱
        embargo: EmbargoConfig（None 時自動載入）
        max_degradation: 最大容許 SR 衰退比例（0.5 = 50%）
        data_dir: 數據根目錄（用於 funding rate 等）
        config_path: validation config 路徑

    Returns:
        EmbargoHoldoutSummary 或 None（embargo 未啟用或數據不足）
    """
    # Lazy import to avoid circular dependency
    from qtrade.backtest.run_backtest import run_symbol_backtest
    from qtrade.data.storage import load_klines

    if embargo is None:
        embargo = load_embargo_config(config_path)

    if not embargo.temporal.enabled:
        logger.info("Temporal embargo disabled, skipping holdout test")
        return None

    cutoff = embargo.cutoff_date

    print("\n" + "=" * 72)
    print("  🔒 Embargo Holdout OOS Test（數據隔離樣本外測試）")
    print(f"     Embargo: 最近 {embargo.temporal.embargo_months} 個月")
    print(f"     Cutoff: {cutoff.strftime('%Y-%m-%d')}")
    print("     IS = cutoff 前 | OOS = cutoff 後（研究從未使用）")
    print("=" * 72)

    results: List[EmbargoHoldoutResult] = []
    import copy

    for symbol in symbols:
        if symbol not in data_paths:
            continue

        try:
            df_full = load_klines(data_paths[symbol])

            # 處理 timezone
            _cutoff = cutoff
            if df_full.index.tz is not None and _cutoff.tz is None:
                _cutoff = _cutoff.tz_localize(df_full.index.tz)

            df_is = df_full[df_full.index <= _cutoff]
            df_oos = df_full[df_full.index > _cutoff]

            if len(df_is) < 720 or len(df_oos) < 168:  # 需要至少 30 天 IS, 7 天 OOS
                print(f"  ⚠️  {symbol}: 數據不足 (IS={len(df_is)}, OOS={len(df_oos)} bars)")
                continue

            # IS backtest
            cfg_is = copy.deepcopy(cfg)
            is_result = run_symbol_backtest(
                symbol, data_paths[symbol], cfg_is, strategy_name,
                data_dir=data_dir,
            )
            # 從 IS result 取 cutoff 前的 equity
            is_equity = is_result.equity()
            if is_equity.index.tz is not None and _cutoff.tz is None:
                _cutoff_eq = _cutoff.tz_localize(is_equity.index.tz)
            else:
                _cutoff_eq = _cutoff
            is_equity_trim = is_equity[is_equity.index <= _cutoff_eq]
            oos_equity_trim = is_equity[is_equity.index > _cutoff_eq]

            if len(is_equity_trim) < 168 or len(oos_equity_trim) < 48:
                print(f"  ⚠️  {symbol}: equity 分割後數據不足")
                continue

            # 計算 IS/OOS Sharpe
            is_returns = is_equity_trim.pct_change().dropna()
            oos_returns = oos_equity_trim.pct_change().dropna()

            ann = np.sqrt(8760)  # hourly → annual
            is_sr = float(
                is_returns.mean() / is_returns.std() * ann
            ) if is_returns.std() > 0 else 0.0
            oos_sr = float(
                oos_returns.mean() / oos_returns.std() * ann
            ) if oos_returns.std() > 0 else 0.0

            # Degradation
            deg = ((is_sr - oos_sr) / max(abs(is_sr), 0.01)) * 100 if abs(is_sr) > 0.01 else 0.0

            # Pass criteria: OOS SR > IS SR × (1 - max_degradation) AND OOS SR > 0
            threshold_sr = is_sr * (1.0 - max_degradation)
            sym_passed = oos_sr > threshold_sr and oos_sr > 0

            result = EmbargoHoldoutResult(
                symbol=symbol,
                is_sr=round(is_sr, 4),
                oos_sr=round(oos_sr, 4),
                degradation_pct=round(deg, 1),
                is_bars=len(is_returns),
                oos_bars=len(oos_returns),
                passed=sym_passed,
            )
            results.append(result)

            icon = "✅" if sym_passed else "❌"
            print(
                f"  {icon} {symbol:>10s}: IS SR={is_sr:+.2f} → OOS SR={oos_sr:+.2f} "
                f"(Δ={deg:+.1f}%) [{len(is_returns)} IS / {len(oos_returns)} OOS bars]"
            )

        except Exception as e:
            print(f"  ❌ {symbol}: embargo holdout 失敗: {e}")

    if not results:
        print("  ⚠️  No symbols had enough data for embargo holdout test")
        return None

    # Summary
    avg_is = np.mean([r.is_sr for r in results])
    avg_oos = np.mean([r.oos_sr for r in results])
    avg_deg = np.mean([r.degradation_pct for r in results])
    n_oos_pos = sum(1 for r in results if r.oos_sr > 0)
    overall_pass = avg_oos > 0 and avg_oos > avg_is * (1.0 - max_degradation)

    print(f"\n  {'─' * 60}")
    print(f"  Average: IS SR={avg_is:.2f} → OOS SR={avg_oos:.2f} (Δ={avg_deg:+.1f}%)")
    print(f"  OOS positive: {n_oos_pos}/{len(results)}")
    icon = "✅ PASS" if overall_pass else "❌ FAIL"
    print(f"  Verdict: {icon}")

    return EmbargoHoldoutSummary(
        results=results,
        avg_is_sr=round(avg_is, 4),
        avg_oos_sr=round(avg_oos, 4),
        avg_degradation_pct=round(avg_deg, 1),
        passed=overall_pass,
        n_oos_positive=n_oos_pos,
        n_total=len(results),
        embargo_cutoff=cutoff.strftime("%Y-%m-%d"),
        embargo_months=embargo.temporal.embargo_months,
    )


def print_embargo_status(embargo: EmbargoConfig | None = None) -> None:
    """印出目前的 embargo 狀態（供研究腳本開頭使用）"""
    if embargo is None:
        embargo = load_embargo_config()

    print("\n" + "─" * 60)
    print("  🔒 DATA EMBARGO STATUS")
    print("─" * 60)
    if embargo.temporal.enabled:
        print(
            f"  ⏰ Temporal: last {embargo.temporal.embargo_months} months locked "
            f"(cutoff: {embargo.cutoff_date.strftime('%Y-%m-%d')})"
        )
    else:
        print("  ⏰ Temporal: DISABLED ⚠️")

    if embargo.symbol.enabled and embargo.symbol.embargo_symbols:
        print(
            f"  🪙 Symbol:   {', '.join(embargo.symbol.embargo_symbols)} "
            f"reserved for cross-symbol OOS"
        )
    else:
        print("  🪙 Symbol:   DISABLED ⚠️")
    print("─" * 60 + "\n")
