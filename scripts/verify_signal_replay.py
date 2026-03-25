#!/usr/bin/env python3
"""
信號重播驗證 — 定期比對回測信號與實盤 SQLite 記錄

核心邏輯：
    1. 載入最新 kline 數據（同 live 使用的 parquet）
    2. 用與 live 相同的路徑（策略 + overlay）計算信號
    3. 從 SQLite 讀取同時段的實盤信號記錄
    4. 逐 bar 比對，報告 mismatch

比對原理：
    - Live 信號路徑: generate_signal() → strategy_func(signal_delay=0) → overlay → signal_value
    - Replay 路徑:   strategy_func(signal_delay=0) → overlay → positions[timestamp]
    - 兩者使用相同策略、相同參數、相同 overlay，差異僅在數據來源（parquet vs API live）

Mismatch 分級：
    - DIRECTION_MISMATCH: 方向不同（backtest=long, live=short）→ CRITICAL
    - MAGNITUDE_MISMATCH: 方向同但差距 > 0.3 → WARNING
    - MINOR_DIFF:         差距 0.1~0.3 → INFO
    - MATCH:              差距 < 0.1 → OK

使用方式：
    # 基本用法（比對最近 7 天）
    PYTHONPATH=src python scripts/verify_signal_replay.py -c config/prod_candidate_simplified.yaml

    # 指定天數
    PYTHONPATH=src python scripts/verify_signal_replay.py -c config/prod_candidate_simplified.yaml --days 3

    # 啟用 Telegram 通知（只在有問題時通知）
    PYTHONPATH=src python scripts/verify_signal_replay.py -c config/prod_candidate_simplified.yaml --notify

    # Cron 模式（靜默，只在 WARN/FAIL 時通知）
    PYTHONPATH=src python scripts/verify_signal_replay.py -c config/prod_candidate_simplified.yaml --notify --quiet

    # 輸出 JSON 報告
    PYTHONPATH=src python scripts/verify_signal_replay.py -c config/prod_candidate_simplified.yaml --output reports/signal_replay.json

建議 cron 設定（每天 UTC 01:00）：
    0 1 * * * cd ~/quant-binance-spot && .venv/bin/python scripts/verify_signal_replay.py \
        -c config/prod_candidate_simplified.yaml --notify --quiet >> logs/signal_replay.log 2>&1
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ── project imports ──
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qtrade.config import load_config, AppConfig
from qtrade.data.storage import load_klines
from qtrade.data.quality import clean_data
from qtrade.strategy import get_strategy
from qtrade.strategy.base import StrategyContext
from qtrade.live.trading_db import TradingDatabase

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("signal_replay")
logger.setLevel(logging.INFO)

# ── 常數 ──
DIRECTION_THRESHOLD = 0.05    # |signal| < 此值視為 flat
MAGNITUDE_WARN_THRESHOLD = 0.3
MINOR_DIFF_THRESHOLD = 0.1
TIMESTAMP_TOLERANCE_H = 2     # timestamp 匹配容許的最大小時差


# ══════════════════════════════════════════════════════════════
#  Section 1: 信號重播引擎
# ══════════════════════════════════════════════════════════════

def _classify_direction(signal: float) -> str:
    """將信號值分類為方向"""
    if signal > DIRECTION_THRESHOLD:
        return "long"
    elif signal < -DIRECTION_THRESHOLD:
        return "short"
    return "flat"


def _compare_single(live_val: float, replay_val: float) -> dict:
    """比對單一 bar 的信號"""
    diff = abs(replay_val - live_val)
    live_dir = _classify_direction(live_val)
    replay_dir = _classify_direction(replay_val)

    if live_dir != replay_dir and live_dir != "flat" and replay_dir != "flat":
        match_type = "DIRECTION_MISMATCH"
    elif diff > MAGNITUDE_WARN_THRESHOLD:
        match_type = "MAGNITUDE_MISMATCH"
    elif diff > MINOR_DIFF_THRESHOLD:
        match_type = "MINOR_DIFF"
    else:
        match_type = "MATCH"

    return {
        "live": round(live_val, 4),
        "replay": round(replay_val, 4),
        "diff": round(diff, 4),
        "live_dir": live_dir,
        "replay_dir": replay_dir,
        "match_type": match_type,
    }


def replay_symbol_signals(
    cfg: AppConfig,
    symbol: str,
    db: TradingDatabase,
    days: int,
) -> dict:
    """
    對單一 symbol 重播信號並與 SQLite 比對。

    Returns:
        Per-symbol 比對結果 dict
    """
    # ── 1. 載入 kline 數據 ──
    # 只載入最近需要的 bars（策略 warmup 300 + 比對天數 × 24h），
    # 大幅降低記憶體用量（Oracle Cloud Free Tier 僅 1GB RAM）
    kline_path = cfg.resolve_kline_path(symbol)
    if not kline_path.exists():
        return {"status": "SKIP", "reason": f"kline not found: {kline_path}"}

    df = load_klines(kline_path)
    df = clean_data(df, fill_method="forward", remove_outliers=False, remove_duplicates=True)

    # 裁切到需要的長度：warmup(500) + 比對期間(days * 24h)
    needed_bars = 500 + days * 24
    if len(df) > needed_bars:
        df = df.iloc[-needed_bars:]

    if len(df) < 50:
        return {"status": "SKIP", "reason": f"insufficient data ({len(df)} bars)"}

    # ── 2. 執行策略（模擬 live 路徑: signal_delay=0）──
    strategy_name = cfg.strategy.name
    params = cfg.strategy.get_params(symbol)

    # 注入 _data_dir（與 base_runner._get_strategy_for_symbol 一致）
    params_copy = copy.deepcopy(params)
    params_copy["_data_dir"] = str(cfg.data_dir)

    ctx = StrategyContext(
        symbol=symbol,
        interval=cfg.market.interval,
        market_type=cfg.market_type_str,
        direction=cfg.direction,
        signal_delay=0,  # Live 模式：信號即時執行
    )

    strategy_func = get_strategy(strategy_name)

    # 移除 StrategyContext 消費的 key（與 generate_signal 一致）
    _consumed_keys = {"_derivatives_data", "_auxiliary_data"}
    clean_params = {k: v for k, v in params_copy.items() if k not in _consumed_keys}

    try:
        positions = strategy_func(df, ctx, clean_params)
    except Exception as e:
        return {"status": "ERROR", "reason": f"strategy execution failed: {e}"}

    # ── 3. 套用 Overlay（與 generate_signal 一致）──
    overlay_cfg = getattr(cfg, '_overlay_cfg', None)
    if overlay_cfg and overlay_cfg.get("enabled", False):
        try:
            from qtrade.strategy.overlays.overlay_pipeline import prepare_and_apply_overlay
            overlay_copy = copy.deepcopy(overlay_cfg)
            positions = prepare_and_apply_overlay(
                positions, df, overlay_copy, symbol,
                data_dir=cfg.data_dir,
            )
        except Exception as e:
            logger.warning(f"  {symbol}: overlay 執行失敗（繼續比對 pre-overlay 信號）: {e}")

    # ── 4. 從 SQLite 讀取實盤信號 ──
    live_signals = db.get_signals(symbol=symbol, days=days, limit=10000)
    if not live_signals:
        return {"status": "SKIP", "reason": f"no live signals in last {days} days"}

    # ── 5. 逐 bar 比對 ──
    comparisons = []
    for sig_record in live_signals:
        ts_str = sig_record["timestamp"]
        live_val = sig_record["signal_value"]

        try:
            ts = pd.Timestamp(ts_str)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")

            # 確保 positions index 也是 timezone-aware
            pos_index = positions.index
            if pos_index.tzinfo is None:
                pos_index = pos_index.tz_localize("UTC")
                positions.index = pos_index

            # 尋找最近的 bar（容許 TIMESTAMP_TOLERANCE_H 小時差）
            if ts in positions.index:
                replay_val = float(positions.loc[ts])
            else:
                time_diffs = abs(positions.index - ts)
                nearest_idx = time_diffs.argmin()
                if time_diffs[nearest_idx] <= pd.Timedelta(hours=TIMESTAMP_TOLERANCE_H):
                    replay_val = float(positions.iloc[nearest_idx])
                else:
                    comparisons.append({
                        "timestamp": ts_str,
                        "match_type": "NO_BAR",
                    })
                    continue

            result = _compare_single(live_val, replay_val)
            result["timestamp"] = ts_str
            comparisons.append(result)

        except Exception as e:
            comparisons.append({
                "timestamp": ts_str,
                "match_type": "ERROR",
                "error": str(e),
            })

    # ── 6. 統計 ──
    total = len(comparisons)
    if total == 0:
        return {"status": "SKIP", "reason": "no comparable signals"}

    counts = {
        "MATCH": 0,
        "MINOR_DIFF": 0,
        "DIRECTION_MISMATCH": 0,
        "MAGNITUDE_MISMATCH": 0,
        "NO_BAR": 0,
        "ERROR": 0,
    }
    for c in comparisons:
        mt = c["match_type"]
        counts[mt] = counts.get(mt, 0) + 1

    match_rate = (counts["MATCH"] + counts["MINOR_DIFF"]) / total * 100
    mismatch_count = counts["DIRECTION_MISMATCH"] + counts["MAGNITUDE_MISMATCH"]
    mismatch_rate = mismatch_count / total if total > 0 else 0

    # Verdict
    if counts["DIRECTION_MISMATCH"] >= 3 or mismatch_rate >= 0.15:
        verdict = "FAIL"
    elif counts["DIRECTION_MISMATCH"] > 0 or mismatch_rate >= 0.05:
        verdict = "WARN"
    else:
        verdict = "PASS"

    # 收集 worst mismatches（方便除錯）
    worst = [
        c for c in comparisons
        if c["match_type"] in ("DIRECTION_MISMATCH", "MAGNITUDE_MISMATCH")
    ][:5]

    return {
        "status": verdict,
        "total_signals": total,
        "counts": counts,
        "match_rate_pct": round(match_rate, 1),
        "mismatch_rate_pct": round(mismatch_rate * 100, 1),
        "worst_mismatches": worst,
    }


def replay_all_signals(cfg: AppConfig, days: int, db_path: Path | None = None) -> dict:
    """
    對所有配置中的 symbols 執行信號重播驗證。

    Returns:
        完整報告 dict（含 overall verdict + per-symbol 結果）
    """
    # 定位 SQLite DB
    if db_path is None:
        db_path = cfg.get_report_dir("live") / "trading.db"

    if not db_path.exists():
        return {
            "overall": "ERROR",
            "error": f"SQLite DB not found: {db_path}",
            "symbols": {},
        }

    db = TradingDatabase(db_path)
    results = {}

    logger.info(f"{'=' * 65}")
    logger.info(f"  🔄 信號重播驗證 — 最近 {days} 天")
    logger.info(f"  策略:   {cfg.strategy.name}")
    logger.info(f"  幣種:   {', '.join(cfg.market.symbols)}")
    logger.info(f"  DB:     {db_path}")
    logger.info(f"{'=' * 65}")

    import gc

    for symbol in cfg.market.symbols:
        logger.info(f"\n  📊 {symbol}...")
        result = replay_symbol_signals(cfg, symbol, db, days)
        results[symbol] = result
        gc.collect()  # 釋放記憶體（Oracle Cloud 1GB RAM 限制）

        status = result.get("status", "?")
        if status == "SKIP":
            logger.info(f"     ⏭️  SKIP: {result.get('reason', '?')}")
        elif status == "ERROR":
            logger.warning(f"     ❌ ERROR: {result.get('reason', '?')}")
        else:
            counts = result.get("counts", {})
            logger.info(
                f"     {_verdict_emoji(status)} {status} — "
                f"match={counts.get('MATCH', 0)}, "
                f"minor={counts.get('MINOR_DIFF', 0)}, "
                f"dir_mm={counts.get('DIRECTION_MISMATCH', 0)}, "
                f"mag_mm={counts.get('MAGNITUDE_MISMATCH', 0)}, "
                f"total={result.get('total_signals', 0)}, "
                f"match_rate={result.get('match_rate_pct', 0)}%"
            )

    # Overall verdict
    verdicts = [r["status"] for r in results.values() if r.get("status") not in ("SKIP", "ERROR")]
    if "FAIL" in verdicts:
        overall = "FAIL"
    elif "WARN" in verdicts:
        overall = "WARN"
    elif verdicts:
        overall = "PASS"
    else:
        overall = "NO_DATA"

    report = {
        "overall": overall,
        "days": days,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "strategy": cfg.strategy.name,
        "config": getattr(cfg, '_config_path', 'unknown'),
        "symbols": results,
    }

    logger.info(f"\n{'=' * 65}")
    logger.info(f"  Overall: {_verdict_emoji(overall)} {overall}")
    logger.info(f"{'=' * 65}")

    return report


def _verdict_emoji(verdict: str) -> str:
    return {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌", "NO_DATA": "🔇"}.get(verdict, "❓")


# ══════════════════════════════════════════════════════════════
#  Section 2: 報告輸出
# ══════════════════════════════════════════════════════════════

def print_report(report: dict) -> None:
    """印出人類可讀報告"""
    overall = report.get("overall", "?")
    days = report.get("days", "?")

    # 錯誤情況（如 DB 不存在）
    if "error" in report:
        print(f"\n❌ Error: {report['error']}")
        return

    print(f"\n{'═' * 65}")
    print(f"  🔄 Signal Replay Verification Report")
    print(f"     策略: {report.get('strategy', '?')}  |  期間: 最近 {days} 天")
    print(f"     時間: {report.get('timestamp', '?')}")
    print(f"{'═' * 65}\n")

    for sym, result in report.get("symbols", {}).items():
        status = result.get("status", "?")
        emoji = _verdict_emoji(status)

        if status in ("SKIP", "ERROR"):
            reason = result.get("reason", "?")
            print(f"  {emoji} {sym:10s}  {status}: {reason}")
            continue

        counts = result.get("counts", {})
        total = result.get("total_signals", 0)
        match_rate = result.get("match_rate_pct", 0)

        print(
            f"  {emoji} {sym:10s}  {status:4s}  "
            f"signals={total:3d}  "
            f"match={match_rate:5.1f}%  "
            f"dir_mm={counts.get('DIRECTION_MISMATCH', 0)}  "
            f"mag_mm={counts.get('MAGNITUDE_MISMATCH', 0)}"
        )

        # 印出 worst mismatches（如果有）
        for mm in result.get("worst_mismatches", []):
            ts = mm.get("timestamp", "?")[:19]
            print(
                f"           └─ {ts}  "
                f"live={mm.get('live', '?'):+.3f}  "
                f"replay={mm.get('replay', '?'):+.3f}  "
                f"diff={mm.get('diff', '?'):.3f}  "
                f"({mm.get('match_type', '?')})"
            )

    print(f"\n{'─' * 65}")
    print(f"  Overall Verdict: {_verdict_emoji(overall)} {overall}")
    print(f"{'─' * 65}\n")


def build_telegram_message(report: dict) -> str:
    """組裝 Telegram 通知訊息（HTML 格式）"""
    overall = report["overall"]
    emoji = _verdict_emoji(overall)
    now = report.get("timestamp", "?")[:19]

    lines = [
        f"{emoji} <b>Signal Replay: {overall}</b>",
        f"期間: 最近 {report.get('days', '?')} 天  |  {now}",
        "",
    ]

    for sym, result in report.get("symbols", {}).items():
        status = result.get("status", "?")
        if status in ("SKIP", "ERROR"):
            lines.append(f"  {sym}: {status}")
            continue

        counts = result.get("counts", {})
        match_rate = result.get("match_rate_pct", 0)
        dir_mm = counts.get("DIRECTION_MISMATCH", 0)

        sym_emoji = _verdict_emoji(status)
        line = f"  {sym_emoji} {sym}: {match_rate:.0f}% match"
        if dir_mm > 0:
            line += f"  ⚠️{dir_mm} dir_mm"
        lines.append(line)

    # Worst mismatches（最多 3 條）
    all_worst = []
    for sym, result in report.get("symbols", {}).items():
        for mm in result.get("worst_mismatches", []):
            all_worst.append((sym, mm))
    if all_worst:
        lines.append("")
        lines.append("<b>Worst Mismatches:</b>")
        for sym, mm in all_worst[:3]:
            ts = mm.get("timestamp", "?")[:16]
            lines.append(
                f"  {sym} {ts}: "
                f"live={mm.get('live', '?'):+.3f} "
                f"replay={mm.get('replay', '?'):+.3f}"
            )

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
#  Section 3: Main
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="信號重播驗證 — 比對回測信號與實盤 SQLite 記錄",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 比對最近 7 天
  PYTHONPATH=src python scripts/verify_signal_replay.py -c config/prod_candidate_simplified.yaml

  # 啟用 Telegram 通知
  PYTHONPATH=src python scripts/verify_signal_replay.py -c config/prod_candidate_simplified.yaml --notify

  # 輸出 JSON
  PYTHONPATH=src python scripts/verify_signal_replay.py -c config/prod_candidate_simplified.yaml --output report.json
        """,
    )

    parser.add_argument("-c", "--config", required=True, help="策略配置文件路徑")
    parser.add_argument("--days", type=int, default=7, help="比對天數（default: 7）")
    parser.add_argument("--db", type=str, default=None, help="SQLite DB 路徑（覆寫自動偵測）")
    parser.add_argument("--notify", action="store_true", help="問題時發送 Telegram 通知")
    parser.add_argument("--notify-on-ok", action="store_true", help="PASS 時也發送通知")
    parser.add_argument("--quiet", action="store_true", help="靜默模式（cron 用）")
    parser.add_argument("--output", type=str, default=None, help="JSON 報告輸出路徑")

    args = parser.parse_args()

    # 靜默模式
    if args.quiet:
        logger.setLevel(logging.WARNING)

    # 載入配置
    cfg = load_config(args.config)

    # DB 路徑
    db_path = Path(args.db) if args.db else None

    # 執行重播
    report = replay_all_signals(cfg, days=args.days, db_path=db_path)

    # 報告輸出
    if not args.quiet:
        print_report(report)

    # JSON 輸出
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"📄 JSON 報告已儲存: {output_path}")

    # Telegram 通知
    overall = report.get("overall", "NO_DATA")
    if args.notify:
        should_notify = overall in ("WARN", "FAIL") or args.notify_on_ok

        if should_notify:
            try:
                from qtrade.monitor.notifier import TelegramNotifier
                notifier = TelegramNotifier()
                if notifier.enabled:
                    msg = build_telegram_message(report)
                    notifier.send(msg)
                    if not args.quiet:
                        print("📱 Telegram 通知已發送")
            except Exception as e:
                logger.warning(f"Telegram 通知失敗: {e}")

    # Exit code
    if overall == "FAIL":
        sys.exit(2)
    elif overall == "WARN":
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
