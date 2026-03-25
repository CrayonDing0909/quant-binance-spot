#!/usr/bin/env python3
"""
Weekly active-buy candidate scan and export.

Purpose:
    1. Read a curated long-only watchlist from config
    2. Compute the faithful weekly Pine reversal signal on local parquet data
    3. Export a stable contract for a separate notification / paper-trading app repo

This script intentionally does NOT place orders or manage account state.
It only answers:
    - what is the signal?
    - when is it executable?
    - how strong is it?
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from qtrade.data.storage import load_klines
from qtrade.indicators.atr import calculate_atr
from qtrade.indicators.macd import calculate_macd
from qtrade.indicators.rsi import calculate_rsi

CONTRACT_VERSION = 2
SIGNAL_FAMILY = "pine_weekly_histogram_reversal"
TIER_LABELS = {
    "core": "Core Watchlist",
    "satellite": "Satellite Watchlist",
    "experimental": "Experimental Watchlist",
    "avoid": "Avoid",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan weekly active-buy candidates and export a contract for a separate app repo."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config/research_active_buy_scan.yaml",
        help="Path to the active-buy scan config.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory from config.",
    )
    return parser.parse_args()


def _load_scan_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError("scan config must be a YAML mapping")
    return raw


def _weekly_tv(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample("W-MON", label="left", closed="left").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    ).dropna()


def _latest_completed_weekly(wk: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    if wk.empty:
        return wk
    last_label = wk.index[-1]
    last_daily_ts = daily.index[-1]
    if last_daily_ts < last_label + pd.Timedelta(days=6):
        return wk.iloc[:-1]
    return wk


def _safe_float(val: Any) -> float | None:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _iso(ts: pd.Timestamp | None) -> str | None:
    if ts is None:
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.isoformat()


def _normalize_tier(raw_tier: str | None) -> str:
    if not raw_tier:
        return "experimental"
    tier = str(raw_tier).strip().lower()
    if tier in TIER_LABELS:
        return tier
    return "experimental"


def _compute_execution_policy(
    *,
    main_signal: bool,
    research_tag: str,
    watchlist_tier: str,
    execution_cfg: dict[str, Any],
) -> dict[str, Any]:
    if not main_signal:
        return {
            "execution_stage_recommendation": "no_action",
            "notify_eligible": False,
            "paper_trade_eligible": False,
            "broker_auto_eligible": False,
        }

    paper_tags = {str(x) for x in execution_cfg.get("paper_trade_tags", ["suitable"])}
    auto_tags = {str(x) for x in execution_cfg.get("broker_auto_tags", [])}
    auto_tiers = {str(x) for x in execution_cfg.get("broker_auto_tiers", [])}
    broker_auto_enabled = bool(execution_cfg.get("broker_auto_enabled", False))

    notify_eligible = research_tag in {"suitable", "experimental"}
    paper_trade_eligible = research_tag in paper_tags
    broker_auto_eligible = (
        broker_auto_enabled
        and research_tag in auto_tags
        and watchlist_tier in auto_tiers
    )

    if broker_auto_eligible:
        execution_stage = "broker_auto_candidate"
    elif paper_trade_eligible:
        execution_stage = "paper_candidate"
    elif notify_eligible:
        execution_stage = "manual_notify"
    else:
        execution_stage = "blocked"

    return {
        "execution_stage_recommendation": execution_stage,
        "notify_eligible": notify_eligible,
        "paper_trade_eligible": paper_trade_eligible,
        "broker_auto_eligible": broker_auto_eligible,
    }


def _compute_priority_score(
    *,
    main_signal: bool,
    rsi_prev: float | None,
    rsi_threshold: float,
    hist_prev1: float | None,
    hist_now: float | None,
    atr_branch_active: bool,
    weights: dict[str, float],
) -> float:
    if not main_signal or rsi_prev is None or hist_prev1 is None or hist_now is None:
        return 0.0

    oversold_depth = max(0.0, min(1.0, (rsi_threshold - rsi_prev) / max(rsi_threshold, 1e-9)))
    denom = max(abs(hist_prev1), 1e-9)
    hist_reversal = max(0.0, min(1.0, (hist_now - hist_prev1) / denom))
    atr_bonus = 1.0 if atr_branch_active else 0.0

    score = (
        weights["oversold_weight"] * oversold_depth
        + weights["histogram_reversal_weight"] * hist_reversal
        + weights["atr_bonus_weight"] * atr_bonus
    )
    return round(score * 100.0, 2)


def _message_notes(
    *,
    signal_status: str,
    symbol_note: str,
    rsi_prev: float | None,
    hist_turn: bool,
    executable_ts: pd.Timestamp | None,
) -> str:
    if signal_status == "buy":
        return (
            f"Weekly active-buy candidate. RSI[1]={rsi_prev:.2f}, "
            f"histogram_turn={hist_turn}, executable_at={_iso(executable_ts)}. {symbol_note}"
        )
    return symbol_note


def _telegram_digest_line(record: dict[str, Any]) -> str | None:
    if record["signal_status"] != "buy":
        return None
    return (
        f"- `{record['symbol']}` | {record['watchlist_priority_label']} | "
        f"score={record['research_priority_score']:.2f} | "
        f"stage={record['execution_stage_recommendation']} | "
        f"exec={record['executable_timestamp']} | "
        f"RSI[1]={record['rsi_prev']:.2f} | tag={record['research_tag']}"
    )


def _scan_symbol(
    *,
    group_name: str,
    asset_class: str,
    data_source: str,
    data_root: Path,
    symbol_cfg: dict[str, Any],
    signal_cfg: dict[str, Any],
    ranking_cfg: dict[str, float],
    execution_cfg: dict[str, Any],
    scan_run_at: pd.Timestamp,
) -> dict[str, Any]:
    symbol = symbol_cfg["symbol"]
    data_path = data_root / f"{symbol}.parquet"
    research_tag = symbol_cfg.get("research_tag", "experimental")
    watchlist_tier = _normalize_tier(symbol_cfg.get("watchlist_tier"))
    symbol_note = symbol_cfg.get("notes", "")

    record: dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "scan_run_at": _iso(scan_run_at),
        "symbol": symbol,
        "group": group_name,
        "asset_class": asset_class,
        "data_source": data_source,
        "signal_family": signal_cfg.get("family", SIGNAL_FAMILY),
        "signal_status": "no_action",
        "signal_timestamp": None,
        "executable_timestamp": None,
        "latest_data_timestamp": None,
        "latest_completed_week": None,
        "rsi_prev": None,
        "hist_prev2": None,
        "hist_prev1": None,
        "hist_now": None,
        "hist_turn": False,
        "atr_branch_active": False,
        "research_priority_score": 0.0,
        "research_rank": None,
        "priority_score": 0.0,
        "rank": None,
        "research_tag": research_tag,
        "watchlist_tier": watchlist_tier,
        "watchlist_priority_label": TIER_LABELS[watchlist_tier],
        "execution_stage_recommendation": "no_action",
        "notify_eligible": False,
        "paper_trade_eligible": False,
        "broker_auto_eligible": False,
        "message_notes": symbol_note,
        "telegram_digest_line": None,
    }

    if not data_path.exists():
        record["message_notes"] = f"Missing local data at {data_path}"
        return record

    df = load_klines(data_path)
    if df.empty or len(df) < 60:
        record["message_notes"] = f"Insufficient local data at {data_path}"
        return record

    wk_full = _weekly_tv(df)
    wk = _latest_completed_weekly(wk_full, df)
    if len(wk) < 30:
        record["message_notes"] = f"Insufficient completed weekly bars for {symbol}"
        return record

    macd_df = calculate_macd(wk["close"], 12, 26, 9)
    hist = macd_df["histogram"]
    rsi = calculate_rsi(wk["close"], 14)

    atr = calculate_atr(wk, 26)
    ema12 = wk["close"].ewm(span=12, adjust=False).mean()
    ema26 = wk["close"].ewm(span=26, adjust=False).mean()
    atrmacd = ((ema12 - ema26) / atr) * 100.0
    atrsignal = atrmacd.ewm(span=9, adjust=False).mean()
    atrhist = atrmacd - atrsignal

    latest_ts = wk.index[-1]
    rsi_prev = _safe_float(rsi.shift(1).loc[latest_ts])
    hist_prev2 = _safe_float(hist.shift(2).loc[latest_ts])
    hist_prev1 = _safe_float(hist.shift(1).loc[latest_ts])
    hist_now = _safe_float(hist.loc[latest_ts])
    atr_branch_active = bool(
        (rsi.shift(1) < signal_cfg["rsi_threshold"]).loc[latest_ts]
        and (atrhist.shift(1) < atrhist).loc[latest_ts]
    )
    hist_turn = bool(
        (hist.shift(2) > hist.shift(1)).loc[latest_ts]
        and (hist.shift(1) < hist).loc[latest_ts]
    )
    main_signal = bool(
        (rsi.shift(1) < signal_cfg["rsi_threshold"]).loc[latest_ts]
        and hist_turn
    )

    executable_ts = latest_ts + pd.Timedelta(days=7)
    priority_score = _compute_priority_score(
        main_signal=main_signal,
        rsi_prev=rsi_prev,
        rsi_threshold=float(signal_cfg["rsi_threshold"]),
        hist_prev1=hist_prev1,
        hist_now=hist_now,
        atr_branch_active=atr_branch_active,
        weights=ranking_cfg,
    )
    execution_policy = _compute_execution_policy(
        main_signal=main_signal,
        research_tag=research_tag,
        watchlist_tier=watchlist_tier,
        execution_cfg=execution_cfg,
    )

    record.update(
        {
            "signal_status": "buy" if main_signal else "no_action",
            "signal_timestamp": _iso(latest_ts),
            "executable_timestamp": _iso(executable_ts),
            "latest_data_timestamp": _iso(df.index[-1]),
            "latest_completed_week": _iso(latest_ts),
            "rsi_prev": rsi_prev,
            "hist_prev2": hist_prev2,
            "hist_prev1": hist_prev1,
            "hist_now": hist_now,
            "hist_turn": hist_turn,
            "atr_branch_active": atr_branch_active,
            "research_priority_score": priority_score,
            "research_rank": None,
            "priority_score": priority_score,
            "message_notes": _message_notes(
                signal_status="buy" if main_signal else "no_action",
                symbol_note=symbol_note,
                rsi_prev=rsi_prev,
                hist_turn=hist_turn,
                executable_ts=executable_ts,
            ),
        }
    )
    record.update(execution_policy)
    record["telegram_digest_line"] = _telegram_digest_line(record)
    return record


def _assign_ranks(records: list[dict[str, Any]]) -> None:
    buys = sorted(
        [r for r in records if r["signal_status"] == "buy"],
        key=lambda r: r["priority_score"],
        reverse=True,
    )
    for rank, rec in enumerate(buys, start=1):
        rec["research_rank"] = rank
        rec["rank"] = rank


def _tier_sort_key(tier: str, tier_order: list[str]) -> int:
    try:
        return tier_order.index(tier)
    except ValueError:
        return len(tier_order)


def _write_summary(records: list[dict[str, Any]], output_dir: Path, scan_run_at: pd.Timestamp) -> None:
    buys = [r for r in records if r["signal_status"] == "buy"]
    lines = [
        f"# Active Buy Weekly Scan",
        "",
        f"- run_at: `{_iso(scan_run_at)}`",
        f"- signal_family: `{SIGNAL_FAMILY}`",
        f"- buy_candidates: `{len(buys)}`",
        "",
    ]
    if buys:
        lines.extend(
            [
                "| Rank | Symbol | Group | Score | Tag | Executable |",
                "|------|--------|-------|-------|-----|------------|",
            ]
        )
        for rec in sorted(buys, key=lambda r: r["research_priority_score"], reverse=True):
            lines.append(
                f"| {rec['research_rank']} | {rec['symbol']} | {rec['group']} | "
                f"{rec['research_priority_score']:.2f} | {rec['research_tag']} | {rec['executable_timestamp']} |"
            )
    else:
        lines.append("No active-buy candidates on the latest completed weekly bar.")

    (output_dir / "latest_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_digest(
    *,
    records: list[dict[str, Any]],
    output_dir: Path,
    scan_run_at: pd.Timestamp,
    digest_cfg: dict[str, Any],
) -> None:
    title = digest_cfg.get("title", "Active Buy Weekly Digest")
    only_buy = bool(digest_cfg.get("only_buy_candidates", True))
    max_per_group = int(digest_cfg.get("max_per_group", 5))
    tier_order = [str(t) for t in digest_cfg.get("tier_order", ["core", "satellite", "experimental", "avoid"])]
    output_file = str(digest_cfg.get("output_file", "latest_digest.md"))

    selected = records
    if only_buy:
        selected = [r for r in records if r["signal_status"] == "buy"]

    selected = sorted(
        selected,
        key=lambda r: (
            _tier_sort_key(r["watchlist_tier"], tier_order),
            -float(r["research_priority_score"]),
            r["symbol"],
        ),
    )

    lines = [
        f"# {title}",
        "",
        f"- scan_run_at: `{_iso(scan_run_at)}`",
        f"- signal_family: `{SIGNAL_FAMILY}`",
        f"- candidates: `{sum(1 for r in selected if r['signal_status'] == 'buy')}`",
        "",
    ]

    if not selected:
        lines.append("本週沒有符合條件的 active-buy candidate。")
    else:
        for tier in tier_order:
            tier_records = [r for r in selected if r["watchlist_tier"] == tier]
            if not tier_records:
                continue
            lines.append(f"## {TIER_LABELS.get(tier, tier.title())}")
            lines.append("")
            for rec in tier_records[:max_per_group]:
                line = rec["telegram_digest_line"]
                if line is None:
                    line = (
                        f"- `{rec['symbol']}` | {rec['watchlist_priority_label']} | "
                        f"no_action | latest_week={rec['latest_completed_week']}"
                    )
                lines.append(line)
            lines.append("")

    (output_dir / output_file).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_outputs(
    *,
    records: list[dict[str, Any]],
    output_dir: Path,
    formats: list[str],
    scan_run_at: pd.Timestamp,
    output_cfg: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = scan_run_at.strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(records)
    payload = {
        "contract_version": CONTRACT_VERSION,
        "signal_family": SIGNAL_FAMILY,
        "scan_run_at": _iso(scan_run_at),
        "records": records,
    }

    if "json" in formats:
        (run_dir / "scan.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        (output_dir / "latest_scan.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if "csv" in formats:
        df.to_csv(run_dir / "scan.csv", index=False)
        df.to_csv(output_dir / "latest_scan.csv", index=False)

    if "parquet" in formats:
        df.to_parquet(run_dir / "scan.parquet", index=False)
        df.to_parquet(output_dir / "latest_scan.parquet", index=False)

    _write_summary(records, output_dir, scan_run_at)
    _write_digest(
        records=records,
        output_dir=output_dir,
        scan_run_at=scan_run_at,
        digest_cfg=output_cfg.get("digest", {}),
    )


def main() -> None:
    args = _parse_args()
    raw = _load_scan_config(args.config)

    signal_cfg = raw["signal"]
    ranking_cfg = raw["ranking"]
    output_cfg = raw["output"]
    execution_cfg = raw.get("execution", {})
    universe_cfg = raw["universe"]["groups"]

    scan_run_at = pd.Timestamp(datetime.now(timezone.utc))
    records: list[dict[str, Any]] = []

    for group_name, group_cfg in universe_cfg.items():
        data_root = Path(group_cfg["data_root"]).resolve()
        for symbol_cfg in group_cfg["symbols"]:
            records.append(
                _scan_symbol(
                    group_name=group_name,
                    asset_class=group_cfg["asset_class"],
                    data_source=group_cfg["data_source"],
                    data_root=data_root,
                    symbol_cfg=symbol_cfg,
                    signal_cfg=signal_cfg,
                    ranking_cfg=ranking_cfg,
                    execution_cfg=execution_cfg,
                    scan_run_at=scan_run_at,
                )
            )

    include_no_action = bool(output_cfg.get("include_no_action", True))
    if not include_no_action:
        records = [r for r in records if r["signal_status"] == "buy"]

    _assign_ranks(records)

    output_dir = Path(args.output_dir or output_cfg["output_dir"]).resolve()
    formats = [fmt.strip() for fmt in output_cfg.get("formats", ["json", "csv"])]
    _write_outputs(
        records=records,
        output_dir=output_dir,
        formats=formats,
        scan_run_at=scan_run_at,
        output_cfg=output_cfg,
    )

    buy_count = sum(1 for r in records if r["signal_status"] == "buy")
    print(f"✅ Active-buy scan completed: {len(records)} records, {buy_count} buy candidates")
    print(f"📦 Output: {output_dir / 'latest_scan.json'}")
    print(f"📝 Summary: {output_dir / 'latest_summary.md'}")
    print(f"💬 Digest: {output_dir / output_cfg.get('digest', {}).get('output_file', 'latest_digest.md')}")


if __name__ == "__main__":
    main()
