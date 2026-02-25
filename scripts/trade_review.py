#!/usr/bin/env python3
"""
交易復盤工具 — Trade Review & Post-Mortem

診斷性工具：比較實盤表現與回測預期，發現信號與執行偏差。
這是唯讀工具，不修改策略、配置或倉位。

使用方式:
    # 復盤最近 7 天交易（預設）
    PYTHONPATH=src python scripts/trade_review.py -c config/prod_candidate_meta_blend.yaml

    # 指定天數
    PYTHONPATH=src python scripts/trade_review.py -c config/prod_candidate_meta_blend.yaml --days 14

    # 只看特定幣種
    PYTHONPATH=src python scripts/trade_review.py -c config/prod_candidate_meta_blend.yaml --symbol BTCUSDT

    # 包含回測對比（較慢，需要跑回測）
    PYTHONPATH=src python scripts/trade_review.py -c config/prod_candidate_meta_blend.yaml --with-replay

    # 輸出到 JSON 檔案
    PYTHONPATH=src python scripts/trade_review.py -c config/prod_candidate_meta_blend.yaml --output report.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ── project imports ──
from qtrade.config import load_config
from qtrade.live.trading_db import TradingDatabase

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("trade_review")
logger.setLevel(logging.INFO)


# ══════════════════════════════════════════════════════════════
#  Section 1: Trade Summary
# ══════════════════════════════════════════════════════════════

def trade_summary(
    db: TradingDatabase,
    days: int,
    symbol_filter: str | None = None,
) -> dict:
    """
    交易摘要：勝率、PnL、per-symbol breakdown。

    Returns:
        {
            "total_trades": int,
            "closed_trades": int,
            "win_rate": float,
            "total_pnl": float,
            "total_fees": float,
            "avg_pnl": float,
            "best_trade": float,
            "worst_trade": float,
            "per_symbol": {symbol: {trades, pnl, fees, win_rate}},
        }
    """
    trades = db.get_trades(symbol=symbol_filter, days=days, limit=10000)

    if not trades:
        return {
            "total_trades": 0,
            "closed_trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "total_fees": 0.0,
            "avg_pnl": 0.0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
            "per_symbol": {},
        }

    total_pnl = 0.0
    total_fees = 0.0
    wins = 0
    losses = 0
    best = 0.0
    worst = 0.0
    sym_data: dict[str, dict] = {}

    for t in trades:
        sym = t["symbol"]
        pnl = t["pnl"]
        fee = t["fee"] or 0.0

        if sym not in sym_data:
            sym_data[sym] = {"trades": 0, "pnl": 0.0, "fees": 0.0, "wins": 0, "losses": 0}

        sym_data[sym]["trades"] += 1
        sym_data[sym]["fees"] += fee
        total_fees += fee

        if pnl is not None:
            total_pnl += pnl
            sym_data[sym]["pnl"] += pnl
            if pnl > 0:
                wins += 1
                sym_data[sym]["wins"] += 1
            elif pnl < 0:
                losses += 1
                sym_data[sym]["losses"] += 1
            best = max(best, pnl)
            worst = min(worst, pnl)

    closed = wins + losses
    per_symbol = {}
    for sym, d in sorted(sym_data.items()):
        sym_closed = d["wins"] + d["losses"]
        per_symbol[sym] = {
            "trades": d["trades"],
            "pnl": round(d["pnl"], 4),
            "fees": round(d["fees"], 4),
            "win_rate": round(d["wins"] / sym_closed, 4) if sym_closed > 0 else 0.0,
        }

    return {
        "total_trades": len(trades),
        "closed_trades": closed,
        "win_rate": round(wins / closed, 4) if closed > 0 else 0.0,
        "total_pnl": round(total_pnl, 4),
        "total_fees": round(total_fees, 4),
        "avg_pnl": round(total_pnl / closed, 4) if closed > 0 else 0.0,
        "best_trade": round(best, 4),
        "worst_trade": round(worst, 4),
        "per_symbol": per_symbol,
    }


# ══════════════════════════════════════════════════════════════
#  Section 2: Signal vs Execution Audit
# ══════════════════════════════════════════════════════════════

def signal_execution_audit(
    db: TradingDatabase,
    days: int,
    symbol_filter: str | None = None,
) -> dict:
    """
    比對信號方向 vs 實際成交方向，找出不一致。

    信號 signal_value > 0 → 應該做多（BUY）
    信號 signal_value < 0 → 應該做空（SELL）
    信號 signal_value == 0 → 應該平倉

    Returns:
        {
            "total_signals": int,
            "action_signals": int,  # 非 HOLD 信號
            "matched": int,
            "mismatched": int,
            "mismatch_details": [...],
            "signal_distribution": {action: count},
        }
    """
    signals = db.get_signals(symbol=symbol_filter, days=days, limit=10000)

    if not signals:
        return {
            "total_signals": 0,
            "action_signals": 0,
            "matched": 0,
            "mismatched": 0,
            "mismatch_details": [],
            "signal_distribution": {},
        }

    # Count signal action distribution
    action_dist: dict[str, int] = {}
    action_signals = 0
    for s in signals:
        action = s["action"]
        action_dist[action] = action_dist.get(action, 0) + 1
        if action != "HOLD":
            action_signals += 1

    # Get trades in the same period for cross-referencing
    trades = db.get_trades(symbol=symbol_filter, days=days, limit=10000)

    # Build a simple trade lookup: {(symbol, hour_bucket): [trades]}
    trade_lookup: dict[tuple[str, str], list] = {}
    for t in trades:
        ts = t["timestamp"][:13]  # YYYY-MM-DDTHH
        key = (t["symbol"], ts)
        trade_lookup.setdefault(key, []).append(t)

    matched = 0
    mismatched = 0
    mismatch_details = []

    for s in signals:
        if s["action"] == "HOLD":
            continue

        sig_val = s["signal_value"]
        sym = s["symbol"]
        ts = s["timestamp"][:13]

        # Find corresponding trade within ±1 hour
        found_trade = None
        for offset in [0, 1, -1]:
            try:
                check_ts = (
                    datetime.fromisoformat(s["timestamp"][:19]) + timedelta(hours=offset)
                ).strftime("%Y-%m-%dT%H")
            except Exception:
                continue
            key = (sym, check_ts)
            if key in trade_lookup:
                found_trade = trade_lookup[key][0]
                break

        if found_trade is None:
            # Signal said to act but no trade found — possible mismatch
            # Could be within rebalance band (not a real mismatch)
            continue

        # Check direction consistency
        # DB stores side as: LONG, SHORT, CLOSE_LONG, CLOSE_SHORT, BUY, SELL
        trade_side = found_trade["side"]

        # Normalize: map signal action + trade side to compatible pairs
        is_match = False
        if s["action"] in ("OPEN_LONG", "INCREASE"):
            is_match = trade_side in ("BUY", "LONG")
        elif s["action"] in ("OPEN_SHORT",):
            is_match = trade_side in ("SELL", "SHORT", "CLOSE_LONG")
            # CLOSE_LONG is valid when going from long → short (first close existing long)
        elif s["action"] in ("CLOSE", "REDUCE", "CLOSE_LONG", "CLOSE_SHORT"):
            is_match = True  # Any close/reduce is directionally valid
        elif s["action"] in ("FLIP_LONG", "FLIP_SHORT"):
            is_match = True  # Flip involves close + open

        if is_match:
            matched += 1
        else:
            mismatched += 1
            mismatch_details.append({
                "timestamp": s["timestamp"][:19],
                "symbol": sym,
                "signal_value": sig_val,
                "signal_action": s["action"],
                "trade_side": trade_side,
                "target_pct": s["target_pct"],
                "current_pct": s["current_pct"],
            })

    return {
        "total_signals": len(signals),
        "action_signals": action_signals,
        "matched": matched,
        "mismatched": mismatched,
        "mismatch_details": mismatch_details[:20],  # Cap at 20
        "signal_distribution": action_dist,
    }


# ══════════════════════════════════════════════════════════════
#  Section 3: Market Regime Context
# ══════════════════════════════════════════════════════════════

def market_regime_context(
    cfg,
    days: int,
    symbol_filter: str | None = None,
) -> dict:
    """
    判斷當前市場 regime（趨勢/盤整/高波動），
    幫助解釋近期策略表現。

    Uses ADX (趨勢強度) and ATR (波動率) to classify regime.

    Returns:
        {
            "symbols": {
                symbol: {
                    "regime": "trending" | "ranging" | "volatile",
                    "adx": float,
                    "atr_pct": float,  # ATR as % of price
                    "return_7d": float,
                    "vol_annualized": float,
                }
            },
            "portfolio_regime": str,
            "explanation": str,
        }
    """
    try:
        import ta
    except ImportError:
        return {
            "symbols": {},
            "portfolio_regime": "unknown",
            "explanation": "需要安裝 ta 套件：pip install ta",
        }

    market_type = cfg.market_type_str
    interval = cfg.market.interval
    symbols = [symbol_filter] if symbol_filter else cfg.market.symbols

    sym_regimes = {}
    regime_counts = {"trending": 0, "ranging": 0, "volatile": 0}

    for sym in symbols:
        data_path = cfg.data_dir / "binance" / market_type / interval / f"{sym}.parquet"
        if not data_path.exists():
            continue

        from qtrade.data.storage import load_klines
        df = load_klines(data_path)
        if df is None or len(df) < 200:
            continue

        # Use last N days of data
        lookback_bars = days * 24  # 1h bars
        df = df.tail(lookback_bars + 100)  # Extra for indicator warmup

        # Calculate ADX
        adx_indicator = ta.trend.ADXIndicator(
            high=df["high"], low=df["low"], close=df["close"], window=14
        )
        adx = adx_indicator.adx().iloc[-1]

        # Calculate ATR as % of price
        atr = ta.volatility.AverageTrueRange(
            high=df["high"], low=df["low"], close=df["close"], window=14
        ).average_true_range().iloc[-1]
        atr_pct = (atr / df["close"].iloc[-1]) * 100

        # Recent return
        recent = df.tail(days * 24)
        if len(recent) > 1:
            ret_period = (recent["close"].iloc[-1] / recent["close"].iloc[0] - 1) * 100
        else:
            ret_period = 0.0

        # Annualized volatility
        returns = df["close"].pct_change().dropna().tail(days * 24)
        vol_ann = returns.std() * np.sqrt(8760) * 100  # 1h bars, 8760 hrs/yr

        # Classify regime
        if adx > 30:
            regime = "trending"
        elif atr_pct > 3.0 or vol_ann > 100:
            regime = "volatile"
        else:
            regime = "ranging"

        regime_counts[regime] += 1
        sym_regimes[sym] = {
            "regime": regime,
            "adx": round(float(adx), 1),
            "atr_pct": round(float(atr_pct), 2),
            "return_period": round(float(ret_period), 2),
            "vol_annualized": round(float(vol_ann), 1),
        }

    # Portfolio-level regime
    if regime_counts["trending"] > regime_counts["ranging"]:
        portfolio_regime = "trending"
    elif regime_counts["volatile"] > 0 and regime_counts["volatile"] >= regime_counts["ranging"]:
        portfolio_regime = "volatile"
    else:
        portfolio_regime = "ranging"

    explanations = {
        "trending": (
            "多數幣種處於趨勢狀態（ADX > 30）。"
            "TSMOM 策略在趨勢市場表現最佳。"
            "如果近期虧損，可能是短期逆轉而非策略失效。"
        ),
        "ranging": (
            "多數幣種處於盤整狀態（ADX < 30）。"
            "TSMOM 策略在盤整市場容易被洗盤。"
            "這是策略的自然弱週期，無需調整。"
        ),
        "volatile": (
            "市場處於高波動狀態。"
            "vol_pause overlay 應該會自動減倉保護。"
            "如果近期虧損加劇，檢查 overlay 是否正常觸發。"
        ),
    }

    return {
        "symbols": sym_regimes,
        "portfolio_regime": portfolio_regime,
        "explanation": explanations.get(portfolio_regime, ""),
    }


# ══════════════════════════════════════════════════════════════
#  Section 4: Backtest Replay Comparison
# ══════════════════════════════════════════════════════════════

def backtest_replay_comparison(
    cfg,
    days: int,
    symbol_filter: str | None = None,
) -> dict:
    """
    在同期間重跑回測，比較回測 PnL vs 實盤 PnL。
    如果顯著偏離 → 可能有 live/backtest 不一致問題。

    Returns:
        {
            "symbols": {
                symbol: {
                    "bt_return_pct": float,
                    "bt_sharpe": float,
                    "bt_trades": int,
                    "bt_win_rate": float,
                }
            },
            "portfolio_bt_return": float,
            "data_period": str,
        }
    """
    from qtrade.backtest.run_backtest import run_symbol_backtest

    market_type = cfg.market_type_str
    symbols = [symbol_filter] if symbol_filter else cfg.market.symbols

    # Calculate replay period
    end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    start_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")

    sym_results = {}
    portfolio_returns = []
    weights = cfg.portfolio.allocation or {}
    total_weight = sum(weights.values()) if weights else len(symbols)

    for sym in symbols:
        data_path = (
            cfg.data_dir / "binance" / market_type
            / cfg.market.interval / f"{sym}.parquet"
        )
        if not data_path.exists():
            logger.warning(f"⚠️  {sym}: data not found ({data_path})")
            continue

        bt_cfg = cfg.to_backtest_dict(symbol=sym)
        # Override date range to replay period
        bt_cfg["start"] = start_date
        bt_cfg["end"] = end_date

        try:
            # Check if data covers the replay period
            from qtrade.data.storage import load_klines as _load_klines
            _check_df = _load_klines(data_path)
            if _check_df is not None and len(_check_df) > 0:
                data_end = _check_df.index[-1]
                replay_start_ts = pd.Timestamp(start_date, tz="UTC")
                if data_end < replay_start_ts:
                    sym_results[sym] = {
                        "error": f"本地數據截止 {data_end.strftime('%Y-%m-%d')}，早於重播起始日",
                    }
                    continue

            res = run_symbol_backtest(
                symbol=sym,
                data_path=data_path,
                cfg=bt_cfg,
                strategy_name=cfg.strategy.name,
                data_dir=cfg.data_dir,
            )

            pf = res.pf
            ret_pct = res.total_return_pct()
            sharpe = res.sharpe()

            # Get trade count and win rate from positions
            try:
                pos_records = pf.positions.records_readable
                n_trades = len(pos_records)
                if n_trades > 0:
                    winning = (pos_records["PnL"] > 0).sum()
                    win_rate = winning / n_trades
                else:
                    win_rate = 0.0
            except Exception:
                n_trades = 0
                win_rate = 0.0

            sym_results[sym] = {
                "bt_return_pct": round(ret_pct, 2),
                "bt_sharpe": round(sharpe, 2),
                "bt_trades": n_trades,
                "bt_win_rate": round(win_rate, 4),
            }

            w = weights.get(sym, 1.0 / len(symbols))
            portfolio_returns.append(ret_pct * w / total_weight)

        except Exception as e:
            logger.error(f"❌ Backtest replay failed for {sym}: {e}")
            sym_results[sym] = {"error": str(e)}

    portfolio_bt_return = sum(portfolio_returns) if portfolio_returns else 0.0

    return {
        "symbols": sym_results,
        "portfolio_bt_return": round(portfolio_bt_return, 2),
        "data_period": f"{start_date} → {end_date}",
    }


# ══════════════════════════════════════════════════════════════
#  Report Formatting
# ══════════════════════════════════════════════════════════════

def _print_horizontal_rule():
    print(f"{'═' * 70}")


def print_trade_summary(summary: dict, days: int):
    """Print Section 1: Trade Summary"""
    _print_horizontal_rule()
    print(f"  📊 Section 1: 交易摘要（最近 {days} 天）")
    _print_horizontal_rule()

    if summary["total_trades"] == 0:
        print("  📭 無交易記錄")
        print()
        return

    wr = summary["win_rate"]
    wr_emoji = "✅" if wr >= 0.5 else "⚠️" if wr >= 0.35 else "❌"
    pnl = summary["total_pnl"]
    pnl_emoji = "📈" if pnl > 0 else "📉"

    print(f"  總交易筆數:  {summary['total_trades']}")
    print(f"  已平倉交易:  {summary['closed_trades']}")
    print(f"  {wr_emoji} 勝率:      {wr:.1%}")
    print(f"  {pnl_emoji} 總 PnL:    ${pnl:+,.2f}")
    print(f"  💰 總手續費:  ${summary['total_fees']:,.2f}")
    print(f"  📊 平均 PnL:  ${summary['avg_pnl']:+,.2f}")
    print(f"  🏆 最佳單筆:  ${summary['best_trade']:+,.2f}")
    print(f"  💀 最差單筆:  ${summary['worst_trade']:+,.2f}")

    if summary["per_symbol"]:
        print(f"\n  {'─' * 60}")
        print(f"  {'幣種':<12} {'交易數':>6} {'PnL':>12} {'手續費':>10} {'勝率':>8}")
        print(f"  {'─' * 60}")
        for sym, d in sorted(summary["per_symbol"].items(), key=lambda x: x[1]["pnl"], reverse=True):
            e = "📈" if d["pnl"] > 0 else "📉"
            print(
                f"  {e} {sym:<10} {d['trades']:>6} "
                f"${d['pnl']:>+10,.2f} ${d['fees']:>8,.2f} "
                f"{d['win_rate']:>7.0%}"
            )

    print()


def print_signal_audit(audit: dict, days: int):
    """Print Section 2: Signal vs Execution Audit"""
    _print_horizontal_rule()
    print(f"  🔍 Section 2: 信號 vs 執行審計（最近 {days} 天）")
    _print_horizontal_rule()

    if audit["total_signals"] == 0:
        print("  📭 無信號記錄")
        print()
        return

    print(f"  總信號數:     {audit['total_signals']}")
    print(f"  動作信號:     {audit['action_signals']}（非 HOLD）")
    print(f"  方向匹配:     {audit['matched']}")
    mismatch_emoji = "✅" if audit["mismatched"] == 0 else "⚠️"
    print(f"  {mismatch_emoji} 方向不一致:  {audit['mismatched']}")

    if audit["signal_distribution"]:
        print(f"\n  信號分布:")
        for action, count in sorted(audit["signal_distribution"].items()):
            pct = count / audit["total_signals"] * 100
            print(f"    {action:<15} {count:>5} ({pct:.1f}%)")

    if audit["mismatch_details"]:
        print(f"\n  ⚠️  不一致的交易:")
        print(f"  {'時間':<20} {'幣種':<10} {'信號':>6} {'動作':<12} {'成交方向':<6}")
        for m in audit["mismatch_details"][:10]:
            print(
                f"  {m['timestamp']:<20} {m['symbol']:<10} "
                f"{m['signal_value']:>+5.0%} {m['signal_action']:<12} "
                f"{m['trade_side']:<6}"
            )

    print()


def print_regime_context(regime: dict, days: int):
    """Print Section 3: Market Regime Context"""
    _print_horizontal_rule()
    print(f"  🌍 Section 3: 市場環境分析（最近 {days} 天）")
    _print_horizontal_rule()

    if not regime["symbols"]:
        print("  📭 無市場數據")
        print()
        return

    regime_emoji = {"trending": "📈", "ranging": "↔️", "volatile": "⚡"}
    regime_zh = {"trending": "趨勢", "ranging": "盤整", "volatile": "高波動"}

    portfolio_r = regime["portfolio_regime"]
    print(f"  {regime_emoji.get(portfolio_r, '❓')} 整體市場: {regime_zh.get(portfolio_r, portfolio_r)}")
    print(f"  💡 {regime['explanation']}")

    print(f"\n  {'─' * 60}")
    print(f"  {'幣種':<10} {'環境':<8} {'ADX':>6} {'ATR%':>7} {'報酬':>8} {'年化波動':>8}")
    print(f"  {'─' * 60}")
    for sym, d in sorted(regime["symbols"].items()):
        r_emoji = regime_emoji.get(d["regime"], "?")
        r_name = regime_zh.get(d["regime"], d["regime"])
        print(
            f"  {sym:<10} {r_emoji}{r_name:<6} "
            f"{d['adx']:>5.1f} {d['atr_pct']:>6.2f}% "
            f"{d['return_period']:>+7.1f}% {d['vol_annualized']:>7.1f}%"
        )

    print()


def print_replay_comparison(replay: dict, days: int):
    """Print Section 4: Backtest Replay Comparison"""
    _print_horizontal_rule()
    print(f"  🔄 Section 4: 回測重播比較（最近 {days} 天）")
    _print_horizontal_rule()

    print(f"  📅 重播區間: {replay['data_period']}")
    print(f"  📊 組合回測回報: {replay['portfolio_bt_return']:+.2f}%")

    if replay["symbols"]:
        print(f"\n  {'─' * 60}")
        print(f"  {'幣種':<10} {'回測回報':>10} {'Sharpe':>8} {'交易數':>6} {'勝率':>8}")
        print(f"  {'─' * 60}")
        for sym, d in sorted(replay["symbols"].items()):
            if "error" in d:
                print(f"  {sym:<10} ❌ {d['error'][:40]}")
                continue
            print(
                f"  {sym:<10} {d['bt_return_pct']:>+9.2f}% "
                f"{d['bt_sharpe']:>7.2f} {d['bt_trades']:>6} "
                f"{d['bt_win_rate']:>7.0%}"
            )

    print(f"\n  💡 如果實盤 PnL 與回測差異 > 20%，建議運行一致性檢查:")
    print(f"     PYTHONPATH=src python scripts/validate_live_consistency.py -c <config>")

    print()


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="交易復盤工具 — 診斷信號與執行差異",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-c", "--config", required=True,
        help="配置檔案路徑",
    )
    parser.add_argument(
        "--days", type=int, default=7,
        help="復盤最近 N 天（預設: 7）",
    )
    parser.add_argument(
        "--symbol", type=str, default=None,
        help="只分析特定幣種",
    )
    parser.add_argument(
        "--with-replay", action="store_true",
        help="包含回測重播比較（較慢）",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="輸出 JSON 報告到檔案",
    )
    parser.add_argument(
        "--db", type=str, default=None,
        help="直接指定 DB 檔案路徑（覆蓋 config）",
    )

    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    # Find trading DB
    if args.db:
        db_path = Path(args.db)
    else:
        db_path = cfg.get_report_dir("live") / "trading.db"

    print()
    _print_horizontal_rule()
    print(f"  📋 交易復盤報告")
    print(f"  📅 分析期間: 最近 {args.days} 天")
    print(f"  📁 策略: {cfg.strategy.name}")
    print(f"  🗄️  資料庫: {db_path}")
    _print_horizontal_rule()
    print()

    # ── Section 1: Trade Summary ──
    report = {}
    if db_path.exists():
        db = TradingDatabase(db_path)

        summary = trade_summary(db, args.days, args.symbol)
        report["trade_summary"] = summary
        print_trade_summary(summary, args.days)

        # ── Section 2: Signal vs Execution Audit ──
        audit = signal_execution_audit(db, args.days, args.symbol)
        report["signal_audit"] = audit
        print_signal_audit(audit, args.days)
    else:
        print(f"  ⚠️  資料庫不存在: {db_path}")
        print(f"     資料庫會在首次實盤交易時自動建立。")
        print(f"     跳過 Section 1 & 2。")
        print()

    # ── Section 3: Market Regime Context ──
    regime = market_regime_context(cfg, args.days, args.symbol)
    report["market_regime"] = regime
    print_regime_context(regime, args.days)

    # ── Section 4: Backtest Replay Comparison (optional) ──
    if args.with_replay:
        replay = backtest_replay_comparison(cfg, args.days, args.symbol)
        report["backtest_replay"] = replay
        print_replay_comparison(replay, args.days)
    else:
        print(f"  💡 使用 --with-replay 可加入回測重播比較（較慢）")
        print()

    # ── Verdict ──
    _print_horizontal_rule()
    print(f"  📝 診斷建議")
    _print_horizontal_rule()

    issues = []
    if "trade_summary" in report:
        ts = report["trade_summary"]
        if ts["total_trades"] == 0:
            issues.append("📭 無交易記錄 — 策略可能尚未運行足夠久")
        elif ts["win_rate"] < 0.35:
            issues.append(f"⚠️  勝率偏低 ({ts['win_rate']:.0%}) — 但 TSMOM 策略正常勝率約 40-55%")
        if ts["total_pnl"] < 0 and "market_regime" in report:
            pr = report["market_regime"].get("portfolio_regime", "")
            if pr == "ranging":
                issues.append("💡 市場盤整中虧損是 TSMOM 的正常弱週期，不建議立即調整")
            elif pr == "volatile":
                issues.append("⚡ 高波動期虧損 — 檢查 vol_pause overlay 是否正常觸發")

    if "signal_audit" in report:
        sa = report["signal_audit"]
        if sa["mismatched"] > 0:
            issues.append(f"🔍 發現 {sa['mismatched']} 筆信號/執行不一致 — 建議調查")

    if not issues:
        issues.append("✅ 未發現明顯異常")

    for issue in issues:
        print(f"  {issue}")
    print()

    # General advice
    print(f"  📌 復盤頻率建議:")
    print(f"     • 每週運行一次: --days 7")
    print(f"     • 月度深度審查: --days 30 --with-replay")
    print(f"     • 如連續 2+ 週虧損，運行 alpha decay 監控:")
    print(f"       PYTHONPATH=src python scripts/monitor_alpha_decay.py -c {args.config}")
    print()

    # ── Output JSON ──
    if args.output:
        # Make all values JSON serializable
        report["meta"] = {
            "config": args.config,
            "days": args.days,
            "symbol": args.symbol,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "strategy": cfg.strategy.name,
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        print(f"  ✅ JSON 報告已輸出: {output_path}")
        print()


if __name__ == "__main__":
    main()
