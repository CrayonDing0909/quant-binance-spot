#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════════
 Dual-Momentum Weekly Signal — Production Cron Job
═══════════════════════════════════════════════════════════════════════════════════

每週一 08:00 UTC+8 執行一次。

工作流程：
    1. 下載 BTC, ETH, SOL, BNB 近 250 天日線數據
    2. 計算 BTC vs SMA(200) → 判斷 Risk-On / Risk-Off
    3. 計算各幣種 90 天動量 → 選出最強資產
    4. 計算波動率定標 → 建議倉位槓桿
    5. 發送 Telegram 通知

手動觸發：
    PYTHONPATH=src python scripts/cron_dual_momentum.py
    PYTHONPATH=src python scripts/cron_dual_momentum.py --dry-run   # 不發送通知

Cron 設定（UTC+8 每週一 08:00 = UTC 每週一 00:00）：
    0 0 * * 1 cd /path/to/project && .venv/bin/python scripts/cron_dual_momentum.py >> logs/dual_momentum.log 2>&1

環境變數（或寫入 .env）：
    DM_TELEGRAM_BOT_TOKEN=xxxx:yyyyy
    DM_TELEGRAM_CHAT_ID=123456789

Author: Quantitative Research Engineer
Date:   2026-02-19
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ── Ensure project root on PYTHONPATH ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd

# ── Optional: load .env ──
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

from qtrade.monitor.notifier import TelegramNotifier
from qtrade.utils.log import get_logger

logger = get_logger("dual_momentum")

# ═════════════════════════════════════════════════════════════════════════════
# Configuration
# ═════════════════════════════════════════════════════════════════════════════
TICKERS = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "SOL": "SOL-USD",
    "BNB": "BNB-USD",
}
SMA_WINDOW = 200
MOMENTUM_WINDOW = 90
VOL_WINDOW = 30
TARGET_VOL = 0.40
MAX_LEVERAGE = 2.0
TRADING_DAYS = 365  # crypto


# ═════════════════════════════════════════════════════════════════════════════
# Data
# ═════════════════════════════════════════════════════════════════════════════
def fetch_prices() -> pd.DataFrame:
    """Download last ~300 days of daily close prices via yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed. Run: pip install yfinance")
        sys.exit(1)

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=SMA_WINDOW + 60)  # extra buffer

    logger.info(f"📥 Fetching prices: {list(TICKERS.keys())} "
                f"({start.date()} → {end.date()})")

    raw = yf.download(
        list(TICKERS.values()),
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )

    if isinstance(raw.columns, pd.MultiIndex):
        closes = raw["Close"]
    else:
        closes = raw[["Close"]].copy()
        closes.columns = [list(TICKERS.values())[0]]

    rename_map = {v: k for k, v in TICKERS.items()}
    closes = closes.rename(columns=rename_map)
    closes = closes.ffill().dropna(how="all")

    logger.info(f"✅ Loaded {len(closes)} days, latest: {closes.index[-1].date()}")
    return closes


# ═════════════════════════════════════════════════════════════════════════════
# Signal Generator
# ═════════════════════════════════════════════════════════════════════════════
def generate_signal(prices: pd.DataFrame) -> dict:
    """
    Generate this week's Dual-Momentum signal.

    Returns dict with:
        regime, selected_asset, momentum_rank, leverage, btc_price,
        btc_sma, vol_annualized, all_momentums
    """
    latest = prices.index[-1]
    btc_close = prices["BTC"]
    btc_price = float(btc_close.iloc[-1])

    # ── 1. Absolute Momentum: BTC vs SMA(200) ──
    btc_sma = float(btc_close.rolling(SMA_WINDOW).mean().iloc[-1])
    risk_on = btc_price > btc_sma
    regime = "🟢 RISK-ON" if risk_on else "🔴 RISK-OFF"
    regime_short = "Risk-On" if risk_on else "Risk-Off"

    # Distance from SMA
    sma_distance = (btc_price / btc_sma - 1) * 100  # percentage

    # ── 2. Relative Momentum: 90-day returns ──
    daily_returns = prices.pct_change()
    momentum = {}
    for asset in TICKERS.keys():
        if asset in prices.columns:
            ret_90d = prices[asset].pct_change(MOMENTUM_WINDOW).iloc[-1]
            if not np.isnan(ret_90d):
                momentum[asset] = ret_90d

    momentum_rank = sorted(momentum.items(), key=lambda x: x[1], reverse=True)

    # ── 3. Asset Selection + Vol Targeting ──
    if risk_on and momentum_rank:
        selected = momentum_rank[0][0]

        # Volatility of selected asset
        vol_30d = daily_returns[selected].rolling(VOL_WINDOW).std().iloc[-1]
        vol_ann = vol_30d * np.sqrt(TRADING_DAYS)

        raw_leverage = TARGET_VOL / vol_ann if vol_ann > 0 else 1.0
        leverage = min(max(raw_leverage, 0.0), MAX_LEVERAGE)
    else:
        selected = "USDT (Cash)"
        vol_ann = 0.0
        leverage = 0.0

    # ── 4. Additional context ──
    # BTC 50-day SMA for trend context
    btc_sma50 = float(btc_close.rolling(50).mean().iloc[-1])
    btc_sma50_dist = (btc_price / btc_sma50 - 1) * 100

    # Fear/Greed proxy: 30-day BTC return
    btc_30d_ret = float(btc_close.pct_change(30).iloc[-1]) * 100

    return {
        "date": latest,
        "regime": regime,
        "regime_short": regime_short,
        "risk_on": risk_on,
        "btc_price": btc_price,
        "btc_sma200": btc_sma,
        "btc_sma50": btc_sma50,
        "sma_distance_pct": sma_distance,
        "sma50_distance_pct": btc_sma50_dist,
        "btc_30d_return": btc_30d_ret,
        "selected_asset": selected,
        "leverage": leverage,
        "vol_annualized": vol_ann,
        "momentum_rank": momentum_rank,
        "all_momentums": momentum,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Telegram Message Formatter
# ═════════════════════════════════════════════════════════════════════════════
def format_signal_message(sig: dict) -> str:
    """Format signal into a clean Telegram-ready HTML message."""
    now = datetime.now(timezone(timedelta(hours=8)))
    date_str = sig["date"].strftime("%Y-%m-%d")

    # Header
    lines = [
        f"📊 <b>Weekly Dual-Momentum Signal</b>",
        f"📅 {date_str} ({now.strftime('%A')})",
        "",
    ]

    # ── Regime ──
    lines.append(f"━━━ <b>Market Regime</b> ━━━")
    lines.append(f"{sig['regime']}")
    lines.append(f"  BTC: ${sig['btc_price']:,.0f}")
    lines.append(f"  SMA(200): ${sig['btc_sma200']:,.0f} "
                 f"({sig['sma_distance_pct']:+.1f}%)")
    lines.append(f"  SMA(50):  ${sig['btc_sma50']:,.0f} "
                 f"({sig['sma50_distance_pct']:+.1f}%)")
    lines.append(f"  BTC 30d:  {sig['btc_30d_return']:+.1f}%")
    lines.append("")

    # ── Recommendation ──
    lines.append(f"━━━ <b>Recommendation</b> ━━━")

    if sig["risk_on"]:
        lines.append(f"✅ <b>全倉 {sig['selected_asset']}</b>")
        lines.append(f"  建議槓桿: <b>{sig['leverage']:.2f}×</b>")
        lines.append(f"  資產年化波動率: {sig['vol_annualized']:.0%}")
        lines.append(f"  (目標波動率 {TARGET_VOL:.0%} → "
                     f"position = {sig['leverage']:.2f}×)")
    else:
        lines.append(f"🛑 <b>全部現金 (USDT)</b>")
        lines.append(f"  等待 BTC 收回 SMA(200) 上方")
        diff = sig['btc_sma200'] - sig['btc_price']
        pct = diff / sig['btc_price'] * 100
        lines.append(f"  距離 SMA(200): 需上漲 {pct:.1f}% (${diff:,.0f})")
    lines.append("")

    # ── Momentum Ranking ──
    lines.append(f"━━━ <b>90-Day Momentum</b> ━━━")
    for i, (asset, mom) in enumerate(sig["momentum_rank"]):
        medal = ["🥇", "🥈", "🥉", "  4."][i] if i < 4 else f"  {i+1}."
        bar = "█" * max(1, int(abs(mom) * 20))
        sign = "+" if mom >= 0 else ""
        highlight = " ◀" if sig["risk_on"] and i == 0 else ""
        lines.append(f"  {medal} {asset:<4s} {sign}{mom:.1%} {bar}{highlight}")
    lines.append("")

    # ── Action Summary ──
    lines.append(f"━━━ <b>Action</b> ━━━")
    if sig["risk_on"]:
        lines.append(f"📌 持有 {sig['selected_asset']}，"
                     f"倉位 {sig['leverage']*100:.0f}%")
        if sig["leverage"] < 1.0:
            cash_pct = (1 - sig["leverage"]) * 100
            lines.append(f"📌 保留 {cash_pct:.0f}% 現金")
        lines.append(f"📌 下週一 08:00 再檢查")
    else:
        lines.append(f"📌 空倉等待")
        lines.append(f"📌 下週一 08:00 再檢查")

    lines.append("")
    lines.append(f"<i>⚙️ SMA={SMA_WINDOW} | Mom={MOMENTUM_WINDOW}d | "
                 f"VolTarget={TARGET_VOL:.0%} | MaxLev={MAX_LEVERAGE:.0f}×</i>")

    return "\n".join(lines)


def format_console_output(sig: dict) -> str:
    """Format signal for console output."""
    lines = [
        "",
        "═" * 60,
        " DUAL-MOMENTUM WEEKLY SIGNAL",
        "═" * 60,
        "",
        f"  Date:          {sig['date'].strftime('%Y-%m-%d')}",
        f"  Regime:        {sig['regime']}",
        "",
        f"  BTC Price:     ${sig['btc_price']:,.2f}",
        f"  BTC SMA(200):  ${sig['btc_sma200']:,.2f} ({sig['sma_distance_pct']:+.1f}%)",
        f"  BTC SMA(50):   ${sig['btc_sma50']:,.2f} ({sig['sma50_distance_pct']:+.1f}%)",
        f"  BTC 30d Ret:   {sig['btc_30d_return']:+.1f}%",
        "",
    ]

    lines.append("  90-Day Momentum Ranking:")
    for i, (asset, mom) in enumerate(sig["momentum_rank"]):
        arrow = "◀" if sig["risk_on"] and i == 0 else ""
        lines.append(f"    {i+1}. {asset:<5s} {mom:+.2%}  {arrow}")

    lines.append("")
    lines.append("  ─── Recommendation ───")

    if sig["risk_on"]:
        lines.append(f"  ✅ ALLOCATE: {sig['selected_asset']}")
        lines.append(f"     Leverage: {sig['leverage']:.2f}×")
        lines.append(f"     Asset Vol: {sig['vol_annualized']:.0%} annualized")
    else:
        lines.append(f"  🛑 HOLD CASH (USDT)")
        diff = sig['btc_sma200'] - sig['btc_price']
        pct = diff / sig['btc_price'] * 100
        lines.append(f"     BTC needs +{pct:.1f}% to reclaim SMA(200)")

    lines.append("")
    lines.append("═" * 60)
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# Notification
# ═════════════════════════════════════════════════════════════════════════════
def send_telegram(message: str, dry_run: bool = False) -> bool:
    """Send signal via Telegram using the project's notifier."""
    if dry_run:
        logger.info("🔇 Dry-run mode — skipping Telegram notification")
        return True

    notifier = TelegramNotifier(
        bot_token=os.getenv("DM_TELEGRAM_BOT_TOKEN",
                            os.getenv("TELEGRAM_BOT_TOKEN", "")),
        chat_id=os.getenv("DM_TELEGRAM_CHAT_ID",
                          os.getenv("TELEGRAM_CHAT_ID", "")),
        prefix="🔄 [Dual-Momentum]",
    )

    if not notifier.enabled:
        logger.warning("⚠️  Telegram not configured. Set DM_TELEGRAM_BOT_TOKEN "
                       "and DM_TELEGRAM_CHAT_ID in .env")
        return False

    success = notifier.send(message, parse_mode="HTML", add_prefix=True)
    if success:
        logger.info("✅ Telegram notification sent!")
    else:
        logger.error("❌ Failed to send Telegram notification")
    return success


# ═════════════════════════════════════════════════════════════════════════════
# Persistence: save signal history
# ═════════════════════════════════════════════════════════════════════════════
def save_signal_log(sig: dict):
    """Append signal to CSV log for audit trail."""
    log_dir = PROJECT_ROOT / "reports" / "dual_momentum"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "signal_history.csv"

    row = {
        "date": sig["date"].strftime("%Y-%m-%d"),
        "regime": sig["regime_short"],
        "btc_price": f"{sig['btc_price']:.2f}",
        "btc_sma200": f"{sig['btc_sma200']:.2f}",
        "sma_dist_pct": f"{sig['sma_distance_pct']:.2f}",
        "selected_asset": sig["selected_asset"],
        "leverage": f"{sig['leverage']:.3f}",
        "vol_ann": f"{sig['vol_annualized']:.4f}",
    }

    # Add momentum columns
    for asset in TICKERS.keys():
        mom = sig["all_momentums"].get(asset, float("nan"))
        row[f"mom_{asset}"] = f"{mom:.4f}" if not np.isnan(mom) else ""

    df_row = pd.DataFrame([row])

    if log_file.exists():
        df_row.to_csv(log_file, mode="a", header=False, index=False)
    else:
        df_row.to_csv(log_file, index=False)

    logger.info(f"📝 Signal logged to {log_file}")


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dual-Momentum Weekly Signal Generator (Cron Job)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Cron setup (UTC+8 Mon 08:00 = UTC Mon 00:00):
  0 0 * * 1 cd /path/to/project && .venv/bin/python scripts/cron_dual_momentum.py

Environment variables:
  DM_TELEGRAM_BOT_TOKEN   Telegram Bot token
  DM_TELEGRAM_CHAT_ID     Telegram Chat ID
  (falls back to TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID)
        """,
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate signal without sending Telegram")
    parser.add_argument("--no-log", action="store_true",
                        help="Skip saving signal to CSV history")
    parser.add_argument("--json", action="store_true",
                        help="Output signal as JSON (for scripting)")
    return parser.parse_args()


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info(" DUAL-MOMENTUM — WEEKLY SIGNAL GENERATOR")
    logger.info(f" {datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')} (UTC+8)")
    logger.info("=" * 60)

    # ── 1. Fetch data ──
    prices = fetch_prices()

    # ── 2. Generate signal ──
    sig = generate_signal(prices)

    # ── 3. Console output ──
    if args.json:
        import json
        output = {
            "date": sig["date"].strftime("%Y-%m-%d"),
            "regime": sig["regime_short"],
            "btc_price": sig["btc_price"],
            "btc_sma200": sig["btc_sma200"],
            "selected_asset": sig["selected_asset"],
            "leverage": round(sig["leverage"], 3),
            "vol_annualized": round(sig["vol_annualized"], 4),
            "momentum_rank": [(a, round(m, 4)) for a, m in sig["momentum_rank"]],
        }
        print(json.dumps(output, indent=2))
    else:
        print(format_console_output(sig))

    # ── 4. Send Telegram ──
    tg_message = format_signal_message(sig)
    send_telegram(tg_message, dry_run=args.dry_run)

    # ── 5. Save history ──
    if not args.no_log:
        save_signal_log(sig)

    logger.info("✅ Done.")


if __name__ == "__main__":
    main()
