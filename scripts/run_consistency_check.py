#!/usr/bin/env python3
"""
Live/Backtest 一致性驗證腳本

使用方法：
    # 驗證過去 7 天
    python scripts/run_consistency_check.py
    
    # 驗證過去 14 天
    python scripts/run_consistency_check.py --days 14
    
    # 指定配置檔
    python scripts/run_consistency_check.py --config config/rsi_adx_atr.yaml
    
    # 只驗證特定交易對
    python scripts/run_consistency_check.py --symbols BTCUSDT ETHUSDT
    
    # 驗證指定期間
    python scripts/run_consistency_check.py --start 2026-01-01 --end 2026-02-01
    
    # 驗證後發送 Telegram 通知
    python scripts/run_consistency_check.py --notify

建議排程（cron）：
    # 每週日 00:00 執行驗證
    0 0 * * 0 cd /path/to/quant-binance-spot && python scripts/run_consistency_check.py --notify
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

# 將專案加入 path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qtrade.config import load_config
from qtrade.validation import ConsistencyValidator, run_consistency_check
from qtrade.monitor.notifier import TelegramNotifier
from qtrade.utils.log import get_logger

logger = get_logger("consistency_check")


def main():
    parser = argparse.ArgumentParser(
        description="Live/Backtest 一致性驗證",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", "-c",
        default="config/rsi_adx_atr.yaml",
        help="配置檔路徑 (default: config/rsi_adx_atr.yaml)",
    )
    parser.add_argument(
        "--days", "-d",
        type=int,
        default=7,
        help="驗證過去 N 天 (default: 7)",
    )
    parser.add_argument(
        "--symbols", "-s",
        nargs="+",
        help="指定交易對 (default: 使用配置檔中的 symbols)",
    )
    parser.add_argument(
        "--start",
        help="開始日期 YYYY-MM-DD (與 --end 搭配使用)",
    )
    parser.add_argument(
        "--end",
        help="結束日期 YYYY-MM-DD (與 --start 搭配使用)",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.05,
        help="信號差異容忍度 (default: 0.05)",
    )
    parser.add_argument(
        "--output", "-o",
        default="reports/validation",
        help="報告輸出目錄 (default: reports/validation)",
    )
    parser.add_argument(
        "--notify",
        action="store_true",
        help="驗證完成後發送 Telegram 通知",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="顯示詳細輸出",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  🔬 Live/Backtest 一致性驗證")
    print("=" * 60)
    print(f"  配置: {args.config}")
    print(f"  輸出: {args.output}")
    
    # 載入配置
    cfg = load_config(args.config)
    symbols = args.symbols or cfg.market.symbols
    
    print(f"  策略: {cfg.strategy.name}")
    print(f"  交易對: {', '.join(symbols)}")
    print(f"  週期: {cfg.market.interval}")
    
    # 決定驗證期間
    if args.start and args.end:
        start_date = args.start
        end_date = args.end
        print(f"  期間: {start_date} → {end_date}")
    else:
        end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        start_date = (datetime.now(timezone.utc) - timedelta(days=args.days)).strftime("%Y-%m-%d")
        print(f"  期間: 過去 {args.days} 天 ({start_date} → {end_date})")
    
    print(f"  容忍度: {args.threshold}")
    print("=" * 60)
    
    # 執行驗證
    validator = ConsistencyValidator(
        strategy_name=cfg.strategy.name,
        params=cfg.strategy.params,
        interval=cfg.market.interval,
        signal_threshold=args.threshold,
        include_details=args.verbose,
    )
    
    results = {}
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for symbol in symbols:
        print(f"\n📊 驗證 {symbol}...")
        
        # 獲取該 symbol 的參數（含覆寫）
        symbol_params = cfg.strategy.get_params(symbol)
        validator.params = symbol_params
        
        try:
            # 檢查是否有 live state 檔案
            live_state_path = Path(f"reports/live/{cfg.strategy.name}/paper_state.json")
            
            if args.start and args.end:
                report = validator.validate_period(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                )
            else:
                report = validator.validate_recent(
                    symbol=symbol,
                    days=args.days,
                    live_state_path=live_state_path,
                )
            
            results[symbol] = report
            
            # 儲存報告
            report_path = output_path / f"consistency_{symbol}_{datetime.now().strftime('%Y%m%d')}.json"
            report.save(report_path)
            
            # 印出摘要
            print(report.summary())
            
        except Exception as e:
            logger.error(f"❌ {symbol} 驗證失敗: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
    
    # 總結
    print("\n" + "=" * 60)
    print("  📋 總結")
    print("=" * 60)
    
    all_passed = True
    summary_lines = []
    
    for symbol, report in results.items():
        status = "✅" if report.is_consistent else "❌"
        all_passed = all_passed and report.is_consistent
        line = f"  {status} {symbol}: {report.consistency_rate:.1%}"
        print(line)
        summary_lines.append(line)
        
        if not report.is_consistent and report.inconsistencies:
            for inc in report.inconsistencies:
                print(f"      ⚠️  {inc.description}")
    
    print("=" * 60)
    
    # 發送 Telegram 通知
    if args.notify:
        notifier = TelegramNotifier()
        if notifier.enabled:
            status_emoji = "✅" if all_passed else "🚨"
            msg = (
                f"{status_emoji} <b>Live/Backtest 一致性驗證</b>\n\n"
                f"策略: {cfg.strategy.name}\n"
                f"期間: {start_date} → {end_date}\n\n"
            )
            
            for symbol, report in results.items():
                emoji = "✅" if report.is_consistent else "❌"
                msg += f"{emoji} {symbol}: {report.consistency_rate:.1%}\n"
            
            if not all_passed:
                msg += "\n⚠️ 請檢查不一致原因，可能有 look-ahead bias 或實作問題"
            
            notifier.send(msg)
            print("\n📱 Telegram 通知已發送")
    
    # 返回 exit code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
