#!/usr/bin/env python3
"""
系統健康檢查腳本

使用方法：
    # 執行檢查並輸出結果
    python scripts/health_check.py
    
    # 只在異常時發送 Telegram 通知
    python scripts/health_check.py --notify
    
    # 總是發送通知（包括正常時）
    python scripts/health_check.py --notify --notify-on-ok
    
    # 指定配置檔
    python scripts/health_check.py --config config/rsi_adx_atr.yaml
    
    # 檢查真實交易模式（檢查 real_state.json）
    python scripts/health_check.py --real
    
    # 輸出 JSON 格式
    python scripts/health_check.py --json

建議 cron 設定（每 30 分鐘檢查一次，異常時通知）：
    # Paper Trading (模擬):
    */30 * * * * cd /path/to/quant-binance-spot && python scripts/health_check.py --notify >> logs/health.log 2>&1
    
    # Real Trading (真實):
    */30 * * * * cd /path/to/quant-binance-spot && python scripts/health_check.py --real --notify >> logs/health.log 2>&1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 將專案加入 path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qtrade.monitor.health import HealthMonitor, run_health_check
from qtrade.config import load_config


def main():
    parser = argparse.ArgumentParser(
        description="系統健康檢查",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", "-c",
        default="config/rsi_adx_atr.yaml",
        help="配置檔路徑 (default: config/rsi_adx_atr.yaml)",
    )
    parser.add_argument(
        "--notify",
        action="store_true",
        help="異常時發送 Telegram 通知",
    )
    parser.add_argument(
        "--notify-on-ok",
        action="store_true",
        help="正常時也發送通知",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="輸出 JSON 格式",
    )
    parser.add_argument(
        "--state-path",
        help="狀態檔路徑（覆寫自動偵測）",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="檢查 real_state.json（真實交易模式）",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="檢查 paper_state.json（模擬交易模式，預設）",
    )
    parser.add_argument(
        "--disk-warning",
        type=float,
        default=0.85,
        help="磁碟使用警告閾值 (default: 0.85)",
    )
    parser.add_argument(
        "--memory-warning",
        type=float,
        default=0.85,
        help="記憶體使用警告閾值 (default: 0.85)",
    )
    parser.add_argument(
        "--stale-minutes",
        type=int,
        default=120,
        help="狀態檔過期分鐘數 (default: 120)",
    )
    
    args = parser.parse_args()
    
    # 決定 state_path
    state_path = None
    if args.state_path:
        state_path = Path(args.state_path)
    else:
        try:
            cfg = load_config(args.config)
            # 根據模式決定 state 檔案名稱
            mode = "real" if args.real else "paper"
            state_path = Path(f"reports/live/{cfg.strategy.name}/{mode}_state.json")
        except Exception:
            pass
    
    # 執行健康檢查
    monitor = HealthMonitor(
        disk_warning_pct=args.disk_warning,
        memory_warning_pct=args.memory_warning,
        state_stale_minutes=args.stale_minutes,
        state_path=state_path,
    )
    
    status = monitor.check_all()
    
    # 輸出結果
    if args.json:
        print(json.dumps(status.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(status.summary())
    
    # 發送通知
    if args.notify:
        from qtrade.monitor.notifier import TelegramNotifier
        notifier = TelegramNotifier()
        
        if notifier.enabled:
            should_notify = not status.ok or args.notify_on_ok
            
            if should_notify:
                notifier.send(status.to_telegram_message())
                if not args.json:
                    print("\n📱 Telegram 通知已發送")
    
    # 根據狀態返回 exit code
    sys.exit(0 if status.ok else 1)


if __name__ == "__main__":
    main()
