#!/usr/bin/env python3
"""
密鑰設置腳本

將敏感憑證安全地存儲到系統 Keychain/Keyring 中。

使用方法：
    python scripts/setup_secrets.py          # 互動式設置
    python scripts/setup_secrets.py --list   # 列出已設置的密鑰
    python scripts/setup_secrets.py --clear  # 清除所有密鑰

安全優勢：
    - 密鑰存儲在系統加密的 Keychain 中（macOS）或 Secret Service（Linux）
    - 不需要在專案中保留 .env 檔案
    - 密鑰不會出現在檔案系統或 Git 歷史中
"""
from __future__ import annotations
import argparse
import getpass
import sys

try:
    import keyring
except ImportError:
    print("❌ 請先安裝 keyring:")
    print("   pip install keyring")
    print()
    print("   或者使用完整安裝:")
    print("   pip install -e '.[security]'")
    sys.exit(1)

# 服務名稱（與 binance_client.py 中的 KEYRING_SERVICE 一致）
SERVICE = "spot_bot"

# 支援的密鑰列表
SUPPORTED_KEYS = [
    ("BINANCE_API_KEY", "Binance API Key"),
    ("BINANCE_API_SECRET", "Binance API Secret"),
    ("TELEGRAM_BOT_TOKEN", "Telegram Bot Token (可選)"),
    ("TELEGRAM_CHAT_ID", "Telegram Chat ID (可選)"),
]


def mask_value(value: str, show: int = 4) -> str:
    """遮蔽密鑰值，只顯示前後幾個字元"""
    if not value:
        return "(未設置)"
    if len(value) <= show * 2:
        return "*" * len(value)
    return value[:show] + "****" + value[-show:]


def list_secrets():
    """列出已設置的密鑰"""
    print("🔐 已設置的密鑰：")
    print("-" * 50)
    
    for key, description in SUPPORTED_KEYS:
        value = keyring.get_password(SERVICE, key)
        status = "✅" if value else "❌"
        masked = mask_value(value) if value else "(未設置)"
        print(f"  {status} {key}")
        print(f"     {description}")
        print(f"     值: {masked}")
        print()


def setup_secrets():
    """互動式設置密鑰"""
    print("🔐 密鑰設置精靈")
    print("=" * 50)
    print("輸入密鑰值（直接按 Enter 跳過該項）")
    print("密鑰將安全存儲到系統 Keychain 中")
    print()
    
    updated = 0
    
    for key, description in SUPPORTED_KEYS:
        current = keyring.get_password(SERVICE, key)
        current_str = f" (目前: {mask_value(current)})" if current else ""
        
        # 使用 getpass 隱藏輸入
        prompt = f"{description}{current_str}\n  {key}: "
        
        if "Secret" in description or "Token" in description:
            value = getpass.getpass(prompt)
        else:
            value = input(prompt)
        
        if value.strip():
            keyring.set_password(SERVICE, key, value.strip())
            print(f"  ✅ {key} 已設置")
            updated += 1
        elif current:
            print(f"  ⏭️  {key} 保持不變")
        else:
            print(f"  ⏭️  {key} 跳過")
        print()
    
    print("-" * 50)
    if updated:
        print(f"✅ 已更新 {updated} 個密鑰")
    else:
        print("ℹ️  沒有變更")
    
    print()
    print("📌 提示：如果你有 .env 檔案，現在可以安全地刪除它了")
    print("   密鑰已存儲在系統 Keychain 中")


def clear_secrets():
    """清除所有密鑰"""
    print("⚠️  即將清除所有密鑰！")
    confirm = input("確定要清除嗎？輸入 'yes' 確認: ")
    
    if confirm.lower() != "yes":
        print("❌ 已取消")
        return
    
    cleared = 0
    for key, _ in SUPPORTED_KEYS:
        try:
            keyring.delete_password(SERVICE, key)
            print(f"  🗑️  {key} 已清除")
            cleared += 1
        except keyring.errors.PasswordDeleteError:
            pass  # 密鑰不存在
    
    print(f"✅ 已清除 {cleared} 個密鑰")


def verify_secrets():
    """驗證必要的密鑰是否已設置"""
    required = ["BINANCE_API_KEY", "BINANCE_API_SECRET"]
    missing = []
    
    for key in required:
        if not keyring.get_password(SERVICE, key):
            missing.append(key)
    
    if missing:
        print("❌ 缺少必要的密鑰：")
        for key in missing:
            print(f"   - {key}")
        print()
        print("請執行: python scripts/setup_secrets.py")
        return False
    
    print("✅ 所有必要的密鑰已設置")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="管理交易機器人的敏感憑證",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例：
    python scripts/setup_secrets.py          # 互動式設置密鑰
    python scripts/setup_secrets.py --list   # 列出已設置的密鑰
    python scripts/setup_secrets.py --verify # 驗證必要密鑰
    python scripts/setup_secrets.py --clear  # 清除所有密鑰
        """,
    )
    
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="列出已設置的密鑰",
    )
    parser.add_argument(
        "--verify", "-v",
        action="store_true",
        help="驗證必要的密鑰是否已設置",
    )
    parser.add_argument(
        "--clear", "-c",
        action="store_true",
        help="清除所有密鑰",
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_secrets()
    elif args.verify:
        sys.exit(0 if verify_secrets() else 1)
    elif args.clear:
        clear_secrets()
    else:
        setup_secrets()


if __name__ == "__main__":
    main()
