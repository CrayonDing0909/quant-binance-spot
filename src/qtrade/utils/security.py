"""
安全檢查模組

啟動時執行安全性檢查，確保：
1. .env 檔案權限正確（僅 owner 可讀）
2. .env 沒有被 Git 追蹤
3. 敏感環境變數格式正確
"""
from __future__ import annotations
import os
import sys
import subprocess
from pathlib import Path

from .log import get_logger

logger = get_logger("security")

# 敏感的環境變數 key
SENSITIVE_KEYS = [
    "BINANCE_API_KEY",
    "BINANCE_API_SECRET",
    "TELEGRAM_BOT_TOKEN",
]


def check_env_file_permissions(env_path: str | Path = ".env") -> list[str]:
    """
    檢查 .env 檔案權限
    
    Returns:
        問題清單（空 = 沒問題）
    """
    issues = []
    env_path = Path(env_path)
    
    if not env_path.exists():
        return issues  # 沒有 .env 檔案，可能用其他方式管理密鑰
    
    # 檢查檔案權限（Unix-like 系統）
    if hasattr(os, "chmod"):
        mode = env_path.stat().st_mode & 0o777
        if mode != 0o600:
            issues.append(
                f"⚠️  {env_path} 權限太寬鬆 ({oct(mode)})，"
                f"建議執行: chmod 600 {env_path}"
            )
    
    return issues


def check_env_not_in_git(env_path: str | Path = ".env") -> list[str]:
    """
    檢查 .env 是否被 Git 追蹤
    
    Returns:
        問題清單（空 = 沒問題）
    """
    issues = []
    env_path = Path(env_path)
    
    if not env_path.exists():
        return issues
    
    # 檢查是否在 Git repo 中
    git_dir = env_path.parent / ".git"
    if not git_dir.exists():
        # 往上層找
        for parent in env_path.absolute().parents:
            if (parent / ".git").exists():
                git_dir = parent / ".git"
                break
    
    if not git_dir.exists():
        return issues  # 不是 Git repo
    
    try:
        result = subprocess.run(
            ["git", "ls-files", str(env_path)],
            capture_output=True,
            text=True,
            cwd=env_path.parent,
            timeout=5,
        )
        if result.stdout.strip():
            issues.append(
                f"🚨 {env_path} 被 Git 追蹤！\n"
                "   請立即執行:\n"
                f"   1. git rm --cached {env_path}\n"
                "   2. 確認 .gitignore 包含 .env\n"
                "   3. 輪換（更新）所有 API 密鑰"
            )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass  # git 不可用，跳過檢查
    
    return issues


def check_env_variables() -> list[str]:
    """
    檢查敏感環境變數格式
    
    Returns:
        問題清單（空 = 沒問題）
    """
    issues = []
    
    for key in SENSITIVE_KEYS:
        val = os.getenv(key, "")
        if not val:
            continue  # 沒設置，可能用其他方式
        
        # 檢查是否太短（可能是 placeholder）
        if len(val) < 10:
            issues.append(f"⚠️  {key} 長度異常短 ({len(val)} chars)，請確認是否正確設置")
        
        # 檢查是否是 placeholder
        placeholders = ["your_", "xxx", "placeholder", "example", "test", "demo"]
        if any(p in val.lower() for p in placeholders):
            issues.append(f"⚠️  {key} 看起來像是 placeholder，請設置真實值")
    
    return issues


def security_check(
    env_path: str | Path = ".env",
    exit_on_critical: bool = True,
) -> bool:
    """
    執行完整的安全檢查
    
    Args:
        env_path: .env 檔案路徑
        exit_on_critical: 遇到嚴重問題時是否直接退出
        
    Returns:
        True = 全部通過, False = 有問題
    """
    all_issues = []
    
    # 執行所有檢查
    all_issues.extend(check_env_file_permissions(env_path))
    all_issues.extend(check_env_not_in_git(env_path))
    all_issues.extend(check_env_variables())
    
    if not all_issues:
        logger.debug("✅ 安全檢查通過")
        return True
    
    # 輸出問題
    logger.warning("🔒 安全檢查發現以下問題：")
    for issue in all_issues:
        logger.warning(issue)
    
    # 檢查是否有嚴重問題
    has_critical = any("🚨" in issue for issue in all_issues)
    
    if has_critical and exit_on_critical:
        logger.error("❌ 發現嚴重安全問題，程式終止")
        sys.exit(1)
    
    return False


def mask_secret(secret: str, show_chars: int = 4) -> str:
    """
    遮蔽敏感字串，只顯示前後幾個字元
    
    Args:
        secret: 原始字串
        show_chars: 前後各顯示幾個字元
        
    Returns:
        遮蔽後的字串，例如 "abcd****wxyz"
    """
    if len(secret) <= show_chars * 2:
        return "*" * len(secret)
    return secret[:show_chars] + "****" + secret[-show_chars:]
