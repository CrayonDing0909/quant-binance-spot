"""
日誌模組

提供統一的 logger 配置，包含：
- SecretFilter: 自動過濾日誌中的敏感資訊（API keys, tokens 等）
"""
from __future__ import annotations
import logging
import re


class SecretFilter(logging.Filter):
    """
    過濾日誌中的敏感資訊
    
    自動偵測並遮蔽：
    - API keys
    - API secrets
    - Tokens
    - 其他長字串憑證
    """
    
    # 正則表達式模式：匹配常見的敏感資訊格式
    PATTERNS = [
        # api_key="xxx" 或 api_key: xxx
        (r'(api[_-]?key[\"\'\s:=]+)[\'\"a-zA-Z0-9_-]{16,}[\'\"]?', r'\1***REDACTED***'),
        # secret="xxx"
        (r'(secret[\"\'\s:=]+)[\'\"a-zA-Z0-9_-]{16,}[\'\"]?', r'\1***REDACTED***'),
        # token="xxx"
        (r'(token[\"\'\s:=]+)[\'\"a-zA-Z0-9_:-]{16,}[\'\"]?', r'\1***REDACTED***'),
        # X-MBX-APIKEY header
        (r'(X-MBX-APIKEY[\"\'\s:=]+)[\'\"a-zA-Z0-9_-]{16,}[\'\"]?', r'\1***REDACTED***'),
        # signature=xxx
        (r'(signature[\"\'\s:=]+)[a-fA-F0-9]{32,}', r'\1***REDACTED***'),
    ]
    
    def filter(self, record: logging.LogRecord) -> bool:
        """過濾並遮蔽日誌訊息中的敏感資訊"""
        if record.msg:
            msg = str(record.msg)
            for pattern, replacement in self.PATTERNS:
                msg = re.sub(pattern, replacement, msg, flags=re.IGNORECASE)
            record.msg = msg
        
        # 也處理 args（格式化參數）
        if record.args:
            new_args = []
            for arg in record.args:
                if isinstance(arg, str):
                    for pattern, replacement in self.PATTERNS:
                        arg = re.sub(pattern, replacement, arg, flags=re.IGNORECASE)
                new_args.append(arg)
            record.args = tuple(new_args)
        
        return True


# 全域的 SecretFilter 實例
_secret_filter = SecretFilter()

# 已配置的 logger 名稱（避免重複配置）
_configured_loggers: set[str] = set()


def get_logger(name: str = "qtrade") -> logging.Logger:
    """
    取得 logger 實例
    
    自動配置：
    - 統一的日誌格式
    - SecretFilter 過濾敏感資訊
    
    Args:
        name: logger 名稱
        
    Returns:
        配置好的 logger
    """
    logger = logging.getLogger(name)
    
    # 避免重複配置
    if name in _configured_loggers:
        return logger
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
        handler.setFormatter(fmt)
        handler.addFilter(_secret_filter)  # 加入敏感資訊過濾
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    else:
        # 如果已有 handler，確保有 SecretFilter
        for handler in logger.handlers:
            if not any(isinstance(f, SecretFilter) for f in handler.filters):
                handler.addFilter(_secret_filter)
    
    _configured_loggers.add(name)
    return logger
