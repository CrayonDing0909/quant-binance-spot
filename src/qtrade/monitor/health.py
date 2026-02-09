"""
健康監控模組

監控項目：
- 磁碟空間
- 記憶體使用
- Trading process 存活狀態
- 狀態檔新鮮度（cron 是否停止）
- Binance API 連通性
- VM 重開機偵測

使用方法：
    # 單次檢查
    monitor = HealthMonitor(
        state_path=Path("reports/live/rsi_adx_atr/paper_state.json")
    )
    status = monitor.check_all()
    print(status.summary())
    
    # 配合 Telegram 告警
    if not status.ok:
        notifier.send_error(status.summary())

建議 cron 設定（每 30 分鐘檢查一次）：
    */30 * * * * cd /path/to/project && python -c "from qtrade.monitor.health import run_health_check; run_health_check()" >> /var/log/trading_health.log 2>&1
"""
from __future__ import annotations

import os
import shutil
import socket
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any

from ..utils.log import get_logger

logger = get_logger("health_monitor")


# ══════════════════════════════════════════════════════════════════════════════
# 資料結構
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class HealthCheck:
    """單一健康檢查結果"""
    name: str
    status: str  # "ok", "warning", "critical"
    message: str
    value: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @property
    def is_ok(self) -> bool:
        return self.status == "ok"
    
    @property
    def is_warning(self) -> bool:
        return self.status == "warning"
    
    @property
    def is_critical(self) -> bool:
        return self.status == "critical"


@dataclass
class HealthStatus:
    """整體健康狀態"""
    ok: bool
    checked_at: datetime
    checks: List[HealthCheck] = field(default_factory=list)
    hostname: str = ""
    uptime_seconds: Optional[float] = None
    
    @property
    def warnings(self) -> List[HealthCheck]:
        return [c for c in self.checks if c.is_warning]
    
    @property
    def criticals(self) -> List[HealthCheck]:
        return [c for c in self.checks if c.is_critical]
    
    def to_dict(self) -> dict:
        result = {
            "ok": self.ok,
            "checked_at": self.checked_at.isoformat(),
            "hostname": self.hostname,
            "uptime_seconds": self.uptime_seconds,
            "checks": [c.to_dict() for c in self.checks],
        }
        return result
    
    def summary(self) -> str:
        """產生摘要文字"""
        status = "✅ 正常" if self.ok else "🚨 異常"
        
        lines = [
            "=" * 50,
            f"  系統健康檢查",
            "=" * 50,
            f"  狀態: {status}",
            f"  時間: {self.checked_at.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"  主機: {self.hostname}",
        ]
        
        if self.uptime_seconds:
            hours = self.uptime_seconds / 3600
            lines.append(f"  運行: {hours:.1f} 小時")
        
        lines.append("-" * 50)
        
        for check in self.checks:
            if check.is_ok:
                emoji = "✅"
            elif check.is_warning:
                emoji = "⚠️"
            else:
                emoji = "🚨"
            
            lines.append(f"  {emoji} {check.name}: {check.message}")
        
        lines.append("=" * 50)
        return "\n".join(lines)
    
    def to_telegram_message(self) -> str:
        """產生 Telegram 格式的訊息"""
        status = "✅ 正常" if self.ok else "🚨 異常"
        
        lines = [
            f"<b>系統健康檢查</b> {status}",
            f"時間: {self.checked_at.strftime('%Y-%m-%d %H:%M UTC')}",
            f"主機: {self.hostname}",
            "",
        ]
        
        for check in self.checks:
            if check.is_ok:
                emoji = "✅"
            elif check.is_warning:
                emoji = "⚠️"
            else:
                emoji = "🚨"
            
            lines.append(f"{emoji} <b>{check.name}</b>: {check.message}")
        
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# 健康監控器
# ══════════════════════════════════════════════════════════════════════════════

class HealthMonitor:
    """
    系統健康監控器
    
    監控項目：
    1. 磁碟空間
    2. 記憶體使用
    3. Trading process 存活
    4. 狀態檔新鮮度（偵測 cron 停止）
    5. Binance API 連通性
    6. VM 重開機偵測
    """
    
    # 預設閾值
    DEFAULT_DISK_WARNING_PCT = 0.85
    DEFAULT_DISK_CRITICAL_PCT = 0.95
    DEFAULT_MEMORY_WARNING_PCT = 0.85
    DEFAULT_MEMORY_CRITICAL_PCT = 0.95
    DEFAULT_STATE_STALE_MINUTES = 120  # 2 小時
    DEFAULT_API_TIMEOUT = 10
    
    def __init__(
        self,
        disk_warning_pct: float = DEFAULT_DISK_WARNING_PCT,
        disk_critical_pct: float = DEFAULT_DISK_CRITICAL_PCT,
        memory_warning_pct: float = DEFAULT_MEMORY_WARNING_PCT,
        memory_critical_pct: float = DEFAULT_MEMORY_CRITICAL_PCT,
        state_stale_minutes: int = DEFAULT_STATE_STALE_MINUTES,
        state_path: Optional[Path] = None,
        pid_file: Optional[Path] = None,
        api_timeout: int = DEFAULT_API_TIMEOUT,
        check_network: bool = True,
    ):
        """
        Args:
            disk_warning_pct: 磁碟使用警告閾值
            disk_critical_pct: 磁碟使用嚴重閾值
            memory_warning_pct: 記憶體使用警告閾值
            memory_critical_pct: 記憶體使用嚴重閾值
            state_stale_minutes: 狀態檔過期分鐘數
            state_path: 狀態檔路徑（用於偵測 cron 停止）
            pid_file: PID 檔路徑（用於偵測 process 存活）
            api_timeout: API 連通測試超時秒數
            check_network: 是否檢查網路連通性
        """
        self.disk_warning_pct = disk_warning_pct
        self.disk_critical_pct = disk_critical_pct
        self.memory_warning_pct = memory_warning_pct
        self.memory_critical_pct = memory_critical_pct
        self.state_stale_minutes = state_stale_minutes
        self.state_path = Path(state_path) if state_path else None
        self.pid_file = Path(pid_file) if pid_file else None
        self.api_timeout = api_timeout
        self.check_network = check_network
        
        # 記錄上次開機時間（用於偵測重開機）
        self._last_boot_time: Optional[float] = None
    
    def check_all(self) -> HealthStatus:
        """執行所有健康檢查"""
        checks = []
        
        # 基礎系統檢查
        checks.append(self._check_disk())
        checks.append(self._check_memory())
        
        # Process 檢查
        process_check = self._check_process_alive()
        if process_check:
            checks.append(process_check)
        
        # 狀態檔新鮮度檢查
        state_check = self._check_state_freshness()
        if state_check:
            checks.append(state_check)
        
        # 重開機偵測
        reboot_check = self._check_reboot()
        if reboot_check:
            checks.append(reboot_check)
        
        # 網路連通性檢查
        if self.check_network:
            checks.append(self._check_binance_api())
        
        # 判斷整體狀態
        has_critical = any(c.is_critical for c in checks)
        
        return HealthStatus(
            ok=not has_critical,
            checked_at=datetime.now(timezone.utc),
            checks=checks,
            hostname=socket.gethostname(),
            uptime_seconds=self._get_uptime(),
        )
    
    # ══════════════════════════════════════════════════════════════════════════
    # 個別檢查
    # ══════════════════════════════════════════════════════════════════════════
    
    def _check_disk(self) -> HealthCheck:
        """檢查磁碟空間"""
        try:
            usage = shutil.disk_usage("/")
            used_pct = usage.used / usage.total
            free_gb = usage.free / (1024 ** 3)
            
            if used_pct >= self.disk_critical_pct:
                return HealthCheck(
                    name="磁碟空間",
                    status="critical",
                    message=f"嚴重不足: {used_pct:.1%} 已使用 (剩餘 {free_gb:.1f}GB)",
                    value=used_pct,
                    details={"free_gb": free_gb, "total_gb": usage.total / (1024**3)},
                )
            elif used_pct >= self.disk_warning_pct:
                return HealthCheck(
                    name="磁碟空間",
                    status="warning",
                    message=f"空間偏低: {used_pct:.1%} 已使用 (剩餘 {free_gb:.1f}GB)",
                    value=used_pct,
                    details={"free_gb": free_gb},
                )
            
            return HealthCheck(
                name="磁碟空間",
                status="ok",
                message=f"正常: {used_pct:.1%} 已使用 (剩餘 {free_gb:.1f}GB)",
                value=used_pct,
            )
        except Exception as e:
            return HealthCheck(
                name="磁碟空間",
                status="warning",
                message=f"無法檢查: {e}",
            )
    
    def _check_memory(self) -> HealthCheck:
        """檢查記憶體使用"""
        try:
            # 嘗試使用 psutil
            try:
                import psutil
                mem = psutil.virtual_memory()
                used_pct = mem.percent / 100
                available_gb = mem.available / (1024 ** 3)
            except ImportError:
                # 沒有 psutil，嘗試讀取 /proc/meminfo (Linux)
                used_pct, available_gb = self._get_memory_from_proc()
            
            if used_pct >= self.memory_critical_pct:
                return HealthCheck(
                    name="記憶體",
                    status="critical",
                    message=f"嚴重不足: {used_pct:.1%} 已使用 (可用 {available_gb:.1f}GB)",
                    value=used_pct,
                )
            elif used_pct >= self.memory_warning_pct:
                return HealthCheck(
                    name="記憶體",
                    status="warning",
                    message=f"使用偏高: {used_pct:.1%} 已使用 (可用 {available_gb:.1f}GB)",
                    value=used_pct,
                )
            
            return HealthCheck(
                name="記憶體",
                status="ok",
                message=f"正常: {used_pct:.1%} 已使用 (可用 {available_gb:.1f}GB)",
                value=used_pct,
            )
        except Exception as e:
            return HealthCheck(
                name="記憶體",
                status="warning",
                message=f"無法檢查: {e}",
            )
    
    def _check_process_alive(self) -> Optional[HealthCheck]:
        """檢查 trading process 是否存活"""
        if not self.pid_file:
            return None
        
        try:
            if not self.pid_file.exists():
                return HealthCheck(
                    name="Trading Process",
                    status="warning",
                    message="PID 檔不存在",
                )
            
            with open(self.pid_file) as f:
                pid = int(f.read().strip())
            
            # 檢查 process 是否存在
            try:
                os.kill(pid, 0)  # 不真的 kill，只是檢查
                return HealthCheck(
                    name="Trading Process",
                    status="ok",
                    message=f"運行中 (PID: {pid})",
                    value=pid,
                )
            except OSError:
                return HealthCheck(
                    name="Trading Process",
                    status="critical",
                    message=f"Process 不存在 (PID: {pid})",
                    value=pid,
                )
        except Exception as e:
            return HealthCheck(
                name="Trading Process",
                status="warning",
                message=f"無法檢查: {e}",
            )
    
    def _check_state_freshness(self) -> Optional[HealthCheck]:
        """
        檢查狀態檔是否過期
        
        用於偵測 cron 停止或程式當掉
        """
        if not self.state_path or not self.state_path.exists():
            return None
        
        try:
            mtime = datetime.fromtimestamp(
                self.state_path.stat().st_mtime, tz=timezone.utc
            )
            age = datetime.now(timezone.utc) - mtime
            age_minutes = age.total_seconds() / 60
            
            if age_minutes >= self.state_stale_minutes:
                return HealthCheck(
                    name="狀態更新",
                    status="critical",
                    message=f"已 {age_minutes:.0f} 分鐘未更新，cron 可能停止",
                    value=age_minutes,
                    details={"last_update": mtime.isoformat()},
                )
            elif age_minutes >= self.state_stale_minutes * 0.8:
                return HealthCheck(
                    name="狀態更新",
                    status="warning",
                    message=f"接近過期: {age_minutes:.0f} 分鐘前更新",
                    value=age_minutes,
                )
            
            return HealthCheck(
                name="狀態更新",
                status="ok",
                message=f"正常: {age_minutes:.0f} 分鐘前更新",
                value=age_minutes,
            )
        except Exception as e:
            return HealthCheck(
                name="狀態更新",
                status="warning",
                message=f"無法檢查: {e}",
            )
    
    def _check_reboot(self) -> Optional[HealthCheck]:
        """
        偵測 VM 重開機
        
        透過比較 boot time 來偵測
        """
        try:
            current_boot_time = self._get_boot_time()
            if current_boot_time is None:
                return None
            
            if self._last_boot_time is None:
                self._last_boot_time = current_boot_time
                return HealthCheck(
                    name="系統啟動",
                    status="ok",
                    message="首次檢查",
                )
            
            if current_boot_time != self._last_boot_time:
                self._last_boot_time = current_boot_time
                boot_dt = datetime.fromtimestamp(current_boot_time, tz=timezone.utc)
                return HealthCheck(
                    name="系統啟動",
                    status="warning",
                    message=f"偵測到重開機: {boot_dt.strftime('%Y-%m-%d %H:%M UTC')}",
                    details={"boot_time": boot_dt.isoformat()},
                )
            
            return None  # 沒有重開機
            
        except Exception as e:
            return HealthCheck(
                name="系統啟動",
                status="warning",
                message=f"無法偵測: {e}",
            )
    
    def _check_binance_api(self) -> HealthCheck:
        """檢查 Binance API 連通性"""
        try:
            import requests
            
            start = time.time()
            resp = requests.get(
                "https://api.binance.com/api/v3/ping",
                timeout=self.api_timeout,
            )
            latency_ms = (time.time() - start) * 1000
            
            if resp.status_code == 200:
                if latency_ms > 2000:
                    return HealthCheck(
                        name="Binance API",
                        status="warning",
                        message=f"連接慢: {latency_ms:.0f}ms",
                        value=latency_ms,
                    )
                return HealthCheck(
                    name="Binance API",
                    status="ok",
                    message=f"正常: {latency_ms:.0f}ms",
                    value=latency_ms,
                )
            
            return HealthCheck(
                name="Binance API",
                status="warning",
                message=f"回應異常: HTTP {resp.status_code}",
                value=resp.status_code,
            )
            
        except requests.exceptions.Timeout:
            return HealthCheck(
                name="Binance API",
                status="critical",
                message=f"連接超時 (>{self.api_timeout}s)",
            )
        except requests.exceptions.ConnectionError:
            return HealthCheck(
                name="Binance API",
                status="critical",
                message="連接失敗",
            )
        except Exception as e:
            return HealthCheck(
                name="Binance API",
                status="warning",
                message=f"檢查失敗: {e}",
            )
    
    # ══════════════════════════════════════════════════════════════════════════
    # 輔助方法
    # ══════════════════════════════════════════════════════════════════════════
    
    def _get_memory_from_proc(self) -> tuple[float, float]:
        """從 /proc/meminfo 讀取記憶體資訊 (Linux)"""
        with open("/proc/meminfo") as f:
            lines = f.readlines()
        
        mem_info = {}
        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                key = parts[0].rstrip(":")
                value = int(parts[1])  # kB
                mem_info[key] = value
        
        total = mem_info.get("MemTotal", 0)
        available = mem_info.get("MemAvailable", mem_info.get("MemFree", 0))
        
        used_pct = (total - available) / total if total > 0 else 0
        available_gb = available / (1024 ** 2)  # kB to GB
        
        return used_pct, available_gb
    
    def _get_boot_time(self) -> Optional[float]:
        """取得系統開機時間"""
        try:
            # 嘗試 psutil
            try:
                import psutil
                return psutil.boot_time()
            except ImportError:
                pass
            
            # Linux: 讀取 /proc/stat
            with open("/proc/stat") as f:
                for line in f:
                    if line.startswith("btime"):
                        return float(line.split()[1])
            
            return None
        except Exception:
            return None
    
    def _get_uptime(self) -> Optional[float]:
        """取得系統運行時間（秒）"""
        try:
            boot_time = self._get_boot_time()
            if boot_time:
                return time.time() - boot_time
            
            # Linux: 讀取 /proc/uptime
            with open("/proc/uptime") as f:
                return float(f.read().split()[0])
        except Exception:
            return None


# ══════════════════════════════════════════════════════════════════════════════
# 便利函數
# ══════════════════════════════════════════════════════════════════════════════

def check_health() -> HealthStatus:
    """
    簡單健康檢查（向後相容）
    
    Returns:
        HealthStatus
    """
    monitor = HealthMonitor(check_network=False)
    return monitor.check_all()


def run_health_check(
    config_path: str = "config/rsi_adx_atr.yaml",
    notify: bool = True,
    notify_on_ok: bool = False,
) -> HealthStatus:
    """
    執行健康檢查並可選發送通知
    
    Args:
        config_path: 配置檔路徑
        notify: 是否發送 Telegram 通知
        notify_on_ok: 正常時是否也發送通知
        
    Returns:
        HealthStatus
    """
    from ..config import load_config
    from .notifier import TelegramNotifier
    
    # 載入配置以取得策略名稱
    try:
        cfg = load_config(config_path)
        strategy_name = cfg.strategy.name
        state_path = Path(f"reports/live/{strategy_name}/paper_state.json")
    except Exception:
        state_path = None
    
    # 執行檢查
    monitor = HealthMonitor(state_path=state_path)
    status = monitor.check_all()
    
    # 輸出結果
    print(status.summary())
    
    # 發送通知
    if notify:
        notifier = TelegramNotifier()
        if notifier.enabled:
            # 只在異常或要求時發送
            if not status.ok or notify_on_ok:
                notifier.send(status.to_telegram_message())
                print("\n📱 Telegram 通知已發送")
    
    return status


# 向後相容的簡單介面
@dataclass
class SimpleHealthStatus:
    ok: bool
    checked_at: datetime


def simple_check_health() -> SimpleHealthStatus:
    """極簡健康檢查（向後相容舊程式碼）"""
    return SimpleHealthStatus(ok=True, checked_at=datetime.now(timezone.utc))
