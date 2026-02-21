"""
Live Watchdog

提供 WebSocket 實盤流程的內建自我健康檢查：
  - run_checks_once()
  - run_periodic_checks()
  - Telegram 告警（含 cooldown 與 recovery）
  - 狀態落盤（latest_status.json / history.jsonl）
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from ..utils.log import get_logger

logger = get_logger("live_watchdog")


DEFAULT_SETTINGS: dict[str, Any] = {
    "enabled": True,
    "interval_sec": 300,
    "alert_cooldown_sec": 1800,
    "notify_phase_change": True,
    "history_limit": 200,
    "output_dir": "reports/live_watchdog",
    "startup_grace_sec": 300,
    "runner_ready_grace_sec": 180,
    # main loop heartbeat（run loop 是否仍在運作）
    "heartbeat_warn_sec": 90,
    "heartbeat_critical_sec": 180,
    # websocket / kline 活性
    "ws_msg_warn_sec": 180,
    "ws_msg_critical_sec": 300,
    "kline_warn_sec": 7200,
    "kline_critical_sec": 14400,
    # data freshness（同時檢查 1h 與 5m parquet）
    "data_intervals": ["1h", "5m"],
    "data_warn_age_sec": {"1h": 7200, "5m": 1800},
    "data_critical_age_sec": {"1h": 14400, "5m": 3600},
    "data_warn_stale_symbols": 2,
    "data_critical_stale_symbols": 5,
    "data_warn_missing_symbols": 1,
    "data_critical_missing_symbols": 3,
    # 錯誤密度
    "error_window_minutes": 15,
    "error_warn_count": 5,
    "error_critical_count": 12,
    "error_scan_tail_lines": 2000,
}


def _status_rank(status: str) -> int:
    return {"ok": 0, "warn": 1, "critical": 2}.get(status, 1)


def _max_status(a: str, b: str) -> str:
    return a if _status_rank(a) >= _status_rank(b) else b


def _interval_to_seconds(interval: str) -> int:
    """簡單解析 Binance interval（如 5m/1h/1d）。"""
    if not interval:
        return 3600
    unit = interval[-1].lower()
    try:
        value = int(interval[:-1])
    except Exception:
        return 3600
    if unit == "m":
        return value * 60
    if unit == "h":
        return value * 3600
    if unit == "d":
        return value * 86400
    return 3600


class LiveWatchdog:
    """WebSocket live watchdog."""

    def __init__(
        self,
        runner: Any,
        cfg: Any,
        notifier: Any = None,
        settings: dict[str, Any] | None = None,
    ):
        self.runner = runner
        self.cfg = cfg
        self.notifier = notifier or getattr(runner, "notifier", None)
        self.pid = os.getpid()

        live_watchdog_cfg = getattr(getattr(cfg, "live", None), "watchdog", {}) or {}
        merged: dict[str, Any] = dict(DEFAULT_SETTINGS)
        merged.update(live_watchdog_cfg)
        if settings:
            merged.update(settings)

        # 向後相容：若未明確配置 data_intervals，預設追蹤策略實際交易週期
        if "data_intervals" not in live_watchdog_cfg:
            market_interval = getattr(getattr(cfg, "market", None), "interval", "1h")
            merged["data_intervals"] = [market_interval]

            # 同步補齊該 interval 的新鮮度閾值（2x / 4x bar）
            sec = _interval_to_seconds(market_interval)
            warn_map = dict(merged.get("data_warn_age_sec", {}))
            critical_map = dict(merged.get("data_critical_age_sec", {}))
            warn_map.setdefault(market_interval, max(sec * 2, 300))
            critical_map.setdefault(market_interval, max(sec * 4, 600))
            merged["data_warn_age_sec"] = warn_map
            merged["data_critical_age_sec"] = critical_map

        self.settings = merged

        self.enabled = bool(self.settings.get("enabled", True))
        self.interval_sec = int(self.settings.get("interval_sec", 300))
        self.alert_cooldown_sec = int(self.settings.get("alert_cooldown_sec", 1800))
        self.history_limit = int(self.settings.get("history_limit", 200))

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_status: dict[str, Any] | None = None
        self._last_check_ts: float | None = None
        self._last_overall_status = "ok"
        self._last_ws_phase: str | None = None
        self._alert_cooldowns: dict[str, float] = {}

        self._output_dir = Path(self.settings.get("output_dir", "reports/live_watchdog"))
        self._latest_path = self._output_dir / "latest_status.json"
        self._history_path = self._output_dir / "history.jsonl"
        self._pid_path = self._output_dir / "watchdog.pid"
        self._log_path = Path(self.settings.get("log_path", "logs/websocket.log"))

    def start_background(self) -> None:
        """背景啟動週期檢查（非阻塞）。"""
        if not self.enabled:
            logger.info("Live Watchdog 已停用（enabled=false）")
            return
        if self._thread and self._thread.is_alive():
            logger.warning("Live Watchdog 已在執行中")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self.run_periodic_checks, daemon=True)
        self._thread.start()
        logger.info(f"🩺 Live Watchdog 已啟動，interval={self.interval_sec}s")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        logger.info("🩺 Live Watchdog 已停止")

    def get_runtime_status(self) -> dict[str, Any]:
        with self._lock:
            last = self._last_status
            return {
                "enabled": self.enabled,
                "interval_sec": self.interval_sec,
                "is_running": bool(self._thread and self._thread.is_alive()),
                "last_check_timestamp": (
                    datetime.fromtimestamp(self._last_check_ts, tz=timezone.utc).isoformat()
                    if self._last_check_ts
                    else None
                ),
                "last_overall_status": last.get("overall_status") if last else None,
                "last_issue_count": len(last.get("issues", [])) if last else 0,
                "last_phase": (
                    (last.get("checks", {}).get("websocket_kline", {}) or {}).get("phase")
                    if last
                    else None
                ),
            }

    def get_last_status(self) -> dict[str, Any] | None:
        with self._lock:
            return self._last_status

    def run_checks_once(self, notify: bool = True) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        overall_status = "ok"
        checks: dict[str, Any] = {}
        issues: list[str] = []

        for check_name, check_result in (
            ("heartbeat", self._check_heartbeat()),
            ("websocket_kline", self._check_websocket_and_kline()),
            ("data_freshness", self._check_data_freshness()),
            ("error_density", self._check_error_density()),
            ("session_uniqueness", self._check_session_uniqueness()),
        ):
            checks[check_name] = check_result
            status = check_result.get("status", "warn")
            overall_status = _max_status(overall_status, status)
            if status != "ok":
                issues.append(f"[{check_name}] {check_result.get('message', '')}")

        result = {
            "timestamp": now.isoformat(),
            "overall_status": overall_status,
            "checks": checks,
            "issues": issues,
        }

        with self._lock:
            self._last_status = result
            self._last_check_ts = time.time()

        self._persist_status(result)
        if notify:
            self._handle_alerting(result)
        return result

    def run_periodic_checks(self) -> None:
        """阻塞式週期檢查循環。"""
        while not self._stop_event.is_set():
            try:
                result = self.run_checks_once(notify=True)
                logger.info(
                    f"Watchdog 檢查完成: status={result['overall_status']} "
                    f"issues={len(result['issues'])}"
                )
            except Exception as e:
                logger.error(f"Watchdog 例外（已降級，不影響交易主流程）: {e}")
            self._stop_event.wait(timeout=self.interval_sec)

    def _check_heartbeat(self) -> dict[str, Any]:
        now_ts = time.time()
        last_loop = float(getattr(self.runner, "_last_main_loop_heartbeat", 0.0) or 0.0)
        if last_loop <= 0:
            return {
                "status": "warn",
                "message": "主迴圈 heartbeat 尚未建立",
                "last_activity_seconds": None,
            }
        age = now_ts - last_loop
        warn_sec = float(self.settings.get("heartbeat_warn_sec", 90))
        critical_sec = float(self.settings.get("heartbeat_critical_sec", 180))
        if age >= critical_sec:
            status = "critical"
            msg = f"主迴圈 heartbeat 過期 {age:.0f}s"
        elif age >= warn_sec:
            status = "warn"
            msg = f"主迴圈 heartbeat 偏舊 {age:.0f}s"
        else:
            status = "ok"
            msg = f"主迴圈 heartbeat 正常 ({age:.0f}s)"
        return {"status": status, "message": msg, "last_activity_seconds": round(age, 1)}

    def _check_websocket_and_kline(self) -> dict[str, Any]:
        now_ts = time.time()
        ws_client = getattr(self.runner, "_ws_client", None)
        is_running = bool(getattr(self.runner, "is_running", False))
        ws_ready = bool(getattr(self.runner, "_ws_ready", False))
        subscriptions_ready = bool(getattr(self.runner, "_subscriptions_ready", False))
        started_at = float(getattr(self.runner, "_started_at", 0.0) or 0.0)
        last_ws_ts = float(getattr(self.runner, "_last_ws_message_time", 0.0) or 0.0)
        last_kline_event_time = float(getattr(self.runner, "_last_kline_event_time", 0.0) or 0.0)

        if started_at <= 0:
            started_at = float(getattr(self.runner, "start_time", 0.0) or 0.0)
        uptime_sec = (now_ts - started_at) if started_at > 0 else None

        ws_age = None if last_ws_ts <= 0 else now_ts - last_ws_ts
        kline_age_sec = None if last_kline_event_time <= 0 else now_ts - last_kline_event_time
        startup_grace_sec = float(self.settings.get("startup_grace_sec", 300))
        runner_ready_grace_sec = float(self.settings.get("runner_ready_grace_sec", 180))

        runner_ready = bool(is_running and ws_client is not None and ws_ready and subscriptions_ready)
        startup_grace_remaining_sec = 0
        runner_ready_grace_remaining_sec = 0
        notes: list[str] = []

        # Phase 1: BOOTSTRAP（啟動寬限內只回 ok）
        if uptime_sec is None or uptime_sec < startup_grace_sec:
            if uptime_sec is None:
                startup_grace_remaining_sec = int(startup_grace_sec)
            else:
                startup_grace_remaining_sec = int(max(0, startup_grace_sec - uptime_sec))
            notes.append("啟動寬限中，暫不判定 websocket_kline 異常")
            if not runner_ready:
                notes.append("runner 尚在初始化（預期行為）")
            return {
                "status": "ok",
                "phase": "bootstrap",
                "phase_detail": "bootstrap_grace",
                "runner_ready": runner_ready,
                "startup_grace_remaining_sec": startup_grace_remaining_sec,
                "runner_ready_grace_remaining_sec": int(runner_ready_grace_sec),
                "message": " | ".join(notes),
                "ws_connected": bool(ws_client is not None and is_running),
                "last_ws_message_age_sec": round(ws_age, 1) if ws_age is not None else None,
                "last_kline_age_sec": round(kline_age_sec, 1) if kline_age_sec is not None else None,
            }

        # runner ready grace：避免剛過 bootstrap 時因資源初始化慢而誤報 critical
        if not runner_ready:
            if uptime_sec is not None:
                runner_ready_grace_remaining_sec = int(max(0, runner_ready_grace_sec - uptime_sec))
            if uptime_sec is None or uptime_sec < runner_ready_grace_sec:
                notes.append("runner 尚未 ready（ready grace 內）")
                return {
                    "status": "ok",
                    "phase": "bootstrap",
                    "phase_detail": "runner_ready_grace",
                    "runner_ready": False,
                    "startup_grace_remaining_sec": 0,
                    "runner_ready_grace_remaining_sec": runner_ready_grace_remaining_sec,
                    "message": " | ".join(notes),
                    "ws_connected": bool(ws_client is not None and is_running),
                    "last_ws_message_age_sec": round(ws_age, 1) if ws_age is not None else None,
                    "last_kline_age_sec": round(kline_age_sec, 1) if kline_age_sec is not None else None,
                }
            notes.append("runner 長時間未 ready，疑似啟動異常")
            return {
                "status": "critical",
                "phase": "stale",
                "phase_detail": "runner_not_ready_timeout",
                "runner_ready": False,
                "startup_grace_remaining_sec": 0,
                "runner_ready_grace_remaining_sec": 0,
                "message": " | ".join(notes),
                "ws_connected": bool(ws_client is not None and is_running),
                "last_ws_message_age_sec": round(ws_age, 1) if ws_age is not None else None,
                "last_kline_age_sec": round(kline_age_sec, 1) if kline_age_sec is not None else None,
            }

        warn_sec = float(self.settings.get("ws_msg_warn_sec", 180))
        critical_sec = float(self.settings.get("ws_msg_critical_sec", 300))
        k_warn = float(self.settings.get("kline_warn_sec", 7200))
        k_critical = float(self.settings.get("kline_critical_sec", 14400))

        # Phase 3: STREAM_STALE（保留真斷線偵測）
        if ws_age is None or ws_age >= critical_sec:
            if ws_age is None:
                notes.append("未收到任何 WS 訊息（超過啟動寬限）")
            else:
                notes.append(f"WS 訊息延遲 {ws_age:.0f}s，疑似斷線")
            return {
                "status": "critical",
                "phase": "stale",
                "phase_detail": "stream_stale",
                "runner_ready": True,
                "startup_grace_remaining_sec": 0,
                "runner_ready_grace_remaining_sec": 0,
                "message": " | ".join(notes),
                "ws_connected": bool(ws_client is not None and is_running),
                "last_ws_message_age_sec": round(ws_age, 1) if ws_age is not None else None,
                "last_kline_age_sec": round(kline_age_sec, 1) if kline_age_sec is not None else None,
            }

        # Phase 2: STREAMING_NO_CLOSE_YET（WS 有消息但尚無收盤 K）
        if kline_age_sec is None:
            status = "warn" if ws_age >= warn_sec else "ok"
            notes.append(f"WS 訊息正常 ({ws_age:.0f}s)")
            notes.append("尚未收到 K 線收盤事件（1h 週期可能正常）")
            return {
                "status": status,
                "phase": "streaming",
                "phase_detail": "streaming_no_close_yet",
                "runner_ready": True,
                "startup_grace_remaining_sec": 0,
                "runner_ready_grace_remaining_sec": 0,
                "message": " | ".join(notes),
                "ws_connected": bool(ws_client is not None and is_running),
                "last_ws_message_age_sec": round(ws_age, 1),
                "last_kline_age_sec": None,
            }

        # Phase: STREAMING（正常運行；同時看 WS 和 K 線新鮮度）
        status = "ok"
        notes.append(f"WS 訊息正常 ({ws_age:.0f}s)")
        if ws_age >= warn_sec:
            status = _max_status(status, "warn")
            notes.append(f"WS 訊息偏舊 {ws_age:.0f}s")

        if kline_age_sec >= k_critical:
            status = _max_status(status, "critical")
            notes.append(f"K 線事件過期 {kline_age_sec/60:.1f}m")
        elif kline_age_sec >= k_warn:
            status = _max_status(status, "warn")
            notes.append(f"K 線事件偏舊 {kline_age_sec/60:.1f}m")
        else:
            notes.append(f"K 線事件正常 ({kline_age_sec/60:.1f}m)")

        return {
            "status": status,
            "phase": "streaming",
            "phase_detail": "streaming_active",
            "runner_ready": True,
            "startup_grace_remaining_sec": 0,
            "runner_ready_grace_remaining_sec": 0,
            "message": " | ".join(notes),
            "ws_connected": bool(ws_client is not None and is_running),
            "last_ws_message_age_sec": round(ws_age, 1) if ws_age is not None else None,
            "last_kline_age_sec": round(kline_age_sec, 1) if kline_age_sec is not None else None,
        }

    def _check_data_freshness(self) -> dict[str, Any]:
        symbols = list(getattr(self.cfg.market, "symbols", []))
        intervals = list(self.settings.get("data_intervals", ["1h", "5m"]))
        base_dir = Path(self.cfg.data_dir) / "binance" / self.cfg.market_type_str
        now_ts = time.time()

        warn_age_cfg = self.settings.get("data_warn_age_sec", {})
        critical_age_cfg = self.settings.get("data_critical_age_sec", {})

        detail: dict[str, Any] = {}
        overall = "ok"
        messages: list[str] = []

        for interval in intervals:
            stale_symbols: list[str] = []
            missing_symbols: list[str] = []
            ages: dict[str, float] = {}

            warn_age = float(warn_age_cfg.get(interval, 7200))
            critical_age = float(critical_age_cfg.get(interval, 14400))
            for sym in symbols:
                # 支援兩種歷史格式：
                # 1) data/binance/futures/5m/BTCUSDT.parquet
                # 2) data/binance/futures/klines/BTCUSDT_5m.parquet
                candidates = [
                    base_dir / interval / f"{sym}.parquet",
                    base_dir / "klines" / f"{sym}_{interval}.parquet",
                ]
                fp = next((p for p in candidates if p.exists()), None)
                if fp is None:
                    missing_symbols.append(sym)
                    continue
                age_sec = now_ts - fp.stat().st_mtime
                ages[sym] = age_sec
                if age_sec >= warn_age:
                    stale_symbols.append(sym)

            warn_stale_n = int(self.settings.get("data_warn_stale_symbols", 2))
            crit_stale_n = int(self.settings.get("data_critical_stale_symbols", 5))
            warn_missing_n = int(self.settings.get("data_warn_missing_symbols", 1))
            crit_missing_n = int(self.settings.get("data_critical_missing_symbols", 3))

            interval_status = "ok"
            if len(missing_symbols) >= crit_missing_n:
                interval_status = "critical"
            elif len(missing_symbols) >= warn_missing_n:
                interval_status = "warn"
            if len(stale_symbols) >= crit_stale_n:
                interval_status = "critical"
            elif len(stale_symbols) >= warn_stale_n:
                interval_status = _max_status(interval_status, "warn")

            # 若最舊 stale 已超 critical age，也直接升級
            if stale_symbols:
                oldest = max(ages[s] for s in stale_symbols if s in ages)
                if oldest >= critical_age:
                    interval_status = _max_status(interval_status, "critical")

            overall = _max_status(overall, interval_status)
            messages.append(
                f"{interval}: stale={len(stale_symbols)}, missing={len(missing_symbols)}, status={interval_status}"
            )
            detail[interval] = {
                "status": interval_status,
                "stale_count": len(stale_symbols),
                "missing_count": len(missing_symbols),
                "stale_symbols": stale_symbols[:20],
                "missing_symbols": missing_symbols[:20],
            }

        return {
            "status": overall,
            "message": " ; ".join(messages),
            "symbols_total": len(symbols),
            "details": detail,
        }

    def _check_error_density(self) -> dict[str, Any]:
        if not self._log_path.exists():
            return {"status": "ok", "message": "找不到 log 檔，略過錯誤密度檢查", "count": 0}

        window_minutes = int(self.settings.get("error_window_minutes", 15))
        warn_n = int(self.settings.get("error_warn_count", 5))
        critical_n = int(self.settings.get("error_critical_count", 12))
        tail_lines = int(self.settings.get("error_scan_tail_lines", 2000))
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)

        with self._log_path.open("r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()[-tail_lines:]

        error_pattern = re.compile(r"(error|exception|traceback|fatal)", flags=re.IGNORECASE)
        ts_pattern = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
        count = 0
        samples: list[str] = []

        for line in lines:
            if not error_pattern.search(line):
                continue
            # 優先依 timestamp 過濾最近 N 分鐘
            include = True
            m = ts_pattern.match(line)
            if m:
                try:
                    ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                    include = ts >= cutoff
                except Exception:
                    include = True
            if include:
                count += 1
                if len(samples) < 5:
                    samples.append(line.strip()[:180])

        if count >= critical_n:
            status = "critical"
        elif count >= warn_n:
            status = "warn"
        else:
            status = "ok"

        return {
            "status": status,
            "message": f"最近 {window_minutes}m 錯誤關鍵字 {count} 次",
            "count": count,
            "samples": samples,
        }

    def _check_session_uniqueness(self) -> dict[str, Any]:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        previous_pid = None
        if self._pid_path.exists():
            try:
                previous_pid = int(self._pid_path.read_text(encoding="utf-8").strip())
            except Exception:
                previous_pid = None

        if previous_pid and previous_pid != self.pid:
            alive = self._pid_alive(previous_pid)
            if alive:
                return {
                    "status": "critical",
                    "message": f"偵測到另一個 active session PID={previous_pid}",
                    "current_pid": self.pid,
                    "other_pid": previous_pid,
                }

        try:
            self._pid_path.write_text(str(self.pid), encoding="utf-8")
        except Exception as e:
            return {"status": "warn", "message": f"無法寫入 watchdog pid 檔: {e}"}

        return {"status": "ok", "message": f"session 唯一性正常 (PID={self.pid})", "current_pid": self.pid}

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def _persist_status(self, status: dict[str, Any]) -> None:
        try:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            with self._latest_path.open("w", encoding="utf-8") as f:
                json.dump(status, f, ensure_ascii=False, indent=2)
            with self._history_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(status, ensure_ascii=False) + "\n")
            self._trim_history_if_needed()
        except Exception as e:
            logger.warning(f"Watchdog 狀態落盤失敗: {e}")

    def _trim_history_if_needed(self) -> None:
        try:
            with self._history_path.open("r", encoding="utf-8") as f:
                lines = f.readlines()
            if len(lines) <= self.history_limit:
                return
            keep = lines[-self.history_limit:]
            with self._history_path.open("w", encoding="utf-8") as f:
                f.writelines(keep)
        except Exception:
            pass

    def _handle_alerting(self, status: dict[str, Any]) -> None:
        if not self.notifier or not getattr(self.notifier, "enabled", False):
            self._last_overall_status = status["overall_status"]
            return

        overall = status.get("overall_status", "ok")
        now_ts = time.time()

        # recovery: warn/critical -> ok
        if self._last_overall_status in {"warn", "critical"} and overall == "ok":
            try:
                self.notifier.send(
                    "✅ <b>Watchdog Recovery</b>\n\n"
                    "系統狀態已從異常恢復到 <b>ok</b>。"
                )
            except Exception:
                pass

        # warn/critical 主動通知（每個 check 做 cooldown）
        if overall in {"warn", "critical"}:
            checks = status.get("checks", {})
            for check_name, detail in checks.items():
                level = detail.get("status", "ok")
                if level not in {"warn", "critical"}:
                    continue
                key = f"{check_name}:{level}"
                last_sent = self._alert_cooldowns.get(key, 0.0)
                if now_ts - last_sent < self.alert_cooldown_sec:
                    continue
                message = (
                    f"{'🚨' if level == 'critical' else '⚠️'} <b>Live Watchdog {level.upper()}</b>\n\n"
                    f"檢查項目: <b>{check_name}</b>\n"
                    f"說明: {detail.get('message', '')}\n"
                    f"時間: {status.get('timestamp', '')}"
                )
                try:
                    self.notifier.send(message)
                    self._alert_cooldowns[key] = now_ts
                except Exception:
                    pass

        # 可選 phase change 通知：bootstrap -> streaming
        ws_check = status.get("checks", {}).get("websocket_kline", {}) or {}
        phase = ws_check.get("phase")
        notify_phase_change = bool(self.settings.get("notify_phase_change", True))
        if (
            notify_phase_change
            and phase
            and self._last_ws_phase == "bootstrap"
            and phase == "streaming"
        ):
            try:
                self.notifier.send(
                    "ℹ️ <b>Live Watchdog Phase Update</b>\n\n"
                    "WebSocket 監控已從 bootstrap 進入 streaming。"
                )
            except Exception:
                pass

        if phase:
            self._last_ws_phase = phase
        self._last_overall_status = overall
