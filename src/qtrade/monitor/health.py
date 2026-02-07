from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class HealthStatus:
    ok: bool
    checked_at: datetime


def check_health() -> HealthStatus:
    return HealthStatus(ok=True, checked_at=datetime.now(timezone.utc))
