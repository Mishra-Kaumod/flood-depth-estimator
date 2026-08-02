"""
In-memory hourly/daily call and spend budget tracker.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class ApiBudgetConfig:
    hourly_call_cap: int = 0
    daily_call_cap: int = 0
    hourly_spend_cap_usd: float = 0.0
    daily_spend_cap_usd: float = 0.0
    estimated_cost_per_call_usd: float = 0.0


class ApiBudgetTracker:
    def __init__(self, config: ApiBudgetConfig | None = None):
        self.config = config or ApiBudgetConfig()
        self._lock = threading.Lock()
        self._hour_bucket = ""
        self._day_bucket = ""
        self._hour_calls = 0
        self._day_calls = 0

    def _ensure_buckets(self, now: datetime) -> None:
        hour_bucket = now.strftime("%Y-%m-%dT%H")
        day_bucket = now.strftime("%Y-%m-%d")

        if hour_bucket != self._hour_bucket:
            self._hour_bucket = hour_bucket
            self._hour_calls = 0
        if day_bucket != self._day_bucket:
            self._day_bucket = day_bucket
            self._day_calls = 0

    def try_consume(self) -> tuple[bool, str]:
        now = datetime.now(timezone.utc)
        with self._lock:
            self._ensure_buckets(now)

            next_hour_calls = self._hour_calls + 1
            next_day_calls = self._day_calls + 1

            if self.config.hourly_call_cap > 0 and next_hour_calls > self.config.hourly_call_cap:
                return False, "hourly_call_cap_exceeded"
            if self.config.daily_call_cap > 0 and next_day_calls > self.config.daily_call_cap:
                return False, "daily_call_cap_exceeded"

            est_hour_spend = next_hour_calls * max(0.0, self.config.estimated_cost_per_call_usd)
            est_day_spend = next_day_calls * max(0.0, self.config.estimated_cost_per_call_usd)
            if self.config.hourly_spend_cap_usd > 0 and est_hour_spend > self.config.hourly_spend_cap_usd:
                return False, "hourly_spend_cap_exceeded"
            if self.config.daily_spend_cap_usd > 0 and est_day_spend > self.config.daily_spend_cap_usd:
                return False, "daily_spend_cap_exceeded"

            self._hour_calls = next_hour_calls
            self._day_calls = next_day_calls
            return True, ""

    def snapshot(self) -> dict[str, int | str]:
        with self._lock:
            return {
                "hour_bucket": self._hour_bucket,
                "day_bucket": self._day_bucket,
                "hour_calls": self._hour_calls,
                "day_calls": self._day_calls,
            }
