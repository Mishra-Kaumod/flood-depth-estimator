"""
Simple in-memory circuit breaker for external dependency calls.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class CircuitBreakerConfig:
    failure_threshold: int = 5
    failure_window_seconds: int = 300
    cooldown_seconds: int = 180


class CircuitBreaker:
    def __init__(self, config: CircuitBreakerConfig | None = None):
        self.config = config or CircuitBreakerConfig()
        self._lock = threading.Lock()
        self._open_until = 0.0
        self._consecutive_failures = 0
        self._failure_window_start = 0.0

    def allow_request(self) -> bool:
        now = time.time()
        with self._lock:
            if now < self._open_until:
                return False
            if self._open_until > 0 and now >= self._open_until:
                self._open_until = 0.0
                self._consecutive_failures = 0
                self._failure_window_start = 0.0
            return True

    def record_success(self) -> None:
        with self._lock:
            self._consecutive_failures = 0
            self._failure_window_start = 0.0
            self._open_until = 0.0

    def record_failure(self) -> None:
        now = time.time()
        with self._lock:
            if (
                self._failure_window_start <= 0
                or (now - self._failure_window_start) > self.config.failure_window_seconds
            ):
                self._failure_window_start = now
                self._consecutive_failures = 1
            else:
                self._consecutive_failures += 1

            if self._consecutive_failures >= self.config.failure_threshold:
                self._open_until = now + float(self.config.cooldown_seconds)

    def state(self) -> dict[str, float | int | bool]:
        now = time.time()
        with self._lock:
            return {
                "open": now < self._open_until,
                "open_until_epoch": self._open_until,
                "consecutive_failures": self._consecutive_failures,
                "failure_window_start_epoch": self._failure_window_start,
            }
