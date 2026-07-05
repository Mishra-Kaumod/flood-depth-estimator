"""
Shared retry middleware for API and queue execution paths.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay_seconds: float = 0.5
    max_delay_seconds: float = 8.0
    jitter_seconds: float = 0.25


def is_transient_error(exc: Exception) -> bool:
    transient_types = [TimeoutError, ConnectionError, OSError]
    try:
        from redis.exceptions import ConnectionError as RedisConnectionError, TimeoutError as RedisTimeoutError

        transient_types.extend([RedisConnectionError, RedisTimeoutError])
    except Exception:
        pass

    return isinstance(exc, tuple(transient_types))


def run_with_retry(
    operation: Callable[[int], T],
    policy: RetryPolicy,
    on_retry: Callable[[int, float, Exception], None] | None = None,
) -> T:
    last_exc: Exception | None = None

    for attempt in range(1, policy.max_attempts + 1):
        try:
            return operation(attempt)
        except Exception as exc:
            last_exc = exc
            if attempt >= policy.max_attempts or not is_transient_error(exc):
                raise

            delay = min(
                policy.max_delay_seconds,
                policy.base_delay_seconds * (2 ** (attempt - 1)),
            ) + random.uniform(0.0, policy.jitter_seconds)

            if on_retry is not None:
                on_retry(attempt, delay, exc)

            time.sleep(delay)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("run_with_retry exited unexpectedly without result")
