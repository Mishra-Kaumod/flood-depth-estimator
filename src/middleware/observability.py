"""
Shared observability middleware: structured logging + lightweight metrics.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Dict, List, TypeVar

T = TypeVar("T")
logger = logging.getLogger(__name__)


@dataclass
class InMemoryMetrics:
    counters: Dict[str, int]
    histograms: Dict[str, List[float]]

    def increment(self, key: str, amount: int = 1) -> None:
        self.counters[key] = self.counters.get(key, 0) + amount

    def observe(self, key: str, value: float) -> None:
        self.histograms.setdefault(key, []).append(value)


METRICS = InMemoryMetrics(counters={}, histograms={})


def _log(level: int, event: str, **fields: object) -> None:
    payload = {"event": event, **fields}
    logger.log(level, json.dumps(payload, default=str))


def observe_execution(
    *,
    event_id: str,
    trace_id: str,
    camera_id: str,
    source: str,
    stage: str,
    attempt: int,
    operation: Callable[[], T],
) -> T:
    start = perf_counter()
    _log(
        logging.INFO,
        "pipeline.start",
        stage=stage,
        event_id=event_id,
        trace_id=trace_id,
        camera_id=camera_id,
        source=source,
        attempt=attempt,
    )
    try:
        result = operation()
        latency_ms = (perf_counter() - start) * 1000.0
        METRICS.increment("events_processed_total")
        METRICS.observe("pipeline_latency_ms", latency_ms)
        _log(
            logging.INFO,
            "pipeline.success",
            stage=stage,
            event_id=event_id,
            trace_id=trace_id,
            camera_id=camera_id,
            source=source,
            attempt=attempt,
            latency_ms=round(latency_ms, 2),
        )
        return result
    except Exception as exc:
        latency_ms = (perf_counter() - start) * 1000.0
        METRICS.increment("events_failed_total")
        _log(
            logging.ERROR,
            "pipeline.failure",
            stage=stage,
            event_id=event_id,
            trace_id=trace_id,
            camera_id=camera_id,
            source=source,
            attempt=attempt,
            latency_ms=round(latency_ms, 2),
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise
