"""
Shared observability middleware: structured logging + lightweight metrics.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from statistics import median
from time import perf_counter
from typing import Callable, Dict, List, TypeVar

T = TypeVar("T")
logger = logging.getLogger(__name__)


@dataclass
class InMemoryMetrics:
    counters: Dict[str, int] = field(default_factory=dict)
    histograms: Dict[str, List[float]] = field(default_factory=dict)
    gauges: Dict[str, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def increment(self, key: str, amount: int = 1) -> None:
        with self._lock:
            self.counters[key] = self.counters.get(key, 0) + amount

    def observe(self, key: str, value: float) -> None:
        with self._lock:
            self.histograms.setdefault(key, []).append(value)

    def set_gauge(self, key: str, value: float) -> None:
        with self._lock:
            self.gauges[key] = float(value)

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            return {
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "histograms": {k: list(v) for k, v in self.histograms.items()},
            }


METRICS = InMemoryMetrics()


def _log(level: int, event: str, **fields: object) -> None:
    payload = {"event": event, **fields}
    logger.log(level, json.dumps(payload, default=str))


def _prom_metric_name(name: str) -> str:
    return str(name).strip().replace(".", "_").replace("-", "_")


def render_prometheus_metrics() -> str:
    snap = METRICS.snapshot()
    lines: List[str] = []

    for key, value in sorted(snap["counters"].items()):
        metric = _prom_metric_name(key)
        lines.append(f"# TYPE {metric} counter")
        lines.append(f"{metric} {int(value)}")

    for key, value in sorted(snap["gauges"].items()):
        metric = _prom_metric_name(key)
        lines.append(f"# TYPE {metric} gauge")
        lines.append(f"{metric} {float(value)}")

    for key, values in sorted(snap["histograms"].items()):
        if not values:
            continue
        metric = _prom_metric_name(key)
        sorted_vals = sorted(float(v) for v in values)
        p95_idx = max(0, int(round(0.95 * len(sorted_vals))) - 1)
        lines.append(f"# TYPE {metric}_count gauge")
        lines.append(f"{metric}_count {len(sorted_vals)}")
        lines.append(f"# TYPE {metric}_sum gauge")
        lines.append(f"{metric}_sum {sum(sorted_vals)}")
        lines.append(f"# TYPE {metric}_avg gauge")
        lines.append(f"{metric}_avg {sum(sorted_vals) / len(sorted_vals)}")
        lines.append(f"# TYPE {metric}_median gauge")
        lines.append(f"{metric}_median {median(sorted_vals)}")
        lines.append(f"# TYPE {metric}_p95 gauge")
        lines.append(f"{metric}_p95 {sorted_vals[p95_idx]}")
        lines.append(f"# TYPE {metric}_max gauge")
        lines.append(f"{metric}_max {max(sorted_vals)}")

    return "\n".join(lines) + "\n"


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
        METRICS.observe("pipeline_latency_ms", latency_ms)
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
