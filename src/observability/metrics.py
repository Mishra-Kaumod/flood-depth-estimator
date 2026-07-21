# src/observability/metrics.py
"""
Prometheus Metrics
===================
Exposes a /metrics endpoint and provides counters/histograms
for every key operation in the pipeline.

Usage (in any module):
    from src.observability.metrics import METRICS
    METRICS.images_processed.inc()
    with METRICS.pipeline_duration.time():
        result = run_pipeline(...)
"""

import logging
import threading
from dataclasses import dataclass

log = logging.getLogger("observability.metrics")

_registry = None


def _get_registry():
    global _registry
    if _registry is None:
        try:
            from prometheus_client import CollectorRegistry
            _registry = CollectorRegistry()
        except ImportError:
            pass
    return _registry


class _NullMetric:
    """No-op metric used when prometheus_client is not installed."""
    def inc(self, *a, **kw): pass
    def dec(self, *a, **kw): pass
    def set(self, *a, **kw): pass
    def observe(self, *a, **kw): pass
    def labels(self, **kw): return self
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def time(self): return self


@dataclass
class FloodWatchMetrics:
    # Pipeline
    images_processed:      object
    images_failed:         object
    pipeline_duration:     object   # histogram in seconds
    batch_size:            object   # histogram

    # Per-stage latency
    stage_duration:        object   # labels: stage=segformer|yolo|depth|fusion|severity|gemini

    # Risk distribution
    risk_counter:          object   # labels: risk_level=...

    # Queue
    queue_depth:           object
    dead_letter_count:     object

    # API
    api_requests:          object   # labels: endpoint, status
    api_latency:           object

    # DB
    db_writes:             object
    db_write_errors:       object


def _build_metrics(registry) -> FloodWatchMetrics:
    try:
        from prometheus_client import Counter, Histogram, Gauge, Summary

        return FloodWatchMetrics(
            images_processed  = Counter("floodwatch_images_processed_total",
                                        "Total images run through pipeline", registry=registry),
            images_failed     = Counter("floodwatch_images_failed_total",
                                        "Images that failed pipeline", registry=registry),
            pipeline_duration = Histogram("floodwatch_pipeline_duration_seconds",
                                          "Full pipeline latency per image",
                                          buckets=[0.1,0.5,1,2,5,10,30],
                                          registry=registry),
            batch_size        = Histogram("floodwatch_batch_size",
                                          "Images per 15-min batch",
                                          buckets=[1,5,10,25,50,100,250,500],
                                          registry=registry),
            stage_duration    = Histogram("floodwatch_stage_duration_seconds",
                                          "Per-stage latency",
                                          labelnames=["stage"],
                                          buckets=[0.01,0.05,0.1,0.5,1,3,10],
                                          registry=registry),
            risk_counter      = Counter("floodwatch_risk_level_total",
                                        "Predictions by risk level",
                                        labelnames=["risk_level"],
                                        registry=registry),
            queue_depth       = Gauge("floodwatch_queue_depth",
                                      "Jobs waiting in Redis queue", registry=registry),
            dead_letter_count = Gauge("floodwatch_dead_letter_total",
                                      "Jobs in dead-letter queue", registry=registry),
            api_requests      = Counter("floodwatch_api_requests_total",
                                        "API requests by endpoint and status",
                                        labelnames=["endpoint","status"],
                                        registry=registry),
            api_latency       = Histogram("floodwatch_api_latency_seconds",
                                          "API response time",
                                          labelnames=["endpoint"],
                                          buckets=[0.05,0.1,0.25,0.5,1,2,5],
                                          registry=registry),
            db_writes         = Counter("floodwatch_db_writes_total",
                                        "Successful DB writes", registry=registry),
            db_write_errors   = Counter("floodwatch_db_write_errors_total",
                                        "Failed DB writes", registry=registry),
        )
    except ImportError:
        log.warning("prometheus_client not installed — metrics disabled. pip install prometheus-client")
        null = _NullMetric()
        return FloodWatchMetrics(**{f.name: null for f in FloodWatchMetrics.__dataclass_fields__.values()})


# ── Singleton ─────────────────────────────────────────────────────────────────
_metrics_instance: FloodWatchMetrics | None = None
_lock = threading.Lock()


def get_metrics() -> FloodWatchMetrics:
    global _metrics_instance
    if _metrics_instance is None:
        with _lock:
            if _metrics_instance is None:
                _metrics_instance = _build_metrics(_get_registry())
    return _metrics_instance


# Convenience alias
METRICS = get_metrics()


def start_metrics_server(port: int = 9090):
    """Start a separate HTTP server exposing /metrics for Prometheus scraping."""
    try:
        from prometheus_client import start_http_server
        start_http_server(port, registry=_get_registry())
        log.info("Prometheus metrics server started on :%d/metrics", port)
    except ImportError:
        log.warning("prometheus_client not installed — metrics server not started")
    except Exception:
        log.exception("Failed to start metrics server on port %d", port)
