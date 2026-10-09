"""Current V6 event adapter for API/queue/EC2 execution.

This module handles event metadata, retries and observability only. It has no
model loading, preprocessing, legacy fallback, depth corrections or averaging.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from typing import Any

from src.event_contract import FloodEvent
from src.middleware.observability import observe_execution
from src.middleware.retry import RetryPolicy, run_with_retry
from src.settings import load_settings_dict
from src.v6_inference import create_v6_pipeline, load_v6_rgb, v6_depth_payload

logger = logging.getLogger(__name__)
_PROCESSOR = None


@dataclass(frozen=True)
class V6EventResult:
    payload: dict[str, Any]

    def to_api_response(self) -> dict[str, Any]:
        return dict(self.payload)


class UnifiedEventProcessor:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_settings_dict(config_path=config_path)
        self.pipeline = create_v6_pipeline()

    def process_event(self, event: FloodEvent) -> V6EventResult:
        rgb = load_v6_rgb(event.image_bytes())
        result = self.pipeline.predict(rgb)
        from src.v6_application_review import review_v6_result
        prediction = review_v6_result(result, event.image_bytes(), event.metadata.get("filename", "camera_upload.jpg"), self.pipeline, config=self.config)
        return V6EventResult({
            **prediction,
            "architecture": "V6",
            "event_id": event.event_id,
            "trace_id": event.trace_id,
            "source": event.source,
            "timestamp": event.timestamp.isoformat(),
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "camera_id": event.camera_id,
            "latitude": event.latitude,
            "longitude": event.longitude,
            "status": "success" if prediction["final_shadow_depth_cm"] is not None else "unavailable",
            "metadata": event.metadata,
        })


def get_processor() -> UnifiedEventProcessor:
    global _PROCESSOR
    if _PROCESSOR is None:
        _PROCESSOR = UnifiedEventProcessor()
    return _PROCESSOR


def execute_event(event: FloodEvent, retry_policy: RetryPolicy | None = None) -> V6EventResult:
    processor = get_processor()
    cfg = processor.config.get("event_processing", {}).get("retry", {})
    policy = retry_policy or RetryPolicy(
        max_attempts=int(cfg.get("max_attempts", 3)),
        base_delay_seconds=float(cfg.get("base_delay_seconds", 0.5)),
        max_delay_seconds=float(cfg.get("max_delay_seconds", 8.0)),
        jitter_seconds=float(cfg.get("jitter_seconds", 0.25)),
    )

    def operation(attempt: int) -> V6EventResult:
        return observe_execution(
            event_id=event.event_id, trace_id=event.trace_id, camera_id=event.camera_id,
            source=event.source, stage="v6_inference", attempt=attempt,
            operation=lambda: processor.process_event(event),
        )

    return run_with_retry(
        operation=operation, policy=policy,
        on_retry=lambda attempt, delay, exc: logger.warning(
            "retrying V6 event=%s attempt=%d delay=%.2fs error=%s",
            event.event_id, attempt + 1, delay, exc,
        ),
    )
