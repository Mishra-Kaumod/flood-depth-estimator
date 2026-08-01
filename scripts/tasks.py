"""
Celery queue adapter for the shared flood event pipeline.
"""

from __future__ import annotations

import logging

from celery import Celery

from src.dlq import get_dead_letter_router
from src.event_contract import FloodEvent, FloodFailureEvent
from src.middleware.retry import RetryPolicy, is_transient_error
from src.pipeline import execute_event

logger = logging.getLogger(__name__)

REDIS_URL = "redis://localhost:6379/0"

celery_app = Celery("flood_tasks", broker=REDIS_URL, backend=REDIS_URL)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_routes={"tasks.infer_flood_depth": {"queue": "flood_inference"}},
    task_soft_time_limit=25,
    task_time_limit=30,
    worker_prefetch_multiplier=1,
)


def _coerce_legacy_payload(event_payload, legacy_args):
    """
    Backward compatibility for old signature:
      infer_flood_depth(camera_id, image_b64, latitude, longitude)
    """
    if isinstance(event_payload, dict):
        return event_payload
    if len(legacy_args) < 3:
        raise ValueError("Legacy task payload requires camera_id, image_b64, latitude, longitude")
    return {
        "source": "queue",
        "camera_id": event_payload,
        "image_b64": legacy_args[0],
        "latitude": legacy_args[1],
        "longitude": legacy_args[2],
    }


@celery_app.task(bind=True, max_retries=3, soft_time_limit=25)
def infer_flood_depth(self, event_payload, *legacy_args):
    payload = _coerce_legacy_payload(event_payload, legacy_args)
    payload["source"] = "queue"
    dlq = get_dead_letter_router()
    max_attempts = int(self.max_retries) + 1

    try:
        event = FloodEvent.model_validate(payload)
    except Exception as exc:
        failure = FloodFailureEvent.from_exception(
            exc=exc,
            stage="queue.validate",
            attempts=1,
            max_attempts=max_attempts,
            retry_exhausted=True,
            event=payload,
            source="queue",
            metadata={"adapter": "tasks.infer_flood_depth"},
        )
        dlq_info = dlq.publish(failure)
        logger.error("DLQ publish for validation failure: %s", dlq_info)
        raise

    retry_policy = RetryPolicy(max_attempts=3, base_delay_seconds=0.5, max_delay_seconds=6.0, jitter_seconds=0.2)
    try:
        result = execute_event(event, retry_policy=retry_policy)
        return result.model_dump(mode="json")
    except Exception as exc:
        # Celery-level retry remains only for transient infrastructure errors.
        if is_transient_error(exc) and self.request.retries < self.max_retries:
            countdown = min(2 ** self.request.retries, 8)
            logger.warning(
                "Transient failure in queue adapter; retrying event=%s retries=%d countdown=%ds error=%s",
                event.event_id,
                self.request.retries + 1,
                countdown,
                exc,
            )
            raise self.retry(exc=exc, countdown=countdown)

        attempts = int(self.request.retries) + 1
        failure = FloodFailureEvent.from_exception(
            exc=exc,
            stage="queue.execute",
            attempts=attempts,
            max_attempts=max_attempts,
            retry_exhausted=True,
            event=event,
            source="queue",
            metadata={"adapter": "tasks.infer_flood_depth"},
        )
        dlq_info = dlq.publish(failure)
        logger.error(
            "Permanent queue failure routed to DLQ event=%s trace=%s dlq=%s",
            failure.event_id,
            failure.trace_id,
            dlq_info,
        )
        raise
