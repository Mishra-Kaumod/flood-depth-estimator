"""
Worker adapter migrated from the old Celery task pattern.

The callable functions are plain Python so Lambda/SQS can reuse them. If Celery
is installed, `celery_app` and task wrappers are also available.
"""

from __future__ import annotations

import base64
import os
from typing import Any

from src.api_service import FloodApiService


def process_camera_event(payload: dict[str, Any]) -> dict[str, Any]:
    image_b64 = payload.get("image_b64")
    if not image_b64:
        raise ValueError("image_b64 is required")

    service = FloodApiService()
    return service.process_camera_upload(
        image_bytes=base64.b64decode(image_b64, validate=True),
        filename=payload.get("filename", "camera_upload.jpg"),
        camera_id=payload.get("camera_id", "intersection_01"),
        latitude=float(payload["latitude"]),
        longitude=float(payload["longitude"]),
        location_name=payload.get("location_name"),
        metadata=payload.get("metadata") or {},
    )


def analyze_temporal_sequence(camera_id: str, time_window_minutes: int = 15) -> dict[str, Any]:
    return FloodApiService().trigger_temporal_analysis(camera_id, time_window_minutes)


try:
    from celery import Celery

    celery_app = Celery("flood_project_cleaned")
    celery_app.conf.broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    celery_app.conf.result_backend = os.getenv("CELERY_RESULT_BACKEND", celery_app.conf.broker_url)

    @celery_app.task(bind=True, max_retries=3)
    def process_camera_event_task(self, payload: dict[str, Any]) -> dict[str, Any]:
        return process_camera_event(payload)

    @celery_app.task(bind=True, max_retries=3)
    def analyze_temporal_sequence_task(
        self,
        camera_id: str,
        time_window_minutes: int = 15,
    ) -> dict[str, Any]:
        return analyze_temporal_sequence(camera_id, time_window_minutes)

except Exception:
    celery_app = None
