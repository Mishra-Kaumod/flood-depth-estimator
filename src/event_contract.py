"""
Canonical event contracts shared across API and queue execution paths.
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
from typing import Any, Dict, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

SCHEMA_VERSION = "1.0"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class FloodEvent(BaseModel):
    schema_version: str = Field(default=SCHEMA_VERSION)
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    source: Literal["api", "queue", "serve"] = "api"
    timestamp: datetime = Field(default_factory=_utc_now)

    camera_id: str = Field(..., min_length=1, max_length=128)
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    image_b64: str = Field(..., description="Base64 encoded image bytes")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, v: str) -> str:
        if v != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema_version '{v}'. Expected '{SCHEMA_VERSION}'."
            )
        return v

    @field_validator("camera_id")
    @classmethod
    def validate_camera_id(cls, v: str) -> str:
        if "/" in v or "\\" in v:
            raise ValueError("camera_id must not contain path separators.")
        return v.strip()

    @field_validator("image_b64")
    @classmethod
    def validate_image_b64(cls, v: str) -> str:
        try:
            decoded = base64.b64decode(v, validate=True)
        except Exception as exc:
            raise ValueError(f"image_b64 must be valid base64: {exc}") from exc
        if len(decoded) < 100:
            raise ValueError("image payload too small (< 100 bytes)")
        return v

    def image_bytes(self) -> bytes:
        return base64.b64decode(self.image_b64)

    def to_task_payload(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")


class FloodResultEvent(BaseModel):
    schema_version: str = Field(default=SCHEMA_VERSION)
    event_id: str
    trace_id: str
    source: Literal["api", "queue", "serve"]
    timestamp: datetime
    processed_at: datetime = Field(default_factory=_utc_now)

    camera_id: str
    latitude: float
    longitude: float

    estimated_depth_meters: float = Field(..., ge=0.0)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    color_code: str
    action_trigger: str
    severity: int = Field(..., ge=1, le=5)
    severity_label: str
    method: str
    window_frame_count: int = Field(default=1, ge=1)

    status: Literal["success", "error"] = "success"
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_api_response(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")


class FloodFailureEvent(BaseModel):
    schema_version: str = Field(default=SCHEMA_VERSION)
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    source: Literal["api", "queue", "serve"] = "api"
    timestamp: datetime = Field(default_factory=_utc_now)
    failed_at: datetime = Field(default_factory=_utc_now)

    status: Literal["error"] = "error"
    stage: str
    error_type: str
    error_message: str
    retry_exhausted: bool = True
    attempts: int = Field(default=1, ge=1)
    max_attempts: int = Field(default=1, ge=1)

    camera_id: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    original_event: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_exception(
        cls,
        *,
        exc: Exception,
        stage: str,
        attempts: int,
        max_attempts: int,
        retry_exhausted: bool,
        event: FloodEvent | Dict[str, Any] | None = None,
        source: Literal["api", "queue", "serve"] = "api",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "FloodFailureEvent":
        base: Dict[str, Any] = {}
        if isinstance(event, FloodEvent):
            base = event.model_dump(mode="json")
        elif isinstance(event, dict):
            base = dict(event)

        return cls(
            schema_version=base.get("schema_version", SCHEMA_VERSION),
            event_id=base.get("event_id", str(uuid4())),
            trace_id=base.get("trace_id", str(uuid4())),
            source=base.get("source", source),
            timestamp=base.get("timestamp", _utc_now()),
            stage=stage,
            error_type=type(exc).__name__,
            error_message=str(exc),
            retry_exhausted=retry_exhausted,
            attempts=attempts,
            max_attempts=max_attempts,
            camera_id=base.get("camera_id"),
            latitude=base.get("latitude"),
            longitude=base.get("longitude"),
            original_event=base,
            metadata=metadata or {},
        )

    def to_api_response(self) -> Dict[str, Any]:
        return self.model_dump(mode="json")
