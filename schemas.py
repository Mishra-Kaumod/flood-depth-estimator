"""
Phase 4: Strict Pydantic data contracts for ingestion and response payloads.
"""
from __future__ import annotations

import base64
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class IngestPayload(BaseModel):
    camera_id: str = Field(..., min_length=1, max_length=128, description="Unique camera/sensor identifier")
    latitude: float = Field(..., ge=-90.0, le=90.0, description="GPS latitude")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="GPS longitude")
    image: str = Field(..., description="Base64-encoded image bytes (JPEG/PNG)")

    @field_validator("image")
    @classmethod
    def validate_image_b64(cls, v: str) -> str:
        try:
            decoded = base64.b64decode(v, validate=True)
        except Exception as exc:
            raise ValueError(f"image must be valid base64: {exc}") from exc
        if len(decoded) < 100:
            raise ValueError("image payload too small (< 100 bytes)")
        return v

    @field_validator("camera_id")
    @classmethod
    def camera_id_no_slashes(cls, v: str) -> str:
        if "/" in v or "\\" in v:
            raise ValueError("camera_id must not contain slashes")
        return v.strip()


class SeverityDetail(BaseModel):
    level: str
    label: str
    color: str
    action: str


class DepthEstimateResponse(BaseModel):
    camera_id: str
    latitude: float
    longitude: float
    estimated_depth_meters: float = Field(..., ge=0.0)
    depth_cm: float = Field(..., ge=0.0)
    model_confidence_score: float = Field(..., ge=0.0, le=1.0)
    dynamic_next_action_trigger: str
    severity: Optional[SeverityDetail] = None
    method: str
    temporal_window: Optional[dict] = None

    @model_validator(mode="after")
    def depth_consistency(self) -> "DepthEstimateResponse":
        expected = round(self.depth_cm / 100.0, 4)
        if abs(self.estimated_depth_meters - expected) > 0.01:
            self.estimated_depth_meters = expected
        return self
