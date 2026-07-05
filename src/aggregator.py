"""
src/aggregator.py
=================
Deliverable 1: Sensor Ingestion Schema & Dynamic Sliding-Window Aggregator.

Reads all tuneable parameters from config/config.yaml so no values are
hardcoded — swap the YAML, restart, behaviour changes.

Classes
-------
SensorPayload           Pydantic ingestion contract (validated at entry).
DepthEstimateResponse   Pydantic output contract (returned to caller).
SlidingWindowAggregator Per-camera 10-minute / 5-frame burst detector.

Usage
-----
    from src.aggregator import SlidingWindowAggregator, SensorPayload

    aggregator = SlidingWindowAggregator()
    result = aggregator.push(payload, depth_cm=42.3, confidence=0.87)
    if result:                         # burst triggered
        print(result.next_action_recommendation)
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# ── Config loading ─────────────────────────────────────────────────────────

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"


def _load_cfg() -> dict:
    with open(_CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _get(cfg: dict, *keys, default=None):
    """Safe nested key getter."""
    node = cfg
    for k in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(k, default)
    return node


_CFG = _load_cfg()

# Aggregator tuneable parameters (read from config; fall back to sane defaults)
_ENV = os.getenv("APP_ENV", "production")
WINDOW_SECONDS: int = int(_get(_CFG, "aggregator", "window_seconds", default=600))
BURST_THRESHOLD: int = int(_get(_CFG, "aggregator", "burst_threshold", default=5))
CONFIDENCE_THRESHOLD: float = float(
    _get(_CFG, "inference", "confidence_threshold", default=0.5)
)


# ── Data Contracts (Pydantic v2) ───────────────────────────────────────────

class SensorPayload(BaseModel):
    """Strict ingestion contract — every camera frame must satisfy this."""

    camera_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Unique sensor/camera identifier (e.g. 'CAM_KRM_001').",
    )
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    image: bytes = Field(..., description="Raw image bytes (JPEG / PNG).")

    @field_validator("image")
    @classmethod
    def image_not_empty(cls, v: bytes) -> bytes:
        if len(v) < 100:
            raise ValueError("image payload too small — minimum 100 bytes required.")
        return v

    @field_validator("camera_id")
    @classmethod
    def no_path_separators(cls, v: str) -> str:
        if "/" in v or "\\" in v:
            raise ValueError("camera_id must not contain path separators.")
        return v.strip()


class DepthEstimateResponse(BaseModel):
    """Output contract — all input metadata plus inference results."""

    # Pass-through from SensorPayload
    camera_id: str
    latitude: float
    longitude: float

    # Inference results
    estimated_flood_depth: float = Field(
        ..., ge=0.0, description="Averaged flood depth in metres."
    )
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    next_action_recommendation: str

    # Window metadata (informational)
    frame_count: int = Field(..., description="Frames in the current burst window.")
    window_seconds: int
    burst_triggered: bool = True


# ── Business Logic ─────────────────────────────────────────────────────────

def _recommend_action(depth_m: float) -> str:
    """
    Map averaged depth (metres) to a municipal action recommendation.
    Thresholds from config/config.yaml  aggregator.thresholds  (if present)
    or sensible Bengaluru-specific defaults.
    """
    thresholds = _get(_CFG, "aggregator", "thresholds") or {}
    critical  = float(thresholds.get("critical_m",  1.00))
    alert     = float(thresholds.get("alert_m",     0.60))
    warning   = float(thresholds.get("warning_m",   0.30))
    advisory  = float(thresholds.get("advisory_m",  0.10))

    if depth_m >= critical:
        return "DEPLOY_EMERGENCY_DIVERSION — depth exceeds 1 m, evacuate low-lying zones immediately."
    elif depth_m >= alert:
        return "ALERT_TRAFFIC_MANAGEMENT — depth 60–100 cm, close affected roads and reroute."
    elif depth_m >= warning:
        return "WARNING_PUMP_ACTIVATION — depth 30–60 cm, activate stormwater pumps."
    elif depth_m >= advisory:
        return "ADVISORY_INCREASE_MONITORING — depth 10–30 cm, heighten sensor polling frequency."
    else:
        return "MONITOR — depth below 10 cm, standard surveillance continues."


# ── Sliding Window Aggregator ──────────────────────────────────────────────

class _Frame:
    """Lightweight internal frame record."""
    __slots__ = ("ts", "depth_cm", "confidence")

    def __init__(self, depth_cm: float, confidence: float):
        self.ts = time.monotonic()
        self.depth_cm = depth_cm
        self.confidence = confidence


class SlidingWindowAggregator:
    """
    Per-camera 10-minute / 5-frame sliding-window aggregator.

    Thread-safety note: this implementation uses a plain Python list per
    camera which is safe for CPython's GIL.  Wrap with asyncio.Lock or
    threading.Lock if you deploy under multiple OS threads without Celery.

    Parameters (all read from config/config.yaml unless overridden):
        window_seconds   How far back to look (default 600 = 10 min).
        burst_threshold  Minimum frames to trigger aggregation (default 5).
    """

    def __init__(
        self,
        window_seconds: int = WINDOW_SECONDS,
        burst_threshold: int = BURST_THRESHOLD,
    ) -> None:
        self.window_seconds = window_seconds
        self.burst_threshold = burst_threshold
        self._windows: dict[str, List[_Frame]] = defaultdict(list)
        logger.info(
            "SlidingWindowAggregator ready — window=%ds burst_threshold=%d",
            window_seconds,
            burst_threshold,
        )

    # ── Public API ─────────────────────────────────────────────────────────

    def push(
        self,
        payload: SensorPayload,
        depth_cm: float,
        confidence: float,
    ) -> Optional[DepthEstimateResponse]:
        """
        Record a new inference result for this camera.

        Returns a DepthEstimateResponse when the burst threshold is met,
        None otherwise.
        """
        frames = self._windows[payload.camera_id]
        frames.append(_Frame(depth_cm=depth_cm, confidence=confidence))
        self._evict_stale(frames)

        logger.debug(
            "camera=%s frames_in_window=%d depth_cm=%.1f",
            payload.camera_id,
            len(frames),
            depth_cm,
        )

        if len(frames) >= self.burst_threshold:
            return self._aggregate(payload, frames)
        return None

    def window_state(self, camera_id: str) -> dict:
        """Return current window stats for a camera without triggering aggregation."""
        frames = self._windows.get(camera_id, [])
        self._evict_stale(frames)
        depths = [f.depth_cm for f in frames]
        return {
            "camera_id": camera_id,
            "frame_count": len(frames),
            "avg_depth_cm": round(sum(depths) / len(depths), 2) if depths else None,
            "burst_ready": len(frames) >= self.burst_threshold,
            "window_seconds": self.window_seconds,
        }

    # ── Internal helpers ───────────────────────────────────────────────────

    def _evict_stale(self, frames: List[_Frame]) -> None:
        cutoff = time.monotonic() - self.window_seconds
        while frames and frames[0].ts < cutoff:
            frames.pop(0)

    def _aggregate(
        self,
        payload: SensorPayload,
        frames: List[_Frame],
    ) -> DepthEstimateResponse:
        depths_m     = [f.depth_cm / 100.0 for f in frames]
        confidences  = [f.confidence for f in frames]
        avg_depth_m  = sum(depths_m) / len(depths_m)
        avg_conf     = sum(confidences) / len(confidences)
        recommendation = _recommend_action(avg_depth_m)

        logger.info(
            "BURST camera=%s frames=%d avg_depth=%.3fm conf=%.3f action=%s",
            payload.camera_id,
            len(frames),
            avg_depth_m,
            avg_conf,
            recommendation.split(" — ")[0],
        )

        return DepthEstimateResponse(
            camera_id=payload.camera_id,
            latitude=payload.latitude,
            longitude=payload.longitude,
            estimated_flood_depth=round(avg_depth_m, 4),
            confidence_score=round(avg_conf, 4),
            next_action_recommendation=recommendation,
            frame_count=len(frames),
            window_seconds=self.window_seconds,
        )
