"""
Sensor ingestion contracts + sliding window aggregator (memory/Redis).

Supports:
1. Strict Pydantic ingestion and output contracts.
2. 5-image / 10-minute sliding window burst logic per camera_id.
3. Event-time processing for out-of-order arrivals.
4. Durable Redis-backed windows for production.
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from src.settings import load_settings_dict

logger = logging.getLogger(__name__)

_CFG = load_settings_dict()
WINDOW_SECONDS: int = int(_CFG.get("aggregator", {}).get("window_seconds", 600))
BURST_THRESHOLD: int = int(_CFG.get("aggregator", {}).get("burst_threshold", 5))
BACKEND: str = str(_CFG.get("aggregator", {}).get("backend", "memory")).lower()
REDIS_URL: str = str(_CFG.get("aggregator", {}).get("redis_url", "redis://localhost:6379/1"))


class SensorPayload(BaseModel):
    camera_id: str = Field(..., min_length=1, max_length=128)
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    image: bytes = Field(..., description="Raw image bytes.")

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
    camera_id: str
    latitude: float
    longitude: float
    estimated_flood_depth: float = Field(..., ge=0.0, description="Averaged depth in metres.")
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    next_action_recommendation: str
    frame_count: int
    window_seconds: int
    burst_triggered: bool = True


def _thresholds() -> dict:
    return _CFG.get("aggregator", {}).get("thresholds", {})


def _recommend_action(depth_m: float) -> str:
    thresholds = _thresholds()
    critical = float(thresholds.get("critical_m", 1.00))
    alert = float(thresholds.get("alert_m", 0.60))
    warning = float(thresholds.get("warning_m", 0.30))
    advisory = float(thresholds.get("advisory_m", 0.10))

    if depth_m >= critical:
        return "DEPLOY_EMERGENCY_DIVERSION"
    if depth_m >= alert:
        return "ALERT_TRAFFIC_MANAGEMENT"
    if depth_m >= warning:
        return "WARNING_PUMP_ACTIVATION"
    if depth_m >= advisory:
        return "ADVISORY_INCREASE_MONITORING"
    return "MONITOR"


def _parse_event_ts(event_ts: Optional[str | float | int]) -> float:
    if event_ts is None:
        return time.time()
    if isinstance(event_ts, (int, float)):
        return float(event_ts)
    ts = event_ts.strip()
    try:
        return float(ts)
    except ValueError:
        if ts.endswith("Z"):
            ts = ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()


class _Frame:
    __slots__ = ("event_ts", "depth_cm", "confidence")

    def __init__(self, event_ts: float, depth_cm: float, confidence: float):
        self.event_ts = event_ts
        self.depth_cm = depth_cm
        self.confidence = confidence


class SlidingWindowAggregator:
    def __init__(
        self,
        window_seconds: int = WINDOW_SECONDS,
        burst_threshold: int = BURST_THRESHOLD,
        backend: str = BACKEND,
        redis_url: str = REDIS_URL,
    ) -> None:
        self.window_seconds = window_seconds
        self.burst_threshold = burst_threshold
        self.backend = backend
        self._windows: dict[str, List[_Frame]] = defaultdict(list)
        self._redis = None

        if self.backend == "redis":
            try:
                import redis

                self._redis = redis.Redis.from_url(redis_url, decode_responses=True)
                self._redis.ping()
                logger.info("SlidingWindowAggregator using Redis backend at %s", redis_url)
            except Exception as e:
                logger.warning("Redis unavailable (%s). Falling back to memory backend.", e)
                self.backend = "memory"

        logger.info(
            "SlidingWindowAggregator ready — backend=%s window=%ds threshold=%d",
            self.backend,
            self.window_seconds,
            self.burst_threshold,
        )

    def push(
        self,
        payload: SensorPayload,
        depth_cm: float,
        confidence: float,
        event_ts: Optional[str | float | int] = None,
    ) -> Optional[DepthEstimateResponse]:
        ts = _parse_event_ts(event_ts)
        if self.backend == "redis" and self._redis is not None:
            return self._push_redis(payload, depth_cm, confidence, ts)
        return self._push_memory(payload, depth_cm, confidence, ts)

    def window_state(self, camera_id: str) -> dict:
        if self.backend == "redis" and self._redis is not None:
            return self._window_state_redis(camera_id)
        return self._window_state_memory(camera_id)

    def _push_memory(
        self,
        payload: SensorPayload,
        depth_cm: float,
        confidence: float,
        ts: float,
    ) -> Optional[DepthEstimateResponse]:
        frames = self._windows[payload.camera_id]
        frames.append(_Frame(event_ts=ts, depth_cm=depth_cm, confidence=confidence))
        self._evict_stale_memory(frames, now=ts)
        frames.sort(key=lambda f: f.event_ts)

        if len(frames) < self.burst_threshold:
            return None

        depths_cm = [f.depth_cm for f in frames]
        confidences = [f.confidence for f in frames]
        return self._build_response(payload, depths_cm, confidences)

    def _push_redis(
        self,
        payload: SensorPayload,
        depth_cm: float,
        confidence: float,
        ts: float,
    ) -> Optional[DepthEstimateResponse]:
        key = f"flood:agg:{payload.camera_id}"
        record = json.dumps({"depth_cm": depth_cm, "confidence": confidence, "event_ts": ts})

        pipe = self._redis.pipeline()
        pipe.zadd(key, {record: ts})
        pipe.zremrangebyscore(key, "-inf", ts - self.window_seconds)
        pipe.zcard(key)
        pipe.expire(key, self.window_seconds + 120)
        _, _, count, _ = pipe.execute()

        if count < self.burst_threshold:
            return None

        raw_frames = self._redis.zrangebyscore(key, ts - self.window_seconds, ts)
        decoded = [json.loads(item) for item in raw_frames]
        depths_cm = [float(item["depth_cm"]) for item in decoded]
        confidences = [float(item["confidence"]) for item in decoded]
        return self._build_response(payload, depths_cm, confidences)

    def _window_state_memory(self, camera_id: str) -> dict:
        frames = self._windows.get(camera_id, [])
        now = time.time()
        self._evict_stale_memory(frames, now=now)
        depths = [f.depth_cm for f in frames]
        return {
            "camera_id": camera_id,
            "frame_count": len(frames),
            "avg_depth_cm": round(sum(depths) / len(depths), 2) if depths else None,
            "burst_ready": len(frames) >= self.burst_threshold,
            "window_seconds": self.window_seconds,
            "backend": "memory",
        }

    def _window_state_redis(self, camera_id: str) -> dict:
        now = time.time()
        key = f"flood:agg:{camera_id}"
        raw_frames = self._redis.zrangebyscore(key, now - self.window_seconds, now)
        decoded = [json.loads(item) for item in raw_frames]
        depths = [float(item["depth_cm"]) for item in decoded]
        return {
            "camera_id": camera_id,
            "frame_count": len(decoded),
            "avg_depth_cm": round(sum(depths) / len(depths), 2) if depths else None,
            "burst_ready": len(decoded) >= self.burst_threshold,
            "window_seconds": self.window_seconds,
            "backend": "redis",
        }

    def _evict_stale_memory(self, frames: List[_Frame], now: float) -> None:
        cutoff = now - self.window_seconds
        while frames and frames[0].event_ts < cutoff:
            frames.pop(0)

    def _build_response(
        self,
        payload: SensorPayload,
        depths_cm: List[float],
        confidences: List[float],
    ) -> DepthEstimateResponse:
        avg_depth_m = (sum(depths_cm) / len(depths_cm)) / 100.0
        avg_conf = sum(confidences) / len(confidences)
        action = _recommend_action(avg_depth_m)
        logger.info(
            "BURST camera=%s frames=%d avg_depth=%.3fm conf=%.3f action=%s",
            payload.camera_id,
            len(depths_cm),
            avg_depth_m,
            avg_conf,
            action,
        )
        return DepthEstimateResponse(
            camera_id=payload.camera_id,
            latitude=payload.latitude,
            longitude=payload.longitude,
            estimated_flood_depth=round(avg_depth_m, 4),
            confidence_score=round(avg_conf, 4),
            next_action_recommendation=action,
            frame_count=len(depths_cm),
            window_seconds=self.window_seconds,
        )
