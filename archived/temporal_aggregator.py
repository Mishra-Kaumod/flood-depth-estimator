"""
Phase 2: Temporal Window Aggregator (5-image / 10-minute sliding window).

Stores per-camera frames in Redis sorted sets (score = Unix timestamp).
When >= BURST_THRESHOLD frames arrive within WINDOW_SECONDS, triggers
batch inference and returns rolling average depth + alert verdict.
"""
from __future__ import annotations

import json
import logging
import time
from statistics import mean, stdev

logger = logging.getLogger(__name__)

WINDOW_SECONDS = 600   # 10-minute sliding window
BURST_THRESHOLD = 5    # frames needed to trigger aggregation
REDIS_KEY_PREFIX = "flood:window:"
REDIS_TTL = WINDOW_SECONDS + 60  # a little beyond window for cleanup


def _get_redis():
    try:
        import redis
        return redis.Redis(host="localhost", port=6379, db=1, decode_responses=True)
    except Exception:
        return None


def push_frame(camera_id: str, depth_cm: float, confidence: float) -> dict | None:
    """
    Push a new depth reading for camera_id into the sliding window.
    Returns aggregated result dict when burst threshold is met, else None.
    """
    r = _get_redis()
    if r is None:
        return None

    now = time.time()
    key = REDIS_KEY_PREFIX + camera_id
    frame = json.dumps({"depth_cm": depth_cm, "confidence": confidence, "ts": now})

    pipe = r.pipeline()
    pipe.zadd(key, {frame: now})
    pipe.zremrangebyscore(key, "-inf", now - WINDOW_SECONDS)
    pipe.zcard(key)
    pipe.expire(key, REDIS_TTL)
    _, _, count, _ = pipe.execute()

    if count < BURST_THRESHOLD:
        return None

    # Read all frames in window
    raw_frames = r.zrangebyscore(key, now - WINDOW_SECONDS, now)
    frames = [json.loads(f) for f in raw_frames]
    depths = [f["depth_cm"] for f in frames]
    confidences = [f["confidence"] for f in frames]

    avg_depth_cm = mean(depths)
    avg_confidence = mean(confidences)
    depth_stdev = stdev(depths) if len(depths) > 1 else 0.0

    verdict = _verdict(avg_depth_cm)

    return {
        "camera_id": camera_id,
        "window_frame_count": len(frames),
        "window_seconds": WINDOW_SECONDS,
        "avg_depth_cm": round(avg_depth_cm, 2),
        "avg_depth_meters": round(avg_depth_cm / 100.0, 4),
        "depth_stdev_cm": round(depth_stdev, 2),
        "avg_confidence": round(avg_confidence, 4),
        "burst_trigger": True,
        "dynamic_next_action_trigger": verdict,
    }


def get_window_state(camera_id: str) -> dict:
    """Return current window state for a camera without triggering aggregation."""
    r = _get_redis()
    if r is None:
        return {"camera_id": camera_id, "error": "redis_unavailable"}

    now = time.time()
    key = REDIS_KEY_PREFIX + camera_id
    raw_frames = r.zrangebyscore(key, now - WINDOW_SECONDS, now)
    frames = [json.loads(f) for f in raw_frames]
    depths = [f["depth_cm"] for f in frames]

    return {
        "camera_id": camera_id,
        "frame_count": len(frames),
        "avg_depth_cm": round(mean(depths), 2) if depths else None,
        "burst_ready": len(frames) >= BURST_THRESHOLD,
    }


def _verdict(avg_depth_cm: float) -> str:
    if avg_depth_cm < 10:
        return "MONITOR"
    elif avg_depth_cm < 30:
        return "ADVISORY"
    elif avg_depth_cm < 60:
        return "WARNING"
    elif avg_depth_cm < 100:
        return "ALERT"
    else:
        return "CRITICAL_EVACUATE"
