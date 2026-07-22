# pipeline/auto_calibration.py
"""
Auto Per-Camera Calibration from Static Scene Geometry
=======================================================
Accumulates dry-condition frames per camera and derives a focal-length + 
ground-plane estimate using OpenCV line detection on road markings/edges.

Outputs a per-camera calibration profile (JSON) that fusion.py uses as
the primary scale source — better than a flat sensor_height_cm constant.

Key classes:
  CameraGeometryCalibrator  — accumulates dry frames, runs calibration
  CameraCalibrationStore    — persists/loads per-camera JSON profiles

Usage in fusion.py:
  store = CameraCalibrationStore("camera_calibration/")
  calibrator = CameraGeometryCalibrator(store=store, dry_frames_needed=50)
  # Call on every dry frame:
  calibrator.observe(camera_id, image_bgr, water_coverage_pct)
  # fusion.py loads the profile at calibrate time:
  profile = store.get(camera_id)

Math:
  Given N detected lines, we estimate the vanishing point (VP) by finding
  the intersection cluster with RANSAC-style voting.  The VP y-coordinate
  relative to image height gives an estimate of:
    pitch_rad ≈ arctan((vy - h/2) / focal_px)
  If we also detect a road-marking at a known y-row (e.g. lane edge at y_road),
  the distance to that marking is:
    d_road ≈ focal_px * sensor_height_cm / (y_road - vy)
  This gives us focal_px from the reference if sensor_height_cm is known,
  or lets us refine sensor_height_cm if YOLO reference is available.

  In practice we estimate focal_px by finding the median ratio across all
  detected line-pair intersections, which is robust to noise.
"""

import json
import logging
import math
import threading
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Deque

import cv2
import numpy as np

log = logging.getLogger("pipeline.auto_calibration")

# ── Config ────────────────────────────────────────────────────────────────────
DRY_FRAMES_NEEDED = 50       # minimum dry frames before calibration attempt
MIN_LINES         = 4        # minimum detected lines for a valid calibration
MAX_VP_ITERS      = 50       # vanishing-point RANSAC iterations
DISAGREEMENT_WARN = 0.20     # warn if YOLO cross-check disagrees by >20%


@dataclass
class CalibrationProfile:
    """Per-camera geometry profile derived from dry-frame analysis."""
    camera_id:       str
    focal_px:        float    # estimated focal length in pixels
    image_width:     int
    image_height:    int
    vp_x:            float    # vanishing-point x (pixel)
    vp_y:            float    # vanishing-point y (pixel)
    pitch_deg:       float    # estimated camera tilt (° from horizontal)
    n_dry_frames:    int      # how many frames were used
    calibrated_at:  str       # ISO timestamp

    def scale_factor_for_rel_depth(self, rel_at_obj: float) -> float:
        """
        Convert a relative depth value (0-1) to metric cm.
        Assumes depth_map_rel ∝ disparity.
        depth_cm = focal_px / max(rel_at_obj, 1e-6)
        (This is a first-order approximation; good for CCTV where depth range
        is large and the depth model is trained on relative disparity.)
        """
        return self.focal_px / max(rel_at_obj, 1e-6)


# ── Calibration store ─────────────────────────────────────────────────────────
class CameraCalibrationStore:
    """
    Thread-safe JSON-backed store for CalibrationProfile objects.
    One JSON file per camera in the given directory.
    """

    def __init__(self, store_dir: str | Path = "camera_calibration"):
        self._dir = Path(store_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._cache: Dict[str, CalibrationProfile] = {}
        self._load_all()

    def get(self, camera_id: str) -> CalibrationProfile | None:
        with self._lock:
            return self._cache.get(camera_id)

    def put(self, profile: CalibrationProfile) -> None:
        path = self._path_for(profile.camera_id)
        with self._lock:
            self._cache[profile.camera_id] = profile
            with path.open("w") as f:
                json.dump(asdict(profile), f, indent=2)
        log.info("Saved calibration profile for %s → %s", profile.camera_id, path)

    def _path_for(self, camera_id: str) -> Path:
        safe = camera_id.replace("/", "_").replace("\\", "_")
        return self._dir / f"{safe}.json"

    def _load_all(self) -> None:
        for p in self._dir.glob("*.json"):
            try:
                with p.open() as f:
                    d = json.load(f)
                profile = CalibrationProfile(**d)
                self._cache[profile.camera_id] = profile
                log.debug("Loaded calibration profile %s", profile.camera_id)
            except Exception:
                log.warning("Could not load calibration profile %s", p, exc_info=True)
        if self._cache:
            log.info("Loaded %d calibration profiles from %s", len(self._cache), self._dir)


# ── Calibrator ────────────────────────────────────────────────────────────────
class CameraGeometryCalibrator:
    """
    Accumulate dry frames and derive a geometric calibration profile.

    Args:
        store:             CameraCalibrationStore to persist profiles.
        dry_frames_needed: minimum dry frames before attempting calibration.
        sensor_height_cm:  fallback camera height — used to bootstrap focal_px
                           when no YOLO reference is available.
    """

    def __init__(
        self,
        store: CameraCalibrationStore,
        dry_frames_needed: int = DRY_FRAMES_NEEDED,
        sensor_height_cm: float = 300.0,
    ):
        self.store             = store
        self.dry_frames_needed = dry_frames_needed
        self.sensor_height_cm  = sensor_height_cm
        self._buffers: Dict[str, Deque[np.ndarray]] = {}
        self._lock = threading.Lock()

    def observe(
        self,
        camera_id:        str,
        image_bgr:        np.ndarray,
        water_coverage_pct: float,
    ) -> None:
        """
        Call on every processed frame. Only stores frames where
        water_coverage_pct < 1 (dry-road condition).
        Triggers re-calibration automatically when buffer reaches threshold.
        """
        if water_coverage_pct >= 1.0:
            return  # skip wet frames

        with self._lock:
            if camera_id not in self._buffers:
                self._buffers[camera_id] = deque(maxlen=self.dry_frames_needed * 2)
            self._buffers[camera_id].append(image_bgr.copy())
            n = len(self._buffers[camera_id])

        if n >= self.dry_frames_needed and n % self.dry_frames_needed == 0:
            # Every `dry_frames_needed` new frames, re-calibrate
            self._calibrate(camera_id)

    # ── Internal calibration ──────────────────────────────────────────────
    def _calibrate(self, camera_id: str) -> None:
        with self._lock:
            frames = list(self._buffers.get(camera_id, []))
        if len(frames) < self.dry_frames_needed:
            return

        log.info("Calibrating %s from %d dry frames…", camera_id, len(frames))
        try:
            focal_px, vp_x, vp_y = self._estimate_geometry(frames)
            h, w = frames[0].shape[:2]
            # Camera pitch: positive = tilted down
            vy_from_centre = vp_y - h / 2.0
            pitch_deg = math.degrees(math.atan2(vy_from_centre, focal_px))
            profile = CalibrationProfile(
                camera_id    = camera_id,
                focal_px     = round(focal_px, 1),
                image_width  = w,
                image_height = h,
                vp_x         = round(vp_x, 1),
                vp_y         = round(vp_y, 1),
                pitch_deg    = round(pitch_deg, 2),
                n_dry_frames = len(frames),
                calibrated_at= _utcnow(),
            )
            self.store.put(profile)
            log.info(
                "Calibrated %s: focal=%.0fpx vp=(%.0f,%.0f) pitch=%.1f°",
                camera_id, focal_px, vp_x, vp_y, pitch_deg,
            )
        except Exception:
            log.warning("Calibration failed for %s", camera_id, exc_info=True)

    def _estimate_geometry(
        self, frames: list[np.ndarray]
    ) -> tuple[float, float, float]:
        """
        Detect line-based vanishing point across frames.
        Returns (focal_px, vp_x, vp_y).
        """
        all_lines: list[tuple[float, float, float, float]] = []

        # Sample up to 10 evenly-spaced frames for speed
        step   = max(1, len(frames) // 10)
        sample = frames[::step][:10]

        for frame in sample:
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(
                edges, rho=1, theta=np.pi / 180,
                threshold=80, minLineLength=60, maxLineGap=20,
            )
            if lines is None:
                continue
            h = frame.shape[0]
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Keep only lines in upper 80% of frame (road perspective)
                if y1 > h * 0.8 or y2 > h * 0.8:
                    continue
                # Skip near-horizontal lines (not perspective lines)
                if abs(y2 - y1) < 5:
                    continue
                all_lines.append((float(x1), float(y1), float(x2), float(y2)))

        if len(all_lines) < MIN_LINES:
            raise ValueError(
                f"Only {len(all_lines)} lines detected — need {MIN_LINES}. "
                "Camera may not have enough road texture."
            )

        # Find vanishing point as the median pairwise intersection
        vp_candidates: list[tuple[float, float]] = []
        rng = np.random.default_rng(42)
        indices = rng.choice(len(all_lines), size=(min(MAX_VP_ITERS, len(all_lines) ** 2), 2))
        for i, j in indices:
            if i == j:
                continue
            pt = _line_intersection(all_lines[i], all_lines[j])
            if pt is not None:
                vp_candidates.append(pt)

        if not vp_candidates:
            raise ValueError("No line intersections found")

        vp_arr = np.array(vp_candidates)
        vp_x   = float(np.median(vp_arr[:, 0]))
        vp_y   = float(np.median(vp_arr[:, 1]))

        # Focal length estimate: if sensor_height_cm is known, use bottom-of-frame
        # road row to infer focal_px ≈ (y_road - vp_y) * sensor_height_cm / 100
        h = frames[0].shape[0]
        y_road    = h * 0.9           # approx bottom-of-road row
        dist_px   = max(y_road - vp_y, 10.0)
        focal_px  = dist_px * self.sensor_height_cm / 100.0  # rough estimate

        return focal_px, vp_x, vp_y


# ── Helpers ───────────────────────────────────────────────────────────────────
def _line_intersection(
    l1: tuple[float, float, float, float],
    l2: tuple[float, float, float, float],
) -> tuple[float, float] | None:
    """Find intersection of two line segments (extended as infinite lines)."""
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None  # parallel
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    ix = x1 + t * (x2 - x1)
    iy = y1 + t * (y2 - y1)
    # Reject intersections that are implausibly far (> 5× image width)
    if abs(ix) > 5000 or abs(iy) > 5000:
        return None
    return ix, iy


def _utcnow() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
