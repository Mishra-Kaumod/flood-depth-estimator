# pipeline/fusion.py
"""
Stage 4 — Fusion Engine
=========================
Inputs : SegFormerResult + YOLOResult + DepthResult + raw image
Output : StructuredFeatures (rich explainable dict for severity model)

Key job:
  1. Use YOLO reference objects to calibrate relative depth → absolute cm
  2. Intersect water mask with depth map to get flood depth at water pixels
  3. Compute explainable features (water %, mean depth, max depth, area, etc.)
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List

from .segformer import SegFormerResult
from .yolo      import YOLOResult
from .depth     import DepthResult

log = logging.getLogger("pipeline.fusion")

# Fallback pixels-per-metre if no YOLO reference found
_FALLBACK_PX_PER_METRE = 120.0


@dataclass
class StructuredFeatures:
    """
    Explainable intermediate features — input to SeverityStage.
    Every value here has a physical meaning, unlike raw pixel values.
    """
    # Water extent
    water_coverage_pct:    float    # % of frame covered by water
    water_pixel_count:     int

    # Depth (in absolute cm, calibrated via YOLO reference)
    mean_flood_depth_cm:   float
    max_flood_depth_cm:    float
    p90_flood_depth_cm:    float    # 90th-percentile depth (robust max)

    # Calibration quality
    calibration_source:    str      # "yolo_<class>" | "fallback"
    calibration_confidence: float   # 0-1

    # Engines used
    seg_engine:            str
    yolo_engine:           str
    depth_engine:          str

    # Raw maps (passed through for visualisation, not fed to severity model)
    water_mask:            np.ndarray = field(repr=False)
    depth_map_cm:          np.ndarray = field(repr=False)   # absolute cm


class FusionStage:

    def __init__(self, sensor_height_cm: float = 300.0):
        """
        sensor_height_cm: approximate camera mounting height above ground.
        Used as secondary calibration fallback.
        """
        self.sensor_height_cm = sensor_height_cm

    def fuse(
        self,
        image_bgr:    np.ndarray,
        seg_result:   SegFormerResult,
        yolo_result:  YOLOResult,
        depth_result: DepthResult,
    ) -> StructuredFeatures:

        depth_map_rel = depth_result.depth_map   # H×W float32 0→1

        # ── Step 1: calibrate depth map to absolute cm ───────────────────────
        depth_map_cm, cal_source, cal_conf = self._calibrate(
            depth_map_rel, yolo_result, image_bgr.shape
        )

        # ── Step 2: extract flood depth only where water mask is True ────────
        water_mask = seg_result.water_mask
        water_depths = depth_map_cm[water_mask]

        if water_depths.size == 0:
            mean_d = max_d = p90_d = 0.0
        else:
            mean_d = float(np.mean(water_depths))
            max_d  = float(np.max(water_depths))
            p90_d  = float(np.percentile(water_depths, 90))

        return StructuredFeatures(
            water_coverage_pct     = seg_result.water_coverage_pct,
            water_pixel_count      = int(water_mask.sum()),
            mean_flood_depth_cm    = round(mean_d, 2),
            max_flood_depth_cm     = round(max_d,  2),
            p90_flood_depth_cm     = round(p90_d,  2),
            calibration_source     = cal_source,
            calibration_confidence = cal_conf,
            seg_engine             = seg_result.engine,
            yolo_engine            = yolo_result.engine,
            depth_engine           = depth_result.engine,
            water_mask             = water_mask,
            depth_map_cm           = depth_map_cm,
        )

    # ── Calibration ──────────────────────────────────────────────────────────
    def _calibrate(
        self,
        depth_map_rel: np.ndarray,
        yolo_result:   YOLOResult,
        img_shape:     tuple,
    ) -> tuple[np.ndarray, str, float]:
        """
        Convert relative 0→1 depth map to absolute centimetres.
        Returns (depth_map_cm, source_label, confidence 0-1).

        Math (YOLO-found branch):
        ─────────────────────────
        Given a reference object with known real-world height H_cm and
        apparent pixel height P_px:

          px_per_cm  = P_px / H_cm          # pixels per cm at the object's depth

        We assume a pinhole camera with focal length ≈ image_height pixels
        (typical for CCTV lenses), giving us the object's metric depth:

          depth_at_obj_cm = image_height * H_cm / P_px   (i.e. image_h / px_per_cm)

        The depth model's relative map is sampled inside the object's bounding
        box to get rel_at_obj (average relative depth at that location). This
        anchors the scale factor:

          scale = depth_at_obj_cm / rel_at_obj

        So that depth_map_cm = depth_map_rel × scale satisfies:
          depth_map_cm[bbox] ≈ depth_at_obj_cm ✓

        Fallback branch (no reference object):
          scale = sensor_height_cm  (flat constant, low-confidence estimate)
        """
        best_obj  = None
        best_conf = 0.0

        for obj in yolo_result.objects:
            if obj.confidence > best_conf and obj.pixel_height > 10:
                best_obj  = obj
                best_conf = obj.confidence

        if best_obj is not None and best_obj.pixel_height > 0:
            # ── Reference-object calibration ─────────────────────────────────
            px_per_cm = best_obj.pixel_height / best_obj.real_height_cm

            # Sample mean relative depth inside the reference object's bbox
            x1, y1, x2, y2 = best_obj.bbox_xyxy
            img_h = depth_map_rel.shape[0]
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(depth_map_rel.shape[1], x2); y2 = min(img_h, y2)
            roi = depth_map_rel[y1:y2, x1:x2]
            rel_at_obj = float(np.mean(roi)) if roi.size > 0 else 0.5

            # Perspective estimate: distance to object in cm
            depth_at_obj_cm = img_h / px_per_cm   # = img_h * real_h / pixel_h

            # Scale so that depth_map_cm at the object equals depth_at_obj_cm
            scale = depth_at_obj_cm / max(rel_at_obj, 1e-6)

            src  = f"yolo_{best_obj.class_name}"
            conf = min(best_obj.confidence, 1.0)
        else:
            # ── Fallback: use sensor mounting height as scale constant ───────
            scale = self.sensor_height_cm
            src   = "fallback"
            conf  = 0.3

        depth_map_cm = (depth_map_rel * scale).astype(np.float32)
        return depth_map_cm, src, round(conf, 3)
