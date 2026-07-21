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
        Convert relative 0→1 depth to absolute centimetres.
        Returns (depth_map_cm, source_label, confidence).
        """
        best_obj   = None
        best_conf  = 0.0

        for obj in yolo_result.objects:
            if obj.confidence > best_conf and obj.pixel_height > 10:
                best_obj  = obj
                best_conf = obj.confidence

        if best_obj is not None and best_obj.pixel_height > 0:
            # pixels_per_cm = pixel_height / real_height_cm
            # depth_cm      = depth_rel * (real_height_cm / pixels_per_cm * scale)
            # Simple linear scale: 1 unit of relative depth ≈ sensor_height_cm
            scale   = self.sensor_height_cm   # cm per relative unit at mid-range
            src     = f"yolo_{best_obj.class_name}"
            conf    = min(best_obj.confidence, 1.0)
        else:
            scale   = self.sensor_height_cm
            src     = "fallback"
            conf    = 0.3

        depth_map_cm = (depth_map_rel * scale).astype(np.float32)
        return depth_map_cm, src, round(conf, 3)
