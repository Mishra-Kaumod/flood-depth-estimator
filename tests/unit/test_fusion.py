# tests/unit/test_fusion.py
"""
Unit tests for FusionStage — depth calibration and feature extraction.
Run: pytest tests/unit/test_fusion.py -v
"""

import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline.fusion    import FusionStage, StructuredFeatures
from pipeline.segformer import SegFormerResult
from pipeline.yolo      import YOLOResult, ReferenceObject
from pipeline.depth     import DepthResult


# ── Fixtures ──────────────────────────────────────────────────────────────────
def make_image(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)

def make_seg(water_pct=30.0, h=480, w=640):
    mask = np.zeros((h, w), dtype=bool)
    # Mark bottom 30% as water
    water_rows = int(h * water_pct / 100)
    mask[-water_rows:, :] = True
    return SegFormerResult(water_mask=mask, water_coverage_pct=water_pct, engine="test")

def make_yolo_with_car(confidence=0.8, pixel_height=120):
    obj = ReferenceObject(
        class_name="car", confidence=confidence, bbox_xyxy=[100,300,500,420],
        real_height_cm=120.0, pixel_height=pixel_height,
    )
    return YOLOResult(objects=[obj], engine="yolov8")

def make_yolo_empty():
    return YOLOResult(objects=[], engine="stub")

def make_depth_uniform(value=0.5, h=480, w=640):
    return DepthResult(
        depth_map=np.full((h, w), value, dtype=np.float32),
        is_metric=False, engine="test"
    )


# ── Tests ─────────────────────────────────────────────────────────────────────
class TestFusionCalibration:

    def test_calibration_source_yolo_when_object_detected(self):
        fusion = FusionStage(sensor_height_cm=300)
        img    = make_image()
        result = fusion.fuse(img, make_seg(), make_yolo_with_car(0.8), make_depth_uniform())
        assert "yolo_car" in result.calibration_source

    def test_calibration_source_fallback_when_no_yolo(self):
        fusion = FusionStage(sensor_height_cm=300)
        img    = make_image()
        result = fusion.fuse(img, make_seg(), make_yolo_empty(), make_depth_uniform())
        assert result.calibration_source == "fallback"

    def test_calibration_confidence_higher_with_yolo(self):
        fusion  = FusionStage(sensor_height_cm=300)
        img     = make_image()
        with_yolo = fusion.fuse(img, make_seg(), make_yolo_with_car(0.9), make_depth_uniform())
        no_yolo   = fusion.fuse(img, make_seg(), make_yolo_empty(),        make_depth_uniform())
        assert with_yolo.calibration_confidence > no_yolo.calibration_confidence

    def test_depth_map_shape_matches_image(self):
        fusion = FusionStage()
        img    = make_image(480, 640)
        result = fusion.fuse(img, make_seg(), make_yolo_empty(), make_depth_uniform())
        assert result.depth_map_cm.shape == (480, 640)

    def test_depth_map_values_are_positive(self):
        fusion = FusionStage(sensor_height_cm=300)
        img    = make_image()
        result = fusion.fuse(img, make_seg(), make_yolo_empty(), make_depth_uniform(0.5))
        assert result.depth_map_cm.min() >= 0


class TestFusionWaterStats:

    def test_no_water_gives_zero_depth(self):
        fusion = FusionStage()
        img    = make_image()
        no_water_seg = SegFormerResult(
            water_mask=np.zeros((480, 640), dtype=bool),
            water_coverage_pct=0.0, engine="test"
        )
        result = fusion.fuse(img, no_water_seg, make_yolo_empty(), make_depth_uniform())
        assert result.mean_flood_depth_cm == 0.0
        assert result.max_flood_depth_cm  == 0.0
        assert result.water_pixel_count   == 0

    def test_full_water_mask_gives_nonzero_depth(self):
        fusion = FusionStage(sensor_height_cm=100)
        img    = make_image()
        full_water = SegFormerResult(
            water_mask=np.ones((480, 640), dtype=bool),
            water_coverage_pct=100.0, engine="test"
        )
        result = fusion.fuse(img, full_water, make_yolo_empty(), make_depth_uniform(0.5))
        assert result.mean_flood_depth_cm > 0
        assert result.p90_flood_depth_cm  >= result.mean_flood_depth_cm

    def test_water_coverage_pct_passed_through(self):
        fusion = FusionStage()
        result = fusion.fuse(make_image(), make_seg(25.0), make_yolo_empty(), make_depth_uniform())
        assert result.water_coverage_pct == 25.0

    def test_engine_names_passed_through(self):
        fusion  = FusionStage()
        seg     = SegFormerResult(np.zeros((480,640),bool), 0, "my_segformer")
        yolo    = YOLOResult([], "my_yolo")
        depth   = DepthResult(np.zeros((480,640),np.float32), False, "my_depth")
        result  = fusion.fuse(make_image(), seg, yolo, depth)
        assert result.seg_engine   == "my_segformer"
        assert result.yolo_engine  == "my_yolo"
        assert result.depth_engine == "my_depth"
