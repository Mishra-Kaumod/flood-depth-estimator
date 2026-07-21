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


# ── Regression tests — calibration must use px_per_cm, not a flat constant ────
class TestCalibrationMath:
    """
    These tests would have caught the bug where scale = sensor_height_cm in
    both branches, making pixel_height / real_height_cm completely irrelevant.

    Correct behaviour: two reference objects at the same relative depth but
    different apparent sizes must produce DIFFERENT depth_map_cm scales.
    """

    def _depth_cm_at_bbox(self, fusion, yolo_result, depth_rel_value=0.5, h=480, w=640):
        """Helper: run calibration and return the depth_map_cm value at the bbox center."""
        depth_map_rel = np.full((h, w), depth_rel_value, dtype=np.float32)
        depth_map_cm, src, conf = fusion._calibrate(depth_map_rel, yolo_result, (h, w, 3))
        obj = yolo_result.objects[0]
        cx = (obj.bbox_xyxy[0] + obj.bbox_xyxy[2]) // 2
        cy = (obj.bbox_xyxy[1] + obj.bbox_xyxy[3]) // 2
        return float(depth_map_cm[cy, cx]), src

    def test_larger_pixel_height_gives_shallower_depth(self):
        """Object appearing larger → closer → smaller depth estimate."""
        fusion = FusionStage(sensor_height_cm=300)
        # Same real height, same confidence — only pixel_height differs
        close_car = YOLOResult(objects=[ReferenceObject(
            class_name="car", confidence=0.9, bbox_xyxy=[100,200,500,440],
            real_height_cm=120.0, pixel_height=240,  # large → nearby
        )], engine="yolov8")
        far_car = YOLOResult(objects=[ReferenceObject(
            class_name="car", confidence=0.9, bbox_xyxy=[100,300,500,420],
            real_height_cm=120.0, pixel_height=120,  # small → far away
        )], engine="yolov8")
        depth_close, _ = self._depth_cm_at_bbox(fusion, close_car)
        depth_far,   _ = self._depth_cm_at_bbox(fusion, far_car)
        assert depth_close < depth_far, (
            f"Larger object (closer) should give smaller depth: "
            f"close={depth_close:.1f}cm, far={depth_far:.1f}cm"
        )

    def test_known_geometry_exact_depth(self):
        """
        Synthetic case with verifiable expected output:
          image_h=480, pixel_height=120px, real_height=120cm
          → px_per_cm = 1.0
          → depth_at_obj_cm = 480 / 1.0 = 480cm
          → rel_at_obj = 0.5 (uniform depth map)
          → scale = 480 / 0.5 = 960
          → depth_map_cm everywhere = 0.5 * 960 = 480cm
        """
        fusion = FusionStage(sensor_height_cm=300)
        yolo = YOLOResult(objects=[ReferenceObject(
            class_name="car", confidence=0.9, bbox_xyxy=[100, 300, 500, 420],
            real_height_cm=120.0, pixel_height=120,
        )], engine="yolov8")
        depth_map_rel = np.full((480, 640), 0.5, dtype=np.float32)
        depth_map_cm, src, _ = fusion._calibrate(depth_map_rel, yolo, (480, 640, 3))
        expected = 480.0   # 480 / (120/120) / 0.5 = 480
        assert abs(float(depth_map_cm[360, 300]) - expected) < 0.5, (
            f"Expected depth_map_cm ≈ {expected}cm, got {depth_map_cm[360, 300]:.2f}cm"
        )
        assert "yolo_car" in src

    def test_same_object_different_real_heights_differ(self):
        """
        Two objects with same pixel_height but different real heights
        must produce different depth estimates (not the same flat scale).
        """
        fusion = FusionStage(sensor_height_cm=300)
        car_yolo = YOLOResult(objects=[ReferenceObject(
            class_name="car", confidence=0.9, bbox_xyxy=[100,300,400,420],
            real_height_cm=120.0, pixel_height=120,
        )], engine="yolov8")
        truck_yolo = YOLOResult(objects=[ReferenceObject(
            class_name="truck", confidence=0.9, bbox_xyxy=[100,300,400,420],
            real_height_cm=380.0, pixel_height=120,  # same pixels, bigger real object
        )], engine="yolov8")
        depth_car,   _ = self._depth_cm_at_bbox(fusion, car_yolo)
        depth_truck, _ = self._depth_cm_at_bbox(fusion, truck_yolo)
        # Truck is taller → same pixel height means it's FARTHER away → deeper depth
        assert depth_truck > depth_car, (
            f"Truck (taller, same pixels) should give deeper estimate: "
            f"truck={depth_truck:.1f}cm, car={depth_car:.1f}cm"
        )

    def test_fallback_uses_sensor_height_not_object_properties(self):
        """Fallback branch must ignore any object properties and use sensor_height_cm."""
        sensor_h = 250.0
        fusion = FusionStage(sensor_height_cm=sensor_h)
        depth_map_rel = np.full((480, 640), 0.6, dtype=np.float32)
        depth_map_cm, src, conf = fusion._calibrate(depth_map_rel, make_yolo_empty(), (480,640,3))
        assert src == "fallback"
        assert conf == 0.3
        np.testing.assert_allclose(depth_map_cm, 0.6 * sensor_h, rtol=1e-5)
