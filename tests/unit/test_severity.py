# tests/unit/test_severity.py
"""
Unit tests for SeverityStage — risk thresholds, action mapping, edge cases.
Run: pytest tests/unit/test_severity.py -v
"""

import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline.severity import SeverityStage, FloodPrediction
from pipeline.fusion   import StructuredFeatures


# ── Helpers ───────────────────────────────────────────────────────────────────
def make_features(
    p90_depth=0.0, mean_depth=0.0, max_depth=0.0,
    water_pct=0.0, cal_source="fallback", cal_confidence=None
):
    # Mirror actual FusionStage behaviour: YOLO-calibrated maps have higher
    # calibration_confidence than fallback (YOLO conf ≈ 0.8, fallback = 0.3)
    if cal_confidence is None:
        cal_confidence = 0.8 if cal_source.startswith("yolo_") else 0.3
    return StructuredFeatures(
        water_coverage_pct    = water_pct,
        water_pixel_count     = int(water_pct * 100),
        mean_flood_depth_cm   = mean_depth,
        max_flood_depth_cm    = max_depth,
        p90_flood_depth_cm    = p90_depth,
        calibration_source    = cal_source,
        calibration_confidence= cal_confidence,
        seg_engine            = "test",
        yolo_engine           = "test",
        depth_engine          = "test",
        water_mask            = np.zeros((10,10), dtype=bool),
        depth_map_cm          = np.zeros((10,10), dtype=np.float32),
    )


def predict(p90=0.0, water_pct=0.0, cal="fallback"):
    stage = SeverityStage()
    return stage.predict(
        features      = make_features(p90, p90*0.8, p90, water_pct, cal),
        location_id   = "LOC_TEST",
        camera_id     = "CAM_TEST",
        latitude      = 12.9716,
        longitude     = 77.5946,
        location_name = "Test",
        timestamp     = "2026-07-19T14:00:00",
        batch_id      = "batch_test",
    )


# ── Risk threshold tests ───────────────────────────────────────────────────────
class TestRiskThresholds:

    def test_zero_depth_is_no_flood(self):
        r = predict(p90=0.0, water_pct=0.0)
        assert r.risk_level    == "NO FLOOD"
        assert r.flood_detected == False

    def test_sub_1pct_water_is_no_flood_regardless_of_depth(self):
        """Very little water coverage should override depth estimate."""
        r = predict(p90=50.0, water_pct=0.5)
        assert r.risk_level == "NO FLOOD"

    def test_10cm_is_low_risk(self):
        r = predict(p90=10.0, water_pct=20.0)
        assert r.risk_level    == "LOW RISK"
        assert r.flood_detected == True

    def test_20cm_is_moderate(self):
        r = predict(p90=20.0, water_pct=25.0)
        assert r.risk_level == "MODERATE"

    def test_50cm_is_high_risk(self):
        r = predict(p90=50.0, water_pct=40.0)
        assert r.risk_level == "HIGH RISK"

    def test_80cm_is_critical(self):
        r = predict(p90=80.0, water_pct=60.0)
        assert r.risk_level == "CRITICAL"

    def test_boundary_exactly_15cm_is_low_risk(self):
        r = predict(p90=15.0, water_pct=15.0)
        assert r.risk_level == "LOW RISK"

    def test_boundary_just_over_35cm_is_high_risk(self):
        r = predict(p90=35.1, water_pct=30.0)
        assert r.risk_level == "HIGH RISK"


# ── Output contract tests ─────────────────────────────────────────────────────
class TestOutputContract:

    def test_all_5_outputs_present(self):
        r = predict(p90=40.0, water_pct=30.0)
        assert hasattr(r, "flood_detected")
        assert hasattr(r, "water_depth_cm")
        assert hasattr(r, "risk_level")
        assert hasattr(r, "recommended_action")
        assert hasattr(r, "confidence_pct")

    def test_confidence_in_valid_range(self):
        for depth in [0, 10, 30, 60, 90]:
            r = predict(p90=depth, water_pct=max(depth/3, 1))
            assert 0 <= r.confidence_pct <= 100, f"confidence out of range at depth={depth}"

    def test_recommended_action_not_empty(self):
        for depth in [0, 10, 30, 60, 90]:
            r = predict(p90=depth, water_pct=max(depth/3, 1))
            assert r.recommended_action, f"empty action at depth={depth}"

    def test_flood_detected_true_when_depth_nonzero(self):
        r = predict(p90=5.0, water_pct=10.0)
        assert r.flood_detected == True

    def test_metadata_passed_through(self):
        r = predict(p90=20.0, water_pct=20.0)
        assert r.camera_id     == "CAM_TEST"
        assert r.location_id   == "LOC_TEST"
        assert r.latitude      == 12.9716
        assert r.longitude     == 77.5946

    def test_yolo_calibration_boosts_confidence(self):
        with_yolo    = predict(p90=30.0, water_pct=25.0, cal="yolo_car")
        without_yolo = predict(p90=30.0, water_pct=25.0, cal="fallback")
        assert with_yolo.confidence_pct > without_yolo.confidence_pct

    def test_to_dict_excludes_numpy_arrays(self):
        r = predict(p90=20.0, water_pct=20.0)
        d = r.to_dict()
        for k, v in d.items():
            assert not isinstance(v, np.ndarray), f"numpy array in to_dict key: {k}"
