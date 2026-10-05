import copy
import unittest

from src.v6_shadow_pipeline import V6ShadowPipeline


def payload(efficientnet=42.0):
    return {
        "depth_cm": 7.0,
        "structured_features": {
            "efficientnet_candidate_depth_cm": efficientnet,
            "road_scene_dry_road_probability": 0.1,
            "road_scene_wet_road_probability": 0.2,
            "road_scene_shallow_flood_probability": 0.3,
            "road_scene_meaningful_flood_probability": 0.4,
            "no_water_probability": 0.1,
            "wet_road_no_water_probability": 0.2,
            "water_coverage_pct": 20.0,
            "near_water_coverage_pct": 10.0,
            "mid_water_coverage_pct": 5.0,
            "far_water_coverage_pct": 4.0,
            "input_image_height_px": 100,
            "reference_object_diagnostics": [{
                "label": "car", "detector_confidence": 0.9, "bbox": [0, 20, 50, 80],
                "area_ratio": 0.1, "water_submersion_ratio": 0.5,
                "waterline_height_ratio": 0.4, "waterline_depth_proxy_cm": 300.0,
            }],
            "dense_depth_relative_p90": 0.9,
            "dense_depth_map_min": 0.1,
            "dense_depth_map_max": 1.0,
            "reference_depth_cm": 500.0,
            "region_depth_cm": 123.0,
            "mask_conditioned_fusion_depth_cm": 250.0,
            "reference_available": True,
            "dense_depth_backend": "depth-anything-v2",
            "reference_detection_backend": "object-detector",
        },
    }


class StubV5:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def predict(self, image):
        self.calls += 1
        return copy.deepcopy(self.result)


class V6ShadowPipelineTests(unittest.TestCase):
    def test_contour_yolo_and_dense_values_cannot_change_v6_depth(self):
        result = V6ShadowPipeline(StubV5(payload(42.0))).predict(object())
        self.assertEqual(result.primary_depth_cm, 42.0)
        self.assertEqual(result.final_shadow_depth_cm, 42.0)
        self.assertEqual(result.contract.advisory["contour_reference_depth_estimate"].authority.value, "DIAGNOSTIC_ONLY")
        self.assertEqual(result.contract.object_diagnostics[0].diagnostic_depth_proxy.authority.value, "DIAGNOSTIC_ONLY")
        self.assertEqual(result.contract.relative_depth["dense_relative_p90"].unit, "relative")

    def test_semantic_extremes_cannot_cap_or_zero_depth(self):
        raw = payload(42.0)
        raw["structured_features"].update({
            "no_water_probability": 0.99,
            "road_scene_dry_road_probability": 0.99,
            "road_scene_meaningful_flood_probability": 0.01,
        })
        result = V6ShadowPipeline(StubV5(raw)).predict(object())
        self.assertEqual(result.final_shadow_depth_cm, 42.0)

    def test_string_boolean_is_reported_missing_not_coerced(self):
        raw = payload(42.0)
        raw["structured_features"]["reference_available"] = "false"
        result = V6ShadowPipeline(StubV5(raw)).predict(object())
        reference_available = result.contract.advisory["reference_available"]
        self.assertFalse(reference_available.available)
        self.assertIn("reference_available", result.contract.malformed_signal_names)

    def test_only_efficientnet_owns_final_depth(self):
        result = V6ShadowPipeline(StubV5(payload(42.0))).predict(object())
        self.assertEqual(result.numerical_owner, "efficientnet_primary_anchor")
        owners = {stage.numerical_owner for stage in result.stages if stage.numerical_owner is not None}
        self.assertEqual(owners, {"efficientnet_primary_anchor"})
        self.assertEqual(result.stages[-1].numerical_depth_after_cm, 42.0)

    def test_missing_primary_stays_missing(self):
        result = V6ShadowPipeline(StubV5(payload(None))).predict(object())
        self.assertIsNone(result.primary_depth_cm)
        self.assertIsNone(result.final_shadow_depth_cm)
        self.assertIn("efficientnet_primary_depth_cm", result.reliability.missing_physical_evidence)

    def test_v5_payload_is_not_mutated(self):
        raw = payload(42.0)
        original = copy.deepcopy(raw)
        stub = StubV5(raw)
        V6ShadowPipeline(stub).predict(object())
        self.assertEqual(raw, original)
        self.assertEqual(stub.calls, 1)
