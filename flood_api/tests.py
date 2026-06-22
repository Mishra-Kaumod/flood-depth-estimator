from django.test import TestCase

from flood_api.models import CameraLocation, FloodInundationTelemetry, PredictionFeedback
from flood_api.services.map_payload import build_dashboard_map_points
from flood_api.services.prediction_policy import harmonize_prediction


class PredictionPolicyTests(TestCase):
    def test_strong_surface_signal_promotes_min_depth(self):
        normalized = harmonize_prediction(
            raw_depth_cm=0.0,
            water_pct=82.0,
            raw_confidence=0.22,
            num_anchors=0,
        )
        self.assertTrue(normalized["is_water_confirmed"])
        self.assertEqual(normalized["depth_cm"], 5.0)

    def test_low_signals_return_dry(self):
        normalized = harmonize_prediction(
            raw_depth_cm=0.0,
            water_pct=1.5,
            raw_confidence=0.2,
            num_anchors=0,
        )
        self.assertFalse(normalized["is_water_confirmed"])
        self.assertEqual(normalized["depth_cm"], 0.0)
        self.assertEqual(normalized["warning"], "No water detected")


class MapPayloadVerdictTests(TestCase):
    def test_feedback_overrides_heuristic_verdict(self):
        cam = CameraLocation.objects.create(
            camera_id="cam-01",
            location_name="Test Junction",
            latitude=12.9716,
            longitude=77.5946,
        )
        telemetry = FloodInundationTelemetry.objects.create(
            image_name="flood_scene.jpg",
            camera=cam,
            strategy_applied="Ensemble",
            surface_water_confirmed_pct=88.0,
            computed_depth_cm=42.0,
            system_confidence_score_pct=77.0,
            detected_reference_objects=[],
            num_reference_objects=0,
            is_water_confirmed=True,
            safety_risk_assessment="Critical - synthetic",
        )
        PredictionFeedback.objects.create(
            telemetry=telemetry,
            feedback_type="rejected",
            reviewer="qa",
        )

        points = build_dashboard_map_points([telemetry])
        self.assertEqual(points[0]["response_verdict"], "Incorrect")
        self.assertEqual(points[0]["verdict_source"], "review_feedback")
