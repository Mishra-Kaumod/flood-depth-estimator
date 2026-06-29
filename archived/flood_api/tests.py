from django.test import TestCase
import os
import tempfile

import cv2
import numpy as np

from flood_api.models import CameraLocation, FloodInundationTelemetry, PredictionFeedback
from flood_api.secure_random_image_views import analyze_image_secure
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


class SecureUploadDepthTests(TestCase):
    def _analyze_synthetic_image(self, water_rows):
        img = np.full((100, 100, 3), 120, dtype=np.uint8)
        if water_rows > 0:
            img[:water_rows, :, 0] = 210
            img[:water_rows, :, 1] = 110
            img[:water_rows, :, 2] = 80

        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            cv2.imwrite(path, img)
            analysis, error = analyze_image_secure(path)
            self.assertIsNone(error)
            self.assertIsNotNone(analysis)
            return analysis
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_moderate_visible_water_returns_non_zero_depth(self):
        analysis = self._analyze_synthetic_image(water_rows=25)
        self.assertGreater(analysis["water_pixels"], 20.0)
        self.assertGreater(analysis["depth_cm"], 0)

    def test_very_low_water_signal_returns_zero_depth(self):
        analysis = self._analyze_synthetic_image(water_rows=2)
        self.assertLess(analysis["water_pixels"], 5.0)
        self.assertEqual(analysis["depth_cm"], 0)
