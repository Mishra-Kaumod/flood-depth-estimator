"""Production-style inference adapter for the cleaned project."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

class ProductionFloodAnalyzer:
    """Use the cleaned analyzer as the base and enrich it with the stronger pipeline."""

    def __init__(self, model_path: str = "severity_model.pth") -> None:
        self.model_path = model_path
        self.base_analyzer = None
        self.reference_estimator = None
        self.production_pipeline = None

    def _get_base_analyzer(self):
        if self.base_analyzer is None:
            from archive.legacy_cli.modules.flood_analyzer import FloodAnalyzer
            self.base_analyzer = FloodAnalyzer(model_path=self.model_path)
        return self.base_analyzer

    def _get_reference_estimator(self):
        if self.reference_estimator is None:
            from src.reference_depth_estimator import ReferenceDepthEstimator
            self.reference_estimator = ReferenceDepthEstimator()
        return self.reference_estimator

    def _get_production_pipeline(self):
        if self.production_pipeline is None:
            from src.segformer_yolo_depthv2_pipeline import get_segformer_yolo_depthv2_pipeline
            self.production_pipeline = get_segformer_yolo_depthv2_pipeline()
        return self.production_pipeline

    def analyze_bgr(self, image: np.ndarray, image_path: str = "<in-memory image>") -> Dict[str, Any]:
        try:
            base_result = self._get_base_analyzer().analyze_bgr(image, image_path)
        except Exception as exc:  # pragma: no cover - runtime dependency fallback
            return {
                "image_path": str(image_path),
                "water_detected": False,
                "final_flood_level": "No Flood Detected",
                "depth_cm": 0,
                "depth_method": "fallback",
                "error": str(exc),
            }

        if not base_result.get("water_detected", False):
            return base_result

        try:
            import cv2
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            production_result = self._get_production_pipeline().predict(rgb_image)
            reference_estimate = self._get_reference_estimator().estimate(rgb_image)
        except Exception as exc:  # pragma: no cover - runtime dependency fallback
            base_result["production_warning"] = str(exc)
            return base_result

        base_result.update(
            {
                "production_depth_cm": production_result.get("depth_cm"),
                "production_confidence": production_result.get("confidence"),
                "production_method": production_result.get("method"),
                "production_severity": production_result.get("severity", {}),
                "production_visual_cues": production_result.get("visual_cues", []),
                "production_trace": production_result.get("pipeline_trace", []),
                "production_action": production_result.get("action_trigger"),
                "production_reference_estimate": reference_estimate,
            }
        )

        if base_result.get("depth_cm") in (None, 0):
            base_result["depth_cm"] = production_result.get("depth_cm", base_result.get("depth_cm"))
            base_result["depth_method"] = production_result.get("method", base_result.get("depth_method"))

        return base_result
