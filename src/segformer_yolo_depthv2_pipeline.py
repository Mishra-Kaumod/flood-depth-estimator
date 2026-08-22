"""
Stage-aligned flood inference pipeline:
RGB -> SegFormer water mask -> YOLOv8 reference objects ->
Depth Anything V2 dense depth proxy -> Fusion engine ->
Calibration/severity model.

The code keeps explicit stage boundaries so UI and APIs can report
traceable execution details for each step.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from src.reference_depth_estimator import ReferenceDepthEstimator
from src.settings import load_settings_dict
from src.water_region_detector import WaterRegionDetector

try:
    from archive.legacy_cli.modules.object_detection import ObjectDetector
except ImportError:  # pragma: no cover - optional runtime fallback
    ObjectDetector = None

logger = logging.getLogger(__name__)

RESIDUAL_FUSION_FEATURE_NAMES = [
    "pipeline_depth_cm",
    "efficientnet_candidate_depth_cm",
    "reference_depth_cm",
    "reference_count",
    "max_reference_submersion",
    "dense_depth_cm",
    "water_coverage_pct",
    "near_water_coverage_pct",
    "mid_water_coverage_pct",
    "far_water_coverage_pct",
    "largest_water_region_pct",
    "region_depth_cm",
    "waterline_pct",
    "immediate_risk",
    "far_water_only",
    "mask_quality_warning",
    "low_water_gate_applied",
    "shallow_water_gate_exception",
    "muddy_water_fallback_applied",
    "full_road_water_no_reference",
]

RESIDUAL_FUSION_BOOL_FEATURES = {
    "immediate_risk",
    "far_water_only",
    "mask_quality_warning",
    "low_water_gate_applied",
    "shallow_water_gate_exception",
    "muddy_water_fallback_applied",
    "full_road_water_no_reference",
}


class ResidualFusionDepthModel(nn.Module):
    def __init__(self, input_dim: int, max_residual_cm: float):
        super().__init__()
        self.max_residual_cm = float(max_residual_cm)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 48),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Linear(24, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor, base_depth_cm: torch.Tensor) -> torch.Tensor:
        residual_cm = self.net(x) * self.max_residual_cm
        return torch.clamp(base_depth_cm + residual_cm, min=0.0, max=180.0)


@dataclass
class ReferenceObject:
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    area_ratio: float
    water_submersion_ratio: float


def _depth_to_severity(depth_cm: float, features: Dict[str, float]) -> Dict[str, Any]:
    coverage = features.get("water_coverage_pct", 0.0) / 100.0
    max_reference_submersion = features.get("max_reference_submersion", 0.0)

    if depth_cm < 5:
        return {"level": "SAFE", "label": "No significant flooding", "color": "#16a34a", "stage": 1}
    if depth_cm < 20 and coverage < 0.35 and max_reference_submersion < 0.5:
        return {
            "level": "WATERLOGGED",
            "label": "Localized waterlogging / no flood",
            "color": "#f59e0b",
            "stage": 2,
        }
    if depth_cm < 20:
        return {"level": "LOW", "label": "Minor flooding", "color": "#ca8a04", "stage": 2}
    if depth_cm < 50:
        return {"level": "MEDIUM", "label": "Moderate flooding", "color": "#ea580c", "stage": 3}
    if depth_cm < 80:
        return {"level": "HIGH", "label": "High flood ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â avoid travel", "color": "#dc2626", "stage": 4}
    return {"level": "CRITICAL", "label": "Severe / dangerous flooding", "color": "#7f1d1d", "stage": 5}


class SegformerYoloDepthV2Pipeline:
    """
    Structured multi-stage pipeline with deterministic stage order.
    """

    def __init__(
        self,
        yolo_weights_path: str = "yolov8n.pt",
        yolo_confidence: float = 0.25,
    ) -> None:
        self.water_detector = WaterRegionDetector()
        self.reference_estimator = ReferenceDepthEstimator()
        self.yolo_weights_path = Path(yolo_weights_path)
        self.yolo_confidence = float(yolo_confidence)
        self._yolo_model = None
        self._yolo_backend = "contour-proxy"
        self.object_detector = None
        self._depth_estimator = None
        self._depth_backend = "dense-depth-proxy"
        self._depth_model_name = "depth-anything/Depth-Anything-V2-Small-hf"
        self._efficientnet_model = None
        self._efficientnet_transform = None
        self._efficientnet_backend = "disabled"
        self._efficientnet_max_depth_cm = 100.0
        #Edit_start
        self._no_water_model = None
        self._no_water_transform = None
        self._no_water_device = torch.device("cpu")
        self._no_water_backend = "disabled"
        #Edit_end
        self._residual_fusion_model = None
        self._residual_fusion_backend = "disabled"
        self._residual_fusion_device = torch.device("cpu")
        self._residual_fusion_feature_names = RESIDUAL_FUSION_FEATURE_NAMES
        self._residual_fusion_feature_mean = None
        self._residual_fusion_feature_std = None
        self._load_yolo_if_available()
        self._load_object_detector_if_available()
        self._load_depth_anything_if_available()
        self._load_efficientnet_signal_if_available()
        #Edit_start
        self._load_no_water_guard_if_available()
        #Edit_end
        self._load_residual_fusion_if_available()

    def _load_yolo_if_available(self) -> None:
        if not self.yolo_weights_path.exists():
            logger.info("YOLO weights missing at %s, using contour proxy", self.yolo_weights_path)
            return
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.info("ultralytics not installed, using contour proxy for reference objects")
            return

        try:
            self._yolo_model = YOLO(str(self.yolo_weights_path))
            self._yolo_backend = "yolov8"
            logger.info("Loaded YOLOv8 reference detector from %s", self.yolo_weights_path)
        except (RuntimeError, OSError, ValueError, pickle.UnpicklingError) as exc:
            logger.warning("YOLO weight load failed (%s). Using contour proxy.", exc)
            self._yolo_model = None
            self._yolo_backend = "contour-proxy"

    def _load_object_detector_if_available(self) -> None:
        if ObjectDetector is None:
            return
        try:
            self.object_detector = ObjectDetector(model_name=str(self.yolo_weights_path))
            logger.info("Loaded improved object detector for inference pipeline")
        except Exception as exc:  # pragma: no cover - optional runtime fallback
            logger.info("Improved object detector unavailable, falling back to contour proxy: %s", exc)
            self.object_detector = None

    def _load_depth_anything_if_available(self) -> None:
        try:
            cfg = load_settings_dict().get("inference", {}).get("depth_model", {})
        except Exception as exc:
            logger.info("Depth model config unavailable, using proxy depth map: %s", exc)
            cfg = {}

        if not bool(cfg.get("enabled", True)):
            return

        backend = str(cfg.get("backend", "depth_anything_v2")).lower()
        if backend in {"proxy", "none", "dense-depth-proxy"}:
            return
        if backend not in {"depth_anything", "depth_anything_v2", "hf_depth_estimation"}:
            logger.warning("Unknown depth_model backend '%s', using proxy depth map", backend)
            return

        self._depth_model_name = str(
            cfg.get("model_name", "depth-anything/Depth-Anything-V2-Small-hf")
        )
        device = int(cfg.get("device", -1))

        try:
            from transformers import pipeline

            self._depth_estimator = pipeline(
                task="depth-estimation",
                model=self._depth_model_name,
                device=device,
            )
            self._depth_backend = f"depth-anything-v2:{self._depth_model_name}"
            logger.info("Loaded Depth Anything V2 backend: %s", self._depth_model_name)
        except Exception as exc:
            logger.warning("Depth Anything V2 unavailable, using proxy depth map: %s", exc)
            self._depth_estimator = None
            self._depth_backend = "dense-depth-proxy"

    def _build_efficientnet_depth_model(self) -> nn.Module:
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        return model

    def _load_efficientnet_signal_if_available(self) -> None:
        try:
            cfg = load_settings_dict().get("inference", {}).get("efficientnet_signal", {})
        except Exception as exc:
            logger.info("EfficientNet signal config unavailable: %s", exc)
            return

        if not bool(cfg.get("enabled", False)):
            return

        model_path = Path(str(cfg.get("model_path", "models/candidate/best_flood_model_water_aware.pth")))
        configured_max_depth_cm = float(cfg.get("max_depth_cm", 100.0))
        if not model_path.exists():
            logger.warning("EfficientNet signal checkpoint missing at %s", model_path)
            return

        try:
            device = torch.device("cuda" if torch.cuda.is_available() and str(cfg.get("device", "cpu")) == "cuda" else "cpu")
            model = self._build_efficientnet_depth_model().to(device)
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            self._efficientnet_max_depth_cm = float(checkpoint.get("max_depth_cm", configured_max_depth_cm)) if isinstance(checkpoint, dict) else configured_max_depth_cm
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            self._efficientnet_model = model
            self._efficientnet_device = device
            self._efficientnet_backend = str(model_path)
            self._efficientnet_transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            logger.info("Loaded EfficientNet depth signal from %s", model_path)
        except Exception as exc:
            logger.warning("EfficientNet depth signal unavailable: %s", exc)
            self._efficientnet_model = None
            self._efficientnet_transform = None
            self._efficientnet_backend = "unavailable"

    #Edit_start
    def _build_no_water_model(self) -> nn.Module:
        model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, 2)
        return model

    def _load_no_water_guard_if_available(self) -> None:
        try:
            cfg = load_settings_dict().get("inference", {}).get("no_water_guard", {})
        except Exception as exc:
            logger.info("No-water guard config unavailable: %s", exc)
            return

        if not bool(cfg.get("enabled", False)):
            return

        model_path = Path(str(cfg.get("model_path", "models/no_water_guard_mobilenet_v3_small.pth")))
        if not model_path.exists():
            logger.warning("No-water guard checkpoint missing at %s", model_path)
            self._no_water_backend = "unavailable"
            return

        try:
            device = torch.device(
                "cuda" if torch.cuda.is_available() and str(cfg.get("device", "cpu")) == "cuda" else "cpu"
            )
            model = self._build_no_water_model().to(device)
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            self._no_water_model = model
            self._no_water_device = device
            self._no_water_backend = str(model_path)
            self._no_water_transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            logger.info("Loaded no-water guard from %s", model_path)
        except Exception as exc:
            logger.warning("No-water guard unavailable: %s", exc)
            self._no_water_model = None
            self._no_water_transform = None
            self._no_water_backend = "unavailable"
    #Edit_end

    def _load_residual_fusion_if_available(self) -> None:
        try:
            cfg = load_settings_dict().get("inference", {}).get("residual_fusion_signal", {})
        except Exception as exc:
            logger.info("Residual fusion config unavailable: %s", exc)
            return

        if not bool(cfg.get("enabled", False)):
            return

        model_path = Path(str(cfg.get("model_path", "models/candidate/residual_fusion_depth_model.pt")))
        if not model_path.exists():
            logger.warning("Residual fusion checkpoint missing at %s", model_path)
            return

        try:
            device = torch.device("cuda" if torch.cuda.is_available() and str(cfg.get("device", "cpu")) == "cuda" else "cpu")
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            feature_names = list(checkpoint.get("feature_names", RESIDUAL_FUSION_FEATURE_NAMES))
            max_residual_cm = float(checkpoint.get("max_residual_cm", cfg.get("max_residual_cm", 35.0)))
            model = ResidualFusionDepthModel(len(feature_names), max_residual_cm).to(device)
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            model.eval()

            feature_mean = np.asarray(checkpoint.get("feature_mean"), dtype=np.float32)
            feature_std = np.asarray(checkpoint.get("feature_std"), dtype=np.float32)
            if feature_mean.shape[0] != len(feature_names) or feature_std.shape[0] != len(feature_names):
                raise ValueError("Residual fusion feature normalization shape mismatch")
            feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)

            self._residual_fusion_model = model
            self._residual_fusion_device = device
            self._residual_fusion_backend = str(model_path)
            self._residual_fusion_feature_names = feature_names
            self._residual_fusion_feature_mean = feature_mean
            self._residual_fusion_feature_std = feature_std
            logger.info("Loaded residual fusion depth model from %s", model_path)
        except Exception as exc:
            logger.warning("Residual fusion depth model unavailable: %s", exc)
            self._residual_fusion_model = None
            self._residual_fusion_backend = "unavailable"
            self._residual_fusion_feature_mean = None
            self._residual_fusion_feature_std = None

    @staticmethod
    def _feature_float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            value = float(value)
            if not np.isfinite(value):
                return default
            return value
        except (TypeError, ValueError):
            return default

    def _residual_fusion_feature_vector(self, pipeline_depth_cm: float, features: Dict[str, Any]) -> np.ndarray:
        values: List[float] = []
        for name in self._residual_fusion_feature_names:
            if name == "pipeline_depth_cm":
                value = float(pipeline_depth_cm)
            elif name == "dense_depth_cm":
                value = self._feature_float(features.get("dense_depth_p90")) * 120.0
            elif name in RESIDUAL_FUSION_BOOL_FEATURES:
                value = 1.0 if bool(features.get(name, False)) else 0.0
            else:
                value = self._feature_float(features.get(name))
            values.append(value)
        return np.asarray(values, dtype=np.float32)

    def _apply_residual_fusion_model(
        self,
        depth_cm: float,
        confidence: float,
        action: str,
        features: Dict[str, Any],
    ) -> Tuple[float, float, str]:
        try:
            cfg = load_settings_dict().get("inference", {}).get("residual_fusion_signal", {})
        except Exception:
            cfg = {}

        if self._residual_fusion_model is None or not bool(cfg.get("apply_corrections", True)):
            features["residual_fusion_status"] = self._residual_fusion_backend
            return depth_cm, confidence, action

        candidate_depth = features.get("efficientnet_candidate_depth_cm")
        if candidate_depth is None:
            features["residual_fusion_status"] = "skipped_no_efficientnet_depth"
            return depth_cm, confidence, action

        skip_low_water = bool(cfg.get("skip_low_water_gate", True))
        if skip_low_water and bool(features.get("low_water_gate_applied", False)) and not bool(features.get("shallow_water_gate_exception", False)):
            features["residual_fusion_status"] = "skipped_low_water_gate"
            return depth_cm, confidence, action

        try:
            raw = self._residual_fusion_feature_vector(depth_cm, features)
            normalized = (raw - self._residual_fusion_feature_mean) / self._residual_fusion_feature_std
            x = torch.tensor(normalized.reshape(1, -1), dtype=torch.float32, device=self._residual_fusion_device)
            base = torch.tensor([[float(candidate_depth)]], dtype=torch.float32, device=self._residual_fusion_device)
            with torch.no_grad():
                fusion_depth = float(self._residual_fusion_model(x, base).squeeze().item())
        except Exception as exc:
            logger.warning("Residual fusion inference failed: %s", exc)
            features["residual_fusion_status"] = "inference_failed"
            return depth_cm, confidence, action

        original_depth = round(float(depth_cm), 2)
        fusion_depth = round(float(np.clip(fusion_depth, 0.0, 180.0)), 2)
        max_change_cm = float(cfg.get("max_live_adjustment_cm", 30.0))
        candidate_depth_value = float(candidate_depth)
        candidate_alignment_cm = float(cfg.get("large_adjustment_candidate_alignment_cm", 15.0))
        large_adjustment_requested = abs(fusion_depth - original_depth) > max_change_cm
        candidate_aligned = abs(fusion_depth - candidate_depth_value) <= candidate_alignment_cm
        allow_large_adjustment = bool(cfg.get("allow_large_candidate_aligned_adjustment", True)) and large_adjustment_requested and candidate_aligned

        if allow_large_adjustment:
            applied_depth = fusion_depth
        else:
            change_cm = float(np.clip(fusion_depth - original_depth, -max_change_cm, max_change_cm))
            applied_depth = round(float(np.clip(original_depth + change_cm, 0.0, 180.0)), 2)

        features["pre_residual_fusion_depth_cm"] = original_depth
        features["residual_fusion_depth_cm"] = fusion_depth
        features["residual_fusion_applied_depth_cm"] = applied_depth
        features["residual_fusion_delta_cm"] = round(applied_depth - original_depth, 2)
        features["residual_fusion_large_adjustment_allowed"] = bool(allow_large_adjustment)
        features["residual_fusion_status"] = "applied"
        features["residual_fusion_model_path"] = self._residual_fusion_backend
        features["final_aggregation_source"] = "residual_fusion_model"
        features["model_agreement_depth_cm"] = applied_depth
        features["final_output_reason"] = (
            f"Residual fusion model adjusted depth from {original_depth:.2f} cm to {applied_depth:.2f} cm "
            "using EfficientNet plus pipeline evidence."
        )

        if allow_large_adjustment:
            features["review_required"] = True
            features["review_reason"] = "Residual fusion and EfficientNet agreed on a large correction against noisy pipeline signals."
        elif abs(applied_depth - original_depth) >= float(cfg.get("review_delta_cm", 20.0)):
            features["review_required"] = True
            features["review_reason"] = "Residual fusion changed the rule-based pipeline depth by more than the review threshold."
        confidence = max(float(confidence), min(0.90, float(confidence) + 0.05))
        action = self._action_for_final_depth(applied_depth, features, action)
        return applied_depth, round(float(np.clip(confidence, 0.0, 0.98)), 4), action
    def _efficientnet_depth_signal(self, image_rgb: np.ndarray) -> Optional[float]:
        if self._efficientnet_model is None or self._efficientnet_transform is None:
            return None
        image = Image.fromarray(image_rgb.astype(np.uint8), mode="RGB")
        tensor = self._efficientnet_transform(image).unsqueeze(0).to(self._efficientnet_device)
        with torch.no_grad():
            return round(float(self._efficientnet_model(tensor).squeeze().item()) * self._efficientnet_max_depth_cm, 2)

    #Edit_start
    def _no_water_guard_signal(self, image_rgb: np.ndarray) -> Optional[float]:
        if self._no_water_model is None or self._no_water_transform is None:
            return None
        image = Image.fromarray(image_rgb.astype(np.uint8), mode="RGB")
        tensor = self._no_water_transform(image).unsqueeze(0).to(self._no_water_device)
        try:
            with torch.no_grad():
                probabilities = torch.softmax(self._no_water_model(tensor), dim=1)
            return float(np.clip(probabilities[0, 0].item(), 0.0, 1.0))
        except Exception as exc:
            logger.warning("No-water guard inference failed: %s", exc)
            return None

    def _apply_no_water_guard(
        self,
        depth_cm: float,
        confidence: float,
        action: str,
        features: Dict[str, Any],
    ) -> Tuple[float, float, str]:
        probability = features.get("no_water_probability")
        if probability is None:
            return depth_cm, confidence, action

        try:
            cfg = load_settings_dict().get("inference", {}).get("no_water_guard", {})
        except Exception:
            cfg = {}
        threshold = float(cfg.get("no_water_threshold", 0.92))
        max_coverage_pct = float(cfg.get("max_water_coverage_pct", 3.0))
        coverage_pct = float(features.get("water_coverage_pct", 0.0))
        #Edit_start
        corroborated = (
            float(probability) >= threshold
            and coverage_pct <= max_coverage_pct
            and not bool(features.get("immediate_risk", False))
            and not bool(features.get("muddy_water_fallback_applied", False))
        )
        #Edit_end
        features["no_water_guard_status"] = "applied" if corroborated else "uncertain"
        features["no_water_guard_corroborated"] = corroborated
        if not corroborated:
            return depth_cm, confidence, action

        features["no_water_guard_applied"] = True
        features["final_output_reason"] = (
            f"No-water guard detected a dry scene with probability {float(probability):.2f}; "
            "existing depth signals were suppressed."
        )
        features["final_aggregation_source"] = "no_water_guard"
        return 0.0, round(float(max(confidence, float(probability))), 4), self._action_for_final_depth(0.0, features, action)
    #Edit_end

    #Edit_start
    def _apply_dry_land_guard(
        self,
        depth_cm: float,
        confidence: float,
        action: str,
        image_rgb: np.ndarray,
        features: Dict[str, Any],
    ) -> Tuple[float, float, str]:
        if not self.water_detector.looks_like_dry_land(image_rgb):
            return depth_cm, confidence, action
        if (
            float(features.get("near_water_coverage_pct", 0.0)) > 1.0
            or bool(features.get("immediate_risk", False))
            or bool(features.get("muddy_water_fallback_applied", False))
            or int(float(features.get("reference_count", 0.0))) > 0
        ):
            return depth_cm, confidence, action
        features["dry_land_guard_applied"] = True
        features["final_aggregation_source"] = "dry_land_guard"
        features["final_output_reason"] = "Cracked dry-land evidence with no near-field water overruled depth signals."
        return 0.0, round(float(max(confidence, 0.90)), 4), self._action_for_final_depth(0.0, features, action)
    #Edit_end

    def _segformer_water_mask(self, image_rgb: np.ndarray) -> Tuple[np.ndarray, float]:
        # SegFormer-aligned stage boundary. Current backend is a lightweight detector.
        water_mask, water_coverage = self.water_detector.detect(image_rgb)
        return (water_mask > 0).astype(np.uint8) * 255, float(water_coverage)

    def _extract_reference_from_yolo(
        self,
        image_rgb: np.ndarray,
        water_mask: np.ndarray,
    ) -> List[ReferenceObject]:
        assert self._yolo_model is not None

        results = self._yolo_model(image_rgb, conf=self.yolo_confidence, verbose=False)[0]
        names = results.names
        h, w = image_rgb.shape[:2]
        target_labels = {"person", "car", "truck", "bus", "motorbike", "motorcycle", "bicycle"}
        refs: List[ReferenceObject] = []

        for box in results.boxes:
            cls_idx = int(box.cls.item())
            conf = float(box.conf.item())
            label = names.get(cls_idx, str(cls_idx)) if isinstance(names, dict) else str(names[cls_idx])
            if label not in target_labels:
                continue

            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h))
            if x2 <= x1 or y2 <= y1:
                continue

            bbox_area = float((x2 - x1) * (y2 - y1))
            area_ratio = bbox_area / float(h * w)
            bbox_mask = water_mask[y1:y2, x1:x2]
            submersion = float((bbox_mask > 0).mean()) if bbox_mask.size else 0.0
            refs.append(
                ReferenceObject(
                    label=label,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    area_ratio=round(area_ratio, 4),
                    water_submersion_ratio=round(submersion, 4),
                )
            )

        refs.sort(key=lambda item: item.area_ratio, reverse=True)
        return refs

    def _extract_reference_from_object_detector(
        self,
        image_rgb: np.ndarray,
        water_mask: np.ndarray,
    ) -> List[ReferenceObject]:
        if self.object_detector is None:
            return []

        detections = self.object_detector.detect_objects(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        h, w = image_rgb.shape[:2]
        refs: List[ReferenceObject] = []
        target_labels = {"person", "car", "truck", "bus", "motorcycle", "bicycle"}

        for det in detections:
            label = det.get("class", "")
            if label not in target_labels:
                continue

            bbox = det.get("bbox", {})
            x1, y1, x2, y2 = int(bbox.get("x1", 0)), int(bbox.get("y1", 0)), int(bbox.get("x2", 0)), int(bbox.get("y2", 0))
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h))
            if x2 <= x1 or y2 <= y1:
                continue

            bbox_area = float((x2 - x1) * (y2 - y1))
            area_ratio = bbox_area / float(h * w)
            bbox_mask = water_mask[y1:y2, x1:x2]
            submersion = float((bbox_mask > 0).mean()) if bbox_mask.size else 0.0
            refs.append(
                ReferenceObject(
                    label=label,
                    confidence=float(det.get("confidence", 0.0)),
                    bbox=(x1, y1, x2, y2),
                    area_ratio=round(area_ratio, 4),
                    water_submersion_ratio=round(submersion, 4),
                )
            )

        refs.sort(key=lambda item: item.area_ratio, reverse=True)
        return refs

    def _extract_reference_from_contours(
        self,
        image_rgb: np.ndarray,
        water_mask: np.ndarray,
    ) -> List[ReferenceObject]:
        contour_summary = self.reference_estimator.detect_reference_objects(image_rgb, water_mask)
        h, w = image_rgb.shape[:2]
        refs: List[ReferenceObject] = []

        for item in contour_summary["vehicles"]:
            x, y, bw, bh = item["rect"]
            y2 = min(h, y + bh)
            x2 = min(w, x + bw)
            bbox_mask = water_mask[y:y2, x:x2]
            submersion = float((bbox_mask > 0).mean()) if bbox_mask.size else 0.0
            refs.append(
                ReferenceObject(
                    label="vehicle",
                    confidence=0.55,
                    bbox=(x, y, x2, y2),
                    area_ratio=round((bw * bh) / float(h * w), 4),
                    water_submersion_ratio=round(submersion, 4),
                )
            )

        for item in contour_summary["people"]:
            x, y, bw, bh = item["rect"]
            y2 = min(h, y + bh)
            x2 = min(w, x + bw)
            bbox_mask = water_mask[y:y2, x:x2]
            submersion = float((bbox_mask > 0).mean()) if bbox_mask.size else 0.0
            refs.append(
                ReferenceObject(
                    label="person",
                    confidence=0.50,
                    bbox=(x, y, x2, y2),
                    area_ratio=round((bw * bh) / float(h * w), 4),
                    water_submersion_ratio=round(submersion, 4),
                )
            )

        refs.sort(key=lambda item: item.area_ratio, reverse=True)
        return refs

    def _yolov8_reference_stage(
        self,
        image_rgb: np.ndarray,
        water_mask: np.ndarray,
    ) -> Tuple[List[ReferenceObject], str]:
        if self.object_detector is not None:
            try:
                refs = self._extract_reference_from_object_detector(image_rgb, water_mask)
                if refs:
                    return refs, "object-detector"
            except (RuntimeError, ValueError, AttributeError) as exc:
                logger.warning("Improved object detector failed, falling back: %s", exc)

        if self._yolo_model is not None:
            try:
                return self._extract_reference_from_yolo(image_rgb, water_mask), self._yolo_backend
            except (RuntimeError, ValueError) as exc:
                logger.warning("YOLO runtime failed, reverting to contour proxy: %s", exc)
        return self._extract_reference_from_contours(image_rgb, water_mask), "contour-proxy"

    def _depth_anything_v2_dense_map(self, image_rgb: np.ndarray, water_mask: np.ndarray) -> np.ndarray:
        """
        Return a normalized monocular depth map.

        Uses a real pretrained Depth Anything V2 model when available. If the
        model cannot be loaded or inference fails, falls back to the old proxy.
        """
        if self._depth_estimator is not None:
            try:
                return self._depth_anything_model_map(image_rgb, water_mask)
            except Exception as exc:
                logger.warning("Depth Anything V2 inference failed, using proxy depth map: %s", exc)
        return self._proxy_dense_depth_map(image_rgb, water_mask)

    def _depth_anything_model_map(self, image_rgb: np.ndarray, water_mask: np.ndarray) -> np.ndarray:
        h, w = image_rgb.shape[:2]
        image = Image.fromarray(image_rgb.astype(np.uint8), mode="RGB")
        output = self._depth_estimator(image)
        depth = output.get("depth") if isinstance(output, dict) else None
        if depth is None:
            raise RuntimeError("Depth Anything V2 did not return a depth map")

        if isinstance(depth, Image.Image):
            depth = depth.resize((w, h), Image.Resampling.BICUBIC)
            depth_arr = np.asarray(depth).astype(np.float32)
        else:
            depth_arr = np.asarray(depth, dtype=np.float32)
            if depth_arr.shape[:2] != (h, w):
                depth_arr = cv2.resize(depth_arr, (w, h), interpolation=cv2.INTER_CUBIC)

        if depth_arr.ndim == 3:
            depth_arr = depth_arr[..., 0]

        depth_min = float(np.min(depth_arr))
        depth_max = float(np.max(depth_arr))
        if depth_max - depth_min > 1e-6:
            depth_arr = (depth_arr - depth_min) / (depth_max - depth_min)
        else:
            depth_arr = np.zeros((h, w), dtype=np.float32)

        depth_map = np.where(water_mask > 0, depth_arr, depth_arr * 0.35)
        return depth_map.astype(np.float32)

    def _proxy_dense_depth_map(self, image_rgb: np.ndarray, water_mask: np.ndarray) -> np.ndarray:
        h, w = image_rgb.shape[:2]
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        smooth = cv2.GaussianBlur(gray, (0, 0), 1.2)
        inv_luma = 1.0 - smooth
        vertical_prior = np.linspace(0.0, 1.0, h, dtype=np.float32).reshape(h, 1)

        texture = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
        tex_max = float(np.max(texture))
        if tex_max > 1e-6:
            texture = texture / tex_max

        dense = (0.50 * vertical_prior) + (0.35 * inv_luma) + (0.15 * texture)
        dense = np.clip(dense, 0.0, 1.0)
        depth_map = np.where(water_mask > 0, dense, dense * 0.35)
        return depth_map.astype(np.float32)
    def _water_regions(self, water_mask: np.ndarray) -> List[dict]:
        contours, _ = cv2.findContours(water_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = water_mask.shape[:2]
        regions: List[dict] = []

        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            area = bw * bh
            if area < 400:
                continue

            regions.append(
                {
                    "bbox": (x, y, bw, bh),
                    "area_pct": float(area) / float(h * w) if h * w > 0 else 0.0,
                    "aspect_ratio": float(bw) / float(max(bh, 1)),
                    "bottom_aligned": ((y + bh) / float(h)) >= 0.65,
                }
            )

        regions.sort(key=lambda item: item["area_pct"], reverse=True)
        return regions

    def _estimate_region_depth(
        self,
        largest_region: dict,
        waterline_pct: float,
        dense_depth_cm: float,
    ) -> float:
        if not largest_region:
            return 0.0

        region_pct = largest_region["area_pct"]
        aspect = largest_region["aspect_ratio"]
        bottom_aligned = largest_region["bottom_aligned"]

        score = 0.70
        if region_pct >= 0.20:
            score += 0.16
        elif region_pct >= 0.10:
            score += 0.08

        if aspect >= 2.0 and bottom_aligned:
            score += 0.10

        if waterline_pct >= 35.0:
            score += 0.08

        score = min(score, 1.1)
        estimate = dense_depth_cm * score
        min_depth = 12.0 if region_pct >= 0.10 else 7.0
        estimate = max(min_depth, estimate)
        return float(np.clip(estimate, 0.0, 180.0))

    def _water_zone_features(self, water_mask: np.ndarray) -> Dict[str, float | bool]:
        h, w = water_mask.shape[:2]
        if h == 0 or w == 0:
            return {
                "near_water_coverage_pct": 0.0,
                "mid_water_coverage_pct": 0.0,
                "far_water_coverage_pct": 0.0,
                "water_touches_bottom": False,
                "far_water_only": False,
                "far_dominant_water": False,
                "broad_mask_warning": False,
                "immediate_risk": False,
                "mask_quality_warning": True,
            }

        water = water_mask > 0
        far = water[: int(h * 0.25), :]
        mid = water[int(h * 0.25): int(h * 0.60), :]
        near = water[int(h * 0.60):, :]
        bottom_band = water[int(h * 0.92):, :]

        far_pct = float(far.mean() * 100.0) if far.size else 0.0
        mid_pct = float(mid.mean() * 100.0) if mid.size else 0.0
        near_pct = float(near.mean() * 100.0) if near.size else 0.0
        bottom_pct = float(bottom_band.mean() * 100.0) if bottom_band.size else 0.0

        water_touches_bottom = bottom_pct >= 3.0
        far_water_only = far_pct >= 5.0 and near_pct < 5.0 and mid_pct < 10.0
        far_dominant_water = far_pct >= near_pct + 30.0 and near_pct < 45.0
        broad_mask_warning = near_pct >= 55.0 and mid_pct >= 45.0 and far_pct >= 45.0
        immediate_risk = (near_pct >= 12.0 or (water_touches_bottom and mid_pct >= 15.0)) and not far_dominant_water
        mask_quality_warning = broad_mask_warning or (near_pct >= 85.0 and mid_pct >= 85.0) or (far_pct + mid_pct + near_pct < 1.0)

        return {
            "near_water_coverage_pct": round(near_pct, 4),
            "mid_water_coverage_pct": round(mid_pct, 4),
            "far_water_coverage_pct": round(far_pct, 4),
            "water_touches_bottom": water_touches_bottom,
            "far_water_only": far_water_only,
            "far_dominant_water": far_dominant_water,
            "broad_mask_warning": broad_mask_warning,
            "immediate_risk": immediate_risk,
            "mask_quality_warning": mask_quality_warning,
        }
    def _fusion_engine(
        self,
        water_mask: np.ndarray,
        water_coverage_pct: float,
        references: List[ReferenceObject],
        dense_depth_map: np.ndarray,
        reference_estimate: Dict[str, Any],
    ) -> Dict[str, float]:
        water_pixels = dense_depth_map[water_mask > 0]
        if water_pixels.size == 0:
            water_pixels = dense_depth_map.reshape(-1)

        h, w = water_mask.shape[:2]
        water_regions = self.water_detector.get_water_bounding_boxes(water_mask)
        largest_water_region_pct = 0.0
        largest_region_aspect = 0.0
        if water_regions:
            region_areas = [bw * bh for (_, _, bw, bh) in water_regions]
            largest_idx = int(np.argmax(region_areas))
            x, y, bw, bh = water_regions[largest_idx]
            largest_water_region_pct = round(float(bw * bh) / float(h * w) * 100.0, 4)
            largest_region_aspect = round(float(bw) / float(max(bh, 1)), 4)

        if water_regions:
            x, y, bw, bh = water_regions[0]
            largest_region = {
                "area_pct": round(float(bw * bh) / float(h * w) if h * w > 0 else 0.0, 4),
                "aspect_ratio": round(float(bw) / float(max(bh, 1)), 4),
                "bottom_aligned": ((y + bh) / float(h)) >= 0.65,
            }
        else:
            largest_region = {}

        waterline_pct = float(reference_estimate.get("waterline_pct", 0.0))
        dense_depth_cm = float(np.percentile(water_pixels, 90)) * 120.0
        region_depth_cm = self._estimate_region_depth(
            largest_region=largest_region,
            waterline_pct=waterline_pct,
            dense_depth_cm=dense_depth_cm,
        )

        zone_features = self._water_zone_features(water_mask)

        features = {
            "water_coverage_pct": round(float(water_coverage_pct), 4),
            "reference_count": float(len(references)),
            "max_reference_submersion": round(
                max((obj.water_submersion_ratio for obj in references), default=0.0),
                4,
            ),
            "dense_depth_mean": round(float(np.mean(water_pixels)), 4),
            "dense_depth_p90": round(float(np.percentile(water_pixels, 90)), 4),
            "dense_depth_p95": round(float(np.percentile(water_pixels, 95)), 4),
            "reference_depth_cm": round(float(reference_estimate.get("depth_cm", 0.0)), 2),
            "largest_water_region_pct": round(largest_water_region_pct, 4),
            "largest_water_region_aspect": largest_region_aspect,
            "waterline_pct": round(waterline_pct, 2),
            "region_depth_cm": round(region_depth_cm, 2),
            **zone_features,
        }
        return features

    def _is_waterlogged(self, depth_cm: float, coverage: float, max_reference_submersion: float) -> bool:
        return (
            depth_cm < 20.0
            and coverage < 0.35
            and max_reference_submersion < 0.5
        ) or (
            depth_cm < 12.0 and coverage < 0.45
        )

    def _calibration_severity_model(self, features: Dict[str, float]) -> Tuple[float, float, str]:
        coverage = features["water_coverage_pct"] / 100.0
        dense_depth_cm = features["dense_depth_p90"] * 120.0
        reference_depth_cm = features["reference_depth_cm"]
        reference_count = features["reference_count"]
        largest_region_pct = features.get("largest_water_region_pct", 0.0) / 100.0
        waterline_pct = features.get("waterline_pct", 0.0) / 100.0
        region_depth_cm = features.get("region_depth_cm", 0.0)
        max_reference_submersion = features.get("max_reference_submersion", 0.0)
        near_pct = features.get("near_water_coverage_pct", 0.0) / 100.0
        mid_pct = features.get("mid_water_coverage_pct", 0.0) / 100.0
        far_pct = features.get("far_water_coverage_pct", 0.0) / 100.0
        roadwide_water_evidence = coverage >= 0.60 and near_pct >= 0.35 and mid_pct >= 0.45
        multi_vehicle_bumper_evidence = (
            reference_count >= 2
            and max_reference_submersion >= 0.65
            and coverage >= 0.50
            and near_pct >= 0.30
            and mid_pct >= 0.40
        )
        strong_reference_water_evidence = roadwide_water_evidence or multi_vehicle_bumper_evidence
        single_strong_vehicle_evidence = reference_count >= 1 and max_reference_submersion >= 0.65 and strong_reference_water_evidence

        if coverage < 0.02:
            depth_cm = 0.0
        elif reference_count > 0 and reference_depth_cm > 0:
            # Reference-object estimates can overstate depth in partial-road scenes:
            # one bounding box crossing a water patch is not the same as the whole
            # road being waist-deep. Keep Depth Anything as a supporting signal,
            # then cap by scene coverage and submersion strength.
            depth_cm = (0.65 * reference_depth_cm) + (0.35 * dense_depth_cm)
            if coverage < 0.75 and max_reference_submersion < 0.70:
                depth_cm = min(depth_cm, 65.0 if single_strong_vehicle_evidence else 45.0)
            elif coverage < 0.85 and max_reference_submersion < 0.90:
                depth_cm = min(depth_cm, 65.0)
            if max_reference_submersion < 0.70 and coverage < 0.70:
                depth_cm = min(depth_cm, 65.0 if single_strong_vehicle_evidence else 55.0)
            if max_reference_submersion < 0.70 and coverage < 0.65:
                depth_cm = min(depth_cm, 65.0 if single_strong_vehicle_evidence else 45.0)
            if max_reference_submersion < 0.60 and coverage < 0.55:
                depth_cm = min(depth_cm, 35.0)
            if max_reference_submersion < 0.50 and coverage < 0.50:
                depth_cm = min(depth_cm, 25.0)
        elif region_depth_cm > 0:
            depth_cm = max(region_depth_cm, min(dense_depth_cm * 0.8, 40.0))
        else:
            # When no reference object is detected, use region-aware fallback
            # so partial road flooding still produces a plausible depth.
            if coverage >= 0.70 or largest_region_pct >= 0.25 or waterline_pct >= 0.40:
                depth_cm = max(20.0, min(dense_depth_cm * 0.9, 35.0))
            elif coverage >= 0.40 or largest_region_pct >= 0.15 or waterline_pct >= 0.30:
                depth_cm = max(12.0, min(dense_depth_cm * 0.8, 28.0))
            elif coverage >= 0.20 or largest_region_pct >= 0.08 or waterline_pct >= 0.20:
                depth_cm = max(8.0, min(dense_depth_cm * 0.75, 20.0))
            else:
                depth_cm = 3.0

        far_water_only = bool(features.get("far_water_only", False))
        far_dominant_water = bool(features.get("far_dominant_water", False))
        broad_mask_warning = bool(features.get("broad_mask_warning", False))
        immediate_risk = bool(features.get("immediate_risk", False))
        water_touches_bottom = bool(features.get("water_touches_bottom", False))
        candidate_depth_cm = features.get("efficientnet_candidate_depth_cm")
        candidate_depth_value = float(candidate_depth_cm) if candidate_depth_cm is not None else None
        shallow_model_agreement = (
            candidate_depth_value is not None
            and 8.0 <= candidate_depth_value <= 35.0
            and 8.0 <= dense_depth_cm <= 35.0
            and abs(candidate_depth_value - dense_depth_cm) <= 12.0
        )
        trace_water_evidence = (
            coverage < 0.05
            and near_pct < 0.05
            and max_reference_submersion < 0.15
            and not immediate_risk
        )
        shallow_water_gate_exception = trace_water_evidence and shallow_model_agreement
        if trace_water_evidence:
            features["trace_water_evidence"] = True
            features["low_water_gate_applied"] = not shallow_water_gate_exception
            features["shallow_water_gate_exception"] = shallow_water_gate_exception
            if shallow_water_gate_exception:
                features["low_water_gate_reason"] = "Tiny water mask, but EfficientNet and dense depth agree on shallow flood water."
            else:
                features["low_water_gate_reason"] = "Very small water mask with no near-field risk or meaningful vehicle submersion."
        full_road_water_no_reference = (
            reference_count < 1
            and coverage >= 0.70
            and near_pct >= 0.70
            and mid_pct >= 0.60
            and immediate_risk
        )
        if full_road_water_no_reference:
            features["full_road_water_no_reference"] = True
            features["review_required"] = True
            features["review_reason"] = "Road surface is broadly covered by water, but no reference object was detected for exact depth."
        strong_vehicle_submersion = multi_vehicle_bumper_evidence or (reference_count >= 2 and max_reference_submersion >= 0.70) or single_strong_vehicle_evidence
        useful_reference_evidence = reference_count >= 1 and max_reference_submersion >= 0.45
        weak_reference_evidence = not useful_reference_evidence

        # For CCTV/apartment feeds, image position is not reliable enough to be
        # a hard decision rule. Far/near gates only cap depth when there is no
        # useful object-submersion evidence.
        if shallow_water_gate_exception:
            depth_cm = max(depth_cm, min(candidate_depth_value or dense_depth_cm, dense_depth_cm, 30.0))
        elif trace_water_evidence:
            depth_cm = min(depth_cm, 5.0)
        elif far_water_only and weak_reference_evidence:
            depth_cm = min(depth_cm, 15.0)
        elif far_dominant_water and weak_reference_evidence:
            depth_cm = min(depth_cm, 20.0)
        elif broad_mask_warning:
            if full_road_water_no_reference:
                depth_cm = max(depth_cm, 45.0)
                depth_cm = min(depth_cm, 50.0)
            elif near_pct >= 0.70 and max_reference_submersion >= 0.85:
                depth_cm = min(depth_cm, 40.0)
            else:
                depth_cm = min(depth_cm, 25.0)
        elif not immediate_risk and near_pct < 0.05 and weak_reference_evidence:
            depth_cm = min(depth_cm, 20.0)
        elif near_pct < 0.12 and mid_pct < 0.20 and max_reference_submersion < 0.60:
            depth_cm = min(depth_cm, 30.0)
        elif near_pct < 0.25 and max_reference_submersion < 0.60:
            depth_cm = min(depth_cm, 45.0)

        depth_cm = float(np.clip(depth_cm, 0.0, 180.0))
        confidence = float(np.clip(0.35 + (coverage * 0.35) + (min(reference_count, 1.0) * 0.30), 0.2, 0.98))
        if bool(features.get("mask_quality_warning", False)):
            confidence = min(confidence, 0.72)

        if (far_water_only and weak_reference_evidence) or (far_dominant_water and weak_reference_evidence) or self._is_waterlogged(depth_cm, coverage, max_reference_submersion):
            action = "Monitor" if depth_cm < 10.0 else "Advisory Monitoring"
        elif not immediate_risk and depth_cm < 30.0:
            action = "Advisory Monitoring"
        elif depth_cm >= 100.0 and immediate_risk and water_touches_bottom:
            action = "Deploy Emergency Diversion"
        elif depth_cm >= 60.0 and (immediate_risk or strong_vehicle_submersion):
            action = "Activate Traffic Management"
        elif depth_cm >= 30.0:
            action = "Issue Municipal Warning"
        elif depth_cm >= 10.0:
            action = "Advisory Monitoring"
        else:
            action = "Monitor"

        return round(depth_cm, 2), round(confidence, 4), action

    def _apply_efficientnet_correction(
        self,
        depth_cm: float,
        confidence: float,
        action: str,
        features: Dict[str, float],
    ) -> Tuple[float, float, str]:
        try:
            cfg = load_settings_dict().get("inference", {}).get("efficientnet_signal", {})
        except Exception:
            cfg = {}

        if not bool(cfg.get("apply_corrections", False)):
            return depth_cm, confidence, action

        candidate_depth = features.get("efficientnet_candidate_depth_cm")
        if candidate_depth is None:
            return depth_cm, confidence, action

        candidate_depth = float(candidate_depth)
        coverage = float(features.get("water_coverage_pct", 0.0)) / 100.0
        near_pct = float(features.get("near_water_coverage_pct", 0.0)) / 100.0
        immediate_risk = bool(features.get("immediate_risk", False))
        delta = candidate_depth - depth_cm

        severe_underestimate = (
            candidate_depth >= 60.0
            and depth_cm < 50.0
            and delta >= 25.0
            and coverage >= 0.20
            and near_pct >= 0.10
            and immediate_risk
        )
        extreme_depth_underestimate = (
            candidate_depth >= 100.0
            and depth_cm < 60.0
            and delta >= 50.0
            and coverage >= 0.08
            and near_pct >= 0.08
            and immediate_risk
            and not bool(features.get("low_water_gate_applied", False))
        )
        if not (severe_underestimate or extreme_depth_underestimate):
            return depth_cm, confidence, action

        corrected_depth = float(np.clip(candidate_depth, 0.0, 180.0))
        corrected_confidence = max(confidence, min(0.84, confidence + 0.12))
        if corrected_depth >= 80.0:
            corrected_action = "Deploy Emergency Diversion" if immediate_risk else "Activate Traffic Management"
        elif corrected_depth >= 60.0:
            corrected_action = "Activate Traffic Management"
        elif corrected_depth >= 30.0:
            corrected_action = "Issue Municipal Warning"
        else:
            corrected_action = action

        features["efficientnet_correction_applied"] = True
        features["pre_correction_depth_cm"] = round(depth_cm, 2)
        features["correction_reason"] = "candidate_extreme_depth_with_visible_risk" if extreme_depth_underestimate else "candidate_severe_depth_with_visible_water"
        return round(corrected_depth, 2), round(corrected_confidence, 4), corrected_action

    def _action_for_final_depth(self, depth_cm: float, features: Dict[str, Any], fallback_action: str) -> str:
        immediate_risk = bool(features.get("immediate_risk", False))
        if depth_cm >= 100.0 and immediate_risk:
            return "Deploy Emergency Diversion"
        if depth_cm >= 60.0 and immediate_risk:
            return "Activate Traffic Management"
        if depth_cm >= 30.0:
            return "Issue Municipal Warning"
        if depth_cm >= 10.0:
            return "Advisory Monitoring"
        if depth_cm > 0.0:
            return "Monitor"
        return fallback_action

    def _record_model_agreement(
        self,
        final_depth_cm: float,
        confidence: float,
        action: str,
        features: Dict[str, Any],
    ) -> Tuple[float, float, str]:
        coverage = float(features.get("water_coverage_pct", 0.0)) / 100.0
        near_pct = float(features.get("near_water_coverage_pct", 0.0)) / 100.0
        immediate_risk = bool(features.get("immediate_risk", False))
        low_water_gate = bool(features.get("low_water_gate_applied", False))
        muddy_fallback = bool(features.get("muddy_water_fallback_applied", False))
        max_submersion = float(features.get("max_reference_submersion", 0.0))
        reference_count = int(float(features.get("reference_count", 0.0)))

        signals: List[Dict[str, Any]] = []

        def add_signal(name: str, depth: Any, trusted: bool, reason: str, weight: float) -> None:
            if depth is None:
                return
            try:
                value = float(depth)
            except (TypeError, ValueError):
                return
            if not np.isfinite(value):
                return
            signals.append({"name": name, "depth_cm": round(value, 2), "trusted": bool(trusted), "reason": reason, "weight": float(weight)})

        calibration_depth = features.get("pre_correction_depth_cm", features.get("calibration_depth_cm", final_depth_cm))
        add_signal("fusion_calibration", calibration_depth, True, "combined water/depth/reference calibration", 0.25)

        candidate_depth = features.get("efficientnet_candidate_depth_cm")
        shallow_exception = bool(features.get("shallow_water_gate_exception", False))
        candidate_trusted = (not low_water_gate or shallow_exception) and (
            shallow_exception or muddy_fallback or immediate_risk or coverage >= 0.08 or (candidate_depth is not None and float(candidate_depth) < 15.0)
        )
        add_signal("efficientnet_candidate", candidate_depth, candidate_trusted, "trained depth model", 0.50)

        reference_depth = features.get("reference_depth_cm")
        reference_trusted = reference_count > 0 and not low_water_gate and (max_submersion >= 0.15 or coverage >= 0.20 or immediate_risk)
        add_signal("reference_objects", reference_depth, reference_trusted, "object/reference depth estimate", 0.15)

        dense_depth = float(features.get("dense_depth_p90", 0.0)) * 120.0
        dense_trusted = (not low_water_gate and (coverage >= 0.20 or near_pct >= 0.10 or immediate_risk)) or shallow_exception
        add_signal("depth_anything_dense", dense_depth, dense_trusted, "monocular dense-depth support", 0.10)

        trusted = [signal for signal in signals if signal["trusted"]]
        tolerance_cm = 25.0
        best_cluster: List[Dict[str, Any]] = []

        candidate_signal = next((signal for signal in trusted if signal["name"] == "efficientnet_candidate"), None)
        if candidate_signal is not None:
            candidate_cluster = [signal for signal in trusted if abs(signal["depth_cm"] - candidate_signal["depth_cm"]) <= tolerance_cm]
            if len(candidate_cluster) >= 2:
                best_cluster = candidate_cluster

        if not best_cluster:
            for signal in trusted:
                cluster = [candidate for candidate in trusted if abs(candidate["depth_cm"] - signal["depth_cm"]) <= tolerance_cm]
                if len(cluster) > len(best_cluster):
                    best_cluster = cluster

        output_depth = float(final_depth_cm)
        output_confidence = float(confidence)
        output_action = action

        if low_water_gate:
            status = "low_water_gate"
            agreed_depth = final_depth_cm
            reason = str(features.get("low_water_gate_reason", "Low-water evidence overruled larger depth signals."))
        elif len(best_cluster) >= 2:
            total_weight = sum(item["weight"] for item in best_cluster) or 1.0
            agreed_depth = sum(item["depth_cm"] * item["weight"] for item in best_cluster) / total_weight
            names = ", ".join(item["name"] for item in best_cluster)
            status = "agreement"
            reason = f"Trusted signals agree within {tolerance_cm:.0f} cm: {names}."

            if candidate_signal is not None:
                candidate_value = float(candidate_signal["depth_cm"])
                candidate_gap = candidate_value - float(final_depth_cm)
                should_prefer_candidate = (
                    candidate_gap >= 12.0
                    and candidate_value >= 45.0
                    and (muddy_fallback or (immediate_risk and coverage >= 0.20) or max_submersion >= 0.35)
                )
                if should_prefer_candidate:
                    output_depth = candidate_value
                    output_confidence = max(output_confidence, min(0.88, output_confidence + 0.10))
                    output_action = self._action_for_final_depth(output_depth, features, action)
                    status = "candidate_selected_by_agreement"
                    reason = (
                        f"Trained EfficientNet depth was trusted and fusion was {candidate_gap:.1f} cm lower; "
                        f"final depth uses EfficientNet with support from: {names}."
                    )
                elif abs(agreed_depth - final_depth_cm) >= 15.0:
                    output_depth = float(np.clip(agreed_depth, 0.0, 180.0))
                    output_confidence = max(output_confidence, min(0.84, output_confidence + 0.06))
                    output_action = self._action_for_final_depth(output_depth, features, action)
                    status = "weighted_model_agreement"
                    reason = f"Final depth uses weighted agreement from trusted signals: {names}."
        elif features.get("efficientnet_correction_applied"):
            agreed_depth = final_depth_cm
            status = "candidate_corrected_with_evidence"
            reason = str(features.get("correction_reason", "trained candidate corrected the calibration estimate"))
        else:
            agreed_depth = final_depth_cm
            status = "weak_agreement"
            reason = "No two trusted model signals were close enough; final output used conservative calibration."

        output_depth = round(float(np.clip(output_depth, 0.0, 180.0)), 2)
        output_confidence = round(float(np.clip(output_confidence, 0.0, 0.98)), 4)

        features["model_signals"] = signals
        features["model_agreement_status"] = status
        features["model_cluster_depth_cm"] = round(float(agreed_depth), 2)
        features["model_agreement_depth_cm"] = output_depth
        features["model_agreement_tolerance_cm"] = tolerance_cm
        features["final_aggregation_source"] = "model_agreement_engine"
        features["final_output_reason"] = reason
        return output_depth, output_confidence, output_action
    def predict(self, image_rgb: np.ndarray) -> Dict[str, Any]:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("predict expects an RGB image array with shape (H, W, 3)")

        trace: List[Dict[str, str]] = []

        water_mask, water_coverage_pct = self._segformer_water_mask(image_rgb)
        muddy_water_fallback_applied = False
        trace.append(
            {
                "stage": "SegFormer",
                "backend": "classical-water-detector",
                "status": "ok",
                "summary": f"water_coverage={water_coverage_pct:.2f}%",
            }
        )

        #Edit_start
        no_water_probability = self._no_water_guard_signal(image_rgb)
        #Edit_end

        efficientnet_depth_cm = self._efficientnet_depth_signal(image_rgb)
        if efficientnet_depth_cm is not None:
            trace.append(
                {
                    "stage": "EfficientNet Candidate",
                    "backend": self._efficientnet_backend,
                    "status": "ok",
                    "summary": f"candidate_depth_cm={efficientnet_depth_cm:.2f}",
                }
            )

        if (
            efficientnet_depth_cm is not None
            and efficientnet_depth_cm >= 35.0
            and water_coverage_pct < 5.0
            #Edit_start
            and not self.water_detector.looks_like_dry_land(image_rgb)
            and not (no_water_probability is not None and no_water_probability >= 0.92)
            #Edit_end
        ):
            muddy_mask = self.water_detector._detect_muddy_floodwater(image_rgb)
            muddy_coverage_pct = float((muddy_mask > 0).mean() * 100.0)
            if muddy_coverage_pct >= 15.0:
                water_mask = (muddy_mask > 0).astype(np.uint8) * 255
                water_coverage_pct = muddy_coverage_pct
                muddy_water_fallback_applied = True
                trace.append(
                    {
                        "stage": "Muddy Water Fallback",
                        "backend": "candidate-gated-brown-water-mask",
                        "status": "ok",
                        "summary": f"water_coverage={water_coverage_pct:.2f}%",
                    }
                )
        #Edit_start
        no_water_status = "disabled" if self._no_water_model is None else "ok"
        no_water_summary = "checkpoint unavailable"
        if no_water_probability is not None:
            no_water_summary = f"no_water_probability={no_water_probability:.3f}"
        trace.append(
            {
                "stage": "No-Water Guard",
                "backend": self._no_water_backend,
                "status": no_water_status,
                "summary": no_water_summary,
            }
        )
        #Edit_end
        references, ref_backend = self._yolov8_reference_stage(image_rgb, water_mask)
        trace.append(
            {
                "stage": "YOLOv8",
                "backend": ref_backend,
                "status": "ok",
                "summary": f"reference_objects={len(references)}",
            }
        )

        dense_depth_map = self._depth_anything_v2_dense_map(image_rgb, water_mask)
        trace.append(
            {
                "stage": "Depth Anything V2",
                "backend": self._depth_backend,
                "status": "ok",
                "summary": f"dense_p90={float(np.percentile(dense_depth_map, 90)):.3f}",
            }
        )

        reference_estimate = self.reference_estimator.estimate(image_rgb)
        features = self._fusion_engine(
            water_mask=water_mask,
            water_coverage_pct=water_coverage_pct,
            references=references,
            dense_depth_map=dense_depth_map,
            reference_estimate=reference_estimate,
        )
        if efficientnet_depth_cm is not None:
            features["efficientnet_candidate_depth_cm"] = efficientnet_depth_cm
            features["fusion_candidate_delta_cm"] = round(abs(float(features.get("region_depth_cm", 0.0)) - efficientnet_depth_cm), 2)
        #Edit_start
        features["no_water_probability"] = no_water_probability
        features["no_water_guard_backend"] = self._no_water_backend
        #Edit_end
        trace.append(
            {
                "stage": "Fusion Engine",
                "backend": "feature-fusion-v1",
                "status": "ok",
                "summary": (
                    f"coverage={features['water_coverage_pct']:.2f}% "
                    f"near={features['near_water_coverage_pct']:.2f}% "
                    f"refs={int(features['reference_count'])} p90={features['dense_depth_p90']:.3f}"
                ),
            }
        )

        depth_cm, confidence, action = self._calibration_severity_model(features)
        features["calibration_depth_cm"] = round(depth_cm, 2)
        if efficientnet_depth_cm is not None:
            features["final_candidate_delta_cm"] = round(abs(depth_cm - efficientnet_depth_cm), 2)
        depth_cm, confidence, action = self._apply_efficientnet_correction(depth_cm, confidence, action, features)
        depth_cm, confidence, action = self._record_model_agreement(depth_cm, confidence, action, features)
        depth_cm, confidence, action = self._apply_residual_fusion_model(depth_cm, confidence, action, features)
        #Edit_start
        depth_cm, confidence, action = self._apply_dry_land_guard(depth_cm, confidence, action, image_rgb, features)
        depth_cm, confidence, action = self._apply_no_water_guard(depth_cm, confidence, action, features)
        if features.get("dry_land_guard_applied"):
            trace.append(
                {
                    "stage": "Dry-Land Decision",
                    "backend": "cracked-land-heuristic",
                    "status": "applied",
                    "summary": "dry lower scene with no near-field water forced depth=0.00 cm",
                }
            )
        elif features.get("no_water_guard_applied"):
            trace.append(
                {
                    "stage": "No-Water Decision",
                    "backend": self._no_water_backend,
                    "status": "applied",
                    "summary": "dry-scene corroboration forced depth=0.00 cm",
                }
            )
        #Edit_end
        if features.get("residual_fusion_status") in {"applied", "skipped_low_water_gate"}:
            trace.append(
                {
                    "stage": "Residual Fusion Model",
                    "backend": self._residual_fusion_backend,
                    "status": str(features.get("residual_fusion_status")),
                    "summary": (
                        f"depth_cm={depth_cm:.2f} "
                        f"delta={float(features.get('residual_fusion_delta_cm', 0.0)):.2f}"
                    ),
                }
            )
        severity = _depth_to_severity(depth_cm, features)
        trace.append(
            {
                "stage": "Calibration/Severity Model",
                "backend": "calibration-v1",
                "status": "ok",
                "summary": f"depth_cm={depth_cm:.2f} severity={severity['level']}",
            }
        )

        top_refs = references[:2]
        ref_cues = [
            f"{obj.label} submersion={obj.water_submersion_ratio:.2f} conf={obj.confidence:.2f}"
            for obj in top_refs
        ]
        stage_cues = [f"{step['stage']}: {step['summary']}" for step in trace]
        if features.get("full_road_water_no_reference"):
            stage_cues.append("Road coverage gate: full-road water with no reference object; depth kept conservative and marked for review")
        if features.get("low_water_gate_applied"):
            stage_cues.append("Low-water gate: tiny water mask with no near-field risk; depth capped as shallow trace water")
        if features.get("final_output_reason"):
            stage_cues.append(f"Model Agreement: {features.get('model_agreement_status')} - {features.get('final_output_reason')}")
        visual_cues = stage_cues + ref_cues

        return {
            "depth_cm": depth_cm,
            "confidence": confidence,
            "severity": severity,
            "method": "segformer_yolov8_depthv2_fusion",
            "visual_cues": visual_cues,
            "label_guide": reference_estimate.get("label_guide", ""),
            "waterline_pct": reference_estimate.get("waterline_pct", 0.0),
            "water_coverage": round(water_coverage_pct / 100.0, 4),
            "action_trigger": action,
            "structured_features": features,
            "pipeline_trace": trace,
        }


_PIPELINE: Optional[SegformerYoloDepthV2Pipeline] = None


def get_segformer_yolo_depthv2_pipeline() -> SegformerYoloDepthV2Pipeline:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = SegformerYoloDepthV2Pipeline()
    return _PIPELINE
