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
    from src.depth_teachers import TeacherEnsemble
except Exception:  # pragma: no cover - optional runtime dependency
    TeacherEnsemble = None

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

class MaskConditionedFusionDepthModel(nn.Module):
    """Checkpoint-compatible loader for FloodDepth-MaskConditionedFusion.pth."""

    def __init__(self) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:  # pragma: no cover - optional model dependency
            raise RuntimeError("timm is required for MaskConditionedFusionDepthModel") from exc
        self.backbone = timm.create_model("efficientnetv2_rw_s", pretrained=False, num_classes=0, global_pool="avg")
        self.vis_proj = nn.Sequential(nn.Linear(1792, 256), nn.ReLU(), nn.LayerNorm(256))
        self.obj_mlp = nn.Sequential(nn.Linear(24, 64), nn.ReLU(), nn.LayerNorm(64), nn.Linear(64, 128), nn.ReLU(), nn.LayerNorm(128))
        self.geo_mlp = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.LayerNorm(32), nn.Linear(32, 64), nn.ReLU(), nn.LayerNorm(64))
        self.fusion = nn.Sequential(nn.Linear(448, 384), nn.ReLU(), nn.LayerNorm(384), nn.Dropout(0.10), nn.Linear(384, 256), nn.ReLU(), nn.LayerNorm(256))
        self.depth_head = nn.Linear(256, 1)
        self.ordinal_head = nn.Linear(256, 5)

    def forward(self, image: torch.Tensor, object_features: torch.Tensor, geometry_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        visual = self.vis_proj(self.backbone(image))
        object_embedding = self.obj_mlp(object_features)
        geometry_embedding = self.geo_mlp(geometry_features)
        fused = self.fusion(torch.cat([visual, object_embedding, geometry_embedding], dim=1))
        return self.depth_head(fused), self.ordinal_head(fused)

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
        return {"level": "HIGH", "label": "High flood ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¾ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¾Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â¦ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦Ãƒâ€šÃ‚Â¡ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â avoid travel", "color": "#dc2626", "stage": 4}
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
        self._mask_conditioned_fusion_model = None
        self._mask_conditioned_fusion_transform = None
        self._mask_conditioned_fusion_device = torch.device("cpu")
        self._mask_conditioned_fusion_backend = "disabled"
        self._mask_conditioned_fusion_target_transform = "log1p"
        self._no_water_model = None
        self._no_water_transform = None
        self._no_water_device = torch.device("cpu")
        self._no_water_backend = "disabled"
        self._wet_road_guard_model = None
        self._wet_road_guard_transform = None
        self._wet_road_guard_backend = "disabled"
        self._residual_fusion_model = None
        self._residual_fusion_backend = "disabled"
        self._residual_fusion_device = torch.device("cpu")
        self._residual_fusion_feature_names = RESIDUAL_FUSION_FEATURE_NAMES
        self._residual_fusion_feature_mean = None
        self._residual_fusion_feature_std = None
        self._teacher_ensemble = None
        self._teacher_backend = "disabled"
        self.gemma_semantic_analyzer = None
        self._load_yolo_if_available()
        self._load_object_detector_if_available()
        self._load_depth_anything_if_available()
        self._load_efficientnet_signal_if_available()
        self._load_mask_conditioned_fusion_if_available()
        self._load_no_water_guard_if_available()
        self._load_residual_fusion_if_available()
        self._load_depth_teachers_if_available()
        self._load_gemma_semantic_analyzer_if_available()

    def _load_gemma_semantic_analyzer_if_available(self) -> None:
        try:
            from src.gemma_semantic_analyzer import GemmaSemanticAnalyzer
            cfg = load_settings_dict().get("inference", {}).get("gemma_semantics", {})
            enabled = bool(cfg.get("enabled", True))
            model_name = str(cfg.get("model", "gemma3:4b"))
            ollama_url = str(cfg.get("ollama_url", "http://localhost:11434"))
            timeout_seconds = float(cfg.get("timeout_seconds", 10.0))

            self.gemma_semantic_analyzer = GemmaSemanticAnalyzer(
                model_name=model_name,
                ollama_url=ollama_url,
                timeout_seconds=timeout_seconds,
                enabled=enabled,
            )
            logger.info("Initialized GemmaSemanticAnalyzer (Ollama model: %s)", model_name)
        except Exception as exc:
            logger.warning("Could not initialize GemmaSemanticAnalyzer: %s", exc)

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


    def _load_mask_conditioned_fusion_if_available(self) -> None:
        try:
            cfg = load_settings_dict().get("inference", {}).get("mask_conditioned_fusion_signal", {})
        except Exception as exc:
            logger.info("Mask-conditioned fusion config unavailable: %s", exc)
            return

        if not bool(cfg.get("enabled", False)):
            return

        model_path = Path(str(cfg.get("model_path", "models/FloodDepth-MaskConditionedFusion.pth")))
        if not model_path.exists():
            logger.warning("Mask-conditioned fusion checkpoint missing at %s", model_path)
            self._mask_conditioned_fusion_backend = "unavailable"
            return

        try:
            device = torch.device("cuda" if torch.cuda.is_available() and str(cfg.get("device", "cpu")) == "cuda" else "cpu")
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            model = MaskConditionedFusionDepthModel().to(device)
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            self._mask_conditioned_fusion_model = model
            self._mask_conditioned_fusion_device = device
            self._mask_conditioned_fusion_backend = str(model_path)
            self._mask_conditioned_fusion_target_transform = str(checkpoint.get("target_transform", "log1p")) if isinstance(checkpoint, dict) else "log1p"
            image_size = int(checkpoint.get("image_size", cfg.get("image_size", 384))) if isinstance(checkpoint, dict) else int(cfg.get("image_size", 384))
            self._mask_conditioned_fusion_transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            logger.info("Loaded mask-conditioned fusion depth signal from %s", model_path)
        except Exception as exc:
            logger.warning("Mask-conditioned fusion signal unavailable: %s", exc)
            self._mask_conditioned_fusion_model = None
            self._mask_conditioned_fusion_transform = None
            self._mask_conditioned_fusion_backend = "unavailable"

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
            device = torch.device("cuda" if torch.cuda.is_available() and str(cfg.get("device", "cpu")) == "cuda" else "cpu")
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
            wet_road_path_value = cfg.get("wet_road_guard_model_path", cfg.get("secondary_model_path"))
            if bool(cfg.get("wet_road_guard_enabled", bool(wet_road_path_value))) and wet_road_path_value:
                wet_road_path = Path(str(wet_road_path_value))
                if wet_road_path.exists() and wet_road_path.resolve() != model_path.resolve():
                    wet_road_model = self._build_no_water_model().to(device)
                    wet_road_checkpoint = torch.load(wet_road_path, map_location=device, weights_only=True)
                    wet_road_state_dict = wet_road_checkpoint.get("model_state_dict", wet_road_checkpoint)
                    wet_road_model.load_state_dict(wet_road_state_dict, strict=True)
                    wet_road_model.eval()
                    self._wet_road_guard_model = wet_road_model
                    self._wet_road_guard_transform = self._no_water_transform
                    self._wet_road_guard_backend = str(wet_road_path)
                    logger.info("Loaded wet-road no-water guard from %s", wet_road_path)
        except Exception as exc:
            logger.warning("No-water guard unavailable: %s", exc)
            self._no_water_model = None
            self._no_water_transform = None
            self._no_water_backend = "unavailable"
    def _load_depth_teachers_if_available(self) -> None:
        try:
            cfg = load_settings_dict().get("inference", {}).get("depth_teachers", {})
        except Exception as exc:
            logger.info("Depth teacher config unavailable: %s", exc)
            return

        if not bool(cfg.get("enabled", False)):
            return
        if TeacherEnsemble is None:
            self._teacher_backend = "unavailable: import failed"
            return

        try:
            device = str(cfg.get("device", load_settings_dict().get("inference", {}).get("device", "cpu")))
            self._teacher_ensemble = TeacherEnsemble(
                device=device,
                lazy_load=bool(cfg.get("lazy_load", True)),
                allow_download=bool(cfg.get("allow_download", True)),
                use_fp16=bool(cfg.get("use_fp16", False)),
            )
            self._teacher_backend = "DepthAnythingV2+DepthPro+Metric3D"
            logger.info("Depth teacher ensemble enabled")
        except Exception as exc:
            logger.warning("Depth teacher ensemble unavailable: %s", exc)
            self._teacher_ensemble = None
            self._teacher_backend = f"unavailable: {exc}"

    def _depth_teacher_features(self, image_rgb: np.ndarray, water_mask: np.ndarray) -> Dict[str, Any]:
        if self._teacher_ensemble is None:
            return {}
        try:
            return self._teacher_ensemble.predict(image_rgb, water_mask=(water_mask > 0))
        except Exception as exc:
            logger.warning("Depth teacher feature extraction failed: %s", exc)
            return {
                "water_region_valid": False,
                "teachers": {},
                "ensemble": {},
                "meta": {"available_teacher_count": 0, "total_teachers": 3, "error": str(exc)},
            }

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

        try:
            guard_cfg = load_settings_dict().get("inference", {}).get("no_water_guard", {})
        except Exception:
            guard_cfg = {}
        no_water_probability = self._feature_float(features.get("no_water_probability"))
        wet_road_probability = self._feature_float(
            features.get("wet_road_no_water_probability", features.get("secondary_no_water_probability"))
        )
        no_water_threshold = float(guard_cfg.get("no_water_threshold", 0.99))
        wet_road_threshold = float(guard_cfg.get("wet_road_guard_threshold", guard_cfg.get("secondary_override_threshold", 1.0)))
        coverage_pct = float(features.get("water_coverage_pct", 0.0))
        near_coverage_pct = float(features.get("near_water_coverage_pct", 0.0))
        reference_submersion = float(features.get("max_reference_submersion", 0.0))
        no_water_guard_high = no_water_probability >= no_water_threshold or wet_road_probability >= wet_road_threshold
        low_risk_no_water_scene = (
            coverage_pct <= 1.0
            and near_coverage_pct <= 1.0
            and reference_submersion < 0.20
            and not bool(features.get("immediate_risk", False))
            and not bool(features.get("muddy_water_fallback_applied", False))
        )
        if no_water_guard_high and low_risk_no_water_scene:
            features["residual_fusion_status"] = "skipped_no_water_guard_candidate"
            return depth_cm, confidence, action
        skip_low_water = bool(cfg.get("skip_low_water_gate", True))
        if skip_low_water and bool(features.get("low_water_gate_applied", False)) and not bool(features.get("shallow_water_gate_exception", False)):
            features["residual_fusion_status"] = "skipped_low_water_gate"
            return depth_cm, confidence, action

        gemma_feats = features.get("gemma_semantic_features") or {}
        gemma_flooded = bool(gemma_feats.get("water_present", False)) and str(gemma_feats.get("scene_type", "")).lower() in ("flooded_road", "flooded_indoor")
        if bool(features.get("no_reference_depth_uncertain", False)) and not gemma_flooded:
            features["residual_fusion_status"] = "skipped_no_reference_depth_uncertain"
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

    def _mask_conditioned_fusion_depth_signal(self, image_rgb: np.ndarray, features: Dict[str, Any]) -> Optional[float]:
        if self._mask_conditioned_fusion_model is None or self._mask_conditioned_fusion_transform is None:
            return None
        object_features = self._mask_conditioned_object_features(features)
        geometry_features = self._mask_conditioned_geometry_features(features)
        image = Image.fromarray(image_rgb.astype(np.uint8), mode="RGB")
        tensor = self._mask_conditioned_fusion_transform(image).unsqueeze(0).to(self._mask_conditioned_fusion_device)
        obj_tensor = torch.tensor(object_features.reshape(1, -1), dtype=torch.float32, device=self._mask_conditioned_fusion_device)
        geo_tensor = torch.tensor(geometry_features.reshape(1, -1), dtype=torch.float32, device=self._mask_conditioned_fusion_device)
        try:
            with torch.no_grad():
                depth_raw, ordinal_raw = self._mask_conditioned_fusion_model(tensor, obj_tensor, geo_tensor)
            raw_depth = float(depth_raw.squeeze().item())
            if self._mask_conditioned_fusion_target_transform.lower() == "log1p":
                depth_cm = float(np.expm1(raw_depth))
            else:
                depth_cm = raw_depth
            features["mask_conditioned_fusion_ordinal_class"] = int(torch.argmax(ordinal_raw, dim=1).item())
            return round(float(np.clip(depth_cm, 0.0, 180.0)), 2)
        except Exception as exc:
            logger.warning("Mask-conditioned fusion inference failed: %s", exc)
            return None

    def _mask_conditioned_object_features(self, features: Dict[str, Any]) -> np.ndarray:
        values = np.zeros(24, dtype=np.float32)
        reference_count = float(features.get("reference_count", 0.0))
        values[0] = min(reference_count / 10.0, 1.0)
        values[1] = float(features.get("max_reference_submersion", 0.0))
        values[2] = min(float(features.get("reference_depth_cm", 0.0)) / 180.0, 1.0)
        values[3] = min(float(features.get("waterline_pct", 0.0)) / 100.0, 1.0)
        values[4] = 1.0 if bool(features.get("immediate_risk", False)) else 0.0
        values[5] = 1.0 if bool(features.get("far_water_only", False)) else 0.0
        values[6] = 1.0 if bool(features.get("mask_quality_warning", False)) else 0.0
        values[7] = 1.0 if bool(features.get("low_water_gate_applied", False)) else 0.0
        values[8] = 1.0 if bool(features.get("muddy_water_fallback_applied", False)) else 0.0
        values[9] = 1.0 if bool(features.get("full_road_water_no_reference", False)) else 0.0
        values[10] = min(float(features.get("dense_depth_p90", 0.0)), 1.0)
        values[11] = min(float(features.get("dense_depth_p95", 0.0)), 1.0)
        values[12] = min(float(features.get("region_depth_cm", 0.0)) / 180.0, 1.0)
        values[13] = min(float(features.get("largest_water_region_pct", 0.0)) / 100.0, 1.0)
        values[14] = min(float(features.get("teacher_agreement", 0.0)), 1.0)
        values[15] = min(float(features.get("teacher_median", 0.0)), 1.0)
        values[16] = min(float(features.get("efficientnet_candidate_depth_cm", 0.0)) / 180.0, 1.0)
        return values

    def _mask_conditioned_geometry_features(self, features: Dict[str, Any]) -> np.ndarray:
        return np.asarray(
            [
                min(float(features.get("water_coverage_pct", 0.0)) / 100.0, 1.0),
                min(float(features.get("near_water_coverage_pct", 0.0)) / 100.0, 1.0),
                min(float(features.get("far_water_coverage_pct", 0.0)) / 100.0, 1.0),
            ],
            dtype=np.float32,
        )

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

    def _wet_road_no_water_guard_signal(self, image_rgb: np.ndarray) -> Optional[float]:
        if self._wet_road_guard_model is None or self._wet_road_guard_transform is None:
            return None
        image = Image.fromarray(image_rgb.astype(np.uint8), mode="RGB")
        tensor = self._wet_road_guard_transform(image).unsqueeze(0).to(self._no_water_device)
        try:
            with torch.no_grad():
                probabilities = torch.softmax(self._wet_road_guard_model(tensor), dim=1)
            return float(np.clip(probabilities[0, 0].item(), 0.0, 1.0))
        except Exception as exc:
            logger.warning("Wet-road no-water guard inference failed: %s", exc)
            return None

    def _apply_no_water_guard(
        self,
        depth_cm: float,
        confidence: float,
        action: str,
        features: Dict[str, Any],
    ) -> Tuple[float, float, str]:
        probability = features.get("no_water_probability")
        wet_road_probability = features.get("wet_road_no_water_probability", features.get("secondary_no_water_probability"))
        if probability is None and wet_road_probability is None:
            return depth_cm, confidence, action

        try:
            cfg = load_settings_dict().get("inference", {}).get("no_water_guard", {})
        except Exception:
            cfg = {}

        primary_probability = float(probability) if probability is not None else 0.0
        threshold = float(cfg.get("no_water_threshold", 0.92))
        wet_road_threshold = float(cfg.get("wet_road_guard_threshold", cfg.get("secondary_override_threshold", 1.0)))
        wet_road_enabled = bool(cfg.get("wet_road_guard_enabled", cfg.get("secondary_override_enabled", False)))
        max_coverage_pct = float(cfg.get("max_water_coverage_pct", 3.0))
        max_near_coverage_pct = float(cfg.get("max_near_water_coverage_pct", max_coverage_pct))
        max_reference_submersion = float(cfg.get("max_no_water_reference_submersion", 0.12))
        wet_road_max_coverage_pct = float(cfg.get("wet_road_guard_max_water_coverage_pct", max_coverage_pct))
        wet_road_max_near_coverage_pct = float(cfg.get("wet_road_guard_max_near_water_coverage_pct", max_near_coverage_pct))

        coverage_pct = float(features.get("water_coverage_pct", 0.0))
        near_coverage_pct = float(features.get("near_water_coverage_pct", 0.0))
        far_coverage_pct = float(features.get("far_water_coverage_pct", 0.0))
        reference_submersion = float(features.get("max_reference_submersion", 0.0))
        has_meaningful_submersion = reference_submersion >= max_reference_submersion
        common_low_risk_scene = (
            not bool(features.get("immediate_risk", False))
            and not bool(features.get("muddy_water_fallback_applied", False))
            and not has_meaningful_submersion
        )

        primary_match = probability is not None and primary_probability >= threshold
        wet_road_match = wet_road_enabled and wet_road_probability is not None and float(wet_road_probability) >= wet_road_threshold
        primary_visual_ok = coverage_pct <= max_coverage_pct and near_coverage_pct <= max_near_coverage_pct and common_low_risk_scene
        wet_road_visual_ok = coverage_pct <= wet_road_max_coverage_pct and near_coverage_pct <= wet_road_max_near_coverage_pct and common_low_risk_scene
        primary_corroborated = primary_match and primary_visual_ok
        wet_road_corroborated = wet_road_match and wet_road_visual_ok
        background_mask_only = (
            primary_match
            and wet_road_match
            and near_coverage_pct <= max_near_coverage_pct
            and far_coverage_pct >= 80.0
            and reference_submersion < max_reference_submersion
            and int(features.get("reference_count", 0)) == 0
            and common_low_risk_scene
        )
        corroborated = primary_corroborated or wet_road_corroborated or background_mask_only

        features["primary_no_water_guard_status"] = self._guard_status(probability, primary_match, primary_corroborated)
        features["primary_no_water_guard_corroborated"] = bool(primary_corroborated)
        features["wet_road_guard_status"] = self._guard_status(wet_road_probability, wet_road_match, wet_road_corroborated)
        features["wet_road_guard_corroborated"] = bool(wet_road_corroborated)
        features["wet_road_guard_low_risk_scene"] = bool(wet_road_visual_ok)
        features["background_mask_no_water_override"] = bool(background_mask_only)
        features["no_water_guard_status"] = "applied" if corroborated else "uncertain"
        features["no_water_guard_corroborated"] = bool(corroborated)
        features["secondary_no_water_override"] = bool(wet_road_corroborated)
        features["no_water_guard_blocked_by_flood_evidence"] = bool((primary_match or wet_road_match) and not corroborated)

        if not corroborated:
            return depth_cm, confidence, action

        use_wet_road_guard = wet_road_corroborated and not primary_corroborated
        decision_source = "background-mask no-water guard" if background_mask_only else ("wet-road no-water guard" if use_wet_road_guard else "no-water guard")
        decision_probability = float(wet_road_probability) if use_wet_road_guard and wet_road_probability is not None else primary_probability
        decision_backend = self._wet_road_guard_backend if use_wet_road_guard else self._no_water_backend
        features["no_water_guard_applied"] = True
        features["water_present_overridden_by_no_water_guard"] = True
        features["no_water_decision_source"] = decision_source
        features["no_water_decision_backend"] = decision_backend
        features["final_output_reason"] = (
            f"{decision_source} detected a low-risk dry/wet-road scene with probability {decision_probability:.2f}; "
            "water/depth signals were suppressed."
        )
        features["final_aggregation_source"] = "no_water_guard"
        features["review_required"] = False
        features["review_reason"] = ""
        return 0.0, round(float(max(confidence, decision_probability)), 4), self._action_for_final_depth(0.0, features, action)

    @staticmethod
    def _guard_status(probability: Any, matched: bool, corroborated: bool) -> str:
        if probability is None:
            return "unavailable"
        if corroborated:
            return "applied"
        if matched:
            return "blocked_by_flood_evidence"
        return "below_threshold"
    def _apply_strong_deep_flood_correction(
        self,
        depth_cm: float,
        confidence: float,
        action: str,
        features: Dict[str, Any],
    ) -> Tuple[float, float, str]:
        try:
            cfg = load_settings_dict().get("inference", {}).get("strong_deep_flood_correction", {})
        except Exception:
            cfg = {}
        if not bool(cfg.get("enabled", False)):
            return depth_cm, confidence, action

        coverage = float(features.get("water_coverage_pct", 0.0))
        near = float(features.get("near_water_coverage_pct", 0.0))
        references = float(features.get("reference_count", 0.0))
        submersion = float(features.get("max_reference_submersion", 0.0))
        reference_depth = float(features.get("reference_depth_cm", 0.0))
        region_depth = float(features.get("region_depth_cm", 0.0))
        strong_evidence = (
            coverage >= float(cfg.get("min_water_coverage_pct", 60.0))
            and near >= float(cfg.get("min_near_water_coverage_pct", 45.0))
            and references >= float(cfg.get("min_reference_count", 1))
            and submersion >= float(cfg.get("min_reference_submersion", 0.80))
            and reference_depth >= float(cfg.get("min_reference_depth_cm", 85.0))
            and region_depth >= float(cfg.get("min_region_depth_cm", 65.0))
        )
        if not strong_evidence:
            return depth_cm, confidence, action

        evidence_depth = (0.65 * reference_depth) + (0.35 * region_depth)
        corrected_depth = round(float(np.clip(max(depth_cm, evidence_depth), 0.0, 180.0)), 2)
        if corrected_depth <= depth_cm:
            return depth_cm, confidence, action
        features["strong_deep_flood_correction_applied"] = True
        features["pre_strong_deep_flood_depth_cm"] = round(float(depth_cm), 2)
        features["strong_deep_flood_depth_cm"] = corrected_depth
        features["final_output_reason"] = "Strong broad-water, near-field, reference-object, and region-depth evidence overruled a conservative deep-flood estimate."
        return corrected_depth, round(float(max(confidence, 0.85)), 4), self._action_for_final_depth(corrected_depth, features, action)

    def _apply_gemma_semantic_correction(
        self,
        depth_cm: float,
        confidence: float,
        action: str,
        features: Dict[str, Any],
    ) -> Tuple[float, float, str]:
        gemma_feats = features.get("gemma_semantic_features")
        max_ref_submersion = float(features.get("max_reference_submersion", 0.0))
        reference_count = int(features.get("reference_count", 0))
        coverage_pct = float(features.get("water_coverage_pct", 0.0))
        near_pct = float(features.get("near_water_coverage_pct", 0.0))
        far_pct = float(features.get("far_water_coverage_pct", 0.0))
        dense_p90 = float(features.get("dense_depth_p90", 0.0))
        dense_depth_cm = float(features.get("depth_anything_dense_depth_cm", 0.0))
        candidate_depth_cm = float(features.get("efficientnet_candidate_depth_cm", 0.0))
        mask_conditioned_depth_cm = float(features.get("mask_conditioned_fusion_depth_cm", 0.0))
        immediate_risk = bool(features.get("immediate_risk", False))

        # Parse Gemma features
        gemma_submersion = None
        is_flooded_scene = False
        is_shallow_scene = False
        gemma_ref_visible = False
        gemma_water_reaches_ref = False
        gemma_waterline_visible = False
        scene_type = "unknown"
        water_present = False

        if gemma_feats and isinstance(gemma_feats, dict):
            water_present = bool(gemma_feats.get("water_present", False))
            scene_type = str(gemma_feats.get("scene_type", "")).lower()
            submersion_frac = gemma_feats.get("approximate_submersion_fraction")
            if submersion_frac is not None:
                gemma_submersion = float(submersion_frac)

            gemma_ref_visible = bool(gemma_feats.get("reference_object_visible", False))
            gemma_water_reaches_ref = bool(gemma_feats.get("water_reaches_reference", False))
            gemma_waterline_visible = bool(gemma_feats.get("waterline_visible", False))
            is_flooded_scene = water_present and scene_type in ("flooded_road", "flooded_indoor")
            is_shallow_scene = scene_type in ("wet_puddle", "dry_land") or (
                gemma_submersion is not None and gemma_submersion <= 0.20
            )

        effective_submersion = gemma_submersion if gemma_submersion is not None else max_ref_submersion

        # --- RULE 1: Ankle/foot level shallow water (tiny coverage, near-zero submersion) ---
        # Relaxed thresholds to prefer Gemma when it reports shallow submersion or a shallow scene.
        if (effective_submersion <= 0.20 or is_shallow_scene or coverage_pct <= 2.0) and max_ref_submersion <= 0.20:
            # Slightly higher shallow cap to capture ankle-to-calf depths (typical ~15-20cm)
            target_shallow_depth = 15.0 if effective_submersion <= 0.08 or coverage_pct <= 1.0 else 18.0
            if depth_cm > target_shallow_depth:
                corrected_depth = round(target_shallow_depth, 2)
                features["gemma_semantic_correction_applied"] = True
                features["pre_gemma_semantic_depth_cm"] = round(float(depth_cm), 2)
                features["gemma_semantic_depth_cm"] = corrected_depth
                features["final_output_reason"] = (
                    f"Ankle/foot level submersion evidence (submersion={effective_submersion:.2f}, coverage={coverage_pct:.2f}%) "
                    f"refined depth prediction to shallow ankle water ({corrected_depth} cm)."
                )
                return corrected_depth, max(confidence, 0.88), self._action_for_final_depth(corrected_depth, features, action)

        # --- RULE 2: Broad road water WITH NO (or non-submerged) reference object submersion = shallow spread layer ---
        # Primary gate previously required YOLO to find no references. Relax gate to allow Gemma to veto
        # reference-based submersion when Gemma reports water not reaching reference objects or reports low submersion.
        yolo_no_submersion = reference_count == 0 and max_ref_submersion <= 0.05
        # Treat Gemma's water_reaches_ref as authoritative regardless of reference_count when available
        effective_water_reaches = bool(gemma_water_reaches_ref)
        # Allow Gemma to indicate "no significant submersion" more permissively
        gemma_no_submersion = (not effective_water_reaches) and (gemma_submersion is None or gemma_submersion <= 0.40)

        # Broad coverage OR Gemma-shallow scene qualifies for this rule
        broad_coverage_no_submersion = (
            (yolo_no_submersion or is_shallow_scene or (water_present and scene_type.startswith("flood")))
            and gemma_no_submersion
            and coverage_pct >= 15.0
            and far_pct <= 10.0  # tolerate a small amount of far-field noise
        )
        logger.info(
            f"[GemmaRule2] yolo_no_sub={yolo_no_submersion} gemma_no_sub={gemma_no_submersion} "
            f"effective_water_reaches={effective_water_reaches} gemma_submersion={gemma_submersion} "
            f"coverage={coverage_pct:.1f}% far={far_pct:.1f}% near={near_pct:.1f}% depth_cm={depth_cm:.1f} "
            f"fires={broad_coverage_no_submersion and depth_cm > 22.0}"
        )
        if broad_coverage_no_submersion and depth_cm > 22.0:
            # Estimate based on near-field ratio: more near coverage = slightly deeper spread
            if near_pct >= 60.0:
                cap = 22.0  # Well-flooded near-field → up to 22 cm
            elif near_pct >= 40.0:
                cap = 18.0
            else:
                cap = 15.0

            corrected_depth = round(min(depth_cm, cap), 2)
            if corrected_depth < depth_cm:
                features["gemma_semantic_correction_applied"] = True
                features["pre_gemma_semantic_depth_cm"] = round(float(depth_cm), 2)
                features["gemma_semantic_depth_cm"] = corrected_depth
                features["final_output_reason"] = (
                    f"Broad road water coverage ({coverage_pct:.1f}%, near={near_pct:.1f}%) "
                    f"with no submerged reference objects (YOLO refs={reference_count}, max_sub={max_ref_submersion:.2f}; "
                    f"Gemma: water_reaches_ref={gemma_water_reaches_ref}, submersion={effective_submersion:.2f}). "
                    f"Interpreted as thin surface water layer; depth capped at {corrected_depth} cm."
                )
                return corrected_depth, max(confidence, 0.82), self._action_for_final_depth(corrected_depth, features, action)

        # --- RULE 3: Deep Flood — only trigger when ACTUAL submersion evidence exists ---
        # Require at least one of: object submerged, Gemma confirms submersion, waterline visible + deep p90
        has_confirmed_submersion = (
            effective_submersion >= 0.30
            or gemma_water_reaches_ref
            or (gemma_ref_visible and effective_submersion > 0.20)
            or max_ref_submersion >= 0.35
        )
        has_broad_deep_water = (
            coverage_pct >= 40.0 and near_pct >= 50.0
            and (gemma_waterline_visible or dense_p90 >= 0.60)
            and has_confirmed_submersion  # Must have submersion evidence for deep correction
        )

        if (is_flooded_scene and has_confirmed_submersion) or has_broad_deep_water:
            if dense_p90 >= 0.45:
                candidates = [d for d in [dense_depth_cm, candidate_depth_cm, mask_conditioned_depth_cm] if d >= 35.0]
                if candidates:
                    expected_depth = float(np.median(candidates))
                    if depth_cm < expected_depth:
                        corrected_depth = round(expected_depth, 2)
                        features["gemma_semantic_correction_applied"] = True
                        features["pre_gemma_semantic_depth_cm"] = round(float(depth_cm), 2)
                        features["gemma_semantic_depth_cm"] = corrected_depth
                        features["final_output_reason"] = (
                            f"Gemma confirmed active flood with submersion evidence (submersion={effective_submersion:.2f}); "
                            f"overruled conservative depth clamp using visual candidate signals."
                        )
                        return corrected_depth, max(confidence, 0.85), self._action_for_final_depth(corrected_depth, features, action)

        return depth_cm, confidence, action

    def _apply_mask_conditioned_high_flood_correction(
        self,
        depth_cm: float,
        confidence: float,
        action: str,
        features: Dict[str, Any],
    ) -> Tuple[float, float, str]:
        try:
            cfg = load_settings_dict().get("inference", {}).get("mask_conditioned_fusion_signal", {})
        except Exception:
            cfg = {}

        if not bool(cfg.get("high_flood_correction_enabled", False)):
            features["mask_conditioned_high_flood_status"] = "disabled"
            return depth_cm, confidence, action

        mask_depth_raw = features.get("mask_conditioned_fusion_depth_cm")
        if mask_depth_raw is None:
            features["mask_conditioned_high_flood_status"] = "missing_signal"
            return depth_cm, confidence, action

        try:
            mask_depth = float(mask_depth_raw)
        except (TypeError, ValueError):
            features["mask_conditioned_high_flood_status"] = "invalid_signal"
            return depth_cm, confidence, action

        coverage = float(features.get("water_coverage_pct", 0.0))
        near = float(features.get("near_water_coverage_pct", 0.0))
        mid = float(features.get("mid_water_coverage_pct", 0.0))
        far = float(features.get("far_water_coverage_pct", 0.0))
        max_submersion = float(features.get("max_reference_submersion", 0.0))
        reference_count = int(float(features.get("reference_count", 0.0)))
        no_water_probability = self._feature_float(features.get("no_water_probability"))
        wet_road_probability = self._feature_float(features.get("wet_road_no_water_probability"))

        min_mask_depth = float(cfg.get("high_flood_min_mask_depth_cm", 75.0))
        max_current_depth = float(cfg.get("high_flood_max_current_depth_cm", 70.0))
        min_gap = float(cfg.get("high_flood_min_gap_cm", 25.0))
        min_coverage = float(cfg.get("high_flood_min_water_coverage_pct", 45.0))
        min_near_or_mid = float(cfg.get("high_flood_min_near_or_mid_pct", 35.0))
        max_no_water_prob = float(cfg.get("high_flood_max_no_water_probability", 0.98))
        max_wet_road_prob = float(cfg.get("high_flood_max_wet_road_probability", 0.98))

        dense_depth_cm = float(features.get("dense_depth_p90", 0.0)) * 120.0
        efficientnet_depth = self._feature_float(features.get("efficientnet_candidate_depth_cm"))
        broad_water_evidence = coverage >= min_coverage and max(near, mid) >= min_near_or_mid
        high_model_agreement = (
            bool(features.get("immediate_risk", False))
            and coverage >= float(cfg.get("high_flood_min_weak_mask_coverage_pct", 12.0))
            and near >= float(cfg.get("high_flood_min_weak_mask_near_pct", 12.0))
            and efficientnet_depth >= float(cfg.get("high_flood_min_efficientnet_depth_cm", 60.0))
            and dense_depth_cm >= float(cfg.get("high_flood_min_dense_depth_cm", 70.0))
        )
        object_or_scene_evidence = bool(features.get("immediate_risk", False)) or reference_count > 0 or max_submersion >= 0.25 or far >= 50.0
        guard_blocked = (
            bool(features.get("low_water_gate_applied", False))
            or bool(features.get("dry_land_guard_applied", False))
            or bool(features.get("water_present_overridden_by_no_water_guard", False))
            or bool(features.get("far_water_only", False))
            or no_water_probability >= max_no_water_prob
            or wet_road_probability >= max_wet_road_prob
        )
        should_apply = (
            mask_depth >= min_mask_depth
            and depth_cm <= max_current_depth
            and (mask_depth - depth_cm) >= min_gap
            and (broad_water_evidence or high_model_agreement)
            and object_or_scene_evidence
            and not guard_blocked
        )

        if not should_apply:
            features["mask_conditioned_high_flood_status"] = "not_applicable"
            return depth_cm, confidence, action

        corrected_depth = round(float(np.clip(mask_depth, 0.0, 180.0)), 2)
        features["mask_conditioned_high_flood_status"] = "applied"
        features["mask_conditioned_high_flood_correction_applied"] = True
        features["pre_mask_conditioned_high_flood_depth_cm"] = round(float(depth_cm), 2)
        features["mask_conditioned_high_flood_corrected_depth_cm"] = corrected_depth
        features["model_agreement_status"] = "mask_conditioned_high_flood_correction"
        features["final_aggregation_source"] = "mask_conditioned_high_flood_correction"
        features["final_output_reason"] = (
            f"MaskConditionedFusion predicted deep flood at {corrected_depth:.2f} cm while current pipeline was {depth_cm:.2f} cm; "
            "broad water evidence or high-model agreement allowed this high-flood correction."
        )
        return corrected_depth, round(float(max(confidence, 0.86)), 4), self._action_for_final_depth(corrected_depth, features, action)
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
        features["final_output_reason"] = "Dry-land visual evidence with no near-field water overruled flood depth signals."
        return 0.0, round(float(max(confidence, 0.90)), 4), self._action_for_final_depth(0.0, features, action)
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
            if max_reference_submersion < 0.50 and coverage < 0.50:
                depth_cm = min(depth_cm, 25.0)
            if max_reference_submersion < 0.20 or coverage < 0.03:
                depth_cm = min(depth_cm, 15.0)
            if max_reference_submersion <= 0.05 and coverage < 0.01:
                depth_cm = min(depth_cm, 12.0)
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
            and 2.0 <= candidate_depth_value <= 50.0
            and (abs(candidate_depth_value - dense_depth_cm) <= 25.0 or dense_depth_cm <= 60.0)
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
                features["low_water_gate_reason"] = "Small water mask, but candidate model and depth signals confirm shallow flood water."
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

        no_reference_depth_uncertain = (
            reference_count < 1
            and not full_road_water_no_reference
            and coverage >= 0.08
            and max_reference_submersion < 0.15
        )
        if no_reference_depth_uncertain:
            features["no_reference_depth_uncertain"] = True
            features["review_required"] = True
            features["review_reason"] = "Floodwater is visible, but no usable scale reference was detected; depth is capped and needs review if operational impact is high."

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
        elif no_reference_depth_uncertain:
            if coverage < 0.60:
                depth_cm = min(depth_cm, 25.0)
            elif coverage < 0.75:
                depth_cm = min(depth_cm, 35.0)
            else:
                depth_cm = min(depth_cm, 45.0)
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
        if bool(features.get("no_reference_depth_uncertain", False)):
            confidence = min(confidence, 0.68)

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
        no_reference_uncertain = bool(features.get("no_reference_depth_uncertain", False))
        candidate_value_for_trust = float(candidate_depth) if candidate_depth is not None else None
        candidate_trusted = (not low_water_gate or shallow_exception) and (
            shallow_exception
            or muddy_fallback
            or (immediate_risk and not (no_reference_uncertain and candidate_value_for_trust is not None and candidate_value_for_trust > 35.0))
            or (coverage >= 0.08 and not (no_reference_uncertain and candidate_value_for_trust is not None and candidate_value_for_trust > 35.0))
            or (candidate_value_for_trust is not None and candidate_value_for_trust < 15.0)
        )
        add_signal("efficientnet_candidate", candidate_depth, candidate_trusted, "trained depth model", 0.35)

        mask_conditioned_depth = features.get("mask_conditioned_fusion_depth_cm")
        mask_conditioned_trusted = bool(features.get("mask_conditioned_fusion_trusted", False))
        # Prefer Gemma: if Gemma semantics indicate shallow flood and mask model predicts shallow depth, promote mask model trust
        gemma_feats = features.get("gemma_semantic_features") if isinstance(features.get("gemma_semantic_features"), dict) else {}
        gemma_water_present = bool(gemma_feats.get("water_present", False)) if gemma_feats else False
        gemma_scene = str(gemma_feats.get("scene_type", "")).lower() if gemma_feats else ""
        try:
            mask_depth_val = float(mask_conditioned_depth) if mask_conditioned_depth is not None else None
        except Exception:
            mask_depth_val = None
        if gemma_water_present and mask_depth_val is not None and mask_depth_val < 35.0 and float(features.get("max_reference_submersion", 0.0)) < 0.4:
            mask_conditioned_trusted = True
        add_signal("mask_conditioned_fusion", mask_conditioned_depth, mask_conditioned_trusted, "experimental mask-conditioned trained model (promoted by Gemma)", 0.20)
        # Add a Gemma semantic signal if Gemma suggests shallow submersion and there's a mask-derived depth
        gemma_signal_depth = mask_depth_val if mask_depth_val is not None else None
        if gemma_water_present and gemma_signal_depth is not None:
            # Promote Gemma to dominant semantic signal
            add_signal("gemma_semantics", gemma_signal_depth, True, "Gemma VLM semantic shallow-depth suggestion", 0.85)

        reference_depth = features.get("reference_depth_cm")
        reference_trusted = reference_count > 0 and not low_water_gate and (max_submersion >= 0.15 or coverage >= 0.20 or immediate_risk)
        # Reduce reference weight to allow Gemma to influence agreement more
        add_signal("reference_objects", reference_depth, reference_trusted, "object/reference depth estimate", 0.10)

        dense_depth = float(features.get("dense_depth_p90", 0.0)) * 120.0
        dense_trusted = ((not low_water_gate and (coverage >= 0.20 or near_pct >= 0.10 or immediate_risk)) or shallow_exception) and not (no_reference_uncertain and dense_depth > 35.0)
        # Slightly lower dense-depth weight
        add_signal("depth_anything_dense", dense_depth, dense_trusted, "monocular dense-depth support", 0.08)

        trusted = [signal for signal in signals if signal["trusted"]]
        tolerance_cm = 25.0
        best_cluster: List[Dict[str, Any]] = []

        # Build unique clusters of trusted signals where members are within tolerance_cm of each other.
        clusters: List[List[Dict[str, Any]]] = []
        seen_keys = set()
        for signal in trusted:
            cluster = [candidate for candidate in trusted if abs(candidate["depth_cm"] - signal["depth_cm"]) <= tolerance_cm]
            if not cluster:
                continue
            # Create a deterministic key for the cluster to avoid duplicates: use (name, depth_cm) pairs
            key = frozenset((c["name"], float(c["depth_cm"])) for c in cluster)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            clusters.append(cluster)

        # Prefer clusters with at least two signals. Choose the cluster with highest total weight.
        valid_clusters = [c for c in clusters if len(c) >= 2]
        if valid_clusters:
            def _cluster_sort_key(c: List[Dict[str, Any]]):
                total_weight = sum(float(s.get("weight", 0.0)) for s in c)
                # Tie-breaker: prefer larger cluster, then deterministic lexicographic order of names
                names = tuple(sorted(s["name"] for s in c))
                return (total_weight, len(c), names)

            # max with the key above selects the cluster with highest total weight; deterministic ties resolved by len and names
            best_cluster = max(valid_clusters, key=_cluster_sort_key)
        else:
            # Fallback to previous behaviour: choose the cluster with the largest number of signals
            for signal in trusted:
                cluster = [candidate for candidate in trusted if abs(candidate["depth_cm"] - signal["depth_cm"]) <= tolerance_cm]
                if len(cluster) > len(best_cluster):
                    best_cluster = cluster

        output_depth = float(final_depth_cm)
        output_confidence = float(confidence)
        output_action = action

        # Gemma shallow-override: if Gemma reports low submersion fraction and the
        # mask-conditioned model predicts a shallow depth, prefer the Gemma/mask
        # shallow outcome over a high candidate when the candidate is much deeper.
        gemma_feats = features.get("gemma_semantic_features") if isinstance(features.get("gemma_semantic_features"), dict) else {}
        gemma_sub = None
        if gemma_feats:
            gemma_sub = gemma_feats.get("approximate_submersion_fraction")
            try:
                gemma_sub = float(gemma_sub) if gemma_sub is not None else None
            except Exception:
                gemma_sub = None

        try:
            mask_depth_val = float(features.get("mask_conditioned_fusion_depth_cm")) if features.get("mask_conditioned_fusion_depth_cm") is not None else None
        except Exception:
            mask_depth_val = None

        # candidate_signal already computed above — re-evaluate here for clarity
        candidate_signal = next((signal for signal in trusted if signal["name"] == "efficientnet_candidate"), None)

        # --- Gemma deep-override: when Gemma reports substantial submersion on a visible reference
        # and the mask-conditioned model predicts deep water, prefer gemma/mask depth.
        if (
            gemma_sub is not None
            and gemma_sub >= 0.40
            and mask_depth_val is not None
            and mask_depth_val >= 35.0
        ):
            output_depth = round(float(np.clip(mask_depth_val, 0.0, 180.0)), 2)
            output_confidence = max(float(confidence), 0.90)
            output_action = self._action_for_final_depth(output_depth, features, action)
            status = "gemma_deep_override"
            reason = (
                f"Gemma reported substantial submersion ({gemma_sub:.2f}) and mask suggested deep water {mask_depth_val:.2f} cm; "
                "preferring Gemma/mask deep outcome."
            )
            features["final_aggregation_source"] = "gemma_deep_override"
            features["final_output_reason"] = reason

        # --- Gemma shallow-override: if Gemma reports low submersion fraction and the
        # mask-conditioned model predicts a shallow depth, prefer the Gemma/mask
        # shallow outcome over a high candidate when the candidate is much deeper.
        if (
            gemma_sub is not None
            and gemma_sub <= 0.25
            and mask_depth_val is not None
            and mask_depth_val < 35.0
            and candidate_signal is not None
            and float(candidate_signal["depth_cm"]) > 40.0
        ):
            override_depth = max(mask_depth_val, 15.0)
            output_depth = round(float(np.clip(override_depth, 0.0, 180.0)), 2)
            output_confidence = max(float(confidence), 0.86)
            output_action = self._action_for_final_depth(output_depth, features, action)
            status = "gemma_shallow_override"
            reason = (
                f"Gemma reported shallow submersion ({gemma_sub:.2f}) and mask/Gemma suggested shallow depth {mask_depth_val:.2f} cm; "
                "preferring shallow outcome."
            )
            features["final_aggregation_source"] = "gemma_shallow_override"
            features["final_output_reason"] = reason

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

        no_water_probability = self._no_water_guard_signal(image_rgb)
        wet_road_no_water_probability = self._wet_road_no_water_guard_signal(image_rgb)
        trace.append(
            {
                "stage": "No-Water Guard",
                "backend": self._no_water_backend,
                "status": "disabled" if self._no_water_model is None else "ok",
                "summary": "checkpoint unavailable" if no_water_probability is None else f"no_water_probability={no_water_probability:.3f}",
            }
        )
        if wet_road_no_water_probability is not None:
            trace.append(
                {
                    "stage": "Wet-Road No-Water Guard",
                    "backend": self._wet_road_guard_backend,
                    "status": "ok",
                    "summary": f"no_water_probability={wet_road_no_water_probability:.3f}",
                }
            )

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
            and not self.water_detector.looks_like_dry_land(image_rgb)
            and not (no_water_probability is not None and no_water_probability >= 0.92)
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

        gemma_semantic_result = None
        if self.gemma_semantic_analyzer is not None:
            gemma_res = self.gemma_semantic_analyzer.analyze(
                image_rgb=image_rgb,
                water_mask=water_mask,
                detected_objects=references,
                water_coverage_pct=water_coverage_pct,
            )
            gemma_status = gemma_res.get("status", "skipped")
            gemma_model = gemma_res.get("model", getattr(self.gemma_semantic_analyzer, "model_name", "gemma3:4b"))
            gemma_feats = gemma_res.get("features")
            gemma_lat = gemma_res.get("latency_ms")

            trace_stage: Dict[str, Any] = {
                "stage": "gemma_semantic_analysis",
                "status": gemma_status,
                "model": gemma_model,
            }
            if gemma_status == "success" and gemma_feats:
                trace_stage["summary"] = f"water_present={gemma_feats.get('water_present')}, scene={gemma_feats.get('scene_type')}"
                trace_stage["features"] = gemma_feats
                trace_stage["latency_ms"] = gemma_lat
                gemma_semantic_result = gemma_feats
            else:
                trace_stage["summary"] = gemma_res.get("reason", "Gemma analysis unavailable or skipped")
                if gemma_lat is not None:
                    trace_stage["latency_ms"] = gemma_lat

            trace.append(trace_stage)

        teacher_features = self._depth_teacher_features(image_rgb, water_mask)
        teacher_ensemble_metrics = teacher_features.get("ensemble", {}) if teacher_features else {}
        teacher_meta = teacher_features.get("meta", {}) if teacher_features else {}
        if self._teacher_ensemble is not None:
            trace.append(
                {
                    "stage": "Depth Teachers",
                    "backend": self._teacher_backend,
                    "status": "ok" if int(teacher_meta.get("available_teacher_count", 0) or 0) > 0 else "degraded",
                    "summary": (
                        f"available={teacher_meta.get('available_teacher_count', 0)}/"
                        f"{teacher_meta.get('total_teachers', 3)} "
                        f"agreement={float(teacher_ensemble_metrics.get('teacher_agreement', 0.0) or 0.0):.3f}"
                    ),
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

        # Quick YOLO-based override: if a detected reference object shows heavy submersion,
        # estimate depth heuristically from typical object dimensions and bypass full agreement.
        try:
            override_applied = False
            for obj in references:
                try:
                    label = getattr(obj, "label", str(getattr(obj, "label", ""))).lower()
                    sub = float(getattr(obj, "water_submersion_ratio", 0.0))
                except Exception:
                    continue
                if sub >= 0.65:
                    # Heuristic per-class typical dimensions (cm)
                    if label in ("car", "truck"):
                        wheel_dia = 55.0 if label == "car" else 70.0
                        est = round(min(sub * wheel_dia, 120.0), 2)
                    elif label in ("bus",):
                        est = round(min(sub * 80.0, 160.0), 2)
                    elif label in ("motorbike", "motorcycle"):
                        est = round(min(sub * 50.0, 100.0), 2)
                    elif label == "person":
                        # Assume knee/leg reference height ~ 50cm
                        est = round(min(max(25.0, sub * 60.0), 120.0), 2)
                    else:
                        est = round(min(sub * 60.0, 120.0), 2)

                    features["yolo_ref_override_applied"] = True
                    features["yolo_ref_override_depth_cm"] = est
                    features["yolo_ref_override_label"] = label
                    override_applied = True
                    break
        except Exception:
            override_applied = False
        if efficientnet_depth_cm is not None:
            features["efficientnet_candidate_depth_cm"] = efficientnet_depth_cm
            features["fusion_candidate_delta_cm"] = round(abs(float(features.get("region_depth_cm", 0.0)) - efficientnet_depth_cm), 2)
        mask_conditioned_fusion_depth_cm = self._mask_conditioned_fusion_depth_signal(image_rgb, features)
        if mask_conditioned_fusion_depth_cm is not None:
            features["mask_conditioned_fusion_depth_cm"] = mask_conditioned_fusion_depth_cm
            trace.append(
                {
                    "stage": "Mask-Conditioned Fusion Candidate",
                    "backend": self._mask_conditioned_fusion_backend,
                    "status": "ok",
                    "summary": f"candidate_depth_cm={mask_conditioned_fusion_depth_cm:.2f}",
                }
            )
        features["no_water_probability"] = no_water_probability
        features["wet_road_no_water_probability"] = wet_road_no_water_probability
        features["secondary_no_water_probability"] = wet_road_no_water_probability
        features["wet_road_guard_backend"] = self._wet_road_guard_backend
        features["no_water_guard_backend"] = self._no_water_backend
        if teacher_features:
            features["depth_teacher_available_count"] = int(teacher_meta.get("available_teacher_count", 0) or 0)
            features["depth_teacher_total_count"] = int(teacher_meta.get("total_teachers", 3) or 3)
            for key in (
                "teacher_mean",
                "teacher_median",
                "teacher_min",
                "teacher_max",
                "teacher_spread",
                "teacher_std",
                "teacher_agreement",
            ):
                value = teacher_ensemble_metrics.get(key)
                if isinstance(value, (int, float)) and np.isfinite(float(value)):
                    features[key] = round(float(value), 6)
            for teacher_name, teacher_stats in teacher_features.get("teachers", {}).items():
                slug = str(teacher_name).lower().replace(" ", "_")
                features[f"{slug}_available"] = 1.0 if teacher_stats.get("available") else 0.0
                if teacher_stats.get("available"):
                    for metric in ("mean", "p50", "p90", "water_mean", "water_p50", "water_p90", "spatial_gradient", "depth_variance"):
                        value = teacher_stats.get(metric)
                        if isinstance(value, (int, float)) and np.isfinite(float(value)):
                            features[f"{slug}_{metric}"] = round(float(value), 6)
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

        if gemma_semantic_result is not None:
            features["gemma_semantic_features"] = gemma_semantic_result

        # If YOLO override applied, use that depth and skip model-agreement/residual fusion stages
        if features.get("yolo_ref_override_applied"):
            depth_cm = float(features.get("yolo_ref_override_depth_cm", 0.0))
            confidence = float(max(0.80, float(features.get("semantic_confidence", 0.5))))
            action = self._action_for_final_depth(depth_cm, features, "Advisory Monitoring")
        else:
            depth_cm, confidence, action = self._calibration_severity_model(features)
            features["calibration_depth_cm"] = round(depth_cm, 2)
            if efficientnet_depth_cm is not None:
                features["final_candidate_delta_cm"] = round(abs(depth_cm - efficientnet_depth_cm), 2)
            depth_cm, confidence, action = self._apply_efficientnet_correction(depth_cm, confidence, action, features)
            depth_cm, confidence, action = self._record_model_agreement(depth_cm, confidence, action, features)
            depth_cm, confidence, action = self._apply_residual_fusion_model(depth_cm, confidence, action, features)
            depth_cm, confidence, action = self._apply_mask_conditioned_high_flood_correction(depth_cm, confidence, action, features)
            depth_cm, confidence, action = self._apply_strong_deep_flood_correction(depth_cm, confidence, action, features)
            depth_cm, confidence, action = self._apply_gemma_semantic_correction(depth_cm, confidence, action, features)
            depth_cm, confidence, action = self._apply_dry_land_guard(depth_cm, confidence, action, image_rgb, features)
            depth_cm, confidence, action = self._apply_no_water_guard(depth_cm, confidence, action, features)
        if features.get("gemma_semantic_correction_applied"):
            trace.append(
                {
                    "stage": "Gemma Semantic Decision",
                    "backend": "gemma-semantics-v1",
                    "status": "applied",
                    "summary": f"gemma semantic analysis corrected depth to {depth_cm:.2f} cm",
                }
            )
        if features.get("dry_land_guard_applied"):
            trace.append(
                {
                    "stage": "Dry-Land Decision",
                    "backend": "dry-land-heuristic",
                    "status": "applied",
                    "summary": "dry scene evidence forced depth=0.00 cm",
                }
            )
        elif features.get("no_water_guard_applied"):
            trace.append(
                {
                    "stage": "No-Water Decision",
                    "backend": str(features.get("no_water_decision_backend", self._no_water_backend)),
                    "status": "applied",
                    "summary": "dry/wet-road guard forced depth=0.00 cm",
                }
            )
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

        if gemma_semantic_result is not None:
            features["gemma_semantic_features"] = gemma_semantic_result

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
            "semantic_features": gemma_semantic_result,
            "pipeline_trace": trace,
            "depth_teachers": teacher_features,
        }


_PIPELINE: Optional[SegformerYoloDepthV2Pipeline] = None


def get_segformer_yolo_depthv2_pipeline() -> SegformerYoloDepthV2Pipeline:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = SegformerYoloDepthV2Pipeline()
    return _PIPELINE












