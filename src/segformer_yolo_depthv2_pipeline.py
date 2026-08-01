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
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.reference_depth_estimator import ReferenceDepthEstimator
from src.water_region_detector import WaterRegionDetector

try:
    from archive.legacy_cli.modules.object_detection import ObjectDetector
except ImportError:  # pragma: no cover - optional runtime fallback
    ObjectDetector = None

logger = logging.getLogger(__name__)


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
        return {"level": "HIGH", "label": "High flood — avoid travel", "color": "#dc2626", "stage": 4}
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
        self._load_yolo_if_available()
        self._load_object_detector_if_available()

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
            # PyTorch>=2.6 defaults to weights_only=True, which breaks legacy YOLO checkpoints.
            os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
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
        Depth Anything V2 stage-compatible dense map.
        Uses a deterministic proxy map to keep the stage executable in constrained envs.
        """
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

        if coverage < 0.02:
            depth_cm = 0.0
        elif reference_count > 0 and reference_depth_cm > 0:
            depth_cm = (0.65 * reference_depth_cm) + (0.35 * dense_depth_cm)
            if max_reference_submersion < 0.80 and coverage < 0.70:
                depth_cm = min(depth_cm, 80.0)
            if max_reference_submersion < 0.70 and coverage < 0.65:
                depth_cm = min(depth_cm, 65.0)
            if max_reference_submersion < 0.60 and coverage < 0.55:
                depth_cm = min(depth_cm, 50.0)
            if max_reference_submersion < 0.50 and coverage < 0.50:
                depth_cm = min(depth_cm, 40.0)
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

        depth_cm = float(np.clip(depth_cm, 0.0, 180.0))
        confidence = float(np.clip(0.35 + (coverage * 0.35) + (min(reference_count, 1.0) * 0.30), 0.2, 0.98))

        if self._is_waterlogged(depth_cm, coverage, max_reference_submersion):
            action = "Monitor"
        elif depth_cm >= 100.0:
            action = "Deploy Emergency Diversion"
        elif depth_cm >= 60.0:
            action = "Activate Traffic Management"
        elif depth_cm >= 30.0:
            action = "Issue Municipal Warning"
        elif depth_cm >= 10.0:
            action = "Advisory Monitoring"
        else:
            action = "Monitor"

        return round(depth_cm, 2), round(confidence, 4), action

    def predict(self, image_rgb: np.ndarray) -> Dict[str, Any]:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("predict expects an RGB image array with shape (H, W, 3)")

        trace: List[Dict[str, str]] = []

        water_mask, water_coverage_pct = self._segformer_water_mask(image_rgb)
        trace.append(
            {
                "stage": "SegFormer",
                "backend": "classical-water-detector",
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
                "backend": "dense-depth-proxy",
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
        trace.append(
            {
                "stage": "Fusion Engine",
                "backend": "feature-fusion-v1",
                "status": "ok",
                "summary": (
                    f"coverage={features['water_coverage_pct']:.2f}% "
                    f"refs={int(features['reference_count'])} p90={features['dense_depth_p90']:.3f}"
                ),
            }
        )

        depth_cm, confidence, action = self._calibration_severity_model(features)
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
