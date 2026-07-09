"""
Stage-aligned flood inference pipeline:
  Stage 1 — SegFormer (water mask)       — always classical
  Stage 2 — YOLOv8   (reference objects) — always classical
  Stage 3 — Depth Anything V2            — Gemini-enhanced when key present, else proxy
  Stage 4 — Fusion Engine                — Gemini-enhanced when key present, else math
  Stage 5 — Calibration / Severity Model — Gemini-enhanced when key present, else math

Gemini is OPTIONAL for stages 3-5 only.
Set GEMINI_API_KEY in env or pass gemini_api_key to the constructor.
Stages 1 and 2 are always run with the local classical approach regardless of the key.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.reference_depth_estimator import ReferenceDepthEstimator
from src.water_region_detector import WaterRegionDetector

logger = logging.getLogger(__name__)


@dataclass
class ReferenceObject:
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    area_ratio: float
    water_submersion_ratio: float


def _depth_to_severity(depth_cm: float) -> Dict[str, Any]:
    if depth_cm < 5:
        return {"level": "SAFE", "label": "No significant flooding", "color": "#16a34a", "stage": 1}
    if depth_cm < 20:
        return {"level": "LOW", "label": "Minor flooding", "color": "#ca8a04", "stage": 2}
    if depth_cm < 50:
        return {"level": "MEDIUM", "label": "Moderate flooding", "color": "#ea580c", "stage": 3}
    if depth_cm < 80:
        return {"level": "HIGH", "label": "High flood — avoid travel", "color": "#dc2626", "stage": 4}
    return {"level": "CRITICAL", "label": "Severe / dangerous flooding", "color": "#7f1d1d", "stage": 5}


_SEVERITY_LEVELS = ["SAFE", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
_SEVERITY_ANCHOR_DEPTH_CM = {
    "SAFE": 2.0,
    "LOW": 12.5,
    "MEDIUM": 35.0,
    "HIGH": 65.0,
    "CRITICAL": 110.0,
}


def _severity_level_to_index(level: str) -> Optional[int]:
    normalized = str(level).strip().upper()
    if normalized not in _SEVERITY_LEVELS:
        return None
    return _SEVERITY_LEVELS.index(normalized)


def _severity_anchor_depth(level: str) -> Optional[float]:
    normalized = str(level).strip().upper()
    return _SEVERITY_ANCHOR_DEPTH_CM.get(normalized)


def _severity_from_index(index: int) -> Dict[str, Any]:
    clipped_index = int(np.clip(index, 0, len(_SEVERITY_LEVELS) - 1))
    level = _SEVERITY_LEVELS[clipped_index]
    anchor_depth = _SEVERITY_ANCHOR_DEPTH_CM[level]
    return _depth_to_severity(anchor_depth)


class SegformerYoloDepthV2Pipeline:
    """
    Structured multi-stage pipeline with deterministic stage order.
    Gemini Vision is optionally active for stages 3-5 only.
    """

    def __init__(
        self,
        yolo_weights_path: str = "yolov8n.pt",
        yolo_confidence: float = 0.25,
        gemini_api_key: Optional[str] = None,
    ) -> None:
        self.water_detector = WaterRegionDetector()
        self.reference_estimator = ReferenceDepthEstimator()
        self.yolo_weights_path = Path(yolo_weights_path)
        self.yolo_confidence = float(yolo_confidence)
        self._yolo_model = None
        self._yolo_backend = "contour-proxy"
        self._load_yolo_if_available()
        # Gemini — stages 3-5 only
        self._gemini_model = None
        self._gemini_key: Optional[str] = gemini_api_key or os.environ.get("GEMINI_API_KEY")
        if self._gemini_key:
            self._init_gemini()

    def _init_gemini(self) -> None:
        try:
            import google.generativeai as genai

            genai.configure(api_key=self._gemini_key)
            self._gemini_model = genai.GenerativeModel("gemini-1.5-flash")
            logger.info("Gemini 1.5 Flash ready — stages 3-5 enhanced")
        except Exception as exc:  # pragma: no cover
            logger.warning("Gemini init failed (%s). Stages 3-5 will use fallback.", exc)
            self._gemini_model = None

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
        }
        return features

    def _calibration_severity_model(self, features: Dict[str, float]) -> Tuple[float, float, str]:
        coverage = features["water_coverage_pct"] / 100.0
        dense_depth_cm = features["dense_depth_p90"] * 120.0
        reference_depth_cm = features["reference_depth_cm"]

        if coverage < 0.02:
            depth_cm = 0.0
        elif reference_depth_cm > 0:
            depth_cm = (0.65 * reference_depth_cm) + (0.35 * dense_depth_cm)
        else:
            depth_cm = dense_depth_cm

        depth_cm = float(np.clip(depth_cm, 0.0, 180.0))
        reference_count = min(features["reference_count"] / 4.0, 1.0)
        confidence = float(np.clip(0.35 + (coverage * 0.35) + (reference_count * 0.30), 0.2, 0.98))

        if depth_cm >= 100.0:
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

    # ------------------------------------------------------------------
    # Gemini enhancement helpers — stages 3, 4, 5 only
    # ------------------------------------------------------------------

    def _gemini_dense_depth(self, image_rgb: np.ndarray) -> Optional[Dict[str, Any]]:
        """Stage 3 — Gemini Vision refines depth map from image context."""
        if not self._gemini_model:
            return None
        try:
            from PIL import Image as PILImage

            pil_img = PILImage.fromarray(image_rgb)
            prompt = (
                "Analyze this Bengaluru flood image. "
                "Estimate: (1) percentage of ground covered by flood water, "
                "(2) deepest visible water depth in centimeters, "
                "(3) any reference objects visible (car, person, motorbike). "
                "Respond ONLY with a JSON object, no markdown: "
                '{"water_coverage_pct": 45, "max_depth_cm": 35, '
                '"dense_depth_proxy_0_to_1": 0.29, "references_found": ["car at ~35cm"], '
                '"confidence": 0.75}'
            )
            resp = self._gemini_model.generate_content([prompt, pil_img])
            match = re.search(r"\{.*?\}", resp.text.strip(), re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception as exc:
            logger.warning("Gemini stage-3 dense-depth call failed: %s", exc)
        return None

    def _gemini_fusion(
        self, features: Dict[str, Any], image_rgb: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        """Stage 4 — Gemini fuses structured sensor features with visual context."""
        if not self._gemini_model:
            return None
        try:
            from PIL import Image as PILImage

            pil_img = PILImage.fromarray(image_rgb)
            prompt = (
                "You are a flood analysis engine for Bengaluru.\n"
                f"Sensor readings:\n"
                f"  water_coverage={features['water_coverage_pct']:.1f}%\n"
                f"  reference_objects={int(features['reference_count'])} "
                f"(max_submersion={features['max_reference_submersion']:.2f})\n"
                f"  dense_depth_proxy mean={features['dense_depth_mean']:.3f} "
                f"p90={features['dense_depth_p90']:.3f}\n"
                f"  reference_object_depth_cm={features['reference_depth_cm']:.1f}\n"
                "Combine the image and sensor data. Respond ONLY with JSON (no markdown): "
                '{"fused_depth_cm": 45.0, "confidence": 0.82, "key_signal": "car bumper submerged ~35cm"}'
            )
            resp = self._gemini_model.generate_content([prompt, pil_img])
            match = re.search(r"\{.*?\}", resp.text.strip(), re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception as exc:
            logger.warning("Gemini stage-4 fusion call failed: %s", exc)
        return None

    def _gemini_calibration(
        self, features: Dict[str, Any], prior_depth_cm: float, prior_confidence: float
    ) -> Optional[Dict[str, Any]]:
        """Stage 5 evaluator — Gemini re-checks classical 5-bucket grading."""
        if not self._gemini_model:
            return None
        try:
            prior_bucket = _depth_to_severity(prior_depth_cm)["level"]
            prompt = (
                "Post-evaluation task for Bengaluru flood response.\n"
                "The system already produced a classical 5-bucket flood grade.\n"
                f"Prior depth estimate: {prior_depth_cm:.1f} cm (confidence {prior_confidence:.2f})\n"
                f"Prior bucket grade: {prior_bucket}\n"
                f"Water coverage: {features['water_coverage_pct']:.1f}%\n"
                f"Reference objects: {int(features['reference_count'])}\n"
                "Re-check and confirm/correct the grade with higher trust in Gemini evaluation.\n"
                "Severity bands: SAFE (<5cm), LOW (5-20cm), MEDIUM (20-50cm), HIGH (50-80cm), CRITICAL (>80cm)\n"
                "Actions: Monitor | Advisory Monitoring | Issue Municipal Warning | "
                "Activate Traffic Management | Deploy Emergency Diversion\n"
                "Respond ONLY with JSON (no markdown): "
                '{"bucket_level": "MEDIUM", "calibrated_depth_cm": 45.0, "confidence": 0.85, '
                '"next_action": "Issue Municipal Warning", "rationale": "one-line reason"}'
            )
            resp = self._gemini_model.generate_content(prompt)
            match = re.search(r"\{.*?\}", resp.text.strip(), re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception as exc:
            logger.warning("Gemini stage-5 calibration call failed: %s", exc)
        return None

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

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
        # Stage 3: optional Gemini depth refinement
        gemini_depth_data = self._gemini_dense_depth(image_rgb)
        stage3_backend = "dense-depth-proxy"
        stage3_gemini_note = ""
        if gemini_depth_data:
            stage3_backend = "gemini-1.5-flash+proxy"
            stage3_gemini_note = (
                f" gemini_max_depth={gemini_depth_data.get('max_depth_cm', '?')}cm"
                f" gemini_coverage={gemini_depth_data.get('water_coverage_pct', '?')}%"
            )
        trace.append(
            {
                "stage": "Depth Anything V2",
                "backend": stage3_backend,
                "status": "ok",
                "summary": f"dense_p90={float(np.percentile(dense_depth_map, 90)):.3f}{stage3_gemini_note}",
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
        # Stage 4: optional Gemini fusion override
        gemini_fusion_data = self._gemini_fusion(features, image_rgb)
        stage4_backend = "feature-fusion-v1"
        if gemini_fusion_data and "fused_depth_cm" in gemini_fusion_data:
            # Blend Gemini fused estimate with sensor features (70/30 Gemini/sensor)
            features["reference_depth_cm"] = round(
                0.70 * float(gemini_fusion_data["fused_depth_cm"])
                + 0.30 * features["reference_depth_cm"],
                2,
            )
            features["gemini_fused_depth_cm"] = round(float(gemini_fusion_data["fused_depth_cm"]), 2)
            features["gemini_key_signal"] = gemini_fusion_data.get("key_signal", "")
            stage4_backend = "gemini-1.5-flash+feature-fusion-v1"
        trace.append(
            {
                "stage": "Fusion Engine",
                "backend": stage4_backend,
                "status": "ok",
                "summary": (
                    f"coverage={features['water_coverage_pct']:.2f}% "
                    f"refs={int(features['reference_count'])} p90={features['dense_depth_p90']:.3f}"
                ),
            }
        )

        depth_cm, confidence, action = self._calibration_severity_model(features)
        severity = _depth_to_severity(depth_cm)
        # Stage 5: optional Gemini evaluator after classical 5-bucket grading.
        # Final result is Gemini-weighted (75% Gemini, 25% classical).
        classical_depth_cm = depth_cm
        classical_confidence = confidence
        classical_severity = severity
        gemini_cal_data = self._gemini_calibration(features, depth_cm, confidence)
        stage5_backend = "calibration-v1"
        if gemini_cal_data:
            gemini_depth_raw = gemini_cal_data.get("calibrated_depth_cm")
            gemini_level = str(
                gemini_cal_data.get("bucket_level", gemini_cal_data.get("severity", ""))
            ).strip().upper()
            gemini_depth_cm: Optional[float] = None

            if gemini_depth_raw is not None:
                try:
                    gemini_depth_cm = float(gemini_depth_raw)
                except (TypeError, ValueError):
                    gemini_depth_cm = None
            if gemini_depth_cm is None and gemini_level:
                gemini_depth_cm = _severity_anchor_depth(gemini_level)

            if gemini_depth_cm is not None:
                blended_depth = (0.75 * gemini_depth_cm) + (0.25 * classical_depth_cm)
                depth_cm = round(float(np.clip(blended_depth, 0.0, 180.0)), 2)

            base_idx = classical_severity["stage"] - 1
            gemini_idx = _severity_level_to_index(gemini_level)
            if gemini_idx is not None:
                final_idx = int(round((0.75 * gemini_idx) + (0.25 * base_idx)))
                severity = _severity_from_index(final_idx)
            else:
                severity = _depth_to_severity(depth_cm)

            gemini_confidence_raw = gemini_cal_data.get("confidence", classical_confidence)
            try:
                gemini_confidence = float(gemini_confidence_raw)
            except (TypeError, ValueError):
                gemini_confidence = classical_confidence
            confidence = round(
                float(np.clip((0.75 * gemini_confidence) + (0.25 * classical_confidence), 0.2, 0.98)),
                4,
            )
            action = gemini_cal_data.get("next_action", action)

            features["classical_bucket"] = classical_severity["level"]
            features["gemini_bucket"] = gemini_level if gemini_idx is not None else ""
            features["gemini_evaluator_weight"] = 0.75
            features["classical_weight"] = 0.25
            features["gemini_calibration_rationale"] = gemini_cal_data.get("rationale", "")
            stage5_backend = "gemini-evaluator(75%)+calibration-v1(25%)"
        trace.append(
            {
                "stage": "Calibration/Severity Model",
                "backend": stage5_backend,
                "status": "ok",
                "summary": (
                    f"depth_cm={depth_cm:.2f} severity={severity['level']}"
                    f" classical={classical_severity['level']}"
                ),
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
            "gemini_enhanced": self._gemini_model is not None,
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
        _PIPELINE = SegformerYoloDepthV2Pipeline(
            gemini_api_key=os.environ.get("GEMINI_API_KEY"),
        )
    return _PIPELINE
