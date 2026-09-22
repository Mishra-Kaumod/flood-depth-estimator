"""
Stage-aligned flood inference pipeline:
  Stage 1 — SegFormer (water mask)       — always classical
  Stage 2 — YOLOv8   (reference objects) — always classical
  Stage 3 — Depth Anything V2            — model-backed when available, else proxy
  Stage 4 — Fusion Engine                — learned multimodal checkpoint
  Stage 5 — Severity Model               — policy mapping from learned depth

Gemini is OPTIONAL as an auxiliary signal only.
Set GEMINI_API_KEY in env or pass gemini_api_key to the constructor.
Stages 1 and 2 are always run with the local classical approach regardless of the key.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.middleware.budget_tracker import ApiBudgetConfig, ApiBudgetTracker
from src.middleware.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from src.middleware.observability import METRICS
from src.middleware.retry import RetryPolicy, run_with_retry
from src.reference_depth_estimator import ReferenceDepthEstimator
from src.settings import load_settings_dict
from src.water_region_detector import WaterRegionDetector
from src.depth_teachers import TeacherEnsemble
# Centralized severity helpers (canonical source-of-truth)
from src.geospatial_classifier import classify_depth_dict, severity_anchor_depth

logger = logging.getLogger(__name__)


@dataclass
class ReferenceObject:
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    area_ratio: float
    water_submersion_ratio: float


def _depth_to_severity(depth_cm: float) -> Dict[str, Any]:
    try:
        # Use centralized classifier to ensure a single source of truth
        return classify_depth_dict(depth_cm)
    except Exception:
        # Fallback to legacy mapping
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
    try:
        val = severity_anchor_depth(level)
        if val is not None:
            return val
    except Exception:
        pass
    normalized = str(level).strip().upper()
    return _SEVERITY_ANCHOR_DEPTH_CM.get(normalized)


def _severity_from_index(index: int) -> Dict[str, Any]:
    clipped_index = int(np.clip(index, 0, len(_SEVERITY_LEVELS) - 1))
    level = _SEVERITY_LEVELS[clipped_index]
    anchor_depth = _SEVERITY_ANCHOR_DEPTH_CM[level]
    return _depth_to_severity(anchor_depth)


KNOWN_REFERENCE_LABELS = {
    "person", "car", "truck", "bus", "motorbike", "motorcycle", "bicycle",
    # contour-proxy generic labels
    "vehicle",
}


class _MaskConditionedFusionRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        import timm

        self.backbone = timm.create_model(
            "efficientnetv2_rw_s",
            pretrained=False,
            num_classes=0,
            global_pool="avg",
        )
        self.vis_proj = nn.Sequential(
            nn.Linear(1792, 256),
            nn.SiLU(),
            nn.LayerNorm(256),
        )
        self.obj_mlp = nn.Sequential(
            nn.Linear(24, 64),
            nn.SiLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 128),
            nn.SiLU(),
            nn.LayerNorm(128),
        )
        self.geo_mlp = nn.Sequential(
            nn.Linear(3, 32),
            nn.SiLU(),
            nn.LayerNorm(32),
            nn.Linear(32, 64),
            nn.SiLU(),
            nn.LayerNorm(64),
        )
        self.fusion = nn.Sequential(
            nn.Linear(448, 384),
            nn.SiLU(),
            nn.LayerNorm(384),
            nn.Dropout(0.1),
            nn.Linear(384, 256),
            nn.SiLU(),
            nn.LayerNorm(256),
        )
        self.depth_head = nn.Linear(256, 1)
        self.ordinal_head = nn.Linear(256, 5)

    def forward(
        self,
        image_tensor: torch.Tensor,
        object_features: torch.Tensor,
        geometry_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        visual_features = self.backbone(image_tensor)
        if visual_features.ndim > 2:
            visual_features = torch.flatten(visual_features, start_dim=1)
        visual_features = self.vis_proj(visual_features)
        object_features = self.obj_mlp(object_features)
        geometry_features = self.geo_mlp(geometry_features)
        fused = self.fusion(torch.cat([visual_features, object_features, geometry_features], dim=1))
        depth_logits = self.depth_head(fused).squeeze(1)
        ordinal_logits = self.ordinal_head(fused)
        return depth_logits, ordinal_logits

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
        settings = load_settings_dict()
        inference_cfg = settings.get("inference", {})
        depth_teacher_cfg = inference_cfg.get("depth_teachers", {}) if isinstance(inference_cfg, dict) else {}
        teachers_enabled_cfg = bool(depth_teacher_cfg.get("enabled", False)) if isinstance(depth_teacher_cfg, dict) else False
        teachers_enabled_env = os.environ.get("FLOOD_DEPTH_TEACHERS_ENABLED", "").strip().lower()
        if teachers_enabled_env in {"1", "true", "yes", "on"}:
            self.teacher_ensemble = TeacherEnsemble(device=str(inference_cfg.get("device", "cpu")).strip().lower())
            logger.info("Depth teacher ensemble enabled")
        elif teachers_enabled_env in {"0", "false", "no", "off"}:
            self.teacher_ensemble = None
            logger.info("Depth teacher ensemble disabled via FLOOD_DEPTH_TEACHERS_ENABLED")
        elif teachers_enabled_cfg:
            self.teacher_ensemble = TeacherEnsemble(device=str(inference_cfg.get("device", "cpu")).strip().lower())
            logger.info("Depth teacher ensemble enabled via config")
        else:
            self.teacher_ensemble = None
            logger.info("Depth teacher ensemble disabled")
        gemini_cfg = inference_cfg.get("gemini", {})
        retry_cfg = gemini_cfg.get("retry", {})
        circuit_cfg = gemini_cfg.get("circuit_breaker", {})
        budget_cfg = gemini_cfg.get("budget", {})

        self._gemini_model = None
        self._gemini_key: Optional[str] = gemini_api_key or os.environ.get("GEMINI_API_KEY")
        self._gemini_timeout_seconds = float(gemini_cfg.get("timeout_seconds", 8.0))
        self._gemini_retry_policy = RetryPolicy(
            max_attempts=int(retry_cfg.get("max_attempts", 2)),
            base_delay_seconds=float(retry_cfg.get("base_delay_seconds", 0.3)),
            max_delay_seconds=float(retry_cfg.get("max_delay_seconds", 2.0)),
            jitter_seconds=float(retry_cfg.get("jitter_seconds", 0.1)),
        )
        self._gemini_circuit_breaker = CircuitBreaker(
            CircuitBreakerConfig(
                failure_threshold=int(circuit_cfg.get("failure_threshold", 5)),
                failure_window_seconds=int(circuit_cfg.get("failure_window_seconds", 300)),
                cooldown_seconds=int(circuit_cfg.get("cooldown_seconds", 180)),
            )
        )
        self._gemini_budget = ApiBudgetTracker(
            ApiBudgetConfig(
                hourly_call_cap=int(budget_cfg.get("hourly_call_cap", 0)),
                daily_call_cap=int(budget_cfg.get("daily_call_cap", 0)),
                hourly_spend_cap_usd=float(budget_cfg.get("hourly_spend_cap_usd", 0.0)),
                daily_spend_cap_usd=float(budget_cfg.get("daily_spend_cap_usd", 0.0)),
                estimated_cost_per_call_usd=float(budget_cfg.get("estimated_cost_per_call_usd", 0.0)),
            )
        )
        self._gemini_quota_per_minute = int(budget_cfg.get("verified_quota_per_minute", 0))
        if self._gemini_key:
            self._init_gemini()

        # Learned fusion checkpoint used for final depth prediction.
        self._torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._fusion_model: Optional[_MaskConditionedFusionRegressor] = None
        self._fusion_model_path = self._resolve_fusion_model_path(inference_cfg)
        self._fusion_image_size = 384
        self._fusion_target_transform = "identity"
        self._fusion_load_error: Optional[str] = None
        self._load_learned_fusion_model()

        # Depth Anything V2 (metric depth feature source). Falls back to proxy if unavailable.
        self._depthv2_processor: Optional[Any] = None
        self._depthv2_model: Optional[Any] = None
        self._depthv2_model_ref = str(
            depth_teacher_cfg.get("depth_anything_v2", {}).get("model", "depth-anything/Depth-Anything-V2-Small-hf")
        ).strip()
        self._depthv2_revision = str(depth_teacher_cfg.get("depth_anything_v2", {}).get("revision", "main")).strip() or None
        allow_download_env = os.environ.get("FLOOD_DEPTH_TEACHERS_ALLOW_DOWNLOAD", "").strip().lower()
        self._depthv2_allow_download = allow_download_env in {"1", "true", "yes", "on"}
        self._depthv2_init_attempted = False

    def _init_gemini(self) -> None:
        try:
            import google.generativeai as genai

            genai.configure(api_key=self._gemini_key)
            self._gemini_model = genai.GenerativeModel("gemini-1.5-flash")
            logger.info("Gemini 1.5 Flash ready — stages 3-5 enhanced")
        except Exception as exc:  # pragma: no cover
            logger.warning("Gemini init failed (%s). Stages 3-5 will use fallback.", exc)
            self._gemini_model = None

    def _gemini_generate_content(self, content: Any, stage_name: str) -> Optional[str]:
        if not self._gemini_model:
            return None

        if not self._gemini_circuit_breaker.allow_request():
            METRICS.increment("gemini_circuit_breaker_open_total")
            METRICS.increment("gemini_fallback_total")
            return None

        def call_once(_attempt: int):
            can_call, reason = self._gemini_budget.try_consume()
            if not can_call:
                METRICS.increment("gemini_budget_block_total")
                raise RuntimeError(f"Gemini budget limit reached: {reason}")
            METRICS.increment("gemini_call_total")
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(self._gemini_model.generate_content, content)
                try:
                    resp = future.result(timeout=self._gemini_timeout_seconds)
                except FutureTimeoutError as exc:
                    METRICS.increment("gemini_timeout_total")
                    raise TimeoutError(
                        f"Gemini {stage_name} call exceeded {self._gemini_timeout_seconds:.1f}s timeout"
                    ) from exc
            return resp

        try:
            response = run_with_retry(
                operation=call_once,
                policy=self._gemini_retry_policy,
                on_retry=lambda _attempt, _delay, _exc: METRICS.increment("gemini_retry_total"),
            )
            self._gemini_circuit_breaker.record_success()
            METRICS.increment("gemini_success_total")
            text = getattr(response, "text", "")
            return text if isinstance(text, str) else None
        except Exception as exc:
            self._gemini_circuit_breaker.record_failure()
            METRICS.increment("gemini_fallback_total")
            logger.warning("Gemini %s call failed, falling back to classical path: %s", stage_name, exc)
            return None

    @staticmethod
    def _extract_json_object(response_text: str) -> Optional[Dict[str, Any]]:
        if not response_text:
            return None
        match = re.search(r"\{.*?\}", response_text.strip(), re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group())
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    @staticmethod
    def _resolve_fusion_model_path(inference_cfg: Dict[str, Any]) -> Path:
        configured = str(inference_cfg.get("fusion_model_path", "")).strip()
        if configured:
            candidate = Path(configured).expanduser()
            if not candidate.is_absolute():
                candidate = (Path(__file__).resolve().parent.parent / candidate).resolve()
            return candidate
        return (Path(__file__).resolve().parent.parent / "models" / "FloodDepth-MaskConditionedFusion.pth").resolve()

    def _load_learned_fusion_model(self) -> None:
        if not self._fusion_model_path.exists():
            self._fusion_load_error = f"checkpoint not found at {self._fusion_model_path}"
            logger.warning(
                "Learned fusion checkpoint missing at %s; depth will use reference fallback.",
                self._fusion_model_path,
            )
            return
        try:
            checkpoint = torch.load(self._fusion_model_path, map_location=self._torch_device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(self._fusion_model_path, map_location=self._torch_device)
        except Exception as exc:
            self._fusion_load_error = str(exc)
            logger.warning("Failed to load learned fusion checkpoint (%s)", exc)
            return

        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            self._fusion_image_size = int(checkpoint.get("image_size", self._fusion_image_size))
            self._fusion_target_transform = str(checkpoint.get("target_transform", "identity")).strip().lower()
        else:
            state_dict = checkpoint

        try:
            model = _MaskConditionedFusionRegressor().to(self._torch_device)
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            self._fusion_model = model
            self._fusion_load_error = None
            logger.info(
                "Loaded learned fusion checkpoint from %s (image_size=%s, target_transform=%s)",
                self._fusion_model_path,
                self._fusion_image_size,
                self._fusion_target_transform,
            )
        except Exception as exc:
            self._fusion_model = None
            self._fusion_load_error = str(exc)
            logger.warning("Learned fusion checkpoint is incompatible (%s)", exc)

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

    def _segformer_water_mask(
        self, image_rgb: np.ndarray
    ) -> Tuple[np.ndarray, float, float, List[str]]:
        # SegFormer-aligned stage. detect_validated() also scrubs false positives.
        mask, water_pct, water_conf, water_flags = self.water_detector.detect_validated(image_rgb)
        return (mask > 0).astype(np.uint8) * 255, float(water_pct), float(water_conf), water_flags

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

    def _load_depth_anything_v2_if_available(self) -> None:
        if self._depthv2_init_attempted:
            return
        self._depthv2_init_attempted = True
        if not self._depthv2_model_ref:
            logger.info("Depth Anything V2 model identifier missing; using dense-depth proxy.")
            return
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except Exception as exc:
            logger.info("transformers unavailable (%s); using dense-depth proxy.", exc)
            return
        try:
            kwargs: Dict[str, Any] = {"local_files_only": not self._depthv2_allow_download}
            if self._depthv2_revision:
                kwargs["revision"] = self._depthv2_revision
            self._depthv2_processor = AutoImageProcessor.from_pretrained(self._depthv2_model_ref, **kwargs)
            self._depthv2_model = AutoModelForDepthEstimation.from_pretrained(self._depthv2_model_ref, **kwargs)
            self._depthv2_model = self._depthv2_model.to(self._torch_device)
            self._depthv2_model.eval()
            logger.info("Loaded Depth Anything V2 model for dense metric depth: %s", self._depthv2_model_ref)
        except Exception as exc:
            logger.warning("Depth Anything V2 load failed (%s); using dense-depth proxy.", exc)
            self._depthv2_model = None
            self._depthv2_processor = None

    def _depth_anything_v2_dense_map(
        self, image_rgb: np.ndarray, water_mask: np.ndarray
    ) -> Tuple[np.ndarray, str, bool]:
        """
        Returns (depth_map, backend_name, is_metric_depth_map).
        Metric maps preserve model output scale (no per-image min-max normalization).
        """
        self._load_depth_anything_v2_if_available()
        if self._depthv2_model is not None and self._depthv2_processor is not None:
            try:
                inputs = self._depthv2_processor(images=image_rgb, return_tensors="pt")
                tensor_inputs: Dict[str, Any] = {}
                for k, v in inputs.items():
                    tensor_inputs[k] = v.to(self._torch_device) if torch.is_tensor(v) else v
                with torch.inference_mode():
                    output = self._depthv2_model(**tensor_inputs)
                depth_tensor = getattr(output, "predicted_depth", None)
                if depth_tensor is None:
                    depth_tensor = getattr(output, "depth", None)
                if depth_tensor is None:
                    raise RuntimeError("Depth Anything output does not include predicted_depth/depth")
                if depth_tensor.ndim == 3:
                    depth_tensor = depth_tensor.unsqueeze(1)
                depth_tensor = F.interpolate(
                    depth_tensor,
                    size=image_rgb.shape[:2],
                    mode="bilinear",
                    align_corners=False,
                )
                depth_map = depth_tensor.squeeze().detach().cpu().numpy().astype(np.float32)
                depth_map = np.where(np.isfinite(depth_map), depth_map, 0.0)
                depth_map = np.maximum(depth_map, 0.0)
                return depth_map, "depth-anything-v2", True
            except Exception as exc:
                logger.warning("Depth Anything V2 inference failed (%s); falling back to proxy.", exc)

        h, _w = image_rgb.shape[:2]
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
        return depth_map.astype(np.float32), "dense-depth-proxy", False

    def _fusion_engine(
        self,
        water_mask: np.ndarray,
        water_coverage_pct: float,
        references: List[ReferenceObject],
        dense_depth_map: np.ndarray,
        reference_estimate: Dict[str, Any],
        dense_depth_is_metric: bool,
    ) -> Dict[str, Any]:
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
            "dense_depth_is_metric": 1.0 if dense_depth_is_metric else 0.0,
        }
        return features

    @staticmethod
    def _normalize_zero_one(value: float, scale_if_percent: bool = False) -> float:
        v = float(value)
        if scale_if_percent and v > 1.0:
            v = v / 100.0
        return float(np.clip(v, 0.0, 1.0))

    @staticmethod
    def _normalize_depth_feature(depth_value: float, is_metric: bool) -> float:
        val = max(0.0, float(depth_value))
        if is_metric:
            return float(np.clip(np.log1p(val) / np.log1p(5.0), 0.0, 1.0))
        return float(np.clip(val, 0.0, 1.0))

    def _encode_fusion_image(self, image_rgb: np.ndarray) -> torch.Tensor:
        size = int(max(64, self._fusion_image_size))
        resized = cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (resized - mean) / std
        chw = np.transpose(normalized, (2, 0, 1))
        return torch.from_numpy(chw).unsqueeze(0).to(self._torch_device)

    def _build_object_feature_vector(
        self,
        image_rgb: np.ndarray,
        references: List[ReferenceObject],
        reference_estimate: Dict[str, Any],
        water_coverage_pct: float,
        water_mask: np.ndarray,
        dense_depth_map: np.ndarray,
        dense_depth_is_metric: bool,
        teacher_features: Dict[str, Any],
    ) -> np.ndarray:
        h, w = image_rgb.shape[:2]
        count = len(references)
        confidences = [float(r.confidence) for r in references]
        submersions = [float(r.water_submersion_ratio) for r in references]
        areas = [float(r.area_ratio) for r in references]
        top_ref = references[0] if references else None

        person_present = 1.0 if any(r.label == "person" for r in references) else 0.0
        vehicle_present = 1.0 if any(r.label in {"car", "truck", "bus", "motorbike", "motorcycle", "bicycle", "vehicle"} for r in references) else 0.0

        if top_ref is not None:
            x1, y1, x2, y2 = top_ref.bbox
            bbox_h = self._normalize_zero_one((y2 - y1) / max(1.0, float(h)))
            bbox_w = self._normalize_zero_one((x2 - x1) / max(1.0, float(w)))
            bbox_bottom = self._normalize_zero_one(y2 / max(1.0, float(h)))
            bbox_cx = self._normalize_zero_one(((x1 + x2) * 0.5) / max(1.0, float(w)))
            bbox_cy = self._normalize_zero_one(((y1 + y2) * 0.5) / max(1.0, float(h)))
        else:
            bbox_h = bbox_w = bbox_bottom = bbox_cx = bbox_cy = 0.0

        water_pixels = dense_depth_map[water_mask > 0]
        if water_pixels.size == 0:
            water_pixels = dense_depth_map.reshape(-1)
        dense_p50 = float(np.percentile(water_pixels, 50))
        dense_p90 = float(np.percentile(water_pixels, 90))

        ref_depth_cm = float(reference_estimate.get("depth_cm", 0.0) or 0.0)
        ref_depth_valid = 1.0 if ref_depth_cm > 0.0 else 0.0
        ref_depth_norm = float(np.clip(ref_depth_cm / 200.0, 0.0, 1.0))
        waterline_norm = self._normalize_zero_one(float(reference_estimate.get("waterline_pct", 0.0) or 0.0), scale_if_percent=True)

        teacher_meta = teacher_features.get("meta", {})
        teacher_ensemble = teacher_features.get("ensemble", {})
        teacher_available_ratio = float(
            teacher_meta.get("available_teacher_count", 0)
        ) / max(1.0, float(teacher_meta.get("total_teachers", 3)))
        teacher_agreement = self._normalize_zero_one(float(teacher_ensemble.get("teacher_agreement", 0.0) or 0.0))
        teacher_spread = self._normalize_zero_one(float(teacher_ensemble.get("teacher_spread", 0.0) or 0.0))

        vector = np.array(
            [
                self._normalize_zero_one(float(count) / 5.0),
                self._normalize_zero_one(max(submersions) if submersions else 0.0),
                self._normalize_zero_one(float(np.mean(submersions)) if submersions else 0.0),
                self._normalize_zero_one(max(areas) if areas else 0.0),
                self._normalize_zero_one(float(np.mean(areas)) if areas else 0.0),
                self._normalize_zero_one(max(confidences) if confidences else 0.0),
                self._normalize_zero_one(float(np.mean(confidences)) if confidences else 0.0),
                ref_depth_norm,
                ref_depth_valid,
                person_present,
                vehicle_present,
                bbox_h,
                bbox_w,
                bbox_bottom,
                bbox_cx,
                bbox_cy,
                waterline_norm,
                self._normalize_zero_one(water_coverage_pct, scale_if_percent=True),
                self._normalize_depth_feature(dense_p50, dense_depth_is_metric),
                self._normalize_depth_feature(dense_p90, dense_depth_is_metric),
                teacher_agreement,
                teacher_spread,
                self._normalize_zero_one(teacher_available_ratio),
                1.0 if count == 0 else 0.0,
            ],
            dtype=np.float32,
        )
        return vector

    def _build_geo_feature_vector(
        self,
        water_coverage_pct: float,
        water_mask: np.ndarray,
        dense_depth_map: np.ndarray,
        dense_depth_is_metric: bool,
    ) -> np.ndarray:
        water_pixels = dense_depth_map[water_mask > 0]
        if water_pixels.size == 0:
            water_pixels = dense_depth_map.reshape(-1)
        p50 = float(np.percentile(water_pixels, 50))
        p90 = float(np.percentile(water_pixels, 90))
        geo = np.array(
            [
                self._normalize_zero_one(water_coverage_pct, scale_if_percent=True),
                self._normalize_depth_feature(p50, dense_depth_is_metric),
                self._normalize_depth_feature(p90, dense_depth_is_metric),
            ],
            dtype=np.float32,
        )
        return geo

    def _predict_depth_with_learned_fusion(
        self,
        image_rgb: np.ndarray,
        object_features: np.ndarray,
        geometry_features: np.ndarray,
    ) -> Tuple[float, float]:
        if self._fusion_model is None:
            raise RuntimeError(
                "learned fusion checkpoint unavailable"
                + (f": {self._fusion_load_error}" if self._fusion_load_error else "")
            )
        image_tensor = self._encode_fusion_image(image_rgb)
        object_tensor = torch.from_numpy(object_features).unsqueeze(0).to(self._torch_device)
        geo_tensor = torch.from_numpy(geometry_features).unsqueeze(0).to(self._torch_device)
        with torch.inference_mode():
            depth_logits, ordinal_logits = self._fusion_model(image_tensor, object_tensor, geo_tensor)

        depth_value = float(depth_logits.squeeze().detach().cpu().item())
        if self._fusion_target_transform == "log1p":
            depth_cm = float(np.expm1(depth_value))
        else:
            depth_cm = depth_value
        if not np.isfinite(depth_cm):
            raise RuntimeError("learned fusion produced non-finite depth")
        depth_cm = max(depth_cm, 0.0)

        ordinal_probs = torch.softmax(ordinal_logits, dim=1)
        confidence = float(ordinal_probs.max().detach().cpu().item())
        return depth_cm, float(np.clip(confidence, 0.0, 1.0))

    @staticmethod
    def _severity_action_model(depth_cm: float) -> Tuple[Dict[str, Any], str]:
        severity = _depth_to_severity(depth_cm)
        action_by_level = {
            "SAFE": "Monitor",
            "LOW": "Advisory Monitoring",
            "MEDIUM": "Issue Municipal Warning",
            "HIGH": "Activate Traffic Management",
            "CRITICAL": "Deploy Emergency Diversion",
        }
        return severity, action_by_level.get(str(severity.get("level", "")).upper(), "Monitor")


    def _segformer_scene_comment(
        self,
        water_mask: "np.ndarray",
        water_coverage_pct: float,
        dense_depth_map: "np.ndarray",
        dense_depth_is_metric: bool = False,
        water_confidence: float = 1.0,
        water_flags: Optional[List[str]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Returns (plain_text_comment, structured_detail_dict).
        Called when no YOLO reference objects are available so the
        operator understands the basis for the estimate.
        structured_detail_dict keys:
          water_pct      float   — coverage percentage
          water_level    str     — "minimal" | "partial" | "moderate" | "extensive" | "near-total"
          water_position str     — short distribution label
          bot_pct        float   — % of bottom-half image that is water
          top_pct        float   — % of top-half image that is water
          depth_p90      float   — p90 depth cue (relative proxy or meters)
          depth_level    str     — "shallow" | "moderate" | "significant" | "deep"
        """
        h = water_mask.shape[0]

        # --- Water extent ---
        if water_coverage_pct < 5:
            extent = "minimal flood traces"
            water_level = "minimal"
        elif water_coverage_pct < 20:
            extent = f"partial flooding ({water_coverage_pct:.0f}% of scene)"
            water_level = "partial"
        elif water_coverage_pct < 50:
            extent = f"moderate flooding ({water_coverage_pct:.0f}% of scene)"
            water_level = "moderate"
        elif water_coverage_pct < 75:
            extent = f"extensive flooding ({water_coverage_pct:.0f}% of scene)"
            water_level = "extensive"
        else:
            extent = f"near-total inundation ({water_coverage_pct:.0f}% of scene)"
            water_level = "near-total"

        # --- Vertical distribution ---
        top_pct = float(np.mean(water_mask[:h // 2] > 0)) * 100
        bot_pct = float(np.mean(water_mask[h // 2:] > 0)) * 100
        if bot_pct > top_pct * 1.5:
            position = "ground level (lower frame)"
            pos_icon = "⬇"
        elif top_pct > bot_pct * 1.2:
            position = "throughout scene incl. upper frame"
            pos_icon = "↕"
        else:
            position = "evenly distributed"
            pos_icon = "↔"

        # --- DepthV2 depth signal ---
        water_pixels = dense_depth_map[water_mask > 0]
        if water_pixels.size == 0:
            water_pixels = dense_depth_map.reshape(-1)
        p90 = float(np.percentile(water_pixels, 90))
        if dense_depth_is_metric:
            if p90 < 0.15:
                depth_signal = "shallow depth cues (~<15cm)"
                depth_level = "shallow"
            elif p90 < 0.40:
                depth_signal = "moderate depth cues (~{:.2f}m)".format(p90)
                depth_level = "moderate"
            elif p90 < 1.00:
                depth_signal = "significant depth cues (~{:.2f}m)".format(p90)
                depth_level = "significant"
            else:
                depth_signal = "deep flood cues (~{:.2f}m)".format(p90)
                depth_level = "deep"
        elif p90 < 0.20:
            depth_signal = "shallow depth cues"
            depth_level = "shallow"
        elif p90 < 0.40:
            depth_signal = "moderate depth cues (p90≈{:.2f})".format(p90)
            depth_level = "moderate"
        elif p90 < 0.65:
            depth_signal = "significant depth cues (p90≈{:.2f})".format(p90)
            depth_level = "significant"
        else:
            depth_signal = "deep flood cues (p90≈{:.2f})".format(p90)
            depth_level = "deep"

        comment = (
            f"SegFormer detects {extent}, {pos_icon} {position}. "
            f"DepthV2: {depth_signal}. "
            "No real-world scale anchor — visual depth cues only. "
            "Add a car, person or motorbike for a calibrated reading."
        )
        # ── Validation flags from _validate_and_refine ──────────────
        wflags = water_flags or []
        water_conf_label: str
        if water_confidence >= 0.80:
            water_conf_label = "high"
        elif water_confidence >= 0.55:
            water_conf_label = "moderate"
        else:
            water_conf_label = "low"

        # Tighten comment when detection is suspect
        if water_confidence < 0.55:
            qual = " (low-confidence detection — may be dry surface)"
        elif water_confidence < 0.80:
            qual = " (moderate-confidence detection)"
        else:
            qual = ""

        comment = (
            f"SegFormer detects {extent}{qual}, {pos_icon} {position}. "
            f"DepthV2: {depth_signal}. "
            "No real-world scale anchor — visual depth cues only. "
            "Add a car, person or motorbike for a calibrated reading."
        )
        detail = {
            "water_pct": round(water_coverage_pct, 1),
            "water_level": water_level,
            "water_position": position,
            "pos_icon": pos_icon,
            "top_pct": round(top_pct, 1),
            "bot_pct": round(bot_pct, 1),
            "depth_p90": round(p90, 3),
            "depth_level": depth_level,
            "water_confidence": round(water_confidence, 2),
            "water_conf_label": water_conf_label,
            "water_flags": wflags,
        }
        return comment, detail

    # ------------------------------------------------------------------
    # Gemini enhancement helpers — optional advisory side-channel
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
            response_text = self._gemini_generate_content([prompt, pil_img], stage_name="stage-3")
            payload = self._extract_json_object(response_text or "")
            if payload is not None:
                return payload
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
            response_text = self._gemini_generate_content([prompt, pil_img], stage_name="stage-4")
            payload = self._extract_json_object(response_text or "")
            if payload is not None:
                return payload
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
            response_text = self._gemini_generate_content(prompt, stage_name="stage-5")
            payload = self._extract_json_object(response_text or "")
            if payload is not None:
                return payload
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
        METRICS.increment("pipeline_predictions_total")

        water_mask, water_coverage_pct, water_confidence, water_flags = (
            self._segformer_water_mask(image_rgb)
        )
        trace.append(
            {
                "stage": "SegFormer",
                "backend": "classical-water-detector",
                "status": "ok",
                "summary": (
                    f"water_coverage={water_coverage_pct:.2f}% "
                    f"water_confidence={water_confidence:.2f}"
                    + (f" flags={water_flags}" if water_flags else "")
                ),
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

        # When no known reference objects are detected YOLO cannot provide a
        # real-world scale anchor.  We do NOT refuse — instead we continue with
        # SegFormer + DepthV2 only, flag the result as low-confidence, and
        # describe what SegFormer sees so the operator still gets useful context.
        no_reference = len(references) == 0
        if no_reference:
            logger.info(
                "No YOLO reference objects found — estimate will use SegFormer + DepthV2 only. "
                "Confidence will be capped at 0.55."
            )

        dense_depth_map, stage3_backend, dense_depth_is_metric = self._depth_anything_v2_dense_map(
            image_rgb,
            water_mask,
        )
        # Stage 3: optional Gemini depth refinement
        gemini_depth_data = self._gemini_dense_depth(image_rgb)
        stage3_gemini_note = ""
        if gemini_depth_data:
            stage3_backend = f"gemini-1.5-flash+{stage3_backend}"
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

        teacher_features: Dict[str, Any] = {
            "water_region_valid": False,
            "teachers": {},
            "ensemble": {},
            "meta": {"available_teacher_count": 0, "total_teachers": 3},
        }
        teacher_ensemble_metrics: Dict[str, Any] = {}
        if self.teacher_ensemble is not None:
            try:
                teacher_features = self.teacher_ensemble.predict(image_rgb, water_mask=(water_mask > 0))
                teacher_ensemble_metrics = teacher_features.get("ensemble", {})
                teacher_meta = teacher_features.get("meta", {})
                trace.append(
                    {
                        "stage": "Depth Teachers",
                        "backend": "DepthAnythingV2+DepthPro+Metric3D",
                        "status": "ok",
                        "summary": (
                            f"available={teacher_meta.get('available_teacher_count', 0)}/"
                            f"{teacher_meta.get('total_teachers', 3)} "
                            f"agreement={teacher_ensemble_metrics.get('teacher_agreement', 0.0):.3f}"
                        ),
                    }
                )
            except Exception as exc:
                logger.warning("Depth teacher feature extraction failed: %s", exc)
                trace.append(
                    {
                        "stage": "Depth Teachers",
                        "backend": "DepthAnythingV2+DepthPro+Metric3D",
                        "status": "degraded",
                        "summary": f"teacher extraction failed: {exc}",
                    }
                )
        else:
            trace.append(
                {
                    "stage": "Depth Teachers",
                    "backend": "DepthAnythingV2+DepthPro+Metric3D",
                    "status": "skipped",
                    "summary": "teacher ensemble disabled",
                }
            )

        reference_estimate = self.reference_estimator.estimate(image_rgb)
        features = self._fusion_engine(
            water_mask=water_mask,
            water_coverage_pct=water_coverage_pct,
            references=references,
            dense_depth_map=dense_depth_map,
            reference_estimate=reference_estimate,
            dense_depth_is_metric=dense_depth_is_metric,
        )

        if teacher_ensemble_metrics:
            for key in (
                "teacher_mean",
                "teacher_median",
                "teacher_min",
                "teacher_max",
                "teacher_spread",
                "teacher_std",
                "teacher_agreement",
            ):
                val = teacher_ensemble_metrics.get(key)
                if isinstance(val, (int, float)):
                    features[key] = round(float(val), 6)

        for t_name, t_stats in teacher_features.get("teachers", {}).items():
            slug = t_name.lower()
            features[f"{slug}_available"] = 1.0 if t_stats.get("available") else 0.0
            if t_stats.get("available"):
                for metric in ("mean", "p50", "p90", "water_mean", "water_p50", "water_p90", "spatial_gradient", "depth_variance"):
                    val = t_stats.get(metric)
                    if isinstance(val, (int, float)):
                        features[f"{slug}_{metric}"] = round(float(val), 6)

        object_vector = self._build_object_feature_vector(
            image_rgb=image_rgb,
            references=references,
            reference_estimate=reference_estimate,
            water_coverage_pct=water_coverage_pct,
            water_mask=water_mask,
            dense_depth_map=dense_depth_map,
            dense_depth_is_metric=dense_depth_is_metric,
            teacher_features=teacher_features,
        )
        geometry_vector = self._build_geo_feature_vector(
            water_coverage_pct=water_coverage_pct,
            water_mask=water_mask,
            dense_depth_map=dense_depth_map,
            dense_depth_is_metric=dense_depth_is_metric,
        )
        features["fusion_object_features"] = [round(float(v), 6) for v in object_vector.tolist()]
        features["fusion_geometry_features"] = [round(float(v), 6) for v in geometry_vector.tolist()]
        features["fusion_model_path"] = str(self._fusion_model_path)
        features["fusion_target_transform"] = self._fusion_target_transform
        features["depth_backbone"] = stage3_backend

        stage4_backend = "learned-mask-conditioned-fusion"
        depth_cm: float
        confidence: float
        try:
            depth_cm, confidence = self._predict_depth_with_learned_fusion(
                image_rgb=image_rgb,
                object_features=object_vector,
                geometry_features=geometry_vector,
            )
            features["fusion_model_active"] = 1.0
        except Exception as exc:
            logger.warning("Learned fusion inference failed (%s); using reference fallback.", exc)
            depth_cm = float(reference_estimate.get("depth_cm", 0.0) or 0.0)
            confidence = float(reference_estimate.get("confidence", 0.35) or 0.35)
            stage4_backend = "reference-depth-fallback"
            features["fusion_model_active"] = 0.0
            features["fusion_fallback_reason"] = str(exc)
        trace.append(
            {
                "stage": "Fusion Engine",
                "backend": stage4_backend,
                "status": "ok",
                "summary": (
                    f"coverage={features['water_coverage_pct']:.2f}% "
                    f"refs={int(features['reference_count'])} depth_cm={depth_cm:.2f}"
                ),
            }
        )

        depth_cm = round(float(depth_cm), 2)
        confidence = round(float(np.clip(confidence, 0.0, 1.0)), 4)
        severity, action = self._severity_action_model(depth_cm)
        stage5_backend = "severity-mapping-only"
        trace.append(
            {
                "stage": "Calibration/Severity Model",
                "backend": stage5_backend,
                "status": "ok",
                "summary": (
                    f"depth_cm={depth_cm:.2f} severity={severity['level']}"
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

        # When no YOLO objects were found, cap confidence and attach a plain-English
        # description of what SegFormer + DepthV2 observed so the operator has context.
        scene_comment: str = ""
        no_ref_detail: Dict[str, Any] = {}
        scale_anchor: str = "yolo_reference"
        water_detection_unreliable = False
        suppressed_depth_reason = ""
        provisional_depth_cm: Optional[float] = None
        if no_reference:
            confidence = round(float(np.clip(confidence, 0.0, 0.55)), 4)
            scale_anchor = "none"
            scene_comment, no_ref_detail = self._segformer_scene_comment(
                water_mask=water_mask,
                water_coverage_pct=water_coverage_pct,
                dense_depth_map=dense_depth_map,
                dense_depth_is_metric=dense_depth_is_metric,
                water_confidence=water_confidence,
                water_flags=water_flags,
            )
            # Further cap confidence when water detection itself is low-quality
            if water_confidence < 0.55:
                confidence = round(float(np.clip(confidence, 0.0, 0.35)), 4)
            elif water_confidence < 0.80:
                confidence = round(float(np.clip(confidence, 0.0, 0.45)), 4)

            # Hard guard: if water signal itself is likely false-positive, do not
            # surface a numeric depth/severity estimate at all.
            if water_confidence < 0.45:
                water_detection_unreliable = True
                METRICS.increment("depth_suppressed_total")
                provisional_depth_cm = float(depth_cm)
                suppressed_depth_reason = (
                    "Water mask quality is too low for a reliable depth estimate. "
                    "Likely dry-surface false positive (e.g., textured road/soil)."
                )
                depth_cm = None
                severity = None
                confidence = round(float(np.clip(confidence, 0.0, 0.20)), 4)
                action = "Retake image with clearly visible flood water and a reference object."

        return {
            "depth_cm": depth_cm,
            "confidence": confidence,
            "severity": severity,
            "method": "segformer_depthv2_only" if no_reference else "segformer_yolov8_depthv2_fusion",
            "scale_anchor": scale_anchor,
            "no_reference_warning": no_reference,
            "scene_comment": scene_comment,
            "no_ref_detail": no_ref_detail,
            "water_confidence": water_confidence,
            "water_flags": water_flags,
            "water_detection_unreliable": water_detection_unreliable,
            "suppressed_depth_reason": suppressed_depth_reason,
            "provisional_depth_cm": provisional_depth_cm,
            "gemini_enhanced": self._gemini_model is not None,
            "gemini_quota_per_minute": self._gemini_quota_per_minute,
            "gemini_circuit_breaker_state": self._gemini_circuit_breaker.state(),
            "gemini_budget_state": self._gemini_budget.snapshot(),
            "visual_cues": visual_cues,
            "label_guide": reference_estimate.get("label_guide", ""),
            "waterline_pct": reference_estimate.get("waterline_pct", 0.0),
            "water_coverage": round(water_coverage_pct / 100.0, 4),
            "action_trigger": action,
            "structured_features": features,
            "pipeline_trace": trace,
            "depth_teachers": teacher_features,
        }


_PIPELINE: Optional[SegformerYoloDepthV2Pipeline] = None


def get_segformer_yolo_depthv2_pipeline() -> SegformerYoloDepthV2Pipeline:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = SegformerYoloDepthV2Pipeline(
            gemini_api_key=os.environ.get("GEMINI_API_KEY"),
        )
    return _PIPELINE
