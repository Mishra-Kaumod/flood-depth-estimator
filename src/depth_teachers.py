"""
Canonical depth-teacher feature extraction module.

This module provides optional teacher depth maps (Depth Anything V2, Depth Pro,
Metric3D V2) and converts them into robust teacher features for flood-depth
fusion models.

Runtime guarantees:
- No archived/* imports
- No flood_depth.* imports
- Each teacher is loaded independently
- Teacher failure does not crash ensemble prediction
- No implicit teacher substitution when a teacher fails
"""

from __future__ import annotations

import copy
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    from src.settings import load_settings_dict
except Exception:  # pragma: no cover
    from settings import load_settings_dict  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class _TeacherBackend:
    model_ref: str
    loader: Callable[[], Dict[str, Any]]
    predictor: Callable[[Dict[str, Any], np.ndarray], np.ndarray]
    loaded_payload: Optional[Dict[str, Any]] = None
    loaded: bool = False


class TeacherEnsemble:
    """
    Optional teacher ensemble for depth-feature extraction.

    Required API:
      teachers = TeacherEnsemble(device="cuda")
      teachers.status
      teachers.predict(image_path, water_mask=None)

    Notes:
    - image_path may be a filesystem path OR an RGB numpy array.
    - Teacher model loading is independent per teacher.
    - By default, remote downloads are disabled to keep behavior deterministic.
      Set FLOOD_DEPTH_TEACHERS_ALLOW_DOWNLOAD=1 to enable.
    """

    TEACHER_NAMES: Tuple[str, str, str] = ("DepthAnythingV2", "DepthPro", "Metric3D")

    def __init__(
        self,
        device: str = "cuda",
        *,
        use_fp16: Optional[bool] = None,
        lazy_load: Optional[bool] = None,
        allow_download: Optional[bool] = None,
        teacher_overrides: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> None:
        self.device = self._resolve_device(device)
        self.use_fp16 = self._resolve_use_fp16(use_fp16)
        self.lazy_load = self._resolve_lazy_load(lazy_load)
        self.allow_download = self._resolve_allow_download(allow_download)

        self._teacher_overrides: Mapping[str, Mapping[str, Any]] = teacher_overrides or {}
        self._teacher_backends: Dict[str, _TeacherBackend] = {}
        self._status: Dict[str, Dict[str, Any]] = {}
        self._probed = False

        cfg = self._load_teacher_cfg()
        self._register_teachers(cfg)

        if not self.lazy_load:
            self._probe_all()

    @property
    def status(self) -> Dict[str, Dict[str, Any]]:
        """Return per-teacher load status with independent failure details."""
        if not self._probed:
            self._probe_all()
        return copy.deepcopy(self._status)

    def predict(
        self,
        image_path: Union[str, Path, np.ndarray],
        water_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Compute teacher features for the given image.

        Returns:
          {
            "water_region_valid": bool,
            "teachers": {
              "DepthAnythingV2": { ... required stats ... },
              "DepthPro": { ... },
              "Metric3D": { ... },
            },
            "ensemble": {
              "teacher_mean": ...,
              "teacher_median": ...,
              "teacher_min": ...,
              "teacher_max": ...,
              "teacher_spread": ...,
              "teacher_std": ...,
              "teacher_agreement": ...,
            },
            "meta": { ... }
          }
        """
        image_rgb = self._load_image_rgb(image_path)
        h, w = image_rgb.shape[:2]

        mask_valid, mask_bool = self._validate_water_mask(water_mask, (h, w))

        teachers_out: Dict[str, Dict[str, Any]] = {}
        normalized_region_maps: Dict[str, np.ndarray] = {}
        normalized_region_means: Dict[str, float] = {}

        for name in self.TEACHER_NAMES:
            teacher_status = self._status.get(name, {"available": False, "model": "", "error": "not_initialized"})
            row: Dict[str, Any] = {
                "available": bool(teacher_status.get("available", False)),
                "model": teacher_status.get("model"),
                "error": teacher_status.get("error"),
            }

            if not self._ensure_loaded(name):
                teachers_out[name] = row
                continue

            backend = self._teacher_backends[name]
            assert backend.loaded_payload is not None

            try:
                raw_map = backend.predictor(backend.loaded_payload, image_rgb)
            except Exception as exc:  # pragma: no cover - defensive path
                row["available"] = False
                row["error"] = f"predict failed: {exc}"
                teachers_out[name] = row
                logger.warning("Teacher %s prediction failed: %s", name, exc)
                continue

            if raw_map.shape != (h, w):
                row["available"] = False
                row["error"] = f"invalid depth map shape: {raw_map.shape}, expected {(h, w)}"
                teachers_out[name] = row
                continue

            raw_map = raw_map.astype(np.float32)
            finite = np.isfinite(raw_map)
            valid_ratio = float(finite.mean()) if finite.size else 0.0

            safe_map = raw_map.copy()
            if not finite.all():
                safe_map[~finite] = np.nanmedian(safe_map[finite]) if finite.any() else 0.0

            norm_map, norm_meta = self._robust_normalize(safe_map)

            global_vals = safe_map.reshape(-1)
            if mask_valid:
                region_vals = safe_map[mask_bool]
                region_norm = norm_map[mask_bool]
            else:
                region_vals = global_vals
                region_norm = norm_map.reshape(-1)

            global_stats = self._quantile_stats(global_vals)
            water_stats = self._quantile_stats(region_vals)

            row.update(
                {
                    # Requested canonical names
                    "global_mean": global_stats["mean"],
                    "global_median": global_stats["median"],
                    "p10": global_stats["p10"],
                    "p25": global_stats["p25"],
                    "p50": global_stats["p50"],
                    "p75": global_stats["p75"],
                    "p90": global_stats["p90"],
                    "water_depth_mean": water_stats["mean"],
                    "water_depth_median": water_stats["median"],
                    "water_p10": water_stats["p10"],
                    "water_p25": water_stats["p25"],
                    "water_p50": water_stats["p50"],
                    "water_p75": water_stats["p75"],
                    "water_p90": water_stats["p90"],
                    "valid_pixel_ratio": round(valid_ratio, 6),
                    # Backward-compatible aliases for existing pipeline consumers
                    "mean": global_stats["mean"],
                    "median": global_stats["median"],
                    "water_mean": water_stats["mean"],
                    "water_median": water_stats["median"],
                    "spatial_gradient": self._spatial_gradient(norm_map),
                    "depth_variance": float(np.nanvar(global_vals)),
                    "normalization": norm_meta,
                }
            )

            teachers_out[name] = row
            normalized_region_maps[name] = region_norm.astype(np.float32)
            normalized_region_means[name] = float(np.nanmean(region_norm))

        ensemble = self._aggregate_teacher_metrics(normalized_region_maps, normalized_region_means)

        return {
            "water_region_valid": bool(mask_valid),
            "teachers": teachers_out,
            "ensemble": ensemble,
            "meta": {
                "device": self.device,
                "use_fp16": self.use_fp16,
                "available_teacher_count": int(sum(1 for v in teachers_out.values() if v.get("available"))),
                "total_teachers": len(self.TEACHER_NAMES),
                "mask_pixels": int(mask_bool.sum()) if mask_bool is not None else 0,
            },
        }

    # ------------------------------------------------------------------
    # Teacher registration and loading
    # ------------------------------------------------------------------

    def _register_teachers(self, cfg: Dict[str, Any]) -> None:
        for name in self.TEACHER_NAMES:
            self._status[name] = {
                "available": False,
                "model": "",
                "error": "not_loaded",
            }

        specs = {
            "DepthAnythingV2": cfg.get("DepthAnythingV2", {}),
            "DepthPro": cfg.get("DepthPro", {}),
            "Metric3D": cfg.get("Metric3D", {}),
        }

        for name, spec in specs.items():
            override = self._teacher_overrides.get(name, {})
            model_ref = str(override.get("model") or spec.get("model") or "").strip()

            loader: Callable[[], Dict[str, Any]]
            predictor: Callable[[Dict[str, Any], np.ndarray], np.ndarray]

            if override:
                loader = override.get("loader", lambda: {"kind": "mock"})
                predictor = override.get("predictor", lambda _payload, img: np.zeros(img.shape[:2], dtype=np.float32))
            else:
                loader = lambda model_ref=model_ref: self._load_transformers_teacher(model_ref)
                predictor = self._predict_with_transformers_teacher

            self._teacher_backends[name] = _TeacherBackend(
                model_ref=model_ref,
                loader=loader,
                predictor=predictor,
            )
            self._status[name]["model"] = model_ref
            if not model_ref:
                self._status[name]["error"] = "no model identifier/checkpoint configured"

    def _probe_all(self) -> None:
        for name in self.TEACHER_NAMES:
            self._ensure_loaded(name)
        self._probed = True

    def _ensure_loaded(self, teacher_name: str) -> bool:
        backend = self._teacher_backends[teacher_name]
        if backend.loaded:
            return bool(self._status[teacher_name].get("available", False))

        if not backend.model_ref:
            self._status[teacher_name]["available"] = False
            self._status[teacher_name]["error"] = "no model identifier/checkpoint configured"
            backend.loaded = True
            backend.loaded_payload = None
            return False

        try:
            payload = backend.loader()
            backend.loaded_payload = payload
            backend.loaded = True
            self._status[teacher_name]["available"] = True
            self._status[teacher_name]["error"] = None
            return True
        except Exception as exc:
            backend.loaded_payload = None
            backend.loaded = True
            self._status[teacher_name]["available"] = False
            self._status[teacher_name]["error"] = str(exc)
            return False

    def _load_transformers_teacher(self, model_ref: str) -> Dict[str, Any]:
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except Exception as exc:
            raise RuntimeError(f"transformers not available: {exc}") from exc

        kwargs: Dict[str, Any] = {"local_files_only": not self.allow_download}
        processor = AutoImageProcessor.from_pretrained(model_ref, **kwargs)
        model = AutoModelForDepthEstimation.from_pretrained(model_ref, **kwargs)
        model = model.to(self.device)
        model.eval()
        if self.use_fp16 and self.device.startswith("cuda"):
            model = model.half()

        return {
            "processor": processor,
            "model": model,
            "model_ref": model_ref,
        }

    def _predict_with_transformers_teacher(self, payload: Dict[str, Any], image_rgb: np.ndarray) -> np.ndarray:
        processor = payload["processor"]
        model = payload["model"]

        inputs = processor(images=image_rgb, return_tensors="pt")
        tensor_inputs: Dict[str, Any] = {}
        for k, v in inputs.items():
            if torch.is_tensor(v):
                t = v.to(self.device)
                if self.use_fp16 and self.device.startswith("cuda") and t.dtype == torch.float32:
                    t = t.half()
                tensor_inputs[k] = t
            else:
                tensor_inputs[k] = v

        with torch.inference_mode():
            if self.use_fp16 and self.device.startswith("cuda"):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(**tensor_inputs)
            else:
                outputs = model(**tensor_inputs)

        depth_tensor = self._extract_depth_tensor(outputs)
        if depth_tensor.ndim == 2:
            depth_tensor = depth_tensor.unsqueeze(0).unsqueeze(0)
        elif depth_tensor.ndim == 3:
            depth_tensor = depth_tensor.unsqueeze(1)
        elif depth_tensor.ndim != 4:
            raise RuntimeError(f"Unsupported depth tensor shape: {tuple(depth_tensor.shape)}")

        h, w = image_rgb.shape[:2]
        depth_tensor = F.interpolate(depth_tensor.float(), size=(h, w), mode="bicubic", align_corners=False)
        depth_np = depth_tensor[0, 0].detach().cpu().numpy().astype(np.float32)
        return depth_np

    @staticmethod
    def _extract_depth_tensor(outputs: Any) -> torch.Tensor:
        if hasattr(outputs, "predicted_depth"):
            return outputs.predicted_depth
        if hasattr(outputs, "depth"):
            return outputs.depth
        if isinstance(outputs, Mapping):
            for key in ("predicted_depth", "depth", "output", "last_hidden_state"):
                if key in outputs and torch.is_tensor(outputs[key]):
                    return outputs[key]
        if isinstance(outputs, (tuple, list)) and outputs:
            if torch.is_tensor(outputs[0]):
                return outputs[0]
        raise RuntimeError("Teacher output does not contain a depth tensor")

    # ------------------------------------------------------------------
    # Config + utility helpers
    # ------------------------------------------------------------------

    def _load_teacher_cfg(self) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {}
        try:
            settings = load_settings_dict()
            cfg = (
                settings.get("inference", {})
                .get("depth_teachers", {})
            ) or {}
        except Exception:
            cfg = {}

        def pick(model_env: str, ckpt_env: str, cfg_node: Mapping[str, Any], default_model: str) -> str:
            env_ckpt = os.getenv(ckpt_env, "").strip()
            env_model = os.getenv(model_env, "").strip()
            cfg_ckpt = str(cfg_node.get("checkpoint", "")).strip() if cfg_node else ""
            cfg_model = str(cfg_node.get("model", "")).strip() if cfg_node else ""
            return env_ckpt or env_model or cfg_ckpt or cfg_model or default_model

        da_cfg = cfg.get("depth_anything_v2", {}) if isinstance(cfg, Mapping) else {}
        dp_cfg = cfg.get("depth_pro", {}) if isinstance(cfg, Mapping) else {}
        m3_cfg = cfg.get("metric3d", {}) if isinstance(cfg, Mapping) else {}

        return {
            "DepthAnythingV2": {
                "model": pick(
                    "FLOOD_DEPTH_ANYTHING_V2_MODEL",
                    "FLOOD_DEPTH_ANYTHING_V2_CHECKPOINT",
                    da_cfg,
                    "depth-anything/Depth-Anything-V2-Small-hf",
                )
            },
            "DepthPro": {
                "model": pick(
                    "FLOOD_DEPTH_PRO_MODEL",
                    "FLOOD_DEPTH_PRO_CHECKPOINT",
                    dp_cfg,
                    "apple/DepthPro-hf",
                )
            },
            "Metric3D": {
                "model": pick(
                    "FLOOD_METRIC3D_MODEL",
                    "FLOOD_METRIC3D_CHECKPOINT",
                    m3_cfg,
                    "",
                )
            },
        }

    @staticmethod
    def _resolve_device(device: str) -> str:
        requested = str(device).strip().lower()
        if requested == "cuda" and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _resolve_use_fp16(self, use_fp16: Optional[bool]) -> bool:
        if use_fp16 is not None:
            return bool(use_fp16) and self.device == "cuda"
        env = os.getenv("FLOOD_DEPTH_TEACHERS_USE_FP16", "0").strip().lower()
        return env in {"1", "true", "yes", "on"} and self.device == "cuda"

    @staticmethod
    def _resolve_lazy_load(lazy_load: Optional[bool]) -> bool:
        if lazy_load is not None:
            return bool(lazy_load)
        env = os.getenv("FLOOD_DEPTH_TEACHERS_LAZY_LOAD", "1").strip().lower()
        return env in {"1", "true", "yes", "on"}

    @staticmethod
    def _resolve_allow_download(allow_download: Optional[bool]) -> bool:
        if allow_download is not None:
            return bool(allow_download)
        env = os.getenv("FLOOD_DEPTH_TEACHERS_ALLOW_DOWNLOAD", "0").strip().lower()
        return env in {"1", "true", "yes", "on"}

    @staticmethod
    def _load_image_rgb(image_path: Union[str, Path, np.ndarray]) -> np.ndarray:
        if isinstance(image_path, np.ndarray):
            image_rgb = image_path
            if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
                raise ValueError("image array must have shape (H, W, 3)")
            return image_rgb.astype(np.uint8, copy=False)

        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"image not found: {path}")
        image = Image.open(path).convert("RGB")
        return np.array(image)

    @staticmethod
    def _validate_water_mask(water_mask: Optional[np.ndarray], hw: Tuple[int, int]) -> Tuple[bool, Optional[np.ndarray]]:
        h, w = hw
        if water_mask is None:
            return False, None

        mask = np.asarray(water_mask)
        if mask.ndim == 3:
            mask = mask[..., 0]
        if mask.ndim != 2:
            return False, None
        if mask.shape != (h, w):
            return False, None

        mask_bool = mask.astype(np.float32) > 0.5
        if int(mask_bool.sum()) == 0:
            return False, mask_bool
        return True, mask_bool

    @staticmethod
    def _quantile_stats(values: np.ndarray) -> Dict[str, float]:
        vals = np.asarray(values, dtype=np.float32).reshape(-1)
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            return {
                "mean": 0.0,
                "median": 0.0,
                "p10": 0.0,
                "p25": 0.0,
                "p50": 0.0,
                "p75": 0.0,
                "p90": 0.0,
            }
        return {
            "mean": float(np.mean(finite)),
            "median": float(np.median(finite)),
            "p10": float(np.percentile(finite, 10)),
            "p25": float(np.percentile(finite, 25)),
            "p50": float(np.percentile(finite, 50)),
            "p75": float(np.percentile(finite, 75)),
            "p90": float(np.percentile(finite, 90)),
        }

    @staticmethod
    def _robust_normalize(depth_map: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        arr = np.asarray(depth_map, dtype=np.float32)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return np.zeros_like(arr, dtype=np.float32), {
                "method": "robust_p5_p95",
                "q05": 0.0,
                "q95": 0.0,
                "eps_fallback": True,
            }

        q05 = float(np.percentile(finite, 5))
        q95 = float(np.percentile(finite, 95))
        eps = 1e-6
        if (q95 - q05) < eps:
            mn = float(np.min(finite))
            mx = float(np.max(finite))
            if (mx - mn) < eps:
                return np.zeros_like(arr, dtype=np.float32), {
                    "method": "robust_p5_p95",
                    "q05": q05,
                    "q95": q95,
                    "eps_fallback": True,
                }
            norm = (arr - mn) / (mx - mn)
            norm = np.clip(norm, 0.0, 1.0).astype(np.float32)
            return norm, {
                "method": "minmax_fallback",
                "min": mn,
                "max": mx,
                "eps_fallback": True,
            }

        norm = (arr - q05) / (q95 - q05)
        norm = np.clip(norm, 0.0, 1.0).astype(np.float32)
        return norm, {
            "method": "robust_p5_p95",
            "q05": q05,
            "q95": q95,
            "eps_fallback": False,
        }

    @staticmethod
    def _spatial_gradient(norm_map: np.ndarray) -> float:
        arr = np.asarray(norm_map, dtype=np.float32)
        gy, gx = np.gradient(arr)
        mag = np.sqrt(gx * gx + gy * gy)
        return float(np.nanmean(mag))

    def _aggregate_teacher_metrics(
        self,
        region_maps: Mapping[str, np.ndarray],
        region_means: Mapping[str, float],
    ) -> Dict[str, Any]:
        if not region_means:
            return {
                "teacher_mean": 0.0,
                "teacher_median": 0.0,
                "teacher_min": 0.0,
                "teacher_max": 0.0,
                "teacher_spread": 0.0,
                "teacher_std": 0.0,
                "teacher_agreement": 0.0,
            }

        vals = np.array(list(region_means.values()), dtype=np.float32)
        teacher_mean = float(np.mean(vals))
        teacher_median = float(np.median(vals))
        teacher_min = float(np.min(vals))
        teacher_max = float(np.max(vals))
        teacher_spread = float(teacher_max - teacher_min)
        teacher_std = float(np.std(vals))

        if len(region_maps) <= 1:
            agreement = 1.0
        else:
            names = list(region_maps.keys())
            pairwise_mae = []
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    a = np.asarray(region_maps[names[i]], dtype=np.float32).reshape(-1)
                    b = np.asarray(region_maps[names[j]], dtype=np.float32).reshape(-1)
                    n = min(a.size, b.size)
                    if n == 0:
                        continue
                    if n > 20000:
                        step = int(np.ceil(n / 20000.0))
                        a = a[::step]
                        b = b[::step]
                    pairwise_mae.append(float(np.mean(np.abs(a - b))))
            if pairwise_mae:
                # Agreement in [0,1]: lower normalized disagreement => higher agreement.
                agreement = float(np.clip(1.0 - np.mean(pairwise_mae), 0.0, 1.0))
            else:
                agreement = 0.0

        return {
            "teacher_mean": teacher_mean,
            "teacher_median": teacher_median,
            "teacher_min": teacher_min,
            "teacher_max": teacher_max,
            "teacher_spread": teacher_spread,
            "teacher_std": teacher_std,
            "teacher_agreement": agreement,
        }


__all__ = ["TeacherEnsemble"]
