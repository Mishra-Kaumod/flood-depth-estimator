"""Canonical production depth-teacher feature extraction module."""

from __future__ import annotations

import copy
import inspect
import logging
import os
import platform
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Union

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
    Teacher feature extractor.

    Required API:
      teachers = TeacherEnsemble(device="cuda")
      teachers.status
      teachers.predict(image_path, water_mask=None)
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
        self._cfg = self._load_teacher_cfg()
        self._probed = False

        self._register_teachers(self._cfg)
        if not self.lazy_load:
            self._probe_all()

    @property
    def status(self) -> Dict[str, Dict[str, Any]]:
        if not self._probed:
            self._probe_all()
        return copy.deepcopy(self._status)

    def diagnose(self) -> Dict[str, Any]:
        def _ver(pkg: str) -> Optional[str]:
            try:
                m = __import__(pkg)
                return getattr(m, "__version__", "unknown")
            except Exception:
                return None

        gpu = None
        if torch.cuda.is_available():
            try:
                gpu = torch.cuda.get_device_name(0)
            except Exception:
                gpu = "unknown"

        return {
            "python_version": platform.python_version(),
            "torch_version": getattr(torch, "__version__", None),
            "transformers_version": _ver("transformers"),
            "huggingface_hub_version": _ver("huggingface_hub"),
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu_name": gpu,
            "cache_paths": {
                "HF_HOME": os.getenv("HF_HOME"),
                "TRANSFORMERS_CACHE": os.getenv("TRANSFORMERS_CACHE"),
                "HUGGINGFACE_HUB_CACHE": os.getenv("HUGGINGFACE_HUB_CACHE"),
                "TORCH_HOME": os.getenv("TORCH_HOME"),
            },
            "teacher_config": copy.deepcopy(self._cfg),
            "teacher_status": self.status,
        }

    def predict(self, image_path: Union[str, Path, np.ndarray], water_mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
        image_rgb = self._load_image_rgb(image_path)
        h, w = image_rgb.shape[:2]
        water_valid, water_mask_bool = self._validate_water_mask(water_mask, (h, w))

        teacher_rows: Dict[str, Dict[str, Any]] = {}
        norm_region_maps: Dict[str, np.ndarray] = {}

        for name in self.TEACHER_NAMES:
            base = self._status.get(name, {"available": False, "model": "", "error": "not_initialized"})
            row: Dict[str, Any] = {
                "available": bool(base.get("available", False)),
                "model": base.get("model"),
                "error": base.get("error"),
            }

            if not self._ensure_loaded(name):
                row["available"] = bool(self._status[name].get("available", False))
                row["error"] = self._status[name].get("error")
                teacher_rows[name] = row
                continue

            backend = self._teacher_backends[name]
            assert backend.loaded_payload is not None
            try:
                raw = backend.predictor(backend.loaded_payload, image_rgb)
            except Exception as exc:
                msg = f"predict failed: {exc}"
                self._status[name]["available"] = False
                self._status[name]["error"] = msg
                row["available"] = False
                row["error"] = msg
                teacher_rows[name] = row
                continue

            if raw.shape != (h, w):
                msg = f"invalid depth map shape: {raw.shape}, expected {(h, w)}"
                self._status[name]["available"] = False
                self._status[name]["error"] = msg
                row["available"] = False
                row["error"] = msg
                teacher_rows[name] = row
                continue

            raw = raw.astype(np.float32)
            finite = np.isfinite(raw)
            valid_pixel_ratio = float(finite.mean()) if finite.size else 0.0
            safe = raw.copy()
            if not finite.all():
                fill = float(np.nanmedian(safe[finite])) if finite.any() else 0.0
                safe[~finite] = fill

            norm, norm_meta = self._robust_normalize(safe)
            global_vals = safe.reshape(-1)
            if water_valid:
                region_vals = safe[water_mask_bool]
                region_norm = norm[water_mask_bool]
            else:
                region_vals = global_vals
                region_norm = norm.reshape(-1)

            gs = self._quantile_stats(global_vals)
            ws = self._quantile_stats(region_vals)

            row.update(
                {
                    "global_mean": gs["mean"],
                    "global_median": gs["median"],
                    "p10": gs["p10"],
                    "p25": gs["p25"],
                    "p50": gs["p50"],
                    "p75": gs["p75"],
                    "p90": gs["p90"],
                    "water_depth_mean": ws["mean"],
                    "water_depth_median": ws["median"],
                    "water_p10": ws["p10"],
                    "water_p25": ws["p25"],
                    "water_p50": ws["p50"],
                    "water_p75": ws["p75"],
                    "water_p90": ws["p90"],
                    "valid_pixel_ratio": round(valid_pixel_ratio, 6),
                    "spatial_gradient": self._spatial_gradient(norm),
                    "depth_variance": float(np.nanvar(global_vals)),
                    "normalization": norm_meta,
                    # compatibility aliases
                    "mean": gs["mean"],
                    "median": gs["median"],
                    "water_mean": ws["mean"],
                    "water_median": ws["median"],
                }
            )
            if name == "DepthPro":
                row["metric_depth_mean_m"] = gs["mean"]
                row["metric_depth_median_m"] = gs["median"]

            self._status[name]["available"] = True
            self._status[name]["error"] = None
            row["available"] = True
            row["error"] = None
            teacher_rows[name] = row
            norm_region_maps[name] = region_norm.astype(np.float32)

        ensemble = self._aggregate_teacher_metrics(norm_region_maps)
        available_rows = [r for r in teacher_rows.values() if r.get("available")]
        features = self._aggregate_requested_features(available_rows, ensemble)

        return {
            "water_region_valid": bool(water_valid),
            "teachers": teacher_rows,
            "ensemble": ensemble,
            "features": features,
            "meta": {
                "device": self.device,
                "use_fp16": self.use_fp16,
                "available_teacher_count": int(len(available_rows)),
                "total_teachers": len(self.TEACHER_NAMES),
                "mask_pixels": int(water_mask_bool.sum()) if water_mask_bool is not None else 0,
            },
        }

    def _register_teachers(self, cfg: Dict[str, Any]) -> None:
        for name in self.TEACHER_NAMES:
            self._status[name] = {"available": False, "model": "", "error": "not_loaded"}

        da = cfg.get("DepthAnythingV2", {})
        dp = cfg.get("DepthPro", {})
        m3 = cfg.get("Metric3D", {})

        specs = {
            "DepthAnythingV2": {
                "model": str(da.get("model", "")).strip(),
                "loader": lambda model=str(da.get("model", "")).strip(), rev=(str(da.get("revision", "")).strip() or None): self._load_depth_anything_v2(model, rev),
                "predictor": self._predict_transformers_depth,
            },
            "DepthPro": {
                "model": str(dp.get("model", "")).strip(),
                "loader": lambda model=str(dp.get("model", "")).strip(), ckpt=(str(dp.get("checkpoint", "")).strip() or None): self._load_depth_pro(model, ckpt),
                "predictor": self._predict_depth_pro,
            },
            "Metric3D": {
                "model": str(m3.get("model", "")).strip(),
                "loader": lambda model=str(m3.get("model", "")).strip(), var=(str(m3.get("variant", "metric3d_vit_small")).strip() or "metric3d_vit_small"), ckpt=(str(m3.get("checkpoint", "")).strip() or None): self._load_metric3d(model, var, ckpt),
                "predictor": self._predict_metric3d,
            },
        }

        for name in self.TEACHER_NAMES:
            override = self._teacher_overrides.get(name, {})
            if override:
                model_ref = str(override.get("model") or specs[name]["model"]).strip()
                loader = override.get("loader")
                predictor = override.get("predictor")
                if not callable(loader):
                    loader = specs[name]["loader"]
                if not callable(predictor):
                    predictor = specs[name]["predictor"]
            else:
                model_ref = specs[name]["model"]
                loader = specs[name]["loader"]
                predictor = specs[name]["predictor"]

            self._teacher_backends[name] = _TeacherBackend(
                model_ref=model_ref,
                loader=loader,
                predictor=predictor,
            )
            self._status[name]["model"] = model_ref

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

    def _load_depth_anything_v2(self, model_ref: str, revision: Optional[str]) -> Dict[str, Any]:
        if not model_ref:
            raise RuntimeError("DepthAnythingV2 model identifier is empty")
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except Exception as exc:
            raise RuntimeError(f"transformers import failed: {exc}") from exc

        cache_dir = os.getenv("HUGGINGFACE_HUB_CACHE") or os.getenv("HF_HOME") or os.getenv("TRANSFORMERS_CACHE")
        force_download = os.getenv("FLOOD_DEPTH_TEACHERS_FORCE_DOWNLOAD", "0").strip().lower() in {"1", "true", "yes", "on"}

        source: Union[str, Path] = model_ref
        if "/" in model_ref and not Path(model_ref).exists():
            try:
                from huggingface_hub import snapshot_download
                source = snapshot_download(
                    repo_id=model_ref,
                    revision=revision,
                    local_files_only=not self.allow_download,
                    force_download=force_download and self.allow_download,
                    cache_dir=cache_dir,
                    allow_patterns=["*.json", "*.safetensors", "*.bin", "*.txt", "*.model"],
                )
            except Exception:
                source = model_ref

        kwargs: Dict[str, Any] = {"local_files_only": not self.allow_download}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        if revision:
            kwargs["revision"] = revision

        processor = AutoImageProcessor.from_pretrained(str(source), **kwargs)
        model = AutoModelForDepthEstimation.from_pretrained(str(source), **kwargs)
        model = model.to(self.device)
        model.eval()
        if self.use_fp16 and self.device == "cuda":
            model = model.half()
        return {"kind": "transformers", "processor": processor, "model": model, "model_ref": model_ref, "resolved_source": str(source)}

    def _load_depth_pro(self, model_ref: str, checkpoint: Optional[str]) -> Dict[str, Any]:
        if not model_ref:
            raise RuntimeError("DepthPro model identifier is empty")
        try:
            import depth_pro
        except Exception as exc:
            raise RuntimeError(f"depth_pro import failed: {exc}") from exc

        create_fn = getattr(depth_pro, "create_model_and_transforms", None)
        if create_fn is None:
            raise RuntimeError("depth_pro.create_model_and_transforms not found")

        kwargs: Dict[str, Any] = {}
        sig = inspect.signature(create_fn)
        if "device" in sig.parameters:
            kwargs["device"] = torch.device(self.device)
        if "precision" in sig.parameters and self.use_fp16 and self.device == "cuda":
            kwargs["precision"] = torch.float16
        if checkpoint:
            for k in ("checkpoint", "checkpoint_uri", "checkpoint_path"):
                if k in sig.parameters:
                    kwargs[k] = checkpoint
                    break

        created = create_fn(**kwargs)
        if not isinstance(created, tuple) or len(created) < 2:
            raise RuntimeError("Depth Pro factory did not return (model, transform)")

        model, transform = created[0], created[1]
        if hasattr(model, "to"):
            model = model.to(self.device)
        if hasattr(model, "eval"):
            model.eval()

        return {"kind": "depth_pro", "module": depth_pro, "model": model, "transform": transform, "model_ref": model_ref, "checkpoint": checkpoint}

    def _load_metric3d(self, model_ref: str, variant: str, checkpoint: Optional[str]) -> Dict[str, Any]:
        if not model_ref:
            raise RuntimeError("Metric3D model identifier is empty")
        if checkpoint and not Path(checkpoint).exists():
            raise RuntimeError(f"Metric3D checkpoint not found: {checkpoint}")

        try:
            import metric3d  # type: ignore
            create_fn = getattr(metric3d, "create_model_and_transforms", None)
            if create_fn is not None:
                kwargs: Dict[str, Any] = {}
                sig = inspect.signature(create_fn)
                if "variant" in sig.parameters:
                    kwargs["variant"] = variant
                if "checkpoint" in sig.parameters and checkpoint:
                    kwargs["checkpoint"] = checkpoint
                out = create_fn(**kwargs)
                if isinstance(out, tuple) and len(out) >= 2:
                    model, transform = out[0], out[1]
                    if hasattr(model, "to"):
                        model = model.to(self.device)
                    if hasattr(model, "eval"):
                        model.eval()
                    return {"kind": "metric3d_pkg", "module": metric3d, "model": model, "transform": transform, "model_ref": model_ref, "variant": variant, "checkpoint": checkpoint}
        except Exception:
            pass

        try:
            model = torch.hub.load(model_ref, variant, pretrained=(checkpoint is None), trust_repo=True)
            if checkpoint:
                state = torch.load(checkpoint, map_location="cpu", weights_only=False)
                sd = state.get("state_dict", state) if isinstance(state, dict) else state
                model.load_state_dict(sd, strict=False)
            model = model.to(self.device)
            model.eval()
            return {"kind": "metric3d_hub", "module": None, "model": model, "transform": None, "model_ref": model_ref, "variant": variant, "checkpoint": checkpoint}
        except Exception as exc:
            raise RuntimeError(
                "Metric3D load failed. Install official Metric3D dependency or configure a valid checkpoint. "
                f"Details: {exc}"
            ) from exc

    def _predict_transformers_depth(self, payload: Dict[str, Any], image_rgb: np.ndarray) -> np.ndarray:
        processor = payload["processor"]
        model = payload["model"]
        inputs = processor(images=image_rgb, return_tensors="pt")

        tensor_inputs: Dict[str, Any] = {}
        for k, v in inputs.items():
            if torch.is_tensor(v):
                t = v.to(self.device)
                if self.use_fp16 and self.device == "cuda" and t.dtype == torch.float32:
                    t = t.half()
                tensor_inputs[k] = t
            else:
                tensor_inputs[k] = v

        with torch.inference_mode():
            if self.use_fp16 and self.device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = model(**tensor_inputs)
            else:
                out = model(**tensor_inputs)

        depth = self._extract_depth_tensor(out)
        return self._resize_depth_to_image(depth, image_rgb.shape[:2])

    def _predict_depth_pro(self, payload: Dict[str, Any], image_rgb: np.ndarray) -> np.ndarray:
        depth_pro = payload["module"]
        model = payload["model"]
        transform = payload["transform"]
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                tmp_path = f.name
            Image.fromarray(image_rgb).save(tmp_path)

            if hasattr(depth_pro, "load_rgb"):
                loaded = depth_pro.load_rgb(tmp_path)
                if isinstance(loaded, tuple):
                    rgb_img = loaded[0]
                    f_px = loaded[2] if len(loaded) > 2 else None
                else:
                    rgb_img = loaded
                    f_px = None
            else:
                rgb_img = Image.open(tmp_path).convert("RGB")
                f_px = None

            x = transform(rgb_img) if callable(transform) else rgb_img
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            if torch.is_tensor(x):
                if x.ndim == 3:
                    x = x.unsqueeze(0)
                x = x.to(self.device)
                if self.use_fp16 and self.device == "cuda" and x.dtype == torch.float32:
                    x = x.half()

            infer_fn = getattr(model, "infer", None)
            with torch.inference_mode():
                if callable(infer_fn):
                    kwargs: Dict[str, Any] = {}
                    sig = inspect.signature(infer_fn)
                    if "f_px" in sig.parameters and f_px is not None:
                        kwargs["f_px"] = f_px
                    out = infer_fn(x, **kwargs)
                else:
                    out = model(x)

            depth = self._extract_depth_tensor(out)
            return self._resize_depth_to_image(depth, image_rgb.shape[:2])
        finally:
            if tmp_path:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def _predict_metric3d(self, payload: Dict[str, Any], image_rgb: np.ndarray) -> np.ndarray:
        model = payload["model"]
        transform = payload.get("transform")

        if callable(transform):
            x = transform(Image.fromarray(image_rgb))
        else:
            x = torch.from_numpy(image_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        x = x.to(self.device)
        if self.use_fp16 and self.device == "cuda" and x.dtype == torch.float32:
            x = x.half()

        with torch.inference_mode():
            out = model(x)
        depth = self._extract_depth_tensor(out)
        return self._resize_depth_to_image(depth, image_rgb.shape[:2])

    @staticmethod
    def _extract_depth_tensor(outputs: Any) -> torch.Tensor:
        if torch.is_tensor(outputs):
            return outputs
        if hasattr(outputs, "predicted_depth"):
            return outputs.predicted_depth
        if hasattr(outputs, "depth"):
            return outputs.depth
        if isinstance(outputs, Mapping):
            for k in ("predicted_depth", "depth", "metric_depth", "depth_map", "output"):
                if k in outputs and torch.is_tensor(outputs[k]):
                    return outputs[k]
        if isinstance(outputs, (tuple, list)):
            for item in outputs:
                if torch.is_tensor(item):
                    return item
                if isinstance(item, Mapping):
                    for k in ("predicted_depth", "depth", "metric_depth", "depth_map"):
                        if k in item and torch.is_tensor(item[k]):
                            return item[k]
        raise RuntimeError("Teacher output does not contain a tensor depth map")

    @staticmethod
    def _resize_depth_to_image(depth: torch.Tensor, hw: Tuple[int, int]) -> np.ndarray:
        h, w = hw
        d = depth
        if d.ndim == 2:
            d = d.unsqueeze(0).unsqueeze(0)
        elif d.ndim == 3:
            d = d.unsqueeze(1)
        elif d.ndim != 4:
            raise RuntimeError(f"Unsupported depth tensor shape: {tuple(d.shape)}")
        d = F.interpolate(d.float(), size=(h, w), mode="bicubic", align_corners=False)
        return d[0, 0].detach().cpu().numpy().astype(np.float32)

    def _load_teacher_cfg(self) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {}
        try:
            settings = load_settings_dict()
            cfg = settings.get("inference", {}).get("depth_teachers", {}) or {}
        except Exception:
            cfg = {}

        da = cfg.get("depth_anything_v2", {}) if isinstance(cfg, Mapping) else {}
        dp = cfg.get("depth_pro", {}) if isinstance(cfg, Mapping) else {}
        m3 = cfg.get("metric3d", {}) if isinstance(cfg, Mapping) else {}

        return {
            "DepthAnythingV2": {
                "model": os.getenv("FLOOD_DEPTH_ANYTHING_V2_MODEL", str(da.get("model", "")).strip() or "depth-anything/Depth-Anything-V2-Small-hf"),
                "revision": os.getenv("FLOOD_DEPTH_ANYTHING_V2_REVISION", str(da.get("revision", "")).strip() or "main"),
            },
            "DepthPro": {
                "model": os.getenv("FLOOD_DEPTH_PRO_MODEL", str(dp.get("model", "")).strip() or "apple/DepthPro"),
                "checkpoint": os.getenv("FLOOD_DEPTH_PRO_CHECKPOINT", str(dp.get("checkpoint", "")).strip()),
            },
            "Metric3D": {
                "model": os.getenv("FLOOD_METRIC3D_MODEL", str(m3.get("model", "")).strip() or "YvanYin/Metric3D"),
                "variant": os.getenv("FLOOD_METRIC3D_VARIANT", str(m3.get("variant", "")).strip() or "metric3d_vit_small"),
                "checkpoint": os.getenv("FLOOD_METRIC3D_CHECKPOINT", str(m3.get("checkpoint", "")).strip()),
            },
        }

    @staticmethod
    def _resolve_device(device: str) -> str:
        d = str(device).strip().lower()
        if d == "cuda" and torch.cuda.is_available():
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
        env = os.getenv("FLOOD_DEPTH_TEACHERS_ALLOW_DOWNLOAD", "1").strip().lower()
        return env in {"1", "true", "yes", "on"}

    @staticmethod
    def _load_image_rgb(image_path: Union[str, Path, np.ndarray]) -> np.ndarray:
        if isinstance(image_path, np.ndarray):
            arr = image_path
            if arr.ndim != 3 or arr.shape[2] != 3:
                raise ValueError("image array must have shape (H, W, 3)")
            return arr.astype(np.uint8, copy=False)
        p = Path(image_path)
        if not p.exists():
            raise FileNotFoundError(f"image not found: {p}")
        return np.array(Image.open(p).convert("RGB"))

    @staticmethod
    def _validate_water_mask(water_mask: Optional[np.ndarray], hw: Tuple[int, int]) -> Tuple[bool, Optional[np.ndarray]]:
        if water_mask is None:
            return False, None
        h, w = hw
        m = np.asarray(water_mask)
        if m.ndim == 3:
            m = m[..., 0]
        if m.ndim != 2 or m.shape != (h, w):
            return False, None
        mb = m.astype(np.float32) > 0.5
        if int(mb.sum()) == 0:
            return False, mb
        return True, mb

    @staticmethod
    def _quantile_stats(values: np.ndarray) -> Dict[str, float]:
        vals = np.asarray(values, dtype=np.float32).reshape(-1)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return {"mean": 0.0, "median": 0.0, "p10": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0}
        return {
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "p10": float(np.percentile(vals, 10)),
            "p25": float(np.percentile(vals, 25)),
            "p50": float(np.percentile(vals, 50)),
            "p75": float(np.percentile(vals, 75)),
            "p90": float(np.percentile(vals, 90)),
        }

    @staticmethod
    def _robust_normalize(depth_map: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        arr = np.asarray(depth_map, dtype=np.float32)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return np.zeros_like(arr, dtype=np.float32), {"method": "robust_p5_p95", "q05": 0.0, "q95": 0.0, "eps_fallback": True}
        q05 = float(np.percentile(finite, 5))
        q95 = float(np.percentile(finite, 95))
        eps = 1e-6
        if (q95 - q05) < eps:
            mn = float(np.min(finite))
            mx = float(np.max(finite))
            if (mx - mn) < eps:
                return np.zeros_like(arr, dtype=np.float32), {"method": "robust_p5_p95", "q05": q05, "q95": q95, "eps_fallback": True}
            n = np.clip((arr - mn) / (mx - mn), 0.0, 1.0).astype(np.float32)
            return n, {"method": "minmax_fallback", "min": mn, "max": mx, "eps_fallback": True}
        n = np.clip((arr - q05) / (q95 - q05), 0.0, 1.0).astype(np.float32)
        return n, {"method": "robust_p5_p95", "q05": q05, "q95": q95, "eps_fallback": False}

    @staticmethod
    def _spatial_gradient(norm_map: np.ndarray) -> float:
        arr = np.asarray(norm_map, dtype=np.float32)
        gy, gx = np.gradient(arr)
        return float(np.nanmean(np.sqrt(gx * gx + gy * gy)))

    @staticmethod
    def _aggregate_teacher_metrics(region_maps: Mapping[str, np.ndarray]) -> Dict[str, float]:
        if not region_maps:
            return {"teacher_mean": 0.0, "teacher_median": 0.0, "teacher_spread": 0.0, "teacher_std": 0.0, "teacher_agreement": 0.0}

        means = np.array([float(np.nanmean(v)) for v in region_maps.values()], dtype=np.float32)
        out = {
            "teacher_mean": float(np.mean(means)),
            "teacher_median": float(np.median(means)),
            "teacher_spread": float(np.max(means) - np.min(means)),
            "teacher_std": float(np.std(means)),
            "teacher_agreement": 1.0,
        }

        if len(region_maps) > 1:
            names = list(region_maps.keys())
            maes = []
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
                    maes.append(float(np.mean(np.abs(a - b))))
            out["teacher_agreement"] = float(np.clip(1.0 - (np.mean(maes) if maes else 1.0), 0.0, 1.0))

        return out

    @staticmethod
    def _aggregate_requested_features(available_rows: list[Dict[str, Any]], ensemble: Dict[str, float]) -> Dict[str, float]:
        if not available_rows:
            return {
                "global_mean": 0.0,
                "global_median": 0.0,
                "p10": 0.0,
                "p25": 0.0,
                "p50": 0.0,
                "p75": 0.0,
                "p90": 0.0,
                "water_depth_mean": 0.0,
                "water_depth_median": 0.0,
                "teacher_mean": float(ensemble.get("teacher_mean", 0.0)),
                "teacher_median": float(ensemble.get("teacher_median", 0.0)),
                "teacher_spread": float(ensemble.get("teacher_spread", 0.0)),
                "teacher_std": float(ensemble.get("teacher_std", 0.0)),
                "teacher_agreement": float(ensemble.get("teacher_agreement", 0.0)),
                "valid_pixel_ratio": 0.0,
            }

        def avg(k: str) -> float:
            vals = [float(r.get(k, 0.0)) for r in available_rows if isinstance(r.get(k), (int, float))]
            return float(np.mean(vals)) if vals else 0.0

        return {
            "global_mean": avg("global_mean"),
            "global_median": avg("global_median"),
            "p10": avg("p10"),
            "p25": avg("p25"),
            "p50": avg("p50"),
            "p75": avg("p75"),
            "p90": avg("p90"),
            "water_depth_mean": avg("water_depth_mean"),
            "water_depth_median": avg("water_depth_median"),
            "teacher_mean": float(ensemble.get("teacher_mean", 0.0)),
            "teacher_median": float(ensemble.get("teacher_median", 0.0)),
            "teacher_spread": float(ensemble.get("teacher_spread", 0.0)),
            "teacher_std": float(ensemble.get("teacher_std", 0.0)),
            "teacher_agreement": float(ensemble.get("teacher_agreement", 0.0)),
            "valid_pixel_ratio": avg("valid_pixel_ratio"),
        }


__all__ = ["TeacherEnsemble"]
