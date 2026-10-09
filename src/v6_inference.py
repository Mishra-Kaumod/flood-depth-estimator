"""Shared V6 entrypoint contract; no depth decisions belong here.

All deployment surfaces use PIL-decoded uint8 RGB, the same pipeline factory,
and V6ShadowPipeline.predict(). Video must save its frame before using this
loader so image uploads and file inference see the same encoded image.
"""
from io import BytesIO
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def load_v6_rgb(source: str | Path | bytes) -> np.ndarray:
    """Decode encoded image bytes or a file as RGB without resizing/normalizing."""
    with Image.open(BytesIO(source) if isinstance(source, bytes) else source) as image:
        return np.asarray(image.convert("RGB"))


def create_v6_pipeline():
    """Single model-construction path, with the existing model defaults intact."""
    from .efficientnet_depth_signal import EfficientNetDepthSignal
    from .v6_shadow_pipeline import V6ShadowPipeline

    return V6ShadowPipeline(EfficientNetDepthSignal(), evidence_collector=V6EvidenceCollector())


class V6EvidenceCollector:
    """Shared diagnostic extraction, never a depth estimator or legacy runner.

    Optional artifacts are local/cache-only: unavailable backends are explicit,
    never silently substituted with contour objects or synthetic dense depth.
    Models are loaded lazily once per pipeline and failures are isolated.
    """

    def __init__(self, config=None):
        from src.settings import load_settings_dict
        self.config = config if config is not None else load_settings_dict()
        self.options = self.config.get("inference", {}).get("v6_evidence", {})
        self._models = {}
        self._load_failures = {}

    def _model(self, name, factory):
        if name in self._load_failures:
            raise RuntimeError(self._load_failures[name])
        if name not in self._models:
            try:
                self._models[name] = factory()
            except Exception as exc:
                self._load_failures[name] = type(exc).__name__
                raise
        return self._models[name]

    def _water(self, rgb, features):
        import hashlib
        from src.water_region_detector import WaterRegionDetector
        detector = self._model("water", WaterRegionDetector)
        mask, _ = detector.detect(rgb)
        mask = np.asarray(mask) > 0
        if mask.shape != rgb.shape[:2]:
            raise ValueError("Water mask shape mismatch")
        h = mask.shape[0]
        features["water_coverage_pct"] = float(mask.mean() * 100)
        for name, band in (("far", mask[:int(h * .25)]),
                           ("mid", mask[int(h * .25):int(h * .60)]),
                           ("near", mask[int(h * .60):])):
            features[name + "_water_coverage_pct"] = float(band.mean() * 100) if band.size else None
        rows = np.flatnonzero(mask.any(axis=1))
        features.update(water_segmentation_backend="classical_water_region_detector",
                        water_mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
                        water_mask_shape=list(mask.shape),
                        waterline_image_row_ratio=float(rows[0] / h) if rows.size else None,
                        mask_quality={"shape_valid": True, "calibrated_quality": None,
                                      "note": "Classical color/contrast mask; quality not calibrated"})
        return mask

    def _objects(self, rgb, mask, features):
        from pathlib import Path
        def load():
            from ultralytics import YOLO
            path = Path(self.options.get("yolo_model_path", "yolov8n.pt"))
            if not path.is_file():
                raise FileNotFoundError("Local YOLO checkpoint unavailable")
            return YOLO(str(path))
        model = self._model("yolo", load)
        # Existing detector native confidence setting, not a correction threshold.
        # Ultralytics NumPy input is BGR; the shared V6 input remains RGB.
        output = model.predict(rgb[..., ::-1].copy(), conf=.25, verbose=False)[0]
        rows = []
        h, w = rgb.shape[:2]
        for box in output.boxes:
            label = str(model.names[int(box.cls.item())])
            if label not in {"car", "person", "bus", "truck", "motorcycle", "bicycle"}:
                continue
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            region = mask[y1:y2, x1:x2] if mask is not None else None
            occupied = np.flatnonzero(region.any(axis=1)) if region is not None else []
            rows.append({"label": label, "detector_confidence": float(box.conf.item()),
                         "bbox": [x1, y1, x2, y2], "area_ratio": (x2-x1)*(y2-y1)/(h*w),
                         "water_submersion_ratio": float(region.mean()) if region is not None else None,
                         "waterline_height_ratio": float((y2-y1-occupied[0])/(y2-y1)) if len(occupied) else None})
        features["reference_object_diagnostics"] = rows
        features["reference_available"] = bool(rows)
        features["reference_count"] = len(rows)
        overlap = [row["water_submersion_ratio"] for row in rows if row["water_submersion_ratio"] is not None]
        features["object_consistency"] = {"count": len(rows), "overlap_std": float(np.std(overlap)) if len(overlap) >= 2 else None,
                                          "note": "Overlap is mask evidence, not physical submersion or metric reliability"}
        features["reference_detection_backend"] = "yolov8_local"

    def _relative(self, rgb, mask, features):
        def load():
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            name = self.options.get("depth_anything_model", "depth-anything/Depth-Anything-V2-Small-hf")
            return (AutoImageProcessor.from_pretrained(name, local_files_only=True),
                    AutoModelForDepthEstimation.from_pretrained(name, local_files_only=True).eval())
        import torch
        processor, model = self._model("relative_depth", load)
        with torch.no_grad():
            depth = model(**processor(images=rgb, return_tensors="pt")).predicted_depth
            depth = torch.nn.functional.interpolate(depth.unsqueeze(1), size=rgb.shape[:2], mode="bicubic", align_corners=False)[0, 0].numpy()
        if not np.isfinite(depth).all() or depth.max() <= depth.min():
            raise ValueError("Invalid relative depth map")
        depth = (depth-depth.min())/(depth.max()-depth.min())
        features.update(dense_depth_relative_p90=float(np.percentile(depth, 90)),
                        dense_depth_map_min=float(depth.min()), dense_depth_map_max=float(depth.max()),
                        dense_relative_water_median=float(np.median(depth[mask])) if mask is not None and mask.any() else None,
                        dense_depth_backend="depth_anything_v2_relative_local")

    def _semantic(self, rgb, features, name, cfg, classes):
        import torch
        from torchvision import models, transforms
        def load():
            checkpoint = torch.load(cfg["model_path"], map_location="cpu", weights_only=True)
            if list(checkpoint.get("class_names", [])) != classes:
                raise ValueError("Semantic class contract mismatch")
            model = models.mobilenet_v3_small(weights=None)
            model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, len(classes))
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            return model.eval()
        model = self._model(name, load)
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                        transforms.Normalize([.485,.456,.406], [.229,.224,.225])])
        with torch.no_grad():
            values = model(transform(Image.fromarray(rgb)).unsqueeze(0)).softmax(dim=1)[0].tolist()
        if name == "road_scene":
            features.update({"road_scene_"+label+"_probability": value for label, value in zip(classes, values)})
        else:
            features[name + "_probability"] = values[0]

    def collect(self, image_rgb):
        from src.v6_shadow_contract import EvidenceBundle
        # Isolate collectors from the primary model's input and each other.
        rgb = np.array(image_rgb, copy=True)
        features, status = {}, {}
        mask = None
        def run(name, enabled, action):
            if not enabled:
                status[name] = {"status": "disabled"}
                return None
            before = dict(features)
            try:
                value = action()
                status[name] = {"status": "available"}
                return value
            except Exception as exc:
                features.clear()
                features.update(before)
                status[name] = {"status": "unavailable", "reason": type(exc).__name__}
                return None
        enabled = self.options.get("enabled", True)
        mask = run("water", enabled and self.options.get("water", True), lambda: self._water(rgb.copy(), features))
        run("yolo", enabled and self.options.get("yolo", True), lambda: self._objects(rgb.copy(), mask, features))
        run("relative_depth", enabled and self.options.get("relative_depth", True), lambda: self._relative(rgb.copy(), mask, features))
        inference = self.config.get("inference", {})
        scene = inference.get("road_scene_classifier", {})
        guards = inference.get("no_water_guard", {})
        semantics = enabled and self.options.get("semantics", True)
        run("road_scene", semantics and scene.get("enabled", False),
            lambda: self._semantic(rgb.copy(), features, "road_scene", scene, ["dry_road","wet_road","shallow_flood","meaningful_flood"]))
        run("no_water", semantics and guards.get("enabled", False),
            lambda: self._semantic(rgb.copy(), features, "no_water", guards, ["no_water","water"]))
        wet = {**guards, "model_path": guards.get("wet_road_guard_model_path", guards.get("secondary_model_path"))}
        run("wet_road_no_water", semantics and guards.get("enabled", False) and bool(wet["model_path"]),
            lambda: self._semantic(rgb.copy(), features, "wet_road_no_water", wet, ["no_water","water"]))
        votes = {}
        scene_keys = ["road_scene_"+name+"_probability" for name in ("dry_road","wet_road","shallow_flood","meaningful_flood")]
        if all(key in features for key in scene_keys):
            votes["road_scene"] = ("dry_road","wet_road","shallow_flood","meaningful_flood")[int(np.argmax([features[key] for key in scene_keys]))]
        for name in ("no_water", "wet_road_no_water"):
            if name+"_probability" in features:
                value = features[name+"_probability"]
                votes[name] = "no_water" if value >= 1-value else "water"
        features["semantic_native_predictions"] = votes
        status["experimental_candidates"] = {"status": "not_collected", "reason": "Legacy region/mask feature contract requires separate validation"}
        # No unsupported scene labels are inferred from coverage or primary depth.
        features["scene_slices"] = {name: None for name in (
            "dry_road", "wet_road_without_flood", "depth_0_10", "depth_10_20",
            "reflections", "muddy_water", "distant_water", "perspective_issues", "occlusion")}
        features["scene_slices"]["label_source"] = "requires_review_or_ground_truth"
        if mask is not None:
            mask.setflags(write=False)
        return EvidenceBundle(features, status, mask)


def finite_depth(value: Any) -> float | None:
    """Serialization only: retain the existing video handling of invalid values."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def v6_depth_payload(result: Any) -> dict[str, Any]:
    """Expose V6's final centimeters; never select, correct, or fuse depth."""
    return {
        "primary_depth_cm": finite_depth(result.primary_depth_cm),
        "final_shadow_depth_cm": finite_depth(result.final_shadow_depth_cm),
        "numerical_owner": result.numerical_owner,
        "final_v6_depth_cm": finite_depth(getattr(result, "final_v6_depth_cm", result.final_shadow_depth_cm)),
    }
