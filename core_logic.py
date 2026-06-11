"""core_logic.py  —  Ensemble Flood Depth Estimator
==================================================
Three independent depth signals fused via confidence-weighted voting:
  Engine A  — Geometric wheel scaling      (Approach 1)
  Engine B  — Depth-ratio anchor fix       (Approach 4 corrected)
  Engine C  — Depth-Anything V2 fallback   (original, math-corrected)
Final depth = weighted average where each engine's weight is scaled
by its own internal confidence score.  An engine that fails or has
low confidence contributes proportionally less to the final answer.
"""

import os

import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Global model cache  (loaded once at import time)
# ---------------------------------------------------------------------------
yolo_model = YOLO("yolov8n.pt")
depth_processor = AutoImageProcessor.from_pretrained(
    "depth-anything/Depth-Anything-V2-Small-hf"
)
depth_model = AutoModelForDepthEstimation.from_pretrained(
    "depth-anything/Depth-Anything-V2-Small-hf"
)
torch.set_num_threads(2)

# ---------------------------------------------------------------------------
# Physical constants  (Indian automotive fleet, measured values)
# ---------------------------------------------------------------------------
VEHICLE_SPECS = {
    # class_name : (wheel_diameter_cm, ground_clearance_cm, typical_height_cm)
    "car": (65, 18, 145),
    "truck": (90, 25, 200),
    "bus": (100, 28, 280),
    "motorcycle": (56, 14, 90),
    "person": (None, None, 170),  # no wheel — use height only
}

# Base weights before confidence adjustment.
# These reflect typical accuracy ceiling per engine:
#   A (geometry)  best when wheel clearly visible
#   B (ratio)     best when anchor object present, geometry unclear
#   C (global)    best when no objects, pure scene depth
BASE_WEIGHTS = {
    "engine_a": 0.50,
    "engine_b": 0.30,
    "engine_c": 0.20,
}

INFERENCE_SIZE = 448

# ---------------------------------------------------------------------------
# TripleEnginePipeline  —  unchanged classifier (kept for flood detection)
# ---------------------------------------------------------------------------
class TripleEnginePipeline:
    def get_water_mask(self, image_matrix):
        """
        PHASE 3 PLACEHOLDER: Semantic Segmentation Engine.
        Currently routes to the legacy Phase 2 classifier until the new model is trained.
        """
        # Fallback to existing logic: Convert probability to a "coverage percentage"
        probability = self.predict_flood_probability(image_matrix)
        simulated_coverage_pct = probability * 100.0
        
        # Return a dummy mask (None) and the simulated coverage percentage
        return None, simulated_coverage_pct

        
    def predict_flood_probability(self, cv2_image_matrix):
        rgb = cv2.cvtColor(cv2_image_matrix, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        tensor = self.classifier_transforms(pil).unsqueeze(0)
        with torch.no_grad():
            out = self.custom_classifier(tensor)
            prob = torch.nn.functional.softmax(out, dim=1)
        return float(prob[0][1])


# ---------------------------------------------------------------------------
# Shared helper: run Depth-Anything V2 once, return normalised depth map
# ---------------------------------------------------------------------------
def _get_depth_map(resized_bgr):
    rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    inputs = depth_processor(images=rgb, return_tensors="pt")
    with torch.no_grad():
        raw = depth_model(**inputs).predicted_depth
        raw = torch.nn.functional.interpolate(
            raw.unsqueeze(1),
            size=(INFERENCE_SIZE, INFERENCE_SIZE),
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    dm = raw.cpu().numpy()
    mn, mx = dm.min(), dm.max()
    return (dm - mn) / (mx - mn) if (mx - mn) > 0 else dm


# ---------------------------------------------------------------------------
# Shared helper: run YOLO, return best anchor dict or None
# ---------------------------------------------------------------------------
def _get_best_anchor(resized_bgr):
    results = yolo_model(resized_bgr, verbose=False)[0]
    best = None
    best_area = 0
    for box in results.boxes:
        label = yolo_model.names[int(box.cls[0])]
        if label not in VEHICLE_SPECS:
            continue
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        area = (x2 - x1) * (y2 - y1)
        if area > best_area:
            best_area = area
            best = {
                "label": label,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "cx": (x1 + x2) // 2,
                "box_h": y2 - y1,
                "box_w": x2 - x1,
                "specs": VEHICLE_SPECS[label],
            }
    return best


# ===========================================================================
# ENGINE A  —  Geometric wheel scaling
# Physical basis: wheel diameter is a known real-world length.
# Pixel height of the wheel → px/cm ratio → waterline offset → depth in cm.
# Confidence driven by: wheel visibility, bounding-box aspect ratio,
#                       waterline clearly below mid-box.
# ===========================================================================
def _engine_a_geometric(anchor, waterline_y, img_h):
    """
    Returns (depth_cm, confidence 0-1) or (None, 0) if not applicable.
    """
    if anchor is None:
        return None, 0.0

    wheel_d_cm, ground_clr_cm, _ = anchor["specs"]
    if wheel_d_cm is None:  # person — no wheel geometry
        return None, 0.0

    box_h = anchor["box_h"]
    wheel_px = box_h * 0.28  # empirical: wheel occupies ~28% of box height
    if wheel_px < 8:  # box too small to be reliable
        return None, 0.0

    px_per_cm = wheel_px / wheel_d_cm
    road_y = anchor["y2"]
    splash_trim = int(box_h * 0.08)  # ignore bottom 8% (splash noise)
    adjusted_waterline = max(waterline_y, anchor["y1"])
    pixel_offset = (road_y - splash_trim) - adjusted_waterline
    if pixel_offset <= 0:
        return 0.0, 0.4  # waterline at or below road — no flood

    depth_cm = pixel_offset / px_per_cm
    size_conf = min(1.0, box_h / (img_h * 0.25))
    depth_conf = 1.0 if depth_cm < 80 else max(0.2, 1 - (depth_cm - 80) / 200)
    aspect_ok = 0.9 if 0.3 < (anchor["box_w"] / max(box_h, 1)) < 3.0 else 0.5
    confidence = round(size_conf * depth_conf * aspect_ok, 3)

    return round(float(depth_cm), 1), confidence


# ===========================================================================
# ENGINE B  —  Depth-ratio anchor fix
# Physical basis: relative depth values at waterline vs road contact are
# proportional to real distances.  Anchor that ratio to known ground clearance.
# More robust than Engine A when wheel boundaries are unclear.
# ===========================================================================
def _engine_b_depth_ratio(anchor, depth_map, waterline_y):
    """
    Returns (depth_cm, confidence 0-1) or (None, 0) if not applicable.
    """
    if anchor is None:
        return None, 0.0

    _, ground_clr_cm, _ = anchor["specs"]
    if ground_clr_cm is None:
        return None, 0.0

    cx = anchor["cx"]
    road_y = min(anchor["y2"], INFERENCE_SIZE - 1)
    wl_y_clamped = min(max(waterline_y, 0), INFERENCE_SIZE - 1)
    cx_clamped = min(max(cx, 0), INFERENCE_SIZE - 1)

    def sample(y, x, r=3):
        y0, y1 = max(0, y - r), min(INFERENCE_SIZE, y + r + 1)
        x0, x1 = max(0, x - r), min(INFERENCE_SIZE, x + r + 1)
        patch = depth_map[y0:y1, x0:x1]
        return float(np.median(patch)) if patch.size else 0.0

    d_road = sample(road_y, cx_clamped)
    d_waterline = sample(wl_y_clamped, cx_clamped)
    if d_road < 1e-4:
        return None, 0.0

    ratio = d_waterline / d_road
    if ratio <= 1.0:
        depth_cm = 0.0
        confidence = 0.35
    else:
        depth_cm = round((ratio - 1.0) * ground_clr_cm * 2.5, 1)
        ratio_conf = min(1.0, (ratio - 1.0) * 3.0)
        size_conf = min(1.0, anchor["box_h"] / (INFERENCE_SIZE * 0.20))
        confidence = round(ratio_conf * size_conf, 3)

    return float(depth_cm), confidence


# ===========================================================================
# ENGINE C  —  Corrected global depth fallback
# Physical basis: lower-third scene depth relative to full-scene median.
# Valid ONLY when no anchor object present.  Weight is intentionally low.
# Compared to original: removed the "× 105" magic number; instead uses
# a scene-relative ratio that at least scales with image content.
# ===========================================================================
def _engine_c_global_fallback(depth_map, anchor):
    """
    Returns (depth_cm, confidence 0-1).
    Confidence is always low — this is a last-resort signal.
    """
    lower_start = int(INFERENCE_SIZE * 0.65)
    lower_zone = depth_map[lower_start:, :]
    upper_zone = depth_map[:lower_start, :]

    lower_mean = float(np.mean(lower_zone))
    upper_mean = float(np.mean(upper_zone))
    if upper_mean < 1e-4:
        return 0.0, 0.05

    scene_ratio = lower_mean / upper_mean
    if scene_ratio < 1.05:
        depth_cm = 0.0
        conf = 0.10
    else:
        depth_cm = round((scene_ratio - 1.0) * 100.0, 1)
        conf = min(0.25, (scene_ratio - 1.0) * 0.5)

    if anchor is not None:
        conf *= 0.3

    return float(depth_cm), round(conf, 3)


# ===========================================================================
# ENSEMBLE FUSION
# Weighted average where effective weight = base_weight × engine_confidence.
# If total effective weight < 0.05, return zero depth (no signal at all).
# ===========================================================================
def _fuse_ensemble(a_depth, a_conf, b_depth, b_conf, c_depth, c_conf):
    engines = [
        ("engine_a", a_depth, a_conf),
        ("engine_b", b_depth, b_conf),
        ("engine_c", c_depth, c_conf),
    ]
    total_w = 0.0
    weighted = 0.0
    breakdown = {}

    for name, depth, conf in engines:
        if depth is None:
            effective_w = 0.0
        else:
            effective_w = BASE_WEIGHTS[name] * conf
        breakdown[name] = {
            "depth_cm": depth,
            "confidence": conf,
            "base_weight": BASE_WEIGHTS[name],
            "effective_w": round(effective_w, 4),
        }
        if depth is not None:
            weighted += effective_w * depth
            total_w += effective_w

    if total_w < 0.05:
        final_depth = 0.0
        final_conf = 0.0
    else:
        final_depth = round(weighted / total_w, 1)
        final_conf = round(total_w / sum(BASE_WEIGHTS.values()), 3)

    for name in breakdown:
        ew = breakdown[name]["effective_w"]
        breakdown[name]["contribution_pct"] = (
            round(ew / total_w * 100, 1) if total_w > 0 else 0.0
        )

    return final_depth, min(1.0, final_conf), breakdown


# ===========================================================================
# Waterline detector — replaces the old hard-coded 90% rule
# Combines horizontal-line score from water_detection with depth-map
# gradient to find the most likely water surface row.
# ===========================================================================
def _find_waterline_y(depth_map, anchor, img_h):
    """
    Returns the best-estimate waterline y-pixel in INFERENCE_SIZE space.
    Strategy:
      1. If anchor: search between y1 and y2 for the row with the steepest
         vertical depth gradient (surface discontinuity).
      2. Fallback: use lower-third median row.
    """
    if anchor:
        search_top = anchor["y1"]
        search_bot = anchor["y2"] - int(anchor["box_h"] * 0.08)
        search_top = max(0, search_top)
        search_bot = min(INFERENCE_SIZE - 1, search_bot)

        if search_bot > search_top + 4:
            col_slice = depth_map[search_top:search_bot, :]
            grad = np.abs(np.diff(col_slice, axis=0))
            row_grad = grad.mean(axis=1)
            peak_local = int(np.argmax(row_grad))
            waterline_y = search_top + peak_local
        else:
            waterline_y = anchor["y2"] - int(anchor["box_h"] * 0.15)
    else:
        waterline_y = int(INFERENCE_SIZE * 0.70)

    return int(np.clip(waterline_y, 0, INFERENCE_SIZE - 1))


# ===========================================================================
# PUBLIC ENTRY POINT  —  drop-in replacement for old estimate_flood_depth()
# ===========================================================================
def estimate_flood_depth(image_array, context_profile=None):
    """
    Ensemble flood depth estimator.  No baseline required.

    Parameters
    ----------
    image_array : np.ndarray   BGR image from cv2
    context_profile : ignored  (kept for API compatibility)

    Returns
    -------
    dict with keys:
        status, estimated_depth_cm, ensemble_confidence,
        calculation_mode, engine_breakdown
    """
    img_h, img_w, _ = image_array.shape
    resized = cv2.resize(
        image_array,
        (INFERENCE_SIZE, INFERENCE_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    anchor = _get_best_anchor(resized)
    depth_map = _get_depth_map(resized)
    waterline_y = _find_waterline_y(depth_map, anchor, img_h)

    a_depth, a_conf = _engine_a_geometric(anchor, waterline_y, INFERENCE_SIZE)
    b_depth, b_conf = _engine_b_depth_ratio(anchor, depth_map, waterline_y)
    c_depth, c_conf = _engine_c_global_fallback(depth_map, anchor)

    final_depth, final_conf, breakdown = _fuse_ensemble(
        a_depth, a_conf, b_depth, b_conf, c_depth, c_conf
    )

    active = [
        n.replace("engine_", "").upper()
        for n, d in breakdown.items()
        if d["contribution_pct"] > 5
    ]
    anchor_str = (
        f"[Anchor: {anchor['label']}]" if anchor else "[No anchor — fallback only]"
    )
    mode = f"Ensemble ({'+'.join(active)}) {anchor_str}"

    return {
        "status": "success",
        "estimated_depth_cm": final_depth,
        "ensemble_confidence": final_conf,
        "calculation_mode": mode,
        "waterline_y_px": waterline_y,
        "engine_breakdown": breakdown,
    }
