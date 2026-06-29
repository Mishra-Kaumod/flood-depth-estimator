import os


def _to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _to_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def harmonize_prediction(raw_depth_cm, water_pct, raw_confidence, num_anchors):
    """
    Normalize core flood signals so downstream API/UI outputs cannot contradict.
    """
    depth_cm = max(0.0, _to_float(raw_depth_cm, 0.0))
    water_pct = max(0.0, min(100.0, _to_float(water_pct, 0.0)))
    confidence = max(0.0, min(1.0, _to_float(raw_confidence, 0.0)))
    anchors = max(0, _to_int(num_anchors, 0))
    moderate_surface_threshold = _to_float(os.getenv("FLOOD_SURFACE_MODERATE_THRESHOLD", "12"), 12.0)
    moderate_confidence_threshold = _to_float(os.getenv("FLOOD_MODERATE_CONF_THRESHOLD", "0.50"), 0.50)
    strong_surface_threshold = _to_float(os.getenv("FLOOD_SURFACE_STRONG_THRESHOLD", "40"), 40.0)
    anchorless_surface_threshold = _to_float(os.getenv("FLOOD_SURFACE_ANCHORLESS_THRESHOLD", "45"), 45.0)

    depth_signal = depth_cm >= 5.0
    moderate_surface_signal = water_pct >= moderate_surface_threshold and confidence >= moderate_confidence_threshold
    strong_surface_signal = water_pct >= strong_surface_threshold
    is_water_confirmed = bool(depth_signal or moderate_surface_signal or strong_surface_signal)
    # Guard against depth-only hallucinations in barren/no-anchor scenes.
    if depth_signal and anchors == 0 and water_pct < 8.0:
        is_water_confirmed = False
        depth_cm = 0.0
    if not depth_signal and anchors == 0 and water_pct < anchorless_surface_threshold:
        is_water_confirmed = False

    if not is_water_confirmed:
        return {
            "depth_cm": 0.0,
            "is_water_confirmed": False,
            "warning": "No water detected",
            "anchor_count": anchors,
        }

    if depth_cm < 1.0 and strong_surface_signal:
        depth_cm = 5.0

    warning = ""
    if anchors == 0 and depth_cm > 20.0:
        warning = "⚠️ NO REFERENCE OBJECTS DETECTED - Depth unvalidated"
    elif anchors == 1 and depth_cm > 20.0:
        warning = "⚠️ Only 1 reference object - consider multi-image sequence"

    return {
        "depth_cm": round(depth_cm, 1),
        "is_water_confirmed": True,
        "warning": warning,
        "anchor_count": anchors,
    }
