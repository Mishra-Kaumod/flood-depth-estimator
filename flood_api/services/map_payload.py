from .location_mapping import pick_bengaluru_point
from .prediction_policy import harmonize_prediction


def depth_to_intensity_scale(depth_cm):
    if depth_cm <= 5:
        return 1
    if depth_cm <= 20:
        return 2
    if depth_cm <= 50:
        return 3
    if depth_cm <= 80:
        return 4
    return 5


def _expected_flood_from_filename(image_name):
    name = (image_name or "").lower()
    positive_markers = ["flood", "inund", "water", "rescue", "storm", "rain", "monsoon"]
    negative_markers = ["dry", "drought", "desert", "no_flood", "noflood"]
    if any(marker in name for marker in negative_markers):
        return False
    if any(marker in name for marker in positive_markers):
        return True
    return None


def _evaluate_verdict(record, depth_cm):
    normalized = harmonize_prediction(
        raw_depth_cm=depth_cm,
        water_pct=record.surface_water_confirmed_pct,
        raw_confidence=(record.system_confidence_score_pct or 0.0) / 100.0,
        num_anchors=record.num_reference_objects,
    )
    predicted_flood = normalized["is_water_confirmed"]
    latest_feedback = record.feedback_entries.order_by("-created_at").first()
    if latest_feedback:
        if latest_feedback.feedback_type in ("accepted", "corrected"):
            return "Correct", "review_feedback"
        if latest_feedback.feedback_type == "rejected":
            return "Incorrect", "review_feedback"

    expected_flood = _expected_flood_from_filename(record.image_name)
    if expected_flood is None:
        return "Needs Review", "unverified"
    return ("Correct", "filename_heuristic") if predicted_flood == expected_flood else ("Incorrect", "filename_heuristic")


def build_dashboard_map_points(records):
    map_points = []
    for record in records:
        camera = record.camera
        camera_id = camera.camera_id if camera else f"unknown_{record.id}"
        fallback_point = pick_bengaluru_point(f"{camera_id}:{record.image_name or record.id}")
        latitude = camera.latitude if camera and camera.latitude is not None else fallback_point["latitude"]
        longitude = camera.longitude if camera and camera.longitude is not None else fallback_point["longitude"]
        location_name = (
            camera.location_name
            if camera and camera.location_name
            else f"{fallback_point['name']} (Randomized)"
        )

        depth_cm = float(record.computed_depth_cm or 0)
        verdict, verdict_source = _evaluate_verdict(record, depth_cm)
        map_points.append({
            "record_id": record.id,
            "camera_id": camera_id,
            "location_name": location_name,
            "latitude": latitude,
            "longitude": longitude,
            "image_name": record.image_name or "unknown_image.jpg",
            "depth_cm": round(depth_cm, 2),
            "intensity_scale": depth_to_intensity_scale(depth_cm),
            "surface_water_confirmed_pct": record.surface_water_confirmed_pct,
            "system_confidence_score_pct": record.system_confidence_score_pct,
            "strategy_applied": record.strategy_applied,
            "safety_risk_assessment": record.safety_risk_assessment,
            "detected_reference_objects": record.detected_reference_objects,
            "is_water_confirmed": record.is_water_confirmed,
            "response_verdict": verdict,
            "verdict_source": verdict_source,
            "timestamp": record.timestamp.isoformat(),
        })
    return map_points
