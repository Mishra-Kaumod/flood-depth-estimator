from src.scene_guard import evaluate_scene_guard


CONFIG = {
    "scene_guard_enabled": True,
    "scene_guard_confidence_threshold": 0.80,
    "scene_guard_margin_threshold": 0.15,
    "scene_guard_max_water_coverage_pct": 12.0,
    "scene_guard_max_near_water_coverage_pct": 8.0,
}


def test_confident_wet_road_forces_zero_in_low_risk_scene():
    result = evaluate_scene_guard({"wet_road": 0.92, "shallow_flood": 0.04, "dry_road": 0.03, "meaningful_flood": 0.01}, 35.0, {}, CONFIG)
    assert result["status"] == "no_flood"
    assert result["depth_override_cm"] == 0.0


def test_uncertain_scene_requires_review_and_preserves_depth():
    result = evaluate_scene_guard({"wet_road": 0.44, "shallow_flood": 0.40, "dry_road": 0.10, "meaningful_flood": 0.06}, 24.0, {}, CONFIG)
    assert result["status"] == "review_required"
    assert result["review_required"] is True
    assert result["depth_override_cm"] is None


def test_wet_road_prediction_with_flood_evidence_requires_review():
    result = evaluate_scene_guard({"wet_road": 0.92, "shallow_flood": 0.03, "dry_road": 0.03, "meaningful_flood": 0.02}, 8.0, {"water_coverage_pct": 25.0}, CONFIG)
    assert result["status"] == "review_required"


def test_confident_shallow_flood_passes_through():
    result = evaluate_scene_guard({"shallow_flood": 0.90, "wet_road": 0.04, "dry_road": 0.03, "meaningful_flood": 0.03}, 8.0, {}, CONFIG)
    assert result["status"] == "flood_pass_through"
    assert result["depth_override_cm"] is None