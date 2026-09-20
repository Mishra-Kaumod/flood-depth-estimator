"""Pure decision policy for the four-class road-scene guard."""

from __future__ import annotations

from typing import Any


def evaluate_scene_guard(
    probabilities: dict[str, float],
    depth_cm: float,
    features: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any] | None:
    if not probabilities or not bool(config.get("scene_guard_enabled", True)):
        return None

    ranked = sorted(probabilities.items(), key=lambda item: float(item[1]), reverse=True)
    scene_name, scene_probability = ranked[0]
    runner_up_probability = float(ranked[1][1]) if len(ranked) > 1 else 0.0
    scene_probability = float(scene_probability)
    scene_margin = scene_probability - runner_up_probability
    scene_confident = (
        scene_probability >= float(config.get("scene_guard_confidence_threshold", 0.80))
        and scene_margin >= float(config.get("scene_guard_margin_threshold", 0.15))
    )
    coverage_pct = float(features.get("water_coverage_pct", 0.0))
    near_coverage_pct = float(features.get("near_water_coverage_pct", 0.0))
    reference_submersion = float(features.get("max_reference_submersion", 0.0))
    strong_flood_evidence = (
        bool(features.get("immediate_risk", False))
        or coverage_pct >= float(config.get("scene_guard_strong_water_coverage_pct", 20.0))
        or near_coverage_pct >= float(config.get("scene_guard_strong_near_water_coverage_pct", 12.0))
        or reference_submersion >= float(config.get("scene_guard_strong_submersion", 0.35))
    )
    low_risk_visuals = (
        not bool(features.get("immediate_risk", False))
        and not bool(features.get("muddy_water_fallback_applied", False))
        and reference_submersion < float(config.get("scene_guard_max_reference_submersion", 0.20))
        and coverage_pct <= float(config.get("scene_guard_max_water_coverage_pct", 12.0))
        and near_coverage_pct <= float(config.get("scene_guard_max_near_water_coverage_pct", 8.0))
    )
    result = {
        "scene": scene_name,
        "probability": round(scene_probability, 6),
        "runner_up_probability": round(runner_up_probability, 6),
        "margin": round(scene_margin, 6),
        "confident": scene_confident,
        "low_risk_visuals": low_risk_visuals,
        "strong_flood_evidence": strong_flood_evidence,
        "depth_override_cm": None,
        "review_required": False,
        "review_reason": "",
    }
    if not scene_confident:
        result["status"] = "review_required"
        result["review_required"] = True
        result["review_reason"] = (
            f"Road-scene classifier is uncertain: {scene_name} probability={scene_probability:.3f}, "
            f"margin={scene_margin:.3f}; depth output was preserved for review."
        )
    elif scene_name in {"dry_road", "wet_road"} and (strong_flood_evidence or not low_risk_visuals):
        result["status"] = "review_required"
        result["review_required"] = True
        result["review_reason"] = (
            f"Road-scene classifier predicted {scene_name}, but visual/depth signals indicate possible flood evidence; "
            "depth output was preserved for review."
        )
    elif scene_name in {"dry_road", "wet_road"}:
        result["status"] = "no_flood"
        result["depth_override_cm"] = 0.0
    elif scene_name in {"shallow_flood", "meaningful_flood"}:
        result["status"] = "flood_pass_through"
    else:
        result["status"] = "review_required"
        result["review_required"] = True
        result["review_reason"] = f"Unknown scene class {scene_name!r}; depth output was preserved for review."
    return result