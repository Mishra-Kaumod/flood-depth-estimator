"""The shared V6 gate skips only corroborated no-water, never zero-fills."""
from unittest.mock import Mock
from unittest.mock import patch
import pytest

import numpy as np

from src.v6_shadow_contract import EvidenceBundle
from src.v6_shadow_pipeline import V6ShadowPipeline
from src.v6_inference import v6_depth_payload
from src.v6_application_review import review_v6_result
from src.v6_image_reporting import build_image_report
from src.llm_judge import LLMJudge, GeminiReviewError


class Primary:
    def __init__(self):
        self.calls = 0

    def predict(self, image):
        self.calls += 1
        return {"structured_features": {"efficientnet_candidate_depth_cm": 42}}


def run(features, status, *, no_water_zero=True, no_reference_na=False):
    primary = Primary()
    collector = Mock()
    collector.config = {"inference": {"no_water_guard": {
        "no_water_threshold": .99, "max_water_coverage_pct": 5,
        "max_near_water_coverage_pct": 5, "wet_road_guard_enabled": True,
        "wet_road_guard_threshold": .995, "wet_road_guard_max_water_coverage_pct": 12,
        "wet_road_guard_max_near_water_coverage_pct": 8},
        "v6_eligibility": {"no_water_zero_enabled": no_water_zero,
                           "no_reference_na_enabled": no_reference_na}}}
    collector.collect.return_value = EvidenceBundle(features, status)
    result = V6ShadowPipeline(primary, collector).predict(np.zeros((8, 8, 3), dtype=np.uint8))
    return result, primary.calls


def evidence(**changes):
    return {**{"water_coverage_pct": 2., "near_water_coverage_pct": 2.,
            "reference_count": 0, "reference_object_diagnostics": [],
            "no_water_probability": .999}, **changes}


def statuses(yolo="available", no_water="available"):
    return {"water": {"status": "available"}, "yolo": {"status": yolo},
            "no_water": {"status": no_water}, "wet_road_no_water": {"status": "disabled"}}


def test_corroborated_no_water_returns_application_zero_without_primary_or_gemini():
    result, calls = run(evidence(), statuses())
    assert calls == 0 and result.water_present is False
    assert result.depth_inference_skipped and result.skip_reason == "no_flood_water_detected"
    payload = v6_depth_payload(result)
    assert payload["primary_depth_cm"] is None and payload["final_v6_depth_cm"] is None
    output = review_v6_result(result, b"image", "a.jpg", config={"inference": {"llm_judge": {"enabled": True}}})
    assert output["application_final_depth_cm"] == 0.0
    assert output["decision_source"] == "no_water_gate"
    assert output["gemini_review"]["status"] == "skipped_eligibility"
    report = build_image_report("dry.jpg", output, result, None)
    assert report["status"] == "no_flood_detected"
    assert report["application_final_depth_cm"] == 0.0
    assert report["final_shadow_depth_cm"] is None
    assert report["error_reason"] == ""


def test_zero_references_alone_does_not_skip():
    result, calls = run(evidence(no_water_probability=.4), statuses())
    assert calls == 1 and result.final_v6_depth_cm == 42
    assert result.reference_eligibility == "none_found"


def test_zero_references_opt_in_returns_na_and_preserves_missing_depth():
    result, calls = run(evidence(no_water_probability=.4), statuses(), no_reference_na=True)
    assert calls == 0 and result.depth_inference_skipped
    assert result.skip_reason == "no_valid_reference_object_detected"
    output = review_v6_result(result, b"image", "a.jpg", config={"inference": {"llm_judge": {"enabled": True}}})
    assert output["primary_depth_cm"] is None
    assert output["final_shadow_depth_cm"] is None
    assert output["application_final_depth_cm"] is None
    assert output["decision_source"] == "no_reference_gate"
    assert output["gemini_review"]["status"] == "skipped_eligibility"
    report = build_image_report("no_ref.jpg", output, result, None)
    assert report["status"] == "skipped_no_reference"
    assert report["application_final_depth_cm"] is None


def test_confirmed_no_water_precedes_zero_reference_na():
    result, calls = run(evidence(), statuses(), no_water_zero=True, no_reference_na=True)
    assert calls == 0 and result.skip_reason == "no_flood_water_detected"
    output = review_v6_result(result, b"image", "dry.jpg", config={"inference": {"llm_judge": {"enabled": False}}})
    assert output["application_final_depth_cm"] == 0.0


def test_controls_are_independent_and_both_disabled_preserve_primary():
    result, calls = run(evidence(), statuses(), no_water_zero=False, no_reference_na=False)
    assert calls == 1 and result.primary_depth_cm == result.final_v6_depth_cm == 42
    assert result.water_gate == "confirmed_no_water_control_disabled"
    no_ref, calls = run(evidence(), statuses(), no_water_zero=False, no_reference_na=True)
    assert calls == 0 and no_ref.skip_reason == "no_valid_reference_object_detected"


def test_yolo_failure_does_not_trigger_na():
    result, calls = run(evidence(no_water_probability=.4), statuses(yolo="unavailable"), no_reference_na=True)
    assert calls == 1 and not result.depth_inference_skipped


class StagedCollector:
    """Spy on the same water -> YOLO -> remaining collector contract as runtime."""
    def __init__(self, *, no_water_probability=.4, reference_count=2, water_failure=False,
                 yolo_failure=False, no_water_zero=True, no_reference_na=True):
        self.calls = []
        self.no_water_probability = no_water_probability
        self.reference_count = reference_count
        self.water_failure = water_failure
        self.yolo_failure = yolo_failure
        self.config = {"inference": {"no_water_guard": {
            "no_water_threshold": .99, "max_water_coverage_pct": 5,
            "max_near_water_coverage_pct": 5, "wet_road_guard_enabled": False},
            "v6_eligibility": {"no_water_zero_enabled": no_water_zero,
                               "no_reference_na_enabled": no_reference_na}}}

    def collect_water(self, image):
        self.calls.append("water")
        if self.water_failure:
            raise RuntimeError("water detector failed")
        return EvidenceBundle({"water_coverage_pct": 2., "near_water_coverage_pct": 2.,
                               "no_water_probability": self.no_water_probability},
                              {"water": {"status": "available"}, "no_water": {"status": "available"}})

    def collect_references(self, image, previous):
        self.calls.append("yolo")
        if self.yolo_failure:
            raise RuntimeError("YOLO failed")
        return EvidenceBundle({**dict(previous.features), "reference_count": self.reference_count,
                               "reference_object_diagnostics": []},
                              {**dict(previous.status), "yolo": {"status": "available"}})

    def collect_remaining(self, image, previous):
        self.calls.append("remaining")
        return previous


@pytest.mark.parametrize("reference_count", [0, 5])
def test_no_water_early_exit_never_calls_yolo_or_depth(reference_count):
    collector = StagedCollector(no_water_probability=.999, reference_count=reference_count)
    primary = Primary()
    with patch("src.v6_shadow_pipeline.controlled_correction") as correction:
        result = V6ShadowPipeline(primary, collector).predict(np.zeros((8, 8, 3), dtype=np.uint8))
        correction.assert_not_called()
    assert collector.calls == ["water"] and primary.calls == 0
    assert result.skip_reason == "no_flood_water_detected" and result.correction_trace == ()
    with patch("src.v6_application_review.LLMJudge") as judge:
        reviewed = review_v6_result(result, b"image", "dry.jpg",
                                    config={"inference": {"llm_judge": {"enabled": True}}})
        judge.assert_not_called()
    assert reviewed["application_final_depth_cm"] == 0.0


def test_zero_references_stop_before_other_depth_models():
    collector = StagedCollector(reference_count=0)
    primary = Primary()
    with patch("src.v6_shadow_pipeline.controlled_correction") as correction:
        result = V6ShadowPipeline(primary, collector).predict(np.zeros((8, 8, 3), dtype=np.uint8))
        correction.assert_not_called()
    assert collector.calls == ["water", "yolo"] and primary.calls == 0
    assert result.skip_reason == "no_valid_reference_object_detected" and result.correction_trace == ()
    with patch("src.v6_application_review.LLMJudge") as judge:
        reviewed = review_v6_result(result, b"image", "no_ref.jpg",
                                    config={"inference": {"llm_judge": {"enabled": True}}})
        judge.assert_not_called()
    assert reviewed["application_final_depth_cm"] is None


@pytest.mark.parametrize("changes", [{"water_failure": True}, {"yolo_failure": True},
                                     {"no_water_probability": None}])
def test_failed_or_uncertain_early_stage_continues(changes):
    collector = StagedCollector(**changes)
    primary = Primary()
    result = V6ShadowPipeline(primary, collector).predict(np.zeros((8, 8, 3), dtype=np.uint8))
    assert collector.calls == ["water", "yolo", "remaining"] and primary.calls == 1
    assert result.final_v6_depth_cm == 42


def test_independent_disabled_controls_still_run_normal_flow():
    collector = StagedCollector(no_water_probability=.999, reference_count=0,
                                no_water_zero=False, no_reference_na=False)
    primary = Primary()
    result = V6ShadowPipeline(primary, collector).predict(np.zeros((8, 8, 3), dtype=np.uint8))
    assert collector.calls == ["water", "yolo", "remaining"] and primary.calls == 1
    assert result.final_v6_depth_cm == 42


def test_yolo_failure_is_not_zero_objects():
    result, calls = run(evidence(no_water_probability=.4), statuses(yolo="unavailable"))
    assert calls == 1 and result.water_present is True
    assert result.reference_eligibility == "unavailable"


def test_missing_classifier_and_flood_evidence_abstain_or_continue():
    uncertain, calls = run(evidence(no_water_probability=None), statuses(no_water="unavailable"))
    assert calls == 1 and uncertain.water_present is None
    flooded, calls = run(evidence(water_coverage_pct=35., near_water_coverage_pct=30.), statuses())
    assert calls == 1 and flooded.water_present is True and flooded.final_v6_depth_cm == 42


def test_conflicting_water_classifiers_do_not_force_zero():
    status = statuses()
    status["wet_road_no_water"] = {"status": "available"}
    result, calls = run(evidence(wet_road_no_water_probability=0.0), status)
    assert calls == 1 and result.water_present is None
    assert result.water_gate == "uncertain_conflicting_classifiers"


def test_disabled_evidence_preserves_baseline_exactly():
    result, calls = run({}, {"water": {"status": "disabled"}, "yolo": {"status": "disabled"}})
    assert calls == 1 and result.primary_depth_cm == result.final_v6_depth_cm == 42


@pytest.mark.parametrize("code", ("model_endpoint", "request_payload", "quota", "authentication"))
def test_gemini_safe_error_category(code):
    judge = LLMJudge({"enabled": True, "google_api_key": "unit-test-placeholder"})
    with patch.object(judge, "_call_google_api", return_value='{"error":"unavailable","error_code":"'+code+'"}'):
        with pytest.raises(GeminiReviewError, match=code):
            judge.judge_v6({}, b"image", "a.jpg")


def test_gemini_nonfinite_context_is_request_payload_error():
    judge = LLMJudge({"enabled": True, "google_api_key": "unit-test-placeholder"})
    with pytest.raises(GeminiReviewError, match="request_payload"):
        judge.judge_v6({"bad": float("nan")}, b"image", "a.jpg")


