"""A V6 metric correction needs approved provenance and stays within both budgets."""
from dataclasses import replace

import numpy as np
import pytest

from src.v6_shadow_contract import MetricCorrectionCandidate, SignalValidationError
from src.v6_shadow_pipeline import V6ShadowPipeline, controlled_correction


class Primary:
    def predict(self, image):
        return {"structured_features": {"efficientnet_candidate_depth_cm": 55.0}}


def fixture_contract(primary=55.0, proposed=52.0, collectors=("water", "yolo")):
    result = V6ShadowPipeline(Primary()).predict(np.zeros((2, 2, 3), dtype=np.uint8))
    candidate = MetricCorrectionCandidate("validated_metric_model", proposed, "freeze-001", "evaluated_road",
                                          ("metric_model", "water", "yolo"), collectors)
    contract = replace(result.contract, metric_candidates=(candidate,),
                       diagnostic_metadata={"collector_status": {name: {"status": "available"} for name in collectors}})
    return contract


def approved(**overrides):
    return {**{"enabled": True, "max_abs_delta_cm": 5.0, "max_relative_delta_fraction": .20,
            "approved_candidates": [{"source": "validated_metric_model", "validation_id": "freeze-001",
                                     "approved_regimes": ["evaluated_road"]}]}, **overrides}


def test_approved_metric_candidate_can_make_one_small_traced_change():
    trace = controlled_correction(55.0, fixture_contract(), approved())
    assert len([item for item in trace if item.accepted]) == 1
    decision = trace[-1]
    assert decision.original_primary_depth_cm == 55
    assert decision.correction_amount_cm == -3
    assert decision.final_v6_depth_cm == 52
    assert decision.correction_source == "validated_metric_model"
    assert decision.evidence_ids


def test_unapproved_and_missing_dependency_abstain():
    contract = fixture_contract()
    assert controlled_correction(55, contract, {**approved(), "approved_candidates": []})[-1].acceptance_or_rejection_reason == "candidate_not_approved_for_regime"
    missing = replace(contract, diagnostic_metadata={"collector_status": {"water": {"status": "available"}, "yolo": {"status": "unavailable"}}})
    assert controlled_correction(55, missing, approved())[-1].acceptance_or_rejection_reason == "required_collector_unavailable"


def test_absolute_and_relative_budgets_reject_without_clamping():
    assert controlled_correction(55, fixture_contract(proposed=49), approved())[-1].acceptance_or_rejection_reason == "proposal_exceeds_correction_budget"
    assert controlled_correction(10, fixture_contract(proposed=7), approved())[-1].acceptance_or_rejection_reason == "proposal_exceeds_correction_budget"
    assert controlled_correction(55, fixture_contract(proposed=52), {**approved(), "max_abs_delta_cm": 6})[-1].acceptance_or_rejection_reason == "invalid_correction_budget"


def test_multiple_metric_candidates_conflict_and_primary_is_unchanged():
    contract = fixture_contract()
    contract = replace(contract, metric_candidates=(contract.metric_candidates[0], replace(contract.metric_candidates[0], proposed_depth_cm=54)))
    decisions = controlled_correction(55, contract, approved())
    assert all(not item.accepted and item.correction_amount_cm == 0 and item.final_v6_depth_cm == 55 for item in decisions)


def test_no_approved_candidate_preserves_active_v6():
    result = V6ShadowPipeline(Primary()).predict(np.zeros((2, 2, 3), dtype=np.uint8))
    assert result.primary_depth_cm == result.final_v6_depth_cm == 55
    assert not result.correction_trace[-1].accepted


@pytest.mark.parametrize("value", [None, "52", float("nan"), float("inf"), -1, True])
def test_invalid_metric_candidate_cannot_enter_contract(value):
    with pytest.raises(SignalValidationError):
        MetricCorrectionCandidate("validated_metric_model", value, "freeze-001", "evaluated_road",
                                  ("metric_model",), ("water",))
