"""Evidence cannot write metrics; one abstaining authority and safe review."""
from dataclasses import FrozenInstanceError
from io import BytesIO
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import pytest

from src.v6_inference import V6EvidenceCollector, v6_depth_payload
from src.v6_shadow_contract import CorrectionTrace, EvidenceBundle, SignalValidationError
from src.v6_shadow_pipeline import V6ShadowPipeline
from src.llm_judge import LLMJudge, GeminiReviewError
from src.v6_application_review import review_v6_result
from scripts.evaluate_v6_future_template import evaluate_rows


class Primary:
    _efficientnet_max_depth_cm = 180
    def predict(self, rgb):
        return {"structured_features": {"efficientnet_candidate_depth_cm": 42}}


def config(**options):
    return {"inference": {"v6_evidence": options, "llm_judge": {"enabled": False}}}


def test_hostile_evidence_cannot_write_metrics_or_semantic_zero():
    collector = Mock()
    collector.collect.return_value = EvidenceBundle({
        "efficientnet_candidate_depth_cm": 0, "final_v6_depth_cm": 0,
        "depth_cm": 0, "road_scene_dry_road_probability": 1,
        "no_water_probability": 1, "water_coverage_pct": 0,
        "dense_depth_relative_p90": .9, "region_depth_cm": 2})
    result = V6ShadowPipeline(Primary(), collector).predict(np.zeros((8, 8, 3), dtype=np.uint8))
    assert result.primary_depth_cm == result.final_v6_depth_cm == result.final_shadow_depth_cm == 42
    assert result.numerical_owner == "efficientnet_primary_anchor"
    assert result.contract.relative_depth["dense_relative_p90"].unit == "relative"
    assert len(result.correction_trace) == 1
    trace = result.correction_trace[0]
    assert trace.proposed_depth_cm == 2 and not trace.accepted and trace.correction_amount_cm == 0
    assert trace.final_v6_depth_cm == trace.original_primary_depth_cm == 42
    assert sum(s.stage == "controlled_correction_policy" for s in result.stages) == 1
    assert [s.stage for s in result.stages[:5]] == ["input", "efficientnet_primary_depth", "segmentation_context", "object_diagnostics", "relative_depth_diagnostics"]
    with pytest.raises(FrozenInstanceError):
        trace.accepted = True


def test_missing_evidence_has_null_signals_and_does_not_fail_primary():
    collector = Mock()
    collector.collect.side_effect = RuntimeError("sensitive exception")
    result = V6ShadowPipeline(Primary(), collector).predict(np.zeros((2, 2, 3)))
    assert result.final_v6_depth_cm == 42
    assert result.contract.semantic_context["water_coverage_pct"].value is None
    assert result.contract.relative_depth["dense_relative_p90"].value is None
    output = review_v6_result(result, b"original", "a.jpg", config=config())
    assert output["diagnostic_evidence"]["collector_status"]["collector"]["reason"] == "RuntimeError"
    assert "sensitive exception" not in json.dumps(output)


def test_classical_water_geometry_is_genuine_and_mask_is_read_only(monkeypatch):
    collector = V6EvidenceCollector(config(water=True, yolo=False, relative_depth=False, semantics=False))
    mask = np.zeros((20, 10), dtype=np.uint8)
    mask[12:] = 255
    detector = Mock()
    detector.detect.return_value = (mask, .0)  # derive coverage from mask, not supplied hint
    collector._models["water"] = detector
    result = collector.collect(np.zeros((20, 10, 3), dtype=np.uint8))
    assert result.features["water_coverage_pct"] == 40
    assert result.features["near_water_coverage_pct"] == 100
    assert result.features["far_water_coverage_pct"] == 0  # measured zero is valid
    assert result.features["water_segmentation_backend"] == "classical_water_region_detector"
    assert not result.water_mask.flags.writeable
    assert all(v is None for k,v in result.features["scene_slices"].items() if k != "label_source")


def test_collector_partial_failure_is_rolled_back_and_other_collectors_run(monkeypatch):
    collector = V6EvidenceCollector(config())
    def failed(rgb, features):
        features["water_coverage_pct"] = 100
        raise OSError("secret")
    monkeypatch.setattr(collector, "_water", failed)
    monkeypatch.setattr(collector, "_objects", lambda rgb, mask, features: features.update(reference_count=0))
    monkeypatch.setattr(collector, "_relative", lambda *args: None)
    result = collector.collect(np.zeros((2, 2, 3), dtype=np.uint8))
    assert "water_coverage_pct" not in result.features
    assert result.status["water"] == {"status":"unavailable", "reason":"OSError"}
    assert result.status["yolo"]["status"] == "available"


def test_yolo_receives_bgr_but_contract_stays_rgb_and_no_cm_proxy():
    collector = V6EvidenceCollector(config())
    rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    rgb[..., 0] = 200
    box = SimpleNamespace(cls=Mock(item=lambda:0), conf=Mock(item=lambda:.8),
                          xyxy=[Mock(tolist=lambda:[1, 1, 9, 9])])
    model = Mock(names={0:"car"})
    model.predict.return_value = [SimpleNamespace(boxes=[box])]
    collector._models["yolo"] = model
    features = {}
    collector._objects(rgb, np.ones((10,10), dtype=bool), features)
    np.testing.assert_array_equal(model.predict.call_args.args[0], rgb[...,::-1])
    assert features["reference_count"] == 1
    assert "waterline_depth_proxy_cm" not in features["reference_object_diagnostics"][0]
    assert features["object_consistency"]["overlap_std"] is None


def test_complete_accepted_trace_schema_but_runtime_never_promotes():
    trace = CorrectionTrace(42,42,40,"validated_fixture",-2,True,"test-only acceptance",("fixture",),40)
    assert trace.final_v6_depth_cm == 40
    with pytest.raises(SignalValidationError):
        CorrectionTrace(42,42,40,"fixture",-2,True,"missing evidence",(),40)
    with pytest.raises(SignalValidationError):
        CorrectionTrace(42,42,40,"fixture",-1,True,"bad arithmetic",("fixture",),40)
    with pytest.raises(SignalValidationError):
        CorrectionTrace(42,42,40,"fixture",-2,False,"rejected",("fixture",),40)
    result = V6ShadowPipeline(Primary()).predict(None)
    assert all(not t.accepted for t in result.correction_trace)


@pytest.mark.parametrize("response,code", [
    ({"error":{"code":401}},"authentication"), ({"error":{"code":429}},"quota"),
    ({"error":{"code":500}},"http_transport"),
    ({"promptFeedback":{"blockReason":"SAFETY"}},"blocked_response"),
    ({"candidates":[{"finishReason":"SAFETY"}]},"blocked_response"),
    ({"candidates":[{"content":{"parts":[{"text":"not JSON"}]}}]},"malformed_json"),
    ({"prediction_correct":"false"},"schema_failure")])
def test_safe_gemini_error_categories(response, code, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY","never-print-this")
    judge = LLMJudge({"enabled":True})
    with patch.object(judge,"_call_google_api",return_value=json.dumps(response)):
        with pytest.raises(GeminiReviewError) as exc:
            judge.judge_v6({},b"image","a.jpg")
    assert exc.value.code == code and "never-print-this" not in str(exc.value)


@pytest.mark.parametrize("failure,code", [(GeminiReviewError("quota"),"quota"),
                                         (TimeoutError("secret"),"timeout"),
                                         (OSError("secret"),"http_transport")])
def test_gemini_error_preserves_primary_final_and_trace(failure,code,monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY","secret")
    result = V6ShadowPipeline(Primary()).predict(None)
    cfg={"inference":{"llm_judge":{"enabled":True,"apply_corrections":True}}}
    with patch.object(LLMJudge,"judge_v6",side_effect=failure):
        output=review_v6_result(result,b"original","a.jpg",config=cfg)
    assert output["gemini_review"]["error_code"] == code
    assert output["primary_depth_cm"] == output["final_v6_depth_cm"] == output["application_final_depth_cm"] == 42
    assert not result.correction_trace[0].accepted


def test_paired_evaluation_counts_improvements_worsening_and_catastrophes():
    rows=[{"actual_depth_cm":35,"primary_depth_cm":42,"final_v6_depth_cm":42,"application_final_depth_cm":0},
          {"actual_depth_cm":35,"primary_depth_cm":0,"final_v6_depth_cm":0,"application_final_depth_cm":35}]
    for row in rows:
        row.update(gemini_enabled=True,gemini_status="reviewed",gemini_correction_applied=True,slice_occlusion=True)
    result=evaluate_rows(rows)
    assert result["paired_rows"]==2
    assert result["gemini"]["improved"]==result["gemini"]["worsened"]==1
    assert result["gemini"]["catastrophes_introduced"]==result["gemini"]["catastrophes_removed"]==1
    assert result["gemini"]["average_improvement_cm"]==35
    assert result["gemini"]["average_worsening_cm"]==28
    assert result["systems"]["primary_depth_cm"]["signed_bias_cm"]==-14
    assert result["slices"]["occlusion"]["primary_depth_cm"]["count"]==2


def test_nonfinite_evaluation_rows_are_unavailable_not_zero():
    result=evaluate_rows([{"actual_depth_cm":10,"primary_depth_cm":"NaN","final_v6_depth_cm":None,"application_final_depth_cm":"inf"}])
    assert result["paired_rows"]==0
    assert all(v is None for v in result["systems"].values())


@pytest.mark.parametrize("split", ["TEST", "INTERNAL_TEST", "EXTERNAL_CHALLENGE", "CONSUMED", "challenge"])
def test_evaluation_cli_rejects_consumed_or_unfrozen_sets(split, tmp_path, monkeypatch):
    from scripts import evaluate_v6_future_template as evaluator
    path=tmp_path/"predictions.csv"
    path.write_text("split,image_id,actual_depth_cm\n"+split+",fixture,10\n",encoding="utf-8")
    monkeypatch.setattr("sys.argv",["evaluate", "--predictions",str(path),"--output",str(tmp_path/"out.json")])
    with pytest.raises(RuntimeError):
        evaluator.main()
    assert not (tmp_path/"out.json").exists()


def test_failed_optional_model_load_is_cached():
    collector=V6EvidenceCollector(config())
    factory=Mock(side_effect=FileNotFoundError("not installed"))
    for _ in range(2):
        with pytest.raises((FileNotFoundError,RuntimeError)):
            collector._model("missing_backend",factory)
    assert factory.call_count==1


def test_mutating_collector_receives_copy_of_original_rgb():
    rgb=np.full((2,2,3),123,dtype=np.uint8)
    def collect(copy):
        copy[:]=0
        return EvidenceBundle()
    result=V6ShadowPipeline(Primary(),SimpleNamespace(collect=collect)).predict(rgb)
    assert np.all(rgb==123)
    assert result.primary_depth_cm==result.final_v6_depth_cm==42
