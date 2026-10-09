"""Mocked application review, immutable V6 ownership and cross-surface parity."""
import base64
import csv
from io import BytesIO
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import pytest

import main as cli
from src import pipeline as events
from src import v6_application_review as application
from src.llm_judge import LLMJudge
from src.v6_inference import load_v6_rgb, v6_depth_payload
from src.v6_shadow_pipeline import V6ShadowPipeline
from src.worker import process_camera_event
from web_app import create_app


def config(enabled=True, corrections=True, video=False):
    return {"inference": {"llm_judge": {"enabled": enabled, "apply_corrections": corrections},
                          "efficientnet_signal": {"max_depth_cm": 180},
                          "video": {"gemini_review_enabled": video}}}


def response(**changes):
    return {"prediction_correct": False, "recommended_depth_cm": 35,
            "visual_depth_estimate_cm": 36, "visual_depth_range_cm": "32–38",
            "visual_confidence": "medium", "review_required": True,
            "reason": "Visible waterline", **changes}


@pytest.fixture
def trained_result():
    class Signals:
        _efficientnet_max_depth_cm = 180
        def predict(self, rgb):
            return {"structured_features": {"efficientnet_candidate_depth_cm": 42}}
    pipeline = V6ShadowPipeline(Signals())
    return pipeline, pipeline.predict(np.zeros((2, 2, 3), dtype=np.uint8))


@pytest.mark.parametrize("enabled,corrections,expected,calls", [(False, True, 42, 0), (True, False, 42, 1), (True, True, 35, 1)])
def test_three_modes_preserve_v6(trained_result, monkeypatch, enabled, corrections, expected, calls):
    pipeline, result = trained_result
    monkeypatch.setenv("GOOGLE_API_KEY", "unit-test-placeholder")
    with patch.object(LLMJudge, "judge_v6", return_value=response()) as judge:
        output = application.review_v6_result(result, b"original", "image.png", pipeline,
                                              config=config(enabled, corrections))
    assert judge.call_count == calls
    assert output["primary_depth_cm"] == output["final_shadow_depth_cm"] == result.primary_depth_cm == result.final_shadow_depth_cm == 42
    assert output["numerical_owner"] == "efficientnet_primary_anchor"
    assert output["application_final_depth_cm"] == expected
    assert output["decision_source"] == ("gemini_review" if enabled and corrections else "v6_pipeline")
    if calls:
        assert judge.call_args.kwargs["image_bytes"] == b"original"
        assert output["gemini_review"]["visual_depth_range_cm"] == "32–38"
    else:
        assert output["application_final_depth_cm"] - result.final_shadow_depth_cm == 0.0


@pytest.mark.parametrize("depth", [None, "bad", "NaN", float("nan"), float("inf"), -float("inf"), -1, 181, True, [], {}])
def test_invalid_depth_falls_back(trained_result, monkeypatch, depth):
    pipeline, result = trained_result
    monkeypatch.setenv("GOOGLE_API_KEY", "unit-test-placeholder")
    with patch.object(LLMJudge, "judge_v6", return_value=response(recommended_depth_cm=depth)):
        output = application.review_v6_result(result, b"image", "a.jpg", pipeline, config=config())
    assert output["application_final_depth_cm"] == 42 and output["decision_source"] == "v6_pipeline"
    assert output["gemini_review"]["status"] == "invalid_depth"
    json.dumps(output, allow_nan=False)


@pytest.mark.parametrize("failure", [TimeoutError("secret-in-error"), OSError("HTTP error secret-in-error"), ValueError("malformed JSON secret-in-error")])
def test_failures_do_not_fail_v6_or_expose_exception(trained_result, monkeypatch, failure):
    pipeline, result = trained_result
    monkeypatch.setenv("GOOGLE_API_KEY", "secret-in-error")
    with patch.object(LLMJudge, "judge_v6", side_effect=failure):
        output = application.review_v6_result(result, b"image", "a.jpg", pipeline, config=config())
    assert output["application_final_depth_cm"] == 42 and output["decision_source"] == "v6_pipeline"
    assert output["gemini_review"]["status"] == "failed"
    assert "secret-in-error" not in json.dumps(output)


def test_missing_key_falls_back_without_request(trained_result):
    pipeline, result = trained_result
    with patch.object(LLMJudge, "judge_v6") as judge:
        output = application.review_v6_result(result, b"image", "a.jpg", pipeline, config=config())
    judge.assert_not_called()
    assert output["application_final_depth_cm"] == 42 and output["gemini_review"]["status"] == "failed"


@pytest.mark.parametrize("wire", ['not JSON', '{"error":{"message":"API failure"}}', '{"candidates":[{"content":{"parts":[{"text":"not JSON"}]}}]}', '{"prediction_correct":"false","recommended_depth_cm":35}'])
def test_strict_client_rejects_api_and_json_failures(wire):
    judge = LLMJudge({"enabled": True, "google_api_key": "unit-test-placeholder"})
    with patch.object(judge, "_call_google_api", return_value=wire), pytest.raises(ValueError):
        judge.judge_v6({}, b"image", "a.jpg")


def test_strict_client_image_context_and_no_zero_fill():
    judge = LLMJudge({"enabled": True, "google_api_key": "unit-test-placeholder"})
    wire = json.dumps({"candidates": [{"content": {"parts": [{"text": json.dumps({"prediction_correct": False, "reason": "uncertain"})}]}}]})
    with patch.object(judge, "_call_google_api", return_value=wire) as call:
        output = judge.judge_v6({"v6_metric_prediction": {"final_shadow_depth_cm": 42}}, b"original-image", "a.png")
    assert "recommended_depth_cm" not in output
    parts = call.call_args.args[0]["contents"][0]["parts"]
    assert base64.b64decode(parts[1]["inline_data"]["data"]) == b"original-image"
    assert parts[1]["inline_data"]["mime_type"] == "image/png"
    assert "v6_metric_prediction" in parts[0]["text"]


@pytest.mark.parametrize("enabled,corrections,final", [(False, True, 42), (True, False, 42), (True, True, 35)])
def test_all_image_application_entrypoints_and_csv(tmp_path, monkeypatch, capsys, enabled, corrections, final):
    image = tmp_path / "image.png"
    Image.fromarray(np.arange(600, dtype=np.uint8).reshape(10, 20, 3)).save(image)
    cfg = config(enabled, corrections)
    monkeypatch.setenv("GOOGLE_API_KEY", "unit-test-placeholder")
    monkeypatch.setattr(application, "load_settings_dict", lambda: cfg)
    monkeypatch.setattr(events, "load_settings_dict", lambda **kwargs: cfg)
    monkeypatch.setattr("src.v6_image_reporting.IMAGE_REPORT_DIRECTORY", tmp_path)
    class Source:
        def predict(self, rgb):
            np.testing.assert_array_equal(rgb, load_v6_rgb(image))
            return {"structured_features": {"efficientnet_candidate_depth_cm": 42}}
    with patch("src.efficientnet_depth_signal.EfficientNetDepthSignal", side_effect=Source), patch.object(events, "_PROCESSOR", None), patch("src.api_service.FloodRepository"), patch.object(LLMJudge, "judge_v6", return_value=response()) as judge:
        main = cli.process_image_cli(str(image), "local", "test", 0, 0, None, None)
        console = capsys.readouterr().out
        client = create_app().test_client()
        ui = client.post("/predict", data={"image": (BytesIO(image.read_bytes()), image.name)}).get_json()
        event = {"image_b64": base64.b64encode(image.read_bytes()).decode(), "latitude": 0, "longitude": 0}
        api_response = client.post("/api/v1/estimate", json=event)
        assert api_response.status_code == 200, api_response.get_json()
        api = api_response.get_json()["result"]
        worker = process_camera_event(event)["result"]
        assert judge.call_count == (4 if enabled else 0)
    for output in (main, ui, api, worker):
        assert output["primary_depth_cm"] == output["final_shadow_depth_cm"] == 42
        assert output["application_final_depth_cm"] == final
        assert output["decision_source"] == ("gemini_review" if enabled and corrections else "v6_pipeline")
        assert output["numerical_owner"] == "efficientnet_primary_anchor"
    row = next(csv.DictReader((tmp_path / "image_v6_prediction.csv").open(encoding="utf-8")))
    assert float(row["application_final_depth_cm"]) == final
    assert float(row["final_shadow_depth_cm"]) == 42
    assert f"Estimated depth: {float(final):.2f} cm" in console
    assert f"Depth source: {'Gemini review' if enabled and corrections else 'EfficientNet'}" in console
    assert "Final depth:" not in console
    assert "42.00 cm" not in console or final == 42
    assert row["gemini_visual_confidence"] == ("medium" if enabled else "")
    assert not any(name in row for name in ("severity", "confidence", "aggregation", "estimated_depth_meters"))


@pytest.mark.parametrize("opt_in", [False, True])
def test_video_opt_in_uses_same_reviewer_and_saved_jpeg(tmp_path, trained_result, monkeypatch, opt_in):
    from scripts.run_v6_shadow_video import process_saved_frames
    from src.v6_video_input import VideoFrame
    pipeline, _ = trained_result
    monkeypatch.setenv("GOOGLE_API_KEY", "unit-test-placeholder")
    monkeypatch.setattr(application, "load_settings_dict", lambda: config(video=opt_in))
    class Reader:
        diagnostics, backend, fps = [], "opencv", 25
        def iter_frames(self, _):
            yield VideoFrame(0, 0, np.zeros((4, 4, 3), dtype=np.uint8), "opencv")
    with patch.object(LLMJudge, "judge_v6", return_value=response()) as judge:
        rows, _ = process_saved_frames(Reader(), pipeline, "video.mp4", tmp_path, 1, 1)
    row = rows[0]
    assert judge.call_count == int(opt_in)
    assert row["final_shadow_depth_cm"] == 42
    assert row["application_final_depth_cm"] == (35 if opt_in else 42)
    if opt_in:
        from pathlib import Path
        assert judge.call_args.kwargs["image_bytes"] == Path(row["saved_frame_path"]).read_bytes()


def test_loaded_bound_and_unavailable_v6(trained_result, monkeypatch):
    pipeline, result = trained_result
    monkeypatch.setenv("GOOGLE_API_KEY", "unit-test-placeholder")
    pipeline._signal_source._efficientnet_max_depth_cm = 100
    with patch.object(LLMJudge, "judge_v6", return_value=response(recommended_depth_cm=120)) as judge:
        output = application.review_v6_result(result, b"image", "a.jpg", pipeline, config=config())
        assert output["decision_source"] == "v6_pipeline"
        missing = SimpleNamespace(primary_depth_cm=None, final_shadow_depth_cm=None, numerical_owner="efficientnet_primary_anchor")
        unavailable = application.review_v6_result(missing, b"image", "a.jpg", config=config())
        assert unavailable["application_final_depth_cm"] is None
        assert judge.call_count == 1


def test_valid_zero_numeric_string_and_secret_redaction(trained_result, monkeypatch):
    pipeline, result = trained_result
    monkeypatch.setenv("GOOGLE_API_KEY", "unit-test-placeholder")
    with patch.object(LLMJudge, "judge_v6", return_value=response(recommended_depth_cm="0", reason="unit-test-placeholder")):
        output = application.review_v6_result(result, b"image", "a.jpg", pipeline, config=config())
    assert output["application_final_depth_cm"] == 0
    assert output["decision_source"] == "gemini_review"
    assert "unit-test-placeholder" not in json.dumps(output)


def test_agreement_cannot_override_v6(trained_result, monkeypatch):
    pipeline, result = trained_result
    monkeypatch.setenv("GOOGLE_API_KEY", "unit-test-placeholder")
    with patch.object(LLMJudge, "judge_v6", return_value=response(prediction_correct=True)):
        output = application.review_v6_result(result, b"image", "a.jpg", pipeline, config=config())
    assert output["application_final_depth_cm"] == 42
    assert not output["gemini_review"]["correction_applied"]


def test_existing_diagnostics_are_context_only(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "unit-test-placeholder")
    class Source:
        def predict(self, _):
            return {"structured_features": {
                "efficientnet_candidate_depth_cm": 42, "water_coverage_pct": 60,
                "near_water_coverage_pct": 50, "mid_water_coverage_pct": 60,
                "far_water_coverage_pct": 70, "reference_depth_cm": 25,
                "dense_depth_relative_p90": 0.7, "input_image_height_px": 100,
                "reference_object_diagnostics": [{"label": "car", "bbox": [0, 0, 20, 80],
                    "detector_confidence": 0.9, "waterline_depth_proxy_cm": 25}]}}
    pipeline = V6ShadowPipeline(Source())
    result = pipeline.predict(np.zeros((100, 100, 3), dtype=np.uint8))
    with patch.object(LLMJudge, "judge_v6", return_value=response()) as judge:
        output = application.review_v6_result(result, b"image", "a.jpg", pipeline, config=config(corrections=False))
    sent = judge.call_args.args[0]
    context = sent["diagnostic_context"]
    assert context["water_coverage_pct"]["value"] == 60
    assert context["dense_relative_p90"]["unit"] == "relative"
    assert "contour_reference_depth_estimate" not in context
    assert context["reference_objects"][0]["object_class"] == "car"
    assert output["application_final_depth_cm"] == output["final_shadow_depth_cm"] == 42
    json.dumps(sent, allow_nan=False)


@pytest.mark.parametrize("enabled", [False, True])
def test_optional_frame_bytes_failure_retains_v6(trained_result, monkeypatch, enabled):
    pipeline, result = trained_result
    monkeypatch.setenv("GOOGLE_API_KEY", "unit-test-placeholder")
    read = Mock(side_effect=OSError("saved bytes unavailable"))
    with patch.object(LLMJudge, "judge_v6") as judge:
        output = application.review_v6_result(result, read, "frame.jpg", pipeline,
                                              video=True, config=config(enabled=enabled, video=True))
    assert read.call_count == int(enabled)
    judge.assert_not_called()
    assert output["application_final_depth_cm"] == output["final_shadow_depth_cm"] == 42
    assert output["gemini_review"]["status"] == ("failed" if enabled else "disabled")
