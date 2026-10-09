"""Console presentation tests; V6 results are provided by the shared caller."""
import json
from pathlib import Path

from src.v6_image_reporting import _actionable_warnings, print_image_report, write_image_debug_report


def sample_report(**updates):
    report = {
        "image_filename": "01_test.jpg",
        "application_final_depth_cm": 74.35,
        "final_shadow_depth_cm": 74.35,
        "numerical_owner": "efficientnet_primary_anchor",
        "decision_source": "v6_pipeline",
        "gemini_enabled": False,
        "gemini_status": "disabled",
        "gemini_correction_applied": False,
        "gemini_prediction_correct": None,
        "gemini_reason": None,
        "gemini_error_reason": None,
        "gemini_error_code": None,
        "diagnostic_evidence": "{}",
        "uncertainty_flags": "[]",
        "controlled_correction_trace": "[]",
    }
    report.update(updates)
    return report


def test_default_output_is_concise_and_uses_application_result(tmp_path, capsys):
    report = sample_report()
    print_image_report(report, tmp_path / "result.csv")
    output = capsys.readouterr().out
    assert "Flood Depth Estimator" in output
    assert "Image: 01_test.jpg" in output
    assert "Estimated depth: 74.35 cm" in output
    assert "Depth source: EfficientNet" in output
    assert "Correction: None — no validated alternative estimate" in output
    assert "Report saved:" in output
    assert "primary_depth_cm" not in output
    assert "checkpoint_sha256" not in output
    assert "Warnings:" not in output


def test_corrected_result_is_displayed_as_application_final(tmp_path, capsys):
    report = sample_report(application_final_depth_cm=35.0, decision_source="gemini_review",
                           gemini_enabled=True, gemini_status="reviewed",
                           gemini_correction_applied=True, gemini_reason="Waterline below wheel hub")
    print_image_report(report, tmp_path / "result.csv")
    output = capsys.readouterr().out
    assert "Estimated depth: 35.00 cm" in output
    assert "Depth source: Gemini review" in output
    assert "Correction: Applied — Waterline below wheel hub" in output


def test_debug_json_preserves_result_payload(tmp_path):
    report = sample_report()
    payload = {"primary_depth_cm": 74.35, "final_shadow_depth_cm": 74.35,
               "diagnostic_evidence": {"reference_objects": [{"bbox": [1, 2, 3, 4]}]}}
    path = write_image_debug_report(report, payload, tmp_path / "result.csv")
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["result"] == payload
    assert document["csv_report_path"].endswith("result.csv")


def test_warnings_are_conditional_and_actionable():
    report = sample_report(diagnostic_evidence=json.dumps({"semantic_disagreement": "native_predictions_disagree"}),
                           gemini_enabled=True, gemini_status="failed", gemini_error_code="missing_key")
    assert _actionable_warnings(report) == [
        "Scene classifiers disagree; review recommended.",
        "Gemini review unavailable: API key missing.",
    ]
