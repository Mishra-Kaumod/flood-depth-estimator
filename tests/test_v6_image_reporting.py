"""Image-run reports preserve shared V6 values and exclude legacy outputs."""
import csv
import hashlib
from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest

import main as cli
from src import v6_image_reporting as reporting
from src.v6_inference import load_v6_rgb, v6_depth_payload
from src.v6_shadow_pipeline import V6ShadowPipeline
from web_app import create_app


@pytest.mark.parametrize("depth", [12.3456789, 0.0, None, float("nan"), float("inf"), -float("inf")])
def test_cli_csv_console_and_ui_match_shared_v6(tmp_path, monkeypatch, capsys, depth):
    image = tmp_path / "input.png"
    Image.fromarray(np.arange(600, dtype=np.uint8).reshape(10, 20, 3)).save(image)
    checkpoint = tmp_path / "checkpoint.fixture"
    checkpoint.write_bytes(b"metadata-only test fixture")
    monkeypatch.setattr(reporting, "IMAGE_REPORT_DIRECTORY", tmp_path / "reports")
    expected_rgb = load_v6_rgb(image)
    class Source:
        _efficientnet_backend = str(checkpoint)
        def predict(self, rgb):
            np.testing.assert_array_equal(rgb, expected_rgb)
            return {"structured_features": {"efficientnet_candidate_depth_cm": depth}}
    pipeline = V6ShadowPipeline(Source())
    expected = v6_depth_payload(pipeline.predict(expected_rgb))
    with patch.object(cli, "create_v6_pipeline", return_value=pipeline):
        result = cli.process_image_cli(str(image), "local", "test", 0, 0, None, None)
    console = capsys.readouterr().out
    output = tmp_path / "reports/input_v6_prediction.csv"
    with output.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(reporting.CSV_FIELDS)
        rows = list(reader)
    assert len(rows) == 1
    row = rows[0]
    assert {key: result[key] for key in expected} == expected
    debug_path = tmp_path / "reports/input_v6_debug.json"
    debug_document = json.loads(debug_path.read_text(encoding="utf-8"))
    assert debug_document["result"] == result
    assert "Report saved:" in console and "Estimated depth:" in console
    assert "primary_depth_cm" not in console and "checkpoint_sha256" not in console
    assert result["application_final_depth_cm"] == expected["final_shadow_depth_cm"]
    for field in ("primary_depth_cm", "final_shadow_depth_cm"):
        value = expected[field]
        assert row[field] == (str(value) if value is not None else "")
    final = result["application_final_depth_cm"]
    assert f"Estimated depth: {f'{final:.2f} cm' if final is not None else 'unavailable'}" in console
    assert row["image_filename"] == "input.png" and row["image_path"] == str(image)
    assert row["checkpoint_path"] == str(checkpoint)
    assert row["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert row["status"] == ("success" if expected["final_shadow_depth_cm"] is not None else "unavailable")
    assert row["prediction_timestamp"] and row["trace_id"]
    assert row["numerical_owner"] == expected["numerical_owner"]
    assert json.loads(row["uncertainty_flags"]) == list(pipeline.predict(expected_rgb).uncertainty.flags)
    banned = ["severity", "aggregation", "fusion", "guard", "resolver", "estimated_depth_meters", "v5"]
    assert not any(word in field.lower() for word in banned for field in row)
    assert not any(word in console.lower() for word in banned)
    assert "Inference: completed; estimate is not independently accuracy-verified." in console
    assert "confidence" not in row and "severity" not in row
    with patch("web_app.create_v6_pipeline", return_value=pipeline):
        response = create_app().test_client().post("/predict",data={"image":(BytesIO(image.read_bytes()),image.name)})
    assert response.status_code == 200
    assert {key: response.get_json()[key] for key in expected} == expected
    assert response.get_json()["application_final_depth_cm"] == expected["final_shadow_depth_cm"]


def test_report_replaces_same_image_without_duplicate_headers(tmp_path, monkeypatch):
    monkeypatch.setattr(reporting,"IMAGE_REPORT_DIRECTORY",tmp_path)
    payload={"primary_depth_cm":1.2,"final_shadow_depth_cm":1.2,"numerical_owner":"efficientnet_primary_anchor"}
    first=reporting.build_image_report("folder/input.jpg",payload,SimpleNamespace(),SimpleNamespace())
    output=reporting.write_image_report(first)
    second=reporting.build_image_report("folder/input.jpg",payload,SimpleNamespace(),SimpleNamespace())
    assert reporting.write_image_report(second)==output
    with output.open(newline="",encoding="utf-8") as handle:rows=list(csv.DictReader(handle))
    assert len(rows)==1 and rows[0]["trace_id"]==second["trace_id"]!=first["trace_id"]
    assert list(tmp_path.glob("*.tmp"))==[]


def test_checkpoint_hash_failure_does_not_change_prediction(tmp_path):
    payload={"primary_depth_cm":7.68,"final_shadow_depth_cm":7.68,"numerical_owner":"efficientnet_primary_anchor"}
    pipeline=SimpleNamespace(_signal_source=SimpleNamespace(_efficientnet_backend=str(tmp_path/'missing.pth')))
    row=reporting.build_image_report("input.jpg",payload,SimpleNamespace(),pipeline)
    assert row["status"]=="success" and row["final_shadow_depth_cm"]==7.68
    assert row["checkpoint_sha256"]=="" and "Checkpoint hash unavailable" in row["error_reason"]


def test_aws_image_report_is_saved_locally_and_mirrored_without_reprediction(tmp_path, monkeypatch):
    monkeypatch.setattr(reporting,"IMAGE_REPORT_DIRECTORY",tmp_path)
    encoded=BytesIO();Image.new("RGB",(4,4)).save(encoded,format="PNG")
    class Source:
        def predict(self,rgb):return {"structured_features":{"efficientnet_candidate_depth_cm":9.25}}
    pipeline=V6ShadowPipeline(Source())
    with patch.object(cli,"get_s3_handler") as handler,patch.object(cli,"read_s3_bytes",return_value=encoded.getvalue()),patch.object(cli,"create_v6_pipeline",return_value=pipeline):
        result=cli.process_image_cli("images/input.png","aws","test",0,0,None,"bucket")
    saved=tmp_path/'input_v6_prediction.csv'
    handler.return_value.write_csv_to_s3.assert_called_once_with(saved.read_text(encoding="utf-8"),saved.as_posix())
    assert result["final_shadow_depth_cm"]==9.25


def test_report_filename_preserves_normal_letters_and_digits(tmp_path, monkeypatch):
    monkeypatch.setattr(reporting,"IMAGE_REPORT_DIRECTORY",tmp_path)
    row={field:"" for field in reporting.CSV_FIELDS}
    row["image_filename"]="066_20CM_flood_x.jpg"
    assert reporting.write_image_report(row).name=="066_20CM_flood_x_v6_prediction.csv"


def test_debug_mode_prints_details_and_sidecar_preserves_payload(tmp_path, monkeypatch, capsys):
    image = tmp_path / "debug.png"
    Image.new("RGB", (3, 3), color=(10, 20, 30)).save(image)
    monkeypatch.setattr(reporting, "IMAGE_REPORT_DIRECTORY", tmp_path / "reports")
    class Source:
        def predict(self, rgb):
            return {"structured_features": {"efficientnet_candidate_depth_cm": 8.5}}
    pipeline = V6ShadowPipeline(Source())
    with patch.object(cli, "create_v6_pipeline", return_value=pipeline):
        result = cli.process_image_cli(str(image), "local", "test", 0, 0, None, None, debug=True)
    console = capsys.readouterr().out
    assert "Complete application result JSON:" in console
    assert '"final_shadow_depth_cm": 8.5' in console
    saved = tmp_path / "reports/debug_v6_debug.json"
    assert json.loads(saved.read_text(encoding="utf-8"))["result"] == result


def test_actionable_warnings_are_conditional_and_safe():
    report = {
        "diagnostic_evidence": json.dumps({"semantic_disagreement": "native_predictions_disagree"}),
        "final_shadow_depth_cm": 42.0,
        "gemini_enabled": True,
        "gemini_status": "failed",
        "gemini_error_code": "missing_key",
    }
    assert reporting._actionable_warnings(report) == [
        "Scene classifiers disagree; review recommended.",
        "Gemini review unavailable: API key missing.",
    ]
    quiet = {"diagnostic_evidence": "{}", "final_shadow_depth_cm": 42.0,
             "gemini_enabled": False, "gemini_status": "disabled"}
    assert reporting._actionable_warnings(quiet) == []
