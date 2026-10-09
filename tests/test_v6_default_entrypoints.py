"""Default CLI/event/API/worker use the same V6 numerical owner and RGB input."""
from io import BytesIO
import base64
import json
from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest

import main as cli
from src import pipeline as events
from src.event_contract import FloodEvent
from src.v6_inference import create_v6_pipeline, load_v6_rgb, v6_depth_payload
from src.worker import process_camera_event
from web_app import create_app


@pytest.fixture(autouse=True)
def isolate_image_reports(tmp_path, monkeypatch):
    monkeypatch.setattr("src.v6_image_reporting.IMAGE_REPORT_DIRECTORY", tmp_path / "reports")


@pytest.mark.parametrize("depth", [12.3456789, 0.0, None, float("nan"), float("inf")])
def test_default_image_api_worker_and_ui_parity(tmp_path, capsys, depth):
    path = tmp_path / "image.png"
    Image.fromarray(np.arange(600, dtype=np.uint8).reshape(10, 20, 3)).save(path)
    rgb = load_v6_rgb(path)
    calls = []

    class PrimarySignals:
        def predict(self, value):
            np.testing.assert_array_equal(value, rgb)
            calls.append(value)
            return {"structured_features": {"efficientnet_candidate_depth_cm": depth}}

    with patch("src.efficientnet_depth_signal.EfficientNetDepthSignal", side_effect=PrimarySignals), patch.object(events, "_PROCESSOR", None), patch("src.api_service.FloodRepository"):
        expected = v6_depth_payload(create_v6_pipeline().predict(rgb))
        with patch("sys.argv", ["main.py", "image", str(path), "--storage", "local"]):
            cli.main()
        output = capsys.readouterr().out
        assert output.startswith("Flood Depth Estimator")
        expected_depth = expected["final_shadow_depth_cm"]
        rendered = f"{expected_depth:.2f} cm" if expected_depth is not None else "unavailable"
        assert f"Estimated depth: {rendered}" in output
        assert "Depth source: EfficientNet" in output
        assert "primary_depth_cm" not in output
        assert "Final depth:" not in output
        client = create_app().test_client()
        ui = client.post("/predict", data={"image": (BytesIO(path.read_bytes()), "image.png")})
        assert ui.status_code == 200
        assert ui.get_json() == expected
        payload = {"image_b64": base64.b64encode(path.read_bytes()).decode(), "latitude": 12, "longitude": 77}
        api = client.post("/api/v1/estimate", json=payload)
        assert api.status_code == 200
        worker = process_camera_event(payload)
        direct_event = events.execute_event(FloodEvent(**payload, camera_id="test"))
        for response in [api.get_json()["result"], worker["result"], direct_event.to_api_response()]:
            assert {key: response[key] for key in expected} == expected
            assert "estimated_depth_meters" not in response
        assert len(calls) == 6


def test_invalid_cli_image_never_constructs_model(tmp_path):
    path = tmp_path / "bad.png"
    path.write_bytes(b"invalid image")
    with patch.object(cli, "create_v6_pipeline") as factory, pytest.raises(Exception, match="identify image"):
        cli.process_image_cli(str(path), "local", "test", 0, 0, None, None)
    factory.assert_not_called()


def test_model_failure_is_explicit_without_fallback(tmp_path):
    path = tmp_path / "input.png"
    Image.new("RGB", (20, 20)).save(path)
    with patch("src.efficientnet_depth_signal.EfficientNetDepthSignal", side_effect=RuntimeError("checkpoint load failed")):
        with pytest.raises(RuntimeError, match="checkpoint load failed"):
            cli.process_image_cli(str(path), "local", "test", 0, 0, None, None)
        response = create_app().test_client().post("/predict", data={"image": (BytesIO(path.read_bytes()), "input.png")})
        assert response.status_code == 500
        assert "checkpoint load failed" in response.get_json()["error"]


def test_primary_factory_does_not_construct_legacy_pipeline():
    with patch("src.efficientnet_depth_signal.EfficientNetDepthSignal") as primary, patch("src.segformer_yolo_depthv2_pipeline.SegformerYoloDepthV2Pipeline", side_effect=AssertionError("legacy constructor called")):
        result = create_v6_pipeline()
        primary.assert_called_once_with()
        assert result._signal_source is primary.return_value


def test_main_video_reuses_saved_frame_flow(tmp_path):
    from src.v6_video_input import VideoFrame
    from scripts.run_v6_shadow_video import process_saved_frames
    frame = np.arange(180, dtype=np.uint8).reshape(6, 10, 3)
    class Reader:
        diagnostics, backend, fps = [], "opencv", 25.0
        def iter_frames(self, _):
            yield VideoFrame(0, 0.0, frame, "opencv")
    class PrimarySignals:
        def predict(self, rgb):
            return {"structured_features": {"efficientnet_candidate_depth_cm": float(rgb.mean())}}
    with patch("src.efficientnet_depth_signal.EfficientNetDepthSignal", side_effect=PrimarySignals), patch("src.v6_video_input.V6VideoInput", Reader):
        csv_path = tmp_path / "out.csv"
        cli.process_video_cli("video.mp4", str(csv_path), 1, "local", "test", 0, 0, None, None)
        import csv
        row = next(csv.DictReader(csv_path.open()))
        saved_rgb = load_v6_rgb(row["saved_frame_path"])
        independent = create_v6_pipeline().predict(saved_rgb)
        assert float(row["primary_depth_cm"]) == independent.primary_depth_cm
        assert float(row["final_shadow_depth_cm"]) == independent.final_shadow_depth_cm


def test_disabled_or_missing_checkpoint_is_explicit():
    from src.efficientnet_depth_signal import EfficientNetDepthSignal
    with patch.object(EfficientNetDepthSignal, "_load_efficientnet_signal_if_available"), pytest.raises(RuntimeError, match="checkpoint could not be loaded"):
        EfficientNetDepthSignal()


def test_invalid_camera_api_image_is_explicit_client_error():
    with patch("src.api_service.FloodRepository"), patch.object(events, "_PROCESSOR", None), patch("src.efficientnet_depth_signal.EfficientNetDepthSignal"):
        response = create_app().test_client().post("/api/v1/estimate", json={
            "image_b64": base64.b64encode(b"not an image" * 20).decode(),
            "latitude": 0, "longitude": 0})
    assert response.status_code == 400
    assert "identify image" in response.get_json()["error"]


def test_s3_image_uses_identical_shared_rgb(tmp_path):
    path = tmp_path / "image.png"
    Image.fromarray(np.arange(600, dtype=np.uint8).reshape(10, 20, 3)).save(path)
    expected = load_v6_rgb(path)
    class PrimarySignals:
        def predict(self, rgb):
            np.testing.assert_array_equal(rgb, expected)
            return {"structured_features": {"efficientnet_candidate_depth_cm": 9.25}}
    with patch("src.efficientnet_depth_signal.EfficientNetDepthSignal", side_effect=PrimarySignals), patch.object(cli, "get_s3_handler"), patch.object(cli, "read_s3_bytes", return_value=path.read_bytes()):
        local = cli.process_image_cli(str(path), "local", "test", 0, 0, None, None)
        aws = cli.process_image_cli("images/input.png", "aws", "test", 0, 0, None, "test-bucket")
    assert local == aws
