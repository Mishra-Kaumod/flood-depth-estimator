"""Exercise actual UI and both CLI main functions with the same saved frame."""
from io import BytesIO
import json
from pathlib import Path
import csv
import sys
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from scripts import run_v6_shadow_comparison as image_cli
from scripts import run_v6_shadow_video as video_cli
from src.v6_inference import create_v6_pipeline, load_v6_rgb
from src.v6_video_input import VideoFrame
from web_app import create_app


@pytest.mark.parametrize("mode", ["fractional", "zero", "missing"])
def test_same_encoded_image_matches_all_active_v6_entrypoints(tmp_path, mode):
    # Video's existing JPEG round trip is part of its input contract.
    frame = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
    image_path = tmp_path / "input.jpg"
    assert cv2.imwrite(str(image_path), frame)
    expected_rgb = load_v6_rgb(image_path)
    calls = []

    class Signals:
        def predict(self, rgb):
            np.testing.assert_array_equal(rgb, expected_rgb)
            calls.append(rgb.copy())
            depth = float(rgb.mean()) / 7 if mode == "fractional" else 0.0 if mode == "zero" else None
            return {"depth_cm": 999, "structured_features": {
                "efficientnet_candidate_depth_cm": depth}}

    class Reader:
        diagnostics, backend, fps = [], "opencv", 25.0
        def iter_frames(self, path):
            yield VideoFrame(0, 0.0, frame, "opencv")

    with patch("src.segformer_yolo_depthv2_pipeline.SegformerYoloDepthV2Pipeline", side_effect=Signals):
        expected = create_v6_pipeline().predict(expected_rgb)
        response = create_app().test_client().post("/predict", data={
            "image": (BytesIO(image_path.read_bytes()), "input.jpg")})
        assert response.status_code == 200
        assert response.get_json()["final_shadow_depth_cm"] == expected.final_shadow_depth_cm

        output = tmp_path / "comparison.json"
        with patch.object(sys, "argv", ["comparison", "--image", str(image_path),
                "--expected-sha256", image_cli.digest(image_path), "--output", str(output)]):
            image_cli.main()
        comparison = json.loads(output.read_text())
        assert comparison["comparison"] == expected.comparison().as_dict()

        video_dir = tmp_path / "video"
        with patch.object(video_cli, "V6VideoInput", Reader), patch.object(sys, "argv", [
                "video", "--video", "synthetic.mp4", "--output-dir", str(video_dir)]):
            video_cli.main()
        with (video_dir / "predictions.csv").open(newline="") as handle:
            row = next(csv.DictReader(handle))
        value = float(row["final_shadow_depth_cm"]) if row["final_shadow_depth_cm"] else None
        assert value == expected.final_shadow_depth_cm
        assert len(calls) == 4
        assert Path(row["saved_frame_path"]).read_bytes() == image_path.read_bytes()


def test_shared_rgb_loader_accepts_grayscale_and_rgba(tmp_path):
    from PIL import Image
    for mode in ("L", "RGBA"):
        path = tmp_path / (mode + ".png")
        Image.new(mode, (3, 2)).save(path)
        from_path = load_v6_rgb(path)
        from_bytes = load_v6_rgb(path.read_bytes())
        assert from_path.shape == (2, 3, 3)
        assert from_path.dtype == np.uint8
        np.testing.assert_array_equal(from_path, from_bytes)
