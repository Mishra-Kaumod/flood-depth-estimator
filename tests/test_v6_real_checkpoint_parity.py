"""Opt-in approved-input real-checkpoint smoke, with no accuracy evaluation."""
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from scripts.run_v6_shadow_video import process_saved_frames
from src.v6_inference import create_v6_pipeline, load_v6_rgb, v6_depth_payload
from src.v6_video_input import V6VideoInput
from web_app import create_app


@pytest.mark.skipif(os.getenv("FLOOD_RUN_REAL_V6_PARITY") != "1", reason="explicit real-checkpoint smoke only")
def test_approved_image_and_saved_video_frame_real_parity():
    previous = json.loads(Path("reports/v6_entrypoint_parity_smoke/parity_results.json").read_text())
    image = Path(previous["image"])
    checkpoint = Path(previous["checkpoint"])
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    assert digest(image) == previous["image_sha256"]
    assert digest(checkpoint) == previous["checkpoint_sha256"]
    output = Path("reports/v6_default_entrypoint_parity_smoke")
    output.mkdir(parents=True, exist_ok=True)
    pipeline = create_v6_pipeline()
    source = pipeline._signal_source
    assert Path(source._efficientnet_backend).resolve() == checkpoint.resolve()
    assert source._efficientnet_model is not None
    assert not source._efficientnet_model.training
    captures = []
    original = pipeline.predict

    def capture(rgb):
        result = original(rgb)
        captures.append({**v6_depth_payload(result), "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest()})
        return result

    pipeline.predict = capture
    rgb = load_v6_rgb(image)
    direct = v6_depth_payload(pipeline.predict(rgb))
    direct_capture = captures[-1]

    def main_cli(path):
        proc = subprocess.run([sys.executable, "main.py", "image", str(path), "--storage", "local"], capture_output=True, text=True, encoding="utf-8", check=True)
        assert "Flood Depth Estimator – V6" in proc.stdout
        return json.loads(proc.stdout.splitlines()[-1])

    main_result = main_cli(image)
    diagnostic_path = output / "image_cli.json"
    subprocess.run([sys.executable, "-m", "scripts.run_v6_shadow_comparison", "--image", str(image), "--expected-sha256", digest(image), "--output", str(diagnostic_path)], check=True)
    diagnostic = json.loads(diagnostic_path.read_text())["comparison"]
    diagnostic_result = {"primary_depth_cm": diagnostic["v6_primary_depth_cm"], "final_shadow_depth_cm": diagnostic["v6_final_shadow_depth_cm"]}
    with patch("web_app.create_v6_pipeline", return_value=pipeline):
        response = create_app().test_client().post("/predict", data={"image": (BytesIO(image.read_bytes()), image.name)})
    assert response.status_code == 200
    ui_result = response.get_json()
    ui_capture = captures[-1]
    assert ui_capture["rgb_sha256"] == direct_capture["rgb_sha256"] == previous["ui"]["rgb_sha256"]
    for result in [main_result, diagnostic_result, ui_result]:
        for field in ["primary_depth_cm", "final_shadow_depth_cm"]:
            assert result[field] == direct[field] == previous["image_cli"][field]
    rows, summary = process_saved_frames(V6VideoInput(), pipeline, previous["video_summary"]["video_path"], output / "video", 1, 1)
    row = next(row for row in rows if row["status"] == "processed")
    video_capture = captures[-1]
    saved = Path(row["saved_frame_path"])
    independent = main_cli(saved)
    saved_direct = v6_depth_payload(pipeline.predict(load_v6_rgb(saved)))
    assert captures[-1]["rgb_sha256"] == video_capture["rgb_sha256"] == previous["video_frame"]["rgb_sha256"]
    assert digest(saved) == previous["saved_frame_sha256"]
    for field in ["primary_depth_cm", "final_shadow_depth_cm"]:
        assert row[field] == independent[field] == saved_direct[field] == previous["video_frame"][field]
    evidence = {
        "image": str(image), "image_sha256": digest(image),
        "checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint),
        "checkpoint_loaded_eval_mode": True,
        "main_image_cli": main_result, "shared_direct": direct_capture,
        "diagnostic_image_cli": diagnostic_result, "ui": ui_capture,
        "video_frame": {**row, "rgb_sha256": video_capture["rgb_sha256"]},
        "independent_saved_frame_main_cli": independent,
        "video_summary": summary,
        "differences_cm": {
            name: {field: result[field] - direct[field] for field in ["primary_depth_cm", "final_shadow_depth_cm"]}
            for name, result in [("main_minus_shared", main_result), ("diagnostic_minus_shared", diagnostic_result), ("ui_minus_shared", ui_result)]
        },
        "video_difference_cm": {field: independent[field] - row[field] for field in ["primary_depth_cm", "final_shadow_depth_cm"]},
    }
    (output / "parity_results.json").write_text(json.dumps(evidence, indent=2))
    print(json.dumps(evidence, indent=2))
