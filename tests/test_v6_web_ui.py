"""V6 UI contract and CLI RGB parity checks without model downloads."""
from io import BytesIO
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from src.v6_shadow_pipeline import V6ShadowPipeline
from web_app import create_app


@pytest.mark.parametrize("depth", [12.3456789, 0.0, None])
def test_upload_matches_v6_cli_contract(depth):
    pixels = np.array([[[233, 42, 7], [12, 67, 201]]], dtype=np.uint8)
    upload = BytesIO()
    Image.fromarray(pixels).save(upload, format="PNG")
    data = upload.getvalue()
    with Image.open(BytesIO(data)) as image:
        cli_rgb = np.asarray(image.convert("RGB"))

    class V5Signals:
        def predict(self, rgb):
            np.testing.assert_array_equal(rgb, cli_rgb)
            return {"depth_cm": 999, "structured_features": {
                "efficientnet_candidate_depth_cm": depth}}

    pipeline = V6ShadowPipeline(V5Signals())
    expected = pipeline.predict(cli_rgb).final_shadow_depth_cm
    with patch("src.efficientnet_depth_signal.EfficientNetDepthSignal", return_value=V5Signals()):
        response = create_app().test_client().post("/predict", data={
            "image": (BytesIO(data), "input.png")})
    assert response.status_code == 200
    assert response.get_json() == {"primary_depth_cm": expected, "final_shadow_depth_cm": expected, "numerical_owner": "efficientnet_primary_anchor"}


def test_home_displays_v6_depth_directly():
    body = create_app().test_client().get("/").get_data(as_text=True)
    assert body.count("Flood Depth Estimator – V6") == 2
    assert 'result.final_shadow_depth_cm + " cm"' in body
    assert "estimated_depth_meters" not in body


@pytest.mark.parametrize("data", [{}, {"image": (BytesIO(b""), "empty.png")},
                                     {"image": (BytesIO(b"invalid"), "bad.png")}])
def test_invalid_uploads(data):
    assert create_app().test_client().post("/predict", data=data).status_code == 400
