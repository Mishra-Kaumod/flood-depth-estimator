# tests/integration/test_api.py
"""
Integration tests for the FastAPI server.
Uses TestClient — no real server process needed.
Run: pytest tests/integration/test_api.py -v

Requires: pip install httpx pytest-asyncio
"""

import io
import os
import sys
from pathlib import Path

import numpy as np
import cv2
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Set test env vars BEFORE importing the app
os.environ["FLOODWATCH_API_KEY"]    = "test-key-abc123"
os.environ["API_KEY_REQUIRED"]      = "false"   # disable auth for tests
os.environ["FLOODWATCH_DB_URL"]     = "postgresql://x:x@localhost/x"  # won't connect in tests
os.environ["REDIS_URL"]             = "redis://localhost:6379/99"       # test DB
os.environ["ENVIRONMENT"]           = "development"
os.environ["GEMINI_API_KEY"]        = ""        # disable Gemini in tests

from fastapi.testclient import TestClient


# ── Fixtures ──────────────────────────────────────────────────────────────────
def make_test_image_bytes(h=480, w=640, flood=True) -> bytes:
    """Create a synthetic test image. Blue lower half = flood."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    if flood:
        img[h//2:, :] = [180, 50, 30]   # BGR: blue-ish water
    else:
        img[:] = [80, 120, 60]           # BGR: dry grey road
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


@pytest.fixture(scope="module")
def client():
    from api.server import app
    with TestClient(app) as c:
        yield c


# ── Health endpoint ───────────────────────────────────────────────────────────
class TestHealth:

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_has_required_fields(self, client):
        body = client.get("/health").json()
        assert "status"         in body
        assert "pipeline_ready" in body
        assert "gemini_enabled" in body
        assert "timestamp"      in body

    def test_health_status_is_ok(self, client):
        assert client.get("/health").json()["status"] == "ok"


# ── Single prediction ─────────────────────────────────────────────────────────
class TestPredictEndpoint:

    def _post(self, client, flood=True, **overrides):
        img_bytes = make_test_image_bytes(flood=flood)
        data = {
            "camera_id":     overrides.get("camera_id",     "CAM_TEST"),
            "location_id":   overrides.get("location_id",   "LOC_001"),
            "latitude":      overrides.get("latitude",      "12.9172"),
            "longitude":     overrides.get("longitude",     "77.6228"),
            "location_name": overrides.get("location_name", "Test Junction"),
        }
        files = {"image": ("test.jpg", io.BytesIO(img_bytes), "image/jpeg")}
        return client.post("/predict", data=data, files=files)

    def test_predict_returns_200(self, client):
        assert self._post(client).status_code == 200

    def test_predict_has_all_5_outputs(self, client):
        body = self._post(client).json()
        assert "flood_detected"     in body
        assert "water_depth_cm"     in body
        assert "risk_level"         in body
        assert "recommended_action" in body
        assert "confidence_pct"     in body

    def test_predict_metadata_echoed(self, client):
        body = self._post(client, camera_id="CAM_SILK").json()
        assert body["camera_id"]   == "CAM_SILK"
        assert body["latitude"]    == pytest.approx(12.9172)
        assert body["longitude"]   == pytest.approx(77.6228)

    def test_predict_risk_level_valid(self, client):
        valid = {"NO FLOOD","LOW RISK","MODERATE","HIGH RISK","CRITICAL"}
        assert self._post(client).json()["risk_level"] in valid

    def test_predict_confidence_in_range(self, client):
        conf = self._post(client).json()["confidence_pct"]
        assert 0 <= conf <= 100

    def test_predict_invalid_image_returns_422(self, client):
        files = {"image": ("bad.jpg", io.BytesIO(b"not an image"), "image/jpeg")}
        data  = {"camera_id":"C","location_id":"L","latitude":"12","longitude":"77","location_name":"T"}
        resp  = client.post("/predict", data=data, files=files)
        assert resp.status_code == 422

    def test_predict_missing_camera_id_returns_422(self, client):
        img_bytes = make_test_image_bytes()
        files = {"image": ("test.jpg", io.BytesIO(img_bytes), "image/jpeg")}
        data  = {"location_id":"L","latitude":"12","longitude":"77"}
        resp  = client.post("/predict", data=data, files=files)
        assert resp.status_code == 422


# ── Batch prediction ──────────────────────────────────────────────────────────
class TestBatchEndpoint:

    def test_batch_with_2_images(self, client):
        files = [
            ("images", ("img1.jpg", io.BytesIO(make_test_image_bytes(flood=True)),  "image/jpeg")),
            ("images", ("img2.jpg", io.BytesIO(make_test_image_bytes(flood=False)), "image/jpeg")),
        ]
        data = {"camera_id":"CAM_B","location_id":"L","latitude":"12.9","longitude":"77.6","location_name":"X"}
        resp = client.post("/predict/batch", data=data, files=files)
        assert resp.status_code == 200
        body = resp.json()
        assert body["batch_size"] == 2
        assert len(body["results"]) == 2

    def test_batch_result_has_risk_level(self, client):
        files = [("images", ("img.jpg", io.BytesIO(make_test_image_bytes()), "image/jpeg"))]
        data  = {"camera_id":"CAM_B","location_id":"L","latitude":"12.9","longitude":"77.6","location_name":"X"}
        result = client.post("/predict/batch", data=data, files=files).json()["results"][0]
        assert "risk_level" in result
