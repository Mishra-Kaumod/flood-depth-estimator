"""Flask endpoint tests for the Gemini-backed app — no API key or network required."""

import io

import pytest

import app as flood_app
from src.gemini_depth_estimator import GeminiRequestError


class StubGemini:
    """Stands in for GeminiDepthEstimator in app tests."""

    def __init__(self, result=None, exc=None, available=True, model_name="gemini-2.0-flash",
                 fallback_models=("gemini-2.5-flash-lite",)):
        self.result = result
        self.exc = exc
        self.available = available
        self.model_name = model_name
        self.fallback_models = list(fallback_models)
        self.calls = 0

    def estimate(self, image):
        self.calls += 1
        if self.exc:
            raise self.exc
        return dict(self.result)


GEMINI_RESULT = {
    "depth_cm": 42.5,
    "depth_range_cm": [32.0, 45.0],
    "confidence": 0.87,
    "reference_objects": [
        {"name": "sedan_bumper", "known_height_cm": 45.0, "waterline_description": "water just below bumper",
         "lower_bound_cm": 32.0, "upper_bound_cm": 45.0, "depth_estimate_cm": 43.0},
    ],
    "scene_analysis": "Street-level view; road flooded; one usable object.",
    "visual_cues": ["car tire fully submerged"],
    "label_guide": "waterline at bumper",
    "waterline_pct": 55.0,
    "water_coverage": 0.6,
}


@pytest.fixture
def client():
    flood_app.app.config["TESTING"] = True
    with flood_app.app.test_client() as c:
        yield c


@pytest.fixture
def gemini_ok(monkeypatch):
    stub = StubGemini(result=GEMINI_RESULT)
    monkeypatch.setattr(flood_app, "_GEMINI_ESTIMATOR", stub)
    return stub


@pytest.fixture
def gemini_failing(monkeypatch):
    stub = StubGemini(exc=GeminiRequestError("HTTP 429: quota"))
    monkeypatch.setattr(flood_app, "_GEMINI_ESTIMATOR", stub)
    return stub


@pytest.fixture
def gemini_unconfigured(monkeypatch):
    stub = StubGemini(available=False)
    monkeypatch.setattr(flood_app, "_GEMINI_ESTIMATOR", stub)
    return stub


def upload(data_bytes, name="flood.jpg", field="image"):
    return {field: (io.BytesIO(data_bytes), name)}


# ── /predict ──────────────────────────────────────────────────────────

class TestPredict:
    def test_gemini_success(self, client, gemini_ok, sample_image_bytes):
        resp = client.post("/predict", data=upload(sample_image_bytes),
                           content_type="multipart/form-data")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["method"] == "gemini"
        assert body["depth_cm"] == 42.5
        assert body["confidence"] == 0.87
        assert body["severity"]["level"] == "MEDIUM"  # 20 ≤ 42.5 < 50
        assert body["visual_cues"] == ["car tire fully submerged"]
        assert body["reference_objects"][0]["name"] == "sedan_bumper"
        assert body["depth_range_cm"] == [32.0, 45.0]
        assert body["scene_analysis"].startswith("Street-level view")
        assert body["backend"] == "gemini"
        assert gemini_ok.calls == 1

    def test_gemini_failure_falls_back_to_cv(self, client, gemini_failing, sample_image_bytes):
        resp = client.post("/predict", data=upload(sample_image_bytes),
                           content_type="multipart/form-data")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["method"] == "reference_object_cv"
        assert isinstance(body["depth_cm"], (int, float))
        assert 0 <= body["depth_cm"] <= 150
        assert gemini_failing.calls == 1

    def test_no_key_uses_cv_without_calling_gemini(self, client, gemini_unconfigured, sample_image_bytes):
        resp = client.post("/predict", data=upload(sample_image_bytes),
                           content_type="multipart/form-data")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["method"] == "reference_object_cv"
        assert body["backend"] == "reference_object_cv"
        assert gemini_unconfigured.calls == 0

    def test_missing_image_field_400(self, client, gemini_ok):
        resp = client.post("/predict", data={}, content_type="multipart/form-data")
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_corrupt_image_400(self, client, gemini_ok):
        resp = client.post("/predict", data=upload(b"not an image"),
                           content_type="multipart/form-data")
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    @pytest.mark.parametrize("depth,level", [
        (0.0, "SAFE"), (4.9, "SAFE"), (5.0, "LOW"), (19.9, "LOW"),
        (20.0, "MEDIUM"), (49.9, "MEDIUM"), (50.0, "HIGH"), (79.9, "HIGH"),
        (80.0, "CRITICAL"), (150.0, "CRITICAL"),
    ])
    def test_severity_mapping(self, client, monkeypatch, sample_image_bytes, depth, level):
        stub = StubGemini(result={**GEMINI_RESULT, "depth_cm": depth})
        monkeypatch.setattr(flood_app, "_GEMINI_ESTIMATOR", stub)
        resp = client.post("/predict", data=upload(sample_image_bytes),
                           content_type="multipart/form-data")
        body = resp.get_json()
        assert body["severity"]["level"] == level
        assert body["severity"]["stage"] in range(1, 6)


# ── /predict-batch ────────────────────────────────────────────────────

class TestPredictBatch:
    def test_batch_with_metadata(self, client, gemini_ok, sample_image_bytes):
        data = {
            "images": [
                (io.BytesIO(sample_image_bytes), "a.jpg"),
                (io.BytesIO(sample_image_bytes), "b.jpg"),
            ],
            "lats": ["12.93", "12.97"],
            "lngs": ["77.62", "77.64"],
            "names": ["Koramangala", "Indiranagar"],
        }
        resp = client.post("/predict-batch", data=data, content_type="multipart/form-data")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["count"] == 2
        first = body["results"][0]
        assert first["status"] == "ok"
        assert first["name"] == "Koramangala"
        assert first["lat"] == 12.93
        assert first["lng"] == 77.62
        assert first["method"] == "gemini"
        assert gemini_ok.calls == 2

    def test_batch_defaults_when_metadata_missing(self, client, gemini_ok, sample_image_bytes):
        data = {"images": [(io.BytesIO(sample_image_bytes), "solo.jpg")]}
        resp = client.post("/predict-batch", data=data, content_type="multipart/form-data")
        result = resp.get_json()["results"][0]
        assert result["lat"] == 12.9716
        assert result["lng"] == 77.5946
        assert result["name"] == "solo.jpg"

    def test_batch_partial_failure(self, client, gemini_ok, sample_image_bytes):
        data = {
            "images": [
                (io.BytesIO(sample_image_bytes), "good.jpg"),
                (io.BytesIO(b"corrupt bytes"), "bad.jpg"),
            ],
        }
        resp = client.post("/predict-batch", data=data, content_type="multipart/form-data")
        assert resp.status_code == 200
        results = resp.get_json()["results"]
        assert results[0]["status"] == "ok"
        assert results[1]["status"] == "error"
        assert "error" in results[1]

    def test_batch_empty(self, client, gemini_ok):
        resp = client.post("/predict-batch", data={}, content_type="multipart/form-data")
        assert resp.status_code == 200
        assert resp.get_json() == {"results": [], "count": 0}

    def test_batch_gemini_down_still_succeeds(self, client, gemini_failing, sample_image_bytes):
        data = {"images": [(io.BytesIO(sample_image_bytes), "a.jpg")]}
        resp = client.post("/predict-batch", data=data, content_type="multipart/form-data")
        results = resp.get_json()["results"]
        assert results[0]["status"] == "ok"
        assert results[0]["method"] == "reference_object_cv"


# ── /health and / ─────────────────────────────────────────────────────

class TestHealthAndIndex:
    def test_health_with_gemini(self, client, gemini_ok):
        body = client.get("/health").get_json()
        assert body["status"] == "ok"
        assert body["gemini_available"] is True
        assert body["active_method"] == "gemini"
        assert body["warning"] is None
        assert body["model"] == "gemini-2.0-flash"
        assert body["fallback_models"] == ["gemini-2.5-flash-lite"]
        assert body["reference_cv_available"] is True

    def test_health_without_gemini(self, client, gemini_unconfigured):
        body = client.get("/health").get_json()
        assert body["status"] == "ok"
        assert body["gemini_available"] is False
        assert body["active_method"] == "reference_object_cv"
        assert body["warning"] == "gemini_api_key_missing"

    def test_index_serves_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"Bengaluru Flood Depth Estimator" in resp.data

    def test_no_torch_dependency(self):
        """The app module must not import torch anywhere."""
        import sys
        assert "torch" not in sys.modules
        source = open(flood_app.__file__, encoding="utf-8").read()
        assert "import torch" not in source
        assert "torchvision" not in source


# ── prediction result contract ────────────────────────────────────────

class TestPredictionContract:
    EXPECTED_KEYS = {
        "depth_cm", "depth_range_cm", "confidence", "severity", "method",
        "visual_cues", "label_guide", "scene_analysis", "waterline_pct", "water_coverage",
        "reference_objects", "model_used",
    }

    def test_gemini_result_shape(self, client, gemini_ok, sample_image_bytes):
        body = client.post("/predict", data=upload(sample_image_bytes),
                           content_type="multipart/form-data").get_json()
        assert self.EXPECTED_KEYS <= set(body.keys())
        assert {"level", "label", "color", "stage"} <= set(body["severity"].keys())

    def test_cv_fallback_result_shape(self, client, gemini_unconfigured, sample_image_bytes):
        body = client.post("/predict", data=upload(sample_image_bytes),
                           content_type="multipart/form-data").get_json()
        assert self.EXPECTED_KEYS <= set(body.keys())
        assert body["reference_objects"] == []  # CV estimator has no catalog calculations
        assert body["depth_range_cm"] == [body["depth_cm"], body["depth_cm"]]
        assert body["scene_analysis"] == ""
