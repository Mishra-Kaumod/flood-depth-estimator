"""Unit tests for src.gemini_depth_estimator — no API key or network required."""

import base64
import json

import pytest

from src.gemini_depth_estimator import (
    DEFAULT_MODEL,
    GEMINI_ENDPOINT,
    GeminiDepthEstimator,
    GeminiRequestError,
    GeminiResponseError,
    GeminiUnavailableError,
    _extract_json,
    _first_number,
)


# ── fakes ────────────────────────────────────────────────────────────

class FakeResponse:
    def __init__(self, status_code=200, json_data=None, text="", raise_on_json=False):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text or (json.dumps(json_data) if json_data is not None else "")
        self._raise_on_json = raise_on_json

    def json(self):
        if self._raise_on_json:
            raise ValueError("not json")
        return self._json_data


class FakeSession:
    def __init__(self, response=None, exc=None, script=None):
        # script: ordered list of FakeResponse objects or Exceptions, one per call.
        # Without a script, every call returns `response` (or raises `exc`).
        self.response = response
        self.exc = exc
        self.script = list(script) if script else None
        self.calls = []

    def post(self, url, params=None, json=None, timeout=None):
        self.calls.append({"url": url, "params": params, "json": json, "timeout": timeout})
        if self.script is not None:
            step = self.script.pop(0)
            if isinstance(step, Exception):
                raise step
            return step
        if self.exc:
            raise self.exc
        return self.response


def gemini_text_response(text):
    """Wrap text the way the generateContent endpoint returns it."""
    return {"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}]}


def good_response():
    return FakeResponse(json_data=gemini_text_response(GOOD_JSON_REPLY))


def make_estimator(reply_text=None, response=None, exc=None, script=None, api_key="test-key",
                   max_retries=0, fallback_models=(), model_name="gemini-pro-latest"):
    if response is None and reply_text is not None:
        response = FakeResponse(json_data=gemini_text_response(reply_text))
    session = FakeSession(response=response, exc=exc, script=script)
    est = GeminiDepthEstimator(
        api_key=api_key, model_name=model_name, session=session,
        max_retries=max_retries, fallback_models=list(fallback_models),
    )
    est._sleep = lambda s: None  # never sleep in tests
    return est, session


GOOD_JSON_REPLY = json.dumps({
    "scene_analysis": "Street-level view; water covers the road; two usable reference objects.",
    "reference_objects": [
        {"name": "sedan_bumper", "known_height_cm": 45, "waterline_description": "water just below bumper",
         "lower_bound_cm": 32, "upper_bound_cm": 45, "depth_estimate_cm": 43},
        {"name": "adult_knee", "known_height_cm": 50, "waterline_description": "water below knee",
         "lower_bound_cm": 30, "upper_bound_cm": 50, "depth_estimate_cm": 42},
    ],
    "depth_range_cm": [32, 45],
    "depth_cm": 42.5,
    "confidence": 0.87,
    "visual_cues": ["car tire fully submerged", "water at bumper"],
    "water_coverage": 0.6,
    "waterline_pct": 55.0,
    "label_guide": "waterline at car bumper ≈ 45 cm",
})


# ── configuration / availability ─────────────────────────────────────

class TestConfiguration:
    def test_from_env_reads_gemini_key(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "abc123")
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        est = GeminiDepthEstimator.from_env()
        assert est.api_key == "abc123"
        assert est.available

    def test_from_env_falls_back_to_google_key(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "g-key")
        est = GeminiDepthEstimator.from_env()
        assert est.api_key == "g-key"

    def test_from_env_model_and_timeout_overrides(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "k")
        monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-pro")
        monkeypatch.setenv("GEMINI_TIMEOUT", "12.5")
        est = GeminiDepthEstimator.from_env()
        assert est.model_name == "gemini-2.5-pro"
        assert est.timeout_s == 12.5

    def test_from_env_defaults(self, monkeypatch):
        for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_MODEL", "GEMINI_TIMEOUT"):
            monkeypatch.delenv(var, raising=False)
        est = GeminiDepthEstimator.from_env()
        assert est.model_name == DEFAULT_MODEL
        assert not est.available

    def test_whitespace_key_is_unavailable(self):
        assert not GeminiDepthEstimator(api_key="   ").available

    def test_estimate_without_key_raises_unavailable(self, sample_image):
        est = GeminiDepthEstimator(api_key="")
        with pytest.raises(GeminiUnavailableError):
            est.estimate(sample_image)


# ── request construction ──────────────────────────────────────────────

class TestRequest:
    def test_url_contains_model_and_key_in_params(self, sample_image):
        est, session = make_estimator(reply_text=GOOD_JSON_REPLY)
        est.model_name = "gemini-2.0-flash"
        est.estimate(sample_image)
        call = session.calls[0]
        assert call["url"] == GEMINI_ENDPOINT.format(model="gemini-2.0-flash")
        assert call["params"] == {"key": "test-key"}
        assert call["timeout"] == est.timeout_s

    def test_payload_has_prompt_and_inline_jpeg(self, sample_image):
        est, session = make_estimator(reply_text=GOOD_JSON_REPLY)
        est.estimate(sample_image)
        payload = session.calls[0]["json"]
        parts = payload["contents"][0]["parts"]
        assert "depth" in parts[0]["text"].lower()
        inline = parts[1]["inline_data"]
        assert inline["mime_type"] == "image/jpeg"
        raw = base64.b64decode(inline["data"])
        assert raw[:3] == b"\xff\xd8\xff"  # JPEG magic bytes

    def test_prompt_uses_model_knowledge_not_catalog(self, sample_image):
        """No hardcoded catalog — the model must use its own knowledge of object sizes."""
        est, session = make_estimator(reply_text=GOOD_JSON_REPLY)
        est.estimate(sample_image)
        prompt = session.calls[0]["json"]["contents"][0]["parts"][0]["text"]
        assert "Reference Object Dimensions" not in prompt  # knowledge_base.py catalog absent
        assert "your own" in prompt and "knowledge" in prompt
        assert "reference_objects" in prompt  # response schema still asks for per-object estimates

    def test_prompt_has_robust_method_steps(self, sample_image):
        """Bracketing, validity rules, and reason-before-answer ordering must be in the prompt."""
        est, session = make_estimator(reply_text=GOOD_JSON_REPLY)
        est.estimate(sample_image)
        prompt = session.calls[0]["json"]["contents"][0]["parts"][0]["text"]
        assert "SCENE ANALYSIS" in prompt
        assert "lower bound" in prompt and "upper bound" in prompt
        assert "VALIDITY RULES" in prompt
        assert "kerbs/sidewalks" in prompt          # ground-level correction
        assert "upright, standing adults" in prompt  # posture rule
        assert "depth_range_cm" in prompt
        assert "IN THIS ORDER" in prompt             # reasoning precedes the answer

    def test_payload_requests_json_output(self, sample_image):
        est, session = make_estimator(reply_text=GOOD_JSON_REPLY)
        est.estimate(sample_image)
        cfg = session.calls[0]["json"]["generationConfig"]
        assert cfg["responseMimeType"] == "application/json"
        assert cfg["temperature"] == 0.0

    def test_rgba_image_is_converted(self, sample_image):
        est, session = make_estimator(reply_text=GOOD_JSON_REPLY)
        rgba = sample_image.convert("RGBA")
        result = est.estimate(rgba)
        assert result["depth_cm"] == 42.5


# ── response parsing ──────────────────────────────────────────────────

class TestParsing:
    def test_clean_json_reply(self, sample_image):
        est, _ = make_estimator(reply_text=GOOD_JSON_REPLY)
        result = est.estimate(sample_image)
        assert result["depth_cm"] == 42.5
        assert result["confidence"] == 0.87
        assert result["visual_cues"] == ["car tire fully submerged", "water at bumper"]
        assert result["water_coverage"] == 0.6
        assert result["waterline_pct"] == 55.0
        assert result["label_guide"] == "waterline at car bumper ≈ 45 cm"
        assert result["scene_analysis"].startswith("Street-level view")
        assert result["depth_range_cm"] == [32.0, 45.0]
        assert result["reference_objects"] == [
            {"name": "sedan_bumper", "known_height_cm": 45.0, "waterline_description": "water just below bumper",
             "lower_bound_cm": 32.0, "upper_bound_cm": 45.0, "depth_estimate_cm": 43.0},
            {"name": "adult_knee", "known_height_cm": 50.0, "waterline_description": "water below knee",
             "lower_bound_cm": 30.0, "upper_bound_cm": 50.0, "depth_estimate_cm": 42.0},
        ]

    def test_missing_reference_objects_defaults_empty(self, sample_image):
        est, _ = make_estimator(reply_text=json.dumps({"depth_cm": 30}))
        result = est.estimate(sample_image)
        assert result["reference_objects"] == []
        assert result["scene_analysis"] == ""
        assert result["depth_range_cm"] == [30.0, 30.0]  # degenerate range from depth

    def test_malformed_reference_objects_items_dropped(self, sample_image):
        reply = json.dumps({
            "depth_cm": 30,
            "reference_objects": [
                "not a dict",
                {"name": "adult_knee", "known_height_cm": "fifty", "depth_estimate_cm": 42},
                {"name": "sedan_bumper", "known_height_cm": 45, "depth_estimate_cm": 999},
            ],
        })
        est, _ = make_estimator(reply_text=reply)
        refs = est.estimate(sample_image)["reference_objects"]
        assert len(refs) == 1  # only the last item survives
        assert refs[0]["name"] == "sedan_bumper"
        assert refs[0]["depth_estimate_cm"] == 150.0  # clipped to MAX_DEPTH_CM
        assert refs[0]["waterline_description"] == ""
        assert refs[0]["lower_bound_cm"] is None  # bounds absent → None, not fabricated
        assert refs[0]["upper_bound_cm"] is None

    def test_non_list_reference_objects_defaults_empty(self, sample_image):
        est, _ = make_estimator(reply_text=json.dumps({"depth_cm": 30, "reference_objects": "a car"}))
        assert est.estimate(sample_image)["reference_objects"] == []

    def test_inverted_depth_range_degrades_to_point(self, sample_image):
        est, _ = make_estimator(reply_text=json.dumps({"depth_cm": 30, "depth_range_cm": [50, 20]}))
        assert est.estimate(sample_image)["depth_range_cm"] == [30.0, 30.0]

    def test_malformed_depth_range_degrades_to_point(self, sample_image):
        est, _ = make_estimator(reply_text=json.dumps({"depth_cm": 30, "depth_range_cm": "20 to 50"}))
        assert est.estimate(sample_image)["depth_range_cm"] == [30.0, 30.0]

    def test_depth_range_clipped(self, sample_image):
        est, _ = make_estimator(reply_text=json.dumps({"depth_cm": 30, "depth_range_cm": [-5, 900]}))
        assert est.estimate(sample_image)["depth_range_cm"] == [0.0, 150.0]

    def test_fenced_json_reply(self, sample_image):
        est, _ = make_estimator(reply_text=f"```json\n{GOOD_JSON_REPLY}\n```")
        assert est.estimate(sample_image)["depth_cm"] == 42.5

    def test_json_embedded_in_prose(self, sample_image):
        text = f"Here is my analysis:\n{GOOD_JSON_REPLY}\nHope that helps!"
        est, _ = make_estimator(reply_text=text)
        assert est.estimate(sample_image)["depth_cm"] == 42.5

    def test_plain_number_fallback(self, sample_image):
        est, _ = make_estimator(reply_text="The flood depth is approximately 45 cm.")
        result = est.estimate(sample_image)
        assert result["depth_cm"] == 45.0
        assert result["confidence"] == 0.5
        assert result["reference_objects"] == []
        assert result["depth_range_cm"] == [45.0, 45.0]
        assert result["scene_analysis"] == ""

    def test_depth_clipped_to_max(self, sample_image):
        est, _ = make_estimator(reply_text=json.dumps({"depth_cm": 999, "confidence": 0.9}))
        assert est.estimate(sample_image)["depth_cm"] == 150.0

    def test_negative_depth_clipped_to_zero(self, sample_image):
        est, _ = make_estimator(reply_text=json.dumps({"depth_cm": -10, "confidence": 0.9}))
        assert est.estimate(sample_image)["depth_cm"] == 0.0

    def test_confidence_clipped(self, sample_image):
        est, _ = make_estimator(reply_text=json.dumps({"depth_cm": 30, "confidence": 5.0}))
        assert est.estimate(sample_image)["confidence"] == 1.0

    def test_missing_optional_fields_get_defaults(self, sample_image):
        est, _ = make_estimator(reply_text=json.dumps({"depth_cm": 30}))
        result = est.estimate(sample_image)
        assert result["confidence"] == 0.6
        assert result["visual_cues"] == []
        assert result["label_guide"] == ""
        assert result["waterline_pct"] == 0.0
        assert result["water_coverage"] == 0.0

    def test_non_list_visual_cues_becomes_empty(self, sample_image):
        est, _ = make_estimator(reply_text=json.dumps({"depth_cm": 30, "visual_cues": "a car"}))
        assert est.estimate(sample_image)["visual_cues"] == []

    def test_non_numeric_optional_fields_get_defaults(self, sample_image):
        est, _ = make_estimator(reply_text=json.dumps({
            "depth_cm": 30, "confidence": "high", "water_coverage": "most", "waterline_pct": None,
        }))
        result = est.estimate(sample_image)
        assert result["confidence"] == 0.6
        assert result["water_coverage"] == 0.0
        assert result["waterline_pct"] == 0.0

    def test_garbage_reply_raises(self, sample_image):
        est, _ = make_estimator(reply_text="I cannot help with that request.")
        with pytest.raises(GeminiResponseError):
            est.estimate(sample_image)

    def test_non_numeric_depth_raises(self, sample_image):
        est, _ = make_estimator(reply_text=json.dumps({"depth_cm": "unknown"}))
        with pytest.raises(GeminiResponseError):
            est.estimate(sample_image)


# ── retries and model fallback ────────────────────────────────────────

class TestRetriesAndFallback:
    def test_transient_503_retried_then_succeeds(self, sample_image):
        est, session = make_estimator(
            script=[FakeResponse(status_code=503, text="high demand"), good_response()],
            max_retries=2,
        )
        result = est.estimate(sample_image)
        assert result["depth_cm"] == 42.5
        assert len(session.calls) == 2

    def test_network_error_retried_then_succeeds(self, sample_image):
        est, session = make_estimator(
            script=[ConnectionError("reset"), good_response()],
            max_retries=1,
        )
        assert est.estimate(sample_image)["depth_cm"] == 42.5
        assert len(session.calls) == 2

    def test_retries_exhausted_raises(self, sample_image):
        est, session = make_estimator(
            script=[FakeResponse(status_code=503, text="x")] * 3,
            max_retries=2,
        )
        with pytest.raises(GeminiRequestError, match="503"):
            est.estimate(sample_image)
        assert len(session.calls) == 3  # 1 attempt + 2 retries

    def test_non_retryable_400_fails_fast(self, sample_image):
        est, session = make_estimator(
            script=[FakeResponse(status_code=400, text="bad request")],
            max_retries=2,
        )
        with pytest.raises(GeminiRequestError, match="400"):
            est.estimate(sample_image)
        assert len(session.calls) == 1  # no retries burned on permanent errors

    def test_daily_quota_skips_retries_and_uses_fallback_model(self, sample_image):
        quota_text = "Quota exceeded for metric: generate_content_free_tier_requests ...PerDay..."
        est, session = make_estimator(
            script=[FakeResponse(status_code=429, text=quota_text), good_response()],
            max_retries=2,
            fallback_models=["gemini-2.5-flash-lite"],
        )
        result = est.estimate(sample_image)
        assert result["depth_cm"] == 42.5
        assert result["model_used"] == "gemini-2.5-flash-lite"
        assert len(session.calls) == 2  # no same-model retries for exhausted daily quota
        assert "gemini-pro-latest" in session.calls[0]["url"]
        assert "gemini-2.5-flash-lite" in session.calls[1]["url"]

    def test_throttle_429_without_perday_is_retried_on_same_model(self, sample_image):
        est, session = make_estimator(
            script=[FakeResponse(status_code=429, text="rate limit, retry shortly"), good_response()],
            max_retries=1,
        )
        result = est.estimate(sample_image)
        assert result["model_used"] == "gemini-pro-latest"
        assert len(session.calls) == 2
        assert session.calls[0]["url"] == session.calls[1]["url"]

    def test_all_models_fail_raises_last_error(self, sample_image):
        est, session = make_estimator(
            script=[FakeResponse(status_code=403, text="forbidden")] * 3,
            fallback_models=["m2", "m3"],
        )
        with pytest.raises(GeminiRequestError, match="403"):
            est.estimate(sample_image)
        assert len(session.calls) == 3  # each model tried once

    def test_primary_success_never_touches_fallbacks(self, sample_image):
        est, session = make_estimator(
            script=[good_response()],
            fallback_models=["m2"],
        )
        result = est.estimate(sample_image)
        assert result["model_used"] == "gemini-pro-latest"
        assert len(session.calls) == 1

    def test_fallback_equal_to_primary_is_deduped(self, sample_image):
        est, session = make_estimator(
            script=[FakeResponse(status_code=403, text="forbidden")],
            fallback_models=["gemini-pro-latest"],
        )
        with pytest.raises(GeminiRequestError):
            est.estimate(sample_image)
        assert len(session.calls) == 1

    def test_backoff_delays_are_exponential(self, sample_image):
        est, session = make_estimator(
            script=[FakeResponse(status_code=503, text="x")] * 3,
            max_retries=2,
        )
        delays = []
        est._sleep = delays.append
        with pytest.raises(GeminiRequestError):
            est.estimate(sample_image)
        assert delays == [1.5, 3.0]

    def test_from_env_retry_and_fallback_config(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "k")
        monkeypatch.setenv("GEMINI_MAX_RETRIES", "5")
        monkeypatch.setenv("GEMINI_RETRY_BASE_S", "0.5")
        monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "a-model, b-model")
        est = GeminiDepthEstimator.from_env()
        assert est.max_retries == 5
        assert est.retry_base_s == 0.5
        assert est.fallback_models == ["a-model", "b-model"]

    def test_from_env_default_fallbacks(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "k")
        monkeypatch.delenv("GEMINI_FALLBACK_MODELS", raising=False)
        est = GeminiDepthEstimator.from_env()
        assert est.fallback_models == ["gemini-2.5-flash-lite", "gemini-2.0-flash"]

    def test_from_env_empty_fallbacks_disables_chain(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "k")
        monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "")
        est = GeminiDepthEstimator.from_env()
        assert est.fallback_models == []


# ── error handling ────────────────────────────────────────────────────

class TestErrors:
    @pytest.mark.parametrize("status", [400, 403, 429, 500, 503])
    def test_http_error_raises_request_error(self, sample_image, status):
        est, _ = make_estimator(response=FakeResponse(status_code=status, text="quota exceeded"))
        with pytest.raises(GeminiRequestError, match=str(status)):
            est.estimate(sample_image)

    def test_network_exception_raises_request_error(self, sample_image):
        est, _ = make_estimator(exc=ConnectionError("dns failure"))
        with pytest.raises(GeminiRequestError, match="dns failure"):
            est.estimate(sample_image)

    def test_non_json_body_raises_response_error(self, sample_image):
        est, _ = make_estimator(response=FakeResponse(text="<html>gateway</html>", raise_on_json=True))
        with pytest.raises(GeminiResponseError, match="non-JSON"):
            est.estimate(sample_image)

    def test_no_candidates_raises_with_block_reason(self, sample_image):
        body = {"candidates": [], "promptFeedback": {"blockReason": "SAFETY"}}
        est, _ = make_estimator(response=FakeResponse(json_data=body))
        with pytest.raises(GeminiResponseError, match="SAFETY"):
            est.estimate(sample_image)

    def test_empty_parts_raises_with_finish_reason(self, sample_image):
        body = {"candidates": [{"content": {"parts": []}, "finishReason": "MAX_TOKENS"}]}
        est, _ = make_estimator(response=FakeResponse(json_data=body))
        with pytest.raises(GeminiResponseError, match="MAX_TOKENS"):
            est.estimate(sample_image)


# ── helpers ───────────────────────────────────────────────────────────

class TestHelpers:
    def test_extract_json_variants(self):
        obj = {"depth_cm": 1}
        assert _extract_json(json.dumps(obj)) == obj
        assert _extract_json(f"```json\n{json.dumps(obj)}\n```") == obj
        assert _extract_json(f"```\n{json.dumps(obj)}\n```") == obj
        assert _extract_json(f"prefix {json.dumps(obj)} suffix") == obj
        assert _extract_json("no json here") is None
        assert _extract_json("") is None
        assert _extract_json("[1, 2, 3]") is None  # list, not object

    def test_first_number(self):
        assert _first_number("about 45 cm") == 45.0
        assert _first_number("depth: 12.75cm today") == 12.75
        assert _first_number("-5 cm") == -5.0
        assert _first_number("no digits") is None
        assert _first_number("") is None
