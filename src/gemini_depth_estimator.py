"""
Gemini Depth Estimator
======================
Estimates flood depth by sending the image to the Gemini generateContent
REST endpoint and parsing a strict-JSON reply. Replaces the local PyTorch
EfficientNet regression model as the primary prediction path.

Configuration (environment variables):

  GEMINI_API_KEY   (or GOOGLE_API_KEY)  — API key; estimator is unavailable without it
  GEMINI_MODEL     — model name, default "gemini-2.0-flash"
  GEMINI_TIMEOUT   — request timeout in seconds, default 30

The estimator raises GeminiError subclasses on any failure so callers can
fall back to the reference-object CV estimator.
"""

import base64
import io
import json
import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)

GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

DEFAULT_MODEL = "gemini-pro-latest"
DEFAULT_TIMEOUT_S = 60.0
MAX_DEPTH_CM = 150.0

PROMPT = (
    "You are a precise flood-depth measurement analyst. Determine the flood water depth in "
    "centimeters at the deepest clearly-flooded part of the roadway in this image. Use your own "
    "accurate knowledge of real-world object dimensions (human anthropometry, vehicles, street "
    "furniture, kerbs, signage, animals, building features) as your measuring instruments. "
    "Typical South Indian flood water is muddy brown. Wet pavement, thin puddles, and "
    "reflections are NOT flooding — report them as depth ≤ 2 cm.\n"
    "\n"
    "METHOD — follow every step, in order:\n"
    "\n"
    "STEP 1 — SCENE ANALYSIS (write this before any numbers): camera viewpoint and distance; "
    "where standing water actually is vs merely wet ground; whether the ground is level or "
    "sloped; which objects are standing IN the water and usable for measurement.\n"
    "\n"
    "STEP 2 — BRACKET WITH EACH USABLE OBJECT. For every measurable object standing in the "
    "water at road level, state the object's true height from your world knowledge, then:\n"
    "   - lower bound: the HIGHEST landmark on it that is clearly UNDER water "
    "→ depth ≥ that landmark's height.\n"
    "   - upper bound: the LOWEST landmark on it that is clearly DRY/above water "
    "→ depth ≤ that landmark's height.\n"
    "   - point estimate INSIDE that bracket: exact landmark match, or "
    "known height × submerged fraction.\n"
    "\n"
    "STEP 3 — VALIDITY RULES (apply strictly):\n"
    "   - Only use objects at the SAME ground level as the water you are measuring. Objects on "
    "kerbs/sidewalks stand above the road — subtract that elevation or skip them.\n"
    "   - Only use upright, standing adults. Skip crouching, sitting, wading-bent people and "
    "children unless clearly identifiable as such.\n"
    "   - An expected object that is fully invisible below the water sets a lower bound equal to "
    "its full height.\n"
    "   - Prefer objects near the camera and mid-frame; distant objects suffer perspective error.\n"
    "   - Weight objects whose dimensions are standardized (kerbs, vehicles, signs) over ones "
    "that vary (children, animals, vegetation).\n"
    "\n"
    "STEP 4 — COMBINE: intersect all brackets into depth_range_cm = [max of lower bounds, min of "
    "upper bounds]. Discard point estimates outside it. depth_cm = median of the survivors and "
    "MUST lie within depth_range_cm.\n"
    "\n"
    "STEP 5 — CONFIDENCE rubric: 0.85-0.95 several agreeing reliable objects; 0.6-0.8 one "
    "solid object or minor disagreement; 0.3-0.5 weak, occluded or distant cues only; ≤0.2 no "
    "usable object (depth guessed from texture/context alone).\n"
    "\n"
    "Respond with ONLY a JSON object, no prose. Produce the keys IN THIS ORDER so the reasoning "
    "comes before the answer:\n"
    '{"scene_analysis": "<3-4 sentences from STEP 1>", '
    '"reference_objects": [{"name": "<object used>", '
    '"known_height_cm": <float, its true height from your world knowledge>, '
    '"waterline_description": "<where the water sits on this object>", '
    '"lower_bound_cm": <float>, "upper_bound_cm": <float>, '
    '"depth_estimate_cm": <float, point estimate from this object>}], '
    '"depth_range_cm": [<float lower>, <float upper>], '
    '"depth_cm": <float 0-150, median estimate, inside depth_range_cm>, '
    '"confidence": <float 0-1>, '
    '"visual_cues": [<short strings describing waterline evidence>], '
    '"water_coverage": <float 0-1 fraction of frame covered by water>, '
    '"waterline_pct": <float 0-100 waterline height as % of the primary reference object>, '
    '"label_guide": "<one-line summary of the calculation>"}'
)


class GeminiError(Exception):
    """Base error for Gemini prediction failures."""


class GeminiUnavailableError(GeminiError):
    """No API key configured — Gemini cannot be called."""


class GeminiRequestError(GeminiError):
    """The HTTP request failed (network, timeout, non-200, quota)."""


class GeminiResponseError(GeminiError):
    """The API replied but the content could not be parsed into a depth."""


def _extract_json(text: str) -> Optional[dict]:
    """Parse a JSON object out of model text, tolerating code fences and surrounding prose."""
    if not text:
        return None
    stripped = text.strip()
    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, re.DOTALL)
    if fence:
        stripped = fence.group(1).strip()
    try:
        obj = json.loads(stripped)
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, ValueError):
        pass
    # Fall back to the first {...} block in the text
    brace = re.search(r"\{.*\}", stripped, re.DOTALL)
    if brace:
        try:
            obj = json.loads(brace.group(0))
            return obj if isinstance(obj, dict) else None
        except (json.JSONDecodeError, ValueError):
            return None
    return None


def _first_number(text: str) -> Optional[float]:
    m = re.search(r"(-?\d+(?:\.\d+)?)", text or "")
    return float(m.group(1)) if m else None


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _parse_reference_objects(raw) -> list:
    """Coerce the model's reference_objects array into clean dicts, dropping malformed items."""
    if not isinstance(raw, list):
        return []
    objects = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            obj = {
                "name": str(item.get("name", "")),
                "known_height_cm": float(item.get("known_height_cm", 0.0)),
                "waterline_description": str(item.get("waterline_description", "")),
                "depth_estimate_cm": round(
                    _clip(float(item.get("depth_estimate_cm", 0.0)), 0.0, MAX_DEPTH_CM), 2
                ),
            }
        except (TypeError, ValueError):
            continue
        for bound in ("lower_bound_cm", "upper_bound_cm"):
            try:
                obj[bound] = round(_clip(float(item[bound]), 0.0, MAX_DEPTH_CM), 2)
            except (KeyError, TypeError, ValueError):
                obj[bound] = None
        objects.append(obj)
    return objects


def _parse_depth_range(raw, depth_cm: float) -> list:
    """Coerce depth_range_cm into a sane [lower, upper] containing nothing weird."""
    try:
        lower, upper = float(raw[0]), float(raw[1])
    except (TypeError, ValueError, IndexError, KeyError):
        return [depth_cm, depth_cm]
    lower = _clip(lower, 0.0, MAX_DEPTH_CM)
    upper = _clip(upper, 0.0, MAX_DEPTH_CM)
    if lower > upper:
        return [depth_cm, depth_cm]
    return [round(lower, 2), round(upper, 2)]


class GeminiDepthEstimator:
    def __init__(
        self,
        api_key: str = "",
        model_name: str = DEFAULT_MODEL,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        session=None,
    ):
        self.api_key = (api_key or "").strip()
        self.model_name = model_name or DEFAULT_MODEL
        self.timeout_s = timeout_s
        self._session = session  # injectable for tests; lazily built otherwise

    @classmethod
    def from_env(cls) -> "GeminiDepthEstimator":
        return cls(
            api_key=os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", ""),
            model_name=os.environ.get("GEMINI_MODEL", DEFAULT_MODEL),
            timeout_s=float(os.environ.get("GEMINI_TIMEOUT", DEFAULT_TIMEOUT_S)),
        )

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    # ── request plumbing ─────────────────────────────────────────────

    def _get_session(self):
        if self._session is None:
            import requests
            self._session = requests.Session()
        return self._session

    @staticmethod
    def _encode_image(image) -> str:
        """PIL Image → base64 JPEG payload for inline_data."""
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _build_payload(self, image) -> dict:
        return {
            "contents": [{
                "parts": [
                    {"text": PROMPT},
                    {"inline_data": {"mime_type": "image/jpeg", "data": self._encode_image(image)}},
                ],
            }],
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json",
            },
        }

    # ── public API ───────────────────────────────────────────────────

    def estimate(self, image) -> dict:
        """
        Predict flood depth for a PIL Image via the Gemini endpoint.
        Returns the same shape as ReferenceDepthEstimator.estimate():
        depth_cm, confidence, visual_cues, label_guide, waterline_pct, water_coverage.
        Raises a GeminiError subclass on any failure.
        """
        if not self.available:
            raise GeminiUnavailableError("GEMINI_API_KEY is not set")

        url = GEMINI_ENDPOINT.format(model=self.model_name)
        try:
            resp = self._get_session().post(
                url,
                params={"key": self.api_key},
                json=self._build_payload(image),
                timeout=self.timeout_s,
            )
        except Exception as exc:
            raise GeminiRequestError(f"Gemini request failed: {exc}") from exc

        if resp.status_code != 200:
            raise GeminiRequestError(f"Gemini HTTP {resp.status_code}: {resp.text[:300]}")

        try:
            body = resp.json()
        except ValueError as exc:
            raise GeminiResponseError("Gemini returned non-JSON body") from exc

        text = self._response_text(body)
        return self._parse_result(text)

    @staticmethod
    def _response_text(body: dict) -> str:
        candidates = body.get("candidates") or []
        if not candidates:
            block = (body.get("promptFeedback") or {}).get("blockReason")
            detail = f" (blocked: {block})" if block else ""
            raise GeminiResponseError(f"Gemini response has no candidates{detail}")
        parts = ((candidates[0].get("content") or {}).get("parts")) or []
        text = "".join(p.get("text", "") for p in parts)
        if not text.strip():
            reason = candidates[0].get("finishReason", "unknown")
            raise GeminiResponseError(f"Gemini candidate has no text (finishReason={reason})")
        return text

    @staticmethod
    def _parse_result(text: str) -> dict:
        parsed = _extract_json(text)
        if parsed is not None and "depth_cm" in parsed:
            try:
                depth_cm = float(parsed["depth_cm"])
            except (TypeError, ValueError):
                raise GeminiResponseError(f"Non-numeric depth_cm in Gemini reply: {parsed.get('depth_cm')!r}")
            try:
                confidence = _clip(float(parsed.get("confidence", 0.6)), 0.0, 1.0)
            except (TypeError, ValueError):
                confidence = 0.6
            cues = parsed.get("visual_cues", [])
            visual_cues = [str(c) for c in cues] if isinstance(cues, list) else []
            try:
                water_coverage = _clip(float(parsed.get("water_coverage", 0.0)), 0.0, 1.0)
            except (TypeError, ValueError):
                water_coverage = 0.0
            try:
                waterline_pct = _clip(float(parsed.get("waterline_pct", 0.0)), 0.0, 100.0)
            except (TypeError, ValueError):
                waterline_pct = 0.0
            label_guide = str(parsed.get("label_guide", ""))
            reference_objects = _parse_reference_objects(parsed.get("reference_objects"))
            scene_analysis = str(parsed.get("scene_analysis", ""))
            depth_range_raw = parsed.get("depth_range_cm")
        else:
            # Last resort: the model ignored the JSON instruction — take the first number as cm.
            depth_cm_raw = _first_number(text)
            if depth_cm_raw is None:
                raise GeminiResponseError(f"Could not parse depth from Gemini reply: {text[:200]!r}")
            depth_cm = depth_cm_raw
            confidence = 0.5
            visual_cues = []
            water_coverage = 0.0
            waterline_pct = 0.0
            label_guide = "parsed from unstructured reply"
            reference_objects = []
            scene_analysis = ""
            depth_range_raw = None

        final_depth = round(_clip(depth_cm, 0.0, MAX_DEPTH_CM), 2)
        return {
            "depth_cm": final_depth,
            "depth_range_cm": _parse_depth_range(depth_range_raw, final_depth),
            "confidence": round(confidence, 4),
            "reference_objects": reference_objects,
            "scene_analysis": scene_analysis,
            "visual_cues": visual_cues,
            "label_guide": label_guide,
            "waterline_pct": waterline_pct,
            "water_coverage": water_coverage,
        }
