"""
LLM judge integration for flood inference validation.

This module calls the Google Generative Language API to validate and optionally
recommend corrections for flood depth outputs.
"""

from __future__ import annotations

import json
import base64
import mimetypes
import os
import urllib.error
import urllib.request
from typing import Any, Dict

DEFAULT_MODEL = "gemini-flash-latest"
DEFAULT_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{DEFAULT_MODEL}:generateContent"
)


class LLMJudge:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.enabled = bool(config.get("enabled", False))
        self.provider = str(config.get("provider", "google")).lower()
        self.model = str(config.get("model", DEFAULT_MODEL))
        self.endpoint = str(config.get("endpoint", DEFAULT_ENDPOINT))
        self.temperature = float(config.get("temperature", 0.0))
        self.max_output_tokens = int(config.get("max_output_tokens", 2048))
        self.apply_corrections = bool(config.get("apply_corrections", False))

        self.api_key = config.get("google_api_key") or os.getenv("GOOGLE_API_KEY")
        if self.enabled and not self.api_key:
            raise ValueError(
                "LLM judge enabled but GOOGLE_API_KEY is missing in the environment"
            )

        if self.enabled and self.provider != "google":
            raise ValueError("Only Google provider is supported by this LLM judge module")

    def judge(
        self,
        prediction: Dict[str, Any],
        image_bytes: bytes | None = None,
        filename: str | None = None,
    ) -> Dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}

        request_payload = self._build_payload(
            prediction,
            image_bytes=image_bytes,
            filename=filename,
        )
        response_text = self._call_google_api(request_payload)
        parsed = self._parse_json_response(response_text)

        if parsed is None:
            return {
                "enabled": True,
                "prediction_correct": None,
                "plausible": None,
                "recommended_depth_cm": None,
                "recommended_severity": None,
                "reason": "Could not parse judge response",
                "raw_response": response_text,
                "parse_failed": True,
            }

        parsed["prediction_correct"] = parsed.get("prediction_correct", parsed.get("plausible"))
        parsed["enabled"] = True
        parsed["raw_response"] = response_text
        parsed["parse_failed"] = False
        return parsed

    def _build_payload(
        self,
        prediction: Dict[str, Any],
        image_bytes: bytes | None = None,
        filename: str | None = None,
    ) -> Dict[str, Any]:
        prompt = self._build_prompt(prediction, has_image=bool(image_bytes))
        parts: list[dict[str, Any]] = [{"text": prompt}]
        if image_bytes:
            parts.append(
                {
                    "inline_data": {
                        "mime_type": self._guess_mime_type(filename),
                        "data": base64.b64encode(image_bytes).decode("ascii"),
                    }
                }
            )
        return {
            "contents": [
                {
                    "parts": parts,
                }
            ],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_output_tokens,
                "responseMimeType": "application/json",
            },
        }

    def _guess_mime_type(self, filename: str | None) -> str:
        if filename:
            mime_type, _ = mimetypes.guess_type(filename)
            if mime_type and mime_type.startswith("image/"):
                return mime_type
        return "image/jpeg"

    def _build_prompt(self, prediction: Dict[str, Any], has_image: bool = False) -> str:
        model_context = {
            "confidence_pct": prediction.get("confidence_pct"),
            "water_coverage_pct": prediction.get("water_coverage_pct"),
            "reference_count": prediction.get("reference_count"),
            "reference_depth_cm": prediction.get("reference_depth_cm"),
            "waterline_pct": prediction.get("waterline_pct"),
            "dense_depth_p90": prediction.get("dense_depth_p90"),
            "near_water_coverage_pct": prediction.get("near_water_coverage_pct"),
            "mid_water_coverage_pct": prediction.get("mid_water_coverage_pct"),
            "far_water_coverage_pct": prediction.get("far_water_coverage_pct"),
            "far_water_only": prediction.get("far_water_only"),
            "immediate_risk": prediction.get("immediate_risk"),
            "mask_quality_warning": prediction.get("mask_quality_warning"),
            "visual_cues": prediction.get("visual_cues", [])[:6],
        }
        pipeline_output = {
            "predicted_depth_cm": prediction.get("depth_cm"),
            "predicted_severity": prediction.get("severity_label"),
            "action": prediction.get("action_trigger"),
        }
        image_instruction = (
            "Inspect the attached image first. Estimate flood depth from visible scene evidence before looking at the pipeline output."
            if has_image
            else "Use only these CV numbers; no image is attached."
        )
        return (
            "You are an independent flood-depth visual reviewer. "
            + image_instruction
            + " Do not automatically trust the model depth. Water coverage alone is not depth. "
            "If there is no clear scale reference such as vehicle wheels, people legs, curb, step, wall, pole, or road edge, mark low visual confidence and prefer a depth range or human review. "
            "For far-road water with dry/visible near road and no close reference, cap recommended depth to a shallow or advisory estimate unless obvious deep submersion is visible. "
            "Compare your independent estimate against the pipeline only after forming the visual estimate. "
            "Return ONLY minified JSON with these exact keys: "
            "prediction_correct, visual_depth_estimate_cm, visual_depth_range_cm, visual_confidence, recommended_depth_cm, recommended_severity, final_depth_cm, final_severity, review_required, reason. "
            "prediction_correct and review_required must be true or false. Depth fields must be numbers in cm. "
            "visual_depth_range_cm must be a short string like '10-25'. visual_confidence must be low, medium, or high. "
            "Use severity NORMAL for 0 cm, ADVISORY for shallow water, WARNING for moderate hazardous road flooding, ALERT or CRITICAL only for deep vehicle/person submersion. "
            "Model context: "
            + json.dumps(model_context, separators=(",", ":"), default=str)
            + " Pipeline output to verify: "
            + json.dumps(pipeline_output, separators=(",", ":"), default=str)
        )

    def _call_google_api(self, payload: Dict[str, Any]) -> str:
        url = self.endpoint
        if "{model}" in url:
            url = url.format(model=self.model)

        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": self.api_key,
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            return json.dumps({"error": str(exc), "body": body})
        except urllib.error.URLError as exc:
            return json.dumps({"error": str(exc)})

    def _parse_json_response(self, response_text: str) -> dict[str, Any] | None:
        parsed_payload = None
        try:
            parsed_payload = json.loads(response_text)
        except json.JSONDecodeError:
            parsed_payload = None

        if isinstance(parsed_payload, dict):
            for key in ("output", "candidates", "responses", "response"):
                contents = parsed_payload.get(key)
                text = self._extract_text_from_output(contents)
                if text:
                    result = self._parse_json_text(text)
                    if result:
                        return result

            if set(parsed_payload.keys()) >= {"prediction_correct", "recommended_depth_cm", "recommended_severity", "final_depth_cm", "final_severity", "reason"}:
                return parsed_payload
            if set(parsed_payload.keys()) >= {"plausible", "recommended_depth_cm", "recommended_severity", "final_depth_cm", "final_severity", "reason"}:
                return parsed_payload

            for key in ("outputText", "text", "content"):
                if key in parsed_payload:
                    result = self._parse_json_text(str(parsed_payload[key]))
                    if result:
                        return result

        result = self._parse_json_text(response_text)
        if result:
            return result

        return self._parse_key_value_response(response_text)

    def _extract_text_from_output(self, contents: Any) -> str | None:
        if isinstance(contents, str):
            return contents
        if isinstance(contents, dict):
            return (
                self._extract_text_from_output(contents.get("output"))
                or self._extract_text_from_output(contents.get("text"))
                or self._extract_text_from_output(contents.get("content"))
                or self._extract_text_from_output(contents.get("parts"))
            )
        if isinstance(contents, list):
            parts = []
            for item in contents:
                if isinstance(item, dict):
                    parts.append(
                        self._extract_text_from_output(
                            item.get("content")
                            or item.get("text")
                            or item.get("output")
                            or item.get("parts")
                        )
                        or ""
                    )
                elif isinstance(item, str):
                    parts.append(item)
            return "".join(parts)
        return None

    def _parse_key_value_response(self, text: str) -> dict[str, Any] | None:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return None

        result = {
            "prediction_correct": None,
            "plausible": None,
            "recommended_depth_cm": None,
            "recommended_severity": None,
            "final_depth_cm": None,
            "final_severity": None,
            "reason": None,
        }

        for line in lines:
            lower_line = line.lower()
            if ("prediction_correct" in lower_line or "plausible" in lower_line) and ":" in line:
                value = line.split(":", 1)[1].strip()
                if value.lower() in {"true", "yes", "y"}:
                    result["prediction_correct"] = True
                    result["plausible"] = True
                elif value.lower() in {"false", "no", "n"}:
                    result["prediction_correct"] = False
                    result["plausible"] = False
            elif "recommended_depth" in lower_line and ":" in line:
                value = line.split(":", 1)[1].strip().rstrip(".,")
                try:
                    result["recommended_depth_cm"] = float(value)
                except ValueError:
                    pass
            elif "final_depth" in lower_line and ":" in line:
                value = line.split(":", 1)[1].strip().rstrip(".,")
                try:
                    result["final_depth_cm"] = float(value)
                except ValueError:
                    pass
            elif "recommended_severity" in lower_line and ":" in line:
                result["recommended_severity"] = line.split(":", 1)[1].strip().strip('"')
            elif "final_severity" in lower_line and ":" in line:
                result["final_severity"] = line.split(":", 1)[1].strip().strip('"')
            elif result["reason"] is None:
                if "reason" in lower_line and ":" in line:
                    result["reason"] = line.split(":", 1)[1].strip().strip('"')

        if result["prediction_correct"] is not None or result["plausible"] is not None:
            if result["recommended_depth_cm"] is None:
                result["recommended_depth_cm"] = 0.0
            if result["recommended_severity"] is None:
                result["recommended_severity"] = "unknown"
            if result["reason"] is None:
                result["reason"] = "No reason provided"
            return result

        return None

    def _parse_json_text(self, text: str) -> dict[str, Any] | None:
        text = text.strip()
        if not text:
            return None
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
        try:
            parsed = json.loads(text)
            if (
                "recommended_depth_cm" in parsed
                and "recommended_severity" in parsed
                and "reason" in parsed
                and ("prediction_correct" in parsed or "plausible" in parsed)
            ):
                return parsed
        except json.JSONDecodeError:
            return None
        return None


