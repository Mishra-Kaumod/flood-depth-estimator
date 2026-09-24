"""
Local Gemma Semantic Analyzer for Flood Scene Understanding.

Communicates with local Ollama runtime to extract structured semantic features
(water presence, waterline visibility, reference object submersion, scene type)
without making external API calls or predicting centimeters directly.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_GEMMA_MODEL = "gemma3:4b"


class GemmaSemanticAnalyzer:
    """
    Local Gemma Semantic Feature Extractor using Ollama runtime.
    
    Extracts high-level qualitative visual features from flood images
    to enrich computer vision pipeline traces and future fusion models.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_GEMMA_MODEL,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        timeout_seconds: float = 35.0,
        enabled: bool = True,
    ) -> None:
        self.model_name = model_name
        self.ollama_url = ollama_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.enabled = enabled

    def is_available(self) -> bool:
        """Check if local Ollama server is running and accessible."""
        if not self.enabled:
            return False
        try:
            req = urllib.request.Request(f"{self.ollama_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                return resp.status == 200
        except Exception:
            return False

    def analyze(
        self,
        image_rgb: np.ndarray,
        water_mask: Optional[np.ndarray] = None,
        detected_objects: Optional[List[Any]] = None,
        water_coverage_pct: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Analyze scene using local Gemma model via Ollama.

        Returns structured JSON features or None if skipped/failed.
        """
        if not self.enabled:
            return {
                "status": "skipped",
                "reason": "Gemma semantic analyzer is disabled in configuration",
                "features": None,
            }

        start_time = time.perf_counter()

        # Check local Ollama server status
        if not self.is_available():
            logger.warning(
                f"Ollama service unavailable at {self.ollama_url}. Skipping Gemma semantic analysis."
            )
            return {
                "status": "skipped",
                "reason": f"Ollama service unavailable at {self.ollama_url}",
                "features": None,
                "latency_ms": round((time.perf_counter() - start_time) * 1000.0, 2),
            }

        try:
            # Encode image to JPEG Base64 for Ollama multimodal API
            image_b64 = self._image_to_base64(image_rgb)
            prompt = self._build_prompt(water_mask=water_mask, detected_objects=detected_objects, water_coverage_pct=water_coverage_pct)

            # Query Ollama local API
            raw_response = self._query_ollama(image_b64=image_b64, prompt=prompt)
            latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

            features = self._parse_and_validate_json(raw_response)
            if features is None:
                return {
                    "status": "failed",
                    "reason": "Failed to parse structured JSON from Gemma response",
                    "raw_response": raw_response,
                    "features": None,
                    "latency_ms": latency_ms,
                }

            return {
                "status": "success",
                "model": self.model_name,
                "features": features,
                "latency_ms": latency_ms,
            }

        except Exception as exc:
            latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            logger.warning(f"Gemma semantic analysis error: {exc}")
            return {
                "status": "failed",
                "reason": str(exc),
                "features": None,
                "latency_ms": latency_ms,
            }

    def _image_to_base64(self, image_rgb: np.ndarray, max_dim: int = 768) -> str:
        """Convert numpy RGB image array to base64 JPEG string."""
        pil_img = Image.fromarray(image_rgb)
        w, h = pil_img.size
        if max(w, h) > max_dim:
            scale = max_dim / float(max(w, h))
            pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _build_prompt(
        self,
        water_mask: Optional[np.ndarray] = None,
        detected_objects: Optional[List[Any]] = None,
        water_coverage_pct: Optional[float] = None,
    ) -> str:
        """Build structured prompt passing computer vision detections as contextual hints."""
        cv_hints = []
        if water_coverage_pct is not None:
            cv_hints.append(f"- SegFormer detected water coverage: {water_coverage_pct:.1f}%")

        if detected_objects:
            obj_descriptions = []
            for obj in detected_objects[:5]:
                label = getattr(obj, "label", str(obj))
                submersion = getattr(obj, "water_submersion_ratio", None)
                conf = getattr(obj, "confidence", None)
                sub_str = f", estimated submersion={submersion:.2f}" if submersion is not None else ""
                conf_str = f" (conf={conf:.2f})" if conf is not None else ""
                obj_descriptions.append(f"  * {label}{conf_str}{sub_str}")
            if obj_descriptions:
                cv_hints.append("- YOLO detected reference objects:\n" + "\n".join(obj_descriptions))

        hint_block = "\n".join(cv_hints) if cv_hints else "None provided."

        prompt = (
            "You are an expert hydrological vision assistant. Analyze this image for flood depth estimation features.\n\n"
            "Computer Vision detections (use as context, but verify visually against image):\n"
            f"{hint_block}\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Do NOT guess exact depth in centimeters.\n"
            "2. Do NOT invent objects that are not visible in the image.\n"
            "3. If no reference object is visible, set reference_object_type to null.\n"
            "4. Return ONLY a valid minified JSON object with these exact keys:\n"
            "{\n"
            '  "water_present": boolean,\n'
            '  "waterline_visible": boolean,\n'
            '  "scene_type": "dry_land" | "wet_puddle" | "flooded_road" | "flooded_indoor" | "unknown",\n'
            '  "reference_object_type": string or null,\n'
            '  "reference_object_visible": boolean,\n'
            '  "water_reaches_reference": boolean,\n'
            '  "approximate_submersion_fraction": float (0.0 to 1.0) or null,\n'
            '  "reference_quality": "good" | "uncertain" | "poor",\n'
            '  "occlusion_level": "low" | "moderate" | "high",\n'
            '  "semantic_confidence": float (0.0 to 1.0)\n'
            "}"
        )
        return prompt

    def _query_ollama(self, image_b64: str, prompt: str) -> str:
        """Call local Ollama REST API via Python urllib."""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
            },
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.ollama_url}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
            res_body = response.read().decode("utf-8")
            parsed_res = json.loads(res_body)
            return parsed_res.get("response", "")

    def _parse_and_validate_json(self, raw_text: str) -> Optional[Dict[str, Any]]:
        """Robustly extract and validate JSON response from Gemma output."""
        if not raw_text or not raw_text.strip():
            return None

        clean_text = raw_text.strip()
        # Extract JSON substring if wrapped in markdown codeblocks or text
        json_match = re.search(r"\{.*\}", clean_text, re.DOTALL)
        if json_match:
            clean_text = json_match.group(0)

        try:
            data = json.loads(clean_text)
            if not isinstance(data, dict):
                return None

            # Enforce expected schema and sanitize types
            validated = {
                "water_present": bool(data.get("water_present", False)),
                "waterline_visible": bool(data.get("waterline_visible", False)),
                "scene_type": str(data.get("scene_type", "unknown")).lower(),
                "reference_object_type": str(data["reference_object_type"]) if data.get("reference_object_type") else None,
                "reference_object_visible": bool(data.get("reference_object_visible", False)),
                "water_reaches_reference": bool(data.get("water_reaches_reference", False)),
                "approximate_submersion_fraction": (
                    float(data["approximate_submersion_fraction"])
                    if data.get("approximate_submersion_fraction") is not None
                    else None
                ),
                "reference_quality": str(data.get("reference_quality", "uncertain")).lower(),
                "occlusion_level": str(data.get("occlusion_level", "low")).lower(),
                "semantic_confidence": round(float(data.get("semantic_confidence", 0.5)), 2),
            }
            return validated

        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.debug(f"JSON validation failed for Gemma response: {exc}")
            return None
