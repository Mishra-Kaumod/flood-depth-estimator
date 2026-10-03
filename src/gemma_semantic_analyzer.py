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
DEFAULT_GEMMA_MODEL = "gemma3:8b"


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
        timeout_seconds: float = 60.0,
        enabled: bool = True,
    ) -> None:
        self.model_name = model_name
        self.ollama_url = ollama_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.enabled = enabled
        self.model_name = self._resolve_model_name(model_name)

    def _resolve_model_name(self, preferred_name: str) -> str:
        if not self.enabled:
            return preferred_name
        try:
            req = urllib.request.Request(f"{self.ollama_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
            installed = []
            for item in payload.get("models", []) or []:
                name = item.get("name") or item.get("model")
                if name:
                    installed.append(str(name))
            if preferred_name in installed:
                return preferred_name
            for candidate in ("gemma3:4b", "gemma3:8b", "gemma3:latest"):
                if candidate in installed:
                    logger.info("Ollama Gemma auto-selected model '%s' (preferred '%s' unavailable)", candidate, preferred_name)
                    return candidate
            return preferred_name
        except Exception:
            return preferred_name

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
            # Encode full image and any detected-object crops to JPEG Base64 for Ollama multimodal API
            images_b64 = []
            full_b64 = self._image_to_base64(image_rgb)
            images_b64.append(full_b64)

            # Create tight crops for up to 6 detected reference objects to help Gemma focus
            if detected_objects:
                h, w = image_rgb.shape[:2]
                for obj in detected_objects[:6]:
                    bbox = None
                    # Support ReferenceObject dataclass or dict-like with bbox
                    if hasattr(obj, "bbox"):
                        bbox = getattr(obj, "bbox")
                    elif isinstance(obj, dict):
                        bbox = obj.get("bbox")
                    if not bbox:
                        continue
                    x1, y1, x2, y2 = bbox if len(bbox) == 4 else (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])
                    # Clamp
                    x1 = max(0, int(x1))
                    y1 = max(0, int(y1))
                    x2 = min(w, int(x2))
                    y2 = min(h, int(y2))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    crop = image_rgb[y1:y2, x1:x2]
                    try:
                        images_b64.append(self._image_to_base64(crop))
                    except Exception:
                        continue

            prompt = self._build_prompt(water_mask=water_mask, detected_objects=detected_objects, water_coverage_pct=water_coverage_pct, num_crops=len(images_b64)-1)

            # Query Ollama local API with multiple images
            raw_response = self._query_ollama(images_b64=images_b64, prompt=prompt)
            latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

            features = self._parse_and_validate_json(raw_response)
            # If Gemma returned semantic-only output (no numeric depths), attempt a short follow-up
            # using YOLO-supplied submersion ratios (if available) to prompt Gemma to produce numeric cm values.
            if features is None:
                # fallback: try to coerce numeric output via a second prompt only if detected_objects provided
                if detected_objects:
                    try:
                        follow_prompt = self._build_followup_prompt_with_yolo(detected_objects)
                        follow_raw = self._query_ollama(images_b64=images_b64, prompt=follow_prompt)
                        follow_features = self._parse_and_validate_json(follow_raw)
                        if follow_features is not None:
                            features = follow_features
                            # merge raw responses for debugging
                            raw_response = raw_response + "\n---FOLLOWUP---\n" + follow_raw
                    except Exception:
                        pass

            if features is None:
                return {
                    "status": "failed",
                    "reason": "Failed to parse structured JSON from Gemma response",
                    "raw_response": raw_response,
                    "features": None,
                    "latency_ms": latency_ms,
                }

            # Always include raw_response for debugging (even on success)
            result = {
                "status": "success",
                "model": self.model_name,
                "features": features,
                "raw_response": raw_response,
                "latency_ms": latency_ms,
            }
            return result

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
        num_crops: int = 0,
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

        crop_info = f"\nImages: image0=full_scene" + (f", crops=1..{num_crops}" if num_crops and num_crops > 0 else "")

        prompt = (
           "You are an expert hydrological vision assistant. Analyze the images for flood depth estimation features.\n\n"
           "Image convention: image0 is the full scene. If present, images 1..N are tight crops of detected reference objects for closer inspection. Use crops to estimate object submersion and waterline.\n\n"
           "Computer Vision detections (use as context, but verify visually against images):\n"
           f"{hint_block}\n\n"
           "CRITICAL INSTRUCTIONS (must follow):\n"
           "1. Use the full-scene image and object crops to judge visible waterline and submersion.\n"
           "2. Estimate the approximate_submersion_fraction for each visible reference object (0.0-1.0).\n"
           "3. Always include the keys 'per_object_depth_cm' and 'gemma_depth_cm' in the JSON output.\n"
           "   - If uncertain, use null instead of a wild guess.\n"
           "   - Use per-object depth estimates in centimeters only for objects you actually see.\n"
           "   - Set gemma_depth_cm to the scene-level flood depth most likely visible in the image.\n"
           "4. Provide gemma_depth_confidence (0.0-1.0) for the numeric estimate.\n"
           "5. Do not invent objects or hidden waterline geometry.\n"
           "6. If no reference object is visible, leave reference_object_type null and use null values where appropriate.\n"
           "7. Return ONLY a valid minified JSON object; no markdown fences or prose.\n\n"
           "Depth-policy: prefer conservative estimates grounded in visible waterline and object submersion. A wheel or ankle immersed in muddy floodwater can be 15-40cm, a lower leg or knee can be 30-80cm, and full leg/waist deep water can be 60-120cm. Do NOT force a deep value unless the image clearly supports it.\n\n"
           "Example output (MUST follow schema exactly):\n"
           "{\n"
           '  "water_present": true,\n'
           '  "waterline_visible": true,\n'
           '  "scene_type": "flooded_road",\n'
           '  "reference_object_type": "car",\n'
           '  "reference_object_visible": true,\n'
           '  "water_reaches_reference": true,\n'
           '  "approximate_submersion_fraction": 0.75,\n'
           '  "per_object_submersion": [0.8, 0.3],\n'
           '  "per_object_depth_cm": [60.0, 15.0],\n'
           '  "gemma_depth_cm": 60.0,\n'
           '  "gemma_depth_confidence": 0.92,\n'
           '  "reference_quality": "good",\n'
           '  "occlusion_level": "low",\n'
           '  "semantic_confidence": 0.90\n'
           "}\n\n"
           "Return only JSON and nothing else. Prefer truthful object-grounded depth over arbitrary cm guesses."
        )
        return prompt

    def _query_ollama(self, images_b64: list, prompt: str, timeout: Optional[float] = None) -> str:
        """Call local Ollama REST API via Python urllib with multiple images.

        timeout: if provided overrides self.timeout_seconds for this call.
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "images": images_b64,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.0,
                "top_p": 1.0,
                "num_ctx": 8192,
            },
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.ollama_url}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        call_timeout = float(timeout) if timeout is not None else float(self.timeout_seconds)
        with urllib.request.urlopen(req, timeout=call_timeout) as response:
            res_body = response.read().decode("utf-8")
            parsed_res = json.loads(res_body)
            # Ollama responses vary by version—capture full response string for debugging and return
            response_text = parsed_res.get("response") or parsed_res.get("text") or parsed_res.get("output") or ""
            return response_text

    def _build_followup_prompt_with_yolo(self, detected_objects: Optional[List[Any]]) -> str:
        """Create a concise follow-up prompt using YOLO-detected object submersion ratios to request numeric depths.

        The prompt instructs Gemma to map submersion fractions to per-object depths using the following typical heights:
        car=70cm, truck=90cm, bus=100cm, motorcycle=60cm, bicycle=60cm, person=75cm.
        Returns ONLY JSON with keys: per_object_depth_cm (list), gemma_depth_cm (number), gemma_depth_confidence (0.0-1.0).
        """
        lines = [
            "You previously returned semantic output but did not provide numeric depths. Provide a short JSON object mapping the provided submersion fractions to per-object depths (cm) and an overall gemma_depth_cm. Use the following typical heights (cm) for mapping:",
            "car=70, truck=90, bus=100, motorcycle=60, bicycle=60, person=75",
            "Return ONLY JSON with keys: per_object_depth_cm (array of numbers or null), gemma_depth_cm (number or null), gemma_depth_confidence (0.0-1.0). If unsure, use null for values.",
            "Do not include any extra text.",
            "Objects:",
        ]
        for i, obj in enumerate(detected_objects[:8]):
            try:
                label = getattr(obj, "label", None) if not isinstance(obj, dict) else obj.get("label")
                sub = getattr(obj, "water_submersion_ratio", None) if not isinstance(obj, dict) else obj.get("water_submersion_ratio")
                label = str(label).lower() if label is not None else "unknown"
                sub_val = f"{float(sub):.2f}" if sub is not None else "null"
            except Exception:
                label = "unknown"
                sub_val = "null"
            lines.append(f" - obj{i}: label={label}, submersion={sub_val}")

        lines.append("Example output:\n{\n  \"per_object_depth_cm\": [60.0, 15.0],\n  \"gemma_depth_cm\": 60.0,\n  \"gemma_depth_confidence\": 0.85\n}")
        return "\n".join(lines)

    @staticmethod
    def _coerce_bool(value: Any, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "y", "1", "water_present", "flooded", "visible"}:
                return True
            if lowered in {"false", "no", "n", "0", "dry", "not_visible", "clear"}:
                return False
        return bool(value)

    @staticmethod
    def _heuristic_depth_from_submersion(submersion: Optional[float], scene_type: str) -> Optional[float]:
        if submersion is None:
            return None
        try:
            submersion = float(submersion)
        except (TypeError, ValueError):
            return None
        if submersion < 0.0:
            return None

        scene = (scene_type or "unknown").lower()
        ankle_like = any(token in scene for token in ("ankle", "foot", "knee", "leg", "puddle", "wet", "shallow"))
        if submersion <= 0.15:
            return 15.0 if ankle_like else 20.0
        if submersion <= 0.35:
            return 30.0 if ankle_like else 35.0
        if submersion <= 0.60:
            return 45.0 if ankle_like else 55.0
        if submersion <= 0.80:
            return 60.0
        return 75.0

    def _parse_and_validate_json(self, raw_text: str) -> Optional[Dict[str, Any]]:
        """Robustly extract and validate JSON response from Gemma output.

        This function first attempts strict JSON parsing. If that fails, it uses
        a lightweight numeric fallback parser to extract gemma_depth_cm,
        per_object_depth_cm, and submersion fractions from freeform text so the
        pipeline can still use best-effort numeric cues.
        """
        if not raw_text or not raw_text.strip():
            return None

        clean_text = raw_text.strip()
        json_match = re.search(r"\{.*\}", clean_text, re.DOTALL)
        if json_match:
            clean_text = json_match.group(0)

        def normalize_semantic(validated: Dict[str, Any]) -> Dict[str, Any]:
            water_present = self._coerce_bool(validated.get("water_present"), default=False)
            scene_type = str(validated.get("scene_type", "unknown") or "unknown").lower()
            submersion = validated.get("approximate_submersion_fraction")
            try:
                submersion = float(submersion) if submersion is not None else None
            except (TypeError, ValueError):
                submersion = None
            ref_visible = self._coerce_bool(validated.get("reference_object_visible"), default=False)
            water_reaches_ref = self._coerce_bool(validated.get("water_reaches_reference"), default=False)
            waterline_visible = self._coerce_bool(validated.get("waterline_visible"), default=False)

            flood_hints = (
                "flood" in scene_type
                or "water" in scene_type
                or "puddle" in scene_type
                or "submerged" in scene_type
                or "submersion" in scene_type
                or "road" in scene_type
            )
            ref_or_submersion_evidence = (
                water_reaches_ref
                or waterline_visible
                or ref_visible
                or flood_hints
                or (submersion is not None and submersion >= 0.20)
            )
            if not water_present and ref_or_submersion_evidence:
                water_present = True
            validated["water_present"] = water_present

            if not scene_type or scene_type == "unknown":
                validated["scene_type"] = "flooded_road" if water_present else "unknown"

            if validated.get("approximate_submersion_fraction") is None:
                per_obj = validated.get("per_object_submersion")
                if isinstance(per_obj, list) and per_obj:
                    try:
                        nums = [float(x) for x in per_obj if x is not None]
                        if nums:
                            validated["approximate_submersion_fraction"] = max(nums)
                    except Exception:
                        pass

            if validated.get("gemma_depth_cm") is None:
                pod = validated.get("per_object_depth_cm")
                if isinstance(pod, list) and pod:
                    try:
                        nums = [float(x) for x in pod if x is not None]
                        if nums:
                            validated["gemma_depth_cm"] = max(nums)
                    except Exception:
                        pass
                if validated.get("gemma_depth_cm") is None and validated.get("approximate_submersion_fraction") is not None:
                    validated["gemma_depth_cm"] = self._heuristic_depth_from_submersion(
                        validated.get("approximate_submersion_fraction"),
                        str(validated.get("scene_type", "unknown") or "unknown"),
                    )

            if validated.get("gemma_depth_confidence") is None:
                confidence = 0.5
                if water_present:
                    confidence += 0.2
                if water_reaches_ref or waterline_visible:
                    confidence += 0.2
                if ref_visible:
                    confidence += 0.1
                validated["gemma_depth_confidence"] = round(min(max(confidence, 0.0), 0.99), 2)

            validated["semantic_confidence"] = validated.get("gemma_depth_confidence", validated.get("semantic_confidence", 0.5))
            return validated

        try:
            data = json.loads(clean_text)
            if not isinstance(data, dict):
                return None

            validated = {
                "water_present": self._coerce_bool(data.get("water_present"), default=False),
                "waterline_visible": self._coerce_bool(data.get("waterline_visible"), default=False),
                "scene_type": str(data.get("scene_type", "unknown") or "unknown").lower(),
                "reference_object_type": str(data["reference_object_type"]) if data.get("reference_object_type") else None,
                "reference_object_visible": self._coerce_bool(data.get("reference_object_visible"), default=False),
                "water_reaches_reference": self._coerce_bool(data.get("water_reaches_reference"), default=False),
                "approximate_submersion_fraction": (
                    float(data["approximate_submersion_fraction"]) if data.get("approximate_submersion_fraction") is not None else None
                ),
                "per_object_submersion": data.get("per_object_submersion"),
                "per_object_depth_cm": data.get("per_object_depth_cm"),
                "gemma_depth_cm": float(data.get("gemma_depth_cm")) if data.get("gemma_depth_cm") is not None else None,
                "gemma_depth_confidence": (
                    round(float(data.get("gemma_depth_confidence")), 2) if data.get("gemma_depth_confidence") is not None else None
                ),
                "reference_quality": str(data.get("reference_quality", "uncertain")).lower(),
                "occlusion_level": str(data.get("occlusion_level", "low")).lower(),
                "semantic_confidence": round(float(data.get("semantic_confidence", 0.5)), 2),
            }

            per_obj = validated.get("per_object_submersion")
            if isinstance(per_obj, list) and per_obj:
                try:
                    nums = [float(x) for x in per_obj if x is not None]
                    if nums:
                        validated["approximate_submersion_fraction"] = max(nums)
                except Exception:
                    pass

            pod = validated.get("per_object_depth_cm")
            if validated.get("gemma_depth_cm") is None and isinstance(pod, list) and pod:
                try:
                    nums = [float(x) for x in pod if x is not None]
                    if nums:
                        validated["gemma_depth_cm"] = max(nums)
                except Exception:
                    pass

            updated = normalize_semantic(validated)
            return updated

        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.debug(f"JSON validation failed for Gemma response: {exc}")
            try:
                gd_match = re.search(r"gemma_depth_cm\"?\s*[:=]\s*([0-9]+\.?[0-9]*)", clean_text, re.IGNORECASE)
                pod_matches = re.findall(r"per_object_depth_cm\"?\s*[:=]\s*\[([^\]]+)\]", clean_text, re.IGNORECASE)
                per_obj_depths = None
                if pod_matches:
                    try:
                        nums = re.findall(r"[0-9]+\.?[0-9]*", pod_matches[0])
                        per_obj_depths = [float(x) for x in nums] if nums else None
                    except Exception:
                        per_obj_depths = None

                pos_matches = re.findall(r"per_object_submersion\"?\s*[:=]\s*\[([^\]]+)\]", clean_text, re.IGNORECASE)
                per_obj_sub = None
                if pos_matches:
                    try:
                        nums = re.findall(r"[0-9]+\.?[0-9]*", pos_matches[0])
                        per_obj_sub = [float(x) for x in nums] if nums else None
                    except Exception:
                        per_obj_sub = None

                gemma_depth = float(gd_match.group(1)) if gd_match else None
                gc_match = re.search(r"gemma_depth_confidence\"?\s*[:=]\s*([0-9]+\.?[0-9]*)", clean_text, re.IGNORECASE)
                gemma_conf = float(gc_match.group(1)) if gc_match else None

                validated = {
                    "water_present": False,
                    "waterline_visible": False,
                    "scene_type": "unknown",
                    "reference_object_type": None,
                    "reference_object_visible": False,
                    "water_reaches_reference": False,
                    "approximate_submersion_fraction": max(per_obj_sub) if isinstance(per_obj_sub, list) and per_obj_sub else None,
                    "per_object_submersion": per_obj_sub,
                    "per_object_depth_cm": per_obj_depths,
                    "gemma_depth_cm": gemma_depth,
                    "gemma_depth_confidence": round(float(gemma_conf), 2) if gemma_conf is not None else None,
                    "reference_quality": "uncertain",
                    "occlusion_level": "low",
                    "semantic_confidence": round(float(gemma_conf), 2) if gemma_conf is not None else 0.5,
                }

                if validated.get("gemma_depth_cm") is None and isinstance(per_obj_depths, list) and per_obj_depths:
                    validated["gemma_depth_cm"] = max(per_obj_depths)
                if validated.get("gemma_depth_cm") is None and validated.get("approximate_submersion_fraction") is not None:
                    validated["gemma_depth_cm"] = self._heuristic_depth_from_submersion(
                        validated.get("approximate_submersion_fraction"), "unknown"
                    )

                if validated.get("gemma_depth_cm") is None and not validated.get("per_object_submersion"):
                    return None

                validated["_parsed_with_fallback"] = True
                return normalize_semantic(validated)
            except Exception as exc2:
                logger.debug(f"Fallback numeric parser failed: {exc2}")
                return None
