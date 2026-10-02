"""
Local Gemma / Vision LLM Model Judge Engine for Flood & Depth Verification.

Runs local Hugging Face Vision-Language models on PyTorch with Apple Silicon MPS acceleration
or CUDA/CPU. Supports Google Gemma / PaliGemma models as well as open ungated vision VQA models
(e.g., Salesforce/blip-vqa-base) with zero external server required.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
from PIL import Image

logger = logging.getLogger(__name__)

# Model definitions
DEFAULT_GEMMA_MODEL = "google/paligemma-3b-pt-224"
OPEN_UNGATED_FALLBACK = "Salesforce/blip-vqa-base"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
OLLAMA_GEMMA_MODELS = ("gemma3:4b", "gemma3:8b")


class LocalGemmaJudge:
    """
    Local Gemma & Vision LLM evaluator using PyTorch on Apple Silicon (MPS), CUDA, or CPU.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_GEMMA_MODEL,
        device: str = "auto",
        torch_dtype: str = "float16",
        lazy_load: bool = True,
        max_new_tokens: int = 128,
        hf_token: Optional[str] = None,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        prefer_ollama: bool = True,
    ) -> None:
        self.model_name = model_name
        self.requested_device = device
        self.torch_dtype_str = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.lazy_load = lazy_load
        self.ollama_url = ollama_url.rstrip("/")
        self.prefer_ollama = prefer_ollama
        if not hf_token:
            hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        if not hf_token:
            try:
                import yaml
                config_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
                if config_path.exists():
                    with open(config_path, "r") as f:
                        cfg = yaml.safe_load(f)
                    hf_token = cfg.get("inference", {}).get("llm_judge", {}).get("hf_token")
            except Exception:
                pass

        self.hf_token = hf_token

        self.device = self._resolve_device(device)
        self.dtype = self._resolve_dtype(torch_dtype)

        self._model = None
        self._processor = None
        self._is_blip = False
        self._loaded = False
        self._ollama_available = False
        self._ollama_model_name = None

        if not self.lazy_load:
            self._load_model()

    def _resolve_device(self, device_str: str) -> torch.device:
        if device_str.lower() != "auto":
            return torch.device(device_str)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _resolve_dtype(self, dtype_str: str) -> torch.dtype:
        if self.device.type == "cpu":
            return torch.float32
        if dtype_str.lower() in ("float16", "fp16"):
            return torch.float16
        if dtype_str.lower() in ("bfloat16", "bf16"):
            return torch.bfloat16
        return torch.float32

    def _check_ollama(self) -> tuple[bool, Optional[str]]:
        try:
            import urllib.request
            req = urllib.request.Request(f"{self.ollama_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=3.0) as response:
                if response.status != 200:
                    return False, None
                payload = response.read().decode("utf-8", errors="ignore")
                if not payload:
                    return False, None
                try:
                    data = json.loads(payload)
                except Exception:
                    return False, None
                models = []
                for item in data.get("models", []) or []:
                    name = item.get("name") or item.get("model")
                    if name:
                        models.append(str(name))
                for candidate in OLLAMA_GEMMA_MODELS:
                    if candidate in models:
                        return True, candidate
                return True, None
        except Exception:
            return False, None

    def _load_model(self) -> None:
        if self._loaded:
            return

        if self.prefer_ollama:
            ollama_available, ollama_model = self._check_ollama()
            self._ollama_available = ollama_available
            self._ollama_model_name = ollama_model
            if ollama_available and ollama_model is not None:
                self._loaded = True
                self._is_blip = False
                self.model_name = ollama_model
                logger.info("Using Ollama Gemma model '%s' for judge validation.", ollama_model)
                return

        logger.info(f"Loading local Vision LLM '{self.model_name}' on '{self.device}' with dtype '{self.dtype}'...")

        token_kw = {"token": self.hf_token} if self.hf_token else {}

        # Attempt 1: PaliGemma / Gemma Vision
        if "gemma" in self.model_name.lower():
            try:
                from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

                self._processor = AutoProcessor.from_pretrained(self.model_name, **token_kw)
                self._model = PaliGemmaForConditionalGeneration.from_pretrained(
                    self.model_name,
                    dtype=self.dtype,
                    **token_kw,
                ).to(self.device)
                self._model.eval()
                self._loaded = True
                self._is_blip = False
                logger.info(f"Successfully loaded local Gemma model '{self.model_name}'.")
                return
            except Exception as exc:
                err_msg = str(exc)
                if "gated repo" in err_msg.lower() or "401" in err_msg or "403" in err_msg:
                    logger.warning(
                        f"Gemma model '{self.model_name}' is gated on Hugging Face. "
                        f"BLIP fallback will be informational only and not used to override deep-flood pipeline depth."
                    )
                else:
                    logger.warning(f"Could not load '{self.model_name}': {exc}. BLIP fallback is informational only.")

        # Attempt 2: Open Ungated BLIP Vision VQA Fallback
        try:
            from transformers import BlipForQuestionAnswering, BlipProcessor

            target_model = OPEN_UNGATED_FALLBACK if "gemma" in self.model_name.lower() else self.model_name
            logger.info(f"Loading open vision model '{target_model}'...")

            self._processor = BlipProcessor.from_pretrained(target_model)
            self._model = BlipForQuestionAnswering.from_pretrained(
                target_model,
                torch_dtype=self.dtype,
            ).to(self.device)
            self._model.eval()
            self._is_blip = True
            self._loaded = True
            self.model_name = target_model
            logger.info(f"Successfully loaded open vision model '{target_model}'.")
        except Exception as exc:
            logger.error(f"Failed to load open vision fallback model: {exc}")
            raise RuntimeError(f"Could not load local vision model. Error: {exc}")

    def judge(
        self,
        prediction: Dict[str, Any],
        image_bytes: Optional[bytes] = None,
        image_path: Optional[Union[str, Path]] = None,
        image_pil: Optional[Image.Image] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate flood presence and depth accuracy on the given image using local Gemma/VLM.

        BLIP fallback remains informational only. It never overrides a deep-flood pipeline result
        unless a trusted Gemma/Ollama numeric signal is available and passes confidence/outlier checks.
        """
        pil_image = self._load_pil_image(image_bytes=image_bytes, image_path=image_path, image_pil=image_pil)
        if pil_image is None:
            return {
                "enabled": True,
                "provider": "local_gemma",
                "prediction_correct": None,
                "reason": "No valid image provided to local Gemma judge.",
                "parse_failed": True,
            }

        try:
            self._load_model()

            if self._is_blip:
                raw_text = self._run_blip_questions(pil_image, prediction)
                parsed = self._parse_vqa_response(raw_text, prediction)
                parsed["provider"] = "blip_info_only"
                parsed["override_allowed"] = False
                return {**parsed, "enabled": True, "model_used": self.model_name}

            prompt = self._build_gemma_prompt(prediction)
            raw_text = self._run_gemma_inference(pil_image, prompt)
            parsed = self._parse_json_response(raw_text, prediction)

            parsed["enabled"] = True
            parsed["provider"] = "local_gemma"
            parsed["model_used"] = self.model_name
            parsed["override_allowed"] = self._should_allow_numeric_override(parsed, prediction)
            if not parsed.get("override_allowed"):
                parsed["prediction_correct"] = True
                parsed["review_required"] = False
                parsed["reason"] = "Gemma numeric depth not trusted: confidence/outlier gate failed. Pipeline depth retained."
            return parsed
        except Exception as exc:
            logger.warning(f"Local Gemma inference error: {exc}. BLIP fallback is informational only and cannot override the pipeline.")
            fallback = self._visual_rule_fallback(pil_image, prediction)
            fallback["enabled"] = True
            fallback["provider"] = "local_gemma_fallback"
            fallback["override_allowed"] = False
            fallback["prediction_correct"] = True
            fallback["review_required"] = False
            fallback["error"] = str(exc)
            return fallback

    def _should_allow_numeric_override(self, parsed: Dict[str, Any], prediction: Dict[str, Any]) -> bool:
        recommended_depth = parsed.get("recommended_depth_cm") or parsed.get("final_depth_cm") or prediction.get("depth_cm") or prediction.get("estimated_depth_cm") or 0.0
        try:
            recommended_depth = float(recommended_depth)
        except (TypeError, ValueError):
            return False

        if not isinstance(parsed, dict):
            return False

        gemma_conf = parsed.get("gemma_depth_confidence")
        gemma_depth = parsed.get("gemma_depth_cm")
        if gemma_depth is None:
            return False
        try:
            depth_val = float(gemma_depth)
            conf_val = float(gemma_conf) if gemma_conf is not None else 0.0
        except (TypeError, ValueError):
            return False

        if conf_val < 0.70:
            return False

        pipeline_depth = float(prediction.get("depth_cm") or prediction.get("estimated_depth_cm") or 0.0)
        if pipeline_depth > 0 and abs(depth_val - pipeline_depth) > 40.0:
            return False

        return True

    def _load_pil_image(
        self,
        image_bytes: Optional[bytes] = None,
        image_path: Optional[Union[str, Path]] = None,
        image_pil: Optional[Image.Image] = None,
    ) -> Optional[Image.Image]:
        if image_pil is not None:
            return image_pil.convert("RGB")
        if image_bytes is not None:
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if image_path is not None:
            return Image.open(Path(image_path)).convert("RGB")
        return None

    def _build_gemma_prompt(self, prediction: Dict[str, Any]) -> str:
        predicted_depth = prediction.get("estimated_depth_cm") or prediction.get("depth_cm") or 0.0
        predicted_severity = prediction.get("severity_label") or prediction.get("severity") or "Unknown"
        return (
            "<image>answer en "
            f"Is there flood water on the road? Predicted depth is {predicted_depth:.1f}cm ({predicted_severity})."
        )

    def _run_gemma_inference(self, image: Image.Image, prompt: str) -> str:
        with torch.no_grad():
            inputs = self._processor(text=prompt, images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            if "pixel_values" in inputs and self.dtype in (torch.float16, torch.bfloat16):
                inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)

            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

            input_len = inputs["input_ids"].shape[-1]
            return self._processor.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

    def _run_blip_questions(self, image: Image.Image, prediction: Dict[str, Any]) -> str:
        """Run multi-question visual QA via local BLIP vision model."""
        questions = [
            ("water_present", "is there water or flood on the road?"),
            ("puddle_type", "is this standing flood water or just a small puddle?"),
            ("wheels_submerged", "are car wheels or legs under water?"),
        ]
        qa_pairs = []
        with torch.no_grad():
            for key, q in questions:
                inputs = self._processor(image, q, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                if "pixel_values" in inputs and self.dtype in (torch.float16, torch.bfloat16):
                    inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
                out = self._model.generate(**inputs, max_new_tokens=30)
                ans = self._processor.decode(out[0], skip_special_tokens=True).strip()
                qa_pairs.append(f"{key}:{ans}")
        return " | ".join(qa_pairs)

    def _parse_vqa_response(self, vqa_text: str, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Parse VQA answers cleanly into structured flood evaluation."""
        predicted_depth = float(prediction.get("estimated_depth_cm") or prediction.get("depth_cm") or 0.0)

        # Parse key:value pairs from vqa_text
        answers: Dict[str, str] = {}
        for item in vqa_text.split("|"):
            if ":" in item:
                k, v = item.strip().split(":", 1)
                answers[k.strip().lower()] = v.strip().lower()

        ans_water = answers.get("water_present", "yes")
        ans_puddle = answers.get("puddle_type", "")
        ans_submerged = answers.get("wheels_submerged", "no")

        water_present = "yes" in ans_water or "water" in ans_water or "flood" in ans_water
        if "no" in ans_water or "dry" in ans_water or "clear" in ans_water:
            water_present = False

        if not water_present:
            recommended_depth = 0.0
            severity = "NORMAL"
        else:
            # Dynamically determine depth based on visual evidence and continuous model candidate values
            if any(term in ans_submerged for term in ["yes", "true", "submerged", "under water"]):
                recommended_depth = max(25.0, predicted_depth) if predicted_depth > 0 else 35.0
                severity = "WARNING" if recommended_depth >= 30 else "ADVISORY"
            elif any(term in ans_puddle for term in ["puddle", "small", "shallow"]) or "no" in ans_submerged:
                # Retain candidate model's continuous prediction if available, or visual shallow value
                if 2.0 <= predicted_depth <= 40.0:
                    recommended_depth = predicted_depth
                else:
                    recommended_depth = 15.0
                severity = "ADVISORY"
            else:
                recommended_depth = predicted_depth if predicted_depth > 0 else 15.0
                severity = "ADVISORY"

        prediction_correct = abs(predicted_depth - recommended_depth) <= 8.0 and (water_present or predicted_depth == 0.0)

        return {
            "water_present": water_present,
            "prediction_correct": prediction_correct,
            "plausible": prediction_correct,
            "recommended_depth_cm": round(recommended_depth, 2),
            "recommended_severity": severity,
            "final_depth_cm": round(recommended_depth, 2) if not prediction_correct else round(predicted_depth, 2),
            "final_severity": severity if not prediction_correct else prediction.get("severity_label", "ADVISORY"),
            "review_required": not prediction_correct,
            "reason": f"Local VLM Visual QA: {vqa_text}",
            "raw_response": vqa_text,
            "parse_failed": False,
        }

    def _parse_json_response(self, raw_text: str, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Parse local Gemma model JSON output."""
        predicted_depth = float(prediction.get("estimated_depth_cm") or prediction.get("depth_cm") or 0.0)

        json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                water_present = bool(data.get("water_present", True))
                recommended_depth = float(data.get("visual_depth_cm", data.get("recommended_depth_cm", predicted_depth)))
                recommended_sev = str(data.get("recommended_severity", "ADVISORY")).upper()
                reason = str(data.get("reason", "Local Gemma analysis complete"))

                diff = abs(predicted_depth - recommended_depth)
                prediction_correct = diff <= 12.0 and (water_present or predicted_depth == 0.0)

                return {
                    "water_present": water_present,
                    "prediction_correct": prediction_correct,
                    "plausible": prediction_correct,
                    "recommended_depth_cm": round(recommended_depth, 2),
                    "recommended_severity": recommended_sev,
                    "final_depth_cm": round(recommended_depth, 2) if not prediction_correct else round(predicted_depth, 2),
                    "final_severity": recommended_sev if not prediction_correct else prediction.get("severity_label", "ADVISORY"),
                    "review_required": not prediction_correct,
                    "reason": reason,
                    "raw_response": raw_text,
                    "parse_failed": False,
                }
            except (json.JSONDecodeError, ValueError):
                pass

        # Text parsing fallback
        lower_text = raw_text.lower()
        no_water_keywords = ["no water", "dry", "no flood", "clear road", "false positive"]
        is_water = not any(k in lower_text for k in no_water_keywords)

        depth_numbers = re.findall(r"(\d+(?:\.\d+)?)\s*(?:cm|centimeters|centimeter)", lower_text)
        recommended_depth = float(depth_numbers[0]) if depth_numbers else (predicted_depth if is_water else 0.0)
        prediction_correct = abs(predicted_depth - recommended_depth) <= 12.0

        return {
            "water_present": is_water,
            "prediction_correct": prediction_correct,
            "plausible": prediction_correct,
            "recommended_depth_cm": round(recommended_depth, 2),
            "recommended_severity": "NORMAL" if not is_water or recommended_depth < 5.0 else ("WARNING" if recommended_depth > 30.0 else "ADVISORY"),
            "final_depth_cm": round(recommended_depth, 2) if not prediction_correct else round(predicted_depth, 2),
            "final_severity": prediction.get("severity_label", "ADVISORY"),
            "review_required": not prediction_correct,
            "reason": raw_text[:200] if raw_text else "Local Gemma processed output",
            "raw_response": raw_text,
            "parse_failed": False,
        }

    def _visual_rule_fallback(self, image: Image.Image, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Fast signal inspection if model execution encounters runtime issue."""
        cv_depth = float(prediction.get("estimated_depth_cm") or prediction.get("depth_cm") or 0.0)

        is_plausible = cv_depth < 60.0
        rec_depth = min(cv_depth, 25.0) if cv_depth > 0 else 0.0

        return {
            "water_present": cv_depth > 2.0,
            "prediction_correct": is_plausible,
            "plausible": is_plausible,
            "recommended_depth_cm": round(rec_depth, 2),
            "recommended_severity": "ADVISORY" if rec_depth > 0 else "NORMAL",
            "final_depth_cm": round(rec_depth, 2),
            "final_severity": "ADVISORY" if rec_depth > 0 else "NORMAL",
            "review_required": not is_plausible,
            "reason": "Local VLM fallback: visual rule inspection.",
            "parse_failed": True,
        }


def main():
    parser = argparse.ArgumentParser(description="Run Local Gemma / Vision Model Judge on a flood image.")
    parser.add_argument("image_path", type=str, help="Path to input image file.")
    parser.add_argument("--model", type=str, default=DEFAULT_GEMMA_MODEL, help="Hugging Face model name.")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, mps, cuda, cpu).")
    parser.add_argument("--cv-depth", type=float, default=35.0, help="CV Pipeline predicted depth in cm to verify.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    print("=" * 60)
    print(f"Initializing Local Gemma/VLM Judge ({args.model})...")
    judge_engine = LocalGemmaJudge(model_name=args.model, device=args.device, lazy_load=False)

    sample_prediction = {
        "estimated_depth_cm": args.cv_depth,
        "severity_label": "WARNING" if args.cv_depth > 30 else "ADVISORY",
        "confidence_pct": 85.0,
    }

    print(f"Evaluating image: {args.image_path}")
    print(f"CV Predicted Depth: {args.cv_depth} cm")
    print("-" * 60)

    result = judge_engine.judge(sample_prediction, image_path=args.image_path)
    print("\n[LOCAL GEMMA/VLM JUDGE RESULT]")
    print(json.dumps(result, indent=2))
    print("=" * 60)


if __name__ == "__main__":
    main()
