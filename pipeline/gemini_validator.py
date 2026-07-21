# pipeline/gemini_validator.py
"""
Optional Stage 6 — Gemini Vision Ensemble Validator
=====================================================
Runs AFTER stages 3 (depth), 4 (fusion), 5 (severity).
Sends the original image + structured features to Gemini Vision API.
Gemini acts as a second "expert" — its estimates are merged with the
model output using a weighted ensemble.

Ensemble logic:
  final_depth  = w_model * model_depth  + w_gemini * gemini_depth
  final_risk   = majority vote (model vs gemini, model wins on tie)
  confidence   = average of both, boosted if they agree

Enable by setting GEMINI_API_KEY in environment or passing api_key=.
"""

import base64
import json
import logging
import os
import re
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger("pipeline.gemini_validator")

# Ensemble weights — model is primary, Gemini is secondary validator
WEIGHT_MODEL  = 0.65
WEIGHT_GEMINI = 0.35

RISK_LEVELS   = ["NO FLOOD", "LOW RISK", "MODERATE", "HIGH RISK", "CRITICAL"]
RISK_TO_DEPTH = {              # midpoint depth cm for each risk band
    "NO FLOOD":  0,
    "LOW RISK":  8,
    "MODERATE":  25,
    "HIGH RISK": 48,
    "CRITICAL":  80,
}

# ── Gemini prompt ─────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """You are a flood analysis expert reviewing CCTV / road camera images for BBMP Bengaluru.

Analyse the image and return a JSON object with EXACTLY these keys:
{
  "flood_detected": true or false,
  "estimated_depth_cm": <number 0-120>,
  "risk_level": one of ["NO FLOOD","LOW RISK","MODERATE","HIGH RISK","CRITICAL"],
  "confidence_pct": <number 0-100>,
  "reasoning": "<one sentence>"
}

Depth guide:
  0 cm        = dry road, no flood
  1–15 cm     = surface water, passable
  15–35 cm    = shallow flood, low vehicles affected
  35–60 cm    = significant flood, most vehicles affected
  60–120 cm   = deep flood, dangerous, evacuation needed

Reply with ONLY valid JSON. No markdown, no explanation outside the JSON."""


@dataclass
class GeminiResult:
    flood_detected:   bool
    estimated_depth_cm: float
    risk_level:       str
    confidence_pct:   float
    reasoning:        str
    raw_response:     str
    success:          bool
    error:            str = ""


@dataclass
class EnsembleResult:
    """Final merged output after model + Gemini ensemble."""
    # Ensemble outputs
    flood_detected:      bool
    water_depth_cm:      float
    risk_level:          str
    confidence_pct:      float
    recommended_action:  str

    # Provenance
    model_depth_cm:      float
    model_risk:          str
    model_confidence:    float

    gemini_depth_cm:     float
    gemini_risk:         str
    gemini_confidence:   float
    gemini_reasoning:    str

    agreement:           bool     # True if model and Gemini agree on risk level
    ensemble_method:     str      # "model_gemini_weighted" | "model_only"


# ─────────────────────────────────────────────────────────────────────────────
class GeminiValidator:
    """
    Optional validator. Safe to construct even if no API key is set —
    .validate() returns None and the pipeline continues unaffected.
    """

    def __init__(self, api_key: str | None = None, model: str = "gemini-1.5-flash"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.model   = model
        self.enabled = bool(self.api_key)
        if self.enabled:
            log.info("Gemini validator enabled (model=%s)", self.model)
        else:
            log.info("Gemini validator disabled — set GEMINI_API_KEY to enable")

    # ── Public ────────────────────────────────────────────────────────────────
    def validate(
        self,
        image_bgr:    np.ndarray,
        model_depth:  float,
        model_risk:   str,
        model_conf:   float,
    ) -> EnsembleResult | None:
        """
        Returns EnsembleResult if Gemini is enabled and succeeds.
        Returns None if disabled or API call fails — caller uses model result as-is.
        """
        if not self.enabled:
            return None

        gemini = self._call_gemini(image_bgr, model_depth, model_risk)
        if not gemini.success:
            log.warning("Gemini call failed: %s — using model result only", gemini.error)
            return None

        return self._ensemble(gemini, model_depth, model_risk, model_conf)

    # ── Gemini API call ───────────────────────────────────────────────────────
    def _call_gemini(
        self, image_bgr: np.ndarray, model_depth: float, model_risk: str
    ) -> GeminiResult:
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)

            # Encode image as JPEG bytes
            _, buf  = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            img_b64 = base64.b64encode(buf.tobytes()).decode()

            context = (
                f"\n\nFor context, our computer vision model estimated:\n"
                f"  depth = {model_depth} cm\n"
                f"  risk  = {model_risk}\n"
                f"Please confirm or correct these estimates based on what you see in the image."
            )

            model_obj = genai.GenerativeModel(self.model)
            response  = model_obj.generate_content([
                _SYSTEM_PROMPT + context,
                {"mime_type": "image/jpeg", "data": img_b64},
            ])
            raw = response.text.strip()
            return self._parse_gemini_response(raw)

        except ImportError:
            return GeminiResult(
                flood_detected=False, estimated_depth_cm=0, risk_level="NO FLOOD",
                confidence_pct=0, reasoning="", raw_response="",
                success=False, error="google-generativeai not installed. pip install google-generativeai",
            )
        except Exception as e:
            return GeminiResult(
                flood_detected=False, estimated_depth_cm=0, risk_level="NO FLOOD",
                confidence_pct=0, reasoning="", raw_response="",
                success=False, error=str(e),
            )

    def _parse_gemini_response(self, raw: str) -> GeminiResult:
        # Strip markdown code fences if present
        clean = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
        try:
            d = json.loads(clean)
            return GeminiResult(
                flood_detected    = bool(d.get("flood_detected", False)),
                estimated_depth_cm= float(d.get("estimated_depth_cm", 0)),
                risk_level        = d.get("risk_level", "NO FLOOD"),
                confidence_pct    = float(d.get("confidence_pct", 50)),
                reasoning         = d.get("reasoning", ""),
                raw_response      = raw,
                success           = True,
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            return GeminiResult(
                flood_detected=False, estimated_depth_cm=0, risk_level="NO FLOOD",
                confidence_pct=0, reasoning="", raw_response=raw,
                success=False, error=f"JSON parse failed: {e}",
            )

    # ── Ensemble ──────────────────────────────────────────────────────────────
    def _ensemble(
        self,
        gemini:       GeminiResult,
        model_depth:  float,
        model_risk:   str,
        model_conf:   float,
    ) -> EnsembleResult:

        # Weighted average depth
        ensemble_depth = round(
            WEIGHT_MODEL * model_depth + WEIGHT_GEMINI * gemini.estimated_depth_cm, 1
        )

        # Risk: use ensemble depth to re-derive, but cap at max of both (safety-first)
        model_idx  = RISK_LEVELS.index(model_risk)  if model_risk  in RISK_LEVELS else 0
        gemini_idx = RISK_LEVELS.index(gemini.risk_level) if gemini.risk_level in RISK_LEVELS else 0

        # Safety-first: take the higher risk between model and Gemini
        ensemble_idx  = max(model_idx, gemini_idx)
        ensemble_risk = RISK_LEVELS[ensemble_idx]

        # Confidence boost if they agree
        agreement = (model_risk == gemini.risk_level)
        base_conf = WEIGHT_MODEL * model_conf + WEIGHT_GEMINI * (gemini.confidence_pct / 100)
        ensemble_conf = min(base_conf * (1.1 if agreement else 0.9), 1.0)

        action = _risk_to_action(ensemble_risk)

        log.info("Ensemble: model=%s(%.0fcm) gemini=%s(%.0fcm) → %s(%.0fcm) agree=%s",
                 model_risk, model_depth,
                 gemini.risk_level, gemini.estimated_depth_cm,
                 ensemble_risk, ensemble_depth, agreement)

        return EnsembleResult(
            flood_detected     = ensemble_depth > 0,
            water_depth_cm     = ensemble_depth,
            risk_level         = ensemble_risk,
            confidence_pct     = round(ensemble_conf * 100, 1),
            recommended_action = action,
            model_depth_cm     = model_depth,
            model_risk         = model_risk,
            model_confidence   = round(model_conf * 100, 1),
            gemini_depth_cm    = gemini.estimated_depth_cm,
            gemini_risk        = gemini.risk_level,
            gemini_confidence  = gemini.confidence_pct,
            gemini_reasoning   = gemini.reasoning,
            agreement          = agreement,
            ensemble_method    = "model_gemini_weighted",
        )


# ── Helpers ───────────────────────────────────────────────────────────────────
_ACTIONS = {
    "NO FLOOD":  "No action required. Normal traffic conditions.",
    "LOW RISK":  "Alert field teams. Monitor every 15 minutes.",
    "MODERATE":  "Deploy water barriers. Divert traffic. Alert residents.",
    "HIGH RISK": "Evacuate affected zones. Close roads. Deploy BBMP emergency teams.",
    "CRITICAL":  "IMMEDIATE EVACUATION. All emergency units deployed. Declare disaster zone.",
}

def _risk_to_action(risk: str) -> str:
    return _ACTIONS.get(risk, "Monitor situation.")
