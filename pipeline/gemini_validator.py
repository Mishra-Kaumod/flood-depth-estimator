# pipeline/gemini_validator.py
"""
Optional Stage 6 — Gemini Vision Ensemble Validator
=====================================================
Runs AFTER stages 3 (depth), 4 (fusion), 5 (severity).

Two-call design (fixes anchoring bias):
  • _call_gemini_blind()   — image only, NO model estimates.
                             Used for ensembling. Independent signal.
  • _call_gemini_review()  — image + model estimates, asks Gemini to flag
                             disagreements. Used for human-review logging ONLY,
                             never for agreement_score or confidence blending.

Ensemble weights are dynamic (confidence-normalised), not fixed constants:
  w_model  = model_conf  / (model_conf  + gemini_conf)
  w_gemini = gemini_conf / (model_conf  + gemini_conf)

This means: if Gemini returns 95% confidence and the severity model returns
35%, Gemini gets ~73% of the depth blend — as it should.

Enable by setting GEMINI_API_KEY in environment or passing api_key=.
"""

import base64
import json
import logging
import os
import re
from dataclasses import dataclass, field

import cv2
import numpy as np

log = logging.getLogger("pipeline.gemini_validator")

RISK_LEVELS   = ["NO FLOOD", "LOW RISK", "MODERATE", "HIGH RISK", "CRITICAL"]
RISK_TO_DEPTH = {              # midpoint depth cm for each risk band
    "NO FLOOD":  0,
    "LOW RISK":  8,
    "MODERATE":  25,
    "HIGH RISK": 48,
    "CRITICAL":  100,
}

# ── Blind prompt — no model context, Gemini forms an independent view ─────────
_BLIND_PROMPT = """You are a flood analysis expert reviewing CCTV / road camera images for BBMP Bengaluru.

Analyse the image and return a JSON object with EXACTLY these keys:
{
  "flood_detected": true or false,
  "estimated_depth_cm": <number 0-200>,
  "risk_level": one of ["NO FLOOD","LOW RISK","MODERATE","HIGH RISK","CRITICAL"],
  "confidence_pct": <number 0-100>,
  "reasoning": "<one sentence>"
}

Depth guide:
  0 cm          = dry road, no flood
  1–15 cm       = surface water, passable by most vehicles
  15–35 cm      = shallow flood, low vehicles affected
  35–60 cm      = significant flood, most vehicles affected
  60–120 cm     = deep flood, dangerous, evacuation required
  120–200 cm    = severe / submerged infrastructure, life-threatening

Reply with ONLY valid JSON. No markdown, no explanation outside the JSON."""

# ── Review prompt — includes model estimate, for disagreement logging only ────
_REVIEW_PROMPT_TMPL = """You are a flood analysis expert performing a second-opinion review for BBMP Bengaluru.

Our computer vision model estimated:
  depth = {model_depth} cm
  risk  = {model_risk}

Examine the image carefully. Does this estimate look correct?
Return a JSON object with EXACTLY these keys:
{{
  "agrees_with_model": true or false,
  "estimated_depth_cm": <your independent estimate, 0-200>,
  "risk_level": one of ["NO FLOOD","LOW RISK","MODERATE","HIGH RISK","CRITICAL"],
  "confidence_pct": <number 0-100>,
  "disagreement_notes": "<specific reason if you disagree, else empty string>"
}}

Reply with ONLY valid JSON. No markdown, no explanation outside the JSON."""


@dataclass
class GeminiResult:
    flood_detected:     bool
    estimated_depth_cm: float
    risk_level:         str
    confidence_pct:     float
    reasoning:          str
    raw_response:       str
    success:            bool
    error:              str = ""


@dataclass
class GeminiReviewResult:
    """Result from the anchored review call — logged but NOT used in ensembling."""
    agrees_with_model:   bool
    estimated_depth_cm:  float
    risk_level:          str
    confidence_pct:      float
    disagreement_notes:  str
    raw_response:        str
    success:             bool
    error:               str = ""


@dataclass
class EnsembleResult:
    """Final merged output after model + Gemini blind ensemble."""
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

    # Ensemble weights actually used (dynamic, confidence-normalised)
    w_model:             float
    w_gemini:            float

    agreement:           bool     # True if risk levels match
    ensemble_method:     str      # "model_gemini_weighted" | "model_only"

    # Optional: anchored review for human audit (None if review disabled)
    review_result:       GeminiReviewResult | None = None


# ─────────────────────────────────────────────────────────────────────────────
class GeminiValidator:
    """
    Optional validator. Safe to construct even if no API key is set —
    .validate() returns None and the pipeline continues unaffected.

    Args:
        enable_review: if True, also call _call_gemini_review() and attach
                       the result to EnsembleResult.review_result for logging.
                       Adds a second API call per image — disable in production
                       if latency is critical.
    """

    def __init__(self, api_key: str | None = None, model: str = "gemini-1.5-flash",
                 enable_review: bool = False):
        self.api_key       = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.model         = model
        self.enabled       = bool(self.api_key)
        self.enable_review = enable_review
        if self.enabled:
            log.info("Gemini validator enabled (model=%s, review=%s)", self.model, enable_review)
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
        Returns EnsembleResult if Gemini is enabled and the blind call succeeds.
        Returns None if disabled or API call fails — caller uses model result as-is.

        Ensembling uses only the blind call (no model context given to Gemini).
        The optional review call is attached as review_result for human audit.
        """
        if not self.enabled:
            return None

        # Primary: blind call — no model context, fully independent
        blind = self._call_gemini_blind(image_bgr)
        if not blind.success:
            log.warning("Gemini blind call failed: %s — using model result only", blind.error)
            return None

        result = self._ensemble(blind, model_depth, model_risk, model_conf)

        # Optional: anchored review for disagreement logging (never touches ensemble)
        if self.enable_review:
            review = self._call_gemini_review(image_bgr, model_depth, model_risk)
            result.review_result = review
            if review.success and not review.agrees_with_model:
                log.warning(
                    "Gemini review disagrees with model (camera depth=%.0f cm): %s",
                    model_depth, review.disagreement_notes,
                )

        return result

    # ── Blind call — image only, no model context ─────────────────────────────
    def _call_gemini_blind(self, image_bgr: np.ndarray) -> GeminiResult:
        """
        Primary call used for ensembling. Sends ONLY the image and the
        blind system prompt — no model estimates, no anchoring.
        """
        img_b64 = self._encode_image(image_bgr)
        if img_b64 is None:
            return self._error_result("Image encoding failed")
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            model_obj = genai.GenerativeModel(self.model)
            response  = model_obj.generate_content([
                _BLIND_PROMPT,
                {"mime_type": "image/jpeg", "data": img_b64},
            ])
            return self._parse_gemini_response(response.text.strip())
        except ImportError:
            return self._error_result(
                "google-generativeai not installed. pip install google-generativeai"
            )
        except Exception as e:
            return self._error_result(str(e))

    # ── Anchored review — for disagreement logging only, NOT ensembling ───────
    def _call_gemini_review(
        self, image_bgr: np.ndarray, model_depth: float, model_risk: str
    ) -> GeminiReviewResult:
        """
        Secondary call that includes the model's estimate.
        Result is logged for human review; never used in agreement_score
        or confidence blending.
        """
        img_b64 = self._encode_image(image_bgr)
        if img_b64 is None:
            return GeminiReviewResult(False, 0, "NO FLOOD", 0, "", "", False, "Image encoding failed")
        prompt = _REVIEW_PROMPT_TMPL.format(model_depth=model_depth, model_risk=model_risk)
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            model_obj = genai.GenerativeModel(self.model)
            response  = model_obj.generate_content([
                prompt,
                {"mime_type": "image/jpeg", "data": img_b64},
            ])
            raw   = response.text.strip()
            clean = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
            d = json.loads(clean)
            return GeminiReviewResult(
                agrees_with_model  = bool(d.get("agrees_with_model", True)),
                estimated_depth_cm = float(d.get("estimated_depth_cm", model_depth)),
                risk_level         = d.get("risk_level", model_risk),
                confidence_pct     = float(d.get("confidence_pct", 50)),
                disagreement_notes = d.get("disagreement_notes", ""),
                raw_response       = raw,
                success            = True,
            )
        except ImportError:
            return GeminiReviewResult(False, 0, "NO FLOOD", 0, "", "", False,
                                      "google-generativeai not installed")
        except Exception as e:
            return GeminiReviewResult(False, 0, "NO FLOOD", 0, "", "", False, str(e))

    # ── Ensemble ──────────────────────────────────────────────────────────────
    def _ensemble(
        self,
        gemini:       GeminiResult,
        model_depth:  float,
        model_risk:   str,
        model_conf:   float,
    ) -> EnsembleResult:
        """
        Confidence-normalised weighted ensemble (no fixed constants).

        Weights:
          w_model  = model_conf  / (model_conf  + gemini_conf_01)
          w_gemini = gemini_conf / (model_conf  + gemini_conf_01)

        This shifts the depth blend toward whichever source is more confident.
        Both weights always sum to 1.0.
        """
        gemini_conf_01 = gemini.confidence_pct / 100.0
        total = model_conf + gemini_conf_01
        if total < 1e-6:   # both sources report zero confidence
            w_model, w_gemini = 0.5, 0.5
        else:
            w_model  = model_conf      / total
            w_gemini = gemini_conf_01  / total

        # Confidence-weighted depth blend
        ensemble_depth = round(w_model * model_depth + w_gemini * gemini.estimated_depth_cm, 1)

        # Safety-first risk: take the higher of the two
        model_idx  = RISK_LEVELS.index(model_risk)         if model_risk         in RISK_LEVELS else 0
        gemini_idx = RISK_LEVELS.index(gemini.risk_level)  if gemini.risk_level  in RISK_LEVELS else 0
        ensemble_idx  = max(model_idx, gemini_idx)
        ensemble_risk = RISK_LEVELS[ensemble_idx]

        # Ensemble confidence: weighted average, boosted when both agree on risk
        agreement     = (model_risk == gemini.risk_level)
        base_conf     = w_model * model_conf + w_gemini * gemini_conf_01
        ensemble_conf = min(base_conf * (1.1 if agreement else 0.9), 1.0)
        action        = _risk_to_action(ensemble_risk)

        log.info(
            "Ensemble(blind): model=%s(%.0fcm,conf=%.0f%%) gemini=%s(%.0fcm,conf=%.0f%%) "
            "→ %s(%.0fcm) w_model=%.2f w_gemini=%.2f agree=%s",
            model_risk, model_depth, model_conf * 100,
            gemini.risk_level, gemini.estimated_depth_cm, gemini.confidence_pct,
            ensemble_risk, ensemble_depth, w_model, w_gemini, agreement,
        )

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
            w_model            = round(w_model, 3),
            w_gemini           = round(w_gemini, 3),
            agreement          = agreement,
            ensemble_method    = "model_gemini_weighted",
        )

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _encode_image(self, image_bgr: np.ndarray) -> str | None:
        try:
            _, buf  = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return base64.b64encode(buf.tobytes()).decode()
        except Exception as e:
            log.error("Image encoding failed: %s", e)
            return None

    @staticmethod
    def _error_result(error: str) -> GeminiResult:
        return GeminiResult(
            flood_detected=False, estimated_depth_cm=0, risk_level="NO FLOOD",
            confidence_pct=0, reasoning="", raw_response="",
            success=False, error=error,
        )

    def _parse_gemini_response(self, raw: str) -> GeminiResult:
        clean = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
        try:
            d = json.loads(clean)
            return GeminiResult(
                flood_detected     = bool(d.get("flood_detected", False)),
                estimated_depth_cm = float(d.get("estimated_depth_cm", 0)),
                risk_level         = d.get("risk_level", "NO FLOOD"),
                confidence_pct     = float(d.get("confidence_pct", 50)),
                reasoning          = d.get("reasoning", ""),
                raw_response       = raw,
                success            = True,
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            return self._error_result(f"JSON parse failed: {e}")


# ── Module-level helpers ───────────────────────────────────────────────────────
_ACTIONS = {
    "NO FLOOD":  "No action required. Normal traffic conditions.",
    "LOW RISK":  "Alert field teams. Monitor every 15 minutes.",
    "MODERATE":  "Deploy water barriers. Divert traffic. Alert residents.",
    "HIGH RISK": "Evacuate affected zones. Close roads. Deploy BBMP emergency teams.",
    "CRITICAL":  "IMMEDIATE EVACUATION. All emergency units deployed. Declare disaster zone.",
}

def _risk_to_action(risk: str) -> str:
    return _ACTIONS.get(risk, "Monitor situation.")
