# pipeline/severity.py
"""
Stage 5 — Calibration / Severity Model
========================================
Input : StructuredFeatures (explainable features from FusionStage)
Output: FloodPrediction (the 5 map outputs)

This is where your trained model (EfficientNet / future calibration model)
plugs in.  Today: rule-based on p90_flood_depth_cm (fully explainable).
Future:    sklearn / XGBoost trained on structured features.
"""

import logging
from dataclasses import dataclass, asdict
from pathlib import Path

from .fusion import StructuredFeatures

log = logging.getLogger("pipeline.severity")

# ── Risk table (depth thresholds in cm) ──────────────────────────────────────
_RISK_TABLE = [
    (0,    "NO FLOOD",  "No action required. Normal traffic conditions."),
    (15,   "LOW RISK",  "Alert field teams. Monitor every 15 minutes."),
    (35,   "MODERATE",  "Deploy water barriers. Divert traffic. Alert residents."),
    (60,   "HIGH RISK", "Evacuate affected zones. Close roads. Deploy BBMP emergency teams."),
    (9999, "CRITICAL",  "IMMEDIATE EVACUATION. All emergency units deployed. Declare disaster zone."),
]


@dataclass
class FloodPrediction:
    """The 5 outputs shown on the Bengaluru map."""
    flood_detected:      bool    # Output 1
    water_depth_cm:      float   # Output 2  (p90 depth at water pixels)
    risk_level:          str     # Output 3
    recommended_action:  str     # Output 4
    confidence_pct:      float   # Output 5

    # Metadata (for DB + map placement — not counted as model outputs)
    location_id:    str
    camera_id:      str
    latitude:       float
    longitude:      float
    location_name:  str
    timestamp:      str
    batch_id:       str

    # Explainability fields (shown in UI detail panel)
    water_coverage_pct:   float
    mean_flood_depth_cm:  float
    max_flood_depth_cm:   float
    calibration_source:   str
    seg_engine:           str
    yolo_engine:          str
    depth_engine:         str

    # Gemini ensemble fields (populated by runner.py; None when Gemini disabled)
    gemini_depth_cm:       float | None = None
    gemini_risk:           str   | None = None
    gemini_confidence:     float | None = None
    gemini_reasoning:      str   | None = None
    gemini_agreement:      bool  | None = None
    gemini_agreement_score: float | None = None  # 0–1 numeric agreement
    ensemble_method:       str   | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        # Remove numpy arrays (water_mask etc.) if accidentally included
        return {k: v for k, v in d.items() if not hasattr(v, '__len__') or isinstance(v, str)}


class SeverityStage:
    """
    Converts StructuredFeatures → FloodPrediction.
    Plug in your trained model here by overriding _score().
    """

    def __init__(self, model_path: str | None = None):
        self._model = None
        if model_path and Path(model_path).exists():
            try:
                self._model = self._load(model_path)
                log.info("Severity model loaded from %s", model_path)
            except Exception:
                log.warning("Severity model load failed — using rule-based", exc_info=True)

    # ── Public ────────────────────────────────────────────────────────────────
    def predict(
        self,
        features:      StructuredFeatures,
        location_id:   str,
        camera_id:     str,
        latitude:      float,
        longitude:     float,
        location_name: str,
        timestamp:     str,
        batch_id:      str,
    ) -> FloodPrediction:

        depth_cm, conf = self._score(features)
        risk, action   = self._depth_to_risk(depth_cm)

        return FloodPrediction(
            flood_detected     = depth_cm > 0,
            water_depth_cm     = depth_cm,
            risk_level         = risk,
            recommended_action = action,
            confidence_pct     = round(conf * 100, 1),
            location_id        = location_id,
            camera_id          = camera_id,
            latitude           = latitude,
            longitude          = longitude,
            location_name      = location_name,
            timestamp          = timestamp,
            batch_id           = batch_id,
            water_coverage_pct  = features.water_coverage_pct,
            mean_flood_depth_cm = features.mean_flood_depth_cm,
            max_flood_depth_cm  = features.max_flood_depth_cm,
            calibration_source  = features.calibration_source,
            seg_engine          = features.seg_engine,
            yolo_engine         = features.yolo_engine,
            depth_engine        = features.depth_engine,
        )

    # ── Scoring (swap for ML model here) ─────────────────────────────────────
    def _load(self, path: str):
        """
        Load the joblib bundle produced by scripts/train_severity_model.py.
        Bundle keys: gbr_mean, gbr_q10, gbr_q90, feature_cols, max_depth_in_train.
        """
        try:
            import joblib
        except ImportError as exc:
            raise ImportError("pip install joblib") from exc
        bundle = joblib.load(path)
        required = {"gbr_mean", "gbr_q10", "gbr_q90", "feature_cols", "max_depth_in_train"}
        missing = required - set(bundle.keys())
        if missing:
            raise ValueError(f"Model bundle missing keys: {missing}")
        return bundle

    def _score(self, f: StructuredFeatures) -> tuple[float, float]:
        """
        Returns (estimated_depth_cm, confidence 0-1).

        When trained model is loaded:
          depth = gbr_mean.predict(feature_vec)
          conf  = 1 − clip((q90 − q10) / max_depth_in_train, 0, 1)
          (narrow prediction interval = high confidence)

        Fallback (no model):
          depth = p90_flood_depth_cm (robust to outliers)
          conf  = calibration_confidence (from FusionStage, not hardcoded buckets)
        """
        if f.water_coverage_pct < 1:
            return 0.0, 0.9   # clearly no flood — short-circuit before any model

        if self._model is not None:
            import numpy as np
            gbr_mean = self._model["gbr_mean"]
            gbr_q10  = self._model["gbr_q10"]
            gbr_q90  = self._model["gbr_q90"]
            feature_cols     = self._model["feature_cols"]
            max_depth_train  = self._model["max_depth_in_train"]

            feat_map = {
                "water_coverage_pct":   f.water_coverage_pct,
                "mean_flood_depth_cm":  f.mean_flood_depth_cm,
                "p90_flood_depth_cm":   f.p90_flood_depth_cm,
                "max_flood_depth_cm":   f.max_flood_depth_cm,
                "calibration_confidence": f.calibration_confidence,
            }
            X = np.array([[feat_map[c] for c in feature_cols]])

            depth   = max(float(gbr_mean.predict(X)[0]), 0.0)
            q10_val = float(gbr_q10.predict(X)[0])
            q90_val = float(gbr_q90.predict(X)[0])
            interval_width = max(q90_val - q10_val, 0.0)
            conf = 1.0 - min(interval_width / max(max_depth_train, 1.0), 1.0)
            return round(depth, 1), round(conf, 3)

        # ── Rule-based fallback (no trained model yet) ────────────────────
        depth = f.p90_flood_depth_cm
        # Use calibration_confidence from FusionStage — it reflects how well
        # the depth map was anchored (YOLO reference vs sensor-height fallback).
        # Avoids hardcoded 0.85/0.55 buckets.
        conf = f.calibration_confidence
        return round(depth, 1), round(conf, 3)

    @staticmethod
    def _depth_to_risk(depth_cm: float) -> tuple[str, str]:
        for threshold, risk, action in _RISK_TABLE:
            if depth_cm <= threshold:
                return risk, action
        return "CRITICAL", _RISK_TABLE[-1][2]
