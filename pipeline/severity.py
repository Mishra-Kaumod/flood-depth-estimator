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
        # import joblib; return joblib.load(path)   # sklearn / XGBoost
        raise NotImplementedError

    def _score(self, f: StructuredFeatures) -> tuple[float, float]:
        """Returns (estimated_depth_cm, confidence 0-1)."""
        if self._model is not None:
            # feature_vec = np.array([[f.water_coverage_pct,
            #                          f.mean_flood_depth_cm,
            #                          f.p90_flood_depth_cm,
            #                          f.max_flood_depth_cm]])
            # depth = float(self._model.predict(feature_vec)[0])
            # conf  = 0.85
            pass

        # Rule-based fallback — use p90 depth (robust to outliers)
        depth  = f.p90_flood_depth_cm
        # Confidence based on calibration source + water coverage
        conf   = 0.85 if "yolo_" in f.calibration_source else 0.55
        if f.water_coverage_pct < 1:
            depth, conf = 0.0, 0.9   # clearly no flood
        return round(depth, 1), round(conf, 3)

    @staticmethod
    def _depth_to_risk(depth_cm: float) -> tuple[str, str]:
        for threshold, risk, action in _RISK_TABLE:
            if depth_cm <= threshold:
                return risk, action
        return "CRITICAL", _RISK_TABLE[-1][2]
