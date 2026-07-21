# pipeline/runner.py
"""
PipelineRunner — wires all 5 stages + optional Gemini ensemble
===============================================================
Call .run_image(camera_image) for a single frame.
Call .run_batch(batch_job)   for a full 15-min batch.
"""

import logging
import cv2
from pathlib import Path
from typing import List

from ingestor            import CameraImage, BatchJob
from .segformer          import SegFormerStage
from .yolo               import YOLOStage
from .depth              import DepthStage
from .fusion             import FusionStage
from .severity           import SeverityStage, FloodPrediction
from .gemini_validator   import GeminiValidator, EnsembleResult
from .temporal           import TemporalSmoother

log = logging.getLogger("pipeline.runner")


class PipelineRunner:
    """
    Initialise once at app start.  All stages share their loaded weights
    across every image in every batch — no repeated model loading.

    Set cfg["pipeline"]["gemini_api_key"] or env GEMINI_API_KEY to enable
    Gemini ensemble validation after stage 5.

    Temporal smoothing (P4) is applied to water_depth_cm per camera_id
    before the prediction is returned, stabilising the live map display.
    """

    def __init__(self, cfg: dict):
        p      = cfg.get("pipeline", {})
        device = p.get("device", "cpu")
        stub   = p.get("stub_mode", False)

        log.info("Loading pipeline stages…")
        self.seg      = SegFormerStage(p.get("segformer_weights"),  device, stub_mode=stub)
        self.yolo     = YOLOStage(p.get("yolo_weights"),            device,
                                   p.get("yolo_conf_threshold", 0.4), stub_mode=stub)
        self.depth    = DepthStage(p.get("depth_weights"),          device, stub_mode=stub)
        self.fusion   = FusionStage(p.get("sensor_height_cm", 300))
        self.severity = SeverityStage(p.get("severity_weights"))

        # Optional Gemini ensemble — safe no-op if key not set
        self.gemini   = GeminiValidator(
            api_key = p.get("gemini_api_key"),
            model   = p.get("gemini_model", "gemini-1.5-flash"),
        )

        # Per-camera temporal smoother (P4)
        self.smoother = TemporalSmoother(
            window_size = p.get("temporal_window_size", 5),
            method      = p.get("temporal_smoothing",  "ema"),
            alpha       = p.get("temporal_alpha",      0.3),
        )

        log.info("Pipeline ready  (gemini=%s, smoother=%s)",
                 self.gemini.enabled, self.smoother.method)

    # ── Single image (from file path) ─────────────────────────────────────────
    def run_image(self, cam_img: CameraImage, batch_id: str = "") -> FloodPrediction:
        img_bgr = cv2.imread(str(cam_img.image_path))
        if img_bgr is None:
            raise ValueError(f"Cannot read image: {cam_img.image_path}")
        return self._run_bgr(
            img_bgr       = img_bgr,
            camera_id     = cam_img.camera_id,
            location_id   = cam_img.location_id,
            latitude      = cam_img.latitude,
            longitude     = cam_img.longitude,
            location_name = cam_img.location_name,
            captured_at   = cam_img.captured_at,
            batch_id      = batch_id,
        )

    # ── Single image (from in-memory BGR array — used by serve.py) ────────────
    def run_b64_image(
        self,
        img_bgr:       "np.ndarray",
        camera_id:     str,
        latitude:      float,
        longitude:     float,
        location_id:   str = "",
        location_name: str = "",
        captured_at:   str = "",
        batch_id:      str = "",
    ) -> FloodPrediction:
        """Accept a pre-decoded BGR image (from base64) instead of a file path."""
        import numpy as np  # noqa: F811
        if img_bgr is None or not isinstance(img_bgr, np.ndarray):
            raise ValueError("img_bgr must be a non-None numpy array")
        return self._run_bgr(
            img_bgr       = img_bgr,
            camera_id     = camera_id,
            location_id   = location_id or camera_id,
            latitude      = latitude,
            longitude     = longitude,
            location_name = location_name or camera_id,
            captured_at   = captured_at,
            batch_id      = batch_id,
        )

    # ── Shared execution core ──────────────────────────────────────────────────
    def _run_bgr(
        self,
        img_bgr:       "np.ndarray",
        camera_id:     str,
        location_id:   str,
        latitude:      float,
        longitude:     float,
        location_name: str,
        captured_at:   str,
        batch_id:      str,
    ) -> FloodPrediction:

        # ── Stages 1–5 ───────────────────────────────────────────────────────
        seg_res    = self.seg.predict(img_bgr)
        yolo_res   = self.yolo.predict(img_bgr)
        depth_res  = self.depth.predict(img_bgr)
        features   = self.fusion.fuse(img_bgr, seg_res, yolo_res, depth_res)
        prediction = self.severity.predict(
            features      = features,
            location_id   = location_id,
            camera_id     = camera_id,
            latitude      = latitude,
            longitude     = longitude,
            location_name = location_name,
            timestamp     = captured_at,
            batch_id      = batch_id,
        )

        # ── Stage 6 (optional) — Gemini ensemble ─────────────────────────────
        ensemble: EnsembleResult | None = self.gemini.validate(
            image_bgr   = img_bgr,
            model_depth = prediction.water_depth_cm,
            model_risk  = prediction.risk_level,
            model_conf  = prediction.confidence_pct / 100,
        )

        if ensemble is not None:
            prediction.flood_detected      = ensemble.flood_detected
            prediction.water_depth_cm      = ensemble.water_depth_cm
            prediction.risk_level          = ensemble.risk_level
            prediction.recommended_action  = ensemble.recommended_action
            prediction.gemini_depth_cm     = ensemble.gemini_depth_cm
            prediction.gemini_risk         = ensemble.gemini_risk
            prediction.gemini_confidence   = ensemble.gemini_confidence
            prediction.gemini_reasoning    = ensemble.gemini_reasoning
            prediction.gemini_agreement    = ensemble.agreement
            prediction.ensemble_method     = ensemble.ensemble_method

            # P5 — Confidence from numeric model/Gemini depth agreement
            # agreement_score = 1 − |model_depth − gemini_depth| / max(both, 1)
            # Replaces hardcoded 0.85/0.55/0.9 buckets with a real signal.
            model_d  = ensemble.model_depth_cm
            gemini_d = ensemble.gemini_depth_cm
            agreement_score = 1.0 - min(
                abs(model_d - gemini_d) / max(model_d, gemini_d, 1.0), 1.0
            )
            prediction.gemini_agreement_score = round(agreement_score, 3)
            # Blend: 70% ensemble confidence + 30% numeric agreement signal
            blended_conf = 0.7 * (ensemble.confidence_pct / 100) + 0.3 * agreement_score
            prediction.confidence_pct = round(min(blended_conf, 1.0) * 100, 1)
        else:
            prediction.gemini_depth_cm      = None
            prediction.gemini_risk          = None
            prediction.gemini_confidence    = None
            prediction.gemini_reasoning     = None
            prediction.gemini_agreement     = None
            prediction.gemini_agreement_score = None
            prediction.ensemble_method      = "model_only"

        # ── P4 — Temporal smoothing per camera ────────────────────────────────
        raw_depth = prediction.water_depth_cm
        prediction.water_depth_cm = self.smoother.smooth(camera_id, raw_depth)
        if raw_depth != prediction.water_depth_cm:
            log.debug("  Temporal smooth %s: %.1f → %.1f cm",
                      camera_id, raw_depth, prediction.water_depth_cm)

        log.info("  %-30s → %-10s %.1f cm (%.0f%% conf) [%s]",
                 camera_id, prediction.risk_level,
                 prediction.water_depth_cm, prediction.confidence_pct,
                 prediction.ensemble_method)
        return prediction

    # ── Full batch ────────────────────────────────────────────────────────────
    def run_batch(self, batch: BatchJob) -> List[FloodPrediction]:
        results = []
        log.info("Running batch %s — %d images", batch.batch_id, len(batch.images))
        for cam_img in batch.images:
            try:
                pred = self.run_image(cam_img, batch.batch_id)
                results.append(pred)
            except Exception:
                log.exception("  Failed on %s", cam_img.camera_id)
        log.info("Batch %s done — %d predictions", batch.batch_id, len(results))
        return results
