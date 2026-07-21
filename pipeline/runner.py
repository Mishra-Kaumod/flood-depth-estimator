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

log = logging.getLogger("pipeline.runner")


class PipelineRunner:
    """
    Initialise once at app start.  All stages share their loaded weights
    across every image in every batch — no repeated model loading.

    Set cfg["pipeline"]["gemini_api_key"] or env GEMINI_API_KEY to enable
    Gemini ensemble validation after stage 5.
    """

    def __init__(self, cfg: dict):
        p      = cfg.get("pipeline", {})
        device = p.get("device", "cpu")

        log.info("Loading pipeline stages…")
        self.seg      = SegFormerStage(p.get("segformer_weights"),  device)
        self.yolo     = YOLOStage(p.get("yolo_weights"),            device,
                                   p.get("yolo_conf_threshold", 0.4))
        self.depth    = DepthStage(p.get("depth_weights"),          device)
        self.fusion   = FusionStage(p.get("sensor_height_cm", 300))
        self.severity = SeverityStage(p.get("severity_weights"))

        # Optional Gemini ensemble — safe no-op if key not set
        self.gemini   = GeminiValidator(
            api_key = p.get("gemini_api_key"),
            model   = p.get("gemini_model", "gemini-1.5-flash"),
        )
        log.info("Pipeline ready  (gemini=%s)", self.gemini.enabled)

    # ── Single image ──────────────────────────────────────────────────────────
    def run_image(self, cam_img: CameraImage, batch_id: str = "") -> FloodPrediction:
        img_bgr = cv2.imread(str(cam_img.image_path))
        if img_bgr is None:
            raise ValueError(f"Cannot read image: {cam_img.image_path}")

        # ── Stages 1–5 ───────────────────────────────────────────────────────
        seg_res    = self.seg.predict(img_bgr)
        yolo_res   = self.yolo.predict(img_bgr)
        depth_res  = self.depth.predict(img_bgr)
        features   = self.fusion.fuse(img_bgr, seg_res, yolo_res, depth_res)
        prediction = self.severity.predict(
            features      = features,
            location_id   = cam_img.location_id,
            camera_id     = cam_img.camera_id,
            latitude      = cam_img.latitude,
            longitude     = cam_img.longitude,
            location_name = cam_img.location_name,
            timestamp     = cam_img.captured_at,
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
            prediction.flood_detected     = ensemble.flood_detected
            prediction.water_depth_cm     = ensemble.water_depth_cm
            prediction.risk_level         = ensemble.risk_level
            prediction.recommended_action = ensemble.recommended_action
            prediction.confidence_pct     = ensemble.confidence_pct
            prediction.gemini_depth_cm    = ensemble.gemini_depth_cm
            prediction.gemini_risk        = ensemble.gemini_risk
            prediction.gemini_confidence  = ensemble.gemini_confidence
            prediction.gemini_reasoning   = ensemble.gemini_reasoning
            prediction.gemini_agreement   = ensemble.agreement
            prediction.ensemble_method    = ensemble.ensemble_method
        else:
            prediction.gemini_depth_cm    = None
            prediction.gemini_risk        = None
            prediction.gemini_confidence  = None
            prediction.gemini_reasoning   = None
            prediction.gemini_agreement   = None
            prediction.ensemble_method    = "model_only"

        log.info("  %-30s → %-10s %.1f cm (%.0f%% conf) [%s]",
                 cam_img.camera_id, prediction.risk_level,
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
